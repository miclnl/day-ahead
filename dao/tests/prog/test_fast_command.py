"""F2c1 contract regressions: no receiver and no financial execution rights."""

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, fields, replace
from importlib import import_module, util
from pathlib import Path
from typing import get_args, get_type_hints
from unittest.mock import Mock

import pytest

from dao.prog.fastctrl import actuator, policy
from dao.prog.fastctrl.policy import BatteryControllerState, Decision
from dao.prog.fastctrl.runner import FastControlRunner, load_state, save_state
from dao.tests.prog.test_fast_actuator import ScriptedGateway, proposal
from dao.tests.prog.test_fast_policy import T0, make_plan
from dao.tests.prog.test_fast_runner import FakeHass, RecordingFallback, make_config
from dao.tests.settlement.test_fast_contract import ready_plan


def contract():
    # Report the missing contract as an assertion, not a collection/import error.
    assert util.find_spec("dao.prog.fastctrl.command") is not None, "canonical command contract missing"
    return import_module("dao.prog.fastctrl.command")


def command(**changes):
    return contract().BatteryCommand(**(dict(
        installation_id="synthetic", battery_name="accu", owner_session_id="session-a",
        owner_epoch=2, sequence=3, command_id="command-a", plan_digest="a" * 64,
        issued_at_ms=1000, execute_before_ms=2000, hold_until_ms=3000,
        setpoint_w=-1200, mode="Aan", stop_at_ms=None,
    ) | changes))


def receipt(**changes):
    return contract().ActuationReceipt(**(dict(
        installation_id="synthetic", battery_name="accu", owner_epoch=2, sequence=3,
        command_id="command-a", plan_digest="a" * 64, accepted_ts=1501,
        status="accepted", helper_results=(), owner_session_id="session-a",
        payload_digest="b" * 64, reason="command_accepted", observed_at_ms=1502,
        effect="complete",
    ) | changes))


def assert_code(call):
    with pytest.raises(ValueError) as error:
        call()
    assert re.fullmatch(r"[a-z][a-z0-9_]*", str(error.value))


@pytest.mark.parametrize("outcomes,status", [
    ({}, "confirmed"), ({"mode": [False]}, "unknown"),
    ({"setpoint": [False], "mode": [False], "stop_inverter": [False]}, "unconfirmed"),
])
def test_legacy_receipt_keeps_exact_shape_identity_and_statuses(outcomes, status):
    result = actuator.execute_battery_command(
        ScriptedGateway(outcomes), make_plan([0.3], [0]), proposal(), T0 + 60.9,
    )
    assert type(result).__name__ == "LegacyActuationReceipt"
    assert type(result) is actuator.LegacyActuationReceipt
    assert [f.name for f in fields(result)] == [
        "command_id", "plan_identity", "timestamp", "battery_index", "battery_name",
        "setpoint_w", "status", "helpers",
    ]
    assert get_args(get_type_hints(type(result))["status"]) == ("confirmed", "unconfirmed", "unknown")
    assert result.status == status and result.status != "accepted"
    assert result.command_id == "v1:1700000000/b0/1700000060"
    assert result.plan_identity == "v1:1700000000"
    assert actuator.HelperResult is contract().HelperResult
    assert actuator.ActuationReceipt is contract().ActuationReceipt
    assert set(get_args(actuator.CommandReceipt)) == {actuator.LegacyActuationReceipt, contract().ActuationReceipt}


@pytest.mark.parametrize("status", ["accepted", "rejected", "", None, True, [], float("nan")])
def test_legacy_receipt_rejects_invalid_status_during_direct_construction(status):
    with pytest.raises(ValueError, match=r"^legacy_receipt_status_invalid$"):
        actuator.LegacyActuationReceipt(
            command_id="v1:1700000000/b0/1700000060", plan_identity="v1:1700000000",
            timestamp=T0 + 60.9, battery_index=0, battery_name="accu",
            setpoint_w=-1200, status=status, helpers=(),
        )


@pytest.mark.parametrize("status", ["accepted", "rejected", "", None, True, [], float("nan")])
def test_legacy_receipt_rejects_invalid_status_during_replace(status):
    value = actuator.LegacyActuationReceipt(
        command_id="v1:1700000000/b0/1700000060", plan_identity="v1:1700000000",
        timestamp=T0 + 60.9, battery_index=0, battery_name="accu",
        setpoint_w=-1200, status="confirmed", helpers=(),
    )
    with pytest.raises(ValueError, match=r"^legacy_receipt_status_invalid$"):
        replace(value, status=status)
    assert value.status == "confirmed"


@pytest.mark.parametrize("status", ["confirmed", "unconfirmed", "unknown"])
def test_legacy_receipt_accepts_all_three_valid_statuses(status):
    value = actuator.LegacyActuationReceipt(
        command_id="v1:1700000000/b0/1700000060", plan_identity="v1:1700000000",
        timestamp=T0 + 60.9, battery_index=0, battery_name="accu",
        setpoint_w=-1200, status=status, helpers=(),
    )
    assert value.status == status
    assert replace(value, status=status) == value


@pytest.mark.parametrize("status,effect,accepted_ts", [
    ("accepted", "complete", 0), ("rejected", "none", None),
    ("unconfirmed", "none", None), ("unconfirmed", "partial", None),
    ("unconfirmed", "unknown", None),
])
def test_canonical_receipt_valid_evidence(status, effect, accepted_ts):
    value = receipt(status=status, effect=effect, accepted_ts=accepted_ts)
    assert (value.status, value.effect, value.accepted_ts) == (status, effect, accepted_ts)
    with pytest.raises(FrozenInstanceError):
        value.status = "confirmed"
    assert not hasattr(value, "__dict__")


@pytest.mark.parametrize("changes", [
    {"accepted_ts": None}, {"accepted_ts": True}, {"accepted_ts": -1},
    {"accepted_ts": float("nan")}, {"effect": "partial"},
    {"status": "rejected", "effect": "none"},
    {"status": "unconfirmed", "effect": "unknown"},
    {"status": "unconfirmed", "effect": "complete", "accepted_ts": None},
    {"status": "rejected", "effect": "partial", "accepted_ts": None},
    {"status": "confirmed"}, {"status": "unknown"}, {"status": []},
    {"owner_epoch": True}, {"sequence": "3"}, {"sequence": 0},
    {"observed_at_ms": -1}, {"observed_at_ms": float("inf")},
    {"installation_id": " "}, {"battery_name": ""}, {"owner_session_id": None},
    {"command_id": ""}, {"plan_digest": "g" * 64}, {"payload_digest": "B" * 64},
    {"reason": "Bad reason"}, {"reason": "bad\n"}, {"reason": "échec"},
    {"helper_results": []}, {"helper_results": (object(),)},
])
def test_canonical_receipt_rejects_invalid_evidence_with_code(changes):
    assert_code(lambda: receipt(**changes))


def test_helpers_cannot_forge_canonical_acceptance_or_no_effect_rejection():
    helper_type = contract().HelperResult
    failed = helper_type("mode", "input_select.mode", False, "write_failed", 2)
    success = replace(failed, ok=True)
    assert_code(lambda: receipt(helper_results=(failed,)))
    assert_code(lambda: receipt(helper_results=(replace(success, ok=1),)))
    assert_code(lambda: receipt(status="rejected", accepted_ts=None, effect="none", helper_results=(success,)))
    assert receipt(helper_results=(success,)).helper_results == (success,)
    assert receipt(status="unconfirmed", accepted_ts=None, effect="partial", helper_results=(success, failed))


@pytest.mark.parametrize("field", ["owner_epoch", "sequence", "issued_at_ms", "execute_before_ms",
                                   "hold_until_ms", "setpoint_w", "stop_at_ms"])
@pytest.mark.parametrize("value", [True, 1.0, "1", float("nan"), float("inf")])
def test_command_ints_are_strict(field, value):
    assert_code(lambda: command(**{field: value}))


@pytest.mark.parametrize("changes", [
    {"owner_epoch": 0}, {"sequence": -1}, {"issued_at_ms": -1}, {"stop_at_ms": -1},
    {"issued_at_ms": 2000}, {"execute_before_ms": 3001},
    {"installation_id": " "}, {"battery_name": ""}, {"owner_session_id": "\t"},
    {"command_id": ""}, {"mode": ""}, {"mode": 1},
    {"plan_digest": "A" * 64}, {"plan_digest": "a" * 63},
])
def test_command_rejects_invalid_payload_and_time_order(changes):
    assert_code(lambda: command(**changes))


def test_command_roundtrip_is_strict_frozen_and_detached():
    value = command(execute_before_ms=3000, stop_at_ms=0)
    payload = value.to_dict()
    assert payload == asdict(value)
    assert type(value).from_dict(json.loads(json.dumps(payload))) == value
    payload["setpoint_w"] = 99
    assert value.setpoint_w == -1200
    with pytest.raises(FrozenInstanceError):
        value.setpoint_w = 99
    assert not hasattr(value, "__dict__")
    for invalid in (None, [], {}, value.to_dict() | {"unexpected": 1},
                    value.to_dict() | {"sequence": True}):
        assert_code(lambda invalid=invalid: type(value).from_dict(invalid))


def other_dtos():
    api = contract()
    return [
        api.RevokeRequest("synthetic", 2, "barrier-a", "operator_stop", 0),
        api.BarrierReceipt("synthetic", "barrier-a", 2, 3, "accepted", 1000,
                           True, "confirmed", "generation-a", "barrier_accepted"),
        api.Deadline(3000, 2000),
        api.ClockSample(1000, 2000, "boot-a", 0, True),
        api.OwnerBinding("synthetic", "receiver-a", "generation-a", "session-a", 2, "barrier-a", 3000),
    ]


def test_other_dtos_are_frozen_slots_and_deadline_uses_only_supplied_time():
    for value in other_dtos():
        assert not hasattr(value, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(value, fields(value)[0].name, None)
    deadline = other_dtos()[2]
    assert [deadline.remaining_ns(now) for now in (-1, 0, 2999, 3000, 3001)] == [3001, 3000, 1, 0, 0]
    for invalid in (True, "0", float("nan")):
        assert_code(lambda invalid=invalid: deadline.remaining_ns(invalid))


@pytest.mark.parametrize("index,changes", [
    (0, {"expected_epoch": True}), (0, {"expected_epoch": 0}),
    (0, {"requested_at_ms": -1}), (0, {"reason": ""}),
    (1, {"revoked_through_epoch": -1}), (1, {"current_epoch": 0}),
    (1, {"accepted_ts": None}), (1, {"accepted_ts": True}),
    (1, {"queues_fenced": False}), (1, {"queues_fenced": 1}),
    (1, {"safe_state": "unknown"}), (1, {"receiver_generation": ""}),
    (1, {"status": "confirmed"}), (1, {"reason": ""}),
    (2, {"monotonic_end_ns": True}), (2, {"execute_before_ms": -1}),
    (3, {"utc_ms": True}), (3, {"boottime_ns": -1}), (3, {"uncertainty_ms": float("nan")}),
    (3, {"boot_id": " "}), (3, {"healthy": 1}),
    (4, {"owner_epoch": 0}), (4, {"owner_not_after_ms": -1}), (4, {"barrier_id": ""}),
])
def test_other_dtos_reject_invalid_values_with_code(index, changes):
    assert_code(lambda: replace(other_dtos()[index], **changes))


@pytest.mark.parametrize("status", ["rejected", "unconfirmed"])
def test_nonaccepted_barrier_has_no_acceptance_timestamp(status):
    value = other_dtos()[1]
    assert_code(lambda: replace(value, status=status))
    assert replace(value, status=status, accepted_ts=None, queues_fenced=False, safe_state="unknown")


def test_plan_digest_covers_full_snapshot_deterministically_across_processes():
    api = contract()
    plan = ready_plan()
    expected = hashlib.sha256(json.dumps(plan.to_dict(), sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    assert api.plan_digest(plan) == api.plan_digest(plan) == expected
    script = ("import json,sys; from dao.prog.fastctrl.plan import FastPlan; "
              "from dao.prog.fastctrl.command import plan_digest; "
              "print(plan_digest(FastPlan.from_dict(json.load(sys.stdin))))")
    results = [subprocess.run([sys.executable, "-c", script], input=json.dumps(plan.to_dict()),
                              text=True, capture_output=True, check=True).stdout.strip() for _ in range(2)]
    assert results == [expected, expected]
    assert actuator.plan_identity(plan) == "v2:1798758900:" + "b" * 64


@pytest.mark.parametrize("part", ["profile", "setpoint", "context", "replay_hash"])
def test_plan_digest_distinguishes_content_not_just_timestamp(part):
    api = contract()
    plan = ready_plan()
    other = copy.deepcopy(plan)
    if part == "profile":
        step = other.intervals[0].batteries[0]
        other.intervals[0].batteries[0] = replace(step, on_power_w=1000.0, mean_ac_power_w=1000.0,
                                                ac_power_w=1000.0)
    elif part == "setpoint":
        other.specs[0] = replace(other.specs[0], max_charge_w=4000.0)
    elif part == "context":
        other.financial_context = replace(other.financial_context, not_after_ts=other.financial_context.not_after_ts - 1)
    else:
        other.financial_context = replace(other.financial_context, replay_basis_hash="c" * 64)
    assert plan.created_ts == other.created_ts
    assert api.plan_digest(plan) != api.plan_digest(other)
    if part != "replay_hash":
        assert actuator.plan_identity(plan) == actuator.plan_identity(other)


def test_digest_propagates_original_plan_validation_error_unchanged():
    api = contract()
    original = ValueError("synthetic original plan validation error")

    class InvalidPlan:
        def to_dict(self):
            raise original

    with pytest.raises(ValueError) as error:
        api.plan_digest(InvalidPlan())
    assert error.value is original
    assert str(error.value) == "synthetic original plan validation error"
    assert error.value.__cause__ is None


def test_digest_codes_nonserializable_snapshot_and_preserves_original_cause(monkeypatch):
    api = contract()
    snapshot = {"unsupported": object()}

    class InvalidSnapshotPlan:
        def to_dict(self):
            return snapshot

    with pytest.raises(TypeError) as original:
        json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    dumps = json.dumps
    causes = []

    def observe_serialization_error(*args, **kwargs):
        try:
            return dumps(*args, **kwargs)
        except TypeError as cause:
            causes.append(cause)
            raise

    with monkeypatch.context() as patcher:
        patcher.setattr(api.json, "dumps", observe_serialization_error)
        with pytest.raises(ValueError) as error:
            api.plan_digest(InvalidSnapshotPlan())
    assert str(error.value) == "plan_snapshot_invalid"
    assert len(causes) == 1
    assert error.value.__cause__ is causes[0]
    assert type(error.value.__cause__) is TypeError
    assert str(error.value.__cause__) == str(original.value)


def test_digest_propagates_plan_validation_and_codes_json_nonfinite_error():
    api = contract()
    plan = ready_plan()
    plan.financial_context = None
    with pytest.raises(ValueError):
        api.plan_digest(plan)
    legacy = make_plan([0.3], [0])
    legacy.price_average = float("nan")
    with pytest.raises(ValueError) as error:
        api.plan_digest(legacy)
    assert str(error.value) == "plan_snapshot_invalid"
    assert type(error.value.__cause__) is ValueError


def state(pending=True):
    value = BatteryControllerState(last_command_w=12.0, last_command_ts=0.5,
                                   held_setpoint_w=13.0, restore_pending=True)
    if pending:
        value.pending_command = command()
    return value


def test_no_pending_command_leaves_every_fact_untouched():
    value = state(False)
    before = copy.deepcopy(value)
    policy.apply_actuation_receipt(value, receipt())
    assert value == before


@pytest.mark.parametrize("changes", [
    {"installation_id": "other"}, {"battery_name": "other"}, {"owner_epoch": 3},
    {"sequence": 4}, {"command_id": "other"}, {"plan_digest": "c" * 64},
])
def test_each_receipt_identity_mismatch_preserves_all_state(changes):
    value = state()
    before = copy.deepcopy(value)
    policy.apply_actuation_receipt(value, receipt(**changes))
    assert value == before


@pytest.mark.parametrize("stop_at_ms,cleared", [(None, True), (0, False), (2500, False)])
def test_acceptance_applies_stored_pending_values_not_later_decision(stop_at_ms, cleared):
    value = state()
    value.pending_command = command(stop_at_ms=stop_at_ms)
    later_decision = proposal(setpoint_w=999, stop_inverter="2099-01-01 00:00:00")
    assert later_decision.setpoint_w != value.pending_command.setpoint_w
    before = value.to_dict()
    policy.apply_actuation_receipt(value, receipt())
    assert value.to_dict() == before | dict(
        last_command_w=-1200.0, last_command_ts=1.501, stop_inverter_cleared=cleared,
        held_setpoint_w=-1200.0, held_stop_inverter_cleared=cleared, pending_command=None,
    )
    assert value.restore_pending is True


@pytest.mark.parametrize("status,effect", [("rejected", "none"), ("unconfirmed", "partial")])
def test_nonacceptance_keeps_pending_command_and_confirmed_facts(status, effect):
    value = state()
    before = copy.deepcopy(value)
    policy.apply_actuation_receipt(value, receipt(status=status, effect=effect, accepted_ts=None))
    assert value == before


def test_legacy_receipt_cannot_update_canonical_pending_state():
    value = state()
    before = copy.deepcopy(value)
    legacy = actuator.execute_battery_command(ScriptedGateway(), make_plan([0.3], [0]), proposal(), T0)
    with pytest.raises(TypeError):
        policy.apply_actuation_receipt(value, legacy)
    assert value == before


def test_old_status_remains_readable_and_pending_command_roundtrips(tmp_path):
    old = BatteryControllerState.from_dict({"last_command_w": -123.0})
    assert getattr(old, "pending_command", "missing") is None
    assert old.last_command_w == -123.0
    value = policy.ControllerState(batteries=[state()])
    path = str(tmp_path / "state.json")
    save_state(value, path)
    assert load_state(path).to_dict() == value.to_dict()
    assert load_state(path).battery(0).pending_command == command()
    invalid = value.battery(0).to_dict()
    invalid["pending_command"]["sequence"] = True
    assert_code(lambda: BatteryControllerState.from_dict(invalid))


@pytest.mark.parametrize("pending", ["corrupt-command", {"command_id": "incomplete"}])
def test_load_state_recovers_fresh_state_from_corrupt_pending_command_file(tmp_path, pending):
    path = tmp_path / "corrupt-state.json"
    payload = policy.ControllerState(batteries=[state()], saved_today_eur=42.0,
                                     last_tick_ts=123.0, day_key="synthetic-old-state").to_dict()
    payload["batteries"][0]["pending_command"] = pending
    path.write_text(json.dumps(payload))
    restored = load_state(str(path))
    assert restored == policy.ControllerState()
    assert restored.batteries == []
    assert restored.battery(0).pending_command is None
    assert restored.battery(0).last_command_w == 0.0
    assert restored.saved_today_eur == restored.last_tick_ts == 0.0


@pytest.mark.parametrize("kind", ["write", "write_false", "missing_spec", "missing_helpers", "shadow"])
def test_runner_returns_tuple_and_preserves_last_receipts(tmp_path, kind):
    plan = make_plan([0.3], [0])
    battery = proposal(write=kind != "write_false", index=99 if kind == "missing_spec" else 0)
    if kind == "missing_helpers":
        plan.specs[0] = replace(plan.specs[0], setpoint_entity=None, mode_entity=None, stop_inverter_entity=None)
    runner = FastControlRunner(FakeHass(), make_config(), plan_path=str(tmp_path / "plan.json"),
                               state_path=str(tmp_path / "state.json"))
    gateway = ScriptedGateway()
    runner.gateway = gateway
    runner.last_receipts = ["previous"]
    result = runner._actuate(Decision(T0, [battery]), plan, "shadow" if kind == "shadow" else "active")
    assert isinstance(result, tuple), "_actuate must return receipts rather than implicit None"
    assert list(result) == runner.last_receipts
    if kind == "write":
        assert len(result) == 1 and type(result[0]).__name__ == "LegacyActuationReceipt"
        assert [call[0] for call in gateway.calls] == ["setpoint", "mode", "stop_inverter"]
    else:
        assert result == () and gateway.calls == []


@pytest.mark.parametrize("mode", ["active", "shadow", "off"])
def test_public_runner_ready_v2_is_still_hard_denied_without_helpercalls(tmp_path, monkeypatch, mode):
    from dao.prog.fastctrl import runner as runner_module

    clock = [15_000_000_000]
    monkeypatch.setattr(runner_module.time, "clock_gettime_ns", lambda _: clock[0])
    plan = ready_plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.to_dict()))
    hass, sink = FakeHass(), RecordingFallback()
    runner = FastControlRunner(hass, make_config(mode=mode), fallback=sink,
                               plan_path=str(path), state_path=str(tmp_path / "state.json"))
    assert runner.tick(plan.created_ts) is None
    assert [r.reason for r in sink.requests] == ["actuation_gates_unproven"]
    assert runner.last_receipts == []
    assert hass.values == hass.options == hass.published == hass.switches == {}
    assert hass.services == []
    assert hass.template_calls == hass.state_calls == 0

    # The real admitted read established financial cadence. Phase three must
    # defer before another publication lock/read or any fallback/sensor I/O.
    read = Mock(wraps=runner_module.read_publication)
    acquire = Mock(wraps=runner_module.publication_lock)
    monkeypatch.setattr(runner_module, "read_publication", read)
    monkeypatch.setattr(runner_module, "publication_lock", acquire)
    clock[0] = 18_000_000_000
    assert runner.tick(plan.created_ts) is None
    assert runner.last_gate.allowed is False
    assert runner.last_gate.reason == "admission_deferred"
    read.assert_not_called()
    acquire.assert_not_called()
    assert [r.reason for r in sink.requests] == ["actuation_gates_unproven"]
    assert runner.last_receipts == []
    assert hass.values == hass.options == hass.published == hass.switches == {}
    assert hass.services == []
    assert hass.template_calls == hass.state_calls == 0


def test_command_import_is_pure_without_policy_runner_plan_or_io_stack():
    api = contract()
    tree = ast.parse(Path(api.__file__).read_text())
    allowed = {"__future__", "dataclasses", "hashlib", "json", "typing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name in allowed for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed
    # Isolate the module: fastctrl/__init__.py already eagerly exports policy
    # and plan. That existing package facade is not command.py's dependency.
    result = subprocess.run([sys.executable, "-c", (
        "import sys,importlib.util; "
        f"spec=importlib.util.spec_from_file_location('synthetic_command', {api.__file__!r}); "
        "module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; "
        "spec.loader.exec_module(module); "
        "assert not any(x in sys.modules for x in "
        "('dao.prog.fastctrl.policy', 'dao.prog.fastctrl.actuator', 'dao.prog.fastctrl.plan', "
        "'dao.prog.fastctrl.runner', 'sqlalchemy', 'pandas', 'mip'))"
    )], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
