"""F2c2: deterministic last-writer races, using production transition rules."""

import ast
import copy
import hashlib
import inspect
import json
import re
from dataclasses import FrozenInstanceError, fields, replace
from importlib import import_module, util
from pathlib import Path

import pytest

from dao.prog.fastctrl.command import BatteryCommand, Deadline, HelperResult, RevokeRequest


def receiver():
    # Deliberate assertion RED, never a collection/fixture import error.
    assert util.find_spec("dao.prog.fastctrl.receiver") is not None, "F2c2 receiver module missing"
    return import_module("dao.prog.fastctrl.receiver")


def state(**changes):
    return receiver().ReceiverState(**(dict(
        schema_version=1, installation_id="sandbox", receiver_id="receiver-a",
        receiver_generation="generation-a", anchor_checkpoint="checkpoint-40",
        epoch_high_water=7, sequence_high_water=40, bound_owner_session_id="owner-a",
        bound_owner_epoch=7, owner_not_after_ms=9000, commands=(), inhibited=False,
        queue_known=True, pending_barrier_id=None, last_barrier=None, queue_generation="queue-a",
    ) | changes))


def command(**changes):
    return BatteryCommand(**(dict(
        installation_id="sandbox", battery_name="battery-a", owner_session_id="owner-a",
        owner_epoch=7, sequence=41, command_id="command-a", plan_digest="a" * 64,
        issued_at_ms=1000, execute_before_ms=2000, hold_until_ms=3000,
        setpoint_w=1200, mode="charge", stop_at_ms=None,
    ) | changes))


DEADLINE = Deadline(100_000_000_000, 10000)


def fake(tmp_path, **changes):
    initial = state(**changes)
    module = import_module("dao.tests.prog.fakes.fenced_receiver")
    return module.FencedReceiver(tmp_path, initial_state=initial, utc_ms=1001)


def query(gateway, cmd, **changes):
    return gateway.query(**(dict(
        installation_id=cmd.installation_id, owner_epoch=cmd.owner_epoch,
        sequence=cmd.sequence, command_id=cmd.command_id,
        payload_digest=receiver().payload_digest(cmd), deadline=DEADLINE,
    ) | changes))


def revoke(gateway, **changes):
    request = RevokeRequest(**(dict(
        installation_id="sandbox", expected_epoch=7, barrier_id="barrier-a",
        reason="operator_stop", requested_at_ms=gateway.utc_ms,
    ) | changes))
    return gateway.revoke_and_stop(request, deadline=DEADLINE)


def bind(gateway, **changes):
    return gateway.bind_owner(**(dict(
        installation_id="sandbox", barrier_id="barrier-a", expected_epoch=8,
        owner_session_id="owner-b", requested_not_after_ms=12000, deadline=DEADLINE,
    ) | changes))


def reserve(current, cmd=None, **changes):
    return receiver().classify_send(current, cmd or command(), **(dict(
        receiver_now_upper_ms=1001, clock_healthy=True, clock_uncertainty_ms=0,
    ) | changes))


def test_01_ordered_old_write_after_barrier_cannot_effect_new_owner_can(tmp_path):
    gateway = fake(tmp_path)
    old = command()
    gateway.enqueue(old)
    assert gateway.deliver().status == "unconfirmed"
    gateway.advance_clock(1000)
    barrier = revoke(gateway)
    assert barrier.status == "accepted"
    assert bind(gateway).owner_epoch == 8
    before = list(gateway.effect_log)
    late = gateway.step_helper("setpoint", command=old)
    assert (late.status, late.reason, late.effect) == ("rejected", "epoch_withdrawn", "none")
    assert gateway.effect_log == before
    fresh = command(owner_session_id="owner-b", owner_epoch=8, sequence=42,
                    command_id="command-b", issued_at_ms=2001, execute_before_ms=4000,
                    hold_until_ms=5000)
    assert gateway.send(fresh, deadline=DEADLINE).status == "accepted"
    assert [item[1] for item in gateway.effect_log] == ["setpoint", "mode", "stop_inverter"]


def test_02_retry_after_timeout_never_has_second_effect(tmp_path):
    gateway = fake(tmp_path)
    cmd = command()
    original = gateway.send(cmd, deadline=DEADLINE)
    assert original.status == "accepted"
    gateway.advance_clock(1000)
    before = list(gateway.effect_log)
    decision = receiver().classify_send(gateway.state, cmd, receiver_now_upper_ms=gateway.receiver_now_upper_ms,
                                        clock_healthy=True, clock_uncertainty_ms=0)
    assert decision.status == "rejected"
    assert decision.reason in ("duplicate_command", "stale_or_duplicate_sequence")
    result = gateway.send(cmd, deadline=DEADLINE)
    assert result == original and gateway.effect_log == before  # historical ACK, not a new effect
    assert query(gateway, cmd).status == "accepted"  # retry cannot erase original proof


def test_03_crash_between_helpers_is_unconfirmed_never_replayed(tmp_path):
    gateway = fake(tmp_path)
    cmd = command()
    gateway.enqueue(cmd)
    gateway.deliver()
    partial = gateway.step_helper("setpoint")
    assert (partial.status, partial.effect, partial.accepted_ts) == ("unconfirmed", "partial", None)
    gateway.crash(persist=True)
    gateway.restart()
    result = query(gateway, cmd)
    assert result.status == "unconfirmed" and result.accepted_ts is None
    assert [r.helper for r in result.helper_results] == ["setpoint"]
    assert len(gateway.effect_log) == 1
    assert gateway.send(cmd, deadline=DEADLINE) == result
    assert len(gateway.effect_log) == 1


def test_04_crash_before_reservation_does_not_consume_sequence(tmp_path):
    gateway = fake(tmp_path)
    gateway.enqueue(command())
    gateway.crash(persist=True)
    gateway.restart()
    assert query(gateway, command()) is None
    fresh = command(command_id="new-command")
    assert gateway.send(fresh, deadline=DEADLINE).status == "accepted"


def test_05_crash_after_reserve_is_pending_not_silently_executed(tmp_path):
    gateway = fake(tmp_path)
    gateway.enqueue(command())
    gateway.deliver()
    gateway.crash(persist=True)
    gateway.restart()
    result = query(gateway, command())
    assert (result.status, result.reason) == ("unconfirmed", "pending_effect")
    assert gateway.state.commands[0].phase == "reserved"
    assert gateway.effect_log == []


def test_06_effect_without_ack_is_durable_unknown_and_query_has_no_effect(tmp_path):
    gateway = fake(tmp_path)
    gateway.lose_ack()
    result = gateway.send(command(), deadline=DEADLINE)
    assert (result.status, result.effect, result.accepted_ts) == ("unconfirmed", "unknown", None)
    gateway.crash(persist=True)
    gateway.restart()
    assert query(gateway, command()) == result
    assert len(gateway.effect_log) == 3


def test_07_restart_keeps_both_high_waters(tmp_path):
    gateway = fake(tmp_path)
    gateway.send(command(), deadline=DEADLINE)
    gateway.crash(persist=True)
    gateway.restart()
    assert (gateway.state.epoch_high_water, gateway.state.sequence_high_water) == (7, 41)
    assert gateway.send(command(command_id="old-sequence"), deadline=DEADLINE).reason == "stale_or_duplicate_sequence"


def test_08_file_restore_does_not_restore_independent_anchor(tmp_path):
    gateway = fake(tmp_path)
    saved = gateway.snapshot_files()
    gateway.send(command(), deadline=DEADLINE)
    revoke(gateway)
    anchor = gateway.anchor_path.read_bytes()
    gateway.restore_files(saved)
    assert gateway.anchor_path.read_bytes() == anchor
    reason = gateway.restart()
    assert reason in ("rolled_back_state", "anchor_checkpoint_mismatch")
    assert gateway.state.inhibited and not gateway.state.queue_known
    assert (gateway.state.epoch_high_water, gateway.state.sequence_high_water) == (8, 41)
    assert gateway.send(command(sequence=42), deadline=DEADLINE).reason == "receiver_inhibited"


@pytest.mark.parametrize("corrupt", [False, True])
def test_09_missing_or_corrupt_state_uses_anchor_not_defaults(tmp_path, corrupt):
    gateway = fake(tmp_path)
    gateway.send(command(), deadline=DEADLINE)
    if corrupt:
        gateway.state_path.write_text('{"broken":true}')
    else:
        gateway.state_path.unlink()
    assert gateway.restart() == "state_missing_or_corrupt"
    assert gateway.state.inhibited and not gateway.state.queue_known
    assert (gateway.state.epoch_high_water, gateway.state.sequence_high_water) == (7, 41)


def test_10_transport_receipt_is_not_acceptance(tmp_path):
    gateway = fake(tmp_path)
    assert gateway.enqueue(command()) is None
    assert query(gateway, command()) is None
    pending = gateway.deliver()
    assert (pending.status, pending.accepted_ts, pending.effect) == ("unconfirmed", None, "none")
    assert gateway.effect_log == []


def test_11_low_sequence_rejected(tmp_path):
    gateway = fake(tmp_path)
    assert gateway.send(command(sequence=40), deadline=DEADLINE).reason == "stale_or_duplicate_sequence"
    assert gateway.effect_log == []


def test_12_changed_query_payload_cannot_certify_command(tmp_path):
    gateway = fake(tmp_path)
    original = gateway.send(command(), deadline=DEADLINE)
    altered = command(setpoint_w=1234)
    result = query(gateway, altered)
    assert (result.status, result.reason, result.effect) == ("accepted", "identity_mismatch", "complete")
    assert result.helper_results == original.helper_results and len(result.helper_results) == 3
    assert (result.command_id, result.owner_epoch, result.sequence, result.plan_digest) == (
        "command-a", 7, 41, "a" * 64)
    assert result.payload_digest == original.payload_digest
    assert result.payload_digest != receiver().payload_digest(altered)
    assert len(gateway.effect_log) == 3


@pytest.mark.parametrize("changes,reason", [
    ({"installation_id": "elsewhere"}, "wrong_installation"),
    ({"owner_session_id": "intruder"}, "wrong_owner_session"),
    ({"owner_epoch": 6}, "stale_epoch"),
    ({"owner_epoch": 8}, "epoch_not_granted"),
])
def test_13_14_16_wrong_installation_session_or_epoch(tmp_path, changes, reason):
    gateway = fake(tmp_path)
    result = gateway.send(command(**changes), deadline=DEADLINE)
    assert (result.status, result.reason, result.effect) == ("rejected", reason, "none")
    assert gateway.effect_log == []


def test_15_expired_command_never_effects(tmp_path):
    gateway = fake(tmp_path)
    gateway.advance_clock(999)
    assert gateway.send(command(), deadline=DEADLINE).reason == "command_expired"
    assert gateway.effect_log == []


def test_17_query_exact_ack_after_restart_and_rebind_without_resend(tmp_path):
    gateway = fake(tmp_path)
    cmd = command()
    original = gateway.send(cmd, deadline=DEADLINE)
    gateway.advance_clock(100)
    revoke(gateway)
    bind(gateway)
    gateway.crash(persist=True)
    gateway.restart()
    before = list(gateway.effect_log)
    assert query(gateway, cmd) == original
    assert query(gateway, cmd, command_id="absent") is None
    for change in ({"owner_epoch": 8}, {"sequence": 42}, {"payload_digest": "b" * 64}):
        mismatch = query(gateway, cmd, **change)
        assert mismatch == replace(original, reason="identity_mismatch")
        assert (mismatch.owner_epoch, mismatch.sequence, mismatch.owner_session_id) == (7, 41, "owner-a")
        assert mismatch.observed_at_ms == original.observed_at_ms
    assert query(gateway, cmd, installation_id="wrong") == replace(original, reason="wrong_installation")
    assert gateway.effect_log == before


@pytest.mark.parametrize("healthy,uncertainty,reason", [
    (False, 0, "clock_unhealthy"), (True, None, "clock_uncertainty_unknown"),
    (True, -1, "clock_uncertainty_unknown"),
])
def test_18_clock_failure_fails_closed(tmp_path, healthy, uncertainty, reason):
    gateway = fake(tmp_path)
    gateway.clock_healthy, gateway.clock_uncertainty_ms = healthy, uncertainty
    assert gateway.send(command(), deadline=DEADLINE).reason == reason
    assert gateway.effect_log == []


def test_19_lease_boundary_is_conservative(tmp_path):
    gateway = fake(tmp_path, owner_not_after_ms=1000)
    assert gateway.send(command(), deadline=DEADLINE).reason == "owner_lease_expired"
    assert gateway.effect_log == []


def test_20_unknown_safe_state_withdraws_epoch_but_cannot_bind(tmp_path):
    gateway = fake(tmp_path)
    gateway.queues_drained = gateway.safe_state_confirmed = False
    barrier = revoke(gateway)
    assert (barrier.status, barrier.safe_state, barrier.queues_fenced, barrier.accepted_ts) == (
        "unconfirmed", "unknown", False, None)
    assert gateway.state.epoch_high_water == 8 and gateway.state.inhibited
    with pytest.raises(ValueError, match="^no_confirmed_barrier$"):
        bind(gateway)


def test_21_barrier_fences_every_battery_queue(tmp_path):
    gateway = fake(tmp_path)
    first = command()
    second = command(battery_name="battery-b", sequence=42, command_id="command-b")
    for cmd in (first, second):
        gateway.enqueue(cmd)
        gateway.deliver()
    assert revoke(gateway).queues_fenced
    assert not any(record.phase == "reserved" for record in gateway.state.commands)
    for cmd in (first, second):
        assert gateway.step_helper("setpoint", command=cmd).reason == "epoch_withdrawn"
    assert gateway.effect_log == []


def test_22_higher_sequence_invalidates_waiting_lower_command(tmp_path):
    gateway = fake(tmp_path)
    gateway.enqueue(command())
    gateway.deliver()
    gateway.enqueue(command(sequence=42, command_id="command-b"))
    gateway.deliver()
    low = gateway.state.commands[0]
    assert (low.phase, low.reason) == ("invalidated", "superseded_by_higher_sequence")
    assert gateway.step_helper("setpoint", command=command()).reason == "no_longer_reserved"
    assert gateway.effect_log == []


def test_23_ack_names_exact_command_and_barrier(tmp_path):
    gateway = fake(tmp_path)
    ack = gateway.send(command(), deadline=DEADLINE)
    assert (ack.command_id, ack.owner_epoch, ack.sequence) == ("command-a", 7, 41)
    assert ack.accepted_ts == 1001 and all(result.ok for result in ack.helper_results)
    barrier = revoke(gateway)
    assert (barrier.barrier_id, barrier.revoked_through_epoch, barrier.current_epoch) == ("barrier-a", 7, 8)
    assert barrier.receiver_generation == "generation-a" and barrier.accepted_ts == 1001


def test_24_receiver_is_pure_and_only_fake_implements_gateway():
    module = receiver()
    source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in imports:
        if isinstance(node, ast.ImportFrom):
            assert node.module in {"__future__", "dataclasses", "hashlib", "json", "typing", "command"}
        else:
            assert all(alias.name in {"dataclasses", "hashlib", "json", "typing"} for alias in node.names)
    methods = {"inspect", "revoke_and_stop", "bind_owner", "send", "query"}
    implementations = []
    for path in Path("dao").rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and methods <= {
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            }:
                implementations.append((str(path), node.name))
    assert sorted(implementations) == [
        ("dao/prog/fastctrl/receiver.py", "FencedDeviceGateway"),
        ("dao/tests/prog/fakes/fenced_receiver.py", "FencedReceiver"),
    ]


@pytest.mark.parametrize("changes,command_changes,kwargs,reason", [
    ({"inhibited": True, "queue_known": False}, {"installation_id": "wrong"}, {}, "receiver_inhibited"),
    ({"queue_known": False}, {"installation_id": "wrong"}, {}, "queue_unknown"),
    ({}, {"installation_id": "wrong", "owner_session_id": "wrong"}, {}, "wrong_installation"),
    ({"bound_owner_session_id": None}, {"owner_epoch": 6}, {}, "wrong_owner_session"),
    ({}, {"owner_epoch": 6, "sequence": 1}, {}, "stale_epoch"),
    ({}, {"owner_epoch": 8, "sequence": 1}, {}, "epoch_not_granted"),
    ({}, {"sequence": 40}, {"clock_healthy": False}, "stale_or_duplicate_sequence"),
    ({}, {}, {"receiver_now_upper_ms": 2000, "clock_healthy": False}, "command_expired"),
    ({"owner_not_after_ms": 999}, {}, {"clock_healthy": False}, "owner_lease_expired"),
    ({}, {}, {"clock_healthy": False, "clock_uncertainty_ms": None}, "clock_unhealthy"),
    ({}, {}, {"clock_uncertainty_ms": None}, "clock_uncertainty_unknown"),
])
def test_send_refusal_precedence_and_canonical_rejected_records(changes, command_changes, kwargs, reason):
    original = state(**changes)
    decision = reserve(original, command(**command_changes), **kwargs)
    assert (decision.status, decision.reason, decision.record.phase) == ("rejected", reason, "rejected")
    assert decision.state.epoch_high_water == original.epoch_high_water
    assert decision.state.sequence_high_water == original.sequence_high_water
    assert original.commands == ()


def test_send_id_collision_and_rejection_retention():
    module = receiver()
    rejected = reserve(state(), receiver_now_upper_ms=2000)
    retry = reserve(rejected.state)
    assert retry.reason == "duplicate_command" and retry.state == rejected.state
    changed = reserve(rejected.state, command(setpoint_w=999))
    assert changed.reason == "payload_mismatch" and changed.state == rejected.state
    assert module.classify_query(rejected.state, installation_id="sandbox", owner_epoch=7,
                                 sequence=41, command_id="command-a",
                                 payload_digest=module.payload_digest(command())).reason == "command_expired"


def test_effect_fence_precedence_at_final_writer():
    module = receiver()
    reserved = reserve(state()).state
    assert module.fence_effect(replace(reserved, epoch_high_water=8), command(), receiver_now_upper_ms=3000,
                               clock_healthy=True, clock_uncertainty_ms=0) == (False, "epoch_withdrawn")
    assert module.fence_effect(reserved, command(), receiver_now_upper_ms=2000,
                               clock_healthy=True, clock_uncertainty_ms=0) == (False, "command_expired")
    assert module.fence_effect(state(), command(), receiver_now_upper_ms=1001,
                               clock_healthy=True, clock_uncertainty_ms=0) == (False, "no_longer_reserved")
    assert module.fence_effect(reserved, command(), receiver_now_upper_ms=1999,
                               clock_healthy=True, clock_uncertainty_ms=0)[0]


@pytest.mark.parametrize("outcome,status,phase,effect,accepted", [
    ("complete", "accepted", "accepted", "complete", 1002),
    ("partial", "unconfirmed", "unknown", "partial", None),
    ("unknown", "unconfirmed", "unknown", "unknown", None),
    ("none", "rejected", "rejected", "none", None),
])
def test_effect_translation(outcome, status, phase, effect, accepted):
    decision = receiver().classify_effect(reserve(state()).state, command(), receiver_now_upper_ms=1002, outcome=outcome,
                                          clock_healthy=True, clock_uncertainty_ms=0)
    assert (decision.status, decision.record.phase, decision.receipt.effect, decision.receipt.accepted_ts) == (status, phase, effect, accepted)
    assert decision.receipt.observed_at_ms == 1002


@pytest.mark.parametrize("phase,status,effect,reason", [
    ("reserved", "unconfirmed", "none", "pending_effect"),
    ("invalidated", "rejected", "none", "invalidated"),
    ("unknown", "unconfirmed", "none", "effect_unknown"),
    ("rejected", "rejected", "none", "command_expired"),
])
def test_query_translation(phase, status, effect, reason):
    module = receiver()
    record = replace(reserve(state()).record, phase=phase, reason=reason)
    result = module.classify_query(state(commands=(record,)), installation_id="sandbox", owner_epoch=7,
                                   sequence=41, command_id="command-a", payload_digest=record.payload_digest)
    assert (result.status, result.effect, result.reason) == (status, effect, reason)
    assert result.observed_at_ms == record.observed_at_ms


def test_revoke_mismatch_is_immutable_and_bind_caps_lease(tmp_path):
    gateway = fake(tmp_path)
    before = gateway.state
    assert revoke(gateway, expected_epoch=6).reason == "epoch_mismatch"
    assert gateway.state == before
    revoke(gateway)
    binding = bind(gateway, requested_not_after_ms=100000)
    assert binding.owner_not_after_ms == gateway.utc_ms + gateway.receiver_max_lease_ms
    assert gateway.state.bound_owner_epoch == 8 and not gateway.state.inhibited


@pytest.mark.parametrize("changes,reason", [
    ({"installation_id": "wrong", "barrier_id": "wrong"}, "wrong_installation"),
    ({"barrier_id": "wrong", "expected_epoch": 9}, "barrier_mismatch"),
    ({"expected_epoch": 9, "owner_session_id": ""}, "epoch_mismatch"),
    ({"owner_session_id": ""}, "invalid_owner_session"),
])
def test_bind_refusal_precedence(tmp_path, changes, reason):
    gateway = fake(tmp_path)
    revoke(gateway)
    before = gateway.state
    with pytest.raises(ValueError, match=f"^{reason}$"):
        bind(gateway, **changes)
    assert gateway.state == before


def test_recovery_checks_checkpoint_then_high_waters_and_keeps_unknown_queue():
    module = receiver()
    blocked, reason = module.receiver_recovery_state(state(), anchor_epoch_high_water=9,
        anchor_sequence_high_water=80, anchor_checkpoint="checkpoint-40")
    assert reason == "rolled_back_state"
    assert (blocked.epoch_high_water, blocked.sequence_high_water) == (9, 80)
    assert blocked.inhibited and not blocked.queue_known and blocked.bound_owner_session_id is None
    loaded = state(queue_known=False)
    recovered, _ = module.receiver_recovery_state(loaded, anchor_epoch_high_water=7,
        anchor_sequence_high_water=40, anchor_checkpoint=loaded.anchor_checkpoint)
    assert recovered == loaded


def test_payload_digest_exact_canonical_payload_and_not_plan_digest():
    cmd = command()
    payload = cmd.to_dict()
    del payload["plan_digest"]
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert receiver().payload_digest(cmd) == expected
    assert receiver().payload_digest(replace(cmd, plan_digest="b" * 64)) == expected
    for field, value in (("setpoint_w", 1), ("mode", "discharge"), ("battery_name", "battery-b")):
        assert receiver().payload_digest(replace(cmd, **{field: value})) != expected


def assert_code(call):
    with pytest.raises(ValueError) as error:
        call()
    assert re.fullmatch(r"[a-z][a-z0-9_]*", str(error.value))


def test_state_record_snapshot_frozen_strict_json_roundtrip():
    module = receiver()
    reserved = reserve(state()).state
    revoked = module.classify_revoke(reserved, RevokeRequest("sandbox", 7, "barrier-a", "stop", 1002),
                                     receiver_now_upper_ms=1002, queues_drained=True, safe_state_confirmed=True).state
    for obj in (reserved, revoked, reserved.commands[0], module.receiver_snapshot(revoked)):
        assert not hasattr(obj, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(obj, fields(obj)[0].name, "changed")
        data = obj.to_dict()
        assert type(obj).from_dict(json.loads(json.dumps(data))) == obj
        for field in fields(obj):
            missing = copy.deepcopy(data)
            del missing[field.name]
            assert_code(lambda obj=obj, missing=missing: type(obj).from_dict(missing))
        assert_code(lambda obj=obj, data=data: type(obj).from_dict(data | {"extra": 1}))
        assert_code(lambda obj=obj: type(obj).from_dict([]))


@pytest.mark.parametrize("field", ["schema_version", "epoch_high_water", "sequence_high_water",
                                    "bound_owner_epoch", "owner_not_after_ms"])
@pytest.mark.parametrize("value", [True, "1", 1.0, -1, None])
def test_state_integer_fields_strict(field, value):
    assert_code(lambda: state(**{field: value}))


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"inhibited": 1}, {"queue_known": 0}, {"commands": []},
    {"commands": (object(),)}, {"installation_id": ""}, {"receiver_id": " "},
    {"receiver_generation": None}, {"anchor_checkpoint": ""}, {"queue_generation": 1},
    {"bound_owner_session_id": ""}, {"pending_barrier_id": ""}, {"last_barrier": {}},
])
def test_state_invalid_schema_values(changes):
    assert_code(lambda: state(**changes))


@pytest.mark.parametrize("changes", [
    {"command_id": ""}, {"battery_name": ""}, {"owner_session_id": None},
    {"plan_digest": "A" * 64}, {"payload_digest": "g" * 64}, {"phase": "confirmed"},
    {"reason": "Bad reason"}, {"reason": "reason\n"}, {"accepted_ts": 1001},
    {"phase": "accepted"}, {"effect": "invalid"}, {"helper_results": []},
    {"helper_results": (HelperResult("setpoint", "dummy", 1, None, 1),)},
    {"observed_at_ms": True}, {"issued_at_ms": 1.0}, {"execute_before_ms": "2000"},
    {"hold_until_ms": -1}, {"owner_epoch": True}, {"sequence": False},
])
def test_record_invalid_values(changes):
    assert_code(lambda: replace(reserve(state()).record, **changes))


def test_nested_serialization_rejects_unknown_missing_and_bad_helper_fields():
    original = reserve(state()).state
    helper = HelperResult("setpoint", "dummy.setpoint", True, None, 1)
    original = replace(original, commands=(replace(original.commands[0], helper_results=(helper,)),))
    data = original.to_dict()
    mutations = []
    for path in ("record", "helper"):
        for mode in ("extra", "missing", "boolean_int"):
            mutated = copy.deepcopy(data)
            target = mutated["commands"][0]
            if path == "helper":
                target = target["helper_results"][0]
            if mode == "extra":
                target["extra"] = 1
            elif mode == "missing":
                target.pop(next(iter(target)))
            else:
                target["attempts" if path == "helper" else "sequence"] = True
            mutations.append(mutated)
    for mutated in mutations:
        assert_code(lambda mutated=mutated: type(original).from_dict(mutated))


def test_clock_uncertainty_applies_at_delivery_and_again_at_helper(tmp_path):
    gateway = fake(tmp_path)
    gateway.enqueue(command())
    gateway.deliver()
    gateway.clock_uncertainty_ms = 999
    result = gateway.step_helper("setpoint")
    assert result.reason == "command_expired" and gateway.effect_log == []


def test_crash_without_extra_persistence_keeps_durable_reservation(tmp_path):
    gateway = fake(tmp_path)
    gateway.enqueue(command())
    gateway.deliver()
    saved = gateway.snapshot_files()
    gateway.crash(persist=False)
    assert gateway.state_path.is_file(), "a crash must not delete an already durable reservation"
    assert gateway.snapshot_files() == saved
    gateway.restart()
    assert query(gateway, command()).status == "unconfirmed"
    assert gateway.state.sequence_high_water == 41 and gateway.effect_log == []


def test_backward_clock_is_unhealthy(tmp_path):
    gateway = fake(tmp_path)
    gateway.advance_clock(-1)
    assert gateway.send(command(), deadline=DEADLINE).reason == "clock_unhealthy"


def test_send_does_not_consume_some_other_queued_command(tmp_path):
    gateway = fake(tmp_path)
    waiting = command()
    gateway.enqueue(waiting)
    fresh = command(command_id="fresh", sequence=42)
    assert gateway.send(fresh, deadline=DEADLINE).status == "accepted"
    assert gateway.queue == [waiting]
    assert {item[0] for item in gateway.effect_log} == {"fresh"}


def test_helper_failure_returns_partial_even_with_other_work_waiting(tmp_path):
    gateway = fake(tmp_path)
    gateway.enqueue(command())
    gateway.deliver()
    gateway.helper_ok["mode"] = False
    fresh = command(command_id="fresh", sequence=42)
    result = gateway.send(fresh, deadline=DEADLINE)
    assert (result.status, result.effect) == ("unconfirmed", "partial")
    assert [(r.helper, r.ok) for r in result.helper_results] == [("setpoint", True), ("mode", False)]
    assert gateway.effect_log == [("fresh", "setpoint", "battery-a", 1200)]


def test_wrong_helper_order_has_no_state_or_device_effect(tmp_path):
    gateway = fake(tmp_path)
    gateway.enqueue(command())
    gateway.deliver()
    before = gateway.state
    with pytest.raises(ValueError, match="^receiver_helper_order_invalid$"):
        gateway.step_helper("mode")
    assert gateway.state == before and gateway.effect_log == []


@pytest.mark.parametrize("outcome", ["partial", "complete"])
def test_failed_helpers_cannot_be_accepted(outcome):
    current = reserve(state()).state
    failed = HelperResult("mode", "dummy.mode", False, "write_failed", 1)
    current = replace(current, commands=(replace(current.commands[0], helper_results=(failed,)),))
    if outcome == "complete":
        assert_code(lambda: receiver().classify_effect(current, command(), receiver_now_upper_ms=1002, outcome=outcome,
                                                       clock_healthy=True, clock_uncertainty_ms=0))
    else:
        result = receiver().classify_effect(current, command(), receiver_now_upper_ms=1002, outcome=outcome,
                                            clock_healthy=True, clock_uncertainty_ms=0)
        assert (result.status, result.receipt.effect) == ("unconfirmed", "partial")


def test_inspection_is_not_binding_or_permission(tmp_path):
    gateway = fake(tmp_path, inhibited=True, queue_known=False)
    before = gateway.state
    snapshot = gateway.inspect(deadline=DEADLINE)
    assert snapshot.inhibited and not snapshot.queue_known
    assert snapshot.bound_owner_not_after_ms == 9000
    assert gateway.state == before
    assert gateway.send(command(), deadline=DEADLINE).reason == "receiver_inhibited"


@pytest.mark.parametrize("field", ["owner_epoch", "sequence"])
def test_query_bool_is_not_integer_identity(field):
    module = receiver()
    cmd = command(owner_epoch=1, sequence=1)
    current = state(epoch_high_water=1, sequence_high_water=0, bound_owner_epoch=1)
    reserved = reserve(current, cmd).state
    accepted = module.classify_effect(reserved, cmd, receiver_now_upper_ms=1002, outcome="complete",
                                      clock_healthy=True, clock_uncertainty_ms=0).state
    args = dict(installation_id="sandbox", owner_epoch=1, sequence=1,
                command_id=cmd.command_id, payload_digest=module.payload_digest(cmd))
    result = module.classify_query(accepted, **(args | {field: True}))
    assert (result.status, result.reason) == ("accepted", "identity_mismatch")


def test_refused_effect_for_other_payload_preserves_reserved_identity():
    module = receiver()
    current = reserve(state()).state
    altered = command(setpoint_w=999)
    refused = module.classify_effect(current, altered, receiver_now_upper_ms=1002, outcome="complete",
                                     clock_healthy=True, clock_uncertainty_ms=0)
    assert (refused.status, refused.reason) == ("rejected", "no_longer_reserved")
    assert refused.state == current


def test_revoke_epoch_mismatch_has_first_priority():
    request = RevokeRequest("elsewhere", 6, "barrier", "stop", 1001)
    decision = receiver().classify_revoke(state(), request, receiver_now_upper_ms=1001,
                                         queues_drained=True, safe_state_confirmed=True)
    assert decision.reason == "epoch_mismatch" and decision.state == state()


@pytest.mark.parametrize("mode_failed", [False, True])
def test_25_partial_write_then_revoke_preserves_effect_and_helpers(tmp_path, mode_failed):
    gateway = fake(tmp_path)
    cmd = command()
    gateway.enqueue(cmd)
    gateway.deliver()
    original = gateway.step_helper("setpoint")
    if mode_failed:
        gateway.helper_ok["mode"] = False
        original = gateway.step_helper("mode")
    saved = gateway.state.commands[0]
    assert (original.status, original.effect) == ("unconfirmed", "partial")
    before = list(gateway.effect_log)
    revoke(gateway)
    stored = gateway.state.commands[0]
    assert stored.phase == ("unknown" if mode_failed else "invalidated")
    assert stored.effect == "partial" and stored.helper_results == saved.helper_results
    result = query(gateway, cmd)
    assert (result.status, result.effect, result.accepted_ts) == ("unconfirmed", "partial", None)
    assert result.helper_results == saved.helper_results
    assert result.observed_at_ms == saved.observed_at_ms
    if mode_failed:
        assert result == original
    else:
        assert result.reason == "invalidated"
    gateway.crash(persist=True)
    gateway.restart()
    assert query(gateway, cmd) == result
    assert gateway.effect_log == before


@pytest.mark.parametrize("refusal", ["epoch_withdrawn", "command_expired", "no_longer_reserved"])
def test_26_fenced_execution_after_partial_retains_stored_evidence(tmp_path, refusal):
    gateway = fake(tmp_path)
    cmd = command()
    gateway.enqueue(cmd)
    gateway.deliver()
    gateway.step_helper("setpoint")
    if refusal == "epoch_withdrawn":
        revoke(gateway)
    elif refusal == "command_expired":
        gateway.advance_clock(999)
    else:
        gateway.enqueue(command(sequence=42, command_id="higher"))
        gateway.deliver()
    stored = gateway.state.commands[0]
    before = list(gateway.effect_log)
    result = gateway.step_helper("mode", command=cmd)
    assert (result.status, result.effect, result.reason) == ("unconfirmed", "partial", refusal)
    assert result.helper_results == stored.helper_results
    assert result.observed_at_ms == stored.observed_at_ms and result.accepted_ts is None
    assert gateway.state.commands[0].effect == "partial"
    assert gateway.state.commands[0].helper_results == stored.helper_results
    assert query(gateway, cmd) == result
    assert gateway.effect_log == before


def test_27_epoch_zero_denies_revoke_bind_and_nonzero_without_bootstrap(tmp_path):
    gateway = fake(tmp_path, epoch_high_water=0, sequence_high_water=0, bound_owner_epoch=0,
                   bound_owner_session_id=None, owner_not_after_ms=0, inhibited=True, queue_known=False)
    before = gateway.state
    # F2c1 cannot represent current_epoch=0 in a BarrierReceipt. Fail closed
    # with the prescribed code, not a fabricated epoch-1 receipt.
    with pytest.raises(ValueError, match="^epoch_mismatch$"):
        revoke(gateway, expected_epoch=1)
    assert gateway.state == before
    with pytest.raises(ValueError, match="^no_confirmed_barrier$"):
        bind(gateway, expected_epoch=0)
    assert gateway.state == before
    denied = gateway.send(command(owner_epoch=1, sequence=1), deadline=DEADLINE)
    assert (denied.status, denied.effect) == ("rejected", "none")
    assert gateway.state.epoch_high_water == gateway.state.sequence_high_water == 0
    assert gateway.state.bound_owner_session_id is None and gateway.effect_log == []
    gateway.crash(persist=True)
    gateway.restart()
    assert gateway.state.epoch_high_water == 0 and gateway.state.inhibited


def test_28_unknown_command_with_wrong_installation_is_no_ack(tmp_path):
    gateway = fake(tmp_path)
    gateway.send(command(), deadline=DEADLINE)
    before = gateway.state
    effects = list(gateway.effect_log)
    assert query(gateway, command(), command_id="unknown", installation_id="foreign") is None
    known = query(gateway, command(), installation_id="foreign")
    assert (known.status, known.reason) == ("accepted", "wrong_installation")
    assert gateway.state == before and gateway.effect_log == effects


@pytest.mark.parametrize("outcome,status,effect", [
    ("accepted", "accepted", "complete"),
    ("partial", "unconfirmed", "partial"),
    ("crash", "unconfirmed", "unknown"),
])
def test_original_ack_equals_query_field_by_field_after_restart(tmp_path, outcome, status, effect):
    gateway = fake(tmp_path)
    if outcome == "partial":
        gateway.helper_ok["mode"] = False
    elif outcome == "crash":
        gateway.lose_ack()
    original = gateway.send(command(), deadline=DEADLINE)
    assert (original.status, original.effect) == (status, effect)
    effects = list(gateway.effect_log)
    gateway.advance_clock(100)
    gateway.crash(persist=True)
    gateway.restart()
    result = query(gateway, command())
    assert result is not None
    for field in fields(original):
        assert getattr(result, field.name) == getattr(original, field.name), field.name
    assert gateway.effect_log == effects


@pytest.mark.parametrize("phase", ["reserved", "rejected", "invalidated", "unknown"])
@pytest.mark.parametrize("effect", ["none", "partial", "unknown"])
@pytest.mark.parametrize("mismatch,override_reason", [
    ({}, None), ({"installation_id": "foreign"}, "wrong_installation"),
    ({"payload_digest": "b" * 64}, "identity_mismatch"),
])
def test_amended_query_table_uses_stored_effect_and_helpers(phase, effect, mismatch, override_reason):
    module = receiver()
    helpers = () if effect == "none" else (HelperResult("setpoint", "dummy.setpoint", True, None, 1),)
    record = replace(reserve(state()).record, phase=phase, effect=effect,
                     reason="stored_reason", helper_results=helpers)
    current = state(commands=(record,))
    assert type(current).from_dict(current.to_dict()) == current
    args = dict(installation_id="sandbox", owner_epoch=7, sequence=41,
                command_id=record.command_id, payload_digest=record.payload_digest)
    result = module.classify_query(current, **(args | mismatch))
    status = "rejected" if phase in ("rejected", "invalidated") and effect == "none" else "unconfirmed"
    reason = {"reserved": "pending_effect", "invalidated": "invalidated"}.get(phase, "stored_reason")
    reason = override_reason or reason
    assert (result.status, result.effect, result.reason) == (status, effect, reason)
    assert result.helper_results == helpers
    assert result.accepted_ts is None and result.observed_at_ms == record.observed_at_ms


def test_refused_duplicate_effect_returns_existing_accepted_ack_without_write(tmp_path):
    gateway = fake(tmp_path)
    original = gateway.send(command(), deadline=DEADLINE)
    gateway.advance_clock(1000)
    before = gateway.state
    effects = list(gateway.effect_log)
    result = gateway.step_helper("setpoint", command=command())
    assert result == original
    assert gateway.state == before and gateway.effect_log == effects


def test_fence_without_record_records_proven_no_effect():
    module = receiver()
    result = module.classify_effect(state(), command(), receiver_now_upper_ms=1002, outcome="none",
                                    clock_healthy=True, clock_uncertainty_ms=0)
    assert (result.status, result.reason) == ("rejected", "no_longer_reserved")
    assert result.record.phase == "rejected" and result.record.effect == "none"
    assert result.record.helper_results == ()
    assert result.state.commands == (result.record,)


@pytest.mark.parametrize("outcome", ["accepted", "partial", "crash"])
def test_denied_retry_preserves_prior_outcome_without_reentering_helpers(tmp_path, outcome):
    gateway = fake(tmp_path)
    if outcome == "partial":
        gateway.helper_ok["mode"] = False
    elif outcome == "crash":
        gateway.lose_ack()
    original = gateway.send(command(), deadline=DEADLINE)
    before = gateway.state
    effects = list(gateway.effect_log)
    retry = gateway.send(command(), deadline=DEADLINE)
    assert retry == original
    assert gateway.state == before and gateway.effect_log == effects


def test_retry_of_live_partial_reservation_does_not_resume_helpers(tmp_path):
    gateway = fake(tmp_path)
    gateway.enqueue(command())
    gateway.deliver()
    original = gateway.step_helper("setpoint")
    before = gateway.state
    retry = gateway.send(command(), deadline=DEADLINE)
    assert retry == original
    assert gateway.state == before
    assert len(gateway.effect_log) == 1


@pytest.mark.parametrize("stored_effect", ["partial", "unknown"])
def test_no_new_effect_cannot_erase_prior_effect(stored_effect):
    module = receiver()
    current = reserve(state()).state
    helpers = (HelperResult("setpoint", "dummy.setpoint", True, None, 1),)
    current = replace(current, commands=(replace(current.commands[0], effect=stored_effect, helper_results=helpers),))
    result = module.classify_effect(current, command(), receiver_now_upper_ms=1002, outcome="none",
                                    clock_healthy=True, clock_uncertainty_ms=0)
    assert (result.status, result.receipt.effect, result.receipt.helper_results) == (
        "unconfirmed", stored_effect, helpers)


@pytest.mark.parametrize("outcome", ["accepted", "partial", "unknown"])
@pytest.mark.parametrize("mismatch,reason", [
    ({"payload_digest": "b" * 64}, "identity_mismatch"),
    ({"owner_epoch": 8}, "identity_mismatch"),
    ({"sequence": 42}, "identity_mismatch"),
    ({"installation_id": "foreign"}, "wrong_installation"),
])
def test_29_mismatching_query_preserves_stored_effect_evidence(tmp_path, outcome, mismatch, reason):
    gateway = fake(tmp_path)
    if outcome == "partial":
        gateway.helper_ok["mode"] = False
    elif outcome == "unknown":
        gateway.lose_ack()
    original = gateway.send(command(), deadline=DEADLINE)
    before = gateway.state
    effects = list(gateway.effect_log)
    result = query(gateway, command(), **mismatch)
    assert result == replace(original, reason=reason)
    assert result.installation_id == "sandbox"
    assert gateway.state == before and gateway.effect_log == effects


@pytest.mark.parametrize("partial", [False, True])
def test_30_blocking_restore_invalidates_reservations_and_blocks_late_callback(tmp_path, partial):
    gateway = fake(tmp_path)
    cmd = command()
    gateway.enqueue(cmd)
    gateway.deliver()
    if partial:
        gateway.step_helper("setpoint")
    original = gateway.state.commands[0]
    snapshot = gateway.snapshot_files()
    gateway.send(command(command_id="higher", sequence=42), deadline=DEADLINE)
    anchor = gateway.anchor_path.read_bytes()
    gateway.restore_files(snapshot)
    assert gateway.restart() == "anchor_checkpoint_mismatch"
    assert gateway.state.inhibited and not gateway.state.queue_known
    stored = gateway.state.commands[0]
    assert (stored.phase, stored.reason) == ("invalidated", "anchor_checkpoint_mismatch")
    assert (stored.effect, stored.helper_results) == (original.effect, original.helper_results)
    assert (gateway.state.epoch_high_water, gateway.state.sequence_high_water) == (7, 42)
    assert gateway.anchor_path.read_bytes() == anchor
    before = list(gateway.effect_log)
    result = gateway.step_helper("mode" if partial else "setpoint", command=cmd)
    assert result.reason == "receiver_inhibited"
    assert result.effect == original.effect and result.helper_results == original.helper_results
    assert gateway.effect_log == before


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("healthy,uncertainty,reason", [
    (False, 0, "clock_unhealthy"),
    (True, None, "clock_uncertainty_unknown"),
    (True, -1, "clock_uncertainty_unknown"),
])
def test_31_clock_failure_after_reservation_prevents_every_helper(tmp_path, partial, healthy, uncertainty, reason):
    gateway = fake(tmp_path)
    cmd = command()
    gateway.enqueue(cmd)
    gateway.deliver()
    if partial:
        gateway.step_helper("setpoint")
    stored = gateway.state.commands[0]
    before = list(gateway.effect_log)
    gateway.clock_healthy, gateway.clock_uncertainty_ms = healthy, uncertainty
    result = gateway.step_helper("mode" if partial else "setpoint", command=cmd)
    assert result.reason == reason
    expected = ("unconfirmed", "partial") if partial else ("rejected", "none")
    assert (result.status, result.effect) == expected
    assert result.helper_results == stored.helper_results
    assert result.observed_at_ms == stored.observed_at_ms
    assert gateway.effect_log == before


def test_32_bind_refuses_epoch_not_confirmed_by_exact_barrier():
    module = receiver()
    barrier = module.BarrierReceipt("sandbox", "barrier-a", 7, 8, "accepted", 1000,
                                    True, "confirmed", "generation-a", "accepted")
    current = state(epoch_high_water=9, bound_owner_epoch=0, bound_owner_session_id=None,
                    last_barrier=barrier, inhibited=True)
    assert type(current).from_dict(current.to_dict()) == current
    result = module.classify_bind(current, installation_id="sandbox", barrier_id="barrier-a",
                                  expected_epoch=9, owner_session_id="owner-b", requested_not_after_ms=12000,
                                  receiver_max_lease_ms=5000, receiver_now_upper_ms=1001)
    assert (result.status, result.reason, result.binding) == ("rejected", "barrier_epoch_mismatch", None)
    assert result.state == current


@pytest.mark.parametrize("changes,healthy,uncertainty,now,reason", [
    ({"epoch_high_water": 8, "inhibited": True, "queue_known": False}, False, None, 3000, "epoch_withdrawn"),
    ({"inhibited": True, "queue_known": False}, False, None, 3000, "receiver_inhibited"),
    ({"queue_known": False}, False, None, 3000, "queue_unknown"),
    ({}, False, None, 3000, "clock_unhealthy"),
    ({}, True, None, 3000, "clock_uncertainty_unknown"),
    ({}, True, -1, 3000, "clock_uncertainty_unknown"),
    ({}, True, True, 3000, "clock_uncertainty_unknown"),
    ({"commands": ()}, True, 0, 2000, "command_expired"),
    ({"commands": ()}, True, 0, 1001, "no_longer_reserved"),
    ({"sequence_high_water": 42}, True, 0, 1001, "no_longer_reserved"),
])
def test_amended_fence_order_and_sequence_high_water(changes, healthy, uncertainty, now, reason):
    module = receiver()
    current = replace(reserve(state()).state, **changes)
    assert module.fence_effect(current, command(), receiver_now_upper_ms=now,
                               clock_healthy=healthy, clock_uncertainty_ms=uncertainty) == (False, reason)


@pytest.mark.parametrize("function", ["fence_effect", "classify_effect", "classify_helper"])
def test_effect_rules_require_explicit_keyword_only_clock_evidence(function):
    parameters = inspect.signature(getattr(receiver(), function)).parameters
    for name in ("clock_healthy", "clock_uncertainty_ms"):
        assert parameters[name].default is inspect.Parameter.empty
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("reason", ["state_missing_or_corrupt", "anchor_checkpoint_mismatch", "rolled_back_state", "recovered"])
def test_recovery_invalidates_only_reserved_and_preserves_all_effect_evidence(reason):
    module = receiver()
    helper = HelperResult("setpoint", "dummy.setpoint", True, None, 1)
    base = reserve(state()).record
    records = (
        replace(base, effect="partial", helper_results=(helper,)),
        replace(base, command_id="second", sequence=42),
        replace(base, command_id="third", sequence=43, phase="accepted", accepted_ts=1001,
                effect="complete", helper_results=(helper,)),
        replace(base, command_id="fourth", sequence=44, phase="unknown", effect="unknown", helper_results=(helper,)),
    )
    loaded = state(commands=records, sequence_high_water=44)
    anchor_epoch, anchor_sequence, checkpoint = 7, 44, loaded.anchor_checkpoint
    if reason == "state_missing_or_corrupt":
        loaded = None
        anchor_epoch, anchor_sequence = 9, 70
    elif reason == "anchor_checkpoint_mismatch":
        # Untrusted loaded values cannot replace the independent anchor values.
        loaded = replace(loaded, epoch_high_water=10, sequence_high_water=80)
        anchor_epoch, anchor_sequence, checkpoint = 9, 70, "new-checkpoint"
    elif reason == "rolled_back_state":
        anchor_epoch, anchor_sequence = 9, 70
    recovered, actual = module.receiver_recovery_state(loaded, anchor_epoch_high_water=anchor_epoch,
        anchor_sequence_high_water=anchor_sequence, anchor_checkpoint=checkpoint)
    assert actual == reason
    assert (recovered.epoch_high_water, recovered.sequence_high_water) == (anchor_epoch, anchor_sequence)
    if reason == "recovered":
        assert recovered == loaded
    else:
        assert recovered.inhibited and not recovered.queue_known
        assert recovered.bound_owner_session_id is None
        expected = () if loaded is None else tuple(
            replace(record, phase="invalidated", reason=reason) if record.phase == "reserved" else record
            for record in records)
        assert recovered.commands == expected


def test_bind_epoch_mismatch_precedes_barrier_epoch_mismatch_and_owner_validation():
    module = receiver()
    barrier = module.BarrierReceipt("sandbox", "barrier-a", 7, 8, "accepted", 1000,
                                    True, "confirmed", "generation-a", "accepted")
    current = state(epoch_high_water=9, last_barrier=barrier)
    args = dict(installation_id="sandbox", barrier_id="barrier-a", owner_session_id="",
                requested_not_after_ms=12000, receiver_max_lease_ms=5000, receiver_now_upper_ms=1001)
    result = module.classify_bind(current, expected_epoch=10, **args)
    assert (result.status, result.reason) == ("rejected", "epoch_mismatch")
    result = module.classify_bind(current, expected_epoch=9, **args)
    assert (result.status, result.reason) == ("rejected", "barrier_epoch_mismatch")
