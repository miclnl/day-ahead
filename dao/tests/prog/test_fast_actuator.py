"""Synthetic helper acceptance; a receipt is deliberately not a measurement."""

from dataclasses import asdict, replace
from importlib import import_module

import pytest

from dao.prog.fastctrl.policy import BatteryDecision
from dao.tests.prog.test_fast_policy import T0, make_plan
from dao.tests.settlement.test_fast_contract import ready_plan


class ScriptedGateway:
    """Only the external helper boundary is replaced, including read failures."""

    def __init__(self, outcomes=None, values=None):
        self.outcomes = {key: list(value) for key, value in (outcomes or {}).items()}
        self.values = values or {}
        self.calls = []

    def _write(self, helper, entity, value):
        self.calls.append((helper, entity, value))
        outcomes = self.outcomes.get(helper, [True])
        result = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(result, Exception):
            raise result
        return result

    def write_number(self, entity, value):
        return self._write("setpoint", entity, value)

    def select_option(self, entity, value):
        return self._write("mode", entity, value)

    def set_datetime(self, entity, value):
        return self._write("stop_inverter", entity, value)

    def read(self, entities):
        self.calls.append(("read", tuple(entities)))
        if isinstance(self.values, Exception):
            raise self.values
        return {entity: (self.values.get(entity), T0) for entity in entities}


def proposal(**overrides):
    return replace(BatteryDecision(0, "accu", -1200, 0, True, True, "override",
                                   mode="Aan", stop_inverter="2000-01-01 00:00:00"),
                   **overrides)


def actuator():
    # API absence is reported separately from the existing-code causal REDs.
    return import_module("dao.prog.fastctrl.actuator")


def test_receipt_is_deterministic_traceable_and_not_a_measurement():
    plan = make_plan([0.3], [0])
    gateway = ScriptedGateway()
    execute = actuator().execute_battery_command
    receipt = execute(gateway, plan, proposal(), T0 + 60.9)
    assert receipt.command_id == "v1:1700000000/b0/1700000060"
    assert receipt.plan_identity == "v1:1700000000"
    assert receipt.timestamp == T0 + 60.9
    assert (receipt.battery_index, receipt.battery_name, receipt.setpoint_w) == (0, "accu", -1200)
    assert receipt.status == "confirmed"
    assert [(r.helper, r.entity_id, r.ok, r.error) for r in receipt.helpers] == [
        ("setpoint", "input_number.feedin", True, None),
        ("mode", "input_select.mode", True, None),
        ("stop_inverter", "input_datetime.stop", True, None),
    ]
    assert [c[0] for c in gateway.calls] == ["setpoint", "mode", "stop_inverter"]
    assert execute(ScriptedGateway(), plan, proposal(), T0 + 60.9) == receipt
    assert set(asdict(receipt)) == {"command_id", "plan_identity", "timestamp", "battery_index",
                                    "battery_name", "setpoint_w", "status", "helpers"}


def test_v2_identity_includes_replay_basis_without_granting_rights():
    plan = ready_plan()
    identity = actuator().plan_identity(plan)
    assert identity == "v2:1798758900:" + "b" * 64
    other = replace(plan, financial_context=replace(plan.financial_context, replay_basis_hash="c" * 64))
    assert actuator().plan_identity(other) != identity


@pytest.mark.parametrize("missing", ["context", "hash"])
def test_f2b_r22_invalid_v2_identity_has_no_helper_side_effect(missing):
    plan = ready_plan()
    plan.specs[0] = replace(plan.specs[0], setpoint_entity="input_number.feedin")
    if missing == "context":
        plan.financial_context = None
    else:
        object.__setattr__(plan.financial_context, "replay_basis_hash", "")
    gateway = ScriptedGateway()
    with pytest.raises(ValueError, match="actuation_identity_invalid"):
        actuator().execute_battery_command(gateway, plan, proposal(), T0)
    assert gateway.calls == []


@pytest.mark.parametrize("outcomes,values,expected", [
    ([True], {}, (True, 1, None)),
    ([RuntimeError("secret-token"), True], {}, (True, 2, "write_failed")),
    ([False], {"input_number.feedin": "-1200"}, (True, 2, "write_failed")),
    ([False], RuntimeError("secret-token"), (False, 2, "readback_unreadable")),
    ([False], {}, (False, 2, "readback_unreadable")),
    ([False], {"input_number.feedin": "-1199"}, (False, 2, "readback_mismatch")),
])
def test_f2b_r23_helper_attempts_and_sanitized_error(outcomes, values, expected):
    gateway = ScriptedGateway({"setpoint": outcomes}, values)
    receipt = actuator().execute_battery_command(gateway, make_plan([0.3], [0]), proposal(), T0)
    helper = receipt.helpers[0]
    assert helper.error == expected[2]
    assert (helper.ok, helper.attempts, helper.error) == expected
    assert "secret-token" not in repr(receipt)


@pytest.mark.parametrize("outcomes,status", [
    ({"mode": [False]}, "unknown"),
    ({"setpoint": [False], "mode": [False], "stop_inverter": [False]}, "unconfirmed"),
])
def test_non_atomic_failures_are_recorded_with_bounded_retry_and_one_read(outcomes, status):
    gateway = ScriptedGateway(outcomes)
    receipt = actuator().execute_battery_command(gateway, make_plan([0.3], [0]), proposal(), T0)
    assert receipt.status == status
    for helper in ("setpoint", "mode", "stop_inverter"):
        assert sum(c[0] == helper for c in gateway.calls) == (2 if helper in outcomes else 1)
    assert sum(c[0] == "read" for c in gateway.calls) == len(outcomes)
    assert all(r.error and not r.ok for r in receipt.helpers if r.helper in outcomes)


def test_first_failure_then_success_needs_no_readback():
    gateway = ScriptedGateway({"mode": [False, True]})
    receipt = actuator().execute_battery_command(gateway, make_plan([0.3], [0]), proposal(), T0)
    assert receipt.status == "confirmed"
    assert [c[0] for c in gateway.calls] == ["setpoint", "mode", "mode", "stop_inverter"]


@pytest.mark.parametrize("helper,entity,value", [
    ("setpoint", "input_number.feedin", "-1200.0"),
    ("mode", "input_select.mode", "Aan"),
    ("stop_inverter", "input_datetime.stop", "2000-01-01 00:00:00"),
])
def test_one_matching_readback_confirms_helper_acceptance(helper, entity, value):
    gateway = ScriptedGateway({helper: [False]}, {entity: value})
    receipt = actuator().execute_battery_command(gateway, make_plan([0.3], [0]), proposal(), T0)
    assert receipt.status == "confirmed"
    assert sum(c[0] == helper for c in gateway.calls) == 2
    assert [c for c in gateway.calls if c[0] == "read"] == [("read", (entity,))]


@pytest.mark.parametrize("values", [{}, {"input_number.feedin": "-1199"},
                                   {"input_number.feedin": "nan"},
                                   {"input_number.feedin": "unavailable"}, RuntimeError("unreadable")])
def test_missing_unreadable_or_different_readback_fails_closed(values):
    gateway = ScriptedGateway({"setpoint": [False]}, values)
    receipt = actuator().execute_battery_command(gateway, make_plan([0.3], [0]), proposal(), T0)
    assert receipt.status == "unknown"
    assert receipt.helpers[0].ok is False
    assert sum(c[0] == "read" for c in gateway.calls) == 1


def test_exceptions_are_failed_helpers_not_implicit_acceptance():
    gateway = ScriptedGateway({"mode": [RuntimeError("failed")]})
    receipt = actuator().execute_battery_command(gateway, make_plan([0.3], [0]), proposal(), T0)
    assert receipt.status == "unknown"
    assert receipt.helpers[1].error
    assert sum(c[0] == "mode" for c in gateway.calls) == 2


def test_readback_compares_datetime_values_not_display_precision():
    gateway = ScriptedGateway({"stop_inverter": [False]},
                              {"input_datetime.stop": "2026-01-15 18:07:00"})
    receipt = actuator().execute_battery_command(
        gateway, make_plan([0.3], [0]), proposal(stop_inverter="2026-01-15 18:07"), T0)
    assert receipt.status == "confirmed"
    assert sum(c[0] == "read" for c in gateway.calls) == 1


@pytest.mark.parametrize("value", ["2026-01-15 18:07:01", "unavailable", None, "not a date"])
def test_datetime_readback_does_not_accept_different_or_unreadable_value(value):
    gateway = ScriptedGateway({"stop_inverter": [False]}, {"input_datetime.stop": value})
    receipt = actuator().execute_battery_command(
        gateway, make_plan([0.3], [0]), proposal(stop_inverter="2026-01-15 18:07"), T0)
    assert receipt.status == "unknown"



@pytest.mark.parametrize("kind", ["write_false", "no_entities", "optional_only"])
def test_no_helper_write_means_no_receipt(kind):
    plan = make_plan([0.3], [0])
    battery = proposal(write=kind != "write_false")
    if kind == "no_entities":
        plan.specs[0] = replace(plan.specs[0], setpoint_entity=None, mode_entity=None,
                                stop_inverter_entity=None)
    elif kind == "optional_only":
        plan.specs[0] = replace(plan.specs[0], setpoint_entity=None)
        battery = proposal(mode=None, stop_inverter=None)
    gateway = ScriptedGateway()
    assert actuator().execute_battery_command(gateway, plan, battery, T0) is None
    assert gateway.calls == []
