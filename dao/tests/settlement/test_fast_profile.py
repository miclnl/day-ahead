"""F2a: solver energy, not the legacy command text, owns the v2 profile."""

import pytest

from dao.prog.fastctrl import plan as plans
from dao.tests.settlement.test_fast_contract import ready_plan

FIELDS = ("mean_ac_power_w", "on_power_w", "active_from_ts", "stop_ts")


@pytest.mark.parametrize("operation", ["deepcopy_step", "deepcopy_interval", "replace_step",
                                        "replace_interval_same_span", "replace_interval_new_span", "reuse_steps"])
def test_fix1_profile_binding_has_exclusive_span_ownership(operation):
    from copy import deepcopy
    from dataclasses import replace

    a = 1000
    step = plans.BatteryPlanStep(1000, 80, 70, mean_ac_power_w=1000, on_power_w=1000,
                                 active_from_ts=a, stop_ts=None)
    row = plans.PlanInterval(a, a + 900, 0.1, 0.1, batteries=[step])
    if operation == "deepcopy_step":
        copied = deepcopy(step)
        assert plans.profile_energy_kwh(copied, a, a + 900) == pytest.approx(0.25)
        other = replace(row, end_ts=a + 450, batteries=[copied])
        assert plans.profile_energy_kwh(copied, a, a + 900) == pytest.approx(0.25)
    elif operation == "deepcopy_interval":
        copied = deepcopy(row)
        assert copied.battery(0) is not step
        assert plans.profile_energy_kwh(copied.battery(0), a, a + 900) == pytest.approx(0.25)
        other = replace(copied, end_ts=a + 450)
        assert plans.profile_energy_kwh(copied.battery(0), a, a + 900) == pytest.approx(0.25)
    elif operation == "replace_step":
        copied = replace(step)
        with pytest.raises(ValueError, match="profile_interval_required"):
            plans.profile_energy_kwh(copied, a, a + 900)
        other = replace(row, end_ts=a + 450, batteries=[copied])
    elif operation == "replace_interval_same_span":
        other = replace(row, price_import=0.2)
    elif operation == "replace_interval_new_span":
        other = replace(row, end_ts=a + 450)
    else:
        # Reusing even the same list must not mutate the original row's slots.
        other = plans.PlanInterval(a, a + 450, 0.1, 0.1, batteries=row.batteries)
    assert plans.profile_energy_kwh(step, a, a + 900) == pytest.approx(0.25)
    expected = 0.25 if operation == "replace_interval_same_span" else 0.125
    assert plans.profile_energy_kwh(other.battery(0), a, a + 900) == pytest.approx(expected)
    if operation != "replace_interval_same_span":
        assert other.battery(0) is not step
        assert row.battery(0) is step


def test_fix1_legacy_step_reuse_does_not_change_v1_wire():
    step = plans.BatteryPlanStep(123.4, 50, 51, stop_inverter="legacy")
    row = plans.PlanInterval(1000, 1900, 0.1, 0.1, batteries=[step])
    before = row.to_dict()
    other = plans.PlanInterval(1000, 1450, 0.1, 0.1, batteries=row.batteries)
    assert other.battery(0) is step and row.to_dict() == before


def shaped_plan(*, stop_none=False):
    plan = ready_plan()
    interval = plan.intervals[0]
    step = interval.battery(0)
    # Dynamic assignment permits a causal pre-feature assertion, not a constructor error.
    step.ac_power_w = step.mean_ac_power_w = 1000.0
    step.on_power_w = 2000.0
    step.active_from_ts = interval.start_ts + 450 if stop_none else interval.start_ts
    step.stop_ts = None if stop_none else interval.start_ts + 450
    return plan


def test_v2_roundtrip_preserves_required_profile_without_rounding():
    plan = shaped_plan()
    step = plan.intervals[0].battery(0)
    step.ac_power_w = step.mean_ac_power_w = 1000.123456789
    step.on_power_w = 2000.246913578
    payload = plan.to_dict()
    wire = payload["intervals"][0]["batteries"][0]
    assert all(name in wire for name in FIELDS)
    assert plans.FastPlan.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize("missing", FIELDS)
def test_v2_missing_profile_is_explicitly_rejected(missing):
    payload = ready_plan().to_dict()
    payload["intervals"][0]["batteries"][0].pop(missing, None)
    with pytest.raises(ValueError, match="profile"):
        plans.FastPlan.from_dict(payload)


@pytest.mark.parametrize("field,value", [
    ("on_power_w", 1500.0), ("mean_ac_power_w", 999.0),
    ("active_from_ts", -1), ("stop_ts", -1),
    ("active_from_ts", True), ("on_power_w", float("nan")),
])
def test_strict_v2_rejects_inconsistent_or_out_of_span_profile(field, value):
    plan = shaped_plan()
    setattr(plan.intervals[0].battery(0), field, value)
    with pytest.raises(ValueError):
        plan.to_dict()


@pytest.mark.parametrize("stop_none", [False, True])
def test_pure_helpers_observe_half_open_window_and_integral(stop_none):
    assert callable(getattr(plans, "profile_power_at", None)), "profile helper missing"
    assert callable(getattr(plans, "profile_energy_kwh", None)), "profile integrator missing"
    plan = shaped_plan(stop_none=stop_none)
    interval = plans.FastPlan.from_dict(plan.to_dict()).intervals[0]
    step = interval.battery(0)
    a, b = interval.start_ts, interval.end_ts
    assert plans.profile_power_at(step, a - 1) == 0.0
    assert plans.profile_power_at(step, b) == 0.0
    assert plans.profile_power_at(step, a + 450) == (2000.0 if stop_none else 0.0)
    assert plans.profile_energy_kwh(step, a - 100, b + 100) == pytest.approx(0.25)
    assert sum(plans.profile_power_at(step, t) for t in range(a, b)) / 3600000 == pytest.approx(0.25)


def test_legacy_without_profile_stays_readable_and_has_no_financial_context():
    legacy = plans.FastPlan(1, 900, intervals=[plans.PlanInterval(
        1, 901, 0.1, 0.1, batteries=[plans.BatteryPlanStep(123.4, 50, 50, stop_inverter="legacy")])])
    wire = legacy.to_dict()
    assert not set(FIELDS).intersection(wire["intervals"][0]["batteries"][0])
    restored = plans.FastPlan.from_dict(wire)
    assert restored.to_dict() == wire and restored.financial_context is None


def test_producer_validation_compares_to_solver_not_a_self_consistent_wrong_mean(monkeypatch):
    plan = ready_plan()
    original = plans.BatteryPlanStep

    def wrong_but_internally_consistent_profile(**kwargs):
        # A profile with half the actual solver energy would pass a wire-only
        # mean/integral comparison. The producer must reject it independently.
        for key in ("ac_power_w", "mean_ac_power_w", "on_power_w"):
            kwargs[key] *= 0.5
        return original(**kwargs)

    monkeypatch.setattr(plans, "BatteryPlanStep", wrong_but_internally_consistent_profile)
    with pytest.raises(ValueError, match="profile_solver_energy_mismatch"):
        plans.build_plan(
            plan.created_ts, 900, plan.specs, [row.start_ts for row in plan.intervals],
            [0.25] * 3, [0.1] * 3, [0.1] * 3, [0.25] * 3, [0.0] * 3,
            [[1.0] * 3], [[50.0] * 4], financial_context=plan.financial_context,
            battery_charge_kw=[[1.0] * 3], battery_discharge_kw=[[0.0] * 3],
        )
