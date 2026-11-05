"""Behavioural tests for the fast control decision logic.

The policy is a pure function, so every scenario below is a complete
specification of what the layer must do: build a plan, hand it a measurement,
assert the setpoint. Where an analytical answer exists the test checks against
that rather than against a recorded value.
"""

import pytest

from dao.prog.fastctrl.plan import (
    BatteryPlanStep,
    BatterySpec,
    FastPlan,
    PlanInterval,
)
from dao.prog.fastctrl.policy import (
    BatteryMeasurement,
    ControllerState,
    CostModel,
    FastControlPolicy,
    Measurement,
    PolicyLimits,
    apply_accepted_command,
    estimate_storage_value,
)

T0 = 1_700_000_000


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("limiter", ["deadband", "rate_limited"])
def test_fix1_local_stop_affects_only_legacy_decisions(version, limiter):
    from copy import deepcopy

    from dao.tests.settlement.test_fast_contract import ready_plan

    plan = ready_plan()
    plan.price_average = 0.3
    plan.specs[0].minimum_power_w = 1000
    for row in plan.intervals:
        row.price_import, row.price_export = 0.3, 0.0
        step = row.battery(0)
        step.ac_power_w = step.mean_ac_power_w = step.on_power_w = -400
    plan = FastPlan.from_dict(plan.to_dict())
    if version == 1:
        plan.format_version, plan.financial_context = 1, None
    limits = PolicyLimits(min_benefit=0, deadband=5000 if limiter == "deadband" else 50,
                          min_command_interval=60, urgent_deviation=10000)
    now = plan.intervals[0].start_ts + 60
    decisions = []
    for stop_text in (None, "2026-01-15 18:07"):
        candidate = deepcopy(plan)
        candidate.intervals[0].battery(0).stop_inverter = stop_text
        state = ControllerState()
        state.battery(0).held_setpoint_w = -400
        state.battery(0).last_change_ts = now - 10
        binding = accept_policy_command(candidate, state, -400, now - 10) if version == 2 else None
        decision = FastControlPolicy(limits).decide(
            candidate, measure(grid_w=3400, battery_w=-400, timestamp=now), state,
            owner_binding=binding).batteries[0]
        decisions.append((decision.setpoint_w, decision.write, decision.reason))
    assert decisions[0][0:2] == (-400, False)
    assert limiter in decisions[0][2]
    if version == 2:
        assert decisions[1] == decisions[0]
    else:
        assert decisions[1][0] == -3800
        assert decisions[1][1] is True
        assert limiter not in decisions[1][2]


def test_policy_fallback_and_accounting_use_window_not_mean_or_stop_text():
    from dataclasses import replace

    from dao.tests.settlement.test_fast_profile import shaped_plan

    plan = shaped_plan()
    interval = plan.intervals[0]
    interval.battery(0).stop_inverter = "not a profile source"
    a = interval.start_ts
    policy, state = FastControlPolicy(PolicyLimits()), ControllerState()
    before = Measurement(a + 400, 0.0, [BatteryMeasurement(soc=50, power_w=0, valid=True)])
    after = replace(before, timestamp=a + 700)
    policy.account(state, plan, before, "synthetic")
    policy.account(state, plan, after, "synthetic")
    assert state.battery(0).interval_deviation_kwh == pytest.approx(-1 / 36)
    assert state.battery(0).daily_deviation_kwh == pytest.approx(1 / 36)
    decision = policy.decide(plan, replace(after, grid_valid=False), state)
    assert decision.batteries[0].plan_w == 0
    # Also exercise the normal residual/disabled path and release before the stop.
    state.battery(0).override_active = True
    decision = policy.decide(plan, before, state, enabled=[False])
    assert decision.batteries[0].plan_w == 2000
    assert decision.batteries[0].stop_inverter is None


def test_limited_v2_still_hits_the_separate_actuation_gate():
    from dataclasses import replace

    from dao.prog.fastctrl.publication import GatePolicy, PlanRead, following_verdict
    from dao.tests.settlement.test_fast_contract import ready_plan

    plan = ready_plan()
    plan.financial_context = replace(plan.financial_context, economics_supported=False,
                                     issues=("bonus_prefix_unknown",))
    verdict = following_verdict(PlanRead("ready", plan, "ready"), policy=GatePolicy(None),
                                now_ts=plan.created_ts)
    assert not verdict.allowed and verdict.reason == "actuation_gates_unproven"


def make_spec(**overrides) -> BatterySpec:
    defaults = dict(
        name="accu",
        capacity_kwh=10.0,
        max_charge_w=5000.0,
        max_discharge_w=5000.0,
        minimum_power_w=0.0,
        soc_min=20.0,
        soc_max=95.0,
        cycle_cost=0.01,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        setpoint_entity="input_number.feedin",
        mode_entity="input_select.mode",
        stop_inverter_entity="input_datetime.stop",
        soc_entity="sensor.soc",
    )
    defaults.update(overrides)
    return BatterySpec(**defaults)


def make_plan(
    prices,
    battery_w,
    socs=None,
    specs=None,
    interval_s=900,
    created_ts=T0,
    export_discount=0.18,
    house_w=None,
):
    """Build a plan from parallel lists, one entry per interval."""
    specs = specs or [make_spec()]
    socs = socs or [50.0] * (len(prices) + 1)
    if not isinstance(battery_w[0], (list, tuple)):
        battery_w = [[value] for value in battery_w]
    house_w = house_w if house_w is not None else [500.0] * len(prices)

    intervals = []
    for index, price in enumerate(prices):
        steps = [
            BatteryPlanStep(
                ac_power_w=float(value),
                soc_begin=float(socs[index]),
                soc_end=float(socs[index + 1]),
            )
            for value in battery_w[index]
        ]
        total_battery = sum(s.ac_power_w for s in steps)
        intervals.append(
            PlanInterval(
                start_ts=T0 + index * interval_s,
                end_ts=T0 + (index + 1) * interval_s,
                price_import=price,
                price_export=max(0.0, price - export_discount),
                grid_w=house_w[index] + total_battery,
                house_w=house_w[index],
                batteries=steps,
            )
        )
    return FastPlan(
        created_ts=created_ts,
        interval_s=interval_s,
        specs=specs,
        intervals=intervals,
        price_average=sum(prices) / len(prices),
    )


def measure(grid_w, soc=50.0, battery_w=0.0, timestamp=T0 + 60, count=1, **kwargs):
    return Measurement(
        timestamp=timestamp,
        grid_w=grid_w,
        batteries=[
            BatteryMeasurement(soc=soc, power_w=battery_w, valid=True)
            for _ in range(count)
        ],
        **kwargs,
    )


def run(plan, measurement, limits=None, state=None, enabled=None):
    policy = FastControlPolicy(limits or PolicyLimits(min_benefit=0.0, deadband=0.0))
    state = state or ControllerState()
    return policy.decide(plan, measurement, state, enabled), state


def accepted_facts(controller):
    return (controller.last_command_w, controller.last_command_ts,
            controller.stop_inverter_cleared)


def test_f2b_r17_two_ticks_match_base_exactly():
    plan = make_plan([0.3] * 4, [-250] * 4)
    policy, state = FastControlPolicy(PolicyLimits(min_benefit=0)), ControllerState()
    observed = []
    for tick, grid in [(60, 0), (120, -105)]:
        battery = policy.decide(
            plan, measure(grid_w=grid, battery_w=-250, timestamp=T0 + tick), state,
        ).batteries[0]
        apply_accepted_command(state.battery(0), battery)
        observed.append((battery.setpoint_w, battery.write, battery.reason))
    assert observed == [(-250, False, "deadband"), (-250, False, "deadband")]
    assert accepted_facts(state.battery(0)) == (0, 0, False)


@pytest.mark.parametrize("fallback", [False, True])
def test_f2b_r19_pending_restore_bypasses_deadband_not_command_interval(fallback):
    plan = make_plan([0.3] * 4, [-400] * 4)
    step = plan.intervals[0].battery(0)
    step.mode, step.stop_inverter = "Aan", "2030-01-01 00:00:00"
    policy = FastControlPolicy(PolicyLimits(min_benefit=0, deadband=150, min_command_interval=60))
    state = ControllerState()
    controller = state.battery(0)
    controller.restore_pending = True
    controller.held_setpoint_w = -400
    controller.last_change_ts = T0 + 300
    for tick, expected_write in [(330, False), (360, True)]:
        sample = measure(grid_w=0, battery_w=-400, timestamp=T0 + tick, grid_valid=not fallback)
        decision = policy.decide(plan, sample, state).batteries[0]
        assert decision.write is expected_write
        if expected_write:
            assert decision.restores is True
            assert (decision.mode, decision.stop_inverter) == ("Aan", "2030-01-01 00:00:00")
        else:
            assert "rate_limited" in decision.reason
        assert controller.restore_pending is True  # Policy never discharges the obligation.


def test_f2b_r19_r20_new_state_fields_roundtrip_with_defaults():
    from dao.prog.fastctrl.policy import BatteryControllerState

    state = BatteryControllerState.from_dict({"restore_pending": True, "held_stop_inverter_cleared": True})
    assert state.to_dict().get("restore_pending") is True
    assert state.to_dict().get("held_stop_inverter_cleared") is True
    old = BatteryControllerState.from_dict({})
    assert old.restore_pending is old.held_stop_inverter_cleared is False


def test_f2b_held_state_roundtrip_and_old_state_default():
    from dao.prog.fastctrl.policy import BatteryControllerState

    state = BatteryControllerState.from_dict({"held_setpoint_w": -675, "last_command_w": 123})
    assert state.to_dict().get("held_setpoint_w") == -675
    assert BatteryControllerState.from_dict(state.to_dict()).held_setpoint_w == -675
    old = BatteryControllerState.from_dict({"last_command_w": 123})
    assert old.to_dict().get("held_setpoint_w") == 0.0
    assert old.last_command_w == 123


@pytest.mark.parametrize("valid,power", [(True, None), (False, 9999)])
def test_f2b_account_uses_held_power_when_measurement_is_missing_or_stale(valid, power):
    plan = make_plan([0.3] * 4, [0] * 4)
    policy, state = FastControlPolicy(PolicyLimits()), ControllerState()
    controller = state.battery(0)
    controller.last_command_w, controller.held_setpoint_w = 777, -1500
    first = measure(grid_w=1000, battery_w=power, timestamp=T0 + 60)
    first.batteries[0].valid = valid
    policy.account(state, plan, first, "fixed")
    second = measure(grid_w=1000, battery_w=power, timestamp=T0 + 120)
    second.batteries[0].valid = valid
    policy.account(state, plan, second, "fixed")
    assert controller.interval_deviation_kwh == -0.025
    assert controller.daily_deviation_kwh == 0.025
    assert controller.last_command_w == 777


@pytest.mark.parametrize("path", ["missing_measurement", "ramp", "disabled", "no_step"])
def test_f2b_control_law_reads_held_not_accepted_power(path):
    plan = make_plan([0.3] * 4, [0] * 4)
    policy = FastControlPolicy(PolicyLimits(min_benefit=0, deadband=150,
                                          max_ramp=500 if path == "ramp" else None))
    state = ControllerState()
    controller = state.battery(0)
    controller.last_command_w, controller.held_setpoint_w = 777, -1000
    if path == "disabled":
        for row in plan.intervals:
            row.battery(0).ac_power_w = -1000
    sample = measure(grid_w=3000 if path == "ramp" else 1000,
                     battery_w=None if path == "missing_measurement" else -1000,
                     timestamp=T0 + (5000 if path == "no_step" else 60))
    decision = policy.decide(plan, sample, state, enabled=[path != "disabled"])
    expected = {"missing_measurement": -2000, "ramp": -1500, "disabled": -1000, "no_step": -1000}
    assert decision.batteries[0].setpoint_w == expected[path]
    if path in ("disabled", "no_step"):
        assert decision.batteries[0].write is False
    assert controller.last_command_w == 777


@pytest.mark.parametrize("path", ["override", "fallback", "release"])
def test_f2b_policy_only_proposes_accepted_command(path):
    plan = make_plan([0.3] * 4, [0] * 4)
    state = ControllerState()
    controller = state.battery(0)
    controller.last_command_w, controller.last_change_ts, controller.last_command_ts = -250, T0 - 20, T0 - 20
    if path != "override":
        controller.override_active = controller.stop_inverter_cleared = True
    before = accepted_facts(controller)
    policy = FastControlPolicy(PolicyLimits(min_benefit=0, deadband=0, release_time=0))
    sample = measure(grid_w=1500 if path == "override" else 0, grid_valid=path != "fallback")
    decision = (policy.release(plan, sample, state, "mode_off") if path == "release"
                else policy.decide(plan, sample, state)).batteries[0]
    assert decision.write
    assert accepted_facts(controller) == before
    assert decision.accepted is not None
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        decision.accepted.last_command_w = 123
    apply_accepted_command(controller, decision)
    assert accepted_facts(controller) == (decision.setpoint_w, T0 + 60, path == "override")
    assert controller.last_change_ts == T0 + 60  # Issued by the policy, not the receipt.
    assert controller.held_setpoint_w == decision.setpoint_w


@pytest.mark.parametrize("releasing", [False, True])
def test_f2b_plan_reconciliation_updates_held_not_accepted_state(releasing):
    plan = make_plan([0.3] * 4, [500] * 4)
    state = ControllerState()
    controller = state.battery(0)
    controller.last_command_w, controller.last_change_ts, controller.last_command_ts = -250, T0 - 20, T0 - 20
    controller.held_setpoint_w = -250
    controller.override_active = controller.stop_inverter_cleared = releasing
    before = accepted_facts(controller)
    decision, _ = run(plan, measure(grid_w=500, battery_w=500),
                      PolicyLimits(min_benefit=1e9, deadband=150, release_time=0), state)
    battery = decision.batteries[0]
    assert battery.setpoint_w == 500
    assert battery.write is releasing  # Release still restores mode/stop.
    assert accepted_facts(controller) == before
    assert controller.override_active is False
    assert controller.quiet_since is None
    if not releasing:
        assert battery.accepted is None
        assert controller.held_setpoint_w == 500
    else:
        assert controller.held_setpoint_w == -250  # Release still awaits confirmation.



class TestCostModel:
    """The solve must be exact, not approximately right."""

    @pytest.mark.parametrize(
        "house,price_import,price_export,storage_value",
        [
            (3000.0, 0.30, 0.10, 0.25),
            (-2500.0, 0.30, 0.10, 0.25),
            (0.0, 0.30, 0.10, 0.40),
            (1500.0, -0.05, -0.20, 0.10),
            (4000.0, 0.45, 0.45, 0.30),
        ],
    )
    def test_matches_brute_force(
        self, house, price_import, price_export, storage_value
    ):
        cost = CostModel(
            house_w=house,
            price_import=price_import,
            price_export=price_export,
            storage_value=storage_value,
            cycle_cost=0.01,
            round_trip_efficiency=0.9,
        )
        low, high = -5000.0, 5000.0
        analytic = cost.minimise(low, high)
        grid = [low + step * (high - low) / 20000 for step in range(20001)]
        brute = min(grid, key=cost)
        assert cost(analytic) <= cost(brute) + 1e-9

    def test_convexity_holds_for_realistic_prices(self):
        cost = CostModel(3000.0, 0.30, 0.10, 0.25, 0.01, 0.9)
        points = [-5000 + i * 100 for i in range(101)]
        values = [cost(p) for p in points]
        slopes = [
            (values[i + 1] - values[i]) / 100.0 for i in range(len(values) - 1)
        ]
        assert all(
            slopes[i + 1] >= slopes[i] - 1e-12 for i in range(len(slopes) - 1)
        )


class TestNormalRegime:
    """Ordinary prices: export < storage value < import."""

    def test_unexpected_load_is_covered_from_the_battery(self):
        # Plan: idle battery, 500 W house. Reality: 3500 W house.
        plan = make_plan([0.30] * 8, [0.0] * 8)
        decision, _ = run(plan, measure(grid_w=3500.0))
        battery = decision.batteries[0]
        assert battery.override
        # Cover the whole deviation: grid goes to zero.
        assert battery.setpoint_w == pytest.approx(-3500.0, abs=1.0)

    def test_unexpected_surplus_is_absorbed(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        decision, _ = run(plan, measure(grid_w=-2200.0))
        battery = decision.batteries[0]
        assert battery.override
        assert battery.setpoint_w == pytest.approx(2200.0, abs=1.0)

    def test_planned_grid_charge_is_reduced_by_an_unexpected_load(self):
        # Plan charges 3000 W during a cheap interval; an oven starts.
        plan = make_plan([0.14] * 8, [3000.0] * 8, house_w=[400.0] * 8)
        decision, _ = run(plan, measure(grid_w=6400.0, battery_w=3000.0))
        battery = decision.batteries[0]
        # house = 6400 - 3000 = 3400 W. Cancelling the deviation means the
        # battery stops charging and covers the load instead.
        assert battery.setpoint_w < 3000.0
        assert battery.override

    def test_no_deviation_means_no_override(self):
        plan = make_plan([0.30] * 8, [-1000.0] * 8, house_w=[1000.0] * 8)
        decision, _ = run(plan, measure(grid_w=0.0, battery_w=-1000.0))
        assert not decision.batteries[0].override
        assert decision.batteries[0].setpoint_w == pytest.approx(-1000.0, abs=1.0)


class TestPriceRegimes:
    """The same formula must produce the right behaviour in every regime."""

    def test_negative_export_price_makes_charging_very_attractive(self):
        plan = make_plan([0.05] * 8, [0.0] * 8, export_discount=0.30)
        assert plan.intervals[0].price_export == 0.0
        plan.intervals[0].price_export = -0.15
        decision, _ = run(plan, measure(grid_w=-1800.0))
        assert decision.batteries[0].setpoint_w == pytest.approx(1800.0, abs=1.0)

    def test_price_spike_later_makes_the_layer_hold_its_energy(self):
        # Now is cheap, a spike is coming and the plan saves the battery for it.
        prices = [0.10, 0.10, 0.60, 0.60, 0.60, 0.10, 0.10, 0.10]
        socs = [90.0, 90.0, 90.0, 60.0, 30.0, 25.0, 25.0, 25.0, 25.0]
        plan = make_plan(prices, [0.0] * 8, socs=socs)
        decision, _ = run(plan, measure(grid_w=3000.0, soc=90.0))
        battery = decision.batteries[0]
        # Storage is worth more than the current cheap import price, so it must
        # not be spent now.
        assert battery.storage_value > plan.intervals[0].price_import
        assert battery.setpoint_w >= -1.0

    def test_expensive_now_and_cheap_later_makes_the_layer_spend(self):
        prices = [0.60, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        socs = [90.0] * 4 + [40.0] * 5
        plan = make_plan(prices, [0.0] * 8, socs=socs)
        decision, _ = run(plan, measure(grid_w=3000.0, soc=90.0))
        assert decision.batteries[0].setpoint_w == pytest.approx(-3000.0, abs=1.0)

    def test_export_price_above_import_is_clamped(self):
        plan = make_plan([0.20] * 4, [0.0] * 4)
        for interval in plan.intervals:
            interval.price_export = 0.50
        decision, _ = run(plan, measure(grid_w=1000.0))
        assert decision.price_export <= decision.price_import


class TestStorageValue:
    def test_plan_mode_is_capped_by_the_refill_price(self):
        # A 0.60 spike is coming, but the plan can refill at 0.10 first.
        prices = [0.10, 0.10, 0.60, 0.60]
        socs = [50.0, 50.0, 50.0, 20.0, 20.0]
        plan = make_plan(prices, [0.0] * 4, socs=socs)
        value = estimate_storage_value(plan, T0 + 10, 0, PolicyLimits())
        assert value == pytest.approx(0.10 / 0.90, rel=1e-6)

    def test_plan_mode_uses_the_best_use_when_no_cheap_refill_exists(self):
        prices = [0.50, 0.55, 0.60]
        socs = [80.0, 60.0, 40.0, 20.0]
        plan = make_plan(prices, [0.0] * 3, socs=socs)
        value = estimate_storage_value(plan, T0 + 10, 0, PolicyLimits())
        # min(0.9 * 0.60, 0.50 / 0.9) = min(0.54, 0.5555) = 0.54
        assert value == pytest.approx(0.54, rel=1e-6)

    def test_fixed_mode_wins(self):
        plan = make_plan([0.30] * 4, [0.0] * 4)
        limits = PolicyLimits(storage_value_mode="fixed", storage_value_fixed=0.123)
        assert estimate_storage_value(plan, T0 + 10, 0, limits) == 0.123

    def test_average_mode(self):
        plan = make_plan([0.20, 0.40], [0.0, 0.0])
        limits = PolicyLimits(storage_value_mode="average", round_trip_efficiency=1.0)
        assert estimate_storage_value(plan, T0 + 10, 0, limits) == pytest.approx(0.30)

    def test_never_below_the_export_price(self):
        plan = make_plan([0.10] * 3, [0.0] * 3, export_discount=-0.40)
        value = estimate_storage_value(plan, T0 + 10, 0, PolicyLimits())
        assert value >= plan.intervals[0].price_export


class TestBudgets:
    def test_energy_budget_pulls_the_setpoint_back_to_the_plan(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=0.5)
        state = ControllerState()
        state.battery(0).interval_deviation_kwh = -0.5  # budget fully spent
        decision, _ = run(plan, measure(grid_w=3000.0), limits, state)
        assert decision.batteries[0].setpoint_w == pytest.approx(0.0, abs=1.0)
        assert "energy_budget" in decision.batteries[0].reason

    def test_energy_budget_still_allows_moving_back_towards_the_plan(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=0.5)
        state = ControllerState()
        state.battery(0).interval_deviation_kwh = -0.5
        # Surplus now: charging restores the planned state of charge.
        decision, _ = run(plan, measure(grid_w=-2000.0), limits, state)
        assert decision.batteries[0].setpoint_w == pytest.approx(2000.0, abs=1.0)

    def test_daily_throughput_budget_stops_the_layer(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(
            min_benefit=0.0, deadband=0.0, daily_extra_throughput=2.0
        )
        state = ControllerState()
        state.battery(0).daily_deviation_kwh = 2.0
        decision, _ = run(plan, measure(grid_w=3000.0), limits, state)
        assert decision.batteries[0].setpoint_w == pytest.approx(0.0, abs=1.0)
        assert "daily_budget" in decision.batteries[0].reason

    def test_budget_of_zero_disables_the_limit(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(
            min_benefit=0.0,
            deadband=0.0,
            energy_budget=0.0,
            daily_extra_throughput=0.0,
        )
        state = ControllerState()
        state.battery(0).interval_deviation_kwh = -99.0
        state.battery(0).daily_deviation_kwh = 99.0
        decision, _ = run(plan, measure(grid_w=3000.0), limits, state)
        assert decision.batteries[0].setpoint_w == pytest.approx(-3000.0, abs=1.0)


class TestStateOfCharge:
    def test_an_empty_battery_is_not_discharged_further(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        decision, _ = run(plan, measure(grid_w=3000.0, soc=21.0))
        assert decision.batteries[0].setpoint_w >= -1.0

    def test_a_full_battery_is_not_charged_further(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        decision, _ = run(plan, measure(grid_w=-3000.0, soc=95.0))
        assert decision.batteries[0].setpoint_w <= 1.0

    def test_the_taper_band_softens_the_limit(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        # soc_min 20 + margin 2 = 22, taper band 3 -> half power at 23.5
        decision, _ = run(plan, measure(grid_w=5000.0, soc=23.5))
        assert -3000.0 < decision.batteries[0].setpoint_w < -1500.0

    def test_the_plan_stays_feasible_even_outside_the_working_range(self):
        # The optimizer may charge to 98 while the fast layer stops at 93.
        plan = make_plan([0.10] * 4, [4000.0] * 4)
        decision, _ = run(plan, measure(grid_w=4500.0, soc=94.0, battery_w=4000.0))
        assert decision.batteries[0].setpoint_w >= 0.0


class TestSafety:
    def test_a_stale_grid_sensor_falls_back_to_the_plan(self):
        plan = make_plan([0.30] * 8, [-1500.0] * 8)
        decision, _ = run(plan, measure(grid_w=3000.0, grid_valid=False))
        assert decision.reason == "sensor_stale"
        assert decision.batteries[0].setpoint_w == pytest.approx(-1500.0)

    def test_a_timestamp_outside_the_plan_falls_back(self):
        plan = make_plan([0.30] * 2, [0.0] * 2)
        decision, _ = run(plan, measure(grid_w=3000.0, timestamp=T0 + 99999))
        assert decision.reason == "plan_expired"

    def test_a_disabled_battery_is_left_alone(self):
        plan = make_plan([0.30] * 8, [-800.0] * 8)
        decision, _ = run(plan, measure(grid_w=3000.0), enabled=[False])
        assert decision.batteries[0].reason == "disabled"
        assert decision.batteries[0].setpoint_w == pytest.approx(-800.0)

    def test_peak_shaving_overrules_the_economics(self):
        # Cheap now and a spike later, so the economics say "hold".
        prices = [0.10, 0.10, 0.60, 0.60]
        socs = [90.0, 90.0, 90.0, 40.0, 40.0]
        plan = make_plan(prices, [0.0] * 4, socs=socs)
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0, max_grid_import=2000.0)
        decision, _ = run(plan, measure(grid_w=6000.0, soc=90.0), limits)
        battery = decision.batteries[0]
        assert battery.setpoint_w == pytest.approx(-4000.0, abs=1.0)
        assert "peak_shave" in battery.reason

    def test_peak_shaving_respects_an_empty_battery(self):
        plan = make_plan([0.10] * 4, [0.0] * 4)
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0, max_grid_import=2000.0)
        decision, _ = run(plan, measure(grid_w=6000.0, soc=20.0), limits)
        assert decision.batteries[0].setpoint_w >= -1.0

    def test_planned_export_is_never_turned_into_import(self):
        """The core guard rail of the default configuration.

        Even when the valuation makes buying look profitable -- cheap now, a
        spike later -- the layer may not create grid import that the plan did
        not ask for. Bulk arbitrage stays the optimizer's job, because only the
        optimizer sees the constraints that made it decide otherwise.
        """
        prices = [0.02, 0.02, 0.60, 0.60]
        socs = [40.0, 40.0, 40.0, 30.0, 30.0]
        plan = make_plan(prices, [0.0] * 4, socs=socs, house_w=[500.0] * 4)
        decision, _ = run(plan, measure(grid_w=500.0, soc=40.0))
        battery = decision.batteries[0]
        # The valuation does say the energy is worth more than 0.02 ...
        assert battery.storage_value > 0.02
        # ... but buying it is refused, because the house is importing.
        assert battery.setpoint_w <= 1.0

    def test_grid_charging_is_refused_by_default(self):
        # A deliberately high fixed storage value would make buying attractive,
        # but the default configuration forbids turning import into storage.
        plan = make_plan([0.10] * 4, [0.0] * 4, house_w=[500.0] * 4)
        limits = PolicyLimits(
            min_benefit=0.0,
            deadband=0.0,
            storage_value_mode="fixed",
            storage_value_fixed=0.50,
        )
        decision, _ = run(plan, measure(grid_w=500.0, soc=40.0), limits)
        assert decision.batteries[0].setpoint_w <= 1.0
        assert "no_grid_charge" in decision.batteries[0].reason

    def test_grid_charging_is_allowed_when_configured(self):
        plan = make_plan([0.10] * 4, [0.0] * 4, house_w=[500.0] * 4)
        limits = PolicyLimits(
            min_benefit=0.0,
            deadband=0.0,
            storage_value_mode="fixed",
            storage_value_fixed=0.50,
            allow_grid_charge=True,
        )
        decision, _ = run(plan, measure(grid_w=500.0, soc=40.0), limits)
        assert decision.batteries[0].setpoint_w > 1000.0


class TestStability:
    def test_the_deadband_suppresses_small_changes(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(min_benefit=0.0, deadband=200.0)
        state = ControllerState()
        state.battery(0).held_setpoint_w = -1000.0
        # house = 100 - (-1000) = 1100 W, so the ideal setpoint is -1100 W:
        # only 100 W away from the standing command.
        decision, _ = run(plan, measure(grid_w=100.0, battery_w=-1000.0), limits, state)
        assert not decision.batteries[0].write
        assert decision.batteries[0].setpoint_w == -1000.0

    def test_the_minimum_command_interval_is_respected(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(
            min_benefit=0.0,
            deadband=50.0,
            min_command_interval=60.0,
            urgent_deviation=5000.0,
        )
        state = ControllerState()
        state.battery(0).held_setpoint_w = -1000.0
        state.battery(0).last_change_ts = T0 + 50
        decision, _ = run(
            plan,
            measure(grid_w=1500.0, battery_w=-1000.0, timestamp=T0 + 60),
            limits,
            state,
        )
        assert not decision.batteries[0].write
        assert "rate_limited" in decision.batteries[0].reason

    def test_an_urgent_step_bypasses_the_minimum_interval(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(
            min_benefit=0.0,
            deadband=50.0,
            min_command_interval=60.0,
            urgent_deviation=1500.0,
        )
        state = ControllerState()
        state.battery(0).held_setpoint_w = 0.0
        state.battery(0).last_change_ts = T0 + 50
        decision, _ = run(
            plan, measure(grid_w=4000.0, timestamp=T0 + 60), limits, state
        )
        assert decision.batteries[0].write
        assert decision.batteries[0].setpoint_w == pytest.approx(-4000.0, abs=1.0)

    def test_the_minimum_benefit_gate_blocks_marginal_corrections(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(min_benefit=0.10, deadband=0.0)
        # 200 W at a 0.18 spread is 0.036 euro/hour, far below 0.10.
        decision, _ = run(plan, measure(grid_w=200.0), limits)
        assert decision.batteries[0].setpoint_w == pytest.approx(0.0)
        assert "below_min_benefit" in decision.batteries[0].reason

    def test_the_ramp_limit_is_applied(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0, max_ramp=500.0)
        state = ControllerState()
        state.battery(0).held_setpoint_w = 0.0
        decision, _ = run(plan, measure(grid_w=4000.0), limits, state)
        assert decision.batteries[0].setpoint_w == pytest.approx(-500.0, abs=1.0)

    def test_the_override_is_released_after_the_quiet_time(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        limits = PolicyLimits(
            min_benefit=0.0, deadband=0.0, release_deviation=100.0, release_time=120.0
        )
        policy = FastControlPolicy(limits)
        state = ControllerState()

        engaged = policy.decide(plan, measure(grid_w=3000.0), state)
        assert engaged.batteries[0].override
        apply_accepted_command(state.battery(0), engaged.batteries[0])

        quiet = policy.decide(plan, measure(grid_w=0.0, timestamp=T0 + 100), state)
        assert quiet.batteries[0].override  # still inside the quiet window
        apply_accepted_command(state.battery(0), quiet.batteries[0])

        released = policy.decide(plan, measure(grid_w=0.0, timestamp=T0 + 400), state)
        assert not released.batteries[0].override
        assert released.batteries[0].setpoint_w == pytest.approx(0.0)


class TestInverterDeadZone:
    def test_a_setpoint_below_the_minimum_power_snaps_out_of_the_dead_zone(self):
        spec = make_spec(minimum_power_w=1000.0)
        plan = make_plan([0.30] * 8, [0.0] * 8, specs=[spec])
        decision, _ = run(plan, measure(grid_w=400.0))
        assert decision.batteries[0].setpoint_w in (0.0, -1000.0)

    def test_a_large_deviation_still_lands_on_the_exact_value(self):
        spec = make_spec(minimum_power_w=1000.0)
        plan = make_plan([0.30] * 8, [0.0] * 8, specs=[spec])
        decision, _ = run(plan, measure(grid_w=3000.0))
        assert decision.batteries[0].setpoint_w == pytest.approx(-3000.0, abs=1.0)

    def test_the_plan_inside_the_dead_zone_stays_reachable(self):
        spec = make_spec(minimum_power_w=1000.0)
        plan = make_plan([0.30] * 8, [-400.0] * 8, house_w=[400.0] * 8, specs=[spec])
        decision, _ = run(plan, measure(grid_w=0.0, battery_w=-400.0))
        assert decision.batteries[0].setpoint_w == pytest.approx(-400.0, abs=1.0)


class TestMultipleBatteries:
    def test_the_first_battery_absorbs_the_deviation(self):
        specs = [make_spec(name="a"), make_spec(name="b")]
        plan = make_plan(
            [0.30] * 4, [[0.0, 0.0]] * 4, specs=specs, house_w=[500.0] * 4
        )
        decision, _ = run(plan, measure(grid_w=3000.0, count=2))
        assert decision.batteries[0].setpoint_w == pytest.approx(-3000.0, abs=1.0)
        assert decision.batteries[1].setpoint_w == pytest.approx(0.0, abs=1.0)

    def test_the_remainder_spills_over_to_the_second(self):
        specs = [
            make_spec(name="a", max_discharge_w=2000.0),
            make_spec(name="b", max_discharge_w=5000.0),
        ]
        plan = make_plan(
            [0.30] * 4, [[0.0, 0.0]] * 4, specs=specs, house_w=[500.0] * 4
        )
        decision, _ = run(plan, measure(grid_w=5000.0, count=2))
        assert decision.batteries[0].setpoint_w == pytest.approx(-2000.0, abs=1.0)
        assert decision.batteries[1].setpoint_w == pytest.approx(-3000.0, abs=1.0)

    def test_a_disabled_battery_keeps_following_the_plan(self):
        specs = [make_spec(name="a"), make_spec(name="b")]
        plan = make_plan(
            [0.30] * 4, [[0.0, -700.0]] * 4, specs=specs, house_w=[700.0] * 4
        )
        decision, _ = run(
            plan, measure(grid_w=3000.0, count=2), enabled=[True, False]
        )
        assert decision.batteries[1].setpoint_w == pytest.approx(-700.0)
        # Battery b is expected to deliver its planned 700 W, so battery a only
        # has to make up the remaining 2300 W of the 3000 W house load.
        assert decision.batteries[0].setpoint_w == pytest.approx(-2300.0, abs=1.0)


class TestAccounting:
    def test_the_deviation_energy_is_integrated(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        policy = FastControlPolicy(PolicyLimits())
        state = ControllerState()
        state.last_tick_ts = T0
        state.plan_created_ts = plan.created_ts
        state.interval_start_ts = plan.intervals[0].start_ts
        # 3600 W away from plan for 60 s is 0.06 kWh.
        policy.account(
            state,
            plan,
            measure(grid_w=0.0, battery_w=-3600.0, timestamp=T0 + 60),
            "2024-01-01",
        )
        assert state.battery(0).interval_deviation_kwh == pytest.approx(-0.06)
        assert state.battery(0).daily_deviation_kwh == pytest.approx(0.06)

    def test_a_new_interval_resets_the_trust_region(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        policy = FastControlPolicy(PolicyLimits())
        state = ControllerState(day_key="2024-01-01")
        state.plan_created_ts = plan.created_ts
        state.interval_start_ts = plan.intervals[0].start_ts
        state.battery(0).interval_deviation_kwh = -0.4
        state.battery(0).daily_deviation_kwh = 1.2
        policy.account(
            state, plan, measure(grid_w=0.0, timestamp=T0 + 1000), "2024-01-01"
        )
        assert state.battery(0).interval_deviation_kwh == 0.0
        assert state.battery(0).daily_deviation_kwh == 1.2  # daily survives

    def test_a_new_day_resets_the_wear_budget(self):
        plan = make_plan([0.30] * 8, [0.0] * 8)
        policy = FastControlPolicy(PolicyLimits())
        state = ControllerState(day_key="2024-01-01", saved_today_eur=1.5)
        state.battery(0).daily_deviation_kwh = 3.0
        policy.account(state, plan, measure(grid_w=0.0), "2024-01-02")
        assert state.battery(0).daily_deviation_kwh == 0.0
        assert state.saved_today_eur == 0.0


class TestInvariants:
    """Properties that must hold for every input, not just the happy path."""

    @pytest.mark.parametrize("grid_w", [-8000.0, -3000.0, -200.0, 0.0, 500.0, 4000.0, 9000.0])
    @pytest.mark.parametrize("soc", [20.0, 22.0, 50.0, 94.0, 95.0])
    @pytest.mark.parametrize("plan_w", [-4000.0, -500.0, 0.0, 500.0, 4000.0])
    def test_the_setpoint_never_leaves_the_inverter_range(self, grid_w, soc, plan_w):
        plan = make_plan([0.30] * 4, [plan_w] * 4)
        decision, _ = run(plan, measure(grid_w=grid_w, soc=soc, battery_w=plan_w))
        setpoint = decision.batteries[0].setpoint_w
        spec = plan.specs[0]
        assert -spec.max_discharge_w - 1 <= setpoint <= spec.max_charge_w + 1

    @pytest.mark.parametrize("grid_w", [-6000.0, 0.0, 6000.0])
    @pytest.mark.parametrize("price", [-0.10, 0.0, 0.05, 0.30, 0.90])
    def test_the_correction_never_costs_more_than_the_plan(self, grid_w, price):
        plan = make_plan([price] * 4, [0.0] * 4)
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0)
        decision, _ = run(plan, measure(grid_w=grid_w), limits)
        battery = decision.batteries[0]
        # The benefit is measured against doing nothing, so it may never be
        # negative: the plan is always inside the feasible set.
        assert battery.benefit_eur_h >= -1e-9


class TestInverterRestore:
    """The optimizer duty-cycles the inverter through a stop moment when it
    wants less than the minimum power. An override has to clear that stop
    moment, and releasing the override has to put it back."""

    def spec_with_stop(self):
        return make_spec(minimum_power_w=1000.0)

    def plan_with_duty_cycle(self):
        spec = self.spec_with_stop()
        plan = make_plan([0.30] * 8, [-400.0] * 8, house_w=[400.0] * 8, specs=[spec])
        plan.intervals[0].batteries[0].mode = "Aan"
        plan.intervals[0].batteries[0].stop_inverter = "2026-01-15 18:07"
        return plan

    def test_engaging_clears_the_stop_moment(self):
        plan = self.plan_with_duty_cycle()
        decision, state = run(plan, measure(grid_w=3400.0, battery_w=-400.0))
        battery = decision.batteries[0]
        assert battery.override
        assert battery.write
        assert battery.stop_inverter == "2000-01-01 00:00:00"
        assert battery.mode == "Aan"
        assert not state.battery(0).stop_inverter_cleared
        assert battery.accepted.stop_inverter_cleared

    def test_the_deadband_may_not_swallow_the_first_clearing_write(self):
        plan = self.plan_with_duty_cycle()
        limits = PolicyLimits(min_benefit=0.0, deadband=5000.0)
        state = ControllerState()
        state.battery(0).held_setpoint_w = -400.0
        decision, _ = run(
            plan, measure(grid_w=3400.0, battery_w=-400.0), limits, state
        )
        assert decision.batteries[0].write
        assert decision.batteries[0].stop_inverter == "2000-01-01 00:00:00"

    def test_releasing_restores_the_published_command(self):
        plan = self.plan_with_duty_cycle()
        limits = PolicyLimits(
            min_benefit=0.0, deadband=0.0, release_deviation=100.0, release_time=0.0
        )
        policy = FastControlPolicy(limits)
        state = ControllerState()
        engaged = policy.decide(plan, measure(grid_w=3400.0, battery_w=-400.0), state)
        apply_accepted_command(state.battery(0), engaged.batteries[0])
        released = policy.decide(
            plan, measure(grid_w=0.0, battery_w=-400.0, timestamp=T0 + 300), state
        )
        battery = released.batteries[0]
        assert not battery.override
        assert battery.stop_inverter == "2026-01-15 18:07"
        assert battery.mode == "Aan"
        assert state.battery(0).stop_inverter_cleared
        assert not battery.accepted.stop_inverter_cleared

    def test_a_stale_sensor_also_restores_the_published_command(self):
        plan = self.plan_with_duty_cycle()
        policy = FastControlPolicy(PolicyLimits(min_benefit=0.0, deadband=0.0))
        state = ControllerState()
        engaged = policy.decide(plan, measure(grid_w=3400.0, battery_w=-400.0), state)
        apply_accepted_command(state.battery(0), engaged.batteries[0])
        fallback = policy.decide(
            plan,
            measure(grid_w=3400.0, timestamp=T0 + 120, grid_valid=False),
            state,
        )
        battery = fallback.batteries[0]
        assert fallback.reason == "sensor_stale"
        assert battery.write
        assert battery.setpoint_w == pytest.approx(-400.0)
        assert battery.stop_inverter == "2026-01-15 18:07"

    def test_nothing_is_touched_when_no_override_ever_happened(self):
        plan = self.plan_with_duty_cycle()
        decision, _ = run(plan, measure(grid_w=0.0, battery_w=-400.0))
        battery = decision.batteries[0]
        assert not battery.override
        assert battery.mode is None
        assert battery.stop_inverter is None


class TestMissingMeasurements:
    def test_an_unknown_state_of_charge_disables_the_override(self):
        """Without a SoC reading the working range cannot be enforced."""
        plan = make_plan([0.30] * 8, [-500.0] * 8, house_w=[500.0] * 8)
        measurement = Measurement(
            timestamp=T0 + 60,
            grid_w=3500.0,
            batteries=[BatteryMeasurement(soc=None, power_w=-500.0, valid=True)],
        )
        decision, _ = run(plan, measurement)
        assert decision.batteries[0].reason == "soc_unknown"
        assert decision.batteries[0].setpoint_w == pytest.approx(-500.0)

    def test_a_missing_power_sensor_falls_back_to_the_held_setpoint(self):
        plan = make_plan([0.30] * 8, [0.0] * 8, house_w=[500.0] * 8)
        state = ControllerState()
        state.battery(0).held_setpoint_w = -1500.0
        measurement = Measurement(
            timestamp=T0 + 60,
            grid_w=2000.0,
            batteries=[BatteryMeasurement(soc=60.0, power_w=None, valid=True)],
        )
        decision, _ = run(plan, measurement, state=state)
        # house = 2000 - (-1500) = 3500 W
        assert decision.house_w == pytest.approx(3500.0)
        assert decision.batteries[0].setpoint_w == pytest.approx(-3500.0, abs=1.0)

    def test_a_stale_power_sensor_falls_back_to_the_held_setpoint(self):
        plan = make_plan([0.30] * 8, [0.0] * 8, house_w=[500.0] * 8)
        state = ControllerState()
        state.battery(0).held_setpoint_w = -1500.0
        measurement = Measurement(
            timestamp=T0 + 60,
            grid_w=2000.0,
            batteries=[BatteryMeasurement(soc=60.0, power_w=-99999.0, valid=False)],
        )
        decision, _ = run(plan, measurement, state=state)
        assert decision.house_w == pytest.approx(3500.0)


class TestScarcity:
    """The valuation has to know whether the plan actually needs the energy.

    A battery whose planned trough sits far above the working lower limit is
    carrying energy the plan has no use for. Refusing to spend that on an
    evening peak, because some later interval is nominally more expensive,
    would throw away the single most valuable thing the layer can do.
    """

    def scenario(self, trough_soc, soc_now):
        # An evening peak: expensive now, slightly more expensive next hour,
        # cheap after that. The plan discharges gently and bottoms out at
        # trough_soc.
        prices = [0.42, 0.46, 0.28, 0.22]
        socs = [soc_now, soc_now - 3, trough_soc, trough_soc, trough_soc]
        return make_plan(
            prices, [-1000.0] * 4, socs=socs, house_w=[1000.0] * 4
        )

    def test_a_battery_with_slack_covers_the_peak(self):
        plan = self.scenario(trough_soc=62.0, soc_now=78.0)
        decision, _ = run(plan, measure(grid_w=2400.0, soc=78.0, battery_w=-1000.0))
        battery = decision.batteries[0]
        assert battery.override
        # house = 2400 - (-1000) = 3400 W, cover all of it.
        assert battery.setpoint_w == pytest.approx(-3400.0, abs=1.0)

    def test_a_battery_that_the_plan_drains_holds_on(self):
        plan = self.scenario(trough_soc=22.0, soc_now=78.0)
        decision, _ = run(plan, measure(grid_w=2400.0, soc=78.0, battery_w=-1000.0))
        battery = decision.batteries[0]
        # Every kWh is spoken for at the 0.46 peak, so do not spend it at 0.32.
        assert battery.setpoint_w >= -1100.0

    def test_the_valuation_moves_monotonically_with_the_slack(self):
        limits = PolicyLimits()
        values = [
            estimate_storage_value(self.scenario(trough, 78.0), T0 + 10, 0, limits)
            for trough in (22.0, 25.0, 28.0, 32.0, 45.0, 62.0)
        ]
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
        assert values[0] > values[-1]

    def test_a_full_horizon_of_slack_lands_on_the_optimizer_valuation(self):
        plan = self.scenario(trough_soc=90.0, soc_now=90.0)
        value = estimate_storage_value(plan, T0 + 10, 0, PolicyLimits())
        expected = plan.price_average * PolicyLimits().round_trip_efficiency
        assert value == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# F2c5: the financial regime of the policy
# ---------------------------------------------------------------------------


def financial_plan(*, powers=(-2000.0,), economics_supported=True, prices=None):
    """A ready-v2 plan with a rectangular profile and a real import price."""
    from dataclasses import replace

    from dao.tests.settlement.test_fast_contract import ready_plan

    plan = ready_plan()
    intervals = []
    for index, row in enumerate(plan.intervals):
        power = powers[index] if index < len(powers) else powers[-1]
        if prices:
            price = prices[index] if index < len(prices) else prices[-1]
        else:
            price = 0.30
        step = replace(row.battery(0), ac_power_w=power, mean_ac_power_w=power,
                       on_power_w=power, active_from_ts=row.start_ts, stop_ts=None)
        intervals.append(replace(row, price_import=price, price_export=0.0,
                                 batteries=[step]))
    plan = replace(plan, intervals=intervals, price_average=0.30)
    if not economics_supported:
        plan = replace(plan, financial_context=replace(plan.financial_context,
                                                       economics_supported=False))
    return plan


def accept_policy_command(plan, state, power, timestamp):
    """Apply canonical synthetic acceptance; a status flag alone proves nothing."""
    from dao.prog.fastctrl.command import (
        ActuationReceipt,
        BatteryCommand,
        OwnerBinding,
        plan_digest,
    )
    from dao.prog.fastctrl.policy import apply_execution_outcome
    from dao.prog.fastctrl.receiver import payload_digest

    now_ms = int(timestamp * 1000)
    binding = OwnerBinding(plan.financial_context.basis.meter_id, "receiver", "generation",
                           "session", 3, "barrier", now_ms + 3_600_000)
    command = BatteryCommand(binding.installation_id, plan.specs[0].name, "session", 3, 7,
                             "command-a", plan_digest(plan), now_ms, now_ms + 1000,
                             now_ms + 3_600_000, power, "Aan", None)
    receipt = ActuationReceipt(command.installation_id, command.battery_name, 3, 7, "command-a",
                               command.plan_digest, now_ms, "accepted", (), "session",
                               payload_digest(command), "accepted", now_ms, "complete")
    assert apply_execution_outcome(state.battery(0), outcome="accepted", pending=command,
                                   receipt=receipt)
    return binding


def g2_aggregate_plan():
    """Aligned profiles as one internal model, retaining the source identities."""
    from dataclasses import replace

    from dao.prog.fastctrl.simulate import aggregate_spec
    from dao.tests.prog.test_fast_consumer import financial_plan as multi_battery_plan

    source = multi_battery_plan(batteries=2, powers=(-1000.0,))
    source.to_dict()  # The source is a valid ready-v2 plan before aggregation.
    rows = [replace(row, batteries=[replace(
        row.battery(0), ac_power_w=sum(b.ac_power_w for b in row.batteries),
        mean_ac_power_w=sum(b.mean_ac_power_w for b in row.batteries),
        on_power_w=sum(b.on_power_w for b in row.batteries))]) for row in source.intervals]
    return replace(source, specs=[aggregate_spec(source.specs)], intervals=rows)


G2_POLICY_ROUTES = ("account_measured", "account_unknown", "decide_measured", "decide_unknown",
                    "release", "all_plan", "expired")


def g2_policy_call(policy, route, plan, state, sample, binding=None):
    from dataclasses import replace

    if route.endswith("unknown"):
        sample = replace(sample, batteries=[BatteryMeasurement(soc=50, power_w=None)])
    if route.startswith("account_"):
        return policy.account(state, plan, sample, "synthetic", owner_binding=binding)
    if route == "release":
        return policy.release(plan, sample, state, "g2_release", owner_binding=binding)
    if route == "all_plan":
        return policy._all_plan(plan, sample, state, "g2_fallback", plan.interval_at(sample.timestamp),
                                owner_binding=binding)
    if route == "expired":
        sample = replace(sample, timestamp=plan.horizon_end_ts)
    return policy.decide(plan, sample, state, owner_binding=binding)


@pytest.mark.parametrize("route", G2_POLICY_ROUTES)
def test_g2_unbound_aggregate_needs_no_execution_digest(route):
    from dataclasses import replace

    plan = g2_aggregate_plan()
    start = plan.intervals[0].start_ts
    state = ControllerState(last_tick_ts=start, day_key="synthetic")
    controller = state.battery(0)
    controller.held_setpoint_w = -4000.0  # Virtual/history only, never a measured baseline.
    sample = measure(grid_w=4000, battery_w=-1000, timestamp=start + 60)
    policy = FastControlPolicy(PolicyLimits())
    failure = None
    try:
        decision = g2_policy_call(policy, route, plan, state, sample)
    except ValueError as exc:
        failure = str(exc)
    assert failure is None, f"unbound {route} requested an execution digest: {failure}"
    assert controller.held_setpoint_w == -4000.0
    assert controller._current_accepted_command is None
    assert controller.last_accepted_command is None
    assert controller.execution_status == "unknown"
    if route.startswith("account_"):
        # -1000 W measured versus a -2000 W plan for one minute: +1/60 kWh.
        expected = 1 / 60 if route == "account_measured" else 0
        assert controller.interval_deviation_kwh == pytest.approx(expected)
        assert controller.daily_deviation_kwh == pytest.approx(expected)
        return
    assert decision.benefit_eur_h == 0.0
    battery = decision.batteries[0]
    assert battery.write is (route != "expired")
    if route == "decide_measured":
        assert decision.house_w == 5000.0
        assert battery.override is True and battery.benefit_eur_h > 0
        # The simulator's existing virtual acceptance still grants no live right.
        apply_accepted_command(controller, battery)
        again = policy.decide(plan, replace(sample, timestamp=start + 61), state)
        assert again.batteries[0].write is True
        assert again.benefit_eur_h == 0.0
        assert controller._current_accepted_command is None
        assert controller.last_accepted_command is None
        assert controller.execution_status == "unknown"
    elif route != "expired":
        assert battery.setpoint_w == -2000.0
        assert battery.override is False
        if route == "decide_unknown":
            assert decision.reason == "execution_unknown"
    else:
        assert decision.reason == "plan_expired"


@pytest.mark.parametrize("route", G2_POLICY_ROUTES)
def test_g2_binding_keeps_aggregate_validation_strict_without_runtime_command(route):
    from dao.prog.fastctrl.command import OwnerBinding

    plan = g2_aggregate_plan()
    start = plan.intervals[0].start_ts
    binding = OwnerBinding(plan.financial_context.basis.meter_id, "receiver", "generation",
                           "session", 3, "barrier", (plan.horizon_end_ts + 60) * 1000)
    state = ControllerState(last_tick_ts=start, day_key="synthetic")
    sample = measure(grid_w=4000, battery_w=-1000, timestamp=start + 60)
    assert state.battery(0)._current_accepted_command is None
    with pytest.raises(ValueError, match="^Terminal references must identify exactly the plan batteries$"):
        g2_policy_call(FastControlPolicy(PolicyLimits()), route, plan, state, sample, binding)
    assert state.battery(0)._current_accepted_command is None
    assert state.battery(0).last_accepted_command is None


def test_g2_aggregate_remains_unserializable_without_binding():
    with pytest.raises(ValueError, match="^Terminal references must identify exactly the plan batteries$"):
        g2_aggregate_plan().to_dict()


def test_g2_bound_canonical_noop_still_requires_the_exact_digest():
    from dataclasses import replace

    from dao.prog.fastctrl.command import plan_digest
    from dao.prog.fastctrl.policy import current_execution_command

    plan = financial_plan()
    start = plan.intervals[0].start_ts
    state = ControllerState()
    binding = accept_policy_command(plan, state, -2000, start)
    command = state.battery(0).last_accepted_command
    sample = measure(grid_w=0, battery_w=-2000, timestamp=start + 60)
    policy = FastControlPolicy(PolicyLimits())
    assert policy.decide(plan, sample, state, owner_binding=binding).batteries[0].write is False
    assert current_execution_command(state.battery(0), binding=binding,
                                     now_ms=(start + 60) * 1000,
                                     expected_digest=plan_digest(plan)) == command
    changed = replace(plan, price_average=0.31)
    assert current_execution_command(state.battery(0), binding=binding,
                                     now_ms=(start + 60) * 1000,
                                     expected_digest=plan_digest(changed)) is None
    follow = policy.decide(changed, sample, state, owner_binding=binding)
    assert follow.batteries[0].write is True
    assert follow.batteries[0].setpoint_w == -2000.0
    assert state.battery(0).last_accepted_command == command


@pytest.mark.parametrize("power", [-100.25, 100.25])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("path", ["normal", "grid_stale", "disabled", "soc_unknown", "release"])
def test_i4_fractional_profile_is_not_rounded_by_policy(power, partial, path):
    from dataclasses import replace

    plan = financial_plan(powers=(power,), economics_supported=False)
    now = plan.intervals[0].start_ts
    if partial:
        for row in plan.intervals:
            step = row.battery(0)
            step.stop_ts = row.start_ts + 600
            step.ac_power_w = step.mean_ac_power_w = power * 2 / 3
    policy = FastControlPolicy(PolicyLimits())
    sample = measure(grid_w=-power, timestamp=now, soc=None if path == "soc_unknown" else 50)
    if path == "release":
        decision = policy.release(plan, sample, ControllerState(), "mode_off")
    else:
        decision = policy.decide(plan, replace(sample, grid_valid=path != "grid_stale"),
                                 ControllerState(), enabled=[path != "disabled"])
    battery = decision.batteries[0]
    assert battery.setpoint_w == power
    assert battery.plan_w == power
    assert battery.write is True
    assert battery.override is False


def test_f2c5_a_v2_plan_proposes_the_plan_power_instead_of_assuming_it():
    """The optimizer wrote nothing, so the plan power is a command."""
    plan = financial_plan()
    now = plan.intervals[0].start_ts
    state = ControllerState()
    decision = FastControlPolicy(PolicyLimits()).decide(
        plan, measure(grid_w=2000.0, timestamp=now), state)
    battery = decision.batteries[0]
    assert battery.setpoint_w == -2000
    assert battery.write is True
    assert battery.override is False
    # The in-force setpoint is not reconciled without an acceptance.
    assert state.battery(0).held_setpoint_w == 0.0


def test_f2c5_the_legacy_route_still_assumes_the_plan_power():
    from dataclasses import replace

    plan = replace(financial_plan(), format_version=1, financial_context=None)
    now = plan.intervals[0].start_ts
    state = ControllerState()
    decision = FastControlPolicy(PolicyLimits()).decide(
        plan, measure(grid_w=2000.0, timestamp=now), state)
    battery = decision.batteries[0]
    assert battery.write is False
    assert state.battery(0).held_setpoint_w == -2000.0


def test_f2c5_a_missing_acceptance_is_not_hidden_by_the_deadband():
    plan = financial_plan()
    now = plan.intervals[0].start_ts
    state = ControllerState()
    policy = FastControlPolicy(PolicyLimits())
    first = policy.decide(plan, measure(grid_w=2000.0, timestamp=now), state)
    assert first.batteries[0].write is True
    # Nothing was accepted, so the second tick proposes again.
    second = policy.decide(plan, measure(grid_w=2000.0, timestamp=now + 60), state)
    assert second.batteries[0].write is True
    assert second.batteries[0].setpoint_w == -2000


def test_f2c5_unsupported_economics_forbid_a_deviation_but_not_the_plan():
    plan = financial_plan(economics_supported=False)
    now = plan.intervals[0].start_ts
    state = ControllerState()
    policy = FastControlPolicy(PolicyLimits(min_benefit=0.0))
    decision = policy.decide(plan, measure(grid_w=5000.0, timestamp=now), state)
    battery = decision.batteries[0]
    assert battery.override is False
    assert battery.setpoint_w == -2000
    assert "economics_unsupported" in battery.reason
    # The plan is still followed: it is proposed as a real command.
    assert battery.write is True


def test_f2c5_only_an_accepted_v2_segment_books_a_benefit():
    plan = financial_plan()
    now = plan.intervals[0].start_ts
    state = ControllerState()
    policy = FastControlPolicy(PolicyLimits(min_benefit=0.0))
    decision = policy.decide(plan, measure(grid_w=5000.0, timestamp=now), state)
    assert decision.batteries[0].override is True
    # Nothing is proven yet, so the deviation books no euro claim.
    assert decision.benefit_eur_h == 0.0
    binding = accept_policy_command(plan, state, -5000, now)
    proven = policy.decide(plan, measure(grid_w=5000.0, timestamp=now + 60), state,
                           owner_binding=binding)
    assert proven.batteries[0].override is True
    assert proven.benefit_eur_h > 0.0


def test_f2c5_a_v2_segment_without_an_accepted_command_books_no_deviation():
    from dataclasses import replace

    plan = financial_plan()
    now = plan.intervals[0].start_ts
    state = ControllerState()
    policy = FastControlPolicy(PolicyLimits())
    # No usable battery sensor: the policy would fall back on the held setpoint.
    sample = Measurement(timestamp=now + 100, grid_w=0.0,
                          batteries=[BatteryMeasurement(soc=50.0, power_w=None, valid=True)])
    policy.account(state, plan, sample, "synthetic")
    policy.account(state, plan, replace(sample, timestamp=now + 200), "synthetic")
    # The held setpoint is not a sensor value: the segment stays unknown.
    assert state.battery(0).interval_deviation_kwh == 0.0
    binding = accept_policy_command(plan, state, -1000, now + 200)
    policy.account(state, plan, replace(sample, timestamp=now + 300), "synthetic",
                   owner_binding=binding)
    assert state.battery(0).interval_deviation_kwh != 0.0


def test_f2c5_the_legacy_route_still_falls_back_on_the_held_setpoint():
    from dataclasses import replace

    plan = replace(financial_plan(), format_version=1, financial_context=None)
    now = plan.intervals[0].start_ts
    state = ControllerState()
    policy = FastControlPolicy(PolicyLimits())
    sample = Measurement(timestamp=now + 100, grid_w=0.0,
                          batteries=[BatteryMeasurement(soc=50.0, power_w=None, valid=True)])
    policy.account(state, plan, sample, "synthetic")
    policy.account(state, plan, replace(sample, timestamp=now + 200), "synthetic")
    assert state.battery(0).interval_deviation_kwh != 0.0


def test_f2c5_apply_execution_outcome_follows_the_decision_table():
    from dataclasses import replace

    from dao.prog.fastctrl import policy as policy_module
    from dao.prog.fastctrl.command import ActuationReceipt, BatteryCommand
    from dao.prog.fastctrl.receiver import payload_digest

    # Deliberate assertion RED, never a collection/import error.
    assert hasattr(policy_module, "apply_execution_outcome"), (
        "F2c5 execution outcome application missing")
    apply_execution_outcome = policy_module.apply_execution_outcome

    command = BatteryCommand(installation_id="sandbox", battery_name="accu",
                             owner_session_id="session", owner_epoch=3, sequence=7,
                             command_id="command-a", plan_digest="a" * 64,
                             issued_at_ms=1000, execute_before_ms=2000,
                             hold_until_ms=2000, setpoint_w=-2000, mode="Aan",
                             stop_at_ms=None)
    receipt = ActuationReceipt(installation_id="sandbox", battery_name="accu",
                               owner_epoch=3, sequence=7, command_id="command-a",
                               plan_digest="a" * 64, accepted_ts=1500, status="accepted",
                               helper_results=(), owner_session_id="session",
                               payload_digest=payload_digest(command), reason="accepted",
                               observed_at_ms=1500, effect="complete")
    state = ControllerState().battery(0)
    state.pending_command = command

    # Unknown: nothing changes but the status.
    assert apply_execution_outcome(state, outcome="unknown") is True
    assert state.execution_status == "unknown"
    assert state.pending_command is command
    assert state.recovery_required is False

    # Unconfirmed: the uncertainty, the duty and the pending command remain.
    assert apply_execution_outcome(state, outcome="unconfirmed", receipt=receipt,
                                    pending=command) is True
    assert state.execution_status == "unconfirmed"
    assert state.recovery_required is True
    assert state.pending_command is command
    assert state.last_accepted_command is None

    # A receipt that does not match the stored command mutates nothing.
    wrong = replace(receipt, command_id="command-b")
    assert apply_execution_outcome(state, outcome="accepted", receipt=wrong,
                                    pending=command) is False
    assert state.execution_status == "unconfirmed"
    assert state.recovery_required is True
    assert state.pending_command is command
    assert state.last_command_w == 0.0

    # Another session is not this command either.
    other_session = replace(receipt, owner_session_id="other-session")
    assert apply_execution_outcome(state, outcome="accepted", receipt=other_session,
                                    pending=command) is False
    assert state.execution_status == "unconfirmed"
    assert state.last_accepted_command is None

    # Another payload is not this command either.
    other_payload = replace(receipt, payload_digest="b" * 64)
    assert apply_execution_outcome(state, outcome="accepted", receipt=other_payload,
                                    pending=command) is False
    assert state.execution_status == "unconfirmed"
    assert state.last_accepted_command is None

    # The in-force facts are never touched by an outcome.
    assert apply_execution_outcome(state, outcome="accepted", receipt=receipt,
                                    pending=command) is True
    assert state.execution_status == "accepted"
    assert state.recovery_required is True
    assert state.pending_command is None
    assert state.last_accepted_command is command
    assert state.last_command_w == -2000
    assert state.last_command_ts == 1.5
    assert state.stop_inverter_cleared is True
    assert state.held_setpoint_w == 0.0

    # Only an exact safe barrier ends the recovery duty.
    assert apply_execution_outcome(state, outcome="stopped") is True
    assert state.execution_status == "stopped"
    assert state.recovery_required is False
    assert state.last_accepted_command is command


def test_f2c5_the_controller_state_survives_a_roundtrip():
    from dao.prog.fastctrl.command import BatteryCommand

    command = BatteryCommand(installation_id="sandbox", battery_name="accu",
                             owner_session_id="session", owner_epoch=3, sequence=7,
                             command_id="command-a", plan_digest="a" * 64,
                             issued_at_ms=1000, execute_before_ms=2000,
                             hold_until_ms=2000, setpoint_w=-2000, mode="Aan",
                             stop_at_ms=None)
    state = ControllerState().battery(0)
    state.execution_status = "accepted"
    state.recovery_required = True
    state.last_accepted_command = command
    restored = ControllerState.from_dict({"batteries": [state.to_dict()]}).battery(0)
    assert restored.execution_status == "accepted"
    assert restored.recovery_required is True
    assert restored.last_accepted_command == command


def test_f2c5_unsupported_economics_forbid_the_peak_shaving_override():
    """Without economics there is no deviation at all, not even a peak shave."""
    prices = [0.10, 0.10, 0.60]
    limits = PolicyLimits(min_benefit=0.0, deadband=0.0, max_grid_import=2000.0)

    supported = financial_plan(powers=(0.0,), prices=prices)
    now = supported.intervals[0].start_ts
    decision = FastControlPolicy(limits).decide(
        supported, measure(grid_w=6000.0, timestamp=now), ControllerState())
    battery = decision.batteries[0]
    # The positive control: with economics the peak shave is a real deviation.
    assert battery.setpoint_w == pytest.approx(-4000.0, abs=1.0)
    assert "peak_shave" in battery.reason
    assert battery.override is True

    unsupported = financial_plan(powers=(0.0,), prices=prices, economics_supported=False)
    decision = FastControlPolicy(limits).decide(
        unsupported, measure(grid_w=6000.0, timestamp=now), ControllerState())
    battery = decision.batteries[0]
    # Unsupported economics forbid every deviation: the plan is followed.
    assert battery.override is False
    assert battery.setpoint_w == 0
    assert "peak_shave" not in battery.reason
    assert "economics_unsupported" in battery.reason


def test_f2c5_the_deadband_never_hides_a_v2_profile_transition():
    """The deadband guards an override; a plan transition is a real command."""
    plan = financial_plan(powers=(-2000.0, -1950.0))
    now = plan.intervals[1].start_ts
    state = ControllerState()
    controller = state.battery(0)
    # A confirmed state: the in-force setpoint is the accepted one.
    controller.held_setpoint_w = -2000.0
    controller.execution_status = "accepted"
    controller.last_change_ts = now - 10
    limits = PolicyLimits(min_benefit=0.0, deadband=150.0, min_command_interval=60.0)
    decision = FastControlPolicy(limits).decide(
        plan, measure(grid_w=1950.0, timestamp=now), state)
    battery = decision.batteries[0]
    # The change is far below the deadband, but it is a new plan power.
    assert battery.setpoint_w == -1950
    assert battery.write is True
    assert battery.override is False


def test_f2c5_a_v2_proposal_is_not_an_in_force_change():
    """`last_change_ts` belongs to an accepted command, not to a proposal."""
    from dataclasses import replace

    plan = financial_plan()
    now = plan.intervals[0].start_ts
    limits = PolicyLimits(min_benefit=0.0, deadband=0.0, min_command_interval=0.0)
    state = ControllerState()
    decision = FastControlPolicy(limits).decide(
        plan, measure(grid_w=2000.0, timestamp=now), state)
    # The proposal is a real command, but it is not an in-force change.
    assert decision.batteries[0].write is True
    assert state.battery(0).last_change_ts == 0.0

    legacy = replace(plan, format_version=1, financial_context=None)
    legacy_state = ControllerState()
    legacy_decision = FastControlPolicy(limits).decide(
        legacy, measure(grid_w=2000.0, timestamp=now), legacy_state)
    # The v1 route keeps its behaviour exactly.
    assert legacy_decision.batteries[0].write is True
    assert legacy_state.battery(0).last_change_ts == now


def test_f2c5_unsupported_economics_end_an_existing_override_at_once():
    """Without economics no deviation may persist: it ends in the same tick."""
    plan = financial_plan(economics_supported=False)
    now = plan.intervals[0].start_ts
    state = ControllerState()
    controller = state.battery(0)
    # An existing, young override: the layer is deviating right now.
    controller.override_active = True
    controller.quiet_since = now
    controller.held_setpoint_w = -4000.0
    limits = PolicyLimits(min_benefit=0.0, deadband=0.0, min_command_interval=0.0,
                          release_time=120.0)
    decision = FastControlPolicy(limits).decide(
        plan, measure(grid_w=5000.0, timestamp=now), state)
    battery = decision.batteries[0]
    # The deviation ends now: the plan power is the target, without hysteresis.
    assert battery.override is False
    assert battery.setpoint_w == -2000
    assert battery.write is True
    assert controller.override_active is False


def test_f2c5_the_v1_route_keeps_its_release_hysteresis():
    """The v1 route is unchanged: a young override survives one more tick."""
    from dataclasses import replace

    plan = replace(financial_plan(), format_version=1, financial_context=None)
    now = plan.intervals[0].start_ts
    state = ControllerState()
    controller = state.battery(0)
    controller.override_active = True
    controller.quiet_since = now
    controller.held_setpoint_w = -4000.0
    limits = PolicyLimits(min_benefit=0.0, deadband=0.0, min_command_interval=0.0,
                          release_time=120.0)
    decision = FastControlPolicy(limits).decide(
        plan, measure(grid_w=5000.0, timestamp=now), state)
    battery = decision.batteries[0]
    # The v1 release hysteresis is untouched: the override stays.
    assert battery.override is True


def test_f2c5_a_v2_plan_without_a_confirmed_state_always_proposes():
    """Without a confirmed state a non-zero plan power is always a proposal."""
    plan = financial_plan(powers=(-100.0,))
    now = plan.intervals[0].start_ts
    state = ControllerState()
    controller = state.battery(0)
    # The last command was withdrawn: there is no confirmed in-force state.
    controller.held_setpoint_w = -100.0
    controller.execution_status = "stopped"
    limits = PolicyLimits(min_benefit=0.0, deadband=150.0, min_command_interval=0.0)
    decision = FastControlPolicy(limits).decide(
        plan, measure(grid_w=100.0, timestamp=now), state)
    battery = decision.batteries[0]
    # The plan power is a real command again, not an already in-force fact.
    assert battery.setpoint_w == -100
    assert battery.write is True


def test_f2c5_the_fallback_also_proposes_a_v2_plan_without_a_confirmed_state():
    """The fallback path follows the same rule: the threshold hides nothing."""
    plan = financial_plan(powers=(-100.0,))
    now = plan.intervals[0].start_ts
    state = ControllerState()
    controller = state.battery(0)
    controller.held_setpoint_w = -100.0
    controller.execution_status = "stopped"
    limits = PolicyLimits(min_benefit=0.0, deadband=150.0, min_command_interval=0.0)
    decision = FastControlPolicy(limits).decide(
        plan, measure(grid_w=100.0, timestamp=now, soc=None), state)
    battery = decision.batteries[0]
    assert battery.setpoint_w == -100
    assert battery.write is True


def test_f2c5_the_v1_route_keeps_its_deadband_in_the_fallback():
    """The v1 fallback is unchanged: the threshold still applies there."""
    from dataclasses import replace

    plan = replace(financial_plan(powers=(-100.0,)), format_version=1,
                   financial_context=None)
    now = plan.intervals[0].start_ts
    state = ControllerState()
    controller = state.battery(0)
    controller.held_setpoint_w = -100.0
    limits = PolicyLimits(min_benefit=0.0, deadband=150.0, min_command_interval=0.0)
    decision = FastControlPolicy(limits).decide(
        plan, measure(grid_w=100.0, timestamp=now, soc=None), state)
    battery = decision.batteries[0]
    # The v1 route keeps its behaviour exactly: no command, the threshold wins.
    assert battery.write is False


def test_f2c5_the_v2_fallback_delivers_the_plan_command_despite_the_rate_limit():
    """In v2 the legacy restore/rate-limit path may not hide a plan-following."""
    plan = financial_plan(powers=(-100.0,))
    now = plan.intervals[0].start_ts
    state = ControllerState()
    controller = state.battery(0)
    # An existing deviation with a pending legacy restore and a recent change:
    # the v1 rate-limit path would swallow the plan command.
    controller.override_active = True
    controller.restore_pending = True
    controller.last_change_ts = now - 1
    controller.held_setpoint_w = -100.0
    limits = PolicyLimits(min_benefit=0.0, deadband=150.0, min_command_interval=60.0)
    decision = FastControlPolicy(limits).decide(
        plan, measure(grid_w=100.0, timestamp=now, soc=None), state)
    battery = decision.batteries[0]
    # The plan command is delivered: in v2 the threshold and the rate limit
    # guard an override only, never a plan-following.
    assert battery.setpoint_w == -100
    assert battery.write is True
    assert battery.override is False
    assert "+rate_limited" not in battery.reason


def test_f2c5_the_v1_route_keeps_its_legacy_rate_limit_in_the_fallback():
    """The v1 fallback keeps the legacy restore/rate-limit path."""
    from dataclasses import replace

    plan = replace(financial_plan(powers=(-100.0,)), format_version=1,
                   financial_context=None)
    now = plan.intervals[0].start_ts
    state = ControllerState()
    controller = state.battery(0)
    controller.override_active = True
    controller.restore_pending = True
    controller.last_change_ts = now - 1
    controller.held_setpoint_w = -100.0
    limits = PolicyLimits(min_benefit=0.0, deadband=150.0, min_command_interval=60.0)
    decision = FastControlPolicy(limits).decide(
        plan, measure(grid_w=100.0, timestamp=now, soc=None), state)
    battery = decision.batteries[0]
    # The v1 route keeps its behaviour exactly: the rate limit wins.
    assert battery.write is False
    assert "+rate_limited" in battery.reason
