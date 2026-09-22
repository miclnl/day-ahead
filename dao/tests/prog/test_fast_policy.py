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
    estimate_storage_value,
)

T0 = 1_700_000_000


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
        state.battery(0).last_command_w = -1000.0
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
        state.battery(0).last_command_w = -1000.0
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
        state.battery(0).last_command_w = 0.0
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
        state.battery(0).last_command_w = 0.0
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

        quiet = policy.decide(plan, measure(grid_w=0.0, timestamp=T0 + 100), state)
        assert quiet.batteries[0].override  # still inside the quiet window

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
        assert state.battery(0).stop_inverter_cleared

    def test_the_deadband_may_not_swallow_the_first_clearing_write(self):
        plan = self.plan_with_duty_cycle()
        limits = PolicyLimits(min_benefit=0.0, deadband=5000.0)
        state = ControllerState()
        state.battery(0).last_command_w = -400.0
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
        policy.decide(plan, measure(grid_w=3400.0, battery_w=-400.0), state)
        released = policy.decide(
            plan, measure(grid_w=0.0, battery_w=-400.0, timestamp=T0 + 300), state
        )
        battery = released.batteries[0]
        assert not battery.override
        assert battery.stop_inverter == "2026-01-15 18:07"
        assert battery.mode == "Aan"
        assert not state.battery(0).stop_inverter_cleared

    def test_a_stale_sensor_also_restores_the_published_command(self):
        plan = self.plan_with_duty_cycle()
        policy = FastControlPolicy(PolicyLimits(min_benefit=0.0, deadband=0.0))
        state = ControllerState()
        policy.decide(plan, measure(grid_w=3400.0, battery_w=-400.0), state)
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

    def test_a_missing_power_sensor_falls_back_to_the_last_command(self):
        plan = make_plan([0.30] * 8, [0.0] * 8, house_w=[500.0] * 8)
        state = ControllerState()
        state.battery(0).last_command_w = -1500.0
        measurement = Measurement(
            timestamp=T0 + 60,
            grid_w=2000.0,
            batteries=[BatteryMeasurement(soc=60.0, power_w=None, valid=True)],
        )
        decision, _ = run(plan, measurement, state=state)
        # house = 2000 - (-1500) = 3500 W
        assert decision.house_w == pytest.approx(3500.0)
        assert decision.batteries[0].setpoint_w == pytest.approx(-3500.0, abs=1.0)

    def test_a_stale_power_sensor_falls_back_to_the_last_command(self):
        plan = make_plan([0.30] * 8, [0.0] * 8, house_w=[500.0] * 8)
        state = ControllerState()
        state.battery(0).last_command_w = -1500.0
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
