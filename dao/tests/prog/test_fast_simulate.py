"""Tests for the backtest.

The backtest is the tool the saving estimate comes from, so its battery model
and its cost accounting have to be right, and the comparison has to be fair.
"""

import pytest

from dao.prog.fastctrl.plan import (
    BatteryPlanStep,
    BatterySpec,
    FastPlan,
    PlanInterval,
)
from dao.prog.fastctrl.policy import PolicyLimits
from dao.prog.fastctrl.simulate import (
    BatteryModel,
    Sample,
    aggregate_spec,
    compare,
    simulate,
    synthetic_case,
)

T0 = 1_700_000_000


def make_spec(**overrides) -> BatterySpec:
    defaults = dict(
        name="accu",
        capacity_kwh=10.0,
        max_charge_w=5000.0,
        max_discharge_w=5000.0,
        soc_min=20.0,
        soc_max=95.0,
        cycle_cost=0.01,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    defaults.update(overrides)
    return BatterySpec(**defaults)


def flat_plan(spec, price=0.30, export=0.10, battery_w=0.0, hours=2, house_w=500.0):
    intervals = [
        PlanInterval(
            start_ts=T0 + i * 900,
            end_ts=T0 + (i + 1) * 900,
            price_import=price,
            price_export=export,
            grid_w=house_w + battery_w,
            house_w=house_w,
            batteries=[
                BatteryPlanStep(ac_power_w=battery_w, soc_begin=50.0, soc_end=50.0)
            ],
        )
        for i in range(hours * 4)
    ]
    return FastPlan(
        created_ts=T0,
        interval_s=900,
        specs=[spec],
        intervals=intervals,
        price_average=price,
    )


class TestBatteryModel:
    def test_charging_respects_the_efficiency_chain(self):
        spec = make_spec(charge_efficiency=0.90)
        model = BatteryModel(spec, soc=50.0, inverter_efficiency=1.0)
        power, dc_kwh = model.step(1000.0, 1.0)
        assert power == 1000.0
        assert dc_kwh == pytest.approx(1.0)
        # 1 kWh DC times 0.9 into 10 kWh is 9 percentage points.
        assert model.soc == pytest.approx(59.0)

    def test_discharging_draws_more_than_it_delivers(self):
        spec = make_spec(discharge_efficiency=0.90)
        model = BatteryModel(spec, soc=50.0, inverter_efficiency=1.0)
        power, dc_kwh = model.step(-900.0, 1.0)
        assert power == -900.0
        # 0.9 kWh AC out of a 90 percent efficient battery costs 1 kWh of cells.
        assert model.soc == pytest.approx(40.0)
        assert dc_kwh == pytest.approx(0.9)

    def test_it_stops_at_the_upper_limit(self):
        model = BatteryModel(make_spec(), soc=94.0, inverter_efficiency=1.0)
        power, _ = model.step(5000.0, 1.0)
        assert model.soc == pytest.approx(95.0)
        assert 0 < power < 5000.0

    def test_it_stops_at_the_lower_limit(self):
        model = BatteryModel(make_spec(), soc=21.0, inverter_efficiency=1.0)
        power, _ = model.step(-5000.0, 1.0)
        assert model.soc == pytest.approx(20.0)
        assert -5000.0 < power < 0

    def test_it_never_exceeds_the_inverter_rating(self):
        model = BatteryModel(make_spec(max_charge_w=3000.0), soc=50.0)
        power, _ = model.step(9000.0, 0.01)
        assert power == 3000.0

    def test_an_empty_battery_delivers_nothing(self):
        model = BatteryModel(make_spec(), soc=20.0)
        power, dc_kwh = model.step(-3000.0, 0.25)
        assert power == 0.0
        assert dc_kwh == 0.0


class TestSimulation:
    def test_a_constant_load_reproduces_the_hand_computed_cost(self):
        spec = make_spec()
        plan = flat_plan(spec, price=0.40, export=0.10, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 1000.0) for s in range(61)]
        result = simulate(
            plan, samples, PolicyLimits(), 50.0, "base", use_fast_layer=False
        )
        # 1000 W for one hour at 0.40 euro/kWh.
        assert result.energy_cost == pytest.approx(0.40, rel=1e-6)
        assert result.import_kwh == pytest.approx(1.0, rel=1e-6)
        assert result.throughput_kwh == 0.0

    def test_the_fast_layer_covers_the_load_from_the_battery(self):
        spec = make_spec()
        plan = flat_plan(spec, price=0.40, export=0.10, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 1000.0) for s in range(61)]
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=5.0)
        result = simulate(
            plan, samples, limits, 80.0, "fast", use_fast_layer=True
        )
        assert result.import_kwh < 0.1
        assert result.energy_cost < 0.05
        assert result.throughput_kwh > 0.9

    def test_the_comparison_prices_the_leftover_energy(self):
        """Ending emptier may not look free."""
        spec = make_spec()
        plan = flat_plan(spec, price=0.40, export=0.10, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 1000.0) for s in range(61)]
        result = compare(
            plan,
            samples,
            PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=5.0),
            80.0,
        )
        assert result.fast.soc_end < result.baseline.soc_end
        assert result.fast.terminal_value < result.baseline.terminal_value
        # The raw energy cost drops, but almost all of it is paid back by the
        # energy the battery gave up.
        raw = result.baseline.energy_cost - result.fast.energy_cost
        assert raw > 0.35
        assert result.saving < 0.05 * raw

    def test_a_flat_tariff_leaves_nothing_to_win(self):
        """Self-consumption only pays when prices move or the spread is real.

        With a flat tariff over the whole horizon, discharging now and
        discharging later are worth exactly the same, so the fast layer can
        only lose the wear it causes. The backtest must show that honestly
        instead of booking the avoided import as profit.
        """
        spec = make_spec(cycle_cost=0.02)
        plan = flat_plan(spec, price=0.40, export=0.10, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 1000.0) for s in range(61)]
        result = compare(
            plan,
            samples,
            PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=5.0),
            80.0,
        )
        assert result.saving < 0.0
        assert abs(result.saving) == pytest.approx(
            result.fast.wear_cost, abs=0.03
        )

    def test_the_cost_identity_holds(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=1)
        result = compare(plan, samples, PolicyLimits(), soc)
        for outcome in (result.baseline, result.fast):
            assert outcome.total_cost == pytest.approx(
                outcome.energy_cost + outcome.wear_cost - outcome.terminal_value
            )

    def test_wear_is_charged(self):
        spec = make_spec(cycle_cost=0.05)
        plan = flat_plan(spec, price=0.40, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 2000.0) for s in range(61)]
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=5.0)
        result = simulate(plan, samples, limits, 80.0, "fast", use_fast_layer=True)
        assert result.wear_cost == pytest.approx(
            0.05 * result.throughput_kwh, rel=1e-9
        )

    def test_cycles_are_counted_over_twice_the_capacity(self):
        spec = make_spec(capacity_kwh=10.0)
        plan = flat_plan(spec, hours=1)
        samples = [Sample(T0 + s * 60, 0.0) for s in range(61)]
        result = simulate(
            plan, samples, PolicyLimits(), 50.0, "base", use_fast_layer=False
        )
        result.throughput_kwh = 20.0
        assert result.cycles == pytest.approx(1.0)

    def test_too_few_samples_is_an_error(self):
        plan = flat_plan(make_spec())
        with pytest.raises(ValueError):
            simulate(
                plan, [Sample(T0, 0.0)], PolicyLimits(), 50.0, "x", use_fast_layer=False
            )

    def test_a_plan_without_a_battery_is_an_error(self):
        plan = FastPlan(created_ts=T0, interval_s=900, specs=[], intervals=[])
        with pytest.raises(ValueError):
            simulate(plan, [], PolicyLimits(), 50.0, "x", use_fast_layer=False)


class TestAggregateSpec:
    def test_a_single_battery_is_passed_through(self):
        spec = make_spec()
        assert aggregate_spec([spec]) is spec

    def test_power_and_capacity_add_up(self):
        first = make_spec(capacity_kwh=10.0, max_charge_w=3000.0)
        second = make_spec(capacity_kwh=5.0, max_charge_w=2000.0)
        merged = aggregate_spec([first, second])
        assert merged.capacity_kwh == 15.0
        assert merged.max_charge_w == 5000.0

    def test_efficiencies_are_capacity_weighted(self):
        first = make_spec(capacity_kwh=10.0, charge_efficiency=0.90)
        second = make_spec(capacity_kwh=30.0, charge_efficiency=1.00)
        merged = aggregate_spec([first, second])
        assert merged.charge_efficiency == pytest.approx(0.975)

    def test_an_empty_list_is_an_error(self):
        with pytest.raises(ValueError):
            aggregate_spec([])


class TestSyntheticCase:
    def test_it_produces_a_usable_scenario(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=1)
        assert len(plan.intervals) == 96
        assert len(samples) > 1400
        assert 0 <= soc <= 100
        assert all(i.end_ts > i.start_ts for i in plan.intervals)

    def test_the_appliance_peaks_are_invisible_to_the_plan(self):
        """That invisibility is the whole reason the fast layer exists."""
        spec = make_spec()
        plan, samples, _ = synthetic_case(spec, days=1)
        deviations = []
        for sample in samples[:-1]:
            interval = plan.interval_at(sample.timestamp)
            if interval is not None:
                deviations.append(abs(sample.house_w - interval.house_w))
        assert max(deviations) > 2000.0

    def test_the_fast_layer_saves_money_on_it(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=2)
        result = compare(plan, samples, PolicyLimits(), soc)
        assert result.saving > 0.0
        assert result.fast.import_kwh < result.baseline.import_kwh
        assert result.fast.export_kwh < result.baseline.export_kwh

    def test_a_tight_energy_budget_reduces_both_saving_and_wear(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=2)
        loose = compare(plan, samples, PolicyLimits(energy_budget=2.0), soc)
        tight = compare(plan, samples, PolicyLimits(energy_budget=0.05), soc)
        assert tight.saving < loose.saving
        assert tight.fast.throughput_kwh < loose.fast.throughput_kwh

    def test_a_high_minimum_benefit_reduces_the_number_of_writes(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=2)
        eager = compare(plan, samples, PolicyLimits(min_benefit=0.0), soc)
        lazy = compare(plan, samples, PolicyLimits(min_benefit=0.30), soc)
        assert lazy.fast.writes < eager.fast.writes

    def test_the_report_renders(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=1)
        comparison = compare(plan, samples, PolicyLimits(), soc)
        report = comparison.report()
        assert "Besparing" in report
        assert "Totale kosten" in report
        assert len(comparison.daily_table().splitlines()) >= 3
