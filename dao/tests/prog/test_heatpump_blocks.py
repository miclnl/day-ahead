"""Required heat, not the available horizon, determines power-mode blocks."""

import datetime as dt
import logging
import sys
from types import SimpleNamespace

import mip
import pandas as pd
import pytest

from dao.prog import day_ahead as da
from dao.prog.config.models.devices.boiler import BoilerDisabled, BoilerEnabled
from dao.prog.config.models.devices.heating import HeatingEnabled
from dao.tests.settlement.test_optimization_bonus import ts
from dao.tests.settlement.test_optimization_producer import capture as capture


@pytest.fixture
def power_hp_model(capture, monkeypatch):
    """Reuse the offline capture boundary; build the real HP producer and CBC."""
    instance = capture["instance"]
    start = capture["locals"]["start_dt"]
    batteries = instance.battery_options
    instance.meteo = SimpleNamespace(calc_graaddagen=lambda **kwargs: 1.0)

    ha_writes = []

    def forbidden_write(*args, **kwargs):
        ha_writes.append((args, kwargs))
        pytest.fail("heat-pump producer attempted an external write")

    for name in ("set_value", "call_service", "turn_on", "turn_off", "set_entity_value",
                 "set_entity_state", "set_entity_option", "select_option", "save_df", "export_fast_plan"):
        monkeypatch.setattr(instance, name, forbidden_write)

    def build(*, prices, heat_kwh, minimum_hours, adjustment, demand="eco", run_hours=None,
              boiler=None, boiler_states=None, stages=None, with_batteries=False):
        instance.battery_options = batteries if with_batteries else []
        configured_stages = stages if stages is not None else [{"max_power": 1000, "cop": 4}]
        instance.heating_options = HeatingEnabled.model_validate({
            "adjustment": adjustment,
            "degree days factor": heat_kwh,
            "min run length": minimum_hours,
            "stages": configured_stages,
            "entity hp heat demand": "sensor.synthetic_hp_demand",
            **({"entity hp heat produced": "sensor.synthetic_hp_heat"} if run_hours is not None else {}),
        })
        states = {"sensor.synthetic_hp_demand": demand, "sensor.synthetic_hp_heat": "0"}
        states.update({battery.entity_actual_level: "50" for battery in instance.battery_options})
        if boiler is False:
            instance.boiler_options = BoilerDisabled()
        elif boiler is not None:
            instance.boiler_options = BoilerEnabled.model_validate({
                "entity actual temp.": "sensor.synthetic_boiler_temp",
                "entity setpoint": "input_number.synthetic_boiler_setpoint",
                "entity hysterese": "input_number.synthetic_boiler_hysteresis",
                "switch entity": "switch.synthetic_boiler",
                "cooling rate": 0.5, "heating allowed below": 50,
                "cop": 4, "elec. power": 1000, **boiler,
            })
            states.update({"sensor.synthetic_boiler_temp": "40",
                            "input_number.synthetic_boiler_setpoint": "50",
                            "input_number.synthetic_boiler_hysteresis": "5"})
        states.update(boiler_states or {})

        def get_state(entity_id):
            if entity_id not in states:
                pytest.fail(f"unexpected synthetic HA read: {entity_id}")
            return SimpleNamespace(state=states[entity_id])

        monkeypatch.setattr(instance, "get_state", get_state)
        monkeypatch.setattr(instance, "get_float", da.DaCalc.get_float.__get__(instance))
        steps_per_hour = 3600 // instance.interval_s
        tariffs = [price for price in prices for _ in range(steps_per_hour)]
        grid_start = int(start.timestamp()) // instance.interval_s * instance.interval_s
        stamps = [grid_start + instance.interval_s * u for u in range(len(tariffs))]
        dates = [dt.datetime.fromtimestamp(at) for at in stamps]
        price_data = pd.DataFrame({
            "time": dates, "da_ex": tariffs, "da_cons": tariffs,
            "da_prod": [0.0] * len(tariffs), "da_bonus": [0.0] * len(tariffs),
        })
        forecast = pd.DataFrame({
            "time": stamps, "tijd": dates,
            "glob_rad": [0.0] * len(tariffs), "temp": [5.0] * len(tariffs),
        })
        monkeypatch.setattr(da, "Report", lambda *args: SimpleNamespace(
            get_price_data=lambda *a, **k: price_data.copy(),
            get_heatpump_run_hours=lambda entity: run_hours))
        instance.db_da = SimpleNamespace(get_prognose_data=lambda **kwargs: forecast.copy())
        found = {}
        bind_cost = da.bind_day_ahead_cost
        add_sos, sos = mip.Model.add_sos, []

        def record_sos(model, members, sos_type):
            sos.append((tuple((var.idx, weight) for var, weight in members), sos_type))
            return add_sos(model, members, sos_type)

        class Captured(Exception):
            pass

        def capture_cost(model, **kwargs):
            bind_cost(model, **kwargs)
            found.update(sys._getframe(1).f_locals)
            raise Captured

        with monkeypatch.context() as spies:
            spies.setattr(mip.Model, "add_sos", record_sos)
            spies.setattr(da, "bind_day_ahead_cost", capture_cost)
            with pytest.raises(Captured):
                instance.calc_optimum(_start_dt=start)
        model = found["model"]
        assert model.solver_name == mip.CBC
        assert found["U"] == len(tariffs)
        assert found["degree_days_factor"] == heat_kwh
        assert found["heat_produced"] == 0
        assert found["hour_fraction"] == [
            (stamps[0] + instance.interval_s - int(start.timestamp())) / 3600,
            *([1 / steps_per_hour] * (len(tariffs) - 1)),
        ]
        assert [stage["max_power"] for stage in found["hp_stages"]] == [
            0, *(stage["max_power"] for stage in configured_stages),
        ]
        model.verbose = 0
        model.objective = mip.minimize(found["cost"])
        found["ha_writes"] = ha_writes
        found["sos"] = sos
        return found

    return build


@pytest.mark.parametrize("capture", [{"hourly": True}, {}], indirect=True, ids=["hour", "quarter"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_five_needed_hours_do_not_force_nine(power_hp_model, adjustment):
    """Horizon-sized intermediate blocks force four unnecessary expensive hours."""
    local = power_hp_model(
        prices=[0.10, 0.60, 0.60, 0.10, 0.10, 0.60, 0.60, 0.60, 0.10, 0.10, 0.60, 0.60],
        heat_kwh=20, minimum_hours=1, adjustment=adjustment,
    )
    model = local["model"]
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    assert sum(var.x for var in local["h_hp"]) == pytest.approx(20)
    assert sum(var.x for var in local["c_hp"]) == pytest.approx(5)
    assert sum(flag.x * hours for flag, hours in zip(
        local["hp_on"], local["hour_fraction"], strict=True)) == pytest.approx(5)
    assert [flag.x for flag in local["hp_on"]] == pytest.approx([
        int(price == 0.10) for price in local["pl"]])
    assert local["cost"].x == pytest.approx(0.50)
    assert local["ha_writes"] == []


@pytest.mark.parametrize("capture", [{"hourly": True}, {}], indirect=True, ids=["hour", "quarter"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_single_block_can_exceed_minimum(power_hp_model, adjustment):
    """One five-hour minimum block must still carry all six required hours."""
    local = power_hp_model(
        prices=[0.60, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.60],
        heat_kwh=24, minimum_hours=5, adjustment=adjustment,
    )
    model = local["model"]
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    assert sum(var.x for var in local["h_hp"]) == pytest.approx(24)
    assert sum(var.x for var in local["c_hp"]) == pytest.approx(6)
    assert sum(flag.x * hours for flag, hours in zip(
        local["hp_on"], local["hour_fraction"], strict=True)) == pytest.approx(6)
    assert [flag.x for flag in local["hp_on"]] == pytest.approx([
        int(price == 0.10) for price in local["pl"]])
    assert local["cost"].x == pytest.approx(0.60)
    assert local["ha_writes"] == []


@pytest.mark.parametrize("capture", [{"hourly": True}, {}], indirect=True, ids=["hour", "quarter"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_last_block_keeps_remaining_whole_hours(power_hp_model, adjustment):
    """Splitting five remaining hours into nominal half-hours loses an hourly slot."""
    local = power_hp_model(
        prices=[0.10, 0.60, 0.60, 0.10, 0.10, 0.60, 0.60, 0.60, 0.10, 0.10, 0.10, 0.60],
        heat_kwh=24, minimum_hours=1, adjustment=adjustment,
    )
    model = local["model"]
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    assert sum(var.x for var in local["h_hp"]) == pytest.approx(24)
    assert sum(var.x for var in local["c_hp"]) == pytest.approx(6)
    assert sum(flag.x * hours for flag, hours in zip(
        local["hp_on"], local["hour_fraction"], strict=True)) == pytest.approx(6)
    assert [flag.x for flag in local["hp_on"]] == pytest.approx([
        int(price == 0.10) for price in local["pl"]])
    assert local["cost"].x == pytest.approx(0.60)
    assert local["ha_writes"] == []


def assert_schedule(local, *, heat_kwh, runtime_hours, hp_cost):
    """One native solve; inspect physical output and tariff selection, not status alone."""
    model = local["model"]
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    assert sum(var.x for var in local["h_hp"]) == pytest.approx(heat_kwh)
    assert sum(var.x for var in local["c_hp"]) == pytest.approx(runtime_hours)
    assert sum(flag.x * hours for flag, hours in zip(
        local["hp_on"], local["hour_fraction"], strict=True)) == pytest.approx(runtime_hours)
    assert sum(var.x * price for var, price in zip(
        local["c_hp"], local["pl"], strict=True)) == pytest.approx(hp_cost)
    assert local["ha_writes"] == []


@pytest.mark.parametrize("capture,heat,runtime,cost", [
    pytest.param({"hourly": True}, 12, 3, 0.30, id="hour"),
    pytest.param({}, 9, 2.25, 0.225, id="quarter"),
], indirect=["capture"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_fractional_required_duration_is_not_floored(power_hp_model, adjustment, heat, runtime, cost):
    """Flooring 2.25 required hours loses heat; nominal counts also lose the last quarter."""
    local = power_hp_model(
        prices=[0.10, 0.60, 0.60, 0.60, 0.10, 0.10, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        heat_kwh=9, minimum_hours=1, adjustment=adjustment,
    )
    assert local["heat_needed"] == 9
    assert_schedule(local, heat_kwh=heat, runtime_hours=runtime, hp_cost=cost)
    assert all(flag.x == 0 for flag, price in zip(local["hp_on"], local["pl"], strict=True)
               if price == 0.60)
    assert local["cost"].x == pytest.approx(cost)


@pytest.mark.parametrize("capture", [{"start": ts(minute=7) + 30}], indirect=True)
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_partial_first_quarter_keeps_heat_feasible(power_hp_model, adjustment):
    """Four nominal quarters starting now contain only 52.5 minutes, not one hour."""
    local = power_hp_model(
        prices=[0.60, 0.10, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        heat_kwh=4, minimum_hours=1, adjustment=adjustment, demand="max",
    )
    assert local["hour_fraction"][0] == 0.125
    assert local["heat_needed"] == 4
    assert_schedule(local, heat_kwh=4.5, runtime_hours=1.125, hp_cost=0.55)
    assert [flag.x for flag in local["hp_on"]] == pytest.approx([1] * 5 + [0] * 27)
    assert local["cost"].x == pytest.approx(0.55)


@pytest.mark.parametrize("capture", [{"hourly": True, "start": ts(minute=30)}], indirect=True)
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_partial_first_hour_keeps_minimum(power_hp_model, adjustment):
    """A mandatory half-hour first slot needs the next whole slot to meet a one-hour minimum."""
    local = power_hp_model(
        prices=[0.60, 0.10, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        heat_kwh=4, minimum_hours=1, adjustment=adjustment, demand="max",
    )
    assert_schedule(local, heat_kwh=6, runtime_hours=1.5, hp_cost=0.40)
    assert [flag.x for flag in local["hp_on"]] == pytest.approx([1, 1, 0, 0, 0, 0, 0, 0])
    assert local["cost"].x == pytest.approx(0.40)


@pytest.mark.parametrize("capture,heat,runtime,cost", [
    pytest.param({"hourly": True}, 28, 7, 0.70, id="hour"),
    pytest.param({"start": ts(minute=7) + 30}, 24.5, 6.125, 0.6125, id="partial-quarter"),
], indirect=["capture"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_running_block_keeps_actual_remaining_duration(power_hp_model, adjustment, heat, runtime, cost):
    """A running 1.5-hour block owes 1.5 real hours; truncating either span loses heat."""
    local = power_hp_model(
        prices=[0.10, 0.10, 0.60, 0.60, 0.60, 0.10, 0.10, 0.10, 0.10, 0.10, 0.60, 0.60],
        heat_kwh=24, minimum_hours=3, adjustment=adjustment, run_hours=1.5,
    )
    assert_schedule(local, heat_kwh=heat, runtime_hours=runtime, hp_cost=cost)
    active = [[u for u, flag in enumerate(row) if flag.x > 0.5] for row in local["hp_bl_on"]]
    assert active[0][0] == 0
    assert len(active[0]) == (2 if local["self"].interval_s == 3600 else 7)
    assert active[0][-1] < active[1][0]
    assert all(row == list(range(row[0], row[-1] + 1)) for row in active)
    assert all(flag.x == 0 for flag, price in zip(local["hp_on"], local["pl"], strict=True)
               if price == 0.60)
    assert local["cost"].x == pytest.approx(cost)


@pytest.mark.parametrize("capture", [{"hourly": True}, {}], indirect=True, ids=["hour", "quarter"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
@pytest.mark.parametrize("demand,run_hours,first,cost", [
    ("eco", None, False, 0.10), ("max", None, True, 0.60), ("off", None, False, 0.10),
    ("eco", 0.5, True, 0.60), ("max", 0.5, True, 0.60), ("off", 0.5, False, 0.10),
    ("eco", 2, False, 0.10), ("max", 2, True, 0.60), ("off", 2, False, 0.10),
])
def test_heat_demand_preserves_running_or_price_selected_start(
    power_hp_model, adjustment, demand, run_hours, first, cost,
):
    """Only max or an unfinished eco run forces start0; off must leave the first hour idle."""
    local = power_hp_model(
        prices=[0.60, 0.10, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        heat_kwh=4, minimum_hours=1, adjustment=adjustment, demand=demand, run_hours=run_hours,
    )
    assert_schedule(local, heat_kwh=4, runtime_hours=1, hp_cost=cost)
    steps = 3600 // local["self"].interval_s
    expected_hours = [int(first), int(not first), 0, 0, 0, 0, 0, 0]
    assert [flag.x for flag in local["hp_on"]] == pytest.approx(
        [flag for flag in expected_hours for _ in range(steps)])
    assert local["cost"].x == pytest.approx(cost)


@pytest.mark.parametrize("capture", [{"hourly": True}, {}], indirect=True, ids=["hour", "quarter"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_boiler_blocks_otherwise_cheapest_span(power_hp_model, adjustment):
    """Space heating cannot take the cheap hour already used by the same pump's boiler."""
    local = power_hp_model(
        prices=[0.10, 0.20, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        heat_kwh=4, minimum_hours=1, adjustment=adjustment, boiler={},
    )
    assert_schedule(local, heat_kwh=4, runtime_hours=1, hp_cost=0.20)
    steps = 3600 // local["self"].interval_s
    assert [flag.x for flag in local["boiler_on"]] == pytest.approx([1] * steps + [0] * (7 * steps))
    assert [flag.x for flag in local["hp_on"]] == pytest.approx([0] * steps + [1] * steps + [0] * (6 * steps))
    assert all(hp.x + boiler.x <= 1 for hp, boiler in zip(
        local["hp_on"], local["boiler_on"], strict=True))


@pytest.mark.parametrize("capture,prices,power", [
    pytest.param({"hourly": True}, [0.10] * 8, 125, id="hour"),
    pytest.param({}, [0.10] * 3, 300, id="quarter"),
], indirect=["capture"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_zero_available_hours_keeps_no_block_route(power_hp_model, adjustment, prices, power):
    """An almost horizon-long boiler reservation leaves zero heuristic HP availability."""
    local = power_hp_model(
        prices=prices, heat_kwh=4, minimum_hours=1, adjustment=adjustment,
        boiler={"elec. power": power},
    )
    assert local["hours_avail"] == 0
    assert local["boiler_int"] == local["U"]
    assert local["blocks_num"] == 0
    assert local["heat_needed"] == 0  # Existing physical-cap route, not missing demand.
    assert_schedule(local, heat_kwh=0, runtime_hours=0, hp_cost=0)
    assert [flag.x for flag in local["hp_on"]] == pytest.approx([0] * local["U"])
    # The original assertions establish only heuristic availability, not execution.
    required = 0.1 + 1.1 * (200 * 4.2 + 100 * 0.5) * 10 / (3600 * 4)
    capacity = power * local["self"].interval_s / 3_600_000
    full, tail = divmod(required, capacity)
    expected = [capacity] * int(full) + [tail]
    expected += [0] * (local["U"] - len(expected))
    assert_boiler_output(local, start_index=0, required_kwh=required, expected=expected)


def assert_boiler_output(local, *, start_index, required_kwh, expected):
    """Independent physical/cost checks, not an assertion against the profile helper."""
    assert sum(var.x for var in local["c_b"]) == pytest.approx(required_kwh)
    assert [var.x for var in local["c_b"]] == pytest.approx(expected)
    assert [flag.x for flag in local["boiler_st"]] == pytest.approx([
        int(u == start_index) for u in range(local["U"])])
    assert [flag.x for flag in local["boiler_on"]] == pytest.approx([
        int(energy > 0) for energy in expected])
    power = local["self"].boiler_options.elec_power
    for u, var in enumerate(local["c_b"]):
        capacity = power * (local["interval_ts"][u] + local["self"].interval_s
                            - local["ts"][u]) / 3_600_000
        assert var.ub == pytest.approx(capacity)
        assert -1e-7 <= var.x <= capacity + 1e-7
    assert local["est_needed_intv"][start_index] == sum(energy > 0 for energy in expected)
    boiler_cost = sum(energy * price for energy, price in zip(expected, local["pl"], strict=True))
    assert local["est_elec_cost"][start_index] == pytest.approx(boiler_cost)
    hp_cost = sum(var.x * price for var, price in zip(local["c_hp"], local["pl"], strict=True))
    heat_capacity = 1.1 * (local["self"].boiler_options.volume * 4.2 + 100 * 0.5)
    stored_kwh = required_kwh - local["U"] * local["boiler_cooling"] * heat_capacity / (
        3600 * local["cop_boiler"])
    assert local["cost"].x == pytest.approx(boiler_cost + hp_cost - stored_kwh * local["p_avg"])
    assert all(hp.x + boiler.x <= 1 + 1e-7 for hp, boiler in zip(
        local["hp_on"], local["boiler_on"], strict=True))
    assert local["ha_writes"] == []


def test_long_boiler_reservation_is_not_full_boiler_execution(power_hp_model):
    """Native feasibility alone previously accepted only 0.15 of 0.779861 kWh."""
    local = power_hp_model(
        prices=[0.10] * 3, heat_kwh=4, minimum_hours=1, adjustment="power",
        boiler={"elec. power": 300},
    )
    assert local["boiler_int"] == local["U"] == 12
    model = local["model"]
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    required = 0.1 + 1.1 * (200 * 4.2 + 100 * 0.5) * 10 / (3600 * 4)
    assert required == pytest.approx(0.7798611111111111)
    assert sum(var.x for var in local["c_b"]) == pytest.approx(0.7798611111111111)
    assert_boiler_output(local, start_index=0, required_kwh=required,
                         expected=[0.075] * 10 + [0.0298611111111111, 0])
    assert sum(flag.x for flag in local["boiler_on"]) == 11


@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
@pytest.mark.parametrize("start_index,power,full,tail", [
    (0, 300, 4, 0.003958333333333333),
    (12, 300, 5, 0.0309375),
    (0, 1000, 1, 0.05395833333333333),
    (12, 1000, 1, 0.1559375),
])
def test_selected_boiler_start_binds_every_output_interval(
    power_hp_model, adjustment, start_index, power, full, tail,
):
    """A late output interval must use the selected start's length, not its own length."""
    local = power_hp_model(
        prices=[0.60, 0.10, 0.20, 0.40, 0.30, 0.50], heat_kwh=4, minimum_hours=1,
        adjustment=adjustment, boiler={"elec. power": power},
        boiler_states={"sensor.synthetic_boiler_temp": "47"},
    )
    model = local["model"]
    model += local["boiler_st"][start_index] == 1
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    required = 0.1 + 1.1 * (200 * 4.2 + 100 * 0.5) * (50 - (47 - 0.5 * start_index / 4)) / (3600 * 4)
    expected = [0] * start_index + [power / 4000] * full + [tail]
    expected += [0] * (local["U"] - len(expected))
    assert_boiler_output(local, start_index=start_index, required_kwh=required, expected=expected)
    assert sum(var.x for var in local["h_hp"]) == pytest.approx(4)


@pytest.mark.parametrize("interval,energy", [(10, 0), (11, 0.001)], ids=["underfill", "extra-tail"])
def test_boiler_underfill_and_extra_tail_are_infeasible(power_hp_model, interval, energy):
    """Missing tail energy or energy after completion must contradict the selected cycle."""
    local = power_hp_model(
        prices=[0.10] * 3, heat_kwh=4, minimum_hours=1, adjustment="power",
        boiler={"elec. power": 300},
    )
    model = local["model"]
    model += local["boiler_st"][0] == 1
    model += local["c_b"][interval] == energy
    assert model.optimize() == mip.OptimizationStatus.INFEASIBLE
    assert model.num_solutions == 0
    assert local["ha_writes"] == []


@pytest.mark.parametrize("prices,power,volume,temp,cooling,start_index,expected", [
    pytest.param([0.10] * 3, 1000, 200, 47, 0.5, 10,
                 [0] * 10 + [0.25, 0.1389409722222222], id="last-complete-start"),
    pytest.param([0.10] * 3, 300, 237.44588744588742, 40, 0, 0,
                 [0.075] * 12, id="exact-horizon-end"),
])
def test_boiler_start_at_last_complete_boundary_is_allowed(
    power_hp_model, prices, power, volume, temp, cooling, start_index, expected,
):
    """Requiring one extra nominal slot incorrectly excludes the final complete cycle."""
    local = power_hp_model(
        prices=prices, heat_kwh=4, minimum_hours=1, adjustment="power",
        boiler={"elec. power": power, "volume": volume, "cooling rate": cooling},
        boiler_states={"sensor.synthetic_boiler_temp": str(temp)},
    )
    model = local["model"]
    model += local["boiler_st"][start_index] == 1
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    # The exact-end tank has 1152 kJ/K: 0.1 + 1152 * 10 / (3600 * 4) = 0.9 kWh.
    required = 0.1 + 1.1 * (volume * 4.2 + 100 * 0.5) * (50 - (temp - cooling * start_index / 4)) / (3600 * 4)
    assert_boiler_output(local, start_index=start_index, required_kwh=required, expected=expected)
    assert local["boiler_int"] <= local["U"]


@pytest.mark.parametrize("capture,prices,power,cooling,temp,allowed", [
    pytest.param({}, [0.10] * 3, 200, 0.5, 40, 50, id="too-long"),
    pytest.param({}, [0.10] * 3, 200, 0, 40, 50, id="zero-cooling-too-long"),
    pytest.param({"start": ts(minute=7) + 30}, [0.10] * 3, 270, 0.5, 40, 50,
                 id="partial-first-shortfall"),
    pytest.param({}, [0.10] * 3, 300, 0.5, 47, 46, id="only-premature-starts-fit"),
], indirect=["capture"])
def test_boiler_without_complete_cycle_is_rejected(
    power_hp_model, monkeypatch, prices, power, cooling, temp, allowed,
):
    """An active request cannot be silently dropped or solved with a clipped cycle."""
    def forbidden(*args, **kwargs):
        pytest.fail("incomplete boiler cycle reached solve/cost publication")

    monkeypatch.setattr(mip.Model, "optimize", forbidden)
    monkeypatch.setattr(da, "bind_day_ahead_cost", forbidden)
    with pytest.raises(da.OptimizationInputError, match="^boiler_cycle_exceeds_horizon$"):
        power_hp_model(
            prices=prices, heat_kwh=4, minimum_hours=1, adjustment="power",
            boiler={"elec. power": power, "cooling rate": cooling, "heating allowed below": allowed},
            boiler_states={"sensor.synthetic_boiler_temp": str(temp)},
        )


@pytest.mark.parametrize("boiler,states", [
    pytest.param(None, {}, id="absent"),
    pytest.param(False, {}, id="disabled"),
    pytest.param({"entity boiler enabled": "input_boolean.synthetic_boiler_enabled"},
                 {"input_boolean.synthetic_boiler_enabled": "off"}, id="switched-off"),
    pytest.param({}, {"sensor.synthetic_boiler_temp": "55"}, id="no-heat"),
    pytest.param({"cooling rate": 0}, {"sensor.synthetic_boiler_temp": "50"},
                 id="zero-cooling-no-heat"),
])
def test_boiler_disabled_and_no_heat_bind_zero(power_hp_model, boiler, states):
    """Inactive boiler flags and energy must all be zero, not unused free columns."""
    local = power_hp_model(
        prices=[0.10] * 3, heat_kwh=4, minimum_hours=1, adjustment="power",
        boiler=boiler, boiler_states=states,
    )
    model = local["model"]
    for name in ("boiler_st", "boiler_on", "c_b"):
        forced = model.add_constr(mip.xsum(local[name]) >= 0.001)
        assert model.optimize() == mip.OptimizationStatus.INFEASIBLE
        model.remove(forced)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    for name in ("boiler_st", "boiler_on", "c_b"):
        assert [var.x for var in local[name]] == [0] * local["U"]
    assert local["ha_writes"] == []


@pytest.mark.parametrize("cooling", [0, 0.5])
def test_boiler_instant_start_keeps_only_start_zero(power_hp_model, cooling):
    """Instant-start cannot be delayed by a cheaper later interval, even without cooling."""
    local = power_hp_model(
        prices=[0.60, 0.10, 0.20, 0.30], heat_kwh=4, minimum_hours=1, adjustment="power",
        boiler={"entity instant start": "input_boolean.synthetic_boiler_instant", "cooling rate": cooling},
        boiler_states={"sensor.synthetic_boiler_temp": "47", "input_boolean.synthetic_boiler_instant": "on"},
    )
    model = local["model"]
    forced = model.add_constr(local["boiler_st"][4] == 1)
    assert model.optimize() == mip.OptimizationStatus.INFEASIBLE
    model.remove(forced)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    required = 0.1 + 1.1 * (200 * 4.2 + 100 * 0.5) * 3 / (3600 * 4)
    assert_boiler_output(local, start_index=0, required_kwh=required,
                         expected=[0.25, 0.05395833333333333] + [0] * 14)


def test_boiler_zero_cooling_keeps_temperature_window_unlimited(power_hp_model):
    """With heat allowed but no cooling, no elapsed-time boundary may forbid a later start."""
    local = power_hp_model(
        prices=[0.60, 0.20, 0.10, 0.30], heat_kwh=4, minimum_hours=1, adjustment="power",
        boiler={"cooling rate": 0}, boiler_states={"sensor.synthetic_boiler_temp": "47"},
    )
    assert local["boiler_start"] == 8  # First of the equally cheap, fully fitting starts.
    model = local["model"]
    model += local["boiler_st"][8] == 1
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    required = 0.1 + 1.1 * (200 * 4.2 + 100 * 0.5) * 3 / (3600 * 4)
    assert_boiler_output(local, start_index=8, required_kwh=required,
                         expected=[0] * 8 + [0.25, 0.05395833333333333] + [0] * 6)


@pytest.mark.parametrize("capture,prices,power,expected", [
    pytest.param({"start": ts(minute=7) + 30}, [0.10] * 3, 300,
                 [0.0375] + [0.075] * 9 + [0.0673611111111111, 0], id="partial-quarter"),
    pytest.param({"hourly": True, "start": ts(minute=30)}, [0.10] * 8, 125,
                 [0.0625] + [0.125] * 5 + [0.0923611111111111, 0], id="partial-hour"),
], indirect=["capture"])
def test_boiler_partial_first_interval_keeps_partial_tail(power_hp_model, prices, power, expected):
    """Actual first-slot capacity and a partial tail must survive producer binding."""
    local = power_hp_model(
        prices=prices, heat_kwh=4, minimum_hours=1, adjustment="power",
        boiler={"elec. power": power},
    )
    model = local["model"]
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    required = 0.1 + 1.1 * (200 * 4.2 + 100 * 0.5) * 10 / (3600 * 4)
    assert_boiler_output(local, start_index=0, required_kwh=required, expected=expected)


@pytest.mark.parametrize("capture,heat,runtime,cost", [
    pytest.param({"hourly": True}, 32, 8, 1.30, id="hour"),
    pytest.param({}, 30, 7.5, 1.00, id="quarter"),
], indirect=["capture"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_high_runtime_keeps_physical_no_block_route(power_hp_model, adjustment, heat, runtime, cost):
    """The >80% route must still meet demand without enforcing artificial minimum blocks."""
    local = power_hp_model(
        prices=[0.10] * 7 + [0.60], heat_kwh=heat, minimum_hours=5, adjustment=adjustment,
    )
    assert local["blocks_num"] == 0
    assert_schedule(local, heat_kwh=heat, runtime_hours=runtime, hp_cost=cost)
    steps = 3600 // local["self"].interval_s
    assert [flag.x for flag in local["hp_on"][:7 * steps]] == pytest.approx([1] * (7 * steps))
    assert local["cost"].x == pytest.approx(cost)


@pytest.mark.parametrize("capture", [{"hourly": True}], indirect=True)
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_hourly_thirty_kwh_native_readback(power_hp_model, adjustment, caplog):
    """A two-point SOS2 must not admit half a stage with a binary on flag of one."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        local = power_hp_model(
            prices=[0.10] * 7 + [0.60], heat_kwh=30, minimum_hours=5, adjustment=adjustment,
        )
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.WARNING, "Er ontbreken voor een aantal uur meteogegevens "
         "(en/of dynamische prijzen)\ncontroleer of alle gegevens zijn opgehaald"),
    ]
    assert local["blocks_num"] == 0
    assert local["heat_needed"] == 30
    assert local["hours_avail"] == local["U"] == 8
    assert local["min_run_length"] == 5
    assert local["hour_fraction"] == [1] * 8
    assert [stage["cop"] for stage in local["hp_stages"][1:]] == [4]
    assert all(flag.var_type == mip.BINARY for flag in local["hp_on"])
    assert_schedule(local, heat_kwh=32, runtime_hours=8, hp_cost=1.30)
    assert [flag.x for flag in local["hp_on"]] == pytest.approx([1] * 8)
    assert [tuple(var.x for var in row) for row in local["hp_s_w"]] == [(0, 1)] * 8
    assert local["cost"].x == pytest.approx(1.30)


@pytest.mark.parametrize("capture", [{"hourly": True, "battery": {
    "charge stages": [{"power": 2000, "efficiency": 1}],
    "discharge stages": [{"power": 2000, "efficiency": 1}],
}}], indirect=True)
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_two_stage_hp_omits_only_redundant_sos2(power_hp_model, adjustment):
    """Only HP registration is redundant; even two-member battery SOS2s must survive."""
    local = power_hp_model(
        prices=[0.10] * 7 + [0.60], heat_kwh=30, minimum_hours=5, adjustment=adjustment,
        with_batteries=True,
    )
    assert local["S"] == 2
    expected = [
        (tuple((var.idx, power) for var, power in zip(row, (0, 2), strict=True)), 2)
        for b in range(2) for u in range(8)
        for row in (local["ac_to_dc_w"][b][u], local["ac_from_dc_w"][b][u])
    ]
    assert local["sos"] == expected
    model = local["model"]
    for name in ("dc_to_bat", "dc_from_bat"):
        for row in local[name]:
            for var in row:
                model += var == 0
    assert_schedule(local, heat_kwh=32, runtime_hours=8, hp_cost=1.30)


@pytest.mark.parametrize("capture", [
    pytest.param({"hourly": True}, id="hour"),
    pytest.param({}, id="quarter"),
    pytest.param({"start": ts(minute=7) + 30}, id="partial-quarter"),
], indirect=True)
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_multistage_hp_interpolation_remains_continuous(power_hp_model, adjustment):
    """Discretizing positive stages loses the valid 1500 W / 5 kW heat operating point."""
    local = power_hp_model(
        prices=[0.10] * 7 + [0.60], heat_kwh=30, minimum_hours=5, adjustment=adjustment,
        stages=[{"max_power": 1000, "cop": 4}, {"max_power": 2000, "cop": 3}],
    )
    model, hours = local["model"], local["hour_fraction"][0]
    assert local["blocks_num"] == 0
    assert local["S"] == 3
    assert local["sos"] == [
        (tuple((var.idx, power) for var, power in zip(row, (0, 1000, 2000), strict=True)), 2)
        for row in local["hp_s_w"]
    ]
    model += local["c_hp"][0] == 1.5 * hours
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    assert local["hp_on"][0].x == 1
    assert [var.x for var in local["hp_s_w"][0]] == pytest.approx([0, 0.5, 0.5])
    assert [row[0].x for row in local["p_hp"]] == pytest.approx([0, 500, 1000])
    assert local["h_hp"][0].x == pytest.approx(5 * hours)
    assert sum(var.x for var in local["h_hp"]) >= 30 - 1e-7
    assert local["ha_writes"] == []


@pytest.mark.parametrize("capture", [{"hourly": True}], indirect=True)
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
@pytest.mark.parametrize("positive_stages,expected", [
    pytest.param((1, 2), mip.OptimizationStatus.OPTIMAL, id="adjacent-control"),
    pytest.param((1, 3), mip.OptimizationStatus.INFEASIBLE, id="nonadjacent"),
])
def test_nonadjacent_hp_weights_are_infeasible(power_hp_model, adjustment, positive_stages, expected):
    """Removing multi-stage SOS2 admits a nonadjacent positive-stage mixture."""
    local = power_hp_model(
        prices=[0.10] * 7 + [0.60], heat_kwh=30, minimum_hours=5, adjustment=adjustment,
        stages=[{"max_power": 1000, "cop": 4}, {"max_power": 2000, "cop": 3},
                {"max_power": 3000, "cop": 2}],
    )
    model = local["model"]
    assert local["blocks_num"] == 0
    assert local["S"] == 4
    for s, weight in enumerate(local["hp_s_w"][0]):
        model += weight == (0.5 if s in positive_stages else 0)
    assert model.optimize() == expected
    if expected == mip.OptimizationStatus.OPTIMAL:
        model.check_optimization_results()
        assert local["hp_on"][0].x == 1
        assert local["c_hp"][0].x == pytest.approx(1.5)
        assert local["h_hp"][0].x == pytest.approx(5)
    else:
        assert model.num_solutions == 0
    assert local["ha_writes"] == []


@pytest.mark.parametrize("capture", [{"start": ts(minute=7) + 30}], indirect=True)
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_high_runtime_availability_uses_actual_first_interval(power_hp_model, adjustment):
    """A partial horizon contains 7.875 hours, even when all eight nominal hours are needed."""
    local = power_hp_model(
        prices=[0.10] * 8, heat_kwh=31.5, minimum_hours=5, adjustment=adjustment,
    )
    assert local["hours_avail"] == 7.875
    assert local["blocks_num"] == 0
    assert local["heat_needed"] == 31.5
    assert_schedule(local, heat_kwh=31.5, runtime_hours=7.875, hp_cost=0.7875)
    assert [flag.x for flag in local["hp_on"]] == pytest.approx([1] * 32)


@pytest.mark.parametrize("required,minimum,first,count,expected", [
    pytest.param(0, 3600, 3600, 0, (), id="no-blocks"),
    pytest.param(21600, 18000, 18000, 1, (21600,), id="single-full-duration"),
    pytest.param(1800, 3600, 3600, 1, (3600,), id="single-first-minimum"),
    pytest.param(18000, 3600, 3600, 3, (3600, 7200, 7200), id="five-not-nine"),
    pytest.param(18001, 3600, 3600, 3, (3600, 7200, 7201), id="no-modulo-loss"),
    pytest.param(21600, 3600, 3600, 3, (3600, 7200, 10800), id="upstream-whole-hours"),
    pytest.param(25200, 3600, 3600, 4, (3600, 7200, 7200, 7200), id="several-blocks"),
    pytest.param(18000, 7200, 10800, 3, (10800, 7200), id="reduce-count"),
    pytest.param(21600, 18000, 18000, 2, (21600,), id="reduce-to-single"),
    pytest.param(7200, 10800, 3600, 3, (7200,), id="remaining-minimum-does-not-fit"),
    pytest.param(1800, 3600, 3600, 3, (3600,), id="reduce-to-first-minimum"),
])
def test_block_durations(required, minimum, first, count, expected):
    """Discarding remainder seconds or padding to fit a heuristic count loses heat/runtime."""
    from dao.prog.heatpump_blocks import block_durations

    assert block_durations(
        required_seconds=required, minimum_seconds=minimum,
        first_seconds=first, count=count,
    ) == expected


@pytest.mark.parametrize("intervals,duration,expected", [
    pytest.param((), 3600, (), id="empty"),
    pytest.param(((0, 900),), 0, (), id="no-duration"),
    pytest.param(((0, 3600), (3600, 7200), (7200, 10800)), 3600,
                 ((0, 1), (1, 2), (2, 3)), id="exact-hours"),
    pytest.param(((0, 3600), (3600, 7200), (7200, 10800)), 3601,
                 ((0, 2), (1, 3)), id="remainder-second"),
    pytest.param(((0, 450), (450, 1350), (1350, 2250), (2250, 3150), (3150, 4050)),
                 3600, ((0, 5), (1, 5)), id="partial-first-quarter"),
    pytest.param(((0, 1800), (1800, 5400)), 3600, ((0, 2), (1, 2)), id="partial-first-hour"),
    pytest.param(((0, 450), (450, 1350)), 1350, ((0, 2),), id="exact-horizon-end"),
    pytest.param(((0, 450), (450, 1350)), 1351, (), id="too-short-horizon"),
    pytest.param(((0, 900), (1800, 2700), (2700, 3600)), 1800,
                 ((1, 3),), id="do-not-bridge-gap"),
])
def test_block_spans(intervals, duration, expected):
    """The shortest contiguous UTC span must meet the duration without inventing time."""
    from dao.prog import heatpump_blocks

    assert hasattr(heatpump_blocks, "block_spans"), "actual-interval span helper is missing"
    assert heatpump_blocks.block_spans(intervals, duration) == expected
