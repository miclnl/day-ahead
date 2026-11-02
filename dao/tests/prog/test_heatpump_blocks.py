"""Required heat, not the available horizon, determines power-mode blocks."""

import datetime as dt
import sys
from types import SimpleNamespace

import mip
import pandas as pd
import pytest

from dao.prog import day_ahead as da
from dao.prog.config.models.devices.heating import HeatingEnabled
from dao.tests.settlement.test_optimization_producer import capture as capture


@pytest.fixture
def power_hp_model(capture, monkeypatch):
    """Reuse the offline capture boundary; build the real HP producer and CBC."""
    instance = capture["instance"]
    start = capture["locals"]["start_dt"]
    instance.battery_options = []
    instance.meteo = SimpleNamespace(calc_graaddagen=lambda **kwargs: 1.0)

    def build(*, prices, heat_kwh, minimum_hours, adjustment):
        instance.heating_options = HeatingEnabled.model_validate({
            "adjustment": adjustment,
            "degree days factor": heat_kwh,
            "min run length": minimum_hours,
            "stages": [{"max_power": 1000, "cop": 4}],
        })
        steps_per_hour = 3600 // instance.interval_s
        tariffs = [price for price in prices for _ in range(steps_per_hour)]
        stamps = [int(start.timestamp()) + instance.interval_s * u for u in range(len(tariffs))]
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
            get_price_data=lambda *a, **k: price_data.copy()))
        instance.db_da = SimpleNamespace(get_prognose_data=lambda **kwargs: forecast.copy())
        found = {}
        bind_cost = da.bind_day_ahead_cost

        class Captured(Exception):
            pass

        def capture_cost(model, **kwargs):
            bind_cost(model, **kwargs)
            found.update(sys._getframe(1).f_locals)
            raise Captured

        with monkeypatch.context() as spies:
            spies.setattr(da, "bind_day_ahead_cost", capture_cost)
            with pytest.raises(Captured):
                instance.calc_optimum(_start_dt=start)
        model = found["model"]
        assert model.solver_name == mip.CBC
        assert found["U"] == len(tariffs)
        assert found["heat_needed"] == heat_kwh
        assert found["hour_fraction"] == [1 / steps_per_hour] * len(tariffs)
        assert [stage["max_power"] for stage in found["hp_stages"]] == [0, 1000]
        model.verbose = 0
        model.objective = mip.minimize(found["cost"])
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
