"""The reporting side of the baseload now goes through the history reader."""

from __future__ import annotations

import datetime as dt
import types
from zoneinfo import ZoneInfo

import pytest

from dao.tests.forecast.conftest import HOUR, T0, TZ

ZONE = ZoneInfo(TZ)


@pytest.fixture
def report_with_ha_db(ha_db, monkeypatch):
    """A Report bound to the fake recorder, without loading a configuration."""
    from dao.prog.da_report import Report

    manager, helper = ha_db
    instance = Report.__new__(Report)
    report_config = types.SimpleNamespace(
        entities_grid_consumption=["sensor.test_grid_in"],
        entities_grid_production=["sensor.test_grid_out"],
        entities_solar_production_ac=["sensor.test_pv_power"],
        entities_ev_consumption=[],
        entities_wp_consumption=[],
        entities_boiler_consumption=[],
        entities_machine_consumption=[],
        entities_battery_consumption=["sensor.test_bat_in"],
        entities_battery_production=["sensor.test_bat_out"],
    )
    instance.config = types.SimpleNamespace(
        report=report_config,
        boiler=types.SimpleNamespace(boiler_present=False),
        heating=types.SimpleNamespace(heater_present=False),
        electric_vehicle=[],
        machines=[],
        battery=[types.SimpleNamespace(dc_to_bat_max_power=2200.0, bat_to_dc_max_power=1700.0)],
        solar=[types.SimpleNamespace(total_capacity=3.0)],
        grid=types.SimpleNamespace(max_power=17.0),
        baseload_calc_periode=2,
    )
    instance.db_ha = manager
    instance.time_zone = TZ
    for column, sensors in (
        ("grid_consumption_sensors", report_config.entities_grid_consumption),
        ("grid_production_sensors", report_config.entities_grid_production),
        ("solar_production_ac_sensors", report_config.entities_solar_production_ac),
        ("ev_consumption_sensors", []),
        ("wp_consumption_sensors", []),
        ("boiler_consumption_sensors", []),
        ("machine_consumption_sensors", []),
        ("battery_consumption_sensors", report_config.entities_battery_consumption),
        ("battery_production_sensors", report_config.entities_battery_production),
    ):
        setattr(instance, column, sensors)

    # Two full days of data ending at "now" (the frame covers whole days before today).
    now = dt.datetime.fromtimestamp(T0, tz=ZONE) + dt.timedelta(hours=9)
    hours = range(-48, 1)
    helper.add_energy("sensor.test_grid_in", "kWh", {T0 + i * HOUR: 100.0 + 0.3 * (i + 48) for i in hours})
    helper.add_energy("sensor.test_grid_out", "kWh", {T0 + i * HOUR: 50.0 + 0.1 * (i + 48) for i in hours})
    helper.add_power("sensor.test_pv_power", "W", {T0 + i * HOUR: 800.0 for i in hours})
    helper.add_energy("sensor.test_bat_in", "kWh", {T0 + i * HOUR: 10.0 for i in hours})
    helper.add_energy("sensor.test_bat_out", "kWh", {T0 + i * HOUR: 5.0 for i in hours})

    class FrozenDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is not None else now.replace(tzinfo=None)

    import dao.prog.da_report as da_report

    monkeypatch.setattr(da_report.datetime, "datetime", FrozenDateTime)
    return instance, helper


def test_calc_baseload_frame_uses_power_pv_sensor_via_mean(report_with_ha_db):
    report, _ = report_with_ha_db
    frame = report.calc_baseload_frame()
    assert list(frame.columns) == ["tijd", "weekdag", "uur", "baseload"]
    assert frame["tijd"].dt.tz is not None
    daylight = frame.dropna(subset=["baseload"]).iloc[0]
    # grid in 0.3 - grid out 0.1 + pv mean 0.8 kW x 1 h
    assert daylight["baseload"] == pytest.approx(1.0, abs=1e-6)


def test_check_baseload_sensors_reports_unsupported_unit(report_with_ha_db):
    report, helper = report_with_ha_db
    helper.add_other("sensor.test_soc", "%")
    report.battery_consumption_sensors = ["sensor.test_soc"]
    problems = report.check_baseload_sensors()
    assert any("sensor.test_soc" in p for p in problems)
