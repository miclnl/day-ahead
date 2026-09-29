"""Tests for the history reader over Home Assistant statistics."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from dao.forecast.history import (
    COMPONENT_COLUMNS,
    HistoryReader,
    UnsupportedSensorError,
    baseload_from_components,
    component_groups,
)
from dao.tests.forecast.conftest import HOUR, T0, TZ

ZONE = ZoneInfo(TZ)


def local(ts: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, tz=ZONE)


@pytest.fixture
def reader(ha_db):
    manager, _ = ha_db
    return HistoryReader(manager, TZ)


def test_energy_sensor_uses_sum_deltas_and_wh_factor(ha_db, reader):
    _, helper = ha_db
    helper.add_energy(
        "sensor.test_pv", "Wh", {T0: 1000.0, T0 + HOUR: 1500.0, T0 + 2 * HOUR: 2700.0}
    )
    series = reader.read_energy(["sensor.test_pv"], local(T0), local(T0 + 2 * HOUR))
    assert list(series.round(3)) == [0.5, 1.2]
    assert series.index.tz is not None
    assert series.index[0] == pd.Timestamp(T0, unit="s", tz="UTC").tz_convert(TZ)


def test_energy_sensor_survives_a_meter_reset(ha_db, reader):
    _, helper = ha_db
    helper.add_energy_with_state(
        "sensor.test_grid",
        "kWh",
        {T0: (8000.0, 500.0), T0 + HOUR: (8000.5, 500.5), T0 + 2 * HOUR: (12.0, 512.5)},
    )
    series = reader.read_energy(["sensor.test_grid"], local(T0), local(T0 + 2 * HOUR))
    assert list(series.round(3)) == [0.5, 12.0]


def test_power_sensor_uses_mean_and_logs_once(ha_db, reader, caplog):
    _, helper = ha_db
    helper.add_power("sensor.test_pv_power", "W", {T0: 1200.0, T0 + HOUR: 600.0})
    with caplog.at_level("INFO"):
        series = reader.read_energy(
            ["sensor.test_pv_power"], local(T0), local(T0 + 2 * HOUR)
        )
    assert list(series.round(3)) == [1.2, 0.6]
    assert sum("vermogenssensor" in r.message for r in caplog.records) == 1


def test_unsupported_unit_raises_with_sensor_name(ha_db, reader):
    _, helper = ha_db
    helper.add_other("sensor.test_soc", "%")
    with pytest.raises(UnsupportedSensorError, match="sensor.test_soc"):
        reader.sensor_meta(["sensor.test_soc"])


def test_missing_hours_are_nan_not_zero(ha_db, reader):
    _, helper = ha_db
    helper.add_energy(
        "sensor.test_grid", "kWh", {T0: 10.0, T0 + HOUR: 10.4, T0 + 3 * HOUR: 11.0, T0 + 4 * HOUR: 11.2}
    )
    series = reader.read_energy(["sensor.test_grid"], local(T0), local(T0 + 4 * HOUR))
    assert len(series) == 4
    assert series.iloc[0] == pytest.approx(0.4)
    assert np.isnan(series.iloc[1]) and np.isnan(series.iloc[2])
    assert series.iloc[3] == pytest.approx(0.2)


def test_cap_turns_glitch_into_nan(ha_db, reader, caplog):
    _, helper = ha_db
    helper.add_energy(
        "sensor.test_pv", "kWh", {T0: 100.0, T0 + HOUR: 13586.0, T0 + 2 * HOUR: 13587.0}
    )
    with caplog.at_level("INFO"):
        series = reader.read_energy(
            ["sensor.test_pv"], local(T0), local(T0 + 2 * HOUR), cap_kwh=4.3
        )
    assert np.isnan(series.iloc[0])
    assert series.iloc[1] == pytest.approx(1.0)
    assert any("buiten bereik" in r.message for r in caplog.records)


def _components_frame(reader, helper, *, grid_hours, pv_hours, machines=None):
    helper.add_energy("sensor.test_grid_in", "kWh", grid_hours)
    helper.add_energy("sensor.test_grid_out", "kWh", {t: 0.0 for t in pv_hours})
    helper.add_energy("sensor.test_pv", "kWh", pv_hours)
    helper.add_energy("sensor.test_bat_in", "kWh", {t: 0.0 for t in pv_hours})
    helper.add_energy("sensor.test_bat_out", "kWh", {t: 0.0 for t in pv_hours})
    groups = {c: [] for c in COMPONENT_COLUMNS}
    groups.update(
        {
            "grid_in": ["sensor.test_grid_in"],
            "grid_out": ["sensor.test_grid_out"],
            "pv_ac": ["sensor.test_pv"],
            "bat_in": ["sensor.test_bat_in"],
            "bat_out": ["sensor.test_bat_out"],
        }
    )
    if machines is not None:
        helper.add_energy("sensor.test_wasmachine", "kWh", machines)
        groups["machines"] = ["sensor.test_wasmachine"]
    return reader.read_components(groups, local(T0), local(T0 + 4 * HOUR))


def test_grid_gap_gives_nan_baseload_while_pv_records(ha_db, reader):
    _, helper = ha_db
    pv = {T0 + i * HOUR: 50.0 + 0.5 * i for i in range(5)}
    grid = {T0: 10.0, T0 + HOUR: 10.3, T0 + 4 * HOUR: 11.2}   # hours 1 and 2 missing
    frame = _components_frame(reader, helper, grid_hours=grid, pv_hours=pv)
    base = baseload_from_components(frame)
    assert base.iloc[0] == pytest.approx(0.8)          # 0.3 grid + 0.5 pv
    assert np.isnan(base.iloc[1]) and np.isnan(base.iloc[2])
    assert (base.dropna() >= 0).all()


def test_missing_machine_meter_counts_as_zero(ha_db, reader, caplog):
    _, helper = ha_db
    pv = {T0 + i * HOUR: 50.0 + 0.5 * i for i in range(5)}
    grid = {T0 + i * HOUR: 10.0 + 0.3 * i for i in range(5)}
    machines = {T0: 1.0, T0 + HOUR: 1.2}                # only the first hour has a delta
    frame = _components_frame(reader, helper, grid_hours=grid, pv_hours=pv, machines=machines)
    with caplog.at_level("INFO"):
        base = baseload_from_components(frame)
    assert base.iloc[0] == pytest.approx(0.8 - 0.2)
    assert base.iloc[2] == pytest.approx(0.8)
    assert any("als 0 geteld" in r.message for r in caplog.records)


def test_autumn_dst_hour_appears_twice(ha_db, reader):
    _, helper = ha_db
    start_utc = int(dt.datetime(2026, 10, 24, 22, 0, tzinfo=dt.UTC).timestamp())
    sums = {start_utc + i * HOUR: 100.0 + i for i in range(7)}
    helper.add_energy("sensor.test_grid", "kWh", sums)
    start = dt.datetime(2026, 10, 25, 0, 0, tzinfo=ZONE)
    end = dt.datetime(2026, 10, 25, 4, 0, tzinfo=ZONE)
    series = reader.read_energy(["sensor.test_grid"], start, end)
    local_hours = list(series.index.hour)
    assert local_hours == [0, 1, 2, 2, 3]
    assert len(set(series.index.tz_convert("UTC"))) == 5
    assert series.notna().all()


def test_component_groups_maps_report_config():
    from dao.prog.config.models.report import ReportConfig

    report = ReportConfig(
        **{
            "entities grid consumption": ["sensor.test_grid_in"],
            "entities machine consumption": ["sensor.test_wasmachine"],
        }
    )
    groups = component_groups(report)
    assert set(groups) == set(COMPONENT_COLUMNS)
    assert groups["grid_in"] == ["sensor.test_grid_in"]
    assert groups["machines"] == ["sensor.test_wasmachine"]
    assert groups["ev"] == []


def test_baseload_formula():
    index = pd.date_range("2026-03-02", periods=1, freq="h", tz=TZ)
    frame = pd.DataFrame({c: [1.0] for c in COMPONENT_COLUMNS}, index=index)
    frame["bat_in"] = 2.0
    assert baseload_from_components(frame).iloc[0] == pytest.approx(-4.0)
