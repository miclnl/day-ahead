"""Tests for archive_accuracy: the archive scored against what was measured."""

from __future__ import annotations

import datetime as dt
import json
import types

import pytest

from dao.forecast.evaluate import (
    COMPONENTS,
    AccuracyReport,
    archive_accuracy,
    measured_series,
)
from dao.forecast.history import HistoryReader
from dao.lib.db_manager import DBmanagerObj, forecasts_table
from dao.tests.forecast.conftest import HOUR, TZ

NOW = dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.UTC)


def make_config():
    return types.SimpleNamespace(
        report=types.SimpleNamespace(
            entities_grid_consumption=["sensor.test_grid_in"],
            entities_grid_production=["sensor.test_grid_out"],
            entities_solar_production_ac=["sensor.test_pv"],
            entities_ev_consumption=[],
            entities_wp_consumption=[],
            entities_boiler_consumption=[],
            entities_machine_consumption=[],
            entities_battery_consumption=[],
            entities_battery_production=[],
        ),
        solar=[
            types.SimpleNamespace(
                entities_sensors=["sensor.test_pv"], total_capacity=3.6
            )
        ],
        battery=[],
        grid=types.SimpleNamespace(max_power=None),
    )


@pytest.fixture
def da_db(tmp_path):
    from sqlalchemy import (
        BigInteger,
        Column,
        Float,
        ForeignKey,
        Integer,
        String,
        Table,
        UniqueConstraint,
        insert,
    )

    manager = DBmanagerObj(
        db_dialect="sqlite", db_name="day_ahead.db", db_path=str(tmp_path)
    )
    metadata = manager.metadata
    variabel = Table(
        "variabel",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("code", String(10), unique=True, nullable=False),
        Column("name", String(50), unique=True, nullable=False),
        Column("dim", String(10), nullable=False),
        Column("aggregate", String(3), nullable=False, default="avg"),
    )
    for name in ("values", "prognoses"):
        Table(
            name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
            Column("time", BigInteger, nullable=False),
            Column("value", Float),
            UniqueConstraint("variabel", "time"),
        )
    forecasts_table(metadata)
    metadata.create_all(manager.engine)
    with manager.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {"id": 4, "code": "gr", "name": "Globale straling", "dim": "J/cm2"},
                {"id": 5, "code": "temp", "name": "Temperatuur", "dim": "C"},
                {"id": 11, "code": "base", "name": "Basislast", "dim": "kWh"},
                {"id": 27, "code": "hload", "name": "Geplande huisvraag", "dim": "kWh"},
                {"id": 28, "code": "dni", "name": "Directe straling", "dim": "J/cm2"},
                {"id": 30, "code": "away", "name": "Afwezig", "dim": "-"},
            ],
        )
    return manager


def put_value(db, code, rows):
    from sqlalchemy import Table, insert, select

    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    values = Table("values", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        ident = connection.execute(
            select(variabel.c.id).where(variabel.c.code == code)
        ).scalar_one()
        connection.execute(
            insert(values),
            [{"variabel": ident, "time": t, "value": v} for t, v in rows],
        )


def put_forecast(db, code, rows, lead_bucket=0, source="meteoserver"):
    from sqlalchemy import Table, insert, select

    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    forecasts = Table("forecasts", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        ident = connection.execute(
            select(variabel.c.id).where(variabel.c.code == code)
        ).scalar_one()
        connection.execute(
            insert(forecasts),
            [
                {
                    "variabel": ident,
                    "target_time": t,
                    "lead_bucket": lead_bucket,
                    "issued_time": t - lead_bucket * HOUR,
                    "value": v,
                    "source": source,
                }
                for t, v in rows
            ],
        )


def recent_hours(count: int, offset_days: int = 1):
    """``count`` whole hours ending just before NOW."""
    start = NOW - dt.timedelta(days=offset_days)
    return [int((start + dt.timedelta(hours=i)).timestamp()) for i in range(count)]


def test_measured_base_uses_history_formula(ha_db):
    manager, helper = ha_db
    start = dt.datetime(2026, 6, 20, 0, 0, tzinfo=dt.UTC)
    hours = 24
    grid_in = {}
    total = 0.0
    for i in range(hours + 1):
        grid_in[int((start + dt.timedelta(hours=i)).timestamp())] = total
        total += 0.4
    helper.add_energy("sensor.test_grid_in", "kWh", grid_in)
    helper.add_energy("sensor.test_grid_out", "kWh", {ts: 0.0 for ts in grid_in})
    helper.add_power("sensor.test_pv", "W", {ts: 0.0 for ts in list(grid_in)[:-1]})

    reader = HistoryReader(manager, TZ)
    series = measured_series(
        "base", reader, make_config(), None, start, start + dt.timedelta(hours=hours)
    )

    assert series is not None
    # grid_in 0.4 per hour, nothing else configured -> baseload is 0.4.
    assert series.dropna().iloc[0] == pytest.approx(0.4, abs=1e-6)


def test_report_groups_by_lead_hour_weekday(da_db):
    hours = recent_hours(24)
    put_value(da_db, "temp", [(t, 20.0) for t in hours])
    # Lead 0 is exact; lead 24 is systematically 0.1 too high.
    put_forecast(da_db, "temp", [(t, 20.0) for t in hours], lead_bucket=0)
    put_forecast(da_db, "temp", [(t, 20.1) for t in hours], lead_bucket=24)

    report = archive_accuracy(make_config(), da_db, None, TZ, days=(7,), now=NOW)
    window = report.components["temp"].windows[7]

    assert window["by_lead"][0].bias == pytest.approx(0.0, abs=1e-9)
    assert window["by_lead"][24].bias == pytest.approx(0.1, abs=1e-6)
    assert window["pairs"] == 48
    assert len(window["by_hour"]) == 24
    assert len(window["by_weekday"]) >= 1


def test_report_splits_by_regime(da_db):
    hours = recent_hours(48, offset_days=2)
    put_value(da_db, "temp", [(t, 20.0) for t in hours])
    put_forecast(da_db, "temp", [(t, 20.5) for t in hours], lead_bucket=0)
    # Mark the first of the two days away, at its local midnight.
    first_day = dt.datetime.fromtimestamp(hours[0], tz=dt.UTC).date()
    midnight = int(
        dt.datetime.combine(first_day, dt.time.min, tzinfo=dt.UTC).timestamp()
    )
    put_value(da_db, "away", [(midnight, 1.0)])

    report = archive_accuracy(make_config(), da_db, None, TZ, days=(7,), now=NOW)
    window = report.components["temp"].windows[7]

    assert set(window["by_regime"]) == {"home", "away"}
    assert window["by_regime"]["away"].n > 0
    assert window["by_regime"]["home"].n > 0


def test_report_splits_weather_by_source(da_db):
    hours = recent_hours(12)
    put_value(da_db, "gr", [(t, 100.0) for t in hours])
    put_forecast(da_db, "gr", [(t, 110.0) for t in hours[:6]], source="meteoserver")
    put_forecast(da_db, "gr", [(t, 90.0) for t in hours[6:]], source="openmeteo")

    report = archive_accuracy(make_config(), da_db, None, TZ, days=(7,), now=NOW)
    window = report.components["gr"].windows[7]

    assert set(window["by_source"]) == {"meteoserver", "openmeteo"}
    assert window["by_source"]["meteoserver"].bias == pytest.approx(10.0)
    assert window["by_source"]["openmeteo"].bias == pytest.approx(-10.0)


def test_report_counts_missing_pairs(da_db):
    hours = recent_hours(12)
    # Forecasts for twelve hours, measurements for only the first four.
    put_forecast(da_db, "temp", [(t, 20.0) for t in hours])
    put_value(da_db, "temp", [(t, 20.0) for t in hours[:4]])

    report = archive_accuracy(make_config(), da_db, None, TZ, days=(7,), now=NOW)
    window = report.components["temp"].windows[7]

    assert window["pairs"] == 4
    assert window["missing"] == 8


def test_report_to_dict_json_round_trip(da_db):
    hours = recent_hours(6)
    put_value(da_db, "temp", [(t, 20.0) for t in hours])
    put_forecast(da_db, "temp", [(t, 20.2) for t in hours])

    report = archive_accuracy(make_config(), da_db, None, TZ, days=(7, 28), now=NOW)
    restored = json.loads(json.dumps(report.to_dict()))

    assert restored["days"] == [7, 28]
    assert set(restored["components"]) == set(COMPONENTS)
    temp = restored["components"]["temp"]
    assert temp["unit"] == "\u00b0C"
    assert temp["windows"]["7"]["by_lead"]["0"]["bias"] == pytest.approx(0.2, abs=1e-6)


def test_empty_archive_gives_empty_report_not_error(da_db):
    report = archive_accuracy(make_config(), da_db, None, TZ, days=(7,), now=NOW)

    assert isinstance(report, AccuracyReport)
    assert set(report.components) == set(COMPONENTS)
    for accuracy in report.components.values():
        assert accuracy.windows[7]["pairs"] == 0
        assert accuracy.windows[7]["by_lead"] == {}
