"""Tests for KNMI and Open-Meteo archive observations.

Adapted from dao/tests/prog/test_solar_predictor_weatherdata.py's
get_and_save_knmi_data tests: that method is now a thin delegate to
update_observations, so the coverage moved here to target the real logic
directly. test_import_weatherdata_saves_three_codes_per_source_row stayed
in place -- import_weatherdata (the CSV importer) is unchanged.
"""

from __future__ import annotations

import datetime as dt
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from dao.forecast.weather.observations import (
    fetch_knmi,
    nearest_knmi_station,
    observation_mode,
    update_observations,
)
from dao.forecast.weather.openmeteo import OPENMETEO_ARCHIVE_URL
from dao.lib.db_manager import DBmanagerObj

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def db(tmp_path):
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
    Table(
        "values",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
        Column("time", BigInteger, nullable=False),
        Column("value", Float),
        UniqueConstraint("variabel", "time"),
    )
    metadata.create_all(manager.engine)
    with manager.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {"id": 1, "code": "temp", "name": "Temperatuur", "dim": "C"},
                {"id": 2, "code": "gr", "name": "Globale straling", "dim": "J/cm2"},
                {"id": 3, "code": "winds", "name": "Windsnelheid", "dim": "m/s"},
            ],
        )
    return manager


@pytest.fixture
def requests_stub(monkeypatch):
    """Patches the real ``requests.get`` with a canned Open-Meteo response."""
    payload = json.loads((FIXTURES / "openmeteo_forecast.json").read_text())

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    calls: list[dict] = []

    def stub_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return StubResponse()

    import requests as requests_module

    monkeypatch.setattr(requests_module, "get", stub_get)
    return calls


def stored(db):
    from sqlalchemy import Table, select

    values = Table("values", db.metadata, autoload_with=db.engine)
    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    with db.engine.connect() as connection:
        rows = connection.execute(
            select(variabel.c.code, values.c.time, values.c.value)
            .select_from(values.join(variabel, values.c.variabel == variabel.c.id))
            .order_by(values.c.time, variabel.c.code)
        )
        return [(code, int(time), value) for code, time, value in rows]


def test_nearest_station_picks_closest_aws():
    assert nearest_knmi_station(52.10, 5.18) == 260  # De Bilt
    assert nearest_knmi_station(52.928, 4.781) == 235  # De Kooy


def test_fetch_knmi_does_not_shift_hours(monkeypatch):
    knmi_frame = pd.DataFrame(
        {"Q": [50], "T": [183], "FH": [30]},
        index=pd.to_datetime(["2026-03-15 00:00:00"]),
    )
    monkeypatch.setattr(
        "dao.forecast.weather.observations.knmi.get_hour_data_dataframe",
        lambda stations, start, end, variables: knmi_frame,
    )

    frame = fetch_knmi(260, date(2026, 3, 15), date(2026, 3, 15))

    expected_time = int(pd.Timestamp("2026-03-15 00:00:00", tz="UTC").timestamp())
    assert frame.iloc[0]["time"] == expected_time
    assert frame.iloc[0]["temp"] == pytest.approx(18.3)
    assert frame.iloc[0]["gr"] == pytest.approx(50.0)
    assert frame.iloc[0]["winds"] == pytest.approx(3.0)


def test_update_observations_saves_three_codes_per_row(db, monkeypatch):
    knmi_frame = pd.DataFrame(
        {"T": [100, 110], "Q": [50, 60], "FH": [30, 40]},
        index=pd.to_datetime(["2026-03-15 10:00:00", "2026-03-15 11:00:00"]),
    )
    monkeypatch.setattr(
        "dao.forecast.weather.observations.knmi.get_hour_data_dataframe",
        lambda stations, start, end, variables: knmi_frame,
    )

    count = update_observations(
        db, 52.10, 5.18, "NL", "auto", now=dt.datetime(2026, 3, 16, tzinfo=dt.UTC)
    )

    assert count == 6
    t0 = int(pd.Timestamp("2026-03-15 10:00:00", tz="UTC").timestamp())
    t1 = int(pd.Timestamp("2026-03-15 11:00:00", tz="UTC").timestamp())
    assert stored(db) == [
        ("gr", t0, 50.0),
        ("temp", t0, 10.0),
        ("winds", t0, 3.0),
        ("gr", t1, 60.0),
        ("temp", t1, 11.0),
        ("winds", t1, 4.0),
    ]


def test_update_observations_off_writes_nothing(db):
    count = update_observations(db, 52.10, 5.18, "NL", "off")
    assert count == 0
    assert stored(db) == []


def test_observation_mode_auto_outside_benelux_is_openmeteo():
    assert observation_mode("auto", "DE") == "openmeteo"
    assert observation_mode("auto", "NL") == "knmi"
    assert observation_mode("auto", "BE") == "knmi"
    assert observation_mode("knmi", "DE") == "knmi"
    assert observation_mode("off", "NL") == "off"


def test_update_observations_openmeteo_archive(db, requests_stub):
    count = update_observations(db, 50.0, 8.0, "DE", "auto")

    assert count > 0
    assert {code for code, _, _ in stored(db)} == {"gr", "temp", "winds"}
    assert requests_stub[0]["url"] == OPENMETEO_ARCHIVE_URL
