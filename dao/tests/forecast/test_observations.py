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


def test_fetch_knmi_does_not_shift_the_hour_aggregates(monkeypatch):
    """knmi-py already subtracts one from KNMI's 1-24 hour field, so its
    index is the hour start. Q ("Globale straling per uurvak") and FH
    ("Uurgemiddelde windsnelheid") are aggregates over that slot and belong
    exactly where knmi-py puts them."""
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
    assert frame.iloc[0]["gr"] == pytest.approx(50.0)
    assert frame.iloc[0]["winds"] == pytest.approx(3.0)


def test_fetch_knmi_puts_the_temperature_on_its_own_hour(monkeypatch):
    """KNMI documents T as the temperature "tijdens de waarneming", and the
    observation belonging to hour slot HH is made at the end of it. With
    knmi-py's index at HH-1 the reading lands an hour early, the same
    mismatch Open-Meteo has between its instantaneous and its aggregated
    variables."""
    knmi_frame = pd.DataFrame(
        {"Q": [10, 20, 30], "T": [100, 110, 120], "FH": [30, 30, 30]},
        index=pd.to_datetime(
            ["2026-03-15 00:00:00", "2026-03-15 01:00:00", "2026-03-15 02:00:00"]
        ),
    )
    monkeypatch.setattr(
        "dao.forecast.weather.observations.knmi.get_hour_data_dataframe",
        lambda stations, start, end, variables: knmi_frame,
    )

    frame = fetch_knmi(260, date(2026, 3, 15), date(2026, 3, 15))

    def at(hour: int):
        stamp = int(pd.Timestamp(f"2026-03-15 0{hour}:00:00", tz="UTC").timestamp())
        return frame[frame["time"] == stamp].iloc[0]

    # The slot indexed 01:00 covers 01:00-02:00 and its reading was taken at
    # 02:00, so 11.0 degrees belongs on the 02:00 row.
    assert at(2)["temp"] == pytest.approx(11.0)
    assert at(1)["temp"] == pytest.approx(10.0)
    # ...while the radiation of that same slot stays where it was.
    assert at(1)["gr"] == pytest.approx(20.0)


def test_update_observations_saves_three_codes_per_row(db, monkeypatch):
    """Three rows in, three codes out per hour -- except the first hour of
    the window, whose temperature reading (taken at its end) belongs to the
    hour before the window and was never fetched. That one is left out
    rather than guessed; a trailing window overlaps the previous run, which
    already stored it."""
    knmi_frame = pd.DataFrame(
        {"T": [90, 100, 110], "Q": [40, 50, 60], "FH": [20, 30, 40]},
        index=pd.to_datetime(
            [
                "2026-03-15 09:00:00",
                "2026-03-15 10:00:00",
                "2026-03-15 11:00:00",
            ]
        ),
    )
    monkeypatch.setattr(
        "dao.forecast.weather.observations.knmi.get_hour_data_dataframe",
        lambda stations, start, end, variables: knmi_frame,
    )

    count = update_observations(
        db, 52.10, 5.18, "NL", "auto", now=dt.datetime(2026, 3, 16, tzinfo=dt.UTC)
    )

    assert count == 8
    t0 = int(pd.Timestamp("2026-03-15 09:00:00", tz="UTC").timestamp())
    t1 = int(pd.Timestamp("2026-03-15 10:00:00", tz="UTC").timestamp())
    t2 = int(pd.Timestamp("2026-03-15 11:00:00", tz="UTC").timestamp())
    assert stored(db) == [
        ("gr", t0, 40.0),
        ("winds", t0, 2.0),
        ("gr", t1, 50.0),
        ("temp", t1, 9.0),
        ("winds", t1, 3.0),
        ("gr", t2, 60.0),
        ("temp", t2, 10.0),
        ("winds", t2, 4.0),
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
