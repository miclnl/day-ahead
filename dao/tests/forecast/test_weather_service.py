"""Tests for WeatherService: primary source, Open-Meteo fallback, archiving."""

from __future__ import annotations

import datetime as dt
import json
import types
from pathlib import Path

import pytest

from dao.forecast.weather.service import WeatherService
from dao.lib.db_manager import DBmanagerObj, forecasts_table
from dao.prog.config.models.base import SecretStr
from dao.prog.config.models.weather import WeatherConfig

HOUR = 3600


@pytest.fixture
def db(tmp_path):
    """A day_ahead database with the production schema, weather codes included."""
    from sqlalchemy import (
        BigInteger,
        Column,
        Float,
        ForeignKey,
        Index,
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
    forecasts = forecasts_table(metadata)
    Index("ix_forecasts_target", forecasts.c.target_time)
    metadata.create_all(manager.engine)

    with manager.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {"id": 4, "code": "gr", "name": "Globale straling", "dim": "J/cm2",
                 "aggregate": "avg"},
                {"id": 5, "code": "temp", "name": "Temperatuur", "dim": "C",
                 "aggregate": "avg"},
                {"id": 23, "code": "winds", "name": "Windsnelheid", "dim": "m/s",
                 "aggregate": "avg"},
                {"id": 24, "code": "neersl", "name": "Neerslag", "dim": "mm",
                 "aggregate": "sum"},
                {"id": 28, "code": "dni", "name": "Directe straling", "dim": "J/cm2",
                 "aggregate": "avg"},
                {"id": 29, "code": "dhi", "name": "Diffuse straling", "dim": "J/cm2",
                 "aggregate": "avg"},
            ],
        )
    return manager


def stored(db, tablename):
    from sqlalchemy import Table, select

    table = Table(tablename, db.metadata, autoload_with=db.engine)
    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    time_column = table.c.target_time if tablename == "forecasts" else table.c.time
    columns = [variabel.c.code, time_column, table.c.value]
    if tablename == "forecasts":
        columns.append(table.c.source)
    with db.engine.connect() as connection:
        rows = connection.execute(
            select(*columns)
            .select_from(table.join(variabel, table.c.variabel == variabel.c.id))
            .order_by(time_column, variabel.c.code)
        )
        return [tuple(row) for row in rows]


def make_config(*, has_key=True, fallback="openmeteo", observations="auto"):
    return types.SimpleNamespace(
        meteoserver_key=SecretStr("test-key") if has_key else None,
        meteoserver_model="harmonie",
        meteoserver_attempts=1,
        weather=WeatherConfig(fallback=fallback, observations=observations),
    )


def meteoserver_payload(start_epoch: int, count: int) -> dict:
    rows = []
    for i in range(count):
        epoch = start_epoch + i * HOUR
        rows.append(
            {
                "tijd": str(epoch),
                "tijd_nl": "x",
                "gr": "100",
                "temp": "10",
                "winds": "3",
                "neersl": "0",
            }
        )
    return {"data": rows}


def openmeteo_payload(start_utc: dt.datetime, count: int) -> dict:
    # parse_openmeteo shifts hourly.time back by one hour (Open-Meteo labels
    # the interval's end), so the first entry here has to be start_utc + 1h
    # for the parsed frame to start exactly at start_utc.
    times = []
    for i in range(count):
        moment = start_utc + dt.timedelta(hours=i + 1)
        times.append(moment.strftime("%Y-%m-%dT%H:%M"))
    n = count
    return {
        "hourly": {
            "time": times,
            "shortwave_radiation": [50.0] * n,
            "direct_normal_irradiance": [60.0] * n,
            "diffuse_radiation": [20.0] * n,
            "temperature_2m": [15.0] * n,
            "wind_speed_10m": [3.0] * n,
            "precipitation": [0.0] * n,
        }
    }


class _StubResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} fout")

    def json(self):
        return self._payload


class _StubSession:
    """Dispatches to a meteoserver or an Open-Meteo canned response by URL."""

    def __init__(self, meteoserver=None, meteoserver_status=200, openmeteo=None):
        self.meteoserver = meteoserver
        self.meteoserver_status = meteoserver_status
        self.openmeteo = openmeteo
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        if "meteoserver" in url:
            return _StubResponse(self.meteoserver_status, self.meteoserver)
        return _StubResponse(200, self.openmeteo)


def test_primary_meteoserver_covers_horizon_no_fallback_call():
    now = dt.datetime(2026, 3, 10, 8, 0, tzinfo=dt.UTC)
    start_epoch = int(now.replace(minute=0, second=0, microsecond=0).timestamp())
    session = _StubSession(meteoserver=meteoserver_payload(start_epoch, 72))
    service = WeatherService(
        make_config(),
        db_da=None,
        latitude=52.1,
        longitude=5.2,
        secrets={},
        data_dir=Path("/tmp/opencode/weather_status_1"),
        country="NL",
        now=lambda: now,
        session=session,
    )

    frame, status = service.fetch(horizon_hours=72)

    assert status.primary == "meteoserver"
    assert status.hours_total == 72
    assert status.hours_by_source == {"meteoserver": 72}
    assert all(call["url"].find("meteoserver") >= 0 for call in session.calls)


def test_short_primary_horizon_is_filled_by_openmeteo():
    now = dt.datetime(2026, 3, 10, 8, 0, tzinfo=dt.UTC)
    floored = now.replace(minute=0, second=0, microsecond=0)
    start_epoch = int(floored.timestamp())
    session = _StubSession(
        meteoserver=meteoserver_payload(start_epoch, 48),
        openmeteo=openmeteo_payload(floored, 96),
    )
    service = WeatherService(
        make_config(),
        db_da=None,
        latitude=52.1,
        longitude=5.2,
        secrets={},
        data_dir=Path("/tmp/opencode/weather_status_2"),
        country="NL",
        now=lambda: now,
        session=session,
    )

    frame, status = service.fetch(horizon_hours=72)

    assert status.hours_by_source == {"meteoserver": 48, "openmeteo": 24}
    assert status.hours_total == 72


def test_primary_failure_falls_back_entirely(caplog):
    now = dt.datetime(2026, 3, 10, 8, 0, tzinfo=dt.UTC)
    floored = now.replace(minute=0, second=0, microsecond=0)
    session = _StubSession(
        meteoserver_status=500,
        openmeteo=openmeteo_payload(floored, 96),
    )
    service = WeatherService(
        make_config(),
        db_da=None,
        latitude=52.1,
        longitude=5.2,
        secrets={},
        data_dir=Path("/tmp/opencode/weather_status_3"),
        country="NL",
        now=lambda: now,
        session=session,
    )

    with caplog.at_level("WARNING"):
        frame, status = service.fetch(horizon_hours=72)

    assert status.primary == "meteoserver"
    assert status.primary_error is not None
    assert status.hours_by_source == {"openmeteo": 72}
    assert any("Meteoserver" in message for message in caplog.messages)


def test_no_key_makes_openmeteo_primary():
    now = dt.datetime(2026, 3, 10, 8, 0, tzinfo=dt.UTC)
    floored = now.replace(minute=0, second=0, microsecond=0)
    session = _StubSession(openmeteo=openmeteo_payload(floored, 96))
    service = WeatherService(
        make_config(has_key=False),
        db_da=None,
        latitude=52.1,
        longitude=5.2,
        secrets={},
        data_dir=Path("/tmp/opencode/weather_status_4"),
        country="NL",
        now=lambda: now,
        session=session,
    )

    frame, status = service.fetch(horizon_hours=72)

    assert status.primary == "openmeteo"
    assert all("meteoserver" not in call["url"] for call in session.calls)


def test_fallback_null_disables_gap_fill():
    now = dt.datetime(2026, 3, 10, 8, 0, tzinfo=dt.UTC)
    floored = now.replace(minute=0, second=0, microsecond=0)
    start_epoch = int(floored.timestamp())
    session = _StubSession(
        meteoserver=meteoserver_payload(start_epoch, 48),
        openmeteo=openmeteo_payload(floored, 96),
    )
    service = WeatherService(
        make_config(fallback=None),
        db_da=None,
        latitude=52.1,
        longitude=5.2,
        secrets={},
        data_dir=Path("/tmp/opencode/weather_status_5"),
        country="NL",
        now=lambda: now,
        session=session,
    )

    frame, status = service.fetch(horizon_hours=72)

    assert status.hours_total == 48
    assert status.hours_by_source == {"meteoserver": 48}
    assert all("openmeteo" not in call["url"] for call in session.calls)


def test_update_writes_prognoses_and_archive_with_source(db):
    now = dt.datetime(2026, 3, 10, 8, 0, tzinfo=dt.UTC)
    floored = now.replace(minute=0, second=0, microsecond=0)
    start_epoch = int(floored.timestamp())
    session = _StubSession(meteoserver=meteoserver_payload(start_epoch, 72))
    service = WeatherService(
        make_config(observations="off"),
        db_da=db,
        latitude=52.1,
        longitude=5.2,
        secrets={},
        data_dir=Path("/tmp/opencode/weather_status_6"),
        country="NL",
        now=lambda: now,
        session=session,
    )

    service.update(horizon_hours=72)

    prognose_codes = {code for code, _, _ in stored(db, "prognoses")}
    assert prognose_codes == {"gr", "temp", "winds", "neersl"}

    forecast_rows = stored(db, "forecasts")
    assert forecast_rows
    assert {source for _, _, _, source in forecast_rows} == {"meteoserver"}
    assert {code for code, _, _, _ in forecast_rows} == {"gr", "temp"}


def put_prognose(db, code, rows):
    from sqlalchemy import Table, insert, select

    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    prognoses = Table("prognoses", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        ident = connection.execute(
            select(variabel.c.id).where(variabel.c.code == code)
        ).scalar_one()
        connection.execute(
            insert(prognoses),
            [{"variabel": ident, "time": t, "value": v} for t, v in rows],
        )


def test_get_prognose_fields_outer_merges_optional_codes(db):
    t0 = int(dt.datetime(2026, 3, 10, 10, 0, tzinfo=dt.UTC).timestamp())
    t1 = t0 + HOUR
    put_prognose(db, "gr", [(t0, 100.0), (t1, 110.0)])

    frame = db.get_prognose_fields(["gr", "dni"], t0, t1 + HOUR)

    assert list(frame["gr"]) == [100.0, 110.0]
    assert frame["dni"].isna().all()


def test_update_writes_status_json(tmp_path):
    now = dt.datetime(2026, 3, 10, 8, 0, tzinfo=dt.UTC)
    floored = now.replace(minute=0, second=0, microsecond=0)
    start_epoch = int(floored.timestamp())
    session = _StubSession(meteoserver=meteoserver_payload(start_epoch, 72))
    data_dir = tmp_path / "forecast" / "weather"
    service = WeatherService(
        make_config(observations="off"),
        db_da=None,
        latitude=52.1,
        longitude=5.2,
        secrets={},
        data_dir=data_dir,
        country="NL",
        now=lambda: now,
        session=session,
    )

    service.update(horizon_hours=72)

    status = json.loads((data_dir / "status.json").read_text())
    assert status["primary"] == "meteoserver"
    assert status["hours_total"] == 72
    assert status["hours_by_source"] == {"meteoserver": 72}
