"""Temperature-forecast edge cases in da_meteo.py.

Sun position and radiation-on-a-plane tests moved to
dao/tests/forecast/test_pv_physical.py: that logic now lives in the pvlib
chain, not in Meteo. get_avg_temperature() used to return None on empty data
with no upper time bound, which crashed calc_graaddagen()'s '>= 16'
comparison.
"""

import datetime

import pytest

pytest.importorskip("pandas")

from dao.lib.db_manager import DBmanagerObj  # noqa: E402
from dao.lib.da_meteo import Meteo  # noqa: E402


# -- get_avg_temperature / calc_graaddagen -----------------------------------


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
        "prognoses",
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
            [{"id": 1, "code": "temp", "name": "Temperatuur", "dim": "C"}],
        )
    return manager


def put_temps(db, rows):
    from sqlalchemy import Table, insert

    prognoses = Table("prognoses", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        connection.execute(
            insert(prognoses), [{"variabel": 1, "time": t, "value": v} for t, v in rows]
        )


def make_meteo(db):
    meteo = Meteo.__new__(Meteo)
    meteo.db_da = db
    return meteo


def test_get_avg_temperature_averages_one_day(db):
    day = datetime.datetime(2026, 6, 15)
    t0 = int(day.timestamp())
    put_temps(db, [(t0 + i * 3600, 10.0 + i) for i in range(24)])

    avg = make_meteo(db).get_avg_temperature(day)

    assert avg == pytest.approx(10.0 + sum(range(24)) / 24)


def test_get_avg_temperature_does_not_leak_into_the_next_day(db):
    day = datetime.datetime(2026, 6, 15)
    t0 = int(day.timestamp())
    put_temps(db, [(t0, 10.0), (t0 + 25 * 3600, 30.0)])  # one hour into day 2

    avg = make_meteo(db).get_avg_temperature(day)

    assert avg == pytest.approx(10.0)


def test_get_avg_temperature_returns_none_without_data(db, caplog):
    avg = make_meteo(db).get_avg_temperature(datetime.datetime(2026, 6, 15))
    assert avg is None
    assert "Geen temperatuurprognose" in caplog.text


def test_calc_graaddagen_with_missing_data_returns_zero_instead_of_raising(db, caplog):
    result = make_meteo(db).calc_graaddagen(datetime.datetime(2026, 6, 15))
    assert result == 0.0
    assert "niet te berekenen" in caplog.text


def test_calc_graaddagen_computes_normally_when_data_is_present(db):
    day = datetime.datetime(2026, 1, 15)
    t0 = int(day.timestamp())
    put_temps(db, [(t0 + i * 3600, 4.0) for i in range(24)])  # cold winter day

    result = make_meteo(db).calc_graaddagen(day)

    assert result == pytest.approx(12.0)  # 16 - 4
