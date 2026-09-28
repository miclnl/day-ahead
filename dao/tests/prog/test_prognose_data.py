"""Regression tests for DBmanagerObj.get_prognose_data.

The 15-minute branch used to rebuild the epoch column from the naive local
"tijd" column with ``astype(int) // 1e9``. That was off by the UTC offset and,
under pandas 3 unit inference, produced values in the 1970s. The optimizer feeds
that epoch to the sun position calculation, so the PV forecast for every solar
device without an ML model was wrong in 15-minute mode.
"""

import datetime

import pytest

pytest.importorskip("pandas")

from dao.lib.db_manager import DBmanagerObj  # noqa: E402

HOUR = 3600


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
            [
                {"id": 1, "code": "gr", "name": "Globale straling", "dim": "J/cm2"},
                {"id": 2, "code": "temp", "name": "Temperatuur", "dim": "C"},
            ],
        )
    return manager


def put_prognoses(db, code, rows):
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


def _t0() -> int:
    """A whole hour in the recent past, so the end-date default logic applies."""
    now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    return int((now - datetime.timedelta(hours=6)).timestamp())


def test_hourly_epoch_is_taken_from_the_database(db):
    t0 = _t0()
    put_prognoses(db, "gr", [(t0 + i * HOUR, 10.0 * i) for i in range(4)])
    put_prognoses(db, "temp", [(t0 + i * HOUR, 15.0 + i) for i in range(4)])

    df = db.get_prognose_data(start=t0, end=t0 + 4 * HOUR, interval="1hour")

    assert list(df["time"]) == [t0 + i * HOUR for i in range(4)]
    assert list(df["glob_rad"]) == [0.0, 10.0, 20.0, 30.0]


def test_quarter_epochs_are_derived_from_the_hourly_epoch(db):
    t0 = _t0()
    put_prognoses(db, "gr", [(t0 + i * HOUR, 10.0 * i) for i in range(4)])
    put_prognoses(db, "temp", [(t0 + i * HOUR, 15.0 + i) for i in range(4)])

    df = db.get_prognose_data(start=t0, end=t0 + 4 * HOUR, interval="15min")

    assert list(df.columns) == ["time", "tijd", "temp", "glob_rad"]
    assert len(df) == 16
    assert list(df["time"]) == [t0 + 900 * k for k in range(16)]
    assert str(df["time"].dtype) == "int64"
    # The quarter values of one hour average to the hourly value.
    assert abs(df["glob_rad"].iloc[4:8].mean() - 10.0) < 1e-9
    assert abs(df["temp"].iloc[4:8].mean() - 16.0) < 1e-9
    # The wall clock column stays consistent with the epoch column.
    expected_local = datetime.datetime.fromtimestamp(t0 + 900)
    assert df["tijd"].iloc[1].to_pydatetime() == expected_local


def test_quarter_data_is_joined_on_time_not_on_position(db):
    t0 = _t0()
    # Temperature starts one hour later than radiation: the frames may not be
    # zipped by position, the shared hours must line up on the epoch.
    put_prognoses(db, "gr", [(t0 + i * HOUR, 10.0 * i) for i in range(4)])
    put_prognoses(db, "temp", [(t0 + i * HOUR, 15.0 + i) for i in range(1, 4)])

    df = db.get_prognose_data(start=t0, end=t0 + 4 * HOUR, interval="15min")

    assert list(df["time"]) == [t0 + HOUR + 900 * k for k in range(12)]
    assert abs(df["glob_rad"].iloc[0:4].mean() - 10.0) < 1e-9
    assert abs(df["temp"].iloc[0:4].mean() - 16.0) < 1e-9


def test_too_few_hours_returns_an_empty_frame_instead_of_raising(db):
    t0 = _t0()
    put_prognoses(db, "gr", [(t0, 10.0)])
    put_prognoses(db, "temp", [(t0, 15.0)])

    df = db.get_prognose_data(start=t0, end=t0 + 4 * HOUR, interval="15min")

    assert df.empty
    assert list(df.columns) == ["time", "tijd", "temp", "glob_rad"]
