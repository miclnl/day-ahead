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
    for name in ("prognoses", "values"):
        Table(
            name,
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


def _rows(db, table):
    import pandas as pd
    from sqlalchemy import Table, select

    t = Table(table, db.metadata, autoload_with=db.engine)
    with db.engine.connect() as connection:
        result = connection.execute(select(t.c.variabel, t.c.time, t.c.value).order_by(t.c.time))
        return [tuple(r) for r in result]


def test_savedata_inserts_and_updates_in_one_statement(db):
    import pandas as pd

    t0 = _t0()
    first = pd.DataFrame(
        [[str(t0), "gr", 10.0], [str(t0 + HOUR), "gr", 20.0], [t0, "temp", 15]],
        columns=["time", "code", "value"],
    )
    db.savedata(first, tablename="values")
    assert _rows(db, "values") == [(1, t0, 10.0), (2, t0, 15.0), (1, t0 + HOUR, 20.0)]

    # Same keys again: the values are replaced, no duplicate rows, no error.
    second = pd.DataFrame(
        [[str(t0), "gr", 11.0], [str(t0), "gr", 12.0]], columns=["time", "code", "value"]
    )
    db.savedata(second, tablename="values")
    assert _rows(db, "values") == [(1, t0, 12.0), (2, t0, 15.0), (1, t0 + HOUR, 20.0)]


def test_savedata_skips_garbage_and_unknown_codes(db, caplog):
    import pandas as pd

    t0 = _t0()
    frame = pd.DataFrame(
        [
            [str(t0), "gr", float("nan")],
            [str(t0), "gr", float("inf")],
            [str(t0), "gr", float("-inf")],
            [str(t0), "gr", "abc"],
            [str(t0), "gr", True],
            [str(t0), "nope", 1.0],
            ["not-a-time", "gr", 1.0],
            [str(t0 + HOUR), "gr", 5],
        ],
        columns=["time", "code", "value"],
    )
    db.savedata(frame, tablename="values")
    assert _rows(db, "values") == [(1, t0 + HOUR, 5.0)]
    assert "Onbekende code opslaan data: nope" in caplog.text


def test_savedata_with_an_empty_frame_is_a_no_op(db):
    import pandas as pd

    db.savedata(pd.DataFrame(columns=["time", "code", "value"]), tablename="values")
    assert _rows(db, "values") == []
