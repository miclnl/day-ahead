"""Consolidation of grid data from the Home Assistant recorder into `values`.

The task used to fail three times over: get_latest_present() called dict() on
a SQLAlchemy Row, calc_cost() was called with three arguments but took two,
and the epoch was derived from the naive local "tijd" column, which pandas
reads as UTC. These tests drive consolidate_data() with the recorder and the
tariffs replaced by small frames, against a real SQLite day-ahead database.
"""

import datetime

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from dao.lib.db_manager import DBmanagerObj  # noqa: E402
from dao.prog.da_report import Report  # noqa: E402

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
                {"id": 1, "code": "cons", "name": "Verbruik", "dim": "kWh"},
                {"id": 2, "code": "prod", "name": "Productie", "dim": "kWh"},
                {"id": 3, "code": "cost", "name": "Kosten", "dim": "eur"},
                {"id": 4, "code": "profit", "name": "Opbrengst", "dim": "eur"},
            ],
        )
    return manager


def _hours(start: datetime.datetime, count: int):
    return [start + datetime.timedelta(hours=i) for i in range(count)]


def make_report(db, start, sensor_values, prices):
    """A Report with the recorder and the tariff lookups replaced."""
    report = Report.__new__(Report)
    report.db_da = db
    report.grid_dict = {
        "cons": {"sensors": ["sensor.cons"], "dim": "kWh"},
        "prod": {"sensors": ["sensor.prod"], "dim": "kWh"},
        "cost": {"sensors": "calc", "function": "calc_cost"},
        "profit": {"sensors": "calc", "function": "calc_cost"},
    }
    hours = _hours(start, len(prices))

    def get_sensor_sum(sensors, vanaf, tot, col_name):
        code = "cons" if sensors == ["sensor.cons"] else "prod"
        rows = [
            {
                "tijd": pd.Timestamp(moment),
                "utc": moment.timestamp(),  # recorder start_ts: the true epoch
                col_name: value,
            }
            for moment, value in zip(hours, sensor_values[code])
            if vanaf <= moment < tot
        ]
        frame = pd.DataFrame(rows, columns=["tijd", "utc", col_name])
        frame.index = pd.to_datetime(frame["tijd"])
        return frame

    def get_price_data(vanaf, tot, interval="1hour"):
        rows = [
            {"time": moment, "da_ex": p, "da_cons": p * 1.2, "da_prod": p * 0.9}
            for moment, p in zip(hours, prices)
            if vanaf <= moment < tot
        ]
        return pd.DataFrame(rows, columns=["time", "da_ex", "da_cons", "da_prod"])

    report.get_sensor_sum = get_sensor_sum
    report.get_price_data = get_price_data
    return report


def stored(db, code):
    from sqlalchemy import Table, select

    values = Table("values", db.metadata, autoload_with=db.engine)
    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    query = (
        select(values.c.time, values.c.value)
        .where(values.c.variabel == variabel.c.id, variabel.c.code == code)
        .order_by(values.c.time)
    )
    with db.engine.connect() as connection:
        return [(int(t), round(v, 6)) for t, v in connection.execute(query)]


def test_consolidation_stores_the_recorder_epoch(db):
    start = datetime.datetime(2026, 6, 15, 0, 0)
    report = make_report(
        db, start, {"cons": [1.0, 2.0, 3.0], "prod": [0.5, 0.0, 0.25]}, [0.10, 0.20, 0.30]
    )

    report.consolidate_data(_start=start, _end=start + datetime.timedelta(hours=3))

    t0 = int(start.timestamp())
    assert stored(db, "cons") == [(t0, 1.0), (t0 + HOUR, 2.0), (t0 + 2 * HOUR, 3.0)]
    assert stored(db, "prod") == [(t0, 0.5), (t0 + HOUR, 0.0), (t0 + 2 * HOUR, 0.25)]
    # cost = consumption x consumer tariff, profit = production x producer tariff
    assert stored(db, "cost") == [
        (t0, round(1.0 * 0.12, 6)),
        (t0 + HOUR, round(2.0 * 0.24, 6)),
        (t0 + 2 * HOUR, round(3.0 * 0.36, 6)),
    ]
    assert stored(db, "profit") == [
        (t0, round(0.5 * 0.09, 6)),
        (t0 + HOUR, 0.0),
        (t0 + 2 * HOUR, round(0.25 * 0.27, 6)),
    ]


def test_consolidation_continues_after_the_latest_stored_hour(db):
    start = datetime.datetime(2026, 6, 15, 0, 0)
    report = make_report(
        db, start, {"cons": [1.0, 2.0, 3.0], "prod": [0.0, 0.0, 0.0]}, [0.1, 0.1, 0.1]
    )
    end = start + datetime.timedelta(hours=3)
    # Pretend the first hour was consolidated earlier.
    first = pd.DataFrame(
        [[str(int(start.timestamp())), "cons", 1.0]], columns=["time", "code", "value"]
    )
    db.savedata(first, tablename="values")
    assert report.get_latest_present("cons") == start
    assert report.get_latest_present("cost") == datetime.datetime(2020, 1, 1)

    # No explicit start: cons resumes at the second hour, the rest from scratch.
    report.consolidate_data(_start=None, _end=end)

    t0 = int(start.timestamp())
    assert stored(db, "cons") == [(t0, 1.0), (t0 + HOUR, 2.0), (t0 + 2 * HOUR, 3.0)]
    assert len(stored(db, "cost")) == 3


def test_consolidation_without_prices_skips_cost_but_stores_energy(db, caplog):
    start = datetime.datetime(2026, 6, 15, 0, 0)
    report = make_report(db, start, {"cons": [1.0], "prod": [0.0]}, [0.1])
    report.get_price_data = lambda vanaf, tot, interval="1hour": pd.DataFrame(
        columns=["time", "da_ex", "da_cons", "da_prod"]
    )

    report.consolidate_data(_start=start, _end=start + datetime.timedelta(hours=1))

    assert len(stored(db, "cons")) == 1
    assert stored(db, "cost") == []
    assert "Geen tarieven" in caplog.text
