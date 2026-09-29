"""get_sensor_data's aggregate (maand/dag) query: valid under strict SQL.

The SELECT used to include t2.start_ts and unit_of_measurement bare, next to
func.sum()/func.max() under a GROUP BY. SQLite and MariaDB tolerate that;
PostgreSQL and MySQL 8 (ONLY_FULL_GROUP_BY, the default there) reject it.
"""

import datetime

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sqlalchemy")

from sqlalchemy.dialects import postgresql

from dao.lib.db_manager import DBmanagerObj
from dao.prog.da_report import Report

HOUR = 3600


@pytest.fixture
def db(tmp_path):
    from sqlalchemy import (
        BigInteger,
        Column,
        Float,
        Integer,
        String,
        Table,
        insert,
    )

    manager = DBmanagerObj(
        db_dialect="sqlite", db_name="home-assistant_v2.db", db_path=str(tmp_path)
    )
    metadata = manager.metadata
    statistics_meta = Table(
        "statistics_meta",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("statistic_id", String, unique=True, nullable=False),
        Column("unit_of_measurement", String),
    )
    Table(
        "statistics",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("metadata_id", Integer, nullable=False),
        Column("start_ts", BigInteger, nullable=False),
        Column("state", Float),
        Column("mean", Float),
        Column("sum", Float),
    )
    metadata.create_all(manager.engine)
    with manager.engine.begin() as connection:
        connection.execute(
            insert(statistics_meta),
            [{"id": 1, "statistic_id": "sensor.grid_import", "unit_of_measurement": "kWh"}],
        )
    return manager


def put_hourly_states(db, states):
    from sqlalchemy import Table, insert

    statistics = Table("statistics", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        connection.execute(
            insert(statistics),
            [
                {"metadata_id": 1, "start_ts": ts, "state": state, "mean": state, "sum": state}
                for ts, state in states
            ],
        )


def make_report(db):
    report = Report.__new__(Report)
    report.db_ha = db
    return report


def test_daily_aggregate_sums_the_hourly_deltas(db):
    """t1 ranges over [vanaf - 1h, tot - 1h), paired with t2 = t1 + 3600, so
    the pairs covering [vanaf, tot) at an hourly step are (t0,t0+1h) and
    (t0+1h,t0+2h) when tot = vanaf + 3h.
    """
    day = datetime.datetime(2026, 3, 15)
    t0 = int(day.timestamp())
    put_hourly_states(db, [(t0, 10.0), (t0 + HOUR, 12.0), (t0 + 2 * HOUR, 15.0)])

    result = make_report(db).get_sensor_data(
        "sensor.grid_import", day, day + datetime.timedelta(hours=3), "cons", agg="dag"
    )

    assert len(result) == 1
    assert result["cons"].iloc[0] == pytest.approx(5.0)  # (12-10) + (15-12)
    assert result["tijd"].iloc[0] == day


def test_monthly_aggregate_query_has_no_bare_column_under_group_by():
    """Compiled for PostgreSQL, every selected column must be either the
    GROUP BY key or wrapped in an aggregate function."""
    from sqlalchemy import Table, and_, case, func, select

    # __new__, not DBmanagerObj(...): the constructor eagerly probes a real
    # connection to fail fast, and this test only needs the dialect-aware SQL
    # helpers (month(), month_start(), ...), which read self.db_dialect and
    # self.TARGET_TIMEZONE.
    db = DBmanagerObj.__new__(DBmanagerObj)
    db.db_dialect = "postgresql"
    db.TARGET_TIMEZONE = "Europe/Amsterdam"
    # Build the same shape of query get_sensor_data builds for agg="maand",
    # against a throwaway in-memory metadata (no live connection needed to
    # compile and inspect the SQL text).
    from sqlalchemy import BigInteger, Column, Float, Integer, MetaData, String

    metadata = MetaData()
    statistics = Table(
        "statistics", metadata,
        Column("metadata_id", Integer), Column("start_ts", BigInteger),
        Column("state", Float),
    )
    statistics_meta = Table(
        "statistics_meta", metadata,
        Column("id", Integer), Column("statistic_id", String),
        Column("unit_of_measurement", String),
    )
    t1 = statistics.alias("t1")
    t2 = statistics.alias("t2")
    v1 = statistics_meta.alias("v1")
    column = db.month(t2.c.start_ts).label("maand")
    column2 = func.min(db.month_start(t2.c.start_ts)).label("tijd")
    columns = [
        column,
        column2,
        func.max(db.from_unixtime(t2.c.start_ts)).label("tot"),
        func.min(t2.c.start_ts).label("utc"),
        func.sum(case((t2.c.state > t1.c.state, t2.c.state - t1.c.state), else_=0)).label("cons"),
        func.max(v1.c.unit_of_measurement).label("dim"),
    ]
    query = (
        select(*columns)
        .select_from(
            t1.join(t2, t2.c.start_ts == t1.c.start_ts + 3600).join(
                v1, (v1.c.id == t1.c.metadata_id) & (v1.c.id == t2.c.metadata_id)
            )
        )
        .where(and_(v1.c.statistic_id == "sensor.grid_import", t1.c.state.isnot(None)))
        .group_by("maand")
    )

    compiled = str(query.compile(dialect=postgresql.dialect()))

    # Every column reaching the client is wrapped in min/max/sum, or is the
    # literal group-by expression itself (the strftime-based "maand" label).
    assert "t2.start_ts AS utc" not in compiled
    assert "v1.unit_of_measurement AS dim" not in compiled
    assert "min(t2.start_ts) AS utc" in compiled
    assert "max(v1.unit_of_measurement) AS dim" in compiled
