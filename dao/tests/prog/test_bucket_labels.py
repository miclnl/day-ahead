"""get_da_data / get_energy_balance_data: bucket labels for partial buckets.

Both used to label a bucket with the earliest timestamp that happened to
have data (min(time)), instead of the bucket's own start (the 1st of the
month, midnight, or the hour mark). For a period that does not start on a
bucket boundary -- "dit contractjaar" or "365 dagen" starting mid-month is
the real-world case -- that label did not match generate_df's bucket-start
label, and the whole first bucket silently disappeared from the report.
"""

import datetime

import pytest

pytest.importorskip("pandas")

from dao.lib.db_manager import DBmanagerObj  # noqa: E402
from dao.prog.da_report import Report  # noqa: E402

HOUR = 3600
DAY = 24 * HOUR


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
            insert(variabel), [{"id": 1, "code": "cons", "name": "Verbruik", "dim": "kWh"}]
        )
    return manager


def put_hours(db, code, hours):
    from sqlalchemy import Table, insert, select

    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    values = Table("values", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        ident = connection.execute(
            select(variabel.c.id).where(variabel.c.code == code)
        ).scalar_one()
        connection.execute(
            insert(values), [{"variabel": ident, "time": t, "value": v} for t, v in hours]
        )


def make_report(db):
    report = Report.__new__(Report)
    report.db_da = db
    return report


class TestGetDaData:
    def test_a_full_month_bucket_is_labelled_at_the_start_of_the_month(self, db):
        march = datetime.datetime(2026, 3, 1)
        put_hours(db, "cons", [(int(march.timestamp()) + i * HOUR, 1.0) for i in range(3)])

        result = make_report(db).get_da_data(
            "cons", march, march + datetime.timedelta(days=1), get_interval=None, rep_interval="maand"
        )

        assert result["tijd"].iloc[0] == march

    def test_a_partial_first_month_is_still_labelled_at_the_1st_not_at_the_first_datapoint(self, db):
        """The real-world case: a report period ('dit contractjaar', '365
        dagen') that starts mid-month. Data exists from the 15th onward, but
        the bucket must still be labelled 1 March so it lines up with
        generate_df's own bucket-start label."""
        start = datetime.datetime(2026, 3, 15)
        put_hours(db, "cons", [(int(start.timestamp()) + i * HOUR, 1.0) for i in range(3)])

        result = make_report(db).get_da_data(
            "cons", start, start + datetime.timedelta(days=1), get_interval=None, rep_interval="maand"
        )

        assert result["tijd"].iloc[0] == datetime.datetime(2026, 3, 1)

    def test_a_partial_first_day_is_labelled_at_midnight(self, db):
        start = datetime.datetime(2026, 3, 15, 14, 0)
        put_hours(db, "cons", [(int(start.timestamp()) + i * HOUR, 1.0) for i in range(3)])

        result = make_report(db).get_da_data(
            "cons", start, start + datetime.timedelta(hours=3), get_interval=None, rep_interval="dag"
        )

        assert result["tijd"].iloc[0] == datetime.datetime(2026, 3, 15)

    def test_hourly_buckets_are_labelled_at_the_hour_mark(self, db):
        start = datetime.datetime(2026, 3, 15, 14, 20)
        put_hours(db, "cons", [(int(start.timestamp()), 1.0)])

        result = make_report(db).get_da_data(
            "cons", start, start + datetime.timedelta(hours=1), get_interval=None, rep_interval="uur"
        )

        assert result["tijd"].iloc[0] == datetime.datetime(2026, 3, 15, 14, 0)


class TestGetEnergyBalanceData:
    def _report(self, db, vanaf, tot, interval="maand"):
        report = make_report(db)
        report.periodes = {"test": {"vanaf": vanaf, "tot": tot, "interval": interval}}
        report.energy_balance_dict = {
            "cons": {"dim": "kWh", "sign": "pos", "name": "Verbruik", "sensors": ["sensor.x"]}
        }
        return report

    def test_a_partial_first_month_matches_generate_dfs_bucket_label(self, db):
        start = datetime.datetime(2026, 3, 15)
        # Ends exactly where the data ends, so last_moment == tot and the
        # method returns after the DA-sourced fetch below, without falling
        # through to the Home Assistant recorder branch this test does not
        # set up.
        end = start + datetime.timedelta(hours=3)
        put_hours(db, "cons", [(int(start.timestamp()) + i * HOUR, 1.0) for i in range(3)])

        report = self._report(db, start, end)
        result, _last_moment = report.get_energy_balance_data("test")

        # generate_df labels the March bucket at 1 March even though the
        # period itself starts on the 15th; the DA-sourced value must have
        # landed on that same row instead of being silently dropped.
        assert result.loc[datetime.datetime(2026, 3, 1), "cons"] == pytest.approx(3.0)
