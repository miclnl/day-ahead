"""Timestamps go into the database as epoch integers, converted in Python.

Every range query used to hand SQL a formatted string and let the database
parse it back into an epoch:

    t1.c.time >= self.db_da.unix_timestamp(vanaf.strftime("%Y-%m-%d %H:%M:%S"))

Which zone that string was taken to be in depended on the dialect and on
server settings, so the same query selected a different range on SQLite,
MySQL and PostgreSQL. The columns hold epochs, so the conversion belongs in
Python, once, against the timezone configured in Home Assistant.

PostgreSQL was the one that actually misbehaved. TARGET_TIMEZONE held
config.time_zone all along but was never used -- the single line that
applied it is commented out -- so to_timestamp/to_char rendered and parsed
in the session zone, UTC on a stock server, which is why those
installations saw everything shifted by one or two hours.
"""

import datetime
from zoneinfo import ZoneInfo

import pytest

from dao.lib.db_manager import DBmanagerObj

AMSTERDAM = ZoneInfo("Europe/Amsterdam")


def manager(dialect="sqlite", zone="Europe/Amsterdam"):
    """A DBmanagerObj without a live connection.

    __init__ probes the database to fail fast; these tests only need the
    conversion helpers, which read db_dialect and TARGET_TIMEZONE.
    """
    obj = DBmanagerObj.__new__(DBmanagerObj)
    obj.db_dialect = dialect
    obj.TARGET_TIMEZONE = zone
    return obj


class TestEpoch:
    def test_a_naive_datetime_is_read_as_local_time(self):
        db = manager()

        result = db.epoch(datetime.datetime(2026, 6, 15, 12, 0))

        expected = datetime.datetime(2026, 6, 15, 12, 0, tzinfo=AMSTERDAM)
        assert result == int(expected.timestamp())

    def test_an_aware_datetime_keeps_its_own_offset(self):
        db = manager()
        moment = datetime.datetime(2026, 6, 15, 12, 0, tzinfo=ZoneInfo("UTC"))

        assert db.epoch(moment) == int(moment.timestamp())

    def test_the_configured_zone_is_what_counts_not_the_process_zone(self):
        """Two managers, same wall clock reading, different configured zones:
        the epochs must differ by the offset between them."""
        amsterdam = manager(zone="Europe/Amsterdam")
        utc = manager(zone="UTC")
        moment = datetime.datetime(2026, 6, 15, 12, 0)

        # Amsterdam is UTC+2 in June, so the same local reading is two hours
        # earlier in absolute time.
        assert utc.epoch(moment) - amsterdam.epoch(moment) == 2 * 3600

    def test_summer_and_winter_offsets_are_both_handled(self):
        db = manager()

        winter = db.epoch(datetime.datetime(2026, 1, 15, 12, 0))
        summer = db.epoch(datetime.datetime(2026, 6, 15, 12, 0))

        assert winter == int(
            datetime.datetime(2026, 1, 15, 12, 0, tzinfo=AMSTERDAM).timestamp()
        )
        assert summer == int(
            datetime.datetime(2026, 6, 15, 12, 0, tzinfo=AMSTERDAM).timestamp()
        )
        # A fixed offset would have put these exactly 151 days apart; the DST
        # change makes it an hour less.
        assert (summer - winter) % 86400 == 86400 - 3600

    def test_the_hour_that_occurs_twice_resolves_to_the_first(self):
        """On the October changeover 02:30 happens twice. zoneinfo picks the
        first (fold=0), which is what a report asking for "02:30" means in
        practice: the earlier of the two."""
        db = manager()
        ambiguous = datetime.datetime(2026, 10, 25, 2, 30)

        result = db.epoch(ambiguous)

        first = datetime.datetime(2026, 10, 25, 2, 30, tzinfo=AMSTERDAM, fold=0)
        assert result == int(first.timestamp())

    def test_the_hour_that_does_not_exist_does_not_raise(self):
        """On the March changeover 02:30 never happens. It must still yield a
        usable epoch rather than take a report down with an exception."""
        db = manager()

        result = db.epoch(datetime.datetime(2026, 3, 29, 2, 30))

        assert isinstance(result, int)

    def test_an_unknown_zone_falls_back_to_utc_with_a_warning(self, caplog):
        db = manager(zone="Mars/Olympus_Mons")

        result = db.epoch(datetime.datetime(2026, 6, 15, 12, 0))

        assert result == int(
            datetime.datetime(2026, 6, 15, 12, 0, tzinfo=ZoneInfo("UTC")).timestamp()
        )
        assert "Onbekende tijdzone" in caplog.text


class TestPostgresRendersInTheConfiguredZone:
    """The concrete bug: to_char over a timestamptz renders in the session
    zone, which nothing sets, so a stock server used UTC."""

    @pytest.mark.parametrize(
        "helper",
        ["from_unixtime", "month", "month_start", "day", "day_start", "hour", "hour_start"],
    )
    def test_every_label_helper_pins_the_zone(self, helper):
        from sqlalchemy import BigInteger, Column, MetaData, Table

        db = manager(dialect="postgresql")
        table = Table("values", MetaData(), Column("time", BigInteger))

        sql = str(getattr(db, helper)(table.c.time).compile(
            compile_kwargs={"literal_binds": True}
        ))

        assert "timezone" in sql.lower()
        assert "Europe/Amsterdam" in sql

    def test_sqlite_still_uses_the_process_zone(self):
        """Not a regression, a documented limitation: SQLite cannot render a
        named timezone at all, so it relies on the container's TZ matching
        what Home Assistant is configured with. The supervisor sets it, so
        they normally agree."""
        from sqlalchemy import BigInteger, Column, MetaData, Table

        db = manager(dialect="sqlite")
        table = Table("values", MetaData(), Column("time", BigInteger))

        sql = str(db.hour_start(table.c.time).compile(
            compile_kwargs={"literal_binds": True}
        ))

        assert "localtime" in sql


class TestNoMoreStringRoundTrip:
    def test_unix_timestamp_is_no_longer_used_for_ranges(self):
        """The helper still exists for anything that genuinely needs a
        database-side conversion, but no range query should be building one
        out of a formatted datetime any more."""
        import pathlib

        repo = pathlib.Path(__file__).resolve().parents[3]
        for name in ("dao/lib/db_manager.py", "dao/prog/da_report.py"):
            text = (repo / name).read_text()
            for line in text.splitlines():
                # Strip a trailing comment too: a stale one documenting the
                # old, broken form is exactly what this is looking for.
                code = line.split("#", 1)[0].strip()
                if not code or "``" in code:
                    continue
                if "unix_timestamp(" in code and "def " not in code:
                    assert "strftime" not in code, f"{name}: {code}"


class TestWhichZoneWins:
    """Order of authority: an explicit time_zone in options.json, then what
    Home Assistant reports, then the container's own zone."""

    def test_an_explicit_override_is_kept(self):
        db = DBmanagerObj.__new__(DBmanagerObj)
        db.db_dialect = "sqlite"
        db.TARGET_TIMEZONE = "Asia/Tokyo"

        assert db.tzinfo.key == "Asia/Tokyo"

    def test_no_override_falls_back_to_the_container_zone(self, monkeypatch):
        """config.time_zone is an optional database override and is None on a
        default installation. Passing that None straight through would have
        left the zone unset, and the conversion silently on UTC."""
        from dao.lib import db_manager

        monkeypatch.setenv("TZ", "Asia/Tokyo")
        assert db_manager._container_zone_name() == "Asia/Tokyo"

    def test_the_container_zone_is_read_from_the_environment(self, monkeypatch):
        """The supervisor sets TZ for add-ons, which is why SQLite's
        "localtime" and MySQL's session zone have been right all along."""
        from dao.lib import db_manager

        monkeypatch.delenv("TZ", raising=False)
        # Whatever the host uses; the point is that it resolves to something
        # usable rather than None.
        assert db_manager._container_zone_name()
