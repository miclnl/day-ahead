"""The forecast archive creates its own table when it is missing.

Reported from a live add-on: every optimisation run ended with

    waarschuwing: Prognose-archief niet bijgewerkt: forecasts

The plan itself was fine -- the archive write happens after it and its
failure is caught -- but the message is a NoSuchTableError, whose str() is
nothing but the table name, so it says neither what is wrong nor what to do.

check_db.py creates the table at start-up, but run.sh swallows a failure of
that script with a single log line, so anything going wrong earlier in it
leaves the table uncreated and this warning repeating forever. The table
definition now lives next to the code that writes to it and save_forecasts
creates it on demand.
"""

import time

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, insert, inspect
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from dao.lib.db_manager import DBmanagerObj, forecasts_table


@pytest.fixture
def db(tmp_path):
    """A database with "variabel" but deliberately without "forecasts"."""
    manager = DBmanagerObj(
        db_dialect="sqlite", db_name="day_ahead.db", db_path=str(tmp_path)
    )
    variabel = Table(
        "variabel",
        manager.metadata,
        Column("id", Integer, primary_key=True),
        Column("code", String(10), unique=True, nullable=False),
        Column("name", String(50), nullable=False),
        Column("dim", String(10), nullable=False),
        Column("aggregate", String(3), nullable=False),
    )
    manager.metadata.create_all(manager.engine)
    with manager.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {
                    "id": 1,
                    "code": "gr",
                    "name": "Globale straling",
                    "dim": "J/cm2",
                    "aggregate": "avg",
                }
            ],
        )
    return manager


def a_future_row(code="gr", value=1.0):
    """save_forecasts drops targets in the past, so aim well ahead."""
    return (int(time.time()) + 7200, code, value)


class TestEnsureForecastsTable:
    def test_it_creates_the_table(self, db):
        assert inspect(db.engine).has_table("forecasts") is False

        assert db.ensure_forecasts_table() is True

        assert inspect(db.engine).has_table("forecasts") is True

    def test_it_is_idempotent(self, db):
        assert db.ensure_forecasts_table() is True
        assert db.ensure_forecasts_table() is True

    def test_it_reports_a_failure_instead_of_raising(self, db, monkeypatch, caplog):
        """The archive is a diagnostic, not part of the plan: a database that
        will not take the table must not take the optimisation down with it."""

        def refuse(*args, **kwargs):
            raise RuntimeError("no permission to create tables")

        monkeypatch.setattr(db.metadata, "remove", refuse)

        assert db.ensure_forecasts_table() is False
        assert "forecasts" in caplog.text
        assert "overgeslagen" in caplog.text


class TestForeignKeyTypeFollowsTheReferencedColumn:
    """MySQL and MariaDB only accept a foreign key when both columns have
    exactly the same type, signedness included. Installations from before
    the schema moved into Python carry ``variabel.id`` as ``INT(10)
    UNSIGNED``, so spelling ``Integer`` on the referencing column rendered a
    signed ``INTEGER`` and the server refused the whole table:

        (1005, "Can't create table `day_ahead`.`forecasts`
                (errno: 150 \\"Foreign key constraint is incorrectly formed\\")")

    Every optimiser run then logged that the archive was skipped. SQLite
    ignores the type of a foreign key column entirely, which is why the
    other tests in this file ran green throughout.
    """

    @staticmethod
    def variabel_column_ddl(metadata):
        ddl = str(
            CreateTable(forecasts_table(metadata)).compile(dialect=mysql.dialect())
        )
        return next(
            line.strip().rstrip(",")
            for line in ddl.splitlines()
            if line.strip().startswith("variabel ")
        )

    def test_it_follows_an_unsigned_id_on_an_upgraded_database(self):
        metadata = MetaData()
        Table(
            "variabel",
            metadata,
            Column(
                "id",
                mysql.INTEGER(display_width=10, unsigned=True),
                primary_key=True,
            ),
        )

        assert (
            self.variabel_column_ddl(metadata)
            == "variabel INTEGER(10) UNSIGNED NOT NULL"
        )

    def test_it_follows_a_signed_id_on_a_fresh_database(self):
        metadata = MetaData()
        Table("variabel", metadata, Column("id", Integer, primary_key=True))

        assert self.variabel_column_ddl(metadata) == "variabel INTEGER NOT NULL"


class TestSaveForecastsRecovers:
    def test_a_missing_table_is_created_and_the_rows_land(self, db):
        written = db.save_forecasts([a_future_row()], issued_ts=int(time.time()))

        assert written == 1
        assert inspect(db.engine).has_table("forecasts") is True

    def test_it_does_not_raise_the_opaque_nosuchtable_error(self, db):
        """The regression: NoSuchTableError('forecasts') reached the caller,
        which logged its str() and left the operator with the word
        "forecasts" and nothing else."""
        from sqlalchemy.exc import NoSuchTableError

        try:
            db.save_forecasts([a_future_row()], issued_ts=int(time.time()))
        except NoSuchTableError as exception:  # pragma: no cover
            pytest.fail(f"still raising NoSuchTableError: {exception}")

    def test_it_gives_up_quietly_when_the_table_cannot_be_created(
        self, db, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            db, "ensure_forecasts_table", lambda: False
        )

        written = db.save_forecasts([a_future_row()], issued_ts=int(time.time()))

        assert written == 0

    def test_nothing_is_created_when_there_is_nothing_to_archive(self, db):
        """A row whose target is already in the past carries no information
        about forecast skill and is dropped before the table is touched."""
        written = db.save_forecasts(
            [(int(time.time()) - 7200, "gr", 1.0)], issued_ts=int(time.time())
        )

        assert written == 0
        assert inspect(db.engine).has_table("forecasts") is False

    def test_a_second_call_reuses_the_table(self, db):
        first = db.save_forecasts([a_future_row()], issued_ts=int(time.time()))
        second = db.save_forecasts(
            [a_future_row(value=2.0)], issued_ts=int(time.time())
        )

        assert first == 1
        assert second == 1
