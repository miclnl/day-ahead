"""Additive DDL must preserve legacy data and have portable FK/key definitions."""

import os
import socket

import pytest
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    event,
    inspect,
    select,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.dialects.mysql.mariadb import MariaDBDialect
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex, CreateTable

NAMES = {f"settlement_{name}" for name in ("tariff", "input", "run", "interval", "adjustment", "book", "spot")}


def test_schema_repeat_does_not_touch_values(tmp_path):
    from dao.prog.check_db import CheckDB
    from dao.prog.settlement.repository import SettlementRepository

    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    metadata = MetaData()
    legacy = []
    for name in ("values", "prognoses", "variabel"):
        legacy.append(Table(name, metadata, Column("id", Integer, primary_key=True),
                            Column("code", String(20)), Column("value", Float)))
    metadata.create_all(engine)
    with engine.begin() as connection:
        for item in legacy:
            connection.execute(item.insert(), [{"id": 1, "code": "cost", "value": 0.123456789},
                                               {"id": 2, "code": "profit", "value": -0.5}])
        before = {item.name: connection.execute(select(item)).all() for item in legacy}
    checker = CheckDB.__new__(CheckDB)
    checker.engine = engine
    checker.ensure_settlement_tables()
    checker.ensure_settlement_tables()
    SettlementRepository(engine).ensure_schema()
    with engine.connect() as connection:
        after = {item.name: connection.execute(select(item)).all() for item in legacy}
    assert before == after
    assert set(inspect(engine).get_table_names()) == NAMES | {item.name for item in legacy}
    engine.dispose()


@pytest.mark.parametrize("dialect", [sqlite.dialect(), postgresql.dialect(), MariaDBDialect()])
def test_all_seven_tables_and_indexes_compile_with_matching_foreign_keys(dialect):
    from dao.prog.settlement.repository import _METADATA

    metadata = _METADATA
    assert set(metadata.tables) == NAMES
    for item in metadata.sorted_tables:
        assert str(CreateTable(item).compile(dialect=dialect))
        for index in item.indexes:
            assert str(CreateIndex(index).compile(dialect=dialect))
        for foreign_key in item.foreign_keys:
            assert foreign_key.parent.type.compile(dialect=dialect) == foreign_key.column.type.compile(dialect=dialect)


def test_sqlite_enforces_explicit_positive_input_revision_foreign_key(tmp_path):
    from dao.prog.settlement.repository import SettlementRepository

    engine = create_engine(f"sqlite:///{tmp_path / 'foreign-keys.db'}")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    SettlementRepository(engine).ensure_schema()
    metadata = MetaData()
    metadata.reflect(engine)
    item = metadata.tables["settlement_input"]
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(item.insert().values(meter_id="synthetic", start_ts=1, end_ts=2,
                                                input_revision=0, input_hash="a" * 64,
                                                content_hash="b" * 64, input_json="{}"))
    interval = metadata.tables["settlement_interval"]
    input_fks = [constraint for constraint in interval.foreign_key_constraints
                 if constraint.referred_table.name == "settlement_input"]
    assert len(input_fks) == 1
    assert [element.parent.name for element in input_fks[0].elements] == ["meter_id", "start_ts", "input_revision"]
    engine.dispose()


def local_server_url(backend):
    url_text = os.getenv(f"SETTLEMENT_TEST_{backend}_URL")
    assert url_text, "An explicit test database URL is required"
    url = make_url(url_text)
    assert url.host in {"localhost", "127.0.0.1", "::1", os.getenv("SETTLEMENT_TEST_LOCAL_HOST")} and (
        url.database or "").startswith("settlement_a6_test_"), "Only dedicated, explicitly local A6 test databases are permitted"
    return url


@pytest.mark.parametrize("backend,port", [("POSTGRESQL", 5432), ("MARIADB", 3306)])
def test_available_local_server_database(backend, port):
    """Availability is separate from DDL compilation and the parameterized repository tests."""
    url_text = os.getenv(f"SETTLEMENT_TEST_{backend}_URL")
    if not url_text:
        with socket.socket() as connection:
            connection.settimeout(1)
            reachable = connection.connect_ex(("127.0.0.1", port)) == 0
        pytest.skip(f"No SETTLEMENT_TEST_{backend}_URL; localhost:{port} reachable={reachable}; no server runtime proof")
    engine = create_engine(local_server_url(backend))
    with engine.connect() as connection:
        assert connection.scalar(select(1)) == 1
    engine.dispose()
