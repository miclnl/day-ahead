"""E1d boundaries: one bounded supplier file and a lifetime read-only reader."""

import gc
import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from types import FunctionType

import pytest
from sqlalchemy import MetaData, Row, RowMapping, Table, func, insert, select, text
from sqlalchemy.engine import Connection, CursorResult, Dialect, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import Pool

from dao.prog.settlement import reconciliation_io as reader
from dao.prog.settlement.reconciliation import SupplierPeriod
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_publication import pricing_config
from dao.tests.settlement.test_repository import METER, publish
from dao.tests.settlement.test_repository import calculate as calculate
from dao.tests.settlement.test_repository import inputs as inputs
from dao.tests.settlement.test_repository import repo as repo
from dao.tests.settlement.test_repository import terms as terms

LIMIT = 1024 * 1024


def supplier_document(**changes):
    document = {
        "schema_version": "supplier-period-v1",
        "meter_id": METER,
        "contract_id": "synthetic-2027",
        "timezone": "Europe/Amsterdam",
        "start_ts": START_TS,
        "end_ts": START_TS + 1800,
        "currency": "EUR",
        "energy_unit": "kWh",
        "sign_convention": "cost_positive_revenue_positive",
        "total_scope": "book_close_events_v1",
        "import_kwh": "2",
        "export_kwh": "0",
        "import_amounts": {"ex_vat": "0.20", "vat": "0", "gross": "0.20"},
        "base_export_amounts": {"ex_vat": "0", "vat": "0", "gross": "0"},
        "bonus_amounts": {"ex_vat": "0", "vat": "0", "gross": "0"},
        "credits": [],
        "total": "0.20",
        "rounding": {
            "method": "period_component_half_up_v1",
            "money_quantum": "0.01",
            "money_mode": "ROUND_HALF_UP",
            "aggregation": "sum_unrounded_then_round_each_component",
            "total_rule": "sum_rounded_gross_components",
            "energy_quantum": "0.001",
            "energy_mode": "ROUND_HALF_UP",
            "evidence_ref": "synthetic-rounding-v1",
        },
    }
    document.update(changes)
    return document


def write_supplier(tmp_path, document=None, *, name="supplier.json"):
    path = Path(tmp_path) / name
    if isinstance(document, (bytes, bytearray)):
        path.write_bytes(bytes(document))
    else:
        body = json.dumps(supplier_document() if document is None else document)
        path.write_text(body, encoding="utf-8")
    return path


def options_document(tmp_path, *, meter=METER, database=None):
    pricing = pricing_config().model_dump(mode="json", by_alias=True)
    pricing["settlement"]["meter id"] = meter
    return {
        "database da": database or {"engine": "sqlite", "database": "settlement.db",
                                    "db_path": str(tmp_path)},
        "database ha": {"engine": "sqlite", "database": "recorder.db", "db_path": str(tmp_path)},
        "time_zone": "Europe/Amsterdam",
        "prices": pricing,
    }


def write_options(tmp_path, document=None, *, name="options.json"):
    path = Path(tmp_path) / name
    body = json.dumps(options_document(tmp_path) if document is None else document)
    path.write_text(body, encoding="utf-8")
    return path


def database_section(repository):
    """The reader configuration for the very database the fixture published into."""
    url = repository.engine.url
    if url.get_backend_name() == "sqlite":
        location = Path(url.database)
        return {"engine": "sqlite", "database": location.name, "db_path": str(location.parent)}
    return {"engine": "mysql" if url.get_backend_name() == "mysql" else "postgresql",
            "server": url.host, "port": url.port, "username": url.username,
            "password": url.password, "database": url.database}


def descriptor_count():
    return len(os.listdir("/proc/self/fd"))


def database_descriptors(location):
    """The open descriptors that point at the database file itself."""
    found = []
    for name in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if target == str(location):
            found.append(name)
    return found


def reachable_raw_objects(root, depth=4):
    """Every raw database object that the facade's attributes can reach.

    Only attributes are followed; no guarded operation is called, so the walk
    has no side effect on the book.
    """
    raw_types = (Engine, Connection, CursorResult, Row, RowMapping, Pool, Dialect)
    found = []
    seen = set()

    def walk(obj, path, level):
        if level > depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, raw_types):
            found.append((path, type(obj).__name__))
            return
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            if callable(value):
                continue
            walk(value, f"{path}.{name}", level + 1)

    walk(root, "reader", 0)
    return found


def book_table(repository):
    return Table("settlement_book", MetaData(), autoload_with=repository.engine)


def book_rows(repository):
    table = book_table(repository)
    with repository.engine.connect() as connection:
        return connection.execute(select(table)).mappings().all()


@pytest.fixture
def book(repo, terms, inputs, calculate):
    """One real, published book in the fixture database."""
    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots), snapshots, calculate)
    return repo


def test_e1d_supplier_descriptor_limits(tmp_path, monkeypatch):
    path = write_supplier(tmp_path)
    period = reader.load_supplier(path)
    assert isinstance(period, SupplierPeriod)
    assert period.meter_id == METER and period.import_kwh == Decimal("2")
    assert period.credits == () and period.rounding.method == "period_component_half_up_v1"

    body = json.dumps(supplier_document())
    exact = Path(tmp_path) / "exact.json"
    exact.write_text(body + " " * (LIMIT - len(body)), encoding="utf-8")
    assert exact.stat().st_size == LIMIT
    assert reader.load_supplier(exact).meter_id == METER
    oversize = Path(tmp_path) / "oversize.json"
    oversize.write_text(body + " " * (LIMIT - len(body) + 1), encoding="utf-8")
    assert oversize.stat().st_size == LIMIT + 1
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(oversize)

    link = Path(tmp_path) / "link.json"
    link.symlink_to(path)
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(link)
    directory = Path(tmp_path) / "directory.json"
    directory.mkdir()
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(directory)
    fifo = Path(tmp_path) / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(fifo)
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(Path("/dev/null"))
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(Path(tmp_path) / "absent.json")

    # Every failing path closes its descriptor again.
    before = descriptor_count()
    for refused in (oversize, link, directory, fifo, Path("/dev/null"), Path(tmp_path) / "absent.json"):
        with pytest.raises(reader.SupplierFileError):
            reader.load_supplier(refused)
    assert descriptor_count() == before

    # The file is never copied into a temporary file or cache.
    def forbidden(*args, **kwargs):
        pytest.fail("the supplier file must never be copied")

    monkeypatch.setattr(tempfile, "mkstemp", forbidden)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", forbidden)
    monkeypatch.setattr(tempfile, "TemporaryFile", forbidden)
    listing = sorted(item.name for item in Path(tmp_path).iterdir())
    assert reader.load_supplier(path).meter_id == METER
    assert sorted(item.name for item in Path(tmp_path).iterdir()) == listing


def test_e1d_duplicate_unknown_and_nested_keys(tmp_path):
    body = json.dumps(supplier_document())
    duplicate = Path(tmp_path) / "duplicate.json"
    duplicate.write_text('{"schema_version": "supplier-period-v1", "meter_id": "a",'
                         ' "meter_id": "b"}', encoding="utf-8")
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(duplicate)
    nested = Path(tmp_path) / "nested-duplicate.json"
    nested.write_text(body.replace('"gross": "0.20"', '"gross": "0.20", "gross": "0.21"', 1),
                      encoding="utf-8")
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(nested)

    unknown = supplier_document()
    unknown["iban"] = "NL00SENTINEL"
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(write_supplier(tmp_path, unknown, name="unknown.json"))
    nested_unknown = supplier_document()
    nested_unknown["import_amounts"] = {"ex_vat": "0.20", "vat": "0", "gross": "0.20",
                                        "customer_number": "12345"}
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(write_supplier(tmp_path, nested_unknown, name="nested-unknown.json"))
    missing = {key: value for key, value in supplier_document().items() if key != "total"}
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(write_supplier(tmp_path, missing, name="missing.json"))

    deep = supplier_document()
    deep["credits"] = [{"kind": "spot_zero", "period_start_ts": START_TS,
                         "period_end_ts": START_TS + 1800,
                         "amounts": {"ex_vat": "0", "vat": "0", "gross": {"deeper": {"x": 1}}}}]
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(write_supplier(tmp_path, deep, name="deep.json"))
    excessive = Path(tmp_path) / "excessive.json"
    excessive.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(excessive)

    invalid_utf8 = Path(tmp_path) / "invalid-utf8.json"
    invalid_utf8.write_bytes(b'{"schema_version": "\xff\xfe"}')
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(invalid_utf8)
    for literal in ("NaN", "Infinity", "-Infinity"):
        broken = Path(tmp_path) / f"{literal}.json"
        broken.write_text(f'{{"schema_version": "supplier-period-v1", "total": {literal}}}',
                          encoding="utf-8")
        with pytest.raises(reader.SupplierFileError):
            reader.load_supplier(broken)
    float_value = Path(tmp_path) / "float.json"
    float_value.write_text(body.replace('"total": "0.20"', '"total": 0.2', 1), encoding="utf-8")
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(float_value)


@pytest.mark.parametrize("field,value", [
    ("total", "1e2"),
    ("total", "1E-2"),
    ("total", " 1"),
    ("total", "1_000"),
    ("total", "0" * 64 + "1"),
    ("total", "9" * 37),
    ("total", "0." + "1" * 13),
    ("total", 1),
    ("total", True),
    ("import_kwh", "-0.1"),
    ("import_kwh", "0." + "1" * 10),
    ("export_kwh", "0.1234567890123"),
])
def test_e1d_decimal_and_credit_validation(tmp_path, field, value):
    document = supplier_document()
    document[field] = value
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(write_supplier(tmp_path, document, name="decimal.json"))


def test_e1d_decimal_and_credit_validation_accepts_boundaries(tmp_path):
    document = supplier_document(total="0." + "1" * 12, import_kwh="0." + "1" * 9)
    period = reader.load_supplier(write_supplier(tmp_path, document, name="boundary.json"))
    assert period.total == Decimal("0." + "1" * 12)
    assert period.import_kwh == Decimal("0." + "1" * 9)


def _credit(index, kind="spot_zero"):
    return {"kind": kind, "period_start_ts": START_TS + index * 900,
            "period_end_ts": START_TS + (index + 1) * 900,
            "amounts": {"ex_vat": "0", "vat": "0", "gross": "0"}}


def test_e1d_decimal_and_credit_validation_credits(tmp_path):
    accepted = supplier_document(credits=[_credit(index) for index in range(64)])
    period = reader.load_supplier(write_supplier(tmp_path, accepted, name="credits64.json"))
    assert len(period.credits) == 64
    refused = supplier_document(credits=[_credit(index) for index in range(65)])
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(write_supplier(tmp_path, refused, name="credits65.json"))
    overlapping = supplier_document(credits=[_credit(0), _credit(0)])
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(write_supplier(tmp_path, overlapping, name="overlap.json"))
    reversed_period = supplier_document(credits=[{"kind": "spot_zero",
                                                  "period_start_ts": START_TS + 900,
                                                  "period_end_ts": START_TS,
                                                  "amounts": {"ex_vat": "0", "vat": "0", "gross": "0"}}])
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(write_supplier(tmp_path, reversed_period, name="reversed.json"))
    long_period = supplier_document(end_ts=START_TS + 370 * 86400)
    assert reader.load_supplier(write_supplier(tmp_path, long_period, name="long.json")).end_ts
    too_long = supplier_document(end_ts=START_TS + 370 * 86400 + 1)
    with pytest.raises(reader.SupplierFileError):
        reader.load_supplier(write_supplier(tmp_path, too_long, name="too-long.json"))
    # An unknown kind or rounding method stays inspectable instead of being removed.
    unknown = supplier_document(credits=[_credit(0, kind="future_kind")])
    unknown["rounding"] = dict(unknown["rounding"], method="future_rounding_v9")
    period = reader.load_supplier(write_supplier(tmp_path, unknown, name="unknown-kind.json"))
    assert period.credits[0].kind == "future_kind"
    assert period.rounding.method == "future_rounding_v9"


def test_e1d_config_meter_before_connect(tmp_path, monkeypatch):
    path = write_options(tmp_path, options_document(tmp_path, meter="other-meter"))

    def forbidden(*args, **kwargs):
        pytest.fail("the reader must refuse a foreign meter before it connects")

    monkeypatch.setattr(reader, "create_engine", forbidden)
    with pytest.raises(reader.ReconciliationConfigError):
        with reader.open_reconciliation_reader(path, expected_meter_id=METER):
            pytest.fail("no reader may be produced")

    without_meter = options_document(tmp_path)
    del without_meter["prices"]["settlement"]["meter id"]
    absent = write_options(tmp_path, without_meter, name="options-without-meter.json")
    with pytest.raises(reader.ReconciliationConfigError):
        with reader.open_reconciliation_reader(absent, expected_meter_id=METER):
            pytest.fail("no reader may be produced")

    ambiguous = options_document(tmp_path)
    ambiguous["prices"]["settlement"]["meter_id"] = "another-meter"
    both = write_options(tmp_path, ambiguous, name="options-ambiguous.json")
    with pytest.raises(reader.ReconciliationConfigError):
        with reader.open_reconciliation_reader(both, expected_meter_id=METER):
            pytest.fail("no reader may be produced")

    matching = write_options(tmp_path, options_document(tmp_path), name="options.json")
    with pytest.raises(reader.ReconciliationConfigError):
        with reader.open_reconciliation_reader(matching, expected_meter_id=""):
            pytest.fail("no reader may be produced")


def test_e1d_missing_sqlite_is_not_created(tmp_path):
    path = write_options(tmp_path, options_document(tmp_path))
    target = Path(tmp_path) / "settlement.db"
    assert not target.exists()
    with pytest.raises(reader.ReconciliationReaderError):
        with reader.open_reconciliation_reader(path, expected_meter_id=METER):
            pytest.fail("no reader may be produced")
    assert not target.exists()
    assert not Path(str(target) + "-wal").exists()
    assert not Path(str(target) + "-journal").exists()
    assert sorted(item.name for item in Path(tmp_path).iterdir()) == ["options.json"]


def test_e1d_readonly_every_entry(tmp_path, book, monkeypatch):
    options = write_options(tmp_path, options_document(tmp_path, database=database_section(book)))
    before = book_rows(book)
    table = book_table(book)
    count = select(func.count()).select_from(table)
    opened = {}
    real_create_engine = reader.create_engine

    def recording_create_engine(url, *args, **kwargs):
        opened["url"] = url
        return real_create_engine(url, *args, **kwargs)

    monkeypatch.setattr(reader, "create_engine", recording_create_engine)
    with reader.open_reconciliation_reader(options, expected_meter_id=METER) as repository:
        engine = repository.engine
        assert not hasattr(engine, "raw_connection")
        assert not hasattr(engine, "execute")
        assert not hasattr(engine, "exec_driver_sql")
        # The handle itself is read-only, not only the guard in front of it.
        engine_url = opened["url"]
        if engine_url.get_backend_name() == "sqlite":
            assert engine_url.query.get("mode") == "ro"
            assert "immutable" not in engine_url.query
        # A second connection and a fresh transaction both read.
        with engine.connect() as connection:
            # The connection offers no way back to the raw driver.
            assert not hasattr(connection, "connection")
            assert not hasattr(connection, "exec_driver_sql")
            assert connection.execute(count).scalar() == len(before)
        with engine.begin() as connection:
            assert connection.execute(count).scalar() == len(before)
        # A committed transaction leaves no writable transaction behind.
        with engine.begin() as connection:
            connection.execute(count)
        with engine.begin() as connection:
            assert connection.execute(count).scalar() == len(before)
        # Every write statement is refused by the guard, not by the database.
        with engine.connect() as connection:
            with pytest.raises(reader.ReadOnlyViolation):
                connection.execute(insert(table).values(meter_id=METER))
            with pytest.raises(reader.ReadOnlyViolation):
                connection.execute(text("UPDATE settlement_book SET generation = 9"))
            with pytest.raises(reader.ReadOnlyViolation):
                connection.execute(text("DROP TABLE settlement_book"))
            # A row lock is a write on the book and is refused as well.
            with pytest.raises(reader.ReadOnlyViolation):
                connection.execute(select(table).with_for_update())
        # The repository refuses its write surface before any SQL runs.
        for name in ("ensure_schema", "register_inputs", "register_prices", "commit",
                     "_ensure_book", "_registration", "_transaction", "_lock_book",
                     "_insert_many", "_close_versions"):
            with pytest.raises(reader.ReadOnlyViolation):
                getattr(repository, name)()
        # Schema creation through the reachable engine is refused as well.
        from dao.prog.settlement.repository import _METADATA

        with pytest.raises(reader.ReadOnlyViolation):
            _METADATA.create_all(engine, checkfirst=True)
        # A pool reconnect stays read-only.
        engine.dispose()
        with engine.connect() as connection:
            assert connection.execute(count).scalar() == len(before)
    assert book_rows(book) == before
    # The database itself refuses a write through an independent read-only handle.
    location = Path(book.engine.url.database)
    handle = sqlite3.connect(f"file:{location}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            handle.execute("UPDATE settlement_book SET generation = 9")
    finally:
        handle.close()


def test_e1d_readonly_no_raw_escape(tmp_path, book):
    """The public surface of the reader exposes no raw engine, connection or driver.

    The claim is scoped to the public surface: attributes, methods and result
    objects.  The in-process introspection boundary is a separate, documented
    fact; see ``test_e1d_in_process_introspection_reaches_the_raw_engine``.
    """
    options = write_options(tmp_path, options_document(tmp_path, database=database_section(book)))
    before = book_rows(book)
    table = book_table(book)
    with reader.open_reconciliation_reader(options, expected_meter_id=METER) as repository:
        engine = repository.engine
        # The public repository surface hands out a facade, never the raw engine.
        assert not isinstance(engine, Engine)
        assert not hasattr(engine, "__dict__")
        for name in ("_engine", "_connection", "pool", "dialect", "dbapi", "raw_connection",
                     "execute", "exec_driver_sql", "scalar"):
            assert not hasattr(engine, name), name
        # Nothing raw is reachable through the repository's own attributes.
        assert reachable_raw_objects(repository) == []

        with engine.connect() as connection:
            assert not isinstance(connection, Connection)
            assert not hasattr(connection, "__dict__")
            for name in ("_connection", "connection", "engine", "pool", "dialect", "dbapi",
                         "exec_driver_sql", "raw_connection", "cursor"):
                assert not hasattr(connection, name), name
            assert reachable_raw_objects(connection) == []

            # The read route still works: mapping rows, plain rows and iteration.
            rows = connection.execute(select(table.c.meter_id)).mappings().all()
            assert [row["meter_id"] for row in rows] == [METER]
            assert [row.meter_id for row in connection.execute(select(table.c.meter_id))] == [METER]
            first = connection.execute(select(table.c.meter_id)).mappings().first()
            assert first["meter_id"] == METER

            # A result exposes no route back to the connection that made it.
            result = connection.execute(select(table.c.meter_id))
            assert not isinstance(result, CursorResult)
            assert not hasattr(result, "__dict__")
            for name in ("connection", "engine", "pool", "dbapi", "context", "frozen"):
                assert not hasattr(result, name), name
            assert reachable_raw_objects(result) == []
            assert result.mappings().first()["meter_id"] == METER

            # The write route stays closed on every entry.
            with pytest.raises(reader.ReadOnlyViolation):
                connection.execute(insert(table).values(meter_id=METER))
            with pytest.raises(reader.ReadOnlyViolation):
                connection.execute(text("UPDATE settlement_book SET generation = 9"))
            with pytest.raises(reader.ReadOnlyViolation):
                connection.execute(text("DROP TABLE settlement_book"))
            with pytest.raises(reader.ReadOnlyViolation):
                connection.execute(select(table).with_for_update())

        # A fresh transaction reads as well and never opens a writable one.
        with engine.begin() as transaction:
            assert not isinstance(transaction, Connection)
            assert not hasattr(transaction, "_transaction")
            assert transaction.execute(select(table.c.meter_id)).mappings().first()["meter_id"] == METER
    assert book_rows(book) == before


def test_e1d_in_process_introspection_reaches_the_raw_engine(tmp_path, book):
    """The in-process introspection boundary is real and is documented as such.

    The public surface of the reader is clean (see
    ``test_e1d_readonly_no_raw_escape``).  Inside one process the raw engine is
    nevertheless reachable: the guarded methods capture it in a closure cell,
    the context manager keeps it in a frame and the garbage collector walks to
    it from the facade's own class.  Section 26.7 leaves the wrapper form to
    the builder and prescribes no subprocess boundary, so this route is a
    documented boundary and not a defect that this suite pretends away.
    """
    options = write_options(tmp_path, options_document(tmp_path, database=database_section(book)))
    with reader.open_reconciliation_reader(options, expected_meter_id=METER) as repository:
        engine = repository.engine

        # Route 1: the closure cell of a guarded method.
        cells = engine.connect.__func__.__closure__ or ()
        captured = [cell.cell_contents for cell in cells]
        assert any(isinstance(item, Engine) for item in captured)

        # Route 2: the frame of the context manager that opens a transaction.
        manager = engine.begin()
        with manager:
            frame = getattr(manager.gen, "gi_frame", None)
            assert frame is not None
            assert isinstance(frame.f_locals.get("engine"), Engine)

        # Route 3: the garbage collector walks from the facade's class.
        functions = [value for value in vars(type(engine)).values()
                     if isinstance(value, FunctionType)]
        assert functions
        reached = []
        for function in functions:
            for cell in (function.__closure__ or ()):
                reached.extend(gc.get_referents(cell))
        assert any(isinstance(item, Engine) for item in reached)


def test_e1d_row_mapping_is_immutable_and_hashable(tmp_path, book):
    """The row mapping is immutable and hashable, exactly like the raw RowMapping."""
    options = write_options(tmp_path, options_document(tmp_path, database=database_section(book)))
    table = book_table(book)
    with book.engine.connect() as raw_connection:
        raw = raw_connection.execute(select(table.c.meter_id)).mappings().first()
        assert isinstance(raw, RowMapping)
        # The raw RowMapping is hashable and compares by value; the frozen row
        # keeps that contract instead of silently becoming unhashable.
        assert isinstance(hash(raw), int)
        assert raw == {"meter_id": METER}

    with reader.open_reconciliation_reader(options, expected_meter_id=METER) as repository:
        with repository.engine.connect() as connection:
            row = connection.execute(select(table.c.meter_id)).mappings().first()
            assert isinstance(row, Mapping)
            assert isinstance(hash(row), int)
            assert row == {"meter_id": METER}
            assert row == raw
            assert hash(row) == hash(raw)
            assert row["meter_id"] == METER and row.meter_id == METER
            assert list(row.keys()) == list(raw.keys())

            # Immutable: neither an item nor an attribute can be replaced.
            with pytest.raises(TypeError):
                row["meter_id"] = "other"
            with pytest.raises((AttributeError, TypeError)):
                row.meter_id = "other"

            # No mutable container is reachable from the row.
            assert not hasattr(row, "__dict__")
            for name in dir(row):
                if name.startswith("__"):
                    continue
                value = getattr(row, name)
                assert not isinstance(value, (dict, list, set, bytearray)), name


def test_e1d_row_mapping_refuses_attribute_assignment(tmp_path, book):
    """The frozen row refuses every attribute assignment, like the raw RowMapping.

    The private slots of the row are storage, not a door: they are refused
    exactly like a column attribute, so the row is really frozen after its
    construction and not only "immutable" in the sense of the values.
    """
    options = write_options(tmp_path, options_document(tmp_path, database=database_section(book)))
    table = book_table(book)
    with book.engine.connect() as raw_connection:
        raw = raw_connection.execute(select(table.c.meter_id)).mappings().first()
        # The raw RowMapping refuses every attribute assignment as well.
        with pytest.raises(AttributeError):
            raw.meter_id = "other"
        with pytest.raises(AttributeError):
            raw._data = None

    with reader.open_reconciliation_reader(options, expected_meter_id=METER) as repository:
        with repository.engine.connect() as connection:
            row = connection.execute(select(table.c.meter_id)).mappings().first()

            # A column attribute cannot be replaced.
            with pytest.raises(AttributeError):
                row.meter_id = "other"

            # And neither can a private slot: the row is frozen after construction.
            with pytest.raises(AttributeError):
                row._items = ()
            with pytest.raises(AttributeError):
                row._keys = ()

            # The row still carries exactly the values the reader produced.
            assert row["meter_id"] == METER and row.meter_id == METER
            assert list(row.keys()) == ["meter_id"]
            assert isinstance(hash(row), int)


def test_e1d_server_session_is_read_only_before_the_first_statement():
    """The reader's server session is read-only before the first statement.

    The read-only session is installed on the connect event of the engine that
    the reader opens, so it is in place before any application statement can
    run.  The server-side effect of the statements is an open gate; what is
    proven here is the mechanism, its ordering and its exact wording.
    """
    from sqlalchemy import URL

    recorded = []

    class FakeCursor:
        def execute(self, statement, *args, **kwargs):
            recorded.append(statement)

        def close(self):
            recorded.append("CLOSE")

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def rollback(self):
            recorded.append("ROLLBACK")

    for backend, expected in (
        ("postgresql",
         ["SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"]),
        ("mysql",
         ["SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ",
          "SET SESSION TRANSACTION READ ONLY"]),
    ):
        recorded.clear()
        driver = "postgresql+psycopg2" if backend == "postgresql" else "mysql+pymysql"
        url = URL.create(driver, username="synthetic", host="127.0.0.1", port=5432,
                         database="synthetic_reader")
        engine = reader._reader_engine(url)
        try:
            # The listener is registered when the engine is created, that is
            # before any session and therefore before any statement.
            listeners = [function for function in engine.pool.dispatch.connect.listeners
                         if "readonly_session" in getattr(function, "__qualname__", "")]
            assert len(listeners) == 1
            listeners[0](FakeConnection(), None)
            assert recorded[:len(expected)] == expected
            # The session is left idle, so the first application statement is
            # not swallowed by the setup transaction.
            assert recorded[-1] == "ROLLBACK"
        finally:
            engine.dispose()


def test_e1d_read_route_through_the_public_api(tmp_path, book):
    """Every public read method of the repository still works on the facade."""
    options = write_options(tmp_path, options_document(tmp_path, database=database_section(book)))
    with reader.open_reconciliation_reader(options, expected_meter_id=METER) as repository:
        view = repository.read(METER, START_TS, START_TS + 1800)
        assert view.generation == 1
        assert view.lines and view.net_cost is not None

        batch = repository.read_prices(METER, START_TS, START_TS + 1800)
        assert batch.input_revision == 1

        registered = repository.read_inputs(METER, START_TS, START_TS + 1800)
        assert registered.input_revision == 1

        reference = repository.read_book_reference(METER)
        assert reference.input_revision == 1

        basis = repository.read_reconciliation(METER, START_TS, START_TS + 1800)
        assert basis.view.generation == 1
        assert basis.sources.pin.generation == 1


def test_e1d_readonly_setup_failure_closed(tmp_path, monkeypatch):
    options = write_options(tmp_path, options_document(tmp_path))
    (Path(tmp_path) / "settlement.db").write_bytes(b"")

    def forbidden(*args, **kwargs):
        raise reader.ReconciliationReaderError("database_unavailable")

    monkeypatch.setattr(reader, "create_engine", forbidden)
    with pytest.raises(reader.ReconciliationReaderError):
        with reader.open_reconciliation_reader(options, expected_meter_id=METER):
            pytest.fail("no reader may be produced")

    # A database without the settlement schema fails closed instead of guessing.
    monkeypatch.undo()
    with reader.open_reconciliation_reader(options, expected_meter_id=METER) as repository:
        with pytest.raises(SQLAlchemyError):
            repository.read_reconciliation(METER, START_TS, START_TS + 1800)


def test_e1d_privacy_and_cleanup(tmp_path, caplog, book):
    sentinel = "SUPPLIER-SENTINEL-CONTRACT"
    document = supplier_document(contract_id=sentinel, total="not-a-decimal")
    path = write_supplier(tmp_path, document, name="sentinel.json")
    with pytest.raises(reader.SupplierFileError) as failure:
        reader.load_supplier(path)
    message = str(failure.value)
    assert sentinel not in message
    assert "not-a-decimal" not in message
    assert str(tmp_path) not in message
    assert failure.value.category

    # A secret in the configuration never reaches the reader's error text.
    secret = "SECRET-SENTINEL-VALUE"
    server = {"engine": "mysql", "server": "127.0.0.1", "port": 3306, "username": "synthetic",
              "password": "!secret db_password", "database": "settlement_a6_test_reader"}
    (Path(tmp_path) / "secrets.json").write_text(json.dumps({"other_key": secret}),
                                                 encoding="utf-8")
    options = write_options(tmp_path, options_document(tmp_path, database=server),
                            name="server-options.json")
    with pytest.raises(reader.ReconciliationIOError) as failure:
        with reader.open_reconciliation_reader(options, expected_meter_id=METER):
            pytest.fail("no reader may be produced")
    assert secret not in str(failure.value)
    assert "secrets.json" not in str(failure.value)
    assert str(tmp_path) not in str(failure.value)
    assert secret not in caplog.text

    # The reader disposes its own connections when the context closes.
    book_options = write_options(tmp_path,
                                 options_document(tmp_path, database=database_section(book)),
                                 name="book-options.json")
    location = Path(book.engine.url.database)
    baseline = len(database_descriptors(location))
    with reader.open_reconciliation_reader(book_options, expected_meter_id=METER) as repository:
        with repository.engine.connect() as connection:
            connection.execute(select(func.count()).select_from(book_table(book)))
        # The reader really holds a connection of its own while it is open.
        assert len(database_descriptors(location)) > baseline
    # Closing the reader closes those connections instead of parking them.
    assert len(database_descriptors(location)) == baseline


def test_e1d_secret_reference_resolves_locally(tmp_path):
    """The secret resolution itself is local; the server session behind it is not."""
    from dao.prog.config.models.base import SecretStr

    directory = Path(tmp_path)
    (directory / "secrets.json").write_text(json.dumps({"db_password": "resolved-secret"}),
                                            encoding="utf-8")
    config = directory / "options.json"
    config.write_text("{}", encoding="utf-8")

    # A resolvable reference returns its value; a plain password is used as it is.
    assert reader._resolve_secret(SecretStr("!secret db_password"), config) == "resolved-secret"
    assert reader._resolve_secret(SecretStr("plain-password"), config) == "plain-password"

    # Every unresolvable reference fails closed with one fixed category.
    with pytest.raises(reader.ReconciliationConfigError) as failure:
        reader._resolve_secret(SecretStr("!secret absent_key"), config)
    assert failure.value.category == "config_secret_missing"
    assert "absent_key" not in str(failure.value)
    assert str(directory) not in str(failure.value)

    (directory / "secrets.json").write_text(json.dumps({"db_password": ""}), encoding="utf-8")
    with pytest.raises(reader.ReconciliationConfigError) as failure:
        reader._resolve_secret(SecretStr("!secret db_password"), config)
    assert failure.value.category == "config_secret_missing"

    (directory / "secrets.json").write_text(json.dumps(["db_password"]), encoding="utf-8")
    with pytest.raises(reader.ReconciliationConfigError) as failure:
        reader._resolve_secret(SecretStr("!secret db_password"), config)
    assert failure.value.category == "config_secret_missing"

    (directory / "secrets.json").unlink()
    with pytest.raises(reader.ReconciliationConfigError) as failure:
        reader._resolve_secret(SecretStr("!secret db_password"), config)
    assert failure.value.category == "config_secret_missing"
