"""Bounded supplier file and a lifetime read-only book reader.

Two boundaries live here and nowhere else:

``load_supplier``
    Decodes one regular UTF-8 JSON file through a single descriptor that is
    opened with ``O_NOFOLLOW`` and ``O_NONBLOCK``.  The read is bounded to
    1 MiB, a symlink, directory, FIFO or device is refused instead of
    followed or waited on, and the descriptor is closed on every path.  The
    file is never copied, cached or written anywhere.

``open_reconciliation_reader``
    Opens a second, separately opened connection that can only read.  SQLite
    is opened through the ``mode=ro`` URI (never a create-if-missing connect
    and never ``immutable=1``); a server dialect gets a read-only session
    before its first application statement.  Both are wrapped in a guard that
    refuses DML, DDL and raw access, so no writer or schema route is
    reachable through the returned repository.

    The guard exposes only guarded connect/begin/read/close/dispose.  The raw
    engine, its pool, its dialect and the raw connection are not reachable
    through any attribute of the facade, and a read result is frozen into rows
    that cannot reach the connection that produced them either.

Neither boundary logs, prints or returns supplier content, configuration
values, secrets or absolute paths; every failure carries a fixed category.
"""

import copy
import errno
import json
import os
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import Select, create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.exc import MultipleResultsFound, NoResultFound
from sqlalchemy.sql import visitors
from sqlalchemy.sql.ddl import ExecutableDDLElement
from sqlalchemy.sql.dml import Delete, Insert, Update

from dao.prog.settlement.reconciliation import (
    SupplierAmounts,
    SupplierCredit,
    SupplierPeriod,
    SupplierRounding,
)
from dao.prog.settlement.repository import SettlementRepository

MAX_SUPPLIER_BYTES = 1024 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024

_SUPPLIER_KEYS = frozenset({
    "schema_version", "meter_id", "contract_id", "timezone", "start_ts", "end_ts",
    "currency", "energy_unit", "sign_convention", "total_scope", "import_kwh",
    "export_kwh", "import_amounts", "base_export_amounts", "bonus_amounts",
    "credits", "total", "rounding",
})
_AMOUNT_KEYS = frozenset({"ex_vat", "vat", "gross"})
_CREDIT_KEYS = frozenset({"kind", "period_start_ts", "period_end_ts", "amounts"})
_ROUNDING_KEYS = frozenset({
    "method", "money_quantum", "money_mode", "aggregation", "total_rule",
    "energy_quantum", "energy_mode", "evidence_ref",
})


class ReconciliationIOError(RuntimeError):
    """A sanitised boundary failure; the text is a fixed category, never content."""

    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


class SupplierFileError(ReconciliationIOError):
    """The supplier file does not satisfy the bounded v1 file contract."""


class ReconciliationConfigError(ReconciliationIOError):
    """The configuration cannot start a read-only reconciliation reader."""


class ReconciliationReaderError(ReconciliationIOError):
    """The database behind the configuration cannot be read."""


class ReadOnlyViolation(RuntimeError):
    """A write, DDL or raw-access attempt on the read-only reader."""


def _open_regular(path: Path, error):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise error("file_missing") from None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise error("file_symlink") from None
        raise error("file_unreadable") from None
    return descriptor


def _read_bounded(descriptor: int, limit: int, error) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, limit + 1 - total)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise error("file_oversize")
        chunks.append(chunk)
    return b"".join(chunks)


def _no_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise SupplierFileError("file_duplicate_key")
        document[key] = value
    return document


def _reject_constant(_name):
    raise SupplierFileError("file_nonfinite")


def _reject_float(_text):
    raise SupplierFileError("file_float")


def _decode(text: str):
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys,
                          parse_constant=_reject_constant, parse_float=_reject_float)
    except SupplierFileError:
        raise
    except RecursionError:
        raise SupplierFileError("file_too_deep") from None
    except (ValueError, TypeError):
        raise SupplierFileError("file_invalid_json") from None


def _exact_keys(document, allowed):
    if not isinstance(document, dict):
        raise SupplierFileError("file_invalid_shape")
    keys = set(document)
    if keys - allowed:
        raise SupplierFileError("file_unknown_key")
    if allowed - keys:
        raise SupplierFileError("file_missing_key")


def _amounts(value):
    if value is None:
        return None
    _exact_keys(value, _AMOUNT_KEYS)
    return SupplierAmounts(value["ex_vat"], value["vat"], value["gross"])


def _credit(value):
    _exact_keys(value, _CREDIT_KEYS)
    return SupplierCredit(value["kind"], value["period_start_ts"], value["period_end_ts"],
                          _amounts(value["amounts"]))


def _rounding(value):
    if value is None:
        return None
    _exact_keys(value, _ROUNDING_KEYS)
    return SupplierRounding(value["method"], value["money_quantum"], value["money_mode"],
                             value["aggregation"], value["total_rule"], value["energy_quantum"],
                             value["energy_mode"], value["evidence_ref"])


def _period(document) -> SupplierPeriod:
    _exact_keys(document, _SUPPLIER_KEYS)
    credits = document["credits"]
    if credits is not None:
        if not isinstance(credits, list):
            raise SupplierFileError("file_invalid_shape")
        credits = tuple(_credit(item) for item in credits)
    return SupplierPeriod(
        document["schema_version"], document["meter_id"], document["contract_id"],
        document["timezone"], document["start_ts"], document["end_ts"], document["currency"],
        document["energy_unit"], document["sign_convention"], document["total_scope"],
        document["import_kwh"], document["export_kwh"], _amounts(document["import_amounts"]),
        _amounts(document["base_export_amounts"]), _amounts(document["bonus_amounts"]),
        credits, document["total"], _rounding(document["rounding"]),
    )


def load_supplier(path: Path) -> SupplierPeriod:
    """Read one bounded supplier file into an immutable observation."""
    descriptor = _open_regular(Path(path), SupplierFileError)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SupplierFileError("file_not_regular")
        data = _read_bounded(descriptor, MAX_SUPPLIER_BYTES, SupplierFileError)
    finally:
        os.close(descriptor)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SupplierFileError("file_not_utf8") from None
    document = _decode(text)
    try:
        return _period(document)
    except (TypeError, ValueError):
        raise SupplierFileError("file_invalid_values") from None


def _read_config_bytes(path: Path) -> bytes:
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_CONFIG_BYTES + 1)
    except FileNotFoundError:
        raise ReconciliationConfigError("config_missing") from None
    except OSError:
        raise ReconciliationConfigError("config_unreadable") from None
    if len(data) > MAX_CONFIG_BYTES:
        raise ReconciliationConfigError("config_oversize")
    return data


def _config_document(path: Path) -> dict:
    data = _read_config_bytes(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ReconciliationConfigError("config_not_utf8") from None
    try:
        document = json.loads(text, object_pairs_hook=_config_pairs)
    except ValueError:
        raise ReconciliationConfigError("config_invalid_json") from None
    if not isinstance(document, dict):
        raise ReconciliationConfigError("config_invalid_json")
    return document


def _config_pairs(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ReconciliationConfigError("config_duplicate_key")
        document[key] = value
    return document


def _raw_meter(document) -> str:
    """The meter the operator stated explicitly; never inherited from elsewhere."""
    prices = document.get("prices")
    settlement = prices.get("settlement") if isinstance(prices, dict) else None
    if not isinstance(settlement, dict):
        raise ReconciliationConfigError("config_meter_missing")
    stated = [settlement[key] for key in ("meter id", "meter_id") if key in settlement]
    if not stated:
        raise ReconciliationConfigError("config_meter_missing")
    if len(stated) > 1 and stated[0] != stated[1]:
        raise ReconciliationConfigError("config_meter_ambiguous")
    meter = stated[0]
    if not isinstance(meter, str) or not meter.strip():
        raise ReconciliationConfigError("config_meter_missing")
    return meter


def _validated_config(document):
    from dao.prog.config.loader import validate_config_data

    try:
        # The in-memory validation pattern: no loader, no disk migration.
        return validate_config_data(document)
    except Exception:
        raise ReconciliationConfigError("config_invalid") from None


def _resolve_secret(password, config_path: Path) -> str:
    if not password.is_secret_reference():
        return str(password)
    location = config_path.parent / "secrets.json"
    try:
        data = _read_config_bytes(location)
        secrets = json.loads(data.decode("utf-8"))
    except Exception:
        raise ReconciliationConfigError("config_secret_missing") from None
    if not isinstance(secrets, dict):
        raise ReconciliationConfigError("config_secret_missing")
    value = secrets.get(password.get_secret_key())
    if not isinstance(value, str) or not value:
        raise ReconciliationConfigError("config_secret_missing")
    return value


def _reader_url(model, config_path: Path) -> URL:
    database = model.database_da
    if database.engine == "sqlite":
        location = Path(os.path.abspath(database.db_path or "../data")) / database.database
        if not location.is_file():
            raise ReconciliationReaderError("database_missing")
        return URL.create("sqlite", database=f"file:{quote(str(location), safe='/')}",
                          query={"mode": "ro", "uri": "true"})
    if database.engine == "mysql":
        driver = "mysql+pymysql"
    elif database.engine == "postgresql":
        driver = "postgresql+psycopg2"
    else:
        raise ReconciliationConfigError("database_dialect_unsupported")
    password = None
    if database.password is not None:
        password = _resolve_secret(database.password, config_path)
    return URL.create(driver, username=database.username, password=password, host=database.server,
                      port=database.port, database=database.database)


def _install_readonly_session(engine, backend: str) -> None:
    """Make every transaction of this session read-only before it starts."""

    @event.listens_for(engine, "connect")
    def readonly_session(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        try:
            if backend == "postgresql":
                cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION "
                               "ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            else:
                cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                cursor.execute("SET SESSION TRANSACTION READ ONLY")
        finally:
            cursor.close()
        # Leave the session idle: the statements above must not hold a
        # transaction that would swallow the first application statement.
        rollback = getattr(dbapi_connection, "rollback", None)
        if rollback is not None:
            rollback()


def _reader_engine(url: URL):
    engine = create_engine(url, pool_pre_ping=True)
    backend = url.get_backend_name()
    if backend in ("mysql", "postgresql"):
        _install_readonly_session(engine, backend)
    return engine


def _read_only_statement(statement):
    """Accept only a controlled read; refuse writes, DDL and row locks."""
    if not isinstance(statement, Select):
        raise ReadOnlyViolation("only controlled SELECT statements are allowed")
    if getattr(statement, "_for_update_arg", None) is not None:
        raise ReadOnlyViolation("row locking is not allowed on the read-only reader")
    if getattr(statement, "_prefixes", None):
        raise ReadOnlyViolation("statement prefixes are not allowed on the read-only reader")
    for element in visitors.iterate(statement):
        if isinstance(element, (Insert, Update, Delete, ExecutableDDLElement)):
            raise ReadOnlyViolation("writes are not allowed on the read-only reader")
    return statement


class _ReadOnlyRow(Mapping):
    """One row of a read result, without a route back to the connection.

    The row is a frozen mapping: both ``row["column"]`` and ``row.column``
    work, the values live in an immutable tuple and the row is hashable, just
    like the raw ``RowMapping`` it stands in for.  A plain ``Mapping`` subclass
    would be unhashable, and that would silently deviate from the contract of
    the result it replaces.

    The row is frozen after its construction: every attribute assignment is
    refused, the private slots included, exactly as the raw ``RowMapping``
    refuses it.  A copy of a frozen row is the row itself.
    """

    __slots__ = ("_items", "_keys")

    def __init__(self, values):
        object.__setattr__(self, "_items", tuple(values.items()))
        object.__setattr__(self, "_keys", tuple(values))

    def __setattr__(self, name, value):
        raise AttributeError(f"attribute '{name}' of '{type(self).__name__}' objects is not writable")

    def __getitem__(self, key):
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._items)

    def __hash__(self):
        # The raw RowMapping hashes its values tuple; the frozen row keeps that
        # contract instead of becoming unhashable like a plain Mapping.
        return hash(tuple(value for _, value in self._items))

    def __copy__(self):
        # The row is frozen, so a copy of it is the row itself.
        return self

    def __deepcopy__(self, memo):
        return type(self)({name: copy.deepcopy(value, memo) for name, value in self._items})

    def __reduce__(self):
        return (type(self), (dict(self),))

    def __getattr__(self, name):
        # Column access stays available; no other attribute is reachable.
        if name.startswith("_"):
            raise AttributeError(name)
        for column, value in self._items:
            if column == name:
                return value
        raise AttributeError(name)


class _ReadOnlyResult:
    """A read result that cannot reach the connection that produced it."""

    __slots__ = ("_rows", "_keys")

    def __init__(self, rows, keys):
        self._rows = tuple(rows)
        self._keys = tuple(keys)

    def mappings(self):
        """The mapping view of the very same frozen rows."""
        return _ReadOnlyMappings(self._rows, self._keys)

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def one(self):
        if not self._rows:
            raise NoResultFound("No row was found when one was required")
        if len(self._rows) > 1:
            raise MultipleResultsFound("Multiple rows were found when exactly one was required")
        return self._rows[0]

    def one_or_none(self):
        if len(self._rows) > 1:
            raise MultipleResultsFound("Multiple rows were found when at most one was required")
        return self._rows[0] if self._rows else None

    def scalar(self):
        row = self.first()
        return None if row is None else next(iter(row.values()))

    def keys(self):
        return list(self._keys)

    def partitions(self, size):
        if size <= 0:
            raise ValueError("A partition size must be positive")
        for offset in range(0, len(self._rows), size):
            yield list(self._rows[offset:offset + size])

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


class _ReadOnlyMappings(_ReadOnlyResult):
    """The mapping view of a read result; it holds the same frozen rows."""

    __slots__ = ()


def _read_only_result(result):
    """Freeze a result into rows that cannot reach the connection."""
    mapping = result.mappings()
    return _ReadOnlyResult(tuple(_ReadOnlyRow(dict(row)) for row in mapping.all()),
                           tuple(mapping.keys()))


def _read_only_transaction(transaction):
    """Wrap a raw transaction so that no attribute leads back to it."""

    class _Guarded:
        """A transaction without a route back to the raw connection."""

        __slots__ = ()

        def __enter__(self):
            transaction.__enter__()
            return self

        def __exit__(self, *exc_info):
            return transaction.__exit__(*exc_info)

        def commit(self):
            transaction.commit()

        def rollback(self):
            transaction.rollback()

        def close(self):
            transaction.close()

    return _Guarded()


def _read_only_connection(connection):
    """Wrap a raw connection so that no attribute leads back to it."""

    class _Guarded:
        """Only controlled SELECT statements and read-only transactions."""

        __slots__ = ()

        def execute(self, statement, *args, **kwargs):
            return _read_only_result(connection.execute(_read_only_statement(statement),
                                                        *args, **kwargs))

        def begin(self):
            return _read_only_transaction(connection.begin())

        def close(self):
            connection.close()

        def __enter__(self):
            connection.__enter__()
            return self

        def __exit__(self, *exc_info):
            return connection.__exit__(*exc_info)

    return _Guarded()


def _read_only_engine(engine):
    """Wrap a raw engine so that no attribute leads back to it.

    The facade stores nothing at all: the guarded operations live in the
    closure of this subclass.  There is therefore no attribute - not
    ``_engine``, not ``pool``, not ``dialect`` - from which the raw engine, its
    pool or the raw driver module could be recovered.
    """

    class _Guarded:
        """connect/begin/dispose only; the raw engine stays unreachable."""

        __slots__ = ()

        def connect(self):
            return _read_only_connection(engine.connect())

        @contextmanager
        def begin(self):
            # Entering yields a guarded connection, never the raw transaction.
            with engine.begin() as connection:
                yield _read_only_connection(connection)

        def dispose(self):
            engine.dispose()

        def _run_ddl_visitor(self, *args, **kwargs):
            raise ReadOnlyViolation("schema creation is not available on the read-only reader")

    return _Guarded()


class _ReadOnlySettlementRepository(SettlementRepository):
    """The regular repository surface with every write path removed."""

    def ensure_schema(self):
        raise ReadOnlyViolation("schema creation is not available on the read-only reader")

    def register_inputs(self, *args, **kwargs):
        raise ReadOnlyViolation("input registration is not available on the read-only reader")

    def register_prices(self, *args, **kwargs):
        raise ReadOnlyViolation("price registration is not available on the read-only reader")

    def commit(self, *args, **kwargs):
        raise ReadOnlyViolation("commit is not available on the read-only reader")

    def _ensure_book(self, *args, **kwargs):
        raise ReadOnlyViolation("book initialisation is not available on the read-only reader")

    def _registration(self, *args, **kwargs):
        raise ReadOnlyViolation("registration is not available on the read-only reader")

    def _transaction(self, *args, **kwargs):
        raise ReadOnlyViolation("write transactions are not available on the read-only reader")

    def _lock_book(self, *args, **kwargs):
        raise ReadOnlyViolation("book locking is not available on the read-only reader")

    def _insert_many(self, *args, **kwargs):
        raise ReadOnlyViolation("row insertion is not available on the read-only reader")

    def _close_versions(self, *args, **kwargs):
        raise ReadOnlyViolation("version closing is not available on the read-only reader")


@contextmanager
def open_reconciliation_reader(config_path: Path, *, expected_meter_id: str):
    """Yield a repository that can only read the pinned book.

    The configuration is validated in memory only, its explicitly stated meter
    is compared with *expected_meter_id* before any connection is opened, and
    the database is reached through a separately opened read-only connection.
    A setup failure yields no reader at all.
    """
    if (not isinstance(expected_meter_id, str) or not expected_meter_id.strip()
            or len(expected_meter_id) > 128):
        raise ReconciliationConfigError("config_meter_invalid")
    path = Path(config_path)
    document = _config_document(path)
    if _raw_meter(document) != expected_meter_id:
        raise ReconciliationConfigError("config_meter_mismatch")
    model = _validated_config(document)
    settlement = getattr(getattr(model, "prices", None), "settlement", None)
    meter = getattr(settlement, "meter_id", None)
    if not isinstance(meter, str) or not meter.strip():
        raise ReconciliationConfigError("config_meter_missing")
    if meter != expected_meter_id:
        raise ReconciliationConfigError("config_meter_mismatch")
    url = _reader_url(model, path)
    engine = _reader_engine(url)
    try:
        yield _ReadOnlySettlementRepository(_read_only_engine(engine))
    finally:
        engine.dispose()
