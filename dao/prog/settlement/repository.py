"""Versieerbaar afrekenboek; uitsluitend geregistreerde invoer wordt gepubliceerd.

De korte schrijftransacties vergrendelen eerst de boekrij met een CAS-update.
Metingen ophalen en rekenen gebeuren daarbuiten. Het bronwatermerk is het
hoogste geregistreerde intervaleinde, geen garantie over nog niet opgehaalde
HA-/prijsrevisies. Een gedeeltelijke herberekening wist geen vuile staart.
"""

import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date
from decimal import Decimal, localcontext
from functools import cache
from hashlib import sha256
from time import time
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    bindparam,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

from dao.prog.settlement.tariffs import _canonical_default, resolve_snapshot
from dao.prog.settlement.types import (
    InputBatch,
    InputInterval,
    PeriodAdjustment,
    SettlementLine,
    SettlementResult,
    SettlementView,
    TariffSnapshot,
)

_BATCH_SIZE = 250
_METADATA = MetaData()
_TEXT = Text().with_variant(LONGTEXT(), "mysql", "mariadb")
_TABLE_OPTIONS = {
    "mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_bin",
    "mariadb_engine": "InnoDB", "mariadb_charset": "utf8mb4", "mariadb_collate": "utf8mb4_bin",
}

_TARIFF = Table(
    "settlement_tariff", _METADATA,
    Column("snapshot_hash", String(64), primary_key=True),
    Column("snapshot_json", _TEXT, nullable=False),
    **_TABLE_OPTIONS,
)
_INPUT = Table(
    "settlement_input", _METADATA,
    Column("meter_id", String(128), primary_key=True),
    Column("start_ts", BigInteger, primary_key=True),
    Column("input_revision", BigInteger, primary_key=True),
    Column("end_ts", BigInteger, nullable=False),
    Column("input_hash", String(64), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("input_json", _TEXT, nullable=False),
    CheckConstraint("input_revision > 0", name="ck_settlement_input_revision"),
    CheckConstraint("end_ts > start_ts", name="ck_settlement_input_bounds"),
    **_TABLE_OPTIONS,
)
_RUN = Table(
    "settlement_run", _METADATA,
    Column("run_id", String(36), primary_key=True),
    Column("meter_id", String(128), nullable=False),
    Column("generation", BigInteger, nullable=False),
    Column("input_revision", BigInteger, nullable=False),
    Column("engine_version", String(128), nullable=False),
    Column("input_hash", String(64), nullable=False),
    Column("result_hash", String(64), nullable=False),
    Column("coverage_start_ts", BigInteger, nullable=False),
    Column("coverage_end_ts", BigInteger, nullable=False),
    Column("reason", _TEXT, nullable=False),
    Column("status", String(16), nullable=False),
    Column("snapshot_hashes_json", _TEXT, nullable=False),
    Column("captured_ts", BigInteger, nullable=False),
    Column("source_watermark_ts", BigInteger),
    Column("complete", Boolean, nullable=False),
    Column("known_subtotal", _TEXT, nullable=False),
    Column("net_cost", _TEXT),
    UniqueConstraint("meter_id", "generation", name="uq_settlement_run_generation"),
    CheckConstraint("generation > 0 AND input_revision > 0", name="ck_settlement_run_revisions"),
    **_TABLE_OPTIONS,
)
_INTERVAL = Table(
    "settlement_interval", _METADATA,
    Column("meter_id", String(128), primary_key=True),
    Column("start_ts", BigInteger, primary_key=True),
    Column("valid_from_generation", BigInteger, primary_key=True),
    Column("end_ts", BigInteger, nullable=False),
    Column("valid_to_generation", BigInteger),
    Column("input_revision", BigInteger, nullable=False),
    Column("input_hash", String(64), nullable=False),
    Column("snapshot_hash", String(64), ForeignKey("settlement_tariff.snapshot_hash")),
    Column("run_id", String(36), ForeignKey("settlement_run.run_id"), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("import_cost", _TEXT),
    Column("base_export_revenue", _TEXT),
    Column("bonus_revenue", _TEXT),
    Column("bonus_kwh", _TEXT),
    Column("line_json", _TEXT, nullable=False),
    ForeignKeyConstraint(
        ["meter_id", "start_ts", "input_revision"],
        ["settlement_input.meter_id", "settlement_input.start_ts", "settlement_input.input_revision"],
        name="fk_settlement_interval_input",
    ),
    CheckConstraint("input_revision > 0 AND valid_from_generation > 0", name="ck_settlement_interval_revisions"),
    CheckConstraint("end_ts > start_ts", name="ck_settlement_interval_bounds"),
    CheckConstraint("valid_to_generation IS NULL OR valid_to_generation > valid_from_generation",
                    name="ck_settlement_interval_version"),
    Index("ix_settlement_interval_generation", "meter_id", "valid_from_generation", "valid_to_generation"),
    **_TABLE_OPTIONS,
)
_ADJUSTMENT = Table(
    "settlement_adjustment", _METADATA,
    Column("meter_id", String(128), primary_key=True),
    Column("kind", String(64), primary_key=True),
    Column("period_start_ts", BigInteger, primary_key=True),
    Column("period_end_ts", BigInteger, primary_key=True),
    Column("valid_from_generation", BigInteger, primary_key=True),
    Column("valid_to_generation", BigInteger),
    Column("run_id", String(36), ForeignKey("settlement_run.run_id"), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("tombstone", Boolean, nullable=False),
    Column("credit_ex_vat", _TEXT),
    Column("credit_vat", _TEXT),
    Column("credit_total", _TEXT),
    Column("basis_ex_vat", _TEXT),
    Column("adjustment_json", _TEXT),
    CheckConstraint("valid_from_generation > 0", name="ck_settlement_adjustment_generation"),
    CheckConstraint("period_end_ts > period_start_ts", name="ck_settlement_adjustment_bounds"),
    CheckConstraint("valid_to_generation IS NULL OR valid_to_generation > valid_from_generation",
                    name="ck_settlement_adjustment_version"),
    Index("ix_settlement_adjustment_generation", "meter_id", "valid_from_generation", "valid_to_generation"),
    **_TABLE_OPTIONS,
)
_BOOK = Table(
    "settlement_book", _METADATA,
    Column("meter_id", String(128), primary_key=True),
    Column("generation", BigInteger, nullable=False, server_default="0"),
    Column("input_revision", BigInteger, nullable=False, server_default="0"),
    Column("snapshot_hashes_json", _TEXT, nullable=False),
    Column("source_watermark_ts", BigInteger),
    Column("dirty_from_ts", BigInteger),
    CheckConstraint("generation >= 0 AND input_revision >= 0", name="ck_settlement_book_revisions"),
    **_TABLE_OPTIONS,
)


class ConcurrentSettlementError(RuntimeError):
    """De geregistreerde invoer of financiële generatie is inmiddels gewijzigd."""


def _json(value) -> str:
    return json.dumps(asdict(value) if is_dataclass(value) else value,
                      default=_canonical_default, sort_keys=True, separators=(",", ":"))


def _hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _decimal(value: Decimal | None) -> str | None:
    return None if value is None else _canonical_default(value)


@cache
def _annotations(cls):
    return get_type_hints(cls)


def _restore(value, annotation):
    """Herstel uitsluitend de bekende DTO-typen; geen pickle of dynamische import."""
    if value is None:
        return None
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        return _restore(value, next(kind for kind in args if kind is not type(None)))
    if annotation is Decimal:
        return Decimal(value)
    if annotation is date:
        return date.fromisoformat(value)
    if origin is tuple:
        kinds = (args[0],) * len(value) if args[-1] is Ellipsis else args
        return tuple(_restore(item, kind) for item, kind in zip(value, kinds, strict=True))
    if is_dataclass(annotation):
        return annotation(**{name: _restore(item, _annotations(annotation)[name]) for name, item in value.items()})
    return value


def _load(cls, content: str):
    return _restore(json.loads(content), cls)


def _meter(meter_id: str):
    if not isinstance(meter_id, str) or not meter_id.strip() or len(meter_id) > 128:
        raise ValueError("meter_id must identify one explicit meter (1..128 characters)")


def _revision(value: int):
    if type(value) is not int or not 0 <= value < 2**63 - 1:
        raise ValueError("Expected a nonnegative signed-64-bit revision/generation")


def _bounds(start: int, end: int):
    if type(start) is not int or type(end) is not int or not -(2**63) <= start < end < 2**63:
        raise ValueError("Expected half-open signed-64-bit UTC epoch bounds")


def _input_content(item: InputInterval) -> str:
    content = asdict(item)
    del content["input_revision"]
    return _json(content)


def _valid(table, generation):
    return and_(table.c.valid_from_generation <= generation,
                or_(table.c.valid_to_generation.is_(None), generation < table.c.valid_to_generation))


def _adjustment_key(item):
    return item["kind"], item["period_start_ts"], item["period_end_ts"]


def _insert_if_absent(table, dialect, key):
    if dialect in {"mysql", "mariadb"}:
        from sqlalchemy.dialects.mysql import insert as dialect_insert

        statement = dialect_insert(table)
        return statement.on_duplicate_key_update(**{key: statement.inserted[key]})
    if dialect in {"sqlite", "postgresql"}:
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        else:
            from sqlalchemy.dialects.postgresql import insert as dialect_insert

        return dialect_insert(table).on_conflict_do_nothing(index_elements=[key])
    raise NotImplementedError(f"Unsupported settlement database dialect: {dialect}")


def _totals(lines, adjustments, start, end):
    with localcontext() as context:
        context.prec = 36
        subtotal, cursor, complete = Decimal("0"), start, bool(lines)
        closed = {(item.period_start_ts, item.period_end_ts) for item in adjustments if item.kind == "spot_zero"}
        for line in lines:
            complete &= line.start_ts == cursor and line.end_ts <= end and line.quality != "incomplete"
            cursor = line.end_ts
            for amount, sign in ((line.import_cost, 1), (line.base_export_revenue, -1), (line.bonus_revenue, -1)):
                if amount is None:
                    complete = False
                else:
                    subtotal += sign * amount
            month = line.after_state.month
            if month is not None and line.end_ts == month.period_end_ts:
                complete &= (month.period_start_ts, month.period_end_ts) in closed
        subtotal -= sum((item.credit_total for item in adjustments), Decimal("0"))
        complete &= cursor == end
        return subtotal, bool(complete)


class SettlementRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def ensure_schema(self) -> None:
        """Maak uitsluitend de zes additieve tabellen, onafhankelijk van legacyversies."""
        _METADATA.create_all(self.engine, checkfirst=True)

    @contextmanager
    def _transaction(self):
        try:
            with self.engine.begin() as connection:
                yield connection
        except DBAPIError as exc:
            code = getattr(exc.orig, "sqlstate", None) or getattr(exc.orig, "pgcode", None)
            sqlite_code = getattr(exc.orig, "sqlite_errorcode", 0) & 0xFF
            mysql_code = exc.orig.args[0] if exc.orig.args else None
            if code in {"40001", "40P01"} or sqlite_code in {5, 6} or mysql_code in {1205, 1213}:
                raise ConcurrentSettlementError("Concurrent database writer; reread and recalculate") from exc
            raise

    def _ensure_book(self, meter_id: str):
        # Losse, korte initialisatie vóór de CAS-transactie; ook twee eerste schrijvers zijn veilig.
        values = {"meter_id": meter_id, "generation": 0, "input_revision": 0,
                  "snapshot_hashes_json": "[]"}
        with self._transaction() as connection:
            connection.execute(_insert_if_absent(_BOOK, self.engine.dialect.name, "meter_id"), values)

    @staticmethod
    def _lock_book(connection, meter_id, revision, generation=None):
        predicate = and_(_BOOK.c.meter_id == meter_id, _BOOK.c.input_revision == revision)
        if generation is not None:
            predicate = and_(predicate, _BOOK.c.generation == generation)
        matched = connection.execute(update(_BOOK).where(predicate).values(input_revision=revision)).rowcount
        if matched != 1:
            raise ConcurrentSettlementError("Settlement generation or registered input revision changed")
        return connection.execute(select(_BOOK).where(_BOOK.c.meter_id == meter_id)).mappings().one()

    @staticmethod
    def _latest_inputs(connection, meter_id, revision, start=None, end=None, *, previous=False):
        predicate = and_(_INPUT.c.meter_id == meter_id, _INPUT.c.input_revision <= revision)
        if start is not None:
            predicate = and_(predicate, _INPUT.c.start_ts >= start)
        if end is not None:
            predicate = and_(predicate, _INPUT.c.start_ts < end)
        latest = select(_INPUT.c.start_ts, func.max(_INPUT.c.input_revision).label("revision")).where(
            predicate).group_by(_INPUT.c.start_ts).subquery()
        query = select(_INPUT).join(latest, and_(
            _INPUT.c.start_ts == latest.c.start_ts, _INPUT.c.input_revision == latest.c.revision,
        )).where(_INPUT.c.meter_id == meter_id)
        query = query.order_by(_INPUT.c.start_ts.desc() if previous else _INPUT.c.start_ts)
        if previous:
            query = query.limit(1)
        return connection.execute(query).mappings().all()

    @staticmethod
    def _snapshots(connection, hashes_json):
        hashes = json.loads(hashes_json)
        if not hashes:
            return ()
        by_hash = {row.snapshot_hash: _load(TariffSnapshot, row.snapshot_json) for row in connection.execute(
            select(_TARIFF).where(_TARIFF.c.snapshot_hash.in_(hashes)))}
        return tuple(by_hash[value] for value in hashes)

    @staticmethod
    def _insert_many(connection, table, records, *, immutable_key=None):
        statement = (insert(table) if immutable_key is None else
                     _insert_if_absent(table, connection.dialect.name, immutable_key))
        for offset in range(0, len(records), _BATCH_SIZE):
            connection.execute(statement, records[offset:offset + _BATCH_SIZE])

    def register_inputs(
        self, meter_id: str, inputs: Sequence[InputInterval], snapshots: Sequence[TariffSnapshot],
        expected_input_revision: int,
    ) -> int:
        _meter(meter_id)
        _revision(expected_input_revision)
        prepared, end = [], None
        for item in inputs:
            _bounds(item.start_ts, item.end_ts)
            if item.meter_id != meter_id or (end is not None and item.start_ts < end):
                raise ValueError("Registered input must belong to one meter and be ordered without overlap")
            content = _input_content(item)
            prepared.append({"meter_id": meter_id, "start_ts": item.start_ts, "end_ts": item.end_ts,
                             "input_revision": expected_input_revision + 1, "input_hash": item.input_hash,
                             "content_hash": _hash(content), "input_json": content})
            end = item.end_ts
        ordered = sorted(snapshots, key=lambda item: item.valid_from_ts)
        tariffs = []
        for index, snapshot in enumerate(ordered):
            if snapshot.status != "confirmed":
                raise ValueError("Registration requires explicitly confirmed contract snapshots")
            content = asdict(snapshot)
            del content["snapshot_hash"]
            if _hash(_json(content)) != snapshot.snapshot_hash:
                raise ValueError("Tariff snapshot hash does not match its canonical financial content")
            if index and (ordered[index - 1].valid_to_ts is None
                          or ordered[index - 1].valid_to_ts > snapshot.valid_from_ts):
                raise ValueError("Contract snapshots must not overlap")
            tariffs.append({"snapshot_hash": snapshot.snapshot_hash, "snapshot_json": _json(snapshot)})
        hashes_json = _json([item.snapshot_hash for item in ordered])
        self._ensure_book(meter_id)
        with self._transaction() as connection:
            book = self._lock_book(connection, meter_id, expected_input_revision)
            current = []
            if prepared:
                current = self._latest_inputs(connection, meter_id, expected_input_revision,
                                              prepared[0]["start_ts"], prepared[-1]["end_ts"])
                current += self._latest_inputs(connection, meter_id, expected_input_revision,
                                               end=prepared[0]["start_ts"], previous=True)
            by_start = {row["start_ts"]: row for row in current}
            changed = [row for row in prepared if row["start_ts"] not in by_start
                       or by_start[row["start_ts"]]["content_hash"] != row["content_hash"]]
            combined = {**by_start, **{row["start_ts"]: row for row in prepared}}
            adjacent = sorted(combined.values(), key=lambda row: row["start_ts"])
            if any(left["end_ts"] > right["start_ts"] for left, right in zip(adjacent, adjacent[1:], strict=False)):
                raise ValueError("New input overlaps previously registered intervals")
            contract_changed = hashes_json != book["snapshot_hashes_json"]
            if not changed and not contract_changed:
                return expected_input_revision
            dirty = [row["start_ts"] for row in changed]
            if book["dirty_from_ts"] is not None:
                dirty.append(book["dirty_from_ts"])
            if contract_changed:
                old = self._snapshots(connection, book["snapshot_hashes_json"])
                boundaries = sorted({point for item in (*old, *ordered)
                                     for point in (item.valid_from_ts, item.valid_to_ts) if point is not None})

                def selected(items, at):
                    return next((item.snapshot_hash for item in items if item.valid_from_ts <= at
                                 and (item.valid_to_ts is None or at < item.valid_to_ts)), None)

                dirty.extend(at for at in boundaries if selected(old, at) != selected(ordered, at))
            existing = set(connection.scalars(select(_TARIFF.c.snapshot_hash).where(
                _TARIFF.c.snapshot_hash.in_([item["snapshot_hash"] for item in tariffs]))))
            # Andere meters hebben een eigen boeklock, maar delen dezelfde immutable tarief-PK.
            self._insert_many(connection, _TARIFF, [item for item in tariffs if item["snapshot_hash"] not in existing],
                              immutable_key="snapshot_hash")
            self._insert_many(connection, _INPUT, changed)
            watermarks = [value for value in (book["source_watermark_ts"], end) if value is not None]
            connection.execute(update(_BOOK).where(_BOOK.c.meter_id == meter_id).values(
                input_revision=expected_input_revision + 1, snapshot_hashes_json=hashes_json,
                source_watermark_ts=max(watermarks, default=None), dirty_from_ts=min(dirty, default=None),
            ))
            return expected_input_revision + 1

    def read_inputs(self, meter_id: str, start_ts: int, end_ts: int) -> InputBatch:
        _meter(meter_id)
        _bounds(start_ts, end_ts)
        with self.engine.connect() as connection:
            book = connection.execute(select(_BOOK).where(_BOOK.c.meter_id == meter_id)).mappings().first()
            if book is None:
                return InputBatch(0, (), (), int(time()), None)
            # Eenmaal gepinde revisie; latere registraties veranderen deze immutable rijen niet.
            rows = self._latest_inputs(connection, meter_id, book["input_revision"], start_ts, end_ts)
            items = tuple(_restore({**json.loads(row["input_json"]), "input_revision": row["input_revision"]},
                                   InputInterval) for row in rows)
            return InputBatch(book["input_revision"], items,
                              self._snapshots(connection, book["snapshot_hashes_json"]), int(time()),
                              book["source_watermark_ts"])

    @staticmethod
    def _prepare_result(meter_id, result, batch):
        if batch.input_revision <= 0 or not batch.items:
            raise ValueError("Publication requires positive registered input revisions")
        if not result.engine_version.strip() or len(result.engine_version) > 128:
            raise ValueError("Expected an explicit engine version (1..128 characters)")
        if len(result.lines) != len(batch.items):
            raise ValueError("Every registered batch item requires exactly one result line")
        lines, previous = [], None
        for item, line in zip(batch.items, result.lines, strict=True):
            if (item.meter_id != meter_id or item.input_revision <= 0 or item.input_revision > batch.input_revision
                or (line.meter_id, line.start_ts, line.end_ts, line.input_revision, line.input_hash)
                    != (meter_id, item.start_ts, item.end_ts, item.input_revision, item.input_hash)):
                raise ValueError("Result line does not reference its positive registered input revision")
            try:
                selected = resolve_snapshot(batch.snapshots, item.start_ts, item.end_ts).snapshot_hash
            except ValueError:
                selected = None
            if line.snapshot_hash != selected:
                raise ValueError("Result must use the pinned registered contract selection")
            if (line.after_state.asof_ts != line.end_ts or line.before_state.asof_ts > line.start_ts
                or (previous is not None and (line.start_ts < previous.end_ts or line.before_state != previous.after_state))):
                raise ValueError("Result states must form one chronological series")
            content = _json(line)
            lines.append({"meter_id": meter_id, "start_ts": line.start_ts, "end_ts": line.end_ts,
                          "input_revision": line.input_revision, "input_hash": line.input_hash,
                          "snapshot_hash": line.snapshot_hash, "content_hash": _hash(content), "line_json": content,
                          **{name: _decimal(getattr(line, name)) for name in
                             ("import_cost", "base_export_revenue", "bonus_revenue", "bonus_kwh")}})
            previous = line
        if (result.coverage_start_ts != result.lines[0].before_state.asof_ts
            or result.coverage_end_ts != result.lines[-1].end_ts or result.end_state != result.lines[-1].after_state):
            raise ValueError("Result coverage/end state does not match its lines")
        adjustments = []
        for item in result.adjustments:
            if (not item.kind.strip() or len(item.kind) > 64
                or not result.coverage_start_ts < item.period_end_ts <= result.coverage_end_ts):
                raise ValueError("Adjustment must close inside the result coverage")
            content = _json(item)
            adjustments.append({"meter_id": meter_id, "kind": item.kind, "period_start_ts": item.period_start_ts,
                                "period_end_ts": item.period_end_ts, "tombstone": False,
                                "content_hash": _hash(content), "adjustment_json": content,
                                **{name: _decimal(getattr(item, name)) for name in
                                   ("credit_ex_vat", "credit_vat", "credit_total", "basis_ex_vat")}})
        if len({_adjustment_key(row) for row in adjustments}) != len(adjustments):
            raise ValueError("Duplicate period adjustments")
        subtotal, complete = _totals(result.lines, result.adjustments, result.coverage_start_ts, result.coverage_end_ts)
        if (subtotal != result.known_subtotal or (result.complete and not complete)
            or result.net_cost != (subtotal if result.complete else None)):
            raise ValueError("Result totals/completeness do not match its unrounded components")
        return lines, adjustments

    @staticmethod
    def _close_versions(connection, table, records, generation, keys):
        statement = update(table).where(and_(
            *(table.c[key] == bindparam(f"old_{key}") for key in ("meter_id", *keys, "valid_from_generation")),
            table.c.valid_to_generation.is_(None),
        )).values(valid_to_generation=generation)
        parameters = [{f"old_{key}": row[key] for key in ("meter_id", *keys, "valid_from_generation")}
                      for row in records]
        for offset in range(0, len(parameters), _BATCH_SIZE):
            connection.execute(statement, parameters[offset:offset + _BATCH_SIZE])

    def commit(
        self, meter_id: str, result: SettlementResult, batch: InputBatch, expected_generation: int, reason: str,
    ) -> int:
        _meter(meter_id)
        _revision(expected_generation)
        _revision(batch.input_revision)
        _bounds(result.coverage_start_ts, result.coverage_end_ts)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Publication requires an explicit reason")
        lines, adjustments = self._prepare_result(meter_id, result, batch)
        input_contents = [(item.start_ts, item.input_revision, _hash(_input_content(item))) for item in batch.items]
        hashes_json = _json([item.snapshot_hash for item in sorted(batch.snapshots, key=lambda item: item.valid_from_ts)])
        input_hash = _hash(_json({"revision": batch.input_revision, "items": input_contents, "snapshots": hashes_json}))
        result_hash = _hash(_json({
            "lines": [row["content_hash"] for row in lines], "adjustments": [row["content_hash"] for row in adjustments],
            "engine_version": result.engine_version, "start": result.coverage_start_ts, "end": result.coverage_end_ts,
            "complete": result.complete, "known_subtotal": result.known_subtotal, "net_cost": result.net_cost,
        }))
        self._ensure_book(meter_id)
        with self._transaction() as connection:
            # Eerste statement is een UPDATE: ook SQLite krijgt vóór lezen een echte schrijflock.
            book = self._lock_book(connection, meter_id, batch.input_revision, expected_generation)
            current_inputs = self._latest_inputs(connection, meter_id, batch.input_revision,
                                                 result.coverage_start_ts, result.coverage_end_ts)
            references = [(row["start_ts"], row["input_revision"], row["content_hash"]) for row in current_inputs]
            if (references != input_contents or hashes_json != book["snapshot_hashes_json"]
                or self._snapshots(connection, hashes_json) != tuple(sorted(batch.snapshots, key=lambda item: item.valid_from_ts))
                or batch.source_watermark_ts != book["source_watermark_ts"]):
                raise ValueError("Batch does not match registered input/contract references")
            old_lines = connection.execute(select(_INTERVAL).where(
                _INTERVAL.c.meter_id == meter_id, _INTERVAL.c.start_ts >= result.coverage_start_ts,
                _INTERVAL.c.start_ts < result.coverage_end_ts, _valid(_INTERVAL, expected_generation),
            )).mappings().all()
            old_adjustments = connection.execute(select(_ADJUSTMENT).where(
                _ADJUSTMENT.c.meter_id == meter_id, _ADJUSTMENT.c.period_end_ts > result.coverage_start_ts,
                _ADJUSTMENT.c.period_end_ts <= result.coverage_end_ts, _valid(_ADJUSTMENT, expected_generation),
            )).mappings().all()
            by_start = {row["start_ts"]: row for row in old_lines}
            by_period = {_adjustment_key(row): row for row in old_adjustments}
            changed_lines = [row for row in lines if row["start_ts"] not in by_start
                             or by_start[row["start_ts"]]["content_hash"] != row["content_hash"]]
            periods = {_adjustment_key(row) for row in adjustments}
            for key, old in by_period.items():
                if key not in periods and not old["tombstone"]:
                    adjustments.append({"meter_id": meter_id, "kind": key[0], "period_start_ts": key[1],
                                        "period_end_ts": key[2], "tombstone": True, "content_hash": _hash(_json(key)),
                                        "adjustment_json": None, "credit_ex_vat": None, "credit_vat": None,
                                        "credit_total": None, "basis_ex_vat": None})
            changed_adjustments = [row for row in adjustments if _adjustment_key(row) not in by_period
                                   or by_period[_adjustment_key(row)]["content_hash"] != row["content_hash"]]
            dirty = book["dirty_from_ts"]
            watermark = book["source_watermark_ts"]
            if (dirty is not None and result.coverage_start_ts <= dirty
                and (watermark is None or result.coverage_end_ts >= watermark)):
                dirty = None
            if not changed_lines and not changed_adjustments:
                # Onder de dual-CAS legt dirty=None ook een volledige hervalidatie
                # tegen book.input_revision vast; de oorspronkelijke run blijft immutable.
                connection.execute(update(_BOOK).where(_BOOK.c.meter_id == meter_id).values(dirty_from_ts=dirty))
                return expected_generation
            generation, run_id = expected_generation + 1, str(uuid4())
            connection.execute(insert(_RUN).values(
                run_id=run_id, meter_id=meter_id, generation=generation, input_revision=batch.input_revision,
                engine_version=result.engine_version, input_hash=input_hash, result_hash=result_hash,
                coverage_start_ts=result.coverage_start_ts, coverage_end_ts=result.coverage_end_ts,
                reason=reason, status="published", snapshot_hashes_json=hashes_json, captured_ts=batch.captured_ts,
                source_watermark_ts=batch.source_watermark_ts, complete=result.complete,
                known_subtotal=_decimal(result.known_subtotal), net_cost=_decimal(result.net_cost),
            ))
            self._close_versions(connection, _INTERVAL,
                                 [by_start[row["start_ts"]] for row in changed_lines if row["start_ts"] in by_start],
                                 generation, ("start_ts",))
            self._close_versions(connection, _ADJUSTMENT,
                                 [by_period[_adjustment_key(row)] for row in changed_adjustments
                                  if _adjustment_key(row) in by_period], generation,
                                 ("kind", "period_start_ts", "period_end_ts"))
            for records, table in ((changed_lines, _INTERVAL), (changed_adjustments, _ADJUSTMENT)):
                self._insert_many(connection, table, [{**row, "valid_from_generation": generation,
                                                       "valid_to_generation": None, "run_id": run_id} for row in records])
            # De pointer is de laatste schrijfoperatie en deelt de transactie met alle bovenstaande rijen.
            published = connection.execute(update(_BOOK).where(
                _BOOK.c.meter_id == meter_id, _BOOK.c.generation == expected_generation,
                _BOOK.c.input_revision == batch.input_revision,
            ).values(generation=generation, dirty_from_ts=dirty)).rowcount
            if published != 1:
                raise ConcurrentSettlementError("Settlement pointer publication lost its dual CAS")
            return generation

    def read(self, meter_id: str, start_ts: int, end_ts: int, generation: int | None = None) -> SettlementView:
        _meter(meter_id)
        _bounds(start_ts, end_ts)
        if generation is not None:
            _revision(generation)
        with self.engine.connect() as connection:
            book = connection.execute(select(_BOOK).where(_BOOK.c.meter_id == meter_id)).mappings().first()
            current_generation, revision = (0, 0) if book is None else (book["generation"], book["input_revision"])
            pinned = current_generation if generation is None else generation
            if pinned > current_generation:
                raise ValueError("Requested financial generation has not been published")
            if pinned == 0:
                return SettlementView(meter_id, 0, 0, revision, "unknown", (), (), Decimal("0"), None,
                                      start_ts, end_ts, False, int(time()), None,
                                      None if book is None else book["dirty_from_ts"])
            run = connection.execute(select(_RUN).where(
                _RUN.c.meter_id == meter_id, _RUN.c.generation == pinned,
            )).mappings().one()
            line_rows = connection.execute(select(_INTERVAL.c.line_json).where(
                _INTERVAL.c.meter_id == meter_id, _INTERVAL.c.start_ts >= start_ts,
                _INTERVAL.c.start_ts < end_ts, _valid(_INTERVAL, pinned),
            ).order_by(_INTERVAL.c.start_ts))
            lines = tuple(_load(SettlementLine, row.line_json)
                          for partition in line_rows.partitions(_BATCH_SIZE) for row in partition)
            adjustment_rows = connection.execute(select(_ADJUSTMENT.c.adjustment_json).where(
                _ADJUSTMENT.c.meter_id == meter_id, _ADJUSTMENT.c.period_end_ts > start_ts,
                _ADJUSTMENT.c.period_end_ts <= end_ts, _ADJUSTMENT.c.tombstone.is_(False), _valid(_ADJUSTMENT, pinned),
            ).order_by(_ADJUSTMENT.c.period_end_ts, _ADJUSTMENT.c.kind, _ADJUSTMENT.c.period_start_ts))
            adjustments = tuple(_load(PeriodAdjustment, row.adjustment_json)
                                for partition in adjustment_rows.partitions(_BATCH_SIZE) for row in partition)
            subtotal, complete = _totals(lines, adjustments, start_ts, end_ts)
            # Een schone actuele boekpointer omvat ook financiële no-op-hervalidatie.
            # Historische/vuile generaties behouden de oorspronkelijke auditrevisie.
            fresh = pinned == current_generation and book["dirty_from_ts"] is None
            validated_revision = revision if fresh else run["input_revision"]
            return SettlementView(
                meter_id, pinned, validated_revision, revision, "current" if fresh else "outdated", lines, adjustments,
                subtotal, subtotal if complete else None, start_ts, end_ts, complete, run["captured_ts"],
                run["source_watermark_ts"], book["dirty_from_ts"],
            )
