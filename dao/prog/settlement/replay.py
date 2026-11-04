"""Immutable, content-addressed planning evidence. Never an execution capability.

No database, solver or tariff reconstruction: retain the producer's exact DTOs.
Archive files live beside the plan, named by SHA-256 of canonical UTF-8 bytes.
The trusted local directory and cooperating writers have the same scope as the
publication protocol; this is not protection against arbitrary local writers.
"""

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

from dao.prog.fastctrl.plan import FinancialBasis, _decimal_text, _object, _read_decimal
from dao.prog.fastctrl.publication import PublicationError, canonical_plan_path
from dao.prog.settlement.types import PriceBatch, PriceQuote, SettlementState, TerminalValue


def _canonical_default(value):
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if type(value) is date:
        return value.isoformat()
    raise TypeError("Unsupported replay value")


def _canonical_bytes(data):
    return json.dumps(data, default=_canonical_default, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _decode(value, annotation):
    """Read only the declared frozen DTO tree, with no coercion or defaults."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        for kind in args:
            try:
                return _decode(value, kind)
            except (TypeError, ValueError):
                continue
        raise ValueError("Invalid replay optional value")
    if origin is Literal:
        if not any(type(value) is type(option) and value == option for option in args):
            raise ValueError("Unknown replay literal")
        return value
    if origin is tuple:
        if type(value) is not list:
            raise ValueError("Expected replay array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(item, args[0]) for item in value)
        if len(value) != len(args):
            raise ValueError("Invalid replay tuple length")
        return tuple(_decode(item, kind) for item, kind in zip(value, args, strict=True))
    if annotation is Decimal:
        return _read_decimal(value)
    if annotation is date:
        if type(value) is not str or date.fromisoformat(value).isoformat() != value:
            raise ValueError("Expected canonical replay date")
        return date.fromisoformat(value)
    if is_dataclass(annotation):
        _object(value, tuple(field.name for field in fields(annotation)))
        hints = get_type_hints(annotation)
        return annotation(**{name: _decode(item, hints[name]) for name, item in value.items()})
    if type(value) is not annotation:
        raise ValueError("Unexpected replay value type")
    return value


@dataclass(frozen=True, slots=True)
class ReplayBasis:
    basis: FinancialBasis
    opening: SettlementState
    quotes: tuple[PriceQuote, ...]
    price_batch: PriceBatch
    terminal_value: TerminalValue

    def __post_init__(self):
        for name, kind in (("basis", FinancialBasis), ("opening", SettlementState),
                           ("price_batch", PriceBatch), ("terminal_value", TerminalValue)):
            if type(getattr(self, name)) is not kind:
                raise TypeError(f"replay_{name}_type_invalid")
        if type(self.quotes) not in (list, tuple) or any(type(q) is not PriceQuote for q in self.quotes):
            raise TypeError("replay_quotes_type_invalid")
        object.__setattr__(self, "quotes", tuple(self.quotes))
        basis, batch = self.basis, self.price_batch
        if (batch.meter_id != basis.meter_id or batch.input_revision != basis.input_revision
                or tuple(s.snapshot_hash for s in batch.snapshots) != basis.snapshot_hashes
                or batch.captured_ts != basis.captured_ts or batch.source_watermark_ts != basis.source_watermark_ts
                or batch.start_ts > basis.planning_start_ts or batch.end_ts < basis.horizon_end_ts):
            raise ValueError("replay_basis_mismatch")
        if self.opening.asof_ts != basis.planning_start_ts:
            raise ValueError("replay_opening_mismatch")
        for snapshot in self.snapshots:
            content = asdict(snapshot)
            del content["snapshot_hash"]
            if sha256(_canonical_bytes(content)).hexdigest() != snapshot.snapshot_hash:
                raise ValueError("replay_snapshot_hash_mismatch")
        cursor = basis.planning_start_ts
        for quote in self.quotes:
            if (quote.start_ts != cursor or quote.snapshot not in self.snapshots
                    or quote.start_ts < quote.snapshot.valid_from_ts
                    or (quote.snapshot.valid_to_ts is not None and quote.end_ts > quote.snapshot.valid_to_ts)):
                raise ValueError("replay_quote_selection_mismatch")
            cursor = quote.end_ts
        if not self.quotes or cursor != basis.horizon_end_ts:
            raise ValueError("replay_quote_coverage_mismatch")
        names = [name for name, _ in self.terminal_value.reference_efficiency_by_battery]
        if any(not name.strip() for name in names) or len(names) != len(set(names)):
            raise ValueError("replay_terminal_identity_invalid")

    @property
    def snapshots(self):
        return self.price_batch.snapshots

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self))

    @property
    def content_hash(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()

    def check_context(self, context) -> None:
        if self.basis != context.basis:
            raise ValueError("replay_basis_mismatch")
        if self.terminal_value != context.terminal_value:
            raise ValueError("replay_terminal_mismatch")
        if self.content_hash != context.replay_basis_hash:
            raise ValueError("replay_hash_mismatch")


@dataclass(frozen=True, slots=True)
class ReplayRead:
    status: Literal["known", "unknown"]
    basis: ReplayBasis | None
    reason: str


def replay_archive_path(plan_path: str | Path, replay_basis_hash: str) -> Path:
    if type(replay_basis_hash) is not str or not re.fullmatch("[0-9a-f]{64}", replay_basis_hash):
        raise ValueError("replay_hash_invalid")
    return canonical_plan_path(plan_path).parent / replay_basis_hash


def read_replay_basis(plan_path: str | Path, replay_basis_hash: str, *,
                      financial_basis: FinancialBasis | None = None,
                      terminal_value: TerminalValue | None = None) -> ReplayRead:
    """Absence/corruption/mismatch is explicit unknown, never reconstructed money."""
    try:
        path = replay_archive_path(plan_path, replay_basis_hash)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read()
        if sha256(raw).hexdigest() != replay_basis_hash:
            raise ValueError("replay_hash_mismatch")
        replay = _decode(json.loads(raw), ReplayBasis)
        if raw != replay.canonical_bytes():
            raise ValueError("replay_bytes_noncanonical")
        if financial_basis is not None and replay.basis != financial_basis:
            raise ValueError("replay_basis_mismatch")
        if terminal_value is not None and replay.terminal_value != terminal_value:
            raise ValueError("replay_terminal_mismatch")
        return ReplayRead("known", replay, "replay_basis_verified")
    except FileNotFoundError:
        return ReplayRead("unknown", None, "replay_archive_missing")
    except (PublicationError, OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
        return ReplayRead("unknown", None, f"replay_archive_unknown: {exc}")


def _fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_durable_replay_basis(plan_path, context) -> ReplayBasis:
    """Re-prove durability even after an earlier replace/directory-fsync failure."""
    read = read_replay_basis(plan_path, context.replay_basis_hash,
                             financial_basis=context.basis, terminal_value=context.terminal_value)
    if read.status != "known":
        raise ValueError(read.reason)
    path = replay_archive_path(plan_path, context.replay_basis_hash)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)
    return read.basis


def write_replay_basis(replay: ReplayBasis, *, plan_path: str | Path) -> str:
    """Durable before ready; no capability and no repair of corrupt old evidence."""
    if type(replay) is not ReplayBasis:
        raise TypeError("replay_basis_required")
    raw, digest = replay.canonical_bytes(), replay.content_hash
    path = replay_archive_path(plan_path, digest)
    temporary = None
    try:
        if path.exists() or path.is_symlink():
            existing = read_replay_basis(plan_path, digest)
            if existing.status != "known" or existing.basis != replay:
                raise ValueError("replay_existing_archive_unverifiable")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        else:
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".replay_basis_",
                                             suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        _fsync_directory(path.parent)
        verified = read_replay_basis(plan_path, digest)
        if verified.status != "known" or verified.basis != replay:
            raise ValueError("replay_archive_unverifiable")
        return digest
    except OSError as exc:
        raise ValueError(f"replay_archive_write_failed: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
