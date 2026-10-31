"""Korte lokale publicatiegrens; geen database, solver of actuator-I/O.

Eén host, lokaal filesystem en meewerkende schrijvers zijn voorwaarden. De
DAO-datadirectory moet vertrouwd zijn: dit is geen beveiliging tegen iemand die
die directory kan vervangen. Een blocked-artefact vraagt fallback aan, maar
bewijst geen fysieke stop of eigendomsoverdracht.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

from dao.prog.fastctrl.plan import BlockedPlan, FastPlan, _integer

if TYPE_CHECKING:
    from dao.prog.config.models.pricing import PricingConfig


@dataclass(frozen=True, slots=True)
class GatePolicy:
    required_from_ts: int | None

    def __post_init__(self):
        if self.required_from_ts is not None:
            _integer(self.required_from_ts, "required_from_ts")


@dataclass(frozen=True, slots=True)
class PlanRead:
    status: Literal["ready", "blocked", "missing", "invalid"]
    plan: FastPlan | None
    reason: str


@dataclass(frozen=True, slots=True)
class GateVerdict:
    allowed: bool
    reason: str


class PublicationLockBusy(Exception):
    """Niet wachten: deze job/tick kan geen nieuwe rechten verkrijgen."""


class PublicationError(Exception):
    """Het publicatiebewijs of de duurzame invalidatie is mislukt."""


class PublicationLease:
    """Niet overdraagbaar bewijs, alleen gemaakt door publication_lock()."""

    __slots__ = ("_path", "_pid", "_thread", "_active")

    def __init__(self):
        raise TypeError("Use publication_lock() to obtain a PublicationLease")

    def __setattr__(self, name, value):
        raise AttributeError("PublicationLease cannot be modified")

    @property
    def path(self) -> Path:
        return self._path

    def _check(self):
        if (not self._active or self._pid != os.getpid() or self._thread != threading.get_ident()
                or canonical_plan_path(self._path) != self._path):
            raise PublicationError("Lease is not held by this process/thread/path")


_THREAD_LOCKS: dict[tuple[int, Path], threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _after_fork():
    # Fork kan een gehouden threading.Lock erven, maar geen nieuw procesbewijs.
    global _REGISTRY_LOCK, _THREAD_LOCKS
    _REGISTRY_LOCK = threading.Lock()
    _THREAD_LOCKS = {}


os.register_at_fork(after_in_child=_after_fork)


def canonical_plan_path(path: str | Path) -> Path:
    """Canonicaliseer alleen de parent; de vaste plan-/lockleaf mag geen symlink zijn."""
    try:
        source = Path(path)
        if source.name in ("", ".", ".."):
            raise PublicationError("Expected fixed plan filename")
        result = source.parent.resolve() / source.name
        if result.is_symlink() or Path(str(result) + ".lock").is_symlink():
            raise PublicationError("Plan and lock files must not be symlinks")
        return result
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise PublicationError("Cannot canonicalize plan path") from exc


def gate_policy(pricing: PricingConfig | None) -> GatePolicy:
    contracts = () if pricing is None or pricing.settlement is None else pricing.settlement.contracts
    starts = [int(datetime.combine(contract.valid_from, time.min, ZoneInfo("Europe/Amsterdam")).timestamp())
              for contract in contracts if contract.product == "budget_dynamic_2027"]
    return GatePolicy(min(starts, default=None))


@contextmanager
def publication_lock(path: str | Path) -> Iterator[PublicationLease]:
    plan_path = canonical_plan_path(path)
    with _REGISTRY_LOCK:
        thread_lock = _THREAD_LOCKS.setdefault((os.getpid(), plan_path), threading.Lock())
    if not thread_lock.acquire(blocking=False):
        raise PublicationLockBusy("Publication path is busy in this process")
    fd, lease = None, None
    try:
        # Lockfile wordt nooit verwijderd/vervangen. O_NOFOLLOW sluit leafsymlinks uit.
        fd = os.open(str(plan_path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PublicationLockBusy("Publication path is busy in another process") from exc
        lease = object.__new__(PublicationLease)
        for name, value in (("_path", plan_path), ("_pid", os.getpid()),
                            ("_thread", threading.get_ident()), ("_active", True)):
            object.__setattr__(lease, name, value)
        lease._check()
        yield lease
    except OSError as exc:
        raise PublicationError(f"Publication lock/file I/O failed: {exc}") from exc
    finally:
        if lease is not None:
            object.__setattr__(lease, "_active", False)
        if fd is not None:
            os.close(fd)
        thread_lock.release()


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _finite_tree(value):
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Nonfinite number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def read_publication(path: str | Path) -> PlanRead:
    """Iedere keer volledige inhoud lezen; mtime/inode zijn geen autorisatiecache."""
    try:
        plan_path = canonical_plan_path(path)
        fd = os.open(plan_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, encoding="utf-8") as stream:
            payload = json.load(stream, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        _finite_tree(payload)
        if type(payload) is not dict:
            raise ValueError("Expected plan object")
        if type(payload.get("format_version")) is int and payload["format_version"] == 2 and payload.get("status") == "blocked":
            blocked = BlockedPlan.from_dict(payload)
            return PlanRead("blocked", None, blocked.reason)
        return PlanRead("ready", FastPlan.from_dict(payload), "ready")
    except FileNotFoundError:
        return PlanRead("missing", None, "missing")
    except (PublicationError, OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
        return PlanRead("invalid", None, f"invalid_publication: {exc}")


def _atomic_write(payload: dict, *, lease: PublicationLease) -> None:
    if type(lease) is not PublicationLease:
        raise PublicationError("Expected a live PublicationLease")
    lease._check()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=lease.path.parent,
                                         prefix=".fast_plan_", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        lease._check()
        os.replace(temporary, lease.path)
        directory_fd = os.open(lease.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, ValueError, TypeError) as exc:
        raise PublicationError(f"Atomic publication/flush/fsync failed: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def invalidate_plan(*, lease: PublicationLease, now_ts: int, reason: str) -> None:
    _atomic_write(BlockedPlan(2, now_ts, reason).to_dict(), lease=lease)


def write_publication(plan: FastPlan, *, lease: PublicationLease) -> None:
    """Alleen v1-compatibiliteit; geen weg terug vanuit blocked/v2/onbekend."""
    if type(lease) is not PublicationLease:
        raise PublicationError("Expected a live PublicationLease")
    lease._check()
    existing = read_publication(lease.path)
    if (type(plan) is not FastPlan or plan.format_version != 1 or plan.financial_context is not None
            or existing.status not in ("missing", "ready")
            or (existing.plan is not None and existing.plan.format_version != 1)):
        raise PublicationError("Legacy writer cannot publish/replace financial or unusable artifacts")
    try:
        payload = plan.to_dict()
        _finite_tree(payload)
        FastPlan.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise PublicationError("Invalid legacy plan") from exc
    _atomic_write(payload, lease=lease)


def _write_verified_ready(plan: FastPlan, *, lease: PublicationLease) -> None:
    """Interne adaptergrens: caller heeft de verse boek/configcheck onder deze lease gedaan.

    Geen tweede capabilityprotocol: alleen SettlementPublication gebruikt deze
    helper. Directe/niet-meewerkende bestandschrijvers vallen buiten de garantie.
    """
    if type(plan) is not FastPlan or plan.format_version != 2:
        raise PublicationError("Verified financial publication requires v2")
    _atomic_write(plan.to_dict(), lease=lease)


def following_verdict(read: PlanRead, *, policy: GatePolicy, now_ts: int) -> GateVerdict:
    _integer(now_ts, "now_ts")
    if read.status != "ready" or read.plan is None:
        return GateVerdict(False, read.reason)
    plan = read.plan
    if now_ts < plan.created_ts:
        return GateVerdict(False, "clock_before_creation")
    if plan.format_version == 2:
        context = plan.financial_context
        if context is None:
            return GateVerdict(False, "missing_financial_context")
        if now_ts >= context.not_after_ts:
            return GateVerdict(False, "plan_lease_expired")
        if not context.economics_supported:
            return GateVerdict(False, "economics_unsupported")
        # G1–G3 ontbreken. Ook een synthetisch correcte ready-v2 verleent geen actuatierecht.
        return GateVerdict(False, "actuation_gates_unproven")
    if policy.required_from_ts is not None and now_ts >= policy.required_from_ts:
        return GateVerdict(False, "financial_plan_required")
    if plan.format_version != 1 or plan.interval_at(now_ts) is None:
        return GateVerdict(False, "plan_outside_horizon")
    return GateVerdict(True, "legacy")
