"""Exclusive process ownership and durable operational evidence, not send rights.

Only the caller manages the lifetime flock. This module never acquires/closes
it, never sends commands, and supplies no qualified receiver or provisioning.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

from dao.prog import task_execution
from dao.prog.task_execution import Lease

from .actuator import FENCED_ADAPTER, classify_actuation_adapter
from .command import (
    ActuationReceipt,
    BarrierReceipt,
    BatteryCommand,
    Deadline,
    OwnerBinding,
    RevokeRequest,
    _integer,
    _reason,
    _string,
)
from .publication import (
    OWNER_CYCLE_NS,
    OWNER_START_NS,
    PUBLICATION_PHASE_NS,
    PublicationError,
    PublicationLease,
    PublicationLockBusy,
    canonical_plan_path,
    owner_admission_at,
    publication_lock,
)
from .receiver import FencedDeviceGateway, _helper, _helpers_from_dict, receipt_matches_command


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


@dataclass
class _Admission:
    """Runtime bookkeeping attached to the held lifetime, never durable rights."""

    pid: int
    boot_id: str
    last_ns: int
    first_cycle: int
    claimed_cycle: int = -1
    unhealthy: bool = False


class RecoveryRequired(Exception):
    """Explicit inhibited outcome; start returns it, strict storage raises it."""

    phase = "inhibited"

    def __init__(self, reason: str):
        _reason(reason, "owner_reason_invalid")
        self.reason = reason
        super().__init__(reason)


class MirrorWriteFailed(OSError):
    """Hard persistence error with a cause code the lifecycle hooks preserve."""

    reason = "mirror_write_failed"


def _keys(data, cls):
    if type(data) is not dict or data.keys() != {field.name for field in fields(cls)}:
        raise ValueError("owner_fields_invalid")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("owner_duplicate_field")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class OwnerState:
    schema_version: int
    installation_id: str
    receiver_id: str
    receiver_generation: str
    anchor_checkpoint: str
    owner_session_id: str
    owner_epoch: int
    last_reserved_sequence: int
    phase: Literal["inhibited", "binding", "owned", "unconfirmed", "revoking", "released"]
    pending_command: BatteryCommand | None
    last_receipt: ActuationReceipt | None
    pending_barrier_id: str | None
    last_barrier: BarrierReceipt | None

    def __post_init__(self):
        _integer(self.schema_version, "owner_schema_version_invalid")
        if self.schema_version != 1:
            raise ValueError("owner_schema_version_invalid")
        for name in ("installation_id", "receiver_id", "receiver_generation", "anchor_checkpoint"):
            _string(getattr(self, name), f"owner_{name}_invalid")
        for name in ("owner_epoch", "last_reserved_sequence"):
            _integer(getattr(self, name), f"owner_{name}_invalid")
        if type(self.phase) is not str or self.phase not in (
            "inhibited", "binding", "owned", "unconfirmed", "revoking", "released"
        ):
            raise ValueError("owner_phase_invalid")
        if type(self.owner_session_id) is not str:
            raise ValueError("owner_session_id_invalid")
        if self.phase not in ("inhibited", "released"):
            _string(self.owner_session_id, "owner_session_id_invalid")
        for name, cls in (("pending_command", BatteryCommand), ("last_receipt", ActuationReceipt),
                          ("last_barrier", BarrierReceipt)):
            value = getattr(self, name)
            if value is not None and type(value) is not cls:
                raise ValueError(f"owner_{name}_invalid")
        if self.last_receipt is not None:
            for result in self.last_receipt.helper_results:
                _helper(result)
        if self.pending_barrier_id is not None and type(self.pending_barrier_id) is not str:
            raise ValueError("owner_pending_barrier_id_invalid")

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.last_receipt is not None:
            result["last_receipt"]["helper_results"] = [asdict(item) for item in self.last_receipt.helper_results]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> OwnerState:
        _keys(data, cls)
        values = dict(data)
        if values["pending_command"] is not None:
            values["pending_command"] = BatteryCommand.from_dict(values["pending_command"])
        if values["last_receipt"] is not None:
            receipt = values["last_receipt"]
            _keys(receipt, ActuationReceipt)
            values["last_receipt"] = ActuationReceipt(**(
                receipt | {"helper_results": _helpers_from_dict(receipt["helper_results"])}
            ))
        if values["last_barrier"] is not None:
            _keys(values["last_barrier"], BarrierReceipt)
            values["last_barrier"] = BarrierReceipt(**values["last_barrier"])
        return cls(**values)


class OwnerStore:
    def __init__(self, plan_path: str | Path, installation_id: str, *, lease_dir: Path | None = None):
        _string(installation_id, "owner_installation_id_invalid")
        self.installation_id = installation_id
        self.plan_path = canonical_plan_path(plan_path)
        digest = hashlib.sha256(installation_id.encode("utf-8")).hexdigest()
        self.resource = f"actuator-{digest[:32]}"
        self.path = self.plan_path.parent / "actuation_owner" / f"{digest}.json"
        self.lease_dir = Path(task_execution.LEASE_DIR if lease_dir is None else lease_dir).absolute()

    def holds(self, writer: Lease) -> bool:
        """Check the *held* Linux OFD, without acquiring or unlocking anything.

        fdinfo describes locks on this descriptor's open file description, also
        after pass_fds/exec. A PID, a heartbeat, or an unlocked fd is no proof.
        """
        if type(writer) is not Lease or writer.resource != self.resource or writer.fd < 0:
            return False
        try:
            held = os.fstat(writer.fd)
            expected = os.stat(self.lease_dir / f"{self.resource}.lock", follow_symlinks=False)
            if not stat.S_ISREG(held.st_mode) or (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
                return False
            info = Path(f"/proc/self/fdinfo/{writer.fd}").read_text()
            return any(line.startswith("lock:") and "FLOCK  ADVISORY  WRITE" in line for line in info.splitlines())
        except OSError:
            return False

    def _directory(self, *, create=False):
        if create:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)

    def load_strict(self) -> OwnerState:
        try:
            directory = self._directory()
            try:
                fd = os.open(self.path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                             dir_fd=directory)
                with os.fdopen(fd, "r", encoding="utf-8") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ValueError("owner_mirror_invalid")
                    state = OwnerState.from_dict(json.load(stream, object_pairs_hook=_unique_object))
            finally:
                os.close(directory)
            if state.installation_id != self.installation_id:
                raise ValueError("owner_installation_mismatch")
            return state
        except (OSError, ValueError, UnicodeError) as exc:
            raise RecoveryRequired("mirror_missing_or_corrupt") from exc

    def write(self, state: OwnerState, *, writer: Lease) -> None:
        if not self.holds(writer):
            raise RecoveryRequired("owner_lease_missing")
        if type(state) is not OwnerState or state.installation_id != self.installation_id:
            raise ValueError("owner_state_invalid")
        payload = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        directory = self._directory(create=True)
        temporary = f".owner-{uuid.uuid4().hex}.tmp"
        try:
            os.fchmod(directory, 0o700)
            # Validate an existing destination with NOFOLLOW before replacement.
            try:
                target = os.open(self.path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                                 dir_fd=directory)
            except FileNotFoundError:
                pass
            else:
                try:
                    if not stat.S_ISREG(os.fstat(target).st_mode):
                        raise ValueError("owner_mirror_invalid")
                finally:
                    os.close(target)
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path.name, src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
            finally:
                os.close(directory)


class OwnerCoordinator:
    def __init__(self, store: OwnerStore, *, receiver: FencedDeviceGateway | None = None,
                 now_ns: Callable[[], int] | None = None, now_ms: Callable[[], int] | None = None):
        self.store = store
        self.receiver = receiver
        self.now_ns = now_ns or (lambda: time.clock_gettime_ns(time.CLOCK_BOOTTIME))
        self.now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)
        self.state: OwnerState | None = None
        self._sequence_high_water = 0
        self.writer: Lease | None = None
        self.binding: OwnerBinding | None = None
        self.phase = "inhibited"
        self.recovery_required = True
        self.pending_stop_reason: str | None = None
        self.last_admission_reason: str | None = None
        self._admission: _Admission | None = None
        self._section_lease: PublicationLease | None = None
        self._section_operation: str | None = None
        self._section_thread: int | None = None
        self._section_used = False
        self.section_deadline: Deadline | None = None
        self.transport_deadline: Deadline | None = None

    def _bind_lifetime(self, writer: Lease) -> None:
        if not self.store.holds(writer):
            raise self._inhibit("owner_lease_missing")
        now = self.now_ns()
        try:
            cycle, _, _ = owner_admission_at(now)
        except (TypeError, ValueError):
            raise self._inhibit("clock_unhealthy") from None
        # A new coordinator using the same lifetime cannot return its claim.
        admission = getattr(writer, "_fastctrl_admission", None)
        if admission is None or admission.pid != os.getpid():
            admission = _Admission(os.getpid(), _boot_id(), now, cycle + 1)
            writer._fastctrl_admission = admission
        self.writer, self._admission = writer, admission
        self._clock()

    def _clock(self) -> int:
        now = self.now_ns()
        admission = self._admission
        if (admission is None or admission.pid != os.getpid()
                or admission.unhealthy or not admission.boot_id or _boot_id() != admission.boot_id
                or type(now) is not int or now < admission.last_ns):
            if admission is not None:
                admission.unhealthy = True
            raise self._inhibit("clock_unhealthy")
        admission.last_ns = now
        return now

    @contextmanager
    def final_section(self, *, operation: Literal["start", "send", "stop"],
                      deadline: Deadline | None = None) -> Iterator[PublicationLease]:
        """One admitted installation section, one lock, one non-refundable claim."""
        if operation not in ("start", "send", "stop"):
            raise ValueError("owner_operation_invalid")
        if deadline is not None and type(deadline) is not Deadline:
            raise ValueError("owner_deadline_invalid")
        if self._admission is None:
            self._bind_lifetime(self.writer)
        started = self._clock()
        if not self.store.holds(self.writer):
            raise self._inhibit("owner_lease_missing")
        cycle, allowed, _ = owner_admission_at(started)
        admission = self._admission
        reason = None
        if not allowed or cycle < admission.first_cycle:
            reason = "admission_deferred"
        elif cycle <= admission.claimed_cycle:
            reason = "admission_cycle_used"
        if reason:
            self.last_admission_reason = reason
            raise RecoveryRequired(reason)
        now_ms = self.now_ms()
        end = min(started + OWNER_START_NS, cycle * OWNER_CYCLE_NS + PUBLICATION_PHASE_NS)
        wire_end = now_ms + 1000 if deadline is None else deadline.execute_before_ms
        if deadline is not None:
            end = min(end, deadline.monotonic_end_ns)
        end = min(end, started + max(0, wire_end - now_ms) * 1_000_000)
        section = Deadline(end, wire_end)
        transport = Deadline(min(end, started + 750_000_000), wire_end)
        with publication_lock(self.store.plan_path) as lease:
            acquired = self._clock()
            current_cycle, allowed, _ = owner_admission_at(acquired)
            if current_cycle != cycle or not allowed:
                self.last_admission_reason = "admission_window_missed"
                raise RecoveryRequired("admission_window_missed")
            if cycle <= admission.claimed_cycle:
                raise RecoveryRequired("admission_cycle_used")
            if not self.store.holds(self.writer):
                raise self._inhibit("owner_lease_missing")
            admission.claimed_cycle = cycle
            self._section_lease, self._section_operation = lease, operation
            self._section_thread = threading.get_ident()
            self._section_used = False
            self.section_deadline, self.transport_deadline = section, transport
            self.last_admission_reason = None
            try:
                yield lease
            finally:
                self._section_lease = self._section_operation = self._section_thread = None
                self.section_deadline = self.transport_deadline = None
                self._section_used = False

    def require_final_section(self, lease: PublicationLease, *,
                              operation: Literal["start", "send", "stop"]) -> None:
        if (type(lease) is not PublicationLease or lease is not self._section_lease
                or self._section_operation != operation or self._section_thread != threading.get_ident()
                or lease.path != self.store.plan_path):
            raise RecoveryRequired("owner_section_required")
        try:
            lease._check()
        except PublicationError:
            raise RecoveryRequired("owner_section_required") from None
        if not self.store.holds(self.writer):
            raise self._inhibit("owner_lease_missing")
        self._clock()

    def _stop_in_send_section(self, lease: PublicationLease, *, reason: str,
                              deadline: Deadline) -> BarrierReceipt | None:
        self.require_final_section(lease, operation="send")
        if self._section_used:
            self.pending_stop_reason = reason
            self._inhibit(reason)
            return None
        self._section_operation = "stop"
        return self._stop_locked(lease=lease, reason=reason, deadline=deadline)

    def _require_receiver(self, receiver: FencedDeviceGateway) -> None:
        if classify_actuation_adapter(receiver) != FENCED_ADAPTER:
            raise RecoveryRequired("adapter_unqualified")
        if receiver is not self.receiver:
            raise RecoveryRequired("receiver_binding_mismatch")

    def _covers_next_send(self, binding: OwnerBinding) -> bool:
        now_ns = self._clock()
        next_send = (now_ns // OWNER_CYCLE_NS + 1) * OWNER_CYCLE_NS
        required_ms = (next_send + OWNER_START_NS - now_ns + 999_999) // 1_000_000
        return binding.owner_not_after_ms > self.now_ms() + required_ms

    def _lifecycle_budget(self, deadline: Deadline) -> Deadline:
        if type(deadline) is not Deadline:
            raise ValueError("owner_deadline_invalid")
        wire_end = min(deadline.execute_before_ms, self.section_deadline.execute_before_ms)
        end = min(deadline.monotonic_end_ns, self.section_deadline.monotonic_end_ns,
                  self._clock() + max(0, wire_end - self.now_ms()) * 1_000_000)
        return Deadline(end, wire_end)

    def _inhibit(self, reason):
        self.phase = "inhibited"
        if self.state is not None:
            self.state = replace(self.state, phase="inhibited",
                                 last_reserved_sequence=max(self.state.last_reserved_sequence,
                                                            self._sequence_high_water))
        self.binding = None
        self.recovery_required = True
        return RecoveryRequired(reason)

    def report_failure(self, reason: str, *, durable: bool = False) -> RecoveryRequired:
        """Report a lifecycle failure without granting or restoring any right.

        Without `durable` the outcome stays in memory only, exactly as before:
        the runner hooks call it that way and can therefore never create or
        rewrite a mirror. With `durable` the inhibited phase and the recovery
        duty are written through the existing mirror route, but only when the
        mirror already exists; a mirror is never created here. A write failure
        is never propagated: the in-memory state stays inhibited and the
        recovery duty stays visible. The reason stays visible in the returned
        outcome.
        """
        result = self._inhibit(reason)
        if durable and self.state is not None:
            try:
                self._write(self.state)
            except Exception:
                # The mirror could not be written. The in-memory state stays
                # inhibited and the recovery duty stays visible; the reason
                # stays visible in the returned outcome.
                pass
        return result

    def _write(self, state):
        # A failed replace/fsync may already be visible. Never let a later
        # lifecycle write roll back this process's attempted reservation.
        self._sequence_high_water = max(self._sequence_high_water, state.last_reserved_sequence)
        state = replace(state, last_reserved_sequence=self._sequence_high_water)
        try:
            self.store.write(state, writer=self.writer)
        except RecoveryRequired as exc:
            self.report_failure(exc.reason)
            raise
        except Exception as exc:
            self.report_failure("mirror_write_failed")
            raise MirrorWriteFailed(str(exc)) from exc
        self.state = state
        self.phase = state.phase

    @staticmethod
    def _exact(receipt, request, generation):
        return (type(receipt) is BarrierReceipt and receipt.installation_id == request.installation_id
                and receipt.barrier_id == request.barrier_id
                and receipt.revoked_through_epoch == request.expected_epoch
                and receipt.current_epoch == request.expected_epoch + 1
                and receipt.receiver_generation == generation and receipt.status == "accepted"
                and receipt.queues_fenced is True and receipt.safe_state == "confirmed")

    def start(self, *, writer: Lease, receiver: FencedDeviceGateway, installation_id: str,
              owner_session_id: str, requested_not_after_ms: int,
              deadline: Deadline, now_ms: int) -> OwnerBinding | RecoveryRequired:
        self._inhibit("owner_lease_missing")
        try:
            self._require_receiver(receiver)
            self._bind_lifetime(writer)
            with self.final_section(operation="start", deadline=deadline) as lease:
                return self._start_locked(lease=lease, writer=writer, receiver=receiver,
                    installation_id=installation_id, owner_session_id=owner_session_id,
                    requested_not_after_ms=requested_not_after_ms,
                    deadline=self.section_deadline, now_ms=now_ms)
        except PublicationLockBusy:
            return RecoveryRequired("publication_busy")
        except RecoveryRequired as exc:
            return exc

    def _start_locked(self, *, lease: PublicationLease, writer: Lease,
                      receiver: FencedDeviceGateway, installation_id: str,
                      owner_session_id: str, requested_not_after_ms: int,
                      deadline: Deadline, now_ms: int) -> OwnerBinding | RecoveryRequired:
        self._require_receiver(receiver)
        self.require_final_section(lease, operation="start")
        deadline = self._lifecycle_budget(deadline)
        if writer is not self.writer:
            return self._inhibit("owner_lease_missing")
        if installation_id != self.store.installation_id:
            return self._inhibit("mirror_missing_or_corrupt")
        transport = Deadline(min(deadline.monotonic_end_ns, self.transport_deadline.monotonic_end_ns),
                             min(deadline.execute_before_ms, self.transport_deadline.execute_before_ms))
        if transport.remaining_ns(self._clock()) == 0:
            return self._inhibit("deadline_exceeded")
        try:
            loaded = self.store.load_strict()
        except RecoveryRequired as result:
            # Missing only: unknown metadata is explicitly inhibited, not a
            # fabricated receiver identity or trusted production bootstrap.
            try:
                self.store.path.lstat()
            except FileNotFoundError:
                self._write(OwnerState(1, installation_id, "unknown", "unknown", "unknown", "", 0, 0,
                                       "inhibited", None, None, None, None))
            except OSError:
                return result
            return result
        self._write(replace(loaded, phase="inhibited"))
        try:
            if transport.remaining_ns(self._clock()) == 0:
                return self._inhibit("deadline_exceeded")
            snapshot = receiver.inspect(deadline=transport)
            if (snapshot.installation_id != installation_id or not snapshot.queue_known
                    or snapshot.epoch_high_water < 1):
                return self._inhibit("queue_unknown")
        except RecoveryRequired:
            raise
        except Exception:
            return self._inhibit("queue_unknown")
        try:
            _string(owner_session_id, "owner_session_id_invalid")
            _integer(requested_not_after_ms, "owner_not_after_ms_invalid")
            request = RevokeRequest(installation_id, snapshot.epoch_high_water, uuid.uuid4().hex,
                                    "owner_start", now_ms)
        except ValueError:
            return self._inhibit("bind_refused")
        self._write(replace(self.state, receiver_id=snapshot.receiver_id,
                            receiver_generation=snapshot.receiver_generation,
                            anchor_checkpoint=snapshot.anchor_checkpoint,
                            owner_session_id=owner_session_id, phase="binding",
                            last_reserved_sequence=max(self.state.last_reserved_sequence, snapshot.sequence_high_water),
                            pending_barrier_id=request.barrier_id))
        if transport.remaining_ns(self._clock()) == 0:
            return self._inhibit("barrier_unconfirmed")
        try:
            barrier = receiver.revoke_and_stop(request, deadline=transport)
        except Exception:
            return self._inhibit("barrier_unconfirmed")
        if (not self._exact(barrier, request, snapshot.receiver_generation)
                or transport.remaining_ns(self._clock()) == 0):
            return self._inhibit("barrier_unconfirmed")
        new_epoch = snapshot.epoch_high_water + 1
        try:
            binding = receiver.bind_owner(installation_id=installation_id, barrier_id=request.barrier_id,
                                          expected_epoch=new_epoch, owner_session_id=owner_session_id,
                                          requested_not_after_ms=requested_not_after_ms, deadline=transport)
            if (type(binding) is not OwnerBinding or binding.installation_id != installation_id
                    or binding.receiver_id != snapshot.receiver_id
                    or binding.receiver_generation != snapshot.receiver_generation
                    or binding.owner_session_id != owner_session_id or binding.owner_epoch != new_epoch
                    or binding.barrier_id != request.barrier_id
                    or binding.owner_not_after_ms > requested_not_after_ms
                    or transport.remaining_ns(self._clock()) == 0):
                return self._inhibit("bind_refused")
        except RecoveryRequired:
            raise
        except Exception:
            return self._inhibit("bind_refused")
        if self.now_ms() >= binding.owner_not_after_ms:
            return self._inhibit("owner_lease_expired")
        if not self._covers_next_send(binding):
            return self._inhibit("owner_lease_too_short")
        self._write(replace(self.state, owner_epoch=new_epoch, phase="owned",
                            pending_barrier_id=None, last_barrier=barrier))
        if deadline.remaining_ns(self._clock()) == 0:
            return self._inhibit("deadline_exceeded")
        if self.now_ms() >= binding.owner_not_after_ms:
            return self._inhibit("owner_lease_expired")
        if not self._covers_next_send(binding):
            return self._inhibit("owner_lease_too_short")
        self.binding = binding
        self.pending_stop_reason = None
        self.recovery_required = False
        return binding

    def stop(self, *, reason: str, deadline: Deadline | None = None) -> BarrierReceipt | None:
        """Withdraw local rights immediately; defer the barrier until admitted."""
        _reason(reason, "owner_reason_invalid")
        if deadline is not None and type(deadline) is not Deadline:
            raise ValueError("owner_deadline_invalid")
        if self.phase == "released" and not self.recovery_required:
            self.pending_stop_reason = None
            return self.state.last_barrier
        self.binding = None
        self.recovery_required = True
        self.pending_stop_reason = reason
        if self.state is None or self.state.owner_epoch == 0:
            self.phase = "inhibited"
            return None
        if not self.store.holds(self.writer):
            self._inhibit("owner_lease_missing")
            return None
        entered = False
        try:
            self._require_receiver(self.receiver)
            with self.final_section(operation="stop", deadline=deadline) as lease:
                entered = True
                return self._stop_locked(lease=lease, reason=reason, deadline=self.section_deadline)
        except PublicationLockBusy:
            self.last_admission_reason = "publication_busy"
            return None
        except RecoveryRequired as exc:
            if entered:
                raise
            self.last_admission_reason = exc.reason
            return None

    def _stop_locked(self, *, lease: PublicationLease, reason: str,
                     deadline: Deadline) -> BarrierReceipt | None:
        self._require_receiver(self.receiver)
        self.require_final_section(lease, operation="stop")
        deadline = self._lifecycle_budget(deadline)
        self.binding = None
        self.recovery_required = True
        self.pending_stop_reason = reason
        now_ms = self.now_ms()
        transport = Deadline(min(deadline.monotonic_end_ns, self.transport_deadline.monotonic_end_ns),
                             min(deadline.execute_before_ms, self.transport_deadline.execute_before_ms))
        if transport.remaining_ns(self._clock()) == 0:
            self._inhibit("budget_exhausted")
            return None
        request = RevokeRequest(self.state.installation_id, self.state.owner_epoch,
                                uuid.uuid4().hex, reason, now_ms)
        self._write(replace(self.state, phase="revoking", pending_barrier_id=request.barrier_id))
        try:
            if transport.remaining_ns(self._clock()) == 0:
                self._write(replace(self.state, phase="unconfirmed"))
                return None
            receipt = self.receiver.revoke_and_stop(request, deadline=transport)
        except Exception:
            self._write(replace(self.state, phase="unconfirmed"))
            return None
        exact = (self._exact(receipt, request, self.state.receiver_generation)
                 and transport.remaining_ns(self._clock()) > 0)
        self._write(replace(self.state, phase="released" if exact else "unconfirmed",
                            pending_barrier_id=None if exact else request.barrier_id,
                            last_barrier=receipt if type(receipt) is BarrierReceipt else self.state.last_barrier))
        self.recovery_required = not exact
        if deadline.remaining_ns(self._clock()) == 0:
            self._inhibit("deadline_exceeded")
            return None
        if exact:
            self.pending_stop_reason = None
        return receipt

    def reserve_sequence(self, *, writer: Lease) -> int:
        if not self.store.holds(writer) or writer is not self.writer:
            raise self._inhibit("owner_lease_missing")
        if self.phase != "owned" or self.binding is None:
            raise self._inhibit("bind_refused")
        if self._section_lease is not None:
            self._section_used = True
        sequence = self._sequence_high_water + 1
        self._write(replace(self.state, last_reserved_sequence=sequence))
        return sequence

    def record_pending_command(self, command: BatteryCommand) -> None:
        """Record the immutable pending command durably, before any send.

        The phase stays what the lifecycle determined and the recovery duty is
        never cleared here: this is operational evidence, not a right.
        """
        if type(command) is not BatteryCommand:
            raise ValueError("owner_pending_command_invalid")
        if self.state is None:
            raise RecoveryRequired("owner_state_missing")
        self._write(replace(self.state, pending_command=command))

    def record_receipt(self, receipt: ActuationReceipt, *, terminal: bool = True) -> None:
        """Record the receiver result durably.

        Only a receipt matching the stored pending command may be recorded.
        An accepted or rejected order clears that command; an uncertain order
        stays visible. The phase and the recovery duty are untouched.
        With `terminal=False` the receipt is kept as history only and the
        pending command always stays visible.
        """
        if type(receipt) is not ActuationReceipt:
            raise ValueError("owner_receipt_invalid")
        if type(terminal) is not bool:
            raise ValueError("owner_receipt_terminal_invalid")
        pending = self.state.pending_command if self.state is not None else None
        if pending is None:
            raise RecoveryRequired("pending_command_missing")
        if not receipt_matches_command(receipt, pending):
            raise RecoveryRequired("receipt_identity_mismatch")
        if terminal and receipt.status in ("accepted", "rejected"):
            pending = None
        self._write(replace(self.state, pending_command=pending, last_receipt=receipt))
