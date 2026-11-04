"""Exclusive process ownership and durable operational evidence, not send rights.

Only the caller manages the lifetime flock. This module never acquires/closes
it, never sends commands, and supplies no qualified receiver or provisioning.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

from dao.prog import task_execution
from dao.prog.task_execution import Lease

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
from .receiver import FencedDeviceGateway, _helper, _helpers_from_dict


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
        digest = hashlib.sha256(installation_id.encode("utf-8")).hexdigest()
        self.resource = f"actuator-{digest[:32]}"
        self.path = Path(plan_path).absolute().parent / "actuation_owner" / f"{digest}.json"
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
        if not self.store.holds(writer):
            return RecoveryRequired("owner_lease_missing")
        self.writer = writer
        self.receiver = receiver
        if installation_id != self.store.installation_id:
            return self._inhibit("mirror_missing_or_corrupt")
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
            snapshot = receiver.inspect(deadline=deadline)
            if (snapshot.installation_id != installation_id or not snapshot.queue_known
                    or snapshot.epoch_high_water < 1):
                return self._inhibit("queue_unknown")
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
        if deadline.remaining_ns(self.now_ns()) == 0:
            return self._inhibit("barrier_unconfirmed")
        try:
            barrier = receiver.revoke_and_stop(request, deadline=deadline)
        except Exception:
            return self._inhibit("barrier_unconfirmed")
        if (not self._exact(barrier, request, snapshot.receiver_generation)
                or deadline.remaining_ns(self.now_ns()) == 0):
            return self._inhibit("barrier_unconfirmed")
        new_epoch = snapshot.epoch_high_water + 1
        try:
            binding = receiver.bind_owner(installation_id=installation_id, barrier_id=request.barrier_id,
                                          expected_epoch=new_epoch, owner_session_id=owner_session_id,
                                          requested_not_after_ms=requested_not_after_ms, deadline=deadline)
            if (type(binding) is not OwnerBinding or binding.installation_id != installation_id
                    or binding.receiver_id != snapshot.receiver_id
                    or binding.receiver_generation != snapshot.receiver_generation
                    or binding.owner_session_id != owner_session_id or binding.owner_epoch != new_epoch
                    or binding.barrier_id != request.barrier_id
                    or binding.owner_not_after_ms > requested_not_after_ms
                    or deadline.remaining_ns(self.now_ns()) == 0):
                return self._inhibit("bind_refused")
        except Exception:
            return self._inhibit("bind_refused")
        if self.now_ms() >= binding.owner_not_after_ms:
            return self._inhibit("owner_lease_expired")
        self._write(replace(self.state, owner_epoch=new_epoch, phase="owned",
                            pending_barrier_id=None, last_barrier=barrier))
        if deadline.remaining_ns(self.now_ns()) == 0:
            return self._inhibit("deadline_exceeded")
        if self.now_ms() >= binding.owner_not_after_ms:
            return self._inhibit("owner_lease_expired")
        self.binding = binding
        self.recovery_required = False
        return binding

    def stop(self, *, reason: str, deadline: Deadline | None = None) -> BarrierReceipt | None:
        """Withdraw the epoch; the caller may hand in the remaining section budget.

        Without `deadline` the behaviour is exactly as before: the revocation
        gets its own 750 ms window. With `deadline` the caller keeps the
        authority over the budget; no second window is ever created.
        """
        _reason(reason, "owner_reason_invalid")
        if deadline is not None and type(deadline) is not Deadline:
            raise ValueError("owner_deadline_invalid")
        self.binding = None
        self.recovery_required = True
        if self.state is None or self.state.owner_epoch == 0:
            self.phase = "inhibited"
            return None
        if not self.store.holds(self.writer):
            self._inhibit("owner_lease_missing")
            return None
        now_ms = self.now_ms()
        if deadline is None:
            deadline = Deadline(self.now_ns() + 750_000_000, now_ms + 750)
        request = RevokeRequest(self.state.installation_id, self.state.owner_epoch,
                                uuid.uuid4().hex, reason, now_ms)
        self._write(replace(self.state, phase="revoking", pending_barrier_id=request.barrier_id))
        try:
            receipt = self.receiver.revoke_and_stop(request, deadline=deadline)
        except Exception:
            self._write(replace(self.state, phase="unconfirmed"))
            return None
        exact = (self._exact(receipt, request, self.state.receiver_generation)
                 and deadline.remaining_ns(self.now_ns()) > 0)
        self._write(replace(self.state, phase="released" if exact else "unconfirmed",
                            pending_barrier_id=None if exact else request.barrier_id,
                            last_barrier=receipt if type(receipt) is BarrierReceipt else self.state.last_barrier))
        self.recovery_required = not exact
        return receipt

    def reserve_sequence(self, *, writer: Lease) -> int:
        if not self.store.holds(writer) or writer is not self.writer:
            raise self._inhibit("owner_lease_missing")
        if self.phase != "owned" or self.binding is None:
            raise self._inhibit("bind_refused")
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

        An accepted or rejected order clears the pending command; an uncertain
        order stays visible. The phase and the recovery duty are untouched.
        With `terminal=False` the receipt is kept as history only and the
        pending command always stays visible.
        """
        if type(receipt) is not ActuationReceipt:
            raise ValueError("owner_receipt_invalid")
        if type(terminal) is not bool:
            raise ValueError("owner_receipt_terminal_invalid")
        if self.state is None:
            raise RecoveryRequired("owner_state_missing")
        pending = self.state.pending_command
        if terminal and receipt.status in ("accepted", "rejected"):
            pending = None
        self._write(replace(self.state, pending_command=pending, last_receipt=receipt))
