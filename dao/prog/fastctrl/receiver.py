"""Pure receiving-boundary contracts and transitions; no qualified device adapter.

Time bounds, helper observations, queue evidence and the independent rollback
anchor are inputs supplied by an integration, never inferred here. A reservation
is not acceptance. Only an effect fenced at the last writer can be accepted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Literal, Protocol

from .command import (
    ActuationReceipt,
    BarrierReceipt,
    BatteryCommand,
    Deadline,
    HelperResult,
    OwnerBinding,
    RevokeRequest,
)


def _string(value, code="receiver_string_invalid"):
    if type(value) is not str or not value.strip():
        raise ValueError(code)


def _integer(value, code="receiver_integer_invalid"):
    if type(value) is not int or value < 0:
        raise ValueError(code)


def _boolean(value):
    if type(value) is not bool:
        raise ValueError("receiver_boolean_invalid")


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("receiver_digest_invalid")


def _reason(value):
    if (type(value) is not str or not value or value[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in value)):
        raise ValueError("receiver_reason_invalid")


def _choice(value, choices):
    if type(value) is not str or value not in choices:
        raise ValueError("receiver_choice_invalid")


def _keys(data, cls):
    if type(data) is not dict or data.keys() != {field.name for field in fields(cls)}:
        raise ValueError("receiver_fields_invalid")


def _helper(result):
    if type(result) is not HelperResult:
        raise ValueError("receiver_helper_invalid")
    _choice(result.helper, ("setpoint", "mode", "stop_inverter"))
    _string(result.entity_id)
    _boolean(result.ok)
    _integer(result.attempts)
    if result.attempts < 1:
        raise ValueError("receiver_helper_attempts_invalid")
    if result.error is not None:
        _choice(result.error, ("write_failed", "readback_unreadable", "readback_mismatch"))


def _helpers_from_dict(data):
    if type(data) is not list:
        raise ValueError("receiver_helpers_invalid")
    results = []
    for item in data:
        _keys(item, HelperResult)
        result = HelperResult(**item)
        _helper(result)
        results.append(result)
    return tuple(results)


def _barrier_from_dict(data):
    if data is None:
        return None
    _keys(data, BarrierReceipt)
    return BarrierReceipt(**data)


@dataclass(frozen=True, slots=True)
class ReceiverCommandRecord:
    command_id: str
    battery_name: str
    plan_digest: str
    owner_session_id: str
    owner_epoch: int
    sequence: int
    payload_digest: str
    phase: Literal["reserved", "accepted", "rejected", "invalidated", "unknown"]
    reason: str
    issued_at_ms: int
    execute_before_ms: int
    hold_until_ms: int
    accepted_ts: int | None
    observed_at_ms: int
    effect: Literal["none", "complete", "partial", "unknown"]
    helper_results: tuple[HelperResult, ...]

    def __post_init__(self):
        for name in ("command_id", "battery_name", "owner_session_id"):
            _string(getattr(self, name))
        for name in ("plan_digest", "payload_digest"):
            _digest(getattr(self, name))
        for name in ("owner_epoch", "sequence", "issued_at_ms", "execute_before_ms",
                     "hold_until_ms", "observed_at_ms"):
            _integer(getattr(self, name))
        if self.owner_epoch < 1 or self.sequence < 1:
            raise ValueError("receiver_command_identity_invalid")
        if not self.issued_at_ms < self.execute_before_ms <= self.hold_until_ms:
            raise ValueError("receiver_command_time_order_invalid")
        _choice(self.phase, ("reserved", "accepted", "rejected", "invalidated", "unknown"))
        _choice(self.effect, ("none", "complete", "partial", "unknown"))
        _reason(self.reason)
        if self.accepted_ts is not None:
            _integer(self.accepted_ts)
        if (self.accepted_ts is not None) != (self.phase == "accepted"):
            raise ValueError("receiver_acceptance_invalid")
        if type(self.helper_results) is not tuple:
            raise ValueError("receiver_helpers_invalid")
        for result in self.helper_results:
            _helper(result)
        if self.phase == "accepted" and (self.effect != "complete" or not all(r.ok for r in self.helper_results)):
            raise ValueError("receiver_acceptance_incomplete")
        # Phase is history, not a no-effect certificate. A stored rejection or
        # invalidation can still carry partial/unknown effects (§3bis).
        if self.phase != "accepted" and self.effect == "complete":
            raise ValueError("receiver_unconfirmed_effect_invalid")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["helper_results"] = [asdict(result) for result in self.helper_results]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ReceiverCommandRecord:
        _keys(data, cls)
        return cls(**(data | {"helper_results": _helpers_from_dict(data["helper_results"])}))


@dataclass(frozen=True, slots=True)
class ReceiverState:
    schema_version: int
    installation_id: str
    receiver_id: str
    receiver_generation: str
    anchor_checkpoint: str
    epoch_high_water: int
    sequence_high_water: int
    bound_owner_session_id: str | None
    bound_owner_epoch: int
    owner_not_after_ms: int
    commands: tuple[ReceiverCommandRecord, ...]
    inhibited: bool
    queue_known: bool
    pending_barrier_id: str | None
    last_barrier: BarrierReceipt | None
    queue_generation: str

    def __post_init__(self):
        for name in ("schema_version", "epoch_high_water", "sequence_high_water",
                     "bound_owner_epoch", "owner_not_after_ms"):
            _integer(getattr(self, name))
        if self.schema_version != 1:
            raise ValueError("receiver_schema_version_invalid")
        for name in ("installation_id", "receiver_id", "receiver_generation", "anchor_checkpoint", "queue_generation"):
            _string(getattr(self, name))
        for name in ("bound_owner_session_id", "pending_barrier_id"):
            if getattr(self, name) is not None:
                _string(getattr(self, name))
        _boolean(self.inhibited)
        _boolean(self.queue_known)
        if type(self.commands) is not tuple or any(type(record) is not ReceiverCommandRecord for record in self.commands):
            raise ValueError("receiver_commands_invalid")
        if len({record.command_id for record in self.commands}) != len(self.commands):
            raise ValueError("receiver_duplicate_record")
        if self.last_barrier is not None and type(self.last_barrier) is not BarrierReceipt:
            raise ValueError("receiver_barrier_invalid")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["commands"] = [record.to_dict() for record in self.commands]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ReceiverState:
        _keys(data, cls)
        if type(data["commands"]) is not list:
            raise ValueError("receiver_commands_invalid")
        return cls(**(data | {
            "commands": tuple(ReceiverCommandRecord.from_dict(record) for record in data["commands"]),
            "last_barrier": _barrier_from_dict(data["last_barrier"]),
        }))


@dataclass(frozen=True, slots=True)
class ReceiverSnapshot:
    installation_id: str
    receiver_id: str
    receiver_generation: str
    anchor_checkpoint: str
    epoch_high_water: int
    sequence_high_water: int
    bound_owner_session_id: str | None
    bound_owner_not_after_ms: int
    inhibited: bool
    queue_known: bool
    last_barrier: BarrierReceipt | None
    queue_generation: str

    def __post_init__(self):
        for name in ("installation_id", "receiver_id", "receiver_generation", "anchor_checkpoint", "queue_generation"):
            _string(getattr(self, name))
        for name in ("epoch_high_water", "sequence_high_water", "bound_owner_not_after_ms"):
            _integer(getattr(self, name))
        if self.bound_owner_session_id is not None:
            _string(self.bound_owner_session_id)
        _boolean(self.inhibited)
        _boolean(self.queue_known)
        if self.last_barrier is not None and type(self.last_barrier) is not BarrierReceipt:
            raise ValueError("receiver_barrier_invalid")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ReceiverSnapshot:
        _keys(data, cls)
        return cls(**(data | {"last_barrier": _barrier_from_dict(data["last_barrier"])}))


class FencedDeviceGateway(Protocol):
    def inspect(self, *, deadline: Deadline) -> ReceiverSnapshot: ...
    def revoke_and_stop(self, request: RevokeRequest, *, deadline: Deadline) -> BarrierReceipt: ...
    def bind_owner(self, *, installation_id: str, barrier_id: str, expected_epoch: int,
                   owner_session_id: str, requested_not_after_ms: int, deadline: Deadline) -> OwnerBinding: ...
    def send(self, command: BatteryCommand, *, deadline: Deadline) -> ActuationReceipt: ...
    def query(self, *, installation_id: str, owner_epoch: int, sequence: int,
              command_id: str, payload_digest: str, deadline: Deadline) -> ActuationReceipt | None: ...


@dataclass(frozen=True, slots=True)
class SendDecision:
    status: Literal["reserved", "rejected"]
    reason: str
    state: ReceiverState
    record: ReceiverCommandRecord
    receipt: ActuationReceipt


@dataclass(frozen=True, slots=True)
class EffectDecision:
    status: Literal["accepted", "rejected", "unconfirmed"]
    reason: str
    state: ReceiverState
    record: ReceiverCommandRecord
    receipt: ActuationReceipt


@dataclass(frozen=True, slots=True)
class RevokeDecision:
    status: Literal["accepted", "rejected", "unconfirmed"]
    reason: str
    state: ReceiverState
    receipt: BarrierReceipt


@dataclass(frozen=True, slots=True)
class BindDecision:
    status: Literal["accepted", "rejected"]
    reason: str
    state: ReceiverState
    binding: OwnerBinding | None


def receiver_snapshot(state: ReceiverState) -> ReceiverSnapshot:
    data = {field.name: getattr(state, field.name) for field in fields(ReceiverSnapshot)
            if field.name != "bound_owner_not_after_ms"}
    return ReceiverSnapshot(**data, bound_owner_not_after_ms=state.owner_not_after_ms)


def payload_digest(command: BatteryCommand) -> str:
    data = command.to_dict()
    del data["plan_digest"]
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def receipt_matches_command(receipt: ActuationReceipt, command: BatteryCommand) -> bool:
    """Bind an observation to the exact command, independently of its status."""
    return (type(receipt) is ActuationReceipt and type(command) is BatteryCommand
            and all(getattr(receipt, name) == getattr(command, name) for name in (
                "installation_id", "battery_name", "owner_session_id", "owner_epoch",
                "sequence", "command_id", "plan_digest"))
            and receipt.payload_digest == payload_digest(command))


def _find(state, command_id):
    return next((record for record in state.commands if record.command_id == command_id), None)


def _record(command, now, *, phase="reserved", reason="reserved"):
    return ReceiverCommandRecord(
        command_id=command.command_id, battery_name=command.battery_name,
        plan_digest=command.plan_digest, owner_session_id=command.owner_session_id,
        owner_epoch=command.owner_epoch, sequence=command.sequence, payload_digest=payload_digest(command),
        phase=phase, reason=reason, issued_at_ms=command.issued_at_ms,
        execute_before_ms=command.execute_before_ms, hold_until_ms=command.hold_until_ms,
        accepted_ts=None, observed_at_ms=now, effect="none", helper_results=(),
    )


def _put(state, record):
    if _find(state, record.command_id) is None:
        return replace(state, commands=state.commands + (record,))
    return replace(state, commands=tuple(record if old.command_id == record.command_id else old
                                         for old in state.commands))


def _receipt(installation_id, record, *, status=None, reason=None):
    refused_status = "rejected" if record.effect == "none" else "unconfirmed"
    translation = {
        "accepted": ("accepted", "accepted"),
        "rejected": (refused_status, record.reason),
        "invalidated": (refused_status, "invalidated"),
        "unknown": ("unconfirmed", record.reason),
        "reserved": ("unconfirmed", "pending_effect"),
    }
    default_status, default_reason = translation[record.phase]
    status = status or default_status
    return ActuationReceipt(
        installation_id=installation_id, battery_name=record.battery_name,
        owner_epoch=record.owner_epoch, sequence=record.sequence, command_id=record.command_id,
        plan_digest=record.plan_digest, accepted_ts=record.accepted_ts if status == "accepted" else None,
        status=status, helper_results=() if status == "rejected" else record.helper_results,
        owner_session_id=record.owner_session_id, payload_digest=record.payload_digest,
        reason=reason or default_reason, observed_at_ms=record.observed_at_ms,
        effect=record.effect,
    )


def classify_send(state: ReceiverState, command: BatteryCommand, *, receiver_now_upper_ms: int,
                  clock_healthy: bool, clock_uncertainty_ms: int | None) -> SendDecision:
    _integer(receiver_now_upper_ms)
    _boolean(clock_healthy)
    existing = _find(state, command.command_id)
    checks = (
        (state.inhibited, "receiver_inhibited"),
        (not state.queue_known, "queue_unknown"),
        (command.installation_id != state.installation_id, "wrong_installation"),
        (state.bound_owner_session_id is None or command.owner_session_id != state.bound_owner_session_id, "wrong_owner_session"),
        (command.owner_epoch < state.epoch_high_water, "stale_epoch"),
        (command.owner_epoch > state.epoch_high_water, "epoch_not_granted"),
        (command.sequence <= state.sequence_high_water, "stale_or_duplicate_sequence"),
        (existing is not None and existing.payload_digest != payload_digest(command), "payload_mismatch"),
        (existing is not None, "duplicate_command"),
        (receiver_now_upper_ms >= command.execute_before_ms, "command_expired"),
        (receiver_now_upper_ms > command.hold_until_ms, "command_expired"),
        (state.owner_not_after_ms > 0 and receiver_now_upper_ms > state.owner_not_after_ms, "owner_lease_expired"),
        (not clock_healthy, "clock_unhealthy"),
        (type(clock_uncertainty_ms) is not int or clock_uncertainty_ms < 0, "clock_uncertainty_unknown"),
    )
    reason = next((reason for refused, reason in checks if refused), None)
    if reason is not None:
        if existing is not None and existing.effect != "none":
            # The retry attempt is refused, not re-executed. Its receipt must
            # still report the original command's already observed effects.
            return SendDecision("rejected", reason, state, existing, _receipt(state.installation_id, existing))
        rejected = _record(command, receiver_now_upper_ms, phase="rejected", reason=reason)
        # Never replace the proof for an earlier attempt, especially an accepted one.
        updated = _put(state, rejected) if existing is None else state
        return SendDecision("rejected", reason, updated, rejected, _receipt(command.installation_id, rejected))
    record = _record(command, receiver_now_upper_ms)
    records = tuple(replace(old, phase="invalidated", reason="superseded_by_higher_sequence")
                    if old.phase == "reserved" and old.sequence < command.sequence else old
                    for old in state.commands)
    updated = replace(state, commands=records + (record,), sequence_high_water=command.sequence)
    return SendDecision("reserved", "reserved", updated, record, _receipt(state.installation_id, record))


def fence_effect(state: ReceiverState, command: BatteryCommand, *, receiver_now_upper_ms: int,
                 clock_healthy: bool, clock_uncertainty_ms: int | None) -> tuple[bool, str]:
    _boolean(clock_healthy)
    if state.epoch_high_water != command.owner_epoch:
        return False, "epoch_withdrawn"
    if state.inhibited:
        return False, "receiver_inhibited"
    if not state.queue_known:
        return False, "queue_unknown"
    if command.installation_id != state.installation_id:
        return False, "wrong_installation"
    if command.owner_session_id != state.bound_owner_session_id:
        return False, "wrong_owner_session"
    if command.owner_epoch != state.bound_owner_epoch:
        return False, "epoch_withdrawn"
    if not clock_healthy:
        return False, "clock_unhealthy"
    if type(clock_uncertainty_ms) is not int or clock_uncertainty_ms < 0:
        return False, "clock_uncertainty_unknown"
    _integer(receiver_now_upper_ms)
    if receiver_now_upper_ms >= command.execute_before_ms:
        return False, "command_expired"
    record = _find(state, command.command_id)
    if (record is None or record.phase != "reserved" or record.owner_epoch != command.owner_epoch
            or record.sequence != command.sequence or record.payload_digest != payload_digest(command)
            or record.plan_digest != command.plan_digest or command.sequence != state.sequence_high_water):
        return False, "no_longer_reserved"
    return True, "reserved"


def classify_effect(state: ReceiverState, command: BatteryCommand, *, receiver_now_upper_ms: int,
                    clock_healthy: bool, clock_uncertainty_ms: int | None,
                    outcome: Literal["complete", "partial", "none", "unknown"]) -> EffectDecision:
    _choice(outcome, ("complete", "partial", "none", "unknown"))
    allowed, reason = fence_effect(state, command, receiver_now_upper_ms=receiver_now_upper_ms,
                                    clock_healthy=clock_healthy, clock_uncertainty_ms=clock_uncertainty_ms)
    existing = _find(state, command.command_id)
    if not allowed:
        # Refusing a later writer step cannot certify absence of earlier writes.
        # A completed command keeps its exact historical acceptance; no new
        # effect takes place, and complete can never become unconfirmed.
        if existing is not None and existing.phase == "accepted":
            receipt = _receipt(state.installation_id, existing)
            return EffectDecision(receipt.status, reason, state, existing, receipt)
        different_identity = existing is not None and (
            existing.owner_epoch != command.owner_epoch or existing.sequence != command.sequence
            or existing.payload_digest != payload_digest(command) or existing.plan_digest != command.plan_digest
        )
        if different_identity:
            # Never mutate the original reservation for a mismatched callback.
            status = "rejected" if existing.effect == "none" else "unconfirmed"
            receipt = _receipt(state.installation_id, existing, status=status, reason=reason)
            return EffectDecision(status, reason, state, existing, receipt)
        if existing is None:
            record = _record(command, receiver_now_upper_ms, phase="rejected", reason=reason)
        else:
            record = replace(existing, phase="rejected" if existing.effect == "none" else "unknown",
                             reason=reason, accepted_ts=None)
        receipt = _receipt(state.installation_id, record)
        return EffectDecision(receipt.status, reason, _put(state, record), record, receipt)
    if outcome == "none" and existing.effect in ("partial", "unknown"):
        # No *new* effect does not undo a previous helper write.
        outcome = existing.effect
    status, phase, reason = {
        "complete": ("accepted", "accepted", "accepted"),
        "partial": ("unconfirmed", "unknown", "effect_partial"),
        "unknown": ("unconfirmed", "unknown", "effect_unknown"),
        "none": ("rejected", "rejected", "no_effect"),
    }[outcome]
    record = replace(existing, phase=phase, reason=reason, effect=outcome,
                     accepted_ts=receiver_now_upper_ms if status == "accepted" else None,
                     observed_at_ms=receiver_now_upper_ms,
                     helper_results=() if status == "rejected" else existing.helper_results)
    return EffectDecision(status, reason, _put(state, record), record,
                          _receipt(state.installation_id, record))


def classify_helper(state: ReceiverState, command: BatteryCommand, *, receiver_now_upper_ms: int,
                    clock_healthy: bool, clock_uncertainty_ms: int | None,
                    result: HelperResult, acknowledged: bool = True) -> EffectDecision:
    """Record one last-writer observation; determine completion in production code.

    The environment must call fence_effect immediately before the device part,
    then feed its observation here without another intervening transition.
    """
    _helper(result)
    _boolean(acknowledged)
    allowed, _ = fence_effect(state, command, receiver_now_upper_ms=receiver_now_upper_ms,
                              clock_healthy=clock_healthy, clock_uncertainty_ms=clock_uncertainty_ms)
    if not allowed:
        return classify_effect(state, command, receiver_now_upper_ms=receiver_now_upper_ms, outcome="none",
                               clock_healthy=clock_healthy, clock_uncertainty_ms=clock_uncertainty_ms)
    record = _find(state, command.command_id)
    parts = ("setpoint", "mode", "stop_inverter")
    if len(record.helper_results) >= len(parts) or result.helper != parts[len(record.helper_results)]:
        raise ValueError("receiver_helper_order_invalid")
    helpers = record.helper_results + (result,)
    updated_record = replace(record, helper_results=helpers, effect="partial" if any(r.ok for r in helpers) else "none",
                             observed_at_ms=receiver_now_upper_ms)
    updated = _put(state, updated_record)
    if not result.ok:
        return classify_effect(updated, command, receiver_now_upper_ms=receiver_now_upper_ms,
                               clock_healthy=clock_healthy, clock_uncertainty_ms=clock_uncertainty_ms,
                               outcome="partial" if any(r.ok for r in helpers) else "none")
    if len(helpers) == len(parts):
        return classify_effect(updated, command, receiver_now_upper_ms=receiver_now_upper_ms,
                               clock_healthy=clock_healthy, clock_uncertainty_ms=clock_uncertainty_ms,
                               outcome="complete" if acknowledged else "unknown")
    receipt = _receipt(state.installation_id, updated_record)
    return EffectDecision("unconfirmed", "pending_effect", updated, updated_record, receipt)


def receiver_crash_state(state: ReceiverState, *, receiver_now_upper_ms: int) -> ReceiverState:
    """Persist ambiguity for a crash between helper writes; never replay on recovery."""
    _integer(receiver_now_upper_ms)
    records = tuple(replace(record, phase="unknown", reason="effect_unknown", effect="unknown",
                            accepted_ts=None, observed_at_ms=receiver_now_upper_ms)
                    if record.phase == "reserved" and record.helper_results else record
                    for record in state.commands)
    return replace(state, commands=records)


def classify_revoke(state: ReceiverState, request: RevokeRequest, *, receiver_now_upper_ms: int,
                    queues_drained: bool, safe_state_confirmed: bool) -> RevokeDecision:
    _integer(receiver_now_upper_ms)
    _boolean(queues_drained)
    _boolean(safe_state_confirmed)
    if state.epoch_high_water == 0:
        # F2c1 has no canonical barrier receipt for an unprovisioned epoch.
        # Refuse explicitly instead of inventing current_epoch=1.
        raise ValueError("epoch_mismatch")
    reason = None
    if request.expected_epoch != state.epoch_high_water:
        reason = "epoch_mismatch"
    elif request.installation_id != state.installation_id:
        reason = "wrong_installation"
    if reason is not None:
        receipt = BarrierReceipt(request.installation_id, request.barrier_id, request.expected_epoch,
                                 state.epoch_high_water, "rejected", None, False, "unknown",
                                 state.receiver_generation, reason)
        return RevokeDecision("rejected", reason, state, receipt)
    status = "accepted" if queues_drained and safe_state_confirmed else "unconfirmed"
    reason = "accepted" if status == "accepted" else "barrier_unconfirmed"
    receipt = BarrierReceipt(
        state.installation_id, request.barrier_id, request.expected_epoch, request.expected_epoch + 1,
        status, receiver_now_upper_ms if status == "accepted" else None, queues_drained,
        "confirmed" if safe_state_confirmed else "unknown", state.receiver_generation, reason,
    )
    records = tuple(replace(record, phase="invalidated", reason="barrier_epoch_withdrawn")
                    if record.phase == "reserved" else record for record in state.commands)
    updated = replace(state, epoch_high_water=request.expected_epoch + 1, inhibited=True,
                      bound_owner_session_id=None, commands=records, pending_barrier_id=request.barrier_id,
                      last_barrier=receipt)
    return RevokeDecision(status, reason, updated, receipt)


def classify_bind(state: ReceiverState, *, installation_id: str, barrier_id: str, expected_epoch: int,
                  owner_session_id: str, requested_not_after_ms: int, receiver_max_lease_ms: int,
                  receiver_now_upper_ms: int) -> BindDecision:
    if state.epoch_high_water == 0:
        return BindDecision("rejected", "no_confirmed_barrier", state, None)
    checks = (
        (installation_id != state.installation_id, "wrong_installation"),
        (state.last_barrier is None or state.last_barrier.status != "accepted", "no_confirmed_barrier"),
        (state.last_barrier is not None and barrier_id != state.last_barrier.barrier_id, "barrier_mismatch"),
        (expected_epoch != state.epoch_high_water, "epoch_mismatch"),
        (state.last_barrier is not None and expected_epoch != state.last_barrier.current_epoch, "barrier_epoch_mismatch"),
        (type(owner_session_id) is not str or not owner_session_id.strip(), "invalid_owner_session"),
        (state.bound_owner_epoch == expected_epoch, "barrier_already_bound"),
    )
    reason = next((reason for refused, reason in checks if refused), None)
    if reason is not None:
        return BindDecision("rejected", reason, state, None)
    for value in (expected_epoch, requested_not_after_ms, receiver_max_lease_ms, receiver_now_upper_ms):
        _integer(value)
    not_after = min(requested_not_after_ms, receiver_now_upper_ms + receiver_max_lease_ms)
    binding = OwnerBinding(installation_id, state.receiver_id, state.receiver_generation, owner_session_id,
                           expected_epoch, barrier_id, not_after)
    updated = replace(state, bound_owner_session_id=owner_session_id, bound_owner_epoch=expected_epoch,
                      owner_not_after_ms=not_after, inhibited=False)
    return BindDecision("accepted", "accepted", updated, binding)


def classify_query(state: ReceiverState, *, installation_id: str, owner_epoch: int, sequence: int,
                   command_id: str, payload_digest: str) -> ActuationReceipt | None:
    record = _find(state, command_id)
    # Even a wrong-installation query cannot fabricate identity for an unknown id.
    if record is None:
        return None
    if installation_id != state.installation_id:
        return _receipt(state.installation_id, record, reason="wrong_installation")
    if (type(owner_epoch) is not int or type(sequence) is not int or owner_epoch != record.owner_epoch
            or sequence != record.sequence or payload_digest != record.payload_digest):
        return _receipt(state.installation_id, record, reason="identity_mismatch")
    return _receipt(state.installation_id, record)


def receiver_recovery_state(loaded: ReceiverState | None, *, anchor_epoch_high_water: int,
                             anchor_sequence_high_water: int, anchor_checkpoint: str) -> tuple[ReceiverState, str]:
    _integer(anchor_epoch_high_water)
    _integer(anchor_sequence_high_water)
    _string(anchor_checkpoint)
    if loaded is None:
        # Identity is unavailable, too. These explicit unknown labels grant no right;
        # only the independent anchor supplies high-waters, never a bootstrap epoch.
        blocked = ReceiverState(1, "unknown", "unknown", "unknown", anchor_checkpoint,
                                anchor_epoch_high_water, anchor_sequence_high_water, None, 0, 0, (),
                                True, False, None, None, "unknown")
        return blocked, "state_missing_or_corrupt"
    reason = None
    if loaded.anchor_checkpoint != anchor_checkpoint:
        reason = "anchor_checkpoint_mismatch"
    elif loaded.epoch_high_water < anchor_epoch_high_water or loaded.sequence_high_water < anchor_sequence_high_water:
        reason = "rolled_back_state"
    if reason is not None:
        records = tuple(replace(record, phase="invalidated", reason=reason)
                        if record.phase == "reserved" else record for record in loaded.commands)
        return replace(loaded, inhibited=True, queue_known=False, bound_owner_session_id=None,
                       bound_owner_epoch=0, owner_not_after_ms=0, last_barrier=None, pending_barrier_id=None,
                       commands=records, epoch_high_water=anchor_epoch_high_water,
                       sequence_high_water=anchor_sequence_high_water,
                       anchor_checkpoint=anchor_checkpoint), reason
    return loaded, "recovered"
