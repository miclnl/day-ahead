"""Pure command/evidence contracts, not an execution or ownership boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Literal


def _string(value, code: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(code)


def _integer(value, code: str, minimum: int | None = 0) -> None:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(code)


def _boolean(value, code: str) -> None:
    if type(value) is not bool:
        raise ValueError(code)


def _choice(value, choices: tuple[str, ...], code: str) -> None:
    if type(value) is not str or value not in choices:
        raise ValueError(code)


def _digest(value, code: str) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(code)


def _reason(value, code: str) -> None:
    if (type(value) is not str or not value or value[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in value)):
        raise ValueError(code)


def _acceptance(status, accepted_ts, code: str) -> None:
    _choice(status, ("accepted", "rejected", "unconfirmed"), code)
    if accepted_ts is not None:
        _integer(accepted_ts, code)
    if (accepted_ts is not None) != (status == "accepted"):
        raise ValueError(code)


@dataclass(frozen=True)
class HelperResult:
    helper: Literal["setpoint", "mode", "stop_inverter"]
    entity_id: str
    ok: bool
    error: Literal["write_failed", "readback_unreadable", "readback_mismatch"] | None
    attempts: int


@dataclass(frozen=True, slots=True)
class BatteryCommand:
    installation_id: str
    battery_name: str
    owner_session_id: str
    owner_epoch: int
    sequence: int
    command_id: str
    plan_digest: str
    issued_at_ms: int
    execute_before_ms: int
    hold_until_ms: int
    setpoint_w: int
    mode: str
    stop_at_ms: int | None

    def __post_init__(self) -> None:
        for name in ("installation_id", "battery_name", "owner_session_id", "command_id", "mode"):
            _string(getattr(self, name), f"command_{name}_invalid")
        for name in ("owner_epoch", "sequence"):
            _integer(getattr(self, name), f"command_{name}_invalid", 1)
        _digest(self.plan_digest, "command_plan_digest_invalid")
        for name in ("issued_at_ms", "execute_before_ms", "hold_until_ms"):
            _integer(getattr(self, name), f"command_{name}_invalid")
        _integer(self.setpoint_w, "command_setpoint_w_invalid", None)
        if self.stop_at_ms is not None:
            _integer(self.stop_at_ms, "command_stop_at_ms_invalid")
        if not self.issued_at_ms < self.execute_before_ms <= self.hold_until_ms:
            raise ValueError("command_time_order_invalid")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BatteryCommand:
        if type(data) is not dict or data.keys() != {item.name for item in fields(cls)}:
            raise ValueError("command_fields_invalid")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ActuationReceipt:
    """Acceptance at the last-writer boundary, never a physical measurement."""

    installation_id: str
    battery_name: str
    owner_epoch: int
    sequence: int
    command_id: str
    plan_digest: str
    accepted_ts: int | None
    status: Literal["accepted", "rejected", "unconfirmed"]
    helper_results: tuple[HelperResult, ...]
    owner_session_id: str
    payload_digest: str
    reason: str
    observed_at_ms: int
    effect: Literal["none", "complete", "partial", "unknown"]

    def __post_init__(self) -> None:
        for name in ("installation_id", "battery_name", "command_id", "owner_session_id"):
            _string(getattr(self, name), f"receipt_{name}_invalid")
        for name in ("owner_epoch", "sequence"):
            _integer(getattr(self, name), f"receipt_{name}_invalid", 1)
        for name in ("plan_digest", "payload_digest"):
            _digest(getattr(self, name), f"receipt_{name}_invalid")
        _reason(self.reason, "receipt_reason_invalid")
        _integer(self.observed_at_ms, "receipt_observed_at_ms_invalid")
        _acceptance(self.status, self.accepted_ts, "receipt_acceptance_invalid")
        _choice(self.effect, ("none", "complete", "partial", "unknown"), "receipt_effect_invalid")
        if type(self.helper_results) is not tuple:
            raise ValueError("receipt_helpers_invalid")
        for result in self.helper_results:
            if type(result) is not HelperResult:
                raise ValueError("receipt_helpers_invalid")
            _boolean(result.ok, "receipt_helper_ok_invalid")
        if self.status == "accepted":
            if self.effect != "complete" or not all(result.ok for result in self.helper_results):
                raise ValueError("receipt_acceptance_incomplete")
        elif self.status == "rejected":
            if self.effect != "none" or self.helper_results != ():
                raise ValueError("receipt_rejection_effect_invalid")
        elif self.effect == "complete":
            raise ValueError("receipt_unconfirmed_effect_invalid")


@dataclass(frozen=True, slots=True)
class RevokeRequest:
    installation_id: str
    expected_epoch: int
    barrier_id: str
    reason: str
    requested_at_ms: int

    def __post_init__(self) -> None:
        _string(self.installation_id, "revoke_installation_id_invalid")
        _integer(self.expected_epoch, "revoke_expected_epoch_invalid", 1)
        _string(self.barrier_id, "revoke_barrier_id_invalid")
        _reason(self.reason, "revoke_reason_invalid")
        _integer(self.requested_at_ms, "revoke_requested_at_ms_invalid")


@dataclass(frozen=True, slots=True)
class BarrierReceipt:
    installation_id: str
    barrier_id: str
    revoked_through_epoch: int
    current_epoch: int
    status: Literal["accepted", "rejected", "unconfirmed"]
    accepted_ts: int | None
    queues_fenced: bool
    safe_state: Literal["confirmed", "unknown"]
    receiver_generation: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("installation_id", "barrier_id", "receiver_generation"):
            _string(getattr(self, name), f"barrier_{name}_invalid")
        _integer(self.revoked_through_epoch, "barrier_revoked_through_epoch_invalid")
        _integer(self.current_epoch, "barrier_current_epoch_invalid", 1)
        _acceptance(self.status, self.accepted_ts, "barrier_acceptance_invalid")
        _boolean(self.queues_fenced, "barrier_queues_fenced_invalid")
        _choice(self.safe_state, ("confirmed", "unknown"), "barrier_safe_state_invalid")
        _reason(self.reason, "barrier_reason_invalid")
        if self.status == "accepted" and (not self.queues_fenced or self.safe_state != "confirmed"):
            raise ValueError("barrier_acceptance_incomplete")


@dataclass(frozen=True, slots=True)
class Deadline:
    monotonic_end_ns: int
    execute_before_ms: int

    def __post_init__(self) -> None:
        _integer(self.monotonic_end_ns, "deadline_monotonic_end_ns_invalid")
        _integer(self.execute_before_ms, "deadline_execute_before_ms_invalid")

    def remaining_ns(self, now_ns: int) -> int:
        _integer(now_ns, "deadline_now_ns_invalid", None)
        return max(0, self.monotonic_end_ns - now_ns)


@dataclass(frozen=True, slots=True)
class ClockSample:
    utc_ms: int
    boottime_ns: int
    boot_id: str
    uncertainty_ms: int
    healthy: bool

    def __post_init__(self) -> None:
        for name in ("utc_ms", "boottime_ns", "uncertainty_ms"):
            _integer(getattr(self, name), f"clock_{name}_invalid")
        _string(self.boot_id, "clock_boot_id_invalid")
        _boolean(self.healthy, "clock_healthy_invalid")


@dataclass(frozen=True, slots=True)
class OwnerBinding:
    installation_id: str
    receiver_id: str
    receiver_generation: str
    owner_session_id: str
    owner_epoch: int
    barrier_id: str
    owner_not_after_ms: int

    def __post_init__(self) -> None:
        for name in ("installation_id", "receiver_id", "receiver_generation", "owner_session_id", "barrier_id"):
            _string(getattr(self, name), f"owner_{name}_invalid")
        _integer(self.owner_epoch, "owner_epoch_invalid", 1)
        _integer(self.owner_not_after_ms, "owner_not_after_ms_invalid")


def plan_digest(plan) -> str:
    """Hash the complete validated snapshot; leave plan validation errors intact."""
    snapshot = plan.to_dict()
    try:
        payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("plan_snapshot_invalid") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
