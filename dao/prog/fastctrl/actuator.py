"""Bounded helper acceptance, separate from proposals and physical measurements.

Confirmation means only that the required helpers accepted their values. This
module neither grants execution rights nor touches controller state.

The financial route adds exactly two things here: the **nominal** adapter
classification (who may carry financial command rights at all) and the explicit
machine-readable verdict of an actuation attempt. Neither is inferred from
duck typing, a boolean or a configuration field.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .command import ActuationReceipt, HelperResult, _boolean, _reason
from .plan import NO_STOP_SENTINEL, FastPlan
from .policy import BatteryDecision

#: The generic Home Assistant gateway: it may never carry financial command
#: rights, not even through a canonical actuator call.
LEGACY_ONLY_ADAPTER = "legacy_only"

#: A nominally qualified adapter: the fenced receiver contract.
FENCED_ADAPTER = "fenced"


class FinancialActuationAdapter:
    """Nominal opt-in for adapters that may carry financial command rights.

    Qualification happens by inheriting this class only. It is never
    inferred from the presence of methods, never a boolean and never a
    configuration field.
    """


def classify_actuation_adapter(gateway) -> str:
    """Classify a battery-command adapter; the generic gateway is legacy_only.

    Only a real base class counts, never a structural check: an object that
    merely has the right methods is still ``legacy_only``. A foreign object
    never raises here; it is simply not qualified.
    """
    if isinstance(gateway, FinancialActuationAdapter):
        return FENCED_ADAPTER
    return LEGACY_ONLY_ADAPTER


@dataclass(frozen=True)
class ActuationVerdict:
    """The explicit outcome of an actuation attempt: allowed or refused."""

    allowed: bool
    #: Machine-readable, ``^[a-z][a-z0-9_]*$``.
    reason: str

    def __post_init__(self) -> None:
        _boolean(self.allowed, "actuation_verdict_allowed_invalid")
        _reason(self.reason, "actuation_verdict_reason_invalid")


@dataclass(frozen=True)
class LegacyActuationReceipt:
    """Helper acceptance only: no measured power, SOC or physical-stop claim."""

    command_id: str
    plan_identity: str
    timestamp: float
    battery_index: int
    battery_name: str
    setpoint_w: float
    status: Literal["confirmed", "unconfirmed", "unknown"]
    helpers: tuple[HelperResult, ...]

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in ("confirmed", "unconfirmed", "unknown"):
            raise ValueError("legacy_receipt_status_invalid")


CommandReceipt = ActuationReceipt | LegacyActuationReceipt


def plan_identity(plan: FastPlan) -> str:
    identity = f"v{plan.format_version}:{plan.created_ts}"
    if plan.format_version == 2:
        context = plan.financial_context
        replay_hash = getattr(context, "replay_basis_hash", None)
        if not isinstance(replay_hash, str) or not replay_hash.strip():
            raise ValueError("actuation_identity_invalid: v2 replay basis required")
        identity += f":{replay_hash}"
    return identity


def _execute_helper(
    gateway, helper, entity_id: str, intended, write: Callable[[], bool],
) -> HelperResult:
    # Two attempts total, then exactly one fresh read. No polling or hidden
    # controller bookkeeping; an unreadable value is never an acknowledgement.
    for attempt in range(1, 3):
        try:
            if write() is True:
                return HelperResult(helper, entity_id, True,
                                    None if attempt == 1 else "write_failed", attempt)
        except Exception:  # noqa: BLE001 - failed transport is not acceptance
            pass
    try:
        raw = gateway.read([entity_id]).get(entity_id, (None, 0.0))[0]
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in
                           ("", "unknown", "unavailable", "none", "null")):
            return HelperResult(helper, entity_id, False, "readback_unreadable", 2)
        if helper == "setpoint":
            actual = float(raw)
            if isinstance(raw, bool) or not math.isfinite(actual):
                return HelperResult(helper, entity_id, False, "readback_unreadable", 2)
            matches = actual == intended
        elif helper == "stop_inverter":
            matches = (
                isinstance(raw, str)
                and datetime.fromisoformat(raw) == datetime.fromisoformat(intended)
            )
        else:
            matches = isinstance(raw, str) and raw == intended
        if matches:
            return HelperResult(helper, entity_id, True, "write_failed", 2)
    except Exception:  # noqa: BLE001 - read-back must fail closed
        return HelperResult(helper, entity_id, False, "readback_unreadable", 2)
    return HelperResult(helper, entity_id, False, "readback_mismatch", 2)


def execute_battery_command(
    gateway, plan: FastPlan, battery: BatteryDecision, timestamp: float,
) -> LegacyActuationReceipt | None:
    """Execute one proposal in setpoint → mode → stop order, non-atomically."""
    if not battery.write:
        return None
    spec = plan.spec(battery.index)
    if spec is None:
        return None
    identity = plan_identity(plan)
    helpers = []
    if spec.setpoint_entity:
        helpers.append(_execute_helper(
            gateway, "setpoint", spec.setpoint_entity, battery.setpoint_w,
            lambda: gateway.write_number(spec.setpoint_entity, battery.setpoint_w),
        ))
    if battery.mode is not None and spec.mode_entity:
        helpers.append(_execute_helper(
            gateway, "mode", spec.mode_entity, battery.mode,
            lambda: gateway.select_option(spec.mode_entity, battery.mode),
        ))
    if battery.stop_inverter is not None and spec.stop_inverter_entity:
        value = battery.stop_inverter or NO_STOP_SENTINEL
        helpers.append(_execute_helper(
            gateway, "stop_inverter", spec.stop_inverter_entity, value,
            lambda: gateway.set_datetime(spec.stop_inverter_entity, value),
        ))
    if not helpers:
        return None
    successes = sum(result.ok for result in helpers)
    status = (
        "confirmed" if successes == len(helpers) else ("unknown" if successes else "unconfirmed")
    )
    return LegacyActuationReceipt(
        command_id=f"{identity}/b{battery.index}/{int(timestamp)}",
        plan_identity=identity,
        timestamp=timestamp,
        battery_index=battery.index,
        battery_name=battery.name,
        setpoint_w=battery.setpoint_w,
        status=status,
        helpers=tuple(helpers),
    )
