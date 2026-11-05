"""The fast control decision logic.

This module is deliberately free of Home Assistant, database and configuration
dependencies. It is a pure function of (plan, measurement, controller state)
so that the production loop, the unit tests and the historical backtest all
execute exactly the same code path.

The control law
---------------

Every tick the controller reconstructs the true house load from the meter::

    house = grid_measured - sum(battery_measured)

and then minimises the instantaneous cost rate over the battery setpoint
``p`` of one battery at a time::

    g(p)    = house + p                                 net grid power, + = import
    C(p)    =  price_import * max(g, 0) / 1000          cost of importing
             - price_export * max(-g, 0) / 1000         revenue from exporting
             - storage_value * eta_rt * max(p, 0) / 1000    value of energy stored
             - storage_value        * min(p, 0) / 1000      value of energy spent
             + cycle_cost * abs(p) / 1000               wear

``C`` is convex and piecewise linear in ``p`` as long as
``price_import >= price_export``, with breakpoints only at ``p = 0`` (the wear
and storage kinks) and ``p = -house`` (the grid kink). Its minimum over a
closed interval is therefore attained at one of four candidate points, which
makes the solve exact, branch free and trivially testable.

Why this is not just a self-consumption controller
--------------------------------------------------

``storage_value`` is the marginal worth of a kWh in the battery, derived from
the remaining day-ahead plan. It is the Lagrange multiplier that couples the
single-period problem solved here to the multi-period problem solved by the
optimizer. With a correct multiplier the myopic solution *is* the optimal
solution. In the ordinary price regime
``price_export < storage_value < price_import`` the minimiser lands on
``g = 0``, which is plain self-consumption. During negative prices or a price
spike the same formula flips the behaviour without any special casing.

Because the multiplier is only an estimate, every correction is additionally
confined to an energy trust region around the planned state of charge, and to
a daily wear budget. The plan setpoint is always inside the feasible set, so
the layer can never do worse than doing nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .command import ActuationReceipt, BatteryCommand
from .plan import (
    NO_STOP_SENTINEL,
    BatterySpec,
    FastPlan,
    PlanInterval,
    profile_energy_kwh,
    profile_power_at,
)
from .receiver import receipt_matches_command

#: Fraction of the energy budget over which the allowed power is tapered to
#: zero, instead of dropping to zero in one step.
BUDGET_TAPER_FRACTION = 0.25

#: State of charge band in percent over which the allowed power is tapered to
#: zero as the battery approaches its working limits.
SOC_TAPER_BAND = 3.0

#: Fraction of the capacity over which stored energy is blended from "scarce"
#: to "not needed by the plan" when valuing it.
SCARCITY_BAND_FRACTION = 0.10

#: Setpoints closer to each other than this are treated as identical.
POWER_EPS = 1.0

#: The execution outcomes the financial route can report.
EXECUTION_OUTCOMES = ("unknown", "accepted", "unconfirmed", "stopped")


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return max(low, min(high, value))


def _taper(remaining: float, band: float) -> float:
    """Linear 0..1 ramp used to soften every budget and limit."""
    if remaining <= 0.0:
        return 0.0
    if band <= 0.0:
        return 1.0
    return min(1.0, remaining / band)


@dataclass
class PolicyLimits:
    """Tunables handed to the policy, mirrored from ``FastControlConfig``.

    Kept as a plain dataclass so the policy has no configuration dependency and
    the backtest can sweep parameters without building a Pydantic model.
    """

    storage_value_mode: str = "plan"
    storage_value_fixed: Optional[float] = None
    round_trip_efficiency: float = 0.90
    min_benefit: float = 0.02
    deadband: float = 150.0
    min_command_interval: float = 60.0
    urgent_deviation: float = 1500.0
    max_ramp: Optional[float] = None
    release_deviation: float = 100.0
    release_time: float = 120.0
    energy_budget: float = 0.5
    daily_extra_throughput: float = 4.0
    soc_margin: float = 2.0
    max_grid_import: Optional[float] = None
    allow_grid_charge: bool = False


@dataclass
class BatteryMeasurement:
    """What the controller knows about one battery right now.

    ``power_w`` is optional because not every installation has a sensor for it;
    the controller then falls back on its own last command. ``soc`` is not
    optional in practice: without it the state of charge guard cannot be
    applied, so the controller refuses to override that battery.
    """

    soc: Optional[float] = None
    #: Measured AC power, positive is charging. None when no sensor is wired.
    power_w: Optional[float] = None
    #: False when the power reading is present but stale.
    valid: bool = True


@dataclass
class Measurement:
    """One sample of the plant."""

    timestamp: float
    #: Net grid power in W, positive is import.
    grid_w: float
    batteries: list[BatteryMeasurement] = field(default_factory=list)
    #: Optional, diagnostics only. The control law does not use it because PV
    #: is already contained in the grid measurement.
    pv_w: Optional[float] = None
    #: False when the grid reading is missing or stale, or the plan itself
    #: is too old to be trusted.
    grid_valid: bool = True
    #: Why grid_valid is False, surfaced as Decision.reason. The runner sets
    #: this to "plan_stale" when the plan (not the sensor) is the reason.
    invalid_reason: str = "sensor_stale"

    def battery(self, index: int) -> BatteryMeasurement:
        if 0 <= index < len(self.batteries):
            return self.batteries[index]
        return BatteryMeasurement(valid=False)


@dataclass
class BatteryControllerState:
    """Per battery state that must survive between ticks and across restarts."""

    last_command_w: float = 0.0
    last_command_ts: float = 0.0
    last_change_ts: float = 0.0
    override_active: bool = False
    quiet_since: Optional[float] = None
    #: Signed energy in kWh moved more (+) or less (-) than the plan intended,
    #: accumulated over the current plan interval.
    interval_deviation_kwh: float = 0.0
    #: Absolute deviation throughput in kWh accumulated since midnight.
    daily_deviation_kwh: float = 0.0
    #: Whether the inverter stop moment has been cleared by this layer.
    stop_inverter_cleared: bool = False
    #: Setpoint currently considered in force, including plan reconciliation.
    #: Unlike last_command_w, this is not a helper-acceptance fact.
    held_setpoint_w: float = 0.0
    #: In-force stop ownership, projected in shadow and accepted in active mode.
    held_stop_inverter_cleared: bool = False
    #: A proposed optimizer restore remains due until explicitly confirmed.
    restore_pending: bool = False
    #: Last issued canonical command, cached for exact receipt matching only.
    pending_command: Optional[BatteryCommand] = None
    #: The last known execution outcome of the financial route.
    execution_status: str = "unknown"
    #: There is an unfinished recovery duty: an uncertain or unproven execution.
    recovery_required: bool = False
    #: The identity of the last **confirmed** command.
    last_accepted_command: Optional[BatteryCommand] = None

    def to_dict(self) -> dict:
        return {
            "last_command_w": self.last_command_w,
            "last_command_ts": self.last_command_ts,
            "last_change_ts": self.last_change_ts,
            "override_active": self.override_active,
            "quiet_since": self.quiet_since,
            "interval_deviation_kwh": self.interval_deviation_kwh,
            "daily_deviation_kwh": self.daily_deviation_kwh,
            "stop_inverter_cleared": self.stop_inverter_cleared,
            "held_setpoint_w": self.held_setpoint_w,
            "held_stop_inverter_cleared": self.held_stop_inverter_cleared,
            "restore_pending": self.restore_pending,
            "pending_command": self.pending_command.to_dict() if self.pending_command is not None else None,
            "execution_status": self.execution_status,
            "recovery_required": self.recovery_required,
            "last_accepted_command": (
                self.last_accepted_command.to_dict()
                if self.last_accepted_command is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatteryControllerState":
        state = cls()
        for key, value in (data or {}).items():
            if key == "pending_command" and value is not None:
                value = BatteryCommand.from_dict(value)
            if key == "last_accepted_command" and value is not None:
                value = BatteryCommand.from_dict(value)
            if hasattr(state, key):
                setattr(state, key, value)
        return state


@dataclass
class ControllerState:
    """Controller state shared by all batteries."""

    batteries: list[BatteryControllerState] = field(default_factory=list)
    #: ``created_ts`` of the plan the current interval accounting belongs to.
    plan_created_ts: int = 0
    #: ``start_ts`` of the plan interval the current accounting belongs to.
    interval_start_ts: int = 0
    #: Local date the daily budgets belong to, as ``YYYY-MM-DD``.
    day_key: str = ""
    #: Estimated saving accumulated since midnight, in euro.
    saved_today_eur: float = 0.0
    #: True once any part of today's saved_today_eur came from a tick where
    #: the layer was not actually writing (shadow or off): the figure is
    #: then a projection, not a measured saving, for at least part of today.
    saved_today_is_estimate: bool = False
    last_tick_ts: float = 0.0
    #: Snapshot of the most recent :class:`Decision` attributes, or ``None``
    #: before the first tick. Used to surface the layer's last verdict to the
    #: web UI without recomputing it.
    last_decision: Optional[dict] = None
    #: Rolling log of notable state transitions (override start, release,
    #: budget exhausted, ...). Newest entries are appended at the end; the
    #: runner trims this list to a bounded length on write.
    events: list[dict] = field(default_factory=list)
    #: Sum of all batteries' ``daily_deviation_kwh``. Refreshed by the runner
    #: on every tick so the web UI can surface it without iterating over the
    #: per-battery list.
    daily_extra_throughput_used: float = 0.0
    #: Sum of ``abs(interval_deviation_kwh)`` across all batteries. Counts both
    #: the charge and the discharge side of an override cycle so the budget
    #: cannot be silently exhausted by oscillating around zero.
    energy_budget_used: float = 0.0

    def battery(self, index: int) -> BatteryControllerState:
        while len(self.batteries) <= index:
            self.batteries.append(BatteryControllerState())
        return self.batteries[index]

    def refresh_budget_aggregates(self) -> None:
        """Roll the per-battery deviation totals into the top-level budget fields.

        The web UI reads ``daily_extra_throughput_used`` and
        ``energy_budget_used`` directly so it can render the daily and interval
        budget gauges without iterating over every battery. The runner calls
        this once per tick, after the policy has finished updating each
        battery's deviation counters.
        """
        self.daily_extra_throughput_used = sum(
            b.daily_deviation_kwh for b in self.batteries
        )
        self.energy_budget_used = sum(
            abs(b.interval_deviation_kwh) for b in self.batteries
        )

    def to_dict(self) -> dict:
        return {
            "batteries": [b.to_dict() for b in self.batteries],
            "plan_created_ts": self.plan_created_ts,
            "interval_start_ts": self.interval_start_ts,
            "day_key": self.day_key,
            "saved_today_eur": self.saved_today_eur,
            "saved_today_is_estimate": self.saved_today_is_estimate,
            "last_tick_ts": self.last_tick_ts,
            "last_decision": self.last_decision,
            "events": list(self.events),
            "daily_extra_throughput_used": self.daily_extra_throughput_used,
            "energy_budget_used": self.energy_budget_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ControllerState":
        data = data or {}
        return cls(
            batteries=[
                BatteryControllerState.from_dict(b) for b in data.get("batteries", [])
            ],
            plan_created_ts=int(data.get("plan_created_ts", 0)),
            interval_start_ts=int(data.get("interval_start_ts", 0)),
            day_key=str(data.get("day_key", "")),
            saved_today_eur=float(data.get("saved_today_eur", 0.0)),
            saved_today_is_estimate=bool(data.get("saved_today_is_estimate", False)),
            last_tick_ts=float(data.get("last_tick_ts", 0.0)),
            last_decision=data.get("last_decision"),
            events=list(data.get("events", [])),
            daily_extra_throughput_used=float(
                data.get("daily_extra_throughput_used", 0.0)
            ),
            energy_budget_used=float(data.get("energy_budget_used", 0.0)),
        )


@dataclass(frozen=True)
class AcceptedCommand:
    """Proposed bookkeeping, applied only after helper acceptance."""

    last_command_w: float
    last_command_ts: float
    stop_inverter_cleared: bool


@dataclass
class BatteryDecision:
    """The controller's verdict for one battery."""

    index: int
    name: str
    #: Setpoint to write, in W, positive is charging.
    setpoint_w: float
    #: What the day-ahead plan wanted for this interval.
    plan_w: float
    #: Whether the setpoint must actually be written this tick.
    write: bool
    #: True while the layer is deviating from the plan.
    override: bool
    reason: str
    benefit_eur_h: float = 0.0
    storage_value: float = 0.0
    #: Operating mode to write, None means leave alone.
    mode: Optional[str] = None
    #: Inverter stop moment to write, None means leave alone.
    stop_inverter: Optional[str] = None
    soc: Optional[float] = None
    budget_used_kwh: float = 0.0
    daily_used_kwh: float = 0.0
    accepted: AcceptedCommand | None = None
    #: This tick's helper acceptance, not a physical response or execution right.
    actuation_status: str | None = None
    #: This command restores the optimizer's mode/stop, rather than clearing it.
    restores: bool = False


def apply_actuation_receipt(state: BatteryControllerState, receipt: ActuationReceipt) -> None:
    """Apply exact canonical acceptance from the cached command, without I/O."""
    if type(receipt) is not ActuationReceipt:
        raise TypeError("canonical_receipt_required")
    pending = state.pending_command
    if pending is None:
        return
    for name in ("installation_id", "battery_name", "owner_epoch", "sequence", "command_id", "plan_digest"):
        if getattr(receipt, name) != getattr(pending, name):
            return
    if receipt.status != "accepted":
        return
    state.last_command_w = float(pending.setpoint_w)
    state.last_command_ts = receipt.accepted_ts / 1000.0
    state.stop_inverter_cleared = pending.stop_at_ms is None
    state.held_setpoint_w = float(pending.setpoint_w)
    state.held_stop_inverter_cleared = state.stop_inverter_cleared
    state.pending_command = None


def apply_execution_outcome(state: BatteryControllerState, *,
                             outcome: str,
                             receipt: ActuationReceipt | None = None,
                             pending: BatteryCommand | None = None) -> bool:
    """Fold one financial execution outcome into the controller state.

    Pure: no I/O, no clock and no plan. The decision table is binding:

    ============  =================  ==================  =================  ====================
    outcome       execution_status   recovery_required  pending_command    last_accepted_command
    ============  =================  ==================  =================  ====================
    unknown       unknown            unchanged          unchanged          unchanged
    accepted      accepted           unchanged          cleared            the stored command
    unconfirmed   unconfirmed        True               stays standing     unchanged
    stopped       stopped            False              cleared            unchanged
    ============  =================  ==================  =================  ====================

    The in-force facts (``held_setpoint_w``, ``held_stop_inverter_cleared`` and
    ``last_change_ts``) are never touched here: they belong to the in-force
    reality, not to a receipt. Only an exactly matching receipt proves an
    acceptance; a receipt that does not match the stored pending command
    field by field mutates nothing at all.

    The return value says whether the outcome was applied. Only an applied
    ``accepted`` may unlock the in-force facts in the runner.
    """
    if outcome not in EXECUTION_OUTCOMES:
        raise ValueError("execution_outcome_invalid")
    if receipt is not None and type(receipt) is not ActuationReceipt:
        raise TypeError("canonical_receipt_required")
    if outcome == "unknown":
        # No execution: no owner, an unsuitable adapter, a gate or a proven
        # absence of effect. The recovery duty and the pending command stay.
        state.execution_status = "unknown"
        return True
    if outcome == "stopped":
        # An exact safe barrier ends the recovery duty; nothing else does.
        state.execution_status = "stopped"
        state.recovery_required = False
        state.pending_command = None
        return True
    if outcome == "unconfirmed":
        # Ambiguous: the pending command, the recovery duty and the uncertainty
        # all stay standing until an exact safe barrier.
        state.execution_status = "unconfirmed"
        state.recovery_required = True
        if state.pending_command is None and pending is not None:
            state.pending_command = pending
        return True
    # accepted: only an exactly matching receipt proves it.
    stored = pending if pending is not None else state.pending_command
    if stored is None or receipt is None or receipt.status != "accepted":
        return False
    if not receipt_matches_command(receipt, stored):
        return False
    state.execution_status = "accepted"
    state.pending_command = None
    state.last_accepted_command = stored
    state.last_command_w = float(stored.setpoint_w)
    state.last_command_ts = receipt.accepted_ts / 1000.0
    state.stop_inverter_cleared = stored.stop_at_ms is None
    return True


def apply_accepted_command(controller: BatteryControllerState, decision: BatteryDecision) -> None:
    """Apply the proposal without I/O; the caller owns confirmation."""
    accepted = decision.accepted
    if accepted is None:
        return
    controller.last_command_w = accepted.last_command_w
    controller.last_command_ts = accepted.last_command_ts
    controller.stop_inverter_cleared = accepted.stop_inverter_cleared
    controller.held_setpoint_w = accepted.last_command_w
    controller.held_stop_inverter_cleared = accepted.stop_inverter_cleared


@dataclass
class Decision:
    """The controller's verdict for the whole site."""

    timestamp: float
    batteries: list[BatteryDecision] = field(default_factory=list)
    house_w: float = 0.0
    grid_w: float = 0.0
    plan_grid_w: float = 0.0
    deviation_w: float = 0.0
    price_import: float = 0.0
    price_export: float = 0.0
    reason: str = "plan"
    benefit_eur_h: float = 0.0

    @property
    def override(self) -> bool:
        return any(b.override for b in self.batteries)

    @property
    def writes(self) -> list[BatteryDecision]:
        return [b for b in self.batteries if b.write]

    def as_attributes(self) -> dict:
        """Compact payload suitable for a Home Assistant state attribute set."""
        return {
            "reason": self.reason,
            "override": self.override,
            "house_w": round(self.house_w),
            "grid_w": round(self.grid_w),
            "plan_grid_w": round(self.plan_grid_w),
            "deviation_w": round(self.deviation_w),
            "price_import": round(self.price_import, 4),
            "price_export": round(self.price_export, 4),
            "benefit_eur_h": round(self.benefit_eur_h, 4),
            "batteries": [
                {
                    "name": b.name,
                    "setpoint_w": round(b.setpoint_w),
                    "plan_w": round(b.plan_w),
                    "override": b.override,
                    "reason": b.reason,
                    "soc": b.soc,
                    "storage_value": round(b.storage_value, 4),
                    "budget_used_kwh": round(b.budget_used_kwh, 3),
                    "daily_used_kwh": round(b.daily_used_kwh, 3),
                    "actuation_status": b.actuation_status,
                }
                for b in self.batteries
            ],
        }


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Instantaneous cost rate of one battery setpoint, in euro per hour."""

    house_w: float
    price_import: float
    price_export: float
    storage_value: float
    cycle_cost: float
    round_trip_efficiency: float

    def __call__(self, power_w: float) -> float:
        grid_w = self.house_w + power_w
        grid = (
            self.price_import * max(grid_w, 0.0)
            - self.price_export * max(-grid_w, 0.0)
        ) / 1000.0
        if power_w >= 0.0:
            stored = -self.storage_value * self.round_trip_efficiency * power_w / 1000.0
        else:
            stored = -self.storage_value * power_w / 1000.0
        wear = self.cycle_cost * abs(power_w) / 1000.0
        return grid + stored + wear

    def minimise(self, low: float, high: float) -> float:
        """Exact minimiser over the closed interval ``[low, high]``.

        The cost is convex and piecewise linear with breakpoints at ``p = 0``
        and ``p = -house``, so checking the two endpoints and the two clamped
        breakpoints is sufficient.
        """
        if high < low:
            return low
        candidates = {low, high, _clamp(0.0, low, high), _clamp(-self.house_w, low, high)}
        return min(candidates, key=lambda p: (self(p), abs(p)))


# ---------------------------------------------------------------------------
# storage valuation
# ---------------------------------------------------------------------------


def estimate_storage_value(
    plan: FastPlan,
    now_ts: float,
    battery_index: int,
    limits: PolicyLimits,
) -> float:
    """Marginal value of one kWh of AC-deliverable energy in the battery.

    This is the multiplier that couples the single-period problem the fast
    layer solves to the multi-period problem the optimizer solves. ``plan``
    mode reads it off the remaining day-ahead plan, in two steps.

    **What the energy is worth while the plan still needs it.** Between now and
    the plan's state of charge trough -- the moment the stored energy is
    actually called upon -- an extra kWh is worth the best use the plan can put
    it to, but never more than the cost of simply buying it back, because the
    plan would take the cheaper of the two::

        best_use = eta * max(price_import over the window)
        refill   =       min(price_import over the window) / eta
        scarce   = min(best_use, refill)

    **Whether the plan needs it at all.** If the plan's trough still sits well
    above the working lower limit, the battery has slack: spending a little
    extra now does not shorten a single planned discharge, it only means ending
    the horizon with less. That surplus is worth what the optimizer itself
    books for leftover energy, the average tariff discounted by the round trip
    efficiency. The two are blended over
    ``SCARCITY_BAND_FRACTION`` of the capacity so the valuation never jumps.

    Without this second step the layer would refuse to cover an evening peak
    while sitting on a battery it has no plan for, which is the most valuable
    thing it can do.

    The result is bounded below by the current export price -- energy is never
    worth less than what you can sell it for right now -- and above by the
    highest remaining import price.
    """
    eta = max(0.01, min(1.0, limits.round_trip_efficiency))
    remaining = plan.remaining(now_ts)
    if not remaining:
        return max(0.0, plan.price_average * eta)

    current = remaining[0]
    mode = (limits.storage_value_mode or "plan").lower()

    if mode == "fixed" and limits.storage_value_fixed is not None:
        return float(limits.storage_value_fixed)

    if mode == "average":
        average = sum(i.price_import for i in remaining) / len(remaining)
        return max(current.price_export, average * eta)

    # mode == "plan"
    socs = [i.battery(battery_index).soc_end for i in remaining]
    trough_index = min(range(len(socs)), key=lambda i: socs[i]) if socs else 0
    window = remaining[: trough_index + 1] or remaining

    prices = [i.price_import for i in window]
    scarce_value = min(eta * max(prices), min(prices) / eta)

    spec = plan.spec(battery_index)
    surplus_value = max(current.price_export, plan.price_average * eta)
    scarcity = 1.0
    if spec is not None and spec.capacity_kwh > 0 and socs:
        floor_soc = spec.soc_min + limits.soc_margin
        slack_kwh = (socs[trough_index] - floor_soc) * spec.kwh_per_percent
        band_kwh = max(1e-6, SCARCITY_BAND_FRACTION * spec.capacity_kwh)
        scarcity = 1.0 - _taper(slack_kwh, band_kwh)
    value = scarcity * scarce_value + (1.0 - scarcity) * surplus_value

    ceiling = max(i.price_import for i in remaining)
    return _clamp(value, current.price_export, ceiling)


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------


class FastControlPolicy:
    """Stateless decision maker. All mutable state lives in ``ControllerState``."""

    def __init__(self, limits: PolicyLimits):
        self.limits = limits

    # -- accounting ------------------------------------------------------

    def account(
        self,
        state: ControllerState,
        plan: FastPlan,
        measurement: Measurement,
        day_key: str,
        use_measured: bool = True,
    ) -> None:
        """Integrate the deviation energy and roll over budgets.

        Called once per tick, before :meth:`decide`. The deviation is measured
        against the plan the controller was following during the elapsed
        interval, using the measured battery power when available and the last
        command otherwise.

        In shadow mode the inverter is still following the optimizer, so the
        measured power would show no deviation at all and the budgets would
        never be exercised. Pass ``use_measured=False`` there to account against
        the setpoint the layer *would* have issued.
        """
        now = measurement.timestamp
        previous = state.last_tick_ts
        state.last_tick_ts = now

        if state.day_key != day_key:
            state.day_key = day_key
            state.saved_today_eur = 0.0
            state.saved_today_is_estimate = False
            for battery in state.batteries:
                battery.daily_deviation_kwh = 0.0

        interval = plan.interval_at(now)
        interval_start = interval.start_ts if interval else 0
        plan_changed = state.plan_created_ts != plan.created_ts
        interval_changed = state.interval_start_ts != interval_start
        if plan_changed or interval_changed:
            # A fresh plan re-baselines the state of charge from the real
            # measurement, so any accumulated deviation is already absorbed.
            state.plan_created_ts = plan.created_ts
            state.interval_start_ts = interval_start
            for battery in state.batteries:
                battery.interval_deviation_kwh = 0.0

        if previous <= 0.0 or now <= previous:
            return
        # Skip integration when the gap is suspiciously large: e.g. the
        # add-on was stopped for hours and the state was reloaded from
        # disk. Capping at 600s (10 minutes) would still invent ten minutes
        # of "mid" power consumption that never happened. Treat anything
        # longer than 10 plan intervals as a discontinuity and reset the
        # budget accumulators instead.
        plan_interval_s = plan.interval_s or 3600
        if now - previous > 10 * max(5, plan_interval_s):
            for battery in state.batteries:
                battery.interval_deviation_kwh = 0.0
            return
        elapsed_h = (now - previous) / 3600.0
        if elapsed_h <= 0.0 or interval is None:
            return

        for index in range(len(plan.specs)):
            controller = state.battery(index)
            measured = measurement.battery(index)
            if use_measured and measured.valid and measured.power_w is not None:
                actual_w = measured.power_w
            elif plan.format_version == 2 and controller.execution_status != "accepted":
                # The in-force setpoint is not a sensor reading. Without an
                # accepted command this segment is unknown, not a measured
                # zero, and no deviation is booked for it.
                continue
            else:
                actual_w = controller.held_setpoint_w
            if plan.format_version == 2:
                # Split only at actual profile/interval edges; do not charge the
                # current mean for the entire elapsed tick (or an earlier row).
                for row in plan.intervals:
                    start, end = max(previous, row.start_ts), min(now, row.end_ts)
                    if start >= end:
                        continue
                    step = row.battery(index)
                    edges = sorted({start, end, *(
                        edge for edge in (step.active_from_ts, step.stop_ts)
                        if edge is not None and start < edge < end)})
                    for a, b in zip(edges, edges[1:], strict=False):
                        delta = actual_w * (b - a) / 3600000 - profile_energy_kwh(step, a, b)
                        if row is interval:
                            controller.interval_deviation_kwh += delta
                        controller.daily_deviation_kwh += abs(delta)
                continue
            plan_w = interval.battery(index).ac_power_w
            delta_w = actual_w - plan_w
            controller.interval_deviation_kwh += delta_w * elapsed_h / 1000.0
            controller.daily_deviation_kwh += abs(delta_w) * elapsed_h / 1000.0

    # -- feasible set ----------------------------------------------------

    def _bounds(
        self,
        spec: BatterySpec,
        controller: BatteryControllerState,
        plan_w: float,
        soc: Optional[float],
        reasons: list[str],
    ) -> tuple[float, float]:
        """Feasible setpoint interval, always containing the plan setpoint.

        Keeping the plan inside the interval is the safety invariant of this
        layer: whatever the economics say, the worst it can do is fall back on
        what the optimizer already decided.
        """
        limits = self.limits

        high = spec.max_charge_w
        low = -spec.max_discharge_w

        # Energy trust region around the planned state of charge.
        if limits.energy_budget > 0.0:
            band = max(1e-6, BUDGET_TAPER_FRACTION * limits.energy_budget)
            charge_left = limits.energy_budget - controller.interval_deviation_kwh
            discharge_left = limits.energy_budget + controller.interval_deviation_kwh
            charge_factor = _taper(charge_left, band)
            discharge_factor = _taper(discharge_left, band)
            if charge_factor < 1.0 or discharge_factor < 1.0:
                reasons.append("energy_budget")
            high = min(high, plan_w + charge_factor * spec.max_charge_w)
            low = max(low, plan_w - discharge_factor * spec.max_discharge_w)

        # Daily wear budget.
        if limits.daily_extra_throughput > 0.0:
            band = max(1e-6, BUDGET_TAPER_FRACTION * limits.daily_extra_throughput)
            left = limits.daily_extra_throughput - controller.daily_deviation_kwh
            factor = _taper(left, band)
            if factor < 1.0:
                reasons.append("daily_budget")
            high = min(high, plan_w + factor * spec.max_charge_w)
            low = max(low, plan_w - factor * spec.max_discharge_w)

        # State of charge working range, tighter than the one the optimizer uses.
        if soc is not None:
            soc_high = spec.soc_max - limits.soc_margin
            soc_low = spec.soc_min + limits.soc_margin
            charge_factor = _taper(soc_high - soc, SOC_TAPER_BAND)
            discharge_factor = _taper(soc - soc_low, SOC_TAPER_BAND)
            if charge_factor < 1.0 or discharge_factor < 1.0:
                reasons.append("soc_limit")
            high = min(high, charge_factor * spec.max_charge_w)
            low = max(low, -discharge_factor * spec.max_discharge_w)

        # Slew rate.
        if limits.max_ramp:
            high = min(high, controller.held_setpoint_w + limits.max_ramp)
            low = max(low, controller.held_setpoint_w - limits.max_ramp)

        # The plan is always feasible.
        high = max(high, plan_w)
        low = min(low, plan_w)
        return low, high

    @staticmethod
    def _pieces(
        low: float, high: float, minimum_power: float, plan_w: float
    ) -> list[tuple[float, float]]:
        """Split the feasible interval around the inverter's dead zone.

        Many hybrid inverters cannot run below a few hundred watts. Rather than
        commanding an unachievable setpoint, the solve is run on each reachable
        piece and the cheapest result wins. The plan setpoint is always added as
        a degenerate piece, because the optimizer may legitimately sit inside the
        dead zone and duty-cycle the inverter to realise it.
        """
        if minimum_power <= 0.0:
            return [(low, high)]
        pieces: list[tuple[float, float]] = [(plan_w, plan_w)]
        if low <= -minimum_power:
            pieces.append((low, min(high, -minimum_power)))
        if low <= 0.0 <= high:
            pieces.append((0.0, 0.0))
        if high >= minimum_power:
            pieces.append((max(low, minimum_power), high))
        return pieces

    # -- the decision ----------------------------------------------------

    def decide(
        self,
        plan: FastPlan,
        measurement: Measurement,
        state: ControllerState,
        enabled: Optional[list[bool]] = None,
    ) -> Decision:
        """Compute the setpoints for this tick.

        Batteries are handled greedily in configuration order: the first one
        absorbs as much of the deviation as its own limits allow, the residual
        is offered to the next. Exact for a single battery, and predictable and
        cheap for several.
        """
        now = measurement.timestamp
        interval = plan.interval_at(now)
        if interval is None:
            return self._all_plan(
                plan, measurement, state, "plan_expired", interval=None
            )

        price_import = interval.price_import
        # Guarantees convexity of the grid term. An export price above the
        # import price would imply free money from importing and exporting at
        # the same time, which the meter does not allow anyway.
        price_export = min(interval.price_export, price_import)

        battery_measurements = [measurement.battery(i) for i in range(len(plan.specs))]
        house_w = measurement.grid_w
        for index, measured in enumerate(battery_measurements):
            if measured.valid and measured.power_w is not None:
                house_w -= measured.power_w
            else:
                house_w -= state.battery(index).held_setpoint_w

        economics_supported = True
        if plan.format_version == 2:
            context = plan.financial_context
            economics_supported = bool(context is not None and context.economics_supported)

        decision = Decision(
            timestamp=now,
            house_w=house_w,
            grid_w=measurement.grid_w,
            plan_grid_w=interval.grid_w,
            deviation_w=house_w - interval.house_w,
            price_import=price_import,
            price_export=price_export,
        )

        if not measurement.grid_valid:
            return self._all_plan(plan, measurement, state, measurement.invalid_reason, interval)

        # Batteries are decided one at a time. Each one sees the rest of the
        # site as its load: the house plus whatever the other batteries are
        # expected to do, which is their decision once taken and their plan
        # setpoint until then.
        residual = house_w + sum(
            profile_power_at(interval.battery(i), now) if plan.format_version == 2
            else interval.battery(i).ac_power_w for i in range(len(plan.specs))
        )
        total_benefit = 0.0
        any_override = False

        for index, spec in enumerate(plan.specs):
            controller = state.battery(index)
            step = interval.battery(index)
            plan_w = profile_power_at(step, now) if plan.format_version == 2 else step.ac_power_w
            measured = battery_measurements[index]
            soc = measured.soc
            others_w = residual - plan_w

            skip = None
            if enabled is not None and index < len(enabled) and not enabled[index]:
                skip = "disabled"
            elif soc is None:
                # Without a state of charge reading the working range cannot be
                # enforced, so overriding could run the battery into its limit.
                skip = "soc_unknown"
            if skip is not None:
                decision.batteries.append(
                    self._plan_decision(index, spec, step, controller, soc, skip, now,
                                        profiled=plan.format_version == 2)
                )
                residual = others_w + plan_w
                continue

            storage_value = estimate_storage_value(plan, now, index, self.limits)
            reasons: list[str] = []
            low, high = self._bounds(spec, controller, plan_w, soc, reasons)

            cost = CostModel(
                house_w=others_w,
                price_import=price_import,
                price_export=price_export,
                storage_value=storage_value,
                cycle_cost=spec.cycle_cost,
                round_trip_efficiency=max(
                    0.01, min(1.0, self.limits.round_trip_efficiency)
                ),
            )

            best_power = plan_w
            best_cost = math.inf
            for piece_low, piece_high in self._pieces(
                low, high, spec.minimum_power_w, plan_w
            ):
                candidate = cost.minimise(piece_low, piece_high)
                candidate_cost = cost(candidate)
                if candidate_cost < best_cost - 1e-12:
                    best_cost = candidate_cost
                    best_power = candidate

            # Never buy from the grid to charge beyond the plan unless allowed.
            if not self.limits.allow_grid_charge and best_power > plan_w:
                ceiling = max(plan_w, -others_w)
                if best_power > ceiling:
                    best_power = ceiling
                    reasons.append("no_grid_charge")

            benefit = cost(plan_w) - cost(best_power)

            if not economics_supported:
                # Unsupported economics forbid a deviation, never the plan
                # itself: the plan is still followed, without a benefit claim.
                best_power = plan_w
                benefit = 0.0
                reasons.append("economics_unsupported")

            # Peak shaving overrules the economics, but never the hardware or
            # the state of charge working range. Without economics there is no
            # deviation at all, so the peak shave may not create one either.
            peak_shaved = False
            if economics_supported and self.limits.max_grid_import is not None:
                needed = self.limits.max_grid_import - others_w
                if needed < best_power:
                    hard_low, hard_high = self._hard_bounds(spec, soc, plan_w)
                    shaved = _clamp(needed, hard_low, hard_high)
                    if shaved < best_power - POWER_EPS:
                        best_power = shaved
                        peak_shaved = True
                        reasons.append("peak_shave")

            if not peak_shaved and benefit < self.limits.min_benefit:
                best_power = plan_w
                benefit = 0.0
                reasons.append("below_min_benefit")

            battery_decision = self._stabilise(
                index=index,
                spec=spec,
                step=step,
                controller=controller,
                target_w=best_power,
                plan_w=plan_w,
                now=now,
                soc=soc,
                benefit=benefit,
                storage_value=storage_value,
                reasons=reasons,
                profiled=plan.format_version == 2,
                economics_supported=economics_supported,
            )
            decision.batteries.append(battery_decision)

            # A segment whose execution is not accepted books no euro claim:
            # an unproven or ambiguous execution is not a saving.
            countable = battery_decision.override
            if plan.format_version == 2 and controller.execution_status != "accepted":
                countable = False
            total_benefit += max(0.0, benefit) if countable else 0.0
            any_override = any_override or battery_decision.override
            residual = others_w + battery_decision.setpoint_w

        decision.benefit_eur_h = total_benefit
        decision.reason = "override" if any_override else "plan"
        return decision

    def release(
        self,
        plan: FastPlan,
        measurement: Measurement,
        state: ControllerState,
        reason: str,
    ) -> Decision:
        """Hand every battery back to the optimizer's published command.

        Used when the layer is switched off while an override is active: the
        inverter must not be left on the overridden setpoint with its stop
        moment cleared until the next optimisation happens to run.
        """
        return self._all_plan(
            plan, measurement, state, reason, plan.interval_at(measurement.timestamp)
        )

    def _hard_bounds(
        self, spec: BatterySpec, soc: Optional[float], plan_w: float
    ) -> tuple[float, float]:
        """Inverter and state of charge limits only, ignoring the budgets."""
        high = spec.max_charge_w
        low = -spec.max_discharge_w
        if soc is not None:
            high = min(
                high,
                _taper(spec.soc_max - self.limits.soc_margin - soc, SOC_TAPER_BAND)
                * spec.max_charge_w,
            )
            low = max(
                low,
                -_taper(soc - spec.soc_min - self.limits.soc_margin, SOC_TAPER_BAND)
                * spec.max_discharge_w,
            )
        return min(low, plan_w), max(high, plan_w)

    @staticmethod
    def _confirmed_baseline(controller: BatteryControllerState) -> float:
        """The in-force setpoint of a financial segment, but only when proven.

        Only an accepted command proves an in-force setpoint. Without one
        nothing is in force: the plan power is a proposal, never an already
        in-force fact.
        """
        if controller.execution_status == "accepted":
            return controller.held_setpoint_w
        return 0.0

    def _stabilise(
        self,
        index: int,
        spec: BatterySpec,
        step,
        controller: BatteryControllerState,
        target_w: float,
        plan_w: float,
        now: float,
        soc: Optional[float],
        benefit: float,
        storage_value: float,
        reasons: list[str],
        *,
        profiled: bool = False,
        economics_supported: bool = True,
    ) -> BatteryDecision:
        """Apply hysteresis, deadband and minimum command interval.

        The economic target is recomputed from measurements every tick, so the
        hysteresis deliberately does not freeze the *value* of the setpoint --
        that would keep discharging into a load that has already finished. It
        only governs when the layer declares the override over and restores the
        operating mode and the inverter stop moment the optimizer published.
        """
        limits = self.limits
        deviating = abs(target_w - plan_w) > limits.release_deviation
        baseline_w = controller.held_setpoint_w

        if profiled and not economics_supported:
            # Without economics no deviation is allowed at all, not even an
            # existing one: it ends in this very tick, without the release
            # hysteresis. The target is the plan power.
            controller.quiet_since = None
            override = False
            target_w = plan_w
            if not controller.override_active:
                baseline_w = self._confirmed_baseline(controller)
        elif deviating:
            controller.quiet_since = None
            override = True
        elif controller.override_active:
            if controller.quiet_since is None:
                controller.quiet_since = now
            override = (now - controller.quiet_since) < limits.release_time
            if not override:
                controller.quiet_since = None
                target_w = plan_w
                reasons.append("released")
        else:
            controller.quiet_since = None
            override = False
            if profiled:
                # v2: the optimizer wrote nothing, so the plan power is not an
                # already in-force fact but a proposal. Only a confirmed
                # command proves an in-force setpoint.
                baseline_w = self._confirmed_baseline(controller)
                target_w = plan_w
            else:
                # Reconcile the in-force plan fact across ticks, without claiming
                # helper acceptance or forcing a redundant write.
                controller.held_setpoint_w = plan_w
                baseline_w = plan_w
                target_w = plan_w

        releasing = controller.override_active and not override
        restoring = controller.restore_pending and not override
        change = abs(target_w - baseline_w)
        write = True

        # The legacy optimizer duty-cycles the inverter through the stop moment when it
        # needs less than the minimum power. While overriding, that stop moment
        # must be cleared, so the deadband may not suppress the first write.
        pending_stop = (
            not profiled
            and override
            and not controller.held_stop_inverter_cleared
            and bool(step.stop_inverter)
            and step.stop_inverter != NO_STOP_SENTINEL
        )

        if (not releasing and not pending_stop) or restoring:
            # In the financial regime the thresholds guard an override only: a
            # plan-following command is a real proposal whenever the plan power
            # differs from the in-force setpoint. The legacy regime keeps its
            # behaviour exactly.
            guarded = not profiled or override
            if not restoring and change < limits.deadband and (guarded or change == 0):
                target_w = baseline_w
                write = False
                reasons.append("deadband")
            elif (
                guarded
                and now - controller.last_change_ts < limits.min_command_interval
                and (restoring or limits.urgent_deviation <= 0 or change < limits.urgent_deviation)
            ):
                target_w = baseline_w
                write = False
                reasons.append("rate_limited")

        target_w = round(target_w)
        if write and not profiled:
            # In the financial regime a proposal is not an in-force change: the
            # runner stamps the moment of an applied acceptance instead.
            controller.last_change_ts = now
        controller.override_active = override

        mode: Optional[str] = None
        stop_inverter: Optional[str] = None
        stop_cleared = controller.held_stop_inverter_cleared
        restores = False
        if write:
            if override:
                # While overriding, the inverter must be enabled and must not
                # honour the duty-cycle stop moment the optimizer published for
                # a sub-minimum-power setpoint.
                mode = spec.mode_off if abs(target_w) < POWER_EPS else spec.mode_on
                stop_inverter = NO_STOP_SENTINEL
                stop_cleared = True
            elif controller.held_stop_inverter_cleared or releasing or restoring:
                mode = step.mode if step.mode else (
                    spec.mode_off if abs(target_w) < POWER_EPS else spec.mode_on
                )
                stop_inverter = None if profiled else step.stop_inverter
                stop_cleared = False
                restores = True

        if not reasons:
            reasons.append("override" if override else "plan")

        return BatteryDecision(
            index=index,
            name=spec.name,
            setpoint_w=target_w,
            plan_w=plan_w,
            write=write,
            override=override,
            reason="+".join(dict.fromkeys(reasons)),
            benefit_eur_h=benefit,
            storage_value=storage_value,
            mode=mode,
            stop_inverter=stop_inverter,
            soc=soc,
            budget_used_kwh=controller.interval_deviation_kwh,
            daily_used_kwh=controller.daily_deviation_kwh,
            accepted=AcceptedCommand(target_w, now, stop_cleared) if write else None,
            restores=restores,
        )

    def _plan_decision(
        self,
        index: int,
        spec: BatterySpec,
        step,
        controller: BatteryControllerState,
        soc: Optional[float],
        reason: str,
        now: float,
        *,
        profiled: bool = False,
    ) -> BatteryDecision:
        """Fall back to exactly what the optimizer published."""
        plan_w = round(profile_power_at(step, now) if profiled else step.ac_power_w)
        releasing = (controller.override_active or controller.held_stop_inverter_cleared
                     or controller.restore_pending)
        if profiled:
            # v2: the threshold never suppresses a plan-following. Only a
            # confirmed command proves an in-force setpoint, so without one
            # the plan power is always a proposal. The legacy regime keeps
            # its behaviour exactly.
            baseline_w = self._confirmed_baseline(controller)
            write = releasing or plan_w != baseline_w
        else:
            write = (
                releasing or abs(plan_w - controller.held_setpoint_w) >= self.limits.deadband
            )
        if (not profiled
                and controller.restore_pending
                and now - controller.last_change_ts < self.limits.min_command_interval):
            # The legacy restore/rate-limit path belongs to v1: in the
            # financial regime a plan-following is a proposal that the
            # threshold and the rate limit may never hide.
            write = False
            reason += "+rate_limited"
        if write and not profiled:
            # In the financial regime a proposal is not an in-force change.
            controller.last_change_ts = now
        mode: Optional[str] = None
        stop_inverter: Optional[str] = None
        if releasing:
            mode = (
                step.mode
                if step.mode
                else (spec.mode_off if abs(plan_w) < POWER_EPS else spec.mode_on)
            )
            stop_inverter = None if profiled else step.stop_inverter
        controller.override_active = False
        controller.quiet_since = None
        return BatteryDecision(
            index=index,
            name=spec.name,
            setpoint_w=plan_w,
            plan_w=plan_w,
            write=write,
            override=False,
            reason=reason,
            mode=mode,
            stop_inverter=stop_inverter,
            soc=soc,
            budget_used_kwh=controller.interval_deviation_kwh,
            daily_used_kwh=controller.daily_deviation_kwh,
            accepted=(
                AcceptedCommand(plan_w, now, False if releasing else controller.held_stop_inverter_cleared)
                if write else None
            ),
            restores=releasing,
        )

    def _all_plan(
        self,
        plan: FastPlan,
        measurement: Measurement,
        state: ControllerState,
        reason: str,
        interval: Optional[PlanInterval],
    ) -> Decision:
        decision = Decision(
            timestamp=measurement.timestamp,
            grid_w=measurement.grid_w,
            reason=reason,
            price_import=interval.price_import if interval else 0.0,
            price_export=interval.price_export if interval else 0.0,
            plan_grid_w=interval.grid_w if interval else 0.0,
        )
        for index, spec in enumerate(plan.specs):
            step = interval.battery(index) if interval else None
            controller = state.battery(index)
            if step is None:
                # No plan at all: hold whatever is currently commanded.
                decision.batteries.append(
                    BatteryDecision(
                        index=index,
                        name=spec.name,
                        setpoint_w=controller.held_setpoint_w,
                        plan_w=controller.held_setpoint_w,
                        write=False,
                        override=False,
                        reason=reason,
                        soc=measurement.battery(index).soc,
                    )
                )
                controller.override_active = False
                continue
            decision.batteries.append(
                self._plan_decision(
                    index,
                    spec,
                    step,
                    controller,
                    measurement.battery(index).soc,
                    reason,
                    measurement.timestamp,
                    profiled=plan.format_version == 2,
                )
            )
        return decision
