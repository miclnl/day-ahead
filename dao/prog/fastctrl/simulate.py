"""Backtest the fast control layer on historical data.

Replays a measured house load against a historical day-ahead plan twice: once
with the plan setpoint held constant across each interval, which is what the
optimizer does today, and once through the production
:class:`~dao.prog.fastctrl.policy.FastControlPolicy`. Because both runs share
the same battery model and the same measurements, the difference is a clean
estimate of what the fast layer is worth on your own house.

Two things make the comparison honest:

* The terminal state of charge difference is priced. Without it the fast run
  could "win" simply by ending the window with an emptier battery.
* Battery wear is priced with the same ``cycle cost`` the optimizer uses, so
  extra cycling shows up as a cost, not as free money.

The historical data comes from the Home Assistant recorder. Note that its
default retention is ten days and that long term statistics are hourly, which
is far too coarse to show peak shaving. Run the backtest on the highest
resolution you have and read the resolution warning the loader prints.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal, localcontext
from typing import Callable, Literal, Optional, Sequence

from dao.prog.settlement.replay import ReplayBasis
from dao.prog.settlement.types import PriceQuote, SettlementState, TerminalValue

from .command import BatteryCommand, plan_digest
from .plan import (
    BatterySpec, BatteryPlanStep, FastPlan, PlanInterval,
    profile_energy_kwh, profile_power_at,
)
from .policy import (
    BatteryDecision,
    BatteryMeasurement,
    ControllerState,
    FastControlPolicy,
    Measurement,
    PolicyLimits,
    apply_accepted_command,
)
from .replay_execution import (
    Assumptions,
    BatteryExecutionView,
    ExecutionReplay,
    ExecutionSnapshot,
    ModeledExecution,
    ProposalEvent,
    Reasons,
    RecordedExecution,
    ReplayEvent,
    ReplayInput,
    SocByBattery,
    model_namespace,
)

#: Default AC/DC conversion efficiency used by the simulation battery model when
#: the plan does not carry the inverter's own stage curve.
DEFAULT_INVERTER_EFFICIENCY = 0.95

#: Days of recorder history read in one go. Keeps the peak memory of a backtest
#: at one chunk of one entity instead of the whole window, which matters on the
#: 4 GB of a Home Assistant Yellow.
CHUNK_DAYS = 2


def _iter_chunks(
    start: datetime.datetime, end: datetime.datetime, days: int
) -> "list[tuple[datetime.datetime, datetime.datetime]]":
    """Split ``[start, end)`` into windows of at most *days* days."""
    span = datetime.timedelta(days=max(1, days))
    chunks = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + span, end)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


@dataclass
class Sample:
    """One measured moment of the house, excluding the battery."""

    timestamp: float
    house_w: float


@dataclass
class BatteryModel:
    """A simple, symmetric battery used to replay both scenarios identically."""

    spec: BatterySpec
    soc: float
    inverter_efficiency: float = DEFAULT_INVERTER_EFFICIENCY

    def step(self, command_w: float, dt_h: float) -> tuple[float, float]:
        """Apply *command_w* for *dt_h* hours.

        Returns the AC power that was actually realised and the DC throughput
        in kWh, both of which the caller needs for the cost accounting.
        """
        spec = self.spec
        power = max(-spec.max_discharge_w, min(spec.max_charge_w, command_w))
        if dt_h <= 0.0:
            return 0.0, 0.0

        if power >= 0.0:
            dc_kwh = power * dt_h * self.inverter_efficiency / 1000.0
            stored_kwh = dc_kwh * spec.charge_efficiency
            room_kwh = max(0.0, (spec.soc_max - self.soc) * spec.kwh_per_percent)
            if stored_kwh > room_kwh:
                scale = room_kwh / stored_kwh if stored_kwh > 0 else 0.0
                power *= scale
                dc_kwh *= scale
                stored_kwh = room_kwh
            self.soc += stored_kwh / spec.kwh_per_percent
            return power, dc_kwh

        ac_kwh = -power * dt_h / 1000.0
        dc_kwh = ac_kwh / self.inverter_efficiency
        drawn_kwh = dc_kwh / spec.discharge_efficiency
        available_kwh = max(0.0, (self.soc - spec.soc_min) * spec.kwh_per_percent)
        if drawn_kwh > available_kwh:
            scale = available_kwh / drawn_kwh if drawn_kwh > 0 else 0.0
            power *= scale
            dc_kwh *= scale
            drawn_kwh = available_kwh
        self.soc -= drawn_kwh / spec.kwh_per_percent
        return power, dc_kwh


@dataclass(frozen=True)
class BatteryReplaySegment:
    """One battery's execution evidence and independently modeled gross flows."""

    battery_name: str
    command: BatteryCommand | None
    execution_status: str
    reasons: Reasons
    command_w: float | None
    modeled_ac_w: float | None
    charge_ac_kwh: float | None
    discharge_ac_kwh: float | None
    dc_throughput_kwh: float | None
    cell_delta_kwh: float | None
    soc_begin: float | None
    soc_end: float | None


@dataclass(frozen=True)
class ReplaySegment:
    """A half-open physical slice with nullable site import and export."""

    start_ts: float
    end_ts: float
    house_w: float | None
    batteries: tuple[BatteryReplaySegment, ...]
    import_kwh: float | None
    export_kwh: float | None


def _next_physical_boundary(
    *, start_ts: float, end_ts: float, models: dict[str, BatteryModel | None],
    execution: ExecutionSnapshot,
) -> float:
    """Find saturation without advancing models or quantizing physical time."""
    boundary = end_ts
    for view in execution.batteries:
        model = models[view.battery_name]
        if model is None or view.drive_w is None:
            continue
        spec = model.spec
        power = max(-spec.max_discharge_w, min(spec.max_charge_w, view.drive_w))
        # Invert only the duration of BatteryModel.step's existing cell-energy
        # chain. The integrator still delegates all realized flows to step.
        if power > 0:
            remaining_kwh = max(0.0, (spec.soc_max - model.soc) * spec.kwh_per_percent)
            cell_kw = power * model.inverter_efficiency / 1000.0 * spec.charge_efficiency
        elif power < 0:
            remaining_kwh = max(0.0, (model.soc - spec.soc_min) * spec.kwh_per_percent)
            cell_kw = -power / 1000.0 / model.inverter_efficiency / spec.discharge_efficiency
        else:
            continue
        at_ts = start_ts + remaining_kwh / cell_kw * 3600.0
        # A saturated model already clips to zero; never repeat a zero span.
        if start_ts < at_ts < boundary:
            boundary = at_ts
    return boundary


def _integrate_replay_segment(
    *, start_ts: float, end_ts: float, house_w: float | None,
    models: dict[str, BatteryModel | None], execution: ExecutionSnapshot,
) -> ReplaySegment:
    """Integrate an already split slice, mutating only this run's own models.

    None is an irreversible unknown-SoC latch. Proven zero drive can still have
    known zero flows after that latch, but cannot reconstruct the lost SoC.
    """
    dt_h = (end_ts - start_ts) / 3600.0
    batteries = []
    grid_w = house_w
    for view in execution.batteries:
        model = models[view.battery_name]
        soc_begin = model.soc if model is not None else None
        if view.drive_w is None:
            models[view.battery_name] = model = None
        if model is None:
            actual_w = dc_kwh = cell_delta_kwh = 0.0 if view.drive_w == 0 else None
            soc_end = None
            reasons = tuple(dict.fromkeys((*view.reasons, "replay_soc_missing")))
        else:
            actual_w, dc_kwh = model.step(view.drive_w, dt_h)
            soc_end = model.soc
            cell_delta_kwh = (soc_end - soc_begin) * model.spec.kwh_per_percent
            reasons = view.reasons
        charge_kwh = None if actual_w is None else max(0.0, actual_w) * dt_h / 1000.0
        discharge_kwh = None if actual_w is None else max(0.0, -actual_w) * dt_h / 1000.0
        batteries.append(BatteryReplaySegment(
            battery_name=view.battery_name,
            command=view.command,
            execution_status=view.execution_status,
            reasons=reasons,
            command_w=float(view.command.setpoint_w) if view.command is not None else None,
            modeled_ac_w=actual_w,
            charge_ac_kwh=charge_kwh,
            discharge_ac_kwh=discharge_kwh,
            dc_throughput_kwh=dc_kwh,
            cell_delta_kwh=cell_delta_kwh,
            soc_begin=soc_begin,
            soc_end=soc_end,
        ))
        grid_w = None if grid_w is None or actual_w is None else grid_w + actual_w
    return ReplaySegment(
        start_ts=start_ts, end_ts=end_ts, house_w=house_w, batteries=tuple(batteries),
        import_kwh=None if grid_w is None else max(0.0, grid_w) * dt_h / 1000.0,
        export_kwh=None if grid_w is None else max(0.0, -grid_w) * dt_h / 1000.0,
    )


@dataclass(frozen=True)
class _ReplayBasisView:
    source_plan: FastPlan
    execution_plan: FastPlan
    quotes: tuple[PriceQuote, ...] | None
    opening: SettlementState | None
    terminal: TerminalValue | None
    reasons: Reasons


def _prepare_replay_basis(
    plan: FastPlan, *, start_ts: float, replay_basis: ReplayBasis | None, recorded: bool,
) -> _ReplayBasisView:
    """Keep execution identity separate from the exact archived financial basis."""
    if not plan.specs:
        raise ValueError("het plan bevat geen batterij")
    source = deepcopy(plan)
    execution = deepcopy(source)
    if replay_basis is None:
        reasons = ("replay_basis_required",) if plan.format_version == 2 else ()
        return _ReplayBasisView(source, execution, None, None, None, reasons)
    try:
        if type(replay_basis) is not ReplayBasis or source.format_version != 2 or source.financial_context is None:
            raise ValueError("replay_basis_required")
        replay_basis.check_context(source.financial_context)
        if start_ts != replay_basis.opening.asof_ts:
            raise ValueError("replay_opening_mismatch")
        if len(source.intervals) != len(replay_basis.quotes):
            raise ValueError("replay_quote_coverage_mismatch")
        cursor = replay_basis.basis.planning_start_ts
        intervals = []
        for interval, quote in zip(source.intervals, replay_basis.quotes, strict=True):
            if (interval.start_ts != cursor
                    or (quote.start_ts, quote.end_ts) != (interval.start_ts, interval.end_ts)):
                raise ValueError("replay_quote_coverage_mismatch")
            intervals.append(replace(interval, price_import=float(quote.import_rate),
                                     price_export=float(quote.export_rate)))
            cursor = interval.end_ts
        if cursor != replay_basis.basis.horizon_end_ts:
            raise ValueError("replay_quote_coverage_mismatch")
        if tuple(spec.name for spec in source.specs) != tuple(
                name for name, _ in replay_basis.terminal_value.reference_efficiency_by_battery):
            raise ValueError("replay_terminal_identity_mismatch")
        if not recorded:
            projection = replace(source, intervals=intervals, price_average=sum(
                float(quote.import_rate) for quote in replay_basis.quotes) / len(intervals))
            execution = FastPlan.from_dict(projection.to_dict())
    except (ValueError, TypeError, AttributeError):
        return _ReplayBasisView(source, execution, None, None, None, ("replay_basis_mismatch",))
    return _ReplayBasisView(source, execution, replay_basis.quotes,
                            replay_basis.opening, replay_basis.terminal_value, ())


@dataclass(frozen=True)
class _ReplayRun:
    segments: tuple[ReplaySegment, ...]
    trace: tuple[ReplayEvent, ...]
    soc_start_by_battery: SocByBattery
    soc_end_by_battery: SocByBattery
    writes: int
    override_s: float | None
    execution_kind: Literal["recorded", "modeled", "missing"]
    reasons: Reasons
    assumptions: Assumptions
    source_plan_digest: str | None
    execution_plan_digest: str | None


def _replay_digest(plan: FastPlan) -> str | None:
    try:
        return plan_digest(plan)
    except (ValueError, TypeError, AttributeError):
        return None


def _replay_measurement(segment: ReplaySegment, *, closing: bool) -> Measurement | None:
    """Adapt modeled response only; never let policy fall back to an old held fact."""
    if segment.house_w is None or any(
            battery.modeled_ac_w is None or battery.soc_end is None for battery in segment.batteries):
        return None
    return Measurement(
        timestamp=segment.end_ts if closing else segment.start_ts,
        grid_w=segment.house_w + sum(battery.modeled_ac_w for battery in segment.batteries),
        batteries=[BatteryMeasurement(
            soc=battery.soc_end if closing else battery.soc_begin, power_w=battery.modeled_ac_w,
        ) for battery in segment.batteries],
    )


def _run_replay(
    *, basis: _ReplayBasisView, replay_input: ReplayInput, limits: PolicyLimits,
    use_fast_layer: bool, timezone: datetime.tzinfo | None,
) -> _ReplayRun:
    """Compose the pure execution fold, production policy and per-battery physics."""
    plan = basis.execution_plan
    points = replay_input.sample_points
    start, end = points[0][0], points[-1][0]
    scenario = replay_input.scenario
    recipe = scenario.execution if scenario is not None else None
    modeled = isinstance(recipe, ModeledExecution)
    kind = "missing" if recipe is None else "modeled" if modeled else "recorded"
    reasons = [*basis.reasons, *replay_input.reasons]
    source_digest, execution_digest = _replay_digest(basis.source_plan), _replay_digest(plan)
    if source_digest is None or execution_digest is None:
        reasons.append("execution_binding_mismatch")
    socs = dict(replay_input.soc_by_battery)
    efficiencies = dict(replay_input.inverter_efficiency_by_battery)
    models = {spec.name: None if socs[spec.name] is None else BatteryModel(
        spec, socs[spec.name], efficiencies[spec.name]) for spec in plan.specs}
    state = ControllerState()
    policy = FastControlPolicy(limits)
    execution = None
    assumptions = []
    if scenario is not None:
        assumptions.extend((
            ("scenario_id", scenario.scenario_id),
            ("physics", json.dumps(asdict(scenario.physics), sort_keys=True)),
            ("inverter_efficiency_by_battery", json.dumps(replay_input.inverter_efficiency_by_battery)),
            ("policy_measurements", "modeled_response"),
        ))
        if modeled:
            assumptions.extend((("execution_model", json.dumps(asdict(recipe), sort_keys=True)),
                                ("fresh_policy_start", "true"), ("clock", "healthy_virtual_utc"),
                                ("clock_uncertainty_ms", "0"), ("io_authority", "none")))
        if execution_digest is not None and (not modeled or source_digest is not None):
            namespace = model_namespace(source_plan=basis.source_plan, execution_plan=plan,
                sample_points=points, limits=limits, replay_input=replay_input) if modeled else None
            execution = ExecutionReplay(plan=plan, scenario=scenario, controller=state,
                                         start_ts=start, end_ts=end, namespace=namespace)

    # Policy selectors depend only on the supplied samples and original profiles.
    ticks = {at for at, _ in points[:-1]}
    for interval in plan.intervals:
        ticks.update(edge for edge in (interval.start_ts, interval.end_ts) if start <= edge < end)
        ticks.update(edge for step in interval.batteries for edge in (step.active_from_ts, step.stop_ts)
                     if edge is not None and start <= edge < end)
    tick_indices = {at: index for index, at in enumerate(sorted(ticks))}
    boundaries = {*ticks, end}
    if scenario is not None:
        boundaries.update(at + scenario.physics.house_hold_s for at, _ in points[:-1]
                          if start < at + scenario.physics.house_hold_s < end)
    date = datetime.datetime.fromtimestamp(start, timezone).date() + datetime.timedelta(days=1)
    midnight = datetime.datetime.combine(date, datetime.time(), timezone).timestamp()
    while midnight < end:
        boundaries.add(midnight)
        date += datetime.timedelta(days=1)
        midnight = datetime.datetime.combine(date, datetime.time(), timezone).timestamp()
    boundaries = sorted(boundaries)
    boundary_index = sample_index = 0
    cursor = start
    segments = []
    override_s = 0.0
    policy_known = True
    snapshot = ExecutionSnapshot(None, tuple(BatteryExecutionView(
        spec.name, None, None, "unknown", False, None, False,
        ("execution_input_missing" if recipe is None else "execution_binding_mismatch",),
    ) for spec in plan.specs), 0, (), ())
    while cursor < end:
        while sample_index + 1 < len(points) and points[sample_index + 1][0] <= cursor:
            sample_index += 1
        sample_ts, house_w = points[sample_index]
        if scenario is None or cursor >= sample_ts + scenario.physics.house_hold_s:
            house_w = None
            if scenario is not None:
                reasons.append("replay_house_missing")
        while boundaries[boundary_index] <= cursor:
            boundary_index += 1
        next_edge = boundaries[boundary_index]
        if execution is not None:
            # The preceding slice was closed and accounted before this advance.
            execution.advance(cursor)
            snapshot = execution.snapshot(cursor)
        if modeled and execution is not None and cursor in tick_indices:
            tick = tick_indices[cursor]
            interval = plan.interval_at(cursor)
            if interval is not None and (not use_fast_layer or (plan.format_version == 1 and (
                    cursor == start or cursor == interval.start_ts))):
                proposals = []
                for index, spec in enumerate(plan.specs):
                    step = interval.battery(index)
                    power = profile_power_at(step, cursor) if plan.format_version == 2 else step.ac_power_w
                    current = snapshot.batteries[index].command
                    proposals.append(BatteryDecision(index, spec.name, power, power,
                        current is None or current.setpoint_w != int(round(power)), False, "plan",
                        mode=step.mode, stop_inverter=step.stop_inverter if plan.format_version == 1 else None,
                        soc=None if models[spec.name] is None else models[spec.name].soc))
                execution.submit(tick_index=tick, at_ts=cursor, proposals=tuple(proposals), source="plan")
                execution.advance(cursor)
                snapshot = execution.snapshot(cursor)
            if use_fast_layer and policy_known:
                # A preview on disposable models uses exactly the same clipping
                # chain as integration. No proposal supplies a measured watt.
                preview_end = _next_physical_boundary(start_ts=cursor, end_ts=next_edge,
                                                       models=models, execution=snapshot)
                preview = _integrate_replay_segment(start_ts=cursor, end_ts=preview_end, house_w=house_w,
                    models={name: replace(model) if model is not None else None for name, model in models.items()},
                    execution=snapshot)
                measurement = _replay_measurement(preview, closing=False)
                if measurement is not None:
                    day = _day_key(cursor, timezone)
                    if cursor == start or state.day_key != day:
                        policy.account(state, plan, measurement, day, owner_binding=snapshot.binding)
                    decision = policy.decide(plan, measurement, state, owner_binding=snapshot.binding)
                    execution.submit(tick_index=tick, at_ts=cursor,
                                     proposals=tuple(decision.batteries), source="fast")
                    execution.advance(cursor)
                    snapshot = execution.snapshot(cursor)
        reasons.extend(snapshot.reasons)
        if execution is not None:
            next_edge = execution.next_boundary(cursor, next_edge)
        next_edge = _next_physical_boundary(start_ts=cursor, end_ts=next_edge, models=models, execution=snapshot)
        segment = _integrate_replay_segment(start_ts=cursor, end_ts=next_edge, house_w=house_w,
                                            models=models, execution=snapshot)
        segments.append(segment)
        reasons.extend(reason for battery in segment.batteries for reason in battery.reasons)
        if any(battery.modeled_ac_w is None for battery in segment.batteries):
            override_s = None
        elif override_s is not None and any(view.override and view.command is not None for view in snapshot.batteries):
            override_s += next_edge - cursor
        if modeled and use_fast_layer and policy_known:
            measurement = _replay_measurement(segment, closing=True)
            if measurement is None:
                policy_known = False
                reasons.append("policy_history_unknown")
            else:
                policy.account(state, plan, measurement, _day_key(cursor, timezone), owner_binding=snapshot.binding)
        cursor = next_edge
    if execution is not None:
        execution.advance(end)
        snapshot = execution.snapshot(end)
        # End is excluded: coverage/holds ending here do not spoil the window.
        reasons.extend(reason for reason in snapshot.reasons if reason == "unused_model_response")
    elif isinstance(recipe, RecordedExecution):
        # An invalid plan identity cannot establish acceptance, but keep the raw
        # observations available alongside the explicit unknown physical slices.
        trace = tuple(opening.observation for opening in recipe.opening_acceptances
                      if opening.observation.at_ts < end) + tuple(event for event in recipe.events
                                                                 if start <= event.at_ts < end)
        # Count supplied send evidence even when no acceptance can be derived.
        writes = sum(isinstance(event, ProposalEvent) and event.decision.write and event.command is not None
                     for event in trace)
        writes += sum(opening.sent_at_ts is not None and start <= opening.sent_at_ts < end
                      for opening in recipe.opening_acceptances)
        snapshot = replace(snapshot, trace=deepcopy(trace), writes=writes)
    return _ReplayRun(tuple(segments), snapshot.trace, replay_input.soc_by_battery,
        tuple((spec.name, None if models[spec.name] is None else models[spec.name].soc) for spec in plan.specs),
        snapshot.writes, override_s, kind, tuple(dict.fromkeys(reasons)), tuple(assumptions),
        source_digest, execution_digest)


@dataclass
class DayMetrics:
    """Per day totals, so a saving can be traced back to a specific event."""

    day: str
    energy_cost: float | None = 0.0
    wear_cost: float | None = 0.0
    import_kwh: float | None = 0.0
    export_kwh: float | None = 0.0
    throughput_kwh: float | None = 0.0
    peak_import_w: float | None = 0.0
    writes: int = 0
    override_s: float | None = 0.0


@dataclass
class SimResult:
    """Outcome of one replay."""

    label: str
    energy_cost: float | None = 0.0
    wear_cost: float | None = 0.0
    import_kwh: float | None = 0.0
    export_kwh: float | None = 0.0
    throughput_kwh: float | None = 0.0
    peak_import_w: float | None = 0.0
    writes: int = 0
    override_s: float | None = 0.0
    soc_start: float | None = 0.0
    soc_end: float | None = 0.0
    terminal_value: float | None = 0.0
    capacity_kwh: float = 0.0
    duration_h: float = 0.0
    days: dict[str, DayMetrics] = field(default_factory=dict)
    soc_series: list[tuple[float, float | None]] = field(default_factory=list)
    opening_state: SettlementState | None = None
    replay_terminal_value: TerminalValue | None = None
    execution_kind: Literal["recorded", "modeled", "missing"] = "missing"
    coverage_status: Literal["complete", "partial", "unknown"] = "unknown"
    reasons: Reasons = ()
    assumptions: Assumptions = ()
    source_plan_digest: str | None = None
    execution_plan_digest: str | None = None
    segments: tuple[ReplaySegment, ...] = ()
    trace: tuple[ReplayEvent, ...] = ()
    soc_start_by_battery: SocByBattery = ()
    soc_end_by_battery: SocByBattery = ()
    settlement_comparison_pending: bool = False

    @property
    def total_cost(self) -> float | None:
        """Energy plus wear, corrected for the energy left in the battery."""
        if self.energy_cost is None or self.wear_cost is None or self.terminal_value is None:
            return None
        return self.energy_cost + self.wear_cost - self.terminal_value

    @property
    def cycles(self) -> float | None:
        """Equivalent full cycles, DC throughput over twice the capacity."""
        if self.throughput_kwh is None:
            return None
        if self.capacity_kwh <= 0:
            return 0.0
        return self.throughput_kwh / (2.0 * self.capacity_kwh)

    @property
    def average_soc(self) -> float | None:
        if any(s is None for _, s in self.soc_series):
            return None
        if not self.soc_series:
            return 0.0
        return sum(s for _, s in self.soc_series) / len(self.soc_series)


def _sum_known(values: Sequence[float | None]) -> float | None:
    return None if any(value is None for value in values) else sum(values)


def _summarize_replay(
    *, run: _ReplayRun, basis: _ReplayBasisView, replay_input: ReplayInput,
    label: str, timezone: datetime.tzinfo | None,
) -> SimResult:
    """Summarize complete quantities only; the segments retain known partials."""
    plan = basis.source_plan
    specs = {spec.name: spec for spec in plan.specs}
    indices = {spec.name: index for index, spec in enumerate(plan.specs)}
    capacity = sum(spec.capacity_kwh for spec in plan.specs)
    start, end = replay_input.sample_points[0][0], replay_input.sample_points[-1][0]

    def weighted_soc(socs: SocByBattery) -> float | None:
        values = dict(socs)
        if any(values.get(name) is None for name in specs):
            return None
        return sum(values[name] * spec.capacity_kwh for name, spec in specs.items()) / capacity

    result = SimResult(label, capacity_kwh=capacity, duration_h=(end - start) / 3600,
        soc_start=weighted_soc(run.soc_start_by_battery), soc_end=weighted_soc(run.soc_end_by_battery),
        opening_state=basis.opening, replay_terminal_value=basis.terminal,
        execution_kind=run.execution_kind, reasons=run.reasons, assumptions=run.assumptions,
        source_plan_digest=run.source_plan_digest, execution_plan_digest=run.execution_plan_digest,
        segments=run.segments, trace=run.trace, soc_start_by_battery=run.soc_start_by_battery,
        soc_end_by_battery=run.soc_end_by_battery, writes=run.writes, override_s=run.override_s,
        settlement_comparison_pending=plan.format_version == 2, terminal_value=None)
    result.soc_series.append((start, result.soc_start))
    reasons = [*basis.reasons, *run.reasons]
    financial_basis_known = not basis.reasons and (
        (basis.quotes is not None and basis.opening is not None and basis.terminal is not None)
        if plan.format_version == 2 else run.execution_kind == "modeled")
    quotes = {} if basis.quotes is None else {quote.start_ts: quote for quote in basis.quotes}
    # Only a segment's current accepted command may contribute override time.
    proposals, overrides = {}, {}
    recipe = replay_input.scenario.execution if replay_input.scenario is not None else None
    openings = recipe.opening_acceptances if isinstance(recipe, RecordedExecution) else ()
    sends = []
    for event in run.trace:
        if isinstance(event, ProposalEvent) and event.decision.write and event.command is not None:
            proposals.setdefault(event.command.command_id, event)
            if start <= event.at_ts < end:
                sends.append(event.at_ts)

    def overriding(battery: BatteryReplaySegment, at_ts: float) -> bool:
        command = battery.command
        if command is None:
            return False
        if command.command_id in overrides:
            return overrides[command.command_id]
        proposal = proposals.get(command.command_id)
        # Opening sends admitted by the fold precede in-window retries, even
        # at start. Acceptance itself is already established by this segment.
        from_opening = any(
            opening.command == command and opening.sent_at_ts is not None
            and opening.observation.application is not None
            and opening.sent_at_ts <= opening.observation.at_ts
            <= opening.observation.application.at_ts <= start for opening in openings)
        if proposal is not None and proposal.at_ts <= at_ts and not from_opening:
            overrides[command.command_id] = proposal.decision.override
            return proposal.decision.override
        # Opening acceptances have no proposal. Match the runtime's original
        # plan-at-issue comparison, not the profile at this later segment.
        interval = basis.execution_plan.interval_at(command.issued_at_ms / 1000)
        step = interval.battery(indices[command.battery_name])
        power = (profile_power_at(step, command.issued_at_ms / 1000)
                 if plan.format_version == 2 else step.ac_power_w)
        overrides[command.command_id] = command.setpoint_w != int(round(power))
        return overrides[command.command_id]

    complete, any_known = bool(run.segments), False
    day_energy: dict[str, Decimal | None] = {}
    with localcontext() as context:
        context.prec = 36
        for segment in run.segments:
            day = _day_key(segment.start_ts, timezone)
            metrics = result.days.setdefault(day, DayMetrics(day))
            dt_s = segment.end_ts - segment.start_ts
            batteries = segment.batteries
            site_known = segment.import_kwh is not None and segment.export_kwh is not None
            segment_complete = site_known and segment.house_w is not None and all(
                value is not None for battery in batteries for value in (
                    battery.modeled_ac_w, battery.charge_ac_kwh, battery.discharge_ac_kwh,
                    battery.dc_throughput_kwh, battery.cell_delta_kwh, battery.soc_begin, battery.soc_end))
            complete = complete and segment_complete
            any_known = any_known or site_known or any(b.modeled_ac_w is not None for b in batteries)
            for attribute, value in (
                ("import_kwh", segment.import_kwh), ("export_kwh", segment.export_kwh),
                ("throughput_kwh", _sum_known([b.dc_throughput_kwh for b in batteries])),
            ):
                setattr(metrics, attribute, _sum_known([getattr(metrics, attribute), value]))
            peak = segment.import_kwh * 3600000 / dt_s if site_known else None
            metrics.peak_import_w = (max(metrics.peak_import_w, peak)
                                    if metrics.peak_import_w is not None and peak is not None else None)
            active_overrides = [overriding(b, segment.start_ts) for b in batteries]
            override = (None if any(b.modeled_ac_w is None for b in batteries)
                        else dt_s if any(active_overrides) else 0.0)
            metrics.override_s = _sum_known([metrics.override_s, override])
            result.soc_series.append((segment.end_ts, weighted_soc(tuple(
                (battery.battery_name, battery.soc_end) for battery in batteries))))
            interval = basis.execution_plan.interval_at(segment.start_ts)
            energy_cost = wear_cost = None
            if segment_complete and financial_basis_known and interval is not None:
                quote = quotes.get(interval.start_ts)
                if plan.format_version == 1:
                    import_rate, export_rate = Decimal(str(interval.price_import)), Decimal(str(interval.price_export))
                elif quote is not None and quote.end_ts >= segment.end_ts:
                    import_rate, export_rate = quote.import_rate, quote.export_rate
                else:
                    reasons.append("replay_quote_coverage_mismatch")
                    import_rate = export_rate = None
                if import_rate is not None:
                    energy_cost = (Decimal(str(segment.import_kwh)) * import_rate
                                   - Decimal(str(segment.export_kwh)) * export_rate)
                    wear_cost = sum(specs[b.battery_name].cycle_cost * b.dc_throughput_kwh for b in batteries)
            previous = day_energy.get(day, Decimal(0))
            day_energy[day] = None if previous is None or energy_cost is None else previous + energy_cost
            metrics.wear_cost = _sum_known([metrics.wear_cost, wear_cost])
        for at_ts in sends:
            result.days[_day_key(at_ts, timezone)].writes += 1
        # Extra sends counted by the execution fold are opening evidence at the
        # left boundary; pre-window sends are already excluded by that fold.
        result.days[_day_key(start, timezone)].writes += run.writes - len(sends)
        for day, metrics in result.days.items():
            metrics.energy_cost = None if day_energy[day] is None else float(day_energy[day])
        for attribute in ("import_kwh", "export_kwh", "throughput_kwh"):
            setattr(result, attribute, _sum_known([getattr(day, attribute) for day in result.days.values()]))
        peaks = [day.peak_import_w for day in result.days.values()]
        result.peak_import_w = None if any(peak is None for peak in peaks) else max(peaks)
        complete = complete and result.soc_start is not None and result.soc_end is not None
        result.coverage_status = "complete" if complete else "partial" if any_known else "unknown"
        if complete and financial_basis_known and all(cost is not None for cost in day_energy.values()):
            result.energy_cost = float(sum(day_energy.values()))
            result.wear_cost = _sum_known([day.wear_cost for day in result.days.values()])
            opening, closing = dict(run.soc_start_by_battery), dict(run.soc_end_by_battery)
            if basis.terminal is not None:
                efficiencies = dict(basis.terminal.reference_efficiency_by_battery)
                result.terminal_value = float(sum(
                    (Decimal(str(closing[name])) - Decimal(str(opening[name])))
                    * Decimal(str(spec.capacity_kwh)) / Decimal(100)
                    * efficiencies[name] * basis.terminal.eur_per_ac_kwh for name, spec in specs.items()))
                result.label = f"reconstruction_base_tariffs: {label}"
            else:
                efficiencies = dict(replay_input.inverter_efficiency_by_battery)
                result.terminal_value = sum((closing[name] - opening[name]) * spec.kwh_per_percent
                    * plan.price_average * max(.01, spec.discharge_efficiency * efficiencies[name])
                    for name, spec in specs.items())
        else:
            result.energy_cost = result.wear_cost = None
            result.label = "unknown_reconstruction: " + (", ".join(dict.fromkeys(reasons)) or "comparison_unavailable")
    result.reasons = tuple(dict.fromkeys(reasons))
    return result


@dataclass
class Comparison:
    """Baseline against fast control."""

    baseline: SimResult
    fast: SimResult

    @property
    def saving(self) -> float | str:
        if self.baseline.settlement_comparison_pending or self.fast.settlement_comparison_pending:
            return "unknown_reconstruction: settlement_comparison_pending"
        base_cost, fast_cost = self.baseline.total_cost, self.fast.total_cost
        if base_cost is None or fast_cost is None:
            return "unknown_reconstruction: comparison_unavailable"
        return base_cost - fast_cost

    @property
    def saving_per_day(self) -> float | str:
        saving = self.saving
        if isinstance(saving, str):
            return saving
        days = max(1.0, self.baseline.duration_h / 24.0)
        return saving / days

    @property
    def extra_cycles(self) -> float | None:
        base, fast = self.baseline.cycles, self.fast.cycles
        return None if base is None or fast is None else fast - base

    def report(self) -> str:
        """Human readable summary, ready for the log or the console."""
        saving = self.saving
        metadata = [f"{result.label}: execution_kind={result.execution_kind}; coverage_status={result.coverage_status}"
                    + "".join(f"; {key}={value}" for key, value in result.assumptions)
                    for result in (self.baseline, self.fast)]
        if isinstance(saving, str):
            return "\n".join([saving, *metadata])
        base, fast = self.baseline, self.fast
        days = max(1.0, base.duration_h / 24.0)
        lines = [
            "",
            "=" * 72,
            *metadata,
            f"Fast control backtest over {base.duration_h / 24.0:.1f} dagen",
            "=" * 72,
            f"{'':30}{'zonder':>12}{'met':>12}{'verschil':>14}",
            "-" * 72,
            _row("Energiekosten (euro)", base.energy_cost, fast.energy_cost),
            _row("Slijtagekosten (euro)", base.wear_cost, fast.wear_cost),
            _row("Restwaarde accu (euro)",
                 None if base.terminal_value is None else -base.terminal_value,
                 None if fast.terminal_value is None else -fast.terminal_value),
            _row("Totale kosten (euro)", base.total_cost, fast.total_cost),
            "-" * 72,
            _row("Afname net (kWh)", base.import_kwh, fast.import_kwh),
            _row("Teruglevering (kWh)", base.export_kwh, fast.export_kwh),
            _row("Accu doorzet DC (kWh)", base.throughput_kwh, fast.throughput_kwh),
            _row("Volledige cycli", base.cycles, fast.cycles),
            _row("Piek afname (W)", base.peak_import_w, fast.peak_import_w),
            _row("Gemiddelde SoC (%)", base.average_soc, fast.average_soc),
            _row("Eind SoC (%)", base.soc_end, fast.soc_end),
            "-" * 72,
            f"{'Setpoint-schrijfacties':30}{base.writes:>12.0f}{fast.writes:>12.0f}",
            f"{'Override actief (uur)':30}"
            f"{_number(None if base.override_s is None else base.override_s / 3600, '.1f'):>12}"
            f"{_number(None if fast.override_s is None else fast.override_s / 3600, '.1f'):>12}",
            "=" * 72,
            f"Besparing: {self.saving:.2f} euro over {days:.1f} dagen "
            f"= {self.saving_per_day:.3f} euro/dag "
            f"= {self.saving_per_day * 365:.0f} euro/jaar",
            f"Extra cycli: {_number(self.extra_cycles, '+.2f')} over de periode "
            f"({_number(None if self.extra_cycles is None else self.extra_cycles / days * 365, '+.0f')} per jaar)",
        ]
        if fast.throughput_kwh is not None and base.throughput_kwh is not None and fast.throughput_kwh > base.throughput_kwh > 0:
            extra = fast.throughput_kwh - base.throughput_kwh
            lines.append(
                f"Besparing per extra kWh accudoorzet: "
                f"{self.saving / extra:.3f} euro/kWh"
            )
        lines.append("=" * 72)
        lines.append("")
        return "\n".join(lines)

    def daily_table(self) -> str:
        """Per day breakdown so outliers can be inspected."""
        saving = self.saving
        if isinstance(saving, str):
            return saving
        keys = sorted(set(self.baseline.days) | set(self.fast.days))
        lines = [
            f"{'datum':12}{'zonder':>10}{'met':>10}{'besparing':>11}"
            f"{'cycli':>8}{'piek W':>9}{'writes':>8}"
        ]
        lines.append("-" * 68)
        for key in keys:
            missing = DayMetrics(key, energy_cost=None, wear_cost=None,
                                 throughput_kwh=None, peak_import_w=None)
            base = self.baseline.days.get(key, missing)
            fast = self.fast.days.get(key, missing)
            base_cost = _sum_known([base.energy_cost, base.wear_cost])
            fast_cost = _sum_known([fast.energy_cost, fast.wear_cost])
            difference = None if base_cost is None or fast_cost is None else base_cost - fast_cost
            cycles = (None if fast.throughput_kwh is None else fast.throughput_kwh / (2.0 * self.fast.capacity_kwh)
                      if self.fast.capacity_kwh else 0.0)
            lines.append(
                f"{key:12}{_number(base_cost, '.2f'):>10}{_number(fast_cost, '.2f'):>10}"
                f"{_number(difference, '.3f'):>11}{_number(cycles, '.2f'):>8}"
                f"{_number(fast.peak_import_w, '.0f'):>9}{fast.writes:>8.0f}"
            )
        return "\n".join(lines)


def _number(value: float | None, spec: str) -> str:
    return "unknown" if value is None else format(value, spec)


def _row(label: str, left: float | None, right: float | None) -> str:
    difference = None if left is None or right is None else right - left
    return f"{label:30}{_number(left, '.2f'):>12}{_number(right, '.2f'):>12}{_number(difference, '+.3f'):>14}"


def aggregate_spec(specs: Sequence[BatterySpec], name: str = "totaal") -> BatterySpec:
    """Collapse several batteries into one for the simulation.

    The historical ``prognoses`` table only stores site totals, so the backtest
    necessarily works with an aggregated battery. Power limits and capacity add
    up, efficiencies and cycle cost are capacity weighted.
    """
    if len(specs) == 1:
        return specs[0]
    if not specs:
        raise ValueError("geen batterij om te simuleren")
    capacity = sum(s.capacity_kwh for s in specs) or 1.0

    def weighted(attribute: str) -> float:
        return sum(getattr(s, attribute) * s.capacity_kwh for s in specs) / capacity

    return BatterySpec(
        name=name,
        capacity_kwh=capacity,
        max_charge_w=sum(s.max_charge_w for s in specs),
        max_discharge_w=sum(s.max_discharge_w for s in specs),
        minimum_power_w=min(s.minimum_power_w for s in specs),
        soc_min=weighted("soc_min"),
        soc_max=weighted("soc_max"),
        cycle_cost=weighted("cycle_cost"),
        charge_efficiency=weighted("charge_efficiency"),
        discharge_efficiency=weighted("discharge_efficiency"),
    )


def simulate(
    plan: FastPlan,
    samples: Sequence[Sample],
    limits: PolicyLimits,
    soc_start: float,
    label: str,
    use_fast_layer: bool,
    inverter_efficiency: float = DEFAULT_INVERTER_EFFICIENCY,
    timezone: Optional[datetime.tzinfo] = None,
    *,
    replay_basis: ReplayBasis | None = None,
) -> SimResult:
    """Replay *samples* against *plan*, with or without the fast layer.

    Explicit replay evidence replaces today's tariffs/terminal assumptions. This
    remains a base-tariff physical reconstruction, NOT historical settlement or
    advantage: retain the exact settlement opening without inventing bonus or
    monthly-credit state transitions. F3 owns those trajectories/comparisons.
    No archive, DB or solver reads are performed here or in the tick loop.
    """
    if not plan.specs:
        raise ValueError("het plan bevat geen batterij")
    if replay_basis is not None:
        try:
            if type(replay_basis) is not ReplayBasis or plan.format_version != 2 or plan.financial_context is None:
                raise ValueError("replay_basis_required")
            replay_basis.check_context(plan.financial_context)
            if not samples or samples[0].timestamp != replay_basis.opening.asof_ts:
                raise ValueError("replay_opening_mismatch")
            if len(plan.intervals) != len(replay_basis.quotes):
                raise ValueError("replay_quote_coverage_mismatch")
            intervals, matched_quotes = [], []
            cursor = replay_basis.basis.planning_start_ts
            for interval, quote in zip(plan.intervals, replay_basis.quotes, strict=True):
                if (interval.start_ts != cursor
                        or (quote.start_ts, quote.end_ts) != (interval.start_ts, interval.end_ts)):
                    raise ValueError("replay_quote_coverage_mismatch")
                matched_quotes.append(quote)
                intervals.append(replace(interval, price_import=float(quote.import_rate),
                                         price_export=float(quote.export_rate)))
                cursor = interval.end_ts
            if cursor != replay_basis.basis.horizon_end_ts:
                raise ValueError("replay_quote_coverage_mismatch")
            if tuple(spec.name for spec in plan.specs) != tuple(
                    name for name, _ in replay_basis.terminal_value.reference_efficiency_by_battery):
                raise ValueError("replay_terminal_identity_mismatch")
            plan = replace(plan, intervals=intervals,
                           price_average=sum(float(q.import_rate) for q in matched_quotes) / len(matched_quotes))
        except (ValueError, TypeError, AttributeError):
            return SimResult("unknown_reconstruction: replay_basis_mismatch", energy_cost=None,
                             wear_cost=None, terminal_value=None)
        label = f"reconstruction_base_tariffs: {label}"
    if len(samples) < 2:
        raise ValueError("te weinig meetpunten om te simuleren")

    spec = plan.specs[0] if len(plan.specs) == 1 else aggregate_spec(plan.specs)
    if plan.format_version == 2:
        # Keep every profile edge when using the existing aggregate battery
        # simulator. Differently timed windows cannot be collapsed to one mean.
        sim_intervals = []
        for row in plan.intervals:
            # SOC belongs to the solver interval, not to an invented trajectory
            # inside it. Reuse the legacy aggregation convention for every slice.
            planned_soc = _collapse(row).battery(0)
            edges = sorted({row.start_ts, row.end_ts, *(
                edge for step in row.batteries for edge in (step.active_from_ts, step.stop_ts)
                if edge is not None)})
            for a, b in zip(edges, edges[1:], strict=False):
                power = sum(profile_power_at(step, a) for step in row.batteries)
                mean = sum(profile_energy_kwh(step, a, b) for step in row.batteries) * 3600000 / (b - a)
                sim_intervals.append(replace(row, start_ts=a, end_ts=b, batteries=[BatteryPlanStep(
                    ac_power_w=mean, mean_ac_power_w=mean, on_power_w=power,
                    active_from_ts=a, stop_ts=None,
                    soc_begin=planned_soc.soc_begin, soc_end=planned_soc.soc_end,
                )]))
        # House measurements remain left-held, just as on the legacy path;
        # split the time integration at profile and interval discontinuities.
        boundaries = sorted({edge for row in sim_intervals for edge in (row.start_ts, row.end_ts)})
        split_samples = []
        for sample, following in zip(samples, samples[1:], strict=False):
            split_samples.append(sample)
            split_samples.extend(replace(sample, timestamp=edge) for edge in boundaries
                                 if sample.timestamp < edge < following.timestamp)
        samples = [*split_samples, samples[-1]]
    else:
        sim_intervals = [_collapse(i) for i in plan.intervals]
    sim_plan = FastPlan(
        created_ts=plan.created_ts,
        interval_s=plan.interval_s,
        specs=[spec],
        intervals=sim_intervals,
        strategy=plan.strategy,
        price_average=plan.price_average,
        format_version=plan.format_version,
        financial_context=plan.financial_context,
    )

    battery = BatteryModel(spec, soc_start, inverter_efficiency)
    policy = FastControlPolicy(limits)
    state = ControllerState()
    result = SimResult(
        label=label,
        soc_start=soc_start,
        capacity_kwh=spec.capacity_kwh,
        opening_state=None if replay_basis is None else replay_basis.opening,
        replay_terminal_value=None if replay_basis is None else replay_basis.terminal_value,
    )

    previous_command = 0.0
    for index in range(len(samples) - 1):
        sample = samples[index]
        dt_s = samples[index + 1].timestamp - sample.timestamp
        if dt_s <= 0:
            continue
        dt_h = dt_s / 3600.0
        interval = sim_plan.interval_at(sample.timestamp)
        if interval is None:
            continue

        plan_w = (profile_power_at(interval.battery(0), sample.timestamp)
                  if plan.format_version == 2 else interval.battery(0).ac_power_w)
        if use_fast_layer:
            measurement = Measurement(
                timestamp=sample.timestamp,
                grid_w=sample.house_w + previous_command,
                batteries=[
                    BatteryMeasurement(
                        soc=battery.soc, power_w=previous_command, valid=True
                    )
                ],
                grid_valid=True,
            )
            day_key = _day_key(sample.timestamp, timezone)
            policy.account(state, sim_plan, measurement, day_key)
            decision = policy.decide(sim_plan, measurement, state)
            for proposal in decision.batteries:
                # Simulated helper writes always succeed, using the same
                # acceptance bookkeeping as the production runner.
                apply_accepted_command(state.battery(proposal.index), proposal)
            command = decision.batteries[0].setpoint_w
            wrote = decision.batteries[0].write
            overriding = decision.batteries[0].override
        else:
            command = (profile_energy_kwh(interval.battery(0), sample.timestamp,
                                         samples[index + 1].timestamp) * 1000 / dt_h
                       if plan.format_version == 2 else plan_w)
            wrote = abs(command - previous_command) > 1.0
            overriding = False

        actual_w, dc_kwh = battery.step(command, dt_h)
        previous_command = actual_w

        grid_w = sample.house_w + actual_w
        energy_cost = (
            interval.price_import * max(grid_w, 0.0)
            - interval.price_export * max(-grid_w, 0.0)
        ) * dt_h / 1000.0
        wear_cost = spec.cycle_cost * dc_kwh

        day = _day_key(sample.timestamp, timezone)
        metrics = result.days.setdefault(day, DayMetrics(day))
        metrics.energy_cost += energy_cost
        metrics.wear_cost += wear_cost
        metrics.import_kwh += max(grid_w, 0.0) * dt_h / 1000.0
        metrics.export_kwh += max(-grid_w, 0.0) * dt_h / 1000.0
        metrics.throughput_kwh += dc_kwh
        metrics.peak_import_w = max(metrics.peak_import_w, grid_w)
        metrics.writes += 1 if wrote else 0
        metrics.override_s += dt_s if overriding else 0.0

        result.energy_cost += energy_cost
        result.wear_cost += wear_cost
        result.import_kwh += max(grid_w, 0.0) * dt_h / 1000.0
        result.export_kwh += max(-grid_w, 0.0) * dt_h / 1000.0
        result.throughput_kwh += dc_kwh
        result.peak_import_w = max(result.peak_import_w, grid_w)
        result.writes += 1 if wrote else 0
        result.override_s += dt_s if overriding else 0.0
        result.soc_series.append((sample.timestamp, battery.soc))

    result.soc_end = battery.soc
    result.duration_h = (samples[-1].timestamp - samples[0].timestamp) / 3600.0
    # Value what is left in the cells, otherwise ending emptier looks cheaper.
    # Priced at what it would fetch on delivery: average tariff times the
    # efficiency of getting it back out to AC.
    valuation = plan.price_average * max(
        0.01, spec.discharge_efficiency * inverter_efficiency
    )
    if replay_basis is not None:
        with localcontext() as context:
            context.prec = 36
            # The archived reference already contains cell→bus→AC efficiency.
            capacity = sum(Decimal(str(s.capacity_kwh)) for s in plan.specs)
            valuation = float(replay_basis.terminal_value.eur_per_ac_kwh * sum(
                Decimal(str(s.capacity_kwh)) * efficiency for s, (_, efficiency)
                in zip(plan.specs, replay_basis.terminal_value.reference_efficiency_by_battery, strict=True)) / capacity)
    result.terminal_value = (
        (result.soc_end - result.soc_start) * spec.kwh_per_percent * valuation
    )
    return result


def compare(
    plan: FastPlan,
    samples: Sequence[Sample],
    limits: PolicyLimits,
    soc_start: float,
    inverter_efficiency: float = DEFAULT_INVERTER_EFFICIENCY,
    timezone: Optional[datetime.tzinfo] = None,
) -> Comparison:
    """Run both scenarios on identical inputs."""
    return Comparison(
        baseline=simulate(
            plan,
            samples,
            limits,
            soc_start,
            "day-ahead",
            use_fast_layer=False,
            inverter_efficiency=inverter_efficiency,
            timezone=timezone,
        ),
        fast=simulate(
            plan,
            samples,
            limits,
            soc_start,
            "day-ahead + fast",
            use_fast_layer=True,
            inverter_efficiency=inverter_efficiency,
            timezone=timezone,
        ),
    )


def _collapse(interval: PlanInterval) -> PlanInterval:
    """Merge the per battery plan steps of one interval into a single step."""
    if len(interval.batteries) == 1:
        return interval
    total_power = sum(b.ac_power_w for b in interval.batteries)
    if interval.batteries:
        soc_begin = sum(b.soc_begin for b in interval.batteries) / len(
            interval.batteries
        )
        soc_end = sum(b.soc_end for b in interval.batteries) / len(interval.batteries)
    else:
        soc_begin = soc_end = 0.0
    return PlanInterval(
        start_ts=interval.start_ts,
        end_ts=interval.end_ts,
        price_import=interval.price_import,
        price_export=interval.price_export,
        grid_w=interval.grid_w,
        house_w=interval.house_w,
        pv_w=interval.pv_w,
        batteries=[
            BatteryPlanStep(
                ac_power_w=total_power, soc_begin=soc_begin, soc_end=soc_end
            )
        ],
    )


def _day_key(timestamp: float, timezone: Optional[datetime.tzinfo]) -> str:
    return datetime.datetime.fromtimestamp(timestamp, timezone).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# synthetic scenarios
# ---------------------------------------------------------------------------


def synthetic_case(
    spec: BatterySpec,
    days: int = 1,
    interval_s: int = 900,
    step_s: int = 60,
    prices: Optional[Sequence[float]] = None,
    export_discount: float = 0.18,
    load_profile: Optional[Callable[[float], float]] = None,
    soc_start: float = 50.0,
    seed: int = 7,
) -> tuple[FastPlan, list[Sample], float]:
    """Build a self-contained scenario with a plausible Dutch price shape.

    Used by the tests and by ``da_fast.py demo`` so the machinery can be
    exercised without a populated Home Assistant database. The house load is a
    smooth baseload plus sharp appliance peaks that the plan cannot know about,
    which is exactly the error the fast layer exists to absorb.
    """
    import random

    rng = random.Random(seed)
    if prices is None:
        # A typical winter day: cheap at night, a solar dip midday, an evening peak.
        prices = [
            0.22, 0.20, 0.19, 0.19, 0.20, 0.24, 0.30, 0.34,
            0.31, 0.26, 0.21, 0.17, 0.14, 0.13, 0.15, 0.20,
            0.28, 0.38, 0.46, 0.44, 0.36, 0.30, 0.26, 0.23,
        ]

    start = int(
        datetime.datetime.combine(
            datetime.date.today() - datetime.timedelta(days=days),
            datetime.time(0, 0),
        ).timestamp()
    )
    steps_per_hour = 3600 // interval_s
    total_intervals = days * 24 * steps_per_hour

    def price_at(index: int) -> float:
        return prices[(index // steps_per_hour) % 24]

    def baseload(timestamp: float) -> float:
        hour = datetime.datetime.fromtimestamp(timestamp).hour
        night = 250.0
        day = 450.0
        evening = 700.0
        if hour < 6:
            return night
        if hour < 17:
            return day
        return evening

    def pv(timestamp: float) -> float:
        hour = (
            datetime.datetime.fromtimestamp(timestamp).hour
            + datetime.datetime.fromtimestamp(timestamp).minute / 60.0
        )
        if not 8.0 <= hour <= 18.0:
            return 0.0
        return 3200.0 * math.sin(math.pi * (hour - 8.0) / 10.0) ** 2

    if load_profile is None:

        def load_profile(timestamp: float) -> float:
            hour = datetime.datetime.fromtimestamp(timestamp).hour
            minute = datetime.datetime.fromtimestamp(timestamp).minute
            extra = 0.0
            # Kettle and coffee in the morning.
            if hour == 7 and 10 <= minute < 16:
                extra += 2200.0
            # Oven during the evening peak, the expensive hour.
            if hour == 18 and 5 <= minute < 45:
                extra += 3200.0
            # Induction hob, short and sharp.
            if hour == 18 and 20 <= minute < 30:
                extra += 2400.0
            # Tumble dryer mid afternoon, cheap hour but PV dependent.
            if hour == 14 and 0 <= minute < 50:
                extra += 1800.0
            return extra + rng.gauss(0.0, 60.0)

    # Build the plan from the *forecast*: baseload and PV only. The appliance
    # peaks are deliberately invisible to it, just as they are in reality.
    intervals: list[PlanInterval] = []
    soc = soc_start
    for index in range(total_intervals):
        interval_start = start + index * interval_s
        price_import = price_at(index)
        price_export = max(0.0, price_import - export_discount)
        forecast_house = baseload(interval_start) - pv(interval_start)
        # Charge when cheap and there is room, discharge when expensive.
        if price_import <= 0.18 and soc < spec.soc_max - 5:
            battery_w = min(spec.max_charge_w, 3000.0)
        elif price_import >= 0.36 and soc > spec.soc_min + 5:
            battery_w = -min(spec.max_discharge_w, max(0.0, forecast_house))
        else:
            battery_w = -max(0.0, min(forecast_house, spec.max_discharge_w)) * (
                1.0 if soc > spec.soc_min + 10 else 0.0
            )
        soc_begin = soc
        dt_h = interval_s / 3600.0
        if battery_w >= 0:
            soc += (
                battery_w * dt_h * spec.charge_efficiency / 1000.0
            ) / spec.kwh_per_percent
        else:
            soc -= (
                -battery_w * dt_h / spec.discharge_efficiency / 1000.0
            ) / spec.kwh_per_percent
        soc = max(spec.soc_min, min(spec.soc_max, soc))
        intervals.append(
            PlanInterval(
                start_ts=interval_start,
                end_ts=interval_start + interval_s,
                price_import=price_import,
                price_export=price_export,
                grid_w=forecast_house + battery_w,
                house_w=forecast_house,
                pv_w=pv(interval_start),
                batteries=[
                    BatteryPlanStep(
                        ac_power_w=battery_w, soc_begin=soc_begin, soc_end=soc
                    )
                ],
            )
        )

    plan = FastPlan(
        created_ts=start,
        interval_s=interval_s,
        specs=[spec],
        intervals=intervals,
        strategy="minimize cost",
        price_average=sum(prices) / len(prices),
    )

    samples = [
        Sample(
            timestamp=start + offset,
            house_w=baseload(start + offset)
            + load_profile(start + offset)
            - pv(start + offset),
        )
        for offset in range(0, total_intervals * interval_s + step_s, step_s)
    ]
    return plan, samples, soc_start


# ---------------------------------------------------------------------------
# historical data
# ---------------------------------------------------------------------------


@dataclass
class HistoryWindow:
    """What the loader produced, plus what it had to compromise on."""

    plan: FastPlan
    samples: list[Sample]
    soc_start: float
    resolution_s: int
    source: str
    warnings: list[str] = field(default_factory=list)


class HistoryLoader:
    """Pulls the backtest inputs out of the two databases DAO already has.

    Prices and the historical plans come from DAO's own tables, the measured
    power comes from the Home Assistant recorder. Both recorder layouts are
    supported: the modern ``states_meta`` join and the legacy ``entity_id``
    column.
    """

    def __init__(self, db_ha, db_da, report, time_zone: Optional[str] = None):
        self.db_ha = db_ha
        self.db_da = db_da
        self.report = report
        self.time_zone = time_zone

    # -- recorder --------------------------------------------------------

    def _series(
        self, entity_id: str, start: datetime.datetime, end: datetime.datetime
    ):
        """Measured series for one entity as a ``(timestamp, value)`` list."""
        import pandas as pd
        from sqlalchemy import Table, and_, select

        metadata = self.db_ha.metadata
        engine = self.db_ha.engine
        states = Table("states", metadata, autoload_with=engine)
        start_ts = start.timestamp()
        end_ts = end.timestamp()

        if "metadata_id" in states.c:
            states_meta = Table("states_meta", metadata, autoload_with=engine)
            query = (
                select(states.c.last_updated_ts, states.c.state)
                .select_from(
                    states.join(
                        states_meta,
                        states.c.metadata_id == states_meta.c.metadata_id,
                    )
                )
                .where(
                    and_(
                        states_meta.c.entity_id == entity_id,
                        states.c.last_updated_ts >= start_ts,
                        states.c.last_updated_ts < end_ts,
                    )
                )
                .order_by(states.c.last_updated_ts)
            )
        else:
            query = (
                select(states.c.last_updated_ts, states.c.state)
                .where(
                    and_(
                        states.c.entity_id == entity_id,
                        states.c.last_updated_ts >= start_ts,
                        states.c.last_updated_ts < end_ts,
                    )
                )
                .order_by(states.c.last_updated_ts)
            )

        with engine.connect() as connection:
            rows = connection.execute(query).fetchall()
        frame = pd.DataFrame(rows, columns=["ts", "value"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame.dropna()

    def _series_on_grid(
        self,
        entity_id: str,
        scale: float,
        stamps: Sequence[int],
        start: datetime.datetime,
        end: datetime.datetime,
        chunk_days: int = CHUNK_DAYS,
    ):
        """Read one entity onto the simulation grid, a few days at a time.

        A P1 meter that updates every second produces roughly a million
        recorder rows per fortnight. Pulling that into one DataFrame is
        pointless -- the grid it ends up on has a few thousand points -- and on
        a Home Assistant Yellow or a Raspberry Pi it is enough to exhaust
        memory. Reading per chunk and downsampling immediately keeps the peak
        footprint at one chunk, at the cost of a few extra queries.

        The last reading of a chunk is carried into the next one so the forward
        fill stays correct across the boundary.
        """
        import pandas as pd

        total = None
        carry = None
        rows_read = 0
        for chunk_start, chunk_end in _iter_chunks(start, end, chunk_days):
            sub = [
                s
                for s in stamps
                if chunk_start.timestamp() <= s < chunk_end.timestamp()
            ]
            frame = self._series(entity_id, chunk_start, chunk_end)
            rows_read += len(frame)
            if carry is not None:
                seed = pd.DataFrame(
                    {"ts": [chunk_start.timestamp()], "value": [carry]}
                )
                frame = pd.concat([seed, frame], ignore_index=True)
            if not frame.empty:
                carry = float(frame["value"].iloc[-1])
            if not sub:
                continue
            series = _on_grid(frame, scale, sub)
            if series is None:
                series = pd.Series(index=sub, dtype="float64")
            total = series if total is None else pd.concat([total, series])

        logging.debug(
            f"Backtest: {entity_id} leverde {rows_read} recorder-rijen, "
            f"teruggebracht tot {len(stamps)} rasterpunten"
        )
        if total is None or total.isna().all():
            return None
        return total.reindex(stamps)

    def _sum_entities(
        self,
        entities: Sequence[tuple[str, float]],
        stamps: Sequence[int],
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> Optional[list[Optional[float]]]:
        """Scale, resample and sum several entities onto the simulation grid."""
        import pandas as pd

        total = None
        for entity_id, scale in entities:
            series = self._series_on_grid(entity_id, scale, stamps, start, end)
            if series is None:
                return None
            total = series if total is None else total.add(series, fill_value=0.0)
        if total is None:
            return None
        return [None if pd.isna(value) else float(value) for value in total]

    def _sample_gap(self, entity_id: str, start: datetime.datetime) -> float:
        """Median seconds between two readings, measured on one sample day."""
        frame = self._series(entity_id, start, start + datetime.timedelta(days=1))
        if frame is None or frame.empty:
            return math.inf
        return _median_gap(frame["ts"].tolist())

    # -- plan ------------------------------------------------------------

    def _plan(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        spec: BatterySpec,
        interval_s: int,
    ) -> FastPlan:
        """Rebuild the historical day-ahead plans from the prognoses table."""
        import pandas as pd

        codes = ["bat_in", "bat_out", "cons", "prod", "soc"]
        frames = {}
        for code in codes:
            frame = self.db_da.get_column_data("prognoses", code, start, end)
            frames[code] = frame.set_index("utc")["value"] if len(frame) else None

        prices = self.report.get_price_data(
            start, end, "1hour" if interval_s >= 3600 else "15min"
        )
        prices["ts"] = prices["time"].apply(_local_epoch)
        price_by_ts = prices.set_index("ts")[["da_cons", "da_prod"]]

        stamps = sorted(
            set().union(*[set(f.index) for f in frames.values() if f is not None])
        )
        hour_fraction = interval_s / 3600.0
        intervals: list[PlanInterval] = []
        for stamp in stamps:
            def value(code: str) -> float:
                series = frames.get(code)
                if series is None or stamp not in series.index:
                    return 0.0
                raw = series.loc[stamp]
                raw = raw.iloc[0] if hasattr(raw, "iloc") else raw
                return 0.0 if pd.isna(raw) else float(raw)

            price_row = _nearest(price_by_ts, stamp, interval_s)
            if price_row is None:
                continue
            battery_w = (value("bat_in") - value("bat_out")) * 1000.0 / hour_fraction
            grid_w = (value("cons") - value("prod")) * 1000.0 / hour_fraction
            soc_value = value("soc")
            intervals.append(
                PlanInterval(
                    start_ts=int(stamp),
                    end_ts=int(stamp + interval_s),
                    price_import=float(price_row[0]),
                    price_export=float(price_row[1]),
                    grid_w=grid_w,
                    house_w=grid_w - battery_w,
                    batteries=[
                        BatteryPlanStep(
                            ac_power_w=battery_w,
                            soc_begin=soc_value,
                            soc_end=soc_value,
                        )
                    ],
                )
            )

        average = (
            sum(i.price_import for i in intervals) / len(intervals) if intervals else 0.0
        )
        return FastPlan(
            created_ts=int(start.timestamp()),
            interval_s=interval_s,
            specs=[spec],
            intervals=intervals,
            price_average=average,
        )

    # -- the public entry point ------------------------------------------

    def load(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        spec: BatterySpec,
        grid_entities: Sequence[tuple[str, float]],
        battery_entities: Sequence[tuple[str, float]],
        soc_entity: Optional[str],
        interval_s: int,
        step_s: int = 60,
    ) -> HistoryWindow:
        """Assemble a backtest window.

        ``grid_entities`` and ``battery_entities`` are ``(entity_id, scale)``
        pairs, where the scale carries both the unit conversion and the sign
        convention, so a positive/negative sensor pair is just two entries.
        """
        warnings: list[str] = []
        stamps = list(range(int(start.timestamp()), int(end.timestamp()), step_s))

        grid_w = self._sum_entities(grid_entities, stamps, start, end)
        if grid_w is None:
            raise RuntimeError(
                "geen netvermogen in de Home Assistant recorder gevonden voor de "
                "opgegeven periode; controleer de entiteit en de bewaartermijn "
                "(purge_keep_days)"
            )

        battery_w = None
        if battery_entities:
            battery_w = self._sum_entities(battery_entities, stamps, start, end)
            if battery_w is None:
                warnings.append(
                    "accuvermogen ontbreekt in de recorder; het plan wordt gebruikt "
                    "als schatting van wat de accu deed"
                )
        else:
            warnings.append(
                "geen accuvermogen-sensor geconfigureerd; het plan wordt gebruikt "
                "als schatting van wat de accu deed"
            )

        plan = self._plan(start, end, spec, interval_s)
        if not plan.intervals:
            raise RuntimeError(
                "geen historische planning in de prognoses-tabel voor deze periode"
            )

        samples: list[Sample] = []
        for position, stamp in enumerate(stamps):
            measured_grid = grid_w[position]
            if measured_grid is None:
                continue
            if battery_w is not None and battery_w[position] is not None:
                measured_battery = battery_w[position]
            else:
                interval = plan.interval_at(stamp)
                measured_battery = interval.battery(0).ac_power_w if interval else 0.0
            samples.append(Sample(stamp, measured_grid - measured_battery))

        soc_start = 50.0
        if soc_entity:
            # Only the opening value is needed, so do not drag in the whole
            # window of state changes.
            soc_frame = self._series(
                soc_entity, start, min(start + datetime.timedelta(days=1), end)
            )
            if soc_frame is not None and not soc_frame.empty:
                soc_start = float(soc_frame.iloc[0]["value"])
            else:
                warnings.append("SoC-historie ontbreekt; gestart op 50%")

        median_gap = (
            self._sample_gap(grid_entities[0][0], start) if grid_entities else math.inf
        )
        if median_gap > 300:
            warnings.append(
                f"de recorder levert gemiddeld maar een meting per "
                f"{median_gap:.0f} s; korte verbruikspieken zijn daarmee "
                f"onzichtbaar en de besparing wordt onderschat"
            )

        return HistoryWindow(
            plan=plan,
            samples=samples,
            soc_start=soc_start,
            resolution_s=step_s,
            source="recorder states",
            warnings=warnings,
        )


def _on_grid(frame, scale: float, stamps: Sequence[int]):
    """Forward fill one sparse event series onto a uniform grid.

    Recorder rows only exist when a sensor changes value, so between two rows
    the previous reading still holds.
    """
    import pandas as pd

    if frame is None or frame.empty:
        return None
    series = pd.Series(frame["value"].to_numpy() * scale, index=frame["ts"].to_numpy())
    series = series.groupby(level=0).last().sort_index()
    return series.reindex(series.index.union(stamps)).ffill().reindex(stamps)


def _sum_on_grid(entries, stamps: Sequence[int]) -> Optional[list[Optional[float]]]:
    """Scale, resample and sum several measurement series onto one grid."""
    import pandas as pd

    total = None
    for _entity_id, scale, frame in entries:
        series = _on_grid(frame, scale, stamps)
        if series is None:
            return None
        total = series if total is None else total.add(series, fill_value=0.0)
    if total is None:
        return None
    return [None if pd.isna(value) else float(value) for value in total]


def _local_epoch(value) -> float:
    """Epoch seconds for a naive datetime that denotes *local* time.

    DAO's price frame holds naive local datetimes. ``pandas.Timestamp`` would
    read a naive value as UTC and silently shift everything by the local offset,
    so convert to a plain ``datetime`` first, which uses the local zone.
    """
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    return value.timestamp()


def _nearest(frame, stamp: float, tolerance: int):
    """Row whose index is closest to *stamp*, within *tolerance* seconds."""
    if frame.empty:
        return None
    differences = (frame.index.values - stamp)
    position = abs(differences).argmin()
    if abs(differences[position]) > tolerance:
        return None
    return frame.iloc[position].values


def _median_gap(stamps: Sequence[float]) -> float:
    if len(stamps) < 3:
        return math.inf
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]) if b > a)
    return gaps[len(gaps) // 2] if gaps else math.inf


def log_comparison(comparison: Comparison, show_daily: bool = True) -> None:
    """Print the summary through the logging machinery DAO already sets up."""
    for line in comparison.report().splitlines():
        logging.info(line)
    if show_daily:
        for line in comparison.daily_table().splitlines():
            logging.info(line)
