"""Explicit, transport-free replay inputs; evidence is not execution authority."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Literal

from .command import (
    ActuationReceipt,
    BarrierReceipt,
    BatteryCommand,
    ClockSample,
    Deadline,
    OwnerBinding,
    RevokeRequest,
    plan_digest,
)
from .execution import ExecutionRefused, _bounded_deadline, _command_end_ms
from .plan import NO_STOP_SENTINEL, FastPlan, profile_power_at
from .policy import (
    POWER_EPS,
    AcceptedCommand,
    BatteryDecision,
    ControllerState,
    PolicyLimits,
    apply_execution_outcome,
    current_execution_command,
)
from .receiver import (
    ReceiverState,
    classify_bind,
    classify_effect,
    classify_revoke,
    classify_send,
    receipt_matches_command,
)

Reasons = tuple[str, ...]
SamplePoints = tuple[tuple[float, float], ...]
SocByBattery = tuple[tuple[str, float | None], ...]
EfficiencyByBattery = tuple[tuple[str, float], ...]
Assumptions = tuple[tuple[str, str], ...]


def _number(value, *, minimum=None, maximum=None) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError("replay_value_invalid")


def _text(value, code="replay_value_invalid") -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(code)


def _choice(value, choices, code="replay_value_invalid") -> None:
    if type(value) is not str or value not in choices:
        raise ValueError(code)


def _boolean(value) -> None:
    if type(value) is not bool:
        raise ValueError("replay_value_invalid")


def _typed(value, classes, *, optional=False) -> None:
    if optional and value is None:
        return
    if type(value) not in (classes if type(classes) is tuple else (classes,)):
        raise TypeError("replay_value_invalid")


def _tuple(value) -> tuple:
    if type(value) not in (tuple, list):
        raise TypeError("replay_value_invalid")
    return tuple(value)


def _event_key(event) -> tuple[float, int]:
    _number(event.at_ts)
    if type(event.order) is not int:
        raise ValueError("replay_event_order_invalid")
    return event.at_ts, event.order


def _ordered(events) -> None:
    previous = None
    for event in events:
        key = _event_key(event)
        if previous is not None and key <= previous:
            raise ValueError("replay_event_order_invalid")
        previous = key


def _samples(value) -> SamplePoints:
    points = _tuple(value)
    if len(points) < 2:
        raise ValueError("te weinig meetpunten om te simuleren")
    result = []
    for point in points:
        pair = _tuple(point)
        if len(pair) != 2:
            raise ValueError("replay_value_invalid")
        at_ts, house_w = pair
        _number(at_ts)
        _number(house_w)
        if result and at_ts <= result[-1][0]:
            raise ValueError("replay_time_order_invalid")
        result.append(pair)
    return tuple(result)


def _soc(value) -> None:
    if value is not None:
        _number(value, minimum=0, maximum=100)


def _efficiency(value) -> None:
    _number(value, minimum=0, maximum=1)
    if value == 0:
        raise ValueError("replay_value_invalid")


def _battery_values(value, *, soc=False) -> tuple:
    result = []
    names = set()
    for pair in _tuple(value):
        pair = _tuple(pair)
        if len(pair) != 2:
            raise ValueError("replay_battery_mapping_invalid")
        name, number = pair
        _text(name, "replay_battery_mapping_invalid")
        if name in names:
            raise ValueError("replay_duplicate_identity")
        names.add(name)
        if soc:
            _soc(number)
        else:
            _efficiency(number)
        result.append(pair)
    return tuple(result)


def _decision(value: BatteryDecision) -> None:
    _typed(value, BatteryDecision)
    if type(value.index) is not int or value.index < 0:
        raise ValueError("replay_battery_mapping_invalid")
    _text(value.name, "replay_battery_mapping_invalid")
    for name in ("setpoint_w", "plan_w", "benefit_eur_h", "storage_value",
                 "budget_used_kwh", "daily_used_kwh"):
        _number(getattr(value, name))
    _soc(value.soc)
    for name in ("write", "override", "restores"):
        _boolean(getattr(value, name))
    _typed(value.reason, str)
    for name in ("mode", "stop_inverter", "actuation_status"):
        _typed(getattr(value, name), str, optional=True)
    _typed(value.accepted, AcceptedCommand, optional=True)


@dataclass(frozen=True, slots=True)
class ReplayPhysics:
    version: Literal["battery_model_v1"]
    house_hold_s: float
    stop_response: Literal["unknown", "ideal_programmed_stop"]
    equal_initial_soc: bool = False

    def __post_init__(self) -> None:
        _choice(self.version, ("battery_model_v1",))
        _number(self.house_hold_s, minimum=0)
        if self.house_hold_s == 0:
            raise ValueError("replay_value_invalid")
        _choice(self.stop_response, ("unknown", "ideal_programmed_stop"))
        _boolean(self.equal_initial_soc)


@dataclass(frozen=True, slots=True)
class ReplayAttemptTiming:
    anchor_ts: float
    anchor_clock: ClockSample
    receive_deadline: Deadline
    apply_deadline: Deadline

    def __post_init__(self) -> None:
        _number(self.anchor_ts)
        _typed(self.anchor_clock, ClockSample)
        _typed(self.receive_deadline, Deadline)
        _typed(self.apply_deadline, Deadline)


@dataclass(frozen=True, slots=True)
class ReplayApplication:
    at_ts: float
    order: int
    clock: ClockSample
    outcome: Literal["accepted", "unknown", "unconfirmed"]

    def __post_init__(self) -> None:
        _event_key(self)
        _typed(self.clock, ClockSample)
        _choice(self.outcome, ("accepted", "unknown", "unconfirmed"))


@dataclass(frozen=True, slots=True)
class ProposalEvent:
    at_ts: float
    order: int
    decision: BatteryDecision
    command: BatteryCommand | None
    timing: ReplayAttemptTiming | None

    def __post_init__(self) -> None:
        _event_key(self)
        _decision(self.decision)
        _typed(self.command, BatteryCommand, optional=True)
        _typed(self.timing, ReplayAttemptTiming, optional=True)
        if not self.decision.write and (self.command is not None or self.timing is not None):
            raise ValueError("replay_value_invalid")
        object.__setattr__(self, "decision", deepcopy(self.decision))


@dataclass(frozen=True, slots=True)
class ReceiptEvent:
    at_ts: float
    order: int
    receipt: ActuationReceipt
    received_clock: ClockSample | None
    application: ReplayApplication | None

    def __post_init__(self) -> None:
        _event_key(self)
        _typed(self.receipt, ActuationReceipt)
        _typed(self.received_clock, ClockSample, optional=True)
        _typed(self.application, ReplayApplication, optional=True)
        if self.application is not None and _event_key(self.application) <= _event_key(self):
            raise ValueError("replay_event_order_invalid")


@dataclass(frozen=True, slots=True)
class StopRequestEvent:
    at_ts: float
    order: int
    request: RevokeRequest

    def __post_init__(self) -> None:
        _event_key(self)
        _typed(self.request, RevokeRequest)


@dataclass(frozen=True, slots=True)
class BarrierEvent:
    at_ts: float
    order: int
    receipt: BarrierReceipt

    def __post_init__(self) -> None:
        _event_key(self)
        _typed(self.receipt, BarrierReceipt)


@dataclass(frozen=True, slots=True)
class BindingEvent:
    at_ts: float
    order: int
    binding: OwnerBinding

    def __post_init__(self) -> None:
        _event_key(self)
        _typed(self.binding, OwnerBinding)


ReplayEvent = ProposalEvent | ReceiptEvent | StopRequestEvent | BarrierEvent | BindingEvent


@dataclass(frozen=True, slots=True)
class OpeningAcceptance:
    command: BatteryCommand
    timing: ReplayAttemptTiming | None
    observation: ReceiptEvent
    sent_at_ts: float | None

    def __post_init__(self) -> None:
        _typed(self.command, BatteryCommand)
        _typed(self.timing, ReplayAttemptTiming, optional=True)
        _typed(self.observation, ReceiptEvent)
        if self.sent_at_ts is not None:
            _number(self.sent_at_ts)


@dataclass(frozen=True, slots=True)
class ExecutionCoverage:
    battery_name: str
    start_ts: float
    end_ts: float

    def __post_init__(self) -> None:
        _text(self.battery_name, "replay_battery_mapping_invalid")
        _number(self.start_ts)
        _number(self.end_ts)
        if self.start_ts >= self.end_ts:
            raise ValueError("replay_window_invalid")


@dataclass(frozen=True, slots=True)
class RecordedExecution:
    installation_id: str
    receiver_generation: str
    opening_binding: OwnerBinding | None
    opening_acceptances: tuple[OpeningAcceptance, ...]
    opening_stop: tuple[RevokeRequest, BarrierReceipt] | None
    coverage: tuple[ExecutionCoverage, ...]
    events: tuple[ReplayEvent, ...]

    def __post_init__(self) -> None:
        _text(self.installation_id)
        _text(self.receiver_generation)
        _typed(self.opening_binding, OwnerBinding, optional=True)
        openings, coverage, events = (_tuple(self.opening_acceptances), _tuple(self.coverage), _tuple(self.events))
        for opening in openings:
            _typed(opening, OpeningAcceptance)
        if self.opening_stop is not None:
            stop = _tuple(self.opening_stop)
            if len(stop) != 2:
                raise ValueError("replay_value_invalid")
            _typed(stop[0], RevokeRequest)
            _typed(stop[1], BarrierReceipt)
            object.__setattr__(self, "opening_stop", stop)
        spans = {}
        for span in coverage:
            _typed(span, ExecutionCoverage)
            previous = spans.setdefault(span.battery_name, [])
            if any(span.start_ts < end and start < span.end_ts for start, end in previous):
                raise ValueError("replay_window_invalid")
            previous.append((span.start_ts, span.end_ts))
        for event in events:
            _typed(event, (ProposalEvent, ReceiptEvent, StopRequestEvent, BarrierEvent, BindingEvent))
        _ordered(events)
        # Applications may interleave later events; never reorder the supplied trace.
        keys = set()
        for event in (*events, *(opening.observation for opening in openings)):
            entries = (event, event.application) if type(event) is ReceiptEvent and event.application else (event,)
            for entry in entries:
                key = _event_key(entry)
                if key in keys:
                    raise ValueError("replay_event_order_invalid")
                keys.add(key)
        object.__setattr__(self, "opening_acceptances", openings)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "events", deepcopy(events))


@dataclass(frozen=True, slots=True)
class ModelResponse:
    tick_index: int
    source: Literal["plan", "fast"]
    battery_name: str
    effect: Literal["complete", "none", "partial", "unknown"]
    delay_s: float

    def __post_init__(self) -> None:
        if type(self.tick_index) is not int or self.tick_index < 0:
            raise ValueError("replay_model_selector_invalid")
        _choice(self.source, ("plan", "fast"), "replay_model_selector_invalid")
        _text(self.battery_name, "replay_model_selector_invalid")
        _choice(self.effect, ("complete", "none", "partial", "unknown"))
        _number(self.delay_s, minimum=0)


@dataclass(frozen=True, slots=True)
class ModelStop:
    at_ts: float
    order: int
    queues_fenced: bool
    safe_state_confirmed: bool

    def __post_init__(self) -> None:
        _event_key(self)
        _boolean(self.queues_fenced)
        _boolean(self.safe_state_confirmed)


@dataclass(frozen=True, slots=True)
class ModelBind:
    at_ts: float
    order: int

    def __post_init__(self) -> None:
        _event_key(self)


@dataclass(frozen=True, slots=True)
class ModeledExecution:
    initial_execution: Literal["stopped", "unknown"]
    default_effect: Literal["complete", "none", "partial", "unknown"] | None
    response_delay_s: float = 0
    prepare_delay_s: float = 0
    application_delay_s: float = 0
    responses: tuple[ModelResponse, ...] = ()
    lifecycle: tuple[ModelStop | ModelBind, ...] = ()

    def __post_init__(self) -> None:
        _choice(self.initial_execution, ("stopped", "unknown"))
        if self.default_effect is not None:
            _choice(self.default_effect, ("complete", "none", "partial", "unknown"))
        for delay in (self.response_delay_s, self.prepare_delay_s, self.application_delay_s):
            _number(delay, minimum=0)
        responses, lifecycle = _tuple(self.responses), _tuple(self.lifecycle)
        selectors = set()
        for response in responses:
            _typed(response, ModelResponse)
            selector = (response.tick_index, response.source, response.battery_name)
            if selector in selectors:
                raise ValueError("replay_model_selector_invalid")
            selectors.add(selector)
        for event in lifecycle:
            _typed(event, (ModelStop, ModelBind))
        _ordered(lifecycle)
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "lifecycle", lifecycle)


@dataclass(frozen=True, slots=True)
class ReplayScenario:
    scenario_id: str
    execution: RecordedExecution | ModeledExecution
    physics: ReplayPhysics

    def __post_init__(self) -> None:
        _text(self.scenario_id)
        _typed(self.execution, (RecordedExecution, ModeledExecution))
        _typed(self.physics, ReplayPhysics)
        object.__setattr__(self, "execution", deepcopy(self.execution))


@dataclass(frozen=True, slots=True)
class ReplayInput:
    scenario: ReplayScenario | None
    sample_points: SamplePoints
    soc_by_battery: SocByBattery
    inverter_efficiency_by_battery: EfficiencyByBattery
    reasons: Reasons

    def __post_init__(self) -> None:
        _typed(self.scenario, ReplayScenario, optional=True)
        points = _samples(self.sample_points)
        socs = _battery_values(self.soc_by_battery, soc=True)
        efficiencies = _battery_values(self.inverter_efficiency_by_battery)
        if tuple(name for name, _ in socs) != tuple(name for name, _ in efficiencies):
            raise ValueError("replay_battery_mapping_invalid")
        reasons = _tuple(self.reasons)
        for reason in reasons:
            _text(reason)
        object.__setattr__(self, "scenario", deepcopy(self.scenario))
        object.__setattr__(self, "sample_points", points)
        object.__setattr__(self, "soc_by_battery", socs)
        object.__setattr__(self, "inverter_efficiency_by_battery", efficiencies)
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(reasons)))


def validate_replay_input(*, plan: FastPlan, sample_points: SamplePoints,
                          soc_start: float | Mapping[str, float | None],
                          inverter_efficiency: float | Mapping[str, float],
                          scenario: ReplayScenario | None) -> ReplayInput:
    """Validate shape and retain missing evidence, without proving any execution."""
    _typed(plan, FastPlan)
    if not plan.specs:
        raise ValueError("het plan bevat geen batterij")
    names = tuple(spec.name for spec in plan.specs)
    for name in names:
        _text(name, "replay_battery_mapping_invalid")
    if len(set(names)) != len(names):
        raise ValueError("replay_duplicate_identity")
    points = _samples(sample_points)
    _typed(scenario, ReplayScenario, optional=True)
    execution = scenario.execution if scenario is not None else None
    reasons = []
    if execution is None:
        reasons.append("execution_input_missing")
    elif type(execution) is RecordedExecution:
        for span in execution.coverage:
            if span.battery_name not in names:
                raise ValueError("replay_battery_mapping_invalid")
            if span.start_ts < points[0][0] or span.end_ts > points[-1][0]:
                raise ValueError("replay_window_invalid")
        if not execution.coverage or not (execution.opening_acceptances or execution.opening_stop or execution.events):
            reasons.append("execution_history_missing")
    else:
        for response in execution.responses:
            if response.battery_name not in names:
                raise ValueError("replay_model_selector_invalid")
        # Which explicit selectors are unused/missing depends on actual writes.
        if execution.default_effect is None and not execution.responses:
            reasons.append("execution_model_missing")

    if isinstance(soc_start, Mapping):
        supplied_soc = dict(soc_start)
        if supplied_soc.keys() - set(names):
            raise ValueError("replay_battery_mapping_invalid")
        socs = tuple((name, supplied_soc.get(name)) for name in names)
    else:
        _number(soc_start, minimum=0, maximum=100)
        if len(names) == 1 or (type(execution) is ModeledExecution and scenario.physics.equal_initial_soc):
            socs = tuple((name, soc_start) for name in names)
        else:
            socs = tuple((name, None) for name in names)
            reasons.append("replay_soc_by_battery_required")
    if any(value is None for _, value in socs):
        reasons.append("replay_soc_missing")

    if isinstance(inverter_efficiency, Mapping):
        supplied_efficiencies = dict(inverter_efficiency)
        if supplied_efficiencies.keys() != set(names):
            raise ValueError("replay_battery_mapping_invalid")
        efficiencies = tuple((name, supplied_efficiencies[name]) for name in names)
    else:
        efficiencies = tuple((name, inverter_efficiency) for name in names)
    return ReplayInput(scenario, points, socs, efficiencies, tuple(reasons))


def wire_ms(at_ts: float) -> int:
    """Q only: wire quantization never changes the physical float timeline."""
    return int(round(at_ts * 1000))


def model_namespace(*, source_plan: FastPlan, execution_plan: FastPlan, sample_points: SamplePoints,
                    limits: PolicyLimits, replay_input: ReplayInput) -> str:
    """Local repeatability identity, including the complete declarative recipe."""
    _typed(source_plan, FastPlan)
    _typed(execution_plan, FastPlan)
    _typed(limits, PolicyLimits)
    _typed(replay_input, ReplayInput)
    if replay_input.scenario is None or type(replay_input.scenario.execution) is not ModeledExecution:
        raise ValueError("replay_value_invalid")
    snapshot = dict(version="execution_model_v1", source_plan_digest=plan_digest(source_plan),
                    execution_plan_digest=plan_digest(execution_plan), sample_points=_samples(sample_points),
                    limits=asdict(limits), replay_input=asdict(replay_input))
    # Legacy wire serialization rounds several policy inputs. Retain those
    # in-memory values as well; v2's strict snapshot already preserves them.
    snapshot["source_plan"] = asdict(source_plan) if source_plan.format_version == 1 else source_plan.to_dict()
    snapshot["execution_plan"] = (asdict(execution_plan) if execution_plan.format_version == 1
                                  else execution_plan.to_dict())
    try:
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("replay_value_invalid") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BatteryExecutionView:
    battery_name: str
    command: BatteryCommand | None
    drive_w: float | None
    execution_status: str
    recovery_required: bool
    pending: BatteryCommand | None
    override: bool
    reasons: Reasons


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    binding: OwnerBinding | None
    batteries: tuple[BatteryExecutionView, ...]
    writes: int
    trace: tuple[ReplayEvent, ...]
    reasons: Reasons


@dataclass
class _Attempt:
    command: BatteryCommand
    index: int
    sent_key: tuple[float, int]
    binding: OwnerBinding | None
    timing: ReplayAttemptTiming | None
    decision: BatteryDecision | None
    receive_end: float | None = None
    apply_end: float | None = None
    received: bool = False
    observation: ReceiptEvent | None = None
    open: bool = True
    # Captured on applied acceptance; later policy intent cannot relabel it.
    override: bool = False

    @property
    def cutoff(self) -> float | None:
        return (self.apply_end if self.received else self.receive_end) if self.open else None


@dataclass
class _ModelBudget:
    at_ts: float
    clock: ClockSample
    cursor: float
    failed: bool = False
    sources: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _ModelSend:
    tick_index: int
    source: str
    decision: BatteryDecision
    binding: OwnerBinding | None


@dataclass(frozen=True)
class _ModelEffect:
    command: BatteryCommand
    timing: ReplayAttemptTiming
    effect: str


class ExecutionReplay:
    """Fold supplied evidence locally; no owner, transport or clock acquisition.

    The controller belongs to this run. Only the canonical reducer can establish
    acceptance; the adapter proves the captured pending, clocks and application
    first. The small receiver state used for binding is an evidence mirror, not
    a live receiver or a claim of historical authenticity.
    """

    def __init__(self, *, plan: FastPlan, scenario: ReplayScenario,
                 controller: ControllerState, start_ts: float, end_ts: float,
                 namespace: str | None = None):
        _typed(plan, FastPlan)
        _typed(scenario, ReplayScenario)
        _typed(controller, ControllerState)
        _number(start_ts)
        _number(end_ts)
        if start_ts >= end_ts:
            raise ValueError("replay_window_invalid")
        self._plan, self._scenario = deepcopy(plan), deepcopy(scenario)
        self._execution = self._scenario.execution
        self._model = self._execution if type(self._execution) is ModeledExecution else None
        self._controller = controller
        self._start, self._end, self._now = start_ts, end_ts, start_ts
        self._digest = plan_digest(self._plan)
        self._names = tuple(spec.name for spec in plan.specs)
        if not self._names or len(set(self._names)) != len(self._names):
            raise ValueError("replay_battery_mapping_invalid")
        if self._model is not None:
            self._init_model(namespace)
        self._states = tuple(controller.battery(i) for i in range(len(self._names)))
        for state in self._states:
            state._current_accepted_command = None
        self._reasons = [[] for _ in self._names]
        self._history_reasons = []
        self._trace = []
        self._writes = 0
        self._attempts: dict[str, _Attempt] = {}
        self._observations: dict[tuple[float, int], _Attempt] = {}
        self._receipt_reasons: dict[tuple[float, int], str] = {}
        self._accepted: list[_Attempt | None] = [None] * len(self._names)
        self._safe_since: list[float | None] = [None] * len(self._names)
        self._requests: dict[str, StopRequestEvent] = {}
        self._barrier = None
        self._seen_barriers = set()
        self._receiver_id = None
        self._bound_epoch = 0
        self._last_session = None
        self._binding = self._execution.opening_binding
        if self._binding is not None:
            if not self._binding_matches(self._binding) or self._binding.owner_not_after_ms <= wire_ms(start_ts):
                self._reason("execution_binding_mismatch")
                self._binding = None
            else:
                self._receiver_id = self._binding.receiver_id
                self._bound_epoch = self._binding.owner_epoch
                self._last_session = self._binding.owner_session_id
        self._coverage = {name: [] for name in self._names}
        for span in self._execution.coverage:
            if span.battery_name not in self._names:
                raise ValueError("replay_battery_mapping_invalid")
            if span.start_ts < start_ts or span.end_ts > end_ts:
                raise ValueError("replay_window_invalid")
            self._coverage[span.battery_name].append((span.start_ts, span.end_ts))
        for spans in self._coverage.values():
            spans.sort()
        self._queue = []
        self._position = 0
        self._boundaries = {start_ts, end_ts}
        identities = {}
        sequence = 0
        commands = [opening.command for opening in sorted(
            self._execution.opening_acceptances, key=lambda opening: opening.command.sequence)]
        commands.extend(event.command for event in self._execution.events
                        if type(event) is ProposalEvent and event.command is not None)
        for command in commands:
            previous = identities.get(command.command_id)
            if previous is not None:
                if previous != command:
                    raise ValueError("replay_duplicate_identity")
                continue
            if command.sequence <= sequence:
                raise ValueError("replay_duplicate_identity")
            identities[command.command_id] = command
            sequence = command.sequence
            self._boundaries.update((command.execute_before_ms / 1000, command.hold_until_ms / 1000))
            if command.stop_at_ms is not None:
                self._boundaries.add(command.stop_at_ms / 1000)
        for event in self._execution.events:
            if not start_ts <= event.at_ts <= end_ts:
                raise ValueError("replay_window_invalid")
            if type(event) is ProposalEvent:
                decision = event.decision
                if (decision.index >= len(self._names) or self._names[decision.index] != decision.name
                        or (event.command is not None and event.command.battery_name != decision.name)):
                    raise ValueError("replay_battery_mapping_invalid")
            self._queue.append((*_event_key(event), event, False))
            if type(event) is ReceiptEvent and event.application is not None:
                self._queue.append((*_event_key(event.application), event, True))
            if type(event) is BindingEvent:
                self._boundaries.add(event.binding.owner_not_after_ms / 1000)
        for spans in self._coverage.values():
            for start, end in spans:
                self._boundaries.update((start, end))
        for interval in plan.intervals:
            self._boundaries.update((interval.start_ts, interval.end_ts))
            for step in interval.batteries:
                self._boundaries.update(edge for edge in (step.active_from_ts, step.stop_ts) if edge is not None)
        if self._binding is not None:
            self._boundaries.add(self._binding.owner_not_after_ms / 1000)
        if plan.financial_context is not None:
            self._boundaries.add(plan.financial_context.not_after_ts)
        self._openings(self._execution.opening_acceptances)
        # Input event order was validated by RecordedExecution. Merge all
        # application keys, including opening evidence exactly on window start.
        self._queue.sort(key=lambda item: item[:2])
        self._boundaries.update(item[0] for item in self._queue)
        if self._execution.opening_stop is not None:
            request, barrier = self._execution.opening_stop
            self._request(StopRequestEvent(request.requested_at_ms / 1000, 0, request))
            self._stop(BarrierEvent(start_ts, 0, barrier))
        if self._model is not None:
            # Lifecycle order is declarative. Generated work follows existing
            # lifecycle evidence at the same float time, before the next send.
            for index, event in enumerate(self._model.lifecycle):
                if not start_ts <= event.at_ts <= end_ts:
                    raise ValueError("replay_window_invalid")
                self._enqueue(event.at_ts, index * 3, event)
            if self._model.initial_execution == "stopped":
                self._enqueue(start_ts, -3, ModelStop(start_ts, -3, True, True))
                self._enqueue(start_ts, -1, ModelBind(start_ts, -1))
        self.advance(start_ts)

    def _init_model(self, namespace: str | None) -> None:
        if (type(namespace) is not str or len(namespace) != 64
                or any(character not in "0123456789abcdef" for character in namespace)):
            raise ValueError("replay_value_invalid")
        self._model_root = f"replay/{namespace}"
        context = self._plan.financial_context
        installation = context.basis.meter_id if context is not None else f"{self._model_root}/installation"
        generation = f"{self._model_root}/generation"
        self._model_receiver = ReceiverState(
            1, installation, f"{self._model_root}/receiver", generation, f"{self._model_root}/anchor",
            1, 0, None, 0, 0, (), True, True, None, None, f"{self._model_root}/queue")
        # Complete coverage is a model assumption, never historical evidence.
        # Both variants then use exactly the same evidence consumer below.
        self._execution = RecordedExecution(installation, generation, None, (), None, tuple(
            ExecutionCoverage(name, self._start, self._end) for name in self._names), ())
        end = min(self._end, self._plan.horizon_end_ts)
        if context is not None:
            end = min(end, context.not_after_ts)
        self._model_lease_end = wire_ms(end)
        # Reserve ordered slots for request/barrier/binding before generated
        # responses at the same time. Trace order must also survive re-folding.
        self._model_order = len(self._model.lifecycle) * 3
        self._model_sequence = 0
        self._model_budgets: dict[int, _ModelBudget] = {}
        self._model_attempt_budgets: dict[str, _ModelBudget] = {}
        self._model_responses = {(item.tick_index, item.source, item.battery_name): item
                                 for item in self._model.responses}
        if any(item.battery_name not in self._names for item in self._model.responses):
            raise ValueError("replay_model_selector_invalid")
        self._model_used = set()

    def _order(self) -> int:
        self._model_order += 1
        return self._model_order

    def _enqueue(self, at_ts: float, order: int, event, application=False) -> None:
        self._queue.append((at_ts, order, event, application))
        self._queue[self._position:] = sorted(self._queue[self._position:], key=lambda item: item[:2])
        self._boundaries.add(at_ts)

    def _model_clock(self, at_ts: float, timing: ReplayAttemptTiming | None = None) -> ClockSample:
        mono = round((at_ts - self._start) * 1e9) if timing is None else self._mono(timing, at_ts)
        return ClockSample(wire_ms(at_ts), mono, f"{self._model_root}/boot", 0, True)

    def submit(self, *, tick_index: int, at_ts: float, proposals: tuple[BatteryDecision, ...],
               source: Literal["plan", "fast"]) -> None:
        """Queue model proposals; all sources share the tick's original budget."""
        if self._model is None:
            raise TypeError("replay_value_invalid")
        if type(tick_index) is not int or tick_index < 0:
            raise ValueError("replay_model_selector_invalid")
        _choice(source, ("plan", "fast"), "replay_model_selector_invalid")
        _number(at_ts)
        if not self._now <= at_ts < self._end:
            raise ValueError("replay_time_order_invalid")
        decisions = deepcopy(_tuple(proposals))
        indices = set()
        for decision in decisions:
            _decision(decision)
            if (decision.index >= len(self._names) or self._names[decision.index] != decision.name
                    or decision.index in indices):
                raise ValueError("replay_battery_mapping_invalid")
            indices.add(decision.index)
        budget = self._model_budgets.get(tick_index)
        if budget is None:
            if self._model_budgets and (tick_index <= max(self._model_budgets)
                                       or at_ts <= self._model_budgets[max(self._model_budgets)].at_ts):
                raise ValueError("replay_time_order_invalid")
            budget = _ModelBudget(at_ts, self._model_clock(at_ts), at_ts + self._model.prepare_delay_s)
            self._model_budgets[tick_index] = budget
        if budget.at_ts != at_ts or source in budget.sources:
            raise ValueError("replay_model_selector_invalid")
        self.advance(at_ts)
        budget.sources.add(source)
        for decision in sorted(decisions, key=lambda item: item.index):
            if not decision.write:
                event = ProposalEvent(at_ts, self._order(), decision, None, None)
                self._enqueue(at_ts, event.order, event)
                continue
            task = _ModelSend(tick_index, source, decision, self._binding)
            # Sends come after response/application work at an equal float time.
            # This also drains zero-latency work without creating a Q(t)/1000 gap.
            self._enqueue(budget.cursor, math.inf, task)
            response = self._model_responses.get((tick_index, source, decision.name))
            effect = response.effect if response is not None else self._model.default_effect
            delay = response.delay_s if response is not None else self._model.response_delay_s
            budget.cursor = (budget.cursor + delay + self._model.application_delay_s if effect is not None
                             else max(budget.cursor, at_ts + .75))

    def _model_stop(self, event: ModelStop, order: int) -> None:
        receiver = self._model_receiver
        request = RevokeRequest(receiver.installation_id, receiver.epoch_high_water,
                                f"{self._model_root}/barrier/{receiver.epoch_high_water}",
                                "model_stop", wire_ms(event.at_ts))
        self._fold(StopRequestEvent(event.at_ts, order, request))
        result = classify_revoke(receiver, request, receiver_now_upper_ms=wire_ms(event.at_ts),
                                 queues_drained=event.queues_fenced, safe_state_confirmed=event.safe_state_confirmed)
        self._model_receiver = result.state
        self._binding = None
        self._safe_since = [None] * len(self._names)
        for attempt in self._attempts.values():
            if attempt.open:
                self._model_attempt_budgets[attempt.command.command_id].failed = True
        self._fold(BarrierEvent(event.at_ts, order + 1, result.receipt))

    def _model_bind(self, event: ModelBind, order: int) -> None:
        receiver = self._model_receiver
        now = wire_ms(event.at_ts)
        if now >= self._model_lease_end or event.at_ts >= self._model_lease_end / 1000:
            self._reason("execution_context_expired")
            return
        result = classify_bind(
            receiver, installation_id=receiver.installation_id, barrier_id=receiver.pending_barrier_id or "missing",
            expected_epoch=receiver.epoch_high_water,
            owner_session_id=f"{self._model_root}/owner/{receiver.epoch_high_water}",
            requested_not_after_ms=self._model_lease_end, receiver_max_lease_ms=self._model_lease_end - now,
            receiver_now_upper_ms=now)
        self._model_receiver = result.state
        if result.binding is None:
            self._reason("execution_binding_mismatch")
        else:
            self._fold(BindingEvent(event.at_ts, order, result.binding))

    def _model_send(self, task: _ModelSend, at_ts: float) -> None:
        budget, decision, binding = self._model_budgets[task.tick_index], task.decision, task.binding
        if budget.failed or any(attempt.open for attempt in self._attempts.values()) or any(
                state.recovery_required for state in self._states):
            budget.failed = True
            self._reason("execution_unconfirmed", decision.index)
            return
        if binding is None or binding != self._binding:
            budget.failed = True
            self._reason("execution_binding_mismatch", decision.index)
            return
        issued = wire_ms(budget.at_ts)
        try:
            interval = self._plan.interval_at(budget.at_ts)
            if interval is None or interval is not self._plan.interval_at(issued / 1000):
                raise ExecutionRefused("command_window_invalid")
            if self._plan.financial_context is not None:
                step = interval.battery(decision.index)
                if any(budget.at_ts < edge <= issued / 1000 for edge in (step.active_from_ts, step.stop_ts)
                       if edge is not None):
                    raise ExecutionRefused("command_window_invalid")
                end = _command_end_ms(self._plan, battery_name=decision.name, at_ms=issued,
                                      owner_end_ms=binding.owner_not_after_ms)
            else:
                end = min(wire_ms(interval.end_ts), wire_ms(self._end), binding.owner_not_after_ms)
            receive = _bounded_deadline(Deadline(budget.clock.boottime_ns + 750_000_000, end),
                                        clock=budget.clock, wire_end_ms=end)
            apply = _bounded_deadline(Deadline(budget.clock.boottime_ns + 1_000_000_000, end),
                                      clock=budget.clock, wire_end_ms=end)
            timing = ReplayAttemptTiming(budget.at_ts, budget.clock, receive, apply)
            clock = self._model_clock(at_ts, timing)
            timing = ReplayAttemptTiming(budget.at_ts, budget.clock,
                _bounded_deadline(receive, clock=clock, wire_end_ms=end),
                _bounded_deadline(apply, clock=clock, wire_end_ms=end))
        except ExecutionRefused:
            budget.failed = True
            self._reason("execution_deadline_exceeded", decision.index)
            return
        setpoint, mode, stop = self._command_payload(decision, issued)
        self._model_sequence += 1
        command = BatteryCommand(binding.installation_id, decision.name, binding.owner_session_id,
            binding.owner_epoch, self._model_sequence,
            f"{self._model_root}/{binding.owner_epoch}/{self._model_sequence}", self._digest,
            issued, end, end, setpoint, mode, stop)
        self._model_attempt_budgets[command.command_id] = budget
        self._boundaries.update((end / 1000,))
        if stop is not None:
            self._boundaries.add(stop / 1000)
        self._fold(ProposalEvent(at_ts, self._order(), decision, command, timing))
        result = classify_send(self._model_receiver, command, receiver_now_upper_ms=clock.utc_ms,
                               clock_healthy=True, clock_uncertainty_ms=0)
        self._model_receiver = result.state
        selector = (task.tick_index, task.source, decision.name)
        response = self._model_responses.get(selector)
        if response is not None:
            self._model_used.add(selector)
        if result.status == "rejected":
            self._model_receipt(result.receipt, timing, at_ts)
            return
        effect = response.effect if response is not None else self._model.default_effect
        delay = response.delay_s if response is not None else self._model.response_delay_s
        if effect is None:
            self._reason("execution_model_missing", decision.index)
        else:
            self._enqueue(at_ts + delay, self._order(), _ModelEffect(command, timing, effect))

    def _model_effect(self, task: _ModelEffect, at_ts: float) -> None:
        result = classify_effect(self._model_receiver, task.command, receiver_now_upper_ms=wire_ms(at_ts),
                                 clock_healthy=True, clock_uncertainty_ms=0, outcome=task.effect)
        self._model_receiver = result.state
        self._model_receipt(result.receipt, task.timing, at_ts)

    def _model_receipt(self, receipt: ActuationReceipt, timing: ReplayAttemptTiming, at_ts: float) -> None:
        order = self._order()
        applied = at_ts + self._model.application_delay_s
        outcome = "unknown" if receipt.status == "rejected" else receipt.status
        application = ReplayApplication(applied, self._order(), self._model_clock(applied, timing), outcome)
        event = ReceiptEvent(at_ts, order, receipt, self._model_clock(at_ts, timing), application)
        self._fold(replace(event, application=None))
        self._enqueue(applied, application.order, event, True)

    def _model_application(self, event: ReceiptEvent) -> None:
        # Publish the consumer outcome only after the common fold has performed
        # its final checks. A scheduled success is not application evidence.
        outcome = self._fold(event, application=True) or "unconfirmed"
        evidence = replace(event, application=replace(event.application, outcome=outcome))
        for index, raw in enumerate(self._trace):
            if type(raw) is ReceiptEvent and _event_key(raw) == _event_key(event):
                self._trace[index] = evidence
                break

    def _reason(self, reason: str, index: int | None = None) -> None:
        if reason not in self._history_reasons:
            self._history_reasons.append(reason)
        if index is not None and reason not in self._reasons[index]:
            self._reasons[index].append(reason)

    def _binding_matches(self, binding: OwnerBinding) -> bool:
        context = self._plan.financial_context
        return (binding.installation_id == self._execution.installation_id
                and binding.receiver_generation == self._execution.receiver_generation
                and (context is None or binding.installation_id == context.basis.meter_id))

    def _covered(self, index: int, since: float, at_ts: float) -> bool:
        cursor = max(since, self._start)
        for start, end in self._coverage[self._names[index]]:
            if start > cursor:
                return False
            if end > cursor:
                if at_ts < end:
                    return True
                cursor = end
        return False

    def _command_bound(self, command: BatteryCommand, binding: OwnerBinding) -> int:
        if self._plan.financial_context is not None:
            return _command_end_ms(self._plan, battery_name=command.battery_name,
                                   at_ms=command.issued_at_ms, owner_end_ms=binding.owner_not_after_ms)
        interval = self._plan.interval_at(command.issued_at_ms / 1000)
        if interval is None:
            raise ExecutionRefused("command_window_invalid")
        return min(interval.end_ts * 1000, binding.owner_not_after_ms)

    def _command_matches(self, command: BatteryCommand, at_ts: float) -> bool:
        binding = self._binding
        if (binding is None or not self._binding_matches(binding)
                or command.installation_id != binding.installation_id
                or command.owner_epoch != binding.owner_epoch
                or command.owner_session_id != binding.owner_session_id
                or command.plan_digest != self._digest or command.battery_name not in self._names
                or not command.issued_at_ms <= wire_ms(at_ts) < command.execute_before_ms
                or at_ts >= command.execute_before_ms / 1000):
            return False
        try:
            return command.hold_until_ms <= self._command_bound(command, binding)
        except ExecutionRefused:
            return False

    def _payload_matches(self, event: ProposalEvent) -> bool:
        command = event.command
        return (command.battery_name == event.decision.name
                and (command.setpoint_w, command.mode, command.stop_at_ms)
                == self._command_payload(event.decision, command.issued_at_ms))

    def _command_payload(self, decision: BatteryDecision, issued_at_ms: int) -> tuple[int, str, int | None]:
        spec = self._plan.spec(decision.index)
        setpoint = int(round(decision.setpoint_w))
        mode = decision.mode
        if mode is None:
            mode = spec.mode_on if abs(setpoint) >= POWER_EPS else spec.mode_off
        stop = None
        if decision.stop_inverter is not None:
            if decision.stop_inverter != NO_STOP_SENTINEL:
                try:
                    stop = wire_ms(datetime.fromisoformat(decision.stop_inverter).timestamp())
                except ValueError:
                    pass
        else:
            interval = self._plan.interval_at(issued_at_ms / 1000)
            if interval is not None and interval.battery(decision.index).stop_ts is not None:
                stop = int(interval.battery(decision.index).stop_ts) * 1000
        return setpoint, mode, stop

    @staticmethod
    def _mono(timing: ReplayAttemptTiming, at_ts: float) -> int:
        return timing.anchor_clock.boottime_ns + round((at_ts - timing.anchor_ts) * 1e9)

    def _timing_ends(self, attempt: _Attempt, timing: ReplayAttemptTiming | None,
                     at_ts: float) -> tuple[float, float] | None:
        if timing is None or attempt.binding is None:
            return None
        clock = timing.anchor_clock
        receive, apply = timing.receive_deadline, timing.apply_deadline
        if (not clock.healthy or clock.utc_ms != wire_ms(timing.anchor_ts) or timing.anchor_ts > at_ts
                or receive.monotonic_end_ns > apply.monotonic_end_ns
                or receive.execute_before_ms > apply.execute_before_ms):
            return None
        try:
            bound = min(attempt.command.execute_before_ms, self._command_bound(attempt.command, attempt.binding))
            if any(_bounded_deadline(deadline, clock=clock, wire_end_ms=bound) != deadline
                   for deadline in (receive, apply)):
                return None
        except ExecutionRefused:
            return None
        return tuple(min(timing.anchor_ts + (deadline.monotonic_end_ns - clock.boottime_ns) / 1e9,
                         deadline.execute_before_ms / 1000, bound / 1000) for deadline in (receive, apply))

    def _clock_reason(self, attempt: _Attempt, clock: ClockSample | None,
                      at_ts: float, *, application=False) -> str | None:
        timing = attempt.timing
        if (timing is None or clock is None or not clock.healthy
                or clock.boot_id != timing.anchor_clock.boot_id or clock.utc_ms != wire_ms(at_ts)
                or clock.boottime_ns != self._mono(timing, at_ts)):
            return "execution_deadline_unknown"
        deadline = timing.apply_deadline if application else timing.receive_deadline
        end = attempt.apply_end if application else attempt.receive_end
        bound = min(deadline.execute_before_ms, attempt.command.execute_before_ms,
                    self._command_bound(attempt.command, attempt.binding))
        if (end is None or at_ts >= end or deadline.remaining_ns(clock.boottime_ns) <= 0
                or clock.utc_ms + clock.uncertainty_ms >= bound or at_ts >= bound / 1000):
            return "execution_deadline_exceeded"
        return None

    def _finish(self, attempt: _Attempt, reason: str, *, outcome="unconfirmed") -> None:
        if not attempt.open:
            return
        attempt.open = False
        if self._model is not None:
            self._model_attempt_budgets[attempt.command.command_id].failed = True
        self._reason(reason, attempt.index)
        apply_execution_outcome(self._states[attempt.index], outcome=outcome, pending=attempt.command)

    def _send(self, command: BatteryCommand, key: tuple[float, int], timing: ReplayAttemptTiming | None,
              decision: BatteryDecision | None, *, payload_matches=True) -> _Attempt:
        index = self._names.index(command.battery_name) if command.battery_name in self._names else None
        if index is None:
            raise ValueError("replay_battery_mapping_invalid")
        state = self._states[index]
        previous = self._attempts.get(command.command_id)
        if previous is not None:
            if not payload_matches:
                self._reason("execution_binding_mismatch", index)
                self._finish(previous, "execution_binding_mismatch")
                return previous
            if not self._command_matches(command, key[0]):
                self._reason("execution_binding_mismatch", index)
                return previous
            ends = self._timing_ends(previous, timing, key[0])
            old = previous.timing
            if (ends is None or old is None or timing.anchor_clock.boot_id != old.anchor_clock.boot_id
                    or self._mono(old, timing.anchor_ts) != timing.anchor_clock.boottime_ns
                    or timing.receive_deadline.monotonic_end_ns > old.receive_deadline.monotonic_end_ns
                    or timing.apply_deadline.monotonic_end_ns > old.apply_deadline.monotonic_end_ns
                    or timing.receive_deadline.execute_before_ms > old.receive_deadline.execute_before_ms
                    or timing.apply_deadline.execute_before_ms > old.apply_deadline.execute_before_ms):
                self._finish(previous, "execution_deadline_unknown")
            elif previous.open:
                previous.timing = timing
                previous.receive_end, previous.apply_end = ends
                if previous.observation is not None:
                    observed = previous.observation
                    reason = self._clock_reason(previous, observed.received_clock, observed.at_ts)
                    if reason is not None:
                        self._finish(previous, reason)
                self._expire(key[0])
            return previous
        attempt = _Attempt(command, index, key, self._binding, timing, decision)
        self._attempts[command.command_id] = attempt
        self._safe_since[index] = None
        if decision is not None and decision.restores:
            state.restore_pending = True
        if state.recovery_required or any(item.open and item is not attempt and item.index == index
                                          for item in self._attempts.values()):
            self._finish(attempt, "execution_unconfirmed")
            for item in self._attempts.values():
                if item.index == index:
                    item.open = False
            return attempt
        state.pending_command = command
        apply_execution_outcome(state, outcome="unknown")
        if not payload_matches or not self._command_matches(command, key[0]):
            self._finish(attempt, "execution_binding_mismatch")
            return attempt
        ends = self._timing_ends(attempt, timing, key[0])
        if ends is None:
            self._finish(attempt, "execution_deadline_unknown")
        else:
            attempt.receive_end, attempt.apply_end = ends
            self._expire(key[0])
        return attempt

    def _receive(self, event: ReceiptEvent) -> None:
        self._trace.append(event)
        attempt = self._attempts.get(event.receipt.command_id)
        if attempt is None or not attempt.open:
            return
        reason = None
        if not receipt_matches_command(event.receipt, attempt.command) or _event_key(event) <= attempt.sent_key:
            reason = "execution_receipt_mismatch"
        reason = reason or self._clock_reason(attempt, event.received_clock, event.at_ts)
        if (reason is not None and attempt.observation is not None and attempt.observation.receipt == event.receipt
                and (event.application is None or event.application.outcome == "accepted")):
            # A late duplicate is history, not a timeout of the earlier timely
            # observation. A timely duplicate can carry its own application.
            return
        receipt, command = event.receipt, attempt.command
        if (wire_ms(event.at_ts) < receipt.observed_at_ms or receipt.observed_at_ms < command.issued_at_ms
                or (receipt.accepted_ts is not None and not
                    command.issued_at_ms <= receipt.accepted_ts <= receipt.observed_at_ms)
                or (receipt.accepted_ts is not None and receipt.accepted_ts >= command.execute_before_ms)):
            reason = reason or "execution_receipt_mismatch"
        self._observations[_event_key(event)] = attempt
        if reason is not None:
            # A raw observation, including incomplete clock evidence, is not a
            # terminal consumer outcome. Fold its application or expire pending
            # at the original deadline; never run application work at receipt.
            self._receipt_reasons[_event_key(event)] = reason
            self._reason(reason, attempt.index)
            return
        if receipt.status == "accepted":
            attempt.received = True
            if attempt.observation is None:
                attempt.observation = event

    def _apply(self, event: ReceiptEvent) -> str | None:
        attempt = self._observations.get(_event_key(event))
        if attempt is None or not attempt.open:
            return
        application, command = event.application, attempt.command
        state = self._states[attempt.index]
        if state.pending_command != command or state.recovery_required or self._binding != attempt.binding:
            return
        reason = self._clock_reason(attempt, application.clock, application.at_ts, application=True)
        reason = reason or self._receipt_reasons.get(_event_key(event))
        reason = reason or self._clock_reason(attempt, event.received_clock, event.at_ts)
        if reason is not None:
            self._finish(attempt, reason)
        elif event.receipt.status == "rejected" and application.outcome == "unknown":
            self._finish(attempt, "execution_rejected", outcome="unknown")
            return "unknown"
        elif application.outcome != "accepted" or event.receipt.status != "accepted":
            self._finish(attempt, "execution_unconfirmed")
        elif not self._covered(attempt.index, attempt.sent_key[0], application.at_ts):
            self._finish(attempt, "execution_history_missing")
        elif apply_execution_outcome(state, outcome="accepted", pending=command, receipt=event.receipt):
            attempt.open = False
            self._accepted[attempt.index] = attempt
            state.held_setpoint_w = float(command.setpoint_w)
            state.held_stop_inverter_cleared = state.stop_inverter_cleared
            state.last_change_ts = event.receipt.accepted_ts / 1000
            if attempt.decision is not None:
                attempt.override = attempt.decision.override
                if attempt.decision.restores:
                    state.restore_pending = False
            else:
                interval = self._plan.interval_at(command.issued_at_ms / 1000)
                step = interval.battery(attempt.index)
                plan_w = (profile_power_at(step, command.issued_at_ms / 1000)
                          if self._plan.financial_context is not None else step.ac_power_w)
                attempt.override = command.setpoint_w != int(round(plan_w))
            state.override_active = attempt.override
            return "accepted"
        return "unconfirmed"

    def _openings(self, openings: tuple[OpeningAcceptance, ...]) -> None:
        queue = []
        for opening in openings:
            observation, application = opening.observation, opening.observation.application
            if (opening.sent_at_ts is None or application is None
                    or not opening.sent_at_ts <= observation.at_ts <= application.at_ts <= self._start):
                self._trace.append(observation)
                self._reason("execution_application_unproven")
                continue
            queue.extend(((opening.sent_at_ts, -math.inf, opening, False),
                          (*_event_key(observation), observation, False),
                          (*_event_key(application), observation, True)))
        # Only strictly pre-window evidence is folded here. Start-boundary
        # evidence shares the regular queue and its lifecycle/application order.
        for at_ts, order, event, application in sorted(queue, key=lambda item: item[:2]):
            if at_ts == self._start:
                self._queue.append((at_ts, order, event, application))
                continue
            self._expire(at_ts)
            if type(event) is OpeningAcceptance:
                self._send(event.command, (at_ts, order), event.timing, None)
            elif application:
                self._apply(event)
            else:
                self._receive(event)

    def _request(self, event: StopRequestEvent) -> None:
        request = event.request
        binding = self._binding
        epoch = binding.owner_epoch if binding is not None else (
            self._barrier.current_epoch if self._barrier is not None else request.expected_epoch)
        if self._model is not None:
            epoch = self._model_receiver.epoch_high_water
        if (request.installation_id != self._execution.installation_id or request.expected_epoch != epoch
                or request.requested_at_ms > wire_ms(event.at_ts)):
            self._reason("execution_barrier_unproven")
            return
        self._requests[request.barrier_id] = event

    def _stop(self, event: BarrierEvent) -> None:
        barrier = event.receipt
        own = self._requests.get(barrier.barrier_id)
        if barrier.barrier_id in self._seen_barriers:
            return
        if (own is None or event.at_ts < own.at_ts or barrier.status != "accepted"
                or (self._model is None and self._binding is not None
                    and self._binding.owner_epoch != own.request.expected_epoch)
                or (self._model is None and self._barrier is not None
                    and self._barrier.current_epoch != own.request.expected_epoch)
                or (self._model is not None and self._model_receiver.last_barrier != barrier)
                or barrier.installation_id != self._execution.installation_id
                or barrier.receiver_generation != self._execution.receiver_generation
                or barrier.revoked_through_epoch != own.request.expected_epoch
                or barrier.current_epoch != own.request.expected_epoch + 1
                or not barrier.queues_fenced or barrier.safe_state != "confirmed"
                or not own.request.requested_at_ms <= barrier.accepted_ts <= wire_ms(event.at_ts)):
            self._reason("execution_barrier_unproven")
            return
        self._seen_barriers.add(barrier.barrier_id)
        self._barrier = barrier
        self._binding = None
        for index, state in enumerate(self._states):
            state._current_accepted_command = None
            pending = state.pending_command
            if pending is not None and (pending.installation_id != barrier.installation_id
                                       or pending.owner_epoch > barrier.revoked_through_epoch
                                       or pending.issued_at_ms > barrier.accepted_ts):
                continue
            self._safe_since[index] = event.at_ts
            if pending is not None or state.recovery_required:
                apply_execution_outcome(state, outcome="stopped")
            for attempt in self._attempts.values():
                if attempt.index == index:
                    attempt.open = False

    def _bind(self, event: BindingEvent) -> None:
        binding, barrier = event.binding, self._barrier
        now_ms = wire_ms(event.at_ts)
        if (barrier is None or not self._binding_matches(binding)
                or (self._receiver_id is not None and binding.receiver_id != self._receiver_id)
                or binding.owner_session_id == self._last_session
                or binding.owner_not_after_ms <= now_ms or event.at_ts >= binding.owner_not_after_ms / 1000):
            self._reason("execution_binding_mismatch")
            return
        mirror = ReceiverState(
            schema_version=1, installation_id=self._execution.installation_id, receiver_id=binding.receiver_id,
            receiver_generation=self._execution.receiver_generation, anchor_checkpoint="recorded-evidence",
            epoch_high_water=barrier.current_epoch, sequence_high_water=0,
            bound_owner_session_id=self._binding.owner_session_id if self._binding else None,
            bound_owner_epoch=self._bound_epoch, owner_not_after_ms=0, commands=(), inhibited=True,
            queue_known=True, pending_barrier_id=barrier.barrier_id, last_barrier=barrier,
            queue_generation="recorded-evidence")
        result = classify_bind(mirror, installation_id=binding.installation_id, barrier_id=binding.barrier_id,
                               expected_epoch=binding.owner_epoch, owner_session_id=binding.owner_session_id,
                               requested_not_after_ms=binding.owner_not_after_ms,
                               receiver_max_lease_ms=binding.owner_not_after_ms - now_ms,
                               receiver_now_upper_ms=now_ms)
        if result.status != "accepted" or result.binding != binding:
            self._reason("execution_binding_mismatch")
            return
        self._binding = binding
        self._bound_epoch = binding.owner_epoch
        self._last_session = binding.owner_session_id
        self._receiver_id = binding.receiver_id

    def _expire(self, at_ts: float) -> None:
        for attempt in self._attempts.values():
            if attempt.cutoff is not None and attempt.cutoff <= at_ts:
                self._reason("execution_application_unproven" if attempt.received else "execution_receipt_missing",
                             attempt.index)
                self._finish(attempt, "execution_deadline_exceeded")

    def _fold(self, event: ReplayEvent, *, application=False) -> str | None:
        if application:
            return self._apply(event)
        elif type(event) is ReceiptEvent:
            self._receive(event)
        else:
            self._trace.append(event)
            if type(event) is ProposalEvent:
                if event.decision.write:
                    if event.command is None:
                        index = event.decision.index
                        self._safe_since[index] = None
                        apply_execution_outcome(self._states[index], outcome="unconfirmed")
                        for attempt in self._attempts.values():
                            if attempt.index == index:
                                attempt.open = False
                        self._reason("execution_input_missing", index)
                    else:
                        self._writes += 1
                        self._send(event.command, _event_key(event), event.timing, event.decision,
                                   payload_matches=self._payload_matches(event))
            elif type(event) is StopRequestEvent:
                self._request(event)
            elif type(event) is BarrierEvent:
                self._stop(event)
            elif type(event) is BindingEvent:
                self._bind(event)

    def advance(self, at_ts: float) -> None:
        _number(at_ts)
        if at_ts < self._now or at_ts > self._end:
            raise ValueError("replay_time_order_invalid")
        while self._position < len(self._queue):
            time, order, event, application = self._queue[self._position]
            if time > at_ts or time >= self._end:
                break
            self._expire(time)
            self._position += 1
            if type(event) is OpeningAcceptance:
                # The existing opening-send marker captures pending before its
                # receipt. A send on the left boundary is an in-window attempt.
                self._writes += 1
                self._send(event.command, (time, order), event.timing, None)
            elif type(event) is ModelStop:
                self._model_stop(event, order)
            elif type(event) is ModelBind:
                self._model_bind(event, order)
            elif type(event) is _ModelSend:
                self._model_send(event, time)
            elif type(event) is _ModelEffect:
                self._model_effect(event, time)
            elif self._model is not None and application:
                self._model_application(event)
            else:
                self._fold(event, application=application)
        self._expire(at_ts)
        self._now = at_ts

    def next_boundary(self, after_ts: float, before_ts: float) -> float:
        _number(after_ts)
        _number(before_ts)
        if before_ts < after_ts:
            raise ValueError("replay_time_order_invalid")
        lower = max(after_ts, self._now)
        candidates = [edge for edge in self._boundaries if lower < edge <= before_ts]
        candidates.extend(attempt.cutoff for attempt in self._attempts.values()
                          if attempt.cutoff is not None and lower < attempt.cutoff <= before_ts)
        return min(candidates, default=before_ts)

    def snapshot(self, at_ts: float) -> ExecutionSnapshot:
        _number(at_ts)
        if at_ts != self._now:
            raise ValueError("replay_time_order_invalid")
        binding = self._binding
        if binding is not None and (wire_ms(at_ts) >= binding.owner_not_after_ms
                                    or at_ts >= binding.owner_not_after_ms / 1000):
            binding = None
        views = []
        for index, state in enumerate(self._states):
            reasons = list(self._reasons[index])
            accepted = self._accepted[index]
            command = current_execution_command(state, binding=binding, now_ms=wire_ms(at_ts),
                                                expected_digest=self._digest)
            since = accepted.sent_key[0] if accepted is not None else self._start
            covered = self._covered(index, since, at_ts)
            if not covered:
                command = None
                reasons.append("execution_history_missing")
            if command is not None and at_ts >= min(command.hold_until_ms, binding.owner_not_after_ms) / 1000:
                command = None
            drive = float(command.setpoint_w) if command is not None else None
            ideal_stop = self._scenario.physics.stop_response == "ideal_programmed_stop"
            if command is not None and command.stop_at_ms is not None and at_ts >= command.stop_at_ms / 1000:
                drive = 0.0 if ideal_stop else None
            if drive is None and ideal_stop and state.pending_command is None and not state.recovery_required:
                safe = self._safe_since[index]
                if safe is not None and self._covered(index, safe, at_ts):
                    drive = 0.0
                elif (covered and accepted is not None and state.execution_status == "accepted"
                      and self._binding == accepted.binding and at_ts >= accepted.command.hold_until_ms / 1000):
                    drive = 0.0
            if command is None:
                if accepted is not None and state.pending_command is None and self._safe_since[index] is None:
                    reasons.append("execution_context_expired")
                elif state.pending_command is not None and not state.recovery_required:
                    attempt = self._attempts.get(state.pending_command.command_id)
                    if attempt is not None and attempt.open:
                        reasons.append("execution_application_unproven" if attempt.received else "execution_receipt_missing")
                elif accepted is None and self._safe_since[index] is None and not reasons:
                    reasons.append("execution_history_missing")
            views.append(BatteryExecutionView(self._names[index], command, drive, state.execution_status,
                                               state.recovery_required, state.pending_command,
                                               accepted.override if accepted is not None else False,
                                               tuple(dict.fromkeys(reasons))))
        unused = ("unused_model_response",) if (self._model is not None and at_ts == self._end
                                                and self._model_responses.keys() - self._model_used) else ()
        reasons = tuple(dict.fromkeys((*self._history_reasons, *(reason for view in views for reason in view.reasons),
                                        *unused)))
        # Reuse only strictly validated immutable history. A fresh per-snapshot
        # memo leaves mutable decisions/nested data detached, without rebuilding
        # their surrounding command/timing graph or re-running constructors.
        # HelperResult is only shallowly frozen: nonempty helper payloads still
        # need a full defensive copy, without assuming their fields are scalar.
        memo = {}
        for event in self._trace:
            if type(event) is ProposalEvent:
                memo[id(event.command)] = event.command
                memo[id(event.timing)] = event.timing
            elif type(event) is not ReceiptEvent or not event.receipt.helper_results:
                memo[id(event)] = event
        return ExecutionSnapshot(binding, tuple(views), self._writes, deepcopy(tuple(self._trace), memo), reasons)
