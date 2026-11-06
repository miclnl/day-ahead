"""Explicit, transport-free replay inputs; evidence is not execution authority."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from .command import (
    ActuationReceipt,
    BarrierReceipt,
    BatteryCommand,
    ClockSample,
    Deadline,
    OwnerBinding,
    RevokeRequest,
)
from .plan import FastPlan
from .policy import AcceptedCommand, BatteryDecision

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
