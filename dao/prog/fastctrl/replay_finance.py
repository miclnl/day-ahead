"""Pure replay source bindings and content identities, never execution authority.

PV claims describe complete topology/inventory and the shared exogenous house
environment for both alternatives. They are caller_asserted, not verification
of an installation, signature, sensor, supplier or historical energy quantity.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Context, Decimal, localcontext
from hashlib import sha256
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

from dao.prog.settlement.engine import _contract_periods, delivery_year_bounds, settle_series
from dao.prog.settlement.metering import _spot_window, merge_inputs, resolve_spot
from dao.prog.settlement.optimization_economics import battery_wear_cost
from dao.prog.settlement.replay import ReplayBasis
from dao.prog.settlement.tariffs import _midnight, quote, resolve_snapshot
from dao.prog.settlement.types import (
    TARIFF_FIELDS,
    InputInterval,
    MeterInterval,
    PeriodAdjustment,
    SettlementLine,
    SettlementResult,
    TariffSnapshot,
    _FrozenDTO,
)

from . import command, replay_execution
from .command import plan_digest
from .plan import BatterySpec, FastPlan, _decimal_text
from .policy import AcceptedCommand, BatteryDecision
from .replay_execution import RecordedExecution, ReplayEvent, ReplayInput

if TYPE_CHECKING:
    from .simulate import BatteryReplaySegment, ReplaySegment


def _text(value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("source_text_invalid")


def _digest(value: object) -> None:
    if (type(value) is not str or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("source_digest_invalid")


def _utc(value: object) -> None:
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("source_time_invalid")


def _window(start: int, end: int) -> None:
    _utc(start)
    _utc(end)
    if start >= end:
        raise ValueError("source_window_invalid")


def _choice(value: object, choices: tuple[str, ...]) -> None:
    if type(value) is not str or value not in choices:
        raise ValueError("source_choice_invalid")


def _typed(value: object, kind: type) -> None:
    if type(value) is not kind:
        raise TypeError("source_type_invalid")


def _tuple(value: object) -> tuple:
    if type(value) not in (tuple, list):
        raise TypeError("source_sequence_invalid")
    return tuple(value)


def _identities(value: object) -> tuple[str, ...]:
    result = _tuple(value)
    for name in result:
        _text(name)
    if len(set(result)) != len(result):
        raise ValueError("source_duplicate_identity")
    return tuple(sorted(result))


def _reasons(value: object) -> tuple[str, ...]:
    result = _tuple(value)
    for reason in result:
        _text(reason)
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True, slots=True)
class SourceReference:
    reference: str
    version: str
    content_digest: str

    def __post_init__(self) -> None:
        _text(self.reference)
        _text(self.version)
        _digest(self.content_digest)


def _references(value: object) -> tuple[SourceReference, ...]:
    result = _tuple(value)
    if not result:
        raise ValueError("source_references_empty")
    identities = set()
    for reference in result:
        _typed(reference, SourceReference)
        identity = (reference.reference, reference.version)
        if identity in identities:
            raise ValueError("source_duplicate_reference")
        identities.add(identity)
    return tuple(sorted(result, key=lambda ref: (ref.reference, ref.version, ref.content_digest)))


@dataclass(frozen=True, slots=True)
class ReplaySourceSpan:
    start_ts: int
    end_ts: int
    claim: Literal["pv_only_nonbattery_generation", "unknown"]
    evidence_references: tuple[SourceReference, ...]

    def __post_init__(self) -> None:
        _window(self.start_ts, self.end_ts)
        _choice(self.claim, ("pv_only_nonbattery_generation", "unknown"))
        object.__setattr__(self, "evidence_references", _references(self.evidence_references))


@dataclass(frozen=True, slots=True)
class ReplaySourceEvidence:
    """A positive span asserts complete inventory and PV-only non-storage generation.

    It does not assert equal exports in both scenarios, allocate PV quantities,
    or reclassify modeled battery discharge. Gaps and unknown spans are allowed.
    """

    version: Literal["replay_source_v1"]
    installation_id: str
    meter_id: str
    replay_basis_hash: str
    source_plan_digest: str
    environment_digest: str
    battery_names: tuple[str, ...]
    pv_source_ids: tuple[str, ...]
    topology_reference: SourceReference
    asserted_by: str
    asserted_at_ts: int
    evidence_references: tuple[SourceReference, ...]
    coverage_start_ts: int
    coverage_end_ts: int
    spans: tuple[ReplaySourceSpan, ...]
    counterfactual_scope: Literal["both_alternatives", "observed_run_only", "unknown"]

    def __post_init__(self) -> None:
        _choice(self.version, ("replay_source_v1",))
        for value in (self.installation_id, self.meter_id, self.asserted_by):
            _text(value)
        for value in (self.replay_basis_hash, self.source_plan_digest, self.environment_digest):
            _digest(value)
        object.__setattr__(self, "battery_names", _identities(self.battery_names))
        object.__setattr__(self, "pv_source_ids", _identities(self.pv_source_ids))
        _typed(self.topology_reference, SourceReference)
        _utc(self.asserted_at_ts)
        object.__setattr__(self, "evidence_references", _references(self.evidence_references))
        _window(self.coverage_start_ts, self.coverage_end_ts)
        spans = _tuple(self.spans)
        cursor = self.coverage_start_ts
        for span in spans:
            _typed(span, ReplaySourceSpan)
            if span.start_ts < cursor or span.end_ts > self.coverage_end_ts:
                raise ValueError("source_span_order_or_coverage_invalid")
            if span.claim == "pv_only_nonbattery_generation" and not self.pv_source_ids:
                raise ValueError("source_pv_identity_required")
            cursor = span.end_ts
        object.__setattr__(self, "spans", spans)
        _choice(self.counterfactual_scope, ("both_alternatives", "observed_run_only", "unknown"))

    def canonical_bytes(self) -> bytes:
        return _finance_canonical_bytes(self)

    @property
    def content_hash(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplaySourceBinding:
    installation_id: str
    meter_id: str
    replay_basis_hash: str
    source_plan_digest: str
    environment_digest: str | None
    battery_names: tuple[str, ...]
    source_digest: str | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.installation_id)
        _text(self.meter_id)
        _digest(self.replay_basis_hash)
        _digest(self.source_plan_digest)
        for value in (self.environment_digest, self.source_digest):
            if value is not None:
                _digest(value)
        object.__setattr__(self, "battery_names", _identities(self.battery_names))
        object.__setattr__(self, "reasons", _reasons(self.reasons))


@dataclass(frozen=True, slots=True)
class SourceValidation:
    status: Literal["valid", "missing", "mismatch"]
    evidence: ReplaySourceEvidence | None
    binding: ReplaySourceBinding
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _choice(self.status, ("valid", "missing", "mismatch"))
        if self.evidence is not None:
            _typed(self.evidence, ReplaySourceEvidence)
        _typed(self.binding, ReplaySourceBinding)
        object.__setattr__(self, "reasons", _reasons(self.reasons))


# Only transient reads of existing execution/model data are canonicalized. None
# of the new financial DTOs retains a mutable BatterySpec or BatteryDecision.
_CANONICAL_CLASSES = frozenset((
    SourceReference, ReplaySourceSpan, ReplaySourceEvidence, ReplaySourceBinding, SourceValidation,
    ReplayInput, replay_execution.ReplayScenario, replay_execution.ReplayPhysics,
    RecordedExecution, replay_execution.ModeledExecution, replay_execution.ModelResponse,
    replay_execution.ModelStop, replay_execution.ModelBind, replay_execution.ExecutionCoverage,
    replay_execution.OpeningAcceptance, replay_execution.ReplayAttemptTiming, replay_execution.ReplayApplication,
    replay_execution.ProposalEvent, replay_execution.ReceiptEvent, replay_execution.StopRequestEvent,
    replay_execution.BarrierEvent, replay_execution.BindingEvent,
    command.BatteryCommand, command.ActuationReceipt, command.HelperResult, command.RevokeRequest,
    command.BarrierReceipt, command.Deadline, command.ClockSample, command.OwnerBinding,
    BatterySpec, BatteryDecision, AcceptedCommand,
))


def _finance_tree(value: object) -> dict:
    kind = type(value)
    if value is None:
        return {"type": "none"}
    if kind in (str, bool, int):
        return {"type": kind.__name__, "value": value}
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("finance_nonfinite_value")
        return {"type": "float", "value": value.hex()}
    if kind is Decimal:
        return {"type": "decimal", "value": _decimal_text(value)}
    if kind is tuple:
        return {"type": "tuple", "value": [_finance_tree(item) for item in value]}
    if kind in _CANONICAL_CLASSES:
        return {"type": "dataclass", "name": kind.__name__,
                "fields": {field.name: _finance_tree(getattr(value, field.name)) for field in fields(value)}}
    raise TypeError("finance_canonical_type_invalid")


def _finance_canonical_bytes(value: object) -> bytes:
    """Versioned callers supply the domain; scalars never coerce across types."""
    return json.dumps(_finance_tree(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def replay_environment_digest(sample_points: tuple[tuple[float, float], ...], *, house_hold_s: float) -> str:
    """Ordered step samples, known until min(next sample, sample + hold), exclusive.

    The final sample is the exclusive replay boundary. Expiry of the explicit
    hold makes the house input unknown; no interpolation or assumed hold exists.
    """
    points = replay_execution._samples(sample_points)
    replay_execution._number(house_hold_s, minimum=0)
    if house_hold_s == 0:
        raise ValueError("source_house_hold_invalid")
    return sha256(_finance_canonical_bytes(("house_step_hold_v1", ("sample_points", points),
        ("house_hold_s", house_hold_s), ("right_boundary", "exclusive")))).hexdigest()


def _installation_ids(value: object) -> tuple[str, ...]:
    """Recorded bindings/commands/receipts must name the same v2 installation."""
    if type(value) is tuple:
        return tuple(identity for item in value for identity in _installation_ids(item))
    if is_dataclass(value):
        own = (value.installation_id,) if hasattr(value, "installation_id") else ()
        return own + tuple(identity for field in fields(value)
                           for identity in _installation_ids(getattr(value, field.name)))
    return ()


def _source_mismatch(evidence: ReplaySourceEvidence, binding: ReplaySourceBinding) -> bool:
    return (any(getattr(evidence, name) != getattr(binding, name) for name in (
                "installation_id", "meter_id", "replay_basis_hash", "source_plan_digest", "battery_names"))
            or (binding.environment_digest is not None and evidence.environment_digest != binding.environment_digest)
            or evidence.content_hash != binding.source_digest)


def bind_source_evidence(evidence: ReplaySourceEvidence | None, *, source_plan: FastPlan,
                         replay_basis: ReplayBasis, replay_inputs: tuple[ReplayInput, ...]) -> ReplaySourceBinding:
    """Pin common inputs before either run; source topology remains caller asserted."""
    if evidence is not None:
        _typed(evidence, ReplaySourceEvidence)
    _typed(source_plan, FastPlan)
    _typed(replay_basis, ReplayBasis)
    inputs = _tuple(replay_inputs)
    if len(inputs) not in (1, 2):
        raise ValueError("source_replay_inputs_invalid")
    for item in inputs:
        _typed(item, ReplayInput)
    original_digest = plan_digest(source_plan)
    names = _identities(tuple(spec.name for spec in source_plan.specs))
    context = source_plan.financial_context
    if context is None:
        raise ValueError("source_financial_context_required")
    installation = context.basis.meter_id
    reasons = []
    try:
        replay_basis.check_context(context)
    except ValueError:
        reasons.append("source_binding_mismatch")
    sample_bytes = {_finance_canonical_bytes(item.sample_points) for item in inputs}
    if len(sample_bytes) != 1:
        reasons.append("source_binding_mismatch")
    environments = []
    for item in inputs:
        if _identities(tuple(name for name, _ in item.soc_by_battery)) != names:
            reasons.append("source_binding_mismatch")
        scenario = item.scenario
        if scenario is None:
            reasons.append("source_counterfactual_scope_unproven")
            continue
        environments.append(replay_environment_digest(item.sample_points, house_hold_s=scenario.physics.house_hold_s))
        if type(scenario.execution) is RecordedExecution and any(
                identity != installation for identity in _installation_ids(scenario.execution)):
            reasons.append("source_binding_mismatch")
    if len(set(environments)) > 1:
        reasons.append("source_binding_mismatch")
    environment = environments[0] if len(environments) == len(inputs) and len(set(environments)) == 1 else None
    if evidence is None:
        reasons.append("source_evidence_missing")
    elif evidence.counterfactual_scope != "both_alternatives":
        reasons.append("source_counterfactual_scope_unproven")
    binding = ReplaySourceBinding(installation, replay_basis.basis.meter_id, replay_basis.content_hash,
        original_digest, environment, names, None if evidence is None else evidence.content_hash, tuple(reasons))
    if evidence is not None and _source_mismatch(evidence, binding):
        binding = replace(binding, reasons=(*binding.reasons, "source_binding_mismatch"))
    return binding


def validate_source_evidence(evidence: ReplaySourceEvidence | None, binding: ReplaySourceBinding) -> SourceValidation:
    """Wrong bindings block; absent scope supplies no usable positive spans.

    Valid means a correctly bound package, not full interval coverage or external
    verification. Unknown spans and gaps remain available for interval projection.
    """
    _typed(binding, ReplaySourceBinding)
    if evidence is not None:
        _typed(evidence, ReplaySourceEvidence)
    reasons = list(binding.reasons)
    if evidence is not None and _source_mismatch(evidence, binding):
        reasons.append("source_binding_mismatch")
    if "source_binding_mismatch" in reasons:
        return SourceValidation("mismatch", None, binding, tuple(reasons))
    if evidence is None:
        reasons.append("source_evidence_missing")
    if binding.environment_digest is None or (evidence is not None and evidence.counterfactual_scope != "both_alternatives"):
        reasons.append("source_counterfactual_scope_unproven")
    if reasons:
        return SourceValidation("missing", None, binding, tuple(reasons))
    return SourceValidation("valid", evidence, binding, ())


def replay_scenario_digest(*, source_plan: FastPlan, execution_plan: FastPlan,
                           replay_input: ReplayInput, trace: tuple[ReplayEvent, ...]) -> str:
    """Pin the execution recipe and resulting trace, not runtime object identity."""
    _typed(source_plan, FastPlan)
    _typed(execution_plan, FastPlan)
    _typed(replay_input, ReplayInput)
    events = _tuple(trace)
    for event in events:
        if type(event) not in (replay_execution.ProposalEvent, replay_execution.ReceiptEvent,
                              replay_execution.StopRequestEvent, replay_execution.BarrierEvent,
                              replay_execution.BindingEvent):
            raise TypeError("source_trace_type_invalid")
    replay_execution._ordered(events)
    content = ("replay_scenario_v1", ("execution_model", "execution_model_v1"),
        ("battery_model", "battery_model_v1"), ("source_plan_digest", plan_digest(source_plan)),
        ("execution_plan_digest", plan_digest(execution_plan)),
        ("source_battery_specs", tuple(source_plan.specs)), ("execution_battery_specs", tuple(execution_plan.specs)),
        ("replay_input", replay_input), ("trace", events))
    return sha256(_finance_canonical_bytes(content)).hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectedQuantity(_FrozenDTO):
    """Normalized model energy, with a signed raw-minus-normalized residue."""

    raw: Decimal | None
    normalized: Decimal | None
    residue: Decimal | None

    def __post_init__(self) -> None:
        _FrozenDTO.__post_init__(self)
        if self.raw is None:
            if self.normalized is not None or self.residue is not None:
                raise ValueError("projection_quantity_invalid")
        elif (self.normalized is None or self.residue is None or self.raw < 0
              or self.normalized < 0 or self.normalized.as_tuple().exponent < -9):
            raise ValueError("projection_quantity_invalid")
        else:
            with localcontext(Context(prec=36)):
                if self.raw - self.normalized != self.residue:
                    raise ValueError("projection_residue_invalid")


@dataclass(frozen=True, slots=True)
class ProjectionSlice(_FrozenDTO):
    start_ts: int
    end_ts: int
    segment_indices: tuple[int, ...]
    import_energy: ProjectedQuantity
    export_energy: ProjectedQuantity
    battery_discharge: ProjectedQuantity
    dc_charge_by_battery: tuple[tuple[str, Decimal | None], ...]
    dc_discharge_by_battery: tuple[tuple[str, Decimal | None], ...]
    source_span_indices: tuple[int, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _FrozenDTO.__post_init__(self)
        _window(self.start_ts, self.end_ts)
        for indices in (self.segment_indices, self.source_span_indices):
            if tuple(sorted(set(indices))) != indices or any(index < 0 for index in indices):
                raise ValueError("projection_indices_invalid")
        for flows in (self.dc_charge_by_battery, self.dc_discharge_by_battery):
            _identities(tuple(name for name, _ in flows))
            if any(value is not None and value < 0 for _, value in flows):
                raise ValueError("projection_energy_invalid")
        if tuple(name for name, _ in self.dc_charge_by_battery) != tuple(name for name, _ in self.dc_discharge_by_battery):
            raise ValueError("projection_battery_identity_invalid")
        object.__setattr__(self, "reasons", _reasons(self.reasons))


def _physical_type(value: object, name: str) -> None:
    # The producer imports this adapter in later slices. Do not import simulate
    # back at runtime, or retain mutable policy/execution decisions in these DTOs.
    kind = type(value)
    if kind.__module__ != f"{__package__}.simulate" or kind.__name__ != name or not is_dataclass(value):
        raise TypeError("projection_segment_type_invalid")


def _physical_number(value: object, *, energy: bool = False) -> None:
    if value is not None:
        replay_execution._number(value, minimum=0 if energy else None)


def _physical_segments(value: object) -> tuple[ReplaySegment, ...]:
    segments = _tuple(value)
    previous = None
    for segment in segments:
        _physical_type(segment, "ReplaySegment")
        replay_execution._number(segment.start_ts)
        replay_execution._number(segment.end_ts)
        if segment.start_ts >= segment.end_ts or (previous is not None and segment.start_ts < previous):
            raise ValueError("projection_segment_order_invalid")
        previous = segment.end_ts
        _physical_number(segment.house_w)
        _physical_number(segment.import_kwh, energy=True)
        _physical_number(segment.export_kwh, energy=True)
        if type(segment.batteries) is not tuple:
            raise TypeError("projection_batteries_not_immutable")
        for battery in segment.batteries:
            _physical_type(battery, "BatteryReplaySegment")
            _text(battery.battery_name)
            _text(battery.execution_status)
            if type(battery.reasons) is not tuple:
                raise TypeError("projection_reasons_not_immutable")
            _reasons(battery.reasons)
            if battery.command is not None:
                _typed(battery.command, command.BatteryCommand)
            for name in ("charge_ac_kwh", "discharge_ac_kwh", "dc_throughput_kwh"):
                _physical_number(getattr(battery, name), energy=True)
            for name in ("command_w", "modeled_ac_w", "cell_delta_kwh", "soc_begin", "soc_end"):
                _physical_number(getattr(battery, name))
        _identities(tuple(battery.battery_name for battery in segment.batteries))
    return segments


@dataclass(frozen=True, slots=True)
class ReplayProjection:
    version: Literal["replay_projection_v1"]
    start_ts: float
    end_ts: float
    scenario_digest: str
    source: SourceValidation
    projection_digest: str
    segments: tuple[ReplaySegment, ...]
    items: tuple[InputInterval, ...]
    slices: tuple[ProjectionSlice, ...]
    reasons: tuple[str, ...]
    blocking_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _choice(self.version, ("replay_projection_v1",))
        replay_execution._number(self.start_ts)
        replay_execution._number(self.end_ts)
        if self.start_ts >= self.end_ts:
            raise ValueError("projection_window_invalid")
        _digest(self.scenario_digest)
        _digest(self.projection_digest)
        _typed(self.source, SourceValidation)
        object.__setattr__(self, "segments", _physical_segments(self.segments))
        for name, kind in (("items", InputInterval), ("slices", ProjectionSlice)):
            values = _tuple(getattr(self, name))
            for value in values:
                _typed(value, kind)
            object.__setattr__(self, name, values)
        for name in ("reasons", "blocking_reasons"):
            object.__setattr__(self, name, _reasons(getattr(self, name)))


_CANONICAL_CLASSES = _CANONICAL_CLASSES | frozenset((ProjectedQuantity, ProjectionSlice))


def financial_edges(*, start_ts: int, end_ts: int, replay_basis: ReplayBasis) -> tuple[int, ...]:
    """Money boundaries from pinned terms; physical/source events are not edges.

    Calendar dates advance locally, price windows advance by the existing UTC
    identity of _spot_window. Thus both folds survive and days may be 23/25 hours.
    """
    _window(start_ts, end_ts)
    _typed(replay_basis, ReplayBasis)
    edges = {start_ts, end_ts}
    edges.update(range((start_ts // 900 + 1) * 900, end_ts, 900))
    for archived in replay_basis.quotes:
        edges.update((archived.start_ts, archived.end_ts))
    for snapshot in replay_basis.snapshots:
        lower = max(start_ts, snapshot.valid_from_ts)
        upper = min(end_ts, snapshot.valid_to_ts if snapshot.valid_to_ts is not None else end_ts)
        if lower >= upper:
            continue
        edges.update((lower, upper))
        zone = ZoneInfo(snapshot.timezone)
        for name in TARIFF_FIELDS:
            edges.update(_midnight(day, zone) for day, _ in getattr(snapshot, name)
                         if lower <= _midnight(day, zone) <= upper)
        if snapshot.supply_start is not None:
            boundary = _midnight(snapshot.supply_start, zone)
            if lower <= boundary <= upper:
                edges.add(boundary)
        bonus = snapshot.bonus
        if bonus is not None and bonus.enabled:
            boundary = _midnight(bonus.start_date, zone)
            if lower <= boundary <= upper:
                edges.add(boundary)
        day = datetime.fromtimestamp(lower, zone).date()
        last = datetime.fromtimestamp(upper, zone).date()
        while day <= last:
            # Includes month boundaries; never approximate a month by 30 days.
            boundaries = [_midnight(day, zone), _midnight(day + timedelta(days=1), zone)]
            if snapshot.supply_start is not None:
                boundaries.extend(_midnight(value, zone) for value in delivery_year_bounds(snapshot.supply_start, day))
            if bonus is not None and bonus.enabled:
                for hour in dict(bonus.windows)[day.month]:
                    boundary_day = day + timedelta(days=1) if hour == 24 else day
                    boundaries.append(int(datetime.combine(boundary_day, time(hour % 24), zone).timestamp()))
            edges.update(boundary for boundary in boundaries if lower <= boundary <= upper)
            day += timedelta(days=1)
        cursor = lower
        while cursor < upper:
            window_start, window_end, _ = _spot_window(snapshot, cursor, cursor + 1)
            edges.update(boundary for boundary in (window_start, window_end) if lower <= boundary <= upper)
            cursor = window_end
    return tuple(sorted(edge for edge in edges if start_ts <= edge <= end_ts))


def _binary_decimal(value: float | int) -> Decimal:
    return Decimal.from_float(value) if type(value) is float else Decimal(value)


def _parts(value: float | int | None, cuts: tuple[Decimal, ...]) -> tuple[Decimal | None, ...]:
    """Partition one original gross channel, with its final context36 remainder.

    Cuts include the whole physical segment, including portions outside the
    requested window. Clipping must not donate their energy to the last item.
    """
    if value is None:
        return (None,) * (len(cuts) - 1)
    raw = _binary_decimal(value)
    duration, allocated = cuts[-1] - cuts[0], Decimal(0)
    result = []
    for index, (start, end) in enumerate(zip(cuts, cuts[1:], strict=False)):
        part = raw - allocated if index == len(cuts) - 2 else raw * (end - start) / duration
        result.append(part)
        allocated += part
    return tuple(result)


def _dc_directions(battery: BatteryReplaySegment) -> tuple[float | None, float | None]:
    throughput = battery.dc_throughput_kwh
    if throughput is None:
        return None, None
    charge, discharge, response = battery.charge_ac_kwh, battery.discharge_ac_kwh, battery.modeled_ac_w
    charging = (charge is not None and charge > 0) or (response is not None and response > 0)
    discharging = (discharge is not None and discharge > 0) or (response is not None and response < 0)
    if charging and not discharging:
        return throughput, 0
    if discharging and not charging:
        return 0, throughput
    if not charging and not discharging and throughput == 0 and (response == 0 or charge == discharge == 0):
        return 0, 0
    return None, None


def _source_cover(source: SourceValidation, start: int, end: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    evidence = source.evidence
    if evidence is None:
        return (), _reasons((*source.reasons, "source_evidence_missing"))
    indices, reasons, cursor = [], list(source.reasons), start
    for index, span in enumerate(evidence.spans):
        if span.end_ts <= start or span.start_ts >= end:
            continue
        indices.append(index)
        if span.start_ts > cursor:
            reasons.append("source_coverage_gap")
        if span.claim != "pv_only_nonbattery_generation":
            reasons.append("source_evidence_missing")
        cursor = min(end, span.end_ts)
    if cursor < end:
        reasons.append("source_coverage_gap")
    return tuple(indices), _reasons(reasons)


def _add_known(left: Decimal | None, right: Decimal | None) -> Decimal | None:
    return None if left is None or right is None else left + right


def _quantity(raw: Decimal | None, *, discharge: bool = False) -> ProjectedQuantity:
    if raw is None:
        return ProjectedQuantity(None, None, None)
    normalized = raw.quantize(Decimal("1e-9"), rounding=ROUND_CEILING if discharge else ROUND_HALF_EVEN)
    return ProjectedQuantity(raw, normalized, raw - normalized)


def _project_slices(segments: tuple[ReplaySegment, ...], edges: tuple[int, ...], source: SourceValidation) -> tuple[ProjectionSlice, ...]:
    names = source.binding.battery_names
    # Mutable accumulators are local to this projection, never stored in a DTO.
    coverage = [Decimal(0) for _ in edges[:-1]]
    indices = [[] for _ in edges[:-1]]
    channels = [[Decimal(0), Decimal(0), *(Decimal(0) for _ in names)] for _ in edges[:-1]]
    dc_charge = [{name: Decimal(0) for name in names} for _ in edges[:-1]]
    dc_discharge = [{name: Decimal(0) for name in names} for _ in edges[:-1]]
    for segment_index, segment in enumerate(segments):
        if segment.end_ts <= edges[0] or segment.start_ts >= edges[-1]:
            continue
        first, last = bisect_right(edges, segment.start_ts), bisect_left(edges, segment.end_ts)
        cuts = tuple(_binary_decimal(value) for value in (segment.start_ts, *edges[first:last], segment.end_ts))
        batteries = {battery.battery_name: battery for battery in segment.batteries}
        flows = [_parts(segment.import_kwh, cuts), _parts(segment.export_kwh, cuts)]
        charge_parts, discharge_parts = {}, {}
        for name in names:
            battery = batteries.get(name)
            flows.append(_parts(None if battery is None else battery.discharge_ac_kwh, cuts))
            charge, discharge = (None, None) if battery is None else _dc_directions(battery)
            charge_parts[name], discharge_parts[name] = _parts(charge, cuts), _parts(discharge, cuts)
        for part, (start, end) in enumerate(zip(cuts, cuts[1:], strict=False)):
            if start < edges[0] or end > edges[-1]:
                continue
            index = bisect_right(edges, start) - 1
            coverage[index] += end - start
            indices[index].append(segment_index)
            for channel, values in enumerate(flows):
                channels[index][channel] = _add_known(channels[index][channel], values[part])
            for name in names:
                dc_charge[index][name] = _add_known(dc_charge[index][name], charge_parts[name][part])
                dc_discharge[index][name] = _add_known(dc_discharge[index][name], discharge_parts[name][part])
    slices = []
    for index, (start, end) in enumerate(zip(edges, edges[1:], strict=False)):
        span_indices, source_reasons = _source_cover(source, start, end)
        reasons = list(source_reasons)
        if coverage[index] != end - start:
            reasons.append("projection_physical_gap")
            channels[index] = [None] * len(channels[index])
            dc_charge[index] = dc_discharge[index] = dict.fromkeys(names)
        imported, exported, *discharges = channels[index]
        discharge = None if any(value is None for value in discharges) else sum(discharges, Decimal(0))
        if any(value is None for value in (imported, exported, discharge)):
            reasons.append("projection_quantity_unknown")
        if any(value is None for value in (*dc_charge[index].values(), *dc_discharge[index].values())):
            reasons.append("projection_dc_throughput_unknown")
        slices.append(ProjectionSlice(start, end, tuple(indices[index]), _quantity(imported), _quantity(exported),
            _quantity(discharge, discharge=True), tuple(dc_charge[index].items()), tuple(dc_discharge[index].items()),
            span_indices, tuple(reasons)))
    return tuple(slices)


def _price_provenance(slices: tuple[ProjectionSlice, ...], replay_basis: ReplayBasis) -> tuple[tuple[ProjectionSlice, ...], tuple[str, ...]]:
    spots = tuple(row.spot for row in replay_basis.price_batch.items)
    result, blocking = [], []
    for piece in slices:
        start, end = piece.start_ts, piece.end_ts
        archived = tuple(price for price in replay_basis.quotes if price.start_ts <= start and end <= price.end_ts)
        try:
            snapshot = resolve_snapshot(replay_basis.snapshots, start, end)
        except ValueError:
            blocking.append("replay_price_provenance_mismatch")
            result.append(piece)
            continue
        if len(archived) != 1 or archived[0].snapshot != snapshot:
            blocking.append("replay_price_provenance_mismatch")
        resolved = resolve_spot(snapshot, start, end, spots)
        if resolved.price_ex_vat is None:
            # An archived quote cannot fill a hole in its raw price dependency.
            piece = replace(piece, reasons=(*piece.reasons, "replay_price_missing"))
        elif len(archived) == 1 and quote(snapshot, start, end, resolved.price_ex_vat) != replace(archived[0], start_ts=start, end_ts=end):
            blocking.append("replay_price_provenance_mismatch")
        result.append(piece)
    return tuple(result), _reasons(blocking)


def _projection_hash(*, start_ts: float, end_ts: float, segments: tuple[ReplaySegment, ...],
                     scenario_digest: str, source: SourceValidation, slices: tuple[ProjectionSlice, ...],
                     blocking: tuple[str, ...]) -> str:
    # Preserve original float bits, including signed zero and fractional events.
    # Physical DTOs are read transiently; there is no InputInterval/hash cycle.
    physical = tuple(("ReplaySegment", tuple((field.name,
        tuple(("BatteryReplaySegment", tuple((part.name, getattr(battery, part.name)) for part in fields(battery)))
              for battery in segment.batteries) if field.name == "batteries" else getattr(segment, field.name))
        for field in fields(segment))) for segment in segments)
    content = ("replay_projection_v1", start_ts, end_ts, scenario_digest, source, physical, slices, blocking)
    return sha256(_finance_canonical_bytes(content)).hexdigest()


def project_replay_inputs(*, segments: tuple[ReplaySegment, ...], start_ts: float, end_ts: float,
                          replay_basis: ReplayBasis, source: SourceValidation, scenario_digest: str) -> ReplayProjection:
    """Pure context36 model normalization into existing, unregistered engine DTOs."""
    segments = _physical_segments(segments)
    replay_execution._number(start_ts)
    replay_execution._number(end_ts)
    if end_ts <= start_ts:
        raise ValueError("projection_window_invalid")
    _typed(replay_basis, ReplayBasis)
    _typed(source, SourceValidation)
    _digest(scenario_digest)
    blocking = []
    if source.status == "mismatch" or (source.binding.replay_basis_hash != replay_basis.content_hash
            or source.binding.meter_id != replay_basis.basis.meter_id
            or any(b.battery_name not in source.binding.battery_names for segment in segments for b in segment.batteries)):
        blocking.append("source_binding_mismatch")
    integer_window = all(type(value) is int or value.is_integer() for value in (start_ts, end_ts))
    if not integer_window:
        blocking.append("projection_noninteger_window")
    elif start_ts != replay_basis.opening.asof_ts:
        blocking.append("replay_opening_mismatch")
    slices, items = (), ()
    with localcontext(Context(prec=36)):
        if not blocking:
            edges = financial_edges(start_ts=int(start_ts), end_ts=int(end_ts), replay_basis=replay_basis)
            slices = _project_slices(segments, edges, source)
            slices, price_blocking = _price_provenance(slices, replay_basis)
            blocking.extend(price_blocking)
        blocked = _reasons(blocking)
        digest = _projection_hash(start_ts=start_ts, end_ts=end_ts, segments=segments,
                                  scenario_digest=scenario_digest, source=source, slices=slices, blocking=blocked)
        if not blocked:
            readings = []
            common_sources = ("model:replay_projection_v1", f"scenario:{scenario_digest}", f"projection:{digest}",
                "source:missing" if source.evidence is None else f"source_version:{source.evidence.version}",
                "source_trust:caller_asserted")
            if source.binding.source_digest is not None:
                common_sources += (f"source:{source.binding.source_digest}",)
            for piece in slices:
                _, source_reasons = _source_cover(source, piece.start_ts, piece.end_ts)
                quantities = (piece.import_energy.normalized, piece.export_energy.normalized, piece.battery_discharge.normalized)
                readings.append(MeterInterval(piece.start_ts, piece.end_ts, *quantities,
                    "unknown" if source_reasons else "verified_pv",
                    "incomplete" if any(value is None for value in quantities) else "estimated",
                    (*common_sources, *(f"source_span:{index}" for index in piece.source_span_indices),
                     *(f"issue:{reason}" for reason in piece.reasons))))
            items = merge_inputs(tuple(readings), tuple(row.spot for row in replay_basis.price_batch.items),
                                 replay_basis.snapshots, meter_id=source.binding.meter_id)
        reasons = _reasons((*source.reasons, *(reason for piece in slices for reason in piece.reasons), *blocked))
        return ReplayProjection("replay_projection_v1", start_ts, end_ts, scenario_digest, source,
                                digest, segments, items, slices, reasons, blocked)


FinancialStatus = Literal["complete", "partial", "unknown"]
PeriodStatus = Literal["closed", "as_of_open_period", "not_applicable", "unknown"]
ReconciliationMetric = Literal["import_cost", "base_export_revenue", "bonus_revenue", "month_credit",
    "invoice_energy_cost", "wear_cost", "terminal_value", "total_cost", "advantage_eur"]
ReconciliationStatus = Literal["equal", "difference", "unknown"]


def _booked_coefficient(value: Decimal) -> tuple[int, int]:
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError("reconciliation_amount_invalid")
    sign, digits, exponent = value.as_tuple()
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    return (-coefficient if sign else coefficient), exponent


def _sum_booked_coefficients(terms: tuple[tuple[int, int], ...]) -> Decimal:
    """Lossless diagnostic arithmetic only; never revalue a financial booking."""
    if not terms:
        return Decimal(0)
    exponent = min(exponent for _, exponent in terms)
    coefficient = sum(coefficient * 10 ** (power - exponent) for coefficient, power in terms)
    if coefficient == 0:
        return Decimal(0)
    sign, coefficient = int(coefficient < 0), abs(coefficient)
    digits = []
    # No int-to-text limit, Decimal context, or Decimal unary operation.
    while coefficient:
        coefficient, digit = divmod(coefficient, 10)
        digits.append(digit)
    return Decimal((sign, tuple(reversed(digits)), exponent))


def _exact_booked_sum(values: tuple[Decimal, ...]) -> Decimal:
    return _sum_booked_coefficients(tuple(_booked_coefficient(value) for value in _tuple(values)))


@dataclass(frozen=True, slots=True)
class FinancialReconciliation(_FrozenDTO):
    """Observed total-minus-days difference, not proof of its cause or a booking."""

    metric: ReconciliationMetric
    authoritative_total: Decimal | None
    day_sum: Decimal | None
    difference: Decimal | None
    status: ReconciliationStatus
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        # _FrozenDTO accepts decimal text elsewhere; this boundary must not.
        for value in (self.authoritative_total, self.day_sum, self.difference):
            if value is not None:
                _booked_coefficient(value)
        _FrozenDTO.__post_init__(self)
        object.__setattr__(self, "reasons", _reasons(self.reasons))
        if self.status == "unknown":
            valid = self.day_sum is None and self.difference is None and bool(self.reasons)
        else:
            valid = (all(value is not None for value in (self.authoritative_total, self.day_sum, self.difference))
                and not self.reasons and (self.status == "equal") == (self.difference == 0)
                and _exact_booked_sum((self.day_sum, self.difference)) == self.authoritative_total)
        if not valid:
            raise ValueError("reconciliation_contract_invalid")


def _reconcile_days(metric: ReconciliationMetric, total: Decimal | None,
                    daily_values: tuple[Decimal | None, ...], *, reasons: tuple[str, ...] = ()) -> FinancialReconciliation:
    daily_values = _tuple(daily_values)
    for value in (total, *daily_values):
        if value is not None:
            _booked_coefficient(value)
    reasons = _reasons(reasons)
    missing = tuple(reason for condition, reason in (
        (total is None, "reconciliation_total_unknown"),
        (any(value is None for value in daily_values), "reconciliation_day_unknown"),
        (not daily_values or "reconciliation_coverage_unknown" in reasons, "reconciliation_coverage_unknown"),
    ) if condition)
    if missing:
        return FinancialReconciliation(metric, total, None, None, "unknown", _reasons((*missing, *reasons)))
    day_sum = _exact_booked_sum(daily_values)
    coefficient, exponent = _booked_coefficient(day_sum)
    difference = _sum_booked_coefficients((_booked_coefficient(total), (-coefficient, exponent)))
    return FinancialReconciliation(metric, total, day_sum, difference, "equal" if difference == 0 else "difference", ())


@dataclass(frozen=True, slots=True)
class BatteryFinancialTotals(_FrozenDTO):
    battery_name: str
    capacity_kwh: Decimal
    cycle_cost: Decimal
    soc_start: Decimal | None
    soc_end: Decimal | None

    def __post_init__(self) -> None:
        _FrozenDTO.__post_init__(self)
        _text(self.battery_name)
        if self.capacity_kwh <= 0 or self.cycle_cost < 0:
            raise ValueError("financial_battery_parameters_invalid")


@dataclass(frozen=True, slots=True)
class ReplayPhysicalTotals(_FrozenDTO):
    batteries: tuple[BatteryFinancialTotals, ...]

    def __post_init__(self) -> None:
        _FrozenDTO.__post_init__(self)
        _identities(tuple(battery.battery_name for battery in self.batteries))


@dataclass(frozen=True, slots=True)
class FinancialComponent(_FrozenDTO):
    value: Decimal | None
    known_subtotal: Decimal
    status: FinancialStatus
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _FrozenDTO.__post_init__(self)
        object.__setattr__(self, "reasons", _reasons(self.reasons))
        if ((self.value is not None) != (self.status == "complete")
                or (self.value is not None and self.value != self.known_subtotal)):
            raise ValueError("financial_component_coverage_invalid")


@dataclass(frozen=True, slots=True)
class ReplayCostComponents(_FrozenDTO):
    import_cost: FinancialComponent
    base_export_revenue: FinancialComponent
    bonus_revenue: FinancialComponent
    month_credit: FinancialComponent
    invoice_energy_cost: FinancialComponent
    wear_cost: FinancialComponent
    terminal_value: FinancialComponent
    total_cost: FinancialComponent


@dataclass(frozen=True, slots=True)
class FinancialDayKey(_FrozenDTO):
    local_date: str
    timezone: str
    contract_id: str
    contract_start_ts: int | None

    def __post_init__(self) -> None:
        _FrozenDTO.__post_init__(self)
        if date.fromisoformat(self.local_date).isoformat() != self.local_date:
            raise ValueError("financial_day_invalid")
        ZoneInfo(self.timezone)
        _text(self.contract_id)
        if self.contract_start_ts is not None:
            _utc(self.contract_start_ts)


@dataclass(frozen=True, slots=True)
class ReplayFinancialDay(_FrozenDTO):
    key: FinancialDayKey
    start_ts: int
    end_ts: int
    components: ReplayCostComponents
    financial_status: FinancialStatus
    reasons: tuple[str, ...]
    period_status: PeriodStatus


@dataclass(frozen=True, slots=True)
class ReplayFinancialResult(_FrozenDTO):
    projection: ReplayProjection | None
    settlement: SettlementResult | None
    components: ReplayCostComponents
    days: tuple[ReplayFinancialDay, ...]
    financial_status: FinancialStatus
    financial_reasons: tuple[str, ...]
    period_status: PeriodStatus
    source_trust: Literal["caller_asserted", "missing"]
    economics_is_estimate: bool = True
    economics_kind: Literal["settlement_counterfactual"] = "settlement_counterfactual"
    reconciliation: tuple[FinancialReconciliation, ...] = ()


@dataclass(frozen=True, slots=True)
class ReplayDayAdvantage(_FrozenDTO):
    key: FinancialDayKey
    advantage_eur: Decimal | None


@dataclass(frozen=True, slots=True)
class ReplayFinancialComparison(_FrozenDTO):
    advantage_eur: Decimal | None
    financial_status: FinancialStatus
    reasons: tuple[str, ...]
    days: tuple[ReplayDayAdvantage, ...]
    reconciliation: tuple[FinancialReconciliation, ...] = ()


def unknown_replay_financial(*, reasons: tuple[str, ...]) -> ReplayFinancialResult:
    reasons = _reasons(reasons)
    unknown = FinancialComponent(None, Decimal(0), "unknown", reasons)
    components = ReplayCostComponents(*(unknown for _ in fields(ReplayCostComponents)))
    return _with_reconciliation(ReplayFinancialResult(None, None, components, (), "unknown", reasons, "unknown", "missing"))


def _component(amounts: tuple[Decimal | None, ...], reasons: tuple[str, ...]) -> FinancialComponent:
    """Coverage is about every required amount, including known zero amounts."""
    known = tuple(amount for amount in amounts if amount is not None)
    subtotal = sum(known, Decimal(0))
    complete = len(known) == len(amounts)
    return FinancialComponent(subtotal if complete else None, subtotal,
        "complete" if complete else "partial" if known else "unknown", () if complete else _reasons(reasons))


def _invoice_component(value: Decimal | None, subtotal: Decimal,
                       components: tuple[FinancialComponent, ...], reasons: tuple[str, ...]) -> FinancialComponent:
    status = ("complete" if value is not None else
              "partial" if any(component.status != "unknown" for component in components) else "unknown")
    return FinancialComponent(value, subtotal, status, () if value is not None else _reasons(reasons))


def _cost_components(lines: tuple[SettlementLine, ...], credits: tuple[Decimal | None, ...], *,
                     invoice_value: Decimal | None, invoice_subtotal: Decimal,
                     wear: FinancialComponent, terminal: FinancialComponent) -> ReplayCostComponents:
    energy = tuple(_component(tuple(getattr(line, name) for line in lines),
        tuple(issue for line in lines if getattr(line, name) is None for issue in line.issues))
        for name in ("import_cost", "base_export_revenue", "bonus_revenue"))
    credit = _component(credits, ("replay_month_credit_unknown",))
    invoice = _invoice_component(invoice_value, invoice_subtotal, energy,
        (*(issue for line in lines for issue in line.issues), *credit.reasons))
    required = (invoice, wear, terminal)
    complete = all(component.value is not None for component in required)
    subtotal = invoice.known_subtotal + wear.known_subtotal - terminal.known_subtotal
    total = _invoice_component(subtotal if complete else None, subtotal, required,
                                tuple(reason for component in required for reason in component.reasons))
    return ReplayCostComponents(*energy, credit, invoice, wear, terminal, total)


def _wear_components(slices: tuple[ProjectionSlice, ...], totals: ReplayPhysicalTotals) -> tuple[FinancialComponent, ...]:
    costs = {battery.battery_name: battery.cycle_cost for battery in totals.batteries}
    result = []
    for piece in slices:
        amounts, reasons = [], []
        for direction, flows in (("charge", piece.dc_charge_by_battery), ("discharge", piece.dc_discharge_by_battery)):
            for name, throughput in flows:
                if throughput is None:
                    amounts.append(None)
                    reasons.append(f"replay_dc_{direction}_unknown: {name}")
                else:
                    amounts.append(battery_wear_cost(
                        dc_charge_kwh=throughput if direction == "charge" else Decimal(0),
                        dc_discharge_kwh=throughput if direction == "discharge" else Decimal(0),
                        eur_per_dc_bus_throughput_kwh=costs[name]))
        result.append(_component(tuple(amounts), tuple(reasons)))
    return tuple(result)


def _sum_components(components: tuple[FinancialComponent, ...]) -> FinancialComponent:
    subtotal = sum((component.known_subtotal for component in components), Decimal(0))
    complete = all(component.value is not None for component in components)
    return _invoice_component(subtotal if complete else None, subtotal, components,
                              tuple(reason for component in components for reason in component.reasons))


def _terminal_component(totals: ReplayPhysicalTotals, replay_basis: ReplayBasis) -> FinancialComponent:
    archived = replay_basis.terminal_value
    efficiencies = dict(archived.reference_efficiency_by_battery)
    amounts, reasons = [], []
    for battery in totals.batteries:
        if battery.soc_start is None or battery.soc_end is None:
            amounts.append(None)
            for name in ("soc_start", "soc_end"):
                if getattr(battery, name) is None:
                    reasons.append(f"replay_{name}_unknown: {battery.battery_name}")
        else:
            amounts.append((battery.soc_end - battery.soc_start) * battery.capacity_kwh / 100
                           * efficiencies[battery.battery_name] * archived.eur_per_ac_kwh)
    return _component(tuple(amounts), tuple(reasons))


def _closing_credits(lines: tuple[SettlementLine, ...], adjustments: tuple[PeriodAdjustment, ...]) -> tuple[Decimal | None, ...]:
    """Posted credits only. An open period's zero is not a future closing credit."""
    closed = {(adjustment.period_start_ts, adjustment.period_end_ts): adjustment.credit_total
              for adjustment in adjustments}
    credits = []
    for line in lines:
        month = line.after_state.month
        if month is not None and line.end_ts == month.period_end_ts:
            credits.append(closed.get((month.period_start_ts, month.period_end_ts)))
        else:
            credits.append(Decimal(0))
    return tuple(credits)


def _period_status(lines: tuple[SettlementLine, ...], snapshots: tuple[TariffSnapshot, ...],
                    credits: tuple[Decimal | None, ...]) -> PeriodStatus:
    periods = {}
    for line, snapshot, credit in zip(lines, snapshots, credits, strict=True):
        if snapshot.monthly_export_floor == "none":
            continue
        month = line.after_state.month
        if month is None:
            return "unknown"
        key = (month.contract_id, month.period_start_ts, month.period_end_ts)
        periods[key] = ("unknown" if not month.complete or credit is None else
                        "closed" if line.end_ts == month.period_end_ts else "as_of_open_period")
    if not periods:
        return "not_applicable"
    if "unknown" in periods.values():
        return "unknown"
    return "as_of_open_period" if "as_of_open_period" in periods.values() else "closed"


def _require_book_structure(condition: bool) -> None:
    if not condition:
        raise ValueError("replay_reconciliation_structure_invalid")


def _closing_period(line: SettlementLine) -> tuple[int, int] | None:
    month = line.after_state.month
    if month is not None and line.end_ts == month.period_end_ts:
        return month.period_start_ts, month.period_end_ts
    return None


def _validate_book_structure(projection: ReplayProjection, settlement: SettlementResult,
                             replay_basis: ReplayBasis) -> None:
    """Check correspondence and state/period identity, never recompute money."""
    lines = settlement.lines
    _require_book_structure(bool(lines) and len(lines) == len(projection.items) == len(projection.slices))
    _require_book_structure((settlement.coverage_start_ts, settlement.coverage_end_ts)
                            == (projection.start_ts, projection.end_ts)
                            and settlement.coverage_start_ts == replay_basis.opening.asof_ts)
    state, closes = replay_basis.opening, {}
    for line, item, piece in zip(lines, projection.items, projection.slices, strict=True):
        _require_book_structure((line.start_ts, line.end_ts) == (item.start_ts, item.end_ts) == (piece.start_ts, piece.end_ts))
        _require_book_structure((line.meter_id, line.input_revision, line.input_hash)
                                == (item.meter_id, item.input_revision, item.input_hash))
        _require_book_structure(line.before_state == state and line.start_ts == state.asof_ts
                                and line.after_state.asof_ts == line.end_ts)
        try:
            snapshot = resolve_snapshot(replay_basis.snapshots, line.start_ts, line.end_ts)
        except ValueError as exc:
            raise ValueError("replay_reconciliation_structure_invalid") from exc
        _require_book_structure(line.snapshot_hash == snapshot.snapshot_hash)
        month = line.after_state.month
        if month is not None:
            _require_book_structure(month.contract_id == snapshot.contract_id)
        period = _closing_period(line)
        if period is not None:
            _require_book_structure(period not in closes)
            closes[period] = month.contract_id
        state = line.after_state
    _require_book_structure(state == settlement.end_state and state.asof_ts == settlement.coverage_end_ts)
    periods = tuple((adjustment.period_start_ts, adjustment.period_end_ts) for adjustment in settlement.adjustments)
    _require_book_structure(len(set(periods)) == len(periods) and all(period in closes for period in periods))
    if settlement.complete:
        _require_book_structure(set(periods) == set(closes))


def _validate_day_structure(projection: ReplayProjection, settlement: SettlementResult, replay_basis: ReplayBasis,
                            credits: tuple[Decimal | None, ...], wear: tuple[FinancialComponent, ...],
                            terminal: FinancialComponent, groups: tuple[tuple[FinancialDayKey, tuple[int, ...]], ...],
                            days: tuple[ReplayFinancialDay, ...]) -> None:
    """Prove the producer's source-index allocation independently of any delta."""
    lines = settlement.lines
    _require_book_structure(len(credits) == len(wear) == len(lines) == len(projection.slices)
                            and len(groups) == len(days) and bool(days))
    indices = tuple(index for _, group in groups for index in group)
    _require_book_structure(all(type(index) is int for index in indices) and indices == tuple(range(len(lines))))
    _require_book_structure(len({key for key, _ in groups}) == len(groups))
    bounds = _contract_periods(sorted(replay_basis.snapshots, key=lambda snapshot: snapshot.valid_from_ts))
    adjustments = {(adjustment.period_start_ts, adjustment.period_end_ts): adjustment.credit_total
                   for adjustment in settlement.adjustments}
    for line, credit in zip(lines, credits, strict=True):
        period = _closing_period(line)
        _require_book_structure(credit == (Decimal(0) if period is None else adjustments.get(period)))
    with localcontext(Context(prec=36)):
        for (key, group), day in zip(groups, days, strict=True):
            _require_book_structure(bool(group) and day.key == key)
            selected = tuple(lines[index] for index in group)
            _require_book_structure((day.start_ts, day.end_ts) == (selected[0].start_ts, selected[-1].end_ts))
            for line in selected:
                try:
                    snapshot = resolve_snapshot(replay_basis.snapshots, line.start_ts, line.end_ts)
                except ValueError as exc:
                    raise ValueError("replay_reconciliation_structure_invalid") from exc
                zone = ZoneInfo(snapshot.timezone)
                first_day = datetime.fromtimestamp(line.start_ts, zone).date().isoformat()
                last_day = datetime.fromtimestamp(line.end_ts - 1, zone).date().isoformat()
                _require_book_structure(first_day == last_day and key == FinancialDayKey(
                    first_day, snapshot.timezone, snapshot.contract_id, bounds[snapshot.valid_from_ts][0]))
            for name in ("import_cost", "base_export_revenue", "bonus_revenue"):
                expected = _component(tuple(getattr(line, name) for line in selected),
                    tuple(issue for line in selected if getattr(line, name) is None for issue in line.issues))
                _require_book_structure(getattr(day.components, name) == expected)
            _require_book_structure(day.components.month_credit == _component(
                tuple(credits[index] for index in group), ("replay_month_credit_unknown",)))
            _require_book_structure(day.components.wear_cost == _sum_components(tuple(wear[index] for index in group)))
            expected_terminal = terminal if group[-1] == len(lines) - 1 else _component((Decimal(0),), ())
            _require_book_structure(day.components.terminal_value == expected_terminal)


def _reconciliation_coverage_reasons(result: ReplayFinancialResult) -> tuple[str, ...]:
    """Consumer window evidence only: contract assignment needs the producer."""
    days = result.days
    _require_book_structure(len({day.key for day in days}) == len(days))
    ordered = sorted(days, key=lambda day: day.start_ts)
    _require_book_structure(all(left.end_ts <= right.start_ts for left, right in zip(ordered, ordered[1:], strict=False)))
    projection, settlement = result.projection, result.settlement
    unknown = ("reconciliation_coverage_unknown",)
    if (projection is None or settlement is None or projection.blocking_reasons or not days
            or not projection.items or not settlement.lines or tuple(ordered) != days):
        return unknown
    start, end = projection.start_ts, projection.end_ts
    if ((settlement.coverage_start_ts, settlement.coverage_end_ts) != (start, end)
            or (projection.items[0].start_ts, projection.items[-1].end_ts) != (start, end)
            or (settlement.lines[0].start_ts, settlement.lines[-1].end_ts) != (start, end)
            or settlement.end_state.asof_ts != end
            or (days[0].start_ts, days[-1].end_ts) != (start, end)
            or any(left.end_ts != right.start_ts for left, right in zip(days, days[1:], strict=False))):
        return unknown
    return ()


def _with_reconciliation(result: ReplayFinancialResult) -> ReplayFinancialResult:
    coverage = _reconciliation_coverage_reasons(result)
    entries = []
    for field in fields(ReplayCostComponents):
        component = getattr(result.components, field.name)
        entries.append(_reconcile_days(field.name, component.value,
            tuple(getattr(day.components, field.name).value for day in result.days),
            reasons=(*coverage, *component.reasons, *result.financial_reasons)))
    return replace(result, reconciliation=tuple(entries))


def _financial_days(projection: ReplayProjection, settlement: SettlementResult, replay_basis: ReplayBasis,
                     credits: tuple[Decimal | None, ...], wear: tuple[FinancialComponent, ...],
                     terminal: FinancialComponent) -> tuple[ReplayFinancialDay, ...]:
    """Group the one existing book, never replay or reprice a daily opening."""
    bounds = _contract_periods(sorted(replay_basis.snapshots, key=lambda snapshot: snapshot.valid_from_ts))
    groups = {}
    for index, line in enumerate(settlement.lines):
        snapshot = resolve_snapshot(replay_basis.snapshots, line.start_ts, line.end_ts)
        key = FinancialDayKey(datetime.fromtimestamp(line.start_ts, ZoneInfo(snapshot.timezone)).date().isoformat(),
                              snapshot.timezone, snapshot.contract_id, bounds[snapshot.valid_from_ts][0])
        groups.setdefault(key, []).append((index, snapshot))
    days = []
    for key, entries in groups.items():
        indices = tuple(index for index, _ in entries)
        lines = tuple(settlement.lines[index] for index in indices)
        day_credits = tuple(credits[index] for index in indices)
        # Same booking order as the engine; no tariff, bonus or closing logic.
        subtotal = Decimal(0)
        for line, credit in zip(lines, day_credits, strict=True):
            for amount, sign in ((line.import_cost, 1), (line.base_export_revenue, -1),
                                 (line.bonus_revenue, -1), (credit, -1)):
                if amount is not None:
                    subtotal += sign * amount
        invoice_complete = all(line.quality != "incomplete" for line in lines) and all(c is not None for c in day_credits)
        day_terminal = terminal if indices[-1] == len(settlement.lines) - 1 else _component((Decimal(0),), ())
        components = _cost_components(lines, day_credits, invoice_value=subtotal if invoice_complete else None,
            invoice_subtotal=subtotal, wear=_sum_components(tuple(wear[index] for index in indices)), terminal=day_terminal)
        reasons = _reasons((*(reason for index in indices for reason in projection.slices[index].reasons),
                            *components.total_cost.reasons))
        days.append(ReplayFinancialDay(key, lines[0].start_ts, lines[-1].end_ts, components,
            components.total_cost.status, reasons, _period_status(lines, tuple(snapshot for _, snapshot in entries), day_credits)))
    groups = tuple((key, tuple(index for index, _ in entries)) for key, entries in groups.items())
    days = tuple(days)
    _validate_day_structure(projection, settlement, replay_basis, credits, wear, terminal, groups, days)
    return days


def _book_blocking_reasons(projection: ReplayProjection, replay_basis: ReplayBasis,
                            totals: ReplayPhysicalTotals) -> tuple[str, ...]:
    reasons = list(projection.blocking_reasons)
    source = projection.source
    if source.status == "mismatch":
        reasons.append("source_binding_mismatch")
    if source.binding.replay_basis_hash != replay_basis.content_hash:
        reasons.append("replay_basis_mismatch")
    names = tuple(sorted(battery.battery_name for battery in totals.batteries))
    if names != source.binding.battery_names:
        reasons.append("replay_battery_identity_mismatch")
    if names != tuple(sorted(name for name, _ in replay_basis.terminal_value.reference_efficiency_by_battery)):
        reasons.append("replay_terminal_identity_mismatch")
    if not projection.items:
        reasons.append("replay_projection_empty")
    elif not reasons:
        cursor = projection.start_ts
        if cursor != replay_basis.opening.asof_ts or len(projection.items) != len(projection.slices):
            reasons.append("replay_projection_coverage_mismatch")
        else:
            for item, piece in zip(projection.items, projection.slices, strict=True):
                if ((item.start_ts, item.end_ts) != (piece.start_ts, piece.end_ts) or item.start_ts != cursor
                        or item.meter_id != source.binding.meter_id or item.input_revision != 0
                        or tuple(sorted(name for name, _ in piece.dc_charge_by_battery)) != names):
                    reasons.append("replay_projection_coverage_mismatch")
                    break
                cursor = item.end_ts
            if cursor != projection.end_ts:
                reasons.append("replay_projection_coverage_mismatch")
    return _reasons(reasons)


def settle_replay(projection: ReplayProjection, replay_basis: ReplayBasis,
                  physical_totals: ReplayPhysicalTotals) -> ReplayFinancialResult:
    """One independent engine book from the producer's pinned model projection."""
    _typed(projection, ReplayProjection)
    _typed(replay_basis, ReplayBasis)
    _typed(physical_totals, ReplayPhysicalTotals)
    blocking = _book_blocking_reasons(projection, replay_basis, physical_totals)
    trust = "caller_asserted" if projection.source.evidence is not None else "missing"
    if blocking:
        return replace(unknown_replay_financial(reasons=_reasons((*projection.reasons, *blocking))),
                       projection=projection, source_trust=trust)
    with localcontext(Context(prec=36)):
        settlement = settle_series(projection.items, replay_basis.snapshots, replay_basis.opening)
        _validate_book_structure(projection, settlement, replay_basis)
        credits = _closing_credits(settlement.lines, settlement.adjustments)
        wear = _wear_components(projection.slices, physical_totals)
        terminal = _terminal_component(physical_totals, replay_basis)
        components = _cost_components(settlement.lines, credits, invoice_value=settlement.net_cost,
            invoice_subtotal=settlement.known_subtotal, wear=_sum_components(wear), terminal=terminal)
        days = _financial_days(projection, settlement, replay_basis, credits, wear, terminal)
        snapshots = tuple(resolve_snapshot(replay_basis.snapshots, line.start_ts, line.end_ts) for line in settlement.lines)
        result = ReplayFinancialResult(projection, settlement, components, days, components.total_cost.status,
            _reasons((*projection.reasons, *components.total_cost.reasons)),
            _period_status(settlement.lines, snapshots, credits), trust)
        return _with_reconciliation(result)


def compare_financial_results(baseline: ReplayFinancialResult, fast: ReplayFinancialResult) -> ReplayFinancialComparison:
    """Subtract complete model costs; no subtotal savings and no loss clipping."""
    _typed(baseline, ReplayFinancialResult)
    _typed(fast, ReplayFinancialResult)
    reasons = _reasons((*baseline.financial_reasons, *fast.financial_reasons))
    status = ("unknown" if "unknown" in (baseline.financial_status, fast.financial_status) else
              "complete" if baseline.financial_status == fast.financial_status == "complete" else "partial")
    coverage = _reasons((*_reconciliation_coverage_reasons(baseline), *_reconciliation_coverage_reasons(fast)))
    base_days, fast_days = ({day.key: day for day in result.days} for result in (baseline, fast))
    keys = tuple(dict.fromkeys((*base_days, *fast_days)))
    if coverage or base_days.keys() != fast_days.keys() or any(
            (base_days[key].start_ts, base_days[key].end_ts) != (fast_days[key].start_ts, fast_days[key].end_ts) for key in keys):
        status = "unknown"
        coverage = ("reconciliation_coverage_unknown",)
        reasons = _reasons((*reasons, "replay_comparison_coverage_mismatch"))
    with localcontext(Context(prec=36)):
        advantage = baseline.components.total_cost.value - fast.components.total_cost.value if status == "complete" else None
        days = tuple(ReplayDayAdvantage(key,
            base_days[key].components.total_cost.value - fast_days[key].components.total_cost.value
            if status == "complete" else None) for key in keys)
    reconciliation = _reconcile_days("advantage_eur", advantage, tuple(day.advantage_eur for day in days),
                                      reasons=(*coverage, *reasons))
    return ReplayFinancialComparison(advantage, status, reasons, days, (reconciliation,))
