"""Synthetic caller assertions bind replay inputs; they never verify an installation."""

import importlib
import importlib.util
import inspect
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from datetime import date, datetime, time, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Context, Decimal, Inexact, Rounded, localcontext
from fractions import Fraction
from hashlib import sha256
from math import nextafter
from zoneinfo import ZoneInfo

import pytest

from dao.prog.fastctrl.command import OwnerBinding, plan_digest
from dao.prog.fastctrl.policy import BatteryDecision
from dao.prog.fastctrl.replay_execution import (
    BindingEvent,
    ProposalEvent,
    RecordedExecution,
    validate_replay_input,
)
from dao.prog.fastctrl.simulate import BatteryReplaySegment, ReplaySegment
from dao.prog.settlement.engine import settle_series
from dao.prog.settlement.metering import resolve_spot
from dao.prog.settlement.types import BonusState, MonthState, SettlementState, SpotInterval
from dao.tests.prog.test_fast_simulate import modeled_case
from dao.tests.settlement import test_fast_financial_replay as fixtures
from dao.tests.settlement.conftest import MONTH_END_TS, YEAR_END_TS
from dao.tests.settlement.test_publication import NOW, START_TS

D = Decimal


def finance_api():
    name = "dao.prog.fastctrl.replay_finance"
    assert importlib.util.find_spec(name) is not None, "F3b1 replay_finance API missing"
    api = importlib.import_module(name)
    for symbol in ("SourceReference", "ReplaySourceSpan", "ReplaySourceEvidence", "ReplaySourceBinding",
                   "SourceValidation", "replay_environment_digest", "bind_source_evidence",
                   "validate_source_evidence", "replay_scenario_digest", "_finance_canonical_bytes"):
        assert hasattr(api, symbol), f"F3b1 {symbol} missing"
    return api


def source_case(api):
    plan, replay = fixtures.finance_case()
    points = ((NOW, -1200.0), (NOW + 900, -500.0))
    inputs = tuple(validate_replay_input(plan=plan, sample_points=points, soc_start=50,
                    inverter_efficiency=.95, scenario=modeled_case(name, hold=900))
                   for name in ("synthetic-baseline", "synthetic-fast"))
    reference = api.SourceReference("synthetic:installation-declaration", "v1", "a" * 64)
    evidence = api.ReplaySourceEvidence(
        "replay_source_v1", replay.basis.meter_id, replay.basis.meter_id, replay.content_hash,
        plan_digest(plan), api.replay_environment_digest(points, house_hold_s=900),
        ("accu",), ("synthetic:roof-pv",), reference, "synthetic:caller", NOW,
        (reference,), NOW, NOW + 900,
        (api.ReplaySourceSpan(NOW, NOW + 900, "pv_only_nonbattery_generation", (reference,)),),
        "both_alternatives",
    )
    return plan, replay, inputs, evidence


def binding_for(api, plan, replay, inputs, evidence):
    return api.bind_source_evidence(evidence, source_plan=plan, replay_basis=replay, replay_inputs=inputs)


def test_f3b1_source_is_detached_canonical_and_strict():
    api = finance_api()
    plan, replay, inputs, source = source_case(api)
    extra = api.SourceReference("synthetic:inventory", "v2", "b" * 64)
    refs = [extra, source.evidence_references[0]]
    batteries, pv = ["B", "accu"], ["synthetic:roof-pv", "synthetic:garage-pv"]
    spans = [replace(source.spans[0], evidence_references=refs)]
    detached = replace(source, battery_names=batteries, pv_source_ids=pv,
                       evidence_references=refs, spans=spans)
    raw, digest = detached.canonical_bytes(), detached.content_hash
    assert digest == sha256(raw).hexdigest()
    assert b"replay_source_v1" in raw and b"ReplaySourceEvidence" in raw
    reordered = replace(detached, battery_names=tuple(reversed(batteries)),
                        pv_source_ids=tuple(reversed(pv)), evidence_references=tuple(reversed(refs)),
                        spans=(replace(spans[0], evidence_references=tuple(reversed(refs))),))
    assert reordered.canonical_bytes() == raw and reordered.content_hash == digest
    refs.clear()
    batteries.clear()
    pv.clear()
    spans.clear()
    assert detached.canonical_bytes() == raw and detached.content_hash == digest
    assert len(detached.spans[0].evidence_references) == 2

    binding = binding_for(api, plan, replay, inputs, source)
    reasons = ["source_evidence_missing"]
    names = ["accu"]
    binding_copy = replace(binding, reasons=reasons, battery_names=names)
    validation = api.SourceValidation("missing", None, binding_copy, reasons)
    reasons.clear()
    names.clear()
    assert validation.reasons == binding_copy.reasons == ("source_evidence_missing",)
    assert binding_copy.battery_names == ("accu",)
    for obj in (source.evidence_references[0], source.spans[0], detached, binding_copy, validation):
        assert not hasattr(obj, "__dict__")
        assert all(p.default is inspect.Parameter.empty for p in inspect.signature(type(obj)).parameters.values())
        for field in fields(obj):
            with pytest.raises(FrozenInstanceError):
                setattr(obj, field.name, getattr(obj, field.name))
            kwargs = {item.name: getattr(obj, item.name) for item in fields(obj) if item != field}
            with pytest.raises(TypeError):
                type(obj)(**kwargs)


@pytest.mark.parametrize("fault", [
    "version", "installation", "meter", "archive-hash", "plan-hash", "environment-hash",
    "battery-duplicate", "pv-duplicate", "empty-id", "no-pv", "asserted-by", "asserted-bool",
    "coverage-bool", "coverage-float", "coverage-empty", "span-bool", "span-empty", "outside",
    "overlap", "unordered", "claim", "scope", "references-empty", "span-references-empty",
    "reference-empty", "reference-version", "reference-hash", "reference-duplicate",
    "reference-conflict", "topology-type", "spans-type", "identities-type",
])
def test_f3b1_source_rejects_malformed_before_binding(fault):
    api = finance_api()
    _, _, _, source = source_case(api)
    span, reference = source.spans[0], source.evidence_references[0]
    with pytest.raises((TypeError, ValueError)):
        if fault.startswith("reference-") and fault not in ("reference-duplicate", "reference-conflict"):
            field, value = {"reference-empty": ("reference", " "), "reference-version": ("version", ""),
                            "reference-hash": ("content_digest", "A" * 64)}[fault]
            replace(reference, **{field: value})
        elif fault.startswith("span-"):
            changes = {"span-bool": {"start_ts": True}, "span-empty": {"end_ts": NOW},
                       "span-references-empty": {"evidence_references": ()}}[fault]
            replace(span, **changes)
        elif fault == "claim":
            replace(span, claim="verified_pv")
        else:
            changes = {
                "version": {"version": "replay_source_v2"}, "installation": {"installation_id": ""},
                "meter": {"meter_id": 1}, "archive-hash": {"replay_basis_hash": "f" * 63},
                "plan-hash": {"source_plan_digest": "A" * 64},
                "environment-hash": {"environment_digest": None},
                "battery-duplicate": {"battery_names": ("accu", "accu")},
                "pv-duplicate": {"pv_source_ids": ("pv", "pv")}, "empty-id": {"battery_names": ("",)},
                "no-pv": {"pv_source_ids": ()}, "asserted-by": {"asserted_by": " "},
                "asserted-bool": {"asserted_at_ts": True}, "coverage-bool": {"coverage_start_ts": False},
                "coverage-float": {"coverage_end_ts": float(NOW + 900)},
                "coverage-empty": {"coverage_end_ts": NOW},
                "outside": {"coverage_start_ts": NOW + 1}, "overlap": {"spans": (span, span)},
                "unordered": {"spans": (replace(span, start_ts=NOW + 600), replace(span, end_ts=NOW + 300))},
                "scope": {"counterfactual_scope": "verified"}, "references-empty": {"evidence_references": ()},
                "reference-duplicate": {"evidence_references": (reference, reference)},
                "reference-conflict": {"evidence_references": (reference, replace(reference, content_digest="c" * 64))},
                "topology-type": {"topology_reference": "synthetic"}, "spans-type": {"spans": {}},
                "identities-type": {"battery_names": "accu"},
            }[fault]
            replace(source, **changes)


def test_f3b1_canonical_types_and_exact_finite_environment():
    api = finance_api()
    canonical = api._finance_canonical_bytes
    assert len({canonical(value) for value in (True, 1, 1.0, D("1"), "1", None, (1,))}) == 7
    assert canonical(D("1.2300")) == canonical(D("1.23"))
    assert canonical(D("-0.00")) == canonical(D("0"))
    assert b"0.0000001" in canonical(D("1E-7"))
    exact = D("0.123456789012345678901234567890123456")
    with localcontext() as context:
        context.prec = 6
        assert str(exact).encode() in canonical(exact)
    assert (0.1).hex().encode() in canonical(0.1)
    assert canonical(0.0) != canonical(-0.0)
    for invalid in (float("nan"), float("inf"), float("-inf"), D("NaN"), D("Infinity"),
                    {}, [], {1}, object()):
        with pytest.raises((TypeError, ValueError)):
            canonical(invalid)
    points = ((float(NOW), .1), (float(NOW + 900), -0.0))
    digest = api.replay_environment_digest(points, house_hold_s=900)
    assert len(digest) == 64 and digest == api.replay_environment_digest(deepcopy(points), house_hold_s=900)
    for other, hold in ((points, 899), (points, 900.0000000001),
                        (((float(NOW), nextafter(.1, 1)), points[1]), 900),
                        ((points[0], (float(NOW + 900), 0.0)), 900),
                        ((points[0], (nextafter(float(NOW + 900), float("inf")), -0.0)), 900)):
        assert api.replay_environment_digest(other, house_hold_s=hold) != digest
    assert digest != sha256(canonical(("replay_scenario_v1", points, 900))).hexdigest()


@pytest.mark.parametrize("points,hold", [
    (((NOW, 0), (NOW + 900, float("nan"))), 900),
    (((NOW, float("inf")), (NOW + 900, 0)), 900),
    (((float("-inf"), 0), (NOW + 900, 0)), 900),
    (((True, 0), (NOW + 900, 0)), 900), (((NOW, False), (NOW + 900, 0)), 900),
    (((NOW + 900, 0), (NOW, 0)), 900), (((NOW, 0), (NOW, 0)), 900),
    (((NOW, 0),), 900), (((NOW, 0), (NOW + 900, 0)), True),
    (((NOW, 0), (NOW + 900, 0)), 0), (((NOW, 0), (NOW + 900, 0)), -1),
    (((NOW, 0), (NOW + 900, 0)), float("nan")),
    (((NOW, 0), (NOW + 900, 0)), float("inf")),
])
def test_f3b1_environment_rejects_invalid_numbers_and_order(points, hold):
    api = finance_api()
    with pytest.raises((TypeError, ValueError)):
        api.replay_environment_digest(points, house_hold_s=hold)


def test_f3b1_binding_scope_and_missing_are_distinct():
    api = finance_api()
    plan, replay, inputs, source = source_case(api)
    before = deepcopy((plan, replay, inputs, source))
    binding = binding_for(api, plan, replay, inputs, source)
    valid = api.validate_source_evidence(source, binding)
    assert valid.status == "valid" and valid.evidence == source and valid.reasons == ()
    assert binding.installation_id == binding.meter_id == replay.basis.meter_id
    assert binding.replay_basis_hash == replay.content_hash
    assert binding.source_plan_digest == plan_digest(plan)
    assert binding.source_digest == source.content_hash
    assert binding_for(api, plan, replay, inputs[:1], source) == binding
    missing = api.validate_source_evidence(None, binding_for(api, plan, replay, inputs, None))
    assert missing.status == "missing" and missing.evidence is None
    assert "source_evidence_missing" in missing.reasons
    wrong = api.validate_source_evidence(replace(source, meter_id="synthetic:other"), binding)
    assert wrong.status == "mismatch" and "source_binding_mismatch" in wrong.reasons
    for scope in ("observed_run_only", "unknown"):
        unproven = replace(source, counterfactual_scope=scope)
        result = api.validate_source_evidence(unproven, binding_for(api, plan, replay, inputs, unproven))
        assert result.status == "missing" and "source_counterfactual_scope_unproven" in result.reasons
        assert result.evidence is None  # No positive spans available to the next projection stage.
    no_scenario = validate_replay_input(plan=plan, sample_points=inputs[0].sample_points,
                                       soc_start=50, inverter_efficiency=.95, scenario=None)
    for variants in ((no_scenario,), (inputs[0], no_scenario), (no_scenario, inputs[1])):
        unknown = binding_for(api, plan, replay, variants, source)
        assert unknown.environment_digest is None
        result = api.validate_source_evidence(source, unknown)
        assert result.status == "missing" and "source_counterfactual_scope_unproven" in result.reasons
    # Gaps/unknown are well-formed input: interval eligibility belongs to projection.
    for spans in ((), (replace(source.spans[0], claim="unknown"),),
                  (replace(source.spans[0], end_ts=NOW + 300), replace(source.spans[0], start_ts=NOW + 600))):
        partial = replace(source, spans=spans)
        result = api.validate_source_evidence(partial, binding_for(api, plan, replay, inputs, partial))
        assert result.status == "valid" and result.evidence.spans == spans
    topology = replace(source.topology_reference, content_digest="c" * 64)
    changed = replace(source, topology_reference=topology)
    assert api.validate_source_evidence(changed, binding).status == "mismatch"
    assert api.validate_source_evidence(changed, binding_for(api, plan, replay, inputs, changed)).status == "valid"
    assert (plan, replay, inputs, source) == before


@pytest.mark.parametrize("field,value", [
    ("meter_id", "synthetic:other"), ("installation_id", "synthetic:other"),
    ("replay_basis_hash", "d" * 64), ("source_plan_digest", "d" * 64),
    ("environment_digest", "d" * 64), ("battery_names", ("accu", "unmodeled-storage")),
    ("battery_names", ()),
])
def test_f3b1_wrong_package_binding_is_blocking_even_when_scope_is_missing(field, value):
    api = finance_api()
    plan, replay, inputs, source = source_case(api)
    for scope in ("both_alternatives", "observed_run_only"):
        wrong = replace(source, **{field: value, "counterfactual_scope": scope})
        result = api.validate_source_evidence(wrong, binding_for(api, plan, replay, inputs, wrong))
        assert result.status == "mismatch" and "source_binding_mismatch" in result.reasons
        assert result.evidence is None


@pytest.mark.parametrize("fault", ["hold", "samples", "battery", "recorded-installation", "opening-binding", "event-binding"])
def test_f3b1_alternatives_must_share_environment_inventory_and_installation(fault):
    api = finance_api()
    plan, replay, inputs, source = source_case(api)
    other = inputs[1]
    if fault == "hold":
        other = replace(other, scenario=replace(other.scenario, physics=replace(other.scenario.physics, house_hold_s=899)))
    elif fault == "samples":
        other = replace(other, sample_points=((NOW, -1200.1), other.sample_points[1]))
    elif fault == "battery":
        other = replace(other, soc_by_battery=(("B", 50),), inverter_efficiency_by_battery=(("B", .95),))
    else:
        owner = OwnerBinding("synthetic:wrong", "synthetic:receiver", "synthetic:generation",
                             "synthetic:session", 1, "synthetic:barrier", (NOW + 900) * 1000)
        execution = RecordedExecution("synthetic:wrong" if fault == "recorded-installation" else replay.basis.meter_id,
            "synthetic:generation", owner if fault == "opening-binding" else None, (), None, (),
            (BindingEvent(NOW, 1, owner),) if fault == "event-binding" else ())
        other = replace(other, scenario=replace(other.scenario, execution=execution))
    binding = binding_for(api, plan, replay, (inputs[0], other), source)
    result = api.validate_source_evidence(source, binding)
    assert result.status == "mismatch" and "source_binding_mismatch" in result.reasons
    assert result.evidence is None
    assert api.validate_source_evidence(None, binding).status == "mismatch"


@pytest.mark.parametrize("change", ["house", "plan", "archive"])
def test_f3b1_changed_shared_inputs_require_explicit_rebinding(change):
    api = finance_api()
    plan, replay, inputs, source = source_case(api)
    if change == "house":
        inputs = tuple(replace(item, sample_points=((NOW, -2000.0), item.sample_points[1])) for item in inputs)
    elif change == "plan":
        plan = replace(plan, strategy="synthetic:changed-original-plan")
    else:
        replay = replace(replay, price_batch=replace(replay.price_batch, items=tuple(
            replace(item, content_hash="d" * 64) for item in replay.price_batch.items)))
        plan = fixtures.bind_replay(plan, replay)
    binding = binding_for(api, plan, replay, inputs, source)
    assert api.validate_source_evidence(source, binding).status == "mismatch"
    rebound = replace(source, replay_basis_hash=replay.content_hash, source_plan_digest=plan_digest(plan),
                      environment_digest=api.replay_environment_digest(inputs[0].sample_points, house_hold_s=900))
    assert rebound.content_hash != source.content_hash
    assert api.validate_source_evidence(rebound, binding_for(api, plan, replay, inputs, rebound)).status == "valid"


def test_f3b1_scenario_digest_pins_execution_trace_soc_and_full_battery_model():
    api = finance_api()
    plan, replay, inputs, source = source_case(api)
    binding = binding_for(api, plan, replay, inputs, source)
    event = ProposalEvent(NOW, 1, BatteryDecision(0, "accu", 0.0, 0.0, False, False, "synthetic:idle"), None, None)
    trace = (event, replace(event, at_ts=NOW + 1, order=2))

    def digest(execution_plan=plan, replay_input=inputs[0], events=trace, source_plan=plan):
        return api.replay_scenario_digest(source_plan=source_plan, execution_plan=execution_plan,
                                          replay_input=replay_input, trace=events)

    original = digest()
    assert original == digest(deepcopy(plan), deepcopy(inputs[0]), deepcopy(trace), deepcopy(plan))
    assert original != source.content_hash != binding.environment_digest
    assert original != digest(events=trace[:1])
    assert original != digest(events=(replace(event, decision=replace(event.decision, setpoint_w=.1)), trace[1]))
    assert original != digest(events=(event, replace(trace[1], at_ts=nextafter(float(NOW + 1), float("inf")))))
    changed = replace(inputs[0], scenario=replace(inputs[0].scenario,
                      execution=replace(inputs[0].scenario.execution, prepare_delay_s=.25)))
    assert original != digest(replay_input=changed)
    assert binding_for(api, plan, replay, (changed, inputs[1]), source) == binding
    assert original != digest(replay_input=replace(inputs[0], soc_by_battery=(("accu", 51),)))
    assert original != digest(replay_input=replace(inputs[0], inverter_efficiency_by_battery=(("accu", .9),)))
    assert original != digest(execution_plan=replace(plan, strategy="synthetic:execution-plan"))
    assert original != digest(source_plan=replace(plan, strategy="synthetic:source-plan"))
    for name, value in (("capacity_kwh", 11.0), ("max_charge_w", 4900.0), ("max_discharge_w", 4800.0),
                        ("minimum_power_w", 30.0), ("soc_min", 21.0), ("soc_max", 99.0),
                        ("charge_efficiency", .8), ("discharge_efficiency", .7), ("cycle_cost", .01)):
        assert original != digest(execution_plan=replace(plan, specs=[replace(plan.specs[0], **{name: value})]))
    with pytest.raises((TypeError, ValueError)):
        digest(events=tuple(reversed(trace)))
    with pytest.raises((TypeError, ValueError)):
        digest(events=(event, event))
    broken = deepcopy(trace)
    broken[0].decision.setpoint_w = float("nan")
    with pytest.raises((TypeError, ValueError)):
        digest(events=broken)


def test_f3b1_finance_case_has_real_hashes_exact_edges_and_full_native_hours():
    assert hasattr(fixtures, "finance_case"), "F3b1 owning finance_case factory missing"
    plan, replay = fixtures.finance_case()
    assert [(row.start_ts, row.end_ts) for row in plan.intervals] == [(NOW, NOW + 900)]
    assert [(item.spot.start_ts, item.spot.end_ts, item.spot.price_ex_vat) for item in replay.price_batch.items] == [
        (START_TS, START_TS + 3600, D("0.1"))]
    assert replay.opening == SettlementState(NOW, None, None)
    assert plan.financial_context.not_after_ts == NOW + 900
    replay.check_context(plan.financial_context)
    assert type(plan).from_dict(plan.to_dict()) == plan
    assert plan.financial_context.replay_basis_hash == sha256(replay.canonical_bytes()).hexdigest()
    assert all(item.input_revision == 1 and len(item.content_hash) == 64 for item in replay.price_batch.items)
    spot = replay.price_batch.items[0].spot
    assert resolve_spot(replay.snapshots[0], NOW, NOW + 900, (spot,)).price_ex_vat == D("0.1")
    changed = fixtures.canonical_snapshot(replay.snapshots[0], monthly_export_floor="spot_zero")
    assert changed.snapshot_hash != replay.snapshots[0].snapshot_hash
    edges = (NOW + 3600, NOW + 4200, NOW + 4500)
    opening = SettlementState(edges[0], None, None)
    spots = (SpotInterval(START_TS + 3600, START_TS + 7200, D("-0.2"), "synthetic:custom-spot", "estimated"),)
    custom_plan, custom = fixtures.finance_case(edges=edges, snapshots=(changed,), spots=spots, opening=opening)
    custom.check_context(custom_plan.financial_context)
    assert custom.opening == opening
    assert custom.basis.snapshot_hashes == (changed.snapshot_hash,)
    assert custom.basis.captured_ts == custom_plan.created_ts == edges[0]
    assert custom.basis.planning_start_ts == edges[0] and custom.basis.horizon_end_ts == edges[-1]
    assert [(row.start_ts, row.end_ts) for row in custom_plan.intervals] == [(NOW + 3600, NOW + 4200), (NOW + 4200, NOW + 4500)]
    assert [(row.start_ts, row.end_ts) for row in custom.quotes] == [(NOW + 3600, NOW + 4200), (NOW + 4200, NOW + 4500)]
    assert all(row.spot_ex_vat == D("-0.2") for row in custom.quotes)
    assert json.loads(custom.canonical_bytes())["basis"]["snapshot_hashes"] == [changed.snapshot_hash]


def projection_api():
    api = finance_api()
    for symbol in ("ProjectedQuantity", "ProjectionSlice", "ReplayProjection", "financial_edges", "project_replay_inputs"):
        assert hasattr(api, symbol), f"F3b2 {symbol} missing"
    return api


def physical_battery(name="accu", *, charge=0.0, discharge=0.0, dc=0.0, modeled=0.0):
    return BatteryReplaySegment(
        battery_name=name, command=None, execution_status="stopped" if modeled == 0 else "accepted",
        reasons=(), command_w=None, modeled_ac_w=modeled, charge_ac_kwh=charge,
        discharge_ac_kwh=discharge, dc_throughput_kwh=dc, cell_delta_kwh=0.0,
        soc_begin=50.0, soc_end=50.0,
    )


def physical_segment(start=NOW, end=NOW + 900, *, imported=0.0, exported=0.0, batteries=None):
    return ReplaySegment(start_ts=start, end_ts=end, house_w=0.0,
                         batteries=(physical_battery(),) if batteries is None else batteries,
                         import_kwh=imported, export_kwh=exported)


def projection_source(api, plan, replay, *, kind="valid", spans=None, coverage=None):
    start, end = replay.basis.planning_start_ts, replay.basis.horizon_end_ts
    points = ((start, 0.0), (end, 0.0))
    inputs = (validate_replay_input(plan=plan, sample_points=points, soc_start=50,
              inverter_efficiency=.95, scenario=modeled_case("synthetic-projection", hold=end - start)),)
    reference = api.SourceReference("synthetic:projection-topology", "v1", "b" * 64)
    coverage = (start, end) if coverage is None else coverage
    spans = ((start, end, "pv_only_nonbattery_generation"),) if spans is None else spans
    evidence = api.ReplaySourceEvidence(
        version="replay_source_v1", installation_id=replay.basis.meter_id, meter_id=replay.basis.meter_id,
        replay_basis_hash=replay.content_hash, source_plan_digest=plan_digest(plan),
        environment_digest=api.replay_environment_digest(points, house_hold_s=end - start),
        battery_names=tuple(spec.name for spec in plan.specs), pv_source_ids=("synthetic:pv",),
        topology_reference=reference, asserted_by="synthetic:caller", asserted_at_ts=start,
        evidence_references=(reference,), coverage_start_ts=coverage[0], coverage_end_ts=coverage[1],
        spans=tuple(api.ReplaySourceSpan(a, b, claim, (reference,)) for a, b, claim in spans),
        counterfactual_scope="both_alternatives",
    )
    if kind == "missing":
        evidence = None
    elif kind == "mismatch":
        evidence = replace(evidence, meter_id="synthetic:wrong-installation")
    elif kind == "unknown":
        evidence = replace(evidence, spans=(replace(evidence.spans[0], claim="unknown"),))
    elif kind == "gap":
        evidence = replace(evidence, spans=())
    binding = binding_for(api, plan, replay, inputs, evidence)
    source = api.validate_source_evidence(evidence, binding)
    scenario = api.replay_scenario_digest(source_plan=plan, execution_plan=plan, replay_input=inputs[0], trace=())
    return source, scenario


def project_case(api, plan, replay, segments, *, start=None, end=None, **source_options):
    source, scenario = projection_source(api, plan, replay, **source_options)
    return api.project_replay_inputs(segments=segments,
        start_ts=replay.basis.planning_start_ts if start is None else start,
        end_ts=replay.basis.horizon_end_ts if end is None else end,
        replay_basis=replay, source=source, scenario_digest=scenario)


@pytest.mark.parametrize("case", ["counterflow", "ties", "positive", "thirds", "binary", "dc-directions", "fractional-overlap"])
def test_f3b2_binary_overlap_rounding_and_positive_discharge(case):
    # Catches netting, early rounding, epsilon discharge, float-to-text conversion,
    # lost final overlap remainder, and DC direction inferred from site/cell net.
    api = projection_api()
    plan, replay = fixtures.finance_case(battery_names=("accu", "B") if case in ("positive", "dc-directions") else ("accu",))
    if case == "counterflow":
        segments = (physical_segment(end=NOW + 450.25, imported=.125),
                    physical_segment(start=NOW + 450.25, exported=.25))
    elif case == "ties":
        segments = (physical_segment(end=NOW + 450, imported=2**-10),
                    physical_segment(start=NOW + 450, exported=3 * 2**-10))
    elif case == "positive":
        segments = (physical_segment(exported=.25, batteries=(
            physical_battery(charge=1024.0, dc=512.0, modeled=4096000.0),
            physical_battery("B", discharge=2**-40, dc=2**-39, modeled=-2**-38))),)
    elif case == "dc-directions":
        segments = (physical_segment(exported=.25, batteries=(
            physical_battery(charge=.5, dc=.25, modeled=2000.0),
            physical_battery("B", discharge=.125, dc=.5, modeled=-500.0))),)
    elif case == "binary":
        segments = (physical_segment(imported=.1),)
    elif case == "fractional-overlap":
        plan, replay = fixtures.finance_case(edges=(NOW, NOW + 900, NOW + 1800))
        segments = (physical_segment(end=NOW + 450.25),
                    physical_segment(start=NOW + 450.25, end=NOW + 1350.25, exported=.5),
                    physical_segment(start=NOW + 1350.25, end=NOW + 1800))
    else:
        snapshot = fixtures.canonical_snapshot(replay.snapshots[0], price_interval_seconds=900)
        spots = tuple(SpotInterval(NOW + i * 900, NOW + (i + 1) * 900, D(rate), "synthetic:quarter", "estimated")
                      for i, rate in enumerate((".1", ".2", ".3")))
        plan, replay = fixtures.finance_case(edges=(NOW, NOW + 900, NOW + 1800, NOW + 2700),
                                             snapshots=(snapshot,), spots=spots)
        segments = (physical_segment(end=NOW + 2700, imported=1.0),)
    before = deepcopy(segments)
    original_bytes = tuple((float(s.start_ts).hex(), float(s.end_ts).hex(),
                            s.import_kwh.hex(), s.export_kwh.hex()) for s in segments)
    result = project_case(api, plan, replay, segments)
    assert result.blocking_reasons == ()
    assert result.segments is segments and result.segments == before
    assert len(result.items) == (3 if case == "thirds" else 2 if case == "fractional-overlap" else 1)
    item, piece = result.items[0], result.slices[0]
    assert item.input_revision == 0 and item.quality == "estimated" and item.origin == "verified_pv"
    assert item.meter_id == replay.basis.meter_id and len(item.input_hash) == 64
    assert {"model:replay_projection_v1", f"scenario:{result.scenario_digest}",
            f"projection:{result.projection_digest}", f"source:{result.source.evidence.content_hash}",
            "source_version:replay_source_v1", "source_trust:caller_asserted", "source_span:0"}.issubset(item.sources)
    if case == "counterflow":
        assert (item.import_kwh, item.export_kwh) == (D("0.125"), D("0.25"))
        assert piece.segment_indices == (0, 1)
    elif case == "ties":
        assert (item.import_kwh, item.export_kwh) == (D("0.000976562"), D("0.002929688"))
        assert piece.import_energy.residue == D("0.0000000005")
        assert piece.export_energy.residue == D("-0.0000000005")
    elif case == "positive":
        assert item.battery_discharge_kwh == D("0.000000001")
        assert piece.battery_discharge.raw == D.from_float(2**-40)
        assert D("-0.000000001") < piece.battery_discharge.residue < 0
        zero = (physical_segment(batteries=(physical_battery(), physical_battery("B"))),)
        assert project_case(api, plan, replay, zero).items[0].battery_discharge_kwh == D("0")
    elif case == "dc-directions":
        assert piece.dc_charge_by_battery == (("B", D("0")), ("accu", D(".25")))
        assert piece.dc_discharge_by_battery == (("B", D(".5")), ("accu", D("0")))
        assert item.battery_discharge_kwh == D(".125")
    elif case == "binary":
        assert piece.import_energy.raw == D("0.100000000000000005551115123125782702")
        assert piece.import_energy.residue == D("0.000000000000000005551115123125782702")
    elif case == "fractional-overlap":
        assert tuple(s.segment_indices for s in result.slices) == ((0, 1), (1, 2))
        assert tuple(s.export_energy.raw for s in result.slices) == (
            D("0.249861111111111111111111111111111111"),
            D("0.250138888888888888888888888888888889"))
        assert tuple(i.export_kwh for i in result.items) == (D(".249861111"), D(".250138889"))
    else:
        assert tuple(s.import_energy.raw for s in result.slices) == (
            D("0.333333333333333333333333333333333333"),
            D("0.333333333333333333333333333333333333"),
            D("0.333333333333333333333333333333333334"))
        assert tuple(i.import_kwh for i in result.items) == (D(".333333333"),) * 3
        assert tuple(i.spot_ex_vat for i in result.items) == (D(".1"), D(".2"), D(".3"))
        assert result.slices[-1].import_energy.residue == D("0.000000000333333333333333333333333334")
    with localcontext() as context:
        context.prec = 6
        repeated = project_case(api, plan, replay, segments)
    assert repeated == result
    assert original_bytes == tuple((float(s.start_ts).hex(), float(s.end_ts).hex(),
                                    s.import_kwh.hex(), s.export_kwh.hex()) for s in segments)
    for obj in (piece.import_energy, piece, result):
        assert not hasattr(obj, "__dict__")
        assert all(p.default is inspect.Parameter.empty for p in inspect.signature(type(obj)).parameters.values())
        with pytest.raises(FrozenInstanceError):
            setattr(obj, fields(obj)[0].name, None)


@pytest.mark.parametrize("case", ["fractional-start", "fractional-end", "partial", "gaps", "empty",
                                  "battery-unknown", "dc-unknown", "source-gap", "source-unknown",
                                  "source-envelope", "source-adjacent", "opening"])
def test_f3b2_gaps_fractional_window_and_source_conjunction(case, bonus_rules):
    api = projection_api()
    plan, replay = fixtures.finance_case()
    options = {}
    segments = (physical_segment(exported=.25),)
    if case.startswith("fractional"):
        options["start" if case == "fractional-start" else "end"] = NOW + .25 if case == "fractional-start" else NOW + 899.75
    elif case == "partial":
        snapshot = fixtures.canonical_snapshot(replay.snapshots[0],
            bonus=replace(bonus_rules, windows=tuple((month, (0, 24)) for month in range(1, 13))))
        plan, replay = fixtures.finance_case(edges=(NOW, NOW + 450), snapshots=(snapshot,))
        segments = (physical_segment(end=NOW + 450, exported=.25),)
    elif case == "gaps":
        plan, replay = fixtures.finance_case(edges=tuple(NOW + i * 900 for i in range(6)))
        segments = (physical_segment(start=NOW + 900, end=NOW + 1800, imported=.125),
                    physical_segment(start=NOW + 2700, end=NOW + 3600, exported=.25))
    elif case == "empty":
        segments = ()
    elif case in ("battery-unknown", "dc-unknown"):
        plan, replay = fixtures.finance_case(battery_names=("accu", "B"))
        segments = (physical_segment(exported=.25, batteries=(
            physical_battery(discharge=.125, dc=.25, modeled=-500.0),
            physical_battery("B", discharge=None if case == "battery-unknown" else 0.0,
                             dc=None, modeled=None))),)
    elif case.startswith("source-"):
        positive = "pv_only_nonbattery_generation"
        spans = ((NOW, NOW + 300, positive), (NOW + 600, NOW + 900, positive))
        if case == "source-unknown":
            spans = (spans[0], (NOW + 300, NOW + 600, "unknown"), spans[1])
        elif case == "source-envelope":
            spans = ((NOW, NOW + 600, positive),)
            options["coverage"] = (NOW, NOW + 600)
        elif case == "source-adjacent":
            spans = ((NOW, NOW + 300, positive), (NOW + 300, NOW + 900, positive))
        options["spans"] = spans
    elif case == "opening":
        options["start"] = NOW + 1
    result = project_case(api, plan, replay, segments, **options)
    if case.startswith("fractional"):
        assert result.items == () and result.blocking_reasons == ("projection_noninteger_window",)
        assert result.segments is segments
        return
    if case == "opening":
        assert result.items == () and result.blocking_reasons == ("replay_opening_mismatch",)
        return
    assert result.blocking_reasons == ()
    assert result.items[0].start_ts == NOW and result.items[-1].end_ts == replay.basis.horizon_end_ts
    if case in ("gaps", "empty"):
        assert len(result.items) == (5 if case == "gaps" else 1)
        missing = (0, 2, 4) if case == "gaps" else (0,)
        for index in missing:
            item, piece = result.items[index], result.slices[index]
            assert item.import_kwh is item.export_kwh is item.battery_discharge_kwh is None
            assert piece.import_energy.raw is piece.import_energy.residue is None
            assert item.quality == "incomplete"
        if case == "gaps":
            assert result.items[1].import_kwh == D(".125") and result.items[3].export_kwh == D(".25")
    elif case in ("battery-unknown", "dc-unknown"):
        assert result.items[0].battery_discharge_kwh == (None if case == "battery-unknown" else D(".125"))
        assert result.slices[0].dc_charge_by_battery == (("B", None), ("accu", D("0")))
        assert result.slices[0].dc_discharge_by_battery == (("B", None), ("accu", D(".25")))
        assert result.segments[0].batteries[0].discharge_ac_kwh == .125
    elif case == "partial":
        assert len(result.items) == 1 and result.items[0].end_ts - result.items[0].start_ts == 450
        line = settle_series(result.items, replay.snapshots, replay.opening).lines[0]
        assert "bonus_requires_quarter" in line.issues and line.bonus_revenue is None
    else:
        assert len(result.items) == 1  # Source boundaries never manufacture bonus-eligible subclaims.
        assert result.items[0].quality == "estimated"
        assert result.items[0].origin == ("verified_pv" if case == "source-adjacent" else "unknown")
        if case != "source-adjacent":
            reason = "source_evidence_missing" if case == "source-unknown" else "source_coverage_gap"
            assert reason in result.slices[0].reasons and reason in result.reasons


@pytest.mark.parametrize("case", ["spring", "autumn", "contract", "calendar", "leap-year"])
def test_f3b2_contract_calendar_and_dst_edges(case, bonus_rules):
    api = projection_api()
    assert tuple(inspect.signature(api.financial_edges).parameters) == ("start_ts", "end_ts", "replay_basis")
    _, base = fixtures.finance_case()
    zone = ZoneInfo("Europe/Amsterdam")
    if case in ("spring", "autumn"):
        day = date(2027, 3, 28) if case == "spring" else date(2027, 10, 31)
        start = int(datetime.combine(day, time.min, zone).timestamp())
        end = int(datetime.combine(day + timedelta(days=1), time.min, zone).timestamp())
        assert end - start == (23 if case == "spring" else 25) * 3600
        edges = tuple(range(start, end + 1, 900))
        plan, replay = fixtures.finance_case(edges=edges)
        assert api.financial_edges(start_ts=start, end_ts=end, replay_basis=replay) == edges
        result = project_case(api, plan, replay, (physical_segment(start, end, imported=1.0),))
        assert len(result.items) == (92 if case == "spring" else 100)
        moments = [datetime.fromtimestamp(item.start_ts, zone) for item in result.items]
        if case == "autumn":
            folds = [(item.start_ts, moment.fold) for item, moment in zip(result.items, moments, strict=True)
                     if moment.hour == 2 and moment.minute == 0]
            assert folds[1][0] - folds[0][0] == 3600 and [fold for _, fold in folds] == [0, 1]
        else:
            assert not any(moment.hour == 2 for moment in moments)
    elif case == "contract":
        first = fixtures.canonical_snapshot(base.snapshots[0], valid_to_ts=NOW + 450)
        second = fixtures.canonical_snapshot(base.snapshots[0], valid_from_ts=NOW + 450,
            contract_id="synthetic:replacement", price_interval_seconds=900)
        plan, replay = fixtures.finance_case(edges=(NOW, NOW + 450, NOW + 900), snapshots=(first, second))
        assert api.financial_edges(start_ts=NOW, end_ts=NOW + 900, replay_basis=replay) == (NOW, NOW + 450, NOW + 900)
        result = project_case(api, plan, replay, (physical_segment(imported=1.0),))
        assert [(i.end_ts - i.start_ts, i.import_kwh) for i in result.items] == [(450, D(".5")), (450, D(".5"))]
        assert all(i.spot_ex_vat == D(".1") for i in result.items)
    else:
        day = date(2027, 3, 31) if case == "calendar" else date(2028, 2, 28)
        boundary_day = day + timedelta(days=1)
        start = int(datetime.combine(day, time(23, 45), zone).timestamp())
        boundary = int(datetime.combine(boundary_day, time.min, zone).timestamp())
        end = int(datetime.combine(boundary_day, time(2), zone).timestamp())
        bonus = replace(bonus_rules, start_date=boundary_day if case == "calendar" else date(2027, 1, 1),
                        windows=tuple((month, (1, 24)) for month in range(1, 13)))
        snapshot = fixtures.canonical_snapshot(base.snapshots[0], bonus=bonus,
            supply_start=boundary_day if case == "calendar" else date(2024, 2, 29),
            vat_consumption=((date(2020, 1, 1), D("0")), (boundary_day, D("21"))),
            cost_supplier_production=((date(2020, 1, 1), D("0")), (boundary_day, D(".01"))))
        edges = tuple(range(start, end + 1, 900))
        plan, replay = fixtures.finance_case(edges=edges, snapshots=(snapshot,))
        result = project_case(api, plan, replay, (physical_segment(start, end, exported=1.0),))
        assert api.financial_edges(start_ts=start, end_ts=end, replay_basis=replay) == edges
        assert {boundary, boundary + 3600}.issubset(i.start_ts for i in result.items)
        settlement = settle_series(result.items, replay.snapshots, replay.opening)
        assert not any("boundary_requires_split" in issue for line in settlement.lines for issue in line.issues)
        assert settlement.lines[-1].after_state.bonus.period_start_ts == boundary


@pytest.mark.parametrize("case", ["native", "mean", "missing-quarter", "missing-all", "spot", "import",
                                  "export", "components", "bonus", "snapshot"])
def test_f3b2_native_mean_quarters_and_quote_provenance(case):
    api = projection_api()
    plan, replay = fixtures.finance_case()
    quarters = tuple(SpotInterval(START_TS + i * 900, START_TS + (i + 1) * 900,
                                  D(value), "synthetic:quarter", "estimated")
                     for i, value in enumerate((".1", ".2", ".3", ".4")))
    hour = SpotInterval(START_TS, START_TS + 3600, D(".20"), "synthetic:native-hour", "estimated")
    snapshot = fixtures.canonical_snapshot(replay.snapshots[0],
        hour_price_rule="mean_quarters" if case in ("mean", "missing-quarter") else "native")
    spots = (*quarters, hour)
    if case == "missing-quarter":
        spots = (*quarters[:3], hour)
    elif case == "missing-all":
        spots = ()
    plan, replay = fixtures.finance_case(snapshots=(snapshot,), spots=spots)
    if case in ("spot", "import", "export", "components", "bonus"):
        changes = {"spot": {"spot_ex_vat": D(".21")}, "import": {"import_rate": D(".21")},
                   "export": {"export_rate": D(".21")}, "components": {"components": (("invented", D(".20")),)},
                   "bonus": {"bonus_rate": D(".01")}}[case]
        replay = replace(replay, quotes=(replace(replay.quotes[0], **changes),))
    elif case == "snapshot":
        other = fixtures.canonical_snapshot(snapshot, contract_id="synthetic:ambiguous-selection")
        replay = replace(replay, basis=replace(replay.basis, snapshot_hashes=(snapshot.snapshot_hash, other.snapshot_hash)),
                         price_batch=replace(replay.price_batch, snapshots=(snapshot, other)))
        plan = replace(plan, financial_context=replace(plan.financial_context, basis=replay.basis))
    plan = fixtures.bind_replay(plan, replay)
    plan.intervals[0].price_import = plan.intervals[0].price_export = plan.price_average = 999
    result = project_case(api, plan, replay, (physical_segment(imported=.125, exported=.25),))
    if case in ("spot", "import", "export", "components", "bonus", "snapshot"):
        assert result.items == () and result.blocking_reasons == ("replay_price_provenance_mismatch",)
    else:
        assert result.blocking_reasons == () and len(result.items) == 1
        item = result.items[0]
        if case.startswith("missing"):
            assert item.spot_ex_vat is None and item.quality == "incomplete"
            assert "replay_price_provenance_mismatch" not in result.reasons
        else:
            assert item.spot_ex_vat == D(".20" if case == "native" else ".25")
            assert item.quality == "estimated"


@pytest.mark.parametrize("source_kind", ["missing", "unknown", "gap", "mismatch"])
@pytest.mark.parametrize("zero_case", ["outside-window", "spot-zero", "spot-negative", "export-zero", "discharge", "eligible"])
def test_f3b2_missing_source_can_still_have_engine_zero_bonus(source_kind, zero_case, bonus_rules):
    api = projection_api()
    _, replay = fixtures.finance_case()
    rules = bonus_rules if zero_case == "outside-window" else replace(
        bonus_rules, windows=tuple((month, (0, 24)) for month in range(1, 13)))
    snapshot = fixtures.canonical_snapshot(replay.snapshots[0], bonus=rules)
    spot = D("0") if zero_case == "spot-zero" else D("-.1") if zero_case == "spot-negative" else D(".1")
    opening = SettlementState(NOW, BonusState(START_TS, START_TS + 365 * 86400, NOW, D("0")), None)
    plan, replay = fixtures.finance_case(snapshots=(snapshot,), opening=opening,
        spots=(SpotInterval(START_TS, START_TS + 3600, spot, "synthetic:bonus-control", "estimated"),))
    discharge = 2**-40 if zero_case == "discharge" else 0.0
    segments = (physical_segment(exported=0.0 if zero_case == "export-zero" else .25,
        batteries=(physical_battery(discharge=discharge, dc=discharge, modeled=-discharge),)),)
    result = project_case(api, plan, replay, segments, kind=source_kind)
    if source_kind == "mismatch":
        assert result.items == () and result.blocking_reasons == ("source_binding_mismatch",)
        return
    assert result.blocking_reasons == ()
    assert result.items[0].origin == "unknown" and result.items[0].quality == "estimated"
    reason = "source_coverage_gap" if source_kind == "gap" else "source_evidence_missing"
    assert reason in result.reasons and reason in result.slices[0].reasons
    if source_kind == "missing":
        assert "source:missing" in result.items[0].sources
    # Existing engine zero branches are controls, not new settlement behavior.
    line = settle_series(result.items, replay.snapshots, replay.opening).lines[0]
    assert line.bonus_revenue == (None if zero_case == "eligible" else D("0"))


def test_f3b2_projection_digest_binds_original_bits_scenario_and_source():
    api = projection_api()
    plan, replay = fixtures.finance_case()
    source, scenario = projection_source(api, plan, replay)
    segment = physical_segment(imported=.1)
    result = api.project_replay_inputs(segments=(segment,), start_ts=NOW, end_ts=NOW + 900,
        replay_basis=replay, source=source, scenario_digest=scenario)
    unknown, _ = projection_source(api, plan, replay, kind="unknown")
    variants = ((replace(segment, house_w=-0.0), source, scenario),
                (replace(segment, import_kwh=nextafter(.1, 1)), source, scenario),
                (segment, source, "c" * 64), (segment, unknown, scenario))
    for physical, validation, identity in variants:
        changed = api.project_replay_inputs(segments=(physical,), start_ts=NOW, end_ts=NOW + 900,
            replay_basis=replay, source=validation, scenario_digest=identity)
        assert changed.items[0].import_kwh == result.items[0].import_kwh == D(".1")
        assert changed.projection_digest != result.projection_digest
        assert changed.items[0].input_hash != result.items[0].input_hash
    # Detach caller-owned containers at the financial DTO boundary.
    segments, items, slices, reasons = [segment], list(result.items), list(result.slices), ["synthetic:reason"]
    detached = replace(result, segments=segments, items=items, slices=slices, reasons=reasons)
    for sequence in (segments, items, slices, reasons):
        sequence.clear()
    assert detached.segments == (segment,) and detached.items == result.items and detached.slices == result.slices
    assert detached.reasons == ("synthetic:reason",)
    flows = [["accu", D("0")]]
    detached_slice = replace(result.slices[0], dc_charge_by_battery=flows)
    flows[0][1] = D("1")
    assert detached_slice.dc_charge_by_battery == (("accu", D("0")),)


def result_api():
    api = projection_api()
    for symbol in ("FinancialStatus", "PeriodStatus", "BatteryFinancialTotals", "ReplayPhysicalTotals",
                   "FinancialComponent", "ReplayCostComponents", "FinancialDayKey", "ReplayFinancialDay",
                   "ReplayFinancialResult", "ReplayDayAdvantage", "ReplayFinancialComparison",
                   "settle_replay", "unknown_replay_financial", "compare_financial_results"):
        assert hasattr(api, symbol), f"F3b3 {symbol} missing"
    return api


def idle_totals(api, names=("accu",)):
    return api.ReplayPhysicalTotals(tuple(
        api.BatteryFinancialTotals(name, D("10"), D(".03"), D("50"), D("50")) for name in names))


def terminal_case(plan, replay, *, price=".2", efficiencies=(("accu", D(".7")),)):
    terminal = replace(replay.terminal_value, eur_per_ac_kwh=D(price),
                       reference_efficiency_by_battery=efficiencies)
    replay = replace(replay, terminal_value=terminal)
    plan = replace(plan, financial_context=replace(plan.financial_context, terminal_value=terminal))
    return fixtures.bind_replay(plan, replay), replay


@pytest.mark.parametrize("case", ["cap", "unknown-day", "year-reset", "month-close", "month-open",
                                  "month-incomplete", "contract-close", "vat-change", "floor-none"])
def test_f3b3_independent_books_period_close_and_unknown_prefix(case, bonus_rules, monkeypatch):
    # Catches shared/renewed openings, replay-end month closing, and tariff hashes
    # mistaken for contract switches. Engine checks below are positive controls.
    api = projection_api()
    _, original = fixtures.finance_case()
    snapshot = original.snapshots[0]
    if case in ("cap", "unknown-day", "year-reset"):
        rules = replace(bonus_rules, windows=tuple((month, (0, 24)) for month in range(1, 13)))
        snapshot = fixtures.canonical_snapshot(snapshot, bonus=rules, supply_start=date(2027, 1, 1))
        start = NOW if case == "cap" else START_TS + 86400 - 900 if case == "unknown-day" else YEAR_END_TS - 900
        end = start + (900 if case == "year-reset" else 1800)
        opening = SettlementState(start, BonusState(START_TS, YEAR_END_TS, start,
                                  D("4998.5") if case == "cap" else None), None)
        snapshots = (snapshot,)
    else:
        start, end = MONTH_END_TS - 900, MONTH_END_TS
        if case == "month-open":
            start, end = start - 900, end - 900
        elif case == "vat-change":
            start -= 900
        elif case == "contract-close":
            end = START_TS + 14 * 86400 + 12 * 3600
            start = end - 900
        snapshot = fixtures.canonical_snapshot(snapshot,
            monthly_export_floor="none" if case == "floor-none" else "spot_zero")
        snapshots = (snapshot,)
        if case == "contract-close":
            snapshots = (fixtures.canonical_snapshot(snapshot, valid_to_ts=end),
                         fixtures.canonical_snapshot(snapshot, valid_from_ts=end, contract_id="synthetic:next"))
        elif case == "vat-change":
            snapshots = (fixtures.canonical_snapshot(snapshot, valid_to_ts=end - 900,
                            vat_production=((date(2020, 1, 1), D("9")),)),
                         fixtures.canonical_snapshot(snapshot, valid_from_ts=end - 900,
                            vat_production=((date(2020, 1, 1), D("21")),)))
        month = MonthState(snapshot.contract_id, START_TS, MONTH_END_TS, D("-2"),
                           ((D("9"), D("1")), (D("21"), D("1"))), True)
        if case == "month-incomplete":
            month = replace(month, spot_export_eur=None, negative_basis_by_vat=(), complete=False)
        opening = SettlementState(start, None, None if case == "floor-none" else month)
    edges = tuple(range(start, end + 1, 900))
    spots = tuple(SpotInterval(at, at + 3600, D(".20"), "synthetic:books", "estimated")
                  for at in range(start // 3600 * 3600, end, 3600))
    plan, replay = fixtures.finance_case(edges=edges, snapshots=snapshots, spots=spots, opening=opening)
    projections = tuple(project_case(api, plan, replay, tuple(
        physical_segment(a, b, exported=amount) for a, b in zip(edges, edges[1:], strict=False)))
        for amount in ((1.0, .25) if case == "cap" else (.25, .25) if case == "unknown-day" else (0.0, 0.0)))
    assert all(not p.blocking_reasons for p in projections)
    controls = tuple(settle_series(p.items, replay.snapshots, replay.opening) for p in projections)
    if case == "cap":
        assert [r.end_state.bonus.total_export_kwh for r in controls] == [D("5000.5"), D("4999")]
        assert [sum((line.bonus_revenue for line in r.lines), D(0)) for r in controls] == [D(".09"), D(".03")]
    elif case == "unknown-day":
        assert all(r.end_state.bonus.total_export_kwh is None and r.net_cost is None for r in controls)
    elif case == "year-reset":
        assert all(r.end_state.bonus.total_export_kwh == D("0") for r in controls)
    elif case in ("month-close", "contract-close", "vat-change"):
        assert all(len(r.adjustments) == 1 and r.adjustments[0].credit_total == D("2.30") for r in controls)
    else:
        assert all(r.adjustments == () for r in controls)
        if case == "month-incomplete":
            assert all(r.net_cost is None for r in controls)
    api = result_api()
    totals = idle_totals(api)
    before = deepcopy((projections, replay, totals))
    calls = []

    def observe(items, pinned, initial):
        calls.append((items, pinned, initial))
        return settle_series(items, pinned, initial)

    monkeypatch.setattr(api, "settle_series", observe)
    base = api.settle_replay(projections[0], replay, totals)
    fast = api.settle_replay(projections[1], replay, totals)
    assert base.settlement == controls[0] and fast.settlement == controls[1]
    assert api.settle_replay(projections[1], replay, totals) == fast
    assert api.settle_replay(projections[0], replay, totals) == base
    assert len(calls) == 4  # Includes no extra settlement for financial days.
    assert all(pinned is replay.snapshots and initial is opening for _, pinned, initial in calls)
    assert (projections, replay, totals) == before
    expected_period = ("unknown" if case == "month-incomplete" else "as_of_open_period" if case == "month-open"
                       else "closed" if case in ("month-close", "contract-close", "vat-change") else "not_applicable")
    for result in (base, fast):
        assert_result_diagnostics(result)
        assert result.period_status == expected_period
        assert result.components.invoice_energy_cost.value == result.settlement.net_cost
        assert result.source_trust == "caller_asserted" and result.economics_is_estimate is True
        assert result.economics_kind == "settlement_counterfactual"
        if case in ("month-close", "contract-close", "vat-change"):
            assert result.components.month_credit.value == D("2.30")
            credit_days = [day for day in result.days if day.components.month_credit.value]
            assert len(credit_days) == 1
            assert credit_days[0].key.local_date == datetime.fromtimestamp(end - 1, ZoneInfo(snapshot.timezone)).date().isoformat()
        elif case == "month-incomplete":
            assert result.components.month_credit.value is None
            assert result.components.total_cost.value is None
        elif case == "month-open":
            assert result.settlement.end_state.month is not None
            assert result.components.invoice_energy_cost.value == D("0")
            assert result.financial_status == "complete"
        if case == "vat-change":
            assert len(result.days) == 1 and result.days[0].key.contract_start_ts is None


def test_f3b3_component_coverage_and_daily_allocation(bonus_rules):
    # Catches known subtotal relabelled as invoice/advantage, repeated terminal
    # pricing, and opening-prefix credit spread over earlier export days.
    api = projection_api()
    _, replay = fixtures.finance_case()
    rules = replace(bonus_rules, windows=tuple((month, (0, 24)) for month in range(1, 13)))
    snapshot = fixtures.canonical_snapshot(replay.snapshots[0], bonus=rules,
        cost_supplier_consumption=((date(2020, 1, 1), D(".20")),))
    opening = SettlementState(NOW, BonusState(START_TS, YEAR_END_TS, NOW, D("0")), None)
    plan, replay = fixtures.finance_case(snapshots=(snapshot,), opening=opening)
    unknown_projection = project_case(api, plan, replay, (physical_segment(imported=1.0, exported=2.0),), kind="missing")
    known_projection = project_case(api, plan, replay, (physical_segment(imported=1.0, exported=2.0),))
    control = settle_series(unknown_projection.items, replay.snapshots, replay.opening)
    assert control.net_cost is None and control.known_subtotal == D(".10")
    api = result_api()
    partial = api.settle_replay(unknown_projection, replay, idle_totals(api))
    known = api.settle_replay(known_projection, replay, idle_totals(api))
    components = partial.components
    assert components.import_cost.value == D(".30") and components.import_cost.status == "complete"
    assert components.base_export_revenue.value == D(".20")
    assert components.bonus_revenue.value is None and components.bonus_revenue.status == "unknown"
    assert "unknown_bonus_source" in components.bonus_revenue.reasons
    assert components.invoice_energy_cost.value is None and components.invoice_energy_cost.known_subtotal == D(".10")
    assert components.total_cost.value is None and partial.financial_status == "partial"
    assert partial.source_trust == "missing" and known.financial_status == "complete"
    for comparison in (api.compare_financial_results(partial, known), api.compare_financial_results(known, partial)):
        assert comparison.advantage_eur is None and comparison.financial_status == "partial"
        assert comparison.reasons and all(day.advantage_eur is None for day in comparison.days)

    start, end = MONTH_END_TS - 86400 - 900, MONTH_END_TS + 900
    edges = tuple(range(start, end + 1, 900))
    snapshot = fixtures.canonical_snapshot(snapshot, bonus=None, monthly_export_floor="spot_zero")
    month = MonthState(snapshot.contract_id, START_TS, MONTH_END_TS, D("-2"),
                       ((D("9"), D("1")), (D("21"), D("1"))), True)
    plan, replay = fixtures.finance_case(edges=edges, snapshots=(snapshot,), opening=SettlementState(start, None, month))
    plan, replay = terminal_case(plan, replay)
    segments = tuple(physical_segment(a, b, batteries=(physical_battery(charge=.25, dc=.125, modeled=1000.0),))
                     for a, b in zip(edges, edges[1:], strict=False))
    projection = project_case(api, plan, replay, segments)
    totals = api.ReplayPhysicalTotals((api.BatteryFinancialTotals("accu", D("10"), D(".03"), D("40"), D("41")),))
    result = api.settle_replay(projection, replay, totals)
    baseline = api.settle_replay(project_case(api, plan, replay, (physical_segment(start, end),)), replay, idle_totals(api))
    assert result.financial_status == "complete" and result.period_status == "as_of_open_period"
    assert [day.key.local_date for day in result.days] == ["2027-01-30", "2027-01-31", "2027-02-01"]
    assert [day.components.month_credit.value for day in result.days] == [D("0"), D("2.30"), D("0")]
    assert [day.components.terminal_value.value for day in result.days] == [D("0"), D("0"), D(".014")]
    assert [day.components.wear_cost.value for day in result.days] == [D(".00375"), D(".36"), D(".00375")]
    assert [day.period_status for day in result.days] == ["as_of_open_period", "closed", "as_of_open_period"]
    assert sum((day.components.total_cost.value for day in result.days), D(0)) == result.components.total_cost.value
    comparison = api.compare_financial_results(baseline, result)
    assert comparison.financial_status == "complete" and comparison.advantage_eur == D("-.3535")
    assert sum((day.advantage_eur for day in comparison.days), D(0)) == comparison.advantage_eur
    for side in (baseline, result):
        assert_result_diagnostics(side)
    assert_comparison_diagnostics(comparison)


def test_f3b3_own_wear_terminal_once_and_negative_advantage():
    # Catches cell/AC throughput, averaged battery parameters, double efficiency,
    # clipping losses, and recovery of unknown SoC by a later proven zero stop.
    api = result_api()
    plan, replay = fixtures.finance_case(battery_names=("accu", "B"))
    plan, replay = terminal_case(plan, replay, efficiencies=(("accu", D(".7")), ("B", D(".9"))))
    segments = (physical_segment(batteries=(physical_battery(charge=.25, dc=.125, modeled=1000.0),
                    physical_battery("B", discharge=.25, dc=.25, modeled=-1000.0))),)
    projection = project_case(api, plan, replay, segments)
    batteries = (api.BatteryFinancialTotals("accu", D("10"), D(".03"), D("40"), D("41")),
                 api.BatteryFinancialTotals("B", D("20"), D(".09"), D("80"), D("77.5")))
    totals = api.ReplayPhysicalTotals(batteries)
    result = api.settle_replay(projection, replay, totals)
    assert result.components.invoice_energy_cost.value == D("0")
    assert result.components.wear_cost.value == D(".02625")
    assert result.components.terminal_value.value == D("-.076")
    assert result.components.total_cost.value == D(".10225")
    idle_projection = project_case(api, plan, replay, (physical_segment(batteries=(physical_battery(), physical_battery("B"))),))
    baseline = api.settle_replay(idle_projection, replay, idle_totals(api, ("accu", "B")))
    comparison = api.compare_financial_results(baseline, result)
    assert comparison.advantage_eur == D("-.10225")
    assert api.compare_financial_results(result, baseline).advantage_eur == D(".10225")
    assert comparison.days[0].advantage_eur == comparison.advantage_eur
    changed = replace(segments[0], batteries=tuple(replace(battery, dc_throughput_kwh=2 * battery.dc_throughput_kwh)
                                                   for battery in segments[0].batteries))
    more_wear = api.settle_replay(project_case(api, plan, replay, (changed,)), replay, totals)
    assert more_wear.components.terminal_value == result.components.terminal_value
    assert more_wear.components.wear_cost.value == D(".05250")
    for field in ("soc_start", "soc_end"):
        unknown = api.settle_replay(idle_projection, replay,
            api.ReplayPhysicalTotals((replace(batteries[0], **{field: None}), batteries[1])))
        assert unknown.components.wear_cost.value == D("0")
        assert unknown.components.terminal_value.value is None and unknown.components.total_cost.value is None
        assert unknown.financial_status == "partial"
    with localcontext(Context(prec=6)):
        assert api.settle_replay(projection, replay, totals) == result
        assert api.compare_financial_results(baseline, result) == comparison


@pytest.mark.parametrize("fault", ["source", "fractional", "empty", "basis", "battery", "terminal"])
def test_f3b3_blocked_books_never_offer_component_claims(fault):
    api = result_api()
    plan, replay = fixtures.finance_case()
    options = {"kind": "mismatch"} if fault == "source" else {"end": NOW + 899.5} if fault == "fractional" else {}
    projection = project_case(api, plan, replay, (physical_segment(imported=1.0),), **options)
    totals = idle_totals(api)
    if fault == "empty":
        projection = replace(projection, items=(), slices=())
    elif fault == "basis":
        replay = replace(replay, terminal_value=replace(replay.terminal_value, eur_per_ac_kwh=D(".2")))
    elif fault == "battery":
        totals = idle_totals(api, ("other",))
    elif fault == "terminal":
        # A malformed plan is rejected earlier by plan_digest. Exercise the
        # adapter boundary directly with a source rebound to this archive.
        replay = replace(replay, terminal_value=replace(replay.terminal_value,
                         reference_efficiency_by_battery=(("other", D(".7")),)))
        evidence = replace(projection.source.evidence, replay_basis_hash=replay.content_hash)
        binding = replace(projection.source.binding, replay_basis_hash=replay.content_hash,
                          source_digest=evidence.content_hash)
        projection = api.project_replay_inputs(segments=projection.segments, start_ts=NOW, end_ts=NOW + 900,
            replay_basis=replay, source=api.validate_source_evidence(evidence, binding),
            scenario_digest=projection.scenario_digest)
    result = api.settle_replay(projection, replay, totals)
    assert result.settlement is None and result.financial_status == "unknown" and result.days == ()
    assert result.financial_reasons and result.period_status == "unknown"
    reason = {"source": "source_binding_mismatch", "fractional": "projection_noninteger_window",
              "empty": "replay_projection_empty", "basis": "replay_basis_mismatch",
              "battery": "replay_battery_identity_mismatch", "terminal": "replay_terminal_identity_mismatch"}[fault]
    assert reason in result.financial_reasons
    for field in fields(result.components):
        component = getattr(result.components, field.name)
        assert component.value is None and component.known_subtotal == D("0") and component.status == "unknown"


@pytest.mark.parametrize("case", ["missing-source-zero", "missing-price", "gap", "unknown-throughput"])
def test_f3b3_data_unknown_preserves_only_covered_components(case):
    api = result_api()
    plan, replay = fixtures.finance_case(edges=(NOW, NOW + 900, NOW + 1800), spots=() if case == "missing-price" else None)
    segments = (physical_segment(imported=1.0), physical_segment(NOW + 900, NOW + 1800, imported=1.0))
    if case == "gap":
        segments = segments[:1]
    elif case == "unknown-throughput":
        segments = (segments[0], replace(segments[1], batteries=(physical_battery(dc=None),)))
    projection = project_case(api, plan, replay, segments, kind="missing")
    totals = idle_totals(api)
    if case == "gap":
        totals = api.ReplayPhysicalTotals((replace(totals.batteries[0], soc_end=None),))
    result = api.settle_replay(projection, replay, totals)
    assert result.settlement is not None
    assert (result.settlement.coverage_start_ts, result.settlement.coverage_end_ts) == (NOW, NOW + 1800)
    assert result.components.bonus_revenue.value == D("0")
    if case == "missing-source-zero":
        assert result.financial_status == "complete" and result.components.total_cost.value == D(".2")
        assert result.source_trust == "missing"
    else:
        assert result.financial_status == "partial" and result.components.total_cost.value is None
        if case == "gap":
            assert result.components.import_cost.value is None and result.components.import_cost.known_subtotal == D(".1")
            assert result.components.import_cost.status == "partial"
        elif case == "missing-price":
            assert result.components.import_cost.value is None and result.components.import_cost.status == "unknown"
            assert result.components.wear_cost.value == D("0")
        else:
            assert result.components.invoice_energy_cost.value == D(".2")
            assert result.components.wear_cost.value is None and result.components.wear_cost.status == "partial"


def test_f3b3_daily_keys_use_full_contract_segments_not_date_or_tariff_hash():
    api = result_api()
    _, original = fixtures.finance_case()
    start, switch, end = NOW, NOW + 900, NOW + 2700
    first = fixtures.canonical_snapshot(original.snapshots[0], valid_to_ts=switch)
    second = fixtures.canonical_snapshot(original.snapshots[0], valid_from_ts=switch, valid_to_ts=switch + 900,
                                         contract_id="synthetic:UTC", timezone="UTC")
    third = fixtures.canonical_snapshot(second, valid_from_ts=switch + 900, valid_to_ts=None,
                                        cost_supplier_consumption=((date(2020, 1, 1), D(".2")),))
    plan, replay = fixtures.finance_case(edges=(start, switch, switch + 900, end), snapshots=(first, second, third))
    result = api.settle_replay(project_case(api, plan, replay, (physical_segment(start, end, imported=3.0),)),
                               replay, idle_totals(api))
    assert len(result.days) == 2
    a, b = result.days
    # These instants share a UTC range but their date labels differ; another pair
    # below shares the date text and must still retain both contract/zone keys.
    assert (a.key.timezone, a.key.contract_id, a.key.contract_start_ts) == ("Europe/Amsterdam", first.contract_id, None)
    assert (b.key.timezone, b.key.contract_id, b.key.contract_start_ts) == ("UTC", second.contract_id, switch)
    assert (a.start_ts, a.end_ts, b.start_ts, b.end_ts) == (start, switch, switch, end)
    assert [day.components.invoice_energy_cost.value for day in result.days] == [D(".1"), D(".4")]
    start += 7200
    switch += 7200
    end += 7200
    first = fixtures.canonical_snapshot(first, valid_to_ts=switch)
    second = fixtures.canonical_snapshot(second, valid_from_ts=switch, valid_to_ts=switch + 900)
    third = fixtures.canonical_snapshot(third, valid_from_ts=switch + 900)
    plan, replay = fixtures.finance_case(edges=(start, switch, switch + 900, end), snapshots=(first, second, third))
    result = api.settle_replay(project_case(api, plan, replay, (physical_segment(start, end),)), replay, idle_totals(api))
    assert [day.key.local_date for day in result.days] == ["2027-01-01", "2027-01-01"]
    assert len({day.key for day in result.days}) == 2
    assert len(api.compare_financial_results(result, result).days) == 2
    assert_result_diagnostics(result)
    assert_comparison_diagnostics(api.compare_financial_results(result, result))


def test_f3b3_result_dtos_are_detached_and_unknown_has_no_default_claim():
    api = result_api()
    reasons = ["synthetic:unavailable"]
    result = api.unknown_replay_financial(reasons=reasons)
    reasons.clear()
    assert result.projection is result.settlement is None and result.days == ()
    assert result.financial_status == "unknown" and result.financial_reasons == ("synthetic:unavailable",)
    assert result.source_trust == "missing" and result.components.total_cost.value is None
    comparison = api.compare_financial_results(result, result)
    assert comparison.advantage_eur is None and comparison.financial_status == "unknown"
    battery = idle_totals(api).batteries[0]
    batteries = [battery]
    totals = api.ReplayPhysicalTotals(batteries)
    batteries.clear()
    assert totals.batteries == (battery,)
    for dto in (result, comparison, result.components, result.components.import_cost, totals, battery):
        assert not hasattr(dto, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(dto, fields(dto)[0].name, None)
        for field in fields(dto):
            if field.name in ("economics_is_estimate", "economics_kind", "reconciliation"):
                continue
            assert inspect.signature(type(dto)).parameters[field.name].default is inspect.Parameter.empty
    for changes in ({"capacity_kwh": D("-1")}, {"cycle_cost": D("-1")}, {"soc_start": D("NaN")},
                    {"soc_end": 50.0}, {"battery_name": ""}):
        with pytest.raises((ValueError, TypeError)):
            replace(battery, **changes)
    with pytest.raises((TypeError, ValueError)):
        api.ReplayPhysicalTotals((battery, battery))


@pytest.mark.parametrize("quarters", [3, 98])
def test_f3b3_binary_costs_reconcile_financial_days(quarters):
    # F3b4 supplies binary-converted physical totals; those long Decimals must
    # reconcile across financial days just like the short literal witnesses.
    api = result_api()
    start = START_TS + 86400 - 900
    edges = tuple(start + index * 900 for index in range(quarters + 1))
    plan, replay = fixtures.finance_case(edges=edges)
    segments = tuple(physical_segment(a, b, batteries=(physical_battery(charge=.1, dc=.1, modeled=400.0),))
                     for a, b in zip(edges, edges[1:], strict=False))
    totals = api.ReplayPhysicalTotals((api.BatteryFinancialTotals(
        "accu", D.from_float(10.0), D.from_float(.03), D.from_float(40.0), D.from_float(41.0)),))
    result = api.settle_replay(project_case(api, plan, replay, segments), replay, totals)
    baseline = api.settle_replay(project_case(api, plan, replay, (physical_segment(edges[0], edges[-1]),)),
                                 replay, idle_totals(api))
    assert result.financial_status == "complete"
    with localcontext(Context(prec=36)):
        assert sum((day.components.wear_cost.value for day in result.days), D(0)) == result.components.wear_cost.value
        assert sum((day.components.total_cost.value for day in result.days), D(0)) == result.components.total_cost.value
        comparison = api.compare_financial_results(baseline, result)
        assert sum((day.advantage_eur for day in comparison.days), D(0)) == comparison.advantage_eur


@pytest.mark.parametrize("directions", [("charge", "charge", "charge"), ("charge", "discharge", "charge")])
@pytest.mark.parametrize("component", ["wear_cost", "total_cost", "advantage"])
def test_f3b3_fix1_unequal_directional_bookings_reconcile_days(directions, component):
    # I1/M1: a + (a + 4a) crosses a financial day boundary; repeated identical
    # .1 throughput did not expose the differing context36 accumulation trees.
    api = result_api()
    start = START_TS + 86400 - 900
    edges = tuple(start + index * 900 for index in range(4))
    plan, replay = fixtures.finance_case(edges=edges)
    throughputs = (.125, .125, .5)
    segments = tuple(physical_segment(a, b, batteries=(physical_battery(
        charge=amount if direction == "charge" else 0.0,
        discharge=amount if direction == "discharge" else 0.0,
        dc=amount, modeled=4000 * amount * (1 if direction == "charge" else -1)),))
        for a, b, amount, direction in zip(edges, edges[1:], throughputs, directions, strict=False))
    projection = project_case(api, plan, replay, segments)
    for piece, amount, direction in zip(projection.slices, throughputs, directions, strict=True):
        assert piece.dc_charge_by_battery == (("accu", D.from_float(amount) if direction == "charge" else D(0)),)
        assert piece.dc_discharge_by_battery == (("accu", D.from_float(amount) if direction == "discharge" else D(0)),)
    totals = api.ReplayPhysicalTotals((api.BatteryFinancialTotals(
        "accu", D("10"), D.from_float(.03), D("50"), D("50")),))
    result = api.settle_replay(projection, replay, totals)
    baseline = api.settle_replay(project_case(api, plan, replay, (physical_segment(edges[0], edges[-1]),)),
                                 replay, idle_totals(api))
    assert result.financial_status == baseline.financial_status == "complete"
    assert [day.key.local_date for day in result.days] == ["2027-01-01", "2027-01-02"]
    assert result.components.invoice_energy_cost.value == result.components.terminal_value.value == D(0)
    comparison = api.compare_financial_results(baseline, result)
    assert comparison.financial_status == "complete" and comparison.advantage_eur < 0
    with localcontext(Context(prec=36)):
        if component == "advantage":
            day_sum = sum((day.advantage_eur for day in comparison.days), D(0))
            whole = comparison.advantage_eur
        else:
            day_sum = sum((getattr(day.components, component).value for day in result.days), D(0))
            whole = getattr(result.components, component).value
        # Historical context36 values are controls, not the exact diagnosis.
        assert day_sum == D("-.0224999999999999991673327315311325946" if component == "advantage"
                            else ".0224999999999999991673327315311325946")
        assert whole == D("-.0224999999999999991673327315311325947" if component == "advantage"
                          else ".0224999999999999991673327315311325947")
    positive_days = (D(".00374999999999999986122212192185543245"), D(".0187499999999999993061106096092771622"))
    negative_days = (D("-.00374999999999999986122212192185543245"), D("-.0187499999999999993061106096092771622"))
    for name in ("wear_cost", "total_cost"):
        assert tuple(getattr(day.components, name).value for day in result.days) == positive_days
    assert tuple(day.advantage_eur for day in comparison.days) == negative_days
    assert_result_diagnostics(baseline, {field.name: (D(0), D(0), D(0)) for field in fields(baseline.components)})
    cost_pin = (D(".0224999999999999991673327315311325947"), D(".02249999999999999916733273153113259465"), D("5E-38"))
    assert_result_diagnostics(result, {field.name: cost_pin if field.name in ("wear_cost", "total_cost")
                                      else (D(0), D(0), D(0)) for field in fields(result.components)})
    assert_comparison_diagnostics(comparison, (D("-.0224999999999999991673327315311325947"),
        D("-.02249999999999999916733273153113259465"), D("-5E-38")))
    assert_comparison_diagnostics(api.compare_financial_results(result, baseline), cost_pin)


def test_f3b3_fix1_fixed_engine_amounts_require_incompatible_advantage_sums():
    # Two fixed engine lines, one per financial day/month: no wear accumulator
    # or association choice can change these without changing booked amounts.
    api = result_api()
    start = MONTH_END_TS - 900
    edges = (start, MONTH_END_TS, MONTH_END_TS + 900)
    _, seed = fixtures.finance_case(edges=edges)
    snapshot = fixtures.canonical_snapshot(seed.snapshots[0],
        multiplier_consumption=((date(2020, 1, 1), D("1.123456789012")),),
        vat_consumption=((date(2020, 1, 1), D("21.123456789012")),))
    spots = tuple(SpotInterval(at, at + 3600, D(".234567890123"), "synthetic:fix1-proof", "estimated")
                  for at in range(start // 3600 * 3600, edges[-1], 3600))
    plan, replay = fixtures.finance_case(edges=edges, snapshots=(snapshot,), spots=spots)
    assert replay.quotes[0].import_rate == D(".319192877092817063446337600713403057")
    results = []
    for amounts in ((.125, .125), (.125, .25)):
        segments = tuple(physical_segment(a, b, imported=amount)
                         for a, b, amount in zip(edges, edges[1:], amounts, strict=False))
        results.append(api.settle_replay(project_case(api, plan, replay, segments), replay, idle_totals(api)))
    baseline, fast = results
    a = D(".0398991096366021329307922000891753821")
    twice_a = D(".0797982192732042658615844001783507642")
    rounded_three_a = D(".119697328909806398792376600267526146")
    assert [line.import_cost for line in baseline.settlement.lines] == [a, a]
    assert [line.import_cost for line in fast.settlement.lines] == [a, twice_a]
    assert baseline.settlement.net_cost == twice_a
    assert fast.settlement.net_cost == rounded_three_a
    for result in results:
        assert result.financial_status == "complete" and result.settlement.complete
        assert result.settlement.adjustments == () and result.period_status == "not_applicable"
        assert result.components.wear_cost.value == result.components.terminal_value.value == D(0)
        assert result.components.total_cost.value == result.settlement.net_cost
        assert [day.key.local_date for day in result.days] == ["2027-01-31", "2027-02-01"]
        assert [day.components.total_cost.value for day in result.days] == [line.import_cost for line in result.settlement.lines]
    comparison = api.compare_financial_results(baseline, fast)
    with localcontext(Context(prec=36)):
        for result in results:
            assert sum((day.components.total_cost.value for day in result.days), D(0)) == result.components.total_cost.value
        assert comparison.advantage_eur == baseline.settlement.net_cost - fast.settlement.net_cost
        assert [day.advantage_eur for day in comparison.days] == [D(0), a - twice_a]
        assert sum((day.advantage_eur for day in comparison.days), D(0)) == D("-.0398991096366021329307922000891753821")
        assert comparison.advantage_eur == D("-.0398991096366021329307922000891753818")
    energy_metrics = ("import_cost", "invoice_energy_cost", "total_cost")
    assert_result_diagnostics(baseline, {field.name: (twice_a, twice_a, D(0)) if field.name in energy_metrics
                                        else (D(0), D(0), D(0)) for field in fields(baseline.components)})
    assert_result_diagnostics(fast, {field.name: (rounded_three_a, D(".1196973289098063987923766002675261463"), D("-3E-37"))
                                    if field.name in energy_metrics else (D(0), D(0), D(0)) for field in fields(fast.components)})
    assert_comparison_diagnostics(comparison, (D("-.0398991096366021329307922000891753818"),
                                              D("-.0398991096366021329307922000891753821"), D("3E-37")))
    assert_comparison_diagnostics(api.compare_financial_results(fast, baseline),
        (D(".0398991096366021329307922000891753818"), a, D("-3E-37")))


def reconciliation_api():
    api = result_api()
    for name in ("ReconciliationMetric", "ReconciliationStatus", "FinancialReconciliation", "_exact_booked_sum",
                 "_reconcile_days", "_validate_book_structure", "_validate_day_structure", "_reconciliation_coverage_reasons"):
        assert hasattr(api, name), f"F3b3 fix2 {name} missing"
    return api


def assert_diagnostic(diagnostic, total, day_values, pin=None):
    assert diagnostic.authoritative_total == total
    if total is None or not day_values or any(value is None for value in day_values):
        assert diagnostic.status == "unknown" and diagnostic.day_sum is diagnostic.difference is None
        assert diagnostic.reasons
        return
    exact_sum = sum((Fraction(value) for value in day_values), Fraction(0))
    assert Fraction(diagnostic.day_sum) == exact_sum
    assert Fraction(diagnostic.difference) == Fraction(total) - exact_sum
    assert Fraction(total) == exact_sum + Fraction(diagnostic.difference)
    assert diagnostic.status == ("equal" if diagnostic.difference == 0 else "difference")
    assert diagnostic.reasons == ()
    if pin is not None:
        assert (diagnostic.authoritative_total, diagnostic.day_sum, diagnostic.difference) == pin


def assert_result_diagnostics(result, pins=None):
    assert hasattr(result, "reconciliation"), "F3b3 fix2 result reconciliation missing"
    names = tuple(field.name for field in fields(result.components))
    assert tuple(item.metric for item in result.reconciliation) == names
    for name, item in zip(names, result.reconciliation, strict=True):
        assert_diagnostic(item, getattr(result.components, name).value,
                          tuple(getattr(day.components, name).value for day in result.days),
                          None if pins is None else pins[name])


def assert_comparison_diagnostics(comparison, pin=None):
    assert hasattr(comparison, "reconciliation"), "F3b3 fix2 comparison reconciliation missing"
    assert len(comparison.reconciliation) == 1 and comparison.reconciliation[0].metric == "advantage_eur"
    assert_diagnostic(comparison.reconciliation[0], comparison.advantage_eur,
                      tuple(day.advantage_eur for day in comparison.days), pin)


def fix2_book_case(api):
    start, end = MONTH_END_TS - 900, MONTH_END_TS + 1800
    edges = tuple(range(start, end + 1, 900))
    _, seed = fixtures.finance_case()
    snapshot = fixtures.canonical_snapshot(seed.snapshots[0], monthly_export_floor="spot_zero")
    month = MonthState(snapshot.contract_id, START_TS, MONTH_END_TS, D("-2"),
                       ((D("9"), D("1")), (D("21"), D("1"))), True)
    plan, replay = fixtures.finance_case(edges=edges, snapshots=(snapshot,), opening=SettlementState(start, None, month))
    plan, replay = terminal_case(plan, replay)
    segments = tuple(physical_segment(a, b, batteries=(physical_battery(charge=dc, dc=dc, modeled=dc * 4000),))
                     for a, b, dc in zip(edges, edges[1:], (.125, .25, .5), strict=False))
    totals = api.ReplayPhysicalTotals((api.BatteryFinancialTotals("accu", D("10"), D(".03"), D("40"), D("41")),))
    return project_case(api, plan, replay, segments), replay, totals


@pytest.mark.parametrize("precision", [6, 28])
@pytest.mark.parametrize("rounding", [ROUND_DOWN, ROUND_UP])
@pytest.mark.parametrize("traps", [False, True])
@pytest.mark.parametrize("set_flags", [False, True])
def test_f3b3_fix2_exact_diagnostics_and_dto_contract(precision, rounding, traps, set_flags):
    api = reconciliation_api()
    projection, replay, totals = fix2_book_case(api)
    expected = api.settle_replay(projection, replay, totals)
    expected_comparison = api.compare_financial_results(expected, expected)
    long = D((0, (1,) * 5000, -2500))
    opposite = D((1, (1,) * 5000, -2500))
    large_sum = D((0, (1,) + (0,) * 1999 + (1,), -1000))
    invalid = (0.0, 0, True, "0", D("NaN"), D("sNaN"), D("Infinity"), D("-Infinity"))
    with localcontext() as context:
        context.prec, context.rounding = precision, rounding
        context.traps[Inexact] = context.traps[Rounded] = traps
        context.clear_flags()
        context.flags[Inexact] = context.flags[Rounded] = set_flags
        before = (context.prec, context.rounding, context.Emin, context.Emax, context.clamp,
                  dict(context.traps), dict(context.flags))
        for values in ((), (D("-0E-100"), D("0E+2")), (long, opposite)):
            assert api._exact_booked_sum(values).as_tuple() == D(0).as_tuple()
        assert api._exact_booked_sum((long, opposite, D("1E-6000"))) == D("1E-6000")
        assert api._exact_booked_sum((D("1E+1000"), D("1E-1000"))).as_tuple() == large_sum.as_tuple()
        for total, daily, difference in ((D("1"), (D(".3"), D(".7")), D(0)),
                                         (D("1"), (D(".3"), D(".6")), D(".1")),
                                         (D("-1"), (D("-.3"), D("-.6")), D("-.1")),
                                         (D("1000"), (D("1"),), D("999"))):
            item = api._reconcile_days("total_cost", total, daily)
            assert item.difference == difference
            assert_diagnostic(item, total, daily)
        for total, daily, reasons, want in (
            (D(0), (), (), ("reconciliation_coverage_unknown",)),
            (None, (D(0),), (), ("reconciliation_total_unknown",)),
            (D(0), (None,), (), ("reconciliation_day_unknown",)),
            (None, (None,), ("synthetic:source", "reconciliation_coverage_unknown", "synthetic:source"),
             ("reconciliation_total_unknown", "reconciliation_day_unknown", "reconciliation_coverage_unknown", "synthetic:source")),
        ):
            item = api._reconcile_days("total_cost", total, daily, reasons=reasons)
            assert (item.authoritative_total, item.day_sum, item.difference, item.status, item.reasons) == (
                total, None, None, "unknown", want)
        for value in invalid:
            with pytest.raises((TypeError, ValueError)):
                api._exact_booked_sum((D(0), value))
            with pytest.raises((TypeError, ValueError)):
                api._reconcile_days("total_cost", None, (None, value))
            with pytest.raises((TypeError, ValueError)):
                api._reconcile_days("total_cost", value, (None,))
            for field in ("authoritative_total", "day_sum", "difference"):
                kwargs = dict(metric="total_cost", authoritative_total=D(0), day_sum=D(0), difference=D(0), status="equal", reasons=())
                kwargs[field] = value
                with pytest.raises((TypeError, ValueError)):
                    api.FinancialReconciliation(**kwargs)
        actual = api.settle_replay(projection, replay, totals)
        assert actual == expected
        assert api.compare_financial_results(actual, actual) == expected_comparison
        assert_result_diagnostics(actual)
        assert_comparison_diagnostics(expected_comparison)
        assert before == (context.prec, context.rounding, context.Emin, context.Emax, context.clamp,
                          dict(context.traps), dict(context.flags))
    item = api.FinancialReconciliation("wear_cost", D(1), D(".9"), D(".1"), "difference", ())
    for changes in ({"metric": "unknown"}, {"status": "complete"}, {"status": "equal"},
                    {"difference": D(0)}, {"day_sum": None}, {"authoritative_total": None},
                    {"reasons": ("unexpected",)}, {"status": "unknown"},
                    {"status": "unknown", "day_sum": None, "difference": None}):
        with pytest.raises((TypeError, ValueError)):
            replace(item, **changes)
    with pytest.raises((TypeError, ValueError)):
        api._reconcile_days("invented", D(0), (D(0),))
    reasons = ["synthetic:unknown", "synthetic:unknown"]
    unknown = api.FinancialReconciliation("total_cost", None, None, None, "unknown", reasons)
    reasons.clear()
    assert unknown.reasons == ("synthetic:unknown",)
    assert not hasattr(item, "__dict__")
    for field in fields(item):
        assert inspect.signature(type(item)).parameters[field.name].default is inspect.Parameter.empty
        with pytest.raises(FrozenInstanceError):
            setattr(item, field.name, None)
        with pytest.raises(TypeError):
            api.FinancialReconciliation(**{f.name: getattr(item, f.name) for f in fields(item) if f != field})
    for result in (expected, expected_comparison):
        old = type(result)(*(getattr(result, field.name) for field in fields(result) if field.name != "reconciliation"))
        assert old.reconciliation == ()  # Not computed, never an equal diagnosis.
        entries = list(result.reconciliation)
        detached = replace(result, reconciliation=entries)
        entries.clear()
        assert detached.reconciliation == result.reconciliation


@pytest.mark.parametrize("amount", [D(0), D("1E-37")])
@pytest.mark.parametrize("fault", ["drop", "duplicate", "order", "meter", "revision", "hash", "snapshot",
                                  "window", "coverage", "opening", "chain", "end-state",
                                  "adjustment-duplicate", "adjustment-orphan", "adjustment-missing"])
def test_f3b3_fix2_structure_not_residual(fault, amount, monkeypatch):
    api = result_api()
    projection, replay, totals = fix2_book_case(api)
    calls = []

    def corrupt(items, snapshots, opening):
        result = settle_series(items, snapshots, opening)
        calls.append(result)
        lines = tuple(replace(line, import_cost=amount) for line in result.lines)
        if fault == "drop":
            lines = lines[1:]
        elif fault == "duplicate":
            lines = (lines[0], *lines)
        elif fault == "order":
            lines = tuple(reversed(lines))
        elif fault in ("meter", "revision", "hash", "snapshot", "window", "opening", "chain"):
            changes = {"meter": {"meter_id": "synthetic:wrong"}, "revision": {"input_revision": 1},
                "hash": {"input_hash": "0" * 64}, "snapshot": {"snapshot_hash": "0" * 64},
                "window": {"end_ts": lines[0].end_ts - 1},
                "opening": {"before_state": replace(lines[0].before_state, asof_ts=lines[0].start_ts - 1)},
                "chain": {"after_state": replace(lines[0].after_state, month=None)}}[fault]
            lines = (replace(lines[0], **changes), *lines[1:])
        result = replace(result, lines=lines)
        if fault == "coverage":
            result = replace(result, coverage_end_ts=result.coverage_end_ts - 1)
        elif fault == "end-state":
            result = replace(result, end_state=replace(result.end_state, asof_ts=result.coverage_end_ts - 1))
        elif fault == "adjustment-duplicate":
            result = replace(result, adjustments=result.adjustments * 2)
        elif fault == "adjustment-orphan":
            result = replace(result, adjustments=(replace(result.adjustments[0], period_start_ts=START_TS + 1),))
        elif fault == "adjustment-missing":
            result = replace(result, adjustments=())
        return result

    monkeypatch.setattr(api, "settle_series", corrupt)
    with pytest.raises(ValueError, match="^replay_reconciliation_structure_invalid$"):
        api.settle_replay(projection, replay, totals)
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["valid", "index-missing", "index-duplicate", "index-wrong", "index-order", "key",
                                  "start", "end", "day-missing", "day-duplicate", "credit-length", "wear-length", "credit-slot"])
def test_f3b3_fix2_structure_not_residual_day_groups(fault, monkeypatch):
    api = reconciliation_api()
    projection, replay, totals = fix2_book_case(api)
    seen, produced = [], []
    validate, component_producer = api._validate_day_structure, api._cost_components

    def observe(*args):
        seen.append(args)
        return validate(*args)

    def observe_components(lines, credits, **kwargs):
        produced.append((lines, credits, kwargs["wear"], kwargs["terminal"]))
        return component_producer(lines, credits, **kwargs)

    monkeypatch.setattr(api, "_validate_day_structure", observe)
    monkeypatch.setattr(api, "_cost_components", observe_components)
    result = api.settle_replay(projection, replay, totals)
    assert len(seen) == 1
    arguments = list(seen[0])
    assert arguments[:3] == [projection, result.settlement, replay]
    credits, wear, terminal, groups, days = arguments[3:]
    assert tuple(indices for _, indices in groups) == ((0,), (1, 2))
    assert len(produced) == 3
    for (_, indices), (lines, selected_credits, selected_wear, selected_terminal) in zip(groups, produced[1:], strict=True):
        assert lines == tuple(result.settlement.lines[index] for index in indices)
        assert selected_credits == tuple(credits[index] for index in indices)
        assert Fraction(selected_wear.value) == sum((Fraction(wear[index].value) for index in indices), Fraction(0))
        assert selected_terminal == (terminal if indices[-1] == 2 else api.FinancialComponent(D(0), D(0), "complete", ()))
    if fault == "valid":
        validate(*arguments)
        assert_result_diagnostics(result)
        return
    if fault.startswith("index-"):
        variants = {"index-missing": ((groups[0][0], ()), groups[1]),
                    "index-duplicate": ((groups[0][0], (0, 0)), groups[1]),
                    "index-wrong": ((groups[0][0], (1,)), groups[1]),
                    "index-order": (groups[0], (groups[1][0], (2, 1)))}
        arguments[6] = variants[fault]
    elif fault == "key":
        arguments[6] = ((replace(groups[0][0], local_date="2027-01-30"), groups[0][1]), groups[1])
    elif fault in ("start", "end"):
        arguments[7] = (replace(days[0], **{"start_ts": days[0].start_ts + 1} if fault == "start"
                               else {"end_ts": days[0].end_ts - 1}), days[1])
    elif fault == "day-missing":
        arguments[7] = days[1:]
    elif fault == "day-duplicate":
        arguments[7] = (days[0], days[0])
    elif fault == "credit-length":
        arguments[3] = credits[:-1]
    elif fault == "wear-length":
        arguments[4] = wear[:-1]
    else:
        arguments[3] = (D(0), credits[0], credits[2])
    with pytest.raises(ValueError, match="^replay_reconciliation_structure_invalid$"):
        validate(*arguments)


@pytest.mark.parametrize("fault", ["month_credit", "terminal_value", "wear_cost", "tiny-wear"])
def test_f3b3_fix2_structure_not_residual_misallocated_components(fault, monkeypatch):
    api = result_api()
    projection, replay, totals = fix2_book_case(api)
    control = api.settle_replay(projection, replay, totals)
    original = api._cost_components
    injected = []

    def misallocate(lines, credits, **kwargs):
        result = original(lines, credits, **kwargs)
        if len(lines) == len(projection.items):
            return result
        day_index = 0 if lines[0].start_ts == projection.start_ts else 1
        if fault == "tiny-wear":
            if day_index:
                return result
            value = D("0.0037500000000000000000000000000000001")
            assert Fraction(value) - Fraction(result.wear_cost.value) == Fraction(D("1E-37"))
            injected.append(value)
            return replace(result, wear_cost=api.FinancialComponent(value, value, "complete", ()))
        other = getattr(control.days[1 - day_index].components, fault)
        injected.append(other.value)
        return replace(result, **{fault: other})

    monkeypatch.setattr(api, "_cost_components", misallocate)
    with pytest.raises(ValueError, match="^replay_reconciliation_structure_invalid$"):
        api.settle_replay(projection, replay, totals)
    if fault != "tiny-wear":
        assert sum((Fraction(value) for value in injected), Fraction(0)) == sum(
            (Fraction(getattr(day.components, fault).value) for day in control.days), Fraction(0))


@pytest.mark.parametrize("case", ["partial-bonus", "missing-price", "missing-dc", "missing-soc", "binding", "month-prefix"])
def test_f3b3_fix2_unknown_coverage_and_booking_controls(case, bonus_rules, monkeypatch):
    api = result_api()
    plan, replay = fixtures.finance_case()
    options, segments = {}, (physical_segment(imported=1.0, exported=2.0),)
    if case == "partial-bonus":
        rules = replace(bonus_rules, windows=tuple((month, (0, 24)) for month in range(1, 13)))
        snapshot = fixtures.canonical_snapshot(replay.snapshots[0], bonus=rules,
            cost_supplier_consumption=((date(2020, 1, 1), D(".2")),))
        plan, replay = fixtures.finance_case(snapshots=(snapshot,),
            opening=SettlementState(NOW, BonusState(START_TS, YEAR_END_TS, NOW, D(0)), None))
        options["kind"] = "missing"
    elif case == "missing-price":
        plan, replay = fixtures.finance_case(spots=())
    elif case == "missing-dc":
        segments = (physical_segment(batteries=(physical_battery(dc=None),)),)
    elif case == "binding":
        options["kind"] = "mismatch"
    elif case == "month-prefix":
        snapshot = fixtures.canonical_snapshot(replay.snapshots[0], monthly_export_floor="spot_zero")
        plan, replay = fixtures.finance_case(edges=(MONTH_END_TS - 900, MONTH_END_TS), snapshots=(snapshot,))
        segments = (physical_segment(MONTH_END_TS - 900, MONTH_END_TS),)
    projection = project_case(api, plan, replay, segments, **options)
    totals = idle_totals(api)
    if case == "missing-soc":
        totals = api.ReplayPhysicalTotals((replace(totals.batteries[0], soc_end=None),))
    before, raw = deepcopy((projection, replay, totals)), replay.canonical_bytes()
    calls = []

    def observe(items, snapshots, opening):
        calls.append((items, snapshots, opening))
        return settle_series(items, snapshots, opening)

    monkeypatch.setattr(api, "settle_series", observe)
    result = api.settle_replay(projection, replay, totals)
    assert len(calls) == (0 if case == "binding" else 1)
    assert result.financial_status == ("unknown" if case == "binding" else "partial")
    assert_result_diagnostics(result)
    assert result.components.total_cost.value is None
    assert result.reconciliation[-1].status == "unknown"
    assert "reconciliation_total_unknown" in result.reconciliation[-1].reasons
    reason = {"partial-bonus": "unknown_bonus_source", "missing-price": "missing_spot",
              "missing-dc": "replay_dc_charge_unknown: accu", "missing-soc": "replay_soc_end_unknown: accu",
              "binding": "source_binding_mismatch", "month-prefix": "replay_month_credit_unknown"}[case]
    assert reason in result.financial_reasons and reason in result.reconciliation[-1].reasons
    if case == "partial-bonus":
        assert result.components.import_cost.value == D(".30") and result.components.base_export_revenue.value == D(".20")
        assert result.components.invoice_energy_cost.known_subtotal == D(".10")
        assert result.reconciliation[0].status == result.reconciliation[1].status == "equal"
        for index in (2, 4, 7):
            assert "unknown_bonus_source" in result.reconciliation[index].reasons
            assert "reconciliation_day_unknown" in result.reconciliation[index].reasons
    elif case == "month-prefix":
        assert result.settlement.adjustments == () and result.period_status == "unknown"
        assert result.reconciliation[3].status == "unknown"
    other = api.settle_replay(projection, replay, totals)
    assert len(calls) == (0 if case == "binding" else 2)
    comparison = api.compare_financial_results(result, other)
    assert comparison.advantage_eur is None
    assert_comparison_diagnostics(comparison)
    assert reason in comparison.reasons and reason in comparison.reconciliation[0].reasons
    assert len(calls) == (0 if case == "binding" else 2)
    assert (projection, replay, totals) == before and replay.canonical_bytes() == raw
    if case != "binding":
        assert all(snapshots is replay.snapshots and opening is replay.opening for _, snapshots, opening in calls)


@pytest.mark.parametrize("fault", ["missing-day", "key", "window-gap", "window-different", "empty", "empty-both",
                                  "projection", "settlement", "duplicate-key", "overlap"])
def test_f3b3_fix2_unknown_coverage_and_booking_controls_consumer(fault):
    api = result_api()
    projection, replay, totals = fix2_book_case(api)
    result = api.settle_replay(projection, replay, totals)
    first, last = result.days
    if fault == "missing-day":
        broken = replace(result, days=(last,))
    elif fault == "key":
        broken = replace(result, days=(replace(first, key=replace(first.key, contract_id="synthetic:wrong")), last))
    elif fault == "window-gap":
        broken = replace(result, days=(replace(first, end_ts=first.end_ts - 1), last))
    elif fault == "window-different":
        broken = replace(result, days=(replace(first, end_ts=first.end_ts - 1), replace(last, start_ts=last.start_ts - 1)))
    elif fault in ("empty", "empty-both"):
        broken = replace(result, days=())
        if fault == "empty-both":
            result = broken
    elif fault in ("projection", "settlement"):
        broken = replace(result, **{fault: None})
    elif fault == "duplicate-key":
        broken = replace(result, days=(first, replace(last, key=first.key)))
    else:
        broken = replace(result, days=(replace(first, end_ts=first.end_ts + 1), last))
    if fault in ("duplicate-key", "overlap"):
        with pytest.raises(ValueError, match="^replay_reconciliation_structure_invalid$"):
            api.compare_financial_results(result, broken)
        return
    comparison = api.compare_financial_results(result, broken)
    assert comparison.financial_status == "unknown" and comparison.advantage_eur is None
    assert "replay_comparison_coverage_mismatch" in comparison.reasons
    assert_comparison_diagnostics(comparison)
    assert "reconciliation_coverage_unknown" in comparison.reconciliation[0].reasons
