"""Pure replay input: explicit assumptions and detached, uncoerced evidence."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from importlib import import_module, util
from inspect import Parameter, signature
from typing import get_args, get_type_hints

import pytest

from dao.prog.fastctrl.command import (
    ActuationReceipt,
    BarrierReceipt,
    BatteryCommand,
    ClockSample,
    Deadline,
    OwnerBinding,
    RevokeRequest,
    plan_digest,
)
from dao.prog.fastctrl.policy import (
    BatteryDecision,
    ControllerState,
    PolicyLimits,
    apply_execution_outcome,
)
from dao.prog.fastctrl.receiver import payload_digest, receipt_matches_command
from dao.tests.prog.test_fast_actuator import proposal
from dao.tests.prog.test_fast_command import command, other_dtos, receipt
from dao.tests.prog.test_fast_policy import T0, make_plan, make_spec

POINTS = ((T0 + .0006, 400.0), (T0 + 60.0006, -200.0))


def contract():
    # The first RED must be an assertion, never an import/collection failure.
    assert util.find_spec("dao.prog.fastctrl.replay_execution") is not None, "F3a1 replay input API missing"
    return import_module("dao.prog.fastctrl.replay_execution")


def replay_plan(names=("accu",)):
    return make_plan([.3], [[0] * len(names)], specs=[make_spec(name=name) for name in names])


def physics(api, **changes):
    return api.ReplayPhysics(**(dict(version="battery_model_v1", house_hold_s=60.0,
                                    stop_response="unknown") | changes))


def modeled(api, **changes):
    return api.ModeledExecution(**(dict(initial_execution="stopped", default_effect="complete") | changes))


def recorded(api, **changes):
    return api.RecordedExecution(**(dict(
        installation_id="synthetic", receiver_generation="generation-a",
        opening_binding=None, opening_acceptances=(), opening_stop=None, coverage=(), events=(),
    ) | changes))


def scenario(api, execution=None, **physics_changes):
    return api.ReplayScenario("explicit", modeled(api) if execution is None else execution,
                              physics(api, **physics_changes))


def validate(api, **changes):
    return api.validate_replay_input(**(dict(
        plan=replay_plan(), sample_points=POINTS, soc_start=40.0,
        inverter_efficiency=.95, scenario=scenario(api),
    ) | changes))


def examples(api):
    request, barrier, deadline, clock, binding = other_dtos()
    timing = api.ReplayAttemptTiming(T0, clock, deadline, deadline)
    application = api.ReplayApplication(T0 + 1, 2, clock, "accepted")
    observation = api.ReceiptEvent(T0 + 1, 1, receipt(), clock, application)
    events = (
        api.ProposalEvent(T0, 0, proposal(), command(), timing), observation,
        api.StopRequestEvent(T0 + 2, 0, request), api.BarrierEvent(T0 + 2, 1, barrier),
        api.BindingEvent(T0 + 2, 2, binding),
    )
    opening = api.OpeningAcceptance(command(), None,
                                    api.ReceiptEvent(T0 - 1, 0, receipt(), None, None), None)
    coverage = api.ExecutionCoverage("accu", *[point[0] for point in POINTS])
    execution = recorded(api, opening_binding=binding, opening_acceptances=(opening,),
                         opening_stop=(request, barrier), coverage=(coverage,), events=events)
    response = api.ModelResponse(0, "plan", "accu", "partial", .1)
    stop = api.ModelStop(T0 + 2, 0, True, False)
    bind = api.ModelBind(T0 + 2, 1)
    model = modeled(api, responses=(response,), lifecycle=(stop, bind))
    case = scenario(api, execution)
    result = validate(api, scenario=case)
    return {type(item).__name__: item for item in (
        physics(api), timing, application, *events, opening, coverage, execution,
        response, stop, bind, model, case, result,
    )}


def test_f3a1_contract_is_explicit_and_detached():
    api = contract()
    expected = {
        "ReplayPhysics": ("version house_hold_s stop_response equal_initial_soc", {"equal_initial_soc": False}),
        "ReplayAttemptTiming": ("anchor_ts anchor_clock receive_deadline apply_deadline", {}),
        "ReplayApplication": ("at_ts order clock outcome", {}),
        "ProposalEvent": ("at_ts order decision command timing", {}),
        "ReceiptEvent": ("at_ts order receipt received_clock application", {}),
        "StopRequestEvent": ("at_ts order request", {}),
        "BarrierEvent": ("at_ts order receipt", {}),
        "BindingEvent": ("at_ts order binding", {}),
        "OpeningAcceptance": ("command timing observation sent_at_ts", {}),
        "ExecutionCoverage": ("battery_name start_ts end_ts", {}),
        "RecordedExecution": ("installation_id receiver_generation opening_binding opening_acceptances "
                              "opening_stop coverage events", {}),
        "ModelResponse": ("tick_index source battery_name effect delay_s", {}),
        "ModelStop": ("at_ts order queues_fenced safe_state_confirmed", {}),
        "ModelBind": ("at_ts order", {}),
        "ModeledExecution": ("initial_execution default_effect response_delay_s prepare_delay_s "
                             "application_delay_s responses lifecycle", {
                                 "response_delay_s": 0, "prepare_delay_s": 0, "application_delay_s": 0,
                                 "responses": (), "lifecycle": (),
                             }),
        "ReplayScenario": ("scenario_id execution physics", {}),
        "ReplayInput": ("scenario sample_points soc_by_battery inverter_efficiency_by_battery reasons", {}),
    }
    values = examples(api)
    for name, (names, defaults) in expected.items():
        parameters = signature(getattr(api, name)).parameters
        assert tuple(parameters) == tuple(names.split())
        assert {key: p.default for key, p in parameters.items() if p.default is not Parameter.empty} == defaults
        value = values[name]
        with pytest.raises(FrozenInstanceError):
            setattr(value, fields(value)[0].name, None)
    assert set(get_args(api.ReplayEvent)) == {
        api.ProposalEvent, api.ReceiptEvent, api.StopRequestEvent, api.BarrierEvent, api.BindingEvent,
    }
    assert (api.Reasons, api.SamplePoints, api.SocByBattery, api.EfficiencyByBattery, api.Assumptions) == (
        tuple[str, ...], tuple[tuple[float, float], ...], tuple[tuple[str, float | None], ...],
        tuple[tuple[str, float], ...], tuple[tuple[str, str], ...],
    )
    assert get_type_hints(api.ProposalEvent)["decision"] is BatteryDecision
    assert get_type_hints(api.OpeningAcceptance)["command"] is BatteryCommand
    assert get_type_hints(api.ReceiptEvent)["receipt"] is ActuationReceipt
    assert get_type_hints(api.ReplayAttemptTiming)["anchor_clock"] is ClockSample
    assert get_type_hints(api.ReplayAttemptTiming)["receive_deadline"] is Deadline
    assert all(p.kind is Parameter.KEYWORD_ONLY for p in signature(api.validate_replay_input).parameters.values())
    with pytest.raises(TypeError):
        api.ModeledExecution(initial_execution="stopped")
    with pytest.raises(TypeError):
        api.ReplayScenario("implicit-physics", modeled(api))
    for forbidden in ("callback", "checkpoint", "policy_state"):
        with pytest.raises(TypeError):
            api.ReplayScenario("extra", modeled(api), physics(api), **{forbidden: object()})

    decision = proposal()
    event = api.ProposalEvent(T0 + 1, 0, decision, None, None)
    events = [event]
    execution = recorded(api, events=events)
    case = scenario(api, execution)
    points = [list(point) for point in POINTS]
    socs, efficiencies = {"accu": 37.0}, {"accu": .91}
    plan = replay_plan()
    before = deepcopy(plan)
    result = validate(api, plan=plan, scenario=case, sample_points=points,
                      soc_start=socs, inverter_efficiency=efficiencies)
    assert plan == before and decision.setpoint_w == -1200
    decision.setpoint_w = 101
    event.decision.setpoint_w = 102
    execution.events[0].decision.setpoint_w = 103
    case.execution.events[0].decision.setpoint_w = 104
    events.clear()
    points[0][0] = -1
    socs["accu"], efficiencies["accu"] = 88, .2
    assert result.scenario.execution.events[0].decision.setpoint_w == -1200
    assert result.sample_points == POINTS
    assert result.soc_by_battery == (("accu", 37.0),)
    assert result.inverter_efficiency_by_battery == (("accu", .91),)
    assert result.scenario.execution.events[0].command is None
    assert result.scenario.execution.events[0].timing is None


@pytest.mark.parametrize("field,value,code", [
    ("sample_points", ((T0, 1), (T0, 2)), "replay_time_order_invalid"),
    ("sample_points", ((T0 + 1, 1), (T0, 2)), "replay_time_order_invalid"),
    ("sample_points", ((float("nan"), 1), (T0, 2)), "replay_value_invalid"),
    ("sample_points", ((T0, 1), (float("inf"), 2)), "replay_value_invalid"),
    ("sample_points", ((T0, float("nan")), (T0 + 1, 2)), "replay_value_invalid"),
    ("sample_points", ((T0, 1), (T0 + 1, float("inf"))), "replay_value_invalid"),
    ("sample_points", ((T0, 1, 2), (T0 + 1, 2)), "replay_value_invalid"),
    ("sample_points", (), "te weinig meetpunten om te simuleren"),
    ("sample_points", ((T0, 1),), "te weinig meetpunten om te simuleren"),
    ("soc_start", float("nan"), "replay_value_invalid"),
    ("soc_start", float("inf"), "replay_value_invalid"),
    ("soc_start", -1, "replay_value_invalid"),
    ("soc_start", 101, "replay_value_invalid"),
    ("soc_start", True, "replay_value_invalid"),
    ("soc_start", "50", "replay_value_invalid"),
    ("soc_start", {"accu": float("nan")}, "replay_value_invalid"),
    ("soc_start", {"other": 50}, "replay_battery_mapping_invalid"),
    ("inverter_efficiency", 0, "replay_value_invalid"),
    ("inverter_efficiency", -.1, "replay_value_invalid"),
    ("inverter_efficiency", 1.01, "replay_value_invalid"),
    ("inverter_efficiency", float("inf"), "replay_value_invalid"),
    ("inverter_efficiency", float("nan"), "replay_value_invalid"),
    ("inverter_efficiency", True, "replay_value_invalid"),
    ("inverter_efficiency", {}, "replay_battery_mapping_invalid"),
    ("inverter_efficiency", {"other": .9}, "replay_battery_mapping_invalid"),
    ("inverter_efficiency", {"accu": None}, "replay_value_invalid"),
])
def test_f3a1_validation_distinguishes_invalid_from_missing(field, value, code):
    api = contract()
    with pytest.raises((TypeError, ValueError), match=f"^{code}$"):
        validate(api, **{field: value})


@pytest.mark.parametrize("kind,soc,equal,expected,reasons", [
    ("model", {"accu": 23, "second": 79}, False, (23, 79), ()),
    ("model", {"second": 79, "accu": 23}, False, (23, 79), ()),
    ("model", {"accu": 23}, False, (23, None), ("replay_soc_missing",)),
    ("model", {"accu": None}, False, (None, None), ("replay_soc_missing",)),
    ("model", 37, False, (None, None), ("replay_soc_by_battery_required", "replay_soc_missing")),
    ("model", 37, True, (37, 37), ()),
    ("recorded", 37, True, (None, None),
     ("execution_history_missing", "replay_soc_by_battery_required", "replay_soc_missing")),
    ("missing", {"accu": 23}, False, (23, None), ("execution_input_missing", "replay_soc_missing")),
    ("no-effect", {"accu": 23, "second": 79}, False, (23, 79), ("execution_model_missing",)),
])
def test_f3a1_missing_evidence_retains_individual_soc(kind, soc, equal, expected, reasons):
    api = contract()
    execution = recorded(api) if kind == "recorded" else modeled(
        api, default_effect=None if kind == "no-effect" else "complete")
    case = None if kind == "missing" else scenario(api, execution, equal_initial_soc=equal)
    result = validate(api, plan=replay_plan(("accu", "second")), scenario=case, soc_start=soc)
    assert result.soc_by_battery == (("accu", expected[0]), ("second", expected[1]))
    assert result.inverter_efficiency_by_battery == (("accu", .95), ("second", .95))
    assert result.reasons == reasons


def test_f3a1_battery_names_and_mappings_remain_in_plan_order():
    api = contract()
    plan = replay_plan(("accu", "second"))
    result = validate(api, plan=plan, soc_start={"second": 100, "accu": 0},
                      inverter_efficiency={"second": .87, "accu": 1})
    assert result.soc_by_battery == (("accu", 0), ("second", 100))
    assert result.inverter_efficiency_by_battery == (("accu", 1), ("second", .87))
    plan.specs[1].name = "accu"
    with pytest.raises(ValueError, match="^replay_duplicate_identity$"):
        validate(api, plan=plan)
    plan.specs.clear()
    with pytest.raises(ValueError, match="^het plan bevat geen batterij$"):
        validate(api, plan=plan)


@pytest.mark.parametrize("name,field,value,code", [
    ("ReplayPhysics", "version", "battery_model_v2", "replay_value_invalid"),
    ("ReplayPhysics", "house_hold_s", 0, "replay_value_invalid"),
    ("ReplayPhysics", "house_hold_s", float("inf"), "replay_value_invalid"),
    ("ReplayPhysics", "stop_response", None, "replay_value_invalid"),
    ("ReplayPhysics", "equal_initial_soc", 1, "replay_value_invalid"),
    ("ReplayScenario", "scenario_id", "  ", "replay_value_invalid"),
    ("ReplayScenario", "physics", None, "replay_value_invalid"),
    ("ReplayScenario", "execution", lambda: None, "replay_value_invalid"),
    ("ReplayAttemptTiming", "anchor_ts", float("nan"), "replay_value_invalid"),
    ("ReplayAttemptTiming", "anchor_clock", {}, "replay_value_invalid"),
    ("ReplayAttemptTiming", "receive_deadline", None, "replay_value_invalid"),
    ("ReplayAttemptTiming", "apply_deadline", 12, "replay_value_invalid"),
    ("ReplayApplication", "outcome", "rejected", "replay_value_invalid"),
    ("ReplayApplication", "clock", None, "replay_value_invalid"),
    ("ProposalEvent", "decision", {}, "replay_value_invalid"),
    ("ProposalEvent", "command", {}, "replay_value_invalid"),
    ("ProposalEvent", "timing", 1, "replay_value_invalid"),
    ("ReceiptEvent", "receipt", {}, "replay_value_invalid"),
    ("ReceiptEvent", "received_clock", {}, "replay_value_invalid"),
    ("ReceiptEvent", "application", {}, "replay_value_invalid"),
    ("StopRequestEvent", "request", {}, "replay_value_invalid"),
    ("BarrierEvent", "receipt", {}, "replay_value_invalid"),
    ("BindingEvent", "binding", {}, "replay_value_invalid"),
    ("OpeningAcceptance", "sent_at_ts", float("inf"), "replay_value_invalid"),
    ("OpeningAcceptance", "observation", {}, "replay_value_invalid"),
    ("ExecutionCoverage", "end_ts", T0, "replay_window_invalid"),
    ("RecordedExecution", "events", (object(),), "replay_value_invalid"),
    ("RecordedExecution", "opening_stop", (None, None), "replay_value_invalid"),
    ("RecordedExecution", "opening_binding", {}, "replay_value_invalid"),
    ("RecordedExecution", "installation_id", "", "replay_value_invalid"),
    ("RecordedExecution", "receiver_generation", "", "replay_value_invalid"),
    ("ModelResponse", "tick_index", -1, "replay_model_selector_invalid"),
    ("ModelResponse", "tick_index", True, "replay_model_selector_invalid"),
    ("ModelResponse", "source", "other", "replay_model_selector_invalid"),
    ("ModelResponse", "effect", None, "replay_value_invalid"),
    ("ModelResponse", "delay_s", -.01, "replay_value_invalid"),
    ("ModelStop", "queues_fenced", 1, "replay_value_invalid"),
    ("ModelStop", "safe_state_confirmed", None, "replay_value_invalid"),
    ("ModelBind", "at_ts", float("inf"), "replay_value_invalid"),
    ("ModelBind", "order", True, "replay_event_order_invalid"),
    ("ModelBind", "order", .5, "replay_event_order_invalid"),
    ("ModeledExecution", "initial_execution", "accepted", "replay_value_invalid"),
    ("ModeledExecution", "default_effect", "accepted", "replay_value_invalid"),
    ("ModeledExecution", "response_delay_s", float("nan"), "replay_value_invalid"),
    ("ModeledExecution", "prepare_delay_s", -.1, "replay_value_invalid"),
    ("ModeledExecution", "application_delay_s", float("inf"), "replay_value_invalid"),
    ("ModeledExecution", "lifecycle", (receipt(),), "replay_value_invalid"),
    ("ReplayInput", "soc_by_battery", (("accu", -1),), "replay_value_invalid"),
    ("ReplayInput", "inverter_efficiency_by_battery", (("accu", 0),), "replay_value_invalid"),
])
def test_f3a1_typed_constructors_reject_structural_errors(name, field, value, code):
    api = contract()
    with pytest.raises((TypeError, ValueError), match=f"^{code}$"):
        replace(examples(api)[name], **{field: value})


@pytest.mark.parametrize("case", ["duplicate", "reversed", "application-collision", "application-before",
                                  "application-same-key", "two-applications", "opening-collision"])
def test_f3a1_event_and_application_order_is_one_unique_space(case):
    api = contract()
    clock = other_dtos()[3]
    observed = api.ReceiptEvent(T0 + 1, 1, receipt(), None, None)
    later = api.BindingEvent(T0 + 1, 2, other_dtos()[4])
    with pytest.raises(ValueError, match="^replay_event_order_invalid$"):
        if case == "duplicate":
            recorded(api, events=(observed, observed))
        elif case == "reversed":
            recorded(api, events=(later, observed))
        elif case == "application-before":
            replace(observed, application=api.ReplayApplication(T0 + .9, 9, clock, "unknown"))
        elif case == "application-same-key":
            replace(observed, application=api.ReplayApplication(T0 + 1, 1, clock, "unknown"))
        elif case == "opening-collision":
            recorded(api, opening_acceptances=(api.OpeningAcceptance(command(), None, observed, None),),
                     events=(observed,))
        else:
            application = api.ReplayApplication(T0 + 1, 2 if case == "application-collision" else 3,
                                                clock, "accepted")
            first = replace(observed, application=application)
            second = later if case == "application-collision" else replace(
                observed, order=2, application=application)
            recorded(api, events=(first, second))


def test_f3a1_same_float_order_and_missing_clocks_stay_explicit():
    api = contract()
    clock = other_dtos()[3]
    observed = api.ReceiptEvent(T0 + .0006, 1, receipt(), None,
                               api.ReplayApplication(T0 + .0006, 3, clock, "unknown"))
    barrier = api.BarrierEvent(T0 + .0006, 2, other_dtos()[1])
    execution = recorded(api, events=(observed, barrier), opening_acceptances=(
        api.OpeningAcceptance(command(), None, replace(observed, at_ts=T0 - 1, application=None), None),))
    result = validate(api, scenario=scenario(api, execution))
    stored = result.scenario.execution
    assert [(e.at_ts, e.order) for e in stored.events] == [(T0 + .0006, 1), (T0 + .0006, 2)]
    assert stored.events[0].received_clock is None
    assert stored.events[0].application.order == 3
    assert stored.opening_acceptances[0].timing is None
    assert stored.opening_acceptances[0].sent_at_ts is None
    # Unhealthy/wrong clock, command identity and deadline evidence belong to F3a2.
    assert replace(observed, received_clock=replace(clock, healthy=False)).received_clock.healthy is False


@pytest.mark.parametrize("case,code", [
    ("overlap", "replay_window_invalid"), ("before", "replay_window_invalid"),
    ("after", "replay_window_invalid"), ("battery", "replay_battery_mapping_invalid"),
    ("selector-duplicate", "replay_model_selector_invalid"),
    ("selector-battery", "replay_model_selector_invalid"),
    ("lifecycle-reversed", "replay_event_order_invalid"),
    ("lifecycle-duplicate", "replay_event_order_invalid"),
])
def test_f3a1_coverage_and_model_selectors_are_structural(case, code):
    api = contract()
    start, end = POINTS[0][0], POINTS[-1][0]
    with pytest.raises(ValueError, match=f"^{code}$"):
        if case.startswith("selector"):
            response = api.ModelResponse(0, "fast", "other" if case == "selector-battery" else "accu",
                                         "complete", 0)
            model = modeled(api, responses=(response, response) if case == "selector-duplicate" else (response,))
            validate(api, scenario=scenario(api, model))
        elif case.startswith("lifecycle"):
            stop, bind = api.ModelStop(start, 1, True, True), api.ModelBind(start, 2)
            modeled(api, lifecycle=(bind, stop) if case == "lifecycle-reversed" else (stop, stop))
        else:
            coverage = (api.ExecutionCoverage("other" if case == "battery" else "accu",
                                              start - 1 if case == "before" else start,
                                              end + 1 if case == "after" else end),)
            if case == "overlap":
                coverage += (api.ExecutionCoverage("accu", start + 1, end),)
            validate(api, scenario=scenario(api, recorded(api, coverage=coverage)))


def test_f3a1_coverage_can_be_adjacent_or_shared_across_batteries():
    api = contract()
    start, end = POINTS[0][0], POINTS[-1][0]
    coverage = (api.ExecutionCoverage("accu", start, start + 30),
                api.ExecutionCoverage("second", start, end),
                api.ExecutionCoverage("accu", start + 30, end))
    response = api.ModelResponse(999, "fast", "accu", "none", 0)
    assert validate(api, scenario=scenario(api, modeled(api, responses=(response,)))).reasons == ()
    result = validate(api, plan=replay_plan(("accu", "second")), soc_start={"accu": 40, "second": 60},
                      scenario=scenario(api, recorded(api, coverage=coverage)))
    assert result.scenario.execution.coverage == coverage


def test_f3a1_wire_rounding_keeps_original_float_points():
    api = contract()
    result = validate(api)
    assert api.wire_ms(T0 + .0006) == T0 * 1000 + 1
    assert api.wire_ms(.0005) == 0
    assert api.wire_ms(.0015) == 2
    assert result.sample_points == POINTS
    assert result.sample_points[0][0] != api.wire_ms(POINTS[0][0]) / 1000


def execution_api():
    api = contract()
    assert hasattr(api, "ExecutionReplay"), "F3a2 recorded execution fold missing"
    return api


def replay_clock(at_ts, *, anchor=T0):
    return ClockSample(int(round(at_ts * 1000)), 10_000_000_000 + round((at_ts - anchor) * 1e9),
                       "replay-boot", 0, True)


def replay_timing(api, cmd, *, anchor=T0, receive=.75, apply=1):
    clock = replay_clock(anchor, anchor=anchor)
    return api.ReplayAttemptTiming(anchor, clock,
                                   Deadline(clock.boottime_ns + round(receive * 1e9), cmd.execute_before_ms),
                                   Deadline(clock.boottime_ns + round(apply * 1e9), cmd.execute_before_ms))


def replay_command(plan, **changes):
    start = plan.intervals[0].start_ts
    return command(**(dict(
        installation_id=plan.financial_context.basis.meter_id if plan.financial_context else "synthetic",
        plan_digest=plan_digest(plan), issued_at_ms=start * 1000,
        execute_before_ms=(start + 2) * 1000, hold_until_ms=(start + 8) * 1000,
        setpoint_w=1000,
    ) | changes))


def replay_receipt(api, cmd, at_ts, *, apply_ts=None, order=1, apply_order=2,
                   status="accepted", effect="complete", outcome="accepted", accepted_ts=None,
                   anchor=T0):
    raw = receipt(
        installation_id=cmd.installation_id, battery_name=cmd.battery_name,
        owner_session_id=cmd.owner_session_id, owner_epoch=cmd.owner_epoch, sequence=cmd.sequence,
        command_id=cmd.command_id, plan_digest=cmd.plan_digest, payload_digest=payload_digest(cmd),
        accepted_ts=(cmd.issued_at_ms if accepted_ts is None else accepted_ts) if status == "accepted" else None,
        observed_at_ms=int(round(at_ts * 1000)), status=status, effect=effect,
    )
    application = None if apply_ts is None else api.ReplayApplication(
        apply_ts, apply_order, replay_clock(apply_ts, anchor=anchor), outcome)
    return api.ReceiptEvent(at_ts, order, raw, replay_clock(at_ts, anchor=anchor), application)


def send_event(api, cmd, at_ts, timing, *, order=0, **decision_changes):
    return api.ProposalEvent(at_ts, order, proposal(**(dict(
        name=cmd.battery_name, setpoint_w=cmd.setpoint_w, stop_inverter=None,
    ) | decision_changes)), cmd, timing)


def recorded_run(api, *, plan=None, events=(), openings=(), start=T0, end=T0 + 10,
                 coverage=None, stop_response="unknown", controller=None, **execution_changes):
    plan = replay_plan() if plan is None else plan
    cmd = replay_command(plan)
    binding = OwnerBinding(cmd.installation_id, "receiver-a", "generation-a", "session-a", 2,
                           "opening-barrier", int((end + 10) * 1000))
    if coverage is None:
        coverage = tuple(api.ExecutionCoverage(spec.name, start, end) for spec in plan.specs)
    execution = recorded(api, **(dict(
        installation_id=cmd.installation_id, opening_binding=binding,
        events=events, opening_acceptances=openings, coverage=coverage,
    ) | execution_changes))
    controller = ControllerState() if controller is None else controller
    case = scenario(api, execution, stop_response=stop_response)
    before = deepcopy((plan, case))
    run = api.ExecutionReplay(plan=plan, scenario=case, controller=controller,
                              start_ts=start, end_ts=end)
    assert (plan, case) == before
    return run, controller


def view_at(run, at_ts, index=0):
    run.advance(at_ts)
    return run.snapshot(at_ts).batteries[index]


def stop_events(api, *, request_ts=T0 + .25, barrier_ts=T0 + .3, barrier_order=4, **changes):
    request = RevokeRequest("synthetic", 2, "stop-a", "operator_stop", round(request_ts * 1000))
    barrier = BarrierReceipt(**(dict(
        installation_id="synthetic", barrier_id="stop-a", revoked_through_epoch=2, current_epoch=3,
        status="accepted", accepted_ts=round(barrier_ts * 1000), queues_fenced=True,
        safe_state="confirmed", receiver_generation="generation-a", reason="barrier_accepted",
    ) | changes))
    return api.StopRequestEvent(request_ts, 0, request), api.BarrierEvent(barrier_ts, barrier_order, barrier)


def test_f3a2_issued_before_send_and_missing_send_evidence():
    api = execution_api()
    plan = replay_plan()
    cmd = replay_command(plan)
    timing = replay_timing(api, cmd)
    observation = replay_receipt(api, cmd, T0 + .2, apply_ts=T0 + .2)
    original_bytes, digest = cmd.to_dict(), payload_digest(cmd)
    run, state = recorded_run(api, plan=plan, events=(send_event(api, cmd, T0 + .1, timing), observation))
    assert view_at(run, T0).command is None
    pending = view_at(run, T0 + .1)
    assert pending.command is None and pending.drive_w is None and pending.pending == cmd
    assert not pending.recovery_required
    assert view_at(run, T0 + .199).command is None
    accepted = view_at(run, T0 + .2)
    assert accepted.command == cmd and accepted.drive_w == 1000 and accepted.override
    assert state.battery(0).held_setpoint_w == 1000
    assert cmd.to_dict() == original_bytes and payload_digest(cmd) == digest
    assert run.snapshot(T0 + .2).writes == 1

    for sent, expected in ((T0 + .1, cmd), (None, None), (T0 + .21, None)):
        opening = api.OpeningAcceptance(cmd, timing, observation, sent)
        opened, _ = recorded_run(api, plan=plan, start=T0 + .3, openings=(opening,))
        assert view_at(opened, T0 + .3).command == expected
        assert observation in opened.snapshot(T0 + .3).trace
    future = replace(cmd, issued_at_ms=cmd.issued_at_ms + 101)
    refused, _ = recorded_run(api, plan=plan, events=(send_event(api, future, T0 + .1, timing),
        replay_receipt(api, future, T0 + .2, apply_ts=T0 + .2)))
    assert view_at(refused, T0 + .2).command is None
    assert "execution_binding_mismatch" in refused.snapshot(T0 + .2).reasons


@pytest.mark.parametrize("stage,offset", [(stage, offset) for stage in ("receive", "apply")
                                         for offset in (-.001, 0, .001)])
def test_f3a2_effective_deadlines_and_late_application(stage, offset):
    api = execution_api()
    cmd = replay_command(replay_plan())
    timing = replay_timing(api, cmd)
    receive = T0 + .75 + offset if stage == "receive" else T0 + .749
    applied = receive if stage == "receive" else T0 + 1 + offset
    observation = replay_receipt(api, cmd, receive, apply_ts=applied, accepted_ts=cmd.issued_at_ms + 701)
    run, state = recorded_run(api, events=(send_event(api, cmd, T0 + .7, timing), observation))
    assert view_at(run, T0 + .7).pending == cmd
    assert run.next_boundary(T0 + .7, T0 + 2) == min(receive, T0 + .75)
    result = view_at(run, applied)
    if offset < 0:
        assert result.command == cmd and result.drive_w == 1000 and result.pending is None
        assert state.battery(0).held_setpoint_w == 1000
    else:
        assert result.command is None and result.drive_w is None and result.pending == cmd
        assert result.execution_status == "unconfirmed" and result.recovery_required
        assert state.battery(0).held_setpoint_w == 0 and state.battery(0).last_change_ts == 0
        assert "execution_deadline_exceeded" in result.reasons
    assert observation in run.snapshot(applied).trace
    assert run.next_boundary(applied, T0 + 2) > applied


@pytest.mark.parametrize("missing", ["timing", "receipt", "application", "raw-partial"])
def test_f3a2_missing_evidence_expires_at_the_original_cutoff(missing):
    api = execution_api()
    cmd = replay_command(replay_plan())
    timing = None if missing == "timing" else replay_timing(api, cmd)
    events = [send_event(api, cmd, T0 + .7, timing)]
    if missing in ("application", "raw-partial"):
        events.append(replay_receipt(api, cmd, T0 + .72,
                                    **({"status": "unconfirmed", "effect": "partial"}
                                       if missing == "raw-partial" else {})))
    events.append(replay_receipt(api, cmd, T0 + 1.1, apply_ts=T0 + 1.1, order=3, apply_order=4))
    run, state = recorded_run(api, events=tuple(events))
    cutoff = T0 + (1 if missing == "application" else .75)
    if missing != "timing":
        assert not view_at(run, cutoff - .001).recovery_required
        assert run.next_boundary(cutoff - .001, T0 + 2) == cutoff
    else:
        cutoff = T0 + .7
    expired = view_at(run, cutoff)
    assert expired.pending == cmd and expired.recovery_required and expired.command is None
    assert expired.execution_status == "unconfirmed"
    assert ("execution_deadline_unknown" if missing == "timing" else "execution_deadline_exceeded") in expired.reasons
    assert view_at(run, T0 + 1.1).command is None
    assert state.battery(0).last_accepted_command is None and state.battery(0).held_setpoint_w == 0


@pytest.mark.parametrize("application_first", [False, True])
def test_f3a2_subms_order_and_barrier_precedence(application_first):
    api = execution_api()
    at_ts = T0 + .0006
    cmd = replay_command(replay_plan(), issued_at_ms=T0 * 1000 + 1)
    timing = replay_timing(api, cmd)
    observed = replay_receipt(api, cmd, at_ts, apply_ts=at_ts)
    run, _ = recorded_run(api, events=(send_event(api, cmd, at_ts, timing), observed))
    assert run.next_boundary(T0, T0 + .1) == at_ts
    assert view_at(run, at_ts).drive_w == 1000
    assert run.next_boundary(at_ts, T0 + .1) == T0 + .1

    delayed = replace(observed, application=replace(observed.application, at_ts=T0 + .0008,
                                                    clock=replay_clock(T0 + .0008)))
    run, _ = recorded_run(api, events=(send_event(api, cmd, at_ts, timing), delayed))
    assert view_at(run, at_ts).command is None
    assert run.next_boundary(at_ts, T0 + .1) == T0 + .0008
    assert view_at(run, T0 + .0008).drive_w == 1000

    request, barrier = stop_events(api)
    observed = replay_receipt(api, cmd, T0 + .2, apply_ts=T0 + .3,
                              apply_order=3 if application_first else 5)
    run, state = recorded_run(api, events=(send_event(api, cmd, at_ts, timing), observed, request, barrier),
                              stop_response="ideal_programmed_stop")
    stopped = view_at(run, T0 + .3)
    assert stopped.command is None and stopped.drive_w == 0 and not stopped.recovery_required
    assert state.battery(0).held_setpoint_w == (1000 if application_first else 0)
    assert state.battery(0).last_accepted_command == (cmd if application_first else None)


def test_f3a2_no_write_rejected_and_partial_keep_distinct_states():
    api = execution_api()
    plan = replay_plan()
    old = replay_command(plan)
    timing = replay_timing(api, old)
    opening = api.OpeningAcceptance(old, timing, replay_receipt(api, old, T0 + .1, apply_ts=T0 + .1), T0)
    no_write = api.ProposalEvent(T0 + .3, 0, proposal(setpoint_w=1100, write=False, override=False), None, None)
    for status, effect, outcome in (("rejected", "none", "unknown"), ("unconfirmed", "partial", "unconfirmed")):
        new = replace(old, command_id="command-b", sequence=4, setpoint_w=1200, issued_at_ms=(T0 * 1000 + 400))
        events = (no_write, send_event(api, new, T0 + .4, timing, restores=True, override=False),
                  replay_receipt(api, new, T0 + .5, apply_ts=T0 + .6, status=status, effect=effect, outcome=outcome))
        run, state = recorded_run(api, openings=(opening,), start=T0 + .2, events=events)
        unchanged = view_at(run, T0 + .3)
        assert unchanged.drive_w == 1000 and unchanged.command == old and unchanged.override
        assert run.snapshot(T0 + .3).writes == 0
        assert view_at(run, T0 + .4).drive_w is None
        assert state.battery(0).restore_pending
        result = view_at(run, T0 + .6)
        assert result.command is None and result.drive_w is None and result.pending == new
        assert result.recovery_required is (status == "unconfirmed")
        assert result.execution_status == outcome
        assert state.battery(0).last_accepted_command == old and state.battery(0).held_setpoint_w == 1000
        assert state.battery(0).restore_pending
        assert ("execution_rejected" if status == "rejected" else "execution_unconfirmed") in result.reasons


@pytest.mark.parametrize("bad", [
    {"barrier_id": "foreign"}, {"receiver_generation": "foreign"}, {"installation_id": "foreign"},
    {"revoked_through_epoch": 1}, {"current_epoch": 4}, {"accepted_ts": T0 * 1000 - 1},
    {"accepted_ts": T0 * 1000 + 301},
    {"status": "unconfirmed", "accepted_ts": None, "queues_fenced": False},
    {"status": "unconfirmed", "accepted_ts": None, "safe_state": "unknown"},
])
def test_f3a2_stop_fences_old_receipts_and_rebind_is_single_use(bad):
    api = execution_api()
    cmd = replay_command(replay_plan())
    timing = replay_timing(api, cmd)
    partial = replay_receipt(api, cmd, T0 + .1, apply_ts=T0 + .2,
                             status="unconfirmed", effect="partial", outcome="unconfirmed")
    request, wrong = stop_events(api, **bad)
    _, exact = stop_events(api, barrier_ts=T0 + .4)
    new = OwnerBinding("synthetic", "receiver-a", "generation-a", "session-b", 3, "stop-a", (T0 + 20) * 1000)
    events = (send_event(api, cmd, T0, timing), partial, request, wrong, exact,
              api.BindingEvent(T0 + .5, 0, new),
              replay_receipt(api, cmd, T0 + .6, apply_ts=T0 + .6, order=1, apply_order=2),
              api.BindingEvent(T0 + .7, 0, replace(new, owner_session_id="session-c")),
              send_event(api, cmd, T0 + .8, timing))
    run, state = recorded_run(api, events=events, stop_response="ideal_programmed_stop")
    assert view_at(run, T0 + .3).recovery_required
    assert run.snapshot(T0 + .3).binding.owner_epoch == 2
    stopped = view_at(run, T0 + .4)
    assert stopped.execution_status == "stopped" and not stopped.recovery_required and stopped.pending is None
    assert stopped.command is None and stopped.drive_w == 0
    assert run.snapshot(T0 + .4).binding is None
    view_at(run, T0 + .7)
    assert run.snapshot(T0 + .7).binding == new
    assert state.battery(0).last_accepted_command is None and state.battery(0).held_setpoint_w == 0
    assert view_at(run, T0 + .8).command is None
    assert "execution_binding_mismatch" in run.snapshot(T0 + .8).reasons


def test_f3a2_exact_retry_keeps_identity_and_budget():
    api = execution_api()
    cmd = replay_command(replay_plan())
    timing = replay_timing(api, cmd)
    first = send_event(api, cmd, T0 + .1, timing)
    retry = send_event(api, cmd, T0 + .7, timing)
    observed = replay_receipt(api, cmd, T0 + .751, apply_ts=T0 + .751)
    run, state = recorded_run(api, events=(first, retry, observed))
    assert view_at(run, T0 + .7).pending == cmd
    assert run.snapshot(T0 + .7).writes == 2
    assert run.next_boundary(T0 + .7, T0 + 2) == T0 + .75
    assert view_at(run, T0 + .75).recovery_required
    assert view_at(run, T0 + .751).command is None
    assert state.battery(0).last_accepted_command is None
    with pytest.raises(ValueError, match="^replay_duplicate_identity$"):
        run, _ = recorded_run(api, events=(first, replace(retry, command=replace(cmd, setpoint_w=1001))))
        run.advance(T0 + .7)
    extended = replace(retry, timing=replay_timing(api, cmd, anchor=T0 + .7))
    run, _ = recorded_run(api, events=(first, extended, observed))
    assert view_at(run, T0 + .751).command is None
    assert "execution_deadline_unknown" in run.snapshot(T0 + .751).reasons


@pytest.mark.parametrize("flaw", ["clock-boot", "clock-mono", "clock-utc", "clock-health", "clock-missing",
                                  "anchor-health", "anchor-utc", "uncertainty", "mapping",
                                  "receipt-payload", "receipt-session", "application-outcome"])
def test_f3a2_unproven_clock_or_receipt_never_updates_held(flaw):
    api = execution_api()
    cmd = replay_command(replay_plan())
    timing = replay_timing(api, cmd)
    observed = replay_receipt(api, cmd, T0 + .2, apply_ts=T0 + .3)
    if flaw.startswith("clock-"):
        changes = {"clock-boot": {"boot_id": "other"}, "clock-mono": {"boottime_ns": 1},
                   "clock-utc": {"utc_ms": 1}, "clock-health": {"healthy": False}}
        observed = replace(observed, received_clock=None if flaw == "clock-missing" else
                           replace(observed.received_clock, **changes[flaw]))
    elif flaw in ("anchor-health", "anchor-utc"):
        timing = replace(timing, anchor_clock=replace(timing.anchor_clock,
            **({"healthy": False} if flaw == "anchor-health" else {"utc_ms": 1})))
    elif flaw == "uncertainty":
        observed = replace(observed, application=replace(observed.application,
            clock=replace(observed.application.clock, uncertainty_ms=2000)))
    elif flaw == "mapping":
        observed = replace(observed, application=replace(observed.application,
            clock=replace(observed.application.clock, boottime_ns=1)))
    elif flaw.startswith("receipt-"):
        observed = replace(observed, receipt=replace(observed.receipt,
            **({"payload_digest": "0" * 64} if flaw == "receipt-payload" else {"owner_session_id": "foreign"})))
    else:
        observed = replace(observed, application=replace(observed.application, outcome="unknown"))
    run, state = recorded_run(api, events=(send_event(api, cmd, T0 + .1, timing), observed))
    assert view_at(run, T0 + 1).command is None
    assert state.battery(0).held_setpoint_w == 0 and state.battery(0).last_accepted_command is None
    assert observed in run.snapshot(T0 + 1).trace


def test_f3a2_coverage_hold_and_snapshots_do_not_resurrect_history():
    api = execution_api()
    cmd = replay_command(replay_plan(), hold_until_ms=(T0 + 2) * 1000)
    timing = replay_timing(api, cmd)
    observed = replay_receipt(api, cmd, T0 + .2, apply_ts=T0 + .2)
    coverage = (api.ExecutionCoverage("accu", T0, T0 + .4),
                api.ExecutionCoverage("accu", T0 + .5, T0 + 10))
    run, state = recorded_run(api, events=(send_event(api, cmd, T0, timing), observed), coverage=coverage)
    assert view_at(run, T0 + .2).drive_w == 1000
    assert run.next_boundary(T0 + .2, T0 + 1) == T0 + .4
    assert view_at(run, T0 + .4).drive_w is None
    assert view_at(run, T0 + .5).drive_w is None
    before = deepcopy(state)
    first = run.snapshot(T0 + .5)
    assert run.snapshot(T0 + .5) == first and state == before
    assert "execution_history_missing" in first.batteries[0].reasons
    with pytest.raises(ValueError, match="^replay_time_order_invalid$"):
        run.advance(T0 + .4)
    with pytest.raises(ValueError, match="^replay_time_order_invalid$"):
        run.snapshot(T0 + .6)
    for stop_response, expected in (("unknown", None), ("ideal_programmed_stop", 0)):
        run, state = recorded_run(api, events=(send_event(api, cmd, T0, timing), observed),
                                  stop_response=stop_response)
        assert view_at(run, T0 + 1).drive_w == 1000
        assert run.next_boundary(T0 + 1, T0 + 3) == T0 + 2
        expired = view_at(run, T0 + 2)
        assert expired.command is None and expired.drive_w == expected
        assert state.battery(0).last_accepted_command == cmd


def test_f3a2_v2_plan_and_profile_limit_are_not_rewritten():
    from dao.tests.settlement.test_fast_contract import ready_plan

    api = execution_api()
    plan = ready_plan()
    start = plan.intervals[0].start_ts
    plan.intervals[0].battery(0).stop_ts = start + 1
    for hold, accepted in ((start + 1, True), (start + 2, False)):
        cmd = replay_command(plan, execute_before_ms=(start + 1) * 1000, hold_until_ms=hold * 1000)
        timing = replay_timing(api, cmd, anchor=start, apply=.9)
        observation = replay_receipt(api, cmd, start + .2, apply_ts=start + .2, anchor=start)
        run, _ = recorded_run(api, plan=plan, start=start, end=start + 10, events=(
            send_event(api, cmd, start + .1, timing, stop_inverter="2000-01-01 00:00:00"), observation))
        # The sentinel clears the profile stop, but cannot extend its command window.
        result = view_at(run, start + .2)
        assert (result.command == cmd) is accepted
        if not accepted:
            assert "execution_binding_mismatch" in result.reasons


def test_f3a2_duplicate_receipt_is_history_while_original_application_is_pending():
    api = execution_api()
    cmd = replay_command(replay_plan())
    timing = replay_timing(api, cmd)
    original = replay_receipt(api, cmd, T0 + .2, apply_ts=T0 + .9)
    duplicate = replace(original, at_ts=T0 + .8, received_clock=replay_clock(T0 + .8), application=None)
    run, state = recorded_run(api, events=(send_event(api, cmd, T0, timing), original, duplicate))
    assert view_at(run, T0 + .8).pending == cmd
    assert not state.battery(0).recovery_required
    assert view_at(run, T0 + .9).drive_w == 1000
    assert run.snapshot(T0 + .9).trace[-2:] == (original, duplicate)


def test_f3a2_shorter_retry_rechecks_captured_receive_evidence():
    api = execution_api()
    cmd = replay_command(replay_plan())
    timing = replay_timing(api, cmd)
    observation = replay_receipt(api, cmd, T0 + .7, apply_ts=T0 + .9)
    retry = send_event(api, cmd, T0 + .71, replay_timing(api, cmd, receive=.6))
    run, state = recorded_run(api, events=(send_event(api, cmd, T0, timing), observation, retry))
    result = view_at(run, T0 + .9)
    assert result.command is None and result.recovery_required and result.pending == cmd
    assert state.battery(0).held_setpoint_w == 0


def test_f3a2_stale_different_barrier_cannot_withdraw_a_new_context():
    api = execution_api()
    cmd = replay_command(replay_plan())
    request, barrier = stop_events(api)
    late_request = replace(request, at_ts=T0 + .26,
                           request=replace(request.request, barrier_id="older-unanswered-request"))
    late_barrier = replace(barrier, at_ts=T0 + .7,
                           receipt=replace(barrier.receipt, barrier_id="older-unanswered-request"))
    binding = OwnerBinding("synthetic", "receiver-a", "generation-a", "session-b", 3, "stop-a", (T0 + 20) * 1000)
    new = replace(cmd, command_id="new-epoch", sequence=4, owner_epoch=3, owner_session_id="session-b",
                  issued_at_ms=T0 * 1000 + 500)
    events = (request, late_request, barrier, api.BindingEvent(T0 + .4, 0, binding),
              send_event(api, new, T0 + .5, replay_timing(api, new)),
              replay_receipt(api, new, T0 + .6, apply_ts=T0 + .6), late_barrier)
    run, _ = recorded_run(api, events=events, stop_response="ideal_programmed_stop")
    assert view_at(run, T0 + .6).command == new
    result = view_at(run, T0 + .7)
    assert result.command == new and result.drive_w == 1000
    assert run.snapshot(T0 + .7).binding == binding
    assert "execution_barrier_unproven" in run.snapshot(T0 + .7).reasons


def test_f3a2_missing_command_evidence_cannot_leave_an_older_application_open():
    api = execution_api()
    cmd = replay_command(replay_plan())
    observed = replay_receipt(api, cmd, T0 + .2, apply_ts=T0 + .4)
    unknown_send = api.ProposalEvent(T0 + .3, 0, proposal(), None, None)
    run, state = recorded_run(api, events=(send_event(api, cmd, T0, replay_timing(api, cmd)),
                                          observed, unknown_send))
    assert view_at(run, T0 + .3).recovery_required
    assert view_at(run, T0 + .4).command is None
    assert state.battery(0).held_setpoint_w == 0 and state.battery(0).last_accepted_command is None


def test_f3a2_opening_applications_share_the_causal_timeline():
    api = execution_api()
    old = replay_command(replay_plan())
    new = replace(old, command_id="opening-b", sequence=4, issued_at_ms=T0 * 1000 + 300, setpoint_w=1200)
    timing = replay_timing(api, old)
    # A second send while the first application's outcome is still unproven
    # cannot be made into a clean succession by folding whole receipts at once.
    openings = (api.OpeningAcceptance(old, timing,
        replay_receipt(api, old, T0 + .2, apply_ts=T0 + .6), T0),
        api.OpeningAcceptance(new, timing,
        replay_receipt(api, new, T0 + .4, apply_ts=T0 + .5), T0 + .3))
    run, state = recorded_run(api, start=T0 + .7, openings=openings)
    result = view_at(run, T0 + .7)
    assert result.command is None and result.recovery_required
    assert state.battery(0).held_setpoint_w == 0 and state.battery(0).last_accepted_command is None


def test_f3a2_stop_only_clears_the_covered_pending_epoch_and_preserves_other_battery():
    api = execution_api()
    plan = replay_plan(("accu", "second"))
    old = replay_command(plan)
    other = replace(old, command_id="other-epoch", sequence=4, battery_name="second", owner_epoch=4)
    timing = replay_timing(api, old)
    request, barrier = stop_events(api)
    events = (send_event(api, old, T0, timing),
              replay_receipt(api, old, T0 + .1, apply_ts=T0 + .1, status="unconfirmed", effect="partial",
                             outcome="unconfirmed"),
              send_event(api, other, T0 + .2, timing, index=1), request, barrier)
    run, state = recorded_run(api, plan=plan, events=events, stop_response="ideal_programmed_stop")
    result = view_at(run, T0 + .3)
    assert not result.recovery_required and result.drive_w == 0
    second = run.snapshot(T0 + .3).batteries[1]
    assert second.pending == other and second.recovery_required and second.drive_w is None
    assert state.battery(1).last_accepted_command is None


def test_f3a2_retry_proposal_still_has_to_match_the_original_payload():
    api = execution_api()
    cmd = replay_command(replay_plan())
    timing = replay_timing(api, cmd)
    retry = send_event(api, cmd, T0 + .2, timing, setpoint_w=1200)
    run, state = recorded_run(api, events=(send_event(api, cmd, T0, timing), retry,
        replay_receipt(api, cmd, T0 + .3, apply_ts=T0 + .3)))
    assert view_at(run, T0 + .3).command is None
    assert state.battery(0).held_setpoint_w == 0
    assert "execution_binding_mismatch" in run.snapshot(T0 + .3).reasons


@pytest.mark.parametrize("application", [False, True])
def test_f3a2_unproven_receipt_is_not_an_early_terminal_consumer_outcome(application):
    api = execution_api()
    cmd = replay_command(replay_plan())
    observation = replay_receipt(api, cmd, T0 + .2, apply_ts=T0 + .4 if application else None)
    observation = replace(observation, received_clock=None)
    run, _ = recorded_run(api, events=(send_event(api, cmd, T0, replay_timing(api, cmd)), observation))
    cutoff = T0 + (.4 if application else .75)
    result = view_at(run, cutoff - .001)
    assert result.pending == cmd and result.command is None and not result.recovery_required
    assert "execution_deadline_unknown" in result.reasons
    assert view_at(run, cutoff).recovery_required


def test_f3a2_ideal_hold_stop_can_coincide_with_owner_expiration():
    api = execution_api()
    cmd = replay_command(replay_plan(), hold_until_ms=(T0 + 2) * 1000)
    binding = OwnerBinding("synthetic", "receiver-a", "generation-a", "session-a", 2,
                           "opening-barrier", (T0 + 2) * 1000)
    run, state = recorded_run(api, opening_binding=binding, stop_response="ideal_programmed_stop", events=(
        send_event(api, cmd, T0, replay_timing(api, cmd)),
        replay_receipt(api, cmd, T0 + .2, apply_ts=T0 + .2)))
    assert view_at(run, T0 + 1).command == cmd
    expired = view_at(run, T0 + 2)
    assert expired.command is None and expired.drive_w == 0
    assert run.snapshot(T0 + 2).binding is None
    assert state.battery(0).last_accepted_command == cmd


def test_f3a2_one_uncertain_battery_does_not_destroy_independent_acceptance():
    api = execution_api()
    plan = replay_plan(("accu", "second"))
    first = replay_command(plan)
    second = replace(first, command_id="independent", sequence=4, battery_name="second", setpoint_w=-1500)
    timing = replay_timing(api, first)
    run, _ = recorded_run(api, plan=plan, events=(
        send_event(api, first, T0, timing), send_event(api, second, T0, timing, order=1, index=1),
        replay_receipt(api, first, T0 + .1, apply_ts=T0 + .2, status="unconfirmed", effect="partial",
                       outcome="unconfirmed"),
        replay_receipt(api, second, T0 + .1, apply_ts=T0 + .2, order=3, apply_order=4)))
    assert view_at(run, T0 + .2).recovery_required
    other = run.snapshot(T0 + .2).batteries[1]
    assert other.command == second and other.drive_w == -1500 and not other.recovery_required


def test_f3a2_duplicate_with_its_own_timely_application_can_finish_open_pending():
    api = execution_api()
    cmd = replay_command(replay_plan())
    original = replay_receipt(api, cmd, T0 + .2)
    duplicate = replace(original, at_ts=T0 + .3, received_clock=replay_clock(T0 + .3),
        application=api.ReplayApplication(T0 + .4, 2, replay_clock(T0 + .4), "accepted"))
    run, state = recorded_run(api, events=(send_event(api, cmd, T0, replay_timing(api, cmd)), original, duplicate))
    assert view_at(run, T0 + .3).command is None
    assert view_at(run, T0 + .4).command == cmd
    assert state.battery(0).held_setpoint_w == 1000


def test_f3a2_ambiguous_command_battery_mapping_is_rejected_before_folding():
    api = execution_api()
    plan = replay_plan(("accu", "second"))
    cmd = replay_command(plan, battery_name="second")
    event = send_event(api, cmd, T0, replay_timing(api, cmd), name="accu", index=0)
    with pytest.raises(ValueError, match="^replay_battery_mapping_invalid$"):
        recorded_run(api, plan=plan, events=(event,))


@pytest.mark.parametrize("at_start", ["application", "receipt-application", "send-receipt-application"])
@pytest.mark.parametrize("application_first", [False, True])
def test_f3a2_i1_opening_start_events_obey_barrier_order(at_start, application_first):
    api = contract()
    cmd = replay_command(replay_plan())
    start = T0 + .3
    sent = start if at_start == "send-receipt-application" else T0 + .1
    received = T0 + .2 if at_start == "application" else start
    observation = replay_receipt(api, cmd, received, apply_ts=start,
                                 apply_order=2 if application_first else 5,
                                 accepted_ts=round(received * 1000))
    opening = api.OpeningAcceptance(cmd, replay_timing(api, cmd), observation, sent)
    request, barrier = stop_events(api, request_ts=start, barrier_ts=start)
    request = replace(request, order=3)
    run, state = recorded_run(api, start=start, openings=(opening,), events=(request, barrier),
                              stop_response="ideal_programmed_stop")

    battery = state.battery(0)
    assert battery.last_accepted_command == (cmd if application_first else None)
    assert battery.held_setpoint_w == (1000 if application_first else 0)
    assert battery.last_change_ts == (observation.receipt.accepted_ts / 1000 if application_first else 0)
    assert battery.last_command_w == (1000 if application_first else 0)
    assert battery.held_stop_inverter_cleared is application_first
    snapshot = run.snapshot(start)
    view = snapshot.batteries[0]
    assert view.command is None and view.drive_w == 0
    assert view.pending is None and not view.recovery_required
    if not application_first:
        assert view.execution_status == "stopped"
    assert snapshot.trace == (observation, request, barrier)
    assert snapshot.writes == (1 if sent == start else 0)
    assert run.next_boundary(start, start + .1) == start + .1
    run.advance(start)
    assert run.snapshot(start) == snapshot


@pytest.mark.parametrize("send_at_start", [False, True])
def test_f3a2_i1_opening_receipt_after_start_barrier_is_history(send_at_start):
    api = contract()
    cmd = replay_command(replay_plan())
    start = T0 + .3
    observation = replay_receipt(api, cmd, start, order=6, apply_ts=start, apply_order=7)
    opening = api.OpeningAcceptance(cmd, replay_timing(api, cmd), observation,
                                    start if send_at_start else T0 + .1)
    request, barrier = stop_events(api, request_ts=start, barrier_ts=start)
    request = replace(request, order=3)
    run, state = recorded_run(api, start=start, openings=(opening,), events=(request, barrier))
    snapshot = run.snapshot(start)
    assert snapshot.trace == (request, barrier, observation)
    assert state.battery(0).last_accepted_command is None
    assert state.battery(0).held_setpoint_w == 0 and state.battery(0).last_change_ts == 0
    assert snapshot.batteries[0].execution_status == "stopped"
    assert snapshot.batteries[0].pending is None


def model_api():
    api = execution_api()
    assert hasattr(api, "model_namespace"), "F3a3 deterministic model namespace missing"
    assert hasattr(api.ExecutionReplay, "submit"), "F3a3 declarative model submission missing"
    return api


def model_run(api, *, plan=None, case=None, start=T0, end=T0 + 10, **model_changes):
    plan = replay_plan() if plan is None else plan
    case = scenario(api, modeled(api, **model_changes)) if case is None else case
    points = ((start, 400.0), (end, 400.0))
    inputs = validate(api, plan=plan, scenario=case, sample_points=points,
                      soc_start={spec.name: 40.0 for spec in plan.specs})
    namespace = api.model_namespace(source_plan=plan, execution_plan=plan, sample_points=points,
                                    limits=PolicyLimits(), replay_input=inputs)
    state = ControllerState()
    run = api.ExecutionReplay(plan=plan, scenario=case, controller=state, start_ts=start, end_ts=end,
                              namespace=namespace)
    return run, state, namespace


def model_events(api, snapshot, kind):
    return tuple(event for event in snapshot.trace if type(event) is getattr(api, kind))


def test_f3a3_model_namespace_and_receipts_are_scenario_local():
    api = model_api()
    plan = replay_plan()
    inputs = validate(api)
    kwargs = dict(source_plan=plan, execution_plan=plan, sample_points=POINTS,
                  limits=PolicyLimits(), replay_input=inputs)
    before = deepcopy(kwargs)
    namespace = api.model_namespace(**kwargs)
    assert len(namespace) == 64 and set(namespace) <= set("0123456789abcdef")
    assert api.model_namespace(**deepcopy(kwargs)) == namespace
    other_plan = deepcopy(plan)
    other_plan.intervals[0].price_import = .31
    other_spec = deepcopy(plan)
    other_spec.specs[0].capacity_kwh += 1
    variations = (
        {"replay_input": replace(inputs, scenario=replace(inputs.scenario, scenario_id="baseline"))},
        {"limits": replace(kwargs["limits"], deadband=151)},
        {"replay_input": replace(inputs, soc_by_battery=(("accu", 41),))},
        {"replay_input": replace(inputs, inverter_efficiency_by_battery=(("accu", .94),))},
        {"execution_plan": other_spec}, {"execution_plan": other_plan}, {"source_plan": other_plan},
        {"sample_points": ((POINTS[0][0], 401), POINTS[-1])},
        {"replay_input": replace(inputs, scenario=scenario(api, modeled(api, application_delay_s=.1)))},
        {"replay_input": replace(inputs, scenario=scenario(api, modeled(api, responses=(
            api.ModelResponse(999, "fast", "accu", "partial", .1),))))},
        {"replay_input": replace(inputs, scenario=scenario(api, modeled(api, lifecycle=(
            api.ModelStop(T0 + 2, 0, True, False),))))},
        {"replay_input": replace(inputs, scenario=scenario(api, house_hold_s=30))},
    )
    identities = {api.model_namespace(**(kwargs | change)) for change in variations}
    assert len(identities) == len(variations) and namespace not in identities
    assert kwargs == before
    with pytest.raises((TypeError, ValueError)):
        api.model_namespace(**(kwargs | {"replay_input": validate(api, scenario=scenario(api, recorded(api)))}))

    snapshots = []
    for scenario_id in ("fast", "fast", "baseline"):
        case = replace(scenario(api), scenario_id=scenario_id)
        run, state, identity = model_run(api, case=case, start=T0 + .0006)
        decision = proposal(setpoint_w=1200.5, mode=None)
        run.submit(tick_index=0, at_ts=T0 + .0006, proposals=(decision,), source="fast")
        decision.setpoint_w = 9999
        view = view_at(run, T0 + .0006)
        cmd = view.command
        assert cmd is not None and view.drive_w == 1200
        assert cmd.command_id == f"replay/{identity}/{cmd.owner_epoch}/1"
        assert cmd.installation_id == f"replay/{identity}/installation"
        assert cmd.owner_session_id.startswith(f"replay/{identity}/")
        assert cmd.plan_digest == plan_digest(plan) and cmd.issued_at_ms == T0 * 1000 + 1
        snapshot = run.snapshot(T0 + .0006)
        assert snapshot.binding.receiver_generation.startswith(f"replay/{identity}/")
        observed, = model_events(api, snapshot, "ReceiptEvent")
        assert receipt_matches_command(observed.receipt, cmd)
        assert observed.at_ts == observed.application.at_ts == T0 + .0006
        assert observed.receipt.accepted_ts == observed.receipt.observed_at_ms == T0 * 1000 + 1
        assert run.next_boundary(T0 + .0006, T0 + .1) == T0 + .1
        assert state.battery(0).held_setpoint_w == 1200
        snapshots.append(snapshot)
    assert snapshots[0] == snapshots[1]
    first, other = (model_events(api, snapshot, "ReceiptEvent")[0].receipt for snapshot in snapshots[::2])
    foreign_cmd = snapshots[2].batteries[0].command
    assert first.command_id != other.command_id and first.payload_digest != other.payload_digest
    assert not receipt_matches_command(first, foreign_cmd)
    pending = ControllerState().battery(0)
    pending.pending_command = foreign_cmd
    assert not apply_execution_outcome(pending, outcome="accepted", receipt=first)
    assert pending.last_accepted_command is None and pending.pending_command == foreign_cmd


@pytest.mark.parametrize("effect,status,recovery", [
    ("complete", "accepted", False), ("none", "unknown", False),
    ("partial", "unconfirmed", True), ("unknown", "unconfirmed", True),
    (None, "unconfirmed", True),
])
def test_f3a3_explicit_effects_and_missing_selector_never_imply_success(effect, status, recovery):
    api = model_api()
    run, state, _ = model_run(api, default_effect=effect)
    run.submit(tick_index=0, at_ts=T0, proposals=(proposal(),), source="fast")
    result = view_at(run, T0)
    if effect is None:
        assert result.command is None and result.pending is not None and not result.recovery_required
        assert "execution_model_missing" in result.reasons
        assert run.next_boundary(T0, T0 + 2) == T0 + .75
        result = view_at(run, T0 + .75)
    assert result.execution_status == status and result.recovery_required is recovery
    assert result.drive_w == (-1200 if effect == "complete" else None)
    assert state.battery(0).held_setpoint_w == (-1200 if effect == "complete" else 0)
    receipts = model_events(api, run.snapshot(T0 + .75 if effect is None else T0), "ReceiptEvent")
    if effect is None:
        assert not receipts
    else:
        assert len(receipts) == 1
        assert receipts[0].receipt.status == {"complete": "accepted", "none": "rejected",
                                              "partial": "unconfirmed", "unknown": "unconfirmed"}[effect]


def test_f3a3_no_write_does_not_consume_a_selector_or_sequence():
    api = model_api()
    responses = (api.ModelResponse(1, "fast", "accu", "partial", .1),
                 api.ModelResponse(99, "plan", "accu", "complete", 0))
    run, state, _ = model_run(api, responses=responses)
    run.submit(tick_index=0, at_ts=T0, proposals=(proposal(setpoint_w=1000),), source="fast")
    old = view_at(run, T0).command
    run.advance(T0 + 1)
    run.submit(tick_index=1, at_ts=T0 + 1, proposals=(proposal(setpoint_w=1100, write=False),), source="fast")
    assert view_at(run, T0 + 1).command == old
    assert run.snapshot(T0 + 1).writes == 1 and state.battery(0).held_setpoint_w == 1000
    run.advance(T0 + 2)
    run.submit(tick_index=2, at_ts=T0 + 2, proposals=(proposal(setpoint_w=1200),), source="fast")
    assert view_at(run, T0 + 2).command.sequence == 2
    run.advance(T0 + 10)
    snapshot = run.snapshot(T0 + 10)
    assert len(model_events(api, snapshot, "ReceiptEvent")) == 2
    assert "unused_model_response" in snapshot.reasons
    suppressed, = [event for event in model_events(api, snapshot, "ProposalEvent") if not event.decision.write]
    assert suppressed.command is None and suppressed.timing is None


def test_f3a5_policy_no_write_cannot_relabel_accepted_execution_override():
    from dao.prog.fastctrl.policy import BatteryMeasurement, FastControlPolicy, Measurement

    api = model_api()
    plan = make_plan([.4], [1000], specs=[make_spec(cycle_cost=0)], export_discount=.3)
    run, state, _ = model_run(api, plan=plan)
    run.submit(tick_index=0, at_ts=T0, proposals=(BatteryDecision(
        0, "accu", 1000, 1000, True, False, "plan"),), source="plan")
    before = view_at(run, T0)
    assert before.command.setpoint_w == 1000 and before.override is False
    policy = FastControlPolicy(PolicyLimits(
        storage_value_mode="fixed", storage_value_fixed=.3, round_trip_efficiency=1,
        min_benefit=0, deadband=150, release_deviation=50, min_command_interval=0))
    measurement = Measurement(T0, -100, [BatteryMeasurement(soc=50, power_w=1000)])
    policy.account(state, plan, measurement, "2023-11-14", owner_binding=run.snapshot(T0).binding)
    decision = policy.decide(plan, measurement, state, owner_binding=run.snapshot(T0).binding)
    suppressed, = decision.batteries
    assert suppressed.override is True and suppressed.write is False
    assert suppressed.setpoint_w == 1000 and "deadband" in suppressed.reason
    run.submit(tick_index=0, at_ts=T0, proposals=tuple(decision.batteries), source="fast")
    after = view_at(run, T0)
    assert after.command == before.command and run.snapshot(T0).writes == 1
    assert after.override is False  # The accepted plan command was not replaced.


def test_f3a5_suppressed_plan_cannot_clear_accepted_override():
    from dao.prog.fastctrl.policy import BatteryMeasurement, FastControlPolicy, Measurement

    api = model_api()
    plan = make_plan([.4], [1000])
    response = api.ModelResponse(2, "plan", "accu", "complete", .1)
    run, state, _ = model_run(api, plan=plan, responses=(response,))
    run.submit(tick_index=0, at_ts=T0, proposals=(BatteryDecision(
        0, "accu", 1100, 1000, True, True, "override",
        stop_inverter="2023-11-14T22:13:28+00:00"),), source="fast")
    before = view_at(run, T0)
    assert before.command.setpoint_w == 1100 and before.override is True
    run.advance(T0 + 1)
    policy = FastControlPolicy(PolicyLimits(deadband=150))
    measurement = Measurement(T0 + 1, 0, [BatteryMeasurement(soc=50, power_w=1100)])
    # An unsent release changes policy intent, not the accepted execution. The
    # following disabled decision is suppressed against the still-held 1100 W.
    release = policy.release(plan, measurement, state, "disabled", owner_binding=run.snapshot(T0 + 1).binding)
    assert release.batteries[0].write and not release.batteries[0].override
    decision = policy.decide(plan, measurement, state, enabled=[False], owner_binding=run.snapshot(T0 + 1).binding)
    suppressed, = decision.batteries
    assert suppressed.setpoint_w == 1000 and not suppressed.write and not suppressed.override
    run.submit(tick_index=1, at_ts=T0 + 1, proposals=tuple(decision.batteries), source="fast")
    after = view_at(run, T0 + 1)
    assert after.command == before.command and run.snapshot(T0 + 1).writes == 1
    saved = deepcopy(state)
    assert run.snapshot(T0 + 1).batteries[0] == after
    assert state == saved
    assert after.override is True

    # A send is still insufficient: only its applied acceptance changes the
    # accepted provenance. The old override remains historical while pending.
    run.submit(tick_index=2, at_ts=T0 + 2, proposals=(replace(suppressed, write=True),), source="plan")
    pending = view_at(run, T0 + 2)
    assert pending.command is None and pending.pending is not None and pending.override is True
    applied = view_at(run, T0 + 2.1)
    assert applied.command != before.command and applied.command.setpoint_w == 1000
    assert applied.override is False and run.snapshot(T0 + 2.1).writes == 2


@pytest.mark.parametrize("version, setpoint, expected", [(1, 1000, False), (1, 1100, True),
                                                        (2, 2000, False), (2, 1000, True)])
def test_f3a5_opening_override_uses_accepted_profile_not_policy_state(version, setpoint, expected):
    from dao.tests.settlement.test_fast_profile import shaped_plan

    api = execution_api()
    plan = make_plan([.4], [1000]) if version == 1 else shaped_plan()
    issued = plan.intervals[0].start_ts
    cmd = replay_command(plan, setpoint_w=setpoint)
    observation = replay_receipt(api, cmd, issued + .1, apply_ts=issued + .2, anchor=issued)
    opening = api.OpeningAcceptance(cmd, replay_timing(api, cmd, anchor=issued), observation, issued)
    run, state = recorded_run(api, plan=plan, start=issued + 1, end=issued + 5, openings=(opening,))
    before = run.snapshot(issued + 1)
    assert before.batteries[0].command == cmd and before.batteries[0].override is expected
    assert before.writes == 0
    # Isolate the opening (which has no ProposalEvent) from mutable policy
    # intent; the real-policy paths in the two controls above prove that input.
    state.battery(0).override_active = not expected
    saved = deepcopy(state)
    after = run.snapshot(issued + 1)
    assert state == saved
    assert after == before


@pytest.mark.parametrize("delay,accepted", [(.049, True), (.05, False), (.051, False)])
def test_f3a3_one_absolute_budget_for_all_batteries(delay, accepted):
    api = model_api()
    run, state, _ = model_run(api, prepare_delay_s=.7, response_delay_s=delay)
    run.submit(tick_index=0, at_ts=T0, proposals=(proposal(),), source="plan")
    assert run.next_boundary(T0, T0 + 2) == T0 + .7
    assert view_at(run, T0 + .7).pending is not None
    assert view_at(run, T0 + .7).drive_w is None
    result = view_at(run, T0 + .8)
    assert (result.command is not None) is accepted
    assert result.recovery_required is not accepted
    observed, = model_events(api, run.snapshot(T0 + .8), "ReceiptEvent")
    assert observed.receipt.status == "accepted"  # Receiver acceptance is not consumer application.
    assert observed.application.outcome == ("accepted" if accepted else "unconfirmed")
    sent, = model_events(api, run.snapshot(T0 + .8), "ProposalEvent")
    assert sent.command.issued_at_ms == T0 * 1000 and sent.at_ts == T0 + .7
    assert sent.timing.anchor_ts == T0
    assert sent.timing.receive_deadline.monotonic_end_ns - sent.timing.anchor_clock.boottime_ns == 750_000_000
    assert state.battery(0).held_setpoint_w == (-1200 if accepted else 0)


@pytest.mark.parametrize("application_delay,accepted", [(.374, True), (.375, False), (.376, False)])
def test_f3a3_result_application_has_its_own_absolute_cutoff(application_delay, accepted):
    api = model_api()
    # Binary-exact preparation/response make the middle case truly equal to 1s.
    run, state, _ = model_run(api, prepare_delay_s=.5, response_delay_s=.125,
                              application_delay_s=application_delay)
    run.submit(tick_index=0, at_ts=T0, proposals=(proposal(),), source="fast")
    assert view_at(run, T0 + .8).command is None
    observation, = model_events(api, run.snapshot(T0 + .8), "ReceiptEvent")
    assert observation.application is None  # A scheduled application is not yet evidence.
    assert run.next_boundary(T0 + .8, T0 + 2) == min(T0 + 1, T0 + .5 + .125 + application_delay)
    result = view_at(run, T0 + 1.1)
    assert (result.command is not None) is accepted
    assert result.recovery_required is not accepted
    assert state.battery(0).held_setpoint_w == (-1200 if accepted else 0)
    observation, = model_events(api, run.snapshot(T0 + 1.1), "ReceiptEvent")
    assert observation.application.outcome == ("accepted" if accepted else "unconfirmed")


@pytest.mark.parametrize("separate_source", [False, True])
def test_f3a3_successive_batteries_and_sources_share_remaining_budget(separate_source):
    api = model_api()
    plan = replay_plan(("accu", "second"))
    responses = (api.ModelResponse(0, "plan", "accu", "complete", .04),
                 api.ModelResponse(0, "fast" if separate_source else "plan", "second", "complete", .02))
    run, state, _ = model_run(api, plan=plan, prepare_delay_s=.7, responses=responses)
    first, second = proposal(setpoint_w=1000), proposal(index=1, name="second", setpoint_w=-1500)
    run.submit(tick_index=0, at_ts=T0, proposals=(first,) if separate_source else (first, second), source="plan")
    if separate_source:
        run.submit(tick_index=0, at_ts=T0, proposals=(second,), source="fast")
    assert view_at(run, T0 + .73, 1).pending is None
    assert view_at(run, T0 + .74).drive_w == 1000
    assert view_at(run, T0 + .74, 1).pending is not None
    other = view_at(run, T0 + .76, 1)
    assert other.command is None and other.recovery_required
    assert state.battery(0).held_setpoint_w == 1000 and state.battery(1).held_setpoint_w == 0
    sends = model_events(api, run.snapshot(T0 + .76), "ProposalEvent")
    assert [event.command.sequence for event in sends] == [1, 2]
    assert [event.at_ts for event in sends] == [T0 + .7, T0 + .7 + .04]
    assert sends[0].timing.receive_deadline == sends[1].timing.receive_deadline


def test_f3a3_pending_subms_latency_and_terminal_failure_block_new_sends():
    api = model_api()
    start = T0 + .0006
    run, _, _ = model_run(api, start=start, response_delay_s=.0001, application_delay_s=.0001)
    run.submit(tick_index=0, at_ts=start, proposals=(proposal(),), source="fast")
    assert view_at(run, start).drive_w is None
    assert run.next_boundary(start, start + .1) == start + .0001
    assert view_at(run, start + .0001).drive_w is None
    assert view_at(run, start + .0001 + .0001).drive_w == -1200

    plan = replay_plan(("accu", "second"))
    run, _, _ = model_run(api, plan=plan, default_effect="partial", response_delay_s=.2)
    run.submit(tick_index=0, at_ts=T0, proposals=(proposal(), proposal(index=1, name="second")), source="plan")
    run.advance(T0 + .1)
    run.submit(tick_index=1, at_ts=T0 + .1, proposals=(proposal(index=1, name="second"),), source="fast")
    assert view_at(run, T0 + .4).recovery_required
    assert run.snapshot(T0 + .4).writes == 1
    assert run.snapshot(T0 + .4).batteries[1].pending is None


@pytest.mark.parametrize("fenced,safe", [(False, True), (True, False), (True, True)])
def test_f3a3_profile_epoch_and_stop_rules_are_receiver_owned(fenced, safe):
    api = model_api()
    lifecycle = (api.ModelStop(T0 + .1, 0, fenced, safe), api.ModelBind(T0 + .1, 1),
                 api.ModelBind(T0 + .3, 0))
    case = scenario(api, modeled(api, response_delay_s=.2, lifecycle=lifecycle),
                    stop_response="ideal_programmed_stop")
    run, state, _ = model_run(api, case=case)
    opening_binding = run.snapshot(T0).binding
    assert opening_binding is not None and run.snapshot(T0).batteries[0].drive_w == 0
    run.submit(tick_index=0, at_ts=T0, proposals=(proposal(),), source="fast")
    view_at(run, T0 + .3)
    snapshot = run.snapshot(T0 + .3)
    accepted_stop = fenced and safe
    assert snapshot.batteries[0].command is None
    assert state.battery(0).last_accepted_command is None and state.battery(0).held_setpoint_w == 0
    barriers = model_events(api, snapshot, "BarrierEvent")
    assert [event.receipt.status for event in barriers] == ["accepted", "accepted" if accepted_stop else "unconfirmed"]
    bindings = model_events(api, snapshot, "BindingEvent")
    assert len(bindings) == (2 if accepted_stop else 1)
    if accepted_stop:
        assert snapshot.binding.owner_epoch == opening_binding.owner_epoch + 1
        assert snapshot.binding.owner_session_id != opening_binding.owner_session_id
        assert snapshot.batteries[0].drive_w == 0
    observed, = model_events(api, snapshot, "ReceiptEvent")
    assert observed.receipt.status == "rejected" and observed.receipt.reason == "epoch_withdrawn"
    assert "execution_binding_mismatch" in snapshot.reasons  # Reusing a barrier never rebinds.


def test_f3a3_unknown_start_and_expired_profile_do_not_acquire_execution_rights():
    from dao.tests.settlement.test_fast_contract import ready_plan

    api = model_api()
    run, _, _ = model_run(api, initial_execution="unknown", lifecycle=(api.ModelBind(T0, 0),))
    run.submit(tick_index=0, at_ts=T0, proposals=(proposal(),), source="fast")
    snapshot = view_at(run, T0)
    assert snapshot.command is None and snapshot.drive_w is None
    assert run.snapshot(T0).binding is None and run.snapshot(T0).writes == 0

    plan = ready_plan()
    start = plan.intervals[0].start_ts
    plan.intervals[0].battery(0).stop_ts = start + 1
    before = deepcopy(plan)
    run, state, _ = model_run(api, plan=plan, start=start, end=start + 700, response_delay_s=1.1)
    binding = run.snapshot(start).binding
    assert binding.installation_id == plan.financial_context.basis.meter_id
    assert binding.owner_not_after_ms == (start + 600) * 1000
    run.submit(tick_index=0, at_ts=start, proposals=(proposal(setpoint_w=-200.6, mode=None),), source="fast")
    view_at(run, start + 1.1)
    snapshot = run.snapshot(start + 1.1)
    sent, = model_events(api, snapshot, "ProposalEvent")
    assert sent.command.setpoint_w == -201 and sent.command.mode == plan.specs[0].mode_on
    assert sent.command.execute_before_ms == sent.command.hold_until_ms == (start + 1) * 1000
    observed, = model_events(api, snapshot, "ReceiptEvent")
    assert observed.receipt.status == "rejected" and observed.receipt.reason == "command_expired"
    assert snapshot.batteries[0].recovery_required and state.battery(0).held_setpoint_w == 0
    assert plan == before
    recorded_replay, _ = recorded_run(api)
    with pytest.raises((TypeError, ValueError)):
        recorded_replay.submit(tick_index=0, at_ts=T0, proposals=(proposal(),), source="fast")


def test_f3a3_generated_application_order_roundtrips_through_recorded_fold():
    api = model_api()
    case = scenario(api, modeled(api, response_delay_s=.25, application_delay_s=.25,
                                lifecycle=(api.ModelStop(T0 + .5, 0, True, True),)),
                    stop_response="ideal_programmed_stop")
    run, state, _ = model_run(api, case=case)
    run.submit(tick_index=0, at_ts=T0, proposals=(proposal(),), source="fast")
    view_at(run, T0 + .5)
    snapshot = run.snapshot(T0 + .5)
    observation, = model_events(api, snapshot, "ReceiptEvent")
    barrier = model_events(api, snapshot, "BarrierEvent")[-1]
    assert observation.application.at_ts == barrier.at_ts
    assert observation.application.order > barrier.order
    folded, folded_state = recorded_run(
        api, events=snapshot.trace, opening_binding=None, stop_response="ideal_programmed_stop",
        installation_id=barrier.receipt.installation_id, receiver_generation=barrier.receipt.receiver_generation)
    assert view_at(folded, T0 + .5) == snapshot.batteries[0]
    assert folded_state.battery(0).held_setpoint_w == state.battery(0).held_setpoint_w == 0
    assert folded_state.battery(0).last_accepted_command is None


def test_f3a3_namespace_retains_unrounded_legacy_plan_inputs():
    api = model_api()
    plan = replay_plan()
    changed = deepcopy(plan)
    changed.price_average += .00000001
    assert plan_digest(changed) == plan_digest(plan)  # The legacy wire DTO rounds this field.
    kwargs = dict(source_plan=plan, execution_plan=plan, sample_points=POINTS,
                  limits=PolicyLimits(), replay_input=validate(api))
    original = api.model_namespace(**kwargs)
    assert api.model_namespace(**(kwargs | {"execution_plan": changed})) != original
    assert api.model_namespace(**(kwargs | {"source_plan": changed})) != original


def test_f3a3_rounding_across_a_profile_edge_cannot_extend_the_original_window():
    from dao.tests.settlement.test_fast_contract import ready_plan

    api = model_api()
    plan = ready_plan()
    start = plan.intervals[0].start_ts
    plan.intervals[0].battery(0).stop_ts = start + 1
    run, state, _ = model_run(api, plan=plan, start=start, end=start + 10)
    at_ts = start + .9996
    run.submit(tick_index=0, at_ts=at_ts, proposals=(proposal(),), source="fast")
    assert view_at(run, at_ts).command is None
    assert state.battery(0).last_accepted_command is None
    assert run.snapshot(at_ts).writes == 0
