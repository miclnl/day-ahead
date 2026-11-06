"""Pure replay input: explicit assumptions and detached, uncoerced evidence."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from importlib import import_module, util
from inspect import Parameter, signature
from typing import get_args, get_type_hints

import pytest

from dao.prog.fastctrl.command import ActuationReceipt, BatteryCommand, ClockSample, Deadline
from dao.prog.fastctrl.policy import BatteryDecision
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
