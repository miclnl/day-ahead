"""Tests for the backtest.

The backtest is the tool the saving estimate comes from, so its battery model
and its cost accounting have to be right, and the comparison has to be fair.
"""

import pytest

from dao.prog.fastctrl.plan import (
    BatteryPlanStep,
    BatterySpec,
    FastPlan,
    PlanInterval,
)
from dao.prog.fastctrl.policy import PolicyLimits
from dao.prog.fastctrl.simulate import (
    BatteryModel,
    Sample,
    aggregate_spec,
    compare,
    simulate,
    synthetic_case,
)
from dao.tests.settlement.conftest import bonus_rules as bonus_rules

T0 = 1_700_000_000


def modeled_case(scenario_id, *, hold=60, equal_initial_soc=False):
    from dao.prog.fastctrl.replay_execution import ModeledExecution, ReplayPhysics, ReplayScenario

    return ReplayScenario(scenario_id, ModeledExecution(
        initial_execution="stopped", default_effect="complete", prepare_delay_s=0,
        response_delay_s=0, application_delay_s=0,
    ), ReplayPhysics("battery_model_v1", hold, "ideal_programmed_stop", equal_initial_soc))


def test_f3b4_public_nullable_reports_use_financial_days_budget_timezone(monkeypatch):
    # B22-2: changing scenario timezone must still reset the physical day budget.
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    from dao.prog.fastctrl.policy import FastControlPolicy
    from dao.tests.settlement.test_fast_financial_replay import (
        f3b4_limits,
        f3b4_public_case,
        f3b4_source,
    )

    start = int(datetime(2027, 7, 1, 21, 30, tzinfo=UTC).timestamp())
    reset = start + 1800
    plan, replay, samples = f3b4_public_case(edges=tuple(start + 900 * i for i in range(5)))
    source = f3b4_source(plan, replay, samples)
    observations = []
    decide = FastControlPolicy.decide

    def observe(self, actual_plan, measurement, state, **kwargs):
        observations.append((measurement.timestamp, state.day_key, state.battery(0).daily_deviation_kwh))
        return decide(self, actual_plan, measurement, state, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "decide", observe)
    results, budgets = [], []
    for zone in (UTC, ZoneInfo("Europe/Amsterdam")):
        offset = len(observations)
        results.append(compare(plan, samples, f3b4_limits(daily_extra_throughput=.1),
            plan.specs[0].soc_min, timezone=zone, replay_basis=replay,
            baseline_execution=modeled_case("budget-base", hold=900),
            fast_execution=modeled_case("budget-fast", hold=900), source_evidence=source))
        budgets.append(observations[offset:])
    utc, amsterdam = results
    assert budgets[0][1][0] == budgets[1][1][0] == reset - 900
    assert budgets[0][1][2] >= .1 and budgets[1][1][2] >= .1
    assert budgets[0][2][0] == budgets[1][2][0] == reset
    assert budgets[0][2][2] >= .1 and budgets[1][2][2] == 0
    assert utc.fast.segments != amsterdam.fast.segments
    powers = [tuple((s.start_ts, s.batteries[0].command_w, s.batteries[0].modeled_ac_w)
                    for s in result.fast.segments if s.start_ts >= reset) for result in results]
    assert powers[0][0] == (reset, 3000, 3000)
    # The preceding plan-charge quarter has created discharge headroom. After
    # the reset the existing fixed-zero-storage policy chooses -2000 W; UTC
    # retains its exhausted budget and must keep the 3000 W plan command.
    assert powers[1][0] == (reset, -2000, -2000)
    exports = [sum(s.export_kwh for s in result.fast.segments if s.start_ts >= reset) for result in results]
    assert exports == [.5, 1.75]  # UTC .25+.25; Amsterdam 1.5+.25 kWh.
    assert utc.fast.financial_result.projection.items != amsterdam.fast.financial_result.projection.items
    assert list(utc.fast.days) == ["2027-07-01"]
    assert list(amsterdam.fast.days) == ["2027-07-01", "2027-07-02"]


def test_f3b4_public_nullable_reports_use_financial_days_fixed_run():
    # Catches a presentation timezone sneaking into projection or settlement.
    from datetime import UTC
    from importlib import import_module
    from zoneinfo import ZoneInfo

    from dao.prog.fastctrl import replay_finance as finance
    from dao.prog.fastctrl.replay_execution import validate_replay_input
    from dao.tests.settlement.test_fast_financial_replay import (
        f3b4_limits,
        f3b4_public_case,
        f3b4_source,
    )

    api = import_module("dao.prog.fastctrl.simulate")
    plan, replay, samples = f3b4_public_case()
    source = f3b4_source(plan, replay, samples)
    inputs = validate_replay_input(plan=plan, sample_points=tuple((s.timestamp, s.house_w) for s in samples),
        soc_start=plan.specs[0].soc_min, inverter_efficiency=.95, scenario=modeled_case("fixed-run", hold=900))
    basis = api._prepare_replay_basis(plan, start_ts=samples[0].timestamp, replay_basis=replay, recorded=False)
    binding = finance.bind_source_evidence(source, source_plan=basis.source_plan,
                                           replay_basis=replay, replay_inputs=(inputs,))
    validation = finance.validate_source_evidence(source, binding)
    run = api._run_replay(basis=basis, replay_input=inputs, limits=f3b4_limits(), use_fast_layer=False, timezone=UTC)
    results = tuple(api._summarize_replay(run=run, basis=basis, replay_input=inputs, label="fixed",
        timezone=zone, source_binding=binding, source_validation=validation) for zone in (UTC, ZoneInfo("Europe/Amsterdam")))
    left, right = (result.financial_result for result in results)
    assert basis.replay_basis is replay
    assert left.projection.items == right.projection.items
    assert left.projection.slices == right.projection.slices
    assert left.settlement == right.settlement
    assert left.components == right.components and left.days == right.days
    assert results[0].segments is results[1].segments is run.segments
    assert list(results[0].days) == ["2026-12-31"] and list(results[1].days) == ["2027-01-01"]
    assert results[0].days["2026-12-31"].financial_components is None
    attached = results[1].days["2027-01-01"]
    assert attached.financial_components is right.days[0].components
    assert attached.total_cost == float(right.days[0].components.total_cost.value)


def test_f3b4_source_changes_no_physics_or_execution():
    # Missing or wrongly bound evidence must not change commands or physical flows.
    from dataclasses import asdict, replace
    from datetime import UTC

    from dao.tests.settlement.test_fast_financial_replay import (
        f3b4_limits,
        f3b4_public_case,
        f3b4_source,
    )

    plan, replay, samples = f3b4_public_case()
    source = f3b4_source(plan, replay, samples)
    scenario = modeled_case("source-independent", hold=900)
    scenario = replace(scenario, execution=replace(scenario.execution, prepare_delay_s=.25))
    results = tuple(simulate(plan, samples, f3b4_limits(), plan.specs[0].soc_min, "source", False,
        timezone=UTC, replay_basis=replay, execution=scenario, source_evidence=evidence)
        for evidence in (None, source, replace(source, meter_id="synthetic:wrong")))

    def physical(side):
        return {key: {name: value for name, value in asdict(day).items()
            if name not in ("energy_cost", "wear_cost", "terminal_value", "total_cost", "financial_components")}
            for key, day in side.days.items()}

    for result in results[1:]:
        for field in ("segments", "trace", "soc_start_by_battery", "soc_end_by_battery", "soc_series",
                      "writes", "override_s", "import_kwh", "export_kwh", "throughput_kwh", "coverage_status"):
            assert getattr(result, field) == getattr(results[0], field)
        assert physical(result) == physical(results[0])
    assert any(segment.start_ts % 1 == .25 for segment in results[0].segments)
    assert results[-1].financial_status == "unknown" and results[-1].total_cost is None
    assert "source_binding_mismatch" in results[-1].financial_reasons


@pytest.mark.parametrize("fault", ["source", "second-input"])
def test_f3b4_malformed_inputs_rejected_before_either_run(monkeypatch, fault):
    from importlib import import_module

    from dao.tests.settlement.test_fast_financial_replay import f3b4_limits, f3b4_public_case

    api = import_module("dao.prog.fastctrl.simulate")
    plan, replay, samples = f3b4_public_case()
    runs = []
    real_run = api._run_replay

    def observe(**kwargs):
        runs.append(kwargs)
        return real_run(**kwargs)

    monkeypatch.setattr(api, "_run_replay", observe)
    with pytest.raises((TypeError, ValueError), match="^source_type_invalid$" if fault == "source" else None):
        compare(plan, samples, f3b4_limits(), 50, replay_basis=replay,
            baseline_execution=modeled_case("valid", hold=900),
            fast_execution=object() if fault == "second-input" else modeled_case("other", hold=900),
            **({"source_evidence": {"malformed": True}} if fault == "source" else {}))
    assert runs == []


@pytest.mark.parametrize("execution_kind", ["modeled", "recorded"])
@pytest.mark.parametrize("source_kind", ["full", "partial", "mismatch"])
def test_f3b7_public_source_execution_and_finance_are_separate(bonus_rules, execution_kind, source_kind):
    from copy import deepcopy
    from dataclasses import replace
    from decimal import Decimal

    from dao.prog.fastctrl import replay_execution as execution
    from dao.prog.fastctrl.simulate import Comparison
    from dao.tests.prog.test_fast_replay_execution import (
        replay_command,
        replay_receipt,
        replay_timing,
        send_event,
    )
    from dao.tests.settlement.test_fast_financial_replay import (
        f3b4_limits,
        f3b4_public_case,
        f3b4_source,
    )

    plan, replay, samples = f3b4_public_case(bonus_rules)
    start, end = samples[0].timestamp, samples[-1].timestamp
    source = f3b4_source(plan, replay, samples)
    evidence = {"full": source, "partial": None,
                "mismatch": replace(source, meter_id="synthetic:wrong")}[source_kind]
    limits = f3b4_limits()
    if execution_kind == "modeled":
        scenarios = (modeled_case("f3b7-base", hold=900), modeled_case("f3b7-fast", hold=900))
    else:
        events = []
        for index, row in enumerate(plan.intervals):
            at = row.start_ts
            command = replay_command(plan, setpoint_w=3000, command_id=f"f3b7-{index}", sequence=index + 3,
                issued_at_ms=at * 1000, execute_before_ms=(at + 2) * 1000, hold_until_ms=row.end_ts * 1000)
            events.extend((send_event(execution, command, at, replay_timing(execution, command, anchor=at)),
                replay_receipt(execution, command, at, apply_ts=at, anchor=at)))
        sent = events[0]
        recorded = _f3a5_recorded(plan, start, end, events)
        recorded = replace(recorded, physics=replace(recorded.physics, house_hold_s=900))
        scenarios = (recorded, replace(recorded, scenario_id="f3b7-recorded-peer"))
    inputs = (plan, replay, samples, source, evidence, scenarios, limits)
    before = deepcopy(inputs)
    raw = replay.canonical_bytes(), source.canonical_bytes(), plan.to_dict()
    result = compare(plan, samples, limits, plan.specs[0].soc_min, replay_basis=replay,
        baseline_execution=scenarios[0], fast_execution=scenarios[1], source_evidence=evidence)
    control = compare(plan, samples, limits, plan.specs[0].soc_min, replay_basis=replay,
        baseline_execution=scenarios[0], fast_execution=scenarios[1], source_evidence=source)
    for side, full in zip((result.baseline, result.fast), (control.baseline, control.fast), strict=True):
        assert side.execution_kind == execution_kind and side.coverage_status == "complete", side.reasons
        assert side.trace == full.trace and side.segments == full.segments
        assert side.writes == full.writes and side.soc_end_by_battery == full.soc_end_by_battery
        receipts = [event for event in side.trace if isinstance(event, execution.ReceiptEvent)]
        assert receipts and all(event.receipt.status == "accepted" for event in receipts)
        components = side.financial_result.components
        if source_kind == "full":
            assert side.financial_status == "complete" and components.total_cost.value is not None
            assert components.base_export_revenue.value > 0 and components.bonus_revenue.value > 0
            assert side.financial_result.source_trust == "caller_asserted"
        elif source_kind == "partial":
            # Even a complete ACK history cannot certify PV for the bonus book.
            assert side.financial_status == "partial"
            assert "unknown_bonus_source" in side.financial_reasons
            assert components.base_export_revenue.value > 0
            assert components.bonus_revenue.value is None
            assert components.total_cost.value is None and side.total_cost is None
        else:
            assert side.financial_status == "unknown"
            assert "source_binding_mismatch" in side.financial_reasons
            assert components.base_export_revenue.value is components.total_cost.value is None
    base, fast = control.baseline.financial_result, control.fast.financial_result
    assert base.projection.items is not fast.projection.items
    assert base.settlement.end_state is not fast.settlement.end_state
    assert base.projection.source is fast.projection.source
    if execution_kind == "modeled":
        assert base.settlement.end_state.bonus.total_export_kwh == Decimal("4999")
        assert fast.settlement.end_state.bonus.total_export_kwh == Decimal("5000.5")
        assert control.financial_result.advantage_eur == Decimal(".36")
        reverse = Comparison(control.fast, control.baseline)
        assert reverse.financial_result.advantage_eur == Decimal("-.36") and reverse.saving < 0
    else:
        # Source evidence supplies neither acceptance nor recovery of a lost SoC.
        unacked = replace(scenarios[0], execution=replace(scenarios[0].execution, events=(sent,)))
        unknown = tuple(simulate(plan, samples, limits, plan.specs[0].soc_min, "unacked", False,
            replay_basis=replay, execution=unacked, source_evidence=item) for item in (None, source))
        assert unknown[0].trace == unknown[1].trace and unknown[0].segments == unknown[1].segments
        for side in unknown:
            assert side.coverage_status == "unknown"
            assert side.soc_end_by_battery == ((plan.specs[0].name, None),)
            assert side.total_cost is None
            assert not any(isinstance(event, execution.ReceiptEvent) for event in side.trace)
            assert all(segment.batteries[0].soc_end is None for segment in side.segments)
    if source_kind != "full":
        assert result.financial_result.advantage_eur is None
    assert inputs == before
    assert (replay.canonical_bytes(), source.canonical_bytes(), plan.to_dict()) == raw


def _f3a8_recorded_timeline():
    from dataclasses import replace

    from dao.prog.fastctrl import replay_execution as replay
    from dao.prog.fastctrl.policy import BatteryDecision
    from dao.tests.prog.test_fast_replay_execution import (
        replay_command,
        replay_receipt,
        replay_timing,
        send_event,
        stop_events,
    )

    plan = flat_plan(make_spec(charge_efficiency=.9, discharge_efficiency=.8))
    plan.specs.append(make_spec(name="B", capacity_kwh=20, discharge_efficiency=.8))
    for row in plan.intervals:
        row.batteries.append(BatteryPlanStep(-500, 60, 60))
    first = replay_command(plan, hold_until_ms=(T0 + 10) * 1000)
    other = replace(first, battery_name="B", command_id="opening-B", sequence=4, setpoint_w=-500)
    openings = tuple(replay.OpeningAcceptance(
        cmd, replay_timing(replay, cmd),
        replay_receipt(replay, cmd, T0, apply_ts=T0, order=order, apply_order=order + 1), T0,
    ) for cmd, order in ((first, 1), (other, 3)))
    suppressed = replay.ProposalEvent(T0 + 1, 0, BatteryDecision(
        0, "accu", 1100, 0, False, True, "suppressed"), None, None)
    new = replace(first, command_id="delayed-A", sequence=5, setpoint_w=1200,
                  issued_at_ms=(T0 + 2) * 1000, execute_before_ms=(T0 + 4) * 1000)
    # The attempt budget starts before 700 ms preparation, not at send.
    sent = send_event(replay, new, T0 + 2.7, replay_timing(replay, new, anchor=T0 + 2))
    late = replay_receipt(replay, new, T0 + 2.8, apply_ts=T0 + 2.9,
                          accepted_ts=T0 * 1000 + 2701, anchor=T0 + 2)
    request, barrier = stop_events(replay, request_ts=T0 + 3.1, barrier_ts=T0 + 3.2)
    stale = replay_receipt(replay, new, T0 + 3.3, apply_ts=T0 + 3.4,
                           accepted_ts=T0 * 1000 + 2701, anchor=T0 + 2)
    case = _f3a5_recorded(plan, T0, T0 + 12, (suppressed, sent, late, request, barrier, stale))
    case = replace(case, scenario_id="f3a8-recorded",
                   execution=replace(case.execution, opening_acceptances=openings))
    return plan, [Sample(T0, 0), Sample(T0 + 12, 0)], case


def test_f3a8_public_recorded_timeline_keeps_late_ack_history_and_unknown_energy(monkeypatch):
    from copy import deepcopy

    from dao.prog.fastctrl import replay_execution as replay
    from dao.prog.fastctrl.command import plan_digest
    from dao.prog.fastctrl.receiver import payload_digest, receipt_matches_command

    plan, samples, case = _f3a8_recorded_timeline()
    opening, other = (item.command for item in case.execution.opening_acceptances)
    suppressed, sent, late, request, barrier, stale = case.execution.events
    socs, efficiencies = {"accu": 40, "B": 60}, {"accu": .8, "B": .9}
    before = deepcopy((plan, samples, case, socs, efficiencies))
    payload = sent.command.to_dict(), payload_digest(sent.command)
    snapshots = {}
    snapshot = replay.ExecutionReplay.snapshot

    def observe(self, at_ts):
        result = snapshot(self, at_ts)
        snapshots[at_ts] = result
        return result

    monkeypatch.setattr(replay.ExecutionReplay, "snapshot", observe)
    result = simulate(plan, samples, PolicyLimits(), socs, "recorded chain", True,
                      inverter_efficiency=efficiencies, execution=case)
    assert result.execution_kind == "recorded" and result.coverage_status == "partial"
    assert result.source_plan_digest == result.execution_plan_digest == plan_digest(plan)
    assert result.trace == tuple(item.observation for item in case.execution.opening_acceptances) + case.execution.events
    assert result.writes == 3  # Two opening sends and one new send; write=False adds none.
    for at in (T0, T0 + 1, T0 + 2):
        state = snapshots[at]
        assert state.writes == 2
        assert [(b.command, b.drive_w, b.pending, b.recovery_required) for b in state.batteries] == [
            (opening, 1000, None, False), (other, -500, None, False)]
    assert suppressed.command is suppressed.timing is None
    pending = snapshots[sent.at_ts].batteries[0]
    assert (pending.command, pending.drive_w, pending.pending, pending.execution_status,
            pending.recovery_required) == (None, None, sent.command, "unknown", False)
    assert sent.command.issued_at_ms < replay.wire_ms(sent.at_ts)
    assert snapshots[sent.at_ts].writes == 3
    assert late.receipt.status == stale.receipt.status == "accepted"
    assert late.application.outcome == stale.application.outcome == "accepted"
    assert late.receipt.accepted_ts < sent.command.execute_before_ms
    assert receipt_matches_command(late.receipt, sent.command)
    assert receipt_matches_command(stale.receipt, sent.command)
    # Expiry at anchor+.75 leaves only 50 ms after send. Late acceptance and
    # the stop REQUEST cannot clear that pending/recovery state.
    for at in (T0 + 2.75, late.at_ts, late.application.at_ts, request.at_ts):
        state = snapshots[at].batteries[0]
        assert (state.command, state.drive_w, state.pending, state.execution_status,
                state.recovery_required) == (None, None, sent.command, "unconfirmed", True)
        assert "execution_deadline_exceeded" in state.reasons
        peer = snapshots[at].batteries[1]
        assert (peer.command, peer.drive_w, peer.pending, peer.recovery_required) == (other, -500, None, False)
    for at in (barrier.at_ts, stale.at_ts, stale.application.at_ts, T0 + 4, T0 + 10):
        state = snapshots[at]
        assert state.binding is None
        assert all(b.command is b.pending is None and b.drive_w == 0 and not b.recovery_required
                   for b in state.batteries)
        assert state.batteries[0].execution_status == "stopped"

    prefix = [s for s in result.segments if s.end_ts <= sent.at_ts]
    lost = [s for s in result.segments if sent.at_ts <= s.start_ts < barrier.at_ts]
    stopped = [s for s in result.segments if s.start_ts >= barrier.at_ts]
    assert prefix and lost and stopped
    assert prefix[-1].end_ts == lost[0].start_ts == sent.at_ts
    assert lost[-1].end_ts == stopped[0].start_ts == barrier.at_ts
    assert all(s.batteries[0].command_w == s.batteries[0].modeled_ac_w == 1000 for s in prefix)
    assert sum(s.batteries[0].charge_ac_kwh for s in prefix) == pytest.approx(2.7 / 3600)
    assert sum(s.batteries[0].dc_throughput_kwh for s in prefix) == pytest.approx(2.7 / 3600 * .8)
    assert sum(s.batteries[0].cell_delta_kwh for s in prefix) == pytest.approx(2.7 / 3600 * .72)
    assert sum(s.import_kwh for s in prefix) == pytest.approx(2.7 / 7200)
    assert lost[0].batteries[0].soc_begin == prefix[-1].batteries[0].soc_end
    for segment in lost:
        assert segment.import_kwh is segment.export_kwh is None
        battery = segment.batteries[0]
        for field in ("command", "command_w", "modeled_ac_w", "charge_ac_kwh", "discharge_ac_kwh",
                      "dc_throughput_kwh", "cell_delta_kwh", "soc_end"):
            assert getattr(battery, field) is None
    for segment in stopped:
        battery = segment.batteries[0]
        assert battery.soc_begin is battery.soc_end is None
        assert "replay_soc_missing" in battery.reasons
        assert (battery.command_w, battery.modeled_ac_w, battery.charge_ac_kwh, battery.discharge_ac_kwh,
                battery.dc_throughput_kwh, battery.cell_delta_kwh) == (None, 0, 0, 0, 0, 0)
        assert segment.import_kwh == segment.export_kwh == 0
    assert all(s.batteries[0].soc_end is None for s in (*lost, *stopped))
    assert all(s.batteries[1].soc_begin is not None and s.batteries[1].soc_end is not None for s in result.segments)
    assert sum(s.batteries[1].discharge_ac_kwh for s in result.segments) == pytest.approx(3.2 / 7200)
    assert sum(s.batteries[1].dc_throughput_kwh for s in result.segments) == pytest.approx(3.2 / 7200 / .9)
    assert sum(s.batteries[1].cell_delta_kwh for s in result.segments) == pytest.approx(-3.2 / 7200 / .72)
    assert result.soc_end_by_battery == (("accu", None), ("B", pytest.approx(60 - 3.2 / 7200 / .72 / .2)))
    for field in ("soc_end", "import_kwh", "export_kwh", "throughput_kwh", "energy_cost", "wear_cost",
                  "terminal_value", "total_cost"):
        assert getattr(result, field) is None
    assert (sent.command.to_dict(), payload_digest(sent.command)) == payload
    assert (plan, samples, case, socs, efficiencies) == before


@pytest.mark.parametrize("send_at_start", [False, True])
@pytest.mark.parametrize("safe_stop", [False, True])
@pytest.mark.parametrize("peer", [False, True])
def test_f3a_final_i1_public_opening_missing_application_latches_unknown(send_at_start, safe_stop, peer):
    from copy import deepcopy
    from dataclasses import replace

    from dao.prog.fastctrl import replay_execution as replay
    from dao.tests.prog.test_fast_replay_execution import (
        f3a_final_openings,
        replay_receipt,
        replay_timing,
        stop_events,
    )

    start, end = T0 + .6, T0 + 2
    plan = flat_plan(make_spec(charge_efficiency=1, discharge_efficiency=1))
    if peer:
        plan.specs.append(make_spec(name="B", capacity_kwh=20, charge_efficiency=1, discharge_efficiency=1))
        for row in plan.intervals:
            row.batteries.append(BatteryPlanStep(-500, 60, 60))
    openings = f3a_final_openings(replay, plan, start if send_at_start else T0 + .5)
    if peer:
        other = replace(openings[0].command, command_id="final-opening-B", sequence=2,
                        battery_name="B", setpoint_w=-500)
        openings = (replay.OpeningAcceptance(other, replay_timing(replay, other),
            replay_receipt(replay, other, T0 + .05, apply_ts=T0 + .05), T0), *openings)
    late = replay_receipt(replay, openings[-1].command, T0 + 1.1, apply_ts=T0 + 1.1)
    request, barrier = stop_events(replay, request_ts=T0 + 1.2, barrier_ts=T0 + 1.3)
    stale = replay_receipt(replay, openings[-1].command, T0 + 1.4, apply_ts=T0 + 1.4)
    events = (late, request, barrier, stale) if safe_stop else (late,)
    case = _f3a5_recorded(plan, start, end, events)
    case = replace(case, execution=replace(case.execution, opening_acceptances=openings))
    samples = [Sample(start, 0), Sample(end, 0)]
    socs = {"accu": 40, "B": 60} if peer else {"accu": 40}
    before = deepcopy((plan, samples, socs, case))
    result = simulate(plan, samples, PolicyLimits(), socs, "opening without application", True,
                      inverter_efficiency=1, execution=case)

    lost = [s for s in result.segments if not safe_stop or s.start_ts < barrier.at_ts]
    stopped = [s for s in result.segments if safe_stop and s.start_ts >= barrier.at_ts]
    assert lost and bool(stopped) is safe_stop
    assert lost[0].start_ts == start
    assert lost[-1].end_ts == (barrier.at_ts if safe_stop else end)
    assert result.coverage_status == ("partial" if peer or safe_stop else "unknown")
    assert result.execution_kind == "recorded" and result.writes == int(send_at_start)
    assert result.trace == (*[item.observation for item in openings], *events)
    assert lost[0].batteries[0].soc_begin == 40
    for segment in lost:
        battery = segment.batteries[0]
        assert battery.execution_status == ("unknown" if segment.start_ts < T0 + 1 else "unconfirmed")
        assert "execution_application_unproven" in battery.reasons
        for name in ("command", "command_w", "modeled_ac_w", "charge_ac_kwh", "discharge_ac_kwh",
                     "dc_throughput_kwh", "cell_delta_kwh", "soc_end"):
            assert getattr(battery, name) is None
        assert segment.import_kwh is segment.export_kwh is None
    for segment in stopped:
        battery = segment.batteries[0]
        assert battery.execution_status == "stopped" and battery.command is None
        assert battery.soc_begin is battery.soc_end is None
        assert (battery.modeled_ac_w, battery.charge_ac_kwh, battery.discharge_ac_kwh,
                battery.dc_throughput_kwh, battery.cell_delta_kwh) == (0, 0, 0, 0, 0)
        assert segment.import_kwh == segment.export_kwh == 0
    assert result.soc_end_by_battery[0] == ("accu", None)
    assert "execution_deadline_exceeded" in result.reasons
    if peer:
        duration = (barrier.at_ts if safe_stop else end) - start
        discharge = duration / 7200
        assert sum(s.batteries[1].discharge_ac_kwh for s in result.segments) == pytest.approx(discharge)
        assert result.soc_end_by_battery[1] == ("B", pytest.approx(60 - discharge / .2))
        assert all(s.batteries[1].soc_end is not None for s in result.segments)
    for name in ("soc_end", "import_kwh", "export_kwh", "throughput_kwh", "energy_cost",
                 "wear_cost", "terminal_value", "total_cost"):
        assert getattr(result, name) is None
    assert (plan, samples, socs, case) == before


@pytest.mark.parametrize("sequences", [(4, 3), (3, 4)])
def test_f3a_final_i2_public_opening_sequences_cannot_be_sorted_into_success(sequences):
    from dataclasses import replace

    from dao.prog.fastctrl import replay_execution as replay
    from dao.tests.prog.test_fast_replay_execution import (
        replay_command,
        replay_receipt,
        replay_timing,
    )

    plan = flat_plan(make_spec(charge_efficiency=1))
    first = replay_command(plan, sequence=sequences[0])
    second = replace(first, command_id="final-public-second", sequence=sequences[1], setpoint_w=1200)
    openings = tuple(replay.OpeningAcceptance(cmd, replay_timing(replay, cmd),
        replay_receipt(replay, cmd, T0 + at, apply_ts=T0 + at), T0 + at)
        for cmd, at in ((first, .1), (second, .3)))
    case = _f3a5_recorded(plan, T0 + .5, T0 + 1.5, ())
    case = replace(case, execution=replace(case.execution, opening_acceptances=openings))
    samples = [Sample(T0 + .5, 0), Sample(T0 + 1.5, 0)]
    if sequences == (4, 3):
        with pytest.raises(ValueError, match="^replay_duplicate_identity$"):
            simulate(plan, samples, PolicyLimits(), 40, "descending opening", False, execution=case)
    else:
        result = simulate(plan, samples, PolicyLimits(), 40, "increasing opening", False,
                          inverter_efficiency=1, execution=case)
        assert result.coverage_status == "complete" and result.writes == 0
        assert all(s.batteries[0].command == second for s in result.segments)
        assert result.import_kwh == pytest.approx(1.2 / 3600)
        assert result.soc_end_by_battery == (("accu", pytest.approx(40 + 1.2 / 360)),)


def test_f3a7_missing_execution_is_unknown_not_success(caplog):
    import logging

    from dao.prog.fastctrl.simulate import log_comparison

    plan = flat_plan(make_spec(), battery_w=1000)
    samples = [Sample(T0, 500), Sample(T0 + 60, 500)]
    result = simulate(plan, samples, PolicyLimits(), 50, "old positional caller", False, 1, None)
    assert result.label.startswith("unknown_reconstruction:")
    assert result.execution_kind == "missing" and result.coverage_status == "unknown"
    assert "execution_input_missing" in result.reasons
    for attribute in ("import_kwh", "export_kwh", "throughput_kwh", "soc_end", "peak_import_w",
                      "energy_cost", "wear_cost", "terminal_value", "total_cost", "cycles", "average_soc"):
        assert getattr(result, attribute) is None
    assert result.soc_start_by_battery == (("accu", 50),)
    assert result.duration_h == pytest.approx(1 / 60)
    assert result.writes == 0 and result.trace == ()
    assert result.segments and result.days
    comparison = compare(plan, samples, PolicyLimits(), 50, 1, None)
    assert comparison.extra_cycles is None
    for output in (comparison.saving, comparison.saving_per_day,
                   comparison.report(), comparison.daily_table()):
        assert output.startswith("unknown_reconstruction:")
        assert "Besparing:" not in output and "euro" not in output
    with caplog.at_level(logging.INFO):
        log_comparison(comparison)
    assert "execution_input_missing" in caplog.text


@pytest.mark.parametrize("caller", ["demo", "simulate"])
@pytest.mark.parametrize("gap", [False, True])
def test_f3a7_callers_are_explicit_models_and_unknown_reports_are_safe(monkeypatch, caplog, caller, gap):
    import logging
    import sys
    from types import SimpleNamespace

    from dao.prog import da_fast
    from dao.prog.fastctrl.replay_execution import ModeledExecution
    from dao.prog.fastctrl.simulate import HistoryWindow

    cadence = 60 if caller == "demo" else 30
    plan = flat_plan(make_spec(name="totaal"))
    samples = [Sample(T0, 500), Sample(T0 + cadence, 500),
               Sample(T0 + cadence * (3 if gap else 2), 500)]
    window = HistoryWindow(plan, samples, 50, cadence, "synthetic recorder fixture",
                           ["Aangenomen opening-SoC; ontbrekende recorderbron"])
    config = SimpleNamespace(fast_control=SimpleNamespace(grid_power=object(), batteries=[]),
                             time_zone="Europe/Amsterdam", interval="15min")
    loaded, reports, compared = [], [], []
    report = SimpleNamespace(config=config, ha_getter=object(), db_ha=object(), db_da=object())

    def make_report(options):
        reports.append(options)
        return report

    class Loader:
        def __init__(self, db_ha, db_da, actual_report, zone):
            assert (db_ha, db_da, actual_report, zone) == (
                report.db_ha, report.db_da, report, config.time_zone)

        def load(self, **kwargs):
            loaded.append(kwargs)
            return window

    def compare_spy(*args, **kwargs):
        baseline = kwargs.get("baseline_execution")
        fast = kwargs.get("fast_execution")
        assert baseline is not None and fast is not None, "caller omitted explicit model scenarios"
        assert baseline is not fast and baseline.scenario_id != fast.scenario_id
        for scenario in (baseline, fast):
            assert isinstance(scenario.execution, ModeledExecution)
            model, physics = scenario.execution, scenario.physics
            assert (model.initial_execution, model.default_effect) == ("stopped", "complete")
            assert (model.prepare_delay_s, model.response_delay_s, model.application_delay_s) == (0, 0, 0)
            assert (physics.version, physics.house_hold_s, physics.stop_response) == (
                "battery_model_v1", cadence, "ideal_programmed_stop")
        result = compare(*args, **kwargs)
        compared.append(result)
        return result

    monkeypatch.setitem(sys.modules, "da_report", SimpleNamespace(Report=make_report))
    monkeypatch.setattr(da_fast, "HistoryLoader", Loader)
    monkeypatch.setattr(da_fast, "_spec_from_config", lambda *args: plan.specs[0])
    monkeypatch.setattr(da_fast, "_limits_from_config", lambda *args: PolicyLimits())
    monkeypatch.setattr(da_fast, "_entity_scales", lambda sensor: [("sensor.grid", 1)])
    monkeypatch.setattr(da_fast, "synthetic_case", lambda *args, **kwargs: (plan, samples, 50))
    monkeypatch.setattr(da_fast, "compare", compare_spy)
    args = da_fast.build_parser().parse_args([caller, *(["--step", "30"] if caller == "simulate" else [])])
    with caplog.at_level(logging.INFO):
        assert getattr(da_fast, f"cmd_{caller}")(args) == 0
    assert len(compared) == 1
    for result in (compared[0].baseline, compared[0].fast):
        assert result.execution_kind == "modeled"
        assert result.soc_start_by_battery == (("totaal", 50),)
        assert (result.total_cost is None) is gap
    assert "execution_kind=modeled" in caplog.text
    for assumption in ("house_hold_s", "ideal_programmed_stop", "fresh_policy_start",
                       "prepare_delay_s", "response_delay_s", "application_delay_s"):
        assert assumption in caplog.text
    if gap:
        assert "unknown_reconstruction:" in caplog.text
        assert "Besparing:" not in caplog.text
    if caller == "simulate":
        assert reports == [args.options] and len(loaded) == 1
        assert loaded[0]["step_s"] == 30 and loaded[0]["spec"] is plan.specs[0]
        assert window.warnings[0] in caplog.text
        assert "wat-als" in caplog.text.lower() and "aggregaatmodel" in caplog.text.lower()
        assert "begin-SoC=50" in caplog.text
    else:
        assert reports == loaded == []


def test_f3a7_compare_isolates_states_and_never_reuses_recorded_ack():
    from copy import deepcopy

    from dao.prog.fastctrl import replay_execution as replay
    from dao.tests.prog.test_fast_replay_execution import (
        replay_command,
        replay_receipt,
        replay_timing,
        send_event,
    )

    plan = flat_plan(make_spec(charge_efficiency=1, discharge_efficiency=1), battery_w=1000)
    samples = [Sample(T0, 0), Sample(T0 + 60, 0), Sample(T0 + 120, 0)]
    opening, efficiencies = {"accu": 50}, {"accu": 1}
    before = deepcopy((plan, samples, opening, efficiencies))
    kwargs = dict(inverter_efficiency=efficiencies, baseline_execution=modeled_case("baseline"),
                  fast_execution=modeled_case("fast"))
    first = compare(plan, samples, PolicyLimits(), opening, **kwargs)
    repeated = compare(plan, samples, PolicyLimits(), opening, **kwargs)
    assert first == repeated
    assert first.baseline is not first.fast and first.baseline.days is not first.fast.days
    commands = [{event.command.command_id for event in result.trace
                 if isinstance(event, replay.ProposalEvent) and event.command is not None}
                for result in (first.baseline, first.fast)]
    assert all(commands) and commands[0].isdisjoint(commands[1])
    assert first.baseline.soc_end == pytest.approx(50 + 1 / 3)
    assert first.fast.soc_start_by_battery == first.baseline.soc_start_by_battery == (("accu", 50),)
    assert (plan, samples, opening, efficiencies) == before

    command = replay_command(plan, hold_until_ms=(T0 + 120) * 1000)
    events = (send_event(replay, command, T0, replay_timing(replay, command)),
              replay_receipt(replay, command, T0, apply_ts=T0))
    recorded = _f3a5_recorded(plan, T0, T0 + 120, events)
    comparison = compare(plan, samples, PolicyLimits(), opening,
                         inverter_efficiency=efficiencies, baseline_execution=recorded)
    assert comparison.baseline.trace == events and comparison.baseline.writes == 1
    assert comparison.baseline.import_kwh == pytest.approx(1 / 30)
    assert comparison.fast.execution_kind == "missing"
    assert comparison.fast.trace == () and comparison.fast.import_kwh is None
    modeled = compare(plan, samples, PolicyLimits(), opening, inverter_efficiency=efficiencies,
                      baseline_execution=recorded, fast_execution=modeled_case("alternative"))
    assert modeled.baseline == comparison.baseline
    assert all(event.command != command for event in modeled.fast.trace
               if isinstance(event, replay.ProposalEvent))
    # Selecting fast policy cannot rewrite recorded execution into a counterfactual.
    same_recording = simulate(plan, samples, PolicyLimits(), opening, "recorded", True,
                              inverter_efficiency=efficiencies, execution=recorded)
    assert same_recording.segments == comparison.baseline.segments
    assert same_recording.trace == events


def test_f3a7_existing_simulator_no_write_uses_previous_command(monkeypatch):
    from dao.prog.fastctrl.policy import (
        AcceptedCommand,
        BatteryDecision,
        Decision,
        FastControlPolicy,
    )

    proposals = []

    def decide(policy, plan, measurement, state, **kwargs):
        first = measurement.timestamp == T0
        proposal = BatteryDecision(
            0, "accu", 1000 if first else 1100, 0, first, True, "no-write probe",
            mode="Aan", soc=50,
            accepted=AcceptedCommand(1000, T0, True) if first else None,
        )
        proposals.append((proposal.setpoint_w, proposal.write))
        return Decision(measurement.timestamp, batteries=[proposal])

    monkeypatch.setattr(FastControlPolicy, "decide", decide)
    plan = flat_plan(make_spec(charge_efficiency=1, discharge_efficiency=1))
    result = simulate(plan, [Sample(T0, 0), Sample(T0 + 60, 0), Sample(T0 + 120, 0)],
                      PolicyLimits(), 50, "no-write probe", True, inverter_efficiency=1,
                      execution=modeled_case("no-write"))
    assert proposals == [(1000, True), (1100, False)]
    # Both minutes use the accepted 1000 W, never the suppressed 1100 W.
    assert result.import_kwh == pytest.approx(1 / 30)
    assert result.export_kwh == 0
    assert result.throughput_kwh == pytest.approx(1 / 30)
    assert result.soc_end == pytest.approx(50 + 1 / 3)
    assert [segment.batteries[0].command_w for segment in result.segments] == [1000, 1000]
    from dao.prog.fastctrl.replay_execution import ProposalEvent

    suppressed = [event for event in result.trace if isinstance(event, ProposalEvent)
                  and event.at_ts == T0 + 60]
    assert len(suppressed) == 1 and suppressed[0].command is None


def test_f3a7_existing_simulator_does_not_net_battery_throughput():
    from dataclasses import replace

    spec = make_spec(charge_efficiency=1, discharge_efficiency=1)
    plan = flat_plan(spec, battery_w=1000)
    plan.specs.append(replace(spec, name="second"))
    for row in plan.intervals:
        row.batteries.append(BatteryPlanStep(-1000, 50, 50))
    result = simulate(plan, [Sample(T0, 0), Sample(T0 + 900, 0)], PolicyLimits(),
                      50, "opposing batteries", False, inverter_efficiency=1,
                      execution=modeled_case("opposing", hold=900, equal_initial_soc=True))
    # Each battery moves .25 kWh, even though the site exchanges nothing.
    assert result.throughput_kwh == pytest.approx(.5)
    assert result.import_kwh == result.export_kwh == 0
    assert result.wear_cost == pytest.approx(.005)
    assert result.soc_end == pytest.approx(50)
    assert result.soc_end_by_battery == (("accu", 52.5), ("second", 47.5))


def f2b_fixed_synthetic_datetime():
    """Freeze only this module's date/local-zone dependency, not process clocks."""
    import datetime as dt
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("Europe/Amsterdam")

    class FixedDate(dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 4)

    class LocalDatetime(dt.datetime):
        @classmethod
        def combine(cls, date, time, tzinfo=zone):
            return super().combine(date, time, tzinfo)

        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            return super().fromtimestamp(timestamp, zone if tz is None else tz)

    return SimpleNamespace(datetime=LocalDatetime, date=FixedDate,
                           time=dt.time, timedelta=dt.timedelta)


def test_f2b_r17_two_day_synthetic_explicit_model_is_balanced(monkeypatch):
    from importlib import import_module

    module = import_module("dao.prog.fastctrl.simulate")
    monkeypatch.setattr(module, "datetime", f2b_fixed_synthetic_datetime())
    plan, samples, soc = synthetic_case(make_spec(), days=2)
    assert plan.created_ts == 1793574000
    scenario = modeled_case("R17 explicit model")
    result = simulate(plan, samples, PolicyLimits(), soc, "R17 model", True, execution=scenario)
    assert result.coverage_status == "complete" and result.duration_h == 48
    house_kwh = sum(sample.house_w * (following.timestamp - sample.timestamp) / 3600000
                    for sample, following in zip(samples, samples[1:], strict=False))
    charge = sum(b.charge_ac_kwh for segment in result.segments for b in segment.batteries)
    discharge = sum(b.discharge_ac_kwh for segment in result.segments for b in segment.batteries)
    assert result.import_kwh - result.export_kwh == pytest.approx(house_kwh + charge - discharge)
    assert result.throughput_kwh == pytest.approx(charge * .95 + discharge / .95)
    assert (result.soc_end - soc) * .1 == pytest.approx(charge * .95 * .95 - discharge / (.95 * .95))
    assert all(20 <= b.soc_end <= 95 for segment in result.segments for b in segment.batteries)
    assert result.import_kwh > 0 and result.export_kwh > 0 and result.writes > 0


@pytest.mark.parametrize("min_command_interval", [0, 60])
def test_f2b_simulator_applies_explicit_model_acceptance_between_decisions_without_io(monkeypatch, min_command_interval):
    from dao.prog.fastctrl.policy import FastControlPolicy

    observations = []
    original = FastControlPolicy.decide

    def observe(policy, plan, measurement, state, *args, **kwargs):
        controller = state.battery(0)
        before = (controller.last_command_w, controller.last_command_ts)
        decision = original(policy, plan, measurement, state, *args, **kwargs)
        after = (controller.last_command_w, controller.last_command_ts)
        observations.append((before, after))
        return decision

    monkeypatch.setattr(FastControlPolicy, "decide", observe)
    plan = flat_plan(make_spec(charge_efficiency=1, discharge_efficiency=1),
                     price=0.4, export=0.1)
    result = simulate(plan, [Sample(T0, 1000), Sample(T0 + 60, 1000), Sample(T0 + 120, 1000)],
                      PolicyLimits(min_benefit=0, deadband=150, min_command_interval=min_command_interval),
                      80, "acceptance", True,
                      inverter_efficiency=1, execution=modeled_case("acceptance"))
    # The explicit opening zero command participates in the real rate limit.
    # With no rate limit, retain the original two-minute physical oracle.
    previous = 0 if min_command_interval else -1000
    assert observations == [((0, T0), (0, T0)), ((previous, T0), (previous, T0))]
    if min_command_interval:
        assert result.import_kwh == pytest.approx(1 / 60)
        assert result.throughput_kwh == pytest.approx(1 / 60)
        assert result.soc_end == pytest.approx(80 - 1 / 6)
    else:
        assert result.import_kwh == 0
        assert result.throughput_kwh == pytest.approx(1 / 30)
        assert result.soc_end == pytest.approx(80 - 1 / 3)
    assert result.export_kwh == 0
    assert result.writes == 2


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_fix1_simulation_preserves_interval_soc_in_real_storage_valuation(monkeypatch, multiple, split):
    from dataclasses import replace
    from importlib import import_module

    from dao.tests.settlement.test_fast_contract import ready_plan

    policy_module = import_module("dao.prog.fastctrl.policy")
    plan = ready_plan()
    plan.price_average = 0.2
    if multiple:
        plan.specs.append(replace(plan.specs[0], name="second", capacity_kwh=20))
        context = plan.financial_context
        plan.financial_context = replace(context, terminal_value=replace(context.terminal_value,
            reference_efficiency_by_battery=(*context.terminal_value.reference_efficiency_by_battery,
                                             ("second", context.terminal_value.reference_efficiency_by_battery[0][1]))))
    for index, (row, price, pair) in enumerate(zip(plan.intervals, [0.1, 0.4, 0.8],
                                                  [(80, 70), (70, 60), (60, 40)], strict=True)):
        row.price_import, row.price_export = price, 0.01
        step = row.battery(0)
        step.soc_begin, step.soc_end = pair
        if index == 0:
            step.ac_power_w = step.mean_ac_power_w = 1000
            step.on_power_w = 2000 if split else 1000
            step.stop_ts = row.start_ts + 450 if split else None
        if multiple:
            begin, end = [(60, 50), (50, 40), (40, 30)][index]
            other = BatteryPlanStep(0, begin, end, mean_ac_power_w=0, on_power_w=0,
                                    active_from_ts=row.start_ts, stop_ts=None)
            plan.intervals[index] = replace(row, batteries=[step, other])
    plan = FastPlan.from_dict(plan.to_dict())
    observed = []
    original = policy_module.estimate_storage_value

    def observe_value(sim_plan, now, index, limits):
        value = original(sim_plan, now, index, limits)
        observed.append((value, [[(step.soc_begin, step.soc_end) for step in row.batteries]
                                 for row in sim_plan.intervals], sim_plan.financial_context))
        return value

    monkeypatch.setattr(policy_module, "estimate_storage_value", observe_value)
    result = simulate(plan, [Sample(plan.intervals[0].start_ts, 1000), Sample(plan.horizon_end_ts, 1000)],
                      PolicyLimits(), 70, "SOC preservation", use_fast_layer=True,
                      execution=modeled_case("soc-preservation", hold=2700, equal_initial_soc=multiple))
    # With >10% capacity slack above the floor, the existing valuation is the
    # surplus 0.2 * 0.9 = 0.18, not the zero-SOC first-window scarcity value 0.09.
    assert observed[0][0] == pytest.approx(0.18)
    expected = [[(80, 70)], [(70, 60)], [(60, 40)]]
    if multiple:
        for row, pair in zip(expected, [(60, 50), (50, 40), (40, 30)], strict=True):
            row.append(pair)
    assert all(rows == expected and context == plan.financial_context for _, rows, context in observed)
    assert result.duration_h == pytest.approx(0.75)


def test_fix1_nonzero_profiles_with_different_windows_keep_shape_and_energy(monkeypatch):
    from dataclasses import replace

    from dao.prog.fastctrl.policy import FastControlPolicy
    from dao.tests.settlement.test_fast_contract import ready_plan

    plan = ready_plan()
    plan.specs.append(replace(plan.specs[0], name="second"))
    context = plan.financial_context
    # This is a full-interval physical model; ready_plan's 600s context would
    # correctly stop both batteries before the last 300s profile slice.
    plan.financial_context = replace(context, not_after_ts=plan.horizon_end_ts,
        terminal_value=replace(context.terminal_value,
        reference_efficiency_by_battery=(*context.terminal_value.reference_efficiency_by_battery,
                                         ("second", context.terminal_value.reference_efficiency_by_battery[0][1]))))
    a = plan.intervals[0].start_ts
    for index, row in enumerate(plan.intervals):
        first = row.battery(0)
        second = BatteryPlanStep(0, 50, 50, mean_ac_power_w=0, on_power_w=0,
                                 active_from_ts=row.start_ts, stop_ts=None)
        if index == 0:
            first.ac_power_w = first.mean_ac_power_w = 4000 / 3
            first.on_power_w, first.stop_ts = 2000, a + 600
            second.ac_power_w = second.mean_ac_power_w = 2000 / 3
            second.on_power_w, second.active_from_ts = 1000, a + 300
        plan.intervals[index] = replace(row, batteries=[first, second])
    plan = FastPlan.from_dict(plan.to_dict())
    windows = []
    original = FastControlPolicy.account

    def observe_account(policy, state, sim_plan, measurement, day_key, *args, **kwargs):
        windows.append(sim_plan.to_dict())
        return original(policy, state, sim_plan, measurement, day_key, *args, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "account", observe_account)
    result = simulate(plan, [Sample(a, -1500), Sample(a + 900, -1500)],
                      PolicyLimits(min_benefit=1e9), 50, "two live profiles", use_fast_layer=True,
                      inverter_efficiency=1,
                      execution=modeled_case("two profiles", hold=900, equal_initial_soc=True))
    assert windows and all(window == plan.to_dict() for window in windows)
    assert [(s.start_ts, s.end_ts, tuple(b.modeled_ac_w for b in s.batteries)) for s in result.segments] == [
        (a, a + 300, (2000, 0)),
        (a + 300, a + 600, (2000, 1000)),
        (a + 600, a + 900, (0, 1000)),
    ]
    assert result.import_kwh == pytest.approx(1 / 6)
    assert result.export_kwh == pytest.approx(1 / 24)
    assert result.throughput_kwh == pytest.approx(0.5)


@pytest.mark.parametrize("multiple", [False, True])
def test_simulator_integrates_profile_edges_inside_a_sample_without_mean_fallback(multiple):
    from dataclasses import replace

    from dao.tests.settlement.test_fast_profile import shaped_plan

    plan = shaped_plan()
    interval = plan.intervals[0]
    if multiple:
        plan.specs.append(replace(plan.specs[0], name="second"))
        context = plan.financial_context
        plan.financial_context = replace(context, terminal_value=replace(context.terminal_value,
            reference_efficiency_by_battery=(*context.terminal_value.reference_efficiency_by_battery,
                                             ("second", context.terminal_value.reference_efficiency_by_battery[0][1]))))
        for index, row in enumerate(plan.intervals):
            other = BatteryPlanStep(0, 50, 50, mean_ac_power_w=0, on_power_w=0,
                                    active_from_ts=row.start_ts, stop_ts=None)
            plan.intervals[index] = replace(row, batteries=[*row.batteries, other])
    plan = FastPlan.from_dict(plan.to_dict())
    samples = [Sample(interval.start_ts, -500), Sample(interval.end_ts, -500)]
    result = simulate(plan, samples, PolicyLimits(), 50, "profile", use_fast_layer=False,
                      inverter_efficiency=1,
                      execution=modeled_case("profile", hold=900, equal_initial_soc=multiple))
    # 1500 W import for 450s, then 500 W export for 450s (not net 500 W constant).
    assert result.import_kwh == pytest.approx(0.1875)
    assert result.export_kwh == pytest.approx(0.0625)
    assert result.throughput_kwh == pytest.approx(0.25)


def make_spec(**overrides) -> BatterySpec:
    defaults = dict(
        name="accu",
        capacity_kwh=10.0,
        max_charge_w=5000.0,
        max_discharge_w=5000.0,
        soc_min=20.0,
        soc_max=95.0,
        cycle_cost=0.01,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    defaults.update(overrides)
    return BatterySpec(**defaults)


def flat_plan(spec, price=0.30, export=0.10, battery_w=0.0, hours=2, house_w=500.0):
    intervals = [
        PlanInterval(
            start_ts=T0 + i * 900,
            end_ts=T0 + (i + 1) * 900,
            price_import=price,
            price_export=export,
            grid_w=house_w + battery_w,
            house_w=house_w,
            batteries=[
                BatteryPlanStep(ac_power_w=battery_w, soc_begin=50.0, soc_end=50.0)
            ],
        )
        for i in range(hours * 4)
    ]
    return FastPlan(
        created_ts=T0,
        interval_s=900,
        specs=[spec],
        intervals=intervals,
        price_average=price,
    )


class TestBatteryModel:
    def test_charging_respects_the_efficiency_chain(self):
        spec = make_spec(charge_efficiency=0.90)
        model = BatteryModel(spec, soc=50.0, inverter_efficiency=1.0)
        power, dc_kwh = model.step(1000.0, 1.0)
        assert power == 1000.0
        assert dc_kwh == pytest.approx(1.0)
        # 1 kWh DC times 0.9 into 10 kWh is 9 percentage points.
        assert model.soc == pytest.approx(59.0)

    def test_discharging_draws_more_than_it_delivers(self):
        spec = make_spec(discharge_efficiency=0.90)
        model = BatteryModel(spec, soc=50.0, inverter_efficiency=1.0)
        power, dc_kwh = model.step(-900.0, 1.0)
        assert power == -900.0
        # 0.9 kWh AC out of a 90 percent efficient battery costs 1 kWh of cells.
        assert model.soc == pytest.approx(40.0)
        assert dc_kwh == pytest.approx(0.9)

    def test_it_stops_at_the_upper_limit(self):
        model = BatteryModel(make_spec(), soc=94.0, inverter_efficiency=1.0)
        power, _ = model.step(5000.0, 1.0)
        assert model.soc == pytest.approx(95.0)
        assert 0 < power < 5000.0

    def test_it_stops_at_the_lower_limit(self):
        model = BatteryModel(make_spec(), soc=21.0, inverter_efficiency=1.0)
        power, _ = model.step(-5000.0, 1.0)
        assert model.soc == pytest.approx(20.0)
        assert -5000.0 < power < 0

    def test_it_never_exceeds_the_inverter_rating(self):
        model = BatteryModel(make_spec(max_charge_w=3000.0), soc=50.0)
        power, _ = model.step(9000.0, 0.01)
        assert power == 3000.0

    def test_an_empty_battery_delivers_nothing(self):
        model = BatteryModel(make_spec(), soc=20.0)
        power, dc_kwh = model.step(-3000.0, 0.25)
        assert power == 0.0
        assert dc_kwh == 0.0


class TestSimulation:
    def test_a_constant_load_reproduces_the_hand_computed_cost(self):
        spec = make_spec()
        plan = flat_plan(spec, price=0.40, export=0.10, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 1000.0) for s in range(61)]
        result = simulate(
            plan, samples, PolicyLimits(), 50.0, "base", use_fast_layer=False,
            execution=modeled_case("constant-load"),
        )
        # 1000 W for one hour at 0.40 euro/kWh.
        assert result.energy_cost == pytest.approx(0.40, rel=1e-6)
        assert result.import_kwh == pytest.approx(1.0, rel=1e-6)
        assert result.throughput_kwh == 0.0

    def test_the_fast_layer_covers_the_load_from_the_battery(self):
        spec = make_spec()
        plan = flat_plan(spec, price=0.40, export=0.10, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 1000.0) for s in range(61)]
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=5.0)
        result = simulate(
            plan, samples, limits, 80.0, "fast", use_fast_layer=True,
            execution=modeled_case("cover-load"),
        )
        assert result.import_kwh < 0.1
        assert result.energy_cost < 0.05
        assert result.throughput_kwh > 0.9

    def test_the_comparison_prices_the_leftover_energy(self):
        """Ending emptier may not look free."""
        spec = make_spec()
        plan = flat_plan(spec, price=0.40, export=0.10, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 1000.0) for s in range(61)]
        result = compare(
            plan,
            samples,
            PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=5.0),
            80.0,
            baseline_execution=modeled_case("baseline"), fast_execution=modeled_case("fast"),
        )
        assert result.fast.soc_end < result.baseline.soc_end
        assert result.fast.terminal_value < result.baseline.terminal_value
        # The raw energy cost drops, but almost all of it is paid back by the
        # energy the battery gave up.
        raw = result.baseline.energy_cost - result.fast.energy_cost
        assert raw > 0.35
        assert result.saving < 0.05 * raw

    def test_a_flat_tariff_leaves_nothing_to_win(self):
        """Self-consumption only pays when prices move or the spread is real.

        With a flat tariff over the whole horizon, discharging now and
        discharging later are worth exactly the same, so the fast layer can
        only lose the wear it causes. The backtest must show that honestly
        instead of booking the avoided import as profit.
        """
        spec = make_spec(cycle_cost=0.02)
        plan = flat_plan(spec, price=0.40, export=0.10, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 1000.0) for s in range(61)]
        result = compare(
            plan,
            samples,
            PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=5.0),
            80.0,
            baseline_execution=modeled_case("baseline"), fast_execution=modeled_case("fast"),
        )
        assert result.saving < 0.0
        assert abs(result.saving) == pytest.approx(
            result.fast.wear_cost, abs=0.03
        )

    def test_the_cost_identity_holds(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=1)
        result = compare(plan, samples, PolicyLimits(), soc,
                         baseline_execution=modeled_case("baseline"), fast_execution=modeled_case("fast"))
        for outcome in (result.baseline, result.fast):
            assert outcome.total_cost == pytest.approx(
                outcome.energy_cost + outcome.wear_cost - outcome.terminal_value
            )

    def test_wear_is_charged(self):
        spec = make_spec(cycle_cost=0.05)
        plan = flat_plan(spec, price=0.40, battery_w=0.0, hours=1)
        samples = [Sample(T0 + s * 60, 2000.0) for s in range(61)]
        limits = PolicyLimits(min_benefit=0.0, deadband=0.0, energy_budget=5.0)
        result = simulate(plan, samples, limits, 80.0, "fast", use_fast_layer=True,
                          execution=modeled_case("wear"))
        assert result.wear_cost == pytest.approx(
            0.05 * result.throughput_kwh, rel=1e-9
        )

    def test_cycles_are_counted_over_twice_the_capacity(self):
        spec = make_spec(capacity_kwh=10.0)
        plan = flat_plan(spec, hours=1)
        samples = [Sample(T0 + s * 60, 0.0) for s in range(61)]
        result = simulate(
            plan, samples, PolicyLimits(), 50.0, "base", use_fast_layer=False,
            execution=modeled_case("cycles"),
        )
        result.throughput_kwh = 20.0
        assert result.cycles == pytest.approx(1.0)

    def test_too_few_samples_is_an_error(self):
        plan = flat_plan(make_spec())
        with pytest.raises(ValueError):
            simulate(
                plan, [Sample(T0, 0.0)], PolicyLimits(), 50.0, "x", use_fast_layer=False
            )

    def test_a_plan_without_a_battery_is_an_error(self):
        plan = FastPlan(created_ts=T0, interval_s=900, specs=[], intervals=[])
        with pytest.raises(ValueError):
            simulate(plan, [], PolicyLimits(), 50.0, "x", use_fast_layer=False)


class TestAggregateSpec:
    def test_a_single_battery_is_passed_through(self):
        spec = make_spec()
        assert aggregate_spec([spec]) is spec

    def test_power_and_capacity_add_up(self):
        first = make_spec(capacity_kwh=10.0, max_charge_w=3000.0)
        second = make_spec(capacity_kwh=5.0, max_charge_w=2000.0)
        merged = aggregate_spec([first, second])
        assert merged.capacity_kwh == 15.0
        assert merged.max_charge_w == 5000.0

    def test_efficiencies_are_capacity_weighted(self):
        first = make_spec(capacity_kwh=10.0, charge_efficiency=0.90)
        second = make_spec(capacity_kwh=30.0, charge_efficiency=1.00)
        merged = aggregate_spec([first, second])
        assert merged.charge_efficiency == pytest.approx(0.975)

    def test_an_empty_list_is_an_error(self):
        with pytest.raises(ValueError):
            aggregate_spec([])


class TestSyntheticCase:
    def test_it_produces_a_usable_scenario(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=1)
        assert len(plan.intervals) == 96
        assert len(samples) > 1400
        assert 0 <= soc <= 100
        assert all(i.end_ts > i.start_ts for i in plan.intervals)

    def test_the_appliance_peaks_are_invisible_to_the_plan(self):
        """That invisibility is the whole reason the fast layer exists."""
        spec = make_spec()
        plan, samples, _ = synthetic_case(spec, days=1)
        deviations = []
        for sample in samples[:-1]:
            interval = plan.interval_at(sample.timestamp)
            if interval is not None:
                deviations.append(abs(sample.house_w - interval.house_w))
        assert max(deviations) > 2000.0

    def test_the_fast_layer_saves_money_on_it(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=2)
        result = compare(plan, samples, PolicyLimits(), soc,
                         baseline_execution=modeled_case("baseline"), fast_execution=modeled_case("fast"))
        assert result.saving > 0.0
        assert result.fast.import_kwh < result.baseline.import_kwh
        assert result.fast.export_kwh < result.baseline.export_kwh

    def test_a_tight_energy_budget_reduces_both_saving_and_wear(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=2)
        loose = compare(plan, samples, PolicyLimits(energy_budget=2.0), soc,
                        baseline_execution=modeled_case("baseline"), fast_execution=modeled_case("fast"))
        tight = compare(plan, samples, PolicyLimits(energy_budget=0.05), soc,
                        baseline_execution=modeled_case("baseline"), fast_execution=modeled_case("fast"))
        assert tight.saving < loose.saving
        assert tight.fast.throughput_kwh < loose.fast.throughput_kwh

    def test_a_high_minimum_benefit_reduces_the_number_of_writes(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=2)
        eager = compare(plan, samples, PolicyLimits(min_benefit=0.0), soc,
                        baseline_execution=modeled_case("baseline"), fast_execution=modeled_case("fast"))
        lazy = compare(plan, samples, PolicyLimits(min_benefit=0.30), soc,
                       baseline_execution=modeled_case("baseline"), fast_execution=modeled_case("fast"))
        assert lazy.fast.writes < eager.fast.writes

    def test_the_report_renders(self):
        spec = make_spec()
        plan, samples, soc = synthetic_case(spec, days=1)
        comparison = compare(plan, samples, PolicyLimits(), soc,
                             baseline_execution=modeled_case("baseline"), fast_execution=modeled_case("fast"))
        report = comparison.report()
        assert "Besparing" in report
        assert "Totale kosten" in report
        assert len(comparison.daily_table().splitlines()) >= 3


def _f3a4_physics():
    from importlib import import_module

    module = import_module("dao.prog.fastctrl.simulate")
    names = ("BatteryReplaySegment", "ReplaySegment", "_next_physical_boundary",
             "_integrate_replay_segment")
    missing = [name for name in names if not hasattr(module, name)]
    assert not missing, f"F3a4 private physics API missing: {missing}"
    return module


def _f3a4_execution(*powers):
    from dao.prog.fastctrl.replay_execution import BatteryExecutionView, ExecutionSnapshot
    from dao.tests.prog.test_fast_command import command

    views = []
    for index, (name, power) in enumerate(powers):
        accepted = command(battery_name=name, setpoint_w=power,
                           command_id=f"physics-{name}", sequence=index + 1,
                           issued_at_ms=T0 * 1000, execute_before_ms=T0 * 1000 + 1000,
                           hold_until_ms=(T0 + 7200) * 1000,
                           mode="Aan" if power else "Uit")
        views.append(BatteryExecutionView(name, accepted, float(power), "accepted",
                                          False, None, False, ()))
    return ExecutionSnapshot(None, tuple(views), 0, (), ())


def test_f3a4_gross_flows_and_soc_saturation():
    from dataclasses import FrozenInstanceError

    physics = _f3a4_physics()
    a = BatteryModel(make_spec(name="A", charge_efficiency=.9, discharge_efficiency=.95),
                     soc=50, inverter_efficiency=.8)
    b = BatteryModel(make_spec(name="B", capacity_kwh=20,
                              charge_efficiency=.85, discharge_efficiency=.8),
                     soc=50, inverter_efficiency=.9)
    models = {"B": b, "A": a}  # Physical identity must not depend on dict order.
    execution = _f3a4_execution(("A", 1000), ("B", -1000))
    assert physics._next_physical_boundary(start_ts=T0, end_ts=T0 + 900,
                                           models=models, execution=execution) == T0 + 900
    assert (a.soc, b.soc) == (50, 50)
    segment = physics._integrate_replay_segment(start_ts=T0, end_ts=T0 + 900,
                                               house_w=0, models=models, execution=execution)
    first, second = segment.batteries
    assert (first.battery_name, second.battery_name) == ("A", "B")
    assert (segment.start_ts, segment.end_ts, segment.house_w) == (T0, T0 + 900, 0)
    assert segment.import_kwh == segment.export_kwh == 0
    assert (first.modeled_ac_w, second.modeled_ac_w) == (1000, -1000)
    assert (first.charge_ac_kwh, first.discharge_ac_kwh) == (.25, 0)
    assert (second.charge_ac_kwh, second.discharge_ac_kwh) == (0, .25)
    assert first.dc_throughput_kwh == pytest.approx(.2)
    assert first.cell_delta_kwh == pytest.approx(.18)
    assert (first.soc_begin, first.soc_end) == pytest.approx((50, 51.8))
    assert second.dc_throughput_kwh == pytest.approx(.25 / .9)
    assert second.cell_delta_kwh == pytest.approx(-.25 / (.9 * .8))
    assert second.soc_end == pytest.approx(50 - .25 / (.9 * .8) / .2)
    assert (a.soc, b.soc) == (first.soc_end, second.soc_end)
    for result, view in zip(segment.batteries, execution.batteries, strict=True):
        assert result.command is view.command
        assert result.command_w == view.command.setpoint_w
        assert result.execution_status == "accepted"
        assert result.reasons == ()
    with pytest.raises(FrozenInstanceError):
        first.soc_end = 0
    with pytest.raises(FrozenInstanceError):
        segment.import_kwh = 1

    # Averaging the clipped power across the hour would erase the .05 import.
    full = BatteryModel(make_spec(name="A", charge_efficiency=1, discharge_efficiency=1),
                        soc=94, inverter_efficiency=1)
    models = {"A": full}
    execution = _f3a4_execution(("A", 1000))
    split = physics._next_physical_boundary(start_ts=T0, end_ts=T0 + 3600,
                                            models=models, execution=execution)
    assert split == T0 + 360
    assert full.soc == 94
    charging = physics._integrate_replay_segment(start_ts=T0, end_ts=split, house_w=-500,
                                                models=models, execution=execution)
    assert full.soc == 95
    assert physics._next_physical_boundary(start_ts=split, end_ts=T0 + 3600,
                                           models=models, execution=execution) == T0 + 3600
    idle = physics._integrate_replay_segment(start_ts=split, end_ts=T0 + 3600, house_w=-500,
                                            models=models, execution=execution)
    assert charging.import_kwh == pytest.approx(.05)
    assert charging.export_kwh == idle.import_kwh == 0
    assert idle.export_kwh == pytest.approx(.45)
    assert charging.batteries[0].dc_throughput_kwh == pytest.approx(.1)
    assert charging.batteries[0].cell_delta_kwh == pytest.approx(.1)
    assert idle.batteries[0].modeled_ac_w == idle.batteries[0].dc_throughput_kwh == 0
    assert idle.batteries[0].soc_begin == idle.batteries[0].soc_end == 95
    assert idle.batteries[0].command_w == 1000
    assert idle.batteries[0].execution_status == "accepted"

    # The earliest boundary may belong to the other battery, discharging through
    # its own efficiency chain: .1 kWh cells -> .072 kWh AC -> 259.2 seconds.
    a.soc, b.soc = 94, 20.5
    models = {"A": a, "B": b}
    execution = _f3a4_execution(("A", 1000), ("B", -1000))
    split = physics._next_physical_boundary(start_ts=T0, end_ts=T0 + 900,
                                            models=models, execution=execution)
    assert split == pytest.approx(T0 + 259.2, rel=0, abs=1e-6)
    limited = physics._integrate_replay_segment(start_ts=T0, end_ts=split, house_w=0,
                                               models=models, execution=execution)
    assert limited.batteries[1].soc_end == 20
    assert limited.batteries[1].discharge_ac_kwh == pytest.approx(.072)
    assert limited.batteries[1].dc_throughput_kwh == pytest.approx(.08)
    assert limited.batteries[1].cell_delta_kwh == pytest.approx(-.1)
    assert physics._next_physical_boundary(start_ts=split, end_ts=T0 + 900,
                                           models=models, execution=execution) == pytest.approx(
                                               T0 + 500, rel=0, abs=1e-6)


def test_f3a4_unknown_is_not_zero_and_never_reanchors_soc():
    from dataclasses import replace

    physics = _f3a4_physics()
    models = {name: BatteryModel(make_spec(name=name, charge_efficiency=1,
                                          discharge_efficiency=1), 50, 1)
              for name in ("A", "B")}
    execution = _f3a4_execution(("A", 1000), ("B", -1000))
    prefix = physics._integrate_replay_segment(start_ts=T0, end_ts=T0 + 900, house_w=0,
                                              models=models, execution=execution)
    a, b = execution.batteries
    unknown = replace(a, command=None, drive_w=None, execution_status="unconfirmed",
                      recovery_required=True, pending=a.command, reasons=("execution_unconfirmed",))
    unknown_execution = replace(execution, batteries=(unknown, b))
    lost = physics._integrate_replay_segment(start_ts=T0 + 900, end_ts=T0 + 1800, house_w=0,
                                            models=models, execution=unknown_execution)
    assert prefix.batteries[0].soc_end == 52.5
    assert prefix.batteries[0].charge_ac_kwh == .25
    assert lost.batteries[0].soc_begin == 52.5
    assert lost.batteries[0].soc_end is None
    assert lost.batteries[0].command is lost.batteries[0].command_w is None
    assert lost.batteries[0].execution_status == "unconfirmed"
    assert "execution_unconfirmed" in lost.batteries[0].reasons
    for name in ("modeled_ac_w", "charge_ac_kwh", "discharge_ac_kwh",
                 "dc_throughput_kwh", "cell_delta_kwh"):
        assert getattr(lost.batteries[0], name) is None
    assert models["A"] is None
    assert models["B"].soc == lost.batteries[1].soc_end == 45
    assert lost.batteries[1].discharge_ac_kwh == lost.batteries[1].dc_throughput_kwh == .25
    assert lost.import_kwh is lost.export_kwh is None

    stopped = replace(unknown, pending=None, drive_w=0, execution_status="stopped",
                      recovery_required=False, reasons=())
    stop_execution = replace(execution, batteries=(stopped, b))
    assert physics._next_physical_boundary(start_ts=T0 + 1800, end_ts=T0 + 2700,
                                           models=models, execution=stop_execution) == T0 + 2700
    stop = physics._integrate_replay_segment(start_ts=T0 + 1800, end_ts=T0 + 2700, house_w=0,
                                            models=models, execution=stop_execution)
    stopped_result = stop.batteries[0]
    assert stopped_result.command is stopped_result.command_w is None
    assert stopped_result.execution_status == "stopped"
    assert stopped_result.soc_begin is stopped_result.soc_end is None
    assert "replay_soc_missing" in stopped_result.reasons
    assert (stopped_result.modeled_ac_w, stopped_result.charge_ac_kwh,
            stopped_result.discharge_ac_kwh, stopped_result.dc_throughput_kwh,
            stopped_result.cell_delta_kwh) == (0, 0, 0, 0, 0)
    assert (stop.import_kwh, stop.export_kwh) == (0, .25)
    assert models["A"] is None

    # A later accepted command cannot reconstruct the missing energy or SoC.
    resumed = physics._integrate_replay_segment(start_ts=T0 + 2700, end_ts=T0 + 3600,
                                               house_w=0, models=models, execution=execution)
    assert resumed.batteries[0].command is a.command
    assert resumed.batteries[0].command_w == 1000
    assert resumed.batteries[0].modeled_ac_w is None
    assert resumed.batteries[0].soc_begin is resumed.batteries[0].soc_end is None
    assert models["A"] is None
    assert models["B"].soc == 40

    # Missing house data affects the site alone, even with a latched peer.
    gap = physics._integrate_replay_segment(start_ts=T0 + 3600, end_ts=T0 + 4500,
                                           house_w=None, models=models, execution=stop_execution)
    assert gap.house_w is gap.import_kwh is gap.export_kwh is None
    assert gap.batteries[1].cell_delta_kwh == -.25
    assert gap.batteries[1].soc_end == models["B"].soc == 37.5
    assert gap.batteries[1].reasons == ()
    after_gap = physics._integrate_replay_segment(start_ts=T0 + 4500, end_ts=T0 + 5400,
                                                 house_w=0, models=models, execution=stop_execution)
    assert (after_gap.import_kwh, after_gap.export_kwh) == (0, .25)
    assert models["A"] is None
    assert models["B"].soc == 35
    assert execution.batteries == (a, b)


@pytest.mark.parametrize("command_w, realized_w", [(1001, 1000.25), (-1001, -1000.75), (0, 0)])
def test_f3a4_direction_and_fractional_integration(command_w, realized_w):
    from dataclasses import replace

    physics = _f3a4_physics()
    spec = make_spec(name="A", max_charge_w=1000.25, max_discharge_w=1000.75,
                     charge_efficiency=1, discharge_efficiency=1)
    model = BatteryModel(spec, 50, 1)
    execution = _f3a4_execution(("A", command_w))
    start, end = T0 + .0006, T0 + .0014
    assert physics._next_physical_boundary(start_ts=start, end_ts=end,
                                           models={"A": model}, execution=execution) == end
    segment = physics._integrate_replay_segment(start_ts=start, end_ts=end, house_w=0,
                                               models={"A": model}, execution=execution)
    result = segment.batteries[0]
    energy = abs(realized_w) * (end - start) / 3_600_000
    assert (segment.start_ts, segment.end_ts) == (start, end)
    assert result.command_w == command_w
    assert result.modeled_ac_w == realized_w  # No rounding of physical clipping.
    assert result.charge_ac_kwh == pytest.approx(energy if command_w > 0 else 0, abs=1e-15)
    assert result.discharge_ac_kwh == pytest.approx(energy if command_w < 0 else 0, abs=1e-15)
    assert result.charge_ac_kwh * result.discharge_ac_kwh == 0
    assert result.dc_throughput_kwh == pytest.approx(energy, abs=1e-15)
    assert result.cell_delta_kwh == pytest.approx(energy if command_w >= 0 else -energy, abs=1e-15)
    assert segment.import_kwh == result.charge_ac_kwh
    assert segment.export_kwh == result.discharge_ac_kwh

    # A programmed stop can retain a nonzero current command, but its physical
    # drive is zero. Only the command boundary quantizes power, never this helper.
    stopped = replace(execution.batteries[0], drive_w=0)
    stop = physics._integrate_replay_segment(start_ts=end, end_ts=end + .25, house_w=0,
                                            models={"A": model},
                                            execution=replace(execution, batteries=(stopped,)))
    assert stop.batteries[0].command_w == command_w
    assert stop.batteries[0].modeled_ac_w == stop.batteries[0].cell_delta_kwh == 0
    assert stop.batteries[0].soc_begin == stop.batteries[0].soc_end == result.soc_end


def _f3a6_result(plan, points, *, case=None, soc=50, efficiencies=1,
                 replay_basis=None, fast=False, timezone=None, limits=None):
    from importlib import import_module

    from dao.prog.fastctrl.replay_execution import RecordedExecution, validate_replay_input

    api = import_module("dao.prog.fastctrl.simulate")
    inputs = validate_replay_input(plan=plan, sample_points=points, soc_start=soc,
                                   inverter_efficiency=efficiencies, scenario=case)
    basis = api._prepare_replay_basis(plan, start_ts=points[0][0], replay_basis=replay_basis,
        recorded=case is not None and isinstance(case.execution, RecordedExecution))
    run = api._run_replay(basis=basis, replay_input=inputs, limits=limits or PolicyLimits(),
                           use_fast_layer=fast, timezone=timezone)
    assert callable(getattr(api, "_summarize_replay", None)), "F3a6 private summarizer missing"
    return api._summarize_replay(run=run, basis=basis, replay_input=inputs,
                                  label="f3a6", timezone=timezone)


@pytest.mark.parametrize("fault", ["battery", "house", "opening", "missing"])
def test_f3a6_partial_totals_and_consumers_are_unknown(fault, caplog):
    import logging
    from datetime import UTC, datetime

    from dao.prog.fastctrl.replay_execution import ModeledExecution, ModelResponse, ModelStop
    from dao.prog.fastctrl.simulate import Comparison, log_comparison

    midnight = datetime(2026, 11, 6, tzinfo=UTC).timestamp()
    start = midnight - 60
    plan = flat_plan(make_spec(charge_efficiency=1, discharge_efficiency=1))
    plan.created_ts = start
    plan.specs.append(make_spec(name="B", capacity_kwh=20, charge_efficiency=1, discharge_efficiency=1))
    plan.intervals = [PlanInterval(start, start + 120, .4, .1, batteries=[
        BatteryPlanStep(0, 80, 80), BatteryPlanStep(100, 60, 60)])]
    recipe = ModeledExecution("stopped", "complete", responses=(
        ModelResponse(1, "fast", "accu", "unknown", 0),), lifecycle=(
        ModelStop(start + 90, 0, True, True),))
    case = _f3a5_case(execution=recipe) if fault == "battery" else _f3a5_case(hold=30 if fault == "house" else 4000)
    result = _f3a6_result(plan, ((start, 1000), (start + 30, 2000), (start + 120, 0)),
        case=None if fault == "missing" else case, fast=fault == "battery",
        soc={"accu": None if fault == "opening" else 80, "B": 60}, timezone=UTC,
        limits=PolicyLimits(min_benefit=0, deadband=1, min_command_interval=0))
    assert result.coverage_status == ("unknown" if fault == "missing" else "partial")
    assert result.energy_cost is result.wear_cost is result.terminal_value is result.total_cost is None
    if fault == "opening":
        assert result.import_kwh == pytest.approx(1100 * 30 / 3600000 + 2100 * 90 / 3600000)
        assert result.export_kwh == 0 and result.peak_import_w == pytest.approx(2100)
    else:
        assert result.import_kwh is result.export_kwh is result.peak_import_w is None
    assert result.label.startswith("unknown_reconstruction:")
    if fault in ("battery", "missing"):
        assert result.throughput_kwh is result.cycles is None
        assert result.soc_end is result.average_soc is None
        assert result.override_s is None
    elif fault == "opening":
        # Proven zero response preserves throughput, but cannot invent opening/end SoC.
        assert result.soc_start is result.soc_end is result.average_soc is None
        assert result.throughput_kwh == pytest.approx(1 / 300)
    else:
        assert result.throughput_kwh == pytest.approx(1 / 300)
        assert result.soc_end is not None
    if fault == "battery":
        assert result.segments[0].import_kwh == 0
        assert result.segments[-1].batteries[0].modeled_ac_w == 0
        assert result.segments[-1].batteries[0].soc_end is None
        assert any(s.batteries[1].charge_ac_kwh > 0 for s in result.segments)
        assert result.soc_series[0][1] == pytest.approx((80 * 10 + 60 * 20) / 30)
        assert result.soc_series[-1][1] is None
    assert len(result.days) == 2
    final_day = result.days["2026-11-06"]
    assert final_day.energy_cost is final_day.wear_cost is None
    assert sum(day.writes for day in result.days.values()) == result.writes
    for comparison in (Comparison(result, result), Comparison(result, simulate(
            flat_plan(make_spec()), [Sample(T0, 0), Sample(T0 + 60, 0)],
            PolicyLimits(), 50, "old", False))):
        for output in (comparison.saving, comparison.saving_per_day,
                       comparison.report(), comparison.daily_table()):
            assert output.startswith("unknown_reconstruction:")
            assert "Besparing:" not in output and "euro" not in output
        if result.cycles is None:
            assert comparison.extra_cycles is None
        with caplog.at_level(logging.INFO):
            log_comparison(comparison)
    assert "unknown_reconstruction:" in caplog.text
    assert "Besparing:" not in caplog.text


@pytest.mark.parametrize("field", ["import_kwh", "export_kwh", "throughput_kwh",
                                  "peak_import_w", "soc_end", "override_s", "soc_series"])
def test_f3a6_nullable_numeric_report_and_daily_fields(field, caplog):
    import logging

    from dao.prog.fastctrl.simulate import Comparison, DayMetrics, SimResult, log_comparison

    result = SimResult("legacy", capacity_kwh=10, soc_series=[(T0, 50)])
    setattr(result, field, [(T0, None)] if field == "soc_series" else None)
    result.days["2026-11-06"] = DayMetrics("2026-11-06", energy_cost=None,
        wear_cost=None, throughput_kwh=None, peak_import_w=None, override_s=None)
    comparison = Comparison(SimResult("old", capacity_kwh=10), result)
    if field == "throughput_kwh":
        assert result.cycles is comparison.extra_cycles is None
    if field == "soc_series":
        assert result.average_soc is None
    assert "unknown" in comparison.report()
    assert "unknown" in comparison.daily_table()
    with caplog.at_level(logging.INFO):
        log_comparison(comparison)
    assert "unknown" in caplog.text


def test_f3a6_legacy_model_wear_and_negative_difference():
    from dataclasses import replace
    from datetime import UTC

    from dao.prog.fastctrl.simulate import Comparison

    plan = flat_plan(make_spec(charge_efficiency=.9, discharge_efficiency=.95, cycle_cost=.03), battery_w=1000)
    plan.specs.append(make_spec(name="B", capacity_kwh=20, charge_efficiency=.85,
                                discharge_efficiency=.8, cycle_cost=.09))
    plan.intervals = [replace(row, batteries=[*row.batteries, BatteryPlanStep(-1000, 60, 60)])
                      for row in plan.intervals]
    result = _f3a6_result(plan, ((T0, 0), (T0 + 900, 0)), case=_f3a5_case(),
        soc={"accu": 40, "B": 60}, efficiencies={"accu": .8, "B": .9}, timezone=UTC)
    assert result.coverage_status == "complete" and result.execution_kind == "modeled"
    assert result.import_kwh == result.export_kwh == result.peak_import_w == 0
    assert result.throughput_kwh == pytest.approx(.2 + .25 / .9)
    assert result.wear_cost == pytest.approx(.03 * .2 + .09 * .25 / .9)
    assert result.terminal_value == pytest.approx(.18 * .30 * .95 * .8 - .25 / (.9 * .8) * .30 * .8 * .9)
    assert result.soc_start == pytest.approx(160 / 3)
    assert result.cycles == pytest.approx((.2 + .25 / .9) / 60)
    assert result.writes == 2 and next(iter(result.days.values())).writes == 2
    assert result.override_s == next(iter(result.days.values())).override_s == 0
    assert result.source_plan_digest == result.execution_plan_digest
    assert result.trace and dict(result.assumptions)["fresh_policy_start"] == "true"
    idle = replace(plan, intervals=[replace(row, batteries=[BatteryPlanStep(0, 40, 40),
        BatteryPlanStep(0, 60, 60)]) for row in plan.intervals])
    baseline = _f3a6_result(idle, ((T0, 0), (T0 + 900, 0)), case=_f3a5_case(),
        soc={"accu": 40, "B": 60}, efficiencies={"accu": .8, "B": .9}, timezone=UTC)
    comparison = Comparison(baseline, result)
    assert comparison.saving == pytest.approx(-.06496)
    assert comparison.saving_per_day == comparison.saving
    assert comparison.extra_cycles == pytest.approx(result.cycles)
    assert "modeled" in comparison.report() and "fresh_policy_start" in comparison.report()
    assert "-0.031" in comparison.daily_table()  # Daily costs exclude terminal correction.


@pytest.mark.parametrize("retry_delay", [None, .5, 0], ids=["no-retry", "after-start", "at-start"])
def test_f3a6_recorded_opening_and_daily_override_use_accepted_provenance(retry_delay):
    from dataclasses import replace
    from datetime import UTC, datetime

    from dao.prog.fastctrl import replay_execution as execution
    from dao.tests.prog.test_fast_replay_execution import (
        replay_command,
        replay_receipt,
        replay_timing,
        send_event,
    )

    start = int(datetime(2026, 11, 5, 23, 59, 57, tzinfo=UTC).timestamp())
    plan = flat_plan(make_spec(charge_efficiency=1, discharge_efficiency=1))
    plan.created_ts = start
    plan.intervals = [PlanInterval(start, start + 900, .3, .1,
                                    batteries=[BatteryPlanStep(0, 50, 50)])]
    command = replay_command(plan)
    timing = replay_timing(execution, command, anchor=start)
    observation = replay_receipt(execution, command, start, apply_ts=start, anchor=start)
    proposal = send_event(execution, command, start, timing)
    suppressed = execution.ProposalEvent(start + 1, 0,
        replace(proposal.decision, write=False, override=False, setpoint_w=1100), None, None)
    events = (replace(proposal, at_ts=start + retry_delay,
                      decision=replace(proposal.decision, override=False)), suppressed) if retry_delay is not None else (suppressed,)
    case = _f3a5_recorded(plan, start, start + 6, events)
    case = replace(case, execution=replace(case.execution, opening_acceptances=(
        execution.OpeningAcceptance(command, timing, observation, start),)))
    result = _f3a6_result(plan, ((start, 0), (start + 6, 0)), case=case, timezone=UTC)
    assert result.execution_kind == "recorded" and result.coverage_status == "complete"
    assert result.writes == (2 if retry_delay is not None else 1)
    assert [day.writes for day in result.days.values()] == [result.writes, 0]
    assert result.import_kwh == pytest.approx(1 / 600)
    assert result.segments[-1].batteries[0].command == command
    assert result.override_s == 6
    assert [day.override_s for day in result.days.values()] == [3, 3]


def _f3a5_api():
    from importlib import import_module

    module = import_module("dao.prog.fastctrl.simulate")
    names = ("_ReplayBasisView", "_prepare_replay_basis", "_ReplayRun", "_run_replay")
    missing = [name for name in names if not hasattr(module, name)]
    assert not missing, f"F3a5 private orchestration API missing: {missing}"
    return module


def _f3a5_case(*, execution=None, hold=4000):
    from dao.prog.fastctrl.replay_execution import ModeledExecution, ReplayPhysics, ReplayScenario

    return ReplayScenario("f3a5", execution or ModeledExecution("stopped", "complete"),
                          ReplayPhysics("battery_model_v1", hold, "ideal_programmed_stop"))


def _f3a5_run(plan, points, *, case=None, fast=False, replay_basis=None, soc=50,
              limits=None, timezone=None):
    from dao.prog.fastctrl.replay_execution import RecordedExecution, validate_replay_input

    api = _f3a5_api()
    inputs = validate_replay_input(plan=plan, sample_points=points, soc_start=soc,
                                   inverter_efficiency=1, scenario=case)
    basis = api._prepare_replay_basis(plan, start_ts=points[0][0], replay_basis=replay_basis,
                                      recorded=case is not None and isinstance(case.execution, RecordedExecution))
    return api._run_replay(basis=basis, replay_input=inputs, limits=limits or PolicyLimits(),
                           use_fast_layer=fast, timezone=timezone)


def _f3a5_recorded(plan, start, end, events):
    from dao.prog.fastctrl.command import OwnerBinding
    from dao.prog.fastctrl.replay_execution import ExecutionCoverage, RecordedExecution

    installation = plan.financial_context.basis.meter_id if plan.financial_context else "synthetic"
    binding = OwnerBinding(installation, "receiver-a", "generation-a", "session-a", 2,
                           "opening-barrier", int((end + 10) * 1000))
    return _f3a5_case(execution=RecordedExecution(
        installation, "generation-a", binding, (), None,
        tuple(ExecutionCoverage(spec.name, start, end) for spec in plan.specs), tuple(events)))


def test_f3a5_no_write_integrates_current_not_proposal(monkeypatch):
    from dao.prog.fastctrl import replay_execution as replay
    from dao.prog.fastctrl.policy import BatteryDecision, FastControlPolicy
    from dao.tests.prog.test_fast_replay_execution import (
        replay_command,
        replay_receipt,
        replay_timing,
        send_event,
    )

    api = _f3a5_api()
    plan = flat_plan(make_spec(charge_efficiency=1), battery_w=1000)
    command = replay_command(plan)
    events = (send_event(replay, command, T0, replay_timing(replay, command)),
              replay_receipt(replay, command, T0, apply_ts=T0),
              replay.ProposalEvent(T0 + 1, 0, BatteryDecision(
                  0, "accu", 1100, 1000, False, True, "suppressed"), None, None))
    case = _f3a5_recorded(plan, T0, T0 + 2, events)

    def forbidden(*args, **kwargs):
        pytest.fail("recorded execution recomputed policy")

    monkeypatch.setattr(FastControlPolicy, "account", forbidden)
    monkeypatch.setattr(FastControlPolicy, "decide", forbidden)
    run = _f3a5_run(plan, ((T0, 0), (T0 + 2, 0)), case=case, fast=True)
    assert isinstance(run, api._ReplayRun)
    assert [s.batteries[0].modeled_ac_w for s in run.segments] == [1000, 1000]
    assert sum(s.batteries[0].charge_ac_kwh for s in run.segments) == pytest.approx(1 / 1800)
    assert run.soc_end_by_battery == (("accu", pytest.approx(50 + 1 / 180)),)
    assert run.writes == 1 and run.execution_kind == "recorded"
    assert run.trace == events


@pytest.mark.parametrize("delay", [0, .0001])
def test_f3a5_float_edges_are_not_wire_edges(delay):
    from dao.prog.fastctrl.replay_execution import ModeledExecution, ModelStop, ProposalEvent

    _f3a5_api()
    start, end = T0 + .0006, T0 + .1006
    plan = flat_plan(make_spec(charge_efficiency=1), battery_w=1000)
    model = ModeledExecution("stopped", "complete", response_delay_s=delay,
                             lifecycle=(ModelStop(end, 0, True, True),))
    run = _f3a5_run(plan, ((start, 0), (end, 0)), case=_f3a5_case(execution=model))
    assert [(s.start_ts, s.end_ts) for s in run.segments] == (
        [(start, end)] if delay == 0 else [(start, start + delay), (start + delay, end)])
    assert all(event.at_ts < end for event in run.trace)
    assert run.writes == 1
    assert len([event for event in run.trace if isinstance(event, ProposalEvent)]) == 1
    if delay == 0:
        assert run.segments[0].batteries[0].charge_ac_kwh == pytest.approx((end - start) / 3600)
        assert run.soc_end_by_battery[0][1] > 50
    else:
        assert run.segments[0].batteries[0].command is None
        assert run.segments[1].batteries[0].command.setpoint_w == 1000
        assert all(s.batteries[0].modeled_ac_w is None for s in run.segments)
        assert run.soc_end_by_battery == (("accu", None),)


def _f3a5_shaped_pair():
    from dataclasses import replace

    from dao.tests.settlement.test_fast_profile import shaped_plan

    plan = shaped_plan()
    start = plan.intervals[0].start_ts
    # A partial first solver interval, followed by two full intervals.
    plan.interval_s = 3600
    plan.financial_context = replace(plan.financial_context,
        not_after_ts=start + 7200,
        basis=replace(plan.financial_context.basis, horizon_end_ts=start + 9000))
    plan.specs.append(replace(plan.specs[0], name="B", capacity_kwh=20))
    terminal = plan.financial_context.terminal_value
    plan.financial_context = replace(plan.financial_context, terminal_value=replace(terminal,
        reference_efficiency_by_battery=(*terminal.reference_efficiency_by_battery,
                                         ("B", terminal.reference_efficiency_by_battery[0][1]))))
    rows = []
    for index, (a, b) in enumerate(((start, start + 1800), (start + 1800, start + 5400),
                                   (start + 5400, start + 9000))):
        first = BatteryPlanStep(-400 if index == 0 else 0, 80 - index * 10, 70 - index * 10,
                                mean_ac_power_w=-400 if index == 0 else 0,
                                on_power_w=-500 if index == 0 else 0,
                                active_from_ts=a, stop_ts=a + 1440 if index == 0 else None)
        second = BatteryPlanStep(250 if index == 0 else 0, 60 - index * 10, 50 - index * 10,
                                 mean_ac_power_w=250 if index == 0 else 0,
                                 on_power_w=500 if index == 0 else 0,
                                 active_from_ts=a + 600 if index == 0 else a,
                                 stop_ts=a + 1500 if index == 0 else None)
        rows.append(PlanInterval(a, b, .4, .1, batteries=[first, second]))
    plan.intervals = rows
    return FastPlan.from_dict(plan.to_dict())


def test_f3a5_profile_shape_and_original_policy_context(monkeypatch):
    from copy import deepcopy

    from dao.prog.fastctrl.policy import FastControlPolicy
    from dao.prog.fastctrl.replay_execution import ProposalEvent

    _f3a5_api()
    plan = _f3a5_shaped_pair()
    before = deepcopy(plan)
    start = plan.intervals[0].start_ts
    points = ((start, 0), (start + 1900, 0))
    baseline = _f3a5_run(plan, points, case=_f3a5_case(), soc={"accu": 80, "B": 60})
    assert [(s.start_ts - start, s.end_ts - start) for s in baseline.segments] == [
        (0, 600), (600, 1440), (1440, 1500), (1500, 1800), (1800, 1900)]
    assert [tuple(b.modeled_ac_w for b in s.batteries) for s in baseline.segments] == [
        (-500, 0), (-500, 500), (0, 500), (0, 0), (0, 0)]
    assert sum(s.batteries[0].discharge_ac_kwh for s in baseline.segments) == pytest.approx(.2)
    assert sum(s.batteries[1].charge_ac_kwh for s in baseline.segments) == pytest.approx(.125)
    assert any(isinstance(e, ProposalEvent) and e.command is not None and e.at_ts == start + 1800
               for e in baseline.trace)  # Same zero profile still needs a new hold acceptance.

    calls = []
    real_account, real_decide = FastControlPolicy.account, FastControlPolicy.decide

    def account(self, state, actual_plan, measurement, day_key, *args, **kwargs):
        assert actual_plan == before
        calls.append(("account", measurement.timestamp,
                      tuple(b.power_w for b in measurement.batteries)))
        return real_account(self, state, actual_plan, measurement, day_key, *args, **kwargs)

    def decide(self, actual_plan, measurement, state, *args, **kwargs):
        assert actual_plan == before
        assert kwargs["owner_binding"].installation_id == plan.financial_context.basis.meter_id
        calls.append(("decide", measurement.timestamp,
                      tuple(b.power_w for b in measurement.batteries)))
        return real_decide(self, actual_plan, measurement, state, *args, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "account", account)
    monkeypatch.setattr(FastControlPolicy, "decide", decide)
    fast = _f3a5_run(plan, points, case=_f3a5_case(), fast=True, soc={"accu": 80, "B": 60},
                     limits=PolicyLimits(min_benefit=1e9))
    assert [at - start for kind, at, _ in calls if kind == "decide"] == [0, 600, 1440, 1500, 1800]
    # Account closes A's response before the new zero-profile proposal at stop.
    assert ("account", start + 1440, (-500, 500)) in calls
    assert sum(s.batteries[0].discharge_ac_kwh for s in fast.segments) == pytest.approx(.2)
    assert fast.soc_end_by_battery == (("accu", pytest.approx(78)), ("B", pytest.approx(60.625)))
    assert dict(fast.assumptions)["fresh_policy_start"] == "true"
    assert plan == before


def test_f3a5_unknown_soc_is_latched_without_hiding_other_battery(monkeypatch):
    from datetime import UTC, datetime

    from dao.prog.fastctrl.policy import FastControlPolicy
    from dao.prog.fastctrl.replay_execution import (
        ModeledExecution,
        ModelResponse,
        ModelStop,
        ProposalEvent,
    )

    _f3a5_api()
    midnight = datetime(2026, 11, 6, tzinfo=UTC).timestamp()
    start = int(midnight - 60)
    plan = flat_plan(make_spec(charge_efficiency=1, discharge_efficiency=1), battery_w=0)
    plan.created_ts = start
    plan.specs.append(make_spec(name="B", charge_efficiency=1, discharge_efficiency=1))
    plan.intervals = [PlanInterval(start + i * 120, start + (i + 1) * 120, .4, .1,
                                  batteries=[BatteryPlanStep(0, 50, 50), BatteryPlanStep(100, 50, 50)])
                      for i in range(3)]
    model = ModeledExecution("stopped", "complete", responses=(
        ModelResponse(1, "fast", "accu", "unknown", 0),), lifecycle=(
        ModelStop(start + 180, 0, True, True),))
    calls = []
    decide = FastControlPolicy.decide

    def observe(self, actual_plan, measurement, state, *args, **kwargs):
        calls.append(measurement.timestamp)
        return decide(self, actual_plan, measurement, state, *args, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "decide", observe)
    points = tuple((start + offset, house) for offset, house in ((0, 1000), (30, 2000), (240, 0)))
    run = _f3a5_run(plan, points, case=_f3a5_case(execution=model), fast=True,
                     soc={"accu": 80, "B": 60}, timezone=UTC,
                     limits=PolicyLimits(min_benefit=0, deadband=1, min_command_interval=0))
    assert calls == [start, start + 30]
    assert run.segments[0].batteries[0].modeled_ac_w == -1100
    assert any(s.end_ts == midnight for s in run.segments)
    assert all(s.batteries[0].soc_end is None for s in run.segments if s.start_ts >= start + 30)
    assert run.soc_end_by_battery[0] == ("accu", None)
    assert run.soc_end_by_battery[1][1] is not None
    assert any(s.batteries[1].charge_ac_kwh > 0 for s in run.segments if s.start_ts >= start + 30)
    assert run.segments[-1].batteries[0].modeled_ac_w == 0
    assert "policy_history_unknown" in run.reasons
    assert not any(isinstance(e, ProposalEvent) and e.decision.override and e.at_ts > start + 30
                   for e in run.trace)

    # A house gap alone does not destroy either battery's independent SoC.
    gap = _f3a5_run(plan, ((start, 0), (start + 90, 0)),
                     case=_f3a5_case(hold=30), soc={"accu": 80, "B": 60})
    assert gap.segments[-1].house_w is gap.segments[-1].import_kwh is None
    assert all(soc is not None for _, soc in gap.soc_end_by_battery)


def test_f3a5_recorded_digest_and_model_quote_projection_are_separate(monkeypatch):
    from copy import deepcopy
    from dataclasses import replace

    from dao.prog.fastctrl import replay_execution as execution
    from dao.prog.fastctrl.command import plan_digest
    from dao.prog.fastctrl.policy import FastControlPolicy
    from dao.tests.prog.test_fast_replay_execution import (
        replay_command,
        replay_receipt,
        replay_timing,
        send_event,
    )
    from dao.tests.settlement.test_fast_financial_replay import bind_replay, synthetic_replay

    api = _f3a5_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    plan.price_average = 999
    plan.intervals = [replace(row, price_import=999, price_export=999) for row in plan.intervals]
    before = deepcopy(plan)
    start = plan.intervals[0].start_ts
    points = ((start, 1000), (start + 2, 1000))
    recorded_basis = api._prepare_replay_basis(plan, start_ts=start, replay_basis=replay, recorded=True)
    model_basis = api._prepare_replay_basis(plan, start_ts=start, replay_basis=replay, recorded=False)
    assert recorded_basis.execution_plan == recorded_basis.source_plan == before
    assert recorded_basis.quotes == model_basis.quotes == replay.quotes
    assert model_basis.opening is replay.opening and model_basis.terminal is replay.terminal_value
    assert model_basis.execution_plan.price_average == pytest.approx(.1)
    assert all(row.price_import == .1 for row in model_basis.execution_plan.intervals)
    assert model_basis.execution_plan.financial_context == before.financial_context
    assert model_basis.execution_plan.specs == before.specs
    assert [row.batteries for row in model_basis.execution_plan.intervals] == [row.batteries for row in before.intervals]

    cmd = replay_command(plan)
    events = (send_event(execution, cmd, start, replay_timing(execution, cmd, anchor=start)),
              replay_receipt(execution, cmd, start, apply_ts=start, anchor=start))
    recorded = _f3a5_run(plan, points, case=_f3a5_recorded(plan, start, start + 2, events), replay_basis=replay)
    assert recorded.source_plan_digest == recorded.execution_plan_digest == plan_digest(before)
    assert recorded.segments[0].batteries[0].command == cmd
    observed = []
    decide = FastControlPolicy.decide

    def capture(self, actual_plan, *args, **kwargs):
        observed.append(actual_plan)
        return decide(self, actual_plan, *args, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "decide", capture)
    model = _f3a5_run(plan, points, case=_f3a5_case(), replay_basis=replay, fast=True)
    assert observed == [model_basis.execution_plan]
    assert model.source_plan_digest == recorded.source_plan_digest
    assert model.execution_plan_digest == plan_digest(model_basis.execution_plan) != model.source_plan_digest
    assert all(event.command.plan_digest == model.execution_plan_digest for event in model.trace
               if isinstance(event, execution.ProposalEvent) and event.command is not None)
    assert plan == before


@pytest.mark.parametrize("fault", ["missing", "hash", "opening", "truncated", "duplicate", "empty"])
def test_f3a5_financial_mismatch_retains_physics_or_explicit_unknown(fault):
    from dataclasses import replace

    from dao.tests.settlement.test_fast_financial_replay import bind_replay, synthetic_replay

    api = _f3a5_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    start = plan.intervals[0].start_ts
    if fault == "missing":
        replay = None
    elif fault == "hash":
        plan = replace(plan, financial_context=replace(plan.financial_context, replay_basis_hash="f" * 64))
    elif fault == "opening":
        start += 1
    elif fault == "truncated":
        plan.intervals = plan.intervals[:1]
    elif fault == "duplicate":
        plan.intervals[1] = plan.intervals[0]
    else:
        plan.intervals = []
    basis = api._prepare_replay_basis(plan, start_ts=start, replay_basis=replay, recorded=False)
    assert basis.reasons and basis.quotes is basis.opening is basis.terminal is None
    run = _f3a5_run(plan, ((start, 1000), (start + 2, 1000)), case=_f3a5_case(), replay_basis=replay)
    assert set(basis.reasons) <= set(run.reasons)
    if fault in ("truncated", "duplicate", "empty"):
        assert run.source_plan_digest is run.execution_plan_digest is None
        assert run.writes == 0
        assert all(b.modeled_ac_w is None for s in run.segments for b in s.batteries)
    else:
        assert run.source_plan_digest == run.execution_plan_digest
        assert run.segments[0].import_kwh == pytest.approx(2 / 3600)


def test_f3a5_missing_scenario_never_creates_a_model(monkeypatch):
    api = _f3a5_api()

    def forbidden(*args, **kwargs):
        pytest.fail("missing execution input constructed a model")

    monkeypatch.setattr(api, "ExecutionReplay", forbidden)
    run = _f3a5_run(flat_plan(make_spec()), ((T0, 0), (T0 + 60, 0)))
    assert run.execution_kind == "missing" and run.writes == 0 and run.trace == ()
    assert "execution_input_missing" in run.reasons
    assert run.segments[0].batteries[0].modeled_ac_w is None
    assert run.soc_start_by_battery == (("accu", 50),)
    assert run.soc_end_by_battery == (("accu", None),)


def test_f3a5_suppressed_override_has_no_executed_duration():
    from dao.prog.fastctrl.replay_execution import ProposalEvent

    plan = flat_plan(make_spec(charge_efficiency=1, cycle_cost=0), battery_w=1000, price=.4, export=.1)
    limits = PolicyLimits(storage_value_mode="fixed", storage_value_fixed=.3, round_trip_efficiency=1,
                          min_benefit=0, deadband=150, release_deviation=50, min_command_interval=0)
    run = _f3a5_run(plan, ((T0, -1100), (T0 + 30, -1100), (T0 + 60, -1100)),
                     case=_f3a5_case(), fast=True, limits=limits)
    proposals = [event for event in run.trace if isinstance(event, ProposalEvent)]
    assert [(event.decision.write, event.decision.override) for event in proposals] == [
        (True, False), (False, True), (False, True)]
    assert all(s.batteries[0].command.setpoint_w == 1000 for s in run.segments)
    assert sum(s.batteries[0].charge_ac_kwh for s in run.segments) == pytest.approx(1 / 60)
    assert run.writes == 1 and run.override_s == 0


def test_f3a5_account_closes_response_before_current_tick_decision(monkeypatch):
    from dao.prog.fastctrl.policy import FastControlPolicy

    plan = flat_plan(make_spec(charge_efficiency=1, cycle_cost=0), battery_w=1000, price=.4, export=.1)
    limits = PolicyLimits(storage_value_mode="fixed", storage_value_fixed=.3, round_trip_efficiency=1,
                          min_benefit=0, deadband=1, min_command_interval=0)
    calls = []
    account, decide = FastControlPolicy.account, FastControlPolicy.decide

    def capture(kind, state, measurement):
        calls.append((kind, measurement.timestamp, measurement.battery(0).power_w,
                      measurement.grid_w, state.battery(0).last_accepted_command.setpoint_w))

    def observe_account(self, state, actual_plan, measurement, day_key, *args, **kwargs):
        capture("account", state, measurement)
        return account(self, state, actual_plan, measurement, day_key, *args, **kwargs)

    def observe_decide(self, actual_plan, measurement, state, *args, **kwargs):
        capture("decide", state, measurement)
        return decide(self, actual_plan, measurement, state, *args, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "account", observe_account)
    monkeypatch.setattr(FastControlPolicy, "decide", observe_decide)
    run = _f3a5_run(plan, ((T0, -1000), (T0 + 60, -2000), (T0 + 120, -2000)),
                     case=_f3a5_case(), fast=True, limits=limits)
    assert calls == [("account", T0, 1000, 0, 1000), ("decide", T0, 1000, 0, 1000),
                     ("account", T0 + 60, 1000, 0, 1000), ("decide", T0 + 60, 1000, -1000, 1000),
                     ("account", T0 + 120, 2000, 0, 2000)]
    assert sum(s.batteries[0].charge_ac_kwh for s in run.segments) == pytest.approx(.05)
    assert run.soc_end_by_battery == (("accu", pytest.approx(50.5)),)
    assert run.writes == 2 and run.override_s == 60


@pytest.mark.parametrize("delay", [0, .1])
def test_f3a5_latency_does_not_renumber_sample_and_interval_ticks(delay):
    from dao.prog.fastctrl.replay_execution import (
        ModeledExecution,
        ModelResponse,
        ProposalEvent,
        ReceiptEvent,
    )

    plan = flat_plan(make_spec(charge_efficiency=1), battery_w=1000)
    plan.intervals = [PlanInterval(T0 + i, T0 + i + 1, .4, .1,
                                  batteries=[BatteryPlanStep(1000 + i * 1000, 50, 50)]) for i in range(2)]
    execution = ModeledExecution("stopped", "complete", response_delay_s=delay,
        responses=(ModelResponse(2, "plan", "accu", "none", 0),))
    run = _f3a5_run(plan, ((T0, 0), (T0 + .5, 0), (T0 + 2, 0)),
                     case=_f3a5_case(execution=execution))
    proposals = [event for event in run.trace if isinstance(event, ProposalEvent)]
    assert [event.at_ts for event in proposals] == [T0, T0 + .5, T0 + 1]
    assert [event.command.sequence if event.command else None for event in proposals] == [1, None, 2]
    receipts = [event for event in run.trace if isinstance(event, ReceiptEvent)]
    assert [(event.at_ts, event.receipt.status) for event in receipts] == [
        (T0 + delay, "accepted"), (T0 + 1, "rejected")]
    assert "unused_model_response" not in run.reasons


def test_f3a5_dst_hours_and_local_midnights_keep_distinct_utc_boundaries():
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("Europe/Amsterdam")
    start = int(datetime(2026, 10, 24, 21, 30, tzinfo=UTC).timestamp())
    end = start + 26 * 3600
    first_hour, second_hour = start + 3 * 3600, start + 4 * 3600
    first_midnight, second_midnight = start + 1800, end - 1800
    plan = flat_plan(make_spec(), battery_w=0)
    plan.created_ts = start
    plan.intervals = [PlanInterval(start, end, .4, .1, batteries=[BatteryPlanStep(0, 50, 50)])]
    run = _f3a5_run(plan, ((start, 1000), (first_hour, 2000), (second_hour, 3000), (end, 3000)),
                     case=_f3a5_case(hold=48 * 3600), timezone=zone)
    assert [s.start_ts for s in run.segments] == [
        start, first_midnight, first_hour, second_hour, second_midnight]
    assert [(datetime.fromtimestamp(at, zone).hour, datetime.fromtimestamp(at, zone).fold)
            for at in (first_hour, second_hour)] == [(2, 0), (2, 1)]
    assert second_midnight - first_midnight == 25 * 3600
    assert sum(s.import_kwh for s in run.segments) == pytest.approx(71)
    assert run.soc_end_by_battery == (("accu", 50),) and run.writes == 1


@pytest.mark.parametrize("source, expected_writes", [("send", 1), ("opening", 1), ("prior-opening", 0)])
def test_f3a5_invalid_plan_retains_recorded_send_count_without_acceptance(monkeypatch, source, expected_writes):
    from dataclasses import replace

    from dao.prog.fastctrl import replay_execution as execution
    from dao.tests.prog.test_fast_replay_execution import (
        replay_command,
        replay_receipt,
        replay_timing,
        send_event,
    )
    from dao.tests.settlement.test_fast_financial_replay import bind_replay, synthetic_replay

    api = _f3a5_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    issued = plan.intervals[0].start_ts
    start = issued + 1 if source == "prior-opening" else issued
    cmd = replay_command(plan)
    timing = replay_timing(execution, cmd, anchor=issued)
    observation = replay_receipt(execution, cmd, issued, apply_ts=issued, anchor=issued)
    events = (send_event(execution, cmd, issued, timing), observation) if source == "send" else ()
    case = _f3a5_recorded(plan, start, start + 2, events)
    if source != "send":
        case = replace(case, execution=replace(case.execution, opening_acceptances=(
            execution.OpeningAcceptance(cmd, timing, observation, issued),)))
    plan.intervals = plan.intervals[:1]  # No valid canonical identity for this supplied snapshot.

    def forbidden(*args, **kwargs):
        pytest.fail("invalid plan identity acquired an execution runtime")

    monkeypatch.setattr(api, "ExecutionReplay", forbidden)
    run = _f3a5_run(plan, ((start, 0), (start + 2, 0)), case=case, replay_basis=replay)
    assert run.source_plan_digest is run.execution_plan_digest is None
    assert run.trace == (events if events else (observation,))
    assert all(s.batteries[0].command is None and s.batteries[0].modeled_ac_w is None for s in run.segments)
    assert run.writes == expected_writes
