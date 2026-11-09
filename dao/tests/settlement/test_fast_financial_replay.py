"""Pinned replay evidence, never a capability or a historical savings claim."""

import importlib
import importlib.util
import json
import logging
import os
import stat
from dataclasses import FrozenInstanceError, asdict, replace
from decimal import Decimal, localcontext
from hashlib import sha256

import pytest

from dao.prog.fastctrl.plan import FastPlan
from dao.prog.fastctrl.policy import PolicyLimits
from dao.prog.fastctrl.publication import following_verdict, read_publication
from dao.prog.fastctrl.simulate import Comparison, HistoryLoader, Sample, log_comparison, simulate
from dao.prog.settlement.publication import FinancialPlanBlocked
from dao.prog.settlement.replay import ReplayBasis, _canonical_bytes
from dao.prog.settlement.tariffs import make_snapshot, quote
from dao.prog.settlement.types import RegisteredSpot, SettlementState, SpotInterval, TariffSnapshot
from dao.tests.prog.test_fast_simulate import modeled_case
from dao.tests.settlement.test_fast_contract import ready_plan
from dao.tests.settlement.test_publication import NOW, START_TS, pricing_config, replay_for_plan
from dao.tests.settlement.test_publication import system as system

D = Decimal


def replay_api():
    assert importlib.util.find_spec("dao.prog.settlement.replay") is not None, "F1c ReplayBasis module missing"
    return importlib.import_module("dao.prog.settlement.replay")


def synthetic_replay():
    pricing = pricing_config()
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    plan = ready_plan(hashes=(snapshot.snapshot_hash,))
    replay = replay_for_plan(plan, (snapshot,))
    # A native hour contract needs raw hour prices, even for a quarter replay.
    spots = tuple(SpotInterval(at, at + 3600, D(".1"), "synthetic-replay", "estimated")
                  for at in range(NOW // 3600 * 3600, plan.horizon_end_ts, 3600))
    items = tuple(RegisteredSpot(spot, replay.basis.input_revision,
                  sha256(_canonical_bytes(asdict(spot))).hexdigest()) for spot in spots)
    return plan, replace(replay, price_batch=replace(replay.price_batch, items=items,
        start_ts=spots[0].start_ts, end_ts=spots[-1].end_ts))


def bind_replay(plan, replay):
    return replace(plan, financial_context=replace(plan.financial_context, replay_basis_hash=replay.content_hash))


def canonical_snapshot(snapshot: TariffSnapshot, **changes) -> TariffSnapshot:
    """Rehash an explicitly synthetic snapshot using the archive's canonical form."""
    candidate = replace(snapshot, **changes)
    content = asdict(candidate)
    del content["snapshot_hash"]
    return replace(candidate, snapshot_hash=sha256(_canonical_bytes(content)).hexdigest())


def finance_case(*, edges: tuple[int, ...] = (NOW, NOW + 900),
                  snapshots: tuple[TariffSnapshot, ...] | None = None,
                  spots: tuple[SpotInterval, ...] | None = None,
                  opening: SettlementState | None = None,
                  battery_names: tuple[str, ...] = ("accu",)) -> tuple[FastPlan, ReplayBasis]:
    """Synthetic archive with exact plan edges and complete native price windows.

    Explicitly missing raw prices retain the synthetic .1 archived quote, so
    consumers can test missing price evidence without fabricating a raw source.
    """
    from dao.prog.settlement.metering import _spot_window, resolve_spot

    if (len(edges) < 2 or any(type(edge) is not int for edge in edges)
            or any(end <= start for start, end in zip(edges[:-1], edges[1:], strict=True))):
        raise ValueError("Synthetic finance edges must be ordered integer UTC bounds")
    plan, replay = synthetic_replay()
    terminal = replace(replay.terminal_value, reference_efficiency_by_battery=tuple(
        (name, replay.terminal_value.reference_efficiency_by_battery[0][1]) for name in battery_names))
    replay = replace(replay, terminal_value=terminal)
    plan = replace(plan, specs=[replace(plan.specs[0], name=name) for name in battery_names],
                   financial_context=replace(plan.financial_context, terminal_value=terminal))
    snapshots = replay.snapshots if snapshots is None else tuple(snapshots)
    selections = tuple(next(snapshot for snapshot in snapshots if snapshot.valid_from_ts <= start
                           and (snapshot.valid_to_ts is None or end <= snapshot.valid_to_ts))
                       for start, end in zip(edges[:-1], edges[1:], strict=True))
    if spots is None:
        raw = {}
        for start, end, snapshot in zip(edges[:-1], edges[1:], selections, strict=True):
            window_start, window_end, step = _spot_window(snapshot, start, end)
            for at in range(window_start, window_end, step):
                raw[(at, at + step)] = SpotInterval(at, at + step, D("0.1"), "synthetic-finance", "estimated")
        spots = tuple(raw[key] for key in sorted(raw))
    else:
        spots = tuple(spots)
    quotes = []
    rows = []
    for start, end, snapshot in zip(edges[:-1], edges[1:], selections, strict=True):
        resolved = resolve_spot(snapshot, start, end, spots)
        price = quote(snapshot, start, end, D("0.1") if resolved.price_ex_vat is None else resolved.price_ex_vat)
        quotes.append(price)
        rows.append(replace(plan.intervals[0], start_ts=start, end_ts=end,
                             price_import=float(price.import_rate), price_export=float(price.export_rate),
                             batteries=[replace(plan.intervals[0].batteries[0], active_from_ts=start, stop_ts=None)
                                        for _ in battery_names]))
    basis = replace(replay.basis, snapshot_hashes=tuple(snapshot.snapshot_hash for snapshot in snapshots),
                    prefix_start_ts=min(replay.basis.prefix_start_ts, edges[0]), planning_start_ts=edges[0],
                    horizon_end_ts=edges[-1], captured_ts=edges[0], source_watermark_ts=edges[0])
    items = tuple(RegisteredSpot(spot, 1, sha256(_canonical_bytes(asdict(spot))).hexdigest()) for spot in spots)
    batch = replace(replay.price_batch, start_ts=min((edges[0], *(spot.start_ts for spot in spots))),
                    end_ts=max((edges[-1], *(spot.end_ts for spot in spots))), items=items, snapshots=snapshots,
                    captured_ts=basis.captured_ts, source_watermark_ts=basis.source_watermark_ts)
    replay = replace(replay, basis=basis, opening=SettlementState(edges[0], None, None) if opening is None else opening,
                     quotes=tuple(quotes), price_batch=batch)
    plan = replace(plan, created_ts=edges[0],
                   interval_s=max(end - start for start, end in zip(edges[:-1], edges[1:], strict=True)),
                   intervals=rows, price_average=sum(row.price_import for row in rows) / len(rows),
                   financial_context=replace(plan.financial_context, basis=basis, not_after_ts=edges[-1],
                                             price_interval_s=snapshots[0].price_interval_seconds))
    return bind_replay(plan, replay), replay


def f3b4_limits(**changes):
    return PolicyLimits(**(dict(storage_value_mode="fixed", storage_value_fixed=0,
        min_benefit=0, deadband=0, min_command_interval=0, release_time=0,
        energy_budget=0, daily_extra_throughput=0, soc_margin=0) | changes))


def f3b4_public_case(bonus_rules=None, *, edges=(NOW, NOW + 900, NOW + 1800)):
    from dao.prog.settlement.types import BonusState
    from dao.tests.settlement.conftest import YEAR_END_TS
    from dao.tests.settlement.test_fast_replay_finance import terminal_case

    _, seed = finance_case()
    snapshot = seed.snapshots[0]
    if bonus_rules is not None:
        snapshot = canonical_snapshot(snapshot, bonus=replace(bonus_rules,
            windows=tuple((month, (0, 24)) for month in range(1, 13))))
    opening = SettlementState(edges[0], None if bonus_rules is None else
        BonusState(START_TS, YEAR_END_TS, edges[0], D("4998.5")), None)
    spots = tuple(SpotInterval(at, at + 3600, D(".20"), "synthetic:f3b4", "estimated")
                  for at in range(edges[0] // 3600 * 3600, edges[-1], 3600))
    plan, replay = finance_case(edges=edges, snapshots=(snapshot,), spots=spots, opening=opening)
    plan, replay = terminal_case(plan, replay, price="0")
    plan.specs = [replace(plan.specs[0], cycle_cost=0)]
    for row in plan.intervals:
        row.batteries = [replace(row.batteries[0], ac_power_w=3000,
                                mean_ac_power_w=3000, on_power_w=3000)]
    plan = FastPlan.from_dict(plan.to_dict())
    return plan, replay, [Sample(at, -4000) for at in edges]


def f3b4_source(plan, replay, samples, *, hold=900):
    from dao.prog.fastctrl import replay_finance as api
    from dao.prog.fastctrl.command import plan_digest

    reference = api.SourceReference("synthetic:f3b4-topology", "v1", "b" * 64)
    start, end = replay.basis.planning_start_ts, replay.basis.horizon_end_ts
    return api.ReplaySourceEvidence(
        "replay_source_v1", replay.basis.meter_id, replay.basis.meter_id, replay.content_hash,
        plan_digest(plan), api.replay_environment_digest(
            tuple((s.timestamp, s.house_w) for s in samples), house_hold_s=hold),
        tuple(spec.name for spec in plan.specs), ("synthetic:pv",), reference, "synthetic:caller",
        start, (reference,), start, end,
        (api.ReplaySourceSpan(start, end, "pv_only_nonbattery_generation", (reference,)),),
        "both_alternatives")


@pytest.mark.parametrize("quarters, advantage", [(1, D(".195")), (2, D(".36"))])
def test_f3b4_public_compare_uses_two_real_books_and_one_source(monkeypatch, bonus_rules, quarters, advantage):
    # Catches quote-only valuation, a reused cap book, and clipping a negative saving.
    from copy import deepcopy
    from datetime import UTC

    from dao.prog.fastctrl import replay_finance as api
    from dao.prog.fastctrl.simulate import compare

    plan, replay, samples = f3b4_public_case(bonus_rules, edges=tuple(NOW + 900 * i for i in range(quarters + 1)))
    source = f3b4_source(plan, replay, samples)
    baseline, fast = modeled_case("f3b4-base", hold=900), modeled_case("f3b4-fast", hold=900)
    limits = f3b4_limits()
    inputs = (plan, replay, samples, source, baseline, fast, limits)
    before, raw = deepcopy(inputs), (replay.canonical_bytes(), source.canonical_bytes(), json.dumps(plan.to_dict()))
    calls = []
    settle = api.settle_series

    def observe(items, snapshots, opening):
        book = settle(items, snapshots, opening)
        calls.append((items, snapshots, opening, book))
        return book

    monkeypatch.setattr(api, "settle_series", observe)
    result = compare(plan, samples, limits, plan.specs[0].soc_min, timezone=UTC,
        replay_basis=replay, baseline_execution=baseline, fast_execution=fast, source_evidence=source)
    assert len(calls) == 2
    base, fast = result.baseline.financial_result, result.fast.financial_result
    assert base.projection.source is fast.projection.source
    assert base.projection.source.binding is fast.projection.source.binding
    assert base.projection.source.evidence is source
    assert calls[0][0] is not calls[1][0]
    assert base.settlement.end_state is not fast.settlement.end_state
    assert base.settlement == calls[0][3] and fast.settlement == calls[1][3]
    assert all(pinned is replay.snapshots and opening is replay.opening for _, pinned, opening, _ in calls)
    assert tuple(item.export_kwh for item in base.projection.items) == (D(".25"),) * quarters
    assert tuple(item.export_kwh for item in fast.projection.items) == (D("1"),) * quarters
    if quarters == 2:
        assert base.settlement.end_state.bonus.total_export_kwh == D("4999")
        assert fast.settlement.end_state.bonus.total_export_kwh == D("5000.5")
    for side in (result.baseline, result.fast):
        wrapper = side.financial_result
        assert side.coverage_status == side.financial_status == "complete"
        assert side.economics_is_estimate and side.economics_kind == "settlement_counterfactual"
        assert wrapper.source_trust == "caller_asserted"
        assert side.energy_cost == float(wrapper.settlement.net_cost)
        assert side.total_cost == float(wrapper.components.total_cost.value)
        assert not side.settlement_comparison_pending
        assert all(item.input_revision == 0 and item.quality == "estimated"
                   and "source_trust:caller_asserted" in item.sources for item in wrapper.projection.items)
    assert result.financial_result.advantage_eur == advantage
    assert result.saving == float(advantage)
    reverse = Comparison(result.fast, result.baseline)
    assert reverse.financial_result.advantage_eur == (D("-.195") if quarters == 1 else D("-.36"))
    assert reverse.saving < 0
    for comparison in (result, reverse):
        assert "caller_asserted" in comparison.report()
        assert "rekenprecisiediagnose — waargenomen verschil" in comparison.daily_table()
    assert len(calls) == 2  # Neither comparison nor either report opens another book.
    assert inputs == before
    assert (replay.canonical_bytes(), source.canonical_bytes(), json.dumps(plan.to_dict())) == raw


@pytest.mark.parametrize("fault, reason", [
    ("complete", None), ("source", "unknown_bonus_source"), ("basis", "replay_basis_required"),
    ("mismatch", "source_binding_mismatch"), ("fractional", "projection_noninteger_window"),
    ("fractional-start", "projection_noninteger_window"),
])
def test_f3b4_public_nullable_reports_use_financial_days(bonus_rules, fault, reason):
    # Catches totals from partial components and joining physical/financial dates.
    from datetime import UTC

    from dao.prog.fastctrl.simulate import compare

    plan, replay, samples = f3b4_public_case(bonus_rules)
    if fault == "fractional":
        samples[-1] = replace(samples[-1], timestamp=samples[-1].timestamp - .25)
    elif fault == "fractional-start":
        samples[0] = replace(samples[0], timestamp=samples[0].timestamp + .25)
    source = f3b4_source(plan, replay, samples)
    if fault == "mismatch":
        source = replace(source, meter_id="synthetic:wrong")
    result = compare(plan, samples, f3b4_limits(), plan.specs[0].soc_min, timezone=UTC,
        replay_basis=None if fault == "basis" else replay,
        baseline_execution=modeled_case("nullable-base", hold=900),
        fast_execution=modeled_case("nullable-fast", hold=900),
        source_evidence=None if fault == "source" else source)
    for side in (result.baseline, result.fast):
        assert side.coverage_status == "complete"
        assert list(side.days) == ["2026-12-31"]
        assert all(day.financial_components is day.terminal_value is day.total_cost is None
                   and day.energy_cost is day.wear_cost is None for day in side.days.values())
        if reason is not None:
            assert side.energy_cost is side.wear_cost is side.terminal_value is side.total_cost is None
            assert reason in side.financial_reasons
    if fault == "source":
        assert result.fast.financial_status == "partial"
        assert result.fast.financial_result.components.base_export_revenue.value == D(".4")
    outputs = result.report(), result.daily_table()
    for output in outputs:
        assert "rekenprecisiediagnose — waargenomen verschil" in output
        assert "financial_status=" in output
        if reason is not None:
            assert output.startswith("unknown_reconstruction:") and reason in output
            assert "Besparing:" not in output and "euro/jaar" not in output
        else:
            assert "settlement_counterfactual" in output
            assert "eindcorrectie" in output.lower()
    if reason is None:
        assert result.financial_result.advantage_eur == D(".36")
        assert tuple(day.key.local_date for day in result.fast.financial_result.days) == ("2027-01-01",)
        assert "2027-01-01" in outputs[1] and "Europe/Amsterdam" in outputs[1]
        assert "2026-12-31" not in outputs[1]
        assert "geraamde extrapolatie" in outputs[0].lower()
        assert "euro/dag" in outputs[0] and "euro/jaar" in outputs[0]
    else:
        assert result.financial_result.advantage_eur is None
        assert reason in result.saving


def f3b4_report_books(kind):
    """Real F3b3 books; the report must preserve their independent §24.1 pins."""
    from datetime import date

    from dao.prog.fastctrl import replay_finance as api
    from dao.tests.settlement.conftest import MONTH_END_TS
    from dao.tests.settlement.test_fast_replay_finance import (
        idle_totals,
        physical_battery,
        physical_segment,
        project_case,
    )

    start = MONTH_END_TS - 900 if kind == "invoice" else START_TS + 86400 - 900
    edges = tuple(start + 900 * i for i in range(3 if kind == "invoice" else 4))
    plan, replay = finance_case(edges=edges)
    if kind == "invoice":
        snapshot = canonical_snapshot(replay.snapshots[0],
            multiplier_consumption=((date(2020, 1, 1), D("1.123456789012")),),
            vat_consumption=((date(2020, 1, 1), D("21.123456789012")),))
        spots = tuple(SpotInterval(at, at + 3600, D(".234567890123"), "synthetic:f3b4-report", "estimated")
                      for at in range(start // 3600 * 3600, edges[-1], 3600))
        plan, replay = finance_case(edges=edges, snapshots=(snapshot,), spots=spots)
        runs = tuple(tuple(physical_segment(a, b, imported=amount)
                     for a, b, amount in zip(edges, edges[1:], amounts, strict=False))
                     for amounts in ((.125, .125), (.125, .25)))
        totals = idle_totals(api)
    else:
        directions = ("charge", "discharge", "charge") if kind == "mixed" else ("charge",) * 3
        segments = tuple(physical_segment(a, b, batteries=(physical_battery(
            charge=amount if direction == "charge" else 0.0,
            discharge=amount if direction == "discharge" else 0.0, dc=amount,
            modeled=4000 * amount * (1 if direction == "charge" else -1)),))
            for a, b, amount, direction in zip(edges, edges[1:], (.125, .125, .5), directions, strict=False))
        runs = ((physical_segment(edges[0], edges[-1]),), segments)
        totals = api.ReplayPhysicalTotals((api.BatteryFinancialTotals(
            "accu", D("10"), D.from_float(.03), D("50"), D("50")),))
    return tuple(api.settle_replay(project_case(api, plan, replay, segments), replay, totals) for segments in runs)


def f3b4_report_comparison(base, fast):
    from dao.prog.fastctrl.simulate import SimResult

    # Assign additively so pre-implementation RED reaches the real formatter,
    # rather than stopping at an unsupported constructor keyword.
    sides = []
    for label, book in (("baseline", base), ("fast", fast)):
        result = SimResult(label)
        result.financial_result = book
        result.financial_status = book.financial_status
        result.financial_reasons = book.financial_reasons
        result.economics_kind = "settlement_counterfactual"
        result.period_status = book.period_status
        sides.append(result)
    return Comparison(*sides)


@pytest.mark.parametrize("kind", ["invoice", "charge", "mixed"])
@pytest.mark.parametrize("reverse", [False, True])
def test_f3b4_reconciliation_reports_exact_text(kind, reverse):
    # Catches float/cent rendering, hidden small negatives, and rebooked diagnostics.
    from copy import deepcopy
    from fractions import Fraction

    from dao.prog.fastctrl import replay_finance as api

    base, fast = f3b4_report_books(kind)
    if kind == "invoice":
        totals = (D(".0797982192732042658615844001783507642"), D(".119697328909806398792376600267526146"))
        fast_days = (D(".0398991096366021329307922000891753821"), D(".0797982192732042658615844001783507642"))
        delta, advantage_delta = D("-3E-37"), D("3E-37")
        advantage = D("-.0398991096366021329307922000891753818")
    else:
        totals = (D(0), D(".0224999999999999991673327315311325947"))
        fast_days = (D(".00374999999999999986122212192185543245"), D(".0187499999999999993061106096092771622"))
        delta, advantage_delta = D("5E-38"), D("-5E-38")
        advantage = D("-.0224999999999999991673327315311325947")
    assert (base.components.total_cost.value, fast.components.total_cost.value) == totals
    assert tuple(day.components.total_cost.value for day in fast.days) == fast_days
    assert fast.reconciliation[-1].difference == delta
    assert Fraction(totals[1]) == sum(map(Fraction, fast_days)) + Fraction(delta)
    comparison = f3b4_report_comparison(fast, base) if reverse else f3b4_report_comparison(base, fast)
    expected_advantage = advantage.copy_negate() if reverse else advantage
    expected_delta = advantage_delta.copy_negate() if reverse else advantage_delta
    financial = api.compare_financial_results(fast, base) if reverse else api.compare_financial_results(base, fast)
    assert financial.advantage_eur == expected_advantage
    assert financial.reconciliation[0].difference == expected_delta
    before = deepcopy((base, fast, financial))
    report = comparison.report()
    # Wrapper authority also applies to the existing rounded money columns,
    # even if an old constructor left compatibility money fields at zero.
    metric = "Energiekosten" if kind == "invoice" else "Slijtagekosten"
    row = next(line for line in report.splitlines() if line.startswith(metric))
    amounts = (("0.12", "0.08", "-0.040") if reverse else ("0.08", "0.12", "+0.040")) if kind == "invoice" else (
        ("0.02", "0.00", "-0.022") if reverse else ("0.00", "0.02", "+0.022"))
    assert tuple(row.rsplit(maxsplit=3)[-3:]) == amounts
    for output in (report, comparison.daily_table()):
        table, marker, diagnostic = output.partition("rekenprecisiediagnose — waargenomen verschil")
        assert marker and "totaal minus exacte dagsom" in diagnostic and "geen boeking" in diagnostic
        assert "€0.00" not in diagnostic
        for label in ("baseline", "fast", "advantage"):
            assert label in diagnostic
        for label, entries in (("baseline", comparison.baseline.financial_result.reconciliation),
                               ("fast", comparison.fast.financial_result.reconciliation),
                               ("advantage", comparison.financial_result.reconciliation)):
            for entry in entries:
                if entry.status == "equal":
                    assert "exacte aansluiting" in diagnostic
                    continue
                line = next(line for line in diagnostic.splitlines()
                            if label in line and f"{entry.metric}:" in line)
                assert "difference" in line
                assert f"T={entry.authoritative_total}" in line and f"S={entry.day_sum}" in line
                signed = ("+" if entry.difference > 0 else "") + str(entry.difference)
                assert f"Δ={signed} EUR" in line
        assert "Δ=" not in table
    daily = comparison.daily_table().split("rekenprecisiediagnose — waargenomen verschil")[0]
    assert len([line for line in daily.splitlines() if line.startswith("2027-")]) == 2
    for day in base.days:
        assert daily.count(day.key.local_date) == 1
    assert (base, fast, comparison.financial_result) == before
    assert comparison.saving == float(expected_advantage)


@pytest.mark.parametrize("fault", ["partial", "basis", "mismatch", "coverage", "defaults", "structure", "mixed"])
def test_f3b4_reconciliation_reports_unknown_and_defaults(bonus_rules, fault):
    from dao.prog.fastctrl import replay_finance as api
    from dao.prog.fastctrl.simulate import SimResult

    base, fast = f3b4_report_books("invoice")
    reason = None
    if fault in ("basis", "mismatch"):
        reason = "replay_basis_required" if fault == "basis" else "source_binding_mismatch"
        fast = api.unknown_replay_financial(reasons=(reason,))
    elif fault == "partial":
        from dao.prog.fastctrl.simulate import compare

        plan, replay, samples = f3b4_public_case(bonus_rules)
        partial = compare(plan, samples, f3b4_limits(), plan.specs[0].soc_min, replay_basis=replay,
            baseline_execution=modeled_case("partial-base", hold=900),
            fast_execution=modeled_case("partial-fast", hold=900))
        base, fast = partial.baseline.financial_result, partial.fast.financial_result
        reason = "unknown_bonus_source"
    elif fault in ("coverage", "structure"):
        fast = replace(fast, days=fast.days[:-1] if fault == "coverage" else (*fast.days, fast.days[0]))
        reason = "replay_comparison_coverage_mismatch"
    else:
        base, fast = replace(base, reconciliation=()), replace(fast, reconciliation=())
    comparison = f3b4_report_comparison(base, fast)
    if fault == "mixed":
        comparison.baseline = SimResult("legacy model")
        reason = "comparison_basis_mismatch"
    if fault == "structure":
        for method in (comparison.report, comparison.daily_table):
            with pytest.raises(ValueError, match="^replay_reconciliation_structure_invalid$"):
                method()
        return
    for output in (comparison.report(), comparison.daily_table()):
        assert "rekenprecisiediagnose — waargenomen verschil" in output
        diagnostic = output.split("rekenprecisiediagnose — waargenomen verschil", 1)[1]
        for label, entries in (("baseline", () if comparison.baseline.financial_result is None else
                                comparison.baseline.financial_result.reconciliation),
                               ("fast", fast.reconciliation),
                               ("advantage", comparison.financial_result.reconciliation)):
            for entry in entries:
                if entry.status != "unknown":
                    continue
                line = next(line for line in diagnostic.splitlines()
                            if label in line and f"{entry.metric}: unknown" in line)
                assert "S=unknown; Δ=unknown (geen boeking)" in line
                assert f"T={'unknown' if entry.authoritative_total is None else entry.authoritative_total}" in line
                assert all(reason in line for reason in entry.reasons)
        if reason:
            assert reason in output and "unknown" in output
            assert "Besparing:" not in output and "euro/jaar" not in output
            assert comparison.financial_result.advantage_eur is None
        if fault in ("defaults", "mixed"):
            assert "unknown/reconciliation_not_computed" in output
    if fault == "defaults":
        simulation = importlib.import_module("dao.prog.fastctrl.simulate")
        old = api.ReplayFinancialComparison(None, "unknown", ("synthetic:unavailable",), ())
        text = "\n".join(simulation._reconciliation_lines("advantage", old.reconciliation))
        assert "unknown/reconciliation_not_computed" in text
        assert "T=0" not in text and "Δ=0" not in text
    legacy = Comparison(SimResult("legacy model"), SimResult("legacy model"))
    assert legacy.financial_result is None
    for output in (legacy.report(), legacy.daily_table()):
        assert "settlement_counterfactual" not in output
        assert "rekenprecisiediagnose" not in output


def _f3a8_profile_replay():
    from dao.prog.fastctrl.plan import BatteryPlanStep, FastPlan

    plan, replay = synthetic_replay()
    plan.specs = [replace(plan.specs[0], charge_efficiency=.8, discharge_efficiency=.6, cycle_cost=.03),
                  replace(plan.specs[0], name="B", capacity_kwh=20, discharge_efficiency=.5, cycle_cost=.09)]
    terminal = replace(replay.terminal_value,
                       reference_efficiency_by_battery=(("accu", D("0.7")), ("B", D("0.9"))))
    replay = replace(replay, terminal_value=terminal)
    plan.financial_context = replace(plan.financial_context, terminal_value=terminal,
                                     not_after_ts=plan.horizon_end_ts)
    for index, row in enumerate(plan.intervals):
        row.price_import = row.price_export = 999
        row.batteries = [
            BatteryPlanStep(1000 if index == 0 else 0, 40 - index, 39 - index,
                mean_ac_power_w=1000 if index == 0 else 0, on_power_w=1500 if index == 0 else 0,
                active_from_ts=row.start_ts, stop_ts=NOW + 600 if index == 0 else None),
            BatteryPlanStep(-1000 if index == 0 else 0, 80 - index, 79 - index,
                mean_ac_power_w=-1000 if index == 0 else 0, on_power_w=-1500 if index == 0 else 0,
                active_from_ts=NOW + 300 if index == 0 else row.start_ts, stop_ts=None),
        ]
    plan.price_average = 999
    return FastPlan.from_dict(bind_replay(plan, replay).to_dict()), replay


def test_f3a8_public_model_profile_and_terminal_are_independent_per_scenario(monkeypatch):
    from copy import deepcopy

    from dao.prog.fastctrl.command import plan_digest
    from dao.prog.fastctrl.policy import FastControlPolicy
    from dao.prog.fastctrl.receiver import receipt_matches_command
    from dao.prog.fastctrl.replay_execution import ProposalEvent, ReceiptEvent
    from dao.prog.fastctrl.simulate import compare

    plan, replay = _f3a8_profile_replay()
    samples = [Sample(NOW, 0), Sample(NOW + 900, 0)]
    socs, efficiencies = {"accu": 40, "B": 80}, {"accu": .5, "B": 1}
    baseline = modeled_case("f3a8-baseline", hold=900)
    fast = replace(modeled_case("f3a8-fast", hold=900),
                   execution=replace(baseline.execution, prepare_delay_s=.25))
    limits = PolicyLimits(min_benefit=1e9)
    before = deepcopy((plan, replay, samples, socs, efficiencies, baseline, fast, limits))
    policy_plans = []
    decide = FastControlPolicy.decide

    def observe(self, actual_plan, *args, **kwargs):
        policy_plans.append(deepcopy(actual_plan))
        return decide(self, actual_plan, *args, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "decide", observe)
    comparison = compare(plan, samples, limits, socs, inverter_efficiency=efficiencies,
                         replay_basis=replay, baseline_execution=baseline, fast_execution=fast)
    assert len(policy_plans) == 3
    for actual in policy_plans:
        assert actual.format_version == 2 and actual.financial_context == plan.financial_context
        assert actual.specs == plan.specs
        assert [row.batteries for row in actual.intervals] == [row.batteries for row in plan.intervals]
        assert actual.price_average == pytest.approx(.1)
        assert [(row.start_ts, row.end_ts) for row in actual.intervals] == [(q.start_ts, q.end_ts) for q in replay.quotes]
        assert all(row.price_import == row.price_export == .1 for row in actual.intervals)

    command_ids = []
    for result, scenario, delay in ((comparison.baseline, baseline, 0), (comparison.fast, fast, .25)):
        assert result.execution_kind == "modeled" and result.coverage_status == "complete"
        assert result.soc_start_by_battery == (("accu", 40), ("B", 80))
        assert result.opening_state is replay.opening and result.replay_terminal_value is replay.terminal_value
        assert dict(result.assumptions)["scenario_id"] == scenario.scenario_id
        assert json.loads(dict(result.assumptions)["execution_model"])["prepare_delay_s"] == delay
        assert result.source_plan_digest == plan_digest(plan)
        assert result.execution_plan_digest == plan_digest(policy_plans[0]) != result.source_plan_digest
        sends = [e for e in result.trace if isinstance(e, ProposalEvent) and e.command is not None]
        receipts = [e for e in result.trace if isinstance(e, ReceiptEvent)]
        command_ids.append({event.command.command_id for event in sends})
        assert len(sends) == len(receipts) == result.writes == 4
        assert [(e.command.battery_name, e.command.setpoint_w, e.at_ts - NOW) for e in sends
                if e.command.setpoint_w] == [("accu", 1500, delay), ("B", -1500, 300 + delay)]
        for event, receipt in zip(sends, receipts, strict=True):
            assert event.command.plan_digest == result.execution_plan_digest
            assert event.at_ts - event.timing.anchor_ts == delay
            assert event.command.issued_at_ms == round(event.timing.anchor_ts * 1000)
            assert receipt_matches_command(receipt.receipt, event.command)
            assert receipt.receipt.status == receipt.application.outcome == "accepted"
            assert receipt.at_ts == receipt.application.at_ts == event.at_ts
        spans = [(s.start_ts - NOW, s.end_ts - NOW, tuple(b.modeled_ac_w for b in s.batteries))
                 for s in result.segments]
        assert spans == ([(0, 300, (1500, 0)), (300, 600, (1500, -1500)), (600, 900, (0, -1500))]
            if delay == 0 else [(0, .25, (0, 0)), (.25, 300, (1500, 0)), (300, 300.25, (1500, 0)),
                (300.25, 600, (1500, -1500)), (600, 600.25, (0, -1500)), (600.25, 900, (0, -1500))])
        fraction = (600 - delay) / 600
        for index, charge, discharge, dc, cell in ((0, .25, 0, .125, .1), (1, 0, .25, .25, -.5)):
            batteries = [s.batteries[index] for s in result.segments]
            for field, expected in (("charge_ac_kwh", charge), ("discharge_ac_kwh", discharge),
                                    ("dc_throughput_kwh", dc), ("cell_delta_kwh", cell)):
                assert sum(getattr(b, field) for b in batteries) == pytest.approx(expected * fraction)
            assert all(b.charge_ac_kwh * b.discharge_ac_kwh == 0 for b in batteries)
        assert result.soc_end_by_battery == (("accu", pytest.approx(40 + fraction)),
                                             ("B", pytest.approx(80 - 2.5 * fraction)))
        assert result.import_kwh == pytest.approx(.125) and result.export_kwh == pytest.approx(.125)
        assert result.throughput_kwh == pytest.approx(.375 * fraction)
        assert result.wear_cost == pytest.approx((.03 * .125 + .09 * .25) * fraction)
        # .1*.7 - .5*.9 = -.38 AC-reference kWh; exactly one archived conversion.
        assert result.terminal_value == pytest.approx(float(D("-0.38") * replay.terminal_value.eur_per_ac_kwh) * fraction)
        assert result.energy_cost == pytest.approx(0, abs=1e-12)
        assert result.total_cost == pytest.approx(result.wear_cost - result.terminal_value)
        assert not result.settlement_comparison_pending
        assert result.financial_status == "complete"
        assert result.financial_result.settlement.net_cost == D(0)
    assert command_ids[0].isdisjoint(command_ids[1])
    assert comparison.baseline.soc_end_by_battery != comparison.fast.soc_end_by_battery
    assert comparison.baseline.terminal_value != comparison.fast.terminal_value
    assert comparison.saving == comparison.saving_per_day == float(comparison.financial_result.advantage_eur)
    for output in (comparison.report(), comparison.daily_table()):
        assert "settlement_counterfactual" in output and "settlement_comparison_pending" not in output
    changed = compare(plan, samples, limits, socs, inverter_efficiency=efficiencies, replay_basis=replay,
                      baseline_execution=replace(baseline, execution=replace(baseline.execution, prepare_delay_s=.125)),
                      fast_execution=fast)
    assert changed.fast == comparison.fast
    assert changed.baseline.soc_end_by_battery != comparison.baseline.soc_end_by_battery
    assert (plan, replay, samples, socs, efficiencies, baseline, fast, limits) == before


def test_f3a8_replay_has_no_io_or_test_imports(monkeypatch, bonus_rules):
    import ast
    import builtins
    import sys
    from copy import deepcopy
    from pathlib import Path
    from unittest.mock import Mock

    import mip
    import requests
    from sqlalchemy.engine import Engine

    import dao.prog.day_ahead as day_ahead
    from dao.prog import da_fast
    from dao.prog.da_base import DaBase
    from dao.prog.fastctrl import ownership, publication, replay_execution, replay_finance
    from dao.prog.fastctrl.simulate import compare
    from dao.prog.settlement.publication import SettlementPublication
    from dao.prog.settlement.repository import SettlementRepository
    from dao.tests.prog.test_fast_simulate import _f3a8_recorded_timeline

    simulation = importlib.import_module("dao.prog.fastctrl.simulate")
    archive = replay_api()

    def test_dependency(name):
        return any(part in ("tests", "fakes") or part.startswith(("test_", "fake_"))
                   for part in name.split("."))

    for module in (simulation, replay_execution, replay_finance, da_fast):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [f"{node.module or ''}.{alias.name}" for alias in node.names]
            else:
                continue
            assert not any(test_dependency(name) for name in names), (module.__name__, node.lineno, names)

    plan, replay = _f3a8_profile_replay()
    samples = [Sample(NOW, 0), Sample(NOW + 900, 0)]
    socs, efficiencies = {"accu": 40, "B": 80}, {"accu": .5, "B": 1}
    baseline, fast = modeled_case("pure-base", hold=900), modeled_case("pure-fast", hold=900)
    recorded_plan, recorded_samples, recorded = _f3a8_recorded_timeline()
    source_plan, source_replay, source_samples = f3b4_public_case(bonus_rules)
    source = f3b4_source(source_plan, source_replay, source_samples)
    source_scenarios = modeled_case("pure-source-base", hold=900), modeled_case("pure-source-fast", hold=900)
    limits = PolicyLimits(min_benefit=1e9)
    inputs = (plan, replay, samples, socs, efficiencies, baseline, fast, limits,
              recorded_plan, recorded_samples, recorded,
              source_plan, source_replay, source_samples, source, source_scenarios)
    before, archive_bytes = deepcopy(inputs), replay.canonical_bytes()
    source_bytes = source_replay.canonical_bytes(), source.canonical_bytes(), json.dumps(source_plan.to_dict())
    forbidden_calls = []

    def forbid(boundary):
        def fail(*args, **kwargs):
            forbidden_calls.append(boundary)
            pytest.fail(f"replay crossed {boundary}")
        return Mock(side_effect=fail)

    for target, names in (
        (archive, ("read_replay_basis", "write_replay_basis")),
        (mip.Model, ("optimize",)), (day_ahead.DaCalc, ("calc_optimum",)),
        (SettlementRepository, ("__init__", "read_inputs", "read_prices", "commit")),
        (Engine, ("connect", "raw_connection")),
        (HistoryLoader, ("load", "_series")),
        (ownership.OwnerStore, ("__init__",)), (ownership.OwnerCoordinator, ("__init__",)),
        (DaBase, ("__init__", "get_state", "call_service", "set_state")),
        (requests.Session, ("request",)),
        (publication, ("read_publication", "write_publication", "publication_lock", "publication_attempts")),
        (SettlementPublication, ("publish", "block")),
    ):
        for name in names:
            monkeypatch.setattr(target, name, forbid(f"{target.__name__}.{name}"))
    real_import, real_import_module = builtins.__import__, importlib.import_module

    def check_import(caller, names):
        if caller.startswith(("dao.prog.", "prog.")) and any(test_dependency(name) for name in names):
            forbidden_calls.append((caller, names))
            pytest.fail(f"product imported test/fake code: {caller}: {names}")

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        check_import((globals or {}).get("__name__", ""), (name, *(fromlist or ())))
        return real_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name, package=None):
        check_import(sys._getframe(1).f_globals.get("__name__", ""), (name,))
        return real_import_module(name, package)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    comparison = compare(plan, samples, limits, socs, inverter_efficiency=efficiencies,
                         replay_basis=replay, baseline_execution=baseline, fast_execution=fast)
    result = simulate(recorded_plan, recorded_samples, limits, socs, "pure recorded", True,
                      inverter_efficiency=efficiencies, execution=recorded)
    source_comparison = compare(source_plan, source_samples, limits, source_plan.specs[0].soc_min,
        replay_basis=source_replay, source_evidence=source,
        baseline_execution=source_scenarios[0], fast_execution=source_scenarios[1])
    for side in (source_comparison.baseline, source_comparison.fast):
        assert side.financial_status == "complete"
        assert side.financial_result.projection.source.evidence is source
        assert side.financial_result.settlement is not None
        assert side.financial_result.components.bonus_revenue.value > 0
    assert source_comparison.baseline.financial_result.settlement is not source_comparison.fast.financial_result.settlement
    source_comparison.report()
    source_comparison.daily_table()
    assert comparison.baseline.coverage_status == comparison.fast.coverage_status == "complete"
    assert comparison.baseline.throughput_kwh == comparison.fast.throughput_kwh == pytest.approx(.375)
    assert comparison.baseline.trace and comparison.fast.trace
    assert result.execution_kind == "recorded" and result.writes == 3 and result.trace
    assert result.coverage_status == "partial" and result.soc_end_by_battery[0] == ("accu", None)
    assert result.soc_end_by_battery[1][1] is not None
    comparison.report()
    comparison.daily_table()
    assert forbidden_calls == []
    assert inputs == before and replay.canonical_bytes() == archive_bytes
    assert (source_replay.canonical_bytes(), source.canonical_bytes(), json.dumps(source_plan.to_dict())) == source_bytes


def test_archive_is_canonical_content_addressed_immutable_and_opening_sensitive(tmp_path):
    api = replay_api()
    plan, replay = synthetic_replay()
    canonical = replay.canonical_bytes()
    with localcontext() as context:
        context.prec = 6
        assert replay.canonical_bytes() == canonical
    assert b"0.123456789012345678901234567890123456" in canonical
    digest = sha256(canonical).hexdigest()
    assert replay.content_hash == digest
    plan_path = tmp_path / "fast_plan.json"
    assert api.write_replay_basis(replay, plan_path=plan_path) == digest
    path = api.replay_archive_path(plan_path, digest)
    assert path.name == digest and path.read_bytes() == canonical
    assert api.write_replay_basis(replay, plan_path=plan_path) == digest
    read = api.read_replay_basis(plan_path, digest, financial_basis=replay.basis,
                                 terminal_value=replay.terminal_value)
    assert read.status == "known" and read.basis == replay
    assert read.basis.opening == SettlementState(NOW, None, None)
    assert read.basis.price_batch.items[0].spot.source == "synthetic-replay"
    assert read.basis.snapshots[0].snapshot_hash == plan.financial_context.basis.snapshot_hashes[0]
    with pytest.raises(FrozenInstanceError):
        replay.opening = SettlementState(NOW + 1, None, None)
    independent = replace(replay, opening=SettlementState(NOW, None, None))
    assert independent.opening is not replay.opening
    assert independent.canonical_bytes() == canonical
    # Same financial selection, distinct exact opening (not an alias of a shared mutable state).
    from dao.prog.settlement.types import BonusState
    other = replace(replay, opening=SettlementState(NOW, BonusState(START_TS, START_TS + 86400, NOW, D("7")), None))
    assert other.content_hash != digest
    assert api.write_replay_basis(other, plan_path=plan_path) != digest
    later_plan = ready_plan(hashes=replay.basis.snapshot_hashes, start=NOW + 60)
    later_plan.intervals[-1].end_ts = START_TS + 3600
    later = replay_for_plan(later_plan, replay.snapshots)
    assert later.basis.input_revision == replay.basis.input_revision
    assert later.opening.asof_ts == NOW + 60
    assert later.content_hash != digest
    assert api.write_replay_basis(later, plan_path=plan_path) != digest


@pytest.mark.parametrize("fault", ["missing", "corrupt", "hash", "basis", "terminal", "invalid-plan-path"])
def test_unknown_archive_never_falls_back_or_returns_financial_evidence(tmp_path, fault):
    api = replay_api()
    _, replay = synthetic_replay()
    plan_path = tmp_path / "fast_plan.json"
    digest = api.write_replay_basis(replay, plan_path=plan_path)
    path = api.replay_archive_path(plan_path, digest)
    expected_basis, terminal = replay.basis, replay.terminal_value
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(b"broken archive")
    elif fault == "hash":
        path.write_bytes(replay.canonical_bytes() + b" ")
    elif fault == "basis":
        expected_basis = replace(expected_basis, input_revision=2)
    elif fault == "terminal":
        terminal = replace(terminal, eur_per_ac_kwh=D("0.5"))
    else:
        plan_path.symlink_to(tmp_path / "not-a-plan")
    read = api.read_replay_basis(plan_path, digest, financial_basis=expected_basis, terminal_value=terminal)
    assert read.status == "unknown" and read.basis is None
    assert read.reason.startswith("replay_")


def test_archive_roundtrip_rejects_self_consistent_but_noncanonical_bytes(tmp_path):
    api = replay_api()
    _, replay = synthetic_replay()
    raw = json.dumps(json.loads(replay.canonical_bytes()), indent=2).encode()
    digest = sha256(raw).hexdigest()
    api.replay_archive_path(tmp_path / "plan", digest).write_bytes(raw)
    read = api.read_replay_basis(tmp_path / "plan", digest)
    assert read.status == "unknown" and read.basis is None


def test_ready_publication_has_durable_archive_before_ready_replace(system, monkeypatch):
    api = replay_api()
    import dao.prog.settlement.publication as publication

    replay = replay_for_plan(system.plan, (system.snapshot,))
    plan = bind_replay(system.plan, replay)
    ready = publication._write_verified_ready
    observed = []

    def check_before_ready(candidate, *, lease):
        digest = candidate.financial_context.replay_basis_hash
        raw = api.replay_archive_path(system.path, digest).read_bytes()
        assert sha256(raw).hexdigest() == digest
        read = api.read_replay_basis(system.path, digest)
        assert read.status == "known"
        assert read.basis.terminal_value == candidate.financial_context.terminal_value
        observed.append(True)
        return ready(candidate, lease=lease)

    monkeypatch.setattr(publication, "_write_verified_ready", check_before_ready)
    system.adapter.publish(plan, now_ts=NOW, replay_basis=replay)
    assert observed == [True]
    read = read_publication(system.path)
    assert read.status == "ready"
    assert not following_verdict(read, policy=system.adapter.policy(), now_ts=NOW).allowed


@pytest.mark.parametrize("fault", ["file-fsync", "directory-fsync", "verification"])
def test_archive_failure_blocks_instead_of_issuing_a_dangling_ready(system, monkeypatch, fault):
    api = replay_api()
    replay = replay_for_plan(system.plan, (system.snapshot,))
    # New archive; ready from fixture may not be retained on refusal.
    replay = replace(replay, opening=replace(replay.opening, month=None),
                     price_batch=replace(replay.price_batch, items=tuple(reversed(replay.price_batch.items))))
    plan = bind_replay(system.plan, replay)
    original = os.fsync
    failed = []

    def fsync(fd):
        target = os.readlink(f"/proc/self/fd/{fd}")
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        is_archive = ".replay_basis_" in target or target.endswith(replay.content_hash)
        if not failed and ((fault == "file-fsync" and is_archive)
                           or (fault == "directory-fsync" and directory)):
            failed.append(True)
            raise OSError("synthetic_archive_fsync_failure")
        return original(fd)

    if fault == "verification":
        def unreadable(*args, **kwargs):
            return api.ReplayRead("unknown", None, "replay_archive_unverifiable")
        monkeypatch.setattr(api, "read_replay_basis", unreadable)
    else:
        monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(FinancialPlanBlocked, match="replay_"):
        system.adapter.publish(plan, now_ts=NOW, replay_basis=replay)
    read = read_publication(system.path)
    assert read.status == "blocked" and read.plan is None


def test_missing_archive_without_supplied_basis_cannot_republish(system):
    api = replay_api()
    path = api.replay_archive_path(system.path, system.plan.financial_context.replay_basis_hash)
    path.unlink()
    with pytest.raises(FinancialPlanBlocked, match="replay_"):
        system.adapter.publish(system.plan, now_ts=NOW)
    assert read_publication(system.path).status == "blocked"


def test_later_tariff_revision_cannot_change_archived_evidence(system):
    api = replay_api()
    digest = system.plan.financial_context.replay_basis_hash
    path = api.replay_archive_path(system.path, digest)
    before = path.read_bytes()
    pricing = system.pricing.model_copy(deep=True)
    pricing.cost_supplier_consumption = {"2020-01-01": 0.7}
    revised = make_snapshot(pricing, pricing.settlement.contracts[0])
    system.repo.register_inputs(system.plan.financial_context.basis.meter_id, (), (revised,), 1)
    assert path.read_bytes() == before and sha256(before).hexdigest() == digest
    read = api.read_replay_basis(system.path, digest, financial_basis=system.plan.financial_context.basis)
    assert read.status == "known"
    assert read.basis.quotes[0].import_rate == D("0.1")
    assert read.basis.snapshots == (system.snapshot,)


def test_simulate_uses_explicit_archived_quotes_opening_terminal_without_io(monkeypatch):
    from dao.prog.fastctrl.simulate import compare

    api = replay_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    plan.intervals = [replace(i, price_import=999, price_export=999) for i in plan.intervals]
    plan.price_average = 999

    def forbidden(*args, **kwargs):
        pytest.fail("explicit ReplayBasis attempted DB/solver/optimizer/archive I/O")

    import mip

    import dao.prog.day_ahead as day_ahead
    from dao.prog.settlement.repository import SettlementRepository
    monkeypatch.setattr(SettlementRepository, "read_prices", forbidden)
    monkeypatch.setattr(SettlementRepository, "read_inputs", forbidden)
    monkeypatch.setattr(mip.Model, "optimize", forbidden)
    monkeypatch.setattr(day_ahead.DaCalc, "calc_optimum", forbidden)
    monkeypatch.setattr(HistoryLoader, "_series", forbidden)
    monkeypatch.setattr(api, "read_replay_basis", forbidden)
    samples = [Sample(NOW, 1000), Sample(NOW + 900, 1000)]
    result = simulate(plan, samples, PolicyLimits(), 50, "sandbox", False, replay_basis=replay,
                      execution=modeled_case("archived-quotes", hold=900))
    assert result.label.startswith("settlement_counterfactual")
    assert result.energy_cost == pytest.approx(0.025)  # 0.25 kWh * archived EUR 0.10, not 999.
    assert result.opening_state is replay.opening
    assert result.replay_terminal_value is replay.terminal_value
    assert result.soc_start == 50  # Settlement opening is not battery SoC.
    missing_basis = simulate(plan, samples, PolicyLimits(), 50, "missing basis", False,
                             execution=modeled_case("missing-basis", hold=900))
    assert missing_basis.energy_cost is missing_basis.total_cost is None
    assert missing_basis.import_kwh == pytest.approx(.25)
    assert "replay_basis_required" in missing_basis.reasons
    comparison = compare(plan, samples, PolicyLimits(), 50, replay_basis=replay,
                         baseline_execution=modeled_case("archived-baseline", hold=900),
                         fast_execution=modeled_case("archived-fast", hold=900))
    assert comparison.baseline.energy_cost == pytest.approx(.025)
    for side in (comparison.baseline, comparison.fast):
        assert side.opening_state is replay.opening
        assert side.replay_terminal_value is replay.terminal_value
    assert comparison.saving == float(comparison.financial_result.advantage_eur)
    assert comparison.baseline.financial_result.settlement.net_cost == D(".025")


def test_simulate_terminal_uses_archived_ac_reference_once_not_new_tariff_or_inverter():
    replay_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    # Keep the full 900s physical oracle within an explicitly valid context.
    plan.financial_context = replace(plan.financial_context, not_after_ts=plan.horizon_end_ts)
    step = plan.intervals[0].batteries[0]
    step.ac_power_w = step.mean_ac_power_w = step.on_power_w = 1000
    result = simulate(plan, [Sample(NOW, 0), Sample(NOW + 900, 0)], PolicyLimits(), 50,
                      "sandbox", False, inverter_efficiency=1, replay_basis=replay,
                      execution=modeled_case("archived-terminal", hold=900))
    # 0.25 cell kWh added, then valued with exactly archived EUR/kWh * archived eta 0.94.
    assert result.terminal_value == pytest.approx(float(D("0.25") * D("0.94") *
                            D("0.123456789012345678901234567890123456")))


@pytest.mark.parametrize("fault", ["hash", "basis", "terminal"])
def test_simulate_mismatch_is_explicit_unknown_without_a_euro_claim(fault):
    replay_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    context = plan.financial_context
    if fault == "hash":
        context = replace(context, replay_basis_hash="f" * 64)
    elif fault == "basis":
        context = replace(context, basis=replace(context.basis, input_revision=2))
    else:
        context = replace(context, terminal_value=replace(context.terminal_value, eur_per_ac_kwh=D("0.7")))
    plan = replace(plan, financial_context=context)
    result = simulate(plan, [Sample(NOW, 1000), Sample(NOW + 900, 1000)], PolicyLimits(),
                      50, "sandbox", False, replay_basis=replay,
                      execution=modeled_case("mismatch", hold=900))
    assert result.label.startswith("unknown_reconstruction:")
    assert result.energy_cost is result.wear_cost is result.terminal_value is result.total_cost is None


def test_replay_basis_rejects_a_different_financial_selection_or_opening():
    replay_api()
    _, replay = synthetic_replay()
    for changes in ({"basis": replace(replay.basis, input_revision=2)},
                    {"opening": SettlementState(NOW + 1, None, None)},
                    {"quotes": replay.quotes[1:]},
                    {"price_batch": replace(replay.price_batch, snapshots=())}):
        with pytest.raises((TypeError, ValueError)):
            replace(replay, **changes)


@pytest.mark.parametrize("fault", [
    "truncated", "missing-middle", "extra-interval", "duplicate-interval",
    "reordered", "gap", "empty", "fewer-quotes",
])
def test_fix1_replay_requires_complete_bidirectional_plan_quote_coverage(fault):
    plan, replay = synthetic_replay()
    if fault == "truncated":
        plan.intervals = plan.intervals[:1]
    elif fault == "missing-middle":
        plan.intervals = [plan.intervals[0], plan.intervals[2]]
    elif fault == "extra-interval":
        plan.intervals = [*plan.intervals, plan.intervals[-1]]
    elif fault == "duplicate-interval":
        plan.intervals = [plan.intervals[0], plan.intervals[0], plan.intervals[2]]
    elif fault == "reordered":
        plan.intervals.reverse()
    elif fault == "gap":
        plan.intervals[1] = replace(plan.intervals[1], start_ts=NOW + 901)
    elif fault == "empty":
        plan.intervals = []
    else:
        # A valid archive with two quotes covering the same full horizon,
        # against a plan with three intervals: no exact one-to-one pairing.
        replay = replace(replay, quotes=(replace(replay.quotes[0], end_ts=NOW + 1800), replay.quotes[2]))
    plan = bind_replay(plan, replay)
    result = simulate(plan, [Sample(NOW, 1000), Sample(NOW + 900, 1000)], PolicyLimits(),
                      50, "sandbox", True, replay_basis=replay,
                      execution=modeled_case("quote-coverage", hold=900))
    assert result.label.startswith("unknown_reconstruction:"), result
    assert result.energy_cost is result.wear_cost is result.terminal_value is result.total_cost is None


def test_fix1_valid_replay_policy_uses_average_of_exactly_paired_quotes(monkeypatch):
    from dao.prog.fastctrl.policy import FastControlPolicy

    plan, replay = synthetic_replay()
    rates = (D("0.1"), D("0.2"), D("0.6"))
    snapshot = canonical_snapshot(replay.snapshots[0], price_interval_seconds=900)
    edges = (NOW, NOW + 900, NOW + 1800, NOW + 2700)
    spots = tuple(SpotInterval(start, end, rate, "synthetic:quarter-policy", "estimated")
                  for start, end, rate in zip(edges[:-1], edges[1:], rates, strict=True))
    plan, replay = finance_case(edges=edges, snapshots=(snapshot,), spots=spots)
    plan.price_average = 999
    observed = []
    decide = FastControlPolicy.decide

    def capture_policy(self, actual_plan, *args, **kwargs):
        observed.append((actual_plan.price_average, [i.price_import for i in actual_plan.intervals]))
        return decide(self, actual_plan, *args, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "decide", capture_policy)
    samples = [Sample(NOW + i * 900, 1000) for i in range(4)]
    result = simulate(plan, samples, PolicyLimits(), 50, "sandbox", True, replay_basis=replay,
                      execution=modeled_case("quote-projection", hold=900))
    assert result.label.startswith("settlement_counterfactual:")
    assert observed == [(0.3, [0.1, 0.2, 0.6])] * 3
    assert plan.price_average == 999  # The mutable caller plan is not relabelled.


def comparison_with_unknown_result(unknown_side):
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    samples = [Sample(NOW, 1000), Sample(NOW + 900, 1000)]
    known = simulate(plan, samples, PolicyLimits(), 50, "known reconstruction", False,
                     replay_basis=replay, execution=modeled_case("known", hold=900))
    assert known.total_cost is not None
    plan = replace(plan, financial_context=replace(plan.financial_context, replay_basis_hash="f" * 64))
    unknown = simulate(plan, samples, PolicyLimits(), 50, "sandbox", False, replay_basis=replay,
                       execution=modeled_case("unknown", hold=900))
    assert unknown.label.startswith("unknown_reconstruction:") and unknown.total_cost is None
    return Comparison(unknown if unknown_side != "fast" else known,
                      unknown if unknown_side != "baseline" else known)


@pytest.mark.parametrize("unknown_side", ["baseline", "fast", "both"])
@pytest.mark.parametrize("consumer", ["saving", "saving_per_day", "report", "daily_table"])
def test_fix1_comparison_consumers_label_unknown_without_money_or_typeerror(unknown_side, consumer):
    comparison = comparison_with_unknown_result(unknown_side)
    output = getattr(comparison, consumer)
    if callable(output):
        output = output()
    assert isinstance(output, str) and output.startswith("unknown_reconstruction:")
    assert "Besparing:" not in output and "euro/jaar" not in output


@pytest.mark.parametrize("unknown_side", ["baseline", "fast", "both"])
def test_fix1_log_comparison_labels_unknown_without_money_or_typeerror(unknown_side, caplog):
    comparison = comparison_with_unknown_result(unknown_side)
    with caplog.at_level(logging.INFO):
        log_comparison(comparison)
    assert "unknown_reconstruction:" in caplog.text
    assert "Besparing:" not in caplog.text and "euro/jaar" not in caplog.text


@pytest.mark.parametrize("with_replay", [False, True])
def test_fix1_no_batteries_keeps_specific_configuration_error_with_or_without_replay(with_replay):
    plan, replay = synthetic_replay()
    plan = replace(bind_replay(plan, replay), specs=[])
    kwargs = {"replay_basis": replay} if with_replay else {}
    with pytest.raises(ValueError, match="^het plan bevat geen batterij$"):
        simulate(plan, [Sample(NOW, 1000), Sample(NOW + 900, 1000)], PolicyLimits(),
                  50, "sandbox", False, **kwargs)


def test_f3a6_archived_terminal_per_battery_once():
    from dao.prog.fastctrl.plan import BatteryPlanStep, FastPlan
    from dao.tests.prog.test_fast_simulate import _f3a5_case, _f3a6_result

    plan, replay = synthetic_replay()
    plan.specs = [replace(plan.specs[0], charge_efficiency=.8, discharge_efficiency=.6),
                  replace(plan.specs[0], name="B", capacity_kwh=20, discharge_efficiency=.5)]
    terminal = replace(replay.terminal_value,
                       reference_efficiency_by_battery=(("accu", D("0.7")), ("B", D("0.9"))))
    replay = replace(replay, terminal_value=terminal)
    plan.financial_context = replace(plan.financial_context, terminal_value=terminal)
    plan = bind_replay(plan, replay)
    for row in plan.intervals:
        row.batteries = [BatteryPlanStep(1500, 40, 40, mean_ac_power_w=1500, on_power_w=1500,
                                        active_from_ts=row.start_ts, stop_ts=None),
                         BatteryPlanStep(-1500, 80, 80, mean_ac_power_w=-1500, on_power_w=-1500,
                                         active_from_ts=row.start_ts, stop_ts=None)]
    plan = FastPlan.from_dict(plan.to_dict())
    kwargs = dict(case=_f3a5_case(), soc={"accu": 40, "B": 80},
                  efficiencies={"accu": .5, "B": 1}, replay_basis=replay)
    result = _f3a6_result(plan, ((NOW, 0), (NOW + 600, 0)), **kwargs)
    # A stores .1 cell kWh; B draws .5. Each gets only its OWN archived cell→AC eta.
    expected = float(D("-0.38") * D("0.123456789012345678901234567890123456"))
    assert result.terminal_value == expected
    assert result.soc_end_by_battery == (("accu", 41), ("B", 77.5))
    assert result.throughput_kwh == pytest.approx(.375)
    assert result.opening_state is replay.opening
    assert result.replay_terminal_value is terminal
    with localcontext() as context:
        context.prec = 6
        repeated = _f3a6_result(plan, ((NOW, 0), (NOW + 600, 0)), **kwargs)
    assert repeated.terminal_value == expected
    row = plan.intervals[0]
    row.batteries = [replace(step, ac_power_w=1000 if step.on_power_w > 0 else -1000,
                            mean_ac_power_w=1000 if step.on_power_w > 0 else -1000,
                            on_power_w=2 * step.on_power_w, stop_ts=NOW + 300)
                     for step in row.batteries]
    alternative = _f3a6_result(FastPlan.from_dict(plan.to_dict()), ((NOW, 0), (NOW + 600, 0)), **kwargs)
    assert alternative.soc_end_by_battery == result.soc_end_by_battery
    assert alternative.terminal_value == result.terminal_value


@pytest.mark.parametrize("fault", [None, "missing", "hash"])
def test_f3a6_v2_comparison_uses_settlement_or_concrete_unknown(fault, caplog):
    from dao.tests.prog.test_fast_simulate import _f3a5_case, _f3a6_result

    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    plan.price_average = 999
    plan.intervals = [replace(row, price_import=999, price_export=999) for row in plan.intervals]
    if fault == "hash":
        plan.financial_context = replace(plan.financial_context, replay_basis_hash="f" * 64)
    result = _f3a6_result(plan, ((NOW, 1000), (NOW + 900, 1000)), case=_f3a5_case(),
                          replay_basis=None if fault == "missing" else replay)
    assert result.coverage_status == "complete" and result.import_kwh == .25
    assert result.settlement_comparison_pending is False
    if fault is None:
        assert result.label == "settlement_counterfactual: f3a6"
        assert result.financial_status == "complete"
        assert result.financial_result.settlement.net_cost == D(".025")
        assert result.energy_cost == pytest.approx(.025)
        assert result.opening_state is replay.opening
        assert result.replay_terminal_value is replay.terminal_value
        assert result.terminal_value == 0
    else:
        assert result.label.startswith("unknown_reconstruction:")
        assert ("replay_basis_required" if fault == "missing" else "replay_basis_mismatch") in result.reasons
        assert result.energy_cost is result.wear_cost is result.terminal_value is result.total_cost is None
    comparison = Comparison(result, result)
    if fault is None:
        assert comparison.saving == comparison.saving_per_day == 0
    else:
        reason = "replay_basis_required" if fault == "missing" else "replay_basis_mismatch"
        assert reason in comparison.saving and comparison.saving == comparison.saving_per_day
        assert comparison.report().startswith(comparison.saving)
        assert comparison.daily_table().startswith(comparison.saving)
    with caplog.at_level(logging.INFO):
        log_comparison(comparison)
    assert "settlement_comparison_pending" not in caplog.text
    if fault is not None:
        assert "Besparing:" not in caplog.text and "euro/jaar" not in caplog.text
