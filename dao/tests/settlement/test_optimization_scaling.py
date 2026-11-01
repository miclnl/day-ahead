"""Report actual CBC availability separately from strict source correctness."""

import json
import math
import platform
from dataclasses import asdict, replace
from datetime import datetime
from decimal import Decimal, localcontext
from fractions import Fraction
from time import perf_counter
from zoneinfo import ZoneInfo

import mip
import pytest

from dao.prog.day_ahead import (
    _capture_native_stdout,
    bind_day_ahead_finance,
    finalize_day_ahead_finance,
    finish_day_ahead_finance,
)
from dao.prog.settlement.optimization_audit import (
    PlanningSolutionRejected,
    audit_planning_solution,
    read_solver_evidence,
)
from dao.prog.settlement.tariffs import quote
from dao.tests.settlement import conftest as settlement_fixtures
from dao.tests.settlement.test_optimization_bonus import POLICY, ts
from dao.tests.settlement.test_optimization_producer import context_for

D = Decimal
CASES = [(b, t, D(r), False) for t in (96, 192) for b in (0, 1, 4)
         for r in (("0",) if b == 0 else ("0", "0.8", "1", "1000", "100000"))]
CASES += [(b, 192, D("1000"), True) for b in (1, 4)]

# De 37 echte acceptaties uit task-5-scaling.log blijven afzonderlijk beschermd.
PREVIOUSLY_ACCEPTED = {
    (0, t, D("0"), mode) for t in (96, 192) for mode in ("fixed_zero", "free")
} | {
    (b, t, D(r), mode) for b in (1, 4) for t in (96, 192)
    for r in ("0", "0.8") for mode in ("fixed_zero", "free")
} | {
    (b, t, D(r), "fixed_zero") for b in (1, 4) for t in (96, 192) for r in ("1", "1000")
} | {
    (b, 96, D("1000"), "free") for b in (1, 4)
} | {
    (b, t, D("100000"), "fixed_minimum") for b in (1, 4) for t in (96, 192)
} | {
    (b, t, D("1000"), "fixed_minimum") for b, t in ((1, 96), (1, 192), (4, 96))
}


def test_cbc_scaling_case_coverage():
    """Removing difficult cases must not silently weaken availability evidence."""
    assert len(CASES) == 24
    assert len(set(CASES)) == 24
    positive = [(b, t, r, reset) for b, t, r, reset in CASES if b and r >= 1]
    assert len(positive) == 14
    assert sum(t == 192 for _, t, _, _ in positive) == 8
    assert sum(reset for _, _, _, reset in positive) * 3 == 6
    assert sum(bool(b) for b, _, _, _ in CASES) == 22
    assert 2 * len(CASES) + 22 + len(positive) == 84
    assert len(PREVIOUSLY_ACCEPTED) == 37


@pytest.mark.parametrize("battery_count,interval_count,upper_ratio,reset", CASES)
def test_cbc_source_domain_scaling(battery_count, interval_count, upper_ratio, reset, record_property):
    result = _run_case(battery_count=battery_count, interval_count=interval_count,
                       upper_ratio=upper_ratio, reset=reset)
    record_property("cbc_source_domain", json.dumps(result, sort_keys=True))
    print(json.dumps(result, sort_keys=True), flush=True)
    assert {"python", "mip", "B", "T", "upper_ratio", "reset", "scenarios"} <= result.keys()
    assert result["B"] == battery_count and result["T"] == interval_count
    modes = {item["scenario"]: item for item in result["scenarios"]}
    assert {"fixed_zero", "free"} <= modes.keys()
    if battery_count:
        gap = modes["fixed_gap"]
        assert gap["initial"]["status"] == "INFEASIBLE"
        assert gap["initial"]["incumbents"] == 0
        assert gap["final"] is None
        assert gap["additional_solves"] == 0
    if battery_count and upper_ratio >= 1:
        assert modes["free"]["final"] is not None and modes["free"]["final"]["accepted"], modes["free"]
    if reset:
        for mode in ("fixed_zero", "free", "fixed_minimum"):
            assert modes[mode]["final"] is not None and modes[mode]["final"]["accepted"], modes[mode]
    for item in modes.values():
        assert {"variables", "binaries", "constraints", "source_variables", "coefficient_range",
                "build_seconds", "initial", "final", "additional_solves", "refinement",
                "finalize_seconds", "objective_delta", "max_model_upper_cell_kwh"} <= item.keys()
        # Seven cap/eligibility handles plus two grid vars per step; one z at most.
        assert item["variables"] == (battery_count + 9) * interval_count + item["source_variables"]
        assert item["source_variables"] == (3 * interval_count // 4 if battery_count and upper_ratio >= 1 else 0)
        assert item["binaries"] <= 3 * interval_count
        assert item["constraints"] <= (16 + battery_count) * interval_count + 1
        assert D(item["max_model_upper_cell_kwh"]) >= upper_ratio * D("0.001")
        assert item["additional_solves"] in (0, 1)
        for candidate in (item["initial"], item["final"]):
            if candidate is None:
                continue
            assert {"solve_seconds", "audit_seconds", "status", "incumbents", "objective", "bound",
                    "gap", "max_nodes", "max_constraint_residual", "min_positive_D",
                    "max_auxiliary_residual_kwh", "source_rejections", "parity_error_eur",
                    "accepted", "evidence", "export_normalization"} <= candidate.keys()
            if candidate["incumbents"] == 0:
                for name in ("max_constraint_residual", "min_positive_D", "max_auxiliary_residual_kwh",
                             "parity_error_eur", "evidence", "export_normalization"):
                    assert candidate[name] is None
                assert not candidate["accepted"]
            if candidate["accepted"]:
                assert candidate["parity_complete"]
                assert D(candidate["parity_error_eur"]) <= D("0.000001")
                assert Fraction(candidate["max_auxiliary_residual_kwh"]) <= Fraction(1, 10**9)
                assert candidate["source_rejections"] == []
                assert candidate["evidence"]["status"] == candidate["status"]
                assert candidate["evidence"]["objective_value"] == candidate["objective"]
            else:
                assert candidate["parity_error_eur"] is None
        initial, final = item["initial"], item["final"]
        if not reset and (battery_count, interval_count, upper_ratio, item["scenario"]) in PREVIOUSLY_ACCEPTED:
            assert final is not None and final["accepted"], item
        if final is None:
            assert item["refinement"] is None
            assert item["objective_delta"] is None
            continue
        if item["additional_solves"]:
            refinement = item["refinement"]
            assert refinement is not None, item
            assert refinement["policy_version"] == "fixed_choice_refine_v1"
            assert refinement["initial_evidence"] == initial["evidence"]
            assert refinement["final_evidence"] == final["evidence"]
            assert refinement["objective_delta"] == item["objective_delta"]
            assert refinement["fixes"]
            assert {fix["reason"] for fix in refinement["fixes"]} <= {
                "integer_choice", "source_zero", "active_cell_bound"}
            for fix in refinement["fixes"]:
                assert D(fix["lower_before"]) <= D(fix["fixed_value"]) <= D(fix["upper_before"])
            if final["accepted"]:
                assert "refined_original_optimality_unproven" in final["issues"]
        else:
            assert item["refinement"] is None
            assert initial["evidence"] == final["evidence"]
            assert initial["objective"] == final["objective"]
            assert initial["max_constraint_residual"] == final["max_constraint_residual"]
            assert initial["min_positive_D"] == final["min_positive_D"]
            assert D(item["objective_delta"]) == 0


def _finite_text(value):
    return str(value) if value is not None and math.isfinite(value) else None


class _TimedCbcModel(mip.Model):
    """Count and time actual CBC solves, including the finalizer's optional solve."""

    def __init__(self):
        super().__init__(solver_name=mip.CBC)
        self.solve_seconds = []

    def optimize(self):
        began = perf_counter()
        try:
            return super().optimize()
        finally:
            self.solve_seconds.append(perf_counter() - began)


def _evidence_dict(evidence):
    return {name: str(value) if isinstance(value, Decimal) else list(value) if isinstance(value, tuple) else value
            for name, value in asdict(evidence).items()}


def _rejection(exc):
    return {"code": exc.code, "interval_index": exc.interval_index}


def _raw_auxiliary_residual(terms, evidence):
    """Diagnose this known fixture from raw exports, not engine-normalized quantities.

    Source choices are interpreted only when unique within the recorded integer
    tolerance. This diagnostic never accepts a physical source or changes a read.
    """
    bonus = terms.bonus
    cumulative = Fraction(bonus.opening.bonus.total_export_kwh)
    previous_year = None
    maximum = None
    for i, price in enumerate(terms.quotes):
        year = datetime.fromtimestamp(price.start_ts, ZoneInfo("Europe/Amsterdam")).year
        if previous_year is not None and year != previous_year:
            cumulative = Fraction(0)
        cap = Fraction(price.snapshot.bonus.yearly_cap_kwh)
        q_begin = min(cap, cumulative)
        cumulative += Fraction(D(str(bonus.grid_export[i].x)))
        q_end = min(cap, cumulative)
        first = q_end - q_begin
        eligible = Fraction(0)
        if bonus.source.required[i]:
            z = bonus.source.discharge_on[i]
            if z is None:
                on = 0  # Deze fixture heeft dan uitsluitend het exacte nuldomein.
            else:
                raw = Fraction(D(str(z.x)))
                choices = [value for value in (0, 1) if z.lb <= value <= z.ub
                           and abs(raw - value) <= Fraction(evidence.integer_tol)]
                if len(choices) != 1:
                    return None
                on = choices[0]
            if on == 0:
                eligible = first
        references = (("cumulative_export_kwh", cumulative), ("capped_opening_kwh", q_begin),
                      ("capped_export_kwh", q_end), ("first_kwh", first), ("eligible_kwh", eligible))
        for field, reference in references:
            raw = Fraction(D(str(getattr(bonus, field)[i].x)))
            residual = raw - reference
            if maximum is None or abs(residual) > abs(Fraction(maximum["residual_kwh"])):
                maximum = {"interval_index": i, "field": field, "raw_kwh": str(raw),
                           "reference_kwh": str(reference), "residual_kwh": str(residual)}
        previous_year = year
    return maximum


def _candidate_snapshot(terms, *, solve_seconds, cbc_log="", evidence=None):
    """Freeze raw metrics before any later solve can replace the incumbent."""
    model = terms.bonus.source.cell_discharge.model
    item = {
        "status": model.status.name, "incumbents": model.num_solutions,
        "objective": _finite_text(model.objective_value), "bound": _finite_text(model.objective_bound),
        "gap": _finite_text(model.gap), "solve_seconds": solve_seconds, "audit_seconds": None,
        "max_nodes": model.max_nodes, "max_mip_gap_abs": str(model.max_mip_gap_abs),
        "threads": model.threads, "max_constraint_residual": None, "min_positive_D": None,
        "max_auxiliary_residual_kwh": None, "max_auxiliary_residual": None,
        "accepted": False, "source_rejections": [], "evidence": None,
        "parity_complete": None, "parity_error_eur": None, "issues": [], "export_normalization": None,
        "diagnostic_limitations": [],
    }
    if not model.num_solutions or model.status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        item["diagnostic_limitations"].append("solver_solution_unavailable")
        return item, None
    if any(_finite_text(var.x) is None for var in model.vars):
        item["diagnostic_limitations"].append("nonfinite_or_missing_raw_solution")
        return item, None
    item["max_constraint_residual"] = max(c.expr.violation for c in model.constrs)
    cell = terms.bonus.source.cell_discharge
    totals = [sum((Fraction(D(str(row[i].x))) * cell.cell_coefficients[b][i]
                   for b, row in enumerate(cell.bus_power_vars)), Fraction(0))
              for i in range(len(terms.quotes))]
    minimum = min((value for value in totals if value > 0), default=None)
    item["min_positive_D"] = None if minimum is None else str(minimum)
    try:
        if evidence is None:
            evidence = read_solver_evidence(model, cbc_log=cbc_log)
        item["evidence"] = _evidence_dict(evidence)
        maximum = _raw_auxiliary_residual(terms, evidence)
        item["max_auxiliary_residual"] = maximum
        if maximum is not None:
            item["max_auxiliary_residual_kwh"] = str(abs(Fraction(maximum["residual_kwh"])))
        else:
            item["diagnostic_limitations"].append("auxiliary_source_choice_unavailable")
    except PlanningSolutionRejected as exc:
        item["source_rejections"].append(_rejection(exc))
    return item, evidence


def _record_audit(item, result, terms):
    """Only a successful real audit supplies parity/normalization measurements."""
    for i, raw in enumerate(result.source.raw_cell_kwh):
        if terms.bonus.source.required[i]:
            assert raw == 0 or raw >= Fraction(1, 1000)
        if raw > 0:
            assert result.forecast_result.lines[i].bonus_kwh == 0
    minimum = min((value for value in result.source.raw_cell_kwh if value > 0), default=None)
    assert item["min_positive_D"] == (None if minimum is None else str(minimum))
    max_delta = budget = max_budget = max_first_budget = Fraction(0)
    previous_year = None
    for price, var, forecast in zip(terms.quotes, terms.bonus.grid_export, result.forecast, strict=True):
        year = datetime.fromtimestamp(price.start_ts, ZoneInfo("Europe/Amsterdam")).year
        if year != previous_year:
            budget = Fraction(0)
        before = budget
        delta = abs(Fraction(forecast.export_kwh) - Fraction(D(str(var.x))))
        budget += delta
        max_delta, max_budget = max(max_delta, delta), max(max_budget, budget)
        max_first_budget = max(max_first_budget, before + budget)
        previous_year = year
    item.update(accepted=True, parity_complete=result.parity_complete,
                parity_error_eur=str(result.parity_error_eur), issues=list(result.issues),
                export_normalization={"max_absolute_delta_kwh": str(max_delta),
                                      "max_cumulative_budget_kwh": str(max_budget),
                                      "max_first_budget_kwh": str(max_first_budget)})


def _refinement_dict(report):
    if report is None:
        return None
    fixes = [{name: str(value) if isinstance(value, Decimal) else value
              for name in ("var_index", "var_type", "original_value", "lower_before", "upper_before",
                           "fixed_value", "reason") for value in (getattr(fix, name),)}
             for fix in report.recipe.fixes]
    return {"policy_version": report.policy_version,
            "initial_evidence": _evidence_dict(report.initial_evidence),
            "final_evidence": _evidence_dict(report.final_evidence),
            "original_model_digest": report.recipe.original_model_digest,
            "fixes": fixes, "objective_delta": str(report.objective_delta)}


def _run_case(*, battery_count: int, interval_count: int, upper_ratio: Decimal, reset: bool) -> dict[str, object]:
    """Synthetic PV plus bounded gross cell energy; no mocked solver or audit."""
    snapshot = settlement_fixtures.snapshot.__wrapped__()
    rules = settlement_fixtures.bonus_rules.__wrapped__()
    first = ts("2027-12-31" if reset else "2027-01-01", 12)
    intervals = tuple((first + i*900, first + (i+1)*900) for i in range(interval_count))
    context = context_for(snapshot, rules, intervals, total="4999.5")
    spots = tuple(replace(item, spot=replace(item.spot, price_ex_vat=(D("0.1"), D("0.2"), D("-0.05"), D("0.12"))[i%4]))
                  for i, item in enumerate(context.price_batch.items))
    prices = tuple(quote(context.quotes[i].snapshot, a, b, spots[(a-first)//3600].spot.price_ex_vat)
                   for i, (a, b) in enumerate(intervals))
    context = replace(context, quotes=prices, price_batch=replace(context.price_batch, items=spots))
    upper = upper_ratio * D("0.001")
    modes = ["fixed_zero", "free"]
    if battery_count:
        modes.append("fixed_gap")
        if upper_ratio >= 1:
            modes.append("fixed_minimum")
    scenarios = []
    for mode in modes:
        began = perf_counter()
        model = _TimedCbcModel()
        model.verbose = 1
        model.threads = 1
        model.max_nodes = 1500
        model.max_mip_gap_abs = 0.005
        bus_upper = float(upper * 4 / battery_count) if battery_count else 0
        cells = tuple(tuple(model.add_var(ub=bus_upper) for _ in intervals) for _ in range(battery_count))
        imports = tuple(model.add_var(ub=0) for _ in intervals)
        exports = tuple(model.add_var(ub=float(D("1")+upper)) for _ in intervals)
        for i in range(interval_count):
            total = mip.xsum(row[i] * 0.25 for row in cells)
            model += exports[i] == 1 + total
            if mode == "fixed_zero":
                for row in cells:
                    model += row[i] == 0
            elif mode in ("fixed_gap", "fixed_minimum"):
                amount = (0.0005 if mode == "fixed_gap" else 0.001) if i == 0 else 0
                model += total == amount
        if battery_count and mode == "free":
            model += mip.xsum(var * 0.25 for row in cells for var in row) <= float(upper * interval_count / 4)
        terms = bind_day_ahead_finance(model, context=context, intervals=intervals,
            meter_id="synthetic-meter", dc_from_bat=cells, installation_battery_count=battery_count,
            battery_indices=tuple(range(battery_count)),
            no_battery_evidence="synthetic inventory" if not battery_count else None,
            hour_fractions=(D("0.25"),)*interval_count, discharge_efficiencies=(D("1"),)*battery_count,
            grid_import=imports, grid_export=exports,
            export_bounds_kwh=tuple(D(str(var.ub)) for var in exports),
            origins=("verified_pv",)*interval_count, source_policy=POLICY)
        model.objective = terms.energy_cost
        coefficients = [abs(value) for c in model.constrs for value in c.expr.expr.values() if value]
        item = {
            "scenario": mode, "variables": model.num_cols, "binaries": model.num_int,
            "constraints": model.num_rows,
            "source_variables": sum(var is not None for var in terms.bonus.source.discharge_on),
            "max_model_upper_cell_kwh": str(max(terms.bonus.source.cell_discharge.model_upper_bounds_kwh)),
            "coefficient_range": [str(min(coefficients)), str(max(coefficients))],
            "build_seconds": perf_counter()-began, "initial": None, "final": None,
            "additional_solves": 0, "finalize_seconds": None, "refinement": None,
            "objective_delta": None, "finalization_rejections": [], "failed_candidate": None,
        }
        with _capture_native_stdout() as native:
            model.optimize()
        initial, evidence = _candidate_snapshot(terms, solve_seconds=model.solve_seconds[0],
                                                 cbc_log=native["cbc_log"])
        item["initial"] = initial
        if evidence is not None:
            auditing = perf_counter()
            try:
                result = audit_planning_solution(terms, baseline_import_kwh=(D("0"),)*interval_count,
                                                 baseline_export_kwh=(D("1"),)*interval_count, evidence=evidence)
                _record_audit(initial, result, terms)
            except PlanningSolutionRejected as exc:
                initial["source_rejections"].append(_rejection(exc))
            finally:
                initial["audit_seconds"] = perf_counter()-auditing
        if model.num_solutions and model.status in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
            finalizing = perf_counter()
            try:
                selected = finalize_day_ahead_finance(terms, cbc_log=native["cbc_log"])
            except PlanningSolutionRejected as exc:
                item["finalization_rejections"].append(_rejection(exc))
                # Dit is geen selectie/audit. Originele handles blijven bruikbaar
                # voor ruwe diagnostiek, niet als actuele binding na boundmutatie.
                item["failed_candidate"], _ = _candidate_snapshot(terms,
                    solve_seconds=model.solve_seconds[-1] if len(model.solve_seconds) > 1 else 0.0)
                selected = None
            finally:
                item["finalize_seconds"] = perf_counter()-finalizing
                item["additional_solves"] = len(model.solve_seconds)-1
            if selected is not None:
                final, _ = _candidate_snapshot(selected.terms, evidence=selected.evidence,
                    solve_seconds=model.solve_seconds[-1] if item["additional_solves"] else 0.0)
                item["final"] = final
                item["refinement"] = _refinement_dict(selected.refinement)
                with localcontext() as precision:
                    precision.prec = 36
                    item["objective_delta"] = str(D(final["objective"])-D(initial["objective"]))
                auditing = perf_counter()
                try:
                    result = finish_day_ahead_finance(selected, org_l=(0.0,)*interval_count,
                                                      org_t=(-1.0,)*interval_count)
                    _record_audit(final, result, selected.terms)
                    # Financiële uitkomst uitsluitend van de finale canonieke engineaudit.
                    final["canonical_financial_outcome"] = {
                        "engine_energy_cost_eur": str(result.engine_energy_cost_eur),
                        "baseline_energy_cost_eur": str(result.baseline_energy_cost_eur),
                        "savings_energy_eur": str(result.savings_energy_eur),
                    }
                except PlanningSolutionRejected as exc:
                    final["source_rejections"].append(_rejection(exc))
                finally:
                    final["audit_seconds"] = perf_counter()-auditing
        scenarios.append(item)
    return {"python": platform.python_version(), "mip": mip.__version__, "B": battery_count,
            "T": interval_count, "upper_ratio": str(upper_ratio), "reset": reset, "scenarios": scenarios}
