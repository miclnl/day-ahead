"""Candidate audits reject solver residuals, not reinterpret them as source proof."""

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import mip
import pytest

from dao.prog.fastctrl.plan import FinancialBasis
from dao.prog.settlement.engine import settle_series
from dao.prog.settlement.service import PlanningContext
from dao.prog.settlement.types import (
    InputInterval,
    MonthState,
    PriceBatch,
    RegisteredSpot,
    SpotInterval,
)
from dao.tests.settlement.test_optimization_bonus import setup, ts

D = Decimal


def audit_api():
    from dao.prog.settlement import optimization_audit
    return optimization_audit


def financial_case(snapshot, bonus_rules, **kwargs):
    api = audit_api()
    model, bonus = setup(snapshot, bonus_rules, **kwargs)
    imports = tuple(model.add_var(ub=2, name=f"import_{i}") for i in range(len(bonus.quotes)))
    for var in imports:
        model += var == 0
    snapshots = tuple({p.snapshot.snapshot_hash: p.snapshot for p in bonus.quotes}.values())
    first, last = bonus.quotes[0].start_ts, bonus.quotes[-1].end_ts
    spots = {}
    for price in bonus.quotes:
        start = price.start_ts // 3600 * 3600
        spots[start] = RegisteredSpot(SpotInterval(start, start + 3600, price.spot_ex_vat,
            "synthetic-price", "estimated"), 1, "b" * 64)
    basis = FinancialBasis("synthetic-meter", 1, tuple(s.snapshot_hash for s in snapshots),
                           ts(hour=0), first, last, first, first)
    batch = PriceBatch("synthetic-meter", 1, first, last, tuple(spots.values()), snapshots, first, first)
    context = PlanningContext(basis, bonus.opening, bonus.quotes, batch)
    cost = mip.xsum(var * float(price.import_rate) - export * float(price.export_rate)
                   for var, export, price in zip(imports, bonus.grid_export, bonus.quotes, strict=True)) - bonus.revenue
    terms = api.FinancialModelTerms(context, bonus.quotes, imports, bonus, cost)
    model.objective = cost
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    return model, terms


def evidence(model):
    return audit_api().read_solver_evidence(model, cbc_log="")


def run_audit(model, terms):
    return audit_api().audit_planning_solution(terms,
        baseline_import_kwh=(D("0"),) * len(terms.quotes),
        baseline_export_kwh=(D("1"),) * len(terms.quotes), evidence=evidence(model))


def override_solution(monkeypatch, model, values):
    """Only controlled read doubles; these injected values are not CBC results."""
    original = model.solver.var_get_x
    monkeypatch.setattr(model.solver, "var_get_x", lambda var: values.get(var.idx, original(var)))


@pytest.mark.parametrize("energy,bonus", [("0", "0.03"), ("0.0015", "0")])
def test_forecast_engine_three_cents_and_direction(snapshot, bonus_rules, energy, bonus):
    model, terms = financial_case(snapshot, bonus_rules, cells=(energy,))
    result = run_audit(model, terms)
    assert result.forecast_result.lines[0].bonus_revenue == D(bonus)
    assert result.forecast_result.lines[0].bonus_kwh == (D("1") if energy == "0" else D("0"))
    assert result.forecast[0].quality == "estimated"
    assert result.forecast[0].input_revision == 0
    assert result.forecast[0].battery_discharge_kwh == D(energy)
    assert result.parity_complete
    assert result.parity_error_eur <= D("0.000001")
    assert result.evidence.cbc_primal_tol == result.evidence.infeas_tol
    assert "cbc_integer_native_readback_unavailable" in result.evidence.limitations


@pytest.mark.parametrize("power", [1e-12, -1e-12])
def test_source_audit_never_clamps_zero_status_residual(snapshot, bonus_rules, monkeypatch, power):
    model, terms = financial_case(snapshot, bonus_rules)
    proof = evidence(model)
    var = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    override_solution(monkeypatch, model, {var.idx: power})
    with pytest.raises(RuntimeError) as rejected:
        audit_api().validate_source_solution(terms.bonus, evidence=proof)
    assert rejected.value.code == "source_domain_solution_invalid"


@pytest.mark.parametrize("case,code", [
    ("below_minimum", "source_domain_solution_invalid"),
    ("normalizes_to_zero", "source_domain_normalization_invalid"),
])
def test_source_audit_never_clamps_residual_or_rounds_into_domain(snapshot, bonus_rules, monkeypatch, case, code):
    kwargs = {"cells": ("0.0015",)} if case == "below_minimum" else {"spots": ("-0.10",)}
    model, terms = financial_case(snapshot, bonus_rules, **kwargs)
    var = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    proof = evidence(model)
    value = 0.003999999996 if case == "below_minimum" else 1e-12
    override_solution(monkeypatch, model, {var.idx: value})
    with pytest.raises(RuntimeError) as rejected:
        audit_api().validate_source_solution(terms.bonus, evidence=proof)
    assert rejected.value.code == code


def test_negative_zero_is_zero_not_negative_energy(snapshot, bonus_rules, monkeypatch):
    model, terms = financial_case(snapshot, bonus_rules)
    var = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    override_solution(monkeypatch, model, {var.idx: -0.0})
    assert run_audit(model, terms).source.engine_cell_kwh == (D("0"),)


@pytest.mark.parametrize("value", [0.5, float("nan"), float("inf")])
def test_binary_and_solver_evidence_fail_closed(snapshot, bonus_rules, monkeypatch, value):
    model, terms = financial_case(snapshot, bonus_rules)
    proof = evidence(model)
    var = terms.bonus.source.discharge_on[0]
    override_solution(monkeypatch, model, {var.idx: value})
    with pytest.raises(RuntimeError) as rejected:
        audit_api().validate_source_solution(terms.bonus, evidence=proof)
    assert rejected.value.code == "source_domain_solution_invalid"


@pytest.mark.parametrize("case,code", [
    ("backend", "solver_evidence_unavailable"), ("tolerance", "solver_evidence_unavailable"),
    ("native_mismatch", "solver_evidence_unavailable"),
    ("no_incumbent", "solver_solution_unavailable"), ("nan_objective", "solver_solution_unavailable"),
])
def test_unverifiable_solver_evidence_is_rejected(snapshot, bonus_rules, monkeypatch, case, code):
    model, _ = financial_case(snapshot, bonus_rules)
    if case == "backend":
        monkeypatch.setattr(model, "solver_name", "UNKNOWN")
    elif case == "tolerance":
        monkeypatch.setattr(type(model), "integer_tol", property(lambda self: float("nan")))
    elif case == "no_incumbent":
        monkeypatch.setattr(model.solver, "get_num_solutions", lambda: 0)
    elif case == "nan_objective":
        monkeypatch.setattr(model.solver, "get_objective_value", lambda: float("nan"))
    else:
        import mip.cbc as cbc
        native = cbc.cbclib

        class NativeReadbackDouble(SimpleNamespace):
            def __getattr__(self, name):
                return getattr(native, name)

        monkeypatch.setattr(cbc, "cbclib", NativeReadbackDouble(Cbc_getPrimalTolerance=lambda handle: 0.1))
    with pytest.raises(RuntimeError) as rejected:
        evidence(model)
    assert rejected.value.code == code


def test_feasible_is_candidate_not_optimal_and_stale_evidence_rejected(snapshot, bonus_rules, monkeypatch):
    model, terms = financial_case(snapshot, bonus_rules)
    monkeypatch.setattr(model, "_status", mip.OptimizationStatus.FEASIBLE)
    proof = evidence(model)
    assert proof.status == "FEASIBLE"
    assert "solver_optimality_unproven" in proof.limitations
    assert run_audit(model, terms).parity_complete
    with pytest.raises(RuntimeError) as rejected:
        audit_api().validate_source_solution(terms.bonus, evidence=replace(proof, integer_tol=D("0.01")))
    assert rejected.value.code == "solver_evidence_unavailable"


@pytest.mark.parametrize("case", ["eligible", "rate", "quote", "bounds"])
def test_tampered_solution_rates_or_bounds_do_not_get_certified(snapshot, bonus_rules, monkeypatch, case):
    model, terms = financial_case(snapshot, bonus_rules)
    proof = evidence(model)
    if case == "eligible":
        override_solution(monkeypatch, model, {terms.bonus.eligible_kwh[0].idx: 0.0})
    elif case == "bounds":
        terms.bonus.source.cell_discharge.bus_power_vars[0][0].ub = 4
    elif case == "quote":
        prices = (replace(terms.quotes[0], import_rate=D("0.77")),)
        terms = replace(terms, quotes=prices, bonus=replace(terms.bonus, quotes=prices))
    else:
        terms = replace(terms, energy_cost=terms.energy_cost + 1)
    with pytest.raises(RuntimeError) as rejected:
        audit_api().audit_planning_solution(terms, baseline_import_kwh=(D("0"),),
                                            baseline_export_kwh=(D("1"),), evidence=proof)
    assert rejected.value.code in {"source_domain_solution_invalid", "planning_parity_failed", "planning_forecast_invalid"}


@pytest.mark.parametrize("case", ["origin", "history", "partial"])
def test_unknown_engine_bonus_is_not_zero_parity(snapshot, bonus_rules, case):
    kwargs = {"origins": ("unknown",)} if case == "origin" else (
        {"total": None} if case == "history" else {"starts": (ts(minute=5),), "durations": (600,)})
    model, terms = financial_case(snapshot, bonus_rules, **kwargs)
    result = run_audit(model, terms)
    assert terms.bonus.eligible_kwh[0].x == 0
    assert result.forecast_result.lines[0].bonus_revenue is None
    assert not result.parity_complete
    assert result.engine_energy_cost_eur is None
    assert result.savings_energy_eur is None
    assert "planning_parity_incomplete" in result.issues


def test_reset_restores_only_later_engine_lines(snapshot, bonus_rules):
    model, terms = financial_case(snapshot, bonus_rules, cells=("0", "0"), exports=("1", "1"),
        total=None, starts=(ts("2027-12-31", 23, 45), ts("2028-01-01", 0)))
    result = run_audit(model, terms)
    assert result.forecast_result.lines[0].bonus_revenue is None
    assert result.forecast_result.lines[1].bonus_revenue == D("0.03")
    assert not result.parity_complete
    assert result.savings_energy_eur is None


def test_a8_parity_excludes_month_adjustment_and_terminal_value(snapshot, bonus_rules):
    model, terms = financial_case(snapshot, bonus_rules, starts=(ts("2027-01-31", 23, 45),),
                                   spots=("-0.10",))
    month = MonthState("synthetic-2027", ts(hour=0), ts("2027-02-01", 0),
                        D("-0.20"), ((D("0"), D("0.20")),), True)
    opening = replace(terms.context.state, month=month)
    terms = replace(terms, context=replace(terms.context, state=opening),
                    bonus=replace(terms.bonus, opening=opening))
    result = run_audit(model, terms)
    assert result.forecast_result.adjustments[0].credit_total == D("0.30")
    assert result.engine_energy_cost_eur == D("0.10")
    assert result.forecast_result.net_cost == D("-0.20")
    assert result.parity_complete


def test_measured_subwh_discharge_still_excludes_bonus(snapshot, bonus_rules):
    model, terms = financial_case(snapshot, bonus_rules)
    first = run_audit(model, terms).forecast[0]
    measured = InputInterval(**{name: getattr(first, name) for name in first.__dataclass_fields__})
    measured = replace(measured, battery_discharge_kwh=D("0.0002"), quality="measured")
    result = settle_series((measured,), terms.context.price_batch.snapshots, terms.context.state)
    assert result.lines[0].bonus_kwh == D("0")
