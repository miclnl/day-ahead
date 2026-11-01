"""Candidate audits separate exact source proof from bounded auxiliary roundoff."""

import math
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
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
    model, bonus = setup(snapshot, bonus_rules, **kwargs)
    return financial_terms(model, bonus)


def financial_terms(model, bonus):
    api = audit_api()
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
    floats = {index: float(value) for index, value in values.items()}
    monkeypatch.setattr(model.solver, "var_get_x", lambda var: floats.get(var.idx, original(var)))


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
    assert rejected.value.code == {"eligible": "planning_auxiliary_mismatch",
        "bounds": "source_domain_solution_invalid", "rate": "planning_parity_failed",
        "quote": "planning_forecast_invalid"}[case]


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


@pytest.mark.parametrize("case", ["roundoff", "negative_eligible", "positive_source",
                                 "positive_zero", "negative_zero", "below_minimum"])
def test_auxiliary_residual_does_not_relax_source(snapshot, bonus_rules, monkeypatch, case):
    """Auxiliary roundoff may survive, but no cell remainder gains source permission."""
    active = case in ("positive_source", "below_minimum")
    model, terms = financial_case(snapshot, bonus_rules, cells=("0.0015" if active else "0",),
                                  total="5000" if case == "negative_eligible" else "0")
    bonus = terms.bonus
    values = {bonus.first_kwh[0].idx: D("1.0000000000000004"),
              bonus.eligible_kwh[0].idx: D("1.0000000000000007")}
    if case == "negative_eligible":
        values = {bonus.eligible_kwh[0].idx: D("-1e-16")}
    elif active:
        values = {bonus.eligible_kwh[0].idx: D("1e-16")}
    cell = bonus.source.cell_discharge.bus_power_vars[0][0]
    if case in ("positive_zero", "negative_zero", "below_minimum"):
        values[cell.idx] = {"positive_zero": 1e-12, "negative_zero": -1e-12,
                            "below_minimum": 0.003999999996}[case]
    override_solution(monkeypatch, model, values)
    if case in ("positive_zero", "negative_zero", "below_minimum"):
        with pytest.raises(RuntimeError) as rejected:
            run_audit(model, terms)
        assert rejected.value.code == "source_domain_solution_invalid"
        return
    result = run_audit(model, terms)
    assert D(str(bonus.eligible_kwh[0].x)) == values[bonus.eligible_kwh[0].idx]
    assert any("planning_auxiliary_residual" in issue and "eligible_kwh" in issue
               for issue in result.issues)
    if case == "positive_source":
        assert result.forecast_result.lines[0].bonus_kwh == 0
        assert result.forecast_result.lines[0].bonus_revenue == 0


@pytest.mark.parametrize("field", ["cumulative_export_kwh", "capped_opening_kwh",
                                   "capped_export_kwh", "first_kwh", "eligible_kwh"])
@pytest.mark.parametrize("delta,accepted", [("1e-9", True), ("above", False),
                                           ("-1e-9", True), ("below", False),
                                           ("0.01", False), ("NaN", False), ("Infinity", False)])
def test_auxiliary_quantity_mismatch_rejected_at_zero_price(
        snapshot, bonus_rules, monkeypatch, field, delta, accepted):
    """Zero euro parity must not hide a wrong cap quantity, including at the exact budget."""
    model, terms = financial_case(snapshot, bonus_rules, spots=("0",))
    var = getattr(terms.bonus, field)[0]
    original = D(str(var.x))
    if delta in ("above", "below"):
        direction = 1 if delta == "above" else -1
        value = math.nextafter(float(original + direction * D("1e-9")), direction * math.inf)
        assert abs(D(str(value)) - original) > D("1e-9")
    else:
        value = float(original + D(delta))
    override_solution(monkeypatch, model, {var.idx: value})
    if accepted:
        result = run_audit(model, terms)
        assert result.parity_complete
        assert result.model_energy_cost_eur == result.engine_energy_cost_eur == 0
        assert any("planning_auxiliary_residual" in issue and field in issue for issue in result.issues)
    else:
        with pytest.raises(RuntimeError) as rejected:
            run_audit(model, terms)
        assert rejected.value.code == "planning_auxiliary_mismatch"


@pytest.mark.parametrize("case", ["missing", "foreign", "replaced", "duplicate", "unexpected"])
def test_auxiliary_handles_are_bound_not_just_numeric(snapshot, bonus_rules, monkeypatch, case):
    model, terms = financial_case(snapshot, bonus_rules, total=None if case == "unexpected" else "0")
    bonus = terms.bonus
    var = bonus.eligible_kwh[0]
    if case == "foreign":
        other = mip.Model(solver_name=mip.CBC)
        replacement = other.add_var(ub=2)
    elif case == "replaced":
        replacement = mip.Var(model, var.idx)
    elif case == "duplicate":
        replacement = bonus.first_kwh[0]
    else:
        replacement = None
    if case == "unexpected":
        bonus = replace(bonus, first_kwh=(var,))
    else:
        bonus = replace(bonus, eligible_kwh=(replacement,))
    terms = replace(terms, bonus=bonus)
    with pytest.raises(RuntimeError) as rejected:
        run_audit(model, terms)
    assert rejected.value.code == "planning_auxiliary_mismatch"


def test_source_rejects_positive_original_lower_bound(snapshot, bonus_rules, monkeypatch):
    """Being above L does not authorize being below a battery's recorded positive lb."""
    from dao.tests.settlement.test_optimization_refinement import positive_lower_case

    model, terms, proof = positive_lower_case(snapshot, bonus_rules, monkeypatch)
    with pytest.raises(RuntimeError) as rejected:
        audit_api().validate_source_solution(terms.bonus, evidence=proof)
    assert rejected.value.code == "source_domain_solution_invalid"
    assert terms.bonus.source.cell_discharge.bus_power_vars[0][0].lb == 0.005


@pytest.mark.parametrize("choice", [0, 1])
def test_fixed_binary_readback_remains_semantically_binary(snapshot, bonus_rules, monkeypatch, choice):
    """Refinement keeps the original native integer handle, despite CBC's B-to-I label."""
    import mip.cbc as cbc

    from dao.prog.settlement import optimization_refinement as refinement

    kwargs = {"cells": ("0.0015",), "bounds": (D("0.0015"),)} if choice else {}
    model, terms = financial_case(snapshot, bonus_rules, **kwargs)
    binary = terms.bonus.source.discharge_on[0]
    assert binary.var_type == mip.BINARY
    assert audit_api().validate_source_solution(terms.bonus, evidence=evidence(model)).discharge_on == (choice,)

    def forbidden_type_setter(*args, **kwargs):
        pytest.fail("A fixed binary must never be relabelled with a var_type setter")

    monkeypatch.setattr(model.solver, "var_set_var_type", forbidden_type_setter)
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    with monkeypatch.context() as reads:
        override_solution(reads, model, {cell.idx: 0.006000000000000001 if choice else 1e-13})
        recipe = refinement.prepare_source_refinement(terms, evidence=evidence(model))
        rebound = refinement.apply_source_refinement(terms, recipe=recipe)
    assert rebound.bonus.source.discharge_on[0] is binary
    assert binary.lb == binary.ub == choice
    assert binary.var_type == mip.INTEGER
    assert cbc.cbclib.Cbc_isInteger(model.solver._model, binary.idx)
    fixed_digest = refinement.model_digest(model)
    native_type = model.solver.var_get_var_type
    with monkeypatch.context() as reads:
        reads.setattr(model.solver, "var_get_var_type",
                      lambda var: mip.BINARY if var is binary else native_type(var))
        assert refinement.model_digest(model) == fixed_digest
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    result = run_audit(model, rebound)
    assert result.source.discharge_on == (choice,)
    assert result.source.engine_cell_kwh == (D("0.0015") if choice else D("0"),)
    for var, value in ((cell, 0 if choice else 1e-12), (binary, 1 - choice)):
        with monkeypatch.context() as reads:
            override_solution(reads, model, {var.idx: value})
            with pytest.raises(RuntimeError) as rejected:
                audit_api().validate_source_solution(rebound.bonus, evidence=evidence(model))
            assert rejected.value.code == "source_domain_solution_invalid"


@pytest.mark.parametrize("case", ["continuous", "integer_range", "integer_other", "replaced", "foreign"])
def test_fixed_binary_audit_refuses_changed_integrality_or_handle(snapshot, bonus_rules, case):
    from dao.prog.settlement.optimization_refinement import model_digest

    model, terms = financial_case(snapshot, bonus_rules)
    source = terms.bonus.source
    var = source.discharge_on[0]
    digest = model_digest(model)
    if case == "continuous":
        var.var_type = mip.CONTINUOUS
    elif case == "integer_range":
        var.lb, var.ub = 0, 2
    elif case == "integer_other":
        var.lb = var.ub = 2
    elif case == "replaced":
        source = replace(source, discharge_on=(mip.Var(model, var.idx),))
    else:
        other = mip.Model(solver_name=mip.CBC)
        source = replace(source, discharge_on=(other.add_var(var_type=mip.BINARY),))
    if case not in ("replaced", "foreign"):
        assert model_digest(model) != digest
    with pytest.raises(RuntimeError) as rejected:
        audit_api().validate_source_solution(replace(terms.bonus, source=source), evidence=evidence(model))
    assert rejected.value.code == "source_domain_solution_invalid"


def test_cap_normalization_budget_resets_and_unknown_stays_unknown(snapshot, bonus_rules, monkeypatch):
    """A 192-step cap ledger uses actual export normalization, retaining C at K changes."""
    from dao.prog.settlement.optimization import build_bonus_terms
    from dao.prog.settlement.tariffs import quote

    n = 192
    starts = tuple(ts("2027-12-31", 0) + 900 * i for i in range(n))
    rules = replace(bonus_rules, yearly_cap_kwh=D("1"),
                    windows=tuple((m, (0, 24)) for m in range(1, 13)))
    model, old = setup(snapshot, bonus_rules, cells=("0",) * n, exports=("0.01",) * n,
                       starts=starts, total="0.52", rules=rules)
    first_snap = replace(old.quotes[0].snapshot, valid_to_ts=starts[48])
    next_snap = replace(first_snap, snapshot_hash="c" * 64, valid_from_ts=starts[48],
                        valid_to_ts=None, bonus=replace(rules, yearly_cap_kwh=D("2")))
    prices = tuple(quote(first_snap if i < 48 else next_snap, start, start + 900, D("0.10"))
                   for i, start in enumerate(starts))
    bonus = build_bonus_terms(model, prices, old.grid_export, old.source.cell_discharge,
        old.export_bounds_kwh, old.opening, origins=old.origins, source_policy=old.source.policy_version)
    model, terms = financial_terms(model, bonus)
    values = {}
    begin = D("0.52")
    for i in range(n):
        if i == 96:
            begin = D("0")
        # The second year alternates direction: budgets sum absolute deltas, not net deltas.
        raw_export = D("0.00999999951") if i >= 96 and i % 2 else D("0.01000000049")
        end = begin + raw_export
        cap = D("1") if i < 48 else D("2")
        q_begin, q_end = min(begin, cap), min(end, cap)
        for var, value in ((bonus.grid_export[i], raw_export),
                           (bonus.cumulative_export_kwh[i], end),
                           (bonus.capped_opening_kwh[i], q_begin),
                           (bonus.capped_export_kwh[i], q_end),
                           (bonus.first_kwh[i], q_end - q_begin),
                           (bonus.eligible_kwh[i], q_end - q_begin),
                           (bonus.cap_saturated[i], int(end >= cap)),
                           (bonus.opening_cap_saturated[i], int(begin >= cap))):
            values[var.idx] = value
        begin = end
    override_solution(monkeypatch, model, values)
    result = run_audit(model, terms)
    assert result.parity_complete
    assert result.forecast_result.lines[47].bonus_kwh == D("0.01")
    assert abs(D(str(bonus.first_kwh[47].x)) - D("0.01")) > D("1e-9")
    assert result.forecast_result.lines[48].after_state.bonus.total_export_kwh == D("1.01")
    assert result.forecast_result.lines[95].after_state.bonus.total_export_kwh == 0
    assert result.forecast_result.lines[96].after_state.bonus.total_export_kwh == D("0.01")
    for i, count in ((48, 49), (96, 1), (191, 96)):
        diagnostic = next(issue for issue in result.issues
                          if issue.startswith(f"planning_export_normalization:interval={i}:"))
        assert f"cumulative_budget_kwh={Fraction(count * D('0.00000000049'))}" in diagnostic

    real_settle = audit_api().settle_series

    def leaked_budget(items, snapshots, opening):
        settled = real_settle(items, snapshots, opening)
        if items[0].sources != ("optimizer_forecast",):
            return settled
        lines = list(settled.lines)
        line = lines[96]
        after = replace(line.after_state, bonus=replace(line.after_state.bonus,
            total_export_kwh=line.after_state.bonus.total_export_kwh + D("0.000000002")))
        lines[96] = replace(line, after_state=after)
        return replace(settled, lines=tuple(lines))

    with monkeypatch.context() as engine:
        engine.setattr(audit_api(), "settle_series", leaked_budget)
        with pytest.raises(RuntimeError) as rejected:
            run_audit(model, terms)
        assert rejected.value.code == "planning_auxiliary_mismatch"
    unknown_model, unknown = financial_case(snapshot, bonus_rules, total=None)
    unknown_result = run_audit(unknown_model, unknown)
    assert unknown.bonus.cumulative_export_kwh == (None,)
    assert unknown_result.forecast_result.lines[0].bonus_kwh is None
    assert not unknown_result.parity_complete


@pytest.mark.parametrize("source_energy", ["0", "0.0015"])
@pytest.mark.parametrize("n", [1, 192])
def test_engine_quantity_budget_has_no_generic_epsilon(snapshot, bonus_rules, monkeypatch, source_energy, n):
    """Without export normalization even one input-resolution unit is not allowed."""
    model, terms = financial_case(snapshot, bonus_rules, cells=(source_energy,) * n, exports=("1",) * n)
    real_settle = audit_api().settle_series

    def extra_bonus(items, snapshots, opening):
        result = real_settle(items, snapshots, opening)
        if items[0].sources == ("optimizer_forecast",):
            lines = list(result.lines)
            lines[-1] = replace(lines[-1], bonus_kwh=lines[-1].bonus_kwh + D("1e-9"))
            result = replace(result, lines=tuple(lines))
        return result

    monkeypatch.setattr(audit_api(), "settle_series", extra_bonus)
    with pytest.raises(RuntimeError) as rejected:
        run_audit(model, terms)
    assert rejected.value.code == "planning_auxiliary_mismatch"


def test_zero_bonus_percentage_preserves_builder_quantity_semantics(snapshot, bonus_rules):
    """A known zero-rate engine quantity is not a positive optimizer incentive."""
    rules = replace(bonus_rules, percentage=D("0"),
                    windows=tuple((m, (0, 24)) for m in range(1, 13)))
    model, terms = financial_case(snapshot, bonus_rules, rules=rules)
    result = run_audit(model, terms)
    assert not terms.bonus.source.required[0]
    assert terms.bonus.eligible_kwh[0].x == 0
    assert result.forecast_result.lines[0].bonus_kwh == 1
    assert result.forecast_result.lines[0].bonus_revenue == 0
    assert result.parity_complete


@pytest.mark.parametrize("case", ["history", "origin", "partial"])
def test_unknown_engine_quantity_cannot_become_zero(snapshot, bonus_rules, monkeypatch, case):
    kwargs = {"total": None} if case == "history" else (
        {"origins": ("unknown",)} if case == "origin" else {"starts": (ts(minute=5),), "durations": (600,)})
    model, terms = financial_case(snapshot, bonus_rules, **kwargs)
    real_settle = audit_api().settle_series

    def fabricated_zero(items, snapshots, opening):
        result = real_settle(items, snapshots, opening)
        if items[0].sources == ("optimizer_forecast",):
            result = replace(result, lines=(replace(result.lines[0], bonus_kwh=D("0"), bonus_revenue=D("0")),))
        return result

    monkeypatch.setattr(audit_api(), "settle_series", fabricated_zero)
    with pytest.raises(RuntimeError) as rejected:
        run_audit(model, terms)
    assert rejected.value.code == "planning_auxiliary_mismatch"


@pytest.mark.parametrize("early", ["negative", "cell", "origin"])
def test_raw_audit_counts_even_bonus_excluded_export(snapshot, bonus_rules, early):
    """Excluded early export still exhausts the cap independently of later bonus prices."""
    model, terms = financial_case(snapshot, bonus_rules, exports=("1", "1"), total="4999",
        starts=(ts(minute=45), ts(hour=13)),
        cells=("0.0015" if early == "cell" else "0", "0"),
        spots=("-0.10" if early == "negative" else "0.10", "0.20"),
        origins=("unknown" if early == "origin" else "verified_pv", "verified_pv"))
    result = run_audit(model, terms)
    assert result.forecast_result.lines[-1].after_state.bonus.total_export_kwh == D("5001")
    assert result.forecast_result.lines[-1].bonus_kwh == 0
    assert result.forecast_result.lines[-1].bonus_revenue == 0
