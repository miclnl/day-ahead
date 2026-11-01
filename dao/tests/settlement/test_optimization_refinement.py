"""Bound refinement selects a new restricted problem, never accepts cell residuals."""

import math
from dataclasses import FrozenInstanceError, fields, replace
from decimal import Decimal

import mip
import pytest

from dao.prog.settlement.optimization import _validate_cell_binding
from dao.prog.settlement.optimization_audit import PlanningSolutionRejected
from dao.tests.settlement.test_optimization_audit import (
    evidence,
    financial_case,
    override_solution,
)

D = Decimal


def api():
    from dao.prog.settlement import optimization_refinement
    return optimization_refinement


def visible_structure(model):
    """Independently retain handles and all non-bound algebra/settings."""
    def expression(expr):
        return expr.const, expr.sense, tuple(sorted((v.idx, c) for v, c in expr.expr.items()))

    return (
        tuple(id(v) for v in model.vars), tuple(id(c) for c in model.constrs),
        tuple(expression(c.expr) for c in model.constrs), expression(model.objective),
        model.sense, tuple(v.var_type != mip.CONTINUOUS for v in model.vars),
        model.integer_tol, model.infeas_tol, model.opt_tol, model.max_seconds,
        model.max_nodes, model.max_solutions, model.max_mip_gap, model.max_mip_gap_abs,
    )


def prepare_noisy_zero(snapshot, bonus_rules, monkeypatch, **kwargs):
    model, terms = financial_case(snapshot, bonus_rules, **kwargs)
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    override_solution(monkeypatch, model, {cell.idx: 1e-13})
    proof = evidence(model)
    return model, terms, api().prepare_source_refinement(terms, evidence=proof)


@pytest.mark.parametrize("case,value,target,reason", [
    ("zero", 1e-13, "0", "source_zero"),
    ("zero", -1e-13, "0", "source_zero"),
    ("upper", 0.004000000000000001, "0.004", "active_cell_bound"),
    ("lower", 0.003999999999999999, "0.004", "active_cell_bound"),
    ("zero_only", 1e-13, "0", "source_zero"),
    ("fixed", 1e-13, "0", "active_cell_bound"),
])
def test_refinement_only_tightens_authorized_bounds(snapshot, bonus_rules, monkeypatch,
                                                   case, value, target, reason):
    """Only recorded integer choices and original cell endpoints may be fixed."""
    kwargs = {"cells": ("0.001",), "bounds": (D("0.001"),)} if case == "upper" else {}
    if case == "zero_only":
        kwargs = {"bounds": (D("0.0008"),)}
    if case == "fixed":
        kwargs = {"bounds": (D("0"),), "spots": ("-0.10",)}
    if case == "lower":
        kwargs = {"cells": ("0.001",)}
    model, terms = financial_case(snapshot, bonus_rules, **kwargs)
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    if case == "lower":
        cell.lb = 0.004
        from dao.prog.settlement.optimization import make_cell_discharge_input
        old = terms.bonus.source.cell_discharge
        rebound = make_cell_discharge_input(model, old.bus_power_vars, meter_id=old.meter_id,
            installation_battery_count=old.installation_battery_count,
            battery_indices=old.battery_indices, no_battery_evidence=old.no_battery_evidence,
            hour_fractions=old.hour_fractions, discharge_efficiencies=old.discharge_efficiencies)
        terms = replace(terms, bonus=replace(terms.bonus, source=replace(terms.bonus.source,
                                                                     cell_discharge=rebound)))
        assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    proof = evidence(model)
    override_solution(monkeypatch, model, {cell.idx: value})
    before = visible_structure(model)
    bounds = tuple((v.lb, v.ub) for v in model.vars)
    digest = api().model_digest(model)
    recipe = api().prepare_source_refinement(terms, evidence=proof)
    assert api().model_digest(model) == digest
    assert tuple((v.lb, v.ub) for v in model.vars) == bounds
    fixes = {fix.var_index: fix for fix in recipe.fixes}
    assert fixes[cell.idx].var is cell
    assert fixes[cell.idx].original_value == D(str(value))
    assert fixes[cell.idx].fixed_value == D(target)
    assert fixes[cell.idx].reason == reason
    assert {v.idx for v in model.vars if v.var_type in (mip.BINARY, mip.INTEGER)} <= fixes.keys()
    rebound = api().apply_source_refinement(terms, recipe=recipe)
    assert visible_structure(model) == before
    for v, (lower, upper) in zip(model.vars, bounds, strict=True):
        assert lower <= v.lb <= v.ub <= upper
        if v.idx in fixes:
            assert D(str(v.lb)) == D(str(v.ub)) == fixes[v.idx].fixed_value
        else:
            assert (v.lb, v.ub) == (lower, upper)
    assert rebound.bonus.source.cell_discharge.model is model


def test_refinement_rebinds_without_rebuilding_bonus(snapshot, bonus_rules, monkeypatch):
    """Old cell bindings fail; every enclosing field except that cell stays intact."""
    model, terms, recipe = prepare_noisy_zero(snapshot, bonus_rules, monkeypatch)
    original = terms.bonus.source.cell_discharge

    def forbidden_solve(*args, **kwargs):
        pytest.fail("Preparing/applying a recipe must not solve the model")

    monkeypatch.setattr(model, "optimize", forbidden_solve)
    assert api().model_digest(model) == recipe.original_model_digest
    api().prepare_source_refinement(terms, evidence=evidence(model))
    rebound = api().apply_source_refinement(terms, recipe=recipe)
    with pytest.raises(ValueError, match="cell_conversion_invalid"):
        _validate_cell_binding(original, model)
    current = rebound.bonus.source.cell_discharge
    _validate_cell_binding(current, model)
    assert recipe.original_cell is original
    assert current.bus_power_vars[0][0] is original.bus_power_vars[0][0]
    assert current.hour_fractions == original.hour_fractions
    assert current.discharge_efficiencies == original.discharge_efficiencies
    assert current.upper_bounds_kwh == (0,)
    assert rebound.bonus.source.required == (True,)
    assert rebound.bonus.source.discharge_on[0] is terms.bonus.source.discharge_on[0]
    assert rebound.bonus.source.discharge_on[0].var_type == mip.INTEGER
    for old, new, changed in ((terms, rebound, "bonus"), (terms.bonus, rebound.bonus, "source"),
                             (terms.bonus.source, rebound.bonus.source, "cell_discharge")):
        for field in fields(old):
            if field.name != changed:
                assert getattr(new, field.name) is getattr(old, field.name)
    with pytest.raises(FrozenInstanceError):
        recipe.original_model_digest = "changed"
    with pytest.raises(FrozenInstanceError):
        recipe.fixes[0].fixed_value = D("1")


@pytest.mark.parametrize("field,value", [
    ("fixed_value", D("NaN")), ("fixed_value", D("sNaN")), ("fixed_value", D("Infinity")),
    ("original_value", D("sNaN")), ("lower_before", D("NaN")), ("upper_before", D("Infinity")),
    ("var_index", True), ("lower_before", False), ("fixed_value", 0),
])
def test_recipe_rejects_nonfinite_or_untyped_recorded_values(snapshot, bonus_rules, monkeypatch, field, value):
    """Frozen annotations cannot validate forged records before native bound writes."""
    model, terms, recipe = prepare_noisy_zero(snapshot, bonus_rules, monkeypatch)
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    fixes = tuple(replace(f, **{field: value}) if f.var is cell else f for f in recipe.fixes)
    before = tuple((v.lb, v.ub) for v in model.vars)
    with pytest.raises(PlanningSolutionRejected):
        api().apply_source_refinement(terms, recipe=replace(recipe, fixes=fixes))
    assert tuple((v.lb, v.ub) for v in model.vars) == before


def test_apply_detects_unregistered_mutation_during_native_bound_writes(snapshot, bonus_rules, monkeypatch):
    """Post-write verification must reject a hidden change outside the whitelist."""
    model, terms, recipe = prepare_noisy_zero(snapshot, bonus_rules, monkeypatch)
    setter = model.solver.var_set_lb

    def mutate_other_bound(var, value):
        setter(var, value)
        terms.grid_import[0].ub = 3

    monkeypatch.setattr(model.solver, "var_set_lb", mutate_other_bound)
    with pytest.raises(PlanningSolutionRejected):
        api().apply_source_refinement(terms, recipe=recipe)


def test_refinement_includes_near_unviolated_cell_endpoints_and_all_integers(
        snapshot, bonus_rules, monkeypatch):
    """A one-sided repair omits still-near endpoints and leaves CBC residues."""
    model, terms = financial_case(snapshot, bonus_rules, cells=(("0",), ("0",)))
    integer = model.add_var(var_type=mip.INTEGER, lb=2, ub=5)
    model += integer == 3
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    first, second = (row[0] for row in terms.bonus.source.cell_discharge.bus_power_vars)
    override_solution(monkeypatch, model, {first.idx: 1e-13})
    recipe = api().prepare_source_refinement(terms, evidence=evidence(model))
    fixes = {fix.var_index: fix for fix in recipe.fixes}
    assert fixes[second.idx].fixed_value == 0
    assert fixes[integer.idx].fixed_value == 3
    assert fixes[integer.idx].reason == "integer_choice"


def test_refinement_includes_near_but_not_overshot_active_upper_bound(snapshot, bonus_rules, monkeypatch):
    """Whitelisting only violated bounds leaves a new residual in another cell."""
    model, terms = financial_case(snapshot, bonus_rules, cells=("0", "0.001"),
        exports=("1", "1"), bounds=(D("0.001"), D("0.001")))
    first, second = terms.bonus.source.cell_discharge.bus_power_vars[0]
    override_solution(monkeypatch, model, {first.idx: 1e-13, second.idx: math.nextafter(0.004, 0)})
    recipe = api().prepare_source_refinement(terms, evidence=evidence(model))
    fix = next(f for f in recipe.fixes if f.var is second)
    assert fix.fixed_value == D("0.004")
    assert fix.reason == "active_cell_bound"


@pytest.mark.parametrize("case", ["far_zero", "positive_lower_zero", "no_endpoint", "valid",
                                  "normalization_only", "auxiliary_only", "context", "stale_evidence"])
def test_refinement_refuses_unrepairable_or_non_source_failures(snapshot, bonus_rules, monkeypatch, case):
    """Neither arbitrary domain gaps nor unrelated audit failures authorize a recipe."""
    kwargs = {"cells": ("0.0015",)} if case == "no_endpoint" else {}
    if case == "normalization_only":
        kwargs = {"spots": ("-0.10",)}
    model, terms = financial_case(snapshot, bonus_rules, **kwargs)
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    proof = evidence(model)
    if case == "positive_lower_zero":
        cell.lb = 1e-14
        # A fresh binding makes the positive lower bound real, not a stale-DTO error.
        old = terms.bonus.source.cell_discharge
        terms = replace(terms, bonus=replace(terms.bonus, source=replace(terms.bonus.source,
            cell_discharge=replace(old, bus_lower_bounds_kw=((D("1e-14"),),)))))
    values = {cell.idx: 1e-13}
    if case == "far_zero":
        values[cell.idx] = 0.001
    elif case == "no_endpoint":
        values[cell.idx] = 0.0039999999999
    elif case == "valid":
        values = {}
    elif case == "auxiliary_only":
        values = {terms.bonus.eligible_kwh[0].idx: 0.0}
    elif case == "context":
        terms = replace(terms, context=replace(terms.context,
            basis=replace(terms.context.basis, meter_id="other-meter")))
    elif case == "stale_evidence":
        proof = replace(proof, integer_tol=D("0.01"))
    override_solution(monkeypatch, model, values)
    bounds = tuple((v.lb, v.ub) for v in model.vars)
    with pytest.raises(PlanningSolutionRejected):
        api().prepare_source_refinement(terms, evidence=proof)
    assert tuple((v.lb, v.ub) for v in model.vars) == bounds


@pytest.mark.parametrize("case", ["integer_ambiguous", "two_endpoints", "nonfinite_cell",
                                  "nonfinite_other", "nonfinite_integer", "incomplete", "conflict",
                                  "widen", "other_continuous", "wrong_reason", "foreign_handle"])
def test_refinement_rejects_ambiguous_or_nonfinite_choices(snapshot, bonus_rules, monkeypatch, case):
    """Recipes are rederived before any mutation, including all explicit integers."""
    kwargs = {"bounds": (D("0.000000004"),), "spots": ("-0.10",)} if case == "two_endpoints" else {}
    model, terms = financial_case(snapshot, bonus_rules, **kwargs)
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    z = terms.bonus.source.discharge_on[0]
    values = {cell.idx: 1e-13}
    if case == "integer_ambiguous":
        values[z.idx] = 0.5
    elif case == "two_endpoints":
        # Nonrequired intervals have no mandated zero; two distinct nearby bounds are ambiguous.
        values[cell.idx] = -1e-13
    elif case == "nonfinite_cell":
        values[cell.idx] = math.nan
    elif case == "nonfinite_other":
        values[terms.grid_import[0].idx] = math.inf
    elif case == "nonfinite_integer":
        values[z.idx] = -math.inf
    override_solution(monkeypatch, model, values)
    proof = evidence(model)
    before = tuple((v.lb, v.ub) for v in model.vars)
    with pytest.raises(PlanningSolutionRejected):
        if case in {"integer_ambiguous", "two_endpoints", "nonfinite_cell", "nonfinite_other", "nonfinite_integer"}:
            api().prepare_source_refinement(terms, evidence=proof)
        else:
            recipe = api().prepare_source_refinement(terms, evidence=proof)
            fix = next(f for f in recipe.fixes if f.var is cell)
            if case == "incomplete":
                fixes = tuple(f for f in recipe.fixes if f.var is not z)
            elif case == "conflict":
                fixes = (*recipe.fixes, replace(fix, fixed_value=D("0.004")))
            elif case == "widen":
                fixes = tuple(replace(f, fixed_value=D("-0.001")) if f.var is cell else f for f in recipe.fixes)
            elif case == "other_continuous":
                v = terms.grid_import[0]
                fixes = (*recipe.fixes, replace(fix, var=v, var_index=v.idx,
                    original_value=D(str(v.x)), lower_before=D(str(v.lb)), upper_before=D(str(v.ub))))
            elif case == "wrong_reason":
                fixes = tuple(replace(f, reason="active_cell_bound") if f.var is cell else f for f in recipe.fixes)
            else:
                other = mip.Model(solver_name=mip.CBC)
                foreign = other.add_var(ub=cell.ub)
                fixes = tuple(replace(f, var=foreign) if f.var is cell else f for f in recipe.fixes)
            api().apply_source_refinement(terms, recipe=replace(recipe, fixes=fixes))
    assert tuple((v.lb, v.ub) for v in model.vars) == before


@pytest.mark.parametrize("mutation", ["cell_lower", "other_upper", "objective", "sense", "constraint",
                                      "coefficient", "new_var", "new_constraint", "integrality",
                                      "integer_tol", "max_nodes", "raw_value"])
def test_apply_rejects_intervening_mutations(snapshot, bonus_rules, monkeypatch, mutation):
    """A recipe cannot authorize changes made after its original binding was checked."""
    model, terms, recipe = prepare_noisy_zero(snapshot, bonus_rules, monkeypatch)
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    if mutation == "cell_lower":
        cell.lb = 1e-14
    elif mutation == "other_upper":
        terms.grid_import[0].ub = 3
    elif mutation == "objective":
        model.objective = model.objective + 1
    elif mutation == "sense":
        model.sense = mip.MAXIMIZE
    elif mutation == "constraint":
        model.constrs[0].rhs += 0.001
    elif mutation == "coefficient":
        # CBC's constr_set_expr is a no-op; replace the last row at the same idx.
        model.remove(model.constrs[-1])
        model += 2 * terms.grid_import[0] == 0
        assert model.constrs[-1].expr.expr[terms.grid_import[0]] == 2
    elif mutation == "new_var":
        model.add_var(ub=1)
    elif mutation == "new_constraint":
        model += terms.grid_import[0] <= 1
    elif mutation == "integrality":
        terms.bonus.source.discharge_on[0].var_type = mip.CONTINUOUS
    elif mutation == "integer_tol":
        model.integer_tol = 1e-5
    elif mutation == "max_nodes":
        model.max_nodes = 17
    else:
        override_solution(monkeypatch, model, {cell.idx: 2e-13})
    before = tuple((v.lb, v.ub) for v in model.vars)
    with pytest.raises(PlanningSolutionRejected):
        api().apply_source_refinement(terms, recipe=recipe)
    assert tuple((v.lb, v.ub) for v in model.vars) == before


@pytest.mark.parametrize("changed", ["cell", "import", "auxiliary"])
def test_digests_distinguish_model_and_raw_solution_versions(snapshot, bonus_rules, monkeypatch, changed):
    """Unchanged objective/status cannot hide new raw values or coefficients."""
    model, terms = financial_case(snapshot, bonus_rules)
    structure, solution = api().model_digest(model), api().solution_digest(model)
    assert len(structure) == len(solution) == 64
    assert api().model_digest(model) == structure
    assert api().solution_digest(model) == solution
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    var = {"cell": cell, "import": terms.grid_import[0], "auxiliary": terms.bonus.first_kwh[0]}[changed]
    override_solution(monkeypatch, model, {var.idx: var.x + 1e-8})
    assert api().model_digest(model) == structure
    assert api().solution_digest(model) != solution
    model.objective = model.objective + cell * 0.123
    assert api().model_digest(model) != structure
    override_solution(monkeypatch, model, {cell.idx: math.nan})
    with pytest.raises(PlanningSolutionRejected):
        api().solution_digest(model)


def test_model_digest_is_independent_of_object_identity_and_expression_order():
    """Equivalent models must not depend on objectrepr or process-local hashes."""
    models = [mip.Model(solver_name=mip.CBC), mip.Model(solver_name=mip.CBC)]
    for i, model in enumerate(models):
        model.verbose = 0
        x, y = model.add_var(ub=3), model.add_var(var_type=mip.INTEGER, ub=2)
        model += (x + 2 * y if i == 0 else 2 * y + x) <= 4
        model.objective = 3 * x + y + 5 if i == 0 else 5 + y + 3 * x
    assert api().model_digest(models[0]) == api().model_digest(models[1])


def test_model_digest_uses_canonical_integrality_not_bound_dependent_binary_label():
    """Native B-to-I readback must not masquerade as a changed integral domain."""
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    var = model.add_var(var_type=mip.BINARY)
    before = api().model_digest(model)
    var.lb = var.ub = 0
    fixed = api().model_digest(model)
    assert var.var_type == mip.INTEGER
    assert fixed != before
    var.var_type = mip.CONTINUOUS
    assert api().model_digest(model) != fixed
