"""Gecontroleerd boundrecept en lokale versiedigests; geen solve of I/O."""

import hashlib
import json
from dataclasses import dataclass, replace
from decimal import ROUND_FLOOR, Decimal
from fractions import Fraction
from typing import Literal

import mip

from dao.prog.settlement.optimization import (
    CellDischargeInput,
    OptimizationInputError,
    _validate_cell_binding,
    make_cell_discharge_input,
)
from dao.prog.settlement.optimization_audit import (
    FinancialModelTerms,
    PlanningSolutionRejected,
    SolverEvidence,
    _check_context,
    _check_evidence,
    _normalize_energy,
    _number,
    read_solver_evidence,
    validate_source_solution,
)

_INVALID = "source_refinement_invalid"
_MINIMUM = Fraction(1, 1000)
_SETTINGS = (
    "integer_tol", "infeas_tol", "opt_tol", "max_mip_gap", "max_mip_gap_abs",
    "max_seconds", "max_nodes", "max_solutions", "cutoff", "seed", "threads",
    "cuts", "cut_passes", "clique", "preprocess", "pump_passes", "sol_pool_size",
    "round_int_vars", "verbose", "store_search_progress_log",
)


@dataclass(frozen=True, slots=True)
class BoundFix:
    var: mip.Var
    var_index: int
    var_type: str
    original_value: Decimal
    lower_before: Decimal
    upper_before: Decimal
    fixed_value: Decimal
    reason: Literal["integer_choice", "source_zero", "active_cell_bound", "normalization_zero"]


@dataclass(frozen=True, slots=True)
class SourceRefinement:
    model: mip.Model
    original_cell: CellDischargeInput
    original_model_digest: str
    fixes: tuple[BoundFix, ...]


@dataclass(frozen=True, slots=True)
class SourceRefinementReport:
    policy_version: Literal["fixed_choice_refine_v2"]
    initial_evidence: SolverEvidence
    final_evidence: SolverEvidence
    recipe: SourceRefinement
    objective_delta: Decimal


@dataclass(frozen=True, slots=True)
class FinalizedPlanningSolution:
    terms: FinancialModelTerms
    evidence: SolverEvidence
    model_digest: str
    solution_digest: str
    refinement: SourceRefinementReport | None


def _text(value, code=_INVALID):
    number = _number(value, code)
    if number == 0:
        return "0"
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _handle(var, model):
    if (not isinstance(var, mip.Var) or var.model is not model or var.idx < 0
        or var.idx >= len(model.vars) or model.vars[var.idx] is not var):
        raise PlanningSolutionRejected(_INVALID)
    return var.idx


def _expression(expression, model):
    if not isinstance(expression, mip.LinExpr):
        raise PlanningSolutionRejected(_INVALID)
    return (_text(expression.const), expression.sense,
            sorted((_handle(var, model), _text(coefficient))
                   for var, coefficient in expression.expr.items()))


def _model_snapshot(model):
    """Alle zichtbare algebra/instellingen; SOS blijft native en ongewijzigd."""
    if model.cuts_generator is not None or model.lazy_constrs_generator is not None:
        # Een callback kan het exclusief beheerde lokale model alsnog wijzigen.
        raise PlanningSolutionRejected(_INVALID)
    variables = []
    for index, var in enumerate(model.vars):
        if _handle(var, model) != index or var.var_type not in (mip.BINARY, mip.INTEGER, mip.CONTINUOUS):
            raise PlanningSolutionRejected(_INVALID)
        lower, upper = _number(var.lb, _INVALID), _number(var.ub, _INVALID)
        if lower > upper:
            raise PlanningSolutionRejected(_INVALID)
        # CBC leest een gefixeerde BINARY als INTEGER: de integrality blijft gelijk.
        variables.append((index, var.var_type in (mip.BINARY, mip.INTEGER), _text(lower), _text(upper)))
    constraints = []
    for index, constraint in enumerate(model.constrs):
        if constraint.idx != index:
            raise PlanningSolutionRejected(_INVALID)
        constraints.append((index, _expression(constraint.expr, model)))
    settings = [(name, value if isinstance(value := getattr(model, name), bool) else _text(value))
                for name in _SETTINGS]
    settings.extend((("emphasis", model.emphasis.value), ("lp_method", model.lp_method.value)))
    start = None if model.start is None else sorted((_handle(var, model), _text(value))
                                                   for var, value in model.start)
    return {"variables": variables, "constraints": constraints,
            "objective": _expression(model.objective, model), "sense": model.sense,
            "solver": (mip.__version__, model.solver_name), "settings": settings, "start": start}


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def model_digest(model: mip.Model) -> str:
    """Canonieke lokale SHA-256, zonder objectrepr, Var-vergelijking of Python-hash."""
    return _digest(_model_snapshot(model))


def solution_digest(model: mip.Model) -> str:
    """Alle originele eindige .x-waarden, ook buiten objective en celstromen."""
    return _digest([(_handle(var, model), _text(var.x, "planning_solution_digest_invalid"))
                    for var in model.vars])


def _check_terms(terms, evidence):
    if not isinstance(terms, FinancialModelTerms):
        raise PlanningSolutionRejected(_INVALID)
    bonus, cell = terms.bonus, terms.bonus.source.cell_discharge
    model = cell.model
    _check_evidence(model, evidence)
    try:
        _validate_cell_binding(cell, model)
    except (OptimizationInputError, TypeError, ValueError) as exc:
        raise PlanningSolutionRejected(_INVALID) from exc
    n = len(terms.quotes)
    if (terms.quotes != bonus.quotes or len(terms.grid_import) != n
        or any(len(axis) != n for axis in (bonus.source.required, bonus.source.discharge_on,
            bonus.grid_export, bonus.eligible_kwh, bonus.first_kwh, bonus.cumulative_export_kwh,
            bonus.capped_export_kwh, bonus.cap_saturated, bonus.capped_opening_kwh,
            bonus.opening_cap_saturated))):
        raise PlanningSolutionRejected(_INVALID)
    _check_context(terms.context, terms.quotes, bonus.opening, cell.meter_id)
    used = {var.idx for row in cell.bus_power_vars for var in row}
    for var in (*terms.grid_import, *bonus.grid_export):
        if _handle(var, model) in used:
            raise PlanningSolutionRejected(_INVALID)
        used.add(var.idx)
    for axis in (bonus.eligible_kwh, bonus.first_kwh, bonus.cumulative_export_kwh,
                 bonus.capped_export_kwh, bonus.cap_saturated, bonus.capped_opening_kwh,
                 bonus.opening_cap_saturated, bonus.source.discharge_on):
        for var in axis:
            if var is not None:
                if _handle(var, model) in used:
                    raise PlanningSolutionRejected(_INVALID)
                used.add(var.idx)
    _expression(terms.energy_cost, model)
    _expression(bonus.revenue, model)
    return model, cell


def _integer_choice(value, lower, upper, tolerance):
    floor = int(value.to_integral_value(rounding=ROUND_FLOOR))
    candidates = [Decimal(integer) for integer in (floor, floor + 1)
                  if lower <= integer <= upper and abs(Fraction(value) - integer) <= Fraction(tolerance)]
    if len(candidates) != 1:
        raise PlanningSolutionRejected(_INVALID)
    return candidates[0]


def _add_fix(fixes, var, value, target, reason):
    lower, upper = _number(var.lb, _INVALID), _number(var.ub, _INVALID)
    if not target.is_finite() or not lower <= target <= upper:
        raise PlanningSolutionRejected(_INVALID)
    if var.idx in fixes:
        if fixes[var.idx].fixed_value != target:
            raise PlanningSolutionRejected(_INVALID)
        return
    fixes[var.idx] = BoundFix(var, var.idx, var.var_type, value, lower, upper, target, reason)


def _physical_valid(cell, values, statuses, *, normalized=False):
    for i, on in enumerate(statuses):
        if any(not cell.bus_lower_bounds_kw[b][i] <= row[i] <= cell.bus_upper_bounds_kw[b][i]
               for b, row in enumerate(values)):
            return False
        total = sum((Fraction(row[i]) * cell.cell_coefficients[b][i]
                     for b, row in enumerate(values)), Fraction(0))
        if (total > cell.upper_bounds_kwh[i] or (on == 0 and any(row[i] != 0 for row in values))
            or (on == 1 and total < _MINIMUM)):
            return False
        if normalized:
            engine = _normalize_energy(total)
            if ((total > 0 and engine == 0) or Fraction(engine) > cell.upper_bounds_kwh[i]
                or (on == 1 and engine < Decimal("0.001"))):
                return False
    return True


def prepare_source_refinement(terms: FinancialModelTerms, *, evidence: SolverEvidence) -> SourceRefinement:
    """Bereken uitsluitend een volledig controleerbaar recept, zonder modelmutatie."""
    model, cell = _check_terms(terms, evidence)
    digest = model_digest(model)
    values = {var.idx: _number(var.x, _INVALID) for var in model.vars}
    fixes = {}
    for var in model.vars:
        if var.var_type in (mip.BINARY, mip.INTEGER):
            target = _integer_choice(values[var.idx], _number(var.lb, _INVALID),
                                     _number(var.ub, _INVALID), evidence.integer_tol)
            _add_fix(fixes, var, values[var.idx], target, "integer_choice")
    statuses = []
    for i, required in enumerate(terms.bonus.source.required):
        z = terms.bonus.source.discharge_on[i]
        if type(required) is not bool or (required and terms.bonus.source.policy_version != "aggregate_cell_min_1wh_v1"):
            raise PlanningSolutionRejected(_INVALID)
        on = None
        if required:
            if z is None:
                if cell.upper_bounds_kwh[i] >= _MINIMUM:
                    raise PlanningSolutionRejected(_INVALID)
                on = 0
            else:
                if (z.var_type != mip.BINARY and not (z.var_type == mip.INTEGER
                    and z.lb == z.ub and z.lb in (0, 1))):
                    raise PlanningSolutionRejected(_INVALID)
                on = int(fixes[z.idx].fixed_value)
        elif z is not None:
            raise PlanningSolutionRejected(_INVALID)
        statuses.append(on)
    raw = tuple(tuple(values[var.idx] for var in row) for row in cell.bus_power_vars)
    raw_valid = _physical_valid(cell, raw, statuses)
    normalization_zero = set()
    if raw_valid:
        for i, on in enumerate(statuses):
            total = sum((Fraction(row[i]) * cell.cell_coefficients[b][i]
                         for b, row in enumerate(raw)), Fraction(0))
            engine = _normalize_energy(total)
            if ((total > 0 and engine == 0) or Fraction(engine) > cell.upper_bounds_kwh[i]
                or (on == 1 and engine < Decimal("0.001"))):
                if on is not None or not total > 0 or engine != 0:
                    raise PlanningSolutionRejected(_INVALID)
                normalization_zero.add(i)
        if not normalization_zero:
            # Een geldige bron is geen aanleiding voor een optimaliserende extra solve.
            raise PlanningSolutionRejected(_INVALID)
    for i, on in enumerate(statuses):
        for row in cell.bus_power_vars:
            var = row[i]
            value = values[var.idx]
            if on == 0:
                if abs(Fraction(value)) > Fraction(evidence.infeas_tol):
                    raise PlanningSolutionRejected(_INVALID)
                _add_fix(fixes, var, value, Decimal(0), "source_zero")
            else:
                endpoints = {bound for bound in (_number(var.lb, _INVALID), _number(var.ub, _INVALID))
                             if abs(Fraction(value) - Fraction(bound)) <= Fraction(evidence.infeas_tol)}
                if len(endpoints) > 1:
                    raise PlanningSolutionRejected(_INVALID)
                if i in normalization_zero:
                    if endpoints != {Decimal(0)}:
                        raise PlanningSolutionRejected(_INVALID)
                    _add_fix(fixes, var, value, Decimal(0), "normalization_zero")
                elif endpoints:
                    _add_fix(fixes, var, value, endpoints.pop(), "active_cell_bound")
    try:
        validate_source_solution(terms.bonus, evidence=evidence)
    except PlanningSolutionRejected as exc:
        expected = "source_domain_normalization_invalid" if normalization_zero else "source_domain_solution_invalid"
        if exc.code != expected:
            raise
    else:
        # De bestaande audit toetst positieve cel-lb pas in N2; alleen dat gat verklaart succes.
        if not any(value < cell.bus_lower_bounds_kw[b][i]
                   for b, row in enumerate(raw) for i, value in enumerate(row)):
            raise PlanningSolutionRejected(_INVALID)
    projected = tuple(tuple(fixes[var.idx].fixed_value if var.idx in fixes else values[var.idx]
                            for var in row) for row in cell.bus_power_vars)
    if not _physical_valid(cell, projected, statuses, normalized=True):
        # Geen zelfberekend L-punt en geen integerfix als generieke broncorrectie.
        raise PlanningSolutionRejected(_INVALID)
    try:
        model.check_optimization_results()
    except (AssertionError, mip.InfeasibleSolution) as exc:
        raise PlanningSolutionRejected(_INVALID) from exc
    if model_digest(model) != digest:
        raise PlanningSolutionRejected(_INVALID)
    return SourceRefinement(model, cell, digest, tuple(fixes[index] for index in sorted(fixes)))


def _same_fix(actual, expected):
    if (not isinstance(actual, BoundFix) or actual.var is not expected.var
        or type(actual.var_index) is not int
        or any(not isinstance(value := getattr(actual, name), Decimal) or not value.is_finite()
               for name in ("original_value", "lower_before", "upper_before", "fixed_value"))):
        return False
    return all(getattr(actual, name) == getattr(expected, name) for name in (
        "var_index", "var_type", "original_value", "lower_before", "upper_before", "fixed_value", "reason"))


def apply_source_refinement(terms: FinancialModelTerms, *, recipe: SourceRefinement) -> FinancialModelTerms:
    """Valideer alles vóór mutatie; vernauw alleen bounds en herbind alleen de cel-DTO."""
    cell = terms.bonus.source.cell_discharge
    model = cell.model
    if (not isinstance(recipe, SourceRefinement) or recipe.model is not model
        or recipe.original_cell is not cell or recipe.original_model_digest != model_digest(model)):
        raise PlanningSolutionRejected(_INVALID)
    expected = prepare_source_refinement(terms, evidence=read_solver_evidence(model, cbc_log=""))
    if (not isinstance(recipe.fixes, tuple) or len(recipe.fixes) != len(expected.fixes)
        or any(not _same_fix(actual, checked) for actual, checked in zip(recipe.fixes, expected.fixes, strict=True))):
        raise PlanningSolutionRejected(_INVALID)
    before = _model_snapshot(model)
    variables, constraints = tuple(model.vars), tuple(model.constrs)
    targets = {fix.var_index: _text(fix.fixed_value) for fix in recipe.fixes}
    before["variables"] = [(index, integral, targets.get(index, lower), targets.get(index, upper))
                           for index, integral, lower, upper in before["variables"]]
    expected_digest = _digest(before)
    for fix in recipe.fixes:
        fix.var.lb = float(fix.fixed_value)
        fix.var.ub = float(fix.fixed_value)
    if (model_digest(model) != expected_digest
        or len(variables) != len(model.vars) or len(constraints) != len(model.constrs)
        or any(old is not new for old, new in zip(variables, model.vars, strict=True))
        or any(old is not new for old, new in zip(constraints, model.constrs, strict=True))):
        # Niet herstellen met de oude oplossing: de eigenaar moet dit model weggooien.
        raise PlanningSolutionRejected(_INVALID)
    rebound = make_cell_discharge_input(model, cell.bus_power_vars, meter_id=cell.meter_id,
        installation_battery_count=cell.installation_battery_count, battery_indices=cell.battery_indices,
        no_battery_evidence=cell.no_battery_evidence, hour_fractions=cell.hour_fractions,
        discharge_efficiencies=cell.discharge_efficiencies)
    return replace(terms, bonus=replace(terms.bonus, source=replace(terms.bonus.source, cell_discharge=rebound)))
