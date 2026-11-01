"""Onafhankelijke kandidaatcontrole; solverfeasibility is geen bronbewijs."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Literal
from zoneinfo import ZoneInfo

import mip
import mip.cbc as cbc

from dao.prog.settlement.engine import _bonus_period, settle_series
from dao.prog.settlement.metering import resolve_spot
from dao.prog.settlement.optimization import (
    BonusModelTerms,
    OptimizationInputError,
    _validate_cell_binding,
)
from dao.prog.settlement.service import PlanningContext
from dao.prog.settlement.tariffs import bonus_in_window, quote, resolve_snapshot
from dao.prog.settlement.types import InputInterval, PriceQuote, SettlementResult

_AUXILIARY_CODE = "planning_auxiliary_mismatch"
_AUXILIARY_BUDGET = Fraction(Decimal("0.000000001"))


class PlanningSolutionRejected(RuntimeError):
    def __init__(self, code: str, *, interval_index: int | None = None) -> None:
        self.code = code
        self.interval_index = interval_index
        super().__init__(code if interval_index is None else f"{code}: interval {interval_index}")


@dataclass(frozen=True, slots=True)
class SolverEvidence:
    python_mip_version: str
    solver_name: str
    cbc_version: str | None
    integer_tol: Decimal
    infeas_tol: Decimal
    opt_tol: Decimal
    cbc_primal_tol: Decimal | None
    cbc_dual_tol: Decimal | None
    integer_binding: Literal["python_mip_cbc_setDblParam"]
    max_mip_gap_abs: Decimal
    max_nodes: int
    status: str
    num_solutions: int
    objective_value: Decimal
    objective_bound: Decimal | None
    gap: Decimal | None
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceAudit:
    bus_values_kw: tuple[tuple[Decimal, ...], ...]
    raw_cell_kwh: tuple[Fraction, ...]
    engine_cell_kwh: tuple[Decimal, ...]
    discharge_on: tuple[int | None, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinancialModelTerms:
    context: PlanningContext
    quotes: tuple[PriceQuote, ...]
    grid_import: tuple[mip.Var, ...]
    bonus: BonusModelTerms
    energy_cost: mip.LinExpr


@dataclass(frozen=True, slots=True)
class PlanningAuditResult:
    source: SourceAudit
    evidence: SolverEvidence
    forecast: tuple[InputInterval, ...]
    baseline: tuple[InputInterval, ...]
    forecast_result: SettlementResult
    baseline_result: SettlementResult
    model_energy_cost_eur: Decimal
    engine_energy_cost_eur: Decimal | None
    baseline_energy_cost_eur: Decimal | None
    savings_energy_eur: Decimal | None
    parity_error_eur: Decimal | None
    parity_complete: bool
    issues: tuple[str, ...]


def _number(value, code, index=None):
    try:
        if value is None or isinstance(value, bool):
            raise ValueError("Missing numeric value")
        result = Decimal(str(value))
        if not result.is_finite():
            raise ValueError("Nonfinite numeric value")
        return result
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise PlanningSolutionRejected(code, interval_index=index) from exc


def read_solver_evidence(model: mip.Model, *, cbc_log: str) -> SolverEvidence:
    """Lees het geïnspecteerde adapterpad; claim geen native integer-readback."""
    unavailable = "solver_evidence_unavailable"
    if (model.solver_name != mip.CBC or type(model.solver) is not cbc.SolverCbc
        or mip.__version__ != "1.17.6" or not isinstance(cbc_log, str)):
        raise PlanningSolutionRejected(unavailable)
    integer, primal, dual = (_number(value, unavailable) for value in
                             (model.integer_tol, model.infeas_tol, model.opt_tol))
    if not 0 <= integer < Decimal("0.5") or primal < 0 or dual < 0:
        raise PlanningSolutionRejected(unavailable)
    try:
        native_primal = _number(cbc.cbclib.Cbc_getPrimalTolerance(model.solver._model), unavailable)
        native_dual = _number(cbc.cbclib.Cbc_getDualTolerance(model.solver._model), unavailable)
    except (AttributeError, TypeError) as exc:
        raise PlanningSolutionRejected(unavailable) from exc
    if native_primal != primal or native_dual != dual:
        raise PlanningSolutionRejected(unavailable)
    if (model.status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE)
        or model.num_solutions < 1):
        raise PlanningSolutionRejected("solver_solution_unavailable")
    objective = _number(model.objective_value, "solver_solution_unavailable")
    limitations = ["cbc_integer_native_readback_unavailable"]
    optional = []
    for name in ("objective_bound", "gap"):
        try:
            optional.append(_number(getattr(model, name), unavailable))
        except PlanningSolutionRejected:
            optional.append(None)
            limitations.append(f"solver_{name}_unavailable")
    if model.status == mip.OptimizationStatus.FEASIBLE:
        limitations.append("solver_optimality_unproven")
    version = re.search(r"Welcome to the CBC MILP Solver.*?\bVersion:\s*([0-9][\w.+-]*)", cbc_log, re.S)
    if version is None:
        limitations.append("cbc_version_unavailable")
    gap_limit = _number(model.max_mip_gap_abs, unavailable)
    if gap_limit < 0 or type(model.max_nodes) is not int or model.max_nodes < 0:
        raise PlanningSolutionRejected(unavailable)
    return SolverEvidence(mip.__version__, model.solver_name, None if version is None else version[1],
        integer, primal, dual, native_primal, native_dual, "python_mip_cbc_setDblParam",
        gap_limit, model.max_nodes, model.status.name, model.num_solutions, objective,
        optional[0], optional[1], tuple(limitations))


def _check_evidence(model, evidence):
    current = read_solver_evidence(model, cbc_log="")
    if not isinstance(evidence, SolverEvidence):
        raise PlanningSolutionRejected("solver_evidence_unavailable")
    for name in SolverEvidence.__dataclass_fields__:
        if name not in ("cbc_version", "limitations") and getattr(current, name) != getattr(evidence, name):
            raise PlanningSolutionRejected("solver_evidence_unavailable")


def _normalize_energy(value: Fraction) -> Decimal:
    """Half-up op de invoergrens, met rationale som vóór enige afronding."""
    units, remainder = divmod(value.numerator * 10**9, value.denominator)
    if 2 * remainder >= value.denominator:
        units += 1
    return Decimal(f"{units}e-9")


def _handle(var, model, code, index=None):
    if (not isinstance(var, mip.Var) or var.model is not model or var.idx < 0
        or var.idx >= len(model.vars) or model.vars[var.idx] is not var):
        raise PlanningSolutionRejected(code, interval_index=index)
    return var.idx


def _binary_choice(var, model, evidence, code, index):
    _handle(var, model, code, index)
    lower, upper = _number(var.lb, code, index), _number(var.ub, code, index)
    if not ((var.var_type == mip.BINARY and lower == 0 and upper == 1)
            or (var.var_type == mip.INTEGER and lower == upper and lower in (0, 1))):
        raise PlanningSolutionRejected(code, interval_index=index)
    value = Fraction(_number(var.x, code, index))
    choices = [candidate for candidate in (0, 1) if lower <= candidate <= upper
               and abs(value - candidate) <= Fraction(evidence.integer_tol)]
    if len(choices) != 1:
        raise PlanningSolutionRejected(code, interval_index=index)
    return choices[0]


def _validate_physical_source(terms, evidence):
    model = terms.source.cell_discharge.model
    _check_evidence(model, evidence)
    code = "source_domain_solution_invalid"
    cell = terms.source.cell_discharge
    try:
        _validate_cell_binding(cell, model)
    except (OptimizationInputError, TypeError, ValueError) as exc:
        raise PlanningSolutionRejected(code) from exc
    n = len(terms.quotes)
    if any(len(values) != n for values in (terms.source.required, terms.source.discharge_on,
                                          cell.hour_fractions)):
        raise PlanningSolutionRejected(code)
    values = tuple(tuple(_number(var.x, code, i) for i, var in enumerate(row))
                   for row in cell.bus_power_vars)
    totals, normalized, statuses, issues = [], [], [], []
    used = {var.idx for row in cell.bus_power_vars for var in row}
    for i in range(n):
        for b, row in enumerate(values):
            if not cell.bus_lower_bounds_kw[b][i] <= row[i] <= cell.bus_upper_bounds_kw[b][i]:
                raise PlanningSolutionRejected(code, interval_index=i)
        total = sum((Fraction(row[i]) * cell.cell_coefficients[b][i]
                     for b, row in enumerate(values)), Fraction(0))
        if total > cell.upper_bounds_kwh[i]:
            raise PlanningSolutionRejected(code, interval_index=i)
        on = None
        if (type(terms.source.required[i]) is not bool
            or (terms.source.required[i] and terms.source.policy_version != "aggregate_cell_min_1wh_v1")
            or (not terms.source.required[i] and terms.source.discharge_on[i] is not None)):
            raise PlanningSolutionRejected(code, interval_index=i)
        if terms.source.required[i]:
            var = terms.source.discharge_on[i]
            if var is None:
                if cell.upper_bounds_kwh[i] >= Fraction(1, 1000):
                    raise PlanningSolutionRejected(code, interval_index=i)
                on = 0
            else:
                index = _handle(var, model, code, i)
                if index in used:
                    raise PlanningSolutionRejected(code, interval_index=i)
                used.add(index)
                on = _binary_choice(var, model, evidence, code, i)
            if ((on == 0 and any(row[i] != 0 for row in values))
                or (on == 1 and total < Fraction(1, 1000))):
                raise PlanningSolutionRejected(code, interval_index=i)
        engine = _normalize_energy(total)
        if ((total > 0 and engine == 0) or Fraction(engine) > cell.upper_bounds_kwh[i]
            or (on == 1 and engine < Decimal("0.001"))):
            raise PlanningSolutionRejected("source_domain_normalization_invalid", interval_index=i)
        if Fraction(engine) != total:
            issues.append("forecast_energy_normalized")
        totals.append(total)
        normalized.append(engine)
        statuses.append(on)
    return SourceAudit(values, tuple(totals), tuple(normalized), tuple(statuses), tuple(dict.fromkeys(issues)))


def _check_model_solution(model):
    try:
        model.check_optimization_results()
    except (AssertionError, mip.InfeasibleSolution) as exc:
        raise PlanningSolutionRejected("source_domain_solution_invalid") from exc


def validate_source_solution(terms: BonusModelTerms, *, evidence: SolverEvidence) -> SourceAudit:
    source = _validate_physical_source(terms, evidence)
    _check_model_solution(terms.source.cell_discharge.model)
    return source


def _check_context(context, prices, opening, meter_id):
    """Eén gepinde raw prijsselectie, geen nieuwe DB-/configlezing."""
    code = "planning_forecast_invalid"
    try:
        batch, basis = context.price_batch, context.basis
        hashes = tuple(snapshot.snapshot_hash for snapshot in batch.snapshots)
        if (context.state != opening or context.state.asof_ts != prices[0].start_ts
            or basis.meter_id != meter_id or batch.meter_id != meter_id
            or basis.input_revision != batch.input_revision or batch.input_revision <= 0
            or set(basis.snapshot_hashes) != set(hashes) or len(hashes) != len(set(hashes))
            or basis.planning_start_ts != prices[0].start_ts or basis.horizon_end_ts != prices[-1].end_ts
            or batch.start_ts > prices[0].start_ts or batch.end_ts < prices[-1].end_ts):
            raise PlanningSolutionRejected(code)
        for price in prices:
            snapshot = resolve_snapshot(batch.snapshots, price.start_ts, price.end_ts)
            resolved = resolve_spot(snapshot, price.start_ts, price.end_ts,
                                    tuple(item.spot for item in batch.items))
            if (snapshot != price.snapshot or resolved.quality == "incomplete"
                or resolved.price_ex_vat != price.spot_ex_vat
                or quote(snapshot, price.start_ts, price.end_ts, resolved.price_ex_vat) != price):
                raise PlanningSolutionRejected(code)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise PlanningSolutionRejected(code) from exc


def _energy(value, issues, i):
    raw = _number(value, "planning_forecast_invalid", i)
    if raw < 0:
        raise PlanningSolutionRejected("planning_forecast_invalid", interval_index=i)
    normalized = _normalize_energy(Fraction(raw))
    if raw > 0 and normalized == 0:
        raise PlanningSolutionRejected("planning_forecast_invalid", interval_index=i)
    if normalized != raw:
        issues.append("forecast_energy_normalized")
    return normalized


def _series(terms, imports, exports, cell, source, issues):
    if not (len(imports) == len(exports) == len(cell) == len(terms.quotes)):
        raise PlanningSolutionRejected("planning_forecast_invalid")
    return tuple(InputInterval(
        start_ts=price.start_ts, end_ts=price.end_ts, import_kwh=_energy(imp, issues, i),
        export_kwh=_energy(exp, issues, i), battery_discharge_kwh=cell[i],
        origin=terms.bonus.origins[i], quality="estimated", sources=(source,),
        spot_ex_vat=price.spot_ex_vat, spot_source="planning_price_batch", input_revision=0,
        input_hash="", meter_id=terms.bonus.source.cell_discharge.meter_id,
    ) for i, (price, imp, exp) in enumerate(zip(terms.quotes, imports, exports, strict=True)))


def _energy_total(result):
    components = [(line.import_cost, line.base_export_revenue, line.bonus_revenue) for line in result.lines]
    if any(value is None for row in components for value in row):
        return None
    return sum((imp - base - bonus for imp, base, bonus in components), Decimal(0))


def _opening_total(period, start, state):
    if period is None:
        return None
    if start == period[0]:
        return Fraction(0)
    previous = state.bonus
    if (state.asof_ts == start and previous is not None and previous.asof_ts == start
        and (previous.period_start_ts, previous.period_end_ts) == period):
        return None if previous.total_export_kwh is None else Fraction(previous.total_export_kwh)
    return None


def _check_quantity(actual, reference, budget, index):
    """Onbekend blijft onbekend; geen generieke marge naast het benoemde budget."""
    if actual is None or reference is None:
        if actual is not None or reference is not None:
            raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=index)
        return
    value = actual if isinstance(actual, Fraction) else Fraction(_number(actual, _AUXILIARY_CODE, index))
    if abs(value - reference) > budget:
        raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=index)


def _auxiliary_handle(var, present, model, used, index):
    if not present:
        if var is not None:
            raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=index)
        return
    identity = _handle(var, model, _AUXILIARY_CODE, index)
    if identity in used:
        raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=index)
    used.add(identity)


def _auxiliary_quantity(var, reference, model, used, index, field, issues):
    _auxiliary_handle(var, reference is not None, model, used, index)
    if reference is None:
        return
    raw = Fraction(_number(var.x, _AUXILIARY_CODE, index))
    _check_quantity(raw, reference, _AUXILIARY_BUDGET, index)
    residual = raw - reference
    if residual:
        issues.append(f"planning_auxiliary_residual:interval={index}:field={field}:"
                      f"raw_kwh={raw}:reference_kwh={reference}:residual_kwh={residual}")


def _bonus_quantity(price, origin, cell, export, cumulative, first):
    """Onafhankelijke hoeveelheid; een nultarief bewijst geen nulhoeveelheid."""
    rules = price.snapshot.bonus
    if rules is None or not rules.enabled or price.spot_ex_vat <= 0 or export == 0 or cell > 0:
        return Fraction(0)
    if (cumulative is not None and rules.yearly_cap_kwh is not None
        and cumulative >= Fraction(rules.yearly_cap_kwh)):
        return Fraction(0)
    moment = datetime.fromtimestamp(price.start_ts, ZoneInfo(price.snapshot.timezone))
    supply = price.snapshot.supply_start
    if (moment.date() < rules.start_date or (supply is not None and moment.date() < supply)
        or not bonus_in_window(rules.windows, moment)):
        return Fraction(0)
    if (supply is None or cumulative is None or rules.cap_basis == "unconfirmed"
        or rules.vat_percent is None or origin != "verified_pv"
        or price.end_ts - price.start_ts != 900 or price.start_ts % 900):
        return None
    return first


def _check_engine_cap(line, price, period, begin, end, budget_before, budget_after, index):
    if line.before_state.asof_ts != price.start_ts or line.after_state.asof_ts != price.end_ts:
        raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=index)
    _check_quantity(_opening_total(period, price.start_ts, line.before_state), begin, budget_before, index)
    after_period = _bonus_period(price.snapshot, price.end_ts)
    after = line.after_state.bonus
    if after_period is None:
        if after is not None:
            raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=index)
        return
    if (after is None or after.asof_ts != price.end_ts
        or (after.period_start_ts, after.period_end_ts) != after_period):
        raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=index)
    reset = price.end_ts == after_period[0]
    _check_quantity(after.total_export_kwh, Fraction(0) if reset else end,
                    Fraction(0) if reset else budget_after, index)


def _audit_auxiliary_quantities(bonus, source, raw_exports, forecast, result, evidence, used, issues):
    """Reconstrueer C/Q/first/eligible exact uit raw export, niet uit solverhulpen."""
    fields = ("cumulative_export_kwh", "capped_opening_kwh", "capped_export_kwh", "first_kwh",
              "eligible_kwh", "opening_cap_saturated", "cap_saturated", "cap_known", "origins")
    n = len(bonus.quotes)
    if len(result.lines) != n or any(len(getattr(bonus, name)) != n for name in fields):
        raise PlanningSolutionRejected(_AUXILIARY_CODE)
    model = bonus.source.cell_discharge.model
    previous_period = cumulative = normalized_cumulative = None
    budget = Fraction(0)
    for i, (price, raw_export, item, line) in enumerate(zip(bonus.quotes, raw_exports, forecast, result.lines, strict=True)):
        period = _bonus_period(price.snapshot, price.start_ts)
        if i == 0 or period != previous_period:
            cumulative = _opening_total(period, price.start_ts, bonus.opening) if i == 0 else (
                Fraction(0) if period is not None and price.start_ts == period[0] else None)
            normalized_cumulative = cumulative
            budget = Fraction(0)
        known = period is not None and cumulative is not None
        if bonus.cap_known[i] is not known:
            raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=i)
        begin, normalized_begin = cumulative, normalized_cumulative
        export = Fraction(_number(raw_export, "planning_forecast_invalid", i))
        normalized_export = Fraction(item.export_kwh)
        delta = normalized_export - export
        budget_before, budget = budget, budget + abs(delta)
        first_budget = budget_before + budget
        cumulative = None if begin is None else begin + export
        normalized_cumulative = None if normalized_begin is None else normalized_begin + normalized_export
        rules = price.snapshot.bonus
        cap = None if rules is None or rules.yearly_cap_kwh is None else Fraction(rules.yearly_cap_kwh)
        q_begin = None if not known or cap is None else min(cap, begin)
        q_end = None if not known or cap is None else min(cap, cumulative)
        normalized_q_begin = None if not known or cap is None else min(cap, normalized_begin)
        normalized_q_end = None if not known or cap is None else min(cap, normalized_cumulative)
        first = None if not known else export if cap is None else q_end - q_begin
        normalized_first = None if not known else normalized_export if cap is None else normalized_q_end - normalized_q_begin
        eligible = first if bonus.source.required[i] and source.discharge_on[i] == 0 else Fraction(0)
        if eligible is None:
            raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=i)
        for field, reference in (("cumulative_export_kwh", cumulative), ("capped_opening_kwh", q_begin),
                                 ("capped_export_kwh", q_end), ("first_kwh", first), ("eligible_kwh", eligible)):
            _auxiliary_quantity(getattr(bonus, field)[i], reference, model, used, i, field, issues)
        for field, present in (("opening_cap_saturated", q_begin is not None), ("cap_saturated", q_end is not None)):
            var = getattr(bonus, field)[i]
            _auxiliary_handle(var, present, model, used, i)
            if present:
                _binary_choice(var, model, evidence, _AUXILIARY_CODE, i)
        if delta:
            issues.append(f"planning_export_normalization:interval={i}:delta_kwh={delta}:"
                          f"cumulative_budget_kwh={budget}:first_budget_kwh={first_budget}")
        _check_engine_cap(line, price, period, begin, cumulative, budget_before, budget, i)
        # Q is 1-Lipschitz; first heeft de som van het begin- en eindbudget.
        _check_quantity(normalized_q_begin, q_begin, budget_before, i)
        _check_quantity(normalized_q_end, q_end, budget, i)
        _check_quantity(normalized_first, first, first_budget, i)
        canonical = _bonus_quantity(price, bonus.origins[i], source.raw_cell_kwh[i], normalized_export,
                                    normalized_begin, normalized_first)
        _check_quantity(line.bonus_kwh, canonical, Fraction(0), i)
        if bonus.source.required[i]:
            _check_quantity(line.bonus_kwh, eligible, Fraction(0) if source.raw_cell_kwh[i] > 0 else first_budget, i)
        if source.raw_cell_kwh[i] > 0 and line.bonus_revenue != 0:
            raise PlanningSolutionRejected(_AUXILIARY_CODE, interval_index=i)
        previous_period = period


def audit_planning_solution(
    terms: FinancialModelTerms, *, baseline_import_kwh: Sequence[Decimal],
    baseline_export_kwh: Sequence[Decimal], evidence: SolverEvidence,
) -> PlanningAuditResult:
    with localcontext() as context:
        context.prec = 36
        return _audit_planning_solution(terms, tuple(baseline_import_kwh), tuple(baseline_export_kwh), evidence)


def _audit_planning_solution(terms, baseline_import, baseline_export, evidence):
    bonus = terms.bonus
    model = bonus.source.cell_discharge.model
    source = _validate_physical_source(bonus, evidence)
    if terms.quotes != bonus.quotes or len(terms.grid_import) != len(terms.quotes):
        raise PlanningSolutionRejected("planning_forecast_invalid")
    _check_context(terms.context, terms.quotes, bonus.opening, bonus.source.cell_discharge.meter_id)
    used = {v.idx for row in bonus.source.cell_discharge.bus_power_vars for v in row}
    used.update(var.idx for var in bonus.source.discharge_on if var is not None)
    if len(bonus.grid_export) != len(terms.quotes):
        raise PlanningSolutionRejected("planning_forecast_invalid")
    for var in (*terms.grid_import, *bonus.grid_export):
        index = _handle(var, model, "planning_forecast_invalid")
        if index in used:
            raise PlanningSolutionRejected("planning_forecast_invalid")
        used.add(index)
    issues = [*bonus.issues, *source.issues, *evidence.limitations]
    raw_exports = tuple(var.x for var in bonus.grid_export)
    forecast = _series(terms, tuple(var.x for var in terms.grid_import),
                       raw_exports, source.engine_cell_kwh,
                       "optimizer_forecast", issues)
    baseline = _series(terms, baseline_import, baseline_export, (Decimal(0),) * len(terms.quotes),
                       "fixed_no_battery_baseline", issues)
    forecast_result = settle_series(forecast, terms.context.price_batch.snapshots, bonus.opening)
    baseline_result = settle_series(baseline, terms.context.price_batch.snapshots, bonus.opening)
    _audit_auxiliary_quantities(bonus, source, raw_exports, forecast, forecast_result, evidence, used, issues)
    # De gewone modelcontrole blijft volledig, ná de eigen foutclassificatie van hulpvelden.
    _check_model_solution(model)
    errors = []
    expected_model = Decimal(0)
    for i, (price, line) in enumerate(zip(terms.quotes, forecast_result.lines, strict=True)):
        model_values = (_number(terms.grid_import[i].x, "planning_forecast_invalid", i) * price.import_rate,
                        _number(bonus.grid_export[i].x, "planning_forecast_invalid", i) * price.export_rate,
                        _number(bonus.eligible_kwh[i].x, "planning_forecast_invalid", i) * price.bonus_rate)
        expected_model += model_values[0] - model_values[1] - model_values[2]
        for actual, expected in zip(model_values, (line.import_cost, line.base_export_revenue, line.bonus_revenue), strict=True):
            if expected is not None:
                errors.append(abs(actual - expected))
    model_cost = _number(terms.energy_cost.x, "planning_parity_failed")
    errors.append(abs(model_cost - expected_model))
    forecast_cost, baseline_cost = _energy_total(forecast_result), _energy_total(baseline_result)
    if forecast_cost is not None:
        errors.append(abs(model_cost - forecast_cost))
    if any(error > Decimal("0.000001") for error in errors):
        raise PlanningSolutionRejected("planning_parity_failed")
    complete = forecast_cost is not None and baseline_cost is not None
    if not complete:
        issues.append("planning_parity_incomplete")
    for result in (forecast_result, baseline_result):
        for line in result.lines:
            issues.extend(line.issues)
    return PlanningAuditResult(source, evidence, forecast, baseline, forecast_result, baseline_result,
        model_cost, forecast_cost, baseline_cost,
        None if not complete else baseline_cost - forecast_cost,
        max(errors, default=Decimal(0)) if complete else None, complete, tuple(dict.fromkeys(issues)))
