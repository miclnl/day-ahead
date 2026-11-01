"""Pure, explicitly bounded settlement terms for the existing optimizer."""

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Literal
from zoneinfo import ZoneInfo

import mip

from dao.prog.settlement.engine import _bonus_period, _contract_periods, _month_period
from dao.prog.settlement.tariffs import _effective_value, quote, resolve_snapshot
from dao.prog.settlement.types import (
    MonthState,
    Origin,
    PriceQuote,
    SettlementState,
    TariffSnapshot,
    TerminalValue,
)

SourcePolicy = Literal["aggregate_cell_min_1wh_v1"]


class OptimizationInputError(ValueError):
    def __init__(self, code: str, *, interval_index: int | None = None) -> None:
        self.code = code
        self.interval_index = interval_index
        super().__init__(code if interval_index is None else f"{code}: interval {interval_index}")


@dataclass(frozen=True, slots=True)
class CellDischargeInput:
    model: mip.Model
    meter_id: str
    installation_battery_count: int
    battery_indices: tuple[int, ...]
    no_battery_evidence: str | None
    bus_power_vars: tuple[tuple[mip.Var, ...], ...]
    hour_fractions: tuple[Decimal, ...]
    discharge_efficiencies: tuple[Decimal, ...]
    bus_lower_bounds_kw: tuple[tuple[Decimal, ...], ...]
    bus_upper_bounds_kw: tuple[tuple[Decimal, ...], ...]
    cell_coefficients: tuple[tuple[Fraction, ...], ...]
    model_coefficients: tuple[tuple[float, ...], ...]
    per_battery_kwh: tuple[tuple[mip.LinExpr, ...], ...]
    total_kwh: tuple[mip.LinExpr, ...]
    upper_bounds_kwh: tuple[Fraction, ...]
    model_upper_bounds_kwh: tuple[float, ...]


def _outward_float(value: Fraction, *, code: str, interval_index: int | None = None) -> float:
    """Rond een eindige niet-negatieve bound nooit naar binnen af."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise OptimizationInputError(code, interval_index=interval_index) from exc
    if value < 0 or not math.isfinite(result) or (value > 0 and result == 0):
        raise OptimizationInputError(code, interval_index=interval_index)
    while Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise OptimizationInputError(code, interval_index=interval_index)
    return result


def make_cell_discharge_input(
    model: mip.Model, dc_from_bat: Sequence[Sequence[mip.Var]], *,
    meter_id: str, installation_battery_count: int,
    battery_indices: Sequence[int], no_battery_evidence: str | None,
    hour_fractions: Sequence[Decimal], discharge_efficiencies: Sequence[Decimal],
) -> CellDischargeInput:
    """Bind alle originele DC-bus-kW eenmaal aan bruto cel-kWh."""
    count = installation_battery_count
    matrix = tuple(tuple(row) for row in dc_from_bat)
    indices = tuple(battery_indices)
    hours, efficiencies = tuple(hour_fractions), tuple(discharge_efficiencies)
    if (type(count) is not int or count < 0 or len(matrix) != count
        or indices != tuple(range(count)) or any(type(i) is not int for i in indices)
        or not isinstance(meter_id, str) or not meter_id.strip()
        or (count > 0 and no_battery_evidence is not None)
        or (count == 0 and (not isinstance(no_battery_evidence, str) or not no_battery_evidence.strip()))
        or any(len(row) != len(hours) for row in matrix)):
        raise OptimizationInputError("cell_coverage_invalid")
    if (not hours or len(efficiencies) != count
        or any(not isinstance(h, Decimal) or not h.is_finite() or h <= 0 for h in hours)
        or any(not isinstance(e, Decimal) or not e.is_finite() or not 0 < e <= 1
               for e in efficiencies)):
        raise OptimizationInputError("cell_conversion_invalid")

    seen = set()
    lower_bounds, bounds, ratios, coefficients, per_battery = [], [], [], [], []
    for b, row in enumerate(matrix):
        row_lower, row_bounds, row_ratios, row_coefficients, row_terms = [], [], [], [], []
        for i, var in enumerate(row):
            if (not isinstance(var, mip.Var) or var.model is not model or var.idx < 0
                or var.idx >= len(model.vars) or model.vars[var.idx] is not var):
                raise OptimizationInputError("cell_model_identity_invalid", interval_index=i)
            if var.idx in seen:
                raise OptimizationInputError("cell_coverage_invalid", interval_index=i)
            seen.add(var.idx)
            if (not math.isfinite(var.lb) or not math.isfinite(var.ub)
                or var.lb < 0 or var.ub < var.lb or var.ub >= sys.float_info.max):
                raise OptimizationInputError("cell_bounds_invalid", interval_index=i)
            ratio = Fraction(hours[i]) / Fraction(efficiencies[b])
            try:
                coefficient = float(hours[i]) / float(efficiencies[b])
            except (ZeroDivisionError, OverflowError) as exc:
                raise OptimizationInputError("cell_conversion_invalid", interval_index=i) from exc
            if not math.isfinite(coefficient) or coefficient <= 0:
                raise OptimizationInputError("cell_conversion_invalid", interval_index=i)
            row_lower.append(Decimal(str(var.lb)))
            row_bounds.append(Decimal(str(var.ub)))
            row_ratios.append(ratio)
            row_coefficients.append(coefficient)
            row_terms.append(var * coefficient)
        lower_bounds.append(tuple(row_lower))
        bounds.append(tuple(row_bounds))
        ratios.append(tuple(row_ratios))
        coefficients.append(tuple(row_coefficients))
        per_battery.append(tuple(row_terms))

    total, exact_bounds, model_bounds = [], [], []
    for i in range(len(hours)):
        exact = sum((Fraction(bounds[b][i]) * ratios[b][i] for b in range(count)), Fraction(0))
        actual = sum((Fraction.from_float(matrix[b][i].ub) * Fraction.from_float(coefficients[b][i])
                      for b in range(count)), Fraction(0))
        total.append(mip.xsum(per_battery[b][i] for b in range(count)))
        exact_bounds.append(exact)
        model_bounds.append(_outward_float(max(exact, actual), code="cell_bounds_invalid", interval_index=i))
    return CellDischargeInput(
        model, meter_id, count, indices, no_battery_evidence, matrix, hours, efficiencies,
        tuple(lower_bounds), tuple(bounds), tuple(ratios), tuple(coefficients), tuple(per_battery), tuple(total),
        tuple(exact_bounds), tuple(model_bounds),
    )


@dataclass(frozen=True, slots=True)
class SourceDomainTerms:
    policy_version: SourcePolicy | None
    required: tuple[bool, ...]
    discharge_on: tuple[mip.Var | None, ...]
    cell_discharge: CellDischargeInput


@dataclass(frozen=True, slots=True)
class BonusModelTerms:
    quotes: tuple[PriceQuote, ...]
    grid_export: tuple[mip.Var, ...]
    export_bounds_kwh: tuple[Decimal, ...]
    opening: SettlementState
    origins: tuple[Origin, ...]
    cap_known: tuple[bool, ...]
    cumulative_export_kwh: tuple[mip.Var | None, ...]
    capped_export_kwh: tuple[mip.Var | None, ...]
    cap_saturated: tuple[mip.Var | None, ...]
    capped_opening_kwh: tuple[mip.Var | None, ...]
    opening_cap_saturated: tuple[mip.Var | None, ...]
    first_kwh: tuple[mip.Var | None, ...]
    first_bounds_kwh: tuple[Decimal, ...]
    eligible_kwh: tuple[mip.Var, ...]
    revenue: mip.LinExpr
    source: SourceDomainTerms
    issues: tuple[str, ...]


def _expression_signature(expression: mip.LinExpr, model: mip.Model):
    if not isinstance(expression, mip.LinExpr):
        return None
    entries = []
    for var, coefficient in expression.expr.items():
        if var.model is not model or var.idx < 0 or model.vars[var.idx] is not var:
            return None
        entries.append((var.idx, coefficient))
    return expression.const, tuple(sorted(entries))


def _validate_cell_binding(cell: CellDischargeInput, model: mip.Model) -> None:
    """Frozen velden zijn geen bewijs: controleer alle afleidingen opnieuw."""
    if cell.model is not model:
        raise OptimizationInputError("cell_model_identity_invalid")
    checked = make_cell_discharge_input(
        model, cell.bus_power_vars, meter_id=cell.meter_id,
        installation_battery_count=cell.installation_battery_count,
        battery_indices=cell.battery_indices, no_battery_evidence=cell.no_battery_evidence,
        hour_fractions=cell.hour_fractions, discharge_efficiencies=cell.discharge_efficiencies,
    )
    for field in ("bus_lower_bounds_kw", "bus_upper_bounds_kw", "cell_coefficients", "model_coefficients",
                  "upper_bounds_kwh", "model_upper_bounds_kwh"):
        if getattr(cell, field) != getattr(checked, field):
            raise OptimizationInputError("cell_conversion_invalid")
    old = (*cell.total_kwh, *(term for row in cell.per_battery_kwh for term in row))
    new = (*checked.total_kwh, *(term for row in checked.per_battery_kwh for term in row))
    if len(old) != len(new) or any(_expression_signature(a, model) != _expression_signature(b, model)
                                  for a, b in zip(old, new, strict=True)):
        raise OptimizationInputError("cell_conversion_invalid")


def _cap_minimum(model, cumulative, lower, upper, cap, name):
    """Q=min(K,C), met beide richtingen onafhankelijk van de objective."""
    bound = _outward_float(min(cap, upper), code="planning_grid_invalid")
    q = model.add_var(lb=0, ub=bound, name=name)
    saturated = model.add_var(var_type=mip.BINARY, name=f"{name}_saturated")
    cap_float = float(cap)
    m_c = _outward_float(max(Fraction(0), upper - cap), code="planning_grid_invalid")
    m_k = _outward_float(max(Fraction(0), cap - lower), code="planning_grid_invalid")
    model += q <= cumulative
    model += q <= cap_float
    model += q >= cumulative - m_c * saturated
    model += q >= cap_float - m_k * (1 - saturated)
    return q, saturated


def build_bonus_terms(
    model: mip.Model, quotes: Sequence[PriceQuote], grid_export: Sequence[mip.Var],
    cell_discharge: CellDischargeInput, export_bounds_kwh: Sequence[Decimal],
    opening: SettlementState, *, origins: Sequence[Origin],
    source_policy: SourcePolicy | None = None,
) -> BonusModelTerms:
    """Tel alle export; bonus koppelt exact aan cap én originele celstromen."""
    with localcontext() as context:
        context.prec = 36
        return _build_bonus_terms(model, tuple(quotes), tuple(grid_export), cell_discharge,
                                  tuple(export_bounds_kwh), opening, tuple(origins), source_policy)


def _build_bonus_terms(model, quotes, exports, cell, bounds, opening, origins, policy):
    if policy is not None and policy != "aggregate_cell_min_1wh_v1":
        raise OptimizationInputError("source_domain_policy_invalid")
    _validate_cell_binding(cell, model)
    count = len(quotes)
    if any(not isinstance(price, PriceQuote) for price in quotes):
        raise OptimizationInputError("planning_quote_invalid")
    if (not count or len(exports) != count or len(bounds) != count or len(origins) != count
        or len(cell.hour_fractions) != count or not isinstance(opening, SettlementState)
        or opening.asof_ts != quotes[0].start_ts):
        raise OptimizationInputError("planning_grid_invalid")
    seen = {var.idx for row in cell.bus_power_vars for var in row}
    for i, (price, var, bound, origin) in enumerate(zip(quotes, exports, bounds, origins, strict=True)):
        try:
            checked_price = quote(price.snapshot, price.start_ts, price.end_ts, price.spot_ex_vat)
        except (TypeError, ValueError) as exc:
            raise OptimizationInputError("planning_quote_invalid", interval_index=i) from exc
        if origin not in ("verified_pv", "unknown") or price != checked_price:
            raise OptimizationInputError("planning_quote_invalid", interval_index=i)
        if (float(cell.hour_fractions[i]) != (price.end_ts - price.start_ts) / 3600
            or (i and quotes[i - 1].end_ts != price.start_ts)):
            raise OptimizationInputError("planning_grid_invalid", interval_index=i)
        if (not isinstance(var, mip.Var) or var.model is not model or var.idx in seen
            or var.idx < 0 or model.vars[var.idx] is not var
            or not math.isfinite(var.lb) or var.lb < 0 or not math.isfinite(var.ub)
            or var.ub >= sys.float_info.max or var.ub < var.lb
            or not isinstance(bound, Decimal) or not bound.is_finite()
            or bound < Decimal(str(var.ub))):
            raise OptimizationInputError("planning_grid_invalid", interval_index=i)
        seen.add(var.idx)

    cumulative_vars, capped_vars, saturated_vars, opening_vars, opening_saturated = [], [], [], [], []
    first_vars, first_bounds, eligible_vars, source_vars, known_flags, required_flags, issues = [], [], [], [], [], [], []
    previous_period = None
    cumulative = None
    lower = upper = Fraction(0)
    initial = None
    minimum = Fraction(1, 1000)
    for i, price in enumerate(quotes):
        period = _bonus_period(price.snapshot, price.start_ts)
        rules = price.snapshot.bonus
        if period is not None and price.end_ts > period[1]:
            raise OptimizationInputError("planning_grid_invalid", interval_index=i)
        if i == 0 or period != previous_period:
            initial = None
            if period is not None:
                if price.start_ts == period[0]:
                    initial = Decimal(0)
                elif (i == 0 and opening.bonus is not None
                      and opening.bonus.asof_ts == price.start_ts
                      and (opening.bonus.period_start_ts, opening.bonus.period_end_ts) == period):
                    initial = opening.bonus.total_export_kwh
            cumulative = None if initial is None else float(initial)
            lower = upper = Fraction(0) if initial is None else Fraction.from_float(float(initial))
        known = period is not None and cumulative is not None
        known_flags.append(known)
        q_begin = q_end = s_begin = s_end = first = c_end = None
        first_bound = Decimal(0)
        if known:
            cap = None if rules.yearly_cap_kwh is None else Fraction(rules.yearly_cap_kwh)
            first_bound = bounds[i] if cap is None else min(bounds[i], max(Decimal(0), rules.yearly_cap_kwh - initial))
            if cap is not None:
                q_begin, s_begin = _cap_minimum(model, cumulative, lower, upper, cap,
                                               f"settlement_Q_open_{i}")
            lower += Fraction.from_float(exports[i].lb)
            upper += max(Fraction(bounds[i]), Fraction.from_float(exports[i].ub))
            c_end = model.add_var(ub=_outward_float(upper, code="planning_grid_invalid"),
                                  name=f"settlement_C_{i}")
            model += c_end == cumulative + exports[i]
            first = model.add_var(ub=_outward_float(Fraction(first_bound), code="planning_grid_invalid"),
                                  name=f"settlement_first_{i}")
            if cap is None:
                model += first == exports[i]
            else:
                q_end, s_end = _cap_minimum(model, c_end, lower, upper, cap, f"settlement_Q_{i}")
                model += first == q_end - q_begin
            cumulative = c_end
        enabled = rules is not None and rules.enabled
        quarter = price.end_ts - price.start_ts == 900 and price.start_ts % 900 == 0
        basis = (enabled and price.snapshot.status == "confirmed" and rules.cap_basis != "unconfirmed"
                 and rules.vat_percent is not None)
        positive = price.bonus_rate > 0
        required = bool(quarter and basis and positive and known and first_bound > 0
                        and origins[i] == "verified_pv")
        required_flags.append(required)
        if required and policy is None:
            raise OptimizationInputError("source_domain_policy_required", interval_index=i)
        if enabled:
            if not known:
                issues.append("unknown_bonus_history")
            if origins[i] != "verified_pv":
                issues.append("unknown_bonus_source")
            if not basis:
                issues.append("unconfirmed_bonus_basis")
            if not quarter:
                issues.append("bonus_requires_quarter")
                if price.end_ts - price.start_ts == 3600:
                    issues.append("optimizer_bonus_disabled_hourly_grid")
        eligible = model.add_var(ub=_outward_float(Fraction(first_bound if required else 0),
                                                 code="planning_grid_invalid"),
                                 name=f"settlement_eligible_{i}")
        z = None
        if required:
            if cell.upper_bounds_kwh[i] < minimum:
                model += cell.total_kwh[i] == 0
                model += eligible == first
            else:
                z = model.add_var(var_type=mip.BINARY, name=f"settlement_source_{i}")
                scaled = cell.total_kwh[i] * 1000
                ratio = _outward_float(Fraction.from_float(cell.model_upper_bounds_kwh[i]) / minimum,
                                        code="cell_bounds_invalid", interval_index=i)
                f = _outward_float(Fraction(first_bound), code="planning_grid_invalid")
                model += scaled >= z
                model += scaled <= ratio * z
                model += eligible <= first
                model += eligible <= f * (1 - z)
                model += eligible >= first - f * z
        cumulative_vars.append(c_end)
        capped_vars.append(q_end)
        saturated_vars.append(s_end)
        opening_vars.append(q_begin)
        opening_saturated.append(s_begin)
        first_vars.append(first)
        first_bounds.append(first_bound)
        eligible_vars.append(eligible)
        source_vars.append(z)
        previous_period = period
    if policy is not None:
        issues.append(f"planning_source_policy={policy}")
    revenue = mip.xsum(var * float(price.bonus_rate) for var, price in zip(eligible_vars, quotes, strict=True))
    return BonusModelTerms(quotes, exports, bounds, opening, origins, tuple(known_flags),
        tuple(cumulative_vars), tuple(capped_vars), tuple(saturated_vars), tuple(opening_vars),
        tuple(opening_saturated), tuple(first_vars), tuple(first_bounds), tuple(eligible_vars),
        revenue, SourceDomainTerms(policy, tuple(required_flags), tuple(source_vars), cell),
        tuple(dict.fromkeys(issues)))


@dataclass(frozen=True, slots=True)
class MonthFloorTerms:
    """Contractmaand met ex-btw C en bruto EUR-creditondergrens/bovengrens."""

    period_start_ts: int
    period_end_ts: int
    contract_id: str
    interval_indices: tuple[int, ...]
    credit_ex_vat: mip.LinExpr
    credit: mip.LinExpr
    credit_upper: mip.LinExpr
    vat_bounds_percent: tuple[Decimal, Decimal] | None
    exact: bool
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PeriodModelTerms:
    """Bruto EUR-horizonsommen; de bovengrens is diagnostiek, geen extra opbrengst."""

    credit: mip.LinExpr
    credit_upper: mip.LinExpr
    exact: bool
    policy_version: Literal["vat_lower_bound_v1"]
    periods: tuple[MonthFloorTerms, ...]
    issues: tuple[str, ...]


def _validate_period_binding(model, prices, exports, bounds, opening, snapshots):
    if (not prices or len(exports) != len(prices) or len(bounds) != len(prices)
        or not isinstance(opening, SettlementState)):
        raise OptimizationInputError("planning_grid_invalid")
    if (any(not isinstance(price, PriceQuote) for price in prices)
        or any(not isinstance(snapshot, TariffSnapshot) for snapshot in snapshots)):
        raise OptimizationInputError("planning_quote_invalid")
    if opening.asof_ts != prices[0].start_ts:
        raise OptimizationInputError("planning_grid_invalid")
    seen = set()
    for i, (price, var, bound) in enumerate(zip(prices, exports, bounds, strict=True)):
        try:
            snapshot = resolve_snapshot(snapshots, price.start_ts, price.end_ts)
            checked = quote(snapshot, price.start_ts, price.end_ts, price.spot_ex_vat)
        except (TypeError, ValueError) as exc:
            raise OptimizationInputError("planning_quote_invalid", interval_index=i) from exc
        if checked != price:
            raise OptimizationInputError("planning_quote_invalid", interval_index=i)
        if (i and prices[i - 1].end_ts != price.start_ts):
            raise OptimizationInputError("planning_grid_invalid", interval_index=i)
        if (not isinstance(var, mip.Var) or var.model is not model or var.idx in seen
            or var.idx < 0 or var.idx >= len(model.vars) or model.vars[var.idx] is not var
            or not math.isfinite(var.lb) or var.lb < 0 or not math.isfinite(var.ub)
            or var.ub >= sys.float_info.max or var.ub < var.lb
            or not isinstance(bound, Decimal) or not bound.is_finite()
            or bound < Decimal(str(var.ub))):
            raise OptimizationInputError("planning_grid_invalid", interval_index=i)
        _outward_float(Fraction(bound), code="planning_grid_invalid", interval_index=i)
        seen.add(var.idx)


def _validate_month_prefix(opening, price, period):
    month = opening.month
    if month is None:
        return
    rates = set()
    negative = Fraction(0)
    for rate, basis in month.negative_basis_by_vat:
        if (not rate.is_finite() or rate < 0 or not basis.is_finite() or basis < 0
            or rate in rates):
            raise OptimizationInputError("planning_month_state_invalid")
        rates.add(rate)
        negative += Fraction(basis)
    if (month.complete and month.spot_export_eur is not None
        and negative < max(Fraction(0), -Fraction(month.spot_export_eur))):
        raise OptimizationInputError("planning_month_state_invalid")
    natural_end = _month_period(price.snapshot, price.start_ts, None, (None, None))[1]
    if (price.snapshot.monthly_export_floor != "none" and price.start_ts != period[0]
        and (month.contract_id != price.snapshot.contract_id or month.period_start_ts != period[0]
             or not month.period_start_ts <= opening.asof_ts < month.period_end_ts
             or month.period_end_ts > natural_end)):
        raise OptimizationInputError("planning_month_state_invalid")


def _month_floor_period(model, prices, exports, bounds, opening, key, indices):
    start, end, contract, floor = key
    indices = tuple(indices)
    zero = mip.xsum([])

    def result(ex_vat=zero, credit=zero, upper=zero, vat=None, exact=True, issues=()):
        return MonthFloorTerms(start, end, contract, indices, ex_vat, credit, upper, vat, exact, issues)

    if floor == "none":
        return result()
    initial = None
    rates = set()
    if prices[indices[0]].start_ts == start:
        initial = Decimal(0)
    elif indices[0] == 0 and opening.month is not None and opening.month.complete:
        initial = opening.month.spot_export_eur
        rates.update(rate for rate, basis in opening.month.negative_basis_by_vat if basis > 0)
    for i in indices:
        price = prices[i]
        if price.spot_ex_vat < 0 and bounds[i] > 0:
            day = datetime.fromtimestamp(price.start_ts, ZoneInfo(price.snapshot.timezone)).date()
            rate = _effective_value(price.snapshot.vat_production, day)
            if not rate.is_finite() or rate < 0:
                raise OptimizationInputError("planning_month_state_invalid", interval_index=i)
            rates.add(rate)
    if initial is None or prices[indices[-1]].end_ts != end:
        # Nul is hier alleen een weggelaten prikkel, geen bovengrens op de maandnota.
        return result(exact=False, issues=("monthly_credit_omitted_open_or_unknown",))

    initial_float = float(initial)
    if not math.isfinite(initial_float) or (initial != 0 and initial_float == 0):
        raise OptimizationInputError("planning_month_state_invalid")
    lower = upper = Fraction(initial)
    model_lower = model_upper = Fraction.from_float(initial_float)
    components = []
    for i in indices:
        spot, var = prices[i].spot_ex_vat, exports[i]
        coefficient = float(spot)
        if not math.isfinite(coefficient):
            raise OptimizationInputError("planning_quote_invalid", interval_index=i)
        lo = min(Fraction(Decimal(str(var.lb))), Fraction.from_float(var.lb))
        hi = max(Fraction(bounds[i]), Fraction.from_float(var.ub))
        a, b = Fraction(spot) * lo, Fraction(spot) * hi
        lower += min(a, b)
        upper += max(a, b)
        a = Fraction.from_float(coefficient) * Fraction.from_float(var.lb)
        b = Fraction.from_float(coefficient) * Fraction.from_float(var.ub)
        model_lower += min(a, b)
        model_upper += max(a, b)
        components.append(var * coefficient)
    lower, upper = min(lower, model_lower), max(upper, model_upper)
    if lower >= 0:
        return result()  # Structureel bewezen C=0; btwweging is dan niet nodig.
    if not rates:
        raise OptimizationInputError("planning_month_state_invalid")

    # C=max(0,-R), in beide richtingen begrensd zonder hulp van de objective.
    negative_bound = _outward_float(-lower, code="planning_grid_invalid")
    positive_bound = _outward_float(max(Fraction(0), upper), code="planning_grid_invalid")
    revenue = mip.xsum(components) + initial_float
    credit = model.add_var(ub=negative_bound, name=f"settlement_month_credit_{indices[0]}")
    active = model.add_var(var_type=mip.BINARY, name=f"settlement_month_negative_{indices[0]}")
    model += credit >= -revenue
    model += credit <= negative_bound * active
    model += credit <= -revenue + positive_bound * (1 - active)
    vat_bounds = min(rates), max(rates)
    factors = tuple(float(1 + rate / 100) for rate in vat_bounds)
    if any(not math.isfinite(factor) for factor in factors):
        raise OptimizationInputError("planning_month_state_invalid")
    exact = vat_bounds[0] == vat_bounds[1]
    return result(credit * 1, credit * factors[0], credit * factors[1], vat_bounds, exact,
                  () if exact else ("monthly_credit_conservative_mixed_vat",))


def build_month_floor_terms(
    model: mip.Model, quotes: Sequence[PriceQuote], grid_export: Sequence[mip.Var],
    export_bounds_kwh: Sequence[Decimal], opening: SettlementState, *,
    snapshots: Sequence[TariffSnapshot],
) -> PeriodModelTerms:
    """Bouw bruto creditgrenzen met dezelfde gepinde contractkalender als de engine."""
    prices, exports, bounds, snapshots = tuple(quotes), tuple(grid_export), tuple(export_bounds_kwh), tuple(snapshots)
    with localcontext() as context:
        context.prec = 36
        _validate_period_binding(model, prices, exports, bounds, opening, snapshots)
        contract_bounds = _contract_periods(sorted(snapshots, key=lambda snapshot: snapshot.valid_from_ts))
        previous = opening.month
        segments = []
        for i, price in enumerate(prices):
            start, end = _month_period(price.snapshot, price.start_ts, previous,
                                       contract_bounds[price.snapshot.valid_from_ts])
            if i == 0:
                _validate_month_prefix(opening, price, (start, end))
            if price.start_ts < start or price.end_ts > end:
                raise OptimizationInputError("planning_grid_invalid", interval_index=i)
            floor = price.snapshot.monthly_export_floor
            key = start, end, price.snapshot.contract_id, floor
            if not segments or segments[-1][0] != key:
                segments.append((key, []))
            segments[-1][1].append(i)
            # Alleen de kalender doorgeven; financiële afrekening blijft in de engine.
            previous = None if floor == "none" else MonthState(key[2], start, end, None, (), False)
        periods = tuple(_month_floor_period(model, prices, exports, bounds, opening, key, indices)
                        for key, indices in segments)
        return PeriodModelTerms(
            mip.xsum(period.credit for period in periods), mip.xsum(period.credit_upper for period in periods),
            all(period.exact for period in periods), "vat_lower_bound_v1", periods,
            tuple(dict.fromkeys(issue for period in periods for issue in period.issues)),
        )


def terminal_value(quotes: Sequence[PriceQuote], discharge_efficiencies: Sequence[float]) -> TerminalValue:
    """Exportgemiddelde als heuristiek; de cel/DC→AC-conversie blijft expliciet apart."""
    prices = tuple(quotes)
    if not prices:
        raise OptimizationInputError("planning_quote_invalid")
    with localcontext() as context:
        context.prec = 36
        weighted, duration = Decimal(0), 0
        for i, price in enumerate(prices):
            if not isinstance(price, PriceQuote):
                raise OptimizationInputError("planning_quote_invalid", interval_index=i)
            try:
                checked = quote(price.snapshot, price.start_ts, price.end_ts, price.spot_ex_vat)
            except (TypeError, ValueError) as exc:
                raise OptimizationInputError("planning_quote_invalid", interval_index=i) from exc
            if price != checked:
                raise OptimizationInputError("planning_quote_invalid", interval_index=i)
            seconds = price.end_ts - price.start_ts
            weighted += price.export_rate * seconds
            duration += seconds
        return TerminalValue("export_mean_v1", max(Decimal(0), weighted / duration),
                             tuple((str(i), Decimal(str(efficiency)))
                                   for i, efficiency in enumerate(discharge_efficiencies)))
