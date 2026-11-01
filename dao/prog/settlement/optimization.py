"""Pure, explicitly bounded settlement terms for the existing optimizer."""

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Literal

import mip

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
    bounds, ratios, coefficients, per_battery = [], [], [], []
    for b, row in enumerate(matrix):
        row_bounds, row_ratios, row_coefficients, row_terms = [], [], [], []
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
            row_bounds.append(Decimal(str(var.ub)))
            row_ratios.append(ratio)
            row_coefficients.append(coefficient)
            row_terms.append(var * coefficient)
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
        tuple(bounds), tuple(ratios), tuple(coefficients), tuple(per_battery), tuple(total),
        tuple(exact_bounds), tuple(model_bounds),
    )
