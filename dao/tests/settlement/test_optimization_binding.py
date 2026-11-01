"""The optimizer must bind every original cell flow with correct units and bounds."""

import math
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction

import mip
import pytest

D = Decimal


def factory():
    from dao.prog.settlement.optimization import make_cell_discharge_input
    return make_cell_discharge_input


def binding(model, matrix, **overrides):
    return factory()(model, matrix, **{
        "meter_id": "synthetic-meter", "installation_battery_count": len(matrix),
        "battery_indices": tuple(range(len(matrix))), "no_battery_evidence": None,
        "hour_fractions": (D("0.25"), D("0.125")),
        "discharge_efficiencies": (D("0.8"), D("1"))[:len(matrix)],
        **overrides,
    })


def matrix_model():
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    matrix = tuple(tuple(model.add_var(lb=0, ub=bound) for _ in range(2))
                   for bound in (0.0032, 0.004))
    return model, matrix


def test_cell_conversion_and_outward_bounds():
    """Wrong efficiency/time conversion or inward bounds would lose real cell energy."""
    model, matrix = matrix_model()
    result = binding(model, matrix)
    assert result.upper_bounds_kwh == (Fraction(2, 1000), Fraction(1, 1000))
    assert result.cell_coefficients == ((Fraction(5, 16), Fraction(5, 32)),
                                        (Fraction(1, 4), Fraction(1, 8)))
    for i, expected in enumerate((0.002, 0.001)):
        for row in matrix:
            model += row[i] == row[i].ub
        assert result.total_kwh[i].const == 0
        assert result.total_kwh[i].expr[matrix[0][i]] == (0.3125, 0.15625)[i]
        assert result.model_upper_bounds_kwh[i] >= expected
        float_sum = sum(Fraction.from_float(row[i].ub) *
                        Fraction.from_float(result.model_coefficients[b][i])
                        for b, row in enumerate(matrix))
        assert Fraction.from_float(result.model_upper_bounds_kwh[i]) >= float_sum
        assert Fraction.from_float(result.model_upper_bounds_kwh[i]) >= result.upper_bounds_kwh[i]
    model.objective = mip.xsum(result.total_kwh)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert [item.x for item in result.total_kwh] == pytest.approx([0.002, 0.001])
    with pytest.raises(FrozenInstanceError):
        result.meter_id = "changed"


@pytest.mark.parametrize("case,code", [
    ("subset", "cell_coverage_invalid"), ("duplicate", "cell_coverage_invalid"),
    ("order", "cell_coverage_invalid"), ("foreign", "cell_model_identity_invalid"),
    ("missing_evidence", "cell_coverage_invalid"), ("false_evidence", "cell_coverage_invalid"),
])
def test_cell_coverage_requires_all_batteries_or_explicit_none(case, code):
    model, matrix = matrix_model()
    overrides = {}
    if case == "subset":
        matrix = matrix[:1]
        overrides = {"installation_battery_count": 2, "battery_indices": (0, 1)}
    elif case == "duplicate":
        matrix = (matrix[0], matrix[0])
    elif case == "order":
        overrides = {"battery_indices": (1, 0)}
    elif case == "foreign":
        _, matrix = matrix_model()
    elif case == "missing_evidence":
        matrix = ()
    else:
        overrides = {"no_battery_evidence": "not allowed with batteries"}
    with pytest.raises(ValueError) as rejected:
        binding(model, matrix, **overrides)
    assert rejected.value.code == code


def test_explicit_no_battery_preserves_time_axis():
    model = mip.Model(solver_name=mip.CBC)
    result = binding(model, (), no_battery_evidence="synthetic installation inventory")
    assert result.bus_power_vars == ()
    assert len(result.total_kwh) == 2
    assert result.upper_bounds_kwh == (Fraction(0), Fraction(0))
    assert all(item.const == 0 and not item.expr for item in result.total_kwh)


@pytest.mark.parametrize("efficiencies,hours", [
    ((D("0"), D("1")), (D("0.25"), D("0.125"))),
    ((D("1.1"), D("1")), (D("0.25"), D("0.125"))),
    ((D("NaN"), D("1")), (D("0.25"), D("0.125"))),
    ((D("1"), D("1")), (D("0"), D("0.125"))),
    ((D("1"), D("1")), (D("Infinity"), D("0.125"))),
    ((D("1"), D("1")), (D("1e-1000"), D("0.125"))),
])
def test_invalid_cell_conversion_rejected(efficiencies, hours):
    model, matrix = matrix_model()
    with pytest.raises(ValueError) as rejected:
        binding(model, matrix, discharge_efficiencies=efficiencies, hour_fractions=hours)
    assert rejected.value.code == "cell_conversion_invalid"


@pytest.mark.parametrize("negative", [True, False])
def test_invalid_cell_bounds_rejected(negative):
    model, matrix = matrix_model()
    if negative:
        matrix[0][0].lb = -1
    else:
        matrix[0][0].ub = math.inf
    with pytest.raises(ValueError) as rejected:
        binding(model, matrix)
    assert rejected.value.code == "cell_bounds_invalid"
