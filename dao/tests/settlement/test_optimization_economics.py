"""Wear costs use original DC-bus throughput, independently of cell energy."""

import datetime as dt
import importlib
import importlib.util
import logging
from decimal import Decimal, localcontext
from types import SimpleNamespace

import mip
import pytest

import dao.prog.day_ahead as da
from dao.prog.config.models.devices.battery import BatteryConfig
from dao.prog.settlement.optimization_audit import PlanningSolutionRejected
from dao.tests.config.test_battery_stages import BASE
from dao.tests.settlement.test_optimization_audit import financial_case, override_solution
from dao.tests.settlement.test_optimization_producer import (
    capture as capture,
)
from dao.tests.settlement.test_optimization_producer import post_strategy_consumers

D = Decimal


def wear_helper():
    name = "dao.prog.settlement.optimization_economics"
    assert importlib.util.find_spec(name) is not None, "W1 wear helper is missing"
    return importlib.import_module(name).battery_wear_cost


@pytest.mark.parametrize("capacity", [5, 10, 100])
def test_legacy_cycle_cost_keeps_bus_throughput_semantics(capacity):
    """Capacity scaling or counting only one side would change the legacy cost."""
    battery = BatteryConfig.model_validate({**BASE, "capacity": capacity})
    result = wear_helper()(dc_charge_kwh=D("1.25"), dc_discharge_kwh=D("1"),
        eur_per_dc_bus_throughput_kwh=D(str(battery.cycle_cost)))
    assert result == D("0.0225")


@pytest.mark.parametrize("charge,discharge,rate,expected", [
    ("1.25", "0", "0.01", "0.0125"),
    ("0", "1", "0.01", "0.01"),
    ("0", "0", "0.01", "0"),
    ("1.25", "1", "0", "0"),
    ("0.123456789", "0.987654321", "0.123456789012", "0.137174209876159123320"),
])
def test_wear_helper_counts_both_sides_without_rounding(charge, discharge, rate, expected):
    helper = wear_helper()
    with localcontext() as context:
        context.prec = 3
        assert helper(dc_charge_kwh=D(charge), dc_discharge_kwh=D(discharge),
            eur_per_dc_bus_throughput_kwh=D(rate)) == D(expected)


@pytest.mark.parametrize("field", ["dc_charge_kwh", "dc_discharge_kwh", "eur_per_dc_bus_throughput_kwh"])
@pytest.mark.parametrize("invalid", [D("-0.001"), D("NaN"), D("sNaN"), D("Infinity"), D("-Infinity"), 0.01, "0.01", None, True])
def test_wear_helper_rejects_invalid_inputs(field, invalid):
    helper = wear_helper()
    inputs = dict(dc_charge_kwh=D("1.25"), dc_discharge_kwh=D("1"),
                  eur_per_dc_bus_throughput_kwh=D("0.01"))
    inputs[field] = invalid
    with pytest.raises(ValueError, match=field):
        helper(**inputs)


def test_wear_helper_requires_explicit_inputs():
    with pytest.raises(TypeError):
        wear_helper()(dc_charge_kwh=D("1.25"), dc_discharge_kwh=D("1"))


def wear_case(*, rate="0.01", model_cost="0.0225"):
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    charges = [[model.add_var(lb=value, ub=value) for value in (4, 1)]]
    discharges = [[model.add_var(lb=value, ub=value) for value in (0, 4)]]
    costs = [model.add_var(lb=float(model_cost), ub=float(model_cost))]
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    return model, dict(dc_to_bat=charges, dc_from_bat=discharges, hour_fraction=(0.25, 0.25),
                      kwh_cycle_cost=(float(rate),), cycle_cost=costs)


def wear_audit(model, **kwargs):
    assert hasattr(da, "_audit_battery_wear_cost"), "W1 independent throughput audit is missing"
    return da._audit_battery_wear_cost(model, **kwargs)


def test_zero_wear_is_labeled_not_imputed():
    model, args = wear_case(rate="0", model_cost="0")
    wear, issues = wear_audit(model, **args)
    assert wear == D("0")
    assert issues == ("battery_wear_zero_assumption",)


def test_independent_wear_audit_uses_extracted_bus_throughput():
    model, args = wear_case()
    before = da.model_digest(model), da.solution_digest(model)
    with localcontext() as context:
        context.prec = 3
        assert wear_audit(model, **args) == (D("0.0225"), ())
    assert (da.model_digest(model), da.solution_digest(model)) == before


@pytest.mark.parametrize("defect", ["cost", "charge", "discharge", "nonfinite", "negative", "foreign", "shape", "hours", "rate"])
def test_independent_wear_audit_rejects_bad_cost_or_throughput(monkeypatch, defect):
    model, args = wear_case()
    if defect in ("cost", "charge", "discharge", "nonfinite", "negative"):
        var, value = {
            "cost": (args["cycle_cost"][0], 0.02),
            "charge": (args["dc_to_bat"][0][0], 3),
            "discharge": (args["dc_from_bat"][0][1], 3),
            "nonfinite": (args["dc_to_bat"][0][0], float("nan")),
            "negative": (args["dc_from_bat"][0][0], -1e-13),
        }[defect]
        override_solution(monkeypatch, model, {var.idx: value})
    elif defect == "foreign":
        foreign = mip.Model(solver_name=mip.CBC)
        args["cycle_cost"] = [foreign.add_var()]
    elif defect == "shape":
        args["dc_to_bat"] = [[]]
    elif defect == "hours":
        args["hour_fraction"] = (0, 0.25)
    elif defect == "rate":
        args["kwh_cycle_cost"] = (-0.01,)
    with pytest.raises(PlanningSolutionRejected, match="planning_battery_wear_mismatch"):
        wear_audit(model, **args)


def test_wear_audit_checks_horizon_sum_as_well_as_each_battery():
    model, args = wear_case(model_cost="0.0225006")
    args["dc_to_bat"] *= 2
    args["dc_from_bat"] *= 2
    args["kwh_cycle_cost"] *= 2
    args["cycle_cost"] *= 2
    with pytest.raises(PlanningSolutionRejected, match="planning_battery_wear_mismatch"):
        wear_audit(model, **args)


def test_real_producer_wear_terms_use_original_bus_handles(capture):
    """The original MIP must not use nominal capacity or cell efficiency in wear."""
    model, local = capture["model"], capture["locals"]
    for b, cost in enumerate(local["cycle_cost"]):
        rows = [constraint.expr.expr for constraint in model.constrs if cost in constraint.expr.expr]
        assert len(rows) == 1
        row = rows[0]
        assert len(row) == 25  # One cost plus both original flows for all 12 quarters.
        for flows in (local["dc_to_bat"], local["dc_from_bat"]):
            for var in flows[b]:
                assert abs(row[var] / row[cost]) == pytest.approx(0.0025)
    for b in range(2):
        for i in range(12):
            model += local["dc_to_bat"][b][i] == (1 if b == 0 and i < 5 else 0)
            model += local["dc_from_bat"][b][i] == (1 if b == 0 and 5 <= i < 9 else 0)
    model.verbose = 0
    model.objective = mip.xsum(local["cycle_cost"])
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert local["cycle_cost"][0].x == pytest.approx(0.0225)
    assert wear_audit(model, **{key: local[key] for key in (
        "dc_to_bat", "dc_from_bat", "hour_fraction", "kwh_cycle_cost", "cycle_cost")}) == (D("0.0225"), ())


def test_zero_wear_legacy_producer_logs_explicit_issue(capture, monkeypatch, caplog):
    """The unchanged legacy route must label zero too, without inventing a rate."""
    instance = capture["instance"]
    instance.battery_options = [battery.model_copy(update={"cycle_cost": 0}) for battery in instance.battery_options]

    class Captured(Exception):
        pass

    def stop_before_solve(*args, **kwargs):
        raise Captured

    monkeypatch.setattr(da, "bind_day_ahead_finance", stop_before_solve)
    with caplog.at_level(logging.INFO), pytest.raises(Captured):
        instance.calc_optimum(_start_dt=dt.datetime.fromtimestamp(capture["finance"]["intervals"][0][0]))
    assert "battery_wear_zero_assumption" in caplog.text
    assert all(battery.cycle_cost == 0 for battery in instance.battery_options)


@pytest.mark.parametrize("outcome", ["success", "zero", "mismatch", "mutation"])
def test_actual_post_finalization_wear_audit_precedes_preview(snapshot, bonus_rules, monkeypatch, caplog, outcome):
    """A stale cost or audit mutation must block capture and financial preview."""
    model, terms = financial_case(snapshot, bonus_rules)
    discharge = terms.bonus.source.cell_discharge.bus_power_vars
    # This wear-only case uses one interval; charge 5 kW for 0.25 h, no discharge.
    charges = [[model.add_var(lb=5, ub=5)]]
    cell_charge_on = da._bind_battery_cell_direction(model, dc_to_bat=charges, dc_from_bat=discharge,
        installation_battery_count=1, interval_count=1)
    rate = 0 if outcome == "zero" else 0.01
    costs = [model.add_var(lb=0)]
    model += costs[0] == (charges[0][0] + discharge[0][0]) * rate * 0.25
    cost = model.add_var(lb=-10, ub=10)
    model += cost == terms.energy_cost + costs[0]
    model.objective = cost
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert hasattr(da, "_audit_battery_wear_cost"), "W1 independent throughput audit is missing"
    audit = da._audit_battery_wear_cost
    def mutate(*args, **kwargs):
        result = audit(*args, **kwargs)
        model.add_var(ub=1)
        return result

    if outcome == "mutation":
        monkeypatch.setattr(da, "_audit_battery_wear_cost", mutate)
    instance = SimpleNamespace(_debug_capture_vars=True)
    namespace = dict(self=instance, model=model, financial_terms=terms, period_terms=None, cost=cost,
        native={"cbc_log": ""}, B=1, U=1, EV=0, M=0, solar_num=0, pv_dc_num=[0], hour_fraction=[0.25],
        dc_to_bat=charges, dc_from_bat=discharge, cell_charge_on=cell_charge_on,
        kwh_cycle_cost=[0.02 if outcome == "mismatch" else rate], cycle_cost=costs,
        ac_to_dc=charges, ac_from_dc=discharge, c_ev=[], c_ma_u=[], pv_ac=[], solar_prod=[],
        b_l=[0.0], c_b=terms.grid_import, c_hp=terms.grid_import, pl=[0.1], pt=[0.1])
    monkeypatch.setattr(model, "optimize", lambda *a, **k: pytest.fail("extra solve"))
    with caplog.at_level(logging.INFO):
        if outcome in ("mismatch", "mutation"):
            with pytest.raises(PlanningSolutionRejected, match="planning_battery_wear_mismatch|planning_solution_stale"):
                exec(post_strategy_consumers(), da.__dict__, namespace)
            assert not hasattr(instance, "_debug_planning_audit")
            assert "A8 energiecomponent" not in caplog.text
        else:
            exec(post_strategy_consumers(), da.__dict__, namespace)
            assert namespace["battery_wear_eur"] == (D("0") if outcome == "zero" else D("0.0125"))
            assert namespace["battery_wear_issues"] == (("battery_wear_zero_assumption",) if outcome == "zero" else ())
            assert "Batterijslijtage DC-busdoorzet EUR" in caplog.text
            assert (da.model_digest(model), da.solution_digest(model)) == (
                instance._debug_finalized_solution.model_digest, instance._debug_finalized_solution.solution_digest)
