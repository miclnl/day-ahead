"""Winter acceptance uses the original producer, physical handles and final audits."""

import ast
import datetime as dt
import inspect
import json
import sys
import textwrap
import time
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import mip
import pandas as pd
import pytest

import dao.prog.day_ahead as da
from dao.prog.config.models.devices.battery import BatteryConfig
from dao.prog.settlement.tariffs import quote
from dao.prog.settlement.types import MonthState
from dao.tests.settlement.test_optimization_bonus import ts
from dao.tests.settlement.test_optimization_producer import (
    capture as capture,
)
from dao.tests.settlement.test_optimization_producer import (
    context_for,
    post_strategy_consumers,
    pre_strategy_cost_binding,
)

D = Decimal
EUR_AUDIT = D("0.000001")


def strategy_source():
    """Execute the existing strategy and solver settings, not a second optimizer."""
    body = ast.parse(textwrap.dedent(inspect.getsource(da.DaCalc.calc_optimum))).body[0].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "bind_day_ahead_cost") + 1
    end = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "finalized_solution" for t in node.targets))
    # Keep returns valid while executing exactly the producer's strategy body.
    function = ast.FunctionDef(name="run_strategy", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg="model")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[*body[start:end], ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()),
                                                       args=[], keywords=[]))], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    return compile(module, da.__file__, "exec")


def winter_plan(captured, monkeypatch, snapshot, bonus_rules, *, quarters=96,
                peak="0.40", cheap="0.10", wear="0.01", efficiency="0.8",
                pv_per_day="0", demand_per_day="1", grid_charge=True,
                reserve=50, strategy="minimize cost", check_exclusivity_before_audit=False, battery_count=1,
                start_soc=50, start_day="2027-01-16", closed_month=False, mixed_vat=False, known_month=True):
    """Only synthetic read boundaries and explicit scenario limits are replaced."""
    instance = captured["instance"]
    instance._debug_capture_vars = True
    instance.strategy = strategy
    battery = instance.battery_options[0].model_dump()
    battery.update(lower_limit=reserve, optimal_lower_level=reserve,
                   penalty_low_soc=0, cycle_cost=float(wear),
                   bat_to_dc_efficiency=float(efficiency))
    instance.battery_options = [BatteryConfig.model_validate({**battery, "name": f"synthetic-winter-{b}"})
                                for b in range(battery_count)]
    start = ts(start_day, 0)
    stamps = [start + 900 * i for i in range(quarters)]
    dates = [dt.datetime.fromtimestamp(at) for at in stamps]
    spots = tuple(D(cheap if i % 96 < 4 else peak) for i in range(quarters))
    loads = [float(D(demand_per_day) / 4) if 72 <= i % 96 < 76 else 0.0
             for i in range(quarters)]
    pv = [float(D(pv_per_day) / 4) if 72 <= i % 96 < 76 else 0.0
          for i in range(quarters)]
    prices = pd.DataFrame({"time": dates, "da_ex": [float(p) for p in spots],
                           "da_cons": [float(p) for p in spots],
                           "da_prod": [float(p) for p in spots], "da_bonus": [0.0] * quarters})
    forecast = pd.DataFrame({"time": stamps, "tijd": dates,
                             "glob_rad": [0.0] * quarters, "temp": [5.0] * quarters})
    monkeypatch.setattr(da, "Report", lambda *args: SimpleNamespace(get_price_data=lambda *a, **k: prices.copy()))
    instance.db_da = SimpleNamespace(get_prognose_data=lambda **kwargs: forecast.copy())
    instance.calc_solar_predictions = lambda *args: pd.DataFrame({"prediction": pv})
    instance.use_calc_baseload = True
    instance.baseload_service = lambda: SimpleNamespace(
        record_presence=lambda: None, forecast_for_optimizer=lambda *args: loads.copy())
    found = {}

    class Captured(Exception):
        pass

    def capture_cost(model, **kwargs):
        found.update(sys._getframe(1).f_locals)
        raise Captured

    with monkeypatch.context() as spies:
        spies.setattr(da, "bind_day_ahead_cost", capture_cost)
        with pytest.raises(Captured):
            instance.calc_optimum(_start_dt=dates[0], _start_soc=start_soc)
    model = found["model"]
    assert found["settlement_context"] is None
    assert found["financial_terms"] is None
    assert found["B"] == battery_count and found["U"] == quarters
    assert found["EV"] == found["M"] == found["solar_num"] == 0
    assert not instance.boiler_present and not instance.hp_enabled
    # Explicit controlled acceptance scenario, not a production reserve change.
    for row in found["soc"]:
        model += row[-1] == row[0]
    for var in found["c_t"]:
        var.ub = 0
    if not grid_charge:
        for row in found["ac_to_dc"]:
            for var in row:
                var.ub = 0
    intervals = tuple((at, at + 900) for at in stamps)
    context = context_for(snapshot, bonus_rules, intervals, meter="legacy")
    snap = context.quotes[0].snapshot
    if closed_month:
        groups = ((D("9"), D("0.1")), (D("21"), D("0.1"))) if mixed_vat else ((D("21"), D("0.1")),)
        context = replace(context, state=replace(context.state, month=MonthState(snap.contract_id,
            ts("2027-01-01", 0), ts("2027-02-01", 0), D("-0.1"), groups, True)))
    elif known_month:
        # Explicit synthetic complete zero prefix: no unknown amount may be filled with zero by production code.
        context = replace(context, state=replace(context.state, month=MonthState(snap.contract_id,
            ts("2027-01-01", 0), ts("2027-02-01", 0), D("0"), (), True)))
    context = replace(context,
        quotes=tuple(quote(snap, a, b, price) for (a, b), price in zip(intervals, spots, strict=True)),
        price_batch=replace(context.price_batch, items=tuple(
            replace(item, spot=replace(item.spot, price_ex_vat=spots[i * 4]))
            for i, item in enumerate(context.price_batch.items))))
    namespace = {**da.__dict__, **found, "settlement_context": context}
    # The real producer still has context=None. Only its existing pure/source
    # seams receive the immutable synthetic context, without any gate bypass.
    exec(pre_strategy_cost_binding(), namespace)
    assert namespace["terminal_terms"].policy_version == "export_mean_v1"
    assert namespace["period_terms"].issues == (("monthly_credit_conservative_mixed_vat",) if mixed_vat else
        () if closed_month else ("monthly_credit_omitted_open_or_unknown",))
    optimize, solves = model.optimize, []

    def timed_solve(*args, **kwargs):
        start = time.perf_counter()
        status = optimize(*args, **kwargs)
        solves.append({"status": status.name, "incumbents": model.num_solutions,
                       "seconds": time.perf_counter() - start})
        return status

    monkeypatch.setattr(model, "optimize", timed_solve)
    exec(strategy_source(), namespace)
    solved = namespace["run_strategy"](model)
    assert model.num_solutions > 0 and model.status == mip.OptimizationStatus.OPTIMAL
    namespace.update(solved)
    model.check_optimization_results()
    raw = {
        "quarters": quarters, "peak": peak, "cheap": cheap, "battery_count": battery_count,
        "start_cell_kwh": found["soc"][0][0].x * found["one_soc"][0],
        "end_cell_kwh": found["soc"][0][-1].x * found["one_soc"][0],
        "grid_import_kwh": sum(var.x for var in found["c_l"]),
        "grid_export_kwh": sum(var.x for var in found["c_t"]),
        "charge_bus_kwh": sum(v.x * h for v, h in zip(found["dc_to_bat"][0], found["hour_fraction"], strict=True)),
        "discharge_bus_kwh": sum(v.x * h for v, h in zip(found["dc_from_bat"][0], found["hour_fraction"], strict=True)),
        "house_demand_kwh": sum(found["b_l"]), "pv_kwh": sum(pv),
        "other_flexible_load_kwh": sum(var.x for key in ("c_b", "c_hp") for var in found[key]),
        "preference_penalties_eur": sum(var.x for var in namespace["penalty_cost"]),
        "wear_eur": namespace["cycle_cost"][0].x,
        "objective_eur": found["cost"].x,
        "raw_cell_discharge_kw": [(i, var.x) for i, var in enumerate(found["dc_from_bat"][0]) if var.x != 0],
        "simultaneous_cell_intervals": [(i, charge.x, discharge.x) for i, (charge, discharge) in enumerate(
            zip(found["dc_to_bat"][0], found["dc_from_bat"][0], strict=True))
            if charge.x > 0 and discharge.x > 0],
        "source_required_count": sum(namespace["financial_terms"].bonus.source.required),
        "model_digest": da.model_digest(model), "solution_digest": da.solution_digest(model),
    }
    print("WINTER_RAW_EVIDENCE " + json.dumps(raw, sort_keys=True))
    # Raw evidence is not accepted: N4 may select one genuinely new candidate.
    exec(post_strategy_consumers(), namespace)
    selected = instance._debug_finalized_solution
    if check_exclusivity_before_audit:
        assert all(c.x == 0 or d.x == 0 for c, d in
                   zip(found["dc_to_bat"][0], found["dc_from_bat"][0], strict=True))
    assert (da.model_digest(model), da.solution_digest(model)) == (selected.model_digest, selected.solution_digest)
    audit = namespace["financial_audit"]
    assert audit.parity_complete and audit.parity_error_eur <= EUR_AUDIT
    if not closed_month:
        assert namespace["period_audit"].model_credit_eur == 0
        assert namespace["period_audit"].engine_closed_credit_eur == 0
    assert all(line.bonus_revenue == 0 for line in audit.forecast_result.lines)
    assert all(var.x == 0 for var in found["c_t"])
    assert all(var.x == 0 for key in ("c_b", "c_hp") for var in found[key])
    assert all(var.x == 0 for var in namespace["penalty_cost"])
    assert found["soc"][0][-1].x == pytest.approx(found["soc"][0][0].x, abs=1e-7)
    evidence = {
        "quarters": quarters, "peak": peak, "cheap": cheap, "wear_rate": wear,
        "cell_discharge_efficiency": efficiency, "pv_kwh_per_day": pv_per_day,
        "charge_bus_kwh": sum(v.x * h for v, h in zip(found["dc_to_bat"][0], found["hour_fraction"], strict=True)),
        "discharge_bus_kwh": sum(v.x * h for v, h in zip(found["dc_from_bat"][0], found["hour_fraction"], strict=True)),
        "grid_import_kwh": sum(v.x for v in found["c_l"]),
        "start_cell_kwh": found["soc"][0][0].x * found["one_soc"][0],
        "end_cell_kwh": found["soc"][0][-1].x * found["one_soc"][0],
        "energy_savings_eur": str(audit.savings_energy_eur),
        "wear_eur": str(namespace["battery_wear_eur"]), "objective_eur": found["cost"].x,
        "simultaneous_cell_intervals": [(i, charge.x, discharge.x) for i, (charge, discharge) in enumerate(
            zip(found["dc_to_bat"][0], found["dc_from_bat"][0], strict=True))
            if charge.x > 0 and discharge.x > 0],
        "model_digest": selected.model_digest, "solution_digest": selected.solution_digest,
        "battery_count": battery_count, "solves": solves,
        "final_status": selected.evidence.status,
        "refinement_policy": None if selected.refinement is None else selected.refinement.policy_version,
    }
    print("WINTER_EVIDENCE " + json.dumps(evidence, sort_keys=True))
    namespace["winter_evidence"] = evidence
    return namespace


@pytest.mark.parametrize("quarters", [96, 192])
def test_multiple_batteries_keep_winter_planning_available(capture, snapshot, bonus_rules, monkeypatch, quarters):
    """The physical B×T binaries must remain solvable under the original producer limits."""
    result = winter_plan(capture, monkeypatch, snapshot, bonus_rules, quarters=quarters, battery_count=2)
    assert result["financial_audit"].savings_energy_eur == pytest.approx(D("0.275") * (quarters // 96), abs=EUR_AUDIT)
    assert result["battery_wear_eur"] == pytest.approx(D("0.0225") * (quarters // 96), abs=EUR_AUDIT)
    assert result["winter_evidence"]["final_status"] == "OPTIMAL"


@pytest.mark.parametrize("quarters", [96, 192])
@pytest.mark.parametrize("pv,energy_savings,wear,net", [
    ("0", "0.275", "0.0225", "0.2525"),
    ("0.05", "0.26125", "0.021375", "0.239875"),
])
def test_no_pv_profitable_grid_charge_avoids_peak_import(capture, snapshot, bonus_rules, monkeypatch,
                                                       quarters, pv, energy_savings, wear, net):
    """Ignoring either loss or a wear side would inflate this independently derived saving."""
    result = winter_plan(capture, monkeypatch, snapshot, bonus_rules, quarters=quarters, pv_per_day=pv)
    days = quarters // 96
    audit, evidence = result["financial_audit"], result["winter_evidence"]
    useful = D("1") - D(pv)
    assert evidence["charge_bus_kwh"] == pytest.approx(float(useful * D("1.25") * days), abs=1e-6)
    assert evidence["discharge_bus_kwh"] == pytest.approx(float(useful * days), abs=1e-6)
    assert evidence["simultaneous_cell_intervals"] == []
    assert audit.savings_energy_eur == pytest.approx(D(energy_savings) * days, abs=EUR_AUDIT)
    assert result["battery_wear_eur"] == pytest.approx(D(wear) * days, abs=EUR_AUDIT)
    assert audit.savings_energy_eur - result["battery_wear_eur"] == pytest.approx(D(net) * days, abs=EUR_AUDIT)
    # Decimal reference, not a CBC-float comparison or a second efficiency discount.
    assert D("0.40") - D("1.25") * D("0.10") == D("0.275")
    assert (D("1.25") + D("1")) * D("0.01") == D("0.0225")
    assert D("0.275") - D("0.0225") == D("0.2525")
    assert result["planning_economics"]["net_model_advantage_eur"] == pytest.approx(D(net) * days, abs=EUR_AUDIT)


@pytest.mark.parametrize("quarters", [96, 192])
@pytest.mark.parametrize("peak,wear,efficiency", [("0.12", "0.01", "0.8"),
                                                  ("0.40", "0.20", "0.8"),
                                                  ("0.40", "0.01", "0.2")])
def test_small_spread_or_high_wear_does_not_force_cycles(capture, snapshot, bonus_rules, monkeypatch,
                                                       quarters, peak, wear, efficiency):
    """A positive spread is not sufficient once storage losses and both wear sides are paid."""
    result = winter_plan(capture, monkeypatch, snapshot, bonus_rules, quarters=quarters,
                         peak=peak, wear=wear, efficiency=efficiency)
    assert D("1.25") * D("0.10") + D("2.25") * D("0.01") == D("0.1475") > D("0.12")
    assert result["winter_evidence"]["charge_bus_kwh"] == 0
    assert result["winter_evidence"]["discharge_bus_kwh"] == 0
    assert result["financial_audit"].savings_energy_eur == 0
    assert result["battery_wear_eur"] == 0


@pytest.mark.parametrize("restriction", ["grid_charge", "reserve", "consumption_strategy"])
def test_reserve_or_grid_charge_prohibition_blocks_opportunity(capture, snapshot, bonus_rules, monkeypatch, restriction):
    """Explicit user restrictions are reported, not silently replaced to manufacture a cycle."""
    kwargs = {"grid_charge": False} if restriction == "grid_charge" else (
        {"reserve": 100, "start_soc": 100} if restriction == "reserve" else {"strategy": "minimize consumption"})
    result = winter_plan(capture, monkeypatch, snapshot, bonus_rules, **kwargs)
    assert result["winter_evidence"]["charge_bus_kwh"] == 0
    assert result["winter_evidence"]["discharge_bus_kwh"] == 0
    constraints = result["planning_economics"]["constraints"]
    if restriction == "grid_charge":
        assert constraints["grid_charge_possible_by_battery"] == (False,)
    elif restriction == "reserve":
        assert constraints["reserve_soc_percent"] == (100,)
    else:
        assert constraints["strategy"] == "minimize consumption"
        assert capture["instance"].strategy == "minimize consumption"


@pytest.mark.parametrize("closed", [False, True])
def test_winter_floor_and_terminal_are_not_double_counted(capture, snapshot, bonus_rules, monkeypatch, closed):
    """Each trajectory gets its own credit; unchanged physical end energy gets no terminal gain."""
    result = winter_plan(capture, monkeypatch, snapshot, bonus_rules,
        start_day="2027-01-31" if closed else "2027-01-16", closed_month=closed)
    audit, economics = result["financial_audit"], result["planning_economics"]
    assert audit.forecast_result.adjustments == audit.baseline_result.adjustments
    assert sum((a.credit_total for a in audit.forecast_result.adjustments), D("0")) == (D("0.121") if closed else 0)
    assert economics["battery_terminal_cost_eur"] == pytest.approx(D("0"), abs=EUR_AUDIT)
    assert economics["net_model_advantage_eur"] == pytest.approx(D("0.2525"), abs=EUR_AUDIT)
    assert economics["economics_supported"] is False
    assert economics["estimated"] is True


def test_winter_planning_survives_mixed_vat_month(capture, snapshot, bonus_rules, monkeypatch):
    """Mixed VAT is conservatively planned, then independently replayed for both trajectories."""
    result = winter_plan(capture, monkeypatch, snapshot, bonus_rules,
        start_day="2027-01-31", closed_month=True, mixed_vat=True)
    assert result["period_audit"].exact is False
    assert result["period_audit"].model_credit_eur == pytest.approx(D("0.109"), abs=EUR_AUDIT)
    assert result["period_audit"].engine_closed_credit_eur == D("0.115")
    assert result["financial_audit"].baseline_result.adjustments[0].credit_total == D("0.115")
    assert "monthly_credit_conservative_mixed_vat" in result["planning_economics"]["issues"]
    assert result["planning_economics"]["net_model_advantage_eur"] == pytest.approx(D("0.2525"), abs=EUR_AUDIT)


def test_unknown_month_does_not_become_full_net_savings(capture, snapshot, bonus_rules, monkeypatch):
    """Known A8 components can show a saving without certifying an incomplete invoice ledger."""
    result = winter_plan(capture, monkeypatch, snapshot, bonus_rules, known_month=False)
    assert result["financial_audit"].savings_energy_eur == pytest.approx(D("0.275"), abs=EUR_AUDIT)
    assert result["planning_economics"]["net_model_advantage_eur"] is None
    assert "planning_economics_incomplete" in result["planning_economics"]["issues"]


@pytest.mark.parametrize("quarters", [96, 192])
def test_winter_gain_not_from_terminal_depletion(capture, snapshot, bonus_rules, monkeypatch, quarters):
    result = winter_plan(capture, monkeypatch, snapshot, bonus_rules, quarters=quarters)
    assert result["winter_evidence"]["start_cell_kwh"] == result["winter_evidence"]["end_cell_kwh"] == 50
    assert result["planning_economics"]["battery_terminal_cost_eur"] == pytest.approx(D("0"), abs=EUR_AUDIT)
    assert result["planning_economics"]["net_model_advantage_eur"] == pytest.approx(D("0.2525") * (quarters // 96), abs=EUR_AUDIT)


def test_negative_prices_do_not_create_artificial_energy_loops(capture, snapshot, bonus_rules, monkeypatch):
    """No useful load, export or terminal depletion can justify simultaneous physical cell flows."""
    result = winter_plan(capture, monkeypatch, snapshot, bonus_rules, cheap="-1", demand_per_day="0",
                         check_exclusivity_before_audit=True)
    assert result["winter_evidence"]["simultaneous_cell_intervals"] == [], result["winter_evidence"]
    assert result["winter_evidence"]["charge_bus_kwh"] == 0
    assert result["financial_audit"].savings_energy_eur == 0
