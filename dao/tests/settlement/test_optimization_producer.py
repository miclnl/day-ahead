"""Real producer capture stays offline; an A8 audit never grants output rights."""

import datetime as dt
import logging
import socket
import sys
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import mip
import pandas as pd
import pytest

import dao.prog.day_ahead as da
from dao.prog.config.models.base import FlexFloat
from dao.prog.config.models.devices.battery import BatteryConfig
from dao.prog.config.models.production_bonus import ProductionBonusConfig
from dao.prog.fastctrl.plan import FinancialBasis
from dao.prog.settlement.optimization_audit import PlanningSolutionRejected
from dao.prog.settlement.publication import FinancialPlanBlocked
from dao.prog.settlement.service import PlanningContext
from dao.prog.settlement.tariffs import quote
from dao.prog.settlement.types import (
    BonusState,
    MonthState,
    PriceBatch,
    RegisteredSpot,
    SettlementState,
    SpotInterval,
)
from dao.tests.settlement.test_optimization_audit import financial_case, override_solution
from dao.tests.settlement.test_optimization_bonus import POLICY, ts
from dao.tests.settlement.test_publication import pricing_config

D = Decimal


def context_for(snapshot, bonus_rules, intervals, *, meter="synthetic-meter", total="0"):
    first, last = intervals[0][0], intervals[-1][1]
    year = dt.datetime.fromtimestamp(first, dt.UTC).year
    rules = replace(bonus_rules, start_date=dt.date(year, 1, 1),
                    windows=tuple((m, (0, 24)) for m in range(1, 13)))
    snap = snapshot(bonus=rules, valid_from_ts=ts(f"{year}-01-01", 0),
                    supply_start=dt.date(year, 1, 1))
    prices = tuple(quote(snap, a, b, D("0.10")) for a, b in intervals)
    basis = FinancialBasis(meter, 1, (snap.snapshot_hash,), ts(f"{year}-01-01", 0),
                           first, last, first, first)
    spots = tuple(RegisteredSpot(SpotInterval(a, a + 3600, D("0.10"), "synthetic", "estimated"),
                                1, "b" * 64)
                  for a in range(first // 3600 * 3600, last, 3600))
    batch = PriceBatch(meter, 1, first, last, spots, (snap,), first, first)
    opening = SettlementState(first, BonusState(ts(f"{year}-01-01", 0),
        ts(f"{year+1}-01-01", 0), first, D(total)), None)
    return PlanningContext(basis, opening, prices, batch)


@pytest.fixture
def capture(monkeypatch, tmp_path, request):
    """No DaBase construction or external reads; every output surface fails hard."""
    def forbidden(*args, **kwargs):
        raise AssertionError("offline producer attempted I/O")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(da, "FAST_PLAN_FILE", str(tmp_path / "fast_plan.json"))
    instance = da.DaCalc.__new__(da.DaCalc)
    instance.debug = True
    instance.file_name = "synthetic-unused-options"
    instance.interval, instance.interval_s, instance.steps_day = "15min", 900, 96
    instance.interval_name, instance.time_zone = "kwartier", "Europe/Amsterdam"
    instance.log_level = logging.INFO
    instance.config = SimpleNamespace(prices=None, baseload=[getattr(request, "param", 0.0)] * 24,
                                     max_gap=FlexFloat(value=0.005))
    instance.battery_options = [BatteryConfig.model_validate({
        "name": f"synthetic-{b}", "entity actual level": f"sensor.synthetic_{b}", "capacity": 100,
        "charge stages": [{"power": 4000, "efficiency": 1}],
        "discharge stages": [{"power": 4000, "efficiency": 1}],
        "bat_to_dc efficiency": efficiency, "dc_to_bat efficiency": 1,
        "cycle cost": 0.01,
        "solar": [{"name": f"pv-{b}", "max power": 4, "tilt": 30, "orientation": 0, "capacity": 4}],
    }) for b, efficiency in enumerate((0.8, 1.0))]
    instance.ev_options, instance.machines, instance.solar = [], [], []
    instance.boiler_options = instance.heating_options = None
    instance.salderen = False
    instance.grid_max_power = 17.0
    instance.production_bonus = ProductionBonusConfig(enabled=False)
    instance.strategy = "minimize cost"
    instance.use_calc_baseload = False
    instance.baseload_service = lambda: SimpleNamespace(record_presence=forbidden)
    instance.get_float = lambda entity, *args: 50.0
    instance.ha_getter = forbidden
    instance.notify = lambda *args: None  # No notification entity in this fixture.
    for name in ("set_value", "call_service", "turn_on", "turn_off", "set_entity_value",
                 "set_entity_state", "set_entity_option", "select_option", "save_df", "export_fast_plan"):
        monkeypatch.setattr(instance, name, forbidden)
    start = ts()
    stamps = [start + 900 * i for i in range(12)]
    dates = [dt.datetime.fromtimestamp(at) for at in stamps]
    prices = pd.DataFrame({"time": dates, "da_ex": [0.1] * 12,
                           "da_cons": [0.2] * 12, "da_prod": [0.1] * 12, "da_bonus": [0.0] * 12})
    forecast = pd.DataFrame({"time": stamps, "tijd": dates, "glob_rad": [0.0] * 12, "temp": [5.0] * 12})
    monkeypatch.setattr(da, "Report", lambda *args: SimpleNamespace(get_price_data=lambda *a, **k: prices.copy()))
    instance.db_da = SimpleNamespace(get_prognose_data=lambda **kwargs: forecast.copy())
    instance.calc_solar_predictions = lambda *args: pd.DataFrame({"prediction": [0.5] * 12})
    found = {}

    class Captured(Exception):
        pass

    def finance(model, **kwargs):
        found.update(model=model, finance=kwargs, locals=dict(sys._getframe(1).f_locals))
        return None

    def cost(model, **kwargs):
        assert model is found["model"]
        found["cost"] = kwargs
        found["locals"].update(sys._getframe(1).f_locals)
        raise Captured

    # Feature absence, rather than a fixture/network failure, owns the RED.
    assert hasattr(da, "bind_day_ahead_finance")
    assert hasattr(da, "bind_day_ahead_cost")
    with monkeypatch.context() as spies:
        spies.setattr(da, "bind_day_ahead_finance", finance)
        spies.setattr(da, "bind_day_ahead_cost", cost)
        with pytest.raises(Captured):
            instance.calc_optimum(_start_dt=dt.datetime.fromtimestamp(start))
    found["instance"] = instance
    return found


def signature(expr):
    return expr.const, {v.idx: c for v, c in expr.expr.items()}


@pytest.mark.parametrize("capture,cell_power", [(0.0, 0.0), (0.4, 0.0), (0.0, 0.0048)], indirect=["capture"])
def test_real_producer_matrix_and_cost_binding(capture, snapshot, bonus_rules, cell_power):
    model, args, local = capture["model"], capture["finance"], capture["locals"]
    assert args["context"] is None
    assert args["installation_battery_count"] == len(capture["instance"].battery_options) == 2
    assert args["battery_indices"] == (0, 1)
    assert args["hour_fractions"] == (D("0.25"),) * 12
    assert args["discharge_efficiencies"] == (D("0.8"), D("1.0"))
    indices = [v.idx for row in args["dc_from_bat"] for v in row]
    assert len(indices) == len(set(indices)) == 24
    assert indices == [v.idx for row in local["dc_from_bat"] for v in row]
    assert args["export_bounds_kwh"] == tuple(D(str(v.ub)) for v in local["c_t"])
    # Actual SoC rows contain each original handle with h/eta/one_soc, once.
    for b, row in enumerate(args["dc_from_bat"]):
        for i, var in enumerate(row):
            matches = [c.expr.expr for c in model.constrs
                       if local["soc"][b][i+1] in c.expr.expr and var in c.expr.expr]
            assert len(matches) == 1
            assert abs(matches[0][var]) == 0.25 / (0.8, 1.0)[b]
    context = context_for(snapshot, bonus_rules, args["intervals"], meter=args["meter_id"])
    terms = da.bind_day_ahead_finance(model, **{**args, "context": context, "origins": ("verified_pv",) * 12,
                                             "source_policy": POLICY})
    cost_args = capture["cost"]
    assert cost_args["financial_terms"] is None
    old_other = signature(cost_args["other_cost"])
    expected_other = {v.idx: 1.0 for name in ("cycle_cost", "penalty_cost") for v in local[name]}
    for b, coefficient in enumerate((0.08, 0.1)):
        expected_other[local["soc_mid"][b][0].idx] = coefficient
        expected_other[local["soc_mid"][b][12].idx] = -coefficient
    expected_other[local["boiler_temp"][0].idx] = 0.01648148148148148
    expected_other[local["boiler_temp"][12].idx] = -0.01648148148148148
    assert old_other[1] == pytest.approx(expected_other)
    da.bind_day_ahead_cost(model, **{**cost_args, "financial_terms": terms})
    expected = cost_args["cost"] - terms.energy_cost - cost_args["other_cost"]
    assert signature(model.constrs[-1].expr)[1] == signature(expected)[1]
    assert signature(cost_args["other_cost"]) == old_other
    coeffs = terms.energy_cost.expr
    for imp, exp, eligible in zip(terms.grid_import, terms.bonus.grid_export, terms.bonus.eligible_kwh, strict=True):
        assert coeffs[imp] == 0.1
        assert coeffs[exp] == -0.1
        assert coeffs[eligible] == -0.03
    # DC PV pass-through is real and separate from gross cell discharge.
    for b, row in enumerate(args["dc_from_bat"]):
        for i, var in enumerate(row):
            model += var == (cell_power if b == 0 else 0)
            model += local["dc_to_bat"][b][i] == 0
    model.verbose = 0
    model.objective = mip.minimize(cost_args["cost"])
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    if cell_power == 0:
        # Real producer export exposes CBC first/eligible residuals, even at 1 kWh.
        # Keep this availability failure visible; do not clamp the candidate.
        assert terms.bonus.eligible_kwh[0].x == pytest.approx(terms.bonus.grid_export[0].x)
        with pytest.raises(PlanningSolutionRejected, match="source_domain_solution_invalid"):
            da.finish_day_ahead_finance(terms, org_l=[0.0]*12, org_t=[-0.9]*12, cbc_log="")
        return
    result = da.finish_day_ahead_finance(terms, org_l=[0.0]*12, org_t=[-1.0]*12, cbc_log="")
    assert result.parity_complete
    assert result.forecast[0].export_kwh == D("1.0012")  # DC PV plus 0.0048 * 0.25 bus energy.
    assert result.forecast[0].battery_discharge_kwh == D("0.0015")  # Gross, with eta=0.8.
    assert result.forecast_result.lines[0].bonus_revenue == 0
    assert float(result.model_energy_cost_eur) == pytest.approx(terms.energy_cost.x)


def test_baseline_sign_and_independent_cap_state(snapshot, bonus_rules):
    intervals = ((ts(), ts()+900), (ts()+900, ts()+1800))
    context = context_for(snapshot, bonus_rules, intervals, total="4999")
    context = replace(context, state=replace(context.state, month=MonthState(
        "synthetic-2027", ts(hour=0), ts("2027-02-01", 0), D("0"), (), True)))
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    terms = da.bind_day_ahead_finance(model, context=context, intervals=intervals,
        meter_id="synthetic-meter", dc_from_bat=(), installation_battery_count=0,
        battery_indices=(), no_battery_evidence="synthetic inventory",
        hour_fractions=(D("0.25"),)*2, discharge_efficiencies=(),
        grid_import=tuple(model.add_var(ub=0) for _ in intervals),
        grid_export=tuple(model.add_var(lb=1, ub=1) for _ in intervals),
        export_bounds_kwh=(D("1"),)*2, origins=("verified_pv",)*2, source_policy=POLICY)
    model.objective = terms.energy_cost
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    result = da.finish_day_ahead_finance(terms, org_l=(0.0, 0.0), org_t=(0.0, -1.0), cbc_log="")
    assert result.forecast_result.lines[1].bonus_kwh == D("0")
    assert result.baseline_result.lines[1].bonus_kwh == D("1")
    assert result.baseline[1].export_kwh == D("1")
    assert result.forecast_result.lines[0].before_state == result.baseline_result.lines[0].before_state
    assert result.forecast_result.end_state.month != result.baseline_result.end_state.month
    assert result.savings_energy_eur == result.baseline_energy_cost_eur - result.engine_energy_cost_eur
    with pytest.raises(PlanningSolutionRejected, match="planning_forecast_invalid"):
        da.finish_day_ahead_finance(terms, org_l=(0.0, 0.0), org_t=(0.0, 1.0), cbc_log="")


def test_final_selected_solution_audited_before_consumers(snapshot, bonus_rules, monkeypatch):
    model, terms = financial_case(snapshot, bonus_rules, cells=((None,),), exports=("1",))
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    model.objective = cell
    model.optimize()
    first = da.finish_day_ahead_finance(terms, org_l=(0.0,), org_t=(-1.0,), cbc_log="")
    assert first.forecast[0].battery_discharge_kwh == 0
    model += cell >= 0.006
    model.optimize()
    second = da.finish_day_ahead_finance(terms, org_l=(0.0,), org_t=(-1.0,), cbc_log="")
    assert second.forecast[0].battery_discharge_kwh == D("0.0015")
    assert second.forecast_result.lines[0].bonus_kwh == 0
    preview_calls, prognosis_writes, plan_writes, ha_writes = [], [], [], []
    override_solution(monkeypatch, model, {cell.idx: -1e-12})
    with pytest.raises(PlanningSolutionRejected):
        audited = da.finish_day_ahead_finance(terms, org_l=(0.0,), org_t=(-1.0,), cbc_log="")
        preview_calls.append(audited)
        prognosis_writes.append(audited)
        plan_writes.append(audited)
        ha_writes.append(audited)
    assert preview_calls == prognosis_writes == plan_writes == ha_writes == []


def test_new_regime_denied_even_debug_and_valid_offline_audit(capture, snapshot, bonus_rules, monkeypatch):
    _, terms = financial_case(snapshot, bonus_rules)
    assert da.finish_day_ahead_finance(terms, org_l=(0.0,), org_t=(-1.0,), cbc_log="").parity_complete
    instance = capture["instance"]
    instance.config.prices = pricing_config()
    for attr in ("_financial_pricing", "_financial_policy", "_financial_plan_path"):
        if hasattr(instance, attr):
            delattr(instance, attr)
    monkeypatch.setattr(da, "Report", lambda *args: pytest.fail("denied flow reached prices"))
    with pytest.raises(FinancialPlanBlocked):
        instance.calc_optimum(_start_dt=dt.datetime.fromtimestamp(ts()))
    with pytest.raises(FinancialPlanBlocked, match="actuation_gates_unproven"):
        instance._check_financial_output(basis=terms.context.basis, now_ts=ts())


@pytest.mark.parametrize("mode", ["quarter", "rest", "hour", "missing", "mixed", "gap", "stale"])
def test_context_quote_alignment_and_no_legacy_fallback(snapshot, bonus_rules, monkeypatch, mode):
    intervals = tuple((ts()+i*900, ts()+(i+1)*900) for i in range(4))
    context = context_for(snapshot, bonus_rules, intervals)
    actual = ((ts(), ts()+3600),) if mode in ("hour", "missing", "mixed", "gap", "stale") else intervals
    if mode == "rest":
        actual = ((ts()+300, ts()+900), *intervals[1:])
        context = context_for(snapshot, bonus_rules, actual)
    if mode == "missing":
        context = replace(context, quotes=context.quotes[:-1])
    elif mode == "mixed":
        context = replace(context, quotes=(context.quotes[0], replace(context.quotes[1], spot_ex_vat=D("0.2")), *context.quotes[2:]))
    elif mode == "gap":
        context = replace(context, quotes=(context.quotes[0], *context.quotes[2:]))
    elif mode == "stale":
        context = replace(context, price_batch=replace(context.price_batch, input_revision=2))
    model = mip.Model(solver_name=mip.CBC)
    args = dict(context=context, intervals=actual, meter_id="synthetic-meter", dc_from_bat=(),
        installation_battery_count=0, battery_indices=(), no_battery_evidence="synthetic inventory",
        hour_fractions=tuple(D(str((b-a)/3600)) for a,b in actual), discharge_efficiencies=(),
        grid_import=tuple(model.add_var(ub=2) for _ in actual),
        grid_export=tuple(model.add_var(ub=2) for _ in actual), export_bounds_kwh=(D("2"),)*len(actual),
        origins=("verified_pv",)*len(actual), source_policy=POLICY)
    monkeypatch.setattr(da.production_bonus, "add_to_model", lambda *a, **k: pytest.fail("financial fallback"))
    if mode in ("missing", "mixed", "gap", "stale"):
        with pytest.raises(ValueError, match="planning_quote_invalid|planning_grid_invalid"):
            da.bind_day_ahead_finance(model, **args)
    else:
        bound = da.bind_day_ahead_finance(model, **args)
        assert tuple((q.start_ts, q.end_ts) for q in bound.quotes) == actual
        if mode == "hour":
            assert bound.bonus.source.required == (False,)
            assert "optimizer_bonus_disabled_hourly_grid" in bound.bonus.issues
        if mode == "rest":
            assert bound.bonus.source.required == (False, True, True, True)
    cols, rows = model.num_cols, model.num_rows
    assert da.bind_day_ahead_finance(model, **{**args, "context": None}) is None
    assert (model.num_cols, model.num_rows) == (cols, rows)
