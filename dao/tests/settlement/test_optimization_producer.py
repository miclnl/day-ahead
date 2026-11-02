"""Real producer capture stays offline; an A8 audit never grants output rights."""

import ast
import datetime as dt
import inspect
import logging
import socket
import sys
import textwrap
from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import mip
import pandas as pd
import pytest

import dao.prog.day_ahead as da
from dao.prog.config.models.base import FlexFloat
from dao.prog.config.models.devices.battery import BatteryConfig
from dao.prog.config.models.pricing import PricingConfig
from dao.prog.config.models.production_bonus import ProductionBonusConfig
from dao.prog.fastctrl.plan import FinancialBasis
from dao.prog.settlement.optimization import OptimizationInputError
from dao.prog.settlement.optimization_audit import (
    PlanningSolutionRejected,
    validate_source_solution,
)
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
from dao.tests.settlement.test_optimization_periods import (
    audited_period_case,
    financial_period_case,
)
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
    settings = getattr(request, "param", {})
    if isinstance(settings, (int, float)):
        settings = {"baseload": settings}
    if settings.get("hourly"):
        instance.interval, instance.interval_s, instance.steps_day = "1hour", 3600, 24
    instance.config = SimpleNamespace(prices=settings.get("prices"), baseload=[settings.get("baseload", 0.0)] * 24,
                                     max_gap=FlexFloat(value=0.005))
    instance.battery_options = [BatteryConfig.model_validate({
        "name": f"synthetic-{b}", "entity actual level": f"sensor.synthetic_{b}", "capacity": 100,
        "charge stages": [{"power": 2000, "efficiency": 1}, {"power": 4000, "efficiency": 1}],
        "discharge stages": [{"power": 2000, "efficiency": 1}, {"power": 4000, "efficiency": 1}],
        "bat_to_dc efficiency": efficiency, "dc_to_bat efficiency": 1,
        "cycle cost": 0.01,
        "solar": [{"name": f"pv-{b}", "max power": 4, "tilt": 30, "orientation": 0, "capacity": 4}],
        **settings.get("battery", {}),
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
    start = settings.get("start", ts())
    grid_start = start // instance.interval_s * instance.interval_s
    stamps = [grid_start + instance.interval_s * i for i in range(12)]
    dates = [dt.datetime.fromtimestamp(at) for at in stamps]
    prices = pd.DataFrame({"time": dates, "da_ex": [0.1] * 12,
                           "da_cons": [0.2] * 12, "da_prod": [0.1] * 12, "da_bonus": [0.0] * 12})
    forecast = pd.DataFrame({"time": stamps, "tijd": dates, "glob_rad": [0.0] * 12, "temp": [5.0] * 12})
    monkeypatch.setattr(da, "Report", lambda *args: SimpleNamespace(get_price_data=lambda *a, **k: prices.copy()))
    instance.db_da = SimpleNamespace(get_prognose_data=lambda **kwargs: forecast.copy())
    instance.calc_solar_predictions = lambda *args: pd.DataFrame({"prediction": [0.5] * 12})
    found = {}
    add_sos = mip.Model.add_sos

    def record_sos(model, members, sos_type):
        found.setdefault("sos", []).append((model, tuple((var.idx, weight) for var, weight in members), sos_type))
        return add_sos(model, members, sos_type)

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
        spies.setattr(mip.Model, "add_sos", record_sos)
        spies.setattr(da, "bind_day_ahead_finance", finance)
        spies.setattr(da, "bind_day_ahead_cost", cost)
        with pytest.raises(Captured):
            instance.calc_optimum(_start_dt=dt.datetime.fromtimestamp(start))
    found["instance"] = instance
    return found


def signature(expr):
    return expr.const, {v.idx: c for v, c in expr.expr.items()}


@pytest.mark.parametrize("capture", [{"prices": PricingConfig.model_validate(
    pricing_config().model_dump(exclude={"settlement"}))}], indirect=True)
def test_real_producer_accepts_legacy_pricing_without_settlement(capture):
    """Normal legacy pricing must reach cost binding and solve without opting into settlement."""
    model, args = capture["model"], capture["finance"]
    assert capture["instance"]._financial_pricing.settlement is None
    assert args["context"] is None and args["meter_id"] == "legacy" and args["source_policy"] is None
    assert da.bind_day_ahead_finance(model, **args) is None
    assert capture["cost"]["financial_terms"] is None
    da.bind_day_ahead_cost(model, **capture["cost"])
    model.objective = capture["cost"]["cost"]
    model.verbose = 0
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL


@pytest.mark.parametrize("capture,cell_power,residual", [
    (0.0, 0.0, False), (0.4, 0.0, False), (0.0, 0.0048, False),
    (0.0, 0.0, True),
    ({"start": ts("2027-12-31", 23, 45)}, 0.0, True),
    ({"start": ts("2027-12-31", 23, 45)}, 0.0048, False),
], indirect=["capture"])
def test_real_producer_refinement_and_positive_bonus(capture, snapshot, bonus_rules, monkeypatch,
                                                    cell_power, residual):
    model, args, local = capture["model"], capture["finance"], capture["locals"]
    assert args["context"] is None
    assert args["installation_battery_count"] == len(capture["instance"].battery_options) == 2
    assert args["battery_indices"] == (0, 1)
    assert args["hour_fractions"] == (D("0.25"),) * 12
    assert args["discharge_efficiencies"] == (D("0.8"), D("1.0"))
    # Native, nontrivial SOS2 constraints really went into the captured model.
    assert len(capture["sos"]) == 48
    assert all(owner is model and len(members) == 3 and kind == 2
               for owner, members, kind in capture["sos"])
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
    periods = da.build_month_floor_terms(model, terms.quotes, terms.bonus.grid_export,
        terms.bonus.export_bounds_kwh, terms.bonus.opening, snapshots=terms.context.price_batch.snapshots)
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
    da.bind_day_ahead_cost(model, **{**cost_args, "financial_terms": terms, "period_terms": periods})
    expected = cost_args["cost"] - terms.energy_cost + periods.credit - cost_args["other_cost"]
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
    optimize, raw_get = model.optimize, model.solver.var_get_x
    solves = []
    if residual:
        # Controlled initial read defect, not a claim about this CBC invocation.
        override_solution(monkeypatch, model, {args["dc_from_bat"][0][0].idx: 1e-13})

    def resolve():
        solves.append(True)
        monkeypatch.setattr(model.solver, "var_get_x", raw_get)
        return optimize()

    monkeypatch.setattr(model, "optimize", resolve)
    selected = da.finalize_day_ahead_finance(terms, cbc_log="")
    result = da.finish_day_ahead_finance(selected, org_l=[0.0]*12, org_t=[-1.0]*12)
    period_result = da.audit_period_solution(selected.terms, periods, planning_audit=result)
    assert period_result.model_credit_eur == 0
    assert period_result.exact is False
    assert (da.model_digest(model), da.solution_digest(model)) == (selected.model_digest, selected.solution_digest)
    assert len(solves) == int(residual)
    assert (selected.refinement is not None) == residual
    assert signature(cost_args["other_cost"]) == old_other
    validate_source_solution(selected.terms.bonus, evidence=selected.evidence)
    assert result.evidence is selected.evidence
    assert result.parity_complete
    assert result.parity_error_eur <= D("0.000001")
    assert result.forecast[0].battery_discharge_kwh == (D("0.0015") if cell_power else D("0"))
    if cell_power:
        assert result.forecast[0].export_kwh == D("1.0012")  # DC PV plus bus energy.
        assert all(line.bonus_revenue == 0 for line in result.forecast_result.lines)
    else:
        assert terms.bonus.eligible_kwh[0].x == pytest.approx(terms.bonus.grid_export[0].x)
        assert result.forecast[0].export_kwh == (D("0.9") if capture["instance"].config.baseload[0] else D("1"))
        assert all(line.bonus_kwh == item.export_kwh for line, item in zip(
            result.forecast_result.lines, result.forecast, strict=True))
        assert all(line.bonus_revenue > 0 for line in result.forecast_result.lines)
    if args["intervals"][0][0] == ts("2027-12-31", 23, 45):
        assert result.forecast_result.lines[0].after_state.bonus.total_export_kwh == D("0")
        assert result.forecast_result.lines[1].after_state.bonus.total_export_kwh == result.forecast[1].export_kwh
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
    selected = da.finalize_day_ahead_finance(terms, cbc_log="")
    result = da.finish_day_ahead_finance(selected, org_l=(0.0, 0.0), org_t=(0.0, -1.0))
    assert result.forecast_result.lines[1].bonus_kwh == D("0")
    assert result.baseline_result.lines[1].bonus_kwh == D("1")
    assert result.baseline[1].export_kwh == D("1")
    assert result.forecast_result.lines[0].before_state == result.baseline_result.lines[0].before_state
    assert result.forecast_result.end_state.month != result.baseline_result.end_state.month
    assert result.savings_energy_eur == result.baseline_energy_cost_eur - result.engine_energy_cost_eur
    with pytest.raises(PlanningSolutionRejected, match="planning_forecast_invalid"):
        da.finish_day_ahead_finance(selected, org_l=(0.0, 0.0), org_t=(0.0, 1.0))


def test_final_selected_solution_audited_before_consumers(snapshot, bonus_rules, monkeypatch):
    model, terms = financial_case(snapshot, bonus_rules, cells=((None,),), exports=("1",))
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    model.objective = cell
    model.optimize()
    first = da.finish_day_ahead_finance(da.finalize_day_ahead_finance(terms, cbc_log=""), org_l=(0.0,), org_t=(-1.0,))
    assert first.forecast[0].battery_discharge_kwh == 0
    model += cell >= 0.006
    model.optimize()
    second = da.finish_day_ahead_finance(da.finalize_day_ahead_finance(terms, cbc_log=""), org_l=(0.0,), org_t=(-1.0,))
    assert second.forecast[0].battery_discharge_kwh == D("0.0015")
    assert second.forecast_result.lines[0].bonus_kwh == 0
    preview_calls, prognosis_writes, plan_writes, ha_writes = [], [], [], []
    override_solution(monkeypatch, model, {cell.idx: -0.001})
    with pytest.raises(PlanningSolutionRejected):
        audited = da.finish_day_ahead_finance(da.finalize_day_ahead_finance(terms, cbc_log=""), org_l=(0.0,), org_t=(-1.0,))
        preview_calls.append(audited)
        prognosis_writes.append(audited)
        plan_writes.append(audited)
        ha_writes.append(audited)
    assert preview_calls == prognosis_writes == plan_writes == ha_writes == []


def test_new_regime_denied_even_debug_and_valid_offline_audit(capture, snapshot, bonus_rules, monkeypatch):
    _, terms = financial_case(snapshot, bonus_rules)
    assert da.finish_day_ahead_finance(da.finalize_day_ahead_finance(terms, cbc_log=""), org_l=(0.0,), org_t=(-1.0,)).parity_complete
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


def post_strategy_consumers():
    """Execute the real calc consumer block offline, without adding a context bypass.

    The capture fixture independently covers construction. Here real source code
    is compiled, not reimplemented, to exercise extraction and preview without
    running any of the later HA/DB/plan writers or supplying calc a live context.
    """
    body = ast.parse(textwrap.dedent(inspect.getsource(da.DaCalc.calc_optimum))).body[0].body
    arrays = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "old_cost_da" for target in node.targets))
    start = max(i for i, node in enumerate(body[:arrays]) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Attribute)
                and node.test.left.attr == "num_solutions") + 1
    end = next(i for i, node in enumerate(body[arrays:], arrays) if isinstance(node, ast.If)
               and isinstance(node.test, ast.Attribute) and node.test.attr == "boiler_present")
    return compile(ast.Module(body=body[start:end], type_ignores=[]), da.__file__, "exec")


@pytest.mark.parametrize("outcome", ["success", "new_residual", "missing_evidence", "infeasible"])
def test_finalization_reextracts_baseline_and_all_consumers(snapshot, bonus_rules, monkeypatch, caplog, outcome):
    """Moving finalization after any array extraction must expose stale device/PV data."""
    model, terms = financial_case(snapshot, bonus_rules)
    device = model.add_var(ub=1, name="continuous_device_flow")
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    cell_charge_on = da._bind_battery_cell_direction(model, dc_to_bat=((device,),), dc_from_bat=((cell,),),
        installation_battery_count=1, interval_count=1)
    model += cell_charge_on[0][0] == 1
    wear_cost = model.add_var(ub=1)
    model += wear_cost == (device + cell) * 0.0025
    cost = model.add_var(lb=-10, ub=10)
    model += cost == terms.energy_cost
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    raw_get, optimize = model.solver.var_get_x, model.optimize
    override_solution(monkeypatch, model, {cell.idx: 1e-13, device.idx: 0})
    calls = []

    def resolve():
        calls.append(True)
        monkeypatch.setattr(model.solver, "var_get_x", raw_get)
        result = optimize()
        # Another feasible degenerate device value, deliberately independent of
        # objective/status. The real new source solution is not modified.
        override_solution(monkeypatch, model, {device.idx: 0.25, wear_cost.idx: 0.000625})
        if outcome == "new_residual":
            override_solution(monkeypatch, model, {cell.idx: 1e-13})
        elif outcome == "missing_evidence":
            monkeypatch.setattr(model, "solver_name", "UNKNOWN")
        elif outcome == "infeasible":
            monkeypatch.setattr(model, "_status", mip.OptimizationStatus.INFEASIBLE)
            monkeypatch.setattr(model.solver, "get_num_solutions", lambda: 0)
        return result

    monkeypatch.setattr(model, "optimize", resolve)
    instance = SimpleNamespace(_debug_capture_vars=True)
    namespace = dict(self=instance, model=model, financial_terms=terms, period_terms=None, cost=cost,
        native={"cbc_log": "Welcome to the CBC MILP Solver Version: 2.10.12"},
        B=1, U=1, EV=1, M=1, solar_num=1, pv_dc_num=[0], hour_fraction=[0.25],
        dc_to_bat=[[device]], dc_from_bat=[[cell]], cell_charge_on=cell_charge_on,
        battery_terminal_cost=mip.xsum([]), preference_cost=mip.xsum([]), other_terminal_cost=mip.xsum([]),
        planning_constraints=None,
        kwh_cycle_cost=[0.01], cycle_cost=[wear_cost],
        ac_to_dc=[[device]], ac_from_dc=[[device]], c_ev=[[device]], c_ma_u=[[device]],
        pv_ac=[[device]], solar_prod=[[0.5]], b_l=[0.0], c_b=[device], c_hp=[device], pl=[0.1], pt=[0.1])
    if outcome != "success":
        with pytest.raises(PlanningSolutionRejected):
            exec(post_strategy_consumers(), da.__dict__, namespace)
        assert calls == [True]
        assert not {"accu_in_sum", "c_ev_sum", "c_ma_sum", "org_l", "org_t", "financial_audit"} & namespace.keys()
        assert not hasattr(instance, "_debug_planning_audit")
        return
    with caplog.at_level(logging.INFO):
        exec(post_strategy_consumers(), da.__dict__, namespace)
    assert calls == [True]
    assert namespace["accu_in_sum"] == namespace["accu_out_sum"] == [0.0625]
    assert namespace["c_ev_sum"] == namespace["c_ma_sum"] == [0.25]
    assert namespace["solar_hour_sum_opt"] == [0.25]
    assert namespace["battery_wear_eur"] == D("0.000625")
    assert namespace["org_l"] == [0.5]
    assert namespace["org_t"] == [0]
    result = namespace["financial_audit"]
    assert result.baseline[0].import_kwh == D("0.5")
    assert result.engine_energy_cost_eur == D("-0.13")
    assert result.baseline_energy_cost_eur == D("0.05")
    assert result.savings_energy_eur == D("0.18")
    assert instance._debug_planning_audit is result
    assert instance._debug_finalized_solution.evidence is result.evidence
    assert instance._debug_financial_terms is instance._debug_finalized_solution.terms
    assert "A8 energiecomponent: -0.13" in caplog.text
    assert "A8 referentie-energiecomponent: 0.05" in caplog.text
    assert "A8 besparing energiecomponent: 0.18" in caplog.text


def pre_strategy_cost_binding():
    """Use the real producer's pre-solve block, never a new offline optimizer."""
    body = ast.parse(textwrap.dedent(inspect.getsource(da.DaCalc.calc_optimum))).body[0].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "financial_terms" for t in node.targets))
    end = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
               and node.value.func.id == "bind_day_ahead_cost")
    return compile(ast.Module(body=body[start:end+1], type_ignores=[]), da.__file__, "exec")


def new_route_cost_namespace(capture, snapshot, bonus_rules, *, spot="0.20"):
    """Reuse the captured producer and its pinned context, without a live context setter."""
    args = capture["finance"]
    context = context_for(snapshot, bonus_rules, args["intervals"], meter=args["meter_id"])
    snap = replace(context.quotes[0].snapshot, bonus=None)
    context = replace(context,
        quotes=tuple(quote(snap, a, b, D(spot)) for a, b in args["intervals"]),
        price_batch=replace(context.price_batch, snapshots=(snap,), items=tuple(
            replace(item, spot=replace(item.spot, price_ex_vat=D(spot)))
            for item in context.price_batch.items)))
    return {**da.__dict__, **capture["locals"], "settlement_context": context}


@pytest.mark.parametrize("spot,expected", [
    ("0.20", (0.16, 0.20)), ("-0.20", (0, 0)),
    ("0.123456789012", (0.0987654312096, 0.123456789012)),
])
def test_actual_producer_builds_period_and_terminal_before_solve(capture, snapshot, bonus_rules,
                                                               monkeypatch, spot, expected):
    """A new-route terminal replaces p_bat once, including the zero-clipped mean."""
    local = new_route_cost_namespace(capture, snapshot, bonus_rules, spot=spot)
    model = capture["model"]
    monkeypatch.setattr(model, "optimize", lambda *a, **k: pytest.fail("prebuild solve"))
    with localcontext() as decimal_context:
        decimal_context.prec = 3
        exec(pre_strategy_cost_binding(), local)
    assert "period_terms" in local and local["period_terms"] is not None
    assert local["period_terms"].issues == ("monthly_credit_omitted_open_or_unknown",)
    assert local["terminal_terms"].policy_version == "export_mean_v1"
    assert local["terminal_terms"].eur_per_ac_kwh == max(D("0"), D(spot))
    expr = model.constrs[-1].expr.expr
    for b, coefficient in enumerate(expected):
        assert expr.get(local["soc"][b][0], 0) == pytest.approx(-coefficient)
        assert expr.get(local["soc"][b][12], 0) == pytest.approx(coefficient)
        for var in (local["soc_mid"][b][0], local["soc_mid"][b][12]):
            assert expr.get(var, 0) == 0
    assert capture["finance"]["context"] is None  # Actual liveflow remains unbound.


@pytest.mark.parametrize("capture", [
    {"battery": {"lower limit": 20, "optimal lower level": 50}},
    {"battery": {"lower limit": 20, "optimal lower level": 50,
        "discharge stages": [{"power": 2000, "efficiency": 0.9}, {"power": 4000, "efficiency": 0.9}]}},
], indirect=True)
def test_actual_producer_terminal_equal_physical_energy_ignores_soc_allocation(capture, snapshot, bonus_rules):
    """A feasible low/mid reallocation must not create EUR at unchanged stored energy."""
    local = new_route_cost_namespace(capture, snapshot, bonus_rules)
    with localcontext() as decimal_context:
        decimal_context.prec = 3
        exec(pre_strategy_cost_binding(), local)
    model, end = capture["model"], local["U"]
    for b in range(local["B"]):
        assert local["lower_limit"][b] == 20 and local["opt_low_level"][b] == 50
        for u in range(end):
            model += local["dc_to_bat"][b][u] == 0
            model += local["dc_from_bat"][b][u] == 0
        model += local["soc_mid"][b][0] == 0
        model += local["soc_mid"][b][end] == 30
    model.verbose = 0
    model.objective = mip.minimize(local["cost"])
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    for b in range(local["B"]):
        assert [v.x for v in local["soc"][b]] == pytest.approx([50] * (end+1))
        assert local["soc_low"][b][0].x == pytest.approx(50)
        assert local["soc_low"][b][end].x == pytest.approx(20)
    assert local["battery_terminal_cost"].x == pytest.approx(0)
    # No wear/device/boiler/penalty cost: the full cost must not contain the
    # fictitious -EUR 10.8 legacy split value (30 * 0.20 * (0.8 + 1)).
    assert sum(v.x for v in local["cycle_cost"] + local["penalty_cost"]) == pytest.approx(0)
    assert local["boiler_temp"][end].x == local["boiler_temp"][0].x
    assert local["period_terms"].credit.x == 0
    assert local["cost"].x == pytest.approx(local["financial_terms"].energy_cost.x)


@pytest.mark.parametrize("capture", [{"battery": {
    "discharge stages": [{"power": 2000, "efficiency": 0.9}, {"power": 4000, "efficiency": 0.9}],
}}], indirect=True)
def test_actual_producer_terminal_multiplies_reference_factors_as_decimal(capture, snapshot, bonus_rules):
    """Cell 0.8 and inverter 0.9 must give exact reference 0.72 even at caller precision 3."""
    local = new_route_cost_namespace(capture, snapshot, bonus_rules, spot="0.123456789012")
    with localcontext() as decimal_context:
        decimal_context.prec = 3
        exec(pre_strategy_cost_binding(), local)
    assert local["eff_bat_to_dc"] == [0.8, 1.0]
    assert local["avg_eff_dc_to_ac"] == [0.9, 0.9]
    assert local["terminal_terms"].reference_efficiency_by_battery == (("0", D("0.72")), ("1", D("0.9")))
    expr = capture["model"].constrs[-1].expr.expr
    for b, coefficient in enumerate((0.08888888808864, 0.1111111101108)):
        assert expr.get(local["soc"][b][0], 0) == pytest.approx(-coefficient)
        assert expr.get(local["soc"][b][12], 0) == pytest.approx(coefficient)


@pytest.mark.parametrize("capture", [{"battery": {
    "bat_to_dc efficiency": 0.123456789012,
    "discharge stages": [{"power": 2000, "efficiency": 0.9}, {"power": 4000, "efficiency": 0.9}],
}}], indirect=True)
def test_actual_producer_terminal_rejects_overprecise_reference_without_quantizing(capture, snapshot, bonus_rules):
    """The exact 0.1111111101108 product exceeds the unchanged twelve-place DTO policy."""
    local = new_route_cost_namespace(capture, snapshot, bonus_rules)
    with localcontext() as decimal_context:
        decimal_context.prec = 3
        with pytest.raises(ValueError, match="reference efficiency: at most 12 decimal places"):
            exec(pre_strategy_cost_binding(), local)


@pytest.mark.parametrize("outcome", ["success", "audit_failure", "model_mutation", "solution_mutation"])
def test_period_consumer_audit_precedes_preview_and_guards_digests(snapshot, monkeypatch, caplog, outcome):
    """Period audit failure or mutation must stop every financial preview and capture."""
    model, terms, periods = financial_period_case(snapshot)
    cost = model.add_var(lb=-10, ub=10)
    da.bind_day_ahead_cost(model, cost=cost, legacy_energy_cost=mip.xsum([]),
                          other_cost=mip.xsum([]), financial_terms=terms, period_terms=periods)
    model.objective = cost
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert hasattr(da, "audit_period_solution")
    audit_period_solution = da.audit_period_solution
    calls = []

    def audit(*args, **kwargs):
        calls.append(True)
        if outcome == "audit_failure":
            raise PlanningSolutionRejected("planning_period_parity_failed")
        result = audit_period_solution(*args, **kwargs)
        if outcome == "model_mutation":
            model.add_var(ub=1)
        elif outcome == "solution_mutation":
            override_solution(monkeypatch, model, {terms.grid_import[0].idx: 0.1})
        return result

    monkeypatch.setattr(da, "audit_period_solution", audit)
    monkeypatch.setattr(model, "optimize", lambda *a, **k: pytest.fail("extra solve"))
    instance = SimpleNamespace(_debug_capture_vars=True)
    namespace = dict(self=instance, model=model, financial_terms=terms, period_terms=periods, cost=cost,
        native={"cbc_log": ""}, B=0, U=2, EV=0, M=0, solar_num=1, pv_dc_num=[],
        hour_fraction=[1.0, 1.0], ac_to_dc=[], ac_from_dc=[], c_ev=[], c_ma_u=[],
        dc_to_bat=[], dc_from_bat=[], cell_charge_on=(), kwh_cycle_cost=[], cycle_cost=[],
        battery_terminal_cost=mip.xsum([]), preference_cost=mip.xsum([]), other_terminal_cost=mip.xsum([]),
        planning_constraints=None,
        pv_ac=[[terms.grid_import[0], terms.grid_import[1]]], solar_prod=[[0.0, 1.0]],
        b_l=[0.0, 0.0], c_b=terms.grid_import, c_hp=terms.grid_import, pl=[0.1, 0.1], pt=[0.1, 0.1])
    with caplog.at_level(logging.INFO):
        if outcome == "success":
            exec(post_strategy_consumers(), da.__dict__, namespace)
        else:
            with pytest.raises(PlanningSolutionRejected, match="planning_period_parity_failed|planning_solution_stale"):
                exec(post_strategy_consumers(), da.__dict__, namespace)
    assert calls == [True]
    if outcome == "success":
        assert instance._debug_period_terms is periods
        assert instance._debug_period_audit.engine_closed_credit_eur == D("0.1186")
        assert "A9" in caplog.text and "conservatief" in caplog.text
    else:
        assert not hasattr(instance, "_debug_period_audit")
        assert "A8 energiecomponent" not in caplog.text


def test_period_audit_remains_separate_from_a8_public_interfaces(snapshot):
    _, terms, _, selected, audit = audited_period_case(snapshot)
    assert tuple(inspect.signature(da.finalize_day_ahead_finance).parameters) == ("terms", "cbc_log")
    assert tuple(inspect.signature(da.finish_day_ahead_finance).parameters) == ("selected", "org_l", "org_t")
    for value in (terms, selected, audit):
        assert "period_terms" not in value.__dataclass_fields__
        assert "period_audit" not in value.__dataclass_fields__


def test_period_debug_is_cleared_even_when_preflight_denies(capture):
    """A denied later calculation must not leak an earlier successful period audit."""
    instance = capture["instance"]
    instance._debug_capture_vars = True
    names = ("_debug_financial_terms", "_debug_planning_audit", "_debug_finalized_solution",
             "_debug_period_terms", "_debug_period_audit")
    for name in names:
        setattr(instance, name, object())
    instance.config.prices = pricing_config()
    for attr in ("_financial_pricing", "_financial_policy", "_financial_plan_path"):
        if hasattr(instance, attr):
            delattr(instance, attr)
    with pytest.raises(FinancialPlanBlocked):
        instance.calc_optimum(_start_dt=dt.datetime.fromtimestamp(ts()))
    assert all(getattr(instance, name) is None for name in names)


@pytest.mark.parametrize("capture", [{}, {"start": ts(minute=7) + 30}, {"hourly": True}], indirect=True)
@pytest.mark.parametrize("financial", [False, True])
@pytest.mark.parametrize("charge,discharge", [(1, 1), (1.25, 0), (0, 1), (0, 0), (0.37, 0), (0, 0.63)])
def test_real_producer_cell_directions_are_exclusive_and_continuous(
        capture, snapshot, bonus_rules, financial, charge, discharge):
    """Original producer flows, including partial powers, cannot use both cell directions."""
    model, local, args = capture["model"], capture["locals"], capture["finance"]
    if financial:
        da.bind_day_ahead_finance(model, **{**args,
            "context": context_for(snapshot, bonus_rules, args["intervals"], meter=args["meter_id"]),
            "origins": ("unknown",) * local["U"], "source_policy": None})
    assert local["dc_to_bat"][0][0].ub == local["dc_from_bat"][0][0].ub == 8
    assert len(local["cell_charge_on"]) == 2
    for b in range(local["B"]):
        for i in range(local["U"]):
            model += local["dc_to_bat"][b][i] == (charge if (b, i) == (0, 0) else 0)
            model += local["dc_from_bat"][b][i] == (discharge if (b, i) == (0, 0) else 0)
    model.objective = mip.xsum([])
    model.verbose = 0
    assert model.optimize() == (mip.OptimizationStatus.INFEASIBLE if charge and discharge
                                else mip.OptimizationStatus.OPTIMAL)
    if not (charge and discharge):
        # Feasibility proof, not strict acceptance of a constant-objective incumbent.
        # Tiny disabled-direction residues are rejected separately by the final audit tests.
        model.check_optimization_results()
        assert local["dc_to_bat"][0][0].x == pytest.approx(charge, abs=1e-9)
        assert local["dc_from_bat"][0][0].x == pytest.approx(discharge, abs=1e-9)


@pytest.mark.parametrize("mode", ["pass_through", "pv_charge_and_export", "opposite_batteries"])
def test_cell_direction_is_independent_of_inverter_and_other_batteries(capture, snapshot, bonus_rules, mode):
    """An AC export direction must not prohibit DC-PV charging, or another battery's discharge."""
    model, local, args = capture["model"], capture["locals"], capture["finance"]
    terms = da.bind_day_ahead_finance(model, **{**args,
        "context": context_for(snapshot, bonus_rules, args["intervals"], meter=args["meter_id"]),
        "origins": ("verified_pv",) * local["U"], "source_policy": POLICY})
    for b in range(2):
        for i in range(local["U"]):
            model += local["dc_to_bat"][b][i] == (1 if b == i == 0 and mode != "pass_through" else 0)
            model += local["dc_from_bat"][b][i] == (1 if b == 1 and i == 0 and mode == "opposite_batteries" else 0)
    model.objective = terms.energy_cost
    model.verbose = 0
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    selected = da.finalize_day_ahead_finance(terms, cbc_log="")
    da._audit_battery_cell_direction(model, dc_to_bat=local["dc_to_bat"], dc_from_bat=local["dc_from_bat"],
        cell_charge_on=local["cell_charge_on"], installation_battery_count=2, interval_count=local["U"])
    audit = da.finish_day_ahead_finance(selected, org_l=(0.0,) * local["U"], org_t=(-1.0,) * local["U"])
    assert local["ac_from_dc"][0][0].x > 0
    assert audit.forecast[0].battery_discharge_kwh == (D("0.25") if mode == "opposite_batteries" else 0)
    assert audit.forecast_result.lines[0].bonus_kwh == (0 if mode == "opposite_batteries" else audit.forecast[0].export_kwh)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "foreign", "negative", "nonfinite", "integer", "axis"])
def test_cell_direction_binding_rejects_incomplete_or_invalid_handles_before_mutation(monkeypatch, bad):
    model = mip.Model(solver_name=mip.CBC)
    charge, discharge = model.add_var(ub=3), model.add_var(ub=7)
    matrix = ((charge,),)
    if bad == "missing":
        matrix = ()
    elif bad == "duplicate":
        matrix = ((discharge,),)
    elif bad == "foreign":
        other = mip.Model(solver_name=mip.CBC)
        matrix = ((other.add_var(ub=3),),)
    elif bad == "negative":
        charge.lb = -1
    elif bad == "nonfinite":
        # Native CBC converts an infinite bound to a finite maximum; exercise bad readback explicitly.
        get_upper = model.solver.var_get_ub
        monkeypatch.setattr(model.solver, "var_get_ub", lambda var: float("inf") if var is charge else get_upper(var))
    elif bad == "integer":
        charge.var_type = mip.INTEGER
    before = (model.num_cols, model.num_rows)
    with pytest.raises(OptimizationInputError, match="battery_direction_binding_invalid"):
        da._bind_battery_cell_direction(model, dc_to_bat=matrix, dc_from_bat=((discharge,),),
            installation_battery_count=1, interval_count=0 if bad == "axis" else 1)
    assert (model.num_cols, model.num_rows) == before


@pytest.mark.parametrize("charge,discharge", [(0, 0), (0, 1.2), (0.7, 0), (0, 2.9)])
def test_cell_direction_zero_and_unequal_upper_bounds(charge, discharge):
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    assert da._bind_battery_cell_direction(model, dc_to_bat=(), dc_from_bat=(),
        installation_battery_count=0, interval_count=3) == ()
    c, d = model.add_var(ub=0.7), model.add_var(ub=2.9)
    y = da._bind_battery_cell_direction(model, dc_to_bat=((c,),), dc_from_bat=((d,),),
                                      installation_battery_count=1, interval_count=1)
    model += c == charge
    model += d == discharge
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    da._audit_battery_cell_direction(model, dc_to_bat=((c,),), dc_from_bat=((d,),), cell_charge_on=y,
                                    installation_battery_count=1, interval_count=1)
    y[0][0].lb = y[0][0].ub = int(charge > 0)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    da._audit_battery_cell_direction(model, dc_to_bat=((c,),), dc_from_bat=((d,),), cell_charge_on=y,
                                    installation_battery_count=1, interval_count=1)
    d.ub = 0
    if discharge:
        assert model.optimize() == mip.OptimizationStatus.INFEASIBLE


def test_cell_direction_audit_rejects_native_model_check_failure(monkeypatch):
    """A valid direction is not permission to accept a model rejected by its native checker."""
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    c, d = model.add_var(ub=0), model.add_var(ub=0)
    y = da._bind_battery_cell_direction(model, dc_to_bat=((c,),), dc_from_bat=((d,),),
                                      installation_battery_count=1, interval_count=1)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL

    def invalid():
        raise mip.InfeasibleSolution("Controlled native model check failure")

    monkeypatch.setattr(model, "check_optimization_results", invalid)
    with pytest.raises(PlanningSolutionRejected, match="battery_direction_solution_invalid"):
        da._audit_battery_cell_direction(model, dc_to_bat=((c,),), dc_from_bat=((d,),), cell_charge_on=y,
                                        installation_battery_count=1, interval_count=1)


@pytest.mark.parametrize("financial", [False, True])
@pytest.mark.parametrize("fault", ["off_charge", "off_discharge", "negative", "nonfinite", "wrong_y", "fractional_y"])
def test_final_cell_direction_audit_precedes_every_consumer(capture, snapshot, bonus_rules, monkeypatch, financial, fault):
    """Even a tiny disabled-direction value must fail before arrays or an accepted debug selection."""
    model, local = capture["model"], capture["locals"]
    namespace = new_route_cost_namespace(capture, snapshot, bonus_rules) if financial else {**da.__dict__, **local}
    exec(pre_strategy_cost_binding(), namespace)
    for key in ("dc_to_bat", "dc_from_bat"):
        for row in local[key]:
            for var in row:
                model += var == 0
    y = local["cell_charge_on"][0][0]
    model += y == (1 if fault == "off_discharge" else 0)
    model.verbose = 0
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    # Inject only at the direction audit boundary, after genuine financial finalization.
    audit = da._audit_battery_cell_direction

    def inspect_direction(*args, **kwargs):
        if fault in ("off_charge", "negative", "nonfinite"):
            var = local["dc_to_bat"][0][0]
            value = {"off_charge": 1e-12, "negative": -1e-12, "nonfinite": float("nan")}[fault]
        elif fault == "off_discharge":
            var, value = local["dc_from_bat"][0][0], 1e-12
        else:
            var, value = y, 2 if fault == "wrong_y" else 0.5
        override_solution(monkeypatch, model, {var.idx: value})
        return audit(*args, **kwargs)

    namespace.update(_audit_battery_cell_direction=inspect_direction, native={"cbc_log": ""})
    monkeypatch.setattr(model, "optimize", lambda: pytest.fail("Direction audit must not solve"))
    with pytest.raises(PlanningSolutionRejected, match="battery_direction_solution_invalid"):
        exec(post_strategy_consumers(), namespace)
    assert not {"accu_in_sum", "org_l", "org_t", "financial_audit"} & namespace.keys()
    assert not getattr(capture["instance"], "_debug_finalized_solution", None)
