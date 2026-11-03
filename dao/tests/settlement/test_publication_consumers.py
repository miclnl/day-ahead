"""Actual planner/runner boundaries, synthetic DBs and recorded HA calls only."""

import datetime as dt
import logging
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine

import dao.prog.day_ahead as day_ahead
from dao.prog.config.models.base import FlexFloat
from dao.prog.config.models.devices.heating import HeatingEnabled
from dao.prog.config.models.fastcontrol import FastControlConfig
from dao.prog.fastctrl.publication import invalidate_plan, publication_lock, read_publication
from dao.prog.settlement.publication import FinancialPlanBlocked
from dao.tests.settlement.test_optimization_producer import capture as capture
from dao.tests.settlement.test_publication import pricing_config

GATE = 1798758000


@pytest.fixture
def planner(tmp_path, monkeypatch):
    instance = day_ahead.DaCalc.__new__(day_ahead.DaCalc)
    instance.debug = False
    instance.interval_s = 3600
    instance.interval = "1hour"
    instance.log_level = logging.INFO
    instance.file_name = "synthetic-unused-options"
    instance.config = SimpleNamespace(prices=pricing_config(), fast_control=FastControlConfig())
    engine = create_engine(f"sqlite:///{tmp_path / 'planner.db'}")
    instance.db_da = SimpleNamespace(engine=engine)
    instance.ha_calls = []
    # Register every HA surface, including the early average-temperature write.
    for name in ("set_value", "call_service", "turn_on", "turn_off", "set_entity_value",
                 "set_entity_state", "set_entity_option", "select_option"):
        monkeypatch.setattr(instance, name, lambda *a, _name=name, **kw: instance.ha_calls.append((_name, a, kw)))
    path = tmp_path / "fast_plan.json"
    monkeypatch.setattr(day_ahead, "FAST_PLAN_FILE", str(path))
    clock = [GATE]
    monkeypatch.setattr(day_ahead.time, "time", lambda: clock[0])
    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(clock[0], tz)
    monkeypatch.setattr(day_ahead.dt, "datetime", Clock)
    yield instance, path, clock
    engine.dispose()


@pytest.mark.parametrize("mode,before_contract", [("off", False), ("shadow", False), ("active", True)])
@pytest.mark.parametrize("status", ["draft", "confirmed"])
def test_calc_preflight_blocks_before_report_or_any_ha_write(planner, monkeypatch, mode, before_contract, status):
    instance, path, clock = planner
    if mode == "active" and status == "draft":
        with pytest.raises(ValueError, match="draft"):
            pricing_config(mode=mode, status=status)
        assert instance.ha_calls == []
        return
    instance.config.prices = pricing_config(mode=mode, status=status)
    # Off/shadow at the gate still deny before Report/HA. Before the contract,
    # active uses legacy too: an existing invalidation must deny just as early.
    # Active/confirmed at/after the gate has planning rights, tested separately.
    if before_contract:
        clock[0] = GATE - 3600
        with publication_lock(path) as lease:
            invalidate_plan(lease=lease, now_ts=clock[0], reason="registered_financial_inputs_changed")
    report_calls = []
    def report(*args):
        report_calls.append(args)
        raise RuntimeError("price fetch must not own invalidation")
    monkeypatch.setattr(day_ahead, "Report", report)
    with pytest.raises(FinancialPlanBlocked):
        instance.calc_optimum()
    assert read_publication(path).status == "blocked"
    assert instance.ha_calls == report_calls == []


@pytest.mark.parametrize("capture", [{"active": True, "solve": True}], indirect=True)
def test_calc_active_confirmed_reaches_planning_but_never_ha_or_ready_publication(capture):
    # Active/confirmed no longer denies before Report construction/planning.
    # It constructs the legacy object for non-price reads, skips its price API,
    # solves on the real service context and still denies output before any HA write.
    assert "context" in capture, capture.get("error")
    assert capture["prepare_calls"] == [(GATE + 12 * 3600, GATE + 15 * 3600)]
    assert capture["report_constructions"] == [("synthetic-unused-options",)]
    assert capture["report_reads"] == []
    assert str(capture.get("error")) == "actuation_gates_unproven"
    assert capture["output_bases"] == [capture["context"].basis]
    assert capture["ha_writes"] == capture["ready_publications"] == []


@pytest.mark.parametrize("end,blocked", [(GATE, False), (GATE + 1, True)])
def test_legacy_preflight_accepts_only_horizon_ending_at_or_before_gate(planner, end, blocked, caplog):
    instance, path, clock = planner
    clock[0] = GATE - 3600
    if blocked:
        with pytest.raises(FinancialPlanBlocked):
            instance._financial_preflight(planning_start_ts=GATE - 3600, horizon_end_ts=end)
        assert read_publication(path).status == "blocked"
        assert "horizon" in caplog.text
    else:
        instance._financial_preflight(planning_start_ts=GATE - 3600, horizon_end_ts=end)
        assert not path.exists()
    assert instance.ha_calls == []


def test_output_permission_expires_when_calculation_crosses_date(planner):
    instance, path, clock = planner
    clock[0] = GATE - 1
    instance._financial_preflight(planning_start_ts=GATE - 3600, horizon_end_ts=GATE)
    instance._check_financial_output(basis=None, now_ts=GATE - 1)
    with pytest.raises(FinancialPlanBlocked):
        instance._check_financial_output(basis=None, now_ts=GATE)
    assert read_publication(path).status == "blocked"
    assert instance.ha_calls == []


def test_planner_does_not_treat_an_unreadable_prices_property_as_legacy(planner):
    instance, _, _ = planner
    class BrokenConfig:
        @property
        def prices(self):
            raise AttributeError("pricing load failed")
    instance.config = BrokenConfig()
    with pytest.raises((ValueError, AttributeError)):
        instance._financial_preflight(planning_start_ts=GATE, horizon_end_ts=GATE + 3600)


def test_debug_preflight_cannot_mutate_the_operational_artifact(planner):
    instance, path, _ = planner
    instance.config.prices = pricing_config(mode="shadow")
    path.write_text("operator artifact")
    instance.debug = True
    with pytest.raises(FinancialPlanBlocked):
        instance.calc_optimum()
    assert path.read_text() == "operator artifact"
    assert not path.with_name(path.name + ".lock").exists()
    assert instance.ha_calls == []


@pytest.fixture
def solving_planner(planner, monkeypatch):
    """Run the real solve without DaBase construction, live sources or HA."""
    instance, path, clock = planner
    clock[0] = GATE - 12 * 3600
    instance.steps_day = 24
    instance.interval_name = "uur"
    instance.config.baseload = [0.2] * 24
    instance.config.max_gap = FlexFloat(value=0.005)
    instance.config.graphics = SimpleNamespace(style="default", battery_balance=False)
    instance.use_calc_baseload = False
    instance.battery_options = []
    instance.ev_options = []
    instance.solar = []
    instance.machines = []
    instance.boiler_options = None
    instance.heating_options = None
    instance.salderen = False
    instance.grid_max_power = 17.0
    instance.grid = SimpleNamespace()
    instance.strategy = "minimize cost"
    instance.notification_entity = None
    instance.notification_berekening = False
    instance.prices_options = instance.config.prices
    instance.time_zone = "Europe/Amsterdam"
    instance.baseload_service = lambda: SimpleNamespace(record_presence=lambda: None)
    instance.save_df = lambda **kwargs: None  # Isolated forecast-storage boundary.
    start = clock[0]
    stamps = [start + i * 3600 for i in range(12)]
    dates = [dt.datetime.fromtimestamp(at) for at in stamps]
    prices = pd.DataFrame({"time": dates, "da_ex": [0.1] * 12,
                           "da_cons": [0.2] * 12, "da_prod": [0.1] * 12})
    forecast = pd.DataFrame({"time": stamps, "tijd": dates,
                             "glob_rad": [0.0] * 12, "temp": [5.0] * 12})
    monkeypatch.setattr(day_ahead, "Report", lambda *a: SimpleNamespace(get_price_data=lambda *a, **k: prices.copy()))
    instance.db_da.get_prognose_data = lambda **k: forecast.copy()
    return instance, path, clock


def test_real_solver_crossing_gate_aborts_outside_device_catches(solving_planner, monkeypatch):
    instance, path, clock = solving_planner
    original = day_ahead.Model.optimize
    solves = []
    def optimize(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        solves.append(result)
        clock[0] = GATE
        return result
    monkeypatch.setattr(day_ahead.Model, "optimize", optimize)
    with pytest.raises(FinancialPlanBlocked):
        instance.calc_optimum()
    assert solves  # A preflight-only helper cannot satisfy this test.
    assert instance.ha_calls == []
    assert read_publication(path).status == "blocked"


def test_real_heatpump_build_checks_gate_before_early_average_temp_write(solving_planner):
    instance, path, clock = solving_planner
    instance.heating_options = HeatingEnabled(adjustment="on/off", entity_avg_outside_temp="input_number.avg_temp")
    temperature_reads = []
    def temperature(**kwargs):
        temperature_reads.append(kwargs)
        clock[0] = GATE
        return 5.0
    instance.meteo = SimpleNamespace(calc_graaddagen=lambda **k: 3.0, get_avg_temperature=temperature)
    with pytest.raises(FinancialPlanBlocked):
        instance.calc_optimum()
    assert temperature_reads == [{}]
    assert instance.ha_calls == []
    assert read_publication(path).status == "blocked"


@pytest.mark.parametrize("configured", [False, True])
def test_blocked_artifact_before_contract_date_does_not_allow_direct_writes(solving_planner, configured):
    instance, path, clock = solving_planner
    if not configured:
        instance.config.prices = None
    with publication_lock(path) as lease:
        invalidate_plan(lease=lease, now_ts=clock[0], reason="registered_financial_inputs_changed")
    error = None
    try:
        instance.calc_optimum()
    except Exception as exc:
        error = exc
    assert instance.ha_calls == []
    assert isinstance(error, FinancialPlanBlocked)
    assert read_publication(path).status == "blocked"
