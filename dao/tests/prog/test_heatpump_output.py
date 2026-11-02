"""Zero remaining heat must not suppress an allowed curve adjustment.

Real producer/solve/acceptance/output flow, with synthetic inputs and registered
HA/DB sinks only. This is not financial E2E or hardware qualification; the early
on/off temperature write and general notification/presence safety are not covered.
"""

import datetime as dt
import logging
import socket
import sys
from types import SimpleNamespace

import mip
import pandas as pd
import pytest

from dao.prog import day_ahead as da
from dao.prog.config.models.devices.heating import HeatingDisabled, HeatingEnabled
from dao.prog.config.models.grid import GridConfig
from dao.prog.config.models.pricing import PricingConfig
from dao.prog.fastctrl.publication import invalidate_plan, publication_lock, read_publication
from dao.prog.settlement.publication import FinancialPlanBlocked
from dao.tests.prog.test_optimizer_physical_acceptance import ACTUATORS
from dao.tests.settlement.test_optimization_producer import capture as capture
from dao.tests.settlement.test_publication import pricing_config

CURVE_ENTITY = "input_number.synthetic_hp_curve"


class OutputBoundaryReached(BaseException):
    """Stop after HP output, without letting device Exception handlers swallow it."""


@pytest.fixture
def hp_output(capture, monkeypatch, tmp_path, caplog):
    """Reuse hermetic inventory/input boundaries, never a DaBase constructor."""
    instance = capture["instance"]
    start = capture["locals"]["start_dt"]
    instance.battery_options = []
    instance.grid = GridConfig(max_power=17)
    instance._solver_threads = 1
    instance.config.prices = PricingConfig.model_validate(
        pricing_config().model_dump(exclude={"settlement"}))
    for attr in ("_financial_pricing", "_financial_policy", "_financial_plan_path"):
        delattr(instance, attr)
    instance.notification_entity = None
    instance.notification_berekening = False
    monkeypatch.setattr(instance, "notify", da.DaCalc.notify.__get__(instance))
    instance.meteo = SimpleNamespace(calc_graaddagen=lambda **kwargs: 1.0)

    class FrozenDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return start if tz is None else start.astimezone(tz)

    monkeypatch.setattr(da, "dt", SimpleNamespace(
        datetime=FrozenDateTime, timedelta=dt.timedelta, date=dt.date))
    monkeypatch.setattr(da, "time", SimpleNamespace(
        time=lambda: start.timestamp(), perf_counter=da.time.perf_counter))
    plan_path = tmp_path / "fast_plan.json"
    monkeypatch.setattr(da, "FAST_PLAN_FILE", str(plan_path))
    prices = pd.DataFrame({
        "time": capture["locals"]["tijd"], "da_ex": [0.25] * 12,
        "da_cons": [0.25] * 12, "da_prod": [0.0] * 12, "da_bonus": [0.0] * 12,
    })
    monkeypatch.setattr(da, "Report", lambda *args: SimpleNamespace(
        get_price_data=lambda *a, **k: prices.copy(),
        get_heatpump_run_hours=lambda entity: -1))

    def run(*, adjustment="heating curve", mode="allowed", produced=4,
            curve_entity=True, debug=False, blocked=False):
        instance.debug = debug
        # Previous-run state must not select a different regulation at zero heat.
        instance.hp_adjustment = "stale-mode"
        instance.hp_control_allowed = True
        if mode == "absent":
            instance.heating_options = None
        elif mode == "disabled":
            instance.heating_options = HeatingDisabled()
        else:
            instance.heating_options = HeatingEnabled.model_validate({
                "adjustment": adjustment, "degree days factor": 4,
                "entity hp enabled": "input_boolean.synthetic_hp_enabled",
                "entity hp heat produced": "sensor.synthetic_hp_heat",
                "entity hp heat demand": "sensor.synthetic_hp_demand",
                "entity hp switch": "switch.synthetic_hp",
                "entity hp power": "input_number.synthetic_hp_power",
                "entity adjust heating curve": CURVE_ENTITY if curve_entity else None,
                "adjustment factor": 0.04,
                "stages": [{"max_power": 1000, "cop": 4}],
            })
        states = {
            "input_boolean.synthetic_hp_enabled": "off" if mode == "switched-off" else "on",
            "sensor.synthetic_hp_heat": str(produced),
            "sensor.synthetic_hp_demand": "max",
            CURVE_ENTITY: "2.0",
            "switch.synthetic_hp": "off",
            "input_number.synthetic_hp_power": "1.0",
        }
        found = {"events": [], "writes": [], "saved": [], "reads": [], "solves": [],
                 "native_checks": [], "presence": [], "instance": instance}
        events = found["events"]
        bind_cost, optimize = da.bind_day_ahead_cost, mip.Model.optimize
        native_check, audit = mip.Model.check_optimization_results, da._audit_battery_cell_direction
        check_output = instance._check_financial_output
        export = da.DaCalc.export_fast_plan.__get__(instance)

        def get_state(entity_id):
            assert entity_id in states, f"unexpected synthetic HA read: {entity_id}"
            found["reads"].append(entity_id)
            return SimpleNamespace(state=states[entity_id])

        def sink(name):
            def write(*args, **kwargs):
                found["writes"].append((name, args, kwargs))
                events.append(name)
            return write

        def forbidden(*args, **kwargs):
            found["writes"].append(("forbidden_io", args, kwargs))
            pytest.fail("synthetic HP output attempted external I/O")

        def cost(model, **kwargs):
            bind_cost(model, **kwargs)
            found["local"] = dict(sys._getframe(1).f_locals)
            found["model"] = model
            events.append("cost_bound")

        def solve(model, *args, **kwargs):
            found["solves"].append(model)
            assert found["solves"] == [found["model"]], "unexpected extra solve"
            assert model.solver_name == mip.CBC and model.sense == mip.MINIMIZE
            status = optimize(model, *args, **kwargs)
            assert status == mip.OptimizationStatus.OPTIMAL and model.num_solutions == 1
            events.append("solved")
            return status

        def check(model):
            found["native_checks"].append(model)
            events.append("native_check")
            return native_check(model)

        def accept(model, **kwargs):
            result = audit(model, **kwargs)
            events.append("accepted")
            if blocked:
                # Invalidate after preflight/solve: the real final gate must see it.
                with publication_lock(plan_path) as lease:
                    invalidate_plan(lease=lease, now_ts=int(start.timestamp()),
                                    reason="synthetic_b6_block")
                events.append("invalidated")
            return result

        def gate(**kwargs):
            events.append("gate")
            try:
                check_output(**kwargs)
            except FinancialPlanBlocked:
                events.append("gate_rejected")
                raise
            events.append("gate_allowed")

        def save(**kwargs):
            found["saved"].append(kwargs)
            events.append("save_df")  # Legacy prognosis precedes the HA gate; fake DB only.

        def finish(**kwargs):
            # Keep the real debug/no-battery export guard; no plan writer may run.
            export(**kwargs, path=str(plan_path))
            events.append("output_boundary")
            raise OutputBoundaryReached

        with monkeypatch.context() as spies:
            spies.setattr(instance, "get_state", get_state)
            spies.setattr(instance, "get_float", da.DaCalc.get_float.__get__(instance))
            spies.setattr(instance, "baseload_service", lambda: SimpleNamespace(
                record_presence=lambda: found["presence"].append(True)))
            for name in ACTUATORS + ("set_state",):
                spies.setattr(instance, name, sink(name))
            # No grid entities configured: retain the real optional-entity guards.
            for name in ("set_entity_state", "set_entity_value"):
                spies.setattr(instance, name, getattr(da.DaCalc, name).__get__(instance))
            spies.setattr(instance, "save_df", save)
            spies.setattr(instance, "export_fast_plan", finish)
            spies.setattr(instance, "_check_financial_output", gate)
            spies.setattr(da, "write_plan", forbidden)
            spies.setattr(socket, "getaddrinfo", forbidden)
            spies.setattr(socket.socket, "connect", forbidden)
            spies.setattr(socket.socket, "connect_ex", forbidden)
            spies.setattr(da, "bind_day_ahead_cost", cost)
            spies.setattr(mip.Model, "optimize", solve)
            spies.setattr(mip.Model, "check_optimization_results", check)
            spies.setattr(da, "_audit_battery_cell_direction", accept)
            with pytest.raises(FinancialPlanBlocked if blocked else OutputBoundaryReached) as exc:
                instance.calc_optimum()  # A start override would silently force debug.
            found["exception"] = exc.value

        assert instance.debug is debug and instance._financial_pricing.settlement is None
        assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
        assert found["solves"] == found["native_checks"] == [found["model"]]
        assert events[:4] == ["cost_bound", "solved", "native_check", "accepted"]
        assert found["presence"] == ([] if debug else [True])
        assert len(found["saved"]) == (0 if debug else 1)
        assert found["local"]["pl"][0] == found["local"]["p_avg"] == 0.25
        if blocked:
            assert str(found["exception"]) == "synthetic_b6_block"
            assert read_publication(plan_path).status == "blocked"
            assert events[-2:] == ["gate", "gate_rejected"]
        else:
            assert not plan_path.exists()
            assert events[-1] == "output_boundary"
        assert not list(tmp_path.glob(".fast_plan_*.tmp"))
        logging.info("B6 mode=%s adjustment=%s produced=%s debug=%s blocked=%s: events=%s; writes=%s",
                     mode, adjustment, produced, debug, blocked, events, found["writes"])
        return found

    return run


def assert_compressor_idle(found):
    assert found["instance"].hp_enabled is False
    assert [flag.x for flag in found["local"]["hp_on"]] == [0] * 12
    assert [var.x for var in found["local"]["c_hp"]] == [0] * 12


@pytest.mark.parametrize("adjustment", ["on/off", "power", "heating curve"])
@pytest.mark.parametrize("produced", [4, 6], ids=["exactly-met", "exceeded"])
def test_curve_adjustment_with_zero_remaining_heat(hp_output, adjustment, produced):
    """Conflating permission with remaining heat loses the real +2 K -> +1.6 K write."""
    found = hp_output(adjustment=adjustment, produced=produced)
    assert_compressor_idle(found)
    assert found["local"]["heat_needed"] == 0
    assert found["writes"] == [("set_value", (CURVE_ENTITY, 1.6), {})]
    assert found["events"].index("gate_allowed") < found["events"].index("set_value")
    assert found["instance"].hp_control_allowed is True
    assert found["instance"].hp_adjustment == adjustment


@pytest.mark.parametrize("mode", ["absent", "disabled", "switched-off"])
def test_curve_adjustment_requires_control_permission(hp_output, mode):
    """A presence-only guard would write a curve despite an explicit off switch."""
    found = hp_output(mode=mode)
    assert_compressor_idle(found)
    assert found["instance"].hp_control_allowed is False
    assert found["writes"] == []
    assert CURVE_ENTITY not in found["reads"]


@pytest.mark.parametrize("adjustment", ["on/off", "power", "heating curve"])
def test_zero_remaining_heat_without_curve_entity_writes_nothing(hp_output, adjustment):
    found = hp_output(adjustment=adjustment, curve_entity=False)
    assert_compressor_idle(found)
    assert found["instance"].hp_control_allowed is True
    assert found["writes"] == []
    assert CURVE_ENTITY not in found["reads"]


@pytest.mark.parametrize("adjustment", ["on/off", "power", "heating curve"])
def test_debug_curve_adjustment_has_no_writes(hp_output, caplog, adjustment):
    with caplog.at_level(logging.INFO):
        found = hp_output(adjustment=adjustment, debug=True)
    assert_compressor_idle(found)
    assert found["writes"] == found["saved"] == []
    assert CURVE_ENTITY in found["reads"]
    assert "Aanpassing stooklijn zou zijn: 1.60" in caplog.text
    assert "gate" not in found["events"]


@pytest.mark.parametrize("adjustment", ["on/off", "power", "heating curve"])
def test_blocked_curve_adjustment_has_no_writes(hp_output, adjustment):
    found = hp_output(adjustment=adjustment, blocked=True)
    assert_compressor_idle(found)
    assert found["instance"].hp_control_allowed is True
    assert found["writes"] == []
    assert CURVE_ENTITY not in found["reads"]


@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_curve_adjustment_preserves_active_compressor_output(hp_output, adjustment):
    """Separating curve output must not suppress existing switch/power commands."""
    found = hp_output(adjustment=adjustment, produced=0)
    local = found["local"]
    assert found["instance"].hp_control_allowed is True
    assert found["instance"].hp_enabled is True
    assert sum(var.x for var in local["h_hp"]) == pytest.approx(4)
    assert [flag.x for flag in local["hp_on"]] == [1] * 4 + [0] * 8
    assert [var.x for var in local["c_hp"]] == pytest.approx([0.25] * 4 + [0] * 8)
    assert found["writes"] == [
        ("turn_on", ("switch.synthetic_hp",), {}),
        ("set_value", ("input_number.synthetic_hp_power", 1.0), {}),
        ("set_value", (CURVE_ENTITY, 1.6), {}),
    ]
    assert found["events"].index("gate_allowed") < found["events"].index("set_value")
