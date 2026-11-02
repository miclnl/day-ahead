"""Real producer acceptance, not cost-binding capture or a replayed consumer block.

Legacy pricing deliberately leaves the financial harddeny intact. These offline
power/curve and boiler witnesses do not prove on/off, financial E2E or hardware
actuation safety; notifications/presence before solve are separately isolated.
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
from dao.prog.config.models.devices.boiler import BoilerEnabled
from dao.prog.config.models.devices.heating import HeatingEnabled
from dao.prog.config.models.pricing import PricingConfig
from dao.prog.settlement.optimization_audit import PlanningSolutionRejected
from dao.tests.settlement.test_optimization_producer import capture as capture
from dao.tests.settlement.test_publication import pricing_config

DEBUG_RESULTS = (
    "_debug_financial_terms", "_debug_planning_audit", "_debug_finalized_solution",
    "_debug_period_terms", "_debug_period_audit", "_debug_planning_economics",
)
ACTUATORS = (
    "set_value", "call_service", "turn_on", "turn_off", "set_entity_value",
    "set_entity_state", "set_entity_option", "select_option",
)


class OutputReached(Exception):
    """Stop a healthy control at an existing, recorded output boundary."""


@pytest.fixture
def physical_producer(capture, monkeypatch, tmp_path):
    """Reuse only the synthetic inventory; the next calc builds/solves/audits for real."""
    instance = capture["instance"]
    start = capture["locals"]["start_dt"]
    instance.battery_options = []
    instance._solver_threads = 1
    instance._debug_capture_vars = True
    instance.config.prices = PricingConfig.model_validate(
        pricing_config().model_dump(exclude={"settlement"}))
    for attr in ("_financial_pricing", "_financial_policy", "_financial_plan_path"):
        delattr(instance, attr)
    instance.notification_entity = None
    instance.notification_berekening = False
    # Use the real notification method with explicitly absent configuration.
    monkeypatch.setattr(instance, "notify", da.DaCalc.notify.__get__(instance))
    instance.meteo = SimpleNamespace(calc_graaddagen=lambda **kwargs: 1.0)

    class FrozenDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return start if tz is None else start.astimezone(tz)

    # Replace the producer's module reference, not datetime globally (pandas).
    monkeypatch.setattr(da, "dt", SimpleNamespace(
        datetime=FrozenDateTime, timedelta=dt.timedelta, date=dt.date))
    monkeypatch.setattr(da, "time", SimpleNamespace(
        time=lambda: start.timestamp(), perf_counter=da.time.perf_counter))
    plan_path = tmp_path / "fast_plan.json"
    monkeypatch.setattr(da, "FAST_PLAN_FILE", str(plan_path))

    def run(kind, *, debug, adjustment="power", corrupt=False):
        instance.debug = debug
        instance.interval = "1hour" if kind == "hp" else "15min"
        instance.interval_s = 3600 if kind == "hp" else 900
        instance.steps_day = 24 if kind == "hp" else 96
        instance.interval_name = "uur" if kind == "hp" else "kwartier"
        tariffs = [0.10] * 7 + [0.60] if kind == "hp" else [0.10] * 12
        stamps = [int(start.timestamp()) + instance.interval_s * u
                  for u in range(len(tariffs))]
        dates = [dt.datetime.fromtimestamp(at) for at in stamps]
        prices = pd.DataFrame({
            "time": dates, "da_ex": tariffs, "da_cons": tariffs,
            "da_prod": [0.0] * len(tariffs), "da_bonus": [0.0] * len(tariffs),
        })
        forecast = pd.DataFrame({
            "time": stamps, "tijd": dates,
            "glob_rad": [0.0] * len(tariffs), "temp": [5.0] * len(tariffs),
        })
        instance.heating_options = HeatingEnabled.model_validate({
            "adjustment": adjustment, "degree days factor": 30, "min run length": 5,
            "stages": [{"max_power": 1000, "cop": 4}],
        }) if kind == "hp" else None
        instance.boiler_options = BoilerEnabled.model_validate({
            "entity actual temp.": "sensor.synthetic_boiler_temp",
            "entity setpoint": "input_number.synthetic_boiler_setpoint",
            "entity hysterese": "input_number.synthetic_boiler_hysteresis",
            "entity instant start": "input_boolean.synthetic_boiler_instant",
            "switch entity": "switch.synthetic_boiler",
            "cooling rate": 0.5, "heating allowed below": 50,
            "cop": 4, "elec. power": 300,
        }) if kind == "boiler" else None
        states = {
            "sensor.synthetic_boiler_temp": "40",
            "input_number.synthetic_boiler_setpoint": "50",
            "input_number.synthetic_boiler_hysteresis": "5",
            "input_boolean.synthetic_boiler_instant": "on",
        }
        found = {"events": [], "sinks": [], "consumer_reads": [], "presence": [],
                 "solves": [], "native_checks": [], "audit_calls": []}
        events = found["events"]

        def sink(name):
            def visit(*args, **kwargs):
                # Record before raising: some output handlers catch Exception.
                found["sinks"].append((name, args, kwargs))
                events.append(name)
                raise OutputReached(name)
            return visit

        def get_state(entity_id):
            assert entity_id in states, f"unexpected synthetic HA read: {entity_id}"
            return SimpleNamespace(state=states[entity_id])

        def record_presence():
            found["presence"].append(True)  # Explicit synthetic pre-solve fake.

        bind_cost = da.bind_day_ahead_cost
        optimize = mip.Model.optimize
        native_check = mip.Model.check_optimization_results
        audit = da._audit_battery_cell_direction
        log_info = logging.info

        def cost(model, **kwargs):
            bind_cost(model, **kwargs)
            found["local"] = dict(sys._getframe(1).f_locals)
            found["model"] = model
            events.append("cost_bound")
            raw_get = model.solver.var_get_x
            overrides = {}
            found["overrides"] = overrides

            def read(var):
                value = overrides.get(var.idx, raw_get(var))
                # Observe real .x extraction in calc, not reads inside the audit
                # or a second, manually reconstructed result consumer.
                if sys._getframe(2).f_code is da.DaCalc.calc_optimum.__code__:
                    found["consumer_reads"].append((var.idx, value, "accepted" in events))
                return value

            spies.setattr(model.solver, "var_get_x", read)

        def solve(model, *args, **kwargs):
            assert model is found["model"]
            found["solves"].append(True)
            assert found["solves"] == [True], "physical acceptance must not re-solve"
            assert model.solver_name == mip.CBC
            assert model.sense == mip.MINIMIZE
            status = optimize(model, *args, **kwargs)
            assert status == mip.OptimizationStatus.OPTIMAL
            assert model.num_solutions == 1
            events.append("solved")
            local = found["local"]
            assert local["B"] == 0 and local["financial_terms"] is None
            if kind == "hp":
                assert local["heat_needed"] == 30 and local["blocks_num"] == 0
                assert [var.x for var in local["hp_on"]] == [1] * 8
                assert [tuple(var.x for var in row) for row in local["hp_s_w"]] == [(0, 1)] * 8
                assert sum(var.x for var in local["h_hp"]) == pytest.approx(32)
                assert local["cost"].x == pytest.approx(1.30)
                if corrupt:
                    # Only the known HP readback defect; no rows, bounds, status,
                    # objective, tolerance or other physical values are changed.
                    found["overrides"].update({local["hp_on"][0].idx: 1.0,
                                               local["hp_s_w"][0][1].idx: 0.5})
            else:
                assert [var.x for var in local["boiler_st"]] == [1] + [0] * 11
                assert [var.x for var in local["c_b"]] == pytest.approx(
                    [0.075] * 10 + [0.0298611111111111, 0])
                assert sum(var.x for var in local["c_b"]) == pytest.approx(0.7798611111111111)
                if corrupt:
                    found["overrides"][local["c_b"][10].idx] = 0.0
            return status

        def check(model):
            found["native_checks"].append(model)
            events.append("native_check")
            try:
                return native_check(model)
            except mip.InfeasibleSolution as exc:
                found["native_failure"] = exc
                log_info("B5e native checker rejected %s: %s", kind, exc)
                raise

        def inspect_audit(model, **kwargs):
            found["audit_calls"].append(kwargs)
            events.append("audit")
            result = audit(model, **kwargs)
            events.append("accepted")
            return result

        def preview(message, *args, **kwargs):
            if message == "Het programma heeft een optimale oplossing gevonden.":
                local = sys._getframe(1).f_locals
                found["baseline"] = (tuple(local["org_l"]), tuple(local["org_t"]))
                found["sinks"].append(("preview", (), {}))
                events.append("preview")
                log_info(message, *args, **kwargs)
                if debug:
                    raise OutputReached("preview")
                return
            return log_info(message, *args, **kwargs)

        # Deliberately seed previous-run financial/debug results even in legacy
        # mode: a rejected calculation must clear rather than expose them.
        for name in DEBUG_RESULTS:
            setattr(instance, name, object())
        with monkeypatch.context() as spies:
            spies.setattr(da, "Report", lambda *args: SimpleNamespace(
                get_price_data=lambda *a, **k: prices.copy()))
            spies.setattr(instance, "db_da", SimpleNamespace(
                get_prognose_data=lambda **kwargs: forecast.copy()))
            spies.setattr(instance, "baseload_service", lambda: SimpleNamespace(
                record_presence=record_presence))
            spies.setattr(instance, "get_state", get_state)
            spies.setattr(instance, "get_float", da.DaCalc.get_float.__get__(instance))
            for name in ACTUATORS + ("save_df", "export_fast_plan"):
                spies.setattr(instance, name, sink(name))
            for name in ("finish_day_ahead_finance", "audit_period_solution",
                         "planning_cost_breakdown", "write_plan"):
                spies.setattr(da, name, sink(name))
            spies.setattr(socket, "getaddrinfo", sink("network_dns"))
            spies.setattr(socket.socket, "connect", sink("network_connect"))
            spies.setattr(socket.socket, "connect_ex", sink("network_connect_ex"))
            spies.setattr(da, "bind_day_ahead_cost", cost)
            spies.setattr(mip.Model, "optimize", solve)
            spies.setattr(mip.Model, "check_optimization_results", check)
            spies.setattr(da, "_audit_battery_cell_direction", inspect_audit)
            spies.setattr(logging, "info", preview)
            try:
                # No start/SoC override: these would silently enable debug.
                instance.calc_optimum()
            except (PlanningSolutionRejected, OutputReached) as exc:
                found["exception"] = exc

        assert instance.debug is debug
        assert instance._financial_pricing.settlement is None
        assert instance._financial_horizon == (stamps[0], stamps[-1] + instance.interval_s)
        assert found["presence"] == ([] if debug else [True])
        assert found["solves"] == [True]
        assert found["native_checks"] == [found["model"]]
        assert len(found["audit_calls"]) == 1
        assert found["audit_calls"][0]["installation_battery_count"] == 0
        assert all(getattr(instance, name) is None for name in DEBUG_RESULTS)
        assert not plan_path.exists()
        assert not list(tmp_path.glob(".fast_plan_*.tmp"))
        log_info("B5e %s/%s debug=%s corrupt=%s: events=%s; result_reads=%s; sinks=%s",
                 kind, adjustment, debug, corrupt, events, len(found["consumer_reads"]),
                 [name for name, _, _ in found["sinks"]])
        return found

    return run


def assert_rejected_before_consumers(found):
    """Removing/moving the producer acceptance gate must expose a consumer here."""
    exc = found["exception"]
    assert isinstance(exc, PlanningSolutionRejected)
    assert exc.code == "battery_direction_solution_invalid"
    assert isinstance(exc.__cause__, mip.InfeasibleSolution)
    assert exc.__cause__ is found["native_failure"]
    assert "Constraint " in str(exc.__cause__) and "is violated" in str(exc.__cause__)
    assert found["events"] == ["cost_bound", "solved", "audit", "native_check"]
    assert found["consumer_reads"] == found["sinks"] == []
    assert "baseline" not in found


@pytest.mark.parametrize("debug", [True, False], ids=["debug", "nondebug"])
@pytest.mark.parametrize("adjustment", ["power", "heating curve"])
def test_native_hp_violation_aborts_real_producer_before_output(physical_producer, debug, adjustment):
    """A real native HP row violation must propagate out of calc before result reads."""
    found = physical_producer("hp", debug=debug, adjustment=adjustment, corrupt=True)
    assert_rejected_before_consumers(found)


@pytest.mark.parametrize("debug", [True, False], ids=["debug", "nondebug"])
def test_native_boiler_underfill_aborts_real_producer_before_output(physical_producer, debug):
    """The complete selected boiler profile must reject a missing late tail in calc."""
    found = physical_producer("boiler", debug=debug, corrupt=True)
    assert_rejected_before_consumers(found)


@pytest.mark.parametrize("debug", [True, False], ids=["debug", "nondebug"])
@pytest.mark.parametrize("kind,adjustment", [("hp", "power"), ("hp", "heating curve"), ("boiler", "power")])
def test_healthy_real_producer_reaches_result_consumers(physical_producer, debug, kind, adjustment):
    """An early return, preflight denial or missing fixture attribute cannot pass the negatives."""
    found = physical_producer(kind, debug=debug, adjustment=adjustment)
    assert isinstance(found["exception"], OutputReached)
    assert found["overrides"] == {}
    assert "native_failure" not in found
    assert found["events"] == ["cost_bound", "solved", "audit", "native_check", "accepted",
                               "preview", *([] if debug else ["save_df"])]
    assert found["consumer_reads"] and all(accepted for _, _, accepted in found["consumer_reads"])
    assert found["consumer_reads"][0][0] == found["local"]["c_b"][0].idx
    assert found["consumer_reads"][0][1] == pytest.approx(0 if kind == "hp" else 0.075)
    imports, exports = found["baseline"]
    assert imports == pytest.approx([1] * 8 if kind == "hp" else
                                    [0.075] * 10 + [0.0298611111111111, 0])
    assert exports == (0,) * len(imports)
    assert [name for name, _, _ in found["sinks"]] == ["preview", *([] if debug else ["save_df"])]
    if not debug:
        _, _, saved = found["sinks"][-1]
        assert saved["tablename"] == "prognoses" and saved["vintage"] is True
        assert saved["df"]["cons"].tolist() == pytest.approx(imports)
