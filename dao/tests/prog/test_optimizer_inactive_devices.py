"""Inactive EV flags must be constrained, not merely zero in one CBC optimum."""

import datetime as dt
import sys
from types import SimpleNamespace

import mip
import pytest

from dao.prog import day_ahead as da
from dao.prog.config.models.devices.ev import EVConfig
from dao.tests.settlement.test_optimization_producer import capture as capture


@pytest.fixture
def ev_model(capture, monkeypatch):
    """Reuse the offline producer boundaries; keep EV construction and CBC real."""
    instance = capture["instance"]
    start = capture["locals"]["start_dt"]

    def build(deadlines, target_levels):
        states = {}
        instance.ev_options = []
        for e, (minutes, target) in enumerate(zip(deadlines, target_levels, strict=True)):
            config = EVConfig.model_validate({
                "name": f"synthetic-ev-{e}", "capacity": 10,
                "charge three phase": False,
                "charge stages": [{"ampere": 6, "efficiency": 1},
                                  {"ampere": 10, "efficiency": 1}],
                "entity position": f"device_tracker.synthetic_ev_{e}",
                "entity actual level": f"sensor.synthetic_ev_{e}_soc",
                "entity plugged in": f"binary_sensor.synthetic_ev_{e}_plugged",
                "charge switch": f"switch.synthetic_ev_{e}",
                "entity set charging ampere": f"number.synthetic_ev_{e}_ampere",
                "entity stop charging": f"input_datetime.synthetic_ev_{e}_stop",
                "charge scheduler": {
                    "entity set level": f"input_number.synthetic_ev_{e}_target",
                    "entity ready datetime": f"input_datetime.synthetic_ev_{e}_ready",
                },
            })
            instance.ev_options.append(config)
            states.update({
                config.entity_position: "home",
                config.entity_actual_level: "50",
                config.entity_plugged_in: "on",
                config.charge_switch: "off",
                config.entity_set_charging_ampere: "0",
                config.charge_scheduler.entity_set_level: str(target),
                config.charge_scheduler.entity_ready_datetime:
                    (start + dt.timedelta(minutes=minutes)).isoformat(),
            })
        # The shared fixture uses a constant battery SoC reader. Restore the real
        # guarded helpers, with only the external state boundary replaced.
        for battery in instance.battery_options:
            states[battery.entity_actual_level] = "50"

        def get_state(entity_id):
            if entity_id not in states:
                pytest.fail(f"unexpected synthetic HA read: {entity_id}")
            return SimpleNamespace(state=states[entity_id])

        monkeypatch.setattr(instance, "get_state", get_state)
        monkeypatch.setattr(instance, "get_float", da.DaCalc.get_float.__get__(instance))
        found = {}
        bind_cost = da.bind_day_ahead_cost

        class Captured(Exception):
            pass

        def capture_cost(model, **kwargs):
            bind_cost(model, **kwargs)
            found.update(sys._getframe(1).f_locals)
            raise Captured

        with monkeypatch.context() as spies:
            spies.setattr(da, "bind_day_ahead_cost", capture_cost)
            with pytest.raises(Captured):
                instance.calc_optimum(_start_dt=start)
        model = found["model"]
        assert model.solver_name == mip.CBC
        assert found["EV"] == len(deadlines) and found["U"] == 12
        model.verbose = 0
        model.objective = mip.minimize(found["cost"])
        return found

    return build


@pytest.mark.parametrize("deadlines,target_levels,ready_indices,energy_kwh", [
    pytest.param((60,), (63,), (3,), (1.3,), id="scheduled"),
    pytest.param((60,), (50,), (12,), (0.0,), id="no-charge-needed"),
    pytest.param((60, 37.5), (63, 63), (3, 2), (1.3, 1.3), id="two-evs"),
    pytest.param((37.5,), (63,), (2,), (1.3,), id="partial-deadline"),
])
@pytest.mark.parametrize("flag,stage,forbidden", [
    pytest.param("ev_is_on", None, 1, id="on"),
    pytest.param("ev_is_off", None, 0, id="off"),
    pytest.param("stage_on", 0, 1, id="zero-stage"),
    pytest.param("stage_on", 1, 1, id="six-ampere-stage"),
    pytest.param("stage_on", 2, 1, id="ten-ampere-stage"),
])
def test_ev_flags_after_deadline_cannot_be_on(
    ev_model, deadlines, target_levels, ready_indices, energy_kwh, flag, stage, forbidden,
):
    """Removing any inactive flag binding admits a forbidden binary assignment."""
    local = ev_model(deadlines, target_levels)
    model, intervals = local["model"], local["U"]
    assert local["ready_u"] == list(ready_indices)
    assert local["energy_needed"] == pytest.approx(energy_kwh)
    for e, ready in enumerate(ready_indices):
        inactive = range(ready + 1, intervals) if ready < intervals else range(intervals)
        for u in inactive:
            row = local[flag][e] if stage is None else local[flag][e][stage]
            forced = model.add_constr(row[u] == forbidden)
            status = model.optimize()
            model.remove(forced)
            assert status == mip.OptimizationStatus.INFEASIBLE, (e, u, flag, stage, status)

    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    for e, (ready, needed) in enumerate(zip(ready_indices, energy_kwh, strict=True)):
        assert sum(var.x for var in local["ev_accu_in"][e]) == pytest.approx(needed, abs=0.00101)
        inactive = range(ready + 1, intervals) if ready < intervals else range(intervals)
        for u in inactive:
            for name in ("c_ev", "p_ev", "ev_accu_in", "ev_is_on", "ev_is_partial", "ev_boundary_stop"):
                assert local[name][e][u].x == 0
            assert local["ev_is_off"][e][u].x == 1
            assert all(row[u].x == 0 for row in local["stage_on"][e])
        if deadlines[e] == 37.5:
            # 1.3 kWh cannot fit before this last, half-length quarter:
            # 2.3 kW * 0.5 h = 1.15 kWh. The deadline interval stays active,
            # but may supply at most 2.3 kW * 0.125 h = 0.2875 kWh.
            assert 0.148 <= local["c_ev"][e][ready].x <= 0.2875 + 1e-7
            assert any(row[ready].x == 1 for row in local["stage_on"][e][1:])
