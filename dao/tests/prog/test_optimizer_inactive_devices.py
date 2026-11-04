"""Device flags must be allocated once and constrained in the real producer."""

import ast
import copy
import datetime as dt
import inspect
import sys
import textwrap
from types import SimpleNamespace

import mip
import pytest

from dao.prog import day_ahead as da
from dao.prog.config.models.devices.ev import EVConfig
from dao.prog.config.models.devices.heating import HeatingDisabled, HeatingEnabled
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


@pytest.fixture
def hp_model(capture, monkeypatch):
    """Observe real allocations, including handles lost by a later hp_on assignment."""
    instance = capture["instance"]
    start = capture["locals"]["start_dt"]

    def build(mode):
        if mode == "absent":
            instance.heating_options = None
        elif mode == "disabled":
            instance.heating_options = HeatingDisabled()
        else:
            instance.heating_options = HeatingEnabled.model_validate({
                "adjustment": mode if mode in ("on/off", "power", "heating curve") else "power",
                "degree days factor": 0 if mode == "no-heat" else 4,
                "entity hp enabled": "input_boolean.synthetic_hp_enabled",
                "stages": [{"max_power": 1000, "cop": 4}, {"max_power": 2000, "cop": 4}],
            })

        def get_state(entity_id):
            if entity_id != "input_boolean.synthetic_hp_enabled":
                pytest.fail(f"unexpected synthetic HA read: {entity_id}")
            return SimpleNamespace(state="off" if mode == "switched-off" else "on")

        monkeypatch.setattr(instance, "get_state", get_state)
        instance.meteo = SimpleNamespace(calc_graaddagen=lambda **kwargs: 1.0,
                                         get_avg_temperature=lambda **kwargs: 5.0)
        found, allocated = {}, {}
        add_var, bind_cost = mip.Model.add_var, da.bind_day_ahead_cost

        def record_flags(local):
            for flag in local.get("hp_on", ()):
                allocated[flag.idx] = flag

        def record_var(model, *args, **kwargs):
            caller = sys._getframe(1)
            while caller is not None and caller.f_code is not inspect.unwrap(da.DaCalc.calc_optimum).__code__:
                caller = caller.f_back
            if caller is not None:
                record_flags(caller.f_locals)
            return add_var(model, *args, **kwargs)

        class Captured(Exception):
            pass

        def capture_cost(model, **kwargs):
            bind_cost(model, **kwargs)
            found.update(sys._getframe(1).f_locals)
            found["intermediate_hp_on"] = tuple(allocated.values())
            record_flags(found)
            raise Captured

        with monkeypatch.context() as spies:
            spies.setattr(mip.Model, "add_var", record_var)
            spies.setattr(da, "bind_day_ahead_cost", capture_cost)
            with pytest.raises(Captured):
                instance.calc_optimum(_start_dt=start)
        model = found["model"]
        assert model.solver_name == mip.CBC and found["U"] == 12
        model.verbose = 0
        model.objective = mip.minimize(found["cost"])
        found["allocated_hp_on"] = tuple(allocated.values())
        return found

    return build


@pytest.mark.parametrize("mode,enabled,energy_kwh,heat_kwh", [
    pytest.param("absent", False, 0, 0, id="absent"),
    pytest.param("disabled", False, 0, 0, id="disabled"),
    pytest.param("switched-off", False, 0, 0, id="switched-off"),
    pytest.param("no-heat", False, 0, 0, id="no-heat"),
    pytest.param("on/off", True, 1.5, 6, id="on-off"),
    pytest.param("power", True, 1, 4, id="power"),
    pytest.param("heating curve", True, 1, 4, id="heating-curve"),
])
def test_hp_allocates_one_on_flag_per_interval(hp_model, mode, enabled, energy_kwh, heat_kwh):
    """A second allocation leaves real orphan columns; every retained flag must control load."""
    local = hp_model(mode)
    model, flags, allocated = local["model"], local["hp_on"], local["allocated_hp_on"]
    assert local["intermediate_hp_on"], "allocation observer saw no intermediate hp_on handles"
    assert {flag.idx for flag in local["intermediate_hp_on"]} == {flag.idx for flag in allocated}
    assert len(allocated) == local["U"]
    assert {flag.idx for flag in allocated} == {flag.idx for flag in flags}
    used = {var.idx for constraint in model.constrs for var in constraint.expr.expr}
    for flag in allocated:
        assert model.vars[flag.idx] is flag and flag.var_type == mip.BINARY
        assert flag.idx in used

    assert local["self"].hp_enabled is enabled
    if enabled:
        # A disabled flag cannot supply electricity, even under a forced load.
        forced = [model.add_constr(flags[0] == 0), model.add_constr(local["c_hp"][0] >= 0.001)]
        assert model.optimize() == mip.OptimizationStatus.INFEASIBLE
        model.remove(forced)
        # One ordinary hour avoids asserting new B5 block-duration semantics.
        for u, flag in enumerate(flags):
            model += flag == int(u < 4)
    else:
        forced = model.add_constr(mip.xsum(flags) >= 1)
        assert model.optimize() == mip.OptimizationStatus.INFEASIBLE
        model.remove(forced)

    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    model.check_optimization_results()
    assert [flag.x for flag in flags] == [int(enabled and u < 4) for u in range(12)]
    assert [var.x for var in local["c_hp"]] == pytest.approx(
        [energy_kwh / 4 if u < 4 else 0 for u in range(12)])
    assert sum(var.x for var in local["c_hp"]) == pytest.approx(energy_kwh)
    if enabled:
        assert sum(var.x for var in local["h_hp"]) == pytest.approx(heat_kwh)
        if mode != "on/off":
            assert [stage["max_power"] for stage in local["hp_stages"]] == [0, 1000, 2000]
            assert [stage.max_power for stage in local["self"].heating_options.stages] == [1000, 2000]


def test_hp_allocation_observer_exposes_real_orphans_from_second_allocation(hp_model, monkeypatch):
    """A controlled duplicate of the real allocation must not disappear in final capture."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(da.DaCalc.calc_optimum))))
    body = tree.body[0].body
    index = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "hp_on" for target in node.targets))
    body.insert(index, copy.deepcopy(body[index]))
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), da.__file__, "exec"), da.__dict__, namespace)
    monkeypatch.setattr(da.DaCalc, "calc_optimum", namespace["calc_optimum"])

    local = hp_model("absent")
    model, flags, allocated = local["model"], local["hp_on"], local["allocated_hp_on"]
    assert local["intermediate_hp_on"], "allocation observer saw no intermediate hp_on handles"
    assert len(flags) == local["U"] == 12
    assert len(allocated) == 2 * local["U"]
    assert {flag.idx for flag in local["intermediate_hp_on"]} == {flag.idx for flag in allocated}
    final_indices = {flag.idx for flag in flags}
    orphaned = [flag for flag in allocated if flag.idx not in final_indices]
    used = {var.idx for constraint in model.constrs for var in constraint.expr.expr}
    assert len(orphaned) == local["U"]
    assert all(model.vars[flag.idx] is flag and flag.idx not in used for flag in orphaned)
