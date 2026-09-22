"""Tests for the Home Assistant facing side of the fast control layer.

Home Assistant is replaced by a fake that records every call, so these tests
verify the wiring: which entities are read, what gets written, and what happens
when things go wrong.
"""

import datetime
import json

import pytest

from dao.prog.config.models.fastcontrol import FastControlConfig
from dao.prog.fastctrl.plan import (
    BatteryPlanStep,
    BatterySpec,
    FastPlan,
    PlanInterval,
    write_plan,
)
from dao.prog.fastctrl.runner import (
    FastControlRunner,
    HomeAssistantGateway,
    load_state,
    save_state,
)
from dao.prog.fastctrl.policy import ControllerState

T0 = 1_700_000_000


class FakeHass:
    """Minimal stand-in for the hassapi surface DaBase exposes."""

    def __init__(self, states=None, fail_template=False):
        self.states = dict(states or {})
        self.fail_template = fail_template
        self.values = {}
        self.options = {}
        self.services = []
        self.published = {}
        self.switches = {}
        self.template_calls = 0
        self.state_calls = 0
        self.time_zone = "Europe/Amsterdam"
        self.config = type("Config", (), {"fast_control": None, "battery": [1]})()

    def render_template(self, template):
        self.template_calls += 1
        if self.fail_template:
            raise RuntimeError("template API disabled")
        ids = []
        for line in template.splitlines():
            if line.strip().startswith("[{{ states("):
                ids.append(line.split("'")[1])
        return json.dumps([[self.states.get(i), T0] for i in ids])

    def get_state(self, entity_id):
        self.state_calls += 1
        if entity_id not in self.states:
            raise KeyError(entity_id)
        return type(
            "State",
            (),
            {
                "state": self.states[entity_id],
                "last_updated": datetime.datetime.fromtimestamp(T0),
                "last_changed": datetime.datetime.fromtimestamp(T0),
            },
        )()

    def set_value(self, entity_id, value):
        self.values[entity_id] = value

    def select_option(self, entity_id, option):
        self.options[entity_id] = option

    def call_service(self, service, **kwargs):
        self.services.append((service, kwargs))

    def set_state(self, entity_id, state, attributes=None):
        self.published[entity_id] = (state, attributes)

    def turn_on(self, entity_id):
        self.switches[entity_id] = True

    def turn_off(self, entity_id):
        self.switches[entity_id] = False


def make_plan(path, battery_w=0.0, price=0.30, created_ts=T0, house_w=500.0):
    plan = FastPlan(
        created_ts=created_ts,
        interval_s=900,
        specs=[
            BatterySpec(
                name="accu",
                capacity_kwh=10.0,
                max_charge_w=5000.0,
                max_discharge_w=5000.0,
                soc_min=20.0,
                soc_max=95.0,
                cycle_cost=0.01,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                setpoint_entity="input_number.feedin",
                mode_entity="input_select.mode",
                mode_on="Aan",
                mode_off="Uit",
                stop_inverter_entity="input_datetime.stop",
                soc_entity="sensor.soc",
            )
        ],
        intervals=[
            PlanInterval(
                start_ts=T0 + i * 900,
                end_ts=T0 + (i + 1) * 900,
                price_import=price,
                price_export=max(0.0, price - 0.18),
                grid_w=house_w + battery_w,
                house_w=house_w,
                batteries=[
                    BatteryPlanStep(
                        ac_power_w=battery_w,
                        soc_begin=60.0,
                        soc_end=60.0,
                        mode="Aan" if i == 0 else None,
                        stop_inverter="2000-01-01 00:00:00" if i == 0 else None,
                    )
                ],
            )
            for i in range(8)
        ],
        price_average=price,
    )
    write_plan(plan, path)
    return plan


def make_config(mode="active", **overrides):
    data = {
        "mode": mode,
        "grid power": {"entity": "sensor.p1_power"},
        "batteries": [
            {"name": "accu", "actual power": {"entity": "sensor.battery_power"}}
        ],
        "deadband": 0,
        "min benefit": 0.0,
        "min command interval": 0,
    }
    data.update(overrides)
    return FastControlConfig.model_validate(data)


@pytest.fixture
def workspace(tmp_path):
    return {
        "plan_path": str(tmp_path / "fast_plan.json"),
        "state_path": str(tmp_path / "fast_state.json"),
    }


class TestGateway:
    def test_one_templated_call_serves_every_entity(self):
        hass = FakeHass({"sensor.a": "1", "sensor.b": "2"})
        gateway = HomeAssistantGateway(hass)
        result = gateway.read(["sensor.a", "sensor.b"])
        assert result["sensor.a"][0] == "1"
        assert result["sensor.b"][0] == "2"
        assert hass.template_calls == 1
        assert hass.state_calls == 0

    def test_it_falls_back_to_single_reads_and_remembers(self):
        hass = FakeHass({"sensor.a": "1"}, fail_template=True)
        gateway = HomeAssistantGateway(hass)
        assert gateway.read(["sensor.a"])["sensor.a"][0] == "1"
        assert gateway.use_template is False
        gateway.read(["sensor.a"])
        assert hass.template_calls == 1
        assert hass.state_calls == 2

    def test_a_failing_write_does_not_raise(self):
        class Broken(FakeHass):
            def set_value(self, entity_id, value):
                raise RuntimeError("entity is read only")

        gateway = HomeAssistantGateway(Broken())
        assert gateway.write_number("input_number.x", 1) is False

    def test_reading_an_unknown_entity_yields_none(self):
        gateway = HomeAssistantGateway(FakeHass({}, fail_template=True))
        assert gateway.read(["sensor.missing"])["sensor.missing"] == (None, 0.0)


class TestSensorParsing:
    def test_kilowatts_are_converted(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass({"sensor.p1_power": "3.5", "sensor.soc": "60"})
        config = make_config(
            **{"grid power": {"entity": "sensor.p1_power", "unit": "kW"}}
        )
        runner = FastControlRunner(hass, config, **workspace)
        runner._batch = runner.gateway.read(["sensor.p1_power"])
        assert runner._power(config.grid_power, T0 + 60)[0] == pytest.approx(3500.0)

    def test_the_sign_can_be_inverted(self, workspace):
        hass = FakeHass({"sensor.p1_power": "1200"})
        config = make_config(
            **{"grid power": {"entity": "sensor.p1_power", "invert": True}}
        )
        runner = FastControlRunner(hass, config, **workspace)
        runner._batch = runner.gateway.read(["sensor.p1_power"])
        assert runner._power(config.grid_power, T0 + 60)[0] == pytest.approx(-1200.0)

    def test_a_positive_negative_pair_is_combined(self, workspace):
        hass = FakeHass({"sensor.imp": "0", "sensor.exp": "900"})
        config = make_config(
            **{
                "grid power": {
                    "entity positive": "sensor.imp",
                    "entity negative": "sensor.exp",
                }
            }
        )
        runner = FastControlRunner(hass, config, **workspace)
        runner._batch = runner.gateway.read(["sensor.imp", "sensor.exp"])
        assert runner._power(config.grid_power, T0 + 60)[0] == pytest.approx(-900.0)

    @pytest.mark.parametrize("raw", ["unknown", "unavailable", "", "abc", None])
    def test_unusable_states_are_rejected(self, workspace, raw):
        hass = FakeHass({"sensor.p1_power": raw})
        config = make_config()
        runner = FastControlRunner(hass, config, **workspace)
        runner._batch = runner.gateway.read(["sensor.p1_power"])
        assert runner._power(config.grid_power, T0 + 60) == (None, False)

    def test_a_stale_reading_is_flagged(self, workspace):
        hass = FakeHass({"sensor.p1_power": "1000"})
        config = make_config(**{"max sensor age": 30})
        runner = FastControlRunner(hass, config, **workspace)
        runner._batch = runner.gateway.read(["sensor.p1_power"])
        assert runner._power(config.grid_power, T0 + 10)[1] is True
        assert runner._power(config.grid_power, T0 + 999)[1] is False


class TestTick:
    def test_an_unexpected_load_is_written_to_the_inverter(self, workspace):
        make_plan(workspace["plan_path"], battery_w=0.0)
        hass = FakeHass(
            {
                "sensor.p1_power": "3500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        runner = FastControlRunner(hass, make_config(), **workspace)
        decision = runner.tick(T0 + 60)
        assert decision is not None and decision.override
        assert hass.values["input_number.feedin"] == pytest.approx(-3500.0, abs=1.0)
        assert hass.options["input_select.mode"] == "Aan"

    def test_shadow_mode_writes_diagnostics_but_not_the_inverter(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass(
            {
                "sensor.p1_power": "3500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        runner = FastControlRunner(
            hass, make_config(mode="shadow", **{"max sensor age": 999999}), **workspace
        )
        decision = runner.tick(T0 + 60)
        assert decision is not None and decision.override
        assert hass.values == {}
        assert hass.options == {}
        state, attributes = hass.published["sensor.dao_fast_control"]
        assert state.startswith("shadow:")
        assert attributes["deviation_w"] == pytest.approx(3000.0, abs=1.0)

    def test_off_mode_does_nothing(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass({"sensor.p1_power": "3500", "sensor.soc": "60"})
        runner = FastControlRunner(hass, make_config(mode="off"), **workspace)
        assert runner.tick(T0 + 60) is None
        assert hass.values == {}

    def test_the_mode_can_come_from_an_entity(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass(
            {
                "input_select.fast_mode": "shadow",
                "sensor.p1_power": "3500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        runner = FastControlRunner(
            hass, make_config(mode="input_select.fast_mode"), **workspace
        )
        runner.tick(T0 + 60)
        assert hass.values == {}
        hass.states["input_select.fast_mode"] = "active"
        runner.tick(T0 + 120)
        assert "input_number.feedin" in hass.values

    def test_a_missing_plan_is_survivable(self, workspace):
        hass = FakeHass({"sensor.p1_power": "3500"})
        runner = FastControlRunner(hass, make_config(), **workspace)
        assert runner.tick(T0 + 60) is None

    def test_an_old_plan_stops_the_override(self, workspace):
        make_plan(workspace["plan_path"], created_ts=T0 - 100_000)
        hass = FakeHass(
            {
                "sensor.p1_power": "3500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        runner = FastControlRunner(hass, make_config(), **workspace)
        decision = runner.tick(T0 + 60)
        assert decision is not None
        assert decision.reason == "sensor_stale"
        assert not decision.override

    def test_a_missing_grid_sensor_prevents_any_action(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass({"sensor.soc": "60"})
        config = make_config(**{"grid power": {}})
        runner = FastControlRunner(hass, config, **workspace)
        assert runner.tick(T0 + 60) is None

    def test_the_plan_is_reloaded_when_the_file_changes(self, workspace):
        make_plan(workspace["plan_path"], battery_w=0.0)
        hass = FakeHass(
            {
                "sensor.p1_power": "500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        runner = FastControlRunner(hass, make_config(), **workspace)
        runner.tick(T0 + 60)
        first = runner.plan()
        make_plan(workspace["plan_path"], battery_w=-2000.0, created_ts=T0 + 1)
        second = runner.plan()
        assert second is not first
        assert second.intervals[0].battery(0).ac_power_w == -2000.0

    def test_state_survives_a_restart(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass(
            {
                "sensor.p1_power": "3500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        first = FastControlRunner(hass, make_config(), **workspace)
        first.tick(T0 + 60)
        first.tick(T0 + 120)
        saved = first.state.battery(0).last_command_w

        second = FastControlRunner(hass, make_config(), **workspace)
        assert second.state.battery(0).last_command_w == saved

    def test_diagnostic_helpers_are_written(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass(
            {
                "sensor.p1_power": "3500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        config = make_config(
            diagnostics={
                "entity status": "sensor.fast",
                "entity active": "input_boolean.fast_active",
                "entity setpoint": "input_number.fast_setpoint",
                "entity benefit": "input_number.fast_benefit",
            }
        )
        runner = FastControlRunner(hass, config, **workspace)
        runner.tick(T0 + 60)
        assert hass.switches["input_boolean.fast_active"] is True
        assert hass.values["input_number.fast_setpoint"] == pytest.approx(
            -3500.0, abs=1.0
        )
        assert "sensor.fast" in hass.published


class TestStatePersistence:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "state.json")
        state = ControllerState(day_key="2024-05-01", saved_today_eur=1.23)
        state.battery(0).last_command_w = -1234.0
        state.battery(0).daily_deviation_kwh = 2.5
        save_state(state, path)
        restored = load_state(path)
        assert restored.day_key == "2024-05-01"
        assert restored.saved_today_eur == 1.23
        assert restored.battery(0).last_command_w == -1234.0
        assert restored.battery(0).daily_deviation_kwh == 2.5

    def test_a_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("not json at all")
        assert load_state(str(path)).day_key == ""

    def test_a_missing_file_starts_fresh(self, tmp_path):
        assert load_state(str(tmp_path / "absent.json")).saved_today_eur == 0.0


class TestFlashWear:
    """The control loop runs every 15 s; it must not write to disk that often.

    Thousands of small writes a day is the wrong thing to do on the eMMC of a
    Home Assistant Yellow, and actively harmful on the SD card of a Green or a
    Raspberry Pi. The state file only has to survive a restart.
    """

    def runner(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass(
            {
                "sensor.p1_power": "500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        return FastControlRunner(hass, make_config(), **workspace), hass

    def counted(self, monkeypatch):
        """Count how often the state file is actually written."""
        import dao.prog.fastctrl.runner as module

        calls = []
        original = module.save_state
        monkeypatch.setattr(
            module,
            "save_state",
            lambda state, path: calls.append(path) or original(state, path),
        )
        return calls

    def test_twenty_minutes_of_ticking_writes_a_handful_of_times(
        self, workspace, monkeypatch
    ):
        from dao.prog.fastctrl.runner import STATE_SAVE_INTERVAL

        runner, _ = self.runner(workspace)
        calls = self.counted(monkeypatch)
        ticks = 80  # twenty minutes at 15 s
        for step in range(ticks):
            runner._persist_state(T0 + step * 15)
        assert len(calls) <= (ticks * 15) / STATE_SAVE_INTERVAL + 2
        assert len(calls) < ticks / 10

    def test_the_first_call_always_writes(self, workspace, monkeypatch):
        runner, _ = self.runner(workspace)
        calls = self.counted(monkeypatch)
        runner._persist_state(T0)
        assert len(calls) == 1

    def test_an_override_change_forces_a_write(self, workspace, monkeypatch):
        runner, _ = self.runner(workspace)
        calls = self.counted(monkeypatch)
        runner._persist_state(T0)
        runner._persist_state(T0 + 15)
        assert len(calls) == 1
        runner.state.battery(0).override_active = True
        runner._persist_state(T0 + 30)
        assert len(calls) == 2

    def test_a_new_day_forces_a_write(self, workspace, monkeypatch):
        runner, _ = self.runner(workspace)
        calls = self.counted(monkeypatch)
        runner._persist_state(T0)
        runner.state.day_key = "2026-01-02"
        runner._persist_state(T0 + 15)
        assert len(calls) == 2

    def test_a_new_plan_forces_a_write(self, workspace, monkeypatch):
        runner, _ = self.runner(workspace)
        calls = self.counted(monkeypatch)
        runner._persist_state(T0)
        runner.state.plan_created_ts = T0 + 999
        runner._persist_state(T0 + 15)
        assert len(calls) == 2

    def test_the_interval_elapsing_forces_a_write(self, workspace, monkeypatch):
        from dao.prog.fastctrl.runner import STATE_SAVE_INTERVAL

        runner, _ = self.runner(workspace)
        calls = self.counted(monkeypatch)
        runner._persist_state(T0)
        runner._persist_state(T0 + STATE_SAVE_INTERVAL + 1)
        assert len(calls) == 2

    def test_force_always_writes(self, workspace, monkeypatch):
        runner, _ = self.runner(workspace)
        calls = self.counted(monkeypatch)
        runner._persist_state(T0)
        runner._persist_state(T0 + 1, force=True)
        assert len(calls) == 2

    def test_what_is_written_is_still_complete(self, workspace):
        runner, _ = self.runner(workspace)
        runner.state.saved_today_eur = 1.2345
        runner.state.battery(0).daily_deviation_kwh = 2.5
        runner._persist_state(T0, force=True)
        restored = load_state(workspace["state_path"])
        assert restored.saved_today_eur == 1.2345
        assert restored.battery(0).daily_deviation_kwh == 2.5


class TestMeasurement:
    """The loop already reconstructs the true house demand; record it.

    Without a measured counterpart on the same time grid and with the same
    definition, the forecast error of 'hload' cannot be computed at all.
    """

    class RecordingDb:
        def __init__(self):
            self.saved = []

        def savedata(self, df, tablename="values"):
            self.saved.append((tablename, df.values.tolist()))

    def make(self, workspace, interval_battery_w=0.0):
        make_plan(workspace["plan_path"], battery_w=interval_battery_w)
        hass = FakeHass(
            {
                "sensor.p1_power": "1000",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        hass.db_da = self.RecordingDb()
        runner = FastControlRunner(
            hass, make_config(mode="shadow", **{"max sensor age": 999999}), **workspace
        )
        return runner, hass

    def test_the_realised_house_demand_is_written_at_the_interval_boundary(
        self, workspace
    ):
        runner, hass = self.make(workspace)
        # A whole 900 s interval at a steady 1000 W is 0.25 kWh.
        for offset in range(0, 900, 30):
            runner.tick(T0 + offset)
        assert hass.db_da.saved == []  # nothing until the interval closes
        runner.tick(T0 + 905)
        assert hass.db_da.saved
        table, rows = hass.db_da.saved[0]
        assert table == "values"
        codes = {row[1]: row[2] for row in rows}
        assert codes["m_house"] == pytest.approx(0.25, abs=0.02)

    def test_a_partly_covered_interval_is_not_written(self, workspace):
        runner, hass = self.make(workspace)
        # Only the last two minutes of the interval are observed.
        runner.tick(T0 + 780)
        runner.tick(T0 + 810)
        runner.tick(T0 + 905)
        assert hass.db_da.saved == []

    def test_pv_is_recorded_when_a_sensor_is_configured(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass(
            {
                "sensor.p1_power": "1000",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
                "sensor.pv": "2000",
            }
        )
        hass.db_da = self.RecordingDb()
        config = make_config(
            mode="shadow",
            **{"pv power": {"entity": "sensor.pv"}, "max sensor age": 999999},
        )
        runner = FastControlRunner(hass, config, **workspace)
        for offset in range(0, 900, 30):
            runner.tick(T0 + offset)
        runner.tick(T0 + 905)
        codes = {row[1]: row[2] for row in hass.db_da.saved[0][1]}
        assert codes["m_pv"] == pytest.approx(0.5, abs=0.02)

    def test_a_long_gap_does_not_invent_energy(self, workspace):
        runner, hass = self.make(workspace)
        runner.tick(T0 + 0)
        # The add-on was down for ten minutes; integrating across that gap
        # would book energy that was never measured.
        runner.tick(T0 + 600)
        runner.tick(T0 + 905)
        assert hass.db_da.saved == []

    def test_a_missing_database_is_survivable(self, workspace):
        make_plan(workspace["plan_path"])
        hass = FakeHass(
            {
                "sensor.p1_power": "1000",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        runner = FastControlRunner(
            hass, make_config(mode="shadow", **{"max sensor age": 999999}), **workspace
        )
        for offset in range(0, 950, 30):
            runner.tick(T0 + offset)  # must not raise
