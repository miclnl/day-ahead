"""Tests for the Home Assistant facing side of the fast control layer.

Home Assistant is replaced by a fake that records every call, so these tests
verify the wiring: which entities are read, what gets written, and what happens
when things go wrong.
"""

import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dao.prog.config.models.fastcontrol import FastControlConfig
from dao.prog.fastctrl import publication
from dao.prog.fastctrl.plan import (
    BatteryPlanStep,
    BatterySpec,
    FastPlan,
    PlanInterval,
    write_plan,
)
from dao.prog.fastctrl.policy import ControllerState, FastControlPolicy
from dao.prog.fastctrl.runner import (
    FastControlRunner,
    HomeAssistantGateway,
    load_state,
    save_state,
)
from dao.tests.settlement.test_fast_contract import ready_plan
from dao.tests.settlement.test_publication import pricing_config

T0 = 1_700_000_000


@pytest.fixture(autouse=True)
def boottime_at_start_phase(monkeypatch):
    # Existing v2/public-gate probes run at an admitted phase. I7 probes inject
    # their own advancing clock and exercise the real admission predicate.
    monkeypatch.setattr("dao.prog.fastctrl.runner.time.clock_gettime_ns", lambda _: 15_000_000_000)


@pytest.mark.parametrize("owner_present", [False, True])
def test_i7_early_financial_read_and_sensor_overrun(tmp_path, monkeypatch, owner_present):
    """Financial early reads respect cadence, even without an owner."""
    from contextlib import contextmanager

    from dao.prog.fastctrl import runner as runner_module
    from dao.tests.prog.test_fast_consumer import NOW_TS, build

    harness = build(tmp_path)
    clock = [23_000_000_000]
    harness.owner.now_ns = lambda: clock[0]
    if not owner_present:
        harness.runner.owner = None
    monkeypatch.setattr(runner_module.time, "clock_gettime_ns", lambda _: clock[0])
    acquired = []
    real_lock = runner_module.publication_lock

    @contextmanager
    def observe(path):
        with real_lock(path) as lease:
            acquired.append(clock[0])
            yield lease

    monkeypatch.setattr(runner_module, "publication_lock", observe)
    try:
        assert harness.runner.tick(NOW_TS) is None
        assert acquired == []
        assert harness.runner.last_gate.reason == "admission_deferred"
        clock[0] = 25_000_000_000
        assert harness.runner.tick(NOW_TS) is None
        assert acquired == [25_000_000_000]
        assert harness.runner.last_gate.reason == "actuation_gates_unproven"
    finally:
        harness.lease.close()


def test_i7_early_financial_read_and_sensor_overrun_discards_proposal(tmp_path, monkeypatch):
    from dao.prog.fastctrl import runner as runner_module
    from dao.tests.prog.test_fast_consumer import NOW_TS, build

    harness = build(tmp_path)
    clock = [20_000_000_000]
    harness.owner.now_ns = lambda: clock[0]
    # Existing internal-v2 composition seam; admission itself is always real.
    monkeypatch.setattr(runner_module, "following_verdict",
                        lambda *a, **kw: publication.GateVerdict(True, "synthetic"))

    def slow_sensor(ids):
        clock[0] = 21_000_000_000
        return {"sensor.p1_power": ("2000", NOW_TS), "sensor.soc": ("50", NOW_TS)}

    monkeypatch.setattr(harness.runner.gateway, "read", slow_sensor)
    try:
        harness.runner.tick(NOW_TS)
        assert harness.runner.last_actuation.reason == "admission_window_missed"
        assert harness.gateway.effect_log == []
        assert harness.store.load_strict().last_reserved_sequence == 40
    finally:
        harness.lease.close()


def test_i7_unexpected_v2_releases_legacy_read_before_financial_processing(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from dataclasses import replace

    from dao.prog.fastctrl import runner as runner_module
    from dao.tests.prog.test_fast_consumer import NOW_TS, financial_plan

    plan = financial_plan()
    path = tmp_path / "fast_plan.json"
    path.write_text(json.dumps(replace(plan, format_version=1, financial_context=None).to_dict()))
    runner = FastControlRunner(FakeHass(), make_config(), plan_path=path,
        state_path=str(tmp_path / "state.json"), gate_policy=publication.GatePolicy(None))
    clock, acquisitions, verdicts = [23_000_000_000], [], []
    real_lock, real_verdict = runner_module.publication_lock, runner_module.following_verdict

    @contextmanager
    def publish_race(target):
        with real_lock(target) as lease:
            if not acquisitions:
                path.write_text(json.dumps(plan.to_dict()))
            acquisitions.append(clock[0])
            yield lease

    def verdict(*args, **kwargs):
        verdicts.append(clock[0])
        return real_verdict(*args, **kwargs)

    monkeypatch.setattr(runner_module, "publication_lock", publish_race)
    monkeypatch.setattr(runner_module, "following_verdict", verdict)
    monkeypatch.setattr(runner_module.time, "clock_gettime_ns", lambda _: clock[0])
    assert runner.tick(NOW_TS) is None
    assert acquisitions == [23_000_000_000] and verdicts == []
    assert runner.last_gate.reason == "admission_deferred"
    clock[0] = 25_000_000_000
    assert runner.tick(NOW_TS) is None
    assert acquisitions == [23_000_000_000, 25_000_000_000]
    assert verdicts == [25_000_000_000]
    assert runner.last_gate.reason == "actuation_gates_unproven"


def test_i7_shutdown_requested_during_sensors_prevents_new_send(tmp_path, monkeypatch):
    import threading

    from dao.prog.fastctrl import runner as runner_module
    from dao.tests.prog.test_fast_consumer import NOW_TS, build

    harness = build(tmp_path)
    runner = harness.runner
    runner._owner_writer = harness.lease
    stopped = threading.Event()
    monkeypatch.setattr(runner_module, "following_verdict",
                        lambda *a, **k: publication.GateVerdict(True, "synthetic"))
    monkeypatch.setattr(runner_module.time, "time", lambda: NOW_TS)

    def sensor(ids):
        stopped.set()
        return {"sensor.p1_power": ("2000", NOW_TS), "sensor.soc": ("50", NOW_TS)}

    class Wait:
        def wait(self, seconds):
            harness.gateway.advance_clock(round(seconds * 1000))

    monkeypatch.setattr(runner.gateway, "read", sensor)
    monkeypatch.setattr(runner_module.threading, "Event", Wait)
    try:
        runner.run(stopped)
        assert harness.gateway.effect_log == []
        assert runner.last_actuation.reason == "shutdown"
        assert harness.lease.fd == -1
    finally:
        harness.lease.close()


@pytest.mark.parametrize("interval", [5, 10, 15])
def test_i7_loop_started_outside_window_makes_progress(tmp_path, monkeypatch, interval):
    from dao.prog.fastctrl import runner as runner_module
    from dao.tests.prog.test_fast_consumer import NOW, NOW_TS, build

    harness = build(tmp_path, config=make_config(interval=interval))
    harness.runner.owner = None
    clock = [23_000_000_000]
    monkeypatch.setattr(runner_module.time, "clock_gettime_ns", lambda _: clock[0])
    monkeypatch.setattr(runner_module.time, "time", lambda: NOW_TS + (clock[0] - 23_000_000_000) / 1e9)
    monkeypatch.setattr(runner_module.time, "time_ns", lambda: NOW * 1_000_000 + clock[0])
    seen = []
    real_tick = harness.runner.tick

    def tick(now):
        result = real_tick(now)
        seen.append((clock[0], harness.runner.last_gate.reason))
        return result

    class Wait:
        calls = 0

        def is_set(self):
            return self.calls >= 3

        def wait(self, seconds):
            clock[0] += round(seconds * 1e9)
            self.calls += 1
            return self.is_set()

    monkeypatch.setattr(harness.runner, "tick", tick)
    try:
        harness.runner.run(Wait())
        assert any(at == 25_000_000_000 and reason == "actuation_gates_unproven"
                   for at, reason in seen), seen
        assert all(at % 5_000_000_000 < 1_000_000_000 for at, reason in seen
                   if reason == "actuation_gates_unproven"), seen
    finally:
        harness.lease.close()


@pytest.mark.parametrize("owner_state", ["owned", "missing", "inhibited", "expired"])
def test_i4_tick_passes_only_live_owner_context_before_policy(tmp_path, monkeypatch, owner_state):
    from dataclasses import replace

    from dao.prog.fastctrl.publication import GatePolicy
    from dao.tests.prog.test_fast_consumer import NOW, NOW_TS, build, consume, propose

    harness = build(tmp_path)
    try:
        consume(harness, propose(harness.plan, harness.runner.state))
        binding = harness.owner.binding
        # A legacy publication exercises normal tick wiring without opening a v2 gate.
        legacy = replace(harness.plan, format_version=1, financial_context=None)
        harness.path.write_text(json.dumps(legacy.to_dict()))
        harness.runner.gate_policy = GatePolicy(None)
        if owner_state == "missing":
            harness.owner.binding = None
        elif owner_state == "inhibited":
            harness.owner.phase = "inhibited"
        elif owner_state == "expired":
            harness.owner.binding = replace(binding, owner_not_after_ms=NOW)
        observed = []
        account, decide = FastControlPolicy.account, FastControlPolicy.decide

        def account_with_context(self, *args, **kwargs):
            observed.append(("account", kwargs.get("owner_binding")))
            return account(self, *args, **kwargs)

        def decide_with_context(self, *args, **kwargs):
            observed.append(("decide", kwargs.get("owner_binding")))
            return decide(self, *args, **kwargs)

        monkeypatch.setattr(FastControlPolicy, "account", account_with_context)
        monkeypatch.setattr(FastControlPolicy, "decide", decide_with_context)
        harness.runner.tick(NOW_TS)
        expected = binding if owner_state == "owned" else None
        assert observed == [("account", expected), ("decide", expected)]
        if expected is None:
            assert harness.runner.state.battery(0)._current_accepted_command is None
    finally:
        harness.lease.close()


def test_i4_tick_does_not_book_a_gap_hidden_by_the_legacy_saving_cap(tmp_path, monkeypatch):
    import copy
    from dataclasses import replace

    from dao.prog.fastctrl import runner as runner_module
    from dao.prog.fastctrl.command import plan_digest
    from dao.prog.fastctrl.policy import current_execution_command
    from dao.prog.fastctrl.publication import GateVerdict
    from dao.tests.prog.test_fast_consumer import (
        NOW,
        NOW_TS,
        build,
        consume,
        financial_plan,
        propose,
    )

    plan = financial_plan(powers=(-4000.0,))
    first = plan.intervals[0]
    # Keep acceptance live past the final tick without exhausting the default
    # energy budget: a continuous profile in one final, partial hourly interval.
    step = replace(first.battery(0), ac_power_w=-4000.0, mean_ac_power_w=-4000.0,
                   stop_ts=None)
    plan = replace(plan, interval_s=3600, intervals=[replace(
        first, end_ts=NOW_TS + 2700, batteries=[step])])
    harness = build(tmp_path, plan=plan)
    try:
        runner = harness.runner
        assert plan.financial_context.not_after_ts == NOW_TS + 1800
        assert harness.owner.binding.owner_not_after_ms == NOW + 3_600_000
        assert runner.limits().energy_budget == 0.5
        runner.state.last_tick_ts = NOW_TS
        runner.state.day_key = runner._day_key(NOW_TS)
        # A command accepted between the two ticks does not cover the whole gap.
        harness.gateway.advance_clock(100_000)
        receipt = consume(harness, propose(harness.plan, runner.state, timestamp=NOW_TS + 100,
                                            grid_w=5000), now_ms=NOW + 100_000)[0]
        assert receipt.status == "accepted"
        controller = runner.state.battery(0)
        command = controller.last_accepted_command
        assert command.issued_at_ms == receipt.accepted_ts == NOW + 100_000
        assert command.hold_until_ms == NOW + 1_800_000
        harness.gateway.advance_clock(900_000)
        now = NOW_TS + 1000
        assert current_execution_command(
            controller, binding=harness.owner.binding, now_ms=NOW + 1_000_000,
            expected_digest=plan_digest(plan)) is command
        # Synthetic composition of the internal v2 loop only; the public refusal
        # is tested separately and no production gate is changed.
        monkeypatch.setattr(runner_module, "following_verdict",
                            lambda *args, **kwargs: GateVerdict(True, "synthetic"))
        monkeypatch.setattr(runner.gateway, "read", lambda ids: {
            "sensor.p1_power": ("5000", now), "sensor.soc": ("50", now),
            "sensor.battery_power": ("-5000", now)})
        decision = runner.tick(now)
        assert decision.benefit_eur_h > 0
        assert runner.state.saved_today_eur == 0.0
        assert controller.last_accepted_command is command
        assert abs(controller.interval_deviation_kwh) < runner.limits().energy_budget
        # The clipped +400..+1000s gap would book a positive amount; the real
        # +0..+1000s gap must still reject its unaccepted first 100 seconds.
        clipped = copy.copy(runner)
        clipped.state = copy.deepcopy(runner.state)
        clipped._book_saving(decision, elapsed_h=600 / 3600, plan=plan,
                             owner_binding=harness.owner.binding)
        assert clipped.state.saved_today_eur > 0
        assert runner.state.saved_today_eur == 0.0
    finally:
        harness.lease.close()


class FakeHass:
    """Minimal stand-in for the HA client surface DaBase exposes."""

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

    def test_dabase_answers_every_call_the_gateway_makes(self):
        """FakeHass is allowed to be minimal, not to be more capable.

        ``render_template`` existed on the fake from day one but never on
        DaBase, so in production the batched read raised AttributeError on
        its first attempt of every run and the layer quietly degraded to one
        HTTP request per sensor per tick. Derived from the source so a newly
        added call is covered without anyone remembering to list it here.
        """
        import ast
        import inspect

        from dao.prog.da_base import DaBase

        tree = ast.parse(inspect.getsource(HomeAssistantGateway).lstrip())
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "hass"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
        }
        assert used, "the gateway no longer talks to Home Assistant"
        missing = sorted(n for n in used if not callable(getattr(DaBase, n, None)))
        assert not missing, f"DaBase is missing {missing}"


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

    def test_saved_today_is_flagged_as_an_estimate_in_shadow_mode(self, workspace):
        """Shadow mode never writes anything, so any accumulated
        saved_today_eur for the day is a projection, not a measured saving,
        and must be labelled as such."""
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
        runner.tick(T0 + 60)
        assert runner.state.saved_today_is_estimate is True
        _, attributes = hass.published["sensor.dao_fast_control"]
        assert attributes["saved_today_is_estimate"] is True

    def test_saved_today_is_not_an_estimate_in_active_mode(self, workspace):
        make_plan(workspace["plan_path"], battery_w=0.0)
        hass = FakeHass(
            {
                "sensor.p1_power": "3500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        runner = FastControlRunner(hass, make_config(mode="active"), **workspace)
        runner.tick(T0 + 60)
        assert runner.state.saved_today_is_estimate is False
        _, attributes = hass.published["sensor.dao_fast_control"]
        assert attributes["saved_today_is_estimate"] is False

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

    def test_switching_off_during_an_override_restores_the_plan(self, workspace):
        """The layer must not leave the inverter on its overridden setpoint
        with the optimizer's stop moment cleared when it is turned off."""
        make_plan(workspace["plan_path"], battery_w=0.0)
        hass = FakeHass(
            {
                "input_select.fast_mode": "active",
                "sensor.p1_power": "3500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        runner = FastControlRunner(
            hass, make_config(mode="input_select.fast_mode"), **workspace
        )
        decision = runner.tick(T0 + 60)
        assert decision is not None and decision.override
        assert hass.values["input_number.feedin"] == pytest.approx(-3500.0, abs=1.0)
        assert runner.state.battery(0).override_active

        hass.states["input_select.fast_mode"] = "off"
        assert runner.tick(T0 + 120) is None

        assert hass.values["input_number.feedin"] == pytest.approx(0.0, abs=1.0)
        assert hass.options["input_select.mode"] == "Aan"
        assert ("set_datetime", {"entity_id": "input_datetime.stop",
                                 "datetime": "2000-01-01 00:00:00"}) in hass.services
        assert not runner.state.battery(0).override_active
        assert not runner.state.battery(0).stop_inverter_cleared
        kinds = [e["kind"] for e in runner.state.events]
        assert kinds == [
            "override_start",
            "setpoint_change",
            "mode_change",
            "override_end",
            "setpoint_change",
        ]
        # Once released, further ticks in off mode write nothing.
        hass.values.clear()
        assert runner.tick(T0 + 180) is None
        assert hass.values == {}

    def test_switching_off_from_shadow_writes_nothing(self, workspace):
        make_plan(workspace["plan_path"], battery_w=0.0)
        hass = FakeHass(
            {
                "input_select.fast_mode": "shadow",
                "sensor.p1_power": "3500",
                "sensor.battery_power": "0",
                "sensor.soc": "60",
            }
        )
        runner = FastControlRunner(
            hass, make_config(mode="input_select.fast_mode", **{"max sensor age": 999999}),
            **workspace,
        )
        decision = runner.tick(T0 + 60)
        assert decision is not None and decision.override
        hass.states["input_select.fast_mode"] = "off"
        assert runner.tick(T0 + 120) is None
        assert hass.values == {}
        assert hass.options == {}
        assert not runner.state.battery(0).override_active

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
        # Distinct from "sensor_stale": the grid sensor itself is fine here,
        # only the plan is too old to be trusted.
        assert decision.reason == "plan_stale"
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


class RecordingFallback:
    def __init__(self, result="requested", path=None):
        self.requests = []
        self.result = result
        self.path = path

    def request(self, request):
        # Sink-I/O gebeurt na vrijgave van het publicatieslot.
        if self.path is not None:
            with publication.publication_lock(self.path):
                pass
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        if isinstance(self.result, str):
            return publication.FallbackReceipt(self.result, "synthetic gateway acceptance only")
        return self.result


class F2bHass(FakeHass):
    def __init__(self, failures=None, readback=None):
        super().__init__({"sensor.p1_power": "1500", "sensor.soc": "60",
                          "sensor.battery_power": "0", **(readback or {})})
        self.failures = {key: list(value) for key, value in (failures or {}).items()}
        self.attempts = []

    def attempt(self, helper):
        self.attempts.append(helper)
        outcomes = self.failures.get(helper, [True])
        ok = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if not ok:
            raise RuntimeError("synthetic failed helper")

    def set_value(self, entity_id, value):
        if entity_id == "input_number.feedin":
            self.attempt("setpoint")
        super().set_value(entity_id, value)

    def select_option(self, entity_id, option):
        self.attempt("mode")
        super().select_option(entity_id, option)

    def call_service(self, service, **kwargs):
        self.attempt("stop_inverter")
        super().call_service(service, **kwargs)


@pytest.mark.parametrize("failures,status", [
    ({}, "confirmed"),
    ({"mode": [False, True]}, "confirmed"),
    ({"mode": [False]}, "unknown"),
    ({"setpoint": [False], "mode": [False], "stop_inverter": [False]}, "unconfirmed"),
])
def test_f2b_runner_updates_accepted_facts_only_on_confirmation(workspace, failures, status):
    make_plan(workspace["plan_path"])
    hass, sink = F2bHass(failures), RecordingFallback()
    runner = FastControlRunner(hass, make_config(), fallback=sink, **workspace)
    controller = runner.state.battery(0)
    controller.last_command_w, controller.last_change_ts, controller.last_command_ts = -250, T0 - 20, T0 - 20
    controller.held_setpoint_w = -250
    controller.stop_inverter_cleared = True
    before = (controller.last_command_w, controller.last_command_ts)
    decision = runner.tick(T0 + 60)
    after = (controller.last_command_w, controller.last_command_ts)
    assert controller.last_change_ts == T0 + 60
    if status == "confirmed":
        assert after == (-1500, T0 + 60)
        assert controller.stop_inverter_cleared is True
        assert controller.held_setpoint_w == -1500
        assert sink.requests == []
    else:
        assert after == before
        assert controller.override_active is controller.stop_inverter_cleared is True
        assert controller.held_setpoint_w == -250
        assert sink.requests[0].reason == f"actuation_{status}"
        assert runner.fallback_pending is True
    receipt, = runner.last_receipts
    assert receipt.status == status
    assert decision.as_attributes()["batteries"][0]["actuation_status"] == status
    assert runner.state.last_decision["batteries"][0]["actuation_status"] == status
    for helper in ("setpoint", "mode", "stop_inverter"):
        assert hass.attempts.count(helper) == (2 if helper in failures else 1)


def test_f2b_readback_confirmation_reaches_runner_state(workspace):
    make_plan(workspace["plan_path"])
    hass = F2bHass({"mode": [False]}, {"input_select.mode": "Aan"})
    sink = RecordingFallback()
    runner = FastControlRunner(hass, make_config(), fallback=sink, **workspace)
    runner.tick(T0 + 60)
    assert runner.state.battery(0).last_command_w == -1500
    assert runner.last_receipts[0].status == "confirmed"
    assert hass.attempts.count("mode") == 2
    assert hass.template_calls == 2  # initial measurements + one fresh readback
    assert sink.requests == []


@pytest.mark.parametrize("mode", ["active", "shadow"])
def test_f2b_no_writes_preserve_accepted_state_and_clear_this_tick_receipts(workspace, mode):
    make_plan(workspace["plan_path"], battery_w=500)
    hass = F2bHass()
    hass.states.update({"sensor.p1_power": "500", "sensor.battery_power": "500"})
    runner = FastControlRunner(hass, make_config(mode=mode, **{"deadband": 150, "min benefit": 1e9}), **workspace)
    controller = runner.state.battery(0)
    controller.last_command_w, controller.last_command_ts, controller.last_change_ts = -250, T0 - 20, T0 - 20
    before = json.dumps({key: value for key, value in controller.to_dict().items()
                         if key != "held_setpoint_w"}, sort_keys=True)
    runner.last_receipts = ["previous tick"]
    decision = runner.tick(T0 + 60)
    assert not decision.batteries[0].write
    assert json.dumps({key: value for key, value in controller.to_dict().items()
                       if key != "held_setpoint_w"}, sort_keys=True) == before
    assert controller.held_setpoint_w == 500
    assert runner.last_receipts == []
    assert hass.attempts == []


def test_f2b_shadow_proposal_is_not_accepted(workspace):
    make_plan(workspace["plan_path"])
    runner = FastControlRunner(F2bHass(), make_config(mode="shadow"), **workspace)
    decision = runner.tick(T0 + 60)
    assert decision.batteries[0].write
    controller = runner.state.battery(0)
    assert (controller.last_command_w, controller.last_command_ts,
            controller.stop_inverter_cleared) == (0, 0, False)
    assert controller.last_change_ts == T0 + 60
    assert runner.last_receipts == []


def test_f2b_r18_shadow_projection_matches_base_six_ticks_exactly(workspace):
    make_plan(workspace["plan_path"], battery_w=-400, price=0.30)
    hass = FakeHass({"sensor.soc": "60", "sensor.p1_power": "0",
                     "sensor.battery_power": "-400"})
    runner = FastControlRunner(
        hass, make_config(mode="shadow", **{"max sensor age": 999999, "deadband": 150}),
        **workspace,
    )
    controller = runner.state.battery(0)
    controller.last_command_w = -987
    controller.last_change_ts, controller.last_command_ts = T0 - 30, T0 - 20
    accepted_before = (controller.last_command_w, controller.last_command_ts,
                       controller.stop_inverter_cleared)
    observed = []
    for tick in range(1, 7):
        hass.states["sensor.p1_power"] = str(-100 * tick)
        hass.states["sensor.battery_power"] = str(-400 + 10 * tick)
        held_before = controller.held_setpoint_w
        decision = runner.tick(T0 + 60 * tick)
        battery = decision.batteries[0]
        observed.append((battery.setpoint_w, battery.write, battery.reason))
        if not battery.write:
            assert controller.held_setpoint_w == held_before
        assert (controller.last_command_w, controller.last_command_ts,
                controller.stop_inverter_cleared) == accepted_before
        assert runner.last_receipts == []
    assert observed == [
        (-290, True, "override"), (-290, False, "deadband"),
        (-70, True, "override"), (-70, False, "deadband"),
        (150, True, "override"), (150, False, "deadband"),
    ]
    assert controller.held_setpoint_w == 150
    assert "input_number.feedin" not in hass.values
    assert hass.options == {}
    assert hass.services == []


@pytest.mark.parametrize("kind,step_s,reason", [("rate", 30, "rate_limited"),
                                              ("stop", 60, "deadband")])
def test_f2b_r20_shadow_control_law_matches_base_exactly(workspace, kind, step_s, reason):
    plan = make_plan(workspace["plan_path"], battery_w=-400, price=0.3)
    if kind == "stop":
        plan.intervals[0].battery(0).stop_inverter = "2030-01-01 00:00:00"
        write_plan(plan, workspace["plan_path"])
    hass = FakeHass({"sensor.soc": "60"})
    runner = FastControlRunner(hass, make_config(mode="shadow", **{
        "max sensor age": 999999,
        "deadband": 0 if kind == "rate" else 150,
        "min command interval": 60 if kind == "rate" else 0,
    }), **workspace)
    observed = []
    for tick in range(1, 5):
        hass.states.update({"sensor.p1_power": str(-100 * tick),
                            "sensor.battery_power": str(-400 + 10 * tick)})
        battery = runner.tick(T0 + step_s * tick).batteries[0]
        observed.append((battery.setpoint_w, battery.write, battery.reason))
    assert observed == [(-290, True, "override"), (-290, False, reason),
                        (-70, True, "override"), (-70, False, reason)]
    controller = runner.state.battery(0)
    assert controller.last_change_ts == T0 + step_s * 3
    assert controller.held_stop_inverter_cleared is True
    assert (controller.last_command_w, controller.last_command_ts,
            controller.stop_inverter_cleared) == (0, 0, False)


def test_f2b_r19_normal_restore_repeats_until_confirmed(workspace):
    plan = make_plan(workspace["plan_path"], battery_w=-400)
    plan.intervals[0].battery(0).stop_inverter = "2030-01-01 00:00:00"
    write_plan(plan, workspace["plan_path"])
    hass = F2bHass()
    hass.states["sensor.battery_power"] = "-400"
    runner = FastControlRunner(hass, make_config(**{"max sensor age": 999999, "deadband": 150}), **workspace)
    runner.tick(T0 + 60)
    hass.states["sensor.p1_power"] = "0"
    runner.tick(T0 + 180)
    hass.failures = {h: [False] for h in ("setpoint", "mode", "stop_inverter")}
    release = runner.tick(T0 + 300).batteries[0]
    assert (release.write, release.reason, release.mode, release.stop_inverter) == (
        True, "released", "Aan", "2030-01-01 00:00:00")
    assert runner.last_receipts[0].status == "unconfirmed"
    retry = runner.tick(T0 + 360).batteries[0]
    assert retry.write is True
    assert (retry.mode, retry.stop_inverter) == ("Aan", "2030-01-01 00:00:00")
    controller = runner.state.battery(0)
    assert retry.restores is True
    assert controller.restore_pending is True
    hass.failures = {}
    runner.tick(T0 + 420)
    assert controller.restore_pending is False
    assert controller.stop_inverter_cleared is False
    assert controller.held_stop_inverter_cleared is False
    assert runner.tick(T0 + 480).batteries[0].write is False


def test_f2b_r19_off_restore_retries_after_partial_override(workspace):
    make_plan(workspace["plan_path"], battery_w=-400)
    hass = F2bHass({"mode": [False]})
    runner = FastControlRunner(hass, make_config(), **workspace)
    runner.tick(T0 + 60)
    controller = runner.state.battery(0)
    assert controller.override_active is True
    assert controller.stop_inverter_cleared is False
    runner.config = make_config(mode="off")
    runner.tick(T0 + 120)
    assert runner.last_receipts[0].status == "unknown"
    runner.tick(T0 + 180)
    assert runner.last_receipts  # Not silently forgotten with the override flag.
    assert runner.last_receipts[0].status == "unknown"
    assert controller.restore_pending is True
    hass.failures = {}
    runner.tick(T0 + 240)
    assert controller.restore_pending is False


def test_f2b_r25_shadow_off_does_no_release_publish_or_persist(workspace):
    plan = make_plan(workspace["plan_path"], battery_w=-400)
    plan.intervals[0].battery(0).stop_inverter = "2030-01-01 00:00:00"
    write_plan(plan, workspace["plan_path"])
    hass = F2bHass()
    runner = FastControlRunner(hass, make_config(mode="shadow", **{
        "deadband": 150, "max sensor age": 999999,
    }), **workspace)
    for tick in range(1, 5):
        hass.states.update({"sensor.p1_power": str(-100 * tick),
                            "sensor.battery_power": str(-400 + 10 * tick)})
        runner.tick(T0 + 60 * tick)
    controller = runner.state.battery(0)
    assert (controller.override_active, controller.held_stop_inverter_cleared) == (True, True)
    assert (controller.last_command_ts, controller.stop_inverter_cleared,
            controller.restore_pending) == (0, False, False)
    assert hass.attempts == []
    snapshot = json.loads(json.dumps(runner.state.last_decision))
    persisted = Path(workspace["state_path"]).read_bytes()
    # Apart from the required withdrawal of unconfirmed ownership, all state
    # stays as at the last shadow tick (including the diagnostic snapshot).
    expected = runner.state.to_dict()
    expected["batteries"][0]["override_active"] = False
    expected["batteries"][0]["held_stop_inverter_cleared"] = False
    runner.config = make_config(mode="off")
    with (
        patch.object(FastControlPolicy, "release", autospec=True,
                     side_effect=FastControlPolicy.release) as release,
        patch.object(runner, "_publish", wraps=runner._publish) as publish,
        patch.object(runner, "_persist_state", wraps=runner._persist_state) as persist,
    ):
        for tick in range(5, 9):
            assert runner.tick(T0 + 60 * tick) is None
            # BASE has exactly zero diagnostic publications on every off tick.
            assert (publish.call_count, persist.call_count, release.call_count) == (0, 0, 0)
            assert runner.state.last_decision == snapshot
            assert runner.state.to_dict() == expected
            assert Path(workspace["state_path"]).read_bytes() == persisted
            assert hass.attempts == []


@pytest.mark.parametrize("failures,status", [
    ({}, "confirmed"),
    ({"mode": [False]}, "unknown"),
    ({h: [False] for h in ("setpoint", "mode", "stop_inverter")}, "unconfirmed"),
])
def test_f2b_r25_active_off_restores_and_only_retries_failures(workspace, failures, status):
    plan = make_plan(workspace["plan_path"], battery_w=-400)
    plan.intervals[0].battery(0).stop_inverter = "2030-01-01 00:00:00"
    write_plan(plan, workspace["plan_path"])
    hass = F2bHass()
    runner = FastControlRunner(hass, make_config(), **workspace)
    runner.tick(T0 + 60)
    assert runner.last_receipts[0].status == "confirmed"
    controller = runner.state.battery(0)
    assert (controller.last_command_ts, controller.stop_inverter_cleared) == (T0 + 60, True)
    runner.config = make_config(mode="off")
    hass.failures = failures
    with (
        patch.object(runner, "_publish", wraps=runner._publish) as publish,
        patch.object(runner, "_persist_state", wraps=runner._persist_state) as persist,
    ):
        for tick in (2, 3, 4):
            hass.attempts.clear()
            assert runner.tick(T0 + 60 * tick) is None
            if tick > 2 and status == "confirmed":
                assert runner.last_receipts == []
                assert hass.attempts == []
                assert publish.call_count == persist.call_count == 1
                continue
            assert set(hass.attempts) == {"setpoint", "mode", "stop_inverter"}
            assert runner.last_receipts[0].status == status
            assert controller.restore_pending is (status != "confirmed")
            assert controller.stop_inverter_cleared is (status != "confirmed")
            assert runner.state.last_decision["batteries"][0]["actuation_status"] == status
            assert runner.state.last_decision["mode"] == "off"
            assert load_state(workspace["state_path"]).last_decision == runner.state.last_decision
            entity = runner.config.diagnostics.entity_status
            assert hass.published[entity][1]["batteries"][0]["actuation_status"] == status
            assert publish.call_count == persist.call_count == tick - 1
            persist.assert_called_with(T0 + 60 * tick, force=True)
        if status != "confirmed":
            hass.failures = {}
            runner.tick(T0 + 300)
            assert runner.last_receipts[0].status == "confirmed"
            assert controller.restore_pending is False
            calls = publish.call_count
            runner.tick(T0 + 360)
            assert runner.last_receipts == []
            assert publish.call_count == persist.call_count == calls


@pytest.mark.parametrize("fails", [False, True])
def test_f2b_r19_override_clear_does_not_discharge_restore(workspace, fails):
    make_plan(workspace["plan_path"])
    hass = F2bHass({"mode": [False]} if fails else {})
    runner = FastControlRunner(hass, make_config(), **workspace)
    controller = runner.state.battery(0)
    controller.restore_pending = True
    decision = runner.tick(T0 + 60).batteries[0]
    assert decision.override is True
    assert decision.restores is False
    assert controller.restore_pending is True


def test_f2b_r20_denied_gate_revokes_both_stop_facts(workspace):
    runner = FastControlRunner(FakeHass(), make_config(), **workspace)
    controller = runner.state.battery(0)
    controller.stop_inverter_cleared = controller.held_stop_inverter_cleared = True
    runner.tick(T0 + 60)
    assert controller.stop_inverter_cleared is controller.held_stop_inverter_cleared is False


@pytest.mark.parametrize("all_fail,status", [(False, "unknown"), (True, "unconfirmed")])
def test_f2b_r21_failed_off_restore_updates_snapshot_and_operator(workspace, all_fail, status):
    make_plan(workspace["plan_path"], battery_w=-400)
    hass = F2bHass()
    runner = FastControlRunner(hass, make_config(), **workspace)
    runner.tick(T0 + 60)
    assert runner.state.last_decision["batteries"][0]["actuation_status"] == "confirmed"
    hass.failures = {h: [False] for h in (("setpoint", "mode", "stop_inverter") if all_fail else ("mode",))}
    runner.config = make_config(mode="off")
    runner.tick(T0 + 120)
    assert runner.state.last_decision["batteries"][0]["actuation_status"] == status
    assert runner.state.last_decision["mode"] == "off"
    assert load_state(workspace["state_path"]).last_decision == runner.state.last_decision
    entity = runner.config.diagnostics.entity_status
    assert hass.published[entity][1]["batteries"][0]["actuation_status"] == status


def test_f2b_r24_shadow_restart_uses_projection_without_accepting_it(workspace):
    make_plan(workspace["plan_path"], battery_w=-400)
    hass = FakeHass({"sensor.soc": "60", "sensor.p1_power": "-100", "sensor.battery_power": "-390"})
    shadow = FastControlRunner(hass, make_config(mode="shadow", **{"deadband": 150}), **workspace)
    shadow.tick(T0 + 60)
    save_state(shadow.state, workspace["state_path"])
    active = FastControlRunner(hass, make_config(**{"max sensor age": 999999, "deadband": 150}), **workspace)
    controller = active.state.battery(0)
    assert controller.held_setpoint_w == -290
    assert (controller.last_command_w, controller.last_command_ts,
            controller.stop_inverter_cleared) == (0, 0, False)
    assert controller.held_stop_inverter_cleared is True
    hass.states.update({"sensor.p1_power": "-200", "sensor.battery_power": "-380"})
    first = active.tick(T0 + 120).batteries[0]
    assert (first.setpoint_w, first.write, first.reason) == (-290, False, "deadband")
    assert active.last_receipts == []
    assert controller.last_command_ts == 0
    hass.states.update({"sensor.p1_power": "-300", "sensor.battery_power": "-370"})
    active.tick(T0 + 180)
    assert active.last_receipts[0].status == "confirmed"
    assert (controller.last_command_w, controller.last_command_ts,
            controller.stop_inverter_cleared) == (-70, T0 + 180, True)


def test_f2b_r22_invalid_identity_fails_before_helpers_and_escalates(workspace):
    """A v2 plan never reaches the legacy helper route; a v1 plan does."""
    from dataclasses import replace

    from dao.prog.fastctrl.policy import BatteryDecision, Decision

    plan = ready_plan()
    plan.specs[0] = replace(plan.specs[0], setpoint_entity="input_number.feedin")
    plan.financial_context = None
    hass, sink = F2bHass(), RecordingFallback()
    runner = FastControlRunner(hass, make_config(), fallback=sink, **workspace)
    decision = Decision(T0, [BatteryDecision(0, "accu", -1000, 0, True, True, "override")])
    runner._actuate(decision, plan, "active")
    # (1) The financial route refuses a v2 plan without a financial context: no
    # helper call, no receipt and no legacy fallback.
    assert hass.attempts == []
    assert runner.last_receipts == []
    assert runner.last_actuation.allowed is False
    assert runner.last_actuation.reason == "missing_financial_context"
    assert sink.requests == []
    assert decision.as_attributes()["batteries"][0]["actuation_status"] is None
    # (2) A valid v1 plan reaches the legacy helper route: the helpers are
    # called, a legacy receipt is produced and the financial route never ran.
    legacy = make_plan(workspace["plan_path"])
    legacy_hass = F2bHass()
    legacy_runner = FastControlRunner(legacy_hass, make_config(), fallback=sink,
                                      **workspace)
    legacy_runner._actuate(decision, legacy, "active")
    assert legacy_hass.attempts == ["setpoint"]
    assert len(legacy_runner.last_receipts) == 1
    assert legacy_runner.last_receipts[0].plan_identity == "v1:1700000000"
    assert legacy_runner.last_receipts[0].status == "confirmed"
    # The financial route never ran for the v1 plan: there is no verdict.
    assert legacy_runner.last_actuation is None
    assert sink.requests == []


def test_f2b_missing_helper_entities_do_not_accept_proposed_state(workspace):
    from dataclasses import replace

    plan = make_plan(workspace["plan_path"])
    plan.specs[0] = replace(plan.specs[0], setpoint_entity=None, mode_entity=None, stop_inverter_entity=None)
    write_plan(plan, workspace["plan_path"])
    runner = FastControlRunner(F2bHass(), make_config(), **workspace)
    decision = runner.tick(T0 + 60)
    assert decision.batteries[0].write
    assert decision.batteries[0].accepted is not None
    controller = runner.state.battery(0)
    assert (controller.last_command_w, controller.last_command_ts,
            controller.stop_inverter_cleared) == (0, 0, False)
    assert controller.last_change_ts == T0 + 60
    assert runner.last_receipts == []


@pytest.mark.parametrize("off", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_f2b_release_uses_the_same_acceptance_route(workspace, off, fails):
    make_plan(workspace["plan_path"], battery_w=-400)
    hass, sink = F2bHass(), RecordingFallback()
    runner = FastControlRunner(hass, make_config(), fallback=sink, **workspace)
    runner.tick(T0 + 60)
    controller = runner.state.battery(0)
    before = (controller.last_command_w, controller.last_command_ts)
    assert controller.stop_inverter_cleared
    hass.failures = {"mode": [False]} if fails else {}
    if off:
        runner.config = make_config(mode="off")
    else:
        hass.states["sensor.p1_power"] = "unavailable"
    runner.tick(T0 + 120)
    if fails:
        assert (controller.last_command_w, controller.last_command_ts) == before
        assert sink.requests[0].reason == "actuation_unknown"
        assert controller.held_setpoint_w == before[0]
        assert controller.stop_inverter_cleared is True
        # A failed restore must be retried next tick, even in mode off.
        runner.tick(T0 + 121)
        assert runner.last_receipts
        assert runner.last_receipts[0].status == "unknown"
        assert hass.attempts.count("mode") == 5  # engagement + 2 + 2 attempts
    else:
        assert (controller.last_command_w, controller.last_command_ts) == (-400, T0 + 120)
        assert controller.held_setpoint_w == -400
        assert controller.stop_inverter_cleared is False
    assert controller.override_active is False
    assert runner.last_receipts[0].status == ("unknown" if fails else "confirmed")


def test_f2b_event_baseline_uses_held_setpoint_not_last_accepted(workspace):
    from dao.prog.fastctrl.runner import EVENTS_KIND_SETPOINT

    make_plan(workspace["plan_path"], battery_w=-250)
    hass = F2bHass()
    hass.states.update({"sensor.p1_power": "0", "sensor.battery_power": "-250"})
    runner = FastControlRunner(hass, make_config(**{"deadband": 150}), **workspace)
    controller = runner.state.battery(0)
    controller.last_command_w, controller.held_setpoint_w = 777, -250
    decision = runner.tick(T0 + 60)
    assert decision.batteries[0].setpoint_w == -250
    assert decision.batteries[0].write is False
    assert not [e for e in runner.state.events if e["kind"] == EVENTS_KIND_SETPOINT]
    # The direct event path (e.g. off-release) has the same held baseline.
    runner._last_setpoints = ()
    runner._record_events(decision, "active")
    assert not [e for e in runner.state.events if e["kind"] == EVENTS_KIND_SETPOINT]


def test_f2b_actuation_fallback_shares_monotonic_throttle_with_gate(workspace, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("dao.prog.fastctrl.runner.time.monotonic", lambda: clock[0])
    make_plan(workspace["plan_path"])
    sink = RecordingFallback()
    runner = FastControlRunner(F2bHass({"mode": [False]}), make_config(interval=15),
                               fallback=sink, **workspace)
    runner.tick(T0 + 60)
    runner.tick(T0 + 61)
    assert [r.reason for r in sink.requests] == ["actuation_unknown"]
    assert runner.last_receipts[0].timestamp == T0 + 61
    Path(workspace["plan_path"]).unlink()
    runner.tick(T0 + 62)
    assert len(sink.requests) == 1
    assert runner.last_receipts == []
    clock[0] = 115.0
    runner.tick(T0 + 63)
    assert len(sink.requests) == 2



@pytest.mark.parametrize("artifact", ["blocked", "missing", "corrupt", "expired_v1", "expired_v2", "ready_v2", "v1_at_gate"])
@pytest.mark.parametrize("mode", ["off", "shadow", "active"])
def test_denial_never_restores_old_power_or_accounts_benefit(workspace, artifact, mode):
    path = Path(workspace["plan_path"])
    make_plan(path, battery_w=-2300)
    hass = FakeHass()
    sink = RecordingFallback(path=path)
    runner = FastControlRunner(hass, make_config(mode=mode, **{"grid power": {}}),
                               gate_policy=publication.GatePolicy(T0 + (60 if artifact == "v1_at_gate" else 100000)),
                               fallback=sink, **workspace)
    runner.plan()  # Cache the old non-zero plan, including its entity names.
    runner.state.battery(0).override_active = True
    runner.state.battery(0).stop_inverter_cleared = True
    runner.state.battery(0).last_command_w = -3500
    runner.state.saved_today_eur = 1.25
    runner.state.last_tick_ts = T0
    runner.state.last_decision = {"benefit_eur_h": 99}
    now = T0 + 60
    if artifact == "blocked":
        with publication.publication_lock(path) as lease:
            publication.invalidate_plan(lease=lease, now_ts=now, reason="correction")
    elif artifact == "missing":
        path.unlink()
    elif artifact == "corrupt":
        path.write_text("null")
    elif artifact == "expired_v1":
        now = T0 + 8 * 900
    elif artifact in ("expired_v2", "ready_v2"):
        plan = ready_plan()
        path.write_text(json.dumps(plan.to_dict()))
        now = plan.financial_context.not_after_ts if artifact == "expired_v2" else plan.created_ts

    assert runner.tick(now) is None
    request, = sink.requests
    assert (request.power_w, request.mode, request.inhibit_restart, request.requested_ts) == (
        0, "configured_off", True, now,
    )
    if artifact == "ready_v2":
        assert request.reason == "actuation_gates_unproven"
    elif artifact == "expired_v1":
        assert request.reason == "plan_outside_horizon"
    assert runner.last_fallback.status == "requested"
    assert runner.fallback_pending is True
    assert runner.state.last_decision is None
    assert runner.state.saved_today_eur == 1.25
    assert hass.values == hass.options == hass.switches == hass.published == {}
    assert hass.services == []
    assert hass.template_calls == hass.state_calls == 0


@pytest.mark.parametrize("mode", ["off", "shadow", "active"])
@pytest.mark.parametrize("status", ["draft", "confirmed"])
def test_bootstrap_date_gate_is_independent_of_settlement_mode_and_status(workspace, mode, status):
    if mode == "active" and status == "draft":
        with pytest.raises(ValueError, match="draft"):
            pricing_config(mode=mode, status=status)
        return
    gate = 1798758000  # 2027-01-01 00:00 Europe/Amsterdam.
    plan = make_plan(workspace["plan_path"], created_ts=gate - 900)
    for i, interval in enumerate(plan.intervals):
        interval.start_ts = gate - 900 + i * 900
        interval.end_ts = interval.start_ts + 900
    write_plan(plan, workspace["plan_path"])
    hass = FakeHass({"sensor.p1_power": "500", "sensor.soc": "60"})
    hass.config.prices = pricing_config(mode=mode, status=status)
    runner = FastControlRunner(hass, make_config(**{"max sensor age": 999999999}), **workspace)
    assert runner.tick(gate - 1) is not None
    hass.values.clear()
    hass.options.clear()
    hass.published.clear()
    assert runner.tick(gate) is None
    assert runner.last_fallback.status == "unconfirmed"
    assert hass.values == hass.options == hass.published == {}
    # Neither deleting nor restoring v1 after the date recovers legacy rights.
    Path(workspace["plan_path"]).unlink()
    assert runner.tick(gate + 1) is None
    write_plan(plan, workspace["plan_path"])
    assert runner.tick(gate + 2) is None
    assert hass.values == {}


@pytest.mark.parametrize("result", [None, False, {"status": "acknowledged"}, RuntimeError("offline"), "acknowledged"])
def test_fallback_failure_and_gateway_acceptance_do_not_claim_stopping(workspace, result, caplog):
    hass = FakeHass()
    sink = None if result is None else RecordingFallback(result)
    runner = FastControlRunner(hass, make_config(mode="off"), fallback=sink, **workspace)
    assert runner.tick(T0) is None
    assert runner.last_fallback.status == ("acknowledged" if result == "acknowledged" else "unconfirmed")
    assert runner.fallback_pending is True
    assert hass.values == hass.options == hass.published == {}
    if result != "acknowledged":
        assert "unconfirmed" in caplog.text


def test_fallback_retry_uses_monotonic_time_not_wall_clock(workspace, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("dao.prog.fastctrl.runner.time.monotonic", lambda: clock[0])
    sink = RecordingFallback(RuntimeError("offline"))
    runner = FastControlRunner(FakeHass(), make_config(interval=15), fallback=sink, **workspace)
    runner.tick(T0)
    for wall in (T0 + 99999, T0 - 99999, T0):
        runner.tick(wall)
    assert len(sink.requests) == 1
    clock[0] = 114.999
    runner.tick(T0)
    assert len(sink.requests) == 1
    clock[0] = 115.0
    runner.tick(T0)
    assert len(sink.requests) == 2


def test_restart_and_same_mtime_cannot_reuse_cached_authority(workspace):
    path = Path(workspace["plan_path"])
    make_plan(path, battery_w=-2300)
    hass = FakeHass({"sensor.p1_power": "500", "sensor.soc": "60"})
    runner = FastControlRunner(hass, make_config(), **workspace)
    assert runner.tick(T0 + 1) is not None
    previous = path.stat()
    replacement = path.with_suffix(".replacement")
    replacement.write_text("[]")
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    os.replace(replacement, path)
    hass.values.clear()
    hass.options.clear()
    hass.published.clear()
    assert runner.tick(T0 + 2) is None
    restarted = FastControlRunner(hass, make_config(), **workspace)
    assert restarted.tick(T0 + 3) is None
    assert restarted.last_fallback.status == "unconfirmed"
    assert hass.values == hass.options == hass.published == {}


def test_lockbusy_and_read_exception_use_no_cached_plan(workspace, monkeypatch):
    make_plan(workspace["plan_path"], battery_w=-2300)
    sink = RecordingFallback()
    runner = FastControlRunner(FakeHass(), make_config(), fallback=sink, **workspace)
    runner.plan()
    with publication.publication_lock(workspace["plan_path"]):
        assert runner.tick(T0) is None
    assert len(sink.requests) == 1
    assert runner.state.last_decision is None
    monkeypatch.setattr("dao.prog.fastctrl.runner.read_publication", lambda path: (_ for _ in ()).throw(OSError("read")))
    assert runner.tick(T0 + 15) is None
    assert runner.last_fallback is not None


def test_missing_configuration_is_not_a_broken_configuration(workspace):
    # No settlement/prices is ordinary legacy; an invalid loaded config is not.
    runner = FastControlRunner(FakeHass(), make_config(), **workspace)
    assert runner.tick(T0) is None
    assert runner.last_fallback.status == "unconfirmed"
    for config in (None, SimpleNamespace(prices="unreadable"), SimpleNamespace(prices={"settlement": "broken"})):
        hass = FakeHass()
        hass.config = config
        with pytest.raises((ValueError, TypeError, AttributeError)):
            FastControlRunner(hass, make_config(), **workspace)


def test_missing_plan_and_default_sensorless_configuration_are_explicitly_unconfirmed(workspace):
    runner = FastControlRunner(FakeHass(), **workspace)
    assert runner.tick(T0) is None
    assert runner.last_fallback.status == "unconfirmed"
    assert runner.fallback_pending is True


def test_unreadable_prices_property_is_not_treated_as_missing_configuration(workspace):
    class BrokenConfig:
        @property
        def prices(self):
            raise AttributeError("pricing load failed")
    hass = FakeHass()
    hass.config = BrokenConfig()
    with pytest.raises((ValueError, AttributeError)):
        FastControlRunner(hass, make_config(), **workspace)


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


@pytest.mark.parametrize("mode", ["off", "shadow", "active"])
def test_f2c5_the_public_entrypoint_still_refuses_a_ready_v2_plan(workspace, mode):
    """The financial consumer is not reachable from the public route."""
    path = Path(workspace["plan_path"])
    plan = ready_plan()
    path.write_text(json.dumps(plan.to_dict()))
    hass = FakeHass({"sensor.p1_power": "500", "sensor.soc": "60"})
    sink = RecordingFallback(path=path)
    runner = FastControlRunner(hass, make_config(mode=mode), fallback=sink, **workspace)
    assert runner.tick(plan.created_ts) is None
    assert runner.last_gate.allowed is False
    assert runner.last_gate.reason == "actuation_gates_unproven"
    # The consumer was never reached: no verdict, no receipt, no helper call.
    assert runner.last_actuation is None
    assert runner.last_receipts == []
    assert hass.values == hass.options == hass.switches == hass.published == {}
    assert hass.services == []
    assert hass.template_calls == hass.state_calls == 0
    # A restart does not recover the right either.
    restarted = FastControlRunner(hass, make_config(mode=mode), fallback=sink, **workspace)
    assert restarted.tick(plan.created_ts + 1) is None
    assert restarted.last_gate.reason == "actuation_gates_unproven"
    assert restarted.last_actuation is None
    assert restarted.last_receipts == []
    assert hass.values == hass.options == hass.switches == hass.published == {}


def test_f2c5_the_v2_route_refuses_without_an_owner_and_writes_nothing(workspace, monkeypatch):
    from dao.prog.fastctrl.actuator import LegacyActuationReceipt
    from dao.prog.fastctrl.policy import BatteryDecision, Decision

    calls = []
    monkeypatch.setattr("dao.prog.fastctrl.runner.execute_battery_command",
                        lambda *args, **kwargs: calls.append(args))
    plan = ready_plan()
    hass = FakeHass()
    runner = FastControlRunner(hass, make_config(), **workspace)
    decision = Decision(T0, [BatteryDecision(0, "accu", -1000, -2000, True, True, "override")])
    receipts = runner._actuate(decision, plan, "active")
    assert receipts == ()
    assert runner.last_receipts == []
    assert runner.last_actuation.allowed is False
    assert runner.last_actuation.reason == "owner_unavailable"
    # The legacy helper route is never used in the financial regime.
    assert calls == []
    assert not any(isinstance(receipt, LegacyActuationReceipt) for receipt in receipts)
    assert hass.values == hass.options == hass.switches == hass.published == {}
    assert hass.services == []


def test_f2c5_the_generic_gateway_is_explicitly_legacy_only():
    from dao.prog.fastctrl import actuator as actuator_module

    # Deliberate assertion RED, never a collection/import error.
    assert hasattr(actuator_module, "FinancialActuationAdapter"), (
        "F2c5 nominal adapter base class missing")
    assert hasattr(actuator_module, "classify_actuation_adapter"), (
        "F2c5 adapter classification missing")
    assert HomeAssistantGateway.actuation_adapter_class == (
        actuator_module.LEGACY_ONLY_ADAPTER)
    assert actuator_module.classify_actuation_adapter(
        HomeAssistantGateway(FakeHass())) == actuator_module.LEGACY_ONLY_ADAPTER
    # The label is not a release: the generic gateway never inherits the
    # financial adapter contract.
    assert not isinstance(HomeAssistantGateway(FakeHass()),
                          actuator_module.FinancialActuationAdapter)


@pytest.mark.parametrize("mode", ["off", "shadow", "active"])
def test_f2c5_the_production_route_never_reaches_the_execution_boundary(workspace, monkeypatch, mode):
    """The public route never calls the mechanical boundary itself."""
    from dao.prog.fastctrl import execution as execution_boundary

    path = Path(workspace["plan_path"])
    plan = ready_plan()
    path.write_text(json.dumps(plan.to_dict()))
    reached = []
    monkeypatch.setattr(execution_boundary, "owner_section",
                        lambda *args, **kwargs: reached.append("owner_section"))
    monkeypatch.setattr(execution_boundary, "send_locked",
                        lambda *args, **kwargs: reached.append("send_locked"))
    hass = FakeHass({"sensor.p1_power": "500", "sensor.soc": "60"})
    runner = FastControlRunner(hass, make_config(mode=mode), **workspace)
    assert runner.tick(plan.created_ts) is None
    assert runner.last_gate.reason == "actuation_gates_unproven"
    assert reached == []
    # A restart does not reach the boundary either.
    restarted = FastControlRunner(hass, make_config(mode=mode), **workspace)
    assert restarted.tick(plan.created_ts + 1) is None
    assert restarted.last_gate.reason == "actuation_gates_unproven"
    assert reached == []


def test_f2c5_the_v1_diagnostics_keep_writing_an_aliased_entity(workspace):
    """The alias rule is a financial-route rule; the v1 route is unchanged."""
    make_plan(workspace["plan_path"], battery_w=-400)
    hass = FakeHass({"sensor.soc": "60", "sensor.p1_power": "-100",
                     "sensor.battery_power": "-390"})
    config = make_config(diagnostics={"entity setpoint": "input_select.mode"})
    runner = FastControlRunner(hass, config, **workspace)
    runner.tick(T0 + 60)
    # The v1 diagnostics write the aliased entity, exactly as before: the
    # overlap check belongs to the financial route only.
    assert "input_select.mode" in hass.values
    assert hass.values["input_select.mode"] == pytest.approx(
        runner.state.last_decision["batteries"][0]["setpoint_w"], abs=1.0)
