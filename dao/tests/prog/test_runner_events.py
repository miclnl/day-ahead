import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import pytest

from dao.prog.config.models.fastcontrol import FastControlConfig
from dao.prog.fastctrl.plan import (
    BatteryPlanStep,
    BatterySpec,
    FastPlan,
    PlanInterval,
    write_plan,
)
from dao.prog.fastctrl.policy import (
    BatteryDecision,
    ControllerState,
    Decision,
)
from dao.prog.fastctrl.runner import FastControlRunner, save_state

T0 = 1_700_000_000


@pytest.fixture
def workspace(tmp_path):
    return {
        "plan_path": str(tmp_path / "fast_plan.json"),
        "state_path": str(tmp_path / "fast_state.json"),
    }


@pytest.mark.parametrize("site", [.15, -.10, None])
def test_f3b6_event_uses_site_estimate_and_nullable_attributes(workspace, site):
    runner = _make_runner("active", **workspace)
    batteries = [BatteryDecision(i, name, power, 0, True, True, "override",
                                 benefit_eur_h=value)
                 for i, name, power, value in ((0, "A", -1500, .35), (1, "B", 1000, -.20))]
    decision = Decision(123, batteries=batteries, benefit_eur_h=site)
    runner._record_events(decision, "active", (0, 0))
    assert runner.state.events
    for event in runner.state.events:
        assert event["benefit_eur_h"] == site
        assert event["ts"] == 123
        assert event["economics_version"] == "fast_economics_v1"
        assert "economics_status" in event and "economics_kind" in event
        assert event["proposal_benefit_eur_h"] == 0
    attributes = decision.as_attributes()
    assert attributes["benefit_eur_h"] == site
    assert [b["benefit_eur_h"] for b in attributes["batteries"]] == [.35, -.20]


def test_controller_state_roundtrip_includes_new_fields():
    state = ControllerState()
    state.last_decision = {"ts": 1.0, "reason": "plan", "override": False}
    state.events = [{"ts": 1.0, "kind": "override_start"}]
    state.daily_extra_throughput_used = 1.7
    state.energy_budget_used = 0.3

    serialised = state.to_dict()
    assert "last_decision" in serialised
    assert serialised["last_decision"] == {"ts": 1.0, "reason": "plan", "override": False}
    assert serialised["events"] == [{"ts": 1.0, "kind": "override_start"}]
    assert serialised["daily_extra_throughput_used"] == 1.7
    assert serialised["energy_budget_used"] == 0.3

    restored = ControllerState.from_dict(serialised)
    assert {"ts": 1.0, "reason": "plan", "override": False}.items() <= restored.last_decision.items()
    assert {"ts": 1.0, "kind": "override_start"}.items() <= restored.events[0].items()
    assert len(restored.events) == 1
    assert restored.events[0]["historical_economics_status"] == "legacy_estimate_unverified"
    assert restored.last_decision["historical_economics_status"] == "legacy_estimate_unverified"
    assert restored.daily_extra_throughput_used == 1.7
    assert restored.energy_budget_used == 0.3


@dataclass
class FakeRunner:
    state: ControllerState
    last_mode: Optional[str] = None
    last_overrides: tuple = ()
    last_setpoints: tuple = ()

    def append(self, decision, mode):
        new_overrides = tuple(b.override for b in decision.batteries)
        new_setpoints = tuple(b.setpoint_w for b in decision.batteries)
        events = self.state.events

        if mode != self.last_mode:
            events.append({"ts": decision.timestamp, "kind": "mode_change", "mode": mode})
        if self.last_overrides and any(self.last_overrides) and not any(new_overrides):
            events.append({"ts": decision.timestamp, "kind": "override_end", "mode": mode})
        elif (not self.last_overrides or not any(self.last_overrides)) and any(new_overrides):
            events.append({"ts": decision.timestamp, "kind": "override_start", "mode": mode})

        self.last_mode = mode
        self.last_overrides = new_overrides
        self.last_setpoints = new_setpoints


def test_runner_records_events():
    state = ControllerState()
    # Pre-seed last_mode so the first tick isn't logged as a mode_change from None.
    runner = FakeRunner(state, last_mode="shadow")

    plan_decision = Decision(
        timestamp=1.0,
        reason="plan",
        batteries=[BatteryDecision(index=0, name="b1", setpoint_w=0, plan_w=0, write=False, override=False, reason="plan")],
    )
    override_decision = Decision(
        timestamp=2.0,
        reason="override",
        batteries=[BatteryDecision(index=0, name="b1", setpoint_w=-2500, plan_w=0, write=True, override=True, reason="deadband")],
    )

    runner.append(plan_decision, "shadow")
    runner.append(override_decision, "shadow")  # override_start
    runner.append(plan_decision, "shadow")      # override_end

    assert [e["kind"] for e in state.events] == ["override_start", "override_end"]


class _StubHass:
    """Minimal hass surface that satisfies FastControlRunner.__init__.

    _record_events never touches the gateway, so we do not need a real template
    API here. Anything else that __init__ reads is read through the stub too.
    """

    def __init__(self):
        self.time_zone = "Europe/Amsterdam"
        self.config = type("Config", (), {"fast_control": None})()

    def render_template(self, template):  # pragma: no cover - never reached
        return "[]"


def _make_runner(mode: str, state_path: str, plan_path: str = "/nonexistent.json") -> FastControlRunner:
    config = FastControlConfig.model_validate(
        {
            "mode": mode,
            "grid power": {"entity": "sensor.p1_power"},
        }
    )
    return FastControlRunner(_StubHass(), config, plan_path=plan_path, state_path=state_path)


def test_record_events_detects_a_mode_change(workspace):
    """Regression: tick() used to mutate _last_mode before _record_events ran,
    so mode_change events were silently swallowed. _record_events now owns the
    bookkeeping, and the first call with the starting mode must not emit one.
    """
    runner = _make_runner(mode="shadow", state_path=workspace["state_path"])
    # _initial_mode() resolved "shadow", so the ring buffer is empty so far.
    assert runner._last_mode == "shadow"

    neutral = Decision(
        timestamp=1.0,
        reason="plan",
        batteries=[
            BatteryDecision(
                index=0, name="b1", setpoint_w=0, plan_w=0,
                write=False, override=False, reason="plan",
            )
        ],
    )

    # First call matches the initial mode: no event.
    runner._record_events(neutral, "shadow")
    assert runner.state.events == []

    # Mode flips: exactly one mode_change must be appended.
    runner._record_events(neutral, "active")
    assert [e["kind"] for e in runner.state.events] == ["mode_change"]
    assert runner.state.events[0]["mode"] == "active"


def test_record_events_still_handles_override_and_setpoint_transitions(workspace):
    """Cover the other three event kinds on the same instance, to make sure the
    bookkeeping for overrides and setpoints is not affected by the mode_change
    fix.
    """
    runner = _make_runner(mode="active", state_path=workspace["state_path"])
    assert runner._last_mode == "active"

    neutral = Decision(
        timestamp=1.0,
        reason="plan",
        batteries=[
            BatteryDecision(
                index=0, name="b1", setpoint_w=0, plan_w=0,
                write=False, override=False, reason="plan",
            )
        ],
    )
    override_on = Decision(
        timestamp=2.0,
        reason="override",
        batteries=[
            BatteryDecision(
                index=0, name="b1", setpoint_w=-2500, plan_w=0,
                write=True, override=True, reason="deadband",
            )
        ],
    )
    shifted = Decision(
        timestamp=3.0,
        reason="plan",
        batteries=[
            BatteryDecision(
                index=0, name="b1", setpoint_w=500, plan_w=0,
                write=False, override=False, reason="plan",
            )
        ],
    )

    runner._record_events(neutral, "active")      # no event
    runner._record_events(override_on, "active")  # override_start + setpoint_change
    runner._record_events(override_on, "active")  # already active, no event
    runner._record_events(neutral, "active")      # override_end + setpoint_change
    runner._record_events(shifted, "active")      # setpoint_change

    assert [e["kind"] for e in runner.state.events] == [
        "override_start",
        "setpoint_change",
        "override_end",
        "setpoint_change",
        "setpoint_change",
    ]


def test_runner_persists_events():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fast_state.json")
        state = ControllerState()
        state.events.append({"ts": 1.0, "kind": "override_start", "mode": "shadow"})
        save_state(state, path)

        with open(path, "r") as handle:
            payload = json.load(handle)

        assert payload["events"] == [{"ts": 1.0, "kind": "override_start", "mode": "shadow"}]


def test_refresh_budget_aggregates_sums_per_battery_deviation():
    """refresh_budget_aggregates must roll per-battery deviation into the top-level budget fields."""
    state = ControllerState()
    state.battery(0).daily_deviation_kwh = 1.5
    state.battery(0).interval_deviation_kwh = 0.2
    state.battery(1).daily_deviation_kwh = 2.0
    state.battery(1).interval_deviation_kwh = -0.1  # charge side of the cycle

    state.refresh_budget_aggregates()

    assert state.daily_extra_throughput_used == pytest.approx(3.5)   # 1.5 + 2.0
    assert state.energy_budget_used == pytest.approx(0.3)            # abs(0.2) + abs(-0.1)


def _stub_plan() -> FastPlan:
    """Minimal one-battery plan that satisfies tick()'s early-return guards."""
    return FastPlan(
        created_ts=T0,
        interval_s=900,
        specs=[
            BatterySpec(
                name="accu",
                capacity_kwh=10.0,
                max_charge_w=5000.0,
                max_discharge_w=5000.0,
                soc_entity="sensor.soc",
            ),
        ],
        intervals=[
            PlanInterval(
                start_ts=T0,
                end_ts=T0 + 900,
                price_import=0.30,
                price_export=0.12,
                grid_w=500.0,
                house_w=500.0,
                batteries=[
                    BatteryPlanStep(
                        ac_power_w=0.0,
                        soc_begin=60.0,
                        soc_end=60.0,
                    )
                ],
            )
        ],
    )


def test_tick_refreshes_budget_aggregates(workspace, monkeypatch):
    """Regression: the per-battery deviation roll-up must happen from tick(),
    not only on tests that call refresh_budget_aggregates directly. Removing
    the call in runner.tick would leave the web UI gauges stale.
    """
    runner = _make_runner(mode="active", **workspace)
    write_plan(_stub_plan(), workspace["plan_path"])
    runner.state.day_key = "2023-11-14"
    runner.state.plan_created_ts = runner.state.interval_start_ts = T0
    runner.state.battery(0).daily_deviation_kwh = 1.5
    runner.state.battery(0).interval_deviation_kwh = 0.2

    # De echte lokale publicatiecontrole behouden; alleen HA-I/O isoleren.
    monkeypatch.setattr(runner.gateway, "read", lambda ids: {})
    monkeypatch.setattr(runner, "_actuate", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_publish", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_persist_state", lambda *a, **kw: None)

    runner.tick(T0 + 60)

    assert runner.state.daily_extra_throughput_used == pytest.approx(1.5)
    assert runner.state.energy_budget_used == pytest.approx(0.2)
