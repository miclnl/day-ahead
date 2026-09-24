import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import pytest

from dao.prog.config.models.fastcontrol import FastControlConfig
from dao.prog.fastctrl.policy import (
    BatteryDecision,
    ControllerState,
    Decision,
)
from dao.prog.fastctrl.runner import FastControlRunner, save_state


@pytest.fixture
def workspace(tmp_path):
    return {
        "plan_path": str(tmp_path / "fast_plan.json"),
        "state_path": str(tmp_path / "fast_state.json"),
    }


def test_controller_state_roundtrip_includes_new_fields():
    state = ControllerState()
    state.last_decision = {"ts": 1.0, "reason": "plan", "override": False}
    state.events = [{"ts": 1.0, "kind": "override_start"}]

    serialised = state.to_dict()
    assert "last_decision" in serialised
    assert serialised["last_decision"] == {"ts": 1.0, "reason": "plan", "override": False}
    assert serialised["events"] == [{"ts": 1.0, "kind": "override_start"}]

    restored = ControllerState.from_dict(serialised)
    assert restored.last_decision == {"ts": 1.0, "reason": "plan", "override": False}
    assert restored.events == [{"ts": 1.0, "kind": "override_start"}]


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


def _make_runner(mode: str, state_path: str) -> FastControlRunner:
    config = FastControlConfig.model_validate(
        {
            "mode": mode,
            "grid power": {"entity": "sensor.p1_power"},
        }
    )
    return FastControlRunner(_StubHass(), config, plan_path="/nonexistent.json", state_path=state_path)


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
