from dao.prog.fastctrl.policy import ControllerState


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
