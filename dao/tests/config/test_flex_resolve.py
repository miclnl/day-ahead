"""FlexValue.resolve() against unusable Home Assistant states.

Home Assistant reports ``unavailable`` or ``unknown`` while an integration is
starting or reconnecting. A FlexValue bound to such an entity used to raise a
bare ValueError from float(), which took the whole optimisation down.
"""

import pytest

from dao.prog.config.models.base import (
    FlexBool,
    FlexEnum,
    FlexFloat,
    FlexInt,
    FlexResolveError,
    FlexStr,
)


def getter(states):
    def _get(entity_id):
        value = states[entity_id]
        if isinstance(value, Exception):
            raise value
        return value

    return _get


def test_literal_values_are_cast_to_the_resolve_type():
    assert FlexFloat(value=95).resolve(getter({})) == 95.0
    assert isinstance(FlexFloat(value=95).resolve(getter({})), float)
    assert FlexInt(value="95.0").resolve(getter({})) == 95
    assert FlexBool(value="False").resolve(getter({})) is False
    assert FlexStr(value=12).resolve(getter({})) == "12"


def test_entity_state_is_converted():
    states = {"sensor.soc": "42.5", "input_boolean.x": "on", "input_number.n": "7.0"}
    assert FlexFloat(value="sensor.soc").resolve(getter(states)) == 42.5
    assert FlexBool(value="input_boolean.x").resolve(getter(states)) is True
    assert FlexInt(value="input_number.n").resolve(getter(states)) == 7


@pytest.mark.parametrize("state", ["unavailable", "unknown", "", None, "Unknown "])
def test_unusable_state_raises_without_default(state):
    with pytest.raises(FlexResolveError, match="sensor.soc"):
        FlexFloat(value="sensor.soc").resolve(getter({"sensor.soc": state}))


@pytest.mark.parametrize("state", ["unavailable", "unknown", "", None, "abc"])
def test_unusable_state_returns_default_when_given(state, caplog):
    result = FlexFloat(value="sensor.soc").resolve(
        getter({"sensor.soc": state}), default=50.0
    )
    assert result == 50.0
    assert "sensor.soc" in caplog.text


def test_getter_exception_is_reported_not_propagated():
    states = {"sensor.soc": RuntimeError("404 status code returned")}
    with pytest.raises(FlexResolveError, match="404"):
        FlexFloat(value="sensor.soc").resolve(getter(states))
    assert FlexFloat(value="sensor.soc").resolve(getter(states), default=20.0) == 20.0


def test_non_numeric_state_for_number_raises():
    with pytest.raises(FlexResolveError, match="float"):
        FlexFloat(value="sensor.soc").resolve(getter({"sensor.soc": "n/a"}))


def test_none_is_a_valid_default():
    assert (
        FlexFloat(value="sensor.soc").resolve(getter({"sensor.soc": "unknown"}), default=None)
        is None
    )


def test_enum_entity_state_is_returned_as_string():
    states = {"input_select.strategy": "minimize cost"}
    flex = FlexEnum(
        value="input_select.strategy",
        enum_values=["minimize cost", "minimize consumption"],
    )
    assert flex.resolve(getter(states)) == "minimize cost"


def test_enum_entity_state_outside_the_list_raises_or_defaults():
    flex = FlexEnum(
        value="input_select.strategy",
        enum_values=["minimize cost", "minimize consumption"],
    )
    states = {"input_select.strategy": "Minimize Cost"}
    with pytest.raises(FlexResolveError, match="minimize cost"):
        flex.resolve(getter(states))
    assert flex.resolve(getter(states), default="minimize cost") == "minimize cost"
