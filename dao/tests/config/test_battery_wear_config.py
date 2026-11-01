"""Cycle cost help must describe the unchanged runtime throughput unit."""

import pytest
from pydantic import ValidationError

from dao.prog.config.models.devices.battery import BatteryConfig
from dao.tests.config.test_battery_stages import BASE


def test_cycle_cost_help_matches_runtime_unit():
    schema = BatteryConfig.model_json_schema()
    field = schema["properties"]["cycle cost"]
    assert field["x-unit"] == "EUR/kWh DC-bus throughput"
    assert "DC-bus throughput" in field["description"]
    assert "charging and discharging" in field["x-help"]
    assert "full cycle" in field["x-help"] and "not" in field["x-help"]
    assert "zero-wear assumption" in field["x-help"]
    assert "0.83" not in field["x-help"]
    assert "typically" not in field["x-validation-hint"]
    assert "cycle cost" in schema["required"]
    assert "default" not in field


@pytest.mark.parametrize("key", ["cycle cost", "cycle_cost"])
@pytest.mark.parametrize("rate", [0, 0.01, 0.83])
def test_cycle_cost_values_and_aliases_are_not_converted(key, rate):
    values = {name: value for name, value in BASE.items() if name != "cycle cost"}
    battery = BatteryConfig.model_validate({**values, key: rate})
    assert battery.cycle_cost == rate
    assert battery.model_dump(by_alias=True)["cycle cost"] == rate


def test_cycle_cost_is_still_required_and_nonnegative():
    values = {name: value for name, value in BASE.items() if name != "cycle cost"}
    with pytest.raises(ValidationError):
        BatteryConfig.model_validate(values)
    with pytest.raises(ValidationError):
        BatteryConfig.model_validate({**values, "cycle cost": -0.01})
