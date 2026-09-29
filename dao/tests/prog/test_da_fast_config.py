"""_spec_from_config / _limits_from_config: resolving FlexValue fields.

Both used to read FlexValue.value directly, which is the raw config token
(the entity id itself when a field is HA-entity-backed, not a percentage or
a flag), so an entity-backed limit crashed float() and an entity-backed
allow_grid_charge was always truthy (a non-empty entity-id string).
"""

import pytest

from dao.prog.config.models.devices.battery import BatteryConfig
from dao.prog.config.models.fastcontrol import FastControlConfig
from dao.prog.da_fast import _limits_from_config, _spec_from_config


def make_battery(**overrides):
    data = {
        "name": "accu",
        "capacity": 10,
        "entity actual level": "sensor.soc",
        "entity set power feedin": "number.feedin",
        "entity set operating mode": "select.mode",
        "dc_to_bat efficiency": 0.95,
        "bat_to_dc efficiency": 0.95,
        "cycle cost": 0.01,
        "charge stages": [{"power": 0, "efficiency": 1.0}, {"power": 3000, "efficiency": 0.95}],
        "discharge stages": [{"power": 0, "efficiency": 1.0}, {"power": 3000, "efficiency": 0.95}],
        **overrides,
    }
    return BatteryConfig(**data)


class FakeConfig:
    def __init__(self, battery, fast_control):
        self.battery = battery
        self.fast_control = fast_control


def test_spec_uses_literal_limits():
    battery = make_battery(**{"upper limit": 90, "lower limit": 15})
    config = FakeConfig([battery], None)

    spec = _spec_from_config(config, ha_getter=lambda eid: pytest.fail("not an entity"))

    assert spec.soc_min == 15.0
    assert spec.soc_max == 90.0


def test_spec_resolves_entity_backed_limits():
    battery = make_battery(**{"upper limit": "input_number.max_soc", "lower limit": "input_number.min_soc"})
    config = FakeConfig([battery], None)
    states = {"input_number.max_soc": "88", "input_number.min_soc": "12"}

    spec = _spec_from_config(config, ha_getter=lambda eid: states[eid])

    assert spec.soc_min == 12.0
    assert spec.soc_max == 88.0


def test_spec_falls_back_when_the_entity_is_unavailable():
    battery = make_battery(**{"upper limit": "input_number.max_soc"})
    config = FakeConfig([battery], None)

    spec = _spec_from_config(config, ha_getter=lambda eid: "unavailable")

    assert spec.soc_max == 100.0  # the documented default


def test_limits_resolve_entity_backed_allow_grid_charge():
    fast = FastControlConfig.model_validate(
        {
            "grid power": {"entity": "sensor.p1"},
            "allow grid charge": "input_boolean.allow_grid_charge",
        }
    )
    config = FakeConfig([], fast)

    limits = _limits_from_config(config, ha_getter=lambda eid: "on")
    assert limits.allow_grid_charge is True

    limits = _limits_from_config(config, ha_getter=lambda eid: "off")
    assert limits.allow_grid_charge is False


def test_limits_use_the_literal_allow_grid_charge():
    fast = FastControlConfig.model_validate(
        {"grid power": {"entity": "sensor.p1"}, "allow grid charge": True}
    )
    config = FakeConfig([], fast)

    limits = _limits_from_config(config, ha_getter=lambda eid: pytest.fail("not an entity"))
    assert limits.allow_grid_charge is True


def test_limits_resolve_storage_value_mode_and_fixed_value():
    fast = FastControlConfig.model_validate(
        {
            "grid power": {"entity": "sensor.p1"},
            "storage value mode": "fixed",
            "storage value": "input_number.storage_value",
        }
    )
    config = FakeConfig([], fast)

    limits = _limits_from_config(
        config, ha_getter=lambda eid: "0.22" if eid == "input_number.storage_value" else "fixed"
    )

    assert limits.storage_value_mode == "fixed"
    assert limits.storage_value_fixed == pytest.approx(0.22)


def test_limits_storage_value_is_none_without_config():
    fast = FastControlConfig.model_validate({"grid power": {"entity": "sensor.p1"}})
    config = FakeConfig([], fast)

    limits = _limits_from_config(config, ha_getter=lambda eid: pytest.fail("not an entity"))
    assert limits.storage_value_fixed is None
