"""Tests for the fast control configuration model."""

import pytest
from pydantic import ValidationError

from dao.prog.config.models.fastcontrol import (
    FastBatteryLink,
    FastControlConfig,
    PowerSensor,
)
from dao.prog.config.versions.v0 import ConfigurationV0


class TestPowerSensor:
    def test_defaults_are_unconfigured(self):
        sensor = PowerSensor()
        assert not sensor.configured
        assert sensor.entity_ids == []
        assert sensor.unit == "W"
        assert sensor.invert is False

    def test_a_signed_entity(self):
        sensor = PowerSensor.model_validate({"entity": "sensor.p1"})
        assert sensor.configured
        assert sensor.entity_ids == ["sensor.p1"]

    def test_a_positive_negative_pair(self):
        sensor = PowerSensor.model_validate(
            {"entity positive": "sensor.imp", "entity negative": "sensor.exp"}
        )
        assert sensor.entity_ids == ["sensor.imp", "sensor.exp"]

    def test_mixing_both_shapes_is_rejected(self):
        with pytest.raises(ValidationError, match="not both"):
            PowerSensor.model_validate(
                {"entity": "sensor.p1", "entity positive": "sensor.imp"}
            )

    def test_an_invalid_entity_id_is_rejected(self):
        with pytest.raises(ValidationError):
            PowerSensor.model_validate({"entity": "not an entity"})

    def test_an_unknown_unit_is_rejected(self):
        with pytest.raises(ValidationError):
            PowerSensor.model_validate({"entity": "sensor.p1", "unit": "MW"})


class TestFastControlConfig:
    def test_it_is_off_by_default(self):
        config = FastControlConfig()
        assert config.mode.value == "off"
        assert not config.grid_power.configured
        assert config.batteries == []

    def test_the_defaults_are_conservative(self):
        config = FastControlConfig()
        assert config.interval == 15
        assert config.deadband >= 100
        assert config.min_command_interval >= 30
        assert config.energy_budget > 0
        assert config.daily_extra_throughput > 0
        assert config.allow_grid_charge.value is False
        assert config.max_grid_import is None

    def test_spaced_aliases_are_accepted(self):
        config = FastControlConfig.model_validate(
            {
                "mode": "shadow",
                "grid power": {"entity": "sensor.p1"},
                "max sensor age": 90,
                "min command interval": 30,
                "energy budget": 0.8,
                "daily extra throughput": 2.0,
                "storage value mode": "fixed",
                "storage value": 0.31,
                "allow grid charge": True,
                "max grid import": 6000,
            }
        )
        assert config.max_sensor_age == 90
        assert config.min_command_interval == 30
        assert config.energy_budget == 0.8
        assert config.storage_value.value == 0.31
        assert config.allow_grid_charge.value is True
        assert config.max_grid_import == 6000

    def test_the_mode_may_be_an_entity(self):
        config = FastControlConfig.model_validate(
            {"mode": "input_select.dao_fast_mode"}
        )
        assert config.mode.value == "input_select.dao_fast_mode"

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValidationError):
            FastControlConfig.model_validate({"mode": "turbo"})

    @pytest.mark.parametrize(
        "field,value",
        [
            ("interval", 1),
            ("interval", 1000),
            ("round trip efficiency", 1.5),
            ("round trip efficiency", 0.0),
            ("soc margin", 80.0),
            ("deadband", -1),
            ("min benefit", -0.1),
            ("max sensor age", 1),
            ("max plan age", 10),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field, value):
        with pytest.raises(ValidationError):
            FastControlConfig.model_validate({field: value})

    def test_enabled_batteries_are_sorted_by_priority(self):
        config = FastControlConfig.model_validate(
            {
                "batteries": [
                    {"name": "b", "priority": 2},
                    {"name": "a", "priority": 1},
                    {"name": "c", "priority": 0, "enabled": False},
                ]
            }
        )
        assert [b.name for b in config.enabled_batteries] == ["a", "b"]

    def test_a_battery_link_carries_its_measurement(self):
        link = FastBatteryLink.model_validate(
            {
                "name": "Accu1",
                "actual power": {"entity": "sensor.bat_power", "invert": True},
            }
        )
        assert link.enabled is True
        assert link.actual_power.invert is True
        assert link.actual_power.entity_ids == ["sensor.bat_power"]

    def test_diagnostics_default_to_a_self_creating_sensor(self):
        config = FastControlConfig()
        assert config.diagnostics.entity_status == "sensor.dao_fast_control"
        assert config.diagnostics.entity_active is None

    def test_a_roundtrip_through_json_is_stable(self):
        original = FastControlConfig.model_validate(
            {
                "mode": "active",
                "grid power": {"entity": "sensor.p1", "unit": "kW"},
                "batteries": [
                    {"name": "Accu1", "actual power": {"entity": "sensor.bat"}}
                ],
                "storage value": 0.28,
            }
        )
        dumped = original.model_dump(by_alias=True, exclude_none=True)
        assert FastControlConfig.model_validate(dumped) == original


class TestRootIntegration:
    def test_the_section_is_optional(self):
        config = ConfigurationV0.model_validate({"meteoserver-key": "x"})
        assert config.fast_control.mode.value == "off"

    def test_the_spaced_key_is_accepted_and_preserved(self):
        config = ConfigurationV0.model_validate(
            {
                "meteoserver-key": "x",
                "fast control": {
                    "mode": "shadow",
                    "grid power": {"entity": "sensor.p1"},
                },
            }
        )
        assert config.fast_control.mode.value == "shadow"
        dumped = config.model_dump(by_alias=True, exclude_none=True)
        assert dumped["fast control"]["mode"] == "shadow"
        assert dumped["fast control"]["grid power"]["entity"] == "sensor.p1"

    def test_an_existing_config_without_the_section_still_loads(self):
        config = ConfigurationV0.model_validate(
            {"meteoserver-key": "x", "interval": "15min"}
        )
        assert config.interval == "15min"
        assert config.fast_control is not None


class TestBaseloadOptions:
    """Configuration of the baseload estimator."""

    def test_the_defaults_are_the_robust_ones(self):
        from dao.prog.config.models.baseload import BaseloadOptionsConfig

        config = BaseloadOptionsConfig()
        assert config.aggregate == "median"
        assert config.remove_outliers is True
        assert config.half_life_days == 28.0
        assert config.holidays == "sunday"
        assert config.clip_negative is True
        assert config.min_samples >= 1

    def test_spaced_aliases_are_accepted(self):
        from dao.prog.config.models.baseload import BaseloadOptionsConfig

        config = BaseloadOptionsConfig.model_validate(
            {
                "aggregate": "trimmed",
                "trim fraction": 0.1,
                "remove outliers": False,
                "outlier factor": 3.0,
                "half life days": 14.0,
                "holidays": "ignore",
                "clip negative": False,
                "min samples": 5,
            }
        )
        assert config.trim_fraction == 0.1
        assert config.outlier_factor == 3.0
        assert config.min_samples == 5

    def test_the_old_behaviour_can_be_restored(self):
        from dao.prog.config.models.baseload import BaseloadOptionsConfig

        config = BaseloadOptionsConfig.model_validate(
            {
                "aggregate": "mean",
                "remove outliers": False,
                "half life days": None,
                "holidays": "ignore",
            }
        )
        assert config.aggregate == "mean"
        assert config.half_life_days is None

    @pytest.mark.parametrize(
        "field,value",
        [
            ("aggregate", "geometric"),
            ("trim fraction", 0.5),
            ("trim fraction", -0.1),
            ("outlier factor", 0.0),
            ("half life days", 0.0),
            ("holidays", "monday"),
            ("min samples", 0),
        ],
    )
    def test_invalid_values_are_rejected(self, field, value):
        from dao.prog.config.models.baseload import BaseloadOptionsConfig

        with pytest.raises(ValidationError):
            BaseloadOptionsConfig.model_validate({field: value})

    def test_it_is_reachable_from_the_root_config(self):
        config = ConfigurationV0.model_validate(
            {
                "meteoserver-key": "x",
                "baseload options": {"aggregate": "mean"},
            }
        )
        assert config.baseload_options.aggregate == "mean"

    def test_the_root_default_needs_no_configuration(self):
        config = ConfigurationV0.model_validate({"meteoserver-key": "x"})
        assert config.baseload_options.aggregate == "median"


class TestForecastRetention:
    def test_the_default_keeps_two_months(self):
        config = ConfigurationV0.model_validate({"meteoserver-key": "x"})
        assert config.history.forecast_days == 60

    def test_it_can_be_shortened_for_small_storage(self):
        config = ConfigurationV0.model_validate(
            {"meteoserver-key": "x", "history": {"forecast days": 14}}
        )
        assert config.history.forecast_days == 14

    def test_an_absurdly_short_retention_is_rejected(self):
        with pytest.raises(ValidationError):
            ConfigurationV0.model_validate(
                {"meteoserver-key": "x", "history": {"forecast days": 1}}
            )
