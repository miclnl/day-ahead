"""
Tests for configuration migrations.
"""

import pytest
from dao.prog.config.migrations.migrator import migrate_config
from dao.prog.config.migrations.unversioned_to_v0 import migrate_unversioned_to_v0
from dao.prog.config.models.pricing import PricingConfig
from dao.prog.config.models.database import HADatabaseConfig, DatabaseConfig
from dao.prog.config.models.scheduler import SchedulerConfig
from dao.prog.config.migrations.v1_to_v2 import migrate_v1_to_v2

def test_migrate_unversioned_to_v0():
    """Test migration from unversioned config to v0."""
    old_config = {
        "latitude": 52.0,
        "longitude": 5.0,
        "battery": [{"name": "Test", "capacity": 10}]
    }
    
    new_config = migrate_unversioned_to_v0(old_config)
    
    assert new_config["config_version"] == 0
    assert new_config["latitude"] == 52.0
    assert new_config["longitude"] == 5.0
    assert new_config["battery"][0]["name"] == "Test"


def test_migrate_unversioned_to_v0_with_vat():
    """Test migration of prices.vat to prices.vat_consumption and prices.vat_production."""
    old_config = {
        "latitude": 52.0,
        "prices": {
            "source day ahead": "nordpool",
            "energy taxes consumption": {"2020-01-01": 0.0},
            "energy taxes production": {"2020-01-01": 0.0},
            "cost supplier consumption": {"2020-01-01": 0.0},
            "cost supplier production": {"2020-01-01": 0.0},
            "last invoice": "2024-01-01",
            "vat": {"2024-01-01": 21}
        }
    }
    
    new_config = migrate_unversioned_to_v0(old_config)
    
    assert new_config["config_version"] == 0
    assert "vat" not in new_config["prices"]
    assert new_config["prices"]["vat consumption"] == {"2024-01-01": 21}
    assert new_config["prices"]["vat production"] == {"2024-01-01": 21}
    
    # Validate that PricingConfig still works with migrated prices data
    pricing = PricingConfig(**new_config["prices"])
    assert pricing.vat_consumption == {"2024-01-01": 21}
    assert pricing.vat_production == {"2024-01-01": 21}


def test_migrate_unversioned_to_v0_database_engines():
    """Test migration sets database engines to mysql if not specified."""
    old_config = {
        "latitude": 52.0,
        "database ha": {"password": "secret"},
        "database da": {"password": "secret"}
    }
    
    new_config = migrate_unversioned_to_v0(old_config)
    
    assert new_config["config_version"] == 0
    assert new_config["database ha"]["engine"] == "mysql"
    assert new_config["database da"]["engine"] == "mysql"

    # Validate that database models still work with migrated data
    ha_db = HADatabaseConfig(**new_config["database ha"])
    assert ha_db.engine == "mysql"
    assert ha_db.username == "homeassistant"
    da_db = DatabaseConfig(**new_config["database da"])
    assert da_db.engine == "mysql"
    assert da_db.username == "day_ahead"



def test_migrate_unversioned_to_v0_scheduler():
    """Test migration of scheduler from dict format to array format."""
    old_config = {
        "scheduler": {
            "active": "True",
            "0435": "get_day_ahead_prices",
            "0445": "get_meteo_data",
            "0500": "calc_optimum",
        }
    }

    new_config = migrate_unversioned_to_v0(old_config)

    assert new_config["config_version"] == 0
    assert new_config["scheduler"]["active"] is True
    assert len(new_config["scheduler"]["schedule"]) == 3

    # Validate that SchedulerConfig still works with migrated data
    scheduler = SchedulerConfig(**new_config["scheduler"])
    assert scheduler.active is True
    assert len(scheduler.schedule) == 3
    assert scheduler.schedule[0].time == "0435"
    assert scheduler.schedule[0].action == "get_day_ahead_prices"


def test_migrate_unversioned_to_v0_keeps_new_style_scheduler():
    """A document without config_version can already use the schedule list.

    The shipped options_example.json is such a document. Treating every dict
    as the legacy {"HHMM": action} shape turned the "schedule" key into a
    bogus entry and made the example unloadable.
    """
    old_config = {
        "scheduler": {
            "active": True,
            "schedule": [
                {"time": "0544", "action": "get_meteo_data"},
                {"time": "xx15", "action": "calc_optimum"},
            ],
        }
    }

    new_config = migrate_unversioned_to_v0(old_config)

    assert new_config["scheduler"] == old_config["scheduler"]
    scheduler = SchedulerConfig(**new_config["scheduler"])
    assert [e.time for e in scheduler.schedule] == ["0544", "xx15"]
    # The input document is left untouched.
    assert "config_version" not in old_config


def test_migrate_unversioned_to_v0_scheduler_drops_non_time_keys():
    old_config = {
        "scheduler": {
            "active": True,
            "//comment": "runs at night",
            "0435": "get_day_ahead_prices",
            "02xx": "calc_optimum",
        }
    }

    new_config = migrate_unversioned_to_v0(old_config)

    assert [e["time"] for e in new_config["scheduler"]["schedule"]] == ["0435", "02xx"]
    SchedulerConfig(**new_config["scheduler"])


def test_migrate_config_with_target_version():
    """Test migrate_config with explicit target version."""
    config = {
        "latitude": 52.0,
        "longitude": 5.0,
    }
    
    # Migrate to v0 (adds version field)
    migrated = migrate_config(config, target_version=0)
    
    assert migrated["config_version"] == 0
    assert migrated["latitude"] == 52.0


def test_migrate_config_already_at_target():
    """Test that migration is no-op when already at target version."""
    config = {
        "config_version": 0,
        "latitude": 52.0,
    }
    
    migrated = migrate_config(config, target_version=0)
    
    assert migrated["config_version"] == 0
    assert migrated["latitude"] == 52.0



class TestGraphicsKeyMigration:
    def test_prices_delivery_renamed(self):
        config = {"graphics": {"prices delivery": True, "style": "dark_background"}}
        result = migrate_unversioned_to_v0(config)
        assert "prices delivery" not in result["graphics"]
        assert result["graphics"]["prices consumption"] is True

    def test_prices_redelivery_renamed(self):
        config = {"graphics": {"prices redelivery": False}}
        result = migrate_unversioned_to_v0(config)
        assert "prices redelivery" not in result["graphics"]
        assert result["graphics"]["prices production"] is False

    def test_average_delivery_renamed(self):
        config = {"graphics": {"average delivery": True}}
        result = migrate_unversioned_to_v0(config)
        assert "average delivery" not in result["graphics"]
        assert result["graphics"]["average consumption"] is True

    def test_all_three_renamed_together(self):
        config = {"graphics": {
            "prices delivery": True,
            "prices redelivery": False,
            "average delivery": True,
            "style": "dark_background",
        }}
        result = migrate_unversioned_to_v0(config)
        g = result["graphics"]
        assert "prices delivery" not in g
        assert "prices redelivery" not in g
        assert "average delivery" not in g
        assert g["prices consumption"] is True
        assert g["prices production"] is False
        assert g["average consumption"] is True
        assert g["style"] == "dark_background"

    def test_new_key_wins_when_both_present(self):
        # If config already has the new key, old key is dropped and new key is kept
        config = {"graphics": {"prices delivery": False, "prices consumption": True}}
        result = migrate_unversioned_to_v0(config)
        assert "prices delivery" not in result["graphics"]
        assert result["graphics"]["prices consumption"] is True

    def test_no_graphics_section(self):
        config = {}
        result = migrate_unversioned_to_v0(config)
        assert "graphics" not in result

    def test_graphics_without_old_keys(self):
        config = {"graphics": {"prices consumption": True, "style": "dark_background"}}
        result = migrate_unversioned_to_v0(config)
        assert result["graphics"] == {"prices consumption": True, "style": "dark_background"}


class TestSolarOrientationMigration:
    def test_top_level_solar_flat_orientation_normalized(self):
        config = {"solar": [{"name": "roof", "orientation": 270}]}
        result = migrate_unversioned_to_v0(config)
        assert result["solar"][0]["orientation"] == -90

    def test_top_level_solar_string_orientation_normalized(self):
        config = {"solar": [{"name": "roof", "strings": [{"orientation": 270, "tilt": 35}]}]}
        result = migrate_unversioned_to_v0(config)
        assert result["solar"][0]["strings"][0]["orientation"] == -90

    def test_battery_dc_solar_flat_orientation_normalized(self):
        config = {"battery": [{"name": "bat", "solar": [{"name": "dc", "orientation": 315}]}]}
        result = migrate_unversioned_to_v0(config)
        assert result["battery"][0]["solar"][0]["orientation"] == -45

    def test_battery_dc_solar_string_orientation_normalized(self):
        config = {"battery": [{"name": "bat", "solar": [
            {"name": "dc", "strings": [{"orientation": 225, "tilt": 30}]}
        ]}]}
        result = migrate_unversioned_to_v0(config)
        assert result["battery"][0]["solar"][0]["strings"][0]["orientation"] == -135

    def test_orientation_already_in_range_untouched(self):
        config = {"solar": [{"name": "roof", "orientation": 5}]}
        result = migrate_unversioned_to_v0(config)
        assert result["solar"][0]["orientation"] == 5

    def test_orientation_exactly_180_untouched(self):
        config = {"solar": [{"name": "roof", "orientation": 180}]}
        result = migrate_unversioned_to_v0(config)
        assert result["solar"][0]["orientation"] == 180

    def test_negative_orientation_untouched(self):
        config = {"solar": [{"name": "roof", "orientation": -90}]}
        result = migrate_unversioned_to_v0(config)
        assert result["solar"][0]["orientation"] == -90

    def test_multiple_strings_all_normalized(self):
        config = {"solar": [{"name": "roof", "strings": [
            {"orientation": 270, "tilt": 35},
            {"orientation": 90, "tilt": 35},
            {"orientation": 181, "tilt": 35},
        ]}]}
        result = migrate_unversioned_to_v0(config)
        strings = result["solar"][0]["strings"]
        assert strings[0]["orientation"] == -90
        assert strings[1]["orientation"] == 90   # <= 180, untouched
        assert strings[2]["orientation"] == -179

    def test_no_solar_section(self):
        config = {}
        result = migrate_unversioned_to_v0(config)
        assert "solar" not in result


# Example test for future v0→v1 migration (uncomment when implementing):
# def test_v0_to_v1_adds_efficiency():
#     """Test v0→v1 migration adds efficiency to batteries."""
#     old_config = {
#         "config_version": 0,
#         "battery": [{"name": "Test", "capacity": 10, "max_charge_power": 5}]
#     }
#     
#     from dao.prog.config.migrations.v0_to_v1 import migrate_v0_to_v1
#     new_config = migrate_v0_to_v1(old_config)
#     
#     assert new_config['config_version'] == 1
#     assert new_config['battery'][0]['efficiency'] == 0.95
#
#
# def test_migrate_v0_to_v1_via_migrate_config():
#     """Test full migration chain from v0 to v1 using migrate_config."""
#     old_config = {
#         "config_version": 0,
#         "battery": [{"name": "Test", "capacity": 10, "max_charge_power": 5}]
#     }
#     
#     migrated = migrate_config(old_config, target_version=1)
#     
#     assert migrated['config_version'] == 1
#     assert migrated['battery'][0]['efficiency'] == 0.95

def test_migrate_v1_to_v2():
    old_config = {
        "config_version": 1,
        "battery": [
            {"name": "Test1",
             "entity_balance_switch": "input_boolean.nom"},
            {"name": "Test2",
             "entity_grid_setpoint": "input_number.grid_setpoint"}
        ],
        "grid": {
            "max_power": 17.0
        },
    }

    new_config = migrate_v1_to_v2(old_config)

    assert new_config['config_version'] == 2
    assert new_config['grid']["entity balance switch"] == "input_boolean.nom"
    assert new_config['grid']["entity grid setpoint"] == "input_number.grid_setpoint"
    assert "entity_balance_switch" not in new_config["battery"][0]
    assert "entity_grid_setpoint" not in new_config["battery"][1]
    # The input document is not modified.
    assert "entity_balance_switch" in old_config["battery"][0]


def test_migrate_v1_to_v2_moves_the_spaced_aliases_too():
    """Real configurations use the aliases with spaces. The migration only
    looked for the snake_case spelling, so the key lingered in the battery as
    an unknown extra and grid balancing silently stopped working."""
    old_config = {
        "config_version": 1,
        "battery": [
            {
                "name": "Accu",
                "entity balance switch": "input_boolean.balanceer_grid",
                "entity grid setpoint": "input_number.grid_setpoint",
            }
        ],
    }

    new_config = migrate_v1_to_v2(old_config)

    assert new_config["grid"]["entity balance switch"] == "input_boolean.balanceer_grid"
    assert new_config["grid"]["entity grid setpoint"] == "input_number.grid_setpoint"
    assert "entity balance switch" not in new_config["battery"][0]
    assert "entity grid setpoint" not in new_config["battery"][0]

    from dao.prog.config.versions.v2 import ConfigurationV2

    model = ConfigurationV2(**{**new_config, "battery": [], "meteoserver-key": "x"})
    assert model.grid.entity_balance_switch == "input_boolean.balanceer_grid"
    assert model.grid.entity_grid_setpoint == "input_number.grid_setpoint"


def test_migrate_v1_to_v2_keeps_an_existing_grid_value():
    old_config = {
        "config_version": 1,
        "grid": {"entity balance switch": "input_boolean.keep_me"},
        "battery": [{"name": "Accu", "entity balance switch": "input_boolean.old"}],
    }

    new_config = migrate_v1_to_v2(old_config)

    assert new_config["grid"]["entity balance switch"] == "input_boolean.keep_me"
    assert "entity balance switch" not in new_config["battery"][0]


def test_migrate_v1_to_v2_survives_grid_null():
    new_config = migrate_v1_to_v2({"config_version": 1, "grid": None, "battery": []})
    assert new_config["grid"] == {}


def test_migrate_v0_to_v1_renames_the_meteoserver_attempts_alias():
    """The v0 model spelled it 'meteoserver-attemps'; the migration looked for
    'meteo_attemps' and never matched, so a raised retry count fell back to
    the default of 2."""
    from dao.prog.config.migrations.v0_to_v1 import migrate_v0_to_v1
    from dao.prog.config.versions.v1 import ConfigurationV1

    new_config = migrate_v0_to_v1({"config_version": 0, "meteoserver-attemps": 5})

    assert new_config["meteoserver-attempts"] == 5
    assert "meteoserver-attemps" not in new_config
    assert ConfigurationV1(**{**new_config, "meteoserver-key": "x"}).meteoserver_attempts == 5

    kept = migrate_v0_to_v1(
        {"config_version": 0, "meteoserver-attemps": 5, "meteoserver-attempts": 7}
    )
    assert kept["meteoserver-attempts"] == 7
