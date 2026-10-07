"""Tests for the v2 -> v3 migration: ml_prediction becomes model."""

import json
from pathlib import Path

import pytest

from dao.prog.config.loader import CURRENT_VERSION, validate_config_data
from dao.prog.config.migrations.v2_to_v3 import migrate_v2_to_v3

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "dao" / "data" / "options_example.json"


def base_config(**overrides):
    config = {"config_version": 2, "solar": [], "battery": []}
    config.update(overrides)
    return config


def test_ml_prediction_true_becomes_model_ml():
    migrated = migrate_v2_to_v3(
        base_config(solar=[{"name": "roof", "ml_prediction": True}])
    )

    installation = migrated["solar"][0]
    assert installation["model"] == "ml"
    assert "ml_prediction" not in installation
    assert migrated["config_version"] == 3


def test_ml_prediction_false_becomes_model_physical():
    migrated = migrate_v2_to_v3(
        base_config(solar=[{"name": "roof", "ml_prediction": False}])
    )

    assert migrated["solar"][0]["model"] == "physical"


def test_spaced_alias_is_migrated_too():
    """A hand-written options.json may use the spaced spelling."""
    migrated = migrate_v2_to_v3(
        base_config(solar=[{"name": "roof", "ml prediction": True}])
    )

    installation = migrated["solar"][0]
    assert installation["model"] == "ml"
    assert "ml prediction" not in installation


def test_existing_model_wins_over_ml_prediction():
    migrated = migrate_v2_to_v3(
        base_config(solar=[{"name": "roof", "ml_prediction": True, "model": "auto"}])
    )

    installation = migrated["solar"][0]
    assert installation["model"] == "auto"
    assert "ml_prediction" not in installation


def test_battery_solar_is_migrated_too():
    migrated = migrate_v2_to_v3(
        base_config(
            battery=[{"name": "bat", "solar": [{"name": "dc", "ml_prediction": True}]}]
        )
    )

    assert migrated["battery"][0]["solar"][0]["model"] == "ml"


def test_a_config_without_solar_still_bumps_the_version():
    migrated = migrate_v2_to_v3({"config_version": 2})

    assert migrated["config_version"] == 3


def test_full_chain_from_unversioned_reaches_v3():
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.pop("config_version", None)

    config = validate_config_data(raw)

    assert config.config_version == 3
    assert CURRENT_VERSION == 3


def test_the_example_names_its_models_explicitly():
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    installations = list(raw.get("solar", [])) + [
        item
        for battery in raw.get("battery", [])
        for item in battery.get("solar", [])
    ]
    assert all("ml_prediction" not in item for item in installations)

    config = validate_config_data(raw)
    assert all(
        installation.model in ("physical", "ml", "auto")
        for installation in config.solar
    )


def test_forecast_days_default_is_400():
    from dao.prog.config.models.history import HistoryConfig

    assert HistoryConfig().forecast_days == 400


def test_solar_model_defaults_to_physical():
    from dao.prog.config.models.devices.solar import SolarConfig

    installation = SolarConfig(name="roof", tilt=35, orientation=0, capacity=3.0)

    assert installation.model == "physical"
    assert not hasattr(installation, "effective_model")
    with pytest.raises(AttributeError):
        installation.ml_prediction
