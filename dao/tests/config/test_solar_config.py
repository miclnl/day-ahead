"""Tests for the solar configuration's model choice and completeness rules."""

import pytest
from pydantic import ValidationError

from dao.prog.config.models.devices.solar import SolarConfig


def test_flat_config_accepts_capacity_without_yield():
    config = SolarConfig(name="Roof", tilt=35, orientation=0, capacity=3.6)
    assert config.capacity == 3.6
    assert config.yield_factor is None


def test_flat_config_accepts_yield_without_capacity():
    config = SolarConfig(**{"name": "Roof", "tilt": 35, "orientation": 0, "yield": 0.0078})
    assert config.capacity is None
    assert config.yield_factor == pytest.approx(0.0078)


def test_flat_config_without_capacity_or_yield_is_rejected():
    with pytest.raises(ValidationError):
        SolarConfig(name="Roof", tilt=35, orientation=0)


def test_string_without_yield_is_accepted():
    config = SolarConfig(
        name="Roof",
        strings=[{"tilt": 35, "orientation": 0, "capacity": 2.0}],
    )
    assert config.strings[0].yield_factor is None


def test_model_defaults_to_physical():
    """v3 replaced the ml_prediction boolean with an explicit model name;
    an installation that says nothing gets the physical model."""
    installation = SolarConfig(name="Roof", tilt=35, orientation=0, capacity=3.6)
    assert installation.model == "physical"


def test_model_accepts_every_choice():
    for choice in ("physical", "ml", "auto"):
        installation = SolarConfig(
            name="Roof", tilt=35, orientation=0, capacity=3.6, model=choice
        )
        assert installation.model == choice


def test_calibration_defaults_to_scale():
    config = SolarConfig(name="Roof", tilt=35, orientation=0, capacity=3.6)
    assert config.calibration == "scale"
