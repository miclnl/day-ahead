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


def test_effective_model_defaults_from_ml_prediction():
    without_ml = SolarConfig(name="Roof", tilt=35, orientation=0, capacity=3.6)
    assert without_ml.effective_model == "physical"

    with_ml = SolarConfig(
        name="Roof", tilt=35, orientation=0, capacity=3.6, ml_prediction=True
    )
    assert with_ml.effective_model == "ml"


def test_effective_model_explicit_choice_wins():
    config = SolarConfig(
        name="Roof",
        tilt=35,
        orientation=0,
        capacity=3.6,
        ml_prediction=True,
        model="auto",
    )
    assert config.effective_model == "auto"


def test_calibration_defaults_to_scale():
    config = SolarConfig(name="Roof", tilt=35, orientation=0, capacity=3.6)
    assert config.calibration == "scale"
