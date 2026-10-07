"""Tests for the weather source configuration block."""

from dao.prog.config.models.weather import WeatherConfig
from dao.prog.config.versions.v2 import ConfigurationV2


def test_weather_defaults():
    config = WeatherConfig()
    assert config.fallback == "openmeteo"
    assert config.openmeteo_model == "knmi_seamless"
    assert config.observations == "auto"


def test_meteoserver_key_is_optional():
    config = ConfigurationV2(battery=[])
    assert config.meteoserver_key is None
