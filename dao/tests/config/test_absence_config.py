"""Tests for the away-detection block of the baseload configuration."""

from dao.prog.config.models.baseload import BaseloadOptionsConfig


def test_absence_defaults_and_aliases():
    config = BaseloadOptionsConfig(**{"absence": {"entity away": "input_boolean.x"}})
    assert config.absence.entity_away == "input_boolean.x"
    assert config.absence.threshold == 0.4
    assert config.absence.detect is True
    assert config.absence.away_state == "on"
    assert config.absence.entities_presence == []
    assert config.absence.calendar_keywords == ["vakantie", "weg", "afwezig", "holiday"]
    assert config.absence.away_after_hours == 3
    assert config.absence.assume_next_day_after_hours == 24
