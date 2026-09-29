"""Tests for the write-side helpers in the configuration loader.

Everything that writes options.json (the web editors, the fast-control mode
switch) must validate first and write atomically. These tests exercise those
helpers on a copy of the shipped example configuration.
"""

import json
import shutil
from pathlib import Path

import pytest

from dao.prog.config.loader import (
    ConfigValidationError,
    atomic_write_json,
    atomic_write_text,
    set_fast_control_mode,
    validate_config_data,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "data" / "options_example.json"


@pytest.fixture
def options_file(tmp_path):
    target = tmp_path / "options.json"
    shutil.copy(EXAMPLE, target)
    return target


def test_example_config_validates():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    model = validate_config_data(data)
    assert model.config_version == 2


def test_validate_rejects_non_object():
    with pytest.raises(ValueError):
        validate_config_data(["not", "a", "dict"])


def test_validate_rejects_string_version():
    with pytest.raises(ValueError, match="config_version"):
        validate_config_data({"config_version": "2"})


def test_validate_reports_field_errors_readably():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["battery"] = "nope"
    with pytest.raises(ConfigValidationError, match="battery"):
        validate_config_data(data)


def test_atomic_write_replaces_file_and_leaves_no_temp(tmp_path):
    target = tmp_path / "x.json"
    target.write_text("old", encoding="utf-8")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.iterdir()) == [target]
    atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


def test_set_mode_changes_only_the_mode_key(options_file):
    before = json.loads(options_file.read_text(encoding="utf-8"))
    key = "fast control" if "fast control" in before else "fast_control"

    set_fast_control_mode(options_file, "active")

    after = json.loads(options_file.read_text(encoding="utf-8"))
    assert after[key]["mode"] == "active"
    assert {k: v for k, v in before.items() if k != key} == {
        k: v for k, v in after.items() if k != key
    }
    assert {k: v for k, v in before[key].items() if k != "mode"} == {
        k: v for k, v in after[key].items() if k != "mode"
    }


def test_set_mode_rejects_unknown_mode(options_file):
    before = options_file.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Ongeldige modus"):
        set_fast_control_mode(options_file, "turbo")
    assert options_file.read_text(encoding="utf-8") == before


def test_set_mode_refuses_when_bound_to_an_entity(options_file):
    data = json.loads(options_file.read_text(encoding="utf-8"))
    key = "fast control" if "fast control" in data else "fast_control"
    data[key]["mode"] = "input_select.dao_fast_mode"
    options_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="input_select.dao_fast_mode"):
        set_fast_control_mode(options_file, "shadow")


def test_set_mode_adds_section_when_missing(options_file):
    data = json.loads(options_file.read_text(encoding="utf-8"))
    data.pop("fast control", None)
    data.pop("fast_control", None)
    options_file.write_text(json.dumps(data), encoding="utf-8")

    set_fast_control_mode(options_file, "shadow")

    after = json.loads(options_file.read_text(encoding="utf-8"))
    assert after["fast control"]["mode"] == "shadow"
