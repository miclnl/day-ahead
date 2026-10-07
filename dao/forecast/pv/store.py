"""Artefacts for calibrated PV parameters, one file per installation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dao.forecast.baseload.store import read_json, write_json
from dao.forecast.pv.calibrate import CalibrationResult

PV_DIR = "../data/forecast/pv"


def installation_key(name: str) -> str:
    """A filesystem-safe key for an installation's own artefact files."""
    return name.replace(" ", "_").replace("-", "_")


def calibration_path(data_dir: Path, name: str) -> Path:
    return Path(data_dir) / f"{installation_key(name)}.json"


def selection_path(data_dir: Path, name: str) -> Path:
    return Path(data_dir) / f"{installation_key(name)}.selection.json"


def save_calibration(result: CalibrationResult, path: Path) -> None:
    write_json(Path(path), result.to_dict())


def load_calibration(path: Path) -> Optional[CalibrationResult]:
    payload = read_json(Path(path))
    if payload is None:
        return None
    return CalibrationResult.from_dict(payload)
