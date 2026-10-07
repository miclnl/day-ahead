"""Storage for the baseload profile set.

A profile set bundles the seven weekday profiles the estimator in
:mod:`dao.forecast.baseload.profile` produces, plus the optional away
profile and the metadata needed to judge whether it is still fresh. It is
the one file the baseload service reads at forecast time and writes after
each fit.

The store also migrates the old per-weekday files (``baseload_0.json`` ..
``baseload_6.json``) from before this package existed, so an upgrade does
not throw away weeks of history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dao.forecast.baseload.profile import BaseloadProfile, profile_from_file
from dao.prog.config.loader import atomic_write_json

PROFILE_SET_VERSION = 2

PROFILE_FILE = "profile.json"
STATUS_FILE = "status.json"
SELECTION_FILE = "selection.json"


@dataclass
class ProfileSet:
    """Everything the baseload service needs to forecast a day."""

    created: datetime
    period_days: int
    aggregate: str
    home: dict[int, BaseloadProfile] = field(default_factory=dict)
    away: Optional[BaseloadProfile] = None
    away_source: Optional[str] = None
    standby_kwh: Optional[float] = None


def write_json(path: Path, payload: dict) -> None:
    """Atomically write a JSON payload, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def read_json(path: Path) -> Optional[dict]:
    """Read a JSON file, or ``None`` when it does not exist."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _profile_to_payload(profile: BaseloadProfile) -> dict:
    return {
        "values": profile.values,
        "samples": profile.samples,
        "spread": profile.spread,
        "pooled": profile.pooled,
    }


def _profile_from_payload(payload: dict) -> BaseloadProfile:
    return BaseloadProfile(
        values=list(payload["values"]),
        samples=list(payload["samples"]),
        pooled=list(payload["pooled"]),
        spread=list(payload.get("spread", [0.0] * 24)),
    )


def profile_set_to_dict(ps: ProfileSet) -> dict:
    away_payload = None
    if ps.away is not None:
        away_payload = _profile_to_payload(ps.away)
        away_payload["source"] = ps.away_source
    return {
        "version": PROFILE_SET_VERSION,
        "created": ps.created.isoformat(),
        "period_days": ps.period_days,
        "aggregate": ps.aggregate,
        "home": {str(wd): _profile_to_payload(p) for wd, p in ps.home.items()},
        "away": away_payload,
        "standby_kwh": ps.standby_kwh,
    }


def profile_set_from_dict(payload: dict) -> ProfileSet:
    version = payload.get("version")
    if version != PROFILE_SET_VERSION:
        raise ValueError(
            f"profielverzameling heeft versie {version!r}, verwacht "
            f"{PROFILE_SET_VERSION}"
        )

    home = {
        int(weekday): _profile_from_payload(p)
        for weekday, p in payload.get("home", {}).items()
    }

    away_payload = payload.get("away")
    away = None
    away_source = None
    if away_payload is not None:
        away = _profile_from_payload(away_payload)
        away_source = away_payload.get("source")

    return ProfileSet(
        created=datetime.fromisoformat(payload["created"]),
        period_days=payload["period_days"],
        aggregate=payload["aggregate"],
        home=home,
        away=away,
        away_source=away_source,
        standby_kwh=payload.get("standby_kwh"),
    )


def save_profile_set(ps: ProfileSet, data_dir: Path) -> Path:
    path = data_dir / PROFILE_FILE
    write_json(path, profile_set_to_dict(ps))
    return path


def load_profile_set(data_dir: Path) -> Optional[ProfileSet]:
    payload = read_json(data_dir / PROFILE_FILE)
    if payload is None:
        return None
    return profile_set_from_dict(payload)


def migrate_legacy_files(legacy_dir: Path, now: datetime) -> Optional[ProfileSet]:
    """Read the seven pre-v2 ``baseload_<weekday>.json`` files, if all exist."""
    paths = [legacy_dir / f"baseload_{weekday}.json" for weekday in range(7)]
    if not all(path.exists() for path in paths):
        return None

    home: dict[int, BaseloadProfile] = {}
    period_days = 0
    aggregate = "mean"
    for weekday, path in enumerate(paths):
        payload = read_json(path)
        values = profile_from_file(payload)
        if isinstance(payload, dict):
            samples = list(payload.get("samples", [0] * 24))
            pooled = list(payload.get("pooled", [False] * 24))
            period_days = payload.get("period_days", period_days)
            aggregate = payload.get("aggregate", aggregate)
        else:
            samples = [0] * 24
            pooled = [False] * 24
        home[weekday] = BaseloadProfile(values=values, samples=samples, pooled=pooled)

    return ProfileSet(
        created=now, period_days=period_days, aggregate=aggregate, home=home
    )


def profile_age_days(ps: ProfileSet, now: datetime) -> float:
    return (now - ps.created).total_seconds() / 86400.0
