"""Tests for the versioned baseload profile set store."""

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from dao.forecast.baseload.profile import BaseloadProfile
from dao.forecast.baseload.store import (
    PROFILE_FILE,
    ProfileSet,
    load_profile_set,
    migrate_legacy_files,
    profile_age_days,
    save_profile_set,
    write_json,
)

TZ = ZoneInfo("Europe/Amsterdam")


def make_profile(base: float) -> BaseloadProfile:
    return BaseloadProfile(
        values=[round(base + i * 0.01, 3) for i in range(24)],
        samples=[8] * 24,
        pooled=[False] * 24,
        spread=[0.05] * 24,
    )


def test_round_trip_preserves_all_fields(tmp_path):
    created = datetime(2026, 9, 29, 12, 0, tzinfo=TZ)
    home = {wd: make_profile(wd * 0.1) for wd in range(7)}
    away = make_profile(0.05)
    ps = ProfileSet(
        created=created,
        period_days=56,
        aggregate="mean",
        home=home,
        away=away,
        away_source="labels",
        standby_kwh=0.15,
    )

    path = save_profile_set(ps, tmp_path)
    assert path == tmp_path / PROFILE_FILE

    loaded = load_profile_set(tmp_path)

    assert loaded.created == created
    assert loaded.period_days == 56
    assert loaded.aggregate == "mean"
    assert loaded.away_source == "labels"
    assert loaded.standby_kwh == pytest.approx(0.15)
    for wd in range(7):
        assert loaded.home[wd].values == pytest.approx(home[wd].values)
        assert loaded.home[wd].samples == home[wd].samples
        assert loaded.home[wd].pooled == home[wd].pooled
        assert loaded.home[wd].spread == pytest.approx(home[wd].spread)
    assert loaded.away.values == pytest.approx(away.values)


def test_load_returns_none_when_absent(tmp_path):
    assert load_profile_set(tmp_path) is None


def test_wrong_version_raises(tmp_path):
    payload = {
        "version": 3,
        "created": "2026-01-01T00:00:00+01:00",
        "period_days": 1,
        "aggregate": "mean",
        "home": {},
        "away": None,
        "standby_kwh": None,
    }
    write_json(tmp_path / PROFILE_FILE, payload)
    with pytest.raises(ValueError):
        load_profile_set(tmp_path)


def test_legacy_seven_files_migrate(tmp_path):
    for wd in range(6):
        payload = {
            "version": 1,
            "created": "2026-01-01 00:00:00",
            "weekday": wd,
            "period_days": 56,
            "aggregate": "median",
            "baseload": [round(0.1 * wd, 3)] * 24,
            "samples": [8] * 24,
            "pooled": [False] * 24,
        }
        write_json(tmp_path / f"baseload_{wd}.json", payload)
    (tmp_path / "baseload_6.json").write_text(json.dumps([0.9] * 24))

    now = datetime(2026, 3, 1, tzinfo=TZ)
    ps = migrate_legacy_files(tmp_path, now)

    assert ps is not None
    assert ps.away is None
    assert len(ps.home) == 7
    assert ps.home[6].values == pytest.approx([0.9] * 24)
    assert ps.home[6].samples == [0] * 24
    assert ps.home[6].pooled == [False] * 24
    assert ps.home[0].values == pytest.approx([0.0] * 24)
    assert ps.home[0].samples == [8] * 24
    assert ps.period_days == 56
    assert ps.aggregate == "median"


def test_legacy_incomplete_returns_none(tmp_path):
    for wd in range(6):
        write_json(tmp_path / f"baseload_{wd}.json", {"baseload": [0.0] * 24})

    now = datetime(2026, 3, 1, tzinfo=TZ)
    assert migrate_legacy_files(tmp_path, now) is None


def test_profile_age_days(tmp_path):
    created = datetime(2026, 3, 1, 0, 0, tzinfo=TZ)
    now = created + timedelta(hours=36)
    ps = ProfileSet(created=created, period_days=56, aggregate="mean", home={})
    assert profile_age_days(ps, now) == pytest.approx(1.5)
