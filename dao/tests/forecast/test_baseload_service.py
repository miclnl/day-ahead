"""Tests for the baseload service: fit, forecast, and the fallback chain."""

from __future__ import annotations

import datetime as dt
import logging
import types
from zoneinfo import ZoneInfo

import pytest

from dao.forecast.baseload.profile import BaseloadProfile
from dao.forecast.baseload.service import (
    BaseloadService,
    BaseloadUnavailable,
    Regime,
)
from dao.forecast.baseload.store import ProfileSet, save_profile_set, write_json
from dao.tests.forecast.conftest import TZ

ZONE = ZoneInfo(TZ)


def flat_profile(value: float, samples: int = 8) -> BaseloadProfile:
    return BaseloadProfile(
        values=[value] * 24, samples=[samples] * 24, pooled=[False] * 24
    )


def make_config(*, baseload=None, calc_periode=56, aggregate="mean"):
    return types.SimpleNamespace(
        report=types.SimpleNamespace(
            entities_grid_consumption=["sensor.test_grid_in"],
            entities_grid_production=["sensor.test_grid_out"],
            entities_solar_production_ac=["sensor.test_pv"],
            entities_ev_consumption=[],
            entities_wp_consumption=[],
            entities_boiler_consumption=[],
            entities_machine_consumption=[],
            entities_battery_consumption=[],
            entities_battery_production=[],
        ),
        solar=[],
        battery=[],
        grid=types.SimpleNamespace(max_power=None),
        baseload_calc_periode=calc_periode,
        baseload_options=types.SimpleNamespace(
            aggregate=aggregate,
            trim_fraction=0.2,
            remove_outliers=True,
            outlier_factor=2.0,
            half_life_days=28.0,
            holidays="sunday",
            clip_negative=True,
            min_samples=3,
        ),
        baseload=baseload,
    )


def build_service(tmp_path, home, *, away=None, away_source=None, config=None, now=None):
    """A service whose data directory already holds a saved profile set."""
    data_dir = tmp_path / "forecast" / "baseload"
    now = now or (lambda: dt.datetime(2026, 3, 4, 12, 0, tzinfo=ZONE))  # Wednesday
    profile_set = ProfileSet(
        created=dt.datetime(2026, 3, 4, tzinfo=ZONE),
        period_days=56,
        aggregate="mean",
        home=home,
        away=away,
        away_source=away_source,
    )
    save_profile_set(profile_set, data_dir)
    return BaseloadService(
        config or make_config(),
        db_da=None,
        db_ha=None,
        data_dir=data_dir,
        tz=TZ,
        now=now,
    )


@pytest.fixture
def service_with_profiles(tmp_path):
    """Weekday 6 (Sunday) is flat 0.9, every other weekday flat 0.3, away 0.1."""
    home = {wd: flat_profile(0.9 if wd == 6 else 0.3) for wd in range(7)}
    away = flat_profile(0.1)
    return build_service(tmp_path, home, away=away, away_source="labels")


@pytest.fixture
def service_without_profiles(tmp_path):
    data_dir = tmp_path / "forecast" / "baseload"
    config = make_config(baseload=None)
    now = lambda: dt.datetime(2026, 3, 4, 12, 0, tzinfo=ZONE)  # noqa: E731
    return BaseloadService(
        config, db_da=None, db_ha=None, data_dir=data_dir, tz=TZ, now=now
    )


@pytest.fixture
def service_with_history(ha_db, tmp_path):
    """60 days of flat synthetic history ending at midnight of ``now``."""
    manager, helper = ha_db
    now = dt.datetime(2026, 3, 4, 6, 0, tzinfo=ZONE)  # Wednesday morning
    period_days = 60
    tot = now.replace(hour=0, minute=0, second=0, microsecond=0)
    vanaf = tot - dt.timedelta(days=period_days)
    n_hours = period_days * 24

    grid_in = {}
    grid_out = {}
    pv_power = {}
    for i in range(n_hours + 1):
        ts = int((vanaf + dt.timedelta(hours=i)).timestamp())
        grid_in[ts] = 0.3 * i  # cumulative energy counter, +0.3 kWh/hour
        grid_out[ts] = 0.05 * i
    for i in range(n_hours):
        ts = int((vanaf + dt.timedelta(hours=i)).timestamp())
        pv_power[ts] = 100.0  # flat power mean, not cumulative

    helper.add_energy("sensor.test_grid_in", "kWh", grid_in)
    helper.add_energy("sensor.test_grid_out", "kWh", grid_out)
    helper.add_power("sensor.test_pv", "W", pv_power)

    config = make_config(baseload=None, calc_periode=period_days)
    data_dir = tmp_path / "forecast" / "baseload"
    service = BaseloadService(
        config, db_da=None, db_ha=manager, data_dir=data_dir, tz=TZ, now=lambda: now
    )
    return service, data_dir


def test_forecast_uses_effective_weekday_for_holiday(service_with_profiles):
    # 2026-12-25 is a Friday and a public holiday -> folded into Sunday.
    start = dt.datetime(2026, 12, 25, 0, 0, tzinfo=ZONE)
    series = service_with_profiles.forecast(start, 24)
    assert list(series.values) == pytest.approx([0.9] * 24)


def test_forecast_crosses_midnight_with_next_day_profile(tmp_path):
    home = {wd: flat_profile(0.3) for wd in range(7)}
    home[1] = flat_profile(0.5)  # Tuesday
    service = build_service(tmp_path, home)

    start = dt.datetime(2026, 3, 2, 22, 0, tzinfo=ZONE)  # Monday 22:00
    series = service.forecast(start, 6)
    assert list(series.values) == pytest.approx([0.3, 0.3, 0.5, 0.5, 0.5, 0.5])


def test_forecast_uses_target_date_not_now(service_with_profiles):
    # The fixture's "now" is a Wednesday; the target date is a Sunday.
    start = dt.datetime(2026, 3, 1, 0, 0, tzinfo=ZONE)
    series = service_with_profiles.forecast(start, 1)
    assert series.values[0] == pytest.approx(0.9)


def test_regime_away_uses_away_profile(service_with_profiles):
    start = dt.datetime(2026, 3, 2, 0, 0, tzinfo=ZONE)
    series = service_with_profiles.forecast(start, 5, regime=Regime(away=True))
    assert list(series.values) == pytest.approx([0.1] * 5)


def test_regime_away_without_away_profile_uses_standby(tmp_path):
    # Seven distinct weekday values; P10 across them is not any one weekday's own value.
    home = {
        wd: flat_profile(v)
        for wd, v in enumerate([0.2, 0.25, 0.3, 0.35, 0.4, 0.9, 1.5])
    }
    service = build_service(tmp_path, home, away=None)

    start = dt.datetime(2026, 3, 2, 0, 0, tzinfo=ZONE)
    series = service.forecast(start, 1, regime=Regime(away=True))
    assert series.values[0] == pytest.approx(0.23, abs=0.001)


def test_fallback_to_static_baseload(tmp_path, caplog):
    data_dir = tmp_path / "forecast" / "baseload"
    config = make_config(baseload=[0.2] * 24)
    now = lambda: dt.datetime(2026, 3, 4, 12, 0, tzinfo=ZONE)  # noqa: E731
    service = BaseloadService(
        config, db_da=None, db_ha=None, data_dir=data_dir, tz=TZ, now=now
    )

    with caplog.at_level(logging.WARNING):
        series = service.forecast(dt.datetime(2026, 3, 2, 0, 0, tzinfo=ZONE), 3)

    assert list(series.values) == pytest.approx([0.2, 0.2, 0.2])
    assert any("statische baseload" in message for message in caplog.messages)


def test_no_profile_no_static_raises(service_without_profiles):
    with pytest.raises(BaseloadUnavailable):
        service_without_profiles.forecast(
            dt.datetime(2026, 3, 2, 0, 0, tzinfo=ZONE), 3
        )


def test_forecast_for_optimizer_15min_length(service_with_profiles):
    start = dt.datetime(2026, 3, 2, 0, 0, tzinfo=ZONE)  # Monday, flat 0.3
    values = service_with_profiles.forecast_for_optimizer(start, 10, "15min")
    assert len(values) == 10
    assert sum(values[0:4]) == pytest.approx(0.3)


def test_fit_writes_profile_set(service_with_history):
    service, data_dir = service_with_history
    profile_set = service.fit()

    assert len(profile_set.home) == 7
    for profile in profile_set.home.values():
        assert profile.values == pytest.approx([0.35] * 24, abs=0.01)
    assert (data_dir / "profile.json").exists()


def test_profile_set_migrates_legacy_directory(service_without_profiles, tmp_path):
    legacy_dir = tmp_path / "baseload"
    for weekday in range(7):
        payload = {
            "version": 1,
            "created": "2026-01-01 00:00:00",
            "weekday": weekday,
            "period_days": 56,
            "aggregate": "median",
            "baseload": [round(0.1 * weekday, 3)] * 24,
            "samples": [8] * 24,
            "pooled": [False] * 24,
        }
        write_json(legacy_dir / f"baseload_{weekday}.json", payload)

    profile_set = service_without_profiles.profile_set()

    assert profile_set is not None
    assert len(profile_set.home) == 7
    assert (service_without_profiles.data_dir / "profile.json").exists()
