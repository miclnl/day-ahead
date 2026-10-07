"""Tests for the baseload service: fit, forecast, and the fallback chain."""

from __future__ import annotations

import datetime as dt
import json
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


def make_config(
    *,
    baseload=None,
    calc_periode=56,
    aggregate="mean",
    absence_detect=True,
    entities_presence=None,
    entity_away=None,
    entity_calendar=None,
    model="profile",
    ml_min_days=120,
    backtest_days=28,
):
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
            model=model,
            ml_min_days=ml_min_days,
            backtest_days=backtest_days,
            absence=types.SimpleNamespace(
                detect=absence_detect,
                threshold=0.4,
                entities_presence=entities_presence or [],
                entity_away=entity_away,
                away_state="on",
                entity_calendar=entity_calendar,
                calendar_keywords=["vakantie", "weg", "afwezig", "holiday"],
                away_after_hours=3,
                assume_next_day_after_hours=24,
            ),
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


def _synthetic_history_with_gap(ha_db, da_db, tmp_path, *, gap_days: int):
    """60 days of day/night history with a low-consumption gap of
    ``gap_days`` days, twenty days in -- long enough to have history before
    it. Day/night variation matters here: :func:`standby_kwh` reads it off
    the night hours, so a flat day would make "active energy" collapse to
    noise around zero."""
    manager, helper = ha_db
    now = dt.datetime(2026, 3, 4, 6, 0, tzinfo=ZONE)
    period_days = 60
    tot = now.replace(hour=0, minute=0, second=0, microsecond=0)
    vanaf = tot - dt.timedelta(days=period_days)
    n_hours = period_days * 24
    gap_start = vanaf + dt.timedelta(days=20)
    gap_end = gap_start + dt.timedelta(days=gap_days)

    grid_in: dict[int, float] = {}
    total = 0.0
    for i in range(n_hours + 1):
        moment = vanaf + dt.timedelta(hours=i)
        grid_in[int(moment.timestamp())] = total
        if i < n_hours:
            if gap_start <= moment < gap_end:
                step = 0.05
            else:
                step = 0.3 if 7 <= moment.hour < 23 else 0.1
            total += step

    helper.add_energy("sensor.test_grid_in", "kWh", grid_in)
    helper.add_energy("sensor.test_grid_out", "kWh", {ts: 0.0 for ts in grid_in})
    helper.add_power("sensor.test_pv", "W", {ts: 0.0 for ts in list(grid_in)[:-1]})

    config = make_config(baseload=None, calc_periode=period_days)
    data_dir = tmp_path / "forecast" / "baseload"
    service = BaseloadService(
        config, db_da=da_db, db_ha=manager, data_dir=data_dir, tz=TZ, now=lambda: now
    )
    return service, gap_start.date(), gap_end.date()


@pytest.fixture
def service_with_history_and_vacation(ha_db, da_db, tmp_path):
    """A six day dip in consumption, long enough for the "labels" source."""
    return _synthetic_history_with_gap(ha_db, da_db, tmp_path, gap_days=6)


@pytest.fixture
def service_with_history_short_vacation(ha_db, da_db, tmp_path):
    """A two day dip, too short for the "labels" source: standby instead."""
    service, _, _ = _synthetic_history_with_gap(ha_db, da_db, tmp_path, gap_days=2)
    return service


@pytest.fixture
def fake_ha():
    """A stub HA client: configurable entity states, no calendar events."""

    class FakeState:
        def __init__(self, state):
            self.state = state

    class FakeHA:
        def __init__(self):
            self.states: dict[str, str] = {}

        def get_state(self, entity_id):
            return FakeState(self.states.get(entity_id, "unknown"))

        def get_calendar_events(self, entity_id, start, end):
            return []

    return FakeHA()


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


def _service_with_no_rows_in_window(ha_db, tmp_path, *, previous=True):
    """Sensors registered in ``statistics_meta`` but without a single row in
    the window ``fit()`` reads: a recorder purge, a renamed entity, a meter
    that stopped reporting."""
    manager, helper = ha_db
    now = dt.datetime(2026, 3, 4, 6, 0, tzinfo=ZONE)
    period_days = 60
    tot = now.replace(hour=0, minute=0, second=0, microsecond=0)
    vanaf = tot - dt.timedelta(days=period_days)

    stale = {
        int((vanaf - dt.timedelta(hours=i)).timestamp()): 0.3 * (24 - i)
        for i in range(1, 25)
    }
    helper.add_energy("sensor.test_grid_in", "kWh", stale)
    helper.add_energy("sensor.test_grid_out", "kWh", dict.fromkeys(stale, 0.0))
    helper.add_power("sensor.test_pv", "W", dict.fromkeys(stale, 0.0))

    data_dir = tmp_path / "forecast" / "baseload"
    if previous:
        save_profile_set(
            ProfileSet(
                created=dt.datetime(2026, 2, 1, tzinfo=ZONE),
                period_days=period_days,
                aggregate="mean",
                home={wd: flat_profile(0.4) for wd in range(7)},
            ),
            data_dir,
        )
    config = make_config(baseload=None, calc_periode=period_days, absence_detect=False)
    service = BaseloadService(
        config, db_da=None, db_ha=manager, data_dir=data_dir, tz=TZ, now=lambda: now
    )
    return service, data_dir


def test_fit_keeps_the_previous_profile_when_the_window_is_empty(ha_db, tmp_path, caplog):
    service, data_dir = _service_with_no_rows_in_window(ha_db, tmp_path)

    with caplog.at_level(logging.WARNING):
        profile_set = service.fit()

    assert profile_set.home[0].values == [0.4] * 24
    saved = json.loads((data_dir / "profile.json").read_text(encoding="utf-8"))
    assert saved["home"]["0"]["values"] == [0.4] * 24
    assert saved["created"].startswith("2026-02-01")
    assert any(
        "niet herberekend" in record.message and "0 gemeten uren" in record.message
        for record in caplog.records
    )


def test_fit_without_history_and_without_a_previous_profile_raises(ha_db, tmp_path):
    service, data_dir = _service_with_no_rows_in_window(ha_db, tmp_path, previous=False)

    with pytest.raises(BaseloadUnavailable):
        service.fit()

    assert not (data_dir / "profile.json").exists()


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


def test_fit_labels_vacation_and_builds_away_profile(service_with_history_and_vacation):
    service, vacation_start, vacation_end = service_with_history_and_vacation

    profile_set = service.fit()

    assert profile_set.away is not None
    assert profile_set.away_source == "labels"
    assert sum(profile_set.away.values) < 0.6 * sum(profile_set.home[1].values)

    frame = service.db_da.get_column_data(
        "values",
        "away",
        start=dt.datetime.combine(vacation_start, dt.time.min, tzinfo=ZONE),
        end=dt.datetime.combine(vacation_end, dt.time.min, tzinfo=ZONE),
    )
    assert len(frame) == (vacation_end - vacation_start).days
    assert list(frame["value"]) == [1.0] * len(frame)


def test_fit_with_two_away_days_uses_standby(service_with_history_short_vacation):
    profile_set = service_with_history_short_vacation.fit()

    assert profile_set.away is not None
    assert profile_set.away_source == "standby"


def test_forecast_picks_regime_from_entity(service_with_profiles, fake_ha):
    fake_ha.states["input_boolean.away"] = "on"
    service_with_profiles.ha = fake_ha
    service_with_profiles.config.baseload_options.absence.entity_away = (
        "input_boolean.away"
    )

    start = dt.datetime(2026, 3, 2, 0, 0, tzinfo=ZONE)  # Monday, home 0.3, away 0.1
    series = service_with_profiles.forecast(start, 3)

    assert list(series.values) == pytest.approx([0.1, 0.1, 0.1])

    status = json.loads(
        (service_with_profiles.data_dir / "status.json").read_text()
    )
    assert status["regime"] == "away"
    assert status["reason"] == "entity"


def test_current_regime_is_home_when_today_matches_the_profile(ha_db, tmp_path):
    """The whole path, from the recorder to the regime: a household using
    exactly what its profile predicts, read back through the history reader
    with its real NaN tail, must not be called away."""
    manager, helper = ha_db
    now = dt.datetime(2026, 3, 4, 6, 30, tzinfo=ZONE)  # Wednesday morning
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # 0.25 kWh overnight, 0.6 kWh from 06:00: a plausible weekday. Home
    # Assistant has written through 04:00, so 05:00 and 06:00 are missing.
    profile_values = [0.25] * 6 + [0.6] * 18
    grid_in: dict[int, float] = {}
    total = 0.0
    for hour in range(6):
        grid_in[int((midnight + dt.timedelta(hours=hour)).timestamp())] = total
        total += profile_values[hour]

    helper.add_energy("sensor.test_grid_in", "kWh", grid_in)
    helper.add_energy("sensor.test_grid_out", "kWh", dict.fromkeys(grid_in, 0.0))
    helper.add_power("sensor.test_pv", "W", dict.fromkeys(grid_in, 0.0))

    home = {
        wd: BaseloadProfile(
            values=list(profile_values), samples=[8] * 24, pooled=[False] * 24
        )
        for wd in range(7)
    }
    data_dir = tmp_path / "forecast" / "baseload"
    save_profile_set(
        ProfileSet(
            created=dt.datetime(2026, 3, 3, tzinfo=ZONE),
            period_days=56,
            aggregate="mean",
            home=home,
            standby_kwh=0.2,
        ),
        data_dir,
    )
    service = BaseloadService(
        make_config(baseload=None),
        db_da=None,
        db_ha=manager,
        data_dir=data_dir,
        tz=TZ,
        now=lambda: now,
    )

    regime = service.current_regime(now.date())

    assert regime.away is False
    assert regime.reason == "none"


def test_record_presence_writes_fraction(da_db, fake_ha, tmp_path):
    fake_ha.states["person.a"] = "home"
    fake_ha.states["person.b"] = "not_home"
    config = make_config(entities_presence=["person.a", "person.b"])
    now = dt.datetime(2026, 3, 4, 12, 0, tzinfo=ZONE)
    data_dir = tmp_path / "forecast" / "baseload"
    service = BaseloadService(
        config,
        db_da=da_db,
        db_ha=None,
        data_dir=data_dir,
        tz=TZ,
        now=lambda: now,
        ha=fake_ha,
    )

    service.record_presence()

    frame = da_db.get_column_data(
        "values",
        "presence",
        start=now - dt.timedelta(hours=1),
        end=now + dt.timedelta(hours=1),
    )
    assert list(frame["value"]) == [0.5]


def test_record_presence_noop_without_entities(service_with_profiles):
    service_with_profiles.record_presence()  # no entities configured; must not raise


@pytest.fixture
def service_with_long_history(ha_db, tmp_path):
    """150 days of hourly history with a real day/night shape, so "auto"
    has both enough days for the ML model and a pattern to learn."""
    manager, helper = ha_db
    now = dt.datetime(2026, 6, 1, 6, 0, tzinfo=ZONE)
    period_days = 150
    tot = now.replace(hour=0, minute=0, second=0, microsecond=0)
    vanaf = tot - dt.timedelta(days=period_days)
    n_hours = period_days * 24

    grid_in: dict[int, float] = {}
    total = 0.0
    for i in range(n_hours + 1):
        moment = vanaf + dt.timedelta(hours=i)
        grid_in[int(moment.timestamp())] = total
        if i < n_hours:
            hour = moment.hour
            total += 0.8 if 17 <= hour < 21 else (0.3 if 7 <= hour < 23 else 0.1)

    helper.add_energy("sensor.test_grid_in", "kWh", grid_in)
    helper.add_energy("sensor.test_grid_out", "kWh", {ts: 0.0 for ts in grid_in})
    helper.add_power("sensor.test_pv", "W", {ts: 0.0 for ts in list(grid_in)[:-1]})

    config = make_config(
        baseload=None,
        calc_periode=period_days,
        model="auto",
        ml_min_days=120,
        backtest_days=7,
    )
    data_dir = tmp_path / "forecast" / "baseload"
    service = BaseloadService(
        config, db_da=None, db_ha=manager, data_dir=data_dir, tz=TZ, now=lambda: now
    )
    return service, data_dir


def test_fit_auto_writes_selection_json(service_with_long_history):
    service, data_dir = service_with_long_history

    service.fit()

    selection = json.loads((data_dir / "selection.json").read_text())
    assert selection["model"] in ("profile", "ml")
    assert "backtest" in selection["reason"]
    assert set(selection["scores"]) == {"profile", "ml"}
    assert selection["scores"]["ml"]["n"] > 0


def test_forecast_falls_back_to_profile_when_model_missing(tmp_path, caplog):
    """selection.json says ml, but no model was ever saved: the profile
    must answer anyway, with a warning rather than an exception."""
    home = {wd: flat_profile(0.3) for wd in range(7)}
    service = build_service(tmp_path, home, config=make_config(model="ml"))
    write_json(
        service.data_dir / "selection.json",
        {
            "model": "ml",
            "scores": {},
            "decided_at": dt.datetime(2026, 3, 4, tzinfo=ZONE).isoformat(),
            "reason": "geconfigureerd",
        },
    )

    with caplog.at_level(logging.WARNING):
        series = service.forecast(dt.datetime(2026, 3, 2, 0, 0, tzinfo=ZONE), 3)

    assert list(series.values) == pytest.approx([0.3, 0.3, 0.3])
    assert any("ML-model" in message for message in caplog.messages)
