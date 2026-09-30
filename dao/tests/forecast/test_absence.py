"""Tests for labelling away days from history and calibrating the threshold."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from dao.forecast.baseload.absence import (
    CalendarEvent,
    Regime,
    RegimeSignals,
    calibrate_threshold,
    determine_regime,
    label_days,
    standby_kwh,
)

TZ = ZoneInfo("Europe/Amsterdam")


def presence_series(now: datetime, trailing_values: list[float]) -> pd.Series:
    """Hourly presence fraction for the ``len(trailing_values)`` hours up
    to and including ``now``, oldest first."""
    index = pd.date_range(end=now, periods=len(trailing_values), freq="h", tz=TZ)
    return pd.Series(trailing_values, index=index)


def synthetic_year(away_ranges: list[tuple[str, str]]) -> pd.Series:
    """400 days hourly: standby 0.2, +0.3 daytime (07-23h), +0.6 evening
    peak (17-20h); away days sit on standby alone. Small fixed noise."""
    rng = np.random.default_rng(1)
    index = pd.date_range("2026-01-01", periods=400 * 24, freq="h", tz=TZ)
    away_periods = [
        (pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1))
        for start, end in away_ranges
    ]

    values = np.full(len(index), 0.2)
    hours = index.hour
    is_away = np.zeros(len(index), dtype=bool)
    for start, end in away_periods:
        is_away |= (index >= start) & (index < end)

    daytime = (hours >= 7) & (hours < 23) & ~is_away
    evening = (hours >= 17) & (hours < 20) & ~is_away
    values = values + daytime * 0.3 + evening * 0.6
    values = values + rng.normal(0, 0.01, size=len(index))
    return pd.Series(values, index=index)


def test_standby_is_p10_of_night_hours():
    series = synthetic_year([])
    assert standby_kwh(series) == pytest.approx(0.2, abs=0.02)


def test_label_days_finds_the_vacation():
    series = synthetic_year([("2026-08-09", "2026-08-13")])
    labels = label_days(series)
    assert labels.loc[date(2026, 8, 10)]
    assert labels.sum() == 5


def test_label_days_ignores_days_with_missing_hours():
    series = synthetic_year([])
    target = date(2026, 3, 10)
    gap = pd.Timestamp(target, tz=TZ) + pd.Timedelta(hours=12)
    series.loc[gap] = float("nan")

    labels = label_days(series)

    assert target not in labels.index


def test_label_days_needs_history_first():
    series = synthetic_year([])
    labels = label_days(series)
    assert not labels.iloc[:7].any()


def test_calibrate_threshold_recovers_boundary():
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(20)]
    active_fraction = pd.Series(
        [0.9 if i % 2 == 0 else 0.15 for i in range(20)], index=dates
    )
    presence_daily = pd.Series(
        [0.8 if i % 2 == 0 else 0.01 for i in range(20)], index=dates
    )

    threshold = calibrate_threshold(active_fraction, presence_daily)

    # Every grid value (0.20..0.70) separates 0.15 from 0.9 with 100%
    # agreement; ties go to the highest.
    assert threshold == pytest.approx(0.70)


def test_calibrate_threshold_needs_ten_days():
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(9)]
    active_fraction = pd.Series([0.9] * 9, index=dates)
    presence_daily = pd.Series([0.8] * 9, index=dates)

    assert calibrate_threshold(active_fraction, presence_daily) is None


def test_entity_away_wins_over_everything():
    now = datetime(2026, 3, 10, 12, 0, tzinfo=TZ)
    signals = RegimeSignals(
        now=now,
        entity_state="on",
        presence=presence_series(now, [1.0] * 10),  # everyone home
    )
    regime = determine_regime(now.date(), signals)
    assert regime == Regime(away=True, reason="entity")


def test_entity_home_state_gives_home_even_with_calendar():
    now = datetime(2026, 3, 10, 12, 0, tzinfo=TZ)
    event = CalendarEvent(
        start=datetime(2026, 3, 9, tzinfo=TZ),
        end=datetime(2026, 3, 15, tzinfo=TZ),
        summary="Vakantie Zeeland",
    )
    signals = RegimeSignals(now=now, entity_state="off", calendar_events=[event])
    regime = determine_regime(now.date(), signals)
    assert regime == Regime(away=False, reason="entity")


def test_calendar_keyword_marks_tomorrow():
    now = datetime(2026, 3, 10, 12, 0, tzinfo=TZ)
    tomorrow = now.date() + timedelta(days=1)
    event = CalendarEvent(
        start=datetime(2026, 3, 9, 0, 0, tzinfo=TZ),
        end=datetime(2026, 3, 20, 0, 0, tzinfo=TZ),
        summary="Vakantie Zeeland",
    )
    signals = RegimeSignals(now=now, calendar_events=[event])
    regime = determine_regime(tomorrow, signals)
    assert regime.away is True
    assert regime.reason == "calendar"
    assert regime.switch_hour is None


def test_calendar_partial_day_gives_switch_hour():
    now = datetime(2026, 3, 10, 8, 0, tzinfo=TZ)
    event = CalendarEvent(
        start=datetime(2026, 3, 5, 0, 0, tzinfo=TZ),
        end=datetime(2026, 3, 10, 14, 0, tzinfo=TZ),
        summary="Vakantie Zeeland",
    )
    signals = RegimeSignals(now=now, calendar_events=[event])
    regime = determine_regime(now.date(), signals)
    assert regime.away is True
    assert regime.reason == "calendar"
    assert regime.switch_hour == 14


def test_presence_three_hours_gone_marks_rest_of_today():
    now = datetime(2026, 3, 10, 12, 0, tzinfo=TZ)
    presence = presence_series(now, [1.0, 1.0, 0.0, 0.0, 0.0])  # last 3 hours zero
    signals = RegimeSignals(now=now, presence=presence)
    regime = determine_regime(now.date(), signals)
    assert regime.away is True
    assert regime.reason == "presence"
    assert regime.switch_hour == now.hour


def test_presence_two_hours_not_enough():
    now = datetime(2026, 3, 10, 12, 0, tzinfo=TZ)
    presence = presence_series(now, [1.0, 1.0, 1.0, 0.0, 0.0])  # last 2 hours zero
    signals = RegimeSignals(now=now, presence=presence)
    regime = determine_regime(now.date(), signals)
    assert regime == Regime()


def test_presence_24_hours_marks_tomorrow():
    now = datetime(2026, 3, 10, 12, 0, tzinfo=TZ)
    presence = presence_series(now, [0.0] * 24)
    signals = RegimeSignals(now=now, presence=presence)
    tomorrow = now.date() + timedelta(days=1)
    regime = determine_regime(tomorrow, signals)
    assert regime.away is True
    assert regime.reason == "presence"
    assert regime.switch_hour is None


def test_someone_home_breaks_the_run():
    now = datetime(2026, 3, 10, 12, 0, tzinfo=TZ)
    presence = presence_series(now, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    signals = RegimeSignals(now=now, presence=presence)
    regime = determine_regime(now.date(), signals)
    assert regime.away is True
    assert regime.reason == "presence"
    assert regime.switch_hour == now.hour


def test_consumption_rule_switches_after_six():
    now = datetime(2026, 3, 10, 9, 0, tzinfo=TZ)
    signals = RegimeSignals(
        now=now,
        consumption_today=[0.2] * 9,  # roughly standby all morning
        home_profile_today=[0.5] * 24,  # normal use expected
        standby=0.2,
        threshold=0.4,
    )
    regime = determine_regime(now.date(), signals)
    assert regime.away is True
    assert regime.reason == "consumption"
    assert regime.switch_hour == 9


def test_consumption_rule_not_before_six():
    now = datetime(2026, 3, 10, 5, 0, tzinfo=TZ)
    signals = RegimeSignals(
        now=now,
        consumption_today=[0.2] * 5,
        home_profile_today=[0.5] * 24,
        standby=0.2,
        threshold=0.4,
    )
    regime = determine_regime(now.date(), signals)
    assert regime == Regime()


def test_no_signal_is_home():
    now = datetime(2026, 3, 10, 12, 0, tzinfo=TZ)
    signals = RegimeSignals(now=now)
    regime = determine_regime(now.date(), signals)
    assert regime == Regime()
