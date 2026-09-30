"""Tests for labelling away days from history and calibrating the threshold."""

from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from dao.forecast.baseload.absence import (
    calibrate_threshold,
    label_days,
    standby_kwh,
)

TZ = ZoneInfo("Europe/Amsterdam")


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
