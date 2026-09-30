"""Detecting away days from consumption history alone.

Two separate questions, answered separately:

1. Which past days were the household away? Answered purely from the
   consumption history, with no extra sensors needed -- every household
   already has this data the day the estimator is installed.
2. What threshold best matches reality? Answered from ``entities presence``
   once enough days have both a consumption label and a presence reading,
   without needing that from day one.
"""

from __future__ import annotations

import statistics
from typing import Optional

import numpy as np
import pandas as pd

from dao.forecast.baseload.profile import quantile

#: A day is away when its active energy is below this fraction of the
#: median of the preceding window. Overridden once calibration has enough
#: presence history to say something better.
DEFAULT_THRESHOLD = 0.4

#: Local hours a household is almost certainly asleep, used as the floor
#: that daytime activity is measured against.
_NIGHT_HOURS = (1, 2, 3, 4)

#: A day needs at least this many already-processed days before it can be
#: labelled away: with less history the median reference is not trustworthy.
_MIN_HISTORY_DAYS = 7

#: Candidate thresholds for calibration, matching the spec's grid search.
_THRESHOLD_GRID = tuple(round(t, 2) for t in np.arange(0.20, 0.70 + 1e-9, 0.05))


def standby_kwh(series: pd.Series) -> float:
    """The tenth percentile of the night hours (01:00-04:00 local).

    The floor a home never goes below even while occupied, so daytime
    activity can be measured against it rather than against zero. Needs at
    least a full day's worth of night-hour observations (24) or it is
    unknown, not guessed at.
    """
    night = series[series.index.hour.isin(_NIGHT_HOURS)].dropna()
    if len(night) < 24:
        return 0.0
    return round(quantile(sorted(night.values), 0.10), 3)


def active_energy_per_day(series: pd.Series, standby: float) -> pd.Series:
    """Daily total minus what standby alone would have used, date-indexed.

    Only days with all 24 hours measured are trustworthy enough to compare
    against each other; a day with a recorder gap is left out rather than
    scaled up, which would invent activity that was never measured.
    """
    grouped = series.groupby(series.index.date)
    complete = grouped.count() == 24
    totals = grouped.sum()
    result = (totals[complete] - 24 * standby).sort_index()
    result.index.name = None
    return result


def label_days(
    series: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
    window_days: int = 28,
) -> pd.Series:
    """Which days were away, from consumption alone.

    Walks the days in order, keeping a trailing window of already-processed
    days' active energy (home and away both, since "away" is exactly what
    is being decided). A day is away when its active energy falls below
    ``threshold`` times the median of that window. The first few days,
    before the window has enough history to mean anything, are never
    labelled away.
    """
    standby = standby_kwh(series)
    active = active_energy_per_day(series, standby)

    history: list[float] = []
    labels: dict = {}
    for day, value in active.items():
        if len(history) < _MIN_HISTORY_DAYS:
            labels[day] = False
        else:
            reference = statistics.median(history[-window_days:])
            labels[day] = value < threshold * reference
        history.append(value)

    return pd.Series(labels, dtype=bool)


def calibrate_threshold(
    active_fraction: pd.Series, presence_daily: pd.Series
) -> Optional[float]:
    """The threshold that best matches presence, or ``None`` without enough data.

    ``active_fraction`` is a day's active energy divided by the trailing
    median (the same ratio :func:`label_days` compares against a
    threshold); ``presence_daily`` is the mean fraction of configured
    persons present that day. A day counts as actually away when presence
    is below 5%. Needs at least ten days where both are known, otherwise
    the configured threshold is left in place rather than calibrated on too
    little to mean anything. Ties go to the highest threshold in the grid.
    """
    common = active_fraction.index.intersection(presence_daily.index)
    if len(common) < 10:
        return None

    actual_away = presence_daily.loc[common] < 0.05
    fractions = active_fraction.loc[common]

    best_threshold: Optional[float] = None
    best_agreement = -1.0
    for threshold in _THRESHOLD_GRID:
        predicted_away = fractions < threshold
        agreement = float((predicted_away == actual_away).mean())
        if agreement >= best_agreement:
            best_agreement = agreement
            best_threshold = threshold

    return best_threshold
