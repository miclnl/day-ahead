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

import logging
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
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

#: The consumption rule needs at least this fraction of today's elapsed
#: hours actually measured before it will call the household away.
MIN_MEASURED_FRACTION = 0.8


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


@dataclass
class Regime:
    """Whether the target day is expected to be a normal or an away day."""

    away: bool = False
    reason: str = "none"  # "entity" | "calendar" | "presence" | "consumption" | "none"
    switch_hour: Optional[int] = None  # local hour the regime changes; None = whole day


@dataclass
class CalendarEvent:
    """One event from the configured Home Assistant calendar."""

    start: datetime
    end: datetime
    summary: str


@dataclass
class RegimeSignals:
    """Everything :func:`determine_regime` needs, gathered from HA and history."""

    now: datetime  # tz-aware
    entity_state: Optional[str] = None
    away_state: str = "on"
    calendar_events: list[CalendarEvent] = field(default_factory=list)
    keywords: list[str] = field(
        default_factory=lambda: ["vakantie", "weg", "afwezig", "holiday"]
    )
    presence: Optional[pd.Series] = None  # hourly fraction present, up to now
    away_after_hours: int = 3
    assume_next_day_after_hours: int = 24
    consumption_today: Optional[pd.Series] = None  # measured baseload of today so far
    home_profile_today: Optional[list[float]] = None
    standby: float = 0.0
    threshold: float = DEFAULT_THRESHOLD


def _event_matches_keywords(event: CalendarEvent, keywords: list[str]) -> bool:
    summary = event.summary.lower()
    return any(keyword.lower() in summary for keyword in keywords)


def _event_covers_day(event: CalendarEvent, day: date) -> bool:
    return event.start.date() <= day <= event.end.date()


def _calendar_switch_hour(event: CalendarEvent, day: date) -> Optional[int]:
    """The local hour the event's edge falls on within ``day``, or ``None``
    when the event covers the whole day."""
    starts_today = event.start.date() == day
    ends_today = event.end.date() == day
    if starts_today and not ends_today:
        return event.start.hour
    if ends_today and not starts_today:
        return event.end.hour
    if starts_today and ends_today:
        return event.start.hour
    return None


def _trailing_zero_run(presence: pd.Series, now: datetime) -> tuple[int, bool]:
    """Length of the contiguous zero-presence run ending at ``now``, and
    whether that run started on ``now``'s own calendar day."""
    ordered = presence.sort_index()
    today = now.date()
    length = 0
    started_today = True
    for moment in reversed(ordered.index):
        if moment > now:
            continue
        if ordered.loc[moment] > 0:
            break
        length += 1
        if moment.date() != today:
            started_today = False
    return length, started_today


def _consumption_regime(signals: RegimeSignals) -> Optional[Regime]:
    """Away purely from today's consumption so far, or ``None`` to say nothing.

    Only hours that were actually measured are compared, and the profile is
    summed over exactly those same hours. Measured hours come from the
    history reader, which leaves a recorder gap as NaN and has not yet seen
    the one or two most recent hours (Home Assistant writes hour *H* at the
    top of *H+1*). Counting those as "consumed nothing" while the profile
    still expects their full value is what made a household behaving
    exactly as predicted look away.

    Below :data:`MIN_MEASURED_FRACTION` of the elapsed hours the day is too
    patchy to judge at all, and no answer is better than a wrong one: the
    whole remaining horizon would otherwise be planned on the away profile.
    """
    hour = signals.now.hour
    measured = pd.Series(signals.consumption_today).dropna()
    measured = measured[[moment.hour < hour for moment in measured.index]]

    if len(measured) < MIN_MEASURED_FRACTION * hour:
        logging.warning(
            f"Afwezigheid: verbruik van vandaag is te onvolledig om te "
            f"beoordelen ({len(measured)} van {hour} uren gemeten), "
            f"thuis aangenomen"
        )
        return None

    hours = [moment.hour for moment in measured.index]
    consumed = float(measured.sum())
    expected = sum(signals.home_profile_today[h] for h in hours)
    floor = len(hours) * signals.standby

    if consumed - floor < signals.threshold * (expected - floor):
        return Regime(away=True, reason="consumption", switch_hour=hour)
    return None


def determine_regime(day: date, signals: RegimeSignals) -> Regime:
    """Home or away for ``day``, from the configured signals in order of
    precedence: entity, calendar, presence, consumption, none. The first
    signal with something to say wins."""
    if signals.entity_state is not None:
        away = signals.entity_state == signals.away_state
        return Regime(away=away, reason="entity")

    for event in signals.calendar_events:
        if _event_matches_keywords(event, signals.keywords) and _event_covers_day(
            event, day
        ):
            return Regime(
                away=True,
                reason="calendar",
                switch_hour=_calendar_switch_hour(event, day),
            )

    today = signals.now.date()
    if signals.presence is not None and day in (today, today + timedelta(days=1)):
        run_length, started_today = _trailing_zero_run(signals.presence, signals.now)
        if day == today and run_length >= signals.away_after_hours:
            switch_hour = signals.now.hour if started_today else None
            return Regime(away=True, reason="presence", switch_hour=switch_hour)
        if (
            day == today + timedelta(days=1)
            and run_length >= signals.assume_next_day_after_hours
        ):
            return Regime(away=True, reason="presence", switch_hour=None)

    if (
        day == today
        and signals.now.hour >= 6
        and signals.consumption_today is not None
        and signals.home_profile_today is not None
    ):
        regime = _consumption_regime(signals)
        if regime is not None:
            return regime

    return Regime()
