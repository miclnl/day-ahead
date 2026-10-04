"""Walk-forward backtesting for baseload and PV model candidates.

Each candidate sees only data strictly before the day it predicts: the same
discipline a deployed model has, since it can never see today's actual
production before planning today. A candidate that trains (the ML model) is
refit periodically within the window rather than once per day, so a 28-day
backtest finishes in minutes on modest hardware instead of retraining 28
times.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Optional, Protocol

import numpy as np
import pandas as pd


class Candidate(Protocol):
    """One forecasting strategy under comparison."""

    name: str

    def fit(self, history: pd.Series, until: datetime.datetime) -> None:
        """Train (or just remember) on ``history``, strictly before ``until``."""
        ...

    def predict(self, day: date, context: dict) -> np.ndarray:
        """24 hourly values forecast for ``day``."""
        ...


@dataclass
class Score:
    mae: float
    rmse: float
    bias: float
    n: int


@dataclass
class BacktestResult:
    component: str
    days: int
    scores: dict
    winner: str
    perfect_weather: bool
    created: datetime.datetime

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "days": self.days,
            "scores": {
                name: {
                    "mae": score.mae,
                    "rmse": score.rmse,
                    "bias": score.bias,
                    "n": score.n,
                }
                for name, score in self.scores.items()
            },
            "winner": self.winner,
            "perfect_weather": self.perfect_weather,
            "created": self.created.isoformat(),
        }


@dataclass
class Selection:
    """Which model a selector picked, and what it based that on.

    Shared by the baseload and PV selectors so both write the same
    ``selection.json`` shape, and the dashboard can render either without
    knowing which component it came from.
    """

    model: str
    scores: dict
    decided_at: datetime.datetime
    reason: str

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "scores": self.scores,
            "decided_at": self.decided_at.isoformat(),
            "reason": self.reason,
        }


def metrics(forecast: np.ndarray, actual: np.ndarray) -> Score:
    """MAE, RMSE and bias (forecast - actual, so positive means over-forecast)."""
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    diff = forecast - actual
    return Score(
        mae=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff**2))),
        bias=float(np.mean(diff)),
        n=int(len(actual)),
    )


def backtest(
    component: str,
    candidates: list,
    target: pd.Series,
    days: int,
    *,
    end: date,
    retrain_every_days: int = 7,
    context_for_day: Optional[Callable[[date], dict]] = None,
    perfect_weather: bool = False,
) -> BacktestResult:
    """Walk-forward comparison of ``candidates`` on the ``days`` days before ``end``.

    Each day, a candidate not yet fitted or due for its periodic refit
    (every ``retrain_every_days`` days) is fit on ``target`` history
    strictly before that day; its prediction is then compared against
    ``target``'s 24 hourly values for the day. A day with fewer than 24
    values or any NaN among them is skipped for every candidate, so the
    comparison stays on exactly the same days for all of them.
    """
    context_for_day = context_for_day or (lambda day: {})

    # Everything else in this package indexes on tz-aware timestamps; a
    # naive day boundary cannot be compared against those at all (pandas
    # raises), so the boundary follows whatever the target carries.
    target_tz = getattr(target.index, "tz", None)

    last_fit: dict = {}
    forecasts_by_name: dict = {candidate.name: [] for candidate in candidates}
    actuals_by_name: dict = {candidate.name: [] for candidate in candidates}

    day = end - datetime.timedelta(days=days)
    while day < end:
        day_start = pd.Timestamp(datetime.datetime.combine(day, datetime.time.min))
        if target_tz is not None:
            day_start = day_start.tz_localize(target_tz)
        day_end = day_start + pd.Timedelta(days=1)
        day_values = target[(target.index >= day_start) & (target.index < day_end)]

        if len(day_values) != 24 or day_values.isna().any():
            day += datetime.timedelta(days=1)
            continue

        actual = day_values.to_numpy()
        context = context_for_day(day)

        for candidate in candidates:
            fitted_on = last_fit.get(candidate.name)
            if fitted_on is None or (day - fitted_on).days >= retrain_every_days:
                history = target[target.index < day_start]
                candidate.fit(history, day_start.to_pydatetime())
                last_fit[candidate.name] = day

            forecast = np.asarray(candidate.predict(day, context), dtype=float)
            forecasts_by_name[candidate.name].append(forecast)
            actuals_by_name[candidate.name].append(actual)

        day += datetime.timedelta(days=1)

    scores: dict = {}
    for candidate in candidates:
        name = candidate.name
        if not forecasts_by_name[name]:
            scores[name] = Score(mae=float("nan"), rmse=float("nan"), bias=float("nan"), n=0)
            continue
        all_forecast = np.concatenate(forecasts_by_name[name])
        all_actual = np.concatenate(actuals_by_name[name])
        scores[name] = metrics(all_forecast, all_actual)

    scored = {name: score for name, score in scores.items() if score.n > 0}
    if scored:
        # min() keeps the first of equal minima, which -- since scored
        # preserves candidates' own order -- is "ties go to the first
        # candidate" without any extra bookkeeping.
        winner = min(scored, key=lambda name: scored[name].mae)
    else:
        winner = candidates[0].name if candidates else ""

    return BacktestResult(
        component=component,
        days=days,
        scores=scores,
        winner=winner,
        perfect_weather=perfect_weather,
        created=datetime.datetime.now(datetime.UTC),
    )


# ---------------------------------------------------------------------------
# Accuracy of the archive against what was actually measured
# ---------------------------------------------------------------------------

#: Every archived series worth scoring, and the unit it is reported in.
COMPONENTS = ("base", "pv_ac", "pv_dc", "hload", "gr", "dni", "dhi", "temp")

_UNITS = {
    "base": "kWh",
    "pv_ac": "kWh",
    "pv_dc": "kWh",
    "hload": "kWh",
    "gr": "J/cm2",
    "dni": "J/cm2",
    "dhi": "J/cm2",
    "temp": "\u00b0C",
}

#: Weather components are measured in "values", everything else is derived
#: from the Home Assistant meters through the history reader.
_WEATHER_COMPONENTS = ("gr", "dni", "dhi", "temp")


@dataclass
class ComponentAccuracy:
    component: str
    unit: str
    windows: dict  # days -> grouped scores

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "unit": self.unit,
            "windows": {
                str(days): {
                    group: (
                        {str(key): _score_to_dict(score) for key, score in value.items()}
                        if isinstance(value, dict)
                        else value
                    )
                    for group, value in window.items()
                }
                for days, window in self.windows.items()
            },
        }


@dataclass
class AccuracyReport:
    created: datetime.datetime
    days: tuple
    components: dict

    def to_dict(self) -> dict:
        return {
            "created": self.created.isoformat(),
            "days": list(self.days),
            "components": {
                name: accuracy.to_dict() for name, accuracy in self.components.items()
            },
        }


def _score_to_dict(score: Score) -> dict:
    return {"mae": score.mae, "rmse": score.rmse, "bias": score.bias, "n": score.n}


def _empty_window() -> dict:
    return {
        "by_lead": {},
        "by_hour": {},
        "by_weekday": {},
        "by_regime": {},
        "by_source": {},
        "pairs": 0,
        "missing": 0,
    }


def measured_series(
    component: str, reader, config, db_da, start, end
) -> Optional[pd.Series]:
    """What actually happened for ``component`` between ``start`` and ``end``.

    Returns ``None`` when the component cannot be measured at all on this
    installation (no configured sensors, no weather observations), which
    the report records as "no pairs" rather than as an error.
    """
    from dao.forecast.history import (
        baseload_from_components,
        component_caps,
        component_groups,
    )

    if component in _WEATHER_COMPONENTS:
        if db_da is None:
            return None
        try:
            frame = db_da.get_column_data("values", component, start=start, end=end)
        except Exception as ex:  # noqa: BLE001 - an absent series, not a failure
            logging.debug(f"Accuratesse: {component} niet leesbaar: {ex}")
            return None
        if frame is None or len(frame) == 0:
            return None
        index = pd.to_datetime(frame["utc"], unit="s", utc=True)
        return pd.Series(frame["value"].astype(float).to_numpy(), index=index)

    if reader is None:
        return None

    if component in ("pv_ac", "pv_dc"):
        if component == "pv_ac":
            installations = list(config.solar or [])
        else:
            installations = [
                solar
                for battery in (config.battery or [])
                for solar in (battery.solar or [])
            ]
        sensors = [
            sensor
            for installation in installations
            for sensor in (installation.entities_sensors or [])
        ]
        if not sensors:
            return None
        capacity = sum(
            (installation.total_capacity or 0.0) for installation in installations
        )
        return reader.read_energy(
            sensors, start, end, cap_kwh=1.2 * capacity if capacity else None
        )

    groups = component_groups(config.report)
    frame = reader.read_components(groups, start, end, component_caps(config))
    if component == "base":
        return baseload_from_components(frame)
    # hload: what the optimizer plans for the house as a whole, which is
    # the grid exchange corrected for whatever the battery did -- PV and
    # the scheduled devices are deliberately still inside it.
    return (
        frame["grid_in"] - frame["grid_out"] - frame["bat_in"] + frame["bat_out"]
    )


def _away_dates(db_da, start, end) -> set:
    """Dates labelled away, for splitting the report by regime."""
    if db_da is None:
        return set()
    try:
        frame = db_da.get_column_data("values", "away", start=start, end=end)
    except Exception:  # noqa: BLE001 - the split is optional
        return set()
    if frame is None or len(frame) == 0:
        return set()
    moments = pd.to_datetime(frame["utc"], unit="s", utc=True)
    return {
        moment.date()
        for moment, value in zip(moments, frame["value"], strict=True)
        if float(value) >= 0.5
    }


def _window_scores(archive: pd.DataFrame, measured, away_dates: set, tz: str) -> dict:
    """Group one component's archived/measured pairs every way the report shows."""
    window = _empty_window()
    if archive is None or len(archive) == 0:
        return window
    if measured is None or len(measured) == 0:
        window["missing"] = int(len(archive))
        return window

    lookup = {
        int(pd.Timestamp(moment).timestamp()): float(value)
        for moment, value in measured.items()
        if value == value  # skip NaN
    }

    grouped: dict = {
        "by_lead": {},
        "by_hour": {},
        "by_weekday": {},
        "by_regime": {},
        "by_source": {},
    }
    missing = 0
    for row in archive.itertuples():
        actual = lookup.get(int(row.target_time))
        if actual is None:
            missing += 1
            continue
        forecast = float(row.value)
        moment = pd.Timestamp(int(row.target_time), unit="s", tz="UTC").tz_convert(tz)
        keys = {
            "by_lead": int(row.lead_bucket),
            "by_hour": int(moment.hour),
            "by_weekday": int(moment.dayofweek),
            "by_regime": "away" if moment.date() in away_dates else "home",
            "by_source": getattr(row, "source", None) or "onbekend",
        }
        for group, key in keys.items():
            grouped[group].setdefault(key, []).append((forecast, actual))

    window["missing"] = missing
    window["pairs"] = sum(len(pairs) for pairs in grouped["by_lead"].values())
    for group, buckets in grouped.items():
        window[group] = {
            key: metrics(
                np.array([f for f, _ in pairs]), np.array([a for _, a in pairs])
            )
            for key, pairs in buckets.items()
        }
    return window


def archive_accuracy(
    config, db_da, db_ha, tz: str, days: tuple = (7, 28), now=None
) -> AccuracyReport:
    """Compare every archived forecast against what was measured.

    Every component is scored over every window independently, so one
    component with no sensors configured does not take the rest of the
    report down with it.
    """
    from dao.forecast.history import HistoryReader
    from dao.lib.db_manager import LEAD_BUCKETS

    now = now or datetime.datetime.now(datetime.UTC)
    reader = HistoryReader(db_ha, tz) if db_ha is not None else None

    components: dict = {}
    for component in COMPONENTS:
        windows: dict = {}
        for window_days in days:
            start = now - datetime.timedelta(days=window_days)
            try:
                archive = db_da.forecast_rows(
                    [component], list(LEAD_BUCKETS), int(start.timestamp()),
                    int(now.timestamp()),
                )
            except Exception as ex:  # noqa: BLE001 - an empty archive is normal
                logging.debug(f"Accuratesse: archief van {component} niet leesbaar: {ex}")
                archive = None

            if archive is None or len(archive) == 0:
                windows[window_days] = _empty_window()
                continue

            try:
                measured = measured_series(component, reader, config, db_da, start, now)
            except Exception as ex:  # noqa: BLE001 - report "no pairs", not a crash
                logging.debug(f"Accuratesse: {component} niet te meten: {ex}")
                measured = None

            windows[window_days] = _window_scores(
                archive, measured, _away_dates(db_da, start, now), tz
            )

        components[component] = ComponentAccuracy(
            component=component, unit=_UNITS[component], windows=windows
        )

    return AccuracyReport(created=now, days=tuple(days), components=components)
