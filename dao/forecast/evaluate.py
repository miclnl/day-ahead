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
