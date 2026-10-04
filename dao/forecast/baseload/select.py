"""Choosing between the profile estimator and the ML model.

``profile`` and ``ml`` are the operator's own choice and are simply
honoured. ``auto`` lets the data decide: both models forecast every day of
a recent window using only what was known before that day, and the one
with the lower error wins. Below ``ml min days`` of history ``auto`` does
not even run the backtest -- an XGBoost model on a few weeks of data wins
its own training window and loses every week after it.
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from dao.forecast.baseload.profile import build_profile, effective_weekday, iter_samples
from dao.forecast.evaluate import BacktestResult, Selection


def select_model(
    configured: str,
    history_days: int,
    ml_min_days: int,
    backtest_result: Optional[BacktestResult],
) -> Selection:
    """Which model to forecast with, and why."""
    now = datetime.datetime.now(datetime.UTC)

    if configured == "profile":
        return Selection(model="profile", scores={}, decided_at=now, reason="geconfigureerd")
    if configured == "ml":
        return Selection(model="ml", scores={}, decided_at=now, reason="geconfigureerd")

    if history_days < ml_min_days:
        return Selection(
            model="profile",
            scores={},
            decided_at=now,
            reason=(
                f"te weinig historie ({history_days} van {ml_min_days} dagen) "
                f"voor het ML-model"
            ),
        )

    if backtest_result is None:
        return Selection(
            model="profile",
            scores={},
            decided_at=now,
            reason="geen backtest beschikbaar",
        )

    scores = {
        name: {"mae": score.mae, "rmse": score.rmse, "bias": score.bias, "n": score.n}
        for name, score in backtest_result.scores.items()
    }
    return Selection(
        model=backtest_result.winner,
        scores=scores,
        decided_at=now,
        reason=f"backtest over {backtest_result.days} dagen",
    )


class ProfileCandidate:
    """The profile estimator as a backtest candidate."""

    name = "profile"

    def __init__(self, options) -> None:
        self.options = options
        self.profiles: dict = {}

    def fit(self, history: pd.Series, until: datetime.datetime) -> None:
        rows = list(zip(history.index, history.to_numpy(), strict=True))
        grouped = iter_samples(rows, until, self.options.holidays)
        pooled: dict = {}
        for cells in grouped.values():
            for hour, samples in cells.items():
                pooled.setdefault(hour, []).extend(samples)
        self.profiles = {
            weekday: build_profile(grouped.get(weekday, {}), pooled, self.options)
            for weekday in range(7)
        }

    def predict(self, day, context: dict) -> np.ndarray:
        weekday = effective_weekday(day, self.options.holidays)
        profile = self.profiles.get(weekday)
        if profile is None:
            return np.zeros(24)
        return np.asarray(profile.values, dtype=float)


class MLCandidate:
    """The XGBoost model as a backtest candidate.

    ``model_factory`` returns a fresh, untrained :class:`BaseloadMLModel`;
    each refit inside the backtest gets its own, so no information from a
    later window leaks into an earlier one.
    """

    name = "ml"

    def __init__(self, model_factory, temp: pd.Series, away_labels: pd.Series) -> None:
        self.model_factory = model_factory
        self.temp = temp
        self.away_labels = away_labels
        self.model = None
        self._tz = None

    def fit(self, history: pd.Series, until: datetime.datetime) -> None:
        self._tz = getattr(history.index, "tz", None)
        model = self.model_factory()
        try:
            model.train(history, self.temp, self.away_labels)
            self.model = model
        except (ValueError, RuntimeError) as ex:
            # A window that cannot be trained on is a losing candidate, not
            # a failed backtest: the profile still gets scored on this day.
            logging.debug(f"Baseload ML-kandidaat kon niet trainen op {until}: {ex}")
            self.model = None

    def predict(self, day, context: dict) -> np.ndarray:
        if self.model is None:
            return np.zeros(24)
        start = pd.Timestamp(datetime.datetime.combine(day, datetime.time.min))
        if self._tz is not None:
            start = start.tz_localize(self._tz)
        index = pd.date_range(start, periods=24, freq="h")
        temp = context.get("temp")
        away = bool(context.get("away", False))
        return self.model.predict(index, temp, away).to_numpy()
