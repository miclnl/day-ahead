"""Choosing between the physical PV model and the ML model.

``physical`` and ``ml`` are honoured as configured. ``auto`` backtests
both over a recent window and keeps the lower error. The backtest is fed
archived forecasts at lead bucket 12/24 where it has them, so each model
is judged on the kind of weather input it will actually be given; where
the archive is too short it falls back to measured weather and says so
(``perfect_weather``), because a model scored on perfect weather looks
better than it will ever be in production.
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from dao.forecast.evaluate import BacktestResult, Selection


def select_pv_model(
    configured: str, backtest_result: Optional[BacktestResult]
) -> Selection:
    """Which PV model to forecast with, and why."""
    now = datetime.datetime.now(datetime.UTC)

    if configured == "physical":
        return Selection(
            model="physical", scores={}, decided_at=now, reason="geconfigureerd"
        )
    if configured == "ml":
        return Selection(model="ml", scores={}, decided_at=now, reason="geconfigureerd")

    if backtest_result is None:
        return Selection(
            model="physical", scores={}, decided_at=now, reason="geen archief"
        )

    scores = {
        name: {"mae": score.mae, "rmse": score.rmse, "bias": score.bias, "n": score.n}
        for name, score in backtest_result.scores.items()
    }
    reason = f"backtest over {backtest_result.days} dagen"
    if "ml" not in backtest_result.scores:
        # Not a comparison at all: say so, or "physical won the backtest"
        # reads as a verdict on a model that was never run.
        reason += " (geen getraind ML-model om tegen te vergelijken)"
    if backtest_result.perfect_weather:
        reason += " (op waarnemingen, niet op gearchiveerde prognoses)"
    return Selection(
        model=backtest_result.winner, scores=scores, decided_at=now, reason=reason
    )


class PhysicalCandidate:
    """The pvlib model as a backtest candidate.

    It has nothing to learn, so ``fit`` is a no-op; the parameters come
    from the calibration artefact that was current when the backtest
    started.
    """

    name = "physical"

    def __init__(self, params, latitude: float, longitude: float, interval_s: int = 3600):
        self.params = params
        self.latitude = latitude
        self.longitude = longitude
        self.interval_s = interval_s

    def fit(self, history: pd.Series, until: datetime.datetime) -> None:
        return None

    def predict(self, day, context: dict) -> np.ndarray:
        from dao.forecast.pv.physical import simulate

        weather = context.get("weather")
        if weather is None or len(weather) == 0:
            return np.zeros(24)
        produced = simulate(
            self.params, weather, self.latitude, self.longitude, self.interval_s
        )
        return produced.to_numpy()


class MLCandidate:
    """The trained XGBoost model as a backtest candidate.

    Like the physical candidate it does not retrain inside the backtest:
    the model under test is the one on disk, which is what a deployed run
    would use. ``fit`` only records that it was asked.
    """

    name = "ml"

    def __init__(self, predictor, installation):
        self.predictor = predictor
        self.installation = installation

    def fit(self, history: pd.Series, until: datetime.datetime) -> None:
        return None

    def predict(self, day, context: dict) -> np.ndarray:
        weather = context.get("weather")
        if weather is None or len(weather) == 0:
            return np.zeros(24)
        try:
            predicted = self.predictor(self.installation, weather)
        except Exception as ex:  # noqa: BLE001 - a losing candidate, not a crash
            logging.debug(f"PV ML-kandidaat kon {day} niet voorspellen: {ex}")
            return np.zeros(24)
        return np.asarray(predicted, dtype=float)
