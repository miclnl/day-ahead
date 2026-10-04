"""Tests for the walk-forward backtest that the model selectors use."""

from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytest

from dao.forecast.evaluate import backtest, metrics

TZ = "Europe/Amsterdam"


def hourly_target(start: str, days: int, value: float = 0.3) -> pd.Series:
    """A flat hourly series of ``days`` whole days from ``start``."""
    index = pd.date_range(start, periods=days * 24, freq="h", tz=TZ)
    return pd.Series([value] * len(index), index=index)


class ConstantCandidate:
    """Predicts the same value every hour; records how often it was fit."""

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
        self.fit_calls = 0
        self.fit_history_ends: list = []

    def fit(self, history: pd.Series, until: dt.datetime) -> None:
        self.fit_calls += 1
        self.fit_history_ends.append(until)

    def predict(self, day, context) -> np.ndarray:
        return np.full(24, self.value)


def test_metrics_known_values():
    score = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))

    assert score.mae == pytest.approx(1.0)
    assert score.bias == pytest.approx(1.0)
    assert score.rmse == pytest.approx(math.sqrt(5 / 3))
    assert score.n == 3


def test_backtest_refits_weekly():
    # 40 days of history so every day in the window has a past to fit on.
    target = hourly_target("2026-06-01", 40)
    candidate = ConstantCandidate("flat", 0.3)
    end = dt.date(2026, 7, 11)  # 2026-06-01 + 40 days

    result = backtest("baseload", [candidate], target, 28, end=end)

    # Days 0, 7, 14 and 21 of a 28-day window: four refits, not 28.
    assert candidate.fit_calls == 4
    assert result.scores["flat"].n == 28 * 24


def test_backtest_skips_days_with_nan_target():
    target = hourly_target("2026-06-01", 40)
    # Knock out a single hour inside the window; its whole day drops out.
    target.iloc[30 * 24 + 5] = float("nan")
    candidate = ConstantCandidate("flat", 0.3)
    end = dt.date(2026, 7, 11)

    result = backtest("baseload", [candidate], target, 28, end=end)

    assert result.scores["flat"].n == 27 * 24


def test_backtest_picks_lowest_mae():
    target = hourly_target("2026-06-01", 40, value=0.3)
    too_high = ConstantCandidate("A", 0.5)
    exact = ConstantCandidate("B", 0.3)
    end = dt.date(2026, 7, 11)

    result = backtest("baseload", [too_high, exact], target, 28, end=end)

    assert result.winner == "B"
    assert result.scores["B"].mae == pytest.approx(0.0, abs=1e-9)
    assert result.scores["A"].mae == pytest.approx(0.2, abs=1e-9)


def test_backtest_marks_perfect_weather_flag():
    target = hourly_target("2026-06-01", 40)
    candidate = ConstantCandidate("flat", 0.3)
    end = dt.date(2026, 7, 11)

    without = backtest("pv", [candidate], target, 7, end=end)
    with_flag = backtest("pv", [candidate], target, 7, end=end, perfect_weather=True)

    assert without.perfect_weather is False
    assert with_flag.perfect_weather is True


def test_result_to_dict_is_json_serialisable():
    target = hourly_target("2026-06-01", 40)
    candidate = ConstantCandidate("flat", 0.3)
    end = dt.date(2026, 7, 11)

    result = backtest("baseload", [candidate], target, 7, end=end)
    payload = result.to_dict()

    text = json.dumps(payload)
    restored = json.loads(text)
    assert restored["component"] == "baseload"
    assert restored["days"] == 7
    assert restored["winner"] == "flat"
    assert restored["scores"]["flat"]["n"] == 7 * 24
    assert restored["perfect_weather"] is False
