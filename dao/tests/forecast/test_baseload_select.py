"""Tests for the profile | ml | auto baseload selection."""

from __future__ import annotations

import datetime as dt

from dao.forecast.baseload.select import select_model
from dao.forecast.evaluate import BacktestResult, Score


def make_backtest_result(winner: str, days: int = 28) -> BacktestResult:
    return BacktestResult(
        component="baseload",
        days=days,
        scores={
            "profile": Score(mae=0.08, rmse=0.11, bias=-0.01, n=days * 24),
            "ml": Score(mae=0.05, rmse=0.07, bias=0.00, n=days * 24),
        },
        winner=winner,
        perfect_weather=False,
        created=dt.datetime.now(dt.UTC),
    )


def test_profile_configured_is_profile():
    selection = select_model("profile", history_days=400, ml_min_days=120,
                             backtest_result=make_backtest_result("ml"))

    assert selection.model == "profile"
    assert selection.reason == "geconfigureerd"


def test_auto_with_short_history_is_profile():
    selection = select_model("auto", history_days=60, ml_min_days=120,
                             backtest_result=make_backtest_result("ml"))

    assert selection.model == "profile"
    assert "te weinig historie" in selection.reason


def test_auto_takes_backtest_winner():
    selection = select_model("auto", history_days=400, ml_min_days=120,
                             backtest_result=make_backtest_result("ml"))

    assert selection.model == "ml"
    assert "backtest" in selection.reason
    assert selection.scores["ml"]["mae"] == 0.05
    assert selection.scores["profile"]["mae"] == 0.08


def test_auto_without_a_backtest_is_profile():
    selection = select_model("auto", history_days=400, ml_min_days=120,
                             backtest_result=None)

    assert selection.model == "profile"
    assert "geen backtest" in selection.reason


def test_ml_configured_without_model_is_ml_with_reason_configured():
    """An explicit "ml" is honoured here even with no trained model on
    disk; the forecast path is what falls back to the profile, so the
    operator sees their own choice reflected plus a warning, rather than a
    silent downgrade in the selection itself."""
    selection = select_model("ml", history_days=10, ml_min_days=120,
                             backtest_result=None)

    assert selection.model == "ml"
    assert selection.reason == "geconfigureerd"


def test_selection_is_json_serialisable():
    import json

    selection = select_model("auto", history_days=400, ml_min_days=120,
                             backtest_result=make_backtest_result("profile"))

    restored = json.loads(json.dumps(selection.to_dict()))
    assert restored["model"] == "profile"
    assert restored["scores"]["ml"]["mae"] == 0.05
