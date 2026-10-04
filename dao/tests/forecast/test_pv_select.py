"""Tests for the physical | ml | auto PV selection."""

from __future__ import annotations

import datetime as dt
import json

from dao.forecast.evaluate import BacktestResult, Score
from dao.forecast.pv.select import select_pv_model


def make_backtest_result(winner: str, days: int = 28, perfect_weather: bool = False):
    return BacktestResult(
        component="pv",
        days=days,
        scores={
            "physical": Score(mae=0.22, rmse=0.41, bias=0.03, n=days * 24),
            "ml": Score(mae=0.18, rmse=0.33, bias=-0.01, n=days * 24),
        },
        winner=winner,
        perfect_weather=perfect_weather,
        created=dt.datetime.now(dt.UTC),
    )


def test_physical_configured_is_physical():
    selection = select_pv_model("physical", make_backtest_result("ml"))

    assert selection.model == "physical"
    assert selection.reason == "geconfigureerd"


def test_ml_configured_is_ml():
    selection = select_pv_model("ml", None)

    assert selection.model == "ml"
    assert selection.reason == "geconfigureerd"


def test_auto_without_archive_is_physical():
    selection = select_pv_model("auto", None)

    assert selection.model == "physical"
    assert "geen archief" in selection.reason


def test_auto_takes_backtest_winner():
    selection = select_pv_model("auto", make_backtest_result("ml"))

    assert selection.model == "ml"
    assert "backtest" in selection.reason
    assert selection.scores["ml"]["mae"] == 0.18
    assert selection.scores["physical"]["mae"] == 0.22


def test_auto_says_so_when_scored_on_observations():
    """A backtest on measured weather flatters both models; the reason has
    to admit that, or the figure looks like production accuracy."""
    selection = select_pv_model("auto", make_backtest_result("ml", perfect_weather=True))

    assert "waarnemingen" in selection.reason


def test_selection_is_json_serialisable():
    selection = select_pv_model("auto", make_backtest_result("physical"))

    restored = json.loads(json.dumps(selection.to_dict()))
    assert restored["model"] == "physical"
    assert restored["scores"]["ml"]["n"] == 28 * 24
