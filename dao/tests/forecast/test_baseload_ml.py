"""Tests for the XGBoost baseload model."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from dao.forecast.baseload.ml import (
    ML_FEATURES,
    BaseloadMLModel,
    build_features,
    climatological_temp,
)

TZ = "Europe/Amsterdam"
LAT, LON = 52.1, 5.2


def synthetic_history(days: int = 120):
    """Hourly consumption with a clear hour-of-day shape plus mild noise."""
    rng = np.random.default_rng(7)
    index = pd.date_range("2026-01-01", periods=days * 24, freq="h", tz=TZ)
    hours = np.asarray(index.hour)
    # Night 0.15, day 0.35, evening peak 0.8 -- a shape a flat mean cannot match.
    base = np.where(hours < 7, 0.15, np.where(hours < 17, 0.35, 0.8))
    base = np.where(hours >= 22, 0.15, base)
    target = pd.Series(base + rng.normal(0, 0.02, size=len(index)), index=index)
    temp = pd.Series(10.0 + 5 * np.sin(np.arange(len(index)) / 500.0), index=index)
    away_labels = pd.Series(dtype=bool)
    return target, temp, away_labels


def test_features_have_expected_columns_and_no_nan():
    index = pd.date_range("2026-06-01", periods=48, freq="h", tz=TZ)
    temp = pd.Series(15.0, index=index)
    away = pd.Series(0.0, index=index)

    features = build_features(index, temp, away, LAT, LON, "sunday")

    assert list(features.columns) == list(ML_FEATURES)
    assert not features.isna().any().any()
    assert len(features) == 48


def test_train_predict_learns_hour_pattern():
    target, temp, away_labels = synthetic_history(120)
    split = len(target) - 14 * 24
    train_target, test_target = target.iloc[:split], target.iloc[split:]

    model = BaseloadMLModel(LAT, LON, "sunday")
    model.train(train_target, temp.iloc[:split], away_labels)

    predicted = model.predict(test_target.index, temp.iloc[split:], away=False)
    model_mae = float(np.mean(np.abs(predicted.to_numpy() - test_target.to_numpy())))
    flat_mae = float(
        np.mean(np.abs(train_target.mean() - test_target.to_numpy()))
    )

    assert model_mae < flat_mae


def test_predict_with_missing_temp_uses_climatology():
    """A horizon hour without a temperature forecast must still produce a
    number, not a NaN that poisons the whole plan."""
    target, temp, away_labels = synthetic_history(120)
    model = BaseloadMLModel(LAT, LON, "sunday")
    model.train(target, temp, away_labels)

    index = pd.date_range("2026-07-01", periods=24, freq="h", tz=TZ)
    partial_temp = pd.Series(float("nan"), index=index)
    filled = partial_temp.fillna(climatological_temp(temp, 7))

    predicted = model.predict(index, filled, away=False)

    assert len(predicted) == 24
    assert not predicted.isna().any()
    assert (predicted >= 0).all()


def test_save_load_round_trip(tmp_path):
    target, temp, away_labels = synthetic_history(60)
    model = BaseloadMLModel(LAT, LON, "sunday")
    model.train(target, temp, away_labels)
    model.save(tmp_path)

    loaded = BaseloadMLModel.load(tmp_path, LAT, LON, "sunday")

    assert loaded is not None
    index = target.index[-24:]
    np.testing.assert_allclose(
        loaded.predict(index, temp.loc[index], away=False).to_numpy(),
        model.predict(index, temp.loc[index], away=False).to_numpy(),
    )
    meta = json.loads((tmp_path / "model.meta.json").read_text())
    assert meta["features"] == list(ML_FEATURES)
    assert meta["rows"] == model.rows


def test_load_returns_none_without_a_model(tmp_path):
    assert BaseloadMLModel.load(tmp_path, LAT, LON, "sunday") is None


def test_climatological_temp_default_ten_when_empty():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz=TZ))
    assert climatological_temp(empty, 7) == pytest.approx(10.0)

    index = pd.date_range("2026-07-01", periods=48, freq="h", tz=TZ)
    july = pd.Series(21.0, index=index)
    assert climatological_temp(july, 7) == pytest.approx(21.0)
    # A month the series says nothing about falls back to the default.
    assert climatological_temp(july, 1) == pytest.approx(10.0)


def test_away_days_counts_labelled_days():
    target, temp, _ = synthetic_history(60)
    labels = pd.Series(
        {dt.date(2026, 1, 10) + dt.timedelta(days=i): i < 3 for i in range(6)}
    )

    model = BaseloadMLModel(LAT, LON, "sunday")
    model.train(target, temp, labels)

    assert model.away_days == 3
