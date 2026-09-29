"""Model persistence: xgboost's native save_model/load_model instead of
joblib pickling the whole estimator object.

joblib.dump() pickles private attributes of the XGBRegressor wrapper whose
layout has broken across major xgboost releases, so a model trained on one
xgboost version could fail to load after an upgrade. save_model()/
load_model() write the booster's own portable format instead, and a small
sidecar JSON records the feature columns the model was trained with so a
mismatch is caught with a clear error rather than silently misaligned
predictions.
"""

import json

import numpy as np
import pytest
from xgboost import XGBRegressor

from dao.prog.solar_predictor import SolarPredictor, _metadata_path


def make_predictor(feature_columns):
    predictor = SolarPredictor.__new__(SolarPredictor)
    predictor.feature_columns = feature_columns
    return predictor


def fit_and_save(tmp_path, feature_columns, filename="model.json"):
    rng = np.random.default_rng(0)
    X = rng.random((20, len(feature_columns)))
    y = rng.random(20)
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X, y)

    predictor = make_predictor(feature_columns)
    predictor.model = model
    model_path = str(tmp_path / filename)

    # Mirrors the tail end of train(): save_model + sidecar metadata.
    model.save_model(model_path)
    with open(_metadata_path(model_path), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_columns}, f)

    return predictor, model_path


def test_load_model_restores_a_working_predictor(tmp_path):
    columns = ["temperature", "irradiance", "windvelocity"]
    original, model_path = fit_and_save(tmp_path, columns)

    loaded = make_predictor(columns)
    loaded.load_model(model_path)

    assert loaded.is_trained is True
    X = np.random.default_rng(1).random((3, len(columns)))
    np.testing.assert_allclose(
        loaded.model.predict(X), original.model.predict(X)
    )


def test_load_model_rejects_a_feature_column_mismatch(tmp_path):
    trained_with = ["temperature", "irradiance", "windvelocity"]
    _, model_path = fit_and_save(tmp_path, trained_with)

    # Current code expects a different (e.g. extended) feature set.
    loader = make_predictor(trained_with + ["season"])

    with pytest.raises(ValueError, match="feature columns"):
        loader.load_model(model_path)


def test_load_model_without_a_sidecar_still_loads(tmp_path):
    """Backward-compat: a model saved before this change (no .meta.json)
    must still load rather than crash on a missing sidecar file."""
    columns = ["temperature", "irradiance", "windvelocity"]
    rng = np.random.default_rng(2)
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(rng.random((20, len(columns))), rng.random(20))
    model_path = str(tmp_path / "no_sidecar.json")
    model.save_model(model_path)

    loader = make_predictor(columns)
    loader.load_model(model_path)

    assert loader.is_trained is True


def test_load_model_raises_file_not_found_for_a_missing_path(tmp_path):
    loader = make_predictor(["temperature"])
    with pytest.raises(FileNotFoundError):
        loader.load_model(str(tmp_path / "missing.json"))
