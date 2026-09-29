"""ML methodology fixes in SolarPredictor.train() / _load_and_process_solar_data.

1. Resampling solar readings to hourly used pandas' default min_count=0, so
   an hour with zero underlying samples (a recorder or inverter outage)
   summed to 0.0 -- indistinguishable from "no production", which is only
   correct at night. min_count=1 makes an empty bin resolve to NaN, which
   the existing dropna() calls in train() then correctly exclude.
2. GridSearchCV used cv=3 (plain KFold), which for autocorrelated weather/
   production time series can validate a fold on data that precedes the
   data it was trained on -- future information leaking into the score.
   TimeSeriesSplit only ever validates on data after its training fold.
3. warnings.filterwarnings("ignore") used to be set at import time, which
   silenced warnings process-wide for every module that imports
   solar_predictor. It is now scoped to just the GridSearchCV.fit() call.
"""

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dao.prog.solar_predictor import SolarPredictor


def make_predictor(tune_hyperparameters, param_grid=None, parameters=None):
    predictor = SolarPredictor.__new__(SolarPredictor)
    predictor.feature_columns = ["temperature", "irradiance", "windvelocity"]
    predictor.random_state = 42
    predictor.solar_name = "test"
    predictor.config = SimpleNamespace(
        xgboost=SimpleNamespace(
            tune_hyperparameters=tune_hyperparameters,
            param_grid=param_grid,
            parameters=parameters,
        )
    )
    return predictor


def hourly_frame(n_hours, columns):
    index = pd.date_range(
        "2026-01-01", periods=n_hours, freq="h", tz="UTC"
    )
    rng = np.random.default_rng(0)
    data = {col: rng.random(n_hours) for col in columns}
    return pd.DataFrame(data, index=index)


class TestResampleMinCount:
    def test_empty_hour_resamples_to_nan_not_zero(self):
        predictor = make_predictor(tune_hyperparameters=False)
        # Two readings inside the same hour, then a gap of a full hour with
        # no readings at all before the next one.
        solar_df = pd.DataFrame(
            {"solar_kwh": [1.0, 2.0, 3.0]},
            index=pd.to_datetime(
                ["2026-06-01 10:00", "2026-06-01 10:30", "2026-06-01 12:00"]
            ),
        )

        result = predictor._load_and_process_solar_data(solar_df)

        assert result.loc["2026-06-01 10:00", "solar_kwh"] == 3.0
        assert pd.isna(result.loc["2026-06-01 11:00", "solar_kwh"])
        assert result.loc["2026-06-01 12:00", "solar_kwh"] == 3.0


class TestTrainUsesTimeSeriesSplit:
    def test_grid_search_receives_a_time_series_split(self, monkeypatch, tmp_path):
        from sklearn.model_selection import TimeSeriesSplit

        n = 60
        weather = hourly_frame(n, ["temperature", "irradiance", "windvelocity"])
        solar = pd.DataFrame(
            {"solar_kwh": np.random.default_rng(1).random(n)}, index=weather.index
        )

        predictor = make_predictor(
            tune_hyperparameters=True,
            param_grid={"n_estimators": [10], "max_depth": [2]},
        )
        monkeypatch.setattr(
            predictor, "_load_and_process_weather_data", lambda data: weather
        )
        monkeypatch.setattr(
            predictor, "_load_and_process_solar_data", lambda data: solar
        )

        captured = {}
        import sklearn.model_selection as model_selection

        def recording_grid_search_cv(**kwargs):
            captured["cv"] = kwargs.get("cv")
            return model_selection.GridSearchCV(**kwargs)

        monkeypatch.setattr(
            "dao.prog.solar_predictor.GridSearchCV", recording_grid_search_cv
        )

        stats = predictor.train(
            weather_data=weather,
            solar_data=solar,
            model_save_path=str(tmp_path / "model.json"),
            remove_outliers=False,
            tune_hyperparameters=True,
        )

        assert isinstance(captured["cv"], TimeSeriesSplit)
        assert (tmp_path / "model.json").exists()
        assert "best_params" in stats


class TestWarningsScope:
    def test_filterwarnings_is_not_set_at_import_time(self):
        """The old code called warnings.filterwarnings("ignore") when the
        module was imported, silencing warnings for the whole process. It
        must now only apply inside the GridSearchCV.fit() call."""
        import warnings

        import dao.prog.solar_predictor as module

        source = module.__file__
        with open(source) as f:
            text = f.read()

        # The blanket, module-level call must be gone...
        assert 'warnings.filterwarnings("ignore")\n\n\nclass' not in text
        # ...but the module must still scope it somewhere (catch_warnings).
        assert "catch_warnings" in text
