"""Tests for PV ML feature engineering and training-data selection.

Moved from dao/tests/prog/test_solar_predictor_training.py and
test_solar_predictor_persistence.py: the resampling/TimeSeriesSplit/
warnings-scope/persistence fixes those covered are independent of the
feature set and still apply unchanged, now alongside the new
build_features/training_weather coverage they were extended with.
"""

from __future__ import annotations

import datetime as dt
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from dao.forecast.pv.ml import FEATURES, build_features, training_weather
from dao.forecast.pv.physical import Plane, PVParams
from dao.lib.db_manager import DBmanagerObj
from dao.prog.solar_predictor import (
    OutdatedModelError,
    SolarPredictor,
    _metadata_path,
)

TZ = "Europe/Amsterdam"
LAT, LON = 52.1, 5.2


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


def test_build_features_has_all_columns_and_no_nan_outside_dni_dhi():
    times = pd.date_range("2026-06-21 06:00", periods=6, freq="h", tz=TZ)
    weather = pd.DataFrame(
        {
            "ghi": [100.0, 200.0, 300.0, 250.0, 150.0, 50.0],
            "dni": [float("nan")] * 6,
            "dhi": [float("nan")] * 6,
            "temp": [18.0] * 6,
            "wind": [2.0] * 6,
        },
        index=times,
    )
    params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=3.0)])

    features = build_features(weather, LAT, LON, params, 3600)

    assert list(features.columns) == list(FEATURES)
    assert features[["dni", "dhi"]].isna().all().all()
    other_columns = [c for c in FEATURES if c not in ("dni", "dhi")]
    assert not features[other_columns].isna().any().any()


def test_day_of_week_is_not_a_feature():
    assert "day_of_week" not in FEATURES


# ---------------------------------------------------------------------------
# training_weather
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    from sqlalchemy import (
        BigInteger,
        Column,
        Float,
        ForeignKey,
        Integer,
        String,
        Table,
        UniqueConstraint,
        insert,
    )

    from dao.lib.db_manager import forecasts_table

    manager = DBmanagerObj(
        db_dialect="sqlite", db_name="day_ahead.db", db_path=str(tmp_path)
    )
    metadata = manager.metadata
    variabel = Table(
        "variabel",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("code", String(10), unique=True, nullable=False),
        Column("name", String(50), unique=True, nullable=False),
        Column("dim", String(10), nullable=False),
        Column("aggregate", String(3), nullable=False, default="avg"),
    )
    Table(
        "values",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
        Column("time", BigInteger, nullable=False),
        Column("value", Float),
        UniqueConstraint("variabel", "time"),
    )
    forecasts_table(metadata)
    metadata.create_all(manager.engine)
    with manager.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {"id": 4, "code": "gr", "name": "Globale straling", "dim": "J/cm2"},
                {"id": 5, "code": "temp", "name": "Temperatuur", "dim": "C"},
                {"id": 23, "code": "winds", "name": "Windsnelheid", "dim": "m/s"},
                {"id": 28, "code": "dni", "name": "Directe straling", "dim": "J/cm2"},
                {"id": 29, "code": "dhi", "name": "Diffuse straling", "dim": "J/cm2"},
            ],
        )
    return manager


def put_forecast(db, code, lead_bucket, rows):
    from sqlalchemy import Table, insert, select

    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    forecasts = Table("forecasts", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        ident = connection.execute(
            select(variabel.c.id).where(variabel.c.code == code)
        ).scalar_one()
        connection.execute(
            insert(forecasts),
            [
                {
                    "variabel": ident,
                    "target_time": t,
                    "lead_bucket": lead_bucket,
                    "issued_time": t - lead_bucket * 3600,
                    "value": v,
                    "source": "meteoserver",
                }
                for t, v in rows
            ],
        )


def put_value(db, code, rows):
    from sqlalchemy import Table, insert, select

    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    values = Table("values", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        ident = connection.execute(
            select(variabel.c.id).where(variabel.c.code == code)
        ).scalar_one()
        connection.execute(
            insert(values),
            [{"variabel": ident, "time": t, "value": v} for t, v in rows],
        )


def test_training_weather_prefers_archive_when_long_enough(db):
    now = dt.datetime(2026, 6, 21, tzinfo=dt.UTC)
    start = now - dt.timedelta(days=100)
    n_hours = 100 * 24
    for code in ("gr", "dni", "dhi", "temp"):
        rows = [
            (int((start + dt.timedelta(hours=i)).timestamp()), 100.0)
            for i in range(n_hours)
        ]
        put_forecast(db, code, 12, rows)

    weather, source = training_weather(db, start, now, TZ)

    assert source == "archive"
    assert len(weather) > 0


def test_training_weather_falls_back_to_observations(db):
    now = dt.datetime(2026, 6, 21, tzinfo=dt.UTC)
    start = now - dt.timedelta(days=20)
    n_hours = 20 * 24
    for code in ("gr", "dni", "dhi", "temp"):
        rows = [
            (int((start + dt.timedelta(hours=i)).timestamp()), 100.0)
            for i in range(n_hours)
        ]
        put_forecast(db, code, 12, rows)
    for code in ("gr", "temp", "winds"):
        rows = [
            (int((start + dt.timedelta(hours=i)).timestamp()), 50.0)
            for i in range(n_hours)
        ]
        put_value(db, code, rows)

    weather, source = training_weather(db, start, now, TZ)

    assert source == "observations"
    assert len(weather) > 0


# ---------------------------------------------------------------------------
# SolarPredictor.train(): resampling, TimeSeriesSplit, recent-rows subset,
# metadata, warnings scope (moved from test_solar_predictor_training.py)
# ---------------------------------------------------------------------------


def make_predictor(tune_hyperparameters, param_grid=None, parameters=None):
    predictor = SolarPredictor.__new__(SolarPredictor)
    predictor.feature_columns = list(FEATURES)
    predictor.pv_params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=3.0)])
    predictor.latitude = LAT
    predictor.longitude = LON
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
    index = pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC")
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
        weather = hourly_frame(n, list(FEATURES))
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


def test_grid_search_uses_most_recent_rows(monkeypatch, tmp_path):
    n = 8000
    weather = hourly_frame(n, list(FEATURES))
    solar = pd.DataFrame(
        {"solar_kwh": np.random.default_rng(1).random(n)}, index=weather.index
    )

    predictor = make_predictor(
        tune_hyperparameters=True, param_grid={"n_estimators": [10], "max_depth": [2]}
    )
    monkeypatch.setattr(
        predictor, "_load_and_process_weather_data", lambda data: weather
    )
    monkeypatch.setattr(predictor, "_load_and_process_solar_data", lambda data: solar)

    from sklearn.model_selection import GridSearchCV

    captured = {}
    original_fit = GridSearchCV.fit

    def recording_fit(self, X, y=None, **kwargs):
        captured["X"] = X
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(GridSearchCV, "fit", recording_fit)

    predictor.train(
        weather_data=weather,
        solar_data=solar,
        model_save_path=str(tmp_path / "model.json"),
        remove_outliers=False,
        tune_hyperparameters=True,
    )

    X = captured["X"]
    # 8000 rows, test_size 0.2 default -> 6400 training rows; the subset is
    # the last 5000 of those, i.e. positions [1400:6400) of the full frame.
    assert len(X) == 5000
    assert X.index[0] == weather.index[1400]
    assert X.index[-1] == weather.index[6399]


def observation_weather(n_hours: int) -> pd.DataFrame:
    """Weather in exactly the shape the observation branch produces.

    ``update_observations`` writes gr, temp and winds and nothing else, so
    ``training_weather`` hands back a frame whose dni and dhi are NaN for
    every single row. This is the normal case for a fresh install: the
    archive branch only takes over after 90 days.
    """
    times = pd.date_range("2026-03-01", periods=n_hours, freq="h", tz=TZ)
    hours = np.asarray([moment.hour for moment in times], dtype=float)
    ghi = np.clip(700.0 * np.cos((hours - 13.0) / 7.0), 0.0, None)
    return pd.DataFrame(
        {
            "ghi": ghi,
            "dni": float("nan"),
            "dhi": float("nan"),
            "temp": 12.0 + 6.0 * np.sin((hours - 9.0) / 4.0),
            "wind": 3.0,
        },
        index=times,
    )


def test_training_on_observations_keeps_its_rows(tmp_path):
    """build_features leaves dni/dhi NaN on purpose -- XGBoost handles
    missing values natively -- but train() used to drop every row with any
    NaN in it, which on the observation branch is every row there is. The
    model then trained on nothing and the installation silently kept using
    the physical model forever."""
    n = 480
    weather = observation_weather(n)
    predictor = make_predictor(tune_hyperparameters=False)
    # What train_solar_option actually runs with: outlier removal on.
    predictor.create_physics_based_constraints(3.6)
    predictor.log_level = 20

    features = predictor.create_features(weather)
    assert features["dni"].isna().all()  # the premise this test exists for
    assert features["dhi"].isna().all()

    solar = pd.DataFrame(
        {"solar_kwh": features["physical"].to_numpy() * 0.95}, index=weather.index
    )

    stats = predictor.train(
        weather_data=weather,
        solar_data=solar,
        model_save_path=str(tmp_path / "model.json"),
        tune_hyperparameters=False,
    )

    assert stats["training_samples"] > 0.5 * n
    assert (tmp_path / "model.json").exists()


def test_training_still_drops_rows_without_irradiance(tmp_path):
    """Optional means dni and dhi only. A row with no ghi, no temperature or
    no physical prediction has nothing to teach the model and must still
    go."""
    n = 480
    weather = observation_weather(n)
    weather.iloc[:120, weather.columns.get_loc("ghi")] = float("nan")
    predictor = make_predictor(tune_hyperparameters=False)

    features = predictor.create_features(weather)
    solar = pd.DataFrame(
        {"solar_kwh": np.nan_to_num(features["physical"].to_numpy())},
        index=weather.index,
    )

    stats = predictor.train(
        weather_data=weather,
        solar_data=solar,
        model_save_path=str(tmp_path / "model.json"),
        remove_outliers=False,
        tune_hyperparameters=False,
    )

    assert stats["training_samples"] + stats["testing_samples"] == n - 120


def test_model_meta_records_training_source(tmp_path):
    n = 60
    weather = hourly_frame(n, list(FEATURES))
    solar = pd.DataFrame(
        {"solar_kwh": np.random.default_rng(1).random(n)}, index=weather.index
    )
    predictor = make_predictor(tune_hyperparameters=False)
    model_path = str(tmp_path / "model.json")

    predictor.train(
        weather_data=weather,
        solar_data=solar,
        model_save_path=model_path,
        remove_outliers=False,
        tune_hyperparameters=False,
        training_source="archive",
    )

    meta = json.loads(Path(_metadata_path(model_path)).read_text())
    assert meta["training_weather"] == "archive"
    assert meta["feature_columns"] == list(FEATURES)


class TestWarningsScope:
    def test_filterwarnings_is_not_set_at_import_time(self):
        """The old code called warnings.filterwarnings("ignore") when the
        module was imported, silencing warnings for the whole process. It
        must now only apply inside the GridSearchCV.fit() call."""
        import dao.prog.solar_predictor as module

        source = module.__file__
        with open(source) as f:
            text = f.read()

        # The blanket, module-level call must be gone...
        assert 'warnings.filterwarnings("ignore")\n\n\nclass' not in text
        # ...but the module must still scope it somewhere (catch_warnings).
        assert "catch_warnings" in text


# ---------------------------------------------------------------------------
# Model persistence (moved from test_solar_predictor_persistence.py)
# ---------------------------------------------------------------------------


def make_bare_predictor(feature_columns):
    predictor = SolarPredictor.__new__(SolarPredictor)
    predictor.feature_columns = feature_columns
    return predictor


def fit_and_save(tmp_path, feature_columns, filename="model.json"):
    rng = np.random.default_rng(0)
    X = rng.random((20, len(feature_columns)))
    y = rng.random(20)
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X, y)

    predictor = make_bare_predictor(feature_columns)
    predictor.model = model
    model_path = str(tmp_path / filename)

    # Mirrors the tail end of train(): save_model + sidecar metadata.
    model.save_model(model_path)
    with open(_metadata_path(model_path), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_columns}, f)

    return predictor, model_path


def test_load_model_restores_a_working_predictor(tmp_path):
    columns = list(FEATURES)
    original, model_path = fit_and_save(tmp_path, columns)

    loaded = make_bare_predictor(columns)
    loaded.load_model(model_path)

    assert loaded.is_trained is True
    X = np.random.default_rng(1).random((3, len(columns)))
    np.testing.assert_allclose(loaded.model.predict(X), original.model.predict(X))


def test_load_model_rejects_a_feature_column_mismatch(tmp_path):
    trained_with = list(FEATURES)
    _, model_path = fit_and_save(tmp_path, trained_with)

    # Current code expects a different (e.g. extended) feature set.
    loader = make_bare_predictor(trained_with + ["extra"])

    # A dedicated type, not a bare ValueError: an outdated model after an
    # upgrade is a known state with a known remedy, and the service has to
    # tell it apart from a model that is genuinely broken.
    with pytest.raises(OutdatedModelError, match="feature columns"):
        loader.load_model(model_path)
    assert issubclass(OutdatedModelError, ValueError)


def test_load_model_without_a_sidecar_still_loads(tmp_path):
    """Backward-compat: a model saved before this change (no .meta.json)
    must still load rather than crash on a missing sidecar file."""
    columns = list(FEATURES)
    rng = np.random.default_rng(2)
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(rng.random((20, len(columns))), rng.random(20))
    model_path = str(tmp_path / "no_sidecar.json")
    model.save_model(model_path)

    loader = make_bare_predictor(columns)
    loader.load_model(model_path)

    assert loader.is_trained is True


def test_load_model_raises_file_not_found_for_a_missing_path(tmp_path):
    loader = make_bare_predictor(["ghi"])
    with pytest.raises(FileNotFoundError):
        loader.load_model(str(tmp_path / "missing.json"))


def test_build_features_leaks_no_numpy_warnings_into_the_log():
    """Same clear-sky call as simulate(), same divide by cos(zenith) at
    night, same raw RuntimeWarning in the operator's log."""
    weather = observation_weather(24)
    params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=3.0)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features = build_features(weather, LAT, LON, params)

    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime == [], [str(w.message) for w in runtime]
    assert not features["clearsky_ghi"].isna().any()


def test_outlier_detection_leaks_no_numpy_warnings_on_a_flat_bucket():
    """A correlation needs both series to vary. The guard checked only the
    irradiance, so a season-hour bucket where production never changes --
    a shaded hour, a capped inverter, a string that was off -- divided by a
    zero standard deviation and printed "invalid value encountered in
    divide" into the operator's log during every training run."""
    predictor = make_predictor(tune_hyperparameters=False)
    predictor.create_physics_based_constraints(3.6)
    predictor.log_level = 20

    index = pd.date_range("2026-03-01", periods=30 * 24, freq="h", tz=TZ)
    hours = np.asarray([moment.hour for moment in index], dtype=float)
    merged = pd.DataFrame(
        {
            "hour": hours,
            "ghi": np.clip(700.0 * np.cos((hours - 13.0) / 7.0), 0.0, None)
            + np.resize([0.0, 5.0], len(index)),
            "solar_kwh": 0.4,  # never varies: the correlation is undefined
        },
        index=index,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleaned = predictor._detect_outliers(merged)

    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime == [], [str(w.message) for w in runtime]
    # Nothing to flag either: an undefined correlation is not a reason to
    # throw rows away.
    assert len(cleaned) == len(merged)
