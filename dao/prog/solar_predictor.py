"""
Solar Production Prediction Module

This module provides functionality to train XGBoost models for predicting
hourly solar production based on weather data and historical solar output.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
from typing import Optional, Union, Dict, Any
import datetime as dt
import logging
import math

# ML imports
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy import stats
from dao.prog.da_base import DaBase
from dao.prog.config.models.devices.solar import SolarConfig
# import pvlib


def _metadata_path(model_path: str) -> str:
    """Sidecar file next to a saved model holding the feature list it was
    trained with, so a mismatch with the current code shows up as a clear
    error rather than a silently misaligned prediction."""
    return model_path + ".meta.json"


class SolarPredictor(DaBase):
    """
    A comprehensive solar production prediction system using XGBoost.

    This class handles data preprocessing, outlier detection, feature engineering,
    model training, and prediction for hourly solar production forecasting.
    """

    def __init__(
        self,
        solar_name: str = "",
        solar_capacity: float = 5,
        random_state: int = 42,
        # max_hourly_production: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize the SolarPredictor.

        Args:
            random_state: Random state for reproducible results
            max_hourly_production: Optional dictionary mapping hour (0-23) to maximum
                                 expected kWh production for physics-based outlier detection.
                                 If None, uses physics-based constraints for a typical 5kW system at 52°N.
                                 For better accuracy, use create_physics_based_constraints()
                                 with your actual system capacity and location.
        """
        from dao.forecast.pv.ml import FEATURES
        from dao.forecast.pv.physical import PVParams

        super().__init__()
        if self.config is None:
            return
        self.solar_name = solar_name
        self.solar_capacity = solar_capacity
        self.latitude = self.ha_context.latitude
        self.longitude = self.ha_context.longitude
        self.tilt = 45
        self.azimut = 180
        self.random_state = random_state
        self.model = None
        self.feature_columns = list(FEATURES)
        # Overwritten per installation in train_solar_option/predict_solar_device
        # with the current (calibrated when available) physical parameters;
        # this default only matters before either has run once.
        self.pv_params = PVParams(planes=[])
        self.is_trained = False
        self.training_stats = {}
        self.ml_training_start_date = dt.date(2000, 1, 1)
        self.solar_entities = []
        """
        # Set default physics-based constraints for typical residential system
        # Uses 5kW system at 45°N latitude (mid-latitude) as reasonable default
        self.max_hourly_production =(
            self.create_physics_based_constraints(
                solar_capacity,
                system_efficiency=0.8,
                conservative_factor=1.2
            )
        )
        """

    def create_features(self, weather: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering for a weather frame in pv layout (ghi/dni/dhi/
        temp/wind, tz-aware DatetimeIndex): solar geometry, clear-sky ghi,
        calendar features and the physical model's own prediction. See
        dao.forecast.pv.ml.build_features for the full column list (FEATURES)
        and the reasoning behind including the physical model's output.
        """
        from dao.forecast.pv.ml import build_features

        interval_s = getattr(self, "interval_s", 3600)
        return build_features(
            weather, self.latitude, self.longitude, self.pv_params, interval_s
        )

    def create_physics_based_constraints(
        self,
        system_capacity_kw: float,
        system_efficiency: float = 0.8,
        conservative_factor: float = 1.2,
    ) -> Dict[int, float]:
        """
        Create physics-based maximum hourly production constraints based on solar system capacity.

        This calculates theoretical maximum production for each hour based on:
        - System peak capacity (kWp)
        - Solar elevation angles throughout the day
        - System efficiency (inverter losses, temperature derating, etc.)
        - Conservative safety factor for outlier detection

        Args:
            system_capacity_kw: Solar system peak capacity in kW (e.g., 6.5 for 6.5kWp system)
            self.latitude: Installation latitude in degrees (affects sun angles)
                     Examples: 52.0 (Netherlands), 40.7 (New York), 34.0 (Los Angeles)
            self.longitude : deviation from meridian
            system_efficiency: Overall system efficiency (0.0-1.0)
                              Typical: 0.75-0.85 (accounts for inverter losses, temperature, dust, etc.)
            conservative_factor: Safety multiplier for outlier detection (>1.0)
                               Higher = more lenient outlier detection
                               1.2 = 20% above theoretical maximum

        Returns:
            Dictionary mapping hour (0-23) to maximum expected production (kWh)

        Example:
            # For a 8kWp system in Netherlands (52°N latitude)
            constraints = SolarPredictor.create_physics_based_constraints(
                system_capacity_kw=8.0,
                system_efficiency=0.8,
                conservative_factor=1.2
            )
            predictor = SolarPredictor(max_hourly_production=constraints)
        """

        # Solar declination angle (simplified for summer solstice - maximum sun elevation)
        # This gives us the most conservative (highest possible) estimates
        declination = 23.45  # degrees, summer solstice

        constraints = {}

        for hour in range(24):
            # Uren zijn in UTC. Ten oosten van Greenwich valt de zonnemiddag
            # *vroeger* dan 12:00 UTC, dus solar_noon_utc = 12 - lon/15 en de
            # uurhoek wordt (hour - solar_noon_utc) * 15. Voor Nederland
            # (~5,3 gr. oost) zat dit er 42 minuten naast, en de verkeerde kant op.
            solar_noon_utc = 12.0 - self.longitude / 15.0
            hour_angle = 15.0 * (hour - solar_noon_utc)

            # Calculate solar elevation angle
            lat_rad = math.radians(self.latitude)
            dec_rad = math.radians(declination)
            hour_rad = math.radians(hour_angle)

            elevation_rad = math.asin(
                math.sin(lat_rad) * math.sin(dec_rad)
                + math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_rad)
            )
            elevation_deg = math.degrees(elevation_rad)

            if elevation_deg <= 0:
                # Sun is below horizon
                max_production = 0.05  # Small threshold for measurement noise
            else:
                # Calculate relative irradiance based on sun elevation
                # At 90° (zenith), max irradiance ≈ 1000 W/m²
                # Use sine function to approximate atmospheric losses at low angles
                relative_irradiance = math.sin(elevation_rad)

                # Apply additional atmospheric losses for low sun angles
                if elevation_deg < 15:
                    # Significant atmospheric losses near horizon
                    relative_irradiance *= (elevation_deg / 15.0) ** 0.5

                # Calculate theoretical maximum production for this hour
                # Max kWh = System_kW × Relative_Irradiance × Efficiency × 1_hour
                theoretical_max = (
                    system_capacity_kw * relative_irradiance * system_efficiency
                )

                # Apply conservative factor for outlier detection
                max_production = theoretical_max * conservative_factor

            constraints[hour] = max(0.05, max_production)  # Minimum threshold for noise
        self.max_hourly_production = constraints
        return

    def _load_and_process_weather_data(
        self, weather_data: Union[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Load and process weather data.

        Args:
            weather_data: Path to CSV file or pandas DataFrame

        Returns:
            Processed weather DataFrame with features
        """
        if isinstance(weather_data, str):
            if not os.path.exists(weather_data):
                raise FileNotFoundError(f"Weather data file not found: {weather_data}")
            weather_df = pd.read_csv(weather_data)
        else:
            weather_df = weather_data.copy()

        # Ensure datetime column exists
        if "datetime" in weather_df.columns:
            weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
            weather_df = weather_df.set_index("datetime")
        elif not isinstance(weather_df.index, pd.DatetimeIndex):
            raise ValueError(
                "Weather data must have a 'datetime' column or DatetimeIndex"
            )

        # Validate required columns
        # required_cols = ['temperature', 'irradiance', 'windvelocity']
        # missing_cols = [col for col in required_cols if col not in weather_df.columns]
        # if missing_cols:
        #     raise ValueError(f"Weather data missing required columns: {missing_cols}")

        # Create features
        return self.create_features(weather_df)

    def _load_and_process_solar_data(
        self, solar_data: Union[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Load and process solar production data.

        Args:
            solar_data: Path to CSV file or pandas DataFrame

        Returns:
            Processed solar DataFrame
        """
        if isinstance(solar_data, str):
            if not os.path.exists(solar_data):
                raise FileNotFoundError(f"Solar data file not found: {solar_data}")
            solar_df = pd.read_csv(solar_data)
        else:
            solar_df = solar_data.copy()

        # Process datetime index
        if "datetime" in solar_df.columns:
            solar_df["datetime"] = pd.to_datetime(solar_df["datetime"])
            solar_df = solar_df.set_index("datetime")
        elif not isinstance(solar_df.index, pd.DatetimeIndex):
            raise ValueError(
                "Solar data must have a 'datetime' column or DatetimeIndex"
            )

        # Ensure solar_kwh column exists
        if "solar_kwh" not in solar_df.columns:
            raise ValueError("Solar data must contain 'solar_kwh' column")

        # Remove negative values
        solar_df = solar_df[solar_df["solar_kwh"] >= 0]

        # Resample to hourly if needed. min_count=1 (not the default 0): an
        # hour with no underlying samples at all (a recorder or inverter
        # outage) must resample to NaN, not to a fabricated 0. The two
        # dropna() calls a few lines down in train() then correctly exclude
        # it, instead of teaching the model "no production" for an hour
        # where nothing was actually measured. The default made every
        # outage during daylight look like a hard physical zero.
        if len(solar_df) > 0:
            solar_df = solar_df[["solar_kwh"]].resample("h").sum(min_count=1)

        return solar_df

    def _detect_outliers(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive outlier detection for solar production data.

        Uses a three-method approach to identify and remove outliers:

        1. **Statistical outliers**: Z-score > 3 (values more than 3 standard deviations from mean)
        2. **IQR outliers**: Values outside Q1 - 1.5*IQR or Q3 + 1.5*IQR range
        3. **Physics-based outliers**: Values exceeding theoretical maximum production by hour

        A data point is flagged as an outlier only if detected by 2+ methods,
        reducing false positives while catching genuine anomalies.

        Additionally applies seasonal context outlier detection based on the
        correlation between solar production and irradiance within season-hour groups.

        Args:
            merged_data: Merged weather and solar data

        Returns:
            Clean data with outliers removed
        """
        logging.info("Detecting outliers...")
        original_size = len(merged_data)

        # 1. Context-aware outlier detection by hour
        outlier_mask = pd.Series(False, index=merged_data.index)

        for hour in range(24):
            hour_data = merged_data[merged_data["hour"] == hour]
            if len(hour_data) < 10:
                continue

            solar_values = hour_data["solar_kwh"]

            # Statistical outliers (Z-score > 3)
            z_scores = np.abs(stats.zscore(solar_values))
            statistical_outliers = z_scores > 3

            # IQR method
            Q1 = solar_values.quantile(0.25)
            Q3 = solar_values.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = (solar_values < (Q1 - 1.5 * IQR)) | (
                solar_values > (Q3 + 1.5 * IQR)
            )

            # Physics-based constraints: Maximum reasonable solar production by hour
            # These values represent theoretical upper bounds for a typical residential
            # solar installation (4-6kW system) under ideal conditions.
            #
            # Logic:
            # - Night hours (20-05): Virtually no production (0.1 kWh max for measurement noise)
            # - Dawn/Dusk (6, 19): Low production as sun is at low angles
            # - Morning ramp (7-9): Increasing production as sun rises
            # - Peak hours (10-14): Maximum production when sun is highest
            # - Afternoon decline (15-18): Decreasing as sun sets
            #
            # Note: These are conservative estimates and may need adjustment for:
            # - Larger installations (scale proportionally)
            # - Different latitudes (seasonal variation)
            # - Local climate conditions

            # Use configurable physics-based constraints
            physics_outliers = solar_values > self.max_hourly_production.get(hour, 5.5)

            # Combine methods (outlier if flagged by 2+ methods)
            combined_outliers = (
                statistical_outliers.astype(int)
                + iqr_outliers.astype(int)
                + physics_outliers.astype(int)
            ) >= 2

            outlier_mask.loc[hour_data.index] = combined_outliers

        # 2. Seasonal context outlier detection. "season" is not a stored
        # feature since build_features() (day_of_week is deliberately gone
        # too); derived here from the calendar month instead, local to this
        # grouping rather than carried through the whole pipeline.
        seasonal_outlier_mask = pd.Series(False, index=merged_data.index)
        clean_data = merged_data[~outlier_mask]
        season_of = clean_data.index.month.map(
            {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        )

        for season in season_of.unique():
            for hour in range(6, 20):  # Daylight hours only
                mask = (season_of == season) & (clean_data["hour"] == hour)
                season_hour_data = clean_data[mask]

                if len(season_hour_data) < 20:
                    continue

                if season_hour_data["ghi"].std() > 0:
                    irradiance_corr = season_hour_data["solar_kwh"].corr(
                        season_hour_data["ghi"]
                    )

                    if irradiance_corr > 0.5:
                        # Use direct irradiance vs solar production ratio for outlier detection
                        irradiance_ratio = season_hour_data["solar_kwh"] / (
                            season_hour_data["ghi"] + 1e-6
                        )
                        Q1 = irradiance_ratio.quantile(0.25)
                        Q3 = irradiance_ratio.quantile(0.75)
                        IQR = Q3 - Q1
                        ratio_outliers = (irradiance_ratio < (Q1 - 2.0 * IQR)) | (
                            irradiance_ratio > (Q3 + 2.0 * IQR)
                        )
                        seasonal_outlier_mask.loc[season_hour_data.index] = (
                            ratio_outliers
                        )

        # Apply outlier removal
        final_clean_data = clean_data[~seasonal_outlier_mask]

        outliers_removed = original_size - len(final_clean_data)
        if outliers_removed > 0:
            logging.info(
                f"Outliers removed: {outliers_removed} "
                f"({outliers_removed / original_size * 100:.1f}%)"
            )
            if self.log_level >= logging.DEBUG:
                outliers = merged_data[~merged_data.isin(final_clean_data).all(axis=1)]
                logging.debug(f"Detected outliers:\n{outliers.to_string()}")
        return final_clean_data

    def train(
        self,
        weather_data: Union[str, pd.DataFrame],
        solar_data: Union[str, pd.DataFrame],
        model_save_path: str,
        test_size: float = 0.2,
        remove_outliers: bool = True,
        tune_hyperparameters: bool = True,
        training_source: str = "observations",
    ) -> Dict[str, Any]:
        """
        Train the solar prediction model.

        Args:
            weather_data: Path to weather CSV or DataFrame with columns:
                         ['datetime', 'ghi', 'dni', 'dhi', 'temp', 'wind'] (pv layout)
            solar_data: Path to solar CSV or DataFrame with columns:
                       ['datetime', 'solar_kwh']
            model_save_path: Path where to save the trained model
            test_size: Fraction of data to use for testing
            remove_outliers: Whether to apply outlier detection
            tune_hyperparameters: Whether to perform hyperparameter tuning
            training_source: "archive" or "observations" -- which weather
                source training_weather() picked; recorded in the model's
                metadata sidecar so a later mismatch is visible, not guessed at.

        Returns:
            Dictionary with training statistics
        """
        logging.info(
            f"Starting solar prediction model for {self.solar_name} training..."
        )

        # Load and process data
        logging.info("Loading and processing data...")
        weather_features = self._load_and_process_weather_data(weather_data)
        solar_df = self._load_and_process_solar_data(solar_data)

        # Merge datasets
        logging.info("Merging weather and solar data...")
        # Align timezone information
        if weather_features.index.tz is None:
            weather_features.index = weather_features.index.tz_localize(
                "UTC", ambiguous="NaT"
            )
        if solar_df.index.tz is None:
            solar_df.index = solar_df.index.tz_localize("UTC", ambiguous="NaT")

        weather_features = weather_features.dropna()
        solar_df = solar_df.dropna()

        merged_data = weather_features.join(solar_df, how="inner")
        merged_data = merged_data.dropna()

        # drop when irradiance=0 and solar_kwh=0
        # merged_data.query('irradiance > 0 or solar_kwh > 0', inplace=True)

        logging.info(f"Merged dataset: {len(merged_data)} records")
        logging.info(
            f"Date range: {merged_data.index.min()} to {merged_data.index.max()}"
        )
        logging.debug(f"Merged dataset all records:\n {merged_data.to_string()}")

        # Outlier detection
        if remove_outliers:
            merged_data = self._detect_outliers(merged_data)
            logging.info(f"Clean dataset: {len(merged_data)} records")

        # Prepare features and target
        X = merged_data[self.feature_columns].copy()
        y = merged_data["solar_kwh"].copy()

        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]

        # Chronological split (important for time series)
        split_idx = int((1 - test_size) * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logging.info(f"Training samples: {len(X_train)}")
        logging.info(f"Testing samples: {len(X_test)}")

        _xgboost_cfg = self.config.xgboost
        tune_hyperparameters = _xgboost_cfg.tune_hyperparameters
        logging.info(f"Tune hyperparameters: {tune_hyperparameters}")
        # Model training
        if tune_hyperparameters:
            logging.info("Tuning hyperparameters...")
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 6],
                "learning_rate": [0.05, 0.1, 0.15],
                "subsample": [0.8, 0.9],
            }
            param_grid = _xgboost_cfg.param_grid or param_grid
            logging.info(f"Parameter grid: {param_grid}")

            # Use a subset for faster grid search: the most recent rows,
            # not the oldest -- a household's production characteristics
            # (panel soiling, a tree that grew in, an added string) drift
            # over a multi-year training window, so hyperparameters tuned
            # on what is closest to what the model will actually predict
            # generalise better than ones tuned on the oldest data.
            subset_size = min(5000, len(X_train))
            X_train_subset = X_train.iloc[-subset_size:]
            y_train_subset = y_train.iloc[-subset_size:]

            # TimeSeriesSplit, not a plain cv=3 (KFold): with an integer cv,
            # sklearn folds are contiguous but not time-ordered relative to
            # each other, so a model can end up trained on later data and
            # validated on earlier data. For weather/production series with
            # strong day-to-day and seasonal autocorrelation that leaks
            # future information into the score. TimeSeriesSplit only ever
            # validates on data that comes after its training fold.
            grid_search = GridSearchCV(
                estimator=XGBRegressor(
                    random_state=self.random_state, objective="reg:squarederror"
                ),
                param_grid=param_grid,
                cv=TimeSeriesSplit(n_splits=3),
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )

            # GridSearchCV fits hundreds of models here and sklearn/xgboost
            # are chatty about things like convergence and near-constant
            # folds; that used to be silenced process-wide via a
            # module-level warnings.filterwarnings("ignore"), which also
            # hid unrelated warnings everywhere else this module gets
            # imported (the web server, the scheduler). Scope it to just
            # this call instead.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                grid_search.fit(X_train_subset, y_train_subset)
            best_params = grid_search.best_params_
            logging.info(f"Best parameters: {best_params}")
        else:
            # Use default parameters
            best_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
            }
            best_params = _xgboost_cfg.parameters or best_params

        # Train final model
        logging.info("Training final model...")
        logging.info(f"Parameters: {best_params}")
        self.model = XGBRegressor(
            **best_params, random_state=self.random_state, objective="reg:squarederror"
        )
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Store training statistics
        self.training_stats = {
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "feature_importance": dict(
                zip(self.feature_columns, self.model.feature_importances_)
            ),
            "mean_target": y.mean(),
            "std_target": y.std(),
            "best_params": best_params if tune_hyperparameters else "default",
        }

        # Save model. xgboost's own save_model()/load_model() (a "model
        # file", not a pickle of the Python wrapper) survives an xgboost
        # version bump; joblib.dump() pickles the whole estimator object,
        # including private attributes whose layout has broken across major
        # xgboost releases before. The feature list travels alongside it in
        # a small sidecar so a mismatch between the model and the current
        # code's feature_columns is caught with a clear error instead of a
        # silently misaligned prediction.
        os.makedirs(
            os.path.dirname(model_save_path)
            if os.path.dirname(model_save_path)
            else ".",
            exist_ok=True,
        )
        self.model.save_model(model_save_path)
        with open(_metadata_path(model_save_path), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_columns": self.feature_columns,
                    "trained_at": dt.datetime.now().isoformat(),
                    "xgboost_version": xgboost.__version__,
                    "training_weather": training_source,
                },
                f,
                indent=2,
            )
        self.is_trained = True

        logging.info(f"Model training van {self.solar_name} complete")
        logging.info(f"Model saved to: {model_save_path}")
        logging.info(f"Training MAE: {train_mae:.4f} kWh")
        logging.info(f"Testing MAE: {test_mae:.4f} kWh")
        logging.info(f"Training R²: {train_r2:.4f}")
        logging.info(f"Testing R²: {test_r2:.4f}")
        logging.info("Sorted features:")
        importance = self.training_stats["feature_importance"]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_features):
            logging.info(f"  {i + 1}. {feature}: {score:.3f}")
        return self.training_stats

    def predict(
        self, weather_data: Union[Dict[str, float], pd.DataFrame]
    ) -> Union[float, np.ndarray]:
        """
        Make predictions using the trained model.

        Args:
            weather_data: Either a dictionary with single prediction data or DataFrame with multiple predictions,
                        both in pv layout (ghi, dni, dhi, temp, wind) plus a 'datetime' key/column.

        Returns:
            Predicted solar production in kWh
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "Model must be trained before making predictions. Call train() first."
            )

        if isinstance(weather_data, dict):
            # Single prediction - convert to DataFrame and process
            if "datetime" not in weather_data:
                raise ValueError("Single prediction requires 'datetime' key")

            # Create single-row DataFrame
            single_df = pd.DataFrame([weather_data])
            single_df["datetime"] = pd.to_datetime(single_df["datetime"])
            single_df = single_df.set_index("datetime")

            # Process through feature engineering
            processed_df = self.create_features(single_df)

            # Extract features and make prediction
            features = processed_df[self.feature_columns].iloc[0:1]
            prediction = self.model.predict(features)[0]
            return max(0, prediction)  # Ensure non-negative

        else:
            # Multiple predictions
            if not isinstance(weather_data, pd.DataFrame):
                raise ValueError("ned_nl_data must be a dictionary or pandas DataFrame")

            # Process weather data using the standard method
            weather_data = self._load_and_process_weather_data(weather_data)

            # Select required features
            featured_df = weather_data[self.feature_columns]
            if len(featured_df) == 0:
                prediction = []
            else:
                prediction = self.model.predict(featured_df)
            prediction = np.maximum(0, prediction)  # Ensure non-negative
            result = pd.DataFrame(
                {"date_time": featured_df.index, "prediction": prediction}
            )
            result["date_time"] = result["date_time"].dt.tz_convert(self.time_zone)
            return result

    def load_model(self, model_path: str):  # -> 'SolarPredictor':
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model file

        Returns:
            None # SolarPredictor instance with loaded model

        Raises:
            FileNotFoundError: no model at model_path.
            ValueError: the model's sidecar metadata lists a different set
                of feature columns than the current code uses. Loading it
                anyway would silently feed the wrong values into the wrong
                slots instead of failing.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        metadata_path = _metadata_path(model_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            trained_columns = metadata.get("feature_columns")
            if trained_columns is not None and trained_columns != self.feature_columns:
                raise ValueError(
                    f"Model {model_path} was trained with feature columns "
                    f"{trained_columns}, but the current code uses "
                    f"{self.feature_columns}. Retrain before using this model."
                )

        model = XGBRegressor()
        model.load_model(model_path)
        self.model = model
        self.is_trained = True

        return None

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before getting feature importance")

        return dict(zip(self.feature_columns, self.model.feature_importances_))

    def import_weatherdata(self, filename: str):
        """
        get the weatherdata from a local file.
        You get this file from KNMI site: https://www.daggegevens.knmi.nl/klimatologie/uurgegevens
        Place the file in addon_config/prediction/meteo
        When you call "train models", the files in thismap are imported and deleted

        :param filename: the path and the filename
        :return:
        """
        count = 0
        with open(filename) as file:
            while line := file.readline():
                if "# STN," in line:
                    break
                count += 1
        df = pd.read_csv(filename, skiprows=count)
        """
        STN,YYYYMMDD,HH,   FH,    T,    Q 
        275,20220101,    1,   50,  119,    0
        275,20220101,    2,   50,  117,    0
        
        YYYYMMDD,HH:dag einde uur in utc -> HH-1 begin uur utc
        T         : Temperatuur (in 0.1 graden Celsius) -> temp /10
        Q         : Globale straling (in J/cm2) -> gr -
        """
        df = df.rename(columns={"    T": "temp", "    Q": "gr", "   FH": "winds"})
        # Three rows appended per source row with .loc[shape[0]] is O(n^2)
        # (a full copy on every append); a plain list of tuples handed to
        # pd.DataFrame(...) once is O(n). Three years of hourly KNMI data is
        # ~26000 source rows here, so ~78000 appends avoided.
        records = []
        for row in df.itertuples():
            year = int(str(row.YYYYMMDD)[0:4])
            month = int(str(row.YYYYMMDD)[4:6])
            day = int(str(row.YYYYMMDD)[6:8])
            hour = row.HH - 1
            dati = dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc)
            utc = int(dati.timestamp())
            records.append((utc, "temp", row.temp / 10))
            records.append((utc, "gr", row.gr))
            records.append((utc, "winds", row.winds / 10))
        save_df = pd.DataFrame(records, columns=["time", "code", "value"])
        self.db_da.savedata(save_df, tablename="values")
        os.remove(filename)
        return

    def get_solar_data(self, start: dt.datetime, entities: list) -> pd.DataFrame:
        """
        haalt gemeten pv-productie op via de centrale HistoryReader, met
        dezelfde uurresolutie en kolommen (datetime, solar_kwh) als voorheen
        :param start: begindatum
        :param entities: list van sensoren van ha
        :return:
        """
        from zoneinfo import ZoneInfo

        from dao.forecast.history import HistoryReader

        tz = ZoneInfo(self.time_zone)
        now = dt.datetime.now(tz=tz)
        start_aware = start if start.tzinfo is not None else start.replace(tzinfo=tz)
        cap = 1.2 * self.solar_capacity if self.solar_capacity else None

        reader = HistoryReader(self.db_ha, self.time_zone)
        series = reader.read_energy(list(entities), start_aware, now, cap_kwh=cap)
        frame = series.to_frame(name="solar_kwh")
        frame.index.name = "datetime"
        return frame.reset_index()

    def train_solar_option(self, solar_option: SolarConfig, start: dt.datetime):
        """Fetch training weather and measured production for one
        installation, and train its model.

        The weather source (archived forecasts once there is enough of
        them, measured observations otherwise) is picked by
        training_weather() and recorded in the model's metadata sidecar.
        """
        from zoneinfo import ZoneInfo

        from dao.forecast.pv.ml import training_weather

        self.solar_name = solar_option.name.replace(" ", "_").replace("-", "_")
        self.tilt = solar_option.effective_tilt
        self.azimut = solar_option.effective_orientation + 180
        self.solar_capacity = solar_option.total_capacity
        self.solar_entities = solar_option.entities_sensors
        start_date = solar_option.ml_training_start_date
        start_dt = dt.datetime(start_date.year, start_date.month, start_date.day)
        start = max(start, start_dt)
        if not self.solar_entities:
            raise ValueError(
                f"No entities configured in your solar-option of {self.solar_name}"
            )
        self.create_physics_based_constraints(self.solar_capacity)
        self.pv_params, _source = self.pv_service().params_for(solar_option)

        tz = ZoneInfo(self.time_zone)
        start_aware = start if start.tzinfo is not None else start.replace(tzinfo=tz)
        now = dt.datetime.now(tz=tz)
        weather_data, training_source = training_weather(
            self.db_da, start_aware, now, self.time_zone
        )

        solar_data = self.get_solar_data(start=start, entities=self.solar_entities)
        self.train(
            weather_data,
            solar_data,
            "../data/prediction/models/" + self.solar_name + ".json",
            tune_hyperparameters=True,
            training_source=training_source,
        )

    def run_train(self, start: dt.datetime = None):
        """
        traint alle gedefinieerde ml-objecten
        :param start: optionele begindatum om te trainen, anders drie jaar geleden
        :return:
        """
        if start is None:
            now = dt.datetime.now()
            start = dt.datetime(year=now.year - 3, month=now.month, day=now.day)
        solar_options = self.config.solar
        for solar_option in solar_options:
            if solar_option.model in ("ml", "auto") and solar_option.entities_sensors:
                self.train_solar_option(solar_option, start)
        batteries = self.config.battery
        for battery in batteries:
            for solar_option in battery.solar:
                if (
                    solar_option.model in ("ml", "auto")
                    and solar_option.entities_sensors
                ):
                    self.train_solar_option(solar_option, start)

    def predict_solar_device(
        self, solar_option: SolarConfig, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        """
        berekent de voorspelling voor een pv-installatie
        :param solar_option: de configuratie van de installatie
        :param start: start-tijdstip voorspelling
        :param end: eind-tijdstip voorspelling
        :return: dataframe met berekende voorspellingen per uur
        """
        from zoneinfo import ZoneInfo

        from dao.forecast.pv.physical import weather_for_pv

        self.solar_name = solar_option.name.replace(" ", "_").replace("-", "_")
        self.tilt = solar_option.effective_tilt
        self.azimut = solar_option.effective_orientation + 180
        self.solar_capacity = solar_option.total_capacity
        file_name = "../data/prediction/models/" + self.solar_name + ".json"
        if os.path.isfile(file_name):
            self.load_model(file_name)
        else:
            raise FileNotFoundError(
                f"Er is geen model aanwezig voor {self.solar_name},svp eerst trainen."
            )
        self.pv_params, _source = self.pv_service().params_for(solar_option)

        tz = ZoneInfo(self.time_zone)
        start_aware = start if start.tzinfo is not None else start.replace(tzinfo=tz)
        end_aware = end if end.tzinfo is not None else end.replace(tzinfo=tz)
        prog = self.db_da.get_prognose_fields(
            ["gr", "dni", "dhi", "temp", "winds"],
            int(start_aware.timestamp()),
            int(end_aware.timestamp()),
            self.interval,
        )
        weather_data = weather_for_pv(prog, self.time_zone)
        prediction = self.predict(weather_data)
        prediction["prediction"] = prediction["prediction"].clip(lower=0).round(3)
        logging.info(f"ML prediction {self.solar_name}\n{prediction}")
        return prediction


def main():
    arg = sys.argv[1]
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        start_dt = dt.datetime.strptime(arg2, "%Y-%m-%d")
    else:
        start_dt = None
    solar_predictor = SolarPredictor("")
    if arg.lower() == "train":
        solar_predictor.run_train(start=start_dt)


if __name__ == "__main__":
    main()
