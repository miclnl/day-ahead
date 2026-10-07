"""XGBoost baseload model.

Where the profile estimator answers "what does this household normally use
at this hour on this weekday", this answers "what does it use at this hour
given that it is this cold, the sun is this high, and nobody is home" --
the things the profile cannot see because it averages them away. It needs
substantially more history before that extra freedom pays off, which is
what ``ml min days`` guards.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pvlib
import xgboost
from xgboost import XGBRegressor

from dao.forecast.baseload.profile import is_holiday

#: Every column the model is trained and predicted on, in a fixed order.
ML_FEATURES = (
    "hour",
    "weekday",
    "weekend",
    "holiday",
    "doy_sin",
    "doy_cos",
    "temp",
    "sun_elevation",
    "away",
)

#: Deliberately modest: a household's hourly consumption over a year is a
#: few thousand rows, where a deeper or longer-boosted model memorises the
#: training window instead of generalising to next week.
ML_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
    objective="reg:squarederror",
)

#: Below this many labelled away days the model has not seen enough of the
#: away regime to be trusted with it, and the away profile takes over.
MIN_AWAY_DAYS_FOR_ML = 10

#: Fallback when a month has no measured temperature at all: a rough Dutch
#: year-round average, used only so a missing forecast hour cannot drop the
#: whole prediction.
_DEFAULT_TEMP = 10.0

_MODEL_FILE = "model.json"
_META_FILE = "model.meta.json"


def climatological_temp(values_temp: pd.Series, month: int) -> float:
    """The mean measured temperature of ``month``, or 10 degrees when unknown."""
    if values_temp is None or len(values_temp) == 0:
        return _DEFAULT_TEMP
    month_values = values_temp[values_temp.index.month == month].dropna()
    if month_values.empty:
        return _DEFAULT_TEMP
    return float(month_values.mean())


def build_features(
    index: pd.DatetimeIndex,
    temp: pd.Series,
    away: pd.Series,
    latitude: float,
    longitude: float,
    holidays_mode: str,
) -> pd.DataFrame:
    """The feature frame for ``index``.

    Day of year enters as a sine/cosine pair rather than a raw number, so
    31 December and 1 January are neighbours to the model instead of
    opposite extremes.
    """
    # The sun's position halfway through the hour, the same convention the
    # PV model uses, so "sun is up" means up for most of the hour rather
    # than at the instant it started.
    midpoints = index + pd.Timedelta(minutes=30)
    solpos = pvlib.solarposition.get_solarposition(midpoints, latitude, longitude)
    elevation = 90.0 - np.asarray(solpos["apparent_zenith"], dtype=float)

    doy = np.asarray(index.dayofyear, dtype=float)
    weekday = np.asarray(index.dayofweek, dtype=float)

    if holidays_mode == "ignore":
        holiday = np.zeros(len(index), dtype=float)
    else:
        holiday = np.array(
            [1.0 if is_holiday(moment.date()) else 0.0 for moment in index]
        )

    temp_values = (
        pd.Series(temp).reindex(index).astype(float).to_numpy()
        if temp is not None
        else np.full(len(index), float("nan"))
    )
    away_values = (
        pd.Series(away).reindex(index).astype(float).fillna(0.0).to_numpy()
        if away is not None
        else np.zeros(len(index))
    )

    return pd.DataFrame(
        {
            "hour": np.asarray(index.hour, dtype=float),
            "weekday": weekday,
            "weekend": (weekday >= 5).astype(float),
            "holiday": holiday,
            "doy_sin": np.sin(2 * math.pi * doy / 365.25),
            "doy_cos": np.cos(2 * math.pi * doy / 365.25),
            "temp": temp_values,
            "sun_elevation": elevation,
            "away": away_values,
        },
        index=index,
    )[list(ML_FEATURES)]


def _hourly_away(index: pd.DatetimeIndex, away_labels: pd.Series) -> pd.Series:
    """A date-indexed away label spread over that day's hours."""
    if away_labels is None or len(away_labels) == 0:
        return pd.Series(0.0, index=index)
    lookup = {day: bool(value) for day, value in away_labels.items()}
    return pd.Series(
        [1.0 if lookup.get(moment.date(), False) else 0.0 for moment in index],
        index=index,
    )


class BaseloadMLModel:
    """An XGBoost regressor on the baseload, with its own feature contract."""

    def __init__(self, latitude: float, longitude: float, holidays_mode: str) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.holidays_mode = holidays_mode
        self.model: Optional[XGBRegressor] = None
        self.away_days = 0
        self.rows = 0

    def train(
        self, target: pd.Series, temp: pd.Series, away_labels: pd.Series
    ) -> None:
        """Fit on ``target``'s hours. Raises ValueError when nothing is usable."""
        index = pd.DatetimeIndex(target.index)
        away_hourly = _hourly_away(index, away_labels)
        features = build_features(
            index,
            temp,
            away_hourly,
            self.latitude,
            self.longitude,
            self.holidays_mode,
        )
        y = pd.Series(target.to_numpy(), index=index).astype(float)

        usable = ~(features.isna().any(axis=1) | y.isna())
        features = features[usable]
        y = y[usable]
        if len(features) == 0:
            raise ValueError("Baseload ML: geen bruikbare rijen om op te trainen")

        model = XGBRegressor(**ML_PARAMS)
        model.fit(features, y)
        self.model = model
        self.rows = int(len(features))
        self.away_days = (
            int(sum(1 for value in away_labels if bool(value)))
            if away_labels is not None
            else 0
        )

    def predict(
        self, index: pd.DatetimeIndex, temp: pd.Series, away: bool
    ) -> pd.Series:
        """Hourly kWh for ``index``, the whole stretch in one regime."""
        if self.model is None:
            raise ValueError("Baseload ML: model is niet getraind")
        index = pd.DatetimeIndex(index)
        away_series = pd.Series(1.0 if away else 0.0, index=index)
        features = build_features(
            index,
            temp,
            away_series,
            self.latitude,
            self.longitude,
            self.holidays_mode,
        )
        predicted = np.maximum(0.0, self.model.predict(features))
        return pd.Series(predicted, index=index, name="baseload")

    def save(self, data_dir: Path) -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(data_dir / _MODEL_FILE))
        meta = {
            "features": list(ML_FEATURES),
            "trained_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "rows": self.rows,
            "away_days": self.away_days,
            "xgboost_version": xgboost.__version__,
        }
        (data_dir / _META_FILE).write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(
        cls, data_dir: Path, latitude: float, longitude: float, holidays_mode: str
    ) -> Optional[BaseloadMLModel]:
        """The saved model, or ``None`` when there is none or it is stale.

        A feature-set mismatch returns ``None`` rather than raising: the
        forecast path falls back to the profile on anything it cannot use,
        and a model trained against an older feature list is exactly that.
        """
        data_dir = Path(data_dir)
        model_path = data_dir / _MODEL_FILE
        if not model_path.exists():
            return None

        meta_path = data_dir / _META_FILE
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except ValueError:
                meta = {}
            trained_features = meta.get("features")
            if trained_features is not None and trained_features != list(ML_FEATURES):
                logging.warning(
                    f"Baseload ML: model is getraind op {trained_features}, de code "
                    f"gebruikt {list(ML_FEATURES)}; model genegeerd tot het opnieuw "
                    f"is getraind"
                )
                return None

        instance = cls(latitude, longitude, holidays_mode)
        model = XGBRegressor()
        try:
            model.load_model(str(model_path))
        except Exception as ex:  # noqa: BLE001 - a corrupt file falls back too
            logging.warning(f"Baseload ML: model kon niet worden geladen ({ex})")
            return None
        instance.model = model
        instance.away_days = int(meta.get("away_days", 0))
        instance.rows = int(meta.get("rows", 0))
        return instance
