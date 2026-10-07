"""Feature engineering and training-data selection for the PV ML model.

The physical model's own prediction is one of the ML model's inputs: with a
year or so of history and a system that occasionally gets cleaned,
re-angled or has a panel replaced, the physical model already captures the
slow, well-understood part (sun geometry, orientation, capacity); the ML
model only has to learn what the physical model cannot see coming from
weather data alone -- shading from a tree that grew in, a chimney's
shadow, gradual soiling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pvlib

from dao.forecast.pv.physical import (
    PVParams,
    clearsky_irradiance,
    simulate,
    weather_for_pv,
)

#: Every column the ML model is trained and predicted on, in a fixed order.
#: No ``day_of_week``: production does not know what day it is, only where
#: the sun is and what the sky is doing.
FEATURES = (
    "ghi",
    "dni",
    "dhi",
    "temp",
    "wind",
    "zenith",
    "azimuth",
    "clearsky_ghi",
    "hour",
    "doy",
    "physical",
)

#: The only two features allowed to be missing. Meteoserver and KNMI never
#: supply the direct and diffuse components, and ``update_observations``
#: writes gr/temp/winds only, so on the observation branch these are NaN
#: for every row there is. XGBoost learns a default split direction for a
#: missing value, which is exactly the right behaviour here; dropping the
#: rows instead meant dropping all of them.
OPTIONAL_FEATURES = ("dni", "dhi")

#: Features a training row must actually have. A row missing any of these
#: has nothing to teach the model and is dropped.
REQUIRED_FEATURES = tuple(name for name in FEATURES if name not in OPTIONAL_FEATURES)

#: Below this many distinct days, archived forecasts are not enough to
#: train on; a fresh install has no archive yet, and a short one is not
#: representative of what the model will be fed once deployed.
DEFAULT_MIN_ARCHIVE_DAYS = 90

#: Codes read from either source, in the shape weather_for_pv expects.
_WEATHER_CODES = ("gr", "dni", "dhi", "temp", "winds")


def build_features(
    weather: pd.DataFrame,
    latitude: float,
    longitude: float,
    params: PVParams,
    interval_s: int = 3600,
) -> pd.DataFrame:
    """``weather`` (pv layout: ghi/dni/dhi/temp/wind) -> the full feature set.

    dni/dhi pass through unchanged, NaN where the source never supplied
    them (Meteoserver, KNMI) -- XGBoost handles missing values natively, so
    there is no decomposition to do here the way the physical model needs
    one. Solar position, clear-sky ghi and the calendar features are all
    evaluated at the interval midpoint, the same convention the physical
    model uses.
    """
    times = weather.index
    sun_times = times + pd.Timedelta(seconds=interval_s / 2)

    solpos = pvlib.solarposition.get_solarposition(sun_times, latitude, longitude)
    zenith = np.asarray(solpos["apparent_zenith"], dtype=float)
    azimuth = np.asarray(solpos["azimuth"], dtype=float)

    airmass = pvlib.atmosphere.get_relative_airmass(zenith)
    airmass = np.where(np.isnan(airmass), 10.0, airmass)
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass)
    linke_turbidity = np.asarray(
        pvlib.clearsky.lookup_linke_turbidity(sun_times, latitude, longitude),
        dtype=float,
    )
    dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(sun_times), dtype=float)
    clearsky = clearsky_irradiance(
        zenith, airmass_absolute, linke_turbidity, dni_extra
    )

    physical = simulate(params, weather, latitude, longitude, interval_s)

    features = pd.DataFrame(index=times)
    features["ghi"] = weather["ghi"].to_numpy()
    features["dni"] = weather["dni"].to_numpy()
    features["dhi"] = weather["dhi"].to_numpy()
    features["temp"] = weather["temp"].to_numpy()
    features["wind"] = weather["wind"].to_numpy()
    features["zenith"] = zenith
    features["azimuth"] = azimuth
    features["clearsky_ghi"] = np.asarray(clearsky["ghi"], dtype=float)
    features["hour"] = times.hour
    features["doy"] = times.dayofyear
    features["physical"] = physical.to_numpy()
    return features[list(FEATURES)]


def _archive_weather(db_da, start, end, tz: str) -> Optional[pd.DataFrame]:
    archive = db_da.forecast_rows(
        list(_WEATHER_CODES), [12, 24], int(start.timestamp()), int(end.timestamp())
    )
    if archive is None or len(archive) == 0:
        return None
    pivot = (
        archive.pivot_table(
            index="target_time", columns="code", values="value", aggfunc="first"
        )
        .reset_index()
        .rename(columns={"target_time": "time"})
    )
    for code in _WEATHER_CODES:
        if code not in pivot.columns:
            pivot[code] = float("nan")
    return pivot


def _observation_weather(db_da, start, end) -> Optional[pd.DataFrame]:
    columns: dict = {}
    for code in _WEATHER_CODES:
        frame = db_da.get_column_data("values", code, start=start, end=end)
        if frame is None or len(frame) == 0:
            continue
        columns[code] = pd.Series(
            frame["value"].astype(float).to_numpy(),
            index=frame["utc"].astype("int64").to_numpy(),
        )
    if "gr" not in columns or columns["gr"].empty:
        return None
    frame = pd.DataFrame(columns)
    frame.index.name = "time"
    frame = frame.reset_index()
    for code in _WEATHER_CODES:
        if code not in frame.columns:
            frame[code] = float("nan")
    return frame


def training_weather(
    db_da,
    start,
    end,
    tz: str,
    min_archive_days: int = DEFAULT_MIN_ARCHIVE_DAYS,
) -> tuple:
    """Weather to train the ML model on, and which source it came from.

    Training on archived forecasts is better once there is enough of it:
    the model then learns from the same kind of (imperfect) input it will
    be fed at prediction time, rather than from measurements it will never
    see again. Below ``min_archive_days`` distinct days, or with no archive
    at all, falls back to measured observations.
    """
    archive = _archive_weather(db_da, start, end, tz)
    if archive is not None:
        distinct_days = (
            pd.to_datetime(archive["time"], unit="s", utc=True).dt.date.nunique()
        )
        if distinct_days >= min_archive_days:
            return weather_for_pv(archive, tz), "archive"

    observations = _observation_weather(db_da, start, end)
    if observations is None:
        observations = pd.DataFrame(columns=["time", *_WEATHER_CODES])
    return weather_for_pv(observations, tz), "observations"
