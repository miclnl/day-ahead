"""Fetch forecast and archived weather from Open-Meteo.

Open-Meteo is the fallback for both a live forecast (when Meteoserver has no
key configured or is unreachable) and for filling historical gaps (the
archive endpoint, used to calibrate the PV model and to backfill
observations). Both share one parser: the response shape only differs in
which dates it covers.
"""

from __future__ import annotations

import logging
import time as time_module
from datetime import date

import pandas as pd
import requests

from dao.forecast.weather.schema import empty_weather_frame

OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = (
    "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,"
    "temperature_2m,wind_speed_10m,precipitation"
)

#: Falling back to Open-Meteo's own best-guess ensemble when a named model
#: is unavailable for the requested location or has been retired.
_FALLBACK_MODEL = "best_match"


def _numeric_series(values, length: int) -> pd.Series:
    """``values`` (possibly ``None``, possibly containing JSON ``null``) as
    a float series of exactly ``length`` entries."""
    if values is None:
        return pd.Series([float("nan")] * length)
    return pd.Series([float("nan") if v is None else float(v) for v in values])


def parse_openmeteo(payload: dict, source: str = "openmeteo") -> pd.DataFrame:
    """Open-Meteo's JSON response as a weather frame.

    Every row of this package's weather frame describes the hour that
    *starts* at its timestamp. Open-Meteo mixes two conventions within one
    response, both documented in its "Hourly Parameter Definition" table:

    * ``shortwave_radiation``, ``direct_normal_irradiance``,
      ``diffuse_radiation`` and ``precipitation`` are the mean or sum over
      the hour *preceding* the stated time, so the value at ``T`` belongs
      to the row starting at ``T - 1h``. The whole index is shifted back by
      one hour for them.
    * ``temperature_2m`` and ``wind_speed_10m`` are instantaneous at the
      stated time. They are shifted forward by one position to compensate,
      which leaves them on their own timestamp.

    The first row then has no instantaneous reading of its own (its hour
    starts before the response begins); it borrows the next hour's, which
    is one hour of drift on a single edge row rather than a gap the PV
    model would have to fill with a default.
    """
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return empty_weather_frame()

    moments = pd.to_datetime(list(times), utc=True) - pd.Timedelta(hours=1)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    epochs = ((moments - epoch) // pd.Timedelta(seconds=1)).astype("int64")

    length = len(times)

    def instantaneous(key: str) -> pd.Series:
        return _numeric_series(hourly.get(key), length).shift(1).bfill()

    frame = empty_weather_frame()
    frame["time"] = epochs
    frame["gr"] = _numeric_series(hourly.get("shortwave_radiation"), length) * 0.36
    frame["dni"] = _numeric_series(hourly.get("direct_normal_irradiance"), length) * 0.36
    frame["dhi"] = _numeric_series(hourly.get("diffuse_radiation"), length) * 0.36
    frame["temp"] = instantaneous("temperature_2m")
    frame["winds"] = instantaneous("wind_speed_10m")
    frame["neersl"] = _numeric_series(hourly.get("precipitation"), length)
    frame["source"] = source
    return frame


def fetch_openmeteo(
    latitude,
    longitude,
    model: str = "knmi_seamless",
    forecast_days: int = 3,
    attempts: int = 2,
    session=requests,
) -> pd.DataFrame:
    """A weather frame from Open-Meteo's forecast endpoint.

    A named model that Open-Meteo rejects (retired, or not covering this
    location) falls back to ``best_match`` once; every attempt after that
    keeps using whichever model last worked.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": HOURLY_VARS,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "forecast_days": forecast_days,
        "models": model,
    }
    max_attempts = max(1, int(attempts or 0) + 1)
    current_model = model
    fell_back = current_model == _FALLBACK_MODEL

    for attempt in range(1, max_attempts + 1):
        params["models"] = current_model
        try:
            response = session.get(OPENMETEO_FORECAST_URL, params=params, timeout=(5, 30))
            response.raise_for_status()
            return parse_openmeteo(response.json())
        except (requests.RequestException, ValueError) as ex:
            logging.warning(
                f"Open-Meteo poging {attempt} van {max_attempts} mislukt: {ex}"
            )
            if not fell_back:
                logging.warning(
                    f"Open-Meteo model {current_model!r} mislukt, val terug op "
                    f"{_FALLBACK_MODEL}"
                )
                current_model = _FALLBACK_MODEL
                fell_back = True
        if attempt < max_attempts:
            time_module.sleep(min(30, 2**attempt))

    logging.error(f"Geen meteodata ontvangen van Open-Meteo na {max_attempts} pogingen")
    return empty_weather_frame()


def fetch_openmeteo_archive(
    latitude, longitude, start: date, end: date, session=requests
) -> pd.DataFrame:
    """A weather frame from Open-Meteo's historical archive."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": HOURLY_VARS,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    try:
        response = session.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=(5, 30))
        response.raise_for_status()
        return parse_openmeteo(response.json(), source="openmeteo-archive")
    except (requests.RequestException, ValueError) as ex:
        logging.warning(f"Open-Meteo archief mislukt: {ex}")
        return empty_weather_frame()
