"""Fetch hourly forecasts from Meteoserver.

Meteoserver publishes two models under the same response shape: ``harmonie``
(KNMI's short-range model, hourly, ~48-96h ahead) and ``gfs`` (NOAA's,
coarser but reaching further). Neither splits radiation into direct and
diffuse components, so ``dni``/``dhi`` stay ``NaN`` here; the physical PV
model fills those in from the global radiation and the sun's position.
"""

from __future__ import annotations

import logging
import time as time_module

import pandas as pd
import requests

from dao.forecast.weather.schema import WEATHER_COLUMNS, empty_weather_frame

METEOSERVER_URLS = {
    "harmonie": "https://data.meteoserver.nl/api/uurverwachting.php",
    "gfs": "https://data.meteoserver.nl/api/uurverwachting_gfs.php",
}

#: Fields Meteoserver's response must have for a row to be usable.
_REQUIRED_FIELDS = ("tijd", "tijd_nl", "gr", "temp", "winds", "neersl")

#: Meteoserver's own forecast horizon; more rows than this would be unexpected.
_MAX_ROWS = 96


def fetch_meteoserver(
    key: str,
    model: str,
    attempts: int,
    latitude: float,
    longitude: float,
    session=requests,
) -> pd.DataFrame:
    """A weather frame from Meteoserver, or an empty one after every attempt fails.

    ``attempts`` is retries on top of the first request, so up to
    ``attempts + 1`` requests are made, with an exponential pause (capped at
    30s) between them -- the same backoff the pre-package code used, so an
    outage does not turn into a request storm.
    """
    if not key:
        logging.error("Geen meteoserver key geconfigureerd, geen meteodata opgehaald")
        return empty_weather_frame()

    url = METEOSERVER_URLS.get(model, METEOSERVER_URLS["gfs"])
    params = {"lat": str(latitude), "long": str(longitude), "key": key}

    max_attempts = max(1, int(attempts or 0) + 1)
    data = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, params=params, timeout=(5, 30))
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as ex:
            logging.warning(
                f"Meteoserver poging {attempt} van {max_attempts} mislukt: {ex}"
            )
            payload = {}
        if isinstance(payload, dict) and payload.get("data"):
            data = payload["data"]
            break
        if attempt < max_attempts:
            time_module.sleep(min(30, 2**attempt))

    if data is None:
        logging.error(
            f"Geen meteodata ontvangen van meteoserver na {max_attempts} pogingen"
        )
        return empty_weather_frame()

    raw = pd.DataFrame.from_records(data)
    missing = [field for field in _REQUIRED_FIELDS if field not in raw.columns]
    if missing:
        logging.error(f"Meteoserver antwoord mist kolommen {missing}")
        return empty_weather_frame()

    raw = raw[list(_REQUIRED_FIELDS)].iloc[:_MAX_ROWS]
    frame = empty_weather_frame()
    frame["time"] = raw["tijd"].astype("int64")
    frame["gr"] = pd.to_numeric(raw["gr"], errors="coerce")
    frame["temp"] = pd.to_numeric(raw["temp"], errors="coerce")
    frame["winds"] = pd.to_numeric(raw["winds"], errors="coerce")
    frame["neersl"] = pd.to_numeric(raw["neersl"], errors="coerce")
    frame["dni"] = float("nan")
    frame["dhi"] = float("nan")
    frame["source"] = "meteoserver"
    return frame[["time", *WEATHER_COLUMNS, "source"]]
