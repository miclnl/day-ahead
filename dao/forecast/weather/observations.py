"""Weather observations: KNMI for the Netherlands and Belgium, the
Open-Meteo historical archive everywhere else.

Where the forecast sources (Meteoserver, Open-Meteo's forecast endpoint)
predict, this module measures: it backfills the last few days of ``gr``,
``temp`` and ``winds`` in ``values`` so the accuracy report and the PV
calibration have something real to compare a forecast against.
"""

from __future__ import annotations

import datetime
import logging
from datetime import date
from typing import Optional

import knmi
import pandas as pd

from dao.forecast.weather.openmeteo import fetch_openmeteo_archive
from dao.forecast.weather.schema import WEATHER_COLUMNS, empty_weather_frame

#: KNMI's automatic weather stations (AWS) -- the ones that report hourly,
#: as opposed to the manual stations that only report daily. Generated with
#: prog/tst.py/generate_list_knmi_aws.py; check against KNMI's own station
#: list on a new knmi-py release.
KNMI_AWS_STATIONS = (
    215, 235, 240, 249, 251, 257, 260, 267, 269, 270, 273, 275, 277, 278, 279,
    280, 283, 286, 290, 310, 319, 323, 330, 344, 348, 350, 356, 370, 375, 377,
    380,
)

#: Codes update_observations writes; the columns fetch_knmi/fetch_openmeteo_archive
#: both fill in (unlike dni/dhi/neersl, which are either unavailable from KNMI
#: or not needed for the accuracy comparison this feeds).
_OBSERVED_CODES = ("gr", "temp", "winds")


def nearest_knmi_station(latitude: float, longitude: float) -> int:
    """The automatic weather station closest to (latitude, longitude)."""
    best_station: Optional[int] = None
    best_distance: Optional[float] = None
    for station_id in KNMI_AWS_STATIONS:
        station = knmi.stations[station_id]
        distance = (latitude - station.latitude) ** 2 + (
            longitude - station.longitude
        ) ** 2
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_station = station_id
    return best_station


def fetch_knmi(station: int, start: date, end: date) -> pd.DataFrame:
    """Hourly KNMI observations as a weather frame, for ``[start, end]``.

    knmi-py's index is already the hour *start* in UTC -- unlike
    Open-Meteo's, which labels the hour it ends -- so nothing is shifted
    here; doing so would double-correct on top of :func:`parse_openmeteo`.
    Rows with any missing variable are dropped rather than partially
    trusted: KNMI marks a sensor outage as absent data, not as zero.
    """
    frame = knmi.get_hour_data_dataframe(
        [station], start=start, end=end, variables=["Q", "T", "FH"]
    )
    if frame is None or len(frame) == 0:
        return empty_weather_frame()

    frame = frame.dropna()
    if len(frame) == 0:
        return empty_weather_frame()

    moments = pd.to_datetime(frame.index, utc=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    epochs = ((moments - epoch) // pd.Timedelta(seconds=1)).astype("int64")

    result = empty_weather_frame()
    result["time"] = epochs.values
    result["gr"] = frame["Q"].astype(float).values
    result["temp"] = frame["T"].astype(float).values / 10.0
    result["winds"] = frame["FH"].astype(float).values / 10.0
    result["source"] = "knmi"
    return result[["time", *WEATHER_COLUMNS, "source"]]


def observation_mode(mode: str, country: str) -> str:
    """Resolve "auto" against the installation's country; pass the rest through."""
    if mode == "auto":
        return "knmi" if country in ("NL", "BE") else "openmeteo"
    return mode


def update_observations(
    db_da,
    latitude: float,
    longitude: float,
    country: str,
    mode: str,
    days: int = 7,
    now: Optional[datetime.datetime] = None,
) -> int:
    """Fetch the last ``days`` days of observations and save them.

    Covers ``[today - days, today)``: today itself is excluded, since a
    station's data for the day in progress is normally still incomplete.
    Returns the number of (time, code) values written; ``0`` for a source
    that returned nothing, or unconditionally for ``mode == "off"``.
    """
    resolved = observation_mode(mode, country)
    if resolved == "off":
        return 0

    if now is None:
        now = datetime.datetime.now(datetime.UTC)
    today = now.date()
    start = today - datetime.timedelta(days=days)
    end = today - datetime.timedelta(days=1)

    if resolved == "knmi":
        station = nearest_knmi_station(latitude, longitude)
        frame = fetch_knmi(station, start, end)
    else:
        frame = fetch_openmeteo_archive(latitude, longitude, start, end)

    if frame is None or len(frame) == 0:
        return 0

    records = []
    for row in frame.itertuples():
        for code in _OBSERVED_CODES:
            value = getattr(row, code)
            if value == value:  # skip NaN, which compares unequal to itself
                records.append((int(row.time), code, float(value)))

    if not records:
        return 0

    save_frame = pd.DataFrame(records, columns=["time", "code", "value"])
    db_da.savedata(save_frame, tablename="values")
    logging.info(
        f"Waarnemingen ({resolved}): {len(records)} waarden opgeslagen vanaf "
        f"{start} tot {end}"
    )
    return len(records)
