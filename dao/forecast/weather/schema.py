"""Column contract and unit conversion shared by every weather source.

Every weather source (Meteoserver, Open-Meteo, the KNMI/Open-Meteo archive)
converts its own response into this one shape before anything downstream
sees it, so the PV model and the forecast writer never need to know which
source produced a row.
"""

from __future__ import annotations

import pandas as pd

#: Radiation and weather columns every weather frame carries, beyond
#: ``time`` and ``source``. ``gr`` is global radiation; ``dni``/``dhi`` are
#: the direct-normal and diffuse-horizontal components the pvlib model
#: needs and that Meteoserver does not provide.
WEATHER_COLUMNS = ("gr", "dni", "dhi", "temp", "winds", "neersl")


def wm2_to_jcm2h(x):
    """W/m² (Open-Meteo, pvlib) -> J/cm² per hour (the database's unit)."""
    return x * 0.36


def jcm2h_to_wm2(x):
    """J/cm² per hour -> W/m²."""
    return x / 0.36


def empty_weather_frame() -> pd.DataFrame:
    """A weather frame with the right columns and dtypes, but no rows."""
    frame = pd.DataFrame({"time": pd.Series(dtype="int64")})
    for column in WEATHER_COLUMNS:
        frame[column] = pd.Series(dtype="float64")
    frame["source"] = pd.Series(dtype="object")
    return frame


def validate_weather_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check the column contract, sort by time, drop duplicate hours.

    The first row of a duplicated hour wins: sources are queried in
    priority order (Meteoserver before Open-Meteo, say), and the earlier
    call is the one that should not be silently overwritten by a fallback
    that ran only because the first looked short.
    """
    expected = ("time", *WEATHER_COLUMNS, "source")
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"weerframe mist kolommen: {missing}")
    result = df.sort_values("time", kind="stable").drop_duplicates(
        subset="time", keep="first"
    )
    return result.reset_index(drop=True)
