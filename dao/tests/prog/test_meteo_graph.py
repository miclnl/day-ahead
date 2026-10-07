"""The meteo graph's title and hour axis, built from the weather frame.

The frame the weather service hands over has columns
``time, gr, dni, dhi, temp, winds, neersl, source`` and no ``tijd_nl``.
The graph used to read its title from column 2 by position (which is now
``dni``, NaN for every Meteoserver row) and its hour labels from the epoch
in UTC, so the title read "vanaf nan" and the clock was one or two hours
off for half the year.
"""

from __future__ import annotations

import datetime as dt
import types
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from dao.lib.da_meteo import Meteo  # noqa: E402

TZ = "Europe/Amsterdam"
ZONE = ZoneInfo(TZ)


def weather_frame(start: dt.datetime, hours: int = 6) -> pd.DataFrame:
    """Exactly the shape WeatherService.fetch returns."""
    rows = []
    for index in range(hours):
        moment = start + dt.timedelta(hours=index)
        rows.append(
            {
                "time": int(moment.timestamp()),
                "gr": 100.0 + index,
                "dni": float("nan"),
                "dhi": float("nan"),
                "temp": 10.0 + index,
                "winds": 3.0,
                "neersl": 0.0,
                "source": "meteoserver",
            }
        )
    return pd.DataFrame(rows)


def make_meteo() -> Meteo:
    meteo = Meteo.__new__(Meteo)
    meteo.graphics_style = "default"
    meteo.time_zone = TZ
    return meteo


def captured_options(meteo: Meteo, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Run make_graph_meteo with the plot builder stubbed out."""
    seen: dict = {}

    class _StubBuilder:
        def build(self, df, options, show=False):
            seen["df"] = df
            seen["options"] = options
            return types.SimpleNamespace(savefig=lambda path: None)

    import dao.lib.da_meteo as module

    original = module.GraphBuilder
    module.GraphBuilder = _StubBuilder
    try:
        meteo.make_graph_meteo(frame)
    finally:
        module.GraphBuilder = original
    return seen["options"], seen["df"]


def test_title_names_the_first_moment_in_local_time():
    # 2026-07-01 10:00 local is 08:00 UTC: a two-hour offset in summer.
    start = dt.datetime(2026, 7, 1, 10, 0, tzinfo=ZONE)
    options, _df = captured_options(make_meteo(), weather_frame(start))

    assert "nan" not in options["title"].lower()
    assert "2026-07-01 10:00" in options["title"]


def test_hour_axis_is_local_time_not_utc():
    start = dt.datetime(2026, 7, 1, 10, 0, tzinfo=ZONE)
    _options, df = captured_options(make_meteo(), weather_frame(start, hours=4))

    assert list(df["uur"]) == ["10", "11", "12", "13"]


def test_a_frame_that_still_has_tijd_nl_keeps_working():
    """Nothing in this repository produces one any more, but a caller with
    an older frame must not break."""
    start = dt.datetime(2026, 7, 1, 10, 0, tzinfo=ZONE)
    frame = weather_frame(start, hours=3)
    frame["tijd_nl"] = [
        (start + dt.timedelta(hours=index)).strftime("%Y-%m-%d %H:%M")
        for index in range(3)
    ]

    _options, df = captured_options(make_meteo(), frame)

    assert list(df["uur"]) == ["10", "11", "12"]
