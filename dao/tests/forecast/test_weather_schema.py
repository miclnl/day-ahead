"""Tests for the weather column contract and unit conversion."""

import pandas as pd
import pytest

from dao.forecast.weather.schema import (
    WEATHER_COLUMNS,
    empty_weather_frame,
    jcm2h_to_wm2,
    validate_weather_frame,
    wm2_to_jcm2h,
)


def test_conversion_round_trip():
    assert jcm2h_to_wm2(wm2_to_jcm2h(250.0)) == pytest.approx(250.0)


def test_one_kw_hour_is_360_jcm2():
    assert wm2_to_jcm2h(1000.0) == pytest.approx(360.0)


def test_validate_drops_duplicate_times_and_sorts():
    rows = [
        {"time": 300, "source": "test", **{c: 1.0 for c in WEATHER_COLUMNS}},
        {"time": 100, "source": "test", **{c: 2.0 for c in WEATHER_COLUMNS}},
        {"time": 100, "source": "test", **{c: 3.0 for c in WEATHER_COLUMNS}},
    ]
    frame = pd.concat([empty_weather_frame(), pd.DataFrame(rows)], ignore_index=True)

    result = validate_weather_frame(frame)

    assert list(result["time"]) == [100, 300]
    assert result.iloc[0]["gr"] == pytest.approx(2.0)  # first duplicate wins
