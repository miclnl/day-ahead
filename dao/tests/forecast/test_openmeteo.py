"""Tests for the Open-Meteo parser and fetcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from dao.forecast.weather.openmeteo import (
    OPENMETEO_FORECAST_URL,
    fetch_openmeteo,
    parse_openmeteo,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def openmeteo_payload():
    return json.loads((FIXTURES / "openmeteo_forecast.json").read_text())


class _StubResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} fout")

    def json(self):
        return self._payload


class _StubSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if not self._responses:
            raise AssertionError("geen ingeplande responses meer")
        return self._responses.pop(0)


def test_parse_shifts_to_preceding_hour_start(openmeteo_payload):
    frame = parse_openmeteo(openmeteo_payload)
    # hourly.time[6] == "2026-09-30T06:00" -> time epoch of 05:00Z
    assert openmeteo_payload["hourly"]["time"][6] == "2026-09-30T06:00"
    import datetime as dt

    expected = int(dt.datetime(2026, 9, 30, 5, 0, tzinfo=dt.UTC).timestamp())
    assert frame.iloc[6]["time"] == expected


def test_parse_converts_units(openmeteo_payload):
    frame = parse_openmeteo(openmeteo_payload)
    # Hour 13 (index 13) is the peak of the synthetic daytime curve: 600 W/m^2.
    assert openmeteo_payload["hourly"]["shortwave_radiation"][13] == pytest.approx(600.0)
    assert frame.iloc[13]["gr"] == pytest.approx(216.0)  # 600 * 0.36


def test_instantaneous_values_keep_their_own_timestamp():
    """Open-Meteo documents shortwave_radiation, dni, dhi and precipitation
    as the preceding hour's mean or sum, but temperature_2m and
    wind_speed_10m as instantaneous at the stated time. Shifting the whole
    frame back an hour therefore puts temperature and wind an hour early.
    """
    payload = {
        "hourly": {
            "time": [
                "2026-09-30T01:00",
                "2026-09-30T02:00",
                "2026-09-30T03:00",
                "2026-09-30T04:00",
            ],
            "shortwave_radiation": [10.0, 20.0, 30.0, 40.0],
            "direct_normal_irradiance": [1.0, 2.0, 3.0, 4.0],
            "diffuse_radiation": [5.0, 6.0, 7.0, 8.0],
            "temperature_2m": [11.0, 12.0, 13.0, 14.0],
            "wind_speed_10m": [2.0, 3.0, 4.0, 5.0],
            "precipitation": [0.1, 0.2, 0.3, 0.4],
        }
    }

    frame = parse_openmeteo(payload)
    import datetime as dt

    def at(hour_utc: int):
        stamp = int(dt.datetime(2026, 9, 30, hour_utc, tzinfo=dt.UTC).timestamp())
        return frame[frame["time"] == stamp].iloc[0]

    # Row 02:00 covers 02:00-03:00, so it takes the radiation reported at
    # 03:00 (the mean of the hour that just ended)...
    assert at(2)["gr"] == pytest.approx(30.0 * 0.36)
    assert at(2)["dni"] == pytest.approx(3.0 * 0.36)
    assert at(2)["dhi"] == pytest.approx(7.0 * 0.36)
    assert at(2)["neersl"] == pytest.approx(0.3)
    # ...but the temperature and wind reported at 02:00, which are the
    # instantaneous values at that moment.
    assert at(2)["temp"] == pytest.approx(12.0)
    assert at(2)["winds"] == pytest.approx(3.0)


def test_parse_null_becomes_nan(openmeteo_payload):
    frame = parse_openmeteo(openmeteo_payload)
    assert len(frame) == len(openmeteo_payload["hourly"]["time"])
    assert frame.iloc[10]["gr"] != frame.iloc[10]["gr"]  # NaN
    assert frame.iloc[34]["gr"] != frame.iloc[34]["gr"]  # NaN


def test_fetch_retries_with_best_match_on_model_error(openmeteo_payload):
    session = _StubSession(
        [_StubResponse(400), _StubResponse(200, openmeteo_payload)]
    )

    frame = fetch_openmeteo(
        52.09, 5.12, model="knmi_seamless", attempts=2, session=session
    )

    assert len(frame) == len(openmeteo_payload["hourly"]["time"])
    assert len(session.calls) == 2
    assert session.calls[0]["params"]["models"] == "knmi_seamless"
    assert session.calls[1]["params"]["models"] == "best_match"
    assert session.calls[0]["url"] == OPENMETEO_FORECAST_URL


def test_fetch_returns_empty_frame_after_failures(caplog):
    session = _StubSession([_StubResponse(500), _StubResponse(500), _StubResponse(500)])

    with caplog.at_level("ERROR"):
        frame = fetch_openmeteo(
            52.09, 5.12, model="knmi_seamless", attempts=2, session=session
        )

    assert len(frame) == 0
    assert any("Open-Meteo" in message for message in caplog.messages)
