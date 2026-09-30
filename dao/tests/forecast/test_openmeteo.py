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
    assert frame.iloc[13]["winds"] == pytest.approx(
        openmeteo_payload["hourly"]["wind_speed_10m"][13]
    )


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
