"""Tests for the Meteoserver fetcher (moved from dao/tests/lib/test_da_meteo.py:
no test there exercised get_from_meteoserver, so this is the first coverage)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dao.forecast.weather.meteoserver import METEOSERVER_URLS, fetch_meteoserver

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def meteoserver_payload():
    return json.loads((FIXTURES / "meteoserver_uurverwachting.json").read_text())


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _StubSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        return self._responses.pop(0)


def test_fetch_meteoserver_parses_recorded_response(meteoserver_payload):
    session = _StubSession([_StubResponse(meteoserver_payload)])

    frame = fetch_meteoserver(
        key="test-key",
        model="harmonie",
        attempts=1,
        latitude=52.09,
        longitude=5.12,
        session=session,
    )

    assert len(frame) == 48
    assert (frame["source"] == "meteoserver").all()
    assert frame["dni"].isna().all()
    assert frame["dhi"].isna().all()
    assert session.calls[0]["url"] == METEOSERVER_URLS["harmonie"]
    assert session.calls[0]["params"]["key"] == "test-key"


def test_fetch_meteoserver_without_key_returns_empty(caplog):
    with caplog.at_level("ERROR"):
        frame = fetch_meteoserver(
            key="",
            model="harmonie",
            attempts=1,
            latitude=52.09,
            longitude=5.12,
            session=_StubSession([]),
        )

    assert len(frame) == 0
    assert any("meteoserver key" in message for message in caplog.messages)
