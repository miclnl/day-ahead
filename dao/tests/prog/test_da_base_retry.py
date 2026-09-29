"""DaBase retries transient Home Assistant failures, not permanent ones.

calc_optimum reads Home Assistant roughly 40 times per run; a single network
blip or a momentarily overloaded HA instance used to fall straight through
to FlexValue's default (or raise, before that fix), instead of the retry
simply absorbing it. A 401/403/404 is a configuration error retrying cannot
fix, and must fail on the first attempt so the default-fallback logging is
not delayed by several seconds of pointless backoff.
"""

import time

import hassapi as hass
import pytest
import requests
from hassapi.exceptions import NotFound, ServiceUnavailable, Unauthorised

from dao.prog.da_base import DaBase


@pytest.fixture
def instance():
    return DaBase.__new__(DaBase)


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    """These tests verify attempt counts and give-up/no-retry behaviour, not
    the actual backoff timing; skip the real sleep so the suite stays fast.
    tenacity.nap.sleep(seconds) calls time.sleep(seconds); tenacity captured
    a reference to nap.sleep as BaseRetrying's default at import time, so
    patching nap.sleep itself would be too late, but time.sleep is looked
    up fresh on the module object on every call."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


@pytest.fixture
def restore_hass_methods():
    saved = {
        "get_state": hass.Hass.get_state,
        "call_service": hass.Hass.call_service,
        "set_state": hass.Hass.set_state,
    }
    yield
    for name, method in saved.items():
        setattr(hass.Hass, name, method)


def test_get_state_retries_a_transient_connection_error(instance, restore_hass_methods, monkeypatch):
    calls = {"n": 0}

    def flaky(self, entity_id):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectionError("boom")
        return "42"

    monkeypatch.setattr(hass.Hass, "get_state", flaky)

    assert instance.get_state("sensor.x") == "42"
    assert calls["n"] == 3


def test_get_state_does_not_retry_a_permanent_error(instance, restore_hass_methods, monkeypatch):
    calls = {"n": 0}

    def not_found(self, entity_id):
        calls["n"] += 1
        raise NotFound("no such entity")

    monkeypatch.setattr(hass.Hass, "get_state", not_found)

    with pytest.raises(NotFound):
        instance.get_state("sensor.y")
    assert calls["n"] == 1


def test_get_state_gives_up_after_three_attempts(instance, restore_hass_methods, monkeypatch):
    calls = {"n": 0}

    def always_fails(self, entity_id):
        calls["n"] += 1
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr(hass.Hass, "get_state", always_fails)

    with pytest.raises(requests.exceptions.Timeout):
        instance.get_state("sensor.z")
    assert calls["n"] == 3


def test_call_service_retry_also_covers_turn_on_and_set_value(
    instance, restore_hass_methods, monkeypatch
):
    """turn_on/turn_off/select_option/set_value are thin wrappers around
    call_service in hassapi; overriding call_service alone must cover all of
    them through normal method dispatch."""
    calls = {"n": 0}

    def flaky(self, service, entity_id, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ServiceUnavailable("busy")
        return "ok"

    monkeypatch.setattr(hass.Hass, "call_service", flaky)

    assert instance.turn_on("switch.x") == "ok"
    assert calls["n"] == 2


def test_call_service_does_not_retry_unauthorised(instance, restore_hass_methods, monkeypatch):
    calls = {"n": 0}

    def unauthorised(self, service, entity_id, **kwargs):
        calls["n"] += 1
        raise Unauthorised("bad token")

    monkeypatch.setattr(hass.Hass, "call_service", unauthorised)

    with pytest.raises(Unauthorised):
        instance.turn_off("switch.x")
    assert calls["n"] == 1


def test_set_state_also_retries(instance, restore_hass_methods, monkeypatch):
    calls = {"n": 0}

    def flaky(self, entity_id, state, attributes=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.exceptions.ConnectionError("boom")
        return "ok"

    monkeypatch.setattr(hass.Hass, "set_state", flaky)

    assert instance.set_state("sensor.status", "on") == "ok"
    assert calls["n"] == 2
