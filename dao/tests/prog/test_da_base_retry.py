"""DaBase retries transient Home Assistant failures, not permanent ones.

calc_optimum reads Home Assistant roughly 40 times per run; a single network
blip or a momentarily overloaded HA instance used to fall straight through
to FlexValue's default (or raise, before that fix), instead of the retry
simply absorbing it. A 401/403/404 is a configuration error retrying cannot
fix, and must fail on the first attempt so the default-fallback logging is
not delayed by several seconds of pointless backoff.
"""

import time

import pytest
from homeassistant_api.errors import (
    EndpointNotFoundError,
    InternalServerError,
    UnauthorizedError,
)
from niquests.exceptions import ConnectionError as HAConnectionError
from niquests.exceptions import Timeout as HATimeout

from dao.prog.da_base import DaBase


class FakeHAClient:
    """Stand-in for the homeassistant_api.Client instance DaBase composes
    as self._ha_client. Each method below is monkeypatched per test to the
    exact failure/success sequence that test wants to exercise."""


@pytest.fixture
def instance():
    obj = DaBase.__new__(DaBase)
    obj._ha_client = FakeHAClient()
    return obj


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    """These tests verify attempt counts and give-up/no-retry behaviour, not
    the actual backoff timing; skip the real sleep so the suite stays fast.
    tenacity.nap.sleep(seconds) calls time.sleep(seconds); tenacity captured
    a reference to nap.sleep as BaseRetrying's default at import time, so
    patching nap.sleep itself would be too late, but time.sleep is looked
    up fresh on the module object on every call."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


def test_get_state_retries_a_transient_connection_error(instance):
    calls = {"n": 0}

    def flaky(*, entity_id):
        calls["n"] += 1
        if calls["n"] < 3:
            raise HAConnectionError("boom")
        return "42"

    instance._ha_client.get_state = flaky

    assert instance.get_state("sensor.x") == "42"
    assert calls["n"] == 3


def test_get_state_does_not_retry_a_permanent_error(instance):
    calls = {"n": 0}

    def not_found(*, entity_id):
        calls["n"] += 1
        raise EndpointNotFoundError("no such entity")

    instance._ha_client.get_state = not_found

    with pytest.raises(EndpointNotFoundError):
        instance.get_state("sensor.y")
    assert calls["n"] == 1


def test_get_state_gives_up_after_three_attempts(instance):
    calls = {"n": 0}

    def always_fails(*, entity_id):
        calls["n"] += 1
        raise HATimeout("slow")

    instance._ha_client.get_state = always_fails

    with pytest.raises(HATimeout):
        instance.get_state("sensor.z")
    assert calls["n"] == 3


def test_call_service_retry_also_covers_turn_on_and_set_value(instance):
    """turn_on/turn_off/select_option/set_value are thin wrappers around
    call_service; overriding call_service alone must cover all of them
    through normal method dispatch."""
    calls = {"n": 0}

    def flaky(domain, service, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise InternalServerError(503, "busy")
        return "ok"

    instance._ha_client.trigger_service = flaky

    assert instance.turn_on("switch.x") == "ok"
    assert calls["n"] == 2


def test_call_service_does_not_retry_unauthorised(instance):
    calls = {"n": 0}

    def unauthorised(domain, service, **kwargs):
        calls["n"] += 1
        raise UnauthorizedError("bad token")

    instance._ha_client.trigger_service = unauthorised

    with pytest.raises(UnauthorizedError):
        instance.turn_off("switch.x")
    assert calls["n"] == 1


def test_set_state_also_retries(instance):
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        if calls["n"] < 2:
            raise HAConnectionError("boom")
        return "ok"

    instance._ha_client.set_state = flaky

    assert instance.set_state("sensor.status", "on") == "ok"
    assert calls["n"] == 2


def test_call_service_derives_the_domain_from_entity_id(instance):
    """trigger_service() wants the domain as its own argument;
    call_service() must derive it from entity_id rather than require every
    caller to pass it separately (matching the old hassapi call shape)."""
    captured = {}

    def capture(domain, service, **kwargs):
        captured["domain"] = domain
        captured["service"] = service
        captured["kwargs"] = kwargs
        return "ok"

    instance._ha_client.trigger_service = capture

    instance.select_option("select.mode", "eco")

    assert captured == {
        "domain": "select",
        "service": "select_option",
        "kwargs": {"entity_id": "select.mode", "option": "eco"},
    }
