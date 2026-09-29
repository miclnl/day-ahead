"""The fast control status widget polls; polling must stay cheap.

The page refreshes its status every second or so. The v1 equivalent of this
endpoint called ConfigurationLoader.load_and_validate() on every tick,
which re-parsed and re-validated options.json, took the same fcntl.flock
the migration path uses, and could even *write* the file from what looks
like a read-only GET.

v2's endpoint reads only fast_state.json, so it never had that problem.
This keeps it that way: it would be an easy thing to reintroduce by
reaching for the config to render one more field.
"""

import importlib

import pytest

from .conftest import INGRESS, SUPERVISOR


@pytest.fixture
def v2_routes(client):
    return importlib.import_module("app.v2.routes")


def test_polling_the_state_endpoint_does_not_read_the_configuration(
    client, v2_routes, monkeypatch
):
    loads = []
    monkeypatch.setattr(
        v2_routes, "_load_config", lambda: loads.append(1) or None
    )

    response = client.get(
        "/v2/fast-control/state", headers=INGRESS, environ_base=SUPERVISOR
    )

    assert response.status_code == 200
    assert loads == [], "the polling endpoint parsed options.json"


def test_polling_the_state_endpoint_does_not_construct_a_loader(
    client, monkeypatch
):
    """Belt and braces: catch a direct ConfigurationLoader too, not just the
    module's own helper."""
    from dao.prog.config import loader as loader_module

    built = []

    class Tripwire(loader_module.ConfigurationLoader):
        def __init__(self, *args, **kwargs):
            built.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(loader_module, "ConfigurationLoader", Tripwire)

    client.get(
        "/v2/fast-control/state", headers=INGRESS, environ_base=SUPERVISOR
    )

    assert built == []


def test_the_state_endpoint_renders_what_the_runner_wrote(
    client, v2_routes, monkeypatch
):
    monkeypatch.setattr(
        v2_routes,
        "_load_fast_state",
        lambda: {
            "last_decision": {
                "state": "follow_plan",
                "mode": "shadow",
                "reason": "follow_plan",
                "house_w": 1200.0,
                "pv_w": 800.0,
                "benefit_eur_h": 0.02,
                "saved_today_eur": 0.15,
                "saved_today_is_estimate": True,
                "override": False,
            }
        },
    )

    response = client.get(
        "/v2/fast-control/state", headers=INGRESS, environ_base=SUPERVISOR
    )

    assert response.status_code == 200
    assert b"follow_plan" in response.data
