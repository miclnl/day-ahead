"""The v1 fast-control status poll must not re-parse options.json.

/fast_control/state.json is polled every 5 seconds by the page's own JS.
It used to call ConfigurationLoader(...).load_and_validate() on every poll,
which takes an exclusive fcntl.flock() and can write options.json back (the
migration branch) from what looks like a read-only GET.
"""

import pytest

from .conftest import INGRESS, SUPERVISOR


def test_polling_the_state_endpoint_does_not_reparse_options_json(client, monkeypatch):
    import app.routes as routes

    calls = []
    original = routes.ConfigurationLoader.load_and_validate

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(routes.ConfigurationLoader, "load_and_validate", counting)

    for _ in range(3):
        response = client.get(
            "/fast_control/state.json", headers=INGRESS, environ_base=SUPERVISOR
        )
        assert response.status_code == 200

    assert calls == [], (
        "the polling endpoint parsed/validated options.json; it should use "
        "the config already cached at import time"
    )


def test_the_full_page_render_also_uses_the_cached_config(client, monkeypatch):
    import app.routes as routes

    calls = []
    monkeypatch.setattr(
        routes.ConfigurationLoader,
        "load_and_validate",
        lambda self: calls.append(1),
    )

    response = client.get("/fast_control", headers=INGRESS, environ_base=SUPERVISOR)

    assert response.status_code == 200
    assert calls == []


def test_the_mode_still_comes_from_the_cached_config(client):
    """Sanity check that using the cache did not break the feature: the
    example config's fast-control mode must still show up on the page."""
    response = client.get(
        "/fast_control/state.json", headers=INGRESS, environ_base=SUPERVISOR
    )
    assert response.status_code == 200
    assert response.json["mode"] in ("off", "shadow", "active")
