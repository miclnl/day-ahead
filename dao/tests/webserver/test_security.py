"""Security regression tests for the dashboard.

The web server has no login of its own; Home Assistant's ingress is the
authentication. These tests pin down the consequences of that:

* requests that do not come through the supervisor are refused,
* the data directory (secrets.json, database) is not served,
* only PNG graphs are reachable through /images, without traversal,
* the settings editor only touches options.json and secrets.json,
* state-changing forms need a CSRF token,
* the fast-control mode switch edits one key instead of rewriting the file.

The app reads ../data relative to its working directory, so the fixture builds
a small site directory (data copied from the shipped example, app package
linked) and imports the app from there.
"""

import json

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_wtf")

from .conftest import INGRESS, SUPERVISOR, csrf_token as _csrf_token  # noqa: E402


def test_direct_requests_are_refused(client):
    assert client.get("/").status_code == 401
    assert client.get("/", headers=INGRESS).status_code == 401
    assert client.get("/", environ_base=SUPERVISOR).status_code == 401


def test_ingress_requests_are_served(client):
    """The bare ingress path redirects to the dashboard.

    The old interface owned "/", so it now needs a redirect of its own --
    without one, the address Home Assistant's ingress actually opens would
    be a 404.
    """
    response = client.get("/", headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 302
    assert "/v2" in response.headers["Location"]


def test_data_directory_is_not_served_as_static(client):
    for path in ("/static/data/secrets.json", "/static/data/options.json"):
        response = client.get(path, headers=INGRESS, environ_base=SUPERVISOR)
        assert response.status_code == 404, path


def test_image_route_serves_only_png_names(client):
    ok = client.get(
        "/images/calc_2026-09-28__12-00.png", headers=INGRESS, environ_base=SUPERVISOR
    )
    assert ok.status_code == 200
    assert ok.content_type == "image/png"
    for name in ("../secrets.json", "..%2Fsecrets.json", "secrets.json", "x.png.json"):
        response = client.get(f"/images/{name}", headers=INGRESS, environ_base=SUPERVISOR)
        assert response.status_code == 404, name


def test_no_route_takes_a_settings_filename(client):
    """The traversal this used to guard against is now structurally
    impossible.

    The old editor took the file to edit from the url
    (/settings/<filename>), which is what allowed "../evil". The current
    pages are /v2/config and /v2/secrets, each pinned to one fixed path, so
    there is no filename to smuggle anything through.
    """
    from app import app

    for rule in app.url_map.iter_rules():
        assert "settings" not in str(rule), rule


def test_the_config_editor_writes_only_its_own_file(client, site):
    # "site" is module-scoped: its options.json is shared with every other
    # test in this file. "{"nonsense": true}" now validates on its own
    # (every top-level field has a default), so the POST really does
    # overwrite it -- restore the previous content after checking the
    # traversal property this test actually cares about, or every test
    # after this one loads a config with nothing in it.
    options = site / "data" / "options.json"
    before = options.read_text(encoding="utf-8")
    token = _csrf_token(client, "/v2/config")
    try:
        response = client.post(
            "/v2/config",
            data={"config": '{"nonsense": true}', "csrf_token": token},
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert response.status_code == 200
        assert not (site / "evil.json").exists()
        assert not (site / "data" / "evil.json").exists()
    finally:
        options.write_text(before, encoding="utf-8")


def test_the_config_editor_rejects_invalid_config(client, site):
    """A typo in the editor must not be able to leave an options.json the
    scheduler cannot load, which would put it in a restart loop."""
    token = _csrf_token(client, "/v2/config")
    before = (site / "data" / "options.json").read_text(encoding="utf-8")

    response = client.post(
        "/v2/config",
        data={"config": '{"battery": "nope"}', "csrf_token": token},
        headers=INGRESS,
        environ_base=SUPERVISOR,
    )

    assert response.status_code == 200
    assert (site / "data" / "options.json").read_text(encoding="utf-8") == before


def test_unknown_api_task_is_not_echoed(client):
    response = client.get(
        "/api/run/<script>alert(1)</script>", headers=INGRESS, environ_base=SUPERVISOR
    )
    assert response.status_code == 404
    assert b"<script>alert" not in response.data


def test_post_without_csrf_token_is_refused(client):
    response = client.post(
        "/v2/fast-control/mode", data={"mode": "shadow"},
        headers=INGRESS, environ_base=SUPERVISOR,
    )
    assert response.status_code == 400


def test_mode_switch_edits_only_the_mode_key(client, site):
    options = site / "data" / "options.json"
    # Fetch the token first, then read the baseline. Rendering the page loads
    # the configuration, and loading an unversioned options.json migrates it
    # and stamps config_version -- a legitimate one-off write that would
    # otherwise show up here as a difference the mode switch did not cause.
    token = _csrf_token(client, "/v2/fast-control")
    before = json.loads(options.read_text(encoding="utf-8"))
    key = "fast control" if "fast control" in before else "fast_control"

    response = client.post(
        "/v2/fast-control/mode", data={"mode": "active", "csrf_token": token},
        headers=INGRESS, environ_base=SUPERVISOR,
    )

    assert response.status_code == 302
    after = json.loads(options.read_text(encoding="utf-8"))
    assert after[key]["mode"] == "active"
    assert {k: v for k, v in before.items() if k != key} == {
        k: v for k, v in after.items() if k != key
    }


def test_mode_switch_rejects_unknown_mode(client, site):
    token = _csrf_token(client, "/v2/fast-control")
    response = client.post(
        "/v2/fast-control/mode", data={"mode": "turbo", "csrf_token": token},
        headers=INGRESS, environ_base=SUPERVISOR,
    )
    assert response.status_code == 400


def test_switching_the_mode_off_is_written_through(client, site):
    """There used to be two mode switches, one per interface, and the test
    above only covered one of them. There is one now."""
    options = site / "data" / "options.json"
    token = _csrf_token(client, "/v2/fast-control")
    key = (
        "fast control"
        if "fast control" in json.loads(options.read_text())
        else "fast_control"
    )

    response = client.post(
        "/v2/fast-control/mode", data={"mode": "off", "csrf_token": token},
        headers=INGRESS, environ_base=SUPERVISOR,
    )

    assert response.status_code == 302
    assert json.loads(options.read_text(encoding="utf-8"))[key]["mode"] == "off"
