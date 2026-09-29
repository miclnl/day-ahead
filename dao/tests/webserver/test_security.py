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
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_wtf")

REPO = Path(__file__).resolve().parents[3]
WEBSERVER = REPO / "dao" / "webserver"
EXAMPLE = REPO / "dao" / "data" / "options_example.json"

SUPERVISOR = {"REMOTE_ADDR": "172.30.32.2"}
INGRESS = {"X-Ingress-Path": "/api/hassio_ingress/token"}


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """A data directory plus a working directory the app can run from."""
    root = tmp_path_factory.mktemp("site")
    data = root / "data"
    (data / "images").mkdir(parents=True)
    (data / "log").mkdir()
    shutil.copy(EXAMPLE, data / "options.json")
    (data / "secrets.json").write_text('{"db_password": "hunter2"}', encoding="utf-8")
    (data / "images" / "calc_2026-09-28__12-00.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    cwd = root / "webserver"
    cwd.mkdir()
    os.symlink(WEBSERVER / "app", cwd / "app")
    return root


@pytest.fixture(scope="module")
def client(site, tmp_path_factory):
    previous = os.getcwd()
    os.chdir(site / "webserver")
    os.environ["VITE_DEV"] = "1"  # v2 pages: skip the Vite manifest lookup
    os.environ.pop("DAO_ALLOW_DIRECT", None)
    sys.path.insert(0, str(site / "webserver"))
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]
    try:
        from app import app  # noqa: WPS433 - imported here so cwd is the site

        app.config["TESTING"] = True
        yield app.test_client()
    finally:
        os.chdir(previous)
        sys.path.remove(str(site / "webserver"))


def _csrf_token(client, path):
    response = client.get(path, headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 200
    match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    assert match, "page has no csrf token"
    return match.group(1).decode()


def test_direct_requests_are_refused(client):
    assert client.get("/").status_code == 401
    assert client.get("/", headers=INGRESS).status_code == 401
    assert client.get("/", environ_base=SUPERVISOR).status_code == 401


def test_ingress_requests_are_served(client):
    response = client.get("/", headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 200


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


def test_settings_editor_refuses_other_files(client, site):
    token = _csrf_token(client, "/settings/options")
    response = client.post(
        "/settings/options",
        data={"cur_setting": "../evil", "codeinput": "{}", "action": "update",
              "csrf_token": token},
        headers=INGRESS,
        environ_base=SUPERVISOR,
    )
    assert response.status_code == 400
    assert not (site / "evil.json").exists()


def test_settings_editor_rejects_invalid_config(client, site):
    before = (site / "data" / "options.json").read_text(encoding="utf-8")
    token = _csrf_token(client, "/settings/options")
    response = client.post(
        "/settings/options",
        data={"cur_setting": "options", "codeinput": '{"battery": "nope"}',
              "action": "update", "csrf_token": token},
        headers=INGRESS,
        environ_base=SUPERVISOR,
    )
    assert response.status_code == 200
    assert b"Error:" in response.data
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
    before = json.loads(options.read_text(encoding="utf-8"))
    key = "fast control" if "fast control" in before else "fast_control"
    token = _csrf_token(client, "/v2/fast-control")

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


def test_v1_mode_switch_uses_the_same_writer(client, site):
    options = site / "data" / "options.json"
    key = "fast control" if "fast control" in json.loads(options.read_text()) else "fast_control"
    token = _csrf_token(client, "/fast_control")
    response = client.post(
        "/fast_control", data={"mode": "off", "csrf_token": token},
        headers=INGRESS, environ_base=SUPERVISOR,
    )
    assert response.status_code == 200
    assert json.loads(options.read_text(encoding="utf-8"))[key]["mode"] == "off"
