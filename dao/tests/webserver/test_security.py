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

import hashlib
import importlib
import json
import socket
import subprocess
import threading
from html.parser import HTMLParser
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_wtf")

from flask import got_request_exception
from markupsafe import escape

from dao.prog.config.loader import CURRENT_VERSION
from dao.prog.da_base import DaBase
from dao.tests.config.test_atomic_config import (
    _managed_process,
    _message,
    _pause_replace,
    _paused_config_writer,
    _probe_transaction,
    _waiting_config_writer,
)

from .conftest import INGRESS, SUPERVISOR, csrf_token as _csrf_token  # noqa: E402


@pytest.fixture
def editor_sandbox(client, tmp_path, monkeypatch):
    """Real editors and cache, restricted to synthetic files and no backend I/O."""
    routes = importlib.import_module("app.v2.routes")
    paths = {"config": tmp_path / "options.json", "secrets": tmp_path / "secrets.json"}
    paths["config"].write_text(
        json.dumps({"config_version": CURRENT_VERSION, "time_zone": "Europe/Amsterdam"}) + "\n",
        encoding="utf-8",
    )
    paths["secrets"].write_text('{"synthetic": "stored-test-value"}\n', encoding="utf-8")
    monkeypatch.setattr(routes, "app_datapath", str(tmp_path) + "/")
    monkeypatch.setattr(routes, "_config_cache", None)
    monkeypatch.setattr(routes, "_config_cache_loaded", False)

    def forbidden(*args, **kwargs):
        pytest.fail("Editor tests must not start backend I/O or workers")

    monkeypatch.setattr(DaBase, "__init__", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)
    return routes, paths


def _assert_editor_serialization(response, editor, submitted):
    """Assert the serialization binding, not HTML5 parsing or a DOM value."""
    opening = {
        "config": '<textarea name="config" data-code-input-fallback>',
        "secrets": '<textarea data-code-input-fallback name="secrets">',
    }[editor]
    expected = opening + "\n" + str(escape(submitted)) + "</textarea>"
    preserved = expected in response.get_data(as_text=True)
    # A boolean assertion keeps synthetic secret content out of failure logs.
    assert preserved, "The serialized textarea must bind one fixed LF and unchanged escaped input"


def _editor_fields(response):
    """Parse hidden fields only; textarea evidence remains the serialization guard."""
    fields = {}

    class HiddenFields(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "input" and attrs.get("name") in ("csrf_token", "revision"):
                fields[attrs["name"]] = attrs.get("value", "")

    HiddenFields().feed(response.get_data(as_text=True))
    assert fields.get("csrf_token")
    assert len(fields.get("revision", "")) == 64, "Editor GET must provide a byte revision"
    return fields


def _editor_form(client, editor):
    """Use the real GET's CSRF and revision fields, without bypassing either gate."""
    response = client.get(f"/v2/{editor}", headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 200
    return _editor_fields(response)


@pytest.mark.parametrize("editor", ["config", "secrets"])
def test_editor_get_revision_hashes_exact_saved_bytes(client, editor_sandbox, editor):
    _, paths = editor_sandbox
    saved = '\r\n{ "//note": "café" }\r\n'.encode()
    paths[editor].write_bytes(saved)

    form = _editor_form(client, editor)

    assert form["revision"] == hashlib.sha256(saved).hexdigest()
    response = client.get(f"/v2/{editor}", headers=INGRESS, environ_base=SUPERVISOR)
    _assert_editor_serialization(response, editor, saved.decode("utf-8"))


@pytest.mark.parametrize("editor", ["config", "secrets"])
@pytest.mark.parametrize("revision", [None, "", "invalid", "0" * 64])
def test_editor_requires_matching_revision(
    client, editor_sandbox, caplog, editor, revision
):
    routes, paths = editor_sandbox
    token = _csrf_token(client, f"/v2/{editor}")
    cached = routes._cached_config()
    before = {name: path.read_bytes() for name, path in paths.items()}
    submitted = '\n {"//note": "</textarea><script>b8c_sandbox</script>& café"}\t\n'
    form = {editor: submitted, "csrf_token": token}
    if revision is not None:
        form["revision"] = revision
    caplog.clear()

    response = client.post(
        f"/v2/{editor}", data=form, headers=INGRESS, environ_base=SUPERVISOR,
    )

    assert response.status_code == 409
    assert b"Error:" in response.data
    assert b"updated successfully" not in response.data
    unchanged = all(path.read_bytes() == before[name] for name, path in paths.items())
    assert unchanged, "A missing or stale revision must not change either file"
    assert routes._cached_config() is cached
    assert "b8c_sandbox" not in caplog.text, "Conflict input must not be logged"
    _assert_editor_serialization(response, editor, submitted)


@pytest.mark.parametrize("editor", ["config", "secrets"])
def test_stale_editor_form_cannot_overwrite_newer_editor_save(
    client, editor_sandbox, editor
):
    routes, paths = editor_sandbox
    stale_form = _editor_form(client, editor)
    newer = '\n {"config_version": 3, "//note": "newer sandbox save"}\r\n'
    response = client.post(
        f"/v2/{editor}", data={**stale_form, editor: newer},
        headers=INGRESS, environ_base=SUPERVISOR,
    )
    assert response.status_code == 200
    assert b"updated successfully" in response.data
    successful_form = _editor_fields(response)
    assert successful_form["revision"] == hashlib.sha256(newer.encode("utf-8")).hexdigest()
    cached = routes._cached_config()
    updated_form = _editor_form(client, editor)
    assert updated_form["revision"] == hashlib.sha256(newer.encode("utf-8")).hexdigest()
    assert updated_form["revision"] != stale_form["revision"]
    stale_input = '\n {"//note": "stale </textarea>& café"}\t\n'

    response = client.post(
        f"/v2/{editor}", data={**stale_form, editor: stale_input},
        headers=INGRESS, environ_base=SUPERVISOR,
    )

    assert response.status_code == 409
    assert _editor_fields(response)["revision"] == stale_form["revision"]
    unchanged = paths[editor].read_bytes() == newer.encode("utf-8")
    assert unchanged, "A stale editor must preserve the newer saved bytes"
    assert routes._cached_config() is cached
    _assert_editor_serialization(response, editor, stale_input)


def test_mode_change_and_editor_conflict_preserves_newer_data(client, editor_sandbox):
    routes, paths = editor_sandbox
    path = paths["config"]
    form = {
        "csrf_token": _csrf_token(client, "/v2/config"),
        "revision": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    cached = routes._cached_config()
    submitted = '\n {"//note": "stale </textarea>& café"}\t\n'

    def post(connection):
        _probe_transaction(connection, path)
        response = client.post(
            "/v2/config", data={**form, "config": submitted},
            headers=INGRESS, environ_base=SUPERVISOR,
        )
        connection.send(("response", response.status_code))
        assert b"updated successfully" not in response.data
        assert routes._cached_config() is cached
        _assert_editor_serialization(response, "config", submitted)

    with _managed_process(_paused_config_writer, path, "mode") as mode:
        _message(mode, "before_replace")
        mode.send("publish")
        (published_inode,) = _message(mode, "after_replace")
        newer = path.read_bytes()
        with _managed_process(post) as editor:
            (lock_inode,) = _message(editor, "blocked")
            assert lock_inode != published_inode
            mode.send("unlock")
            _message(mode, "done")
            _message(editor, "read")
            assert _message(editor, "response") == (409,)
            _message(editor, "done")

    assert path.read_bytes() == newer
    assert json.loads(newer)["fast control"]["mode"] == "active"


def test_editor_holds_transaction_through_publication(client, editor_sandbox):
    _, paths = editor_sandbox
    path = paths["config"]
    form = _editor_form(client, "config")
    submitted = '{"config_version": 3, "//note": "keep editor change"}\n'

    def post(connection):
        _pause_replace(connection)
        response = client.post(
            "/v2/config", data={**form, "config": submitted},
            headers=INGRESS, environ_base=SUPERVISOR,
        )
        connection.send(("response", response.status_code))
        assert b"Config updated successfully" in response.data

    with _managed_process(post) as editor:
        _message(editor, "before_replace")
        with _managed_process(_waiting_config_writer, path, "mode") as mode:
            _message(mode, "blocked")
            editor.send("publish")
            _message(editor, "after_replace")
            assert path.read_text(encoding="utf-8") == submitted
            editor.send("unlock")
            assert _message(editor, "response") == (200,)
            _message(editor, "done")
            _message(mode, "read")
            _message(mode, "done")

    saved = json.loads(path.read_bytes())
    assert saved["//note"] == "keep editor change"
    assert saved["fast control"]["mode"] == "active"


@pytest.mark.parametrize("editor", ["config", "secrets"])
@pytest.mark.parametrize("failure", ["syntax", "schema", "empty", "whitespace", "io"])
def test_editor_preserves_unsaved_input(
    client, editor_sandbox, monkeypatch, caplog, editor, failure
):
    """A rejected POST must not replace even empty input with saved content."""
    routes, paths = editor_sandbox
    form = _editor_form(client, editor)
    cached = routes._cached_config()
    assert cached is not None
    before = {name: path.read_bytes() for name, path in paths.items()}
    payload = "</textarea><script>window.b8b_sandbox = true</script>& café"
    submitted = {
        "syntax": '\n\n\t{"synthetic": "' + payload + '"} broken  \n',
        "schema": (
            '\n {"solar": 1, "//note": "' + payload + '"}\t\n'
            if editor == "config" else '\n [{"synthetic": "' + payload + '"}]\t\n'
        ),
        "empty": "",
        "whitespace": "\n \t\r\n",
        "io": (
            '\r\n{ "config_version": ' + str(CURRENT_VERSION)
            + ', "//note" : "' + payload + '" }\t \n'
        ),
    }[failure]
    writes = []

    def fail_write(path, text):
        writes.append(path)
        exact_write = text == submitted
        assert exact_write, "The writer must receive the raw submitted text"
        raise OSError("Synthetic sandbox write failure")

    if failure == "io":
        monkeypatch.setattr(routes, "atomic_write_text", fail_write)

    exception_signals = []

    def record_exception(sender, **kwargs):
        exception_signals.append(True)

    caplog.clear()
    with got_request_exception.connected_to(record_exception, client.application):
        response = client.post(
            f"/v2/{editor}", data={**form, editor: submitted},
            headers=INGRESS, environ_base=SUPERVISOR,
        )

    assert response.status_code == 200
    assert b"Error:" in response.data
    assert b"alert-success" not in response.data
    assert b"updated successfully" not in response.data
    unchanged = all(path.read_bytes() == before[name] for name, path in paths.items())
    assert unchanged, "Rejected editor input must leave both files byte-for-byte unchanged"
    assert writes == ([paths[editor]] if failure == "io" else [])
    assert routes._config_cache_loaded
    assert routes._cached_config() is cached
    leaked = "b8b_sandbox" in caplog.text
    assert not leaked, "Editor input must not be logged"
    assert not exception_signals, "Handled editor errors must not reach error telemetry"
    assert b"</textarea><script>" not in response.data
    _assert_editor_serialization(response, editor, submitted)


@pytest.mark.parametrize("editor", ["config", "secrets"])
def test_editor_post_checks_revision_without_replacing_input(
    client, editor_sandbox, monkeypatch, editor
):
    routes, paths = editor_sandbox
    form = _editor_form(client, editor)
    cached = routes._cached_config()
    submitted = '\n {"config_version": ' + str(CURRENT_VERSION) + ', "//note": "saved"}\t\n'

    original_read_bytes = Path.read_bytes
    reads = []

    def read_bytes(path):
        if path == paths[editor]:
            reads.append(path)
        return original_read_bytes(path)

    with monkeypatch.context() as post_only:
        post_only.setattr(Path, "read_bytes", read_bytes)
        response = client.post(
            f"/v2/{editor}", data={**form, editor: submitted},
            headers=INGRESS, environ_base=SUPERVISOR,
        )

    assert response.status_code == 200
    assert b"updated successfully" in response.data
    assert reads == [paths[editor]], "POST reads disk only to check the revision, never as editor input"
    assert paths[editor].read_bytes() == submitted.encode("utf-8")
    _assert_editor_serialization(response, editor, submitted)
    if editor == "config":
        assert not routes._config_cache_loaded
    else:
        assert routes._cached_config() is cached

    # A subsequent GET uses disk, not the preceding POST's text.
    changed = '{"//note": "externally changed sandbox file"}\n'
    paths[editor].write_text(changed, encoding="utf-8")
    response = client.get(f"/v2/{editor}", headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 200
    _assert_editor_serialization(response, editor, changed)


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
    form = _editor_form(client, "config")
    try:
        response = client.post(
            "/v2/config",
            data={**form, "config": '{"nonsense": true}'},
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
    form = _editor_form(client, "config")
    before = (site / "data" / "options.json").read_text(encoding="utf-8")

    response = client.post(
        "/v2/config",
        data={**form, "config": '{"battery": "nope"}'},
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
