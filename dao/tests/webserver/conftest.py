"""Shared fixtures for the webserver tests.

The app reads ../data relative to its working directory, so the fixture
builds a small site directory (data copied from the shipped example, app
package linked) and imports the app from there.
"""

import os
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
