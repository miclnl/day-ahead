"""The Logs page is a second reader of the log directory.

The console reader was confined in fix round 1: it opens the file inside
the log directory itself, without following a link, and accepts only a
regular file. The Logs page is a second reader of that same directory,
and it was still listing every name that matches ``*.log`` and opening it
with a plain ``open()``. A locally placed symlink was therefore served
as a log file, and a directory with a matching name was opened as one.

These tests keep the page working -- ordinary logs are still shown, the
timestamp selection and the file list still work -- and keep the
confinement: a link and a non-regular file are not served.

This is local log confinement: it says nothing about the web server as a
whole and nothing about any device.
"""

import os
import shutil
from pathlib import Path

import pytest

from .conftest import INGRESS, SUPERVISOR

#: The output of the regular log; the page has to show it.
OUTPUT = "the output of the task"

#: The content of the file a symlink points at; the page may not show it.
OUTSIDE = "not the output of the task"


@pytest.fixture(scope="module")
def v2_routes(client):
    """The routes module of the running app, not a second copy of it."""
    import importlib

    return importlib.import_module("app.v2.routes")


@pytest.fixture
def log_dir(site):
    """The log directory of this site."""
    return Path(site) / "data" / "log"


@pytest.fixture(autouse=True)
def empty_log_dir(log_dir):
    """Every test starts with an empty log directory and leaves it that way.

    The site is shared by the whole module, so a log written by one test
    would otherwise show up in the next one.
    """
    before = set(os.listdir(log_dir))
    yield log_dir
    for name in set(os.listdir(log_dir)) - before:
        path = log_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _log_page(client, query: str = "") -> str:
    """The Logs page, as the browser receives it.

    A reader that opens a directory or follows a link raises out of the
    request; that is turned into a plain failure, so the witness of the
    bug stays readable.
    """
    try:
        response = client.get(
            f"/v2/log{query}", headers=INGRESS, environ_base=SUPERVISOR
        )
    except OSError as exception:
        pytest.fail(f"the log reader opened a file it should refuse: {exception}")
    assert response.status_code == 200
    return response.data.decode()


class TestLogPageConfinement:
    """The Logs page serves the log directory, and only plain files in it.

    A name in the log directory may be a link to somewhere else, or a
    directory or a fifo. None of those is a log file of the add-on, and
    none of them is served.
    """

    def test_a_symlink_is_not_served_as_a_log_file(self, client, log_dir, site):
        """A symlink with a matching name is not offered as a log file.

        The link is newer than the ordinary log, so an unconfined reader
        serves the target of the link instead of the log.
        """
        outside = Path(site) / "outside.log"
        outside.write_text(OUTSIDE, encoding="utf-8")
        link = log_dir / "meteo_2026-09-29__12:00:00.log"
        link.symlink_to(outside)
        (log_dir / "meteo_2026-09-29__11:00:00.log").write_text(
            OUTPUT, encoding="utf-8"
        )

        page = _log_page(client)

        assert OUTSIDE not in page, "the page served the target of the symlink"
        assert link.name not in page, "the symlink was offered as a log file"
        assert OUTPUT in page, "the ordinary log was not shown"

    def test_a_symlink_alone_leaves_the_page_empty(self, client, log_dir, site):
        """A log directory that holds nothing but a link shows no log at
        all, rather than the target of that link."""
        outside = Path(site) / "outside.log"
        outside.write_text(OUTSIDE, encoding="utf-8")
        (log_dir / "meteo_2026-09-29__12:00:00.log").symlink_to(outside)

        page = _log_page(client)

        assert OUTSIDE not in page
        assert "no data to show" in page

    def test_a_directory_with_a_log_name_is_not_served(self, client, log_dir):
        """A directory whose name matches the log pattern is not a log
        file, and is not opened as one."""
        (log_dir / "meteo_2026-09-29__12:00:00.log").mkdir()
        (log_dir / "meteo_2026-09-29__11:00:00.log").write_text(
            OUTPUT, encoding="utf-8"
        )

        page = _log_page(client)

        assert "meteo_2026-09-29__12:00:00.log" not in page
        assert OUTPUT in page

    def test_the_file_list_holds_no_link_and_no_directory(self, v2_routes, log_dir):
        """The listing the page is built from holds plain files only."""
        outside = Path(log_dir).parent / "outside.log"
        outside.write_text(OUTSIDE, encoding="utf-8")
        (log_dir / "meteo_2026-09-29__12:00:00.log").symlink_to(outside)
        (log_dir / "meteo_2026-09-29__11:00:00.log").write_text(
            OUTPUT, encoding="utf-8"
        )
        (log_dir / "meteo_2026-09-29__10:00:00.log").mkdir()

        names = [
            entry["name"]
            for entry in v2_routes.get_file_list_with_ts(str(log_dir), "*.log")
        ]

        assert names == ["meteo_2026-09-29__11:00:00.log"]

    def test_a_link_that_appears_after_the_selection_is_not_read(
        self, client, v2_routes, log_dir, site, monkeypatch
    ):
        """The listing is taken first and the file is replaced by a link
        afterwards: the reader itself refuses to follow it."""
        outside = Path(site) / "outside.log"
        outside.write_text(OUTSIDE, encoding="utf-8")
        link = log_dir / "meteo_2026-09-29__12:00:00.log"
        link.symlink_to(outside)

        def listing_with_the_link(path, pattern):
            return [{"name": link.name, "time": 0.0}]

        monkeypatch.setattr(
            v2_routes, "get_file_list_with_ts", listing_with_the_link
        )

        page = _log_page(client)

        assert OUTSIDE not in page, "the reader followed the link"
        assert "Logbestand kon niet worden gelezen." in page

    def test_an_ordinary_log_is_still_served(self, client, log_dir):
        """The confinement may not take the log away from the reader."""
        (log_dir / "meteo_2026-09-29__12:00:00.log").write_text(
            OUTPUT, encoding="utf-8"
        )

        page = _log_page(client)

        assert OUTPUT in page
        assert "meteo_2026-09-29__12:00:00.log" in page

    def test_the_timestamp_selection_still_works(self, client, log_dir):
        """The page still shows the log of the chosen moment."""
        (log_dir / "meteo_2026-09-29__11:00:00.log").write_text(
            "the earlier run", encoding="utf-8"
        )
        (log_dir / "meteo_2026-09-29__12:00:00.log").write_text(
            "the later run", encoding="utf-8"
        )

        page = _log_page(client, "?ts=2026-09-29T11:00:00")

        assert "the earlier run" in page
        assert "the later run" not in page

    def test_the_file_list_navigation_still_works(self, client, log_dir):
        """The page still walks through the logs it found."""
        (log_dir / "meteo_2026-09-29__11:00:00.log").write_text(
            "the earlier run", encoding="utf-8"
        )
        (log_dir / "meteo_2026-09-29__12:00:00.log").write_text(
            "the later run", encoding="utf-8"
        )

        page = _log_page(client, "?i=0")

        assert "the earlier run" in page
        assert "the later run" not in page
