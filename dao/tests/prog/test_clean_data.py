"""DaBase.clean_data(): pruning old logs and graphs.

clean_data used to chdir into each folder to list its files and chdir back
afterwards; an exception between those two calls would have left every other
CWD-relative path in the process (../data, ../prog, ...) resolving from the
wrong directory for the rest of the run. It now uses Path.glob() instead.
"""

import time
import types
from pathlib import Path

import pytest

from dao.prog import da_base
from dao.prog.da_base import DaBase


@pytest.fixture
def instance(tmp_path, monkeypatch):
    (tmp_path / "data" / "log").mkdir(parents=True)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "prog").mkdir()
    monkeypatch.chdir(tmp_path / "prog")
    obj = DaBase.__new__(DaBase)
    obj.history_options = types.SimpleNamespace(save_days=7)
    return obj, tmp_path / "data"


def _age(monkeypatch, days: float) -> None:
    """Make clean_data() see every existing file as *days* old.

    ctime cannot be backdated through os.utime (it reflects the last inode
    change, set by the OS), so the clock is moved forward instead: the
    method only ever computes time.time() - path.stat().st_ctime.
    """
    real_time = time.time()
    monkeypatch.setattr(da_base.time, "time", lambda: real_time + days * 86400)


def test_old_files_matching_the_pattern_are_removed(instance, monkeypatch):
    obj, data = instance
    old_log = data / "log" / "old.log"
    old_log.touch()
    _age(monkeypatch, 10)

    obj.clean_data()

    assert not old_log.exists()


def test_recent_files_are_kept(instance, monkeypatch):
    obj, data = instance
    recent_log = data / "log" / "recent.log"
    recent_log.touch()
    _age(monkeypatch, 1)

    obj.clean_data()

    assert recent_log.exists()


def test_files_that_do_not_match_the_pattern_are_left_alone(instance, monkeypatch):
    obj, data = instance
    other = data / "log" / "old.txt"
    other.touch()
    _age(monkeypatch, 10)

    obj.clean_data()

    assert other.exists()


def test_images_older_than_save_days_are_removed(instance, monkeypatch):
    obj, data = instance
    old_png = data / "images" / "calc_old.png"
    old_png.touch()
    _age(monkeypatch, 10)

    obj.clean_data()

    assert not old_png.exists()


def test_the_working_directory_is_never_changed(instance, monkeypatch):
    obj, data = instance
    import os

    before = os.getcwd()
    old_log = data / "log" / "old.log"
    old_log.touch()
    _age(monkeypatch, 10)

    obj.clean_data()

    assert os.getcwd() == before
