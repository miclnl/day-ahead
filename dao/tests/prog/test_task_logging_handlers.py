"""run_task_function must leave the root logger exactly as it found it.

The old version got this wrong in three ways, all of which these tests pin
down:

* the cleanup sat after a ``try`` that re-raised, so a failing task -- when
  you most want the log -- never reached it;
* ``removeHandler`` was never called, only ``close()``, and only for two of
  the three handlers. A closed FileHandler that is still attached is worse
  than one left open: ``FileHandler.emit`` reopens the file on the next
  record, so the task's log file was reopened and appended to after the task
  had finished (main() logs the pool status after the call returns);
* the NotificationHandler was added outside the ``if logfile:`` block and
  removed nowhere, so a second call in one process pushed every warning to
  Home Assistant twice.
"""

import logging

import pytest

from dao.prog.da_base import DaBase, NotificationHandler


class FakeDb:
    def log_pool_status(self):
        pass


def make_instance(tmp_path, monkeypatch, notification_entity=None, task="meteo"):
    """A DaBase with just enough wired up for run_task_function.

    __init__ needs a live configuration, database and Home Assistant, so it
    is bypassed and only what this method touches is set.
    """
    instance = DaBase.__new__(DaBase)
    instance.tasks = DaBase.generate_tasks()
    instance.log_level = logging.INFO
    instance.config = None
    instance.ha_context = None
    instance.db_da = FakeDb()
    instance.notification_entity = notification_entity
    instance.file_name = None
    monkeypatch.setattr(instance, "set_last_activity", lambda: None)

    # The log directory is addressed as ../data/log relative to the working
    # directory, so run from a scratch directory with that layout.
    (tmp_path / "data" / "log").mkdir(parents=True)
    (tmp_path / "prog").mkdir()
    monkeypatch.chdir(tmp_path / "prog")
    return instance


@pytest.fixture
def clean_root_logger():
    """Restore the root logger around each test.

    A leaked handler here would bleed into every later test in the session
    (duplicate records, a closed file handler being written to), which is
    the very failure mode under test.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield root
    for handler in root.handlers[:]:
        if handler not in saved_handlers:
            root.removeHandler(handler)
    for handler in saved_handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(saved_level)


class TestSuccessPath:
    def test_the_root_logger_is_restored_afterwards(
        self, tmp_path, monkeypatch, clean_root_logger
    ):
        instance = make_instance(tmp_path, monkeypatch)
        monkeypatch.setattr(instance, "get_meteo_data", lambda: None, raising=False)
        before = clean_root_logger.handlers[:]

        instance.run_task_function("meteo")

        assert clean_root_logger.handlers == before

    def test_the_task_log_file_is_written(
        self, tmp_path, monkeypatch, clean_root_logger
    ):
        instance = make_instance(tmp_path, monkeypatch)

        def task():
            logging.info("iets bijzonders")

        monkeypatch.setattr(instance, "get_meteo_data", task, raising=False)

        instance.run_task_function("meteo")

        logs = list((tmp_path / "data" / "log").glob("meteo_*.log"))
        assert len(logs) == 1
        assert "iets bijzonders" in logs[0].read_text()

    def test_nothing_is_appended_after_the_task_finished(
        self, tmp_path, monkeypatch, clean_root_logger
    ):
        """The closed-but-attached FileHandler used to reopen its file on the
        next record, so logging after the call landed in the finished task's
        log. main() does exactly that (db_da.log_pool_status())."""
        instance = make_instance(tmp_path, monkeypatch)
        monkeypatch.setattr(instance, "get_meteo_data", lambda: None, raising=False)

        instance.run_task_function("meteo")

        logfile = next((tmp_path / "data" / "log").glob("meteo_*.log"))
        size_before = logfile.stat().st_size
        logging.warning("na de taak, hoort hier niet in")

        assert logfile.stat().st_size == size_before
        assert "hoort hier niet in" not in logfile.read_text()


class TestFailurePath:
    def test_handlers_are_removed_even_when_the_task_raises(
        self, tmp_path, monkeypatch, clean_root_logger
    ):
        instance = make_instance(tmp_path, monkeypatch)

        def boom():
            raise RuntimeError("taak mislukt")

        monkeypatch.setattr(instance, "get_meteo_data", boom, raising=False)
        before = clean_root_logger.handlers[:]

        with pytest.raises(RuntimeError):
            instance.run_task_function("meteo")

        assert clean_root_logger.handlers == before

    def test_the_traceback_still_reaches_the_task_log(
        self, tmp_path, monkeypatch, clean_root_logger
    ):
        instance = make_instance(tmp_path, monkeypatch)

        def boom():
            raise RuntimeError("taak mislukt")

        monkeypatch.setattr(instance, "get_meteo_data", boom, raising=False)

        with pytest.raises(RuntimeError):
            instance.run_task_function("meteo")

        logfile = next((tmp_path / "data" / "log").glob("meteo_*.log"))
        content = logfile.read_text()
        assert "fout-tracering" in content
        assert "taak mislukt" in content


class TestLoggerLevel:
    def test_the_level_is_raised_for_the_task_and_restored_after(
        self, tmp_path, monkeypatch, clean_root_logger
    ):
        """Handler levels alone do nothing if the logger filters the record
        first, so the method sets the logger level as well rather than
        depending on __init__ having done it."""
        clean_root_logger.setLevel(logging.CRITICAL)
        instance = make_instance(tmp_path, monkeypatch)
        seen = {}

        def task():
            seen["level"] = logging.getLogger().level

        monkeypatch.setattr(instance, "get_meteo_data", task, raising=False)

        instance.run_task_function("meteo")

        assert seen["level"] == logging.INFO
        assert clean_root_logger.level == logging.CRITICAL

    def test_the_level_is_restored_when_the_task_raises(
        self, tmp_path, monkeypatch, clean_root_logger
    ):
        clean_root_logger.setLevel(logging.CRITICAL)
        instance = make_instance(tmp_path, monkeypatch)

        def boom():
            raise RuntimeError("taak mislukt")

        monkeypatch.setattr(instance, "get_meteo_data", boom, raising=False)

        with pytest.raises(RuntimeError):
            instance.run_task_function("meteo")

        assert clean_root_logger.level == logging.CRITICAL


class TestNotificationHandler:
    def test_it_is_removed_again(self, tmp_path, monkeypatch, clean_root_logger):
        instance = make_instance(
            tmp_path, monkeypatch, notification_entity="input_text.dao"
        )
        monkeypatch.setattr(instance, "get_meteo_data", lambda: None, raising=False)
        monkeypatch.setattr(instance, "_raw_set_value", lambda *a: None)

        instance.run_task_function("meteo")

        assert not [
            h for h in clean_root_logger.handlers if isinstance(h, NotificationHandler)
        ]

    def test_two_runs_do_not_leave_two_attached(
        self, tmp_path, monkeypatch, clean_root_logger
    ):
        """Two calls in one process used to leave two NotificationHandlers on
        the root logger, so every warning was pushed to Home Assistant
        twice."""
        instance = make_instance(
            tmp_path, monkeypatch, notification_entity="input_text.dao"
        )
        monkeypatch.setattr(instance, "get_meteo_data", lambda: None, raising=False)
        monkeypatch.setattr(instance, "_raw_set_value", lambda *a: None)

        instance.run_task_function("meteo")
        instance.run_task_function("meteo")

        assert not [
            h for h in clean_root_logger.handlers if isinstance(h, NotificationHandler)
        ]

    def test_it_is_removed_even_when_the_task_raises(
        self, tmp_path, monkeypatch, clean_root_logger
    ):
        instance = make_instance(
            tmp_path, monkeypatch, notification_entity="input_text.dao"
        )

        def boom():
            raise RuntimeError("taak mislukt")

        monkeypatch.setattr(instance, "get_meteo_data", boom, raising=False)
        monkeypatch.setattr(instance, "_raw_set_value", lambda *a: None)

        with pytest.raises(RuntimeError):
            instance.run_task_function("meteo")

        assert not [
            h for h in clean_root_logger.handlers if isinstance(h, NotificationHandler)
        ]


class TestGuards:
    def test_an_unknown_task_is_refused(self, tmp_path, monkeypatch, clean_root_logger):
        instance = make_instance(tmp_path, monkeypatch)
        before = clean_root_logger.handlers[:]

        instance.run_task_function("no_such_task")

        assert clean_root_logger.handlers == before

    def test_a_subprocess_only_task_is_refused_with_the_command(
        self, tmp_path, monkeypatch, clean_root_logger, caplog
    ):
        """fast_once has no DaBase method; it only ever ran as a subprocess.
        Calling it here used to raise AttributeError from getattr."""
        instance = make_instance(tmp_path, monkeypatch)

        instance.run_task_function("fast_once")

        assert "da_fast.py" in caplog.text
