"""The task scheduler: cron translation and exclusive task execution.

DaScheduler needs a live configuration, database and Home Assistant to be
constructed, so these tests build the instance without __init__ and exercise
the parts that hold the scheduling logic.
"""

import datetime
import threading
import time
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("apscheduler")

from dao.prog import da_scheduler  # noqa: E402
from dao.prog.da_scheduler import DaScheduler, cron_trigger  # noqa: E402

TZ = ZoneInfo("Europe/Amsterdam")


def _next(trigger, after):
    return trigger.get_next_fire_time(None, after)


def test_fixed_time_pattern():
    trigger = cron_trigger("0544", TZ)
    after = datetime.datetime(2026, 6, 15, 12, 0, tzinfo=TZ)
    assert _next(trigger, after) == datetime.datetime(2026, 6, 16, 5, 44, tzinfo=TZ)


def test_every_hour_pattern():
    trigger = cron_trigger("xx15", TZ)
    after = datetime.datetime(2026, 6, 15, 12, 16, tzinfo=TZ)
    assert _next(trigger, after) == datetime.datetime(2026, 6, 15, 13, 15, tzinfo=TZ)


def test_every_minute_of_one_hour_pattern():
    trigger = cron_trigger("02xx", TZ)
    after = datetime.datetime(2026, 6, 15, 2, 30, 30, tzinfo=TZ)
    assert _next(trigger, after) == datetime.datetime(2026, 6, 15, 2, 31, tzinfo=TZ)
    after = datetime.datetime(2026, 6, 15, 3, 0, tzinfo=TZ)
    assert _next(trigger, after) == datetime.datetime(2026, 6, 16, 2, 0, tzinfo=TZ)


def _bare_scheduler(monkeypatch, schedule, active=True):
    instance = DaScheduler.__new__(DaScheduler)
    instance.tasks = DaScheduler.generate_tasks()
    instance.active = active
    instance.schedule = schedule
    instance.time_zone = "Europe/Amsterdam"
    instance.fast_control = None
    instance.scheduler = None
    instance._task_locks = {}
    instance._locks_guard = threading.Lock()
    return instance


class Entry:
    def __init__(self, time, action):
        self.time = time
        self.action = action


def test_every_entry_becomes_a_job_including_duplicates(monkeypatch):
    instance = _bare_scheduler(
        monkeypatch,
        [Entry("xx00", "calc_optimum"), Entry("xx00", "get_meteo_data"), Entry("0544", "get_meteo_data")],
    )
    scheduler = instance.build_scheduler()
    names = sorted(job.name for job in scheduler.get_jobs())
    assert names == ["0544 get_meteo_data", "xx00 calc_optimum", "xx00 get_meteo_data"]
    # Job defaults are applied when the scheduler starts; check the configured defaults.
    assert scheduler._job_defaults["max_instances"] == 1
    assert scheduler._job_defaults["misfire_grace_time"] == da_scheduler.MISFIRE_GRACE_S
    assert scheduler._job_defaults["coalesce"] is True


def test_unknown_actions_are_skipped_and_inactive_schedules_are_empty(monkeypatch, caplog):
    instance = _bare_scheduler(monkeypatch, [Entry("xx00", "no_such_task")])
    assert instance.build_scheduler().get_jobs() == []
    assert "no_such_task" in caplog.text

    instance = _bare_scheduler(monkeypatch, [Entry("xx00", "calc_optimum")], active=False)
    assert instance.build_scheduler().get_jobs() == []


def test_the_same_task_never_overlaps_itself(monkeypatch, caplog):
    instance = _bare_scheduler(monkeypatch, [])
    running = threading.Event()
    release = threading.Event()
    runs = []

    def slow_task(key_task):
        runs.append(key_task)
        if key_task == "calc_optimum" and not release.is_set():
            running.set()
            release.wait(5)
        return True

    monkeypatch.setattr(instance, "run_task_process", slow_task)

    first = threading.Thread(target=instance._run_exclusive, args=["calc_optimum"])
    first.start()
    assert running.wait(5)
    # A second start of the same task while the first runs is skipped ...
    instance._run_exclusive("calc_optimum")
    assert runs == ["calc_optimum"]
    assert "overgeslagen" in caplog.text
    # ... but a different task runs alongside it.
    instance._run_exclusive("meteo")
    assert runs == ["calc_optimum", "meteo"]
    release.set()
    first.join(5)
    # Once finished, the task can run again.
    instance._run_exclusive("calc_optimum")
    assert runs[-1] == "calc_optimum"


def test_a_failing_task_does_not_propagate(monkeypatch, caplog):
    instance = _bare_scheduler(monkeypatch, [])

    def boom(key_task):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(instance, "run_task_process", boom)
    instance._run_exclusive("calc_optimum")
    assert "kaboom" in caplog.text
    # The lock is released again after the failure.
    assert not instance._task_locks["calc_optimum"].locked()
