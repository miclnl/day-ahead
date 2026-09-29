"""The task scheduler: cron translation and exclusive task execution.

DaScheduler needs a live configuration, database and Home Assistant to be
constructed, so these tests build the instance without __init__ and exercise
the parts that hold the scheduling logic.
"""

import datetime
import threading
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("apscheduler")

from dao.prog import da_scheduler, task_state  # noqa: E402
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
    instance._last_alive_note = 0.0
    return instance


def cron_jobs(scheduler):
    """The scheduled tasks, without the request-pickup job.

    The pickup job is added for every scheduler, including an inactive one,
    because scheduler.active = false means "run no cron schedule", not
    "ignore the dashboard".
    """
    return [job for job in scheduler.get_jobs() if job.id != "pick-up-task-requests"]


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
    names = sorted(job.name for job in cron_jobs(scheduler))
    assert names == ["0544 get_meteo_data", "xx00 calc_optimum", "xx00 get_meteo_data"]
    # Job defaults are applied when the scheduler starts; check the configured defaults.
    assert scheduler._job_defaults["max_instances"] == 1
    assert scheduler._job_defaults["misfire_grace_time"] == da_scheduler.MISFIRE_GRACE_S
    assert scheduler._job_defaults["coalesce"] is True


def test_unknown_actions_are_skipped_and_inactive_schedules_are_empty(monkeypatch, caplog):
    instance = _bare_scheduler(monkeypatch, [Entry("xx00", "no_such_task")])
    assert cron_jobs(instance.build_scheduler()) == []
    assert "no_such_task" in caplog.text

    instance = _bare_scheduler(monkeypatch, [Entry("xx00", "calc_optimum")], active=False)
    assert cron_jobs(instance.build_scheduler()) == []


def test_the_same_task_never_overlaps_itself(monkeypatch, caplog):
    instance = _bare_scheduler(monkeypatch, [])
    running = threading.Event()
    release = threading.Event()
    runs = []

    def slow_task(key_task, parameters=None):
        runs.append(key_task)
        if key_task == "calc_optimum" and not release.is_set():
            running.set()
            release.wait(5)
        return "done", 0, None

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

    def boom(key_task, parameters=None):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(instance, "run_task_process", boom)
    instance._run_exclusive("calc_optimum")
    assert "kaboom" in caplog.text
    # The claim is released again after the failure, and the outcome is
    # recorded so the dashboard can show that the run failed.
    assert task_state.is_running("calc_optimum") is False
    assert task_state.last_result("calc_optimum")["status"] == "error"


def test_the_scheduler_claim_is_visible_outside_this_process(monkeypatch):
    """The point of moving exclusion into task_state: the dashboard now sees
    a scheduled run. With the old in-process threading.Lock, cron starting
    calc_optimum at 05:44 and a user pressing the button at 05:44 produced
    two optimisation runs writing the same tables."""
    instance = _bare_scheduler(monkeypatch, [])
    running = threading.Event()
    release = threading.Event()

    def slow_task(key_task, parameters=None):
        running.set()
        release.wait(5)
        return "done", 0, None

    monkeypatch.setattr(instance, "run_task_process", slow_task)

    worker = threading.Thread(target=instance._run_exclusive, args=["calc_optimum"])
    worker.start()
    try:
        assert running.wait(5)
        # This is what a dashboard request does before starting a task.
        assert task_state.claim("calc_optimum", "dashboard") is False
        assert task_state.running_tasks()["calc_optimum"]["source"] == "scheduler"
    finally:
        release.set()
        worker.join(5)

    assert task_state.claim("calc_optimum", "dashboard") is True


def test_a_task_started_elsewhere_makes_the_scheduler_skip(monkeypatch, caplog):
    """The other direction: a task the user started from the dashboard must
    not be started a second time when its cron time comes around."""
    instance = _bare_scheduler(monkeypatch, [])
    runs = []
    monkeypatch.setattr(
        instance,
        "run_task_process",
        lambda key, parameters=None: (runs.append(key), ("done", 0, None))[1],
    )

    task_state.claim("calc_optimum", "dashboard")
    instance._run_exclusive("calc_optimum")

    assert runs == []
    assert "overgeslagen" in caplog.text
    assert "dashboard" in caplog.text
