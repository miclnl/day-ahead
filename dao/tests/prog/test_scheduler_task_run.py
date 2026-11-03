"""The scheduler runs every task, including the ones the dashboards ask for.

Tasks used to run inside a gunicorn worker, in a daemon thread. That worker
is recycled on every configuration change -- the watchdog sends gunicorn a
HUP -- which killed the thread while its subprocess carried on, so nothing
was left to notice the result and the task stayed registered as running
until its claim went stale ten minutes later. With two workers a request
could also land in either one.

The dashboards now record a request (a claim in the "pending" state) and
this process, which is long-lived and already owned task execution for the
cron schedule, picks it up.

These tests cover the run itself: process group isolation, cancellation,
claim release on every path, and discovering the log file the task writes.
"""

import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("apscheduler")

from dao.prog import da_scheduler, task_state
from dao.prog.da_scheduler import DaScheduler


class FakeProc:
    """Stands in for subprocess.Popen: runs for one poll, then exits."""

    def __init__(self, cmd=None, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode = 0
        self._killed = False
        self._polls = 0

    def poll(self):
        self._polls += 1
        if self._killed:
            return 0
        return None if self._polls == 1 else 0

    def wait(self):
        return self.returncode

    def kill(self):
        self._killed = True


@pytest.fixture
def instance(monkeypatch, tmp_path):
    obj = DaScheduler.__new__(DaScheduler)
    obj.tasks = DaScheduler.generate_tasks()
    obj.active = True
    obj.schedule = []
    obj.time_zone = "Europe/Amsterdam"
    obj.fast_control = None
    obj.scheduler = None
    obj._last_alive_note = 0.0
    # No real sleeping between polls, and a scratch log directory.
    monkeypatch.setattr(da_scheduler.time, "sleep", lambda seconds: None)
    log_dir = tmp_path / "data" / "log"
    log_dir.mkdir(parents=True)
    monkeypatch.setattr(
        type(obj), "log_dir", property(lambda self: str(log_dir))
    )
    return obj


class TestProcessIsolation:
    def test_the_task_gets_its_own_session(self, instance, monkeypatch):
        """Without this a cancel only reaches the direct child, so anything
        the task spawned is left running as an orphan."""
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured.update(kwargs)
            return FakeProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", fake_popen)
        claim = task_state.claim("meteo", "scheduler")

        instance.run_task_process(claim)

        assert captured.get("start_new_session") is True

    def test_the_working_directory_is_pinned_to_prog(self, instance, monkeypatch):
        """The tasks use CWD-relative paths (../data, ../prog) and misbehave
        silently when started from anywhere else."""
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured.update(kwargs)
            return FakeProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", fake_popen)
        claim = task_state.claim("meteo", "scheduler")

        instance.run_task_process(claim)

        assert captured["cwd"] == DaScheduler.PROG_DIR


class TestParameters:
    def test_they_are_appended_to_the_command(self, instance, monkeypatch):
        """The dashboard collects them, the scheduler builds the command."""
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", fake_popen)
        claim = task_state.claim("fast_control_simulate", "dashboard")

        instance.run_task_process(claim, {"days": "21"})

        assert captured["cmd"][-2:] == ["--days", "21"]


class TestOutcome:
    @pytest.mark.parametrize("raises", [False, True])
    def test_old_runner_cannot_update_or_release_a_replacement(self, instance, monkeypatch, raises):
        old = task_state.claim("meteo", "scheduler")
        replacement_bytes = []
        killed = []

        def replace_during_spawn(cmd, **kwargs):
            path = Path(task_state.STATE_PATH)
            state = json.loads(path.read_text())
            state["running"]["meteo"]["heartbeat"] = time.time() - task_state.STALE_AFTER_S - 1
            path.write_text(json.dumps(state))
            new = task_state.claim("meteo", "dashboard")
            task_state.request_cancel(new)
            replacement_bytes.append(path.read_bytes())
            if raises:
                raise OSError("synthetic spawn failure after replacement")
            return FakeProc()

        monkeypatch.setattr(da_scheduler, "Popen", replace_during_spawn)
        monkeypatch.setattr(da_scheduler.task_registry, "kill_process_group", killed.append)

        instance._run_claimed(old)

        assert replacement_bytes
        assert Path(task_state.STATE_PATH).read_bytes() == replacement_bytes[0]
        assert killed == []
        assert task_state.last_result("meteo") == {}

    def test_a_clean_run_reports_done(self, instance, monkeypatch):
        monkeypatch.setattr(da_scheduler, "Popen", lambda cmd, **kw: FakeProc())
        claim = task_state.claim("meteo", "scheduler")

        status, returncode, _logfile = instance.run_task_process(claim)

        assert (status, returncode) == ("done", 0)

    def test_a_non_zero_exit_reports_error(self, instance, monkeypatch):
        proc = FakeProc()
        proc.returncode = 2
        monkeypatch.setattr(da_scheduler, "Popen", lambda cmd, **kw: proc)
        claim = task_state.claim("meteo", "scheduler")

        status, returncode, _logfile = instance.run_task_process(claim)

        assert (status, returncode) == ("error", 2)

    def test_the_claim_is_released_on_success(self, instance, monkeypatch):
        monkeypatch.setattr(da_scheduler, "Popen", lambda cmd, **kw: FakeProc())
        claim = task_state.claim("meteo", "scheduler")

        instance._run_claimed(claim)

        assert task_state.is_running("meteo") is False
        assert task_state.last_result("meteo")["status"] == "done"

    def test_the_claim_is_released_when_the_run_raises(self, instance, monkeypatch):
        """Otherwise a crashed run blocks the task for the full ten minute
        staleness window."""

        def boom(cmd, **kwargs):
            raise OSError("cannot spawn")

        monkeypatch.setattr(da_scheduler, "Popen", boom)
        claim = task_state.claim("meteo", "scheduler")

        instance._run_claimed(claim)

        assert task_state.is_running("meteo") is False
        assert task_state.last_result("meteo")["status"] == "error"


class TestCancel:
    def test_it_kills_the_whole_process_group(self, instance, monkeypatch):
        proc = FakeProc()
        monkeypatch.setattr(da_scheduler, "Popen", lambda cmd, **kw: proc)
        killed = {}
        monkeypatch.setattr(
            os, "killpg", lambda pgid, sig: killed.update(pgid=pgid, sig=sig)
        )
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)

        claim = task_state.claim("meteo", "dashboard")
        task_state.request_cancel(claim)

        status, _returncode, _logfile = instance.run_task_process(claim)

        assert killed == {"pgid": proc.pid, "sig": signal.SIGKILL}
        assert status == "cancelled"


class TestLogFileDiscovery:
    def test_it_finds_the_log_the_task_wrote(self, instance, monkeypatch):
        """Each task writes its own log through run_task_function; the
        dashboard needs that path to show the output."""
        log_dir = instance.log_dir

        def popen_that_writes(cmd, **kwargs):
            open(os.path.join(log_dir, "meteo_2026-09-29__12:00:00.log"), "w").close()
            return FakeProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", popen_that_writes)
        claim = task_state.claim("meteo", "scheduler")

        _status, _returncode, logfile = instance.run_task_process(claim)

        assert logfile == "../data/log/meteo_2026-09-29__12:00:00.log"

    def test_it_ignores_another_tasks_log(self, instance, monkeypatch):
        """The v2 dashboard used to watch for the newest *.log of any kind,
        which picked up the wrong output when two tasks ran at once. The
        registry knows each task's own prefix."""
        log_dir = instance.log_dir

        def popen_that_writes(cmd, **kwargs):
            open(os.path.join(log_dir, "calc_2026-09-29__12:00:01.log"), "w").close()
            return FakeProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", popen_that_writes)
        claim = task_state.claim("meteo", "scheduler")

        _status, _returncode, logfile = instance.run_task_process(claim)

        assert logfile is None

    def test_a_log_that_was_already_there_is_not_claimed(
        self, instance, monkeypatch
    ):
        """A previous run's log must not be presented as this run's output."""
        open(
            os.path.join(instance.log_dir, "meteo_2026-09-28__08:00:00.log"), "w"
        ).close()
        monkeypatch.setattr(da_scheduler, "Popen", lambda cmd, **kw: FakeProc())
        claim = task_state.claim("meteo", "scheduler")

        _status, _returncode, logfile = instance.run_task_process(claim)

        assert logfile is None


class TestPickUpRequests:
    @pytest.mark.parametrize("key", ["meteo", "no_such_task"])
    def test_stale_pending_snapshot_cannot_take_or_release_replacement(self, instance, monkeypatch, key):
        old = task_state.request(key, "dashboard")
        pending_requests = task_state.pending_requests
        replacement_bytes = []
        jobs = []

        def replace_after_snapshot():
            snapshot = pending_requests()
            task_state.release(old, "done")
            task_state.request(key, "api")
            replacement_bytes.append(Path(task_state.STATE_PATH).read_bytes())
            return snapshot

        monkeypatch.setattr(task_state, "pending_requests", replace_after_snapshot)
        instance.scheduler = type(
            "S", (), {"add_job": lambda self, *args, **kw: jobs.append((args, kw))}
        )()

        instance._pick_up_requests()

        assert replacement_bytes
        assert Path(task_state.STATE_PATH).read_bytes() == replacement_bytes[0]
        assert jobs == []

    def test_a_pending_request_is_taken_and_run(self, instance, monkeypatch):
        ran = []
        monkeypatch.setattr(
            instance,
            "_run_claimed",
            lambda claim, parameters=None: ran.append((claim, parameters)),
        )
        # add_job normally hands it to the thread pool; run it straight away.
        instance.scheduler = type(
            "S", (), {"add_job": lambda self, fn, args=(), **kw: fn(*args)}
        )()

        claim = task_state.request("meteo", source="dashboard", parameters={"x": "1"})
        instance._pick_up_requests()

        assert ran == [(claim, {"x": "1"})]

    def test_taking_it_flips_the_state_so_a_second_poll_skips_it(
        self, instance, monkeypatch
    ):
        instance.scheduler = type(
            "S", (), {"add_job": lambda self, fn, args=(), **kw: None}
        )()

        task_state.request("meteo", source="dashboard")
        instance._pick_up_requests()

        assert task_state.pending_requests() == {}
        assert task_state.running_tasks()["meteo"]["state"] == "running"

    def test_a_request_for_an_unknown_task_is_released(self, instance, caplog):
        task_state.claim(
            "no_such_task", "dashboard", state_name="pending"
        )

        instance._pick_up_requests()

        assert task_state.is_running("no_such_task") is False
        assert "no_such_task" in caplog.text

    def test_polling_records_liveness(self, instance):
        """So the dashboard can say the planner is not running instead of
        letting a request sit there unanswered."""
        assert task_state.scheduler_alive() is None

        instance._pick_up_requests()

        assert task_state.scheduler_alive() is True

    def test_liveness_is_not_rewritten_on_every_poll(self, instance):
        """At a five second interval that would be some seventeen thousand
        small writes a day to what is often an SD card."""
        instance._pick_up_requests()
        first = task_state.read()[task_state.SCHEDULER_SEEN_KEY]

        instance._pick_up_requests()

        assert task_state.read()[task_state.SCHEDULER_SEEN_KEY] == first
