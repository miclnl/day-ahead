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
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from subprocess import TimeoutExpired

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


class EndedProc(FakeProc):
    """A process that was already gone when the runner first polled it."""

    def __init__(self, returncode=0, events=None):
        super().__init__()
        self.returncode = returncode
        self.waits = 0
        self.reaped = False
        self.events = events if events is not None else []

    def poll(self):
        # Polling a process that has ended reaps it. Its pid is gone
        # afterwards, and so is every way to find its process group
        # through that pid.
        self.reaped = True
        self.events.append(("reaped", self.pid))
        return self.returncode

    def wait(self):
        self.waits += 1
        return self.returncode


#: A task process that ends at once and leaves a member of its process
#: group behind. The member writes a heartbeat for as long as it runs.
LEADER_SCRIPT = """
import os, subprocess, sys, time
beat, pidfile, member_script, stopfile = sys.argv[1:]
member = subprocess.Popen([sys.executable, "-c", member_script, beat, stopfile])
with open(pidfile, "w", encoding="utf-8") as handle:
    handle.write(str(member.pid))
while not os.path.exists(beat) or os.path.getsize(beat) == 0:
    if os.path.exists(stopfile):
        break
    time.sleep(0.01)
"""

MEMBER_SCRIPT = """
import os, sys, time
with open(sys.argv[1], "a", buffering=1) as beat:
    while not os.path.exists(sys.argv[2]):
        beat.write("x")
        beat.flush()
        time.sleep(0.05)
"""


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


def _wait(seconds: float) -> None:
    """A wait that really waits.

    The scheduler fixture takes the sleep out of the time module, so a
    test that has to let a process run cannot use ``time.sleep``.
    """
    threading.Event().wait(seconds)


def _wait_for_a_pid(pidfile: Path, seconds: float = 10.0) -> int:
    """The pid that a sandbox process wrote down, once it is there."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return int(pidfile.read_text())
        except (OSError, ValueError):
            if time.monotonic() > deadline:
                raise
            _wait(0.01)


def _assert_process_gone(pid: int, seconds: float = 5.0) -> None:
    """Wait for process exit/reap, checking existence rather than I/O."""
    process = Path(f"/proc/{pid}")
    deadline = time.monotonic() + seconds
    while process.exists() and time.monotonic() < deadline:
        _wait(0.01)
    assert not process.exists(), f"sandbox process {pid} still exists"


@contextmanager
def _sandbox_task_member(monkeypatch, tmp_path):
    """Give the runner a real, ended leader with a member left behind."""
    beat = tmp_path / "beat"
    pidfile = tmp_path / "member.pid"
    stopfile = tmp_path / "stop"
    real_popen = da_scheduler.Popen
    spawned = {}

    def spawn_a_leader_that_leaves_a_member(cmd, **kwargs):
        assert kwargs["start_new_session"] is True
        proc = real_popen(
            [
                sys.executable,
                "-c",
                LEADER_SCRIPT,
                str(beat),
                str(pidfile),
                MEMBER_SCRIPT,
                str(stopfile),
            ],
            **kwargs,
        )
        spawned["leader"] = proc
        spawned["member"] = _wait_for_a_pid(pidfile)
        # Leave the leader waitable for the runner's first poll.
        os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
        return proc

    monkeypatch.setattr(
        da_scheduler, "Popen", spawn_a_leader_that_leaves_a_member
    )
    failed = False
    try:
        yield spawned
    except BaseException:
        failed = True
        raise
    finally:
        # Only request self-exit. A pid/pgid recorded earlier may have been
        # reused, so cleanup must never send a signal to either identity.
        try:
            try:
                stopfile.touch()
            finally:
                leader = spawned.get("leader")
                if leader is not None:
                    leader.wait(timeout=5)
                member = spawned.get("member")
                if member is None and pidfile.exists():
                    member = _wait_for_a_pid(pidfile)
                if member is not None:
                    _assert_process_gone(member)
        except (OSError, TimeoutExpired, AssertionError):
            if not failed:
                raise


def _slot_gate(first_thread_name):
    """Decide the order of two contenders for the same sidecar slot.

    Cancelling a request and the scheduler picking it up contend for the
    same slot, and whoever gets it first owns the claim. Which one that
    is has to be decided rather than hoped for, so this wrapper lets the
    designated thread take the slot first and makes the other one queue
    up behind it. The real slot is still the one that serialises them.
    """
    real_locked = task_state._locked
    first_has_slot = threading.Event()
    second_is_waiting = threading.Event()
    contenders = {"cancel", "pickup"}

    @contextmanager
    def gated():
        name = threading.current_thread().name
        if name not in contenders:
            with real_locked():
                yield
            return
        if name != first_thread_name:
            # The other thread owns this race: wait until it holds the
            # slot, then contend for it.
            first_has_slot.wait(5)
            second_is_waiting.set()
        with real_locked():
            if name == first_thread_name:
                first_has_slot.set()
                second_is_waiting.wait(5)
            yield

    return gated


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
        monkeypatch.setattr(
            da_scheduler.task_registry,
            "kill_process_group",
            lambda proc, group=None: killed.append((proc, group)),
        )

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


class TestCancelBeforeStart:
    """Cancelling a request the scheduler has not started yet.

    The cancel and the pickup of the same request are linearised under the
    same slot, so exactly one of them owns the claim. A request that is
    cancelled while it is still waiting is never spawned; a request the
    scheduler already took is stopped by its runner: the whole process
    group, with a confirmed exit.
    """

    def test_cancelled_pending_request_never_spawns(self, instance, monkeypatch):
        """The task was cancelled before the planner picked it up, so it is
        not started at all -- and it did not fail either."""
        spawned = []

        def fake_popen(cmd, **kwargs):
            spawned.append(cmd)
            return FakeProc()

        monkeypatch.setattr(da_scheduler, "Popen", fake_popen)
        # The fake process has no process group of its own.
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        instance.scheduler = type(
            "S", (), {"add_job": lambda self, fn, args=(), **kw: fn(*args)}
        )()

        claim = task_state.request("meteo", source="dashboard")
        assert task_state.request_cancel(claim) is True

        instance._pick_up_requests()

        assert spawned == []
        assert task_state.is_running("meteo") is False
        assert task_state.last_result("meteo").get("status") == "cancelled"

    @pytest.mark.parametrize("cancel_first", [True, False])
    def test_cancel_start_race_has_one_owner(
        self, instance, monkeypatch, cancel_first
    ):
        """Whoever gets the slot first owns the claim; the other one has no
        effect at all.

        Both orders are exercised, because the order decides the outcome:
        a cancel that wins means the request is never started, a pickup that
        wins means the cancel becomes a running one.
        """
        spawned = []
        killed = []

        def fake_popen(cmd, **kwargs):
            spawned.append(cmd)
            return FakeProc()

        monkeypatch.setattr(da_scheduler, "Popen", fake_popen)
        monkeypatch.setattr(
            da_scheduler.task_registry,
            "kill_process_group",
            lambda proc, group=None: killed.append((proc, group)),
        )
        monkeypatch.setattr(
            task_state,
            "_locked",
            _slot_gate("cancel" if cancel_first else "pickup"),
        )

        claim = task_state.request("meteo", source="dashboard")
        outcome = {}

        def cancel():
            outcome["cancel"] = task_state.request_cancel(claim)

        def pickup():
            outcome["pickup"] = task_state.take_pending(claim)

        workers = [
            threading.Thread(target=cancel, name="cancel"),
            threading.Thread(target=pickup, name="pickup"),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()

        assert outcome["cancel"] is True
        if cancel_first:
            # The cancel owns the claim: the pickup finds nothing and the
            # task is never started.
            assert outcome["pickup"] is None
            assert spawned == []
            assert task_state.last_result("meteo").get("status") == "cancelled"
        else:
            # The pickup owns the claim: the cancel is a running one, so
            # the runner stops the process group and confirms the exit.
            assert outcome["pickup"] is not None
            assert task_state.running_tasks()["meteo"]["cancel"] is True
            instance._run_claimed(claim)
            assert len(spawned) == 1
            assert len(killed) == 1
            assert task_state.last_result("meteo").get("status") == "cancelled"

    def test_a_running_cancel_stops_the_process_group_and_confirms_the_exit(
        self, instance, monkeypatch
    ):
        """The pickup won, so the cancel is a running one.

        A real process group of our own rather than a fake: the point is
        that the signal reaches the process and that the runner waits for
        it, so the outcome is a confirmed exit and not an assumption.
        """
        real_popen = da_scheduler.Popen

        def spawn_a_sleeping_child(cmd, **kwargs):
            return real_popen(
                [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
            )

        monkeypatch.setattr(da_scheduler, "Popen", spawn_a_sleeping_child)

        claim = task_state.request("meteo", source="dashboard")
        assert task_state.take_pending(claim) is not None
        assert task_state.request_cancel(claim) is True

        instance._run_claimed(claim)

        result = task_state.last_result("meteo")
        assert result["status"] == "cancelled"
        assert result["returncode"] == -signal.SIGKILL

    def test_two_poll_cycles_never_start_the_same_request_twice(
        self, instance, monkeypatch
    ):
        """Two polls of the planner see the same request; only one of them
        may start it."""
        spawned = []
        jobs = []

        def fake_popen(cmd, **kwargs):
            spawned.append(cmd)
            return FakeProc()

        monkeypatch.setattr(da_scheduler, "Popen", fake_popen)
        instance.scheduler = type(
            "S", (), {"add_job": lambda self, fn, args=(), **kw: jobs.append((fn, args))}
        )()

        task_state.request("meteo", source="dashboard")

        instance._pick_up_requests()
        instance._pick_up_requests()

        assert len(jobs) == 1
        for function, args in jobs:
            function(*args)
        assert len(spawned) == 1


class TestCancelAfterStart:
    """A cancellation that was accepted while the task was running.

    The runner reads the cancel flag while it polls the process, but a
    task that ends before its first heartbeat never enters that loop. The
    cancellation is answered there as well: the run was called off, so it
    ends as "cancelled" and not as finished or failed.
    """

    @pytest.mark.parametrize("returncode", [0, 2])
    def test_a_cancel_is_answered_when_the_task_ended_before_the_first_heartbeat(
        self, instance, monkeypatch, returncode
    ):
        """The child was gone before the runner's first heartbeat, so the
        polling loop is never entered. The cancellation was accepted, so
        the run ends as "cancelled": not as finished, and not as failed.
        """
        proc = EndedProc(returncode)
        monkeypatch.setattr(da_scheduler, "Popen", lambda cmd, **kw: proc)
        # The fake process has no process group of its own.
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)

        claim = task_state.request("meteo", source="dashboard")
        assert task_state.take_pending(claim) is not None
        assert task_state.request_cancel(claim) is True

        status, code, _logfile = instance.run_task_process(claim)

        assert status == "cancelled"
        assert code == returncode

    def test_a_cancelled_task_that_already_ended_stops_its_group(
        self, instance, monkeypatch
    ):
        """The child was gone before the first heartbeat. The cancellation
        is still answered: the group is stopped -- it can hold members
        that outlived the child -- and the exit is confirmed.

        The identity of the group is recorded while the leader is still
        there. Looking it up through the pid afterwards does not work: a
        pid that has been reaped is gone, and the group it led would be
        left running.
        """
        events = []
        proc = EndedProc(events=events)
        killed = {}

        def getpgid(pid):
            events.append(("getpgid", pid))
            if proc.reaped:
                # A reaped pid is gone: the group cannot be found
                # through it any more.
                raise ProcessLookupError(pid)
            return pid

        monkeypatch.setattr(da_scheduler, "Popen", lambda cmd, **kw: proc)
        monkeypatch.setattr(
            os, "killpg", lambda pgid, sig: killed.update(pgid=pgid, sig=sig)
        )
        monkeypatch.setattr(os, "getpgid", getpgid)

        claim = task_state.request("meteo", source="dashboard")
        assert task_state.take_pending(claim) is not None
        assert task_state.request_cancel(claim) is True

        status, _code, _logfile = instance.run_task_process(claim)

        assert events[0] == ("getpgid", proc.pid), (
            "the group was not recorded while the leader was still there"
        )
        assert killed == {"pgid": proc.pid, "sig": signal.SIGKILL}
        assert proc.waits == 1
        assert status == "cancelled"


class TestGroupStop:
    """A task that ends can leave members of its process group behind.

    A cancel is answered by stopping the whole group, also when the task
    itself has already ended: a member that outlived the task is the only
    thing left of it. The identity of the group is recorded while the
    leader is still there, because a pid that has been reaped is gone --
    and can no longer lead anyone to its group.
    """

    def test_a_member_of_an_ended_task_is_stopped(
        self, instance, monkeypatch, tmp_path
    ):
        """A real task process in a session of its own, which spawns a
        member and ends at once. The member has to be stopped all the
        same.
        """
        claim = task_state.request("meteo", source="dashboard")
        assert task_state.take_pending(claim) is not None
        assert task_state.request_cancel(claim) is True

        real_killpg = os.killpg
        stopped = []
        runner_active = False

        def product_stop(group, sig):
            # Forward only the product's stop, never a cleanup attempt.
            assert runner_active
            assert not (tmp_path / "stop").exists()
            assert group == spawned["leader"].pid != os.getpgrp()
            assert sig == signal.SIGKILL
            stopped.append((group, sig))
            real_killpg(group, sig)

        monkeypatch.setattr(os, "killpg", product_stop)
        with _sandbox_task_member(monkeypatch, tmp_path) as spawned:
            runner_active = True
            try:
                status, code, _logfile = instance.run_task_process(claim)
            finally:
                runner_active = False
            # Before sandbox cleanup: the runner itself must have stopped
            # the member, even if it happens not to write a heartbeat.
            assert not (tmp_path / "stop").exists()
            _assert_process_gone(spawned["member"])

        assert stopped == [(spawned["leader"].pid, signal.SIGKILL)]
        assert status == "cancelled"
        assert code == 0, "the exit of the task was not confirmed"

    @pytest.mark.parametrize("cleanup_error", [None, ProcessLookupError, PermissionError])
    def test_cleanup_stops_an_idle_member_when_the_runner_raises(
        self, instance, monkeypatch, tmp_path, cleanup_error
    ):
        """A live member without heartbeat growth must not escape cleanup."""
        # Write readiness once, then only poll the sandbox stop file.
        monkeypatch.setattr(
            sys.modules[__name__],
            "MEMBER_SCRIPT",
            'import os, sys, time\n'
            'with open(sys.argv[1], "w") as beat:\n'
            '    beat.write("ready")\n'
            'while not os.path.exists(sys.argv[2]):\n'
            '    time.sleep(0.01)\n',
        )
        failure = RuntimeError("runner failed during heartbeat")
        stopfile = tmp_path / "stop"

        def fail_heartbeat(*args, **kwargs):
            assert Path(f"/proc/{spawned['member']}").exists()
            assert not stopfile.exists()
            raise failure

        monkeypatch.setattr(task_state, "heartbeat", fail_heartbeat)
        claim = task_state.claim("meteo", source="scheduler")
        spawned = {}
        signals = []

        def reject_signal(*args):
            signals.append(args)
            # Do not deliver anything, even when checking a broken cleanup.
            raise PermissionError("sandbox cleanup must not signal")

        real_touch = Path.touch

        def touch_then_raise(path, *args, **kwargs):
            real_touch(path, *args, **kwargs)
            if path == stopfile and cleanup_error is not None:
                # The real stop request succeeds; an ensuing cleanup error
                # must not replace the runner's primary exception.
                raise cleanup_error("cleanup failed")

        monkeypatch.setattr(Path, "touch", touch_then_raise)
        monkeypatch.setattr(os, "killpg", reject_signal)
        monkeypatch.setattr(os, "kill", reject_signal)
        try:
            with pytest.raises(RuntimeError) as caught:
                with _sandbox_task_member(monkeypatch, tmp_path) as spawned:
                    instance.run_task_process(claim)
            assert caught.value is failure, "cleanup masked the runner failure"
            assert signals == [], "sandbox cleanup attempted to send a signal"
            assert stopfile.exists()
            _assert_process_gone(spawned["member"])
        finally:
            # Signal-free regression guard, also when the helper is broken.
            real_touch(stopfile)
            leader = spawned.get("leader")
            if leader is not None:
                leader.wait(timeout=5)
            if "member" in spawned:
                _assert_process_gone(spawned["member"])


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

    def test_calc_log_never_selects_calc_debug(self, instance, monkeypatch):
        """The debug optimisation writes "calc_debug_...", which starts with
        "calc_". A prefix match on its own therefore handed the regular
        optimisation the output of the debug run."""
        log_dir = instance.log_dir

        def popen_that_writes(cmd, **kwargs):
            open(os.path.join(log_dir, "calc_2026-09-29__12:00:00.log"), "w").close()
            open(
                os.path.join(log_dir, "calc_debug_2026-09-29__12:00:01.log"), "w"
            ).close()
            return FakeProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", popen_that_writes)
        claim = task_state.claim("calc_optimum", "scheduler")

        _status, _returncode, logfile = instance.run_task_process(claim)

        assert logfile == "../data/log/calc_2026-09-29__12:00:00.log"

    def test_the_debug_task_keeps_its_own_log(self, instance, monkeypatch):
        """The other way round: the debug run is not robbed of its own
        output by the regular one."""
        log_dir = instance.log_dir

        def popen_that_writes(cmd, **kwargs):
            open(os.path.join(log_dir, "calc_2026-09-29__12:00:00.log"), "w").close()
            open(
                os.path.join(log_dir, "calc_debug_2026-09-29__12:00:01.log"), "w"
            ).close()
            return FakeProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", popen_that_writes)
        claim = task_state.claim("calc_optimum_met_debug", "scheduler")

        _status, _returncode, logfile = instance.run_task_process(claim)

        assert logfile == "../data/log/calc_debug_2026-09-29__12:00:01.log"

    def test_immediate_exit_retains_log(self, instance, monkeypatch):
        """A task that finishes in a fraction of a second never reaches the
        polling loop, so its log was never looked up and the output
        disappeared from the dashboard."""
        log_dir = instance.log_dir

        class ImmediateProc(FakeProc):
            def poll(self):
                return 0

        def popen_that_writes(cmd, **kwargs):
            open(os.path.join(log_dir, "meteo_2026-09-29__12:00:00.log"), "w").close()
            return ImmediateProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", popen_that_writes)
        claim = task_state.claim("meteo", "scheduler")

        status, _returncode, logfile = instance.run_task_process(claim)

        assert logfile == "../data/log/meteo_2026-09-29__12:00:00.log"
        assert status == "done"

    def test_the_log_reader_does_not_follow_a_symlink_out_of_the_log_directory(
        self, instance, monkeypatch
    ):
        """A name in the log directory may be a symlink to somewhere else
        entirely. Following it would present a file from outside the
        directory as this task's output."""
        log_dir = Path(instance.log_dir)
        outside = log_dir.parent / "outside.log"
        outside.write_text("not this task's output", encoding="utf-8")

        def popen_that_links(cmd, **kwargs):
            (log_dir / "meteo_2026-09-29__12:00:00.log").symlink_to(outside)
            return FakeProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", popen_that_links)
        claim = task_state.claim("meteo", "scheduler")

        _status, _returncode, logfile = instance.run_task_process(claim)

        assert logfile is None

    @pytest.mark.parametrize("kind", ["directory", "fifo"])
    def test_a_non_regular_file_with_a_task_log_name_is_not_selected(
        self, instance, monkeypatch, kind
    ):
        """A name in the log directory may be a directory or a fifo. Only a
        regular file is a task log; the others are never offered as this
        task's output."""
        log_dir = Path(instance.log_dir)

        def popen_that_creates(cmd, **kwargs):
            name = log_dir / "meteo_2026-09-29__12:00:00.log"
            if kind == "directory":
                name.mkdir()
            else:
                os.mkfifo(name)
            return FakeProc(cmd, **kwargs)

        monkeypatch.setattr(da_scheduler, "Popen", popen_that_creates)
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
