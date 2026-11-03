"""The shared task claim is what stops the scheduler and the dashboard from
running the same task twice.

The interesting property is atomicity across processes: the old dashboard
code read the state, decided nothing was running, and only wrote "running"
once its worker thread got scheduled, so two near-simultaneous requests both
passed the check. claim() does the read-check-write while holding an
exclusive flock, so exactly one caller can win -- which the concurrency test
at the bottom checks with real processes rather than by reading the code.
"""

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from dao.prog import task_state

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def state_in_tmp(tmp_path, monkeypatch):
    """Point the module at a throwaway directory.

    The real paths are relative to the working directory of whichever entry
    point is running, so a test must never be allowed to touch the operator's
    ../data/task_state.json.
    """
    monkeypatch.setattr(task_state, "STATE_PATH", str(tmp_path / "task_state.json"))
    monkeypatch.setattr(task_state, "LOCK_PATH", str(tmp_path / "task_state.lock"))
    return tmp_path


class TestClaim:
    def test_a_first_claim_is_granted(self):
        granted = task_state.claim("calc_optimum", "scheduler")
        assert isinstance(granted, task_state.TaskClaim)
        assert granted.task_key == "calc_optimum" and granted.run_id
        assert task_state.is_running("calc_optimum") is True

    def test_a_second_claim_on_the_same_task_is_refused(self):
        assert task_state.claim("calc_optimum", "scheduler") is not None
        assert task_state.claim("calc_optimum", "dashboard") is None

    def test_a_different_task_may_run_at_the_same_time(self):
        """Matches what the scheduler already allowed: the optimisation does
        not have to wait for ML training. Exclusion is per task."""
        assert task_state.claim("calc_optimum", "scheduler") is not None
        assert task_state.claim("train_ml_predictions", "dashboard") is not None
        assert set(task_state.running_tasks()) == {
            "calc_optimum",
            "train_ml_predictions",
        }

    def test_releasing_frees_the_task_for_a_new_claim(self):
        granted = task_state.claim("meteo", "scheduler")
        task_state.release(granted, "done", returncode=0)
        assert task_state.is_running("meteo") is False
        replacement = task_state.claim("meteo", "dashboard")
        assert replacement is not None and replacement != granted

    def test_the_source_is_recorded(self):
        task_state.claim("meteo", "scheduler")
        assert task_state.running_tasks()["meteo"]["source"] == "scheduler"


class TestRelease:
    def test_the_outcome_is_kept_so_the_dashboard_can_still_show_it(self):
        granted = task_state.claim("prices", "dashboard", logfile="../data/log/prices_x.log")
        task_state.release(granted, "error", returncode=2)

        result = task_state.last_result("prices")
        assert result["status"] == "error"
        assert result["returncode"] == 2
        assert result["logfile"] == "../data/log/prices_x.log"
        assert result["source"] == "dashboard"
        assert result["run_id"] == granted.run_id
        assert result["finished"] > 0

    def test_a_logfile_discovered_while_running_wins(self):
        """Both dashboards only learn the real log file name after the task
        has started (they watch the log directory for a new file), so the
        claim is made without one and it is filled in later."""
        granted = task_state.claim("calc_optimum", "dashboard")
        task_state.heartbeat(granted, logfile="../data/log/calc_late.log")
        task_state.release(granted, "done", returncode=0)

        assert task_state.last_result("calc_optimum")["logfile"] == (
            "../data/log/calc_late.log"
        )

    def test_releasing_a_task_that_is_not_running_is_harmless(self):
        task_state.release("meteo", "done", returncode=0)
        assert task_state.last_result("meteo") == {}
        assert task_state.read() == {"running": {}, "last": {}, "last_update": 0.0}
        assert not Path(task_state.STATE_PATH).exists()


def _age_run(key):
    path = Path(task_state.STATE_PATH)
    state = json.loads(path.read_text())
    state["running"][key]["heartbeat"] = time.time() - task_state.STALE_AFTER_S - 1
    path.write_text(json.dumps(state))


def _age_heartbeat(key, seconds):
    """Make the claim of *key* look *seconds* old."""
    path = Path(task_state.STATE_PATH)
    state = json.loads(path.read_text())
    state["running"][key]["heartbeat"] = time.time() - seconds
    path.write_text(json.dumps(state))


class TestRunIdentity:
    def test_claim_and_request_return_non_reusable_run_id(self, monkeypatch):
        monkeypatch.setattr(task_state.time, "time", lambda: 1800000000.0)
        first = task_state.claim("meteo", "scheduler")
        assert first is not None and not isinstance(first, bool)
        assert first.task_key == "meteo" and first.run_id
        assert task_state.running_tasks()["meteo"]["run_id"] == first.run_id
        assert task_state.request("meteo", "dashboard") is None
        task_state.release(first, "done")
        second = task_state.request("meteo", "dashboard")
        assert second.task_key == "meteo" and second.run_id != first.run_id
        entry = task_state.take_pending(second)
        assert entry["run_id"] == second.run_id and entry["state"] == "running"
        task_state.release(second, "done")
        assert task_state.last_result("meteo")["run_id"] == second.run_id

    @pytest.mark.parametrize("operation", ["release", "heartbeat", "cancel", "pending"])
    def test_stale_cancel_or_release_cannot_touch_replacement_run(self, operation):
        old = task_state.request("meteo", "dashboard")
        _age_run("meteo")
        replacement = task_state.request("meteo", "api")
        assert replacement
        before = Path(task_state.STATE_PATH).read_bytes()

        if operation == "release":
            task_state.release(old, "error", returncode=7, logfile="old.log")
        elif operation == "heartbeat":
            task_state.heartbeat(old, logfile="old.log")
        elif operation == "cancel":
            task_state.request_cancel(old)
        else:
            task_state.take_pending(old)

        assert Path(task_state.STATE_PATH).read_bytes() == before
        assert task_state.last_result("meteo") == {}

    @pytest.mark.parametrize("operation", ["release", "heartbeat", "cancel", "pending"])
    def test_task_key_alone_cannot_update_a_claim(self, operation):
        task_state.request("meteo", "dashboard")
        before = Path(task_state.STATE_PATH).read_bytes()
        if operation == "release":
            task_state.release("meteo", "done")
        elif operation == "heartbeat":
            assert task_state.heartbeat("meteo", logfile="unowned.log") == {}
        elif operation == "cancel":
            assert task_state.request_cancel("meteo") is False
        else:
            assert task_state.take_pending("meteo") is None
        assert Path(task_state.STATE_PATH).read_bytes() == before

    @pytest.mark.parametrize("operation", ["release", "heartbeat", "cancel", "pending"])
    def test_identity_is_compared_after_waiting_for_the_sidecar(self, monkeypatch, operation):
        old = task_state.request("meteo", "dashboard")
        entered = threading.Event()
        locked = task_state._locked

        @contextmanager
        def observed_lock():
            entered.set()
            with locked():
                yield

        monkeypatch.setattr(task_state, "_locked", observed_lock)
        errors = []

        def delayed_update():
            try:
                if operation == "release":
                    task_state.release(old, "error")
                elif operation == "heartbeat":
                    task_state.heartbeat(old, logfile="old.log")
                elif operation == "cancel":
                    task_state.request_cancel(old)
                else:
                    task_state.take_pending(old)
            except BaseException as error:
                errors.append(error)

        with locked():
            worker = threading.Thread(target=delayed_update)
            worker.start()
            assert entered.wait(5)
            path = Path(task_state.STATE_PATH)
            state = json.loads(path.read_text())
            state["running"]["meteo"]["run_id"] = "replacement-under-sidecar"
            task_state._write(state)  # Atomic inode replacement under the same sidecar.
            before = path.read_bytes()
        worker.join(5)
        assert not worker.is_alive() and errors == []
        assert path.read_bytes() == before

    def test_wrong_task_key_with_another_runs_id_is_not_an_identity(self):
        meteo = task_state.request("meteo", "dashboard")
        task_state.request("prices", "dashboard")
        wrong = task_state.TaskClaim("prices", meteo.run_id)
        before = Path(task_state.STATE_PATH).read_bytes()
        task_state.release(wrong, "done")
        assert task_state.heartbeat(wrong) == {}
        assert task_state.request_cancel(wrong) is False
        assert task_state.take_pending(wrong) is None
        assert Path(task_state.STATE_PATH).read_bytes() == before

    def test_old_release_cannot_overwrite_a_newer_finished_result(self):
        old = task_state.claim("meteo", "scheduler")
        task_state.release(old, "done")
        new = task_state.claim("meteo", "scheduler")
        task_state.release(new, "error", returncode=7)
        before = Path(task_state.STATE_PATH).read_bytes()
        task_state.release(old, "cancelled")
        assert Path(task_state.STATE_PATH).read_bytes() == before
        assert task_state.last_result("meteo")["run_id"] == new.run_id

    def test_identityless_legacy_entry_has_no_key_only_update_fallback(self):
        task_state.request("meteo", "dashboard")
        path = Path(task_state.STATE_PATH)
        state = json.loads(path.read_text())
        state["running"]["meteo"].pop("run_id")
        path.write_text(json.dumps(state))
        empty = task_state.TaskClaim("meteo", "")
        before = path.read_bytes()
        task_state.release(empty, "done")
        assert task_state.heartbeat(empty) == {}
        assert task_state.request_cancel(empty) is False
        assert task_state.take_pending(empty) is None
        assert path.read_bytes() == before

    def test_independent_state_does_not_inherit_unowned_release(self, monkeypatch, tmp_path):
        task_state.release("meteo", "done")
        independent = tmp_path / "independent.json"
        monkeypatch.setattr(task_state, "STATE_PATH", str(independent))
        task_state.request("prices", "api")
        saved = json.loads(independent.read_text())
        assert saved["last"] == {}
        assert set(saved["running"]) == {"prices"}


class TestStaleClaims:
    def test_a_claim_nobody_released_stops_blocking_after_the_timeout(self):
        """A task subprocess that is killed -- gunicorn recycling its worker
        on a config change, the watchdog restarting the scheduler, the OOM
        killer -- never releases its claim. Without expiry the task would be
        unstartable until someone deleted the file by hand."""
        task_state.claim("calc_optimum", "dashboard")

        state = json.loads(Path(task_state.STATE_PATH).read_text())
        old = time.time() - task_state.STALE_AFTER_S - 1
        state["running"]["calc_optimum"]["heartbeat"] = old
        Path(task_state.STATE_PATH).write_text(json.dumps(state))

        assert task_state.is_running("calc_optimum") is False
        assert task_state.claim("calc_optimum", "scheduler") is not None

    def test_a_heartbeat_keeps_a_long_task_alive(self):
        granted = task_state.claim("train_ml_predictions", "scheduler")

        state = json.loads(Path(task_state.STATE_PATH).read_text())
        state["running"]["train_ml_predictions"]["heartbeat"] = (
            time.time() - task_state.STALE_AFTER_S - 1
        )
        Path(task_state.STATE_PATH).write_text(json.dumps(state))

        task_state.heartbeat(granted)

        assert task_state.is_running("train_ml_predictions") is True

    def test_one_tasks_heartbeat_does_not_revive_another(self):
        granted = task_state.claim("meteo", "scheduler")
        task_state.claim("prices", "scheduler")

        state = json.loads(Path(task_state.STATE_PATH).read_text())
        state["running"]["prices"]["heartbeat"] = (
            time.time() - task_state.STALE_AFTER_S - 1
        )
        Path(task_state.STATE_PATH).write_text(json.dumps(state))

        task_state.heartbeat(granted)

        assert task_state.is_running("meteo") is True
        assert task_state.is_running("prices") is False


class TestCancel:
    def test_cancel_sets_a_flag_the_runner_can_see(self):
        granted = task_state.claim("calc_optimum", "dashboard")
        assert task_state.request_cancel(granted) is True
        assert task_state.cancel_requested(granted) is True

    def test_the_flag_travels_back_on_the_next_heartbeat(self):
        """A runner polling its subprocess is calling heartbeat anyway, so it
        reads the cancel flag from that same call rather than doing a second
        round trip to disk."""
        granted = task_state.claim("calc_optimum", "dashboard")
        task_state.request_cancel(granted)

        entry = task_state.heartbeat(granted)

        assert entry["cancel"] is True

    def test_cancelling_a_task_that_is_not_running_reports_false(self):
        missing = task_state.TaskClaim("meteo", "nonexistent-run")
        assert task_state.request_cancel(missing) is False
        assert task_state.cancel_requested(missing) is False


class TestPendingCancel:
    """The race between cancelling a request and the scheduler picking it up.

    Both happen under the same sidecar slot, so exactly one of them owns
    the claim. A request that is cancelled while it is still waiting is
    finished there and then: it can never be spawned afterwards, and it
    ends as "cancelled" rather than as a failure.
    """

    def test_cancelling_a_pending_request_finishes_it_as_cancelled(self):
        granted = task_state.request("meteo", "dashboard")

        assert task_state.request_cancel(granted) is True

        assert task_state.pending_requests() == {}
        assert task_state.is_running("meteo") is False
        result = task_state.last_result("meteo")
        assert result.get("status") == "cancelled"
        assert result["run_id"] == granted.run_id

    def test_a_cancelled_request_is_never_taken(self):
        """The pickup of a request that was already cancelled finds nothing:
        the task is not started at all."""
        granted = task_state.request("meteo", "dashboard")
        task_state.request_cancel(granted)

        assert task_state.take_pending(granted) is None
        assert task_state.is_running("meteo") is False
        assert task_state.last_result("meteo").get("status") == "cancelled"

    def test_the_pickup_wins_when_it_happens_first(self):
        """Then the cancel is a running one: the start owns the spawn, and
        the cancel is the flag its runner will answer with a process group
        stop."""
        granted = task_state.request("meteo", "dashboard")

        entry = task_state.take_pending(granted)
        assert entry is not None
        assert task_state.request_cancel(granted) is True

        assert task_state.is_running("meteo") is True
        assert task_state.running_tasks()["meteo"]["cancel"] is True
        assert task_state.last_result("meteo") == {}

    def test_a_cancelled_request_does_not_expire_as_a_failure(self):
        """It was cancelled, so it did not fail. The dashboard has to be able
        to say so."""
        granted = task_state.request("meteo", "dashboard")
        task_state.request_cancel(granted)

        assert task_state.expire_overdue_pending() == []
        assert task_state.last_result("meteo").get("status") == "cancelled"

    def test_a_cancelled_waiting_request_from_an_earlier_version_expires_as_cancelled(
        self,
    ):
        """A claim that was cancelled while it was waiting used to keep a
        cancel flag and nothing else; such a claim must not be reported as
        a failure either."""
        granted = task_state.request("meteo", "dashboard")
        path = Path(task_state.STATE_PATH)
        state = json.loads(path.read_text())
        state["running"]["meteo"]["cancel"] = True
        path.write_text(json.dumps(state))
        _age_heartbeat("meteo", task_state.PENDING_TIMEOUT_S + 1)

        expired = task_state.expire_overdue_pending()

        assert expired == ["meteo"]
        result = task_state.last_result("meteo")
        assert result["status"] == "cancelled"
        assert result["run_id"] == granted.run_id


class TestTakenRequest:
    """A request the scheduler has taken is not a waiting request anymore.

    It belongs to the scheduler from that moment on, so it may not be
    expired like a request nobody picked up, and it is not waiting for
    anything any more either.
    """

    def test_it_is_no_longer_a_pending_request(self):
        granted = task_state.request("meteo", "dashboard")
        task_state.take_pending(granted)

        assert task_state.pending_requests() == {}

    def test_it_does_not_get_the_short_pending_timeout(self):
        """A pending request that nobody picks up is given up after a
        minute. A request the scheduler took is somebody's task now, even
        if the process has not been spawned yet."""
        granted = task_state.request("meteo", "dashboard")
        task_state.take_pending(granted)
        _age_heartbeat("meteo", task_state.PENDING_TIMEOUT_S + 1)

        assert task_state.is_running("meteo") is True

    def test_it_is_not_expired_by_the_pending_expiry(self):
        granted = task_state.request("meteo", "dashboard")
        task_state.take_pending(granted)
        _age_heartbeat("meteo", task_state.PENDING_TIMEOUT_S + 1)

        assert task_state.expire_overdue_pending() == []
        assert task_state.is_running("meteo") is True
        assert task_state.last_result("meteo") == {}


class TestClaimedContextManager:
    def test_context_exit_does_not_release_a_replacement_run(self):
        with task_state.claimed("meteo", "scheduler") as granted:
            assert granted is True
            _age_run("meteo")
            assert task_state.claim("meteo", "dashboard") is not None
            before = Path(task_state.STATE_PATH).read_bytes()
        assert Path(task_state.STATE_PATH).read_bytes() == before
        assert task_state.last_result("meteo") == {}

    def test_it_releases_on_success(self):
        with task_state.claimed("meteo", "scheduler") as granted:
            assert granted is True
            assert task_state.is_running("meteo") is True

        assert task_state.is_running("meteo") is False
        assert task_state.last_result("meteo")["status"] == "done"

    def test_it_releases_and_records_an_error_on_an_exception(self):
        with pytest.raises(RuntimeError):
            with task_state.claimed("meteo", "scheduler") as granted:
                assert granted is True
                raise RuntimeError("task blew up")

        assert task_state.is_running("meteo") is False
        assert task_state.last_result("meteo")["status"] == "error"

    def test_a_refused_claim_does_not_release_the_holders_slot(self):
        """Releasing a claim somebody else owns would hand their task's slot
        away while it is still running, which is exactly the double-run this
        module exists to prevent."""
        task_state.claim("meteo", "scheduler")

        with task_state.claimed("meteo", "dashboard") as granted:
            assert granted is False

        assert task_state.is_running("meteo") is True
        assert task_state.running_tasks()["meteo"]["source"] == "scheduler"


class TestCorruptAndLegacyState:
    def test_a_truncated_state_file_is_treated_as_empty(self):
        """The v1 dashboard wrote this file non-atomically, so a half-written
        file is a shape that really occurred in the field."""
        Path(task_state.STATE_PATH).write_text('{"running": {"calc_optimum"')

        assert task_state.running_tasks() == {}
        assert task_state.claim("calc_optimum", "scheduler") is not None

    def test_a_legacy_flat_state_file_does_not_block_anything(self):
        """The old format was a single flat record of whichever task the
        dashboard was tracking. The writers of that format are gone, so an
        inherited claim would never be released."""
        Path(task_state.STATE_PATH).write_text(
            json.dumps(
                {
                    "status": "running",
                    "task": "calc_zonder_debug",
                    "logfile": "../data/log/calc_old.log",
                    "last_update": time.time(),
                }
            )
        )

        assert task_state.running_tasks() == {}
        assert task_state.claim("calc_optimum", "dashboard") is not None

    def test_a_missing_state_file_is_empty_not_an_error(self):
        assert task_state.running_tasks() == {}

    def test_a_non_dict_state_file_is_treated_as_empty(self):
        Path(task_state.STATE_PATH).write_text("[1, 2, 3]")
        assert task_state.running_tasks() == {}


class TestCrossProcessAtomicity:
    def test_only_one_of_many_processes_wins_the_same_claim(self, state_in_tmp):
        """The real property: the read-check-write happens under an
        exclusive flock, so concurrent processes cannot all pass the "is it
        running" check the way the old dashboard code let them.

        Uses real subprocesses rather than threads: the bug this prevents
        happened across gunicorn workers and the separate scheduler process,
        and a GIL-serialised thread test would not exercise the file lock.
        """
        script = state_in_tmp / "claimer.py"
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            "from dao.prog import task_state\n"
            f"task_state.STATE_PATH = {str(state_in_tmp / 'task_state.json')!r}\n"
            f"task_state.LOCK_PATH = {str(state_in_tmp / 'task_state.lock')!r}\n"
            "barrier = sys.argv[1]\n"
            "import os, time\n"
            "while not os.path.exists(barrier):\n"
            "    time.sleep(0.005)\n"
            "print('WON' if task_state.claim('calc_optimum', 'p') else 'LOST')\n"
        )
        barrier = state_in_tmp / "go"

        # Start them all first, then drop the barrier, so they pile into
        # claim() at the same moment instead of neatly one after another.
        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(barrier)],
                stdout=subprocess.PIPE,
                text=True,
            )
            for _ in range(8)
        ]
        time.sleep(0.4)
        barrier.write_text("go")

        results = [proc.communicate()[0].strip() for proc in procs]

        assert results.count("WON") == 1, results
        assert results.count("LOST") == 7, results
        assert task_state.is_running("calc_optimum") is True


class TestRequests:
    """A request is a claim in the "pending" state, so it excludes everything
    else from the moment the dashboard records it, before the scheduler has
    even seen it."""

    def test_a_request_blocks_cron_from_the_same_task(self):
        assert task_state.request("calc_optimum", "dashboard") is not None
        assert task_state.claim("calc_optimum", "scheduler") is None

    def test_a_request_carries_its_parameters(self):
        task_state.request("prices", "dashboard", parameters={"prijzen_start": "x"})

        entry = task_state.pending_requests()["prices"]
        assert entry["parameters"] == {"prijzen_start": "x"}
        assert entry["state"] == "pending"

    def test_a_running_task_is_not_a_pending_request(self):
        task_state.claim("meteo", "scheduler")
        assert task_state.pending_requests() == {}

    def test_taking_it_flips_the_state(self):
        granted = task_state.request("meteo", "dashboard")

        entry = task_state.take_pending(granted)

        assert entry["state"] == "running"
        assert entry["run_id"] == granted.run_id
        assert task_state.pending_requests() == {}
        assert task_state.is_running("meteo") is True

    def test_only_one_taker_wins(self):
        """Two scheduler poll cycles must not both start the same request."""
        granted = task_state.request("meteo", "dashboard")

        first = task_state.take_pending(granted)
        second = task_state.take_pending(granted)

        assert first is not None
        assert second is None

    def test_taking_something_that_is_not_pending_yields_none(self):
        granted = task_state.claim("meteo", "scheduler")
        assert task_state.take_pending(granted) is None
        assert task_state.take_pending(task_state.TaskClaim("never_requested", "missing")) is None


class TestPendingExpiry:
    def test_a_request_expires_much_sooner_than_a_running_task(self):
        """The scheduler polls every few seconds, so a request still pending
        after a minute means nobody is going to pick it up. Leaving the
        dashboard on "wordt gestart" for the full ten minutes tells the
        operator nothing."""
        task_state.request("meteo", "dashboard")

        state = json.loads(Path(task_state.STATE_PATH).read_text())
        state["running"]["meteo"]["heartbeat"] = (
            time.time() - task_state.PENDING_TIMEOUT_S - 1
        )
        Path(task_state.STATE_PATH).write_text(json.dumps(state))

        assert task_state.pending_requests() == {}
        assert task_state.claim("meteo", "scheduler") is not None

    def test_a_running_task_survives_the_pending_timeout(self):
        task_state.claim("train_ml_predictions", "scheduler")

        state = json.loads(Path(task_state.STATE_PATH).read_text())
        state["running"]["train_ml_predictions"]["heartbeat"] = (
            time.time() - task_state.PENDING_TIMEOUT_S - 1
        )
        Path(task_state.STATE_PATH).write_text(json.dumps(state))

        assert task_state.is_running("train_ml_predictions") is True

    def test_expiring_records_an_outcome_the_dashboard_can_show(self):
        """Letting the claim quietly lapse would leave the dashboard showing
        nothing at all; the operator pressed a button and deserves to know it
        did not happen."""
        granted = task_state.request("meteo", "dashboard")
        state = json.loads(Path(task_state.STATE_PATH).read_text())
        state["running"]["meteo"]["heartbeat"] = (
            time.time() - task_state.PENDING_TIMEOUT_S - 1
        )
        Path(task_state.STATE_PATH).write_text(json.dumps(state))

        expired = task_state.expire_overdue_pending()

        assert expired == ["meteo"]
        result = task_state.last_result("meteo")
        assert result["status"] == "error"
        assert result["run_id"] == granted.run_id
        assert "planner" in result["message"]

    def test_a_fresh_request_is_left_alone(self):
        task_state.request("meteo", "dashboard")

        assert task_state.expire_overdue_pending() == []
        assert task_state.pending_requests().keys() == {"meteo"}

    def test_a_running_task_is_never_expired_this_way(self):
        task_state.claim("meteo", "scheduler")
        assert task_state.expire_overdue_pending() == []
        assert task_state.is_running("meteo") is True


class TestSchedulerLiveness:
    def test_unknown_before_the_scheduler_has_ever_been_seen(self):
        """An installation that has not yet run this version has no such
        field; callers must not report that as "the planner is down"."""
        assert task_state.scheduler_alive() is None

    def test_alive_right_after_being_noted(self):
        task_state.note_scheduler_alive()
        assert task_state.scheduler_alive() is True

    def test_not_alive_once_the_note_is_old(self):
        task_state.note_scheduler_alive()
        state = json.loads(Path(task_state.STATE_PATH).read_text())
        state[task_state.SCHEDULER_SEEN_KEY] = (
            time.time() - task_state.PENDING_TIMEOUT_S - 1
        )
        Path(task_state.STATE_PATH).write_text(json.dumps(state))

        assert task_state.scheduler_alive() is False

    def test_the_note_survives_other_writes(self):
        task_state.note_scheduler_alive()
        granted = task_state.claim("meteo", "scheduler")
        task_state.release(granted, "done")

        assert task_state.scheduler_alive() is True
