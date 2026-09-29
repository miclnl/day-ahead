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
import time
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
        assert task_state.claim("calc_optimum", "scheduler") is True
        assert task_state.is_running("calc_optimum") is True

    def test_a_second_claim_on_the_same_task_is_refused(self):
        assert task_state.claim("calc_optimum", "scheduler") is True
        assert task_state.claim("calc_optimum", "dashboard") is False

    def test_a_different_task_may_run_at_the_same_time(self):
        """Matches what the scheduler already allowed: the optimisation does
        not have to wait for ML training. Exclusion is per task."""
        assert task_state.claim("calc_optimum", "scheduler") is True
        assert task_state.claim("train_ml_predictions", "dashboard") is True
        assert set(task_state.running_tasks()) == {
            "calc_optimum",
            "train_ml_predictions",
        }

    def test_releasing_frees_the_task_for_a_new_claim(self):
        task_state.claim("meteo", "scheduler")
        task_state.release("meteo", "done", returncode=0)
        assert task_state.is_running("meteo") is False
        assert task_state.claim("meteo", "dashboard") is True

    def test_the_source_is_recorded(self):
        task_state.claim("meteo", "scheduler")
        assert task_state.running_tasks()["meteo"]["source"] == "scheduler"


class TestRelease:
    def test_the_outcome_is_kept_so_the_dashboard_can_still_show_it(self):
        task_state.claim("prices", "dashboard", logfile="../data/log/prices_x.log")
        task_state.release("prices", "error", returncode=2)

        result = task_state.last_result("prices")
        assert result["status"] == "error"
        assert result["returncode"] == 2
        assert result["logfile"] == "../data/log/prices_x.log"
        assert result["source"] == "dashboard"
        assert result["finished"] > 0

    def test_a_logfile_discovered_while_running_wins(self):
        """Both dashboards only learn the real log file name after the task
        has started (they watch the log directory for a new file), so the
        claim is made without one and it is filled in later."""
        task_state.claim("calc_optimum", "dashboard")
        task_state.heartbeat("calc_optimum", logfile="../data/log/calc_late.log")
        task_state.release("calc_optimum", "done", returncode=0)

        assert task_state.last_result("calc_optimum")["logfile"] == (
            "../data/log/calc_late.log"
        )

    def test_releasing_a_task_that_is_not_running_is_harmless(self):
        task_state.release("meteo", "done", returncode=0)
        assert task_state.last_result("meteo")["status"] == "done"


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
        assert task_state.claim("calc_optimum", "scheduler") is True

    def test_a_heartbeat_keeps_a_long_task_alive(self):
        task_state.claim("train_ml_predictions", "scheduler")

        state = json.loads(Path(task_state.STATE_PATH).read_text())
        state["running"]["train_ml_predictions"]["heartbeat"] = (
            time.time() - task_state.STALE_AFTER_S - 1
        )
        Path(task_state.STATE_PATH).write_text(json.dumps(state))

        task_state.heartbeat("train_ml_predictions")

        assert task_state.is_running("train_ml_predictions") is True

    def test_one_tasks_heartbeat_does_not_revive_another(self):
        task_state.claim("meteo", "scheduler")
        task_state.claim("prices", "scheduler")

        state = json.loads(Path(task_state.STATE_PATH).read_text())
        state["running"]["prices"]["heartbeat"] = (
            time.time() - task_state.STALE_AFTER_S - 1
        )
        Path(task_state.STATE_PATH).write_text(json.dumps(state))

        task_state.heartbeat("meteo")

        assert task_state.is_running("meteo") is True
        assert task_state.is_running("prices") is False


class TestCancel:
    def test_cancel_sets_a_flag_the_runner_can_see(self):
        task_state.claim("calc_optimum", "dashboard")
        assert task_state.request_cancel("calc_optimum") is True
        assert task_state.cancel_requested("calc_optimum") is True

    def test_the_flag_travels_back_on_the_next_heartbeat(self):
        """A runner polling its subprocess is calling heartbeat anyway, so it
        reads the cancel flag from that same call rather than doing a second
        round trip to disk."""
        task_state.claim("calc_optimum", "dashboard")
        task_state.request_cancel("calc_optimum")

        entry = task_state.heartbeat("calc_optimum")

        assert entry["cancel"] is True

    def test_cancelling_a_task_that_is_not_running_reports_false(self):
        assert task_state.request_cancel("meteo") is False
        assert task_state.cancel_requested("meteo") is False


class TestClaimedContextManager:
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
        assert task_state.claim("calc_optimum", "scheduler") is True

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
        assert task_state.claim("calc_optimum", "dashboard") is True

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
