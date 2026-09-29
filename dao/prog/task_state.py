"""Shared, lock-protected bookkeeping of which tasks are running.

Before this module there were two independent mechanisms and they did not
know about each other:

* the scheduler process excluded a task from overlapping itself with an
  in-memory ``threading.Lock`` (``da_scheduler.py``), invisible to anything
  outside that process;
* both dashboards excluded tasks through ``../data/task_state.json``, which
  the scheduler never wrote or read.

So cron starting ``calc_optimum`` at 05:44 and a user pressing the button at
05:44 produced two optimisation runs, writing the same database tables and
pushing conflicting setpoints to Home Assistant. On top of that the v1
dashboard wrote the shared state file non-atomically while v2 treated a
``JSONDecodeError`` as "nothing is running", so a poll landing during a v1
write opened the guard as well.

Everything that starts a task now claims it here first.

Two properties matter:

* **The claim is atomic.** ``claim()`` takes an exclusive ``fcntl.flock``
  and does the read-check-write inside it, so the check-then-act race that
  let two near-simultaneous requests both pass the "is something running"
  test cannot happen, not even across processes.
* **Exclusion is per task, not global.** That matches what the scheduler
  already did: two different tasks may overlap (the optimisation should not
  have to wait for ML training), the same task never overlaps itself. A
  dashboard that wants to present "one at a time" can still do so by looking
  at :func:`running_tasks`; that is a presentation choice, not the lock.

The lock lives in its own file. The state file is rewritten with
``os.replace``, which swaps the inode, so a lock held on the state file
itself would not exclude the writer of its replacement -- the lock identity
has to be a path nobody replaces.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Optional

#: Both paths are relative to the working directory every entry point uses
#: (dao/prog for the scheduler and the task subprocesses, dao/webserver for
#: gunicorn), which is also how ../data/options.json is addressed everywhere.
STATE_PATH = "../data/task_state.json"
LOCK_PATH = "../data/task_state.lock"

#: A running task that has not been heard from for this long is treated as
#: dead and no longer blocks a new claim. A task subprocess that is killed
#: (gunicorn recycling its worker, the watchdog restarting the scheduler,
#: OOM) never gets to release its claim, and without this the task would be
#: unstartable until someone deleted the file by hand.
STALE_AFTER_S = 600

_EMPTY: dict[str, Any] = {"running": {}, "last": {}, "last_update": 0.0}


@contextmanager
def _locked():
    """Hold an exclusive lock for the duration of a read-modify-write."""
    os.makedirs(os.path.dirname(LOCK_PATH) or ".", exist_ok=True)
    # "a" so the file is created if missing and never truncated: the lock is
    # the file's existence plus the kernel lock on its descriptor, the
    # contents are irrelevant.
    with open(LOCK_PATH, "a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_raw() -> dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return dict(_EMPTY)
    except (json.JSONDecodeError, OSError) as exception:
        logging.warning(f"Taakstatus onleesbaar ({exception}), als leeg behandeld")
        return dict(_EMPTY)
    if not isinstance(data, dict):
        return dict(_EMPTY)
    if "running" not in data:
        # A state file from before this module: a single flat record of the
        # one task the dashboard was tracking. There is no reliable way to
        # know whether that task is still alive, and the old writers are
        # gone, so start clean rather than import a claim nobody will
        # release.
        return dict(_EMPTY)
    running = data.get("running")
    last = data.get("last")
    return {
        "running": running if isinstance(running, dict) else {},
        "last": last if isinstance(last, dict) else {},
        "last_update": data.get("last_update", 0.0),
    }


def _drop_stale(state: dict[str, Any], now: float) -> dict[str, Any]:
    fresh = {}
    for key, entry in state["running"].items():
        heartbeat = entry.get("heartbeat", entry.get("started", 0.0))
        if now - heartbeat > STALE_AFTER_S:
            logging.warning(
                f"Taak {key} stond nog als lopend geregistreerd maar is "
                f"{now - heartbeat:.0f} s niet meer bijgewerkt; claim vrijgegeven"
            )
            continue
        fresh[key] = entry
    state["running"] = fresh
    return state


def _write(state: dict[str, Any]) -> None:
    state["last_update"] = time.time()
    temp_path = STATE_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, STATE_PATH)


def read() -> dict[str, Any]:
    """The current state, with dead claims already filtered out.

    Takes no lock: the state file is only ever replaced atomically, so a
    reader sees either the previous or the next version, never a half-written
    one.
    """
    return _drop_stale(_read_raw(), time.time())


def running_tasks() -> dict[str, Any]:
    """The tasks that currently hold a claim, keyed by canonical task key."""
    return read()["running"]


def is_running(task_key: str) -> bool:
    return task_key in running_tasks()


def claim(
    task_key: str,
    source: str,
    logfile: Optional[str] = None,
    pid: Optional[int] = None,
) -> bool:
    """Register *task_key* as running, unless it already is.

    Returns True when the caller may start the task and is now responsible
    for calling :func:`release` (and :func:`heartbeat` while it runs).
    Returns False when somebody else holds the claim.

    *source* is recorded purely so a log line or the dashboard can say where
    a run came from ("scheduler", "dashboard", "api").
    """
    now = time.time()
    with _locked():
        state = _drop_stale(_read_raw(), now)
        if task_key in state["running"]:
            return False
        state["running"][task_key] = {
            "started": now,
            "heartbeat": now,
            "source": source,
            "logfile": logfile,
            "pid": pid,
            "cancel": False,
        }
        _write(state)
    return True


def heartbeat(task_key: str, logfile: Optional[str] = None) -> dict[str, Any]:
    """Refresh the claim on *task_key* so it is not mistaken for dead.

    Returns the task's entry as it is on disk after the update, so a runner
    that is polling anyway can read the cancel flag from the same call
    instead of a second round trip.
    """
    with _locked():
        state = _read_raw()
        entry = state["running"].get(task_key)
        if entry is None:
            return {}
        entry["heartbeat"] = time.time()
        if logfile is not None:
            entry["logfile"] = logfile
        _write(state)
        return dict(entry)


def release(
    task_key: str,
    status: str,
    returncode: Optional[int] = None,
    logfile: Optional[str] = None,
) -> None:
    """Give up the claim on *task_key* and record how the run ended.

    *status* is one of "done", "error" or "cancelled". The outcome is kept
    under "last" so the dashboard can still show the result of a task that
    is no longer running.
    """
    with _locked():
        state = _read_raw()
        entry = state["running"].pop(task_key, {})
        state["last"][task_key] = {
            "status": status,
            "returncode": returncode,
            "finished": time.time(),
            "started": entry.get("started"),
            "source": entry.get("source"),
            "logfile": logfile if logfile is not None else entry.get("logfile"),
        }
        _write(state)


def request_cancel(task_key: str) -> bool:
    """Ask the process running *task_key* to stop.

    Only sets a flag; the runner notices it on its next heartbeat and kills
    its own subprocess. Returns False when the task is not running.
    """
    with _locked():
        state = _read_raw()
        entry = state["running"].get(task_key)
        if entry is None:
            return False
        entry["cancel"] = True
        _write(state)
    return True


def cancel_requested(task_key: str) -> bool:
    entry = running_tasks().get(task_key)
    return bool(entry and entry.get("cancel"))


def last_result(task_key: str) -> dict[str, Any]:
    """How the previous run of *task_key* ended, or an empty dict."""
    return read()["last"].get(task_key, {})


@contextmanager
def claimed(task_key: str, source: str, logfile: Optional[str] = None):
    """Hold a claim for the duration of a block.

    Yields True when the claim was granted and releases it afterwards,
    recording "error" if the block raised. Yields False when the task was
    already running, and then does not touch the state at all -- releasing a
    claim somebody else owns would hand their task's slot away while it is
    still running.
    """
    granted = claim(task_key, source, logfile=logfile)
    if not granted:
        yield False
        return
    try:
        yield True
    except BaseException:
        release(task_key, "error")
        raise
    else:
        release(task_key, "done")
