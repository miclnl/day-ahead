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

This module owns requests/run identities, not the execution mutex. Actual
writers must hold the process-lifetime lease owned by task_execution.

Two properties matter:

* **The claim is atomic.** ``claim()`` takes an exclusive ``fcntl.flock``
  and does the read-check-write inside it, so the check-then-act race that
  let two near-simultaneous requests both pass the "is something running"
  test cannot happen, not even across processes.
* **Exclusion is per registry resource, not global.** Different tasks may
  overlap (the optimisation should not have to wait for ML training), but
  regular/debug optimisation share their writer resource. A
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
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from dao.prog import task_execution
from dao.prog import tasks as task_registry

#: Both paths are relative to the working directory every entry point uses
#: (dao/prog for the scheduler and the task subprocesses, dao/webserver for
#: gunicorn), which is also how ../data/options.json is addressed everywhere.
STATE_PATH = "../data/task_state.json"
LOCK_PATH = "../data/task_state.lock"

#: After this age a claim is unconfirmed, NOT dead. Recovery additionally
#: requires a successful OS writer-lease acquire; old unleased writers require
#: a quiescent upgrade/explicit termination, never timeout migration.
STALE_AFTER_S = 600

#: A request that the scheduler has not picked up within this many seconds
#: is reported as failed. Much shorter than STALE_AFTER_S on purpose: the
#: scheduler polls every few seconds, so anything longer than this means it
#: is not running, and leaving the dashboard on "wordt gestart" for ten
#: minutes tells the operator nothing.
PENDING_TIMEOUT_S = 60

#: Written by the scheduler while it polls for requests, so the dashboard can
#: say "the planner is not running" instead of letting a request sit there.
SCHEDULER_SEEN_KEY = "scheduler_seen"


@dataclass(frozen=True)
class TaskClaim:
    """Identity of one claim, retained by its runner until release."""

    task_key: str
    run_id: str


def _empty_state() -> dict[str, Any]:
    # Never share nested mappings between independent state files.
    return {"running": {}, "last": {}, "last_update": 0.0}


def _matching_entry(state: dict[str, Any], claim: TaskClaim) -> Optional[dict[str, Any]]:
    """Match both identity fields; mutation callers hold the sidecar lock."""
    if not isinstance(claim, TaskClaim) or not claim.run_id:
        return None
    entry = state["running"].get(claim.task_key)
    if entry is None or entry.get("run_id") != claim.run_id:
        return None
    return entry


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


def _finish(
    state: dict[str, Any],
    claim: TaskClaim,
    status: str,
    returncode: Optional[int] = None,
    logfile: Optional[str] = None,
    message: Optional[str] = None,
) -> bool:
    """Take the claim out of the state and record how the run ended.

    Only the run that holds the identity may finish it, so a stale runner
    cannot record a result for somebody else's run. The caller holds the
    lock.
    """
    entry = _matching_entry(state, claim)
    if entry is None:
        return False
    state["running"].pop(claim.task_key)
    result = {
        "run_id": claim.run_id,
        "status": status,
        "returncode": returncode,
        "finished": time.time(),
        "started": entry.get("started"),
        "source": entry.get("source"),
        "logfile": logfile if logfile is not None else entry.get("logfile"),
    }
    if message is not None:
        result["message"] = message
    state["last"][claim.task_key] = result
    return True


def _read_raw() -> dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return _empty_state()
    except (json.JSONDecodeError, OSError) as exception:
        logging.warning(f"Taakstatus onleesbaar ({exception}), als leeg behandeld")
        return _empty_state()
    if not isinstance(data, dict):
        return _empty_state()
    if "running" not in data:
        result = _empty_state()
        if data.get("status") == "running":
            # No lease evidence: even a free sidecar cannot prove this old
            # writer stopped. Preserve the barrier until quiescent upgrade.
            key = task_registry.resolve(data.get("task", "")) or "*"
            result["running"][key] = {**data, "state": "unconfirmed"}
        return result
    running = data.get("running")
    last = data.get("last")
    result = {
        "running": running if isinstance(running, dict) else {},
        "last": last if isinstance(last, dict) else {},
        "last_update": data.get("last_update", 0.0),
    }
    if data.get(SCHEDULER_SEEN_KEY):
        result[SCHEDULER_SEEN_KEY] = data[SCHEDULER_SEEN_KEY]
    return result


def _age_limit(entry: dict[str, Any]) -> float:
    """How long *entry* may go without an update before it is unconfirmed.

    A pending request gets much less rope than a running task: the
    scheduler polls every few seconds, so a request still pending after a
    minute means nobody is going to pick it up.
    """
    if entry.get("state") == "pending":
        return PENDING_TIMEOUT_S
    return STALE_AFTER_S


def _drop_stale(state: dict[str, Any], now: float) -> dict[str, Any]:
    for key, entry in state["running"].items():
        heartbeat = entry.get("heartbeat", entry.get("started", 0.0))
        limit = _age_limit(entry)
        if now - heartbeat > limit and entry.get("state") != "unconfirmed":
            logging.warning(
                f"Taak {key} is {now - heartbeat:.0f} s niet bijgewerkt; "
                "claim onbevestigd, writerlease blijft leidend"
            )
            entry["state"] = "unconfirmed"
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
    """The current state, with stale claims marked unconfirmed, never erased.

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


def _resource(task_key: str) -> str:
    return task_registry.resource_key(task_key) or task_key


def _resource_entries(state: dict[str, Any], task_key: str) -> dict[str, Any]:
    return {
        key: entry for key, entry in state["running"].items()
        if key == "*" or _resource(key) == _resource(task_key)
    }


def upgrade_safe(task_key: str) -> bool:
    """Read-only legacy barrier; a free sidecar cannot prove old writer exit.

    Like read(), this observes an atomic state snapshot. The caller already
    holds the writer lease, so compatible claim writers cannot acquire their
    probe. Do not create a bookkeeping lock merely to inspect a direct CLI.
    """
    return all(
        entry.get("lease_protocol") == task_execution.LEASE_PROTOCOL
        for entry in _resource_entries(_read_raw(), task_key).values()
    )


def execution_allowed(claim: TaskClaim) -> bool:
    """Revalidate identity AFTER acquiring the lease, before spawning a writer.

    An arbitrarily delayed runner must not execute a superseded claim just
    because it can eventually acquire the OS slot.
    """
    with _locked():
        current = _read_raw()
        entry = _matching_entry(current, claim)
        return bool(
            entry and entry.get("state") != "pending"
            and all(
                item.get("lease_protocol") == task_execution.LEASE_PROTOCOL
                for item in _resource_entries(current, claim.task_key).values()
            )
        )


def claim(
    task_key: str,
    source: str,
    logfile: Optional[str] = None,
    pid: Optional[int] = None,
    state_name: str = "running",
    parameters: Optional[dict[str, Any]] = None,
) -> TaskClaim | None:
    """Register *task_key* as claimed, unless it already is.

    Returns its identity when the caller may start the task and is responsible
    for calling :func:`release` (and :func:`heartbeat` while it runs).
    Returns None when somebody else holds the claim. Updates must use the
    returned identity, never a fresh lookup by task name.

    *source* is recorded purely so a log line or the dashboard can say where
    a run came from ("scheduler", "dashboard", "api").

    *state_name* is "running" for a caller that is about to do the work
    itself, or "pending" for a request the scheduler still has to pick up
    (see :func:`request`). Either way the claim blocks everyone else, so the
    exclusion is the same.

    A grant is bookkeeping, not permission for task I/O without the writer
    lease. Unconfirmed claims can be replaced only with a successful OS probe.
    """
    now = time.time()
    with _locked():
        current = _drop_stale(_read_raw(), now)
        conflicting = _resource_entries(current, task_key)
        if any(
            entry.get("state") != "unconfirmed"
            or entry.get("lease_protocol") != task_execution.LEASE_PROTOCOL
            for entry in conflicting.values()
        ):
            return None
        probe = None
        try:
            if task_registry.resolve(task_key) is not None:
                try:
                    probe = task_execution.acquire(task_key)
                except task_execution.LeaseBusy:
                    return None
            elif conflicting:
                return None  # Unknown resources have no recoverable OS lease.
            # Hold the successful OS probe through the state replacement.
            for key in conflicting:
                current["running"].pop(key)
            granted = TaskClaim(task_key, uuid4().hex)
            current["running"][task_key] = {
                "run_id": granted.run_id,
                "state": state_name,
                "started": now,
                "heartbeat": now,
                "source": source,
                "logfile": logfile,
                "pid": pid,
                "cancel": False,
                "parameters": parameters or {},
                "lease_protocol": task_execution.LEASE_PROTOCOL,
            }
            _write(current)
        finally:
            if probe is not None:
                probe.close()
    return granted


def request(
    task_key: str,
    source: str,
    parameters: Optional[dict[str, Any]] = None,
) -> TaskClaim | None:
    """Ask the scheduler to run *task_key*.

    The dashboards used to run tasks themselves, in a daemon thread inside a
    gunicorn worker. That worker is recycled on every configuration change
    (the watchdog sends it a HUP), which killed the thread while its
    subprocess kept running, with nothing left to notice the result. And
    with two workers, a request could land in either one.

    Now they record what should happen and the scheduler process, which is
    long-lived and already owns task execution for the cron schedule, does
    it. Returns the claim identity, or None when the task is already claimed.
    """
    return claim(task_key, source, state_name="pending", parameters=parameters)


def pending_requests() -> dict[str, Any]:
    """Claims that are waiting for the scheduler to pick them up."""
    return {
        key: entry
        for key, entry in running_tasks().items()
        if entry.get("state") == "pending"
    }


def take_pending(claim: TaskClaim) -> Optional[dict[str, Any]]:
    """Move this exact claim from pending to running and return its entry.

    Only succeeds for a claim that is still pending, so two scheduler
    threads polling at the same time cannot both start the same request.
    Returns None when there was nothing to take.
    """
    now = time.time()
    with _locked():
        current = _drop_stale(_read_raw(), now)
        entry = _matching_entry(current, claim)
        if (entry is None or entry.get("state") != "pending"
                or entry.get("lease_protocol") != task_execution.LEASE_PROTOCOL):
            return None
        entry["state"] = "running"
        entry["heartbeat"] = now
        entry["taken"] = now
        _write(current)
        return dict(entry)


def expire_overdue_pending() -> list[str]:
    """Record overdue requests as failed and give up their claims.

    Called by the scheduler when it resumes polling. A request that has been
    waiting this long means the scheduler was down while it was made, and
    silently running it now would fire an optimisation the operator asked
    for twenty minutes ago. Recording it as failed is both safer and more
    informative than letting the claim quietly expire, which would leave the
    dashboard showing nothing at all.

    Returns the task keys that were expired.
    """
    now = time.time()
    expired = []
    changed = False
    cancelled_keys = set()
    with _locked(), ExitStack() as probes:
        current = _read_raw()
        for key, entry in list(current["running"].items()):
            if entry.get("state") != "pending":
                continue
            if entry.get("lease_protocol") != task_execution.LEASE_PROTOCOL:
                entry["state"] = "unconfirmed"
                changed = True
                continue
            waited = now - entry.get("heartbeat", entry.get("started", now))
            if waited <= PENDING_TIMEOUT_S:
                continue
            if task_registry.resolve(key) is not None:
                try:
                    probe = task_execution.acquire(key)
                except task_execution.LeaseBusy:
                    entry["state"] = "unconfirmed"
                    changed = True
                    continue
                probes.callback(probe.close)
            current["running"].pop(key)
            # A request that was cancelled while it was waiting did not
            # fail; it was called off. An earlier version only set the
            # flag, so a claim can still carry it.
            cancelled = bool(entry.get("cancel"))
            if cancelled:
                cancelled_keys.add(key)
            current["last"][key] = {
                "run_id": entry.get("run_id"),
                "status": "cancelled" if cancelled else "error",
                "returncode": None,
                "finished": now,
                "started": entry.get("started"),
                "source": entry.get("source"),
                "logfile": None,
                "message": (
                    "Geannuleerd voordat de planner hem oppakte."
                    if cancelled
                    else (
                        f"Niet opgepakt binnen {PENDING_TIMEOUT_S} s; "
                        f"de planner draaide op dat moment niet."
                    )
                ),
            }
            expired.append(key)
        if expired or changed:
            _write(current)
    for key in expired:
        if key in cancelled_keys:
            logging.warning(
                f"Aanvraag voor taak {key} is geannuleerd en verlopen; "
                f"hij wordt niet meer uitgevoerd."
            )
        else:
            logging.warning(
                f"Aanvraag voor taak {key} is verlopen: de planner heeft hem niet "
                f"opgepakt en hij wordt niet meer uitgevoerd."
            )
    return expired


def note_scheduler_alive() -> None:
    """Record that the scheduler is polling, for :func:`scheduler_alive`."""
    with _locked():
        current = _read_raw()
        current[SCHEDULER_SEEN_KEY] = time.time()
        _write(current)


def scheduler_alive(within_s: float = PENDING_TIMEOUT_S) -> Optional[bool]:
    """Whether the scheduler has been seen polling recently.

    Returns None when it has never been seen at all, which is what an
    installation that has not yet run this version looks like -- callers
    should treat that as "unknown" rather than "down" and not scare the
    operator with a warning that is really about a missing field.
    """
    seen = read().get(SCHEDULER_SEEN_KEY)
    if not seen:
        return None
    return (time.time() - seen) <= within_s


def heartbeat(claim: TaskClaim, logfile: Optional[str] = None) -> dict[str, Any]:
    """Refresh this exact claim so it is not mistaken for dead.

    Returns the task's entry as it is on disk after the update, so a runner
    that is polling anyway can read the cancel flag from the same call
    instead of a second round trip.
    """
    with _locked():
        state = _read_raw()
        entry = _matching_entry(state, claim)
        if entry is None:
            return {}
        entry["heartbeat"] = time.time()
        if logfile is not None:
            entry["logfile"] = logfile
        _write(state)
        return dict(entry)


def release(
    claim: TaskClaim,
    status: str,
    returncode: Optional[int] = None,
    logfile: Optional[str] = None,
) -> None:
    """Finish this exact claim only after the OS writer resource is free.

    *status* is one of "done", "error" or "cancelled". The outcome is kept
    under "last" so the dashboard can still show the result of a task that
    is no longer running. A missing or superseded identity is a no-op, not
    a result for a run the caller does not own. A still-held writer lease
    keeps the claim unconfirmed instead (also after a leader/runner error).
    """
    with _locked():
        state = _read_raw()
        entry = _matching_entry(state, claim)
        if entry is None:
            return
        if entry.get("lease_protocol") != task_execution.LEASE_PROTOCOL:
            entry["state"] = "unconfirmed"
            _write(state)
            return
        probe = None
        try:
            if task_registry.resolve(claim.task_key) is not None:
                try:
                    probe = task_execution.acquire(claim.task_key)
                except task_execution.LeaseBusy:
                    # A scheduler error/leader exit is not writer exit.
                    # Preserve the exact identity and exclusion, even if an
                    # orphan or writing descendant is all that remains.
                    entry["state"] = "unconfirmed"
                    _write(state)
                    return
            if _finish(state, claim, status, returncode=returncode, logfile=logfile):
                _write(state)
        finally:
            if probe is not None:
                probe.close()


def request_cancel(claim: TaskClaim) -> bool:
    """Ask the runner holding this exact claim to stop.

    A request the scheduler has not picked up yet is finished here. That
    happens under the same lock as the pickup, so the two cannot both
    win: whoever gets the slot first owns the claim. A cancelled request
    is never spawned, and it ends as "cancelled" rather than as a failure.

    A claim that is already running only gets the flag, which its runner
    notices on the next heartbeat and answers with a process group stop.

    Returns False for a missing or superseded identity.
    """
    with _locked():
        state = _read_raw()
        entry = _matching_entry(state, claim)
        if entry is None:
            return False
        if entry.get("lease_protocol") != task_execution.LEASE_PROTOCOL:
            entry["state"] = "unconfirmed"
            entry["cancel"] = True
        elif entry.get("state") == "pending":
            if not _finish(
                state,
                claim,
                "cancelled",
                message="Geannuleerd voordat de planner hem oppakte.",
            ):
                return False
        else:
            entry["cancel"] = True
        _write(state)
    return True


def cancel_requested(claim: TaskClaim) -> bool:
    entry = _matching_entry(read(), claim)
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
        release(granted, "error")
        raise
    else:
        release(granted, "done")
