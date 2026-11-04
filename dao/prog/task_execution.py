"""Process-lifetime writer leases, not heartbeat/PID-based execution mutexes.

The stable sidecar must NEVER be unlinked or replaced. Linux flock belongs to
the open file description: fork/dup/pass_fds share it. Close only our own fd;
LOCK_UN would also unlock a still-writing child. A writing descendant must
retain this fd across fork/exec, or explicitly use Lease.child_kwargs() for
Popen (whose close_fds otherwise discards it). Descendants without the lease
must not perform task I/O. No actuator fencing or settlement CAS is implied.

Upgrade requires quiescence/explicit termination of old unleased writers.
There is deliberately no timeout-based migration for those processes.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from dao.prog import tasks

LEASE_DIR = "../data/task_lease"
LEASE_PROTOCOL = 1
ENV_FD = "DAO_TASK_LEASE_FD"
ENV_RESOURCE = "DAO_TASK_LEASE_RESOURCE"
_active: ContextVar[Lease | None] = ContextVar("task_writer_lease", default=None)


class LeaseError(RuntimeError):
    """Fail closed before task I/O when a lease cannot be established."""


class LeaseBusy(LeaseError):
    """Another open file description owns the resource."""


def lease_path(task_key: str) -> Path:
    resource = tasks.resource_key(task_key)
    if resource is None or Path(resource).name != resource:
        raise LeaseError(f"Onbekende taakresource: {task_key}")
    return Path(LEASE_DIR).absolute() / f"{resource}.lock"


@dataclass
class Lease:
    resource: str
    fd: int

    def close(self) -> None:
        """Drop just this reference; a child/descendant may still own the lease."""
        if self.fd >= 0:
            fd, self.fd = self.fd, -1
            os.close(fd)

    def child_kwargs(self) -> dict:
        """Use at every Popen/run boundary that can produce a task writer."""
        if self.fd < 0:
            raise LeaseError("Writerlease is al gesloten")
        env = dict(os.environ)
        env[ENV_FD] = str(self.fd)
        env[ENV_RESOURCE] = self.resource
        return {"pass_fds": (self.fd,), "env": env}


def acquire(task_key: str) -> Lease:
    """Nonblocking OS acquire, also the authoritative recovery probe."""
    from dao.prog.da_base import DaBase

    path = lease_path(task_key)
    # Bootstrap the app data link before mkdir can occupy its pathname.
    # Derive it from the lease directory so injected sandboxes stay isolated.
    DaBase.make_data_path(path.parent.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise LeaseBusy(f"Taakresource {tasks.resource_key(task_key)} is bezet") from exc
    except BaseException:
        os.close(fd)
        raise
    return Lease(tasks.resource_key(task_key), fd)


def acquire_resource(resource: str, *, lease_dir: Path) -> Lease:
    """Acquire a stable generic sidecar; the caller owns its lifetime."""
    if type(resource) is not str or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", resource) is None:
        raise LeaseError("Ongeldige resource")
    path = Path(lease_dir).absolute() / f"{resource}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise LeaseBusy(f"Resource {resource} is bezet") from exc
    except BaseException:
        os.close(fd)
        raise
    return Lease(resource, fd)


def _inherited(task_key: str) -> Lease | None:
    value = os.environ.get(ENV_FD)
    resource = os.environ.get(ENV_RESOURCE)
    if value is None and resource is None:
        return None
    if resource is not None and resource != tasks.resource_key(task_key):
        # A multi-task CLI may still hold its bootstrap/first-task lease.
        # This descriptor belongs to another resource, not a corrupt lease
        # for the requested one. Acquire the requested resource separately.
        return None
    try:
        if resource != tasks.resource_key(task_key):
            raise ValueError("verkeerde resource")
        fd = int(value)
        if fd < 0:
            raise ValueError("negatieve descriptor")
        held, expected = os.fstat(fd), os.stat(lease_path(task_key))
        if not stat.S_ISREG(held.st_mode) or (held.st_dev, held.st_ino) != (
            expected.st_dev, expected.st_ino
        ):
            raise ValueError("verkeerd sidecar-inode")
        # Same inherited OFD: this does not take a second lease. A bogus env
        # referring to an independently opened fd cannot bypass another writer.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, TypeError, ValueError) as exc:
        raise LeaseError("Ongeldige geërfde writerlease") from exc
    return Lease(resource, fd)


@contextmanager
def execution(task_key: str):
    """Reuse an inherited/nested lease, or acquire before direct CLI task I/O.

    CLI dispatch is single-threaded. Its environment and inheritable fd follow
    a fork/exec descendant; Popen descendants must use child_kwargs explicitly.
    Scheduler threads only use acquire/child_kwargs and never mutate os.environ.
    """
    from dao.prog import task_state

    current = _active.get()
    if current is not None and current.resource == tasks.resource_key(task_key):
        yield current
        return
    lease = _inherited(task_key) or acquire(task_key)
    previous = {key: os.environ.get(key) for key in (ENV_FD, ENV_RESOURCE)}
    token = None
    try:
        if not task_state.upgrade_safe(task_key):
            raise LeaseError("Oude writer zonder lease: quiescent upgrade/expliciete beëindiging vereist")
        # os.exec* preserves this descriptor; close_fds Popen needs pass_fds.
        os.set_inheritable(lease.fd, True)
        os.environ[ENV_FD] = str(lease.fd)
        os.environ[ENV_RESOURCE] = lease.resource
        token = _active.set(lease)
        yield lease
    finally:
        if token is not None:
            _active.reset(token)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        lease.close()
