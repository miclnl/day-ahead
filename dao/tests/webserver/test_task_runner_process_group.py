"""run_and_log spawns a task as its own session leader and, on cancel,
kills the whole process group rather than just the direct child.

Without start_new_session=True a task that forks or execs a helper of its
own could leave that helper running as an orphan after a cancel. The kill
itself lives in dao.prog.tasks because both dashboards cancel tasks and the
two copies had started to differ.

These tests drive the v2 runner, which is the one with a cancel path.
"""

import os
import signal

import pytest

from dao.prog import task_state
from dao.prog import tasks as task_registry


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
def v2_routes(client):
    import importlib

    return importlib.import_module("app.v2.routes")


def test_run_and_log_starts_a_new_session(v2_routes, monkeypatch):
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return FakeProc(cmd, **kwargs)

    monkeypatch.setattr(v2_routes, "Popen", fake_popen)
    monkeypatch.setattr(v2_routes.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(v2_routes, "get_file_list_with_ts", lambda path, pattern: [])

    task_state.claim("meteo", "dashboard")
    v2_routes.run_and_log(["python3", "noop.py"], "meteo")

    assert captured.get("start_new_session") is True


def test_the_claim_is_released_when_the_task_finishes(v2_routes, monkeypatch):
    monkeypatch.setattr(v2_routes, "Popen", lambda cmd, **kw: FakeProc(cmd, **kw))
    monkeypatch.setattr(v2_routes.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(v2_routes, "get_file_list_with_ts", lambda path, pattern: [])

    task_state.claim("meteo", "dashboard")
    v2_routes.run_and_log(["python3", "noop.py"], "meteo")

    assert task_state.is_running("meteo") is False
    assert task_state.last_result("meteo")["status"] == "done"


def test_the_claim_is_released_when_the_task_crashes(v2_routes, monkeypatch):
    """Without this a crashed run would keep its claim until the staleness
    window expired, blocking the task for ten minutes."""

    def boom(cmd, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(v2_routes, "Popen", boom)
    monkeypatch.setattr(v2_routes, "get_file_list_with_ts", lambda path, pattern: [])

    task_state.claim("meteo", "dashboard")
    with pytest.raises(OSError):
        v2_routes.run_and_log(["python3", "noop.py"], "meteo")

    assert task_state.is_running("meteo") is False
    assert task_state.last_result("meteo")["status"] == "error"


def test_cancel_kills_the_whole_process_group(v2_routes, monkeypatch, tmp_path):
    fake_proc = FakeProc(["python3", "noop.py"], start_new_session=True)
    monkeypatch.setattr(v2_routes, "Popen", lambda cmd, **kw: fake_proc)
    monkeypatch.setattr(v2_routes.time, "sleep", lambda seconds: None)

    logfile = tmp_path / "run.log"
    logfile.write_text("partial output")
    log_dir = str(tmp_path)
    monkeypatch.setattr(v2_routes, "app_datapath", str(tmp_path.parent))
    monkeypatch.setattr(
        v2_routes,
        "get_file_list_with_ts",
        lambda path, pattern: [{"name": logfile.name}],
    )

    killed = {}
    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: killed.update(pgid=pgid, sig=sig)
    )
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    task_state.claim("meteo", "dashboard")
    task_state.request_cancel("meteo")
    v2_routes.run_and_log(["python3", "noop.py"], "meteo")

    assert killed == {"pgid": fake_proc.pid, "sig": signal.SIGKILL}
    assert task_state.last_result("meteo")["status"] == "cancelled"


def test_kill_process_group_swallows_already_exited_process(monkeypatch):
    class DeadProc:
        pid = 99999

    def raise_lookup(pid):
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", raise_lookup)

    # Must not raise even though the process is already gone.
    task_registry.kill_process_group(DeadProc())


def test_kill_process_group_logs_other_os_errors(monkeypatch, caplog):
    class Proc:
        pid = 1

    def raise_perm(pid):
        raise PermissionError("not allowed")

    monkeypatch.setattr(os, "getpgid", raise_perm)

    task_registry.kill_process_group(Proc())

    assert "procesgroep" in caplog.text
