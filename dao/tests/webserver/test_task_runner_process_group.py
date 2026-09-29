"""run_and_log() spawns the requested script as its own session leader and,
on cancellation, kills the whole process group rather than just the direct
child. Without start_new_session=True a script that forks or execs a helper
of its own could leave that helper running as an orphan after "cancel"."""

import os
import signal


class FakeProc:
    """Stands in for subprocess.Popen: exits immediately unless kill()'d."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode = 0
        self._killed = False
        self._polls = 0

    def poll(self):
        # First poll: still running, so run_and_log takes the cancellation
        # branch. After that: finished.
        self._polls += 1
        if self._killed:
            return 0
        return None if self._polls == 1 else 0

    def wait(self):
        return self.returncode

    def kill(self):
        self._killed = True


def test_run_and_log_starts_a_new_session(client, monkeypatch):
    import importlib

    v2_routes = importlib.import_module("app.v2.routes")

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return FakeProc(cmd, **kwargs)

    monkeypatch.setattr(v2_routes, "Popen", fake_popen)
    monkeypatch.setattr(
        v2_routes, "get_task_state", lambda: {"status": "idle", "logfile": None}
    )
    monkeypatch.setattr(v2_routes, "save_task_state", lambda state: None)
    monkeypatch.setattr(v2_routes.time, "sleep", lambda seconds: None)

    v2_routes.run_and_log(
        ["python3", "noop.py"], {"logfile": None, "status": "running"}
    )

    assert captured.get("start_new_session") is True


def test_cancel_kills_the_whole_process_group(client, monkeypatch, tmp_path):
    import importlib

    v2_routes = importlib.import_module("app.v2.routes")

    fake_proc = FakeProc(["python3", "noop.py"], start_new_session=True)
    monkeypatch.setattr(v2_routes, "Popen", lambda cmd, **kw: fake_proc)
    monkeypatch.setattr(
        v2_routes,
        "get_task_state",
        lambda: {"status": "cancelled", "logfile": str(logfile)},
    )
    monkeypatch.setattr(v2_routes, "save_task_state", lambda state: None)

    killed = {}

    def fake_killpg(pgid, sig):
        killed["pgid"] = pgid
        killed["sig"] = sig

    monkeypatch.setattr(os, "killpg", fake_killpg)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    logfile = tmp_path / "run.log"
    logfile.write_text("partial output")

    v2_routes.run_and_log(["python3", "noop.py"], {"logfile": str(logfile)})

    assert killed == {"pgid": fake_proc.pid, "sig": signal.SIGKILL}
    assert not logfile.exists()  # cancelled tasks drop their partial log


def test_kill_process_group_swallows_already_exited_process(client, monkeypatch):
    import importlib

    v2_routes = importlib.import_module("app.v2.routes")

    class DeadProc:
        pid = 99999

    def raise_lookup(pid):
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", raise_lookup)

    # Must not raise even though the process is already gone.
    v2_routes._kill_process_group(DeadProc())
