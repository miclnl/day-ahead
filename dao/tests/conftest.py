"""Suite-wide safety nets.

dao.prog.task_state addresses its files relative to the working directory
(``../data/task_state.json``), the same way every entry point addresses
``../data/options.json``. Under pytest the working directory is the
repository root, so that relative path points *outside* the repository, at
the operator's real data directory.

Any test that reaches code taking a task claim -- directly, or indirectly
through DaScheduler._run_exclusive or a dashboard route -- would write
there. Redirecting the paths for every test makes that impossible instead of
relying on each test to remember.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_task_state(tmp_path, monkeypatch):
    from dao.prog import task_state

    monkeypatch.setattr(task_state, "STATE_PATH", str(tmp_path / "task_state.json"))
    monkeypatch.setattr(task_state, "LOCK_PATH", str(tmp_path / "task_state.lock"))
