"""The dashboards and the scheduler exclude each other now.

This is the bug the shared claim exists for: the scheduler used an
in-process threading.Lock and the dashboards used ../data/task_state.json,
which the scheduler never touched. Cron starting calc_optimum at 05:44 and
a user pressing the button at 05:44 produced two optimisation runs, writing
the same database tables and pushing conflicting setpoints to Home
Assistant.

Both dashboards are driven through their real HTTP routes here, so the test
covers the guard as a request actually hits it rather than the helper
underneath.
"""

import pytest

from dao.prog import task_state
from dao.prog import tasks as task_registry

from .conftest import INGRESS, SUPERVISOR, csrf_token


@pytest.fixture(autouse=True)
def no_threads(client, monkeypatch):
    """Keep the routes from actually spawning tasks.

    These tests are about the guard, not about running anything; a real
    thread would start a real day_ahead.py subprocess.
    """
    import importlib

    started = []
    for module_name in ("app.routes", "app.v2.routes"):
        module = importlib.import_module(module_name)

        class FakeThread:
            def __init__(self, target=None, args=(), daemon=None, **kwargs):
                started.append(args)

            def start(self):
                pass

        monkeypatch.setattr(module.threading, "Thread", FakeThread)
    return started


class TestV2Dashboard:
    def test_a_task_the_scheduler_is_running_cannot_be_started(self, client):
        task_state.claim("calc_optimum", source="scheduler")

        response = client.post(
            "/v2/task-exec",
            data={
                "task": "optimize_regular",
                "csrf_token": csrf_token(client, "/v2/tasks"),
            },
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert response.status_code == 409
        assert b"scheduler" in response.data

    def test_a_free_task_starts_and_is_claimed(self, client, no_threads):
        response = client.post(
            "/v2/task-exec",
            data={
                "task": "optimize_regular",
                "csrf_token": csrf_token(client, "/v2/tasks"),
            },
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert response.status_code in (302, 303)
        assert task_state.is_running("calc_optimum") is True
        assert task_state.running_tasks()["calc_optimum"]["source"] == "dashboard"
        # The canonical key is what gets claimed, whichever alias was posted.
        assert no_threads and no_threads[-1][1] == "calc_optimum"

    def test_a_different_task_may_run_alongside(self, client):
        task_state.claim("calc_optimum", source="scheduler")

        response = client.post(
            "/v2/task-exec",
            data={
                "task": "update_meteo",
                "csrf_token": csrf_token(client, "/v2/tasks"),
            },
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert response.status_code in (302, 303)
        assert set(task_state.running_tasks()) == {"calc_optimum", "meteo"}

    def test_an_unknown_task_is_rejected(self, client):
        response = client.post(
            "/v2/task-exec",
            data={
                "task": "definitely_not_a_task",
                "csrf_token": csrf_token(client, "/v2/tasks"),
            },
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert response.status_code == 400
        assert task_state.running_tasks() == {}

    def test_cancel_flags_the_running_task(self, client):
        task_state.claim("calc_optimum", source="dashboard")

        response = client.get(
            "/v2/task-cancel", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        assert task_state.cancel_requested("calc_optimum") is True


class TestV2Api:
    def test_it_refuses_a_task_that_is_already_running(self, client):
        task_state.claim("calc_optimum", source="scheduler")

        response = client.get(
            "/v2/api/run/calc_optimum", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 409
        assert b"scheduler" in response.data

    def test_an_unknown_task_is_a_404(self, client):
        response = client.get(
            "/v2/api/run/nope", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 404


class TestV1Api:
    def test_it_refuses_a_task_that_is_already_running(self, client):
        task_state.claim("calc_optimum", source="scheduler")

        response = client.get(
            "/api/run/calc_zonder_debug", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 409

    def test_a_historical_alias_still_resolves(self, client, monkeypatch):
        """/api/run/<key> URLs are the kind of thing people bookmark or wire
        into an automation, so every old key has to keep working."""
        import importlib

        routes = importlib.import_module("app.routes")
        calls = []

        class Result:
            stdout = "ok"
            stderr = ""
            returncode = 0

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return Result()

        monkeypatch.setattr(routes, "run", fake_run)

        response = client.get(
            "/api/run/get_meteo", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        assert calls == [task_registry.get("meteo")["cmd"]]
        # Claim released afterwards, so the next call is not blocked.
        assert task_state.is_running("meteo") is False

    def test_an_unknown_task_is_a_404(self, client):
        response = client.get(
            "/api/run/nope", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 404


class TestApiTimeoutStaysUnderGunicorn:
    def test_the_cap_is_applied_and_reported(self, client, monkeypatch):
        """Gunicorn kills a worker that does not answer within 120 s. The old
        code capped at 300 s, so the worker died first and the run ended as a
        dead worker plus an orphaned child with no output at all."""
        import importlib
        from subprocess import TimeoutExpired

        routes = importlib.import_module("app.routes")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            raise TimeoutExpired(cmd, kwargs.get("timeout"), output="partial")

        monkeypatch.setattr(routes, "run", fake_run)

        response = client.get(
            "/api/run/get_meteo", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        assert seen["timeout"] == task_registry.API_RUN_TIMEOUT_S
        assert seen["timeout"] < 120
        assert task_state.is_running("meteo") is False
