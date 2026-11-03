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

import importlib
import json
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from dao.prog import da_scheduler, task_state
from dao.prog import tasks as task_registry
from dao.prog.da_base import DaBase

from .conftest import INGRESS, SUPERVISOR, csrf_token


@pytest.fixture(autouse=True)
def no_web_task_workers(client, monkeypatch):
    """Keep HTTP and the sandbox queue real; any worker/backend startup fails."""
    def forbidden(*args, **kwargs):
        pytest.fail("Task HTTP routes must not start processes, threads or backend I/O")

    monkeypatch.setattr(DaBase, "__init__", forbidden)
    monkeypatch.setattr(da_scheduler, "Popen", forbidden)
    monkeypatch.setattr(da_scheduler.DaScheduler, "scheduler_loop", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


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

    def test_a_free_task_is_requested_for_the_scheduler(self, client):
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
        # The canonical key is what gets claimed, whichever alias was posted,
        # and it is pending rather than running: the dashboard no longer runs
        # anything itself.
        entry = task_state.running_tasks()["calc_optimum"]
        assert entry["state"] == "pending"
        assert entry["source"] == "dashboard"
        assert task_state.pending_requests().keys() == {"calc_optimum"}

    def test_the_parameters_travel_with_the_request(self, client):
        """The scheduler builds the command, so it needs the form values the
        dashboard collected."""
        client.post(
            "/v2/task-exec",
            data={
                "task": "fast_simulate",
                "days": "21",
                "csrf_token": csrf_token(client, "/v2/tasks"),
            },
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        entry = task_state.running_tasks()["fast_control_simulate"]
        assert entry["parameters"] == {"days": "21"}

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
    @pytest.mark.parametrize("task", task_registry.TASKS)
    def test_v2_api_run_enqueues_without_process(self, client, task):
        response = client.get(
            f"/v2/api/run/{task}", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 202
        assert response.content_type.startswith("text/plain")
        pending = task_state.pending_requests()
        assert pending.keys() == {task}
        entry = pending[task]
        assert entry["state"] == "pending"
        assert entry["source"] == "api"
        assert entry["parameters"] == {}
        assert entry["pid"] is None
        assert entry["logfile"] is None
        saved = json.loads(Path(task_state.STATE_PATH).read_text(encoding="utf-8"))
        assert saved["running"] == pending
        assert saved["last"] == {}

    @pytest.mark.parametrize("first_api", ["/v2/api/run", "/api/run"])
    @pytest.mark.parametrize("second_api", ["/v2/api/run", "/api/run"])
    @pytest.mark.parametrize("alias", ["calc_zonder_debug", "optimize_regular", "calc"])
    def test_aliases_share_the_canonical_queue_claim(
        self, client, first_api, second_api, alias
    ):
        response = client.get(
            f"{first_api}/{alias}", headers=INGRESS, environ_base=SUPERVISOR
        )
        assert response.status_code == 202
        pending = task_state.pending_requests()
        assert pending.keys() == {"calc_optimum"}
        assert pending["calc_optimum"]["source"] == "api"
        before = Path(task_state.STATE_PATH).read_bytes()

        for spelling in ("calc_optimum", "calc_zonder_debug", "optimize_regular", "calc"):
            response = client.get(
                f"{second_api}/{spelling}", headers=INGRESS, environ_base=SUPERVISOR
            )
            assert response.status_code == 409
            assert response.content_type.startswith("text/plain")
            assert b"api" in response.data
            assert task_state.pending_requests() == pending
            assert Path(task_state.STATE_PATH).read_bytes() == before

    @pytest.mark.parametrize("task", ["calc_optimum", "calc_zonder_debug", "optimize_regular", "calc"])
    def test_it_refuses_a_task_that_is_already_running(self, client, task):
        task_state.claim("calc_optimum", source="scheduler")
        before = Path(task_state.STATE_PATH).read_bytes()

        response = client.get(
            f"/v2/api/run/{task}", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 409
        assert b"scheduler" in response.data
        assert task_state.pending_requests() == {}
        assert Path(task_state.STATE_PATH).read_bytes() == before

    def test_a_different_task_can_still_be_queued(self, client):
        task_state.claim("calc_optimum", source="scheduler")
        holder = task_state.running_tasks()["calc_optimum"]

        response = client.get(
            "/v2/api/run/get_meteo", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 202
        assert task_state.pending_requests().keys() == {"meteo"}
        assert task_state.running_tasks()["calc_optimum"] == holder

    def test_an_unknown_task_is_a_404(self, client):
        response = client.get(
            "/v2/api/run/nope", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 404
        assert task_state.running_tasks() == {}
        assert not Path(task_state.STATE_PATH).exists()

    @pytest.mark.parametrize("api", ["/v2/api/run", "/api/run"])
    @pytest.mark.parametrize("headers, environ", [({}, {}), (INGRESS, {}), ({}, SUPERVISOR)])
    def test_run_requires_ingress_unless_direct_access_is_enabled(
        self, client, api, headers, environ
    ):
        response = client.get(f"{api}/get_meteo", headers=headers, environ_base=environ)

        assert response.status_code == 401
        assert task_state.running_tasks() == {}
        assert not Path(task_state.STATE_PATH).exists()

    @pytest.mark.parametrize("api", ["/v2/api/run", "/api/run"])
    def test_run_accepts_opted_in_direct_access(self, client, monkeypatch, api):
        monkeypatch.setattr(importlib.import_module("app"), "ALLOW_DIRECT", True)

        response = client.get(f"{api}/get_meteo")

        assert response.status_code == 202
        assert task_state.pending_requests().keys() == {"meteo"}
        assert task_state.pending_requests()["meteo"]["source"] == "api"

    def test_v2_run_keeps_its_get_only_contract(self, client):
        response = client.post(
            "/v2/api/run/get_meteo", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 405
        assert task_state.running_tasks() == {}


class TestDocumentedApi:
    """/api/run and /api/report are documented in DOCS.md with worked Home
    Assistant examples, so they kept their paths when the v1 interface was
    removed. They live in app/public_api.py now."""

    def test_post_still_enqueues_without_a_csrf_token(self, client):
        response = client.post(
            "/api/run/get_meteo", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 202
        assert response.content_type.startswith("text/plain")
        assert response.get_data(as_text=True) == "Taak Meteoprognoses ophalen aangevraagd.\n"
        pending = task_state.pending_requests()
        assert pending.keys() == {"meteo"}
        assert pending["meteo"]["state"] == "pending"
        assert pending["meteo"]["source"] == "api"

    def test_it_refuses_a_task_that_is_already_running(self, client):
        task_state.claim("calc_optimum", source="scheduler")

        response = client.get(
            "/api/run/calc_zonder_debug", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 409
        assert b"scheduler" in response.data

    def test_a_historical_alias_still_resolves(self, client):
        """These urls are the kind of thing people wire into an automation,
        so every old key has to keep working."""
        response = client.get(
            "/api/run/get_meteo", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 202
        # The canonical key is what gets claimed, whichever alias was used.
        assert task_state.pending_requests().keys() == {"meteo"}

    def test_it_hands_the_work_to_the_scheduler(self, client):
        """It used to run the task inside the request, which meant it had to
        finish inside gunicorn's per-request timeout or the worker was killed
        and the child orphaned. An optimisation takes minutes, so it never
        could."""
        response = client.get(
            "/api/run/calc_zonder_debug", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 202
        entry = task_state.running_tasks()["calc_optimum"]
        assert entry["state"] == "pending"
        assert entry["source"] == "api"

    def test_the_response_is_plain_text(self, client):
        """It used to render an HTML page. The documented use is a
        fire-and-forget rest_command that does not read the body."""
        response = client.get(
            "/api/run/get_meteo", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.headers["Content-Type"].startswith("text/plain")

    def test_an_unknown_task_is_a_404(self, client):
        response = client.get(
            "/api/run/nope", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 404

    def test_it_does_not_echo_the_path_back(self, client):
        """The segment is attacker controlled."""
        response = client.get(
            "/api/run/<script>alert(1)</script>",
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert response.status_code == 404
        assert b"script" not in response.data

    def test_it_is_still_refused_without_ingress(self, client):
        """Reachable only with allow_direct_access, exactly as before and as
        DOCS.md says."""
        assert client.get("/api/run/get_meteo").status_code == 401
        assert client.get("/api/report/da/vandaag").status_code == 401
