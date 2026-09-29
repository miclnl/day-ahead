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


# No fixture needed to stop the routes running anything: they only record a
# request now. Starting the task is the scheduler process's job, which is
# exactly the point of moving it out of the gunicorn worker.


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


class TestDocumentedApi:
    """/api/run and /api/report are documented in DOCS.md with worked Home
    Assistant examples, so they kept their paths when the v1 interface was
    removed. They live in app/public_api.py now."""

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
