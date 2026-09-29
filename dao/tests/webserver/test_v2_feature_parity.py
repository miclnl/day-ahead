"""The two things v2 had to gain before the v1 interface could be removed.

The review made the removal conditional: "Verwijder de v1 web-UI zodra v2
CO2-rapport en prijsdatum-parameters heeft." Both were genuinely missing:

* the CO2 branch in v2's reports_gen was commented out, and the subject
  picker offered only Grid and Balance;
* the v2 task page had hard-coded buttons with no parameter inputs at all,
  so the price fetch could not be given a date range, the backtest could not
  be given a number of days, and five tasks had no button at all.

These tests keep both from regressing now that v1 is gone and there is no
fallback interface.
"""

import pytest

from dao.prog import tasks as task_registry

from .conftest import INGRESS, SUPERVISOR, csrf_token


@pytest.fixture
def v2_routes(client):
    import importlib

    return importlib.import_module("app.v2.routes")


class TestCo2Report:
    def test_the_subject_is_offered_when_a_sensor_is_configured(
        self, client, v2_routes, monkeypatch
    ):
        monkeypatch.setattr(v2_routes, "co2_available", lambda: True)
        # Only the subject picker is under test; rendering a real report
        # would need a populated database.
        monkeypatch.setattr(
            v2_routes, "reports_gen", lambda *a, **kw: ["<table></table>"]
        )

        response = client.get(
            "/v2/reports?subject=grid", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        assert b'value="co2"' in response.data

    def test_it_is_hidden_when_no_sensor_is_configured(
        self, client, v2_routes, monkeypatch
    ):
        """Without the sensor every CO2 figure is zero, so an empty report is
        worse than no report. The old interface deleted the menu entry for
        exactly this reason."""
        monkeypatch.setattr(v2_routes, "co2_available", lambda: False)
        monkeypatch.setattr(
            v2_routes, "reports_gen", lambda *a, **kw: ["<table></table>"]
        )

        response = client.get(
            "/v2/reports?subject=grid", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert b'value="co2"' not in response.data

    def test_a_bookmarked_co2_url_falls_back_when_the_sensor_is_gone(
        self, client, v2_routes, monkeypatch
    ):
        """Rather than raising "Invalid subject" out of reports_gen."""
        monkeypatch.setattr(v2_routes, "co2_available", lambda: False)
        monkeypatch.setattr(
            v2_routes, "reports_gen", lambda *a, **kw: ["<table></table>"]
        )

        response = client.get(
            "/v2/reports?subject=co2", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200

    def test_co2_does_not_offer_the_forecast_periods(self, v2_routes):
        """There is no forecast of grid intensity, so the periods that look
        into tomorrow cannot be reported on."""
        values = {entry["value"] for entry in v2_routes.period_options("co2")}

        assert not (values & v2_routes.FORECAST_PERIODS)
        assert "vandaag" in values
        assert "vorige maand" in values

    def test_other_subjects_keep_every_period(self, v2_routes):
        values = {entry["value"] for entry in v2_routes.period_options("grid")}

        assert v2_routes.FORECAST_PERIODS <= values

    def test_a_forecast_period_on_co2_is_replaced(
        self, client, v2_routes, monkeypatch
    ):
        monkeypatch.setattr(v2_routes, "co2_available", lambda: True)
        seen = {}

        def fake_reports_gen(subject, view, period, **kwargs):
            seen["period"] = period
            return ["<table></table>"]

        monkeypatch.setattr(v2_routes, "reports_gen", fake_reports_gen)

        client.get(
            "/v2/reports?subject=co2&period=morgen",
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert seen["period"] == "vandaag"

    def test_reports_gen_has_a_live_co2_branch(self, v2_routes):
        """It was commented out; a call must reach calc_co2_emission rather
        than fall through to "Invalid subject"."""
        import inspect

        source = inspect.getsource(v2_routes.reports_gen)

        assert 'elif subject == "co2":' in source
        assert "calc_co2_emission" in source
        assert "co2_graph_options" in source


class TestTaskParameters:
    def test_every_registry_task_has_a_button(self, v2_routes):
        """The hard-coded list left clean, consolidate, forecast_accuracy,
        fast_once and the backtest with no way to start them."""
        offered = {entry["key"] for entry in v2_routes.task_page_entries()}

        assert offered == set(task_registry.TASKS)

    def test_the_price_fetch_offers_a_date_range(self, v2_routes):
        entry = next(
            e for e in v2_routes.task_page_entries() if e["key"] == "prices"
        )

        names = [field["name"] for field in entry["fields"]]
        assert names == ["prijzen_start", "prijzen_tot"]
        assert all(field["type"] == "date" for field in entry["fields"])

    def test_the_backtest_offers_a_number_of_days(self, v2_routes):
        entry = next(
            e
            for e in v2_routes.task_page_entries()
            if e["key"] == "fast_control_simulate"
        )

        assert [field["name"] for field in entry["fields"]] == ["days"]
        assert entry["fields"][0]["default"] == "14"

    def test_a_task_without_parameters_has_no_fields(self, v2_routes):
        entry = next(
            e for e in v2_routes.task_page_entries() if e["key"] == "meteo"
        )

        assert entry["fields"] == []

    def test_the_page_renders_the_inputs(self, client):
        response = client.get(
            "/v2/tasks", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        assert b'name="prijzen_start"' in response.data
        assert b'name="prijzen_tot"' in response.data
        assert b'name="days"' in response.data

    def test_submitted_dates_reach_the_request(self, client):
        from dao.prog import task_state

        client.post(
            "/v2/task-exec",
            data={
                "task": "prices",
                "prijzen_start": "2026-01-01",
                "prijzen_tot": "2026-01-03",
                "csrf_token": csrf_token(client, "/v2/tasks"),
            },
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        entry = task_state.running_tasks()["prices"]
        assert entry["parameters"] == {
            "prijzen_start": "2026-01-01",
            "prijzen_tot": "2026-01-03",
        }

    def test_blank_parameters_are_dropped(self, client):
        """An empty date must not be appended to the command line; the price
        fetch then covers its default range instead of parsing "" as a date."""
        from dao.prog import task_state

        client.post(
            "/v2/task-exec",
            data={
                "task": "prices",
                "prijzen_start": "",
                "prijzen_tot": "  ",
                "csrf_token": csrf_token(client, "/v2/tasks"),
            },
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert task_state.running_tasks()["prices"]["parameters"] == {}

    def test_every_declared_parameter_has_a_field_definition(self):
        """A parameter in the registry with no PARAMETER_FIELDS entry would
        silently never get an input, which is how the price dates went
        missing in the first place."""
        import importlib

        v2_routes = importlib.import_module("app.v2.routes")
        declared = {
            parameter
            for task in task_registry.TASKS.values()
            for parameter in task.get("parameters", ())
        }

        assert declared <= set(v2_routes.PARAMETER_FIELDS)
