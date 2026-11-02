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

import datetime
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dao.lib import da_prices
from dao.prog import task_state
from dao.prog import tasks as task_registry
from dao.prog.config.loader import CURRENT_VERSION, ConfigurationLoader
from dao.prog.da_base import DaBase
from dao.tests.lib.test_da_prices import FakeDb, _values

from .conftest import INGRESS, SUPERVISOR, csrf_token
from .test_security import _editor_form


@pytest.fixture
def v2_routes(client):
    import importlib

    return importlib.import_module("app.v2.routes")


@pytest.fixture
def manual_price_context(v2_routes, monkeypatch):
    """Only context construction and the provider are fake, not the queue/CLI."""
    config = SimpleNamespace(time_zone="Europe/Helsinki")
    backend = SimpleNamespace(time_zone="America/New_York")
    backend_calls = []

    class FrozenDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            instant = datetime.datetime(2026, 6, 14, 22, 30, tzinfo=datetime.UTC)
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    clock = SimpleNamespace(
        datetime=FrozenDatetime, time=datetime.time, timedelta=datetime.timedelta
    )

    def fake_report(file_name):
        backend_calls.append(file_name)
        return backend

    def forbidden(*args, **kwargs):
        pytest.fail("Manual price refresh must not start real backend I/O or task workers")

    monkeypatch.setattr(v2_routes, "_cached_config", lambda: config)
    monkeypatch.setattr(v2_routes, "Report", fake_report)
    monkeypatch.setattr(v2_routes, "datetime", clock)
    monkeypatch.setattr(da_prices, "datetime", clock)
    monkeypatch.setattr(DaBase, "__init__", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)
    return config, backend, backend_calls


class TestCo2Report:
    @pytest.fixture
    def co2_config(self, v2_routes, monkeypatch, tmp_path):
        """Real config/menu/editor, with only backend I/O fenced off."""
        options = tmp_path / "options.json"
        options.write_text(json.dumps({
            "config_version": CURRENT_VERSION,
            "time_zone": "Europe/Amsterdam",
            "report": {},
        }), encoding="utf-8")
        (tmp_path / "secrets.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(v2_routes, "app_datapath", str(tmp_path) + "/")
        monkeypatch.setattr(v2_routes, "_config_cache", None)
        monkeypatch.setattr(v2_routes, "_config_cache_loaded", False)
        # Report rendering needs history; the menu and typed config do not.
        monkeypatch.setattr(v2_routes, "reports_gen", lambda *a, **kw: ["<table></table>"])

        def forbidden(*args, **kwargs):
            pytest.fail("CO2 config/menu tests must not start backend I/O or workers")

        monkeypatch.setattr(DaBase, "__init__", forbidden)
        monkeypatch.setattr(socket, "getaddrinfo", forbidden)
        monkeypatch.setattr(socket, "create_connection", forbidden)
        monkeypatch.setattr(socket.socket, "connect", forbidden)
        monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
        monkeypatch.setattr(subprocess, "Popen", forbidden)
        monkeypatch.setattr(threading.Thread, "start", forbidden)
        return options

    @pytest.mark.parametrize("raw", ["sensor.co2_intensity", ["sensor.co2_intensity"]])
    def test_legacy_co2_config_reaches_report_menu(self, client, co2_config, raw):
        document = json.loads(co2_config.read_text(encoding="utf-8"))
        document["report"] = {"entity co2-intensity": raw}
        co2_config.write_text(json.dumps(document), encoding="utf-8")

        response = client.get(
            "/v2/reports?subject=grid", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        assert b'value="co2"' in response.data

    @pytest.mark.parametrize("key", [
        "co2 intensity sensor", "co2_intensity_sensor", "entity co2-intensity",
    ])
    @pytest.mark.parametrize(("raw", "expected"), [
        ("sensor.co2_intensity", "sensor.co2_intensity"),
        (["sensor.co2_intensity"], "sensor.co2_intensity"),
        (None, None),
        ([], None),
    ])
    def test_co2_config_save_reload_preserves_raw_spelling(
        self, client, v2_routes, co2_config, key, raw, expected
    ):
        # Prime the actual cache: saving must invalidate it, not just write a file.
        assert not v2_routes.co2_available()
        form = _editor_form(client, "config")
        document = json.loads(co2_config.read_text(encoding="utf-8"))
        document["report"] = {key: raw, "//note": "keep this spelling"}
        submitted = json.dumps(document, indent=2)

        response = client.post(
            "/v2/config", data={**form, "config": submitted},
            headers=INGRESS, environ_base=SUPERVISOR,
        )

        assert response.status_code == 200
        assert b"Config updated successfully" in response.data
        assert co2_config.read_text(encoding="utf-8") == submitted
        reloaded = ConfigurationLoader(co2_config).load_and_validate()
        assert reloaded.report.co2_intensity_sensor == expected
        response = client.get("/v2/reports", headers=INGRESS, environ_base=SUPERVISOR)
        assert response.status_code == 200
        assert (b'value="co2"' in response.data) == (expected is not None)

    @pytest.mark.parametrize("report", [
        {"entity co2-intensity": "not_an_entity_id"},
        {"entity co2-intensity": ["not_an_entity_id"]},
        {"entity co2-intensity": ["sensor.co2_intensity", "sensor.other"]},
        {"co2 intensity sensor": "sensor.co2_intensity", "entity co2-intensity": "sensor.other"},
        {"co2_intensity_sensor": "sensor.co2_intensity", "entity co2-intensity": None},
        {"co2 intensity sensor": "sensor.co2_intensity", "co2_intensity_sensor": "sensor.other"},
    ])
    def test_invalid_co2_config_cannot_overwrite_saved_file(self, client, co2_config, report):
        form = _editor_form(client, "config")
        before = co2_config.read_text(encoding="utf-8")
        document = json.loads(before)
        document["report"] = report

        response = client.post(
            "/v2/config", data={**form, "config": json.dumps(document)},
            headers=INGRESS, environ_base=SUPERVISOR,
        )

        assert response.status_code == 200
        assert b"Error:" in response.data
        assert co2_config.read_text(encoding="utf-8") == before

    @pytest.mark.parametrize("key", [
        "co2 intensity sensor", "co2_intensity_sensor", "entity co2-intensity",
    ])
    @pytest.mark.parametrize(("raw", "expected"), [
        ("sensor.co2_intensity", ["sensor.co2_intensity"]),
        (["sensor.co2_intensity"], ["sensor.co2_intensity"]),
        (None, []),
        ([], []),
    ])
    def test_typed_co2_config_keeps_report_sensor_list(
        self, v2_routes, co2_config, monkeypatch, key, raw, expected
    ):
        document = json.loads(co2_config.read_text(encoding="utf-8"))
        document["report"] = {key: raw}
        co2_config.write_text(json.dumps(document), encoding="utf-8")

        def local_config_only(report, file_name):
            # Keep Report.__init__ real, replacing only DaBase's HA/DB startup.
            report.config = ConfigurationLoader(Path(file_name)).load_and_validate()
            report.prices_options = SimpleNamespace(last_invoice=datetime.date(2026, 1, 1))

        monkeypatch.setattr(DaBase, "__init__", local_config_only)
        report = v2_routes.Report(str(co2_config), _now=datetime.datetime(2026, 11, 2, 12))

        assert report.co2_intensity_sensor == expected
        assert report.co2_dict["co2_intensity"]["sensors"] == expected

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

    @pytest.mark.parametrize("task", ["prices", "get_prices", "update_prices"])
    @pytest.mark.parametrize("dates", [
        {"prijzen_start": "2026-01-01", "prijzen_tot": "2026-01-03"},
        {"prijzen_start": "2026-01-01"},
        {"prijzen_tot": "2026-01-03"},
    ])
    def test_submitted_dates_reach_the_request(
        self, client, v2_routes, monkeypatch, task, dates
    ):
        def forbidden(*args, **kwargs):
            pytest.fail("Chosen dates must not need installation context")

        monkeypatch.setattr(v2_routes, "_cached_config", forbidden)
        monkeypatch.setattr(v2_routes, "Report", forbidden)
        response = client.post(
            "/v2/task-exec",
            data={
                "task": task,
                **dates,
                "csrf_token": csrf_token(client, "/v2/tasks"),
            },
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert response.status_code == 302
        assert task_state.pending_requests()["prices"]["parameters"] == dates

    @pytest.mark.parametrize("task, fields, canonical, expected", [
        ("update_meteo", {}, "meteo", {}),
        ("fast_simulate", {"days": "  "}, "fast_control_simulate", {}),
        ("fast_simulate", {"days": "21"}, "fast_control_simulate", {"days": "21"}),
    ])
    def test_other_tasks_do_not_need_installation_context(
        self, client, v2_routes, monkeypatch, task, fields, canonical, expected
    ):
        def forbidden(*args, **kwargs):
            pytest.fail("Other tasks must not need installation context")

        monkeypatch.setattr(v2_routes, "_cached_config", forbidden)
        monkeypatch.setattr(v2_routes, "Report", forbidden)
        response = client.post(
            "/v2/task-exec",
            data={
                "task": task,
                **fields,
                "csrf_token": csrf_token(client, "/v2/tasks"),
            },
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert response.status_code == 302
        assert task_state.pending_requests()[canonical]["parameters"] == expected

    @pytest.mark.parametrize("task", ["prices", "get_prices", "update_prices"])
    @pytest.mark.parametrize("fields", [{}, {"prijzen_start": "", "prijzen_tot": "  "}])
    @pytest.mark.parametrize("config_zone, backend_zone, day, first_epoch", [
        ("Europe/Helsinki", "America/New_York", "2026-06-15", 1781470800),
        (None, "Europe/Helsinki", "2026-06-15", 1781470800),
        ("America/New_York", "Europe/Helsinki", "2026-06-14", 1781409600),
        (None, "America/New_York", "2026-06-14", 1781409600),
    ])
    def test_manual_price_refresh_bypasses_presence_shortcut_via_queue(
        self, client, v2_routes, manual_price_context, monkeypatch,
        task, fields, config_zone, backend_zone, day, first_epoch,
    ):
        config, backend, backend_calls = manual_price_context
        config.time_zone = config_zone
        backend.time_zone = backend_zone
        db = FakeDb(present=datetime.datetime(2100, 1, 1, 23), zone=config_zone or backend_zone)
        prices = da_prices.DaPrices(SimpleNamespace(interval="1hour"), db, country="NL")
        provider_calls = []

        class FakeNordpool:
            def fetch(self, *, areas, end_date, resolution):
                provider_calls.append(end_date.date())
                assert areas == ["NL"] and resolution == 60
                assert end_date.date() in (datetime.date(2026, 6, 14), datetime.date(2026, 6, 15))
                return {"areas": {"NL": {"values": _values(end_date.date(), 24)}}}

        monkeypatch.setattr(da_prices, "Prices", FakeNordpool)
        runner = SimpleNamespace(
            prices=prices, prices_options=SimpleNamespace(source_day_ahead="nordpool")
        )
        # The unchanged CLI/scheduler default still respects already-present data.
        monkeypatch.setattr(sys, "argv", ["day_ahead.py", "prices"])
        DaBase.get_day_ahead_prices(runner)
        assert provider_calls == [] and db.saved == []

        response = client.post(
            "/v2/task-exec",
            data={"task": task, **fields, "csrf_token": csrf_token(client, "/v2/tasks")},
            headers=INGRESS,
            environ_base=SUPERVISOR,
        )

        assert response.status_code == 302
        entry = task_state.pending_requests()["prices"]
        assert entry["source"] == "dashboard"
        assert entry["parameters"] == {"prijzen_start": day}
        assert backend_calls == ([] if config_zone else [v2_routes.app_datapath + "options.json"])
        # Requesting must not execute the fetch; the scheduler owns that work.
        assert provider_calls == [] and db.saved == []
        pending = task_state.take_pending("prices")
        cmd = task_registry.build_cmd("prices", pending["parameters"])
        assert cmd == ["python3", "../prog/day_ahead.py", "prices", day]
        monkeypatch.setattr(sys, "argv", cmd[1:])
        DaBase.get_day_ahead_prices(runner)

        assert provider_calls == [datetime.date(2026, 6, 14), datetime.date(2026, 6, 15)]
        frame = pd.concat(db.saved, ignore_index=True)
        epochs = frame["time"].astype("int64")
        assert len(epochs) == epochs.nunique() == 24
        assert epochs.iloc[0] == first_epoch
        assert epochs.iloc[-1] == first_epoch + 23 * 3600
        assert frame["code"].unique().tolist() == ["da"]

    def test_manual_price_refresh_keeps_ingress_and_csrf_checks(
        self, client, v2_routes, monkeypatch
    ):
        def forbidden(*args, **kwargs):
            pytest.fail("Rejected requests must not read installation context")

        monkeypatch.setattr(v2_routes, "_cached_config", forbidden)
        monkeypatch.setattr(v2_routes, "Report", forbidden)
        response = client.post(
            "/v2/task-exec", data={"task": "prices"},
            headers=INGRESS, environ_base=SUPERVISOR,
        )
        assert response.status_code == 400
        response = client.post(
            "/v2/task-exec",
            data={"task": "prices", "csrf_token": csrf_token(client, "/v2/tasks")},
        )
        assert response.status_code == 401
        assert task_state.pending_requests() == {}

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
