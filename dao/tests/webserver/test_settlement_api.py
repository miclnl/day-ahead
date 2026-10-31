"""The existing ingress boundary and read-only settlement generation query."""

import json
from datetime import datetime
from importlib import import_module

import pytest

from dao.tests.prog.test_consolidate import db as db
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_cli import local_config as local_config
from dao.tests.settlement.test_metering import recorder as recorder
from dao.tests.webserver.conftest import INGRESS, SUPERVISOR


@pytest.mark.parametrize("generation", ["-1", "1.0", "1 OR 1=1", "../../options.json", "", "+1", "true"])
def test_generation_query_is_nonnegative_integer_before_report_construction(client, monkeypatch, generation):
    public_api = import_module("app.public_api")
    def forbidden(*args):
        raise AssertionError("invalid query must not construct a report")
    monkeypatch.setattr(public_api, "Report", forbidden)
    response = client.get("/api/report/settlement/vandaag", query_string={"generation": generation},
                          headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 400


def test_settlement_generation_query_reaches_only_the_read_route(client, monkeypatch):
    public_api = import_module("app.public_api")
    calls = []
    class Report:
        def __init__(self, path):
            pass

        @classmethod
        def settlement_reader(cls, path):
            return cls(path)

        def get_api_data(self, field, period, **kwargs):
            calls.append((field, period, kwargs))
            return json.dumps({"message": "Success", "data": [], "generation": kwargs["generation"]})
    monkeypatch.setattr(public_api, "Report", Report)
    response = client.get("/api/report/settlement/vandaag?generation=0", headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 200 and response.json["generation"] == 0
    assert calls == [("settlement", "vandaag", {"cumulate": False, "generation": 0})]


def test_settlement_api_keeps_existing_access_guard(client, monkeypatch):
    public_api = import_module("app.public_api")
    calls = []
    monkeypatch.setattr(public_api, "Report", lambda *a: calls.append(a))
    response = client.get("/api/report/settlement/vandaag")
    assert response.status_code in (401, 403) and calls == []


def test_real_settlement_api_reads_generation_without_ha_or_file_writes(client, local_config, monkeypatch):
    from dao.prog.da_report import Report
    from dao.prog.da_settlement import load_service
    public_api = import_module("app.public_api")
    service = load_service(local_config.path)
    service.consolidate(START_TS, START_TS + 900, publish=True, reason="api-fixture")
    data = json.loads(local_config.path.read_text())
    data.pop("config_version", None)
    local_config.path.write_text(json.dumps(data))
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only API must not construct an HA client")
    monkeypatch.setattr("dao.prog.da_base.HAClient", forbidden)
    monkeypatch.setattr(public_api, "app_datapath", str(local_config.path.parent))
    original = Report.make_periodes
    monkeypatch.setattr(Report, "make_periodes", lambda self, **kwargs: original(self, datetime(2027, 1, 1)))
    before = {p.name: p.read_bytes() for p in local_config.path.parent.iterdir() if p.is_file()}
    response = client.get("/api/report/settlement/vandaag?generation=1", headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 200
    assert response.json["generation"] == 1 and response.json["net_cost"] is None
    assert response.json["data"][0]["import_cost"] == "0.03"
    assert response.json["data"][0]["bonus_kind"] == "calculated"
    assert {p.name: p.read_bytes() for p in local_config.path.parent.iterdir() if p.is_file()} == before
