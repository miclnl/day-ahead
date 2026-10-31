"""Local CLI is dry-run even for revalue/reconcile unless --publish is explicit."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from dao.lib import db_connections
from dao.tests.prog.test_consolidate import db as db
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_metering import put_grid
from dao.tests.settlement.test_metering import recorder as recorder
from dao.tests.settlement.test_publication import pricing_config
from dao.tests.settlement.test_service import contents
from dao.tests.settlement.test_service import system as system


@pytest.mark.parametrize("command", ["preview", "revalue", "reconcile"])
def test_cli_requires_publish(system, monkeypatch, command, capsys):
    from dao.prog import da_settlement
    monkeypatch.setattr(da_settlement, "load_service", lambda path: system.service)
    before = contents(system)
    assert da_settlement.main([command, "--start", str(START_TS), "--end", str(START_TS + 900)]) == 0
    assert contents(system) == before and not system.path.exists()
    assert "paid" not in capsys.readouterr().out
    if command != "preview":
        assert da_settlement.main([command, "--start", str(START_TS), "--end", str(START_TS + 900),
                                   "--publish", "--reason", "explicit-test"]) == 0
        assert system.repo.read_book_reference(system.meter.meter_id).input_revision > 0


def test_cli_preview_rejects_publish(system, monkeypatch):
    from dao.prog import da_settlement
    monkeypatch.setattr(da_settlement, "load_service", lambda path: system.service)
    with pytest.raises(SystemExit):
        da_settlement.main(["preview", "--publish"])
    assert not system.path.exists()


@pytest.fixture
def local_config(tmp_path, db, recorder, monkeypatch):
    from dao.prog.settlement.repository import SettlementRepository
    db_connections.reset_connections()
    repo = SettlementRepository(db.engine)
    repo.ensure_schema()
    with db.engine.begin() as connection:
        connection.execute(text("INSERT INTO variabel (id,code,name,dim,aggregate) VALUES (5,'da','Prijs','eur','avg')"))
        connection.execute(text('INSERT INTO "values" (variabel,time,value) VALUES (5,:at,0.1)'), {"at": START_TS})
    put_grid(recorder)
    config = tmp_path / "options.json"
    config.write_text(json.dumps({
        "database da": {"engine": "sqlite", "database": "day_ahead.db", "db_path": str(tmp_path)},
        "database ha": {"engine": "sqlite", "database": "recorder.db", "db_path": str(tmp_path)},
        "time_zone": "Europe/Amsterdam", "prices": pricing_config().model_dump(mode="json", by_alias=True),
        "report": {"entities grid consumption": ["sensor.import"], "entities grid production": ["sensor.export"]},
    }))
    (tmp_path / "secrets.json").write_text("{}")
    monkeypatch.setattr("dao.prog.fastctrl.plan.FAST_PLAN_FILE", str(tmp_path / "fast_plan.json"))
    for module in ("service", "repository", "publication"):
        monkeypatch.setattr(f"dao.prog.settlement.{module}.time", lambda: START_TS + 86400)
    yield SimpleNamespace(path=config, db=db, recorder=recorder, repo=repo)
    db_connections.reset_connections()


def test_real_cli_bootstrap_never_migrates_config_or_calls_ha(local_config, monkeypatch, capsys):
    from dao.prog import da_settlement
    def forbidden(*args, **kwargs):
        raise AssertionError("local CLI must not construct an HA client")
    monkeypatch.setattr("dao.prog.da_base.HAClient", forbidden)
    before = {p.name: p.read_bytes() for p in local_config.path.parent.iterdir() if p.is_file()}
    result = da_settlement.main(["preview", "--config", str(local_config.path),
                                 "--start", str(START_TS), "--end", str(START_TS + 900)])
    assert result == 0
    assert json.loads(capsys.readouterr().out)["lines"][0]["import_cost"] == "0.03"
    assert {p.name: p.read_bytes() for p in local_config.path.parent.iterdir() if p.is_file()} == before


def test_registered_cli_command_runs_from_webserver_directory(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "../prog/da_settlement.py", "--help"],
                            cwd=root / "dao/webserver", env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "reconcile" in result.stdout


def test_scheduler_registry_entry_honors_off_without_writing(local_config):
    from dao.prog import da_settlement, tasks
    data = json.loads(local_config.path.read_text())
    data["prices"]["settlement"]["mode"] = "off"
    local_config.path.write_text(json.dumps(data))
    before = {p.name: p.read_bytes() for p in local_config.path.parent.iterdir() if p.is_file()}
    args = tasks.build_cmd("settlement_consolidate")[2:]
    assert da_settlement.main([*args, "--config", str(local_config.path)]) == 0
    assert {p.name: p.read_bytes() for p in local_config.path.parent.iterdir() if p.is_file()} == before
