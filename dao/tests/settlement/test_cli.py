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
from dao.tests.settlement.test_reconciliation_io import book as book
from dao.tests.settlement.test_reconciliation_io import (
    database_section,
    options_document,
    supplier_document,
    write_options,
    write_supplier,
)
from dao.tests.settlement.test_repository import calculate as calculate
from dao.tests.settlement.test_repository import inputs as inputs
from dao.tests.settlement.test_repository import publish
from dao.tests.settlement.test_repository import repo as repo
from dao.tests.settlement.test_repository import terms as terms
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


def run_cli(argv):
    """The CLI exit code, whether the route returns it or argparse raises it."""
    from dao.prog import da_settlement
    try:
        return da_settlement.main(argv)
    except SystemExit as exit_signal:
        return exit_signal.code


def compare_argv(case, *extra, supplier=None):
    return ["compare-invoice", "--supplier", str(supplier or case.supplier),
            "--config", str(case.options), *extra]


@pytest.fixture
def compare_case(tmp_path, book):
    """A published book with its own configuration and a matching supplier file."""
    options = write_options(tmp_path, options_document(tmp_path, database=database_section(book)))
    supplier = write_supplier(tmp_path, supplier_document())
    return SimpleNamespace(directory=Path(tmp_path), options=options, supplier=supplier, book=book)


@pytest.fixture
def compare_history(tmp_path, repo, terms, inputs, calculate):
    """Two published generations, so --generation has something to select."""
    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots), snapshots, calculate)
    publish(repo, inputs(snapshots, imports=("1", "2")), snapshots, calculate,
            revision=1, generation=1)
    options = write_options(tmp_path, options_document(tmp_path, database=database_section(repo)))
    supplier = write_supplier(tmp_path, supplier_document())
    return SimpleNamespace(directory=Path(tmp_path), options=options, supplier=supplier, book=repo)


def test_e1d_compare_rejects_flags_before_loaders(compare_case, monkeypatch):
    from dao.prog.settlement import reconciliation_io

    def forbidden(*args, **kwargs):
        pytest.fail("compare-invoice must refuse its flags before it loads anything")

    monkeypatch.setattr(reconciliation_io, "load_supplier", forbidden)
    monkeypatch.setattr(reconciliation_io, "open_reconciliation_reader", forbidden)
    base = compare_argv(compare_case)
    for extra in (["--publish"], ["--scheduled"], ["--reason", "scheduled"], ["--reason", ""]):
        assert run_cli([*base, *extra]) == 2
    assert run_cli(["compare-invoice", "--config", str(compare_case.options)]) == 2
    assert run_cli(["compare-invoice", "--supplier", str(compare_case.supplier),
                    "--config", str(compare_case.options), "--start", str(START_TS)]) == 2
    assert run_cli(["compare-invoice", "--supplier", str(compare_case.supplier),
                    "--config", str(compare_case.options), "--end", str(START_TS + 1800)]) == 2
    assert run_cli(["preview", "--supplier", str(compare_case.supplier)]) == 2
    assert run_cli(["reconcile", "--generation", "1"]) == 2
    assert run_cli(["revalue", "--supplier", str(compare_case.supplier),
                    "--generation", "1"]) == 2


def test_e1d_compare_crosschecks_bounds_and_generation(compare_history, capsys):
    # Both bounds are accepted only when they equal the supplier bounds exactly.
    assert run_cli(compare_argv(compare_history, "--start", str(START_TS),
                                "--end", str(START_TS + 1800))) == 3
    assert json.loads(capsys.readouterr().out)["pin"]["generation"] == 2
    assert run_cli(compare_argv(compare_history, "--start", str(START_TS),
                                "--end", str(START_TS + 900))) == 2
    assert run_cli(compare_argv(compare_history, "--end", str(START_TS + 1800))) == 2
    # An explicit generation selects that historical book and says so.
    assert run_cli(compare_argv(compare_history, "--generation", "1")) == 4
    report = json.loads(capsys.readouterr().out)
    assert report["pin"]["generation"] == 1
    assert "book_outdated" in report["reasons"]
    # Without a generation the current book pointer decides, and it differs.
    assert run_cli(compare_argv(compare_history)) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["pin"]["generation"] == 2
    assert report["status"] == "different"
    # A generation that was never published is an invalid selection.
    assert run_cli(compare_argv(compare_history, "--generation", "9")) == 2
    assert run_cli(compare_argv(compare_history, "--generation", "not-a-number")) == 2


def test_e1d_compare_exit_precedence(compare_case, tmp_path, capsys, monkeypatch):
    from dao.prog.settlement.repository import ConcurrentSettlementError, SettlementRepository

    # A fully comparable, matching report exits 0 and prints the whole report.
    assert run_cli(compare_argv(compare_case)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "matched"
    assert report["freshness"] == "current"
    assert report["source_trust"] == "registered_book_and_caller_supplied_supplier"
    assert report["supplier_digest"] and report["pin"]["generation"] == 1
    assert report["provenance"]["lines"] and report["total"]["delta"] == "0"

    # One coherent cent of difference is fully comparable: exit 3.
    document = supplier_document()
    document["import_amounts"] = {"ex_vat": "0.20", "vat": "0", "gross": "0.21"}
    document["total"] = "0.21"
    supplier = write_supplier(tmp_path, document, name="one-cent.json")
    assert run_cli(compare_argv(compare_case, supplier=supplier)) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "different"

    # A missing supplier detail is not comparable and outranks the difference: exit 4.
    document = supplier_document()
    document["import_amounts"] = {"ex_vat": "0.20", "vat": "0", "gross": "0.21"}
    document["bonus_amounts"] = {"ex_vat": None, "vat": None, "gross": "0"}
    document["total"] = "0.21"
    supplier = write_supplier(tmp_path, document, name="partial.json")
    assert run_cli(compare_argv(compare_case, supplier=supplier)) == 4
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "not_comparable"

    # An invalid file is refused before the database is touched: exit 2.
    invalid = write_supplier(tmp_path, supplier_document(total="not-a-decimal"), name="invalid.json")
    assert run_cli(compare_argv(compare_case, supplier=invalid)) == 2
    assert capsys.readouterr().out == ""

    # A config meter that does not match is refused before the connection: exit 2.
    foreign = write_options(tmp_path, options_document(tmp_path, meter="another-meter"),
                            name="foreign-options.json")
    assert run_cli(["compare-invoice", "--supplier", str(compare_case.supplier),
                    "--config", str(foreign)]) == 2
    assert capsys.readouterr().out == ""

    # A missing database is an operational failure without a financial report: exit 1.
    absent = write_options(tmp_path, options_document(
        tmp_path, database={"engine": "sqlite", "database": "absent.db", "db_path": str(tmp_path)}),
        name="absent-options.json")
    assert run_cli(["compare-invoice", "--supplier", str(compare_case.supplier),
                    "--config", str(absent)]) == 1
    assert capsys.readouterr().out == ""

    # An unreadable book view is an operational failure without a financial report: exit 1.
    with compare_case.book.engine.begin() as connection:
        connection.execute(text("UPDATE settlement_interval SET line_json = '{broken'"))
    assert run_cli(compare_argv(compare_case)) == 1
    assert capsys.readouterr().out == ""

    # A detected read race is an operational failure without a financial report: exit 1.
    def raced(*args, **kwargs):
        raise ConcurrentSettlementError("Registered book changed during reconciliation read")

    monkeypatch.setattr(SettlementRepository, "read_reconciliation", raced)
    assert run_cli(compare_argv(compare_case)) == 1
    assert capsys.readouterr().out == ""


def test_e1d_compare_reader_exposes_no_raw_write_capability(compare_case, capsys, monkeypatch):
    """The reader the compare route builds exposes no raw write capability."""
    from contextlib import contextmanager

    from sqlalchemy import literal, select
    from sqlalchemy.engine import Connection, CursorResult, Engine

    from dao.prog.settlement import reconciliation_io

    captured = {}
    original = reconciliation_io.open_reconciliation_reader

    @contextmanager
    def capturing_reader(*args, **kwargs):
        with original(*args, **kwargs) as repository:
            captured["repository"] = repository
            yield repository

    monkeypatch.setattr(reconciliation_io, "open_reconciliation_reader", capturing_reader)
    assert run_cli(compare_argv(compare_case)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "matched"

    engine = captured["repository"].engine
    assert not isinstance(engine, Engine)
    for name in ("_engine", "_connection", "pool", "dialect", "dbapi", "raw_connection"):
        assert not hasattr(engine, name), name
    with engine.connect() as connection:
        assert not isinstance(connection, Connection)
        for name in ("_connection", "connection", "engine", "pool", "dialect", "dbapi",
                     "exec_driver_sql", "raw_connection"):
            assert not hasattr(connection, name), name
        result = connection.execute(select(literal(1)))
        assert not isinstance(result, CursorResult)
        for name in ("connection", "engine", "pool", "dbapi", "context"):
            assert not hasattr(result, name), name


def test_e1d_compare_generation_and_read_failures_are_classified(compare_case, capsys, monkeypatch):
    from dao.prog.settlement.repository import SettlementRepository

    # A generation that was never published is an invalid selection: exit 2.
    assert run_cli(compare_argv(compare_case, "--generation", "9")) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": "rejected", "category": "generation_not_published"}

    # A generation outside the supported range is invalid input, not a read failure.
    assert run_cli(compare_argv(compare_case, "--generation", str(2 ** 63))) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": "rejected", "category": "generation_invalid"}

    # Every other ValueError from the reader is an operational read failure.
    def broken_read(*args, **kwargs):
        raise ValueError("stored DTO shape")

    monkeypatch.setattr(SettlementRepository, "read_reconciliation", broken_read)
    assert run_cli(compare_argv(compare_case)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    failure = json.loads(captured.err)
    assert failure["status"] == "error"
    assert failure["category"] != "generation_not_published"

    # A programming error is never reported as a successful comparison either.
    def broken_program(*args, **kwargs):
        raise TypeError("unexpected attribute")

    monkeypatch.setattr(SettlementRepository, "read_reconciliation", broken_program)
    assert run_cli(compare_argv(compare_case)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "error"


def test_e1d_compare_generation_zero_is_not_published(compare_case, capsys, monkeypatch):
    """Generation 0 is the not-published sentinel and never selects a book."""
    from dao.prog.settlement import reconciliation_io

    def forbidden(*args, **kwargs):
        pytest.fail("generation 0 must be refused before the reader connects")

    monkeypatch.setattr(reconciliation_io, "open_reconciliation_reader", forbidden)
    assert run_cli(compare_argv(compare_case, "--generation", "0")) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": "rejected", "category": "generation_not_published"}


def test_e1d_compare_categories_are_fixed_codes(compare_case, tmp_path, capsys, monkeypatch):
    """Every category of the compare route is a fixed code, never a sentence."""
    import re

    from dao.prog.settlement.repository import SettlementRepository

    categories = []

    def category_of(argv):
        code = run_cli(argv)
        captured = capsys.readouterr()
        assert captured.out == ""
        payload = json.loads(captured.err)
        assert payload["status"] in ("rejected", "error")
        # An invalid selection exits 2, an operational failure exits 1.
        assert code == (2 if payload["status"] == "rejected" else 1)
        assert set(payload) == {"status", "category"}
        categories.append(payload["category"])
        return payload["category"]

    # The invalid selections of the route.
    assert category_of([*compare_argv(compare_case), "--publish"]) == "compare_flags_not_allowed"
    assert category_of(["compare-invoice", "--config", str(compare_case.options)]) == "supplier_required"
    assert category_of([*compare_argv(compare_case), "--start", str(START_TS)]) == "bounds_required_together"
    assert category_of([*compare_argv(compare_case), "--start", str(START_TS),
                        "--end", str(START_TS + 900)]) == "bounds_mismatch"
    assert category_of(compare_argv(compare_case, "--generation", "not-a-number")) == "generation_invalid"
    assert category_of(compare_argv(compare_case, "--generation", str(2 ** 63))) == "generation_invalid"
    assert category_of(compare_argv(compare_case, "--generation", "0")) == "generation_not_published"
    assert category_of(compare_argv(compare_case, "--generation", "9")) == "generation_not_published"

    # An invalid supplier file keeps its own fixed category.
    invalid = write_supplier(tmp_path, supplier_document(total="not-a-decimal"), name="invalid.json")
    assert category_of(compare_argv(compare_case, supplier=invalid)) == "file_invalid_values"

    # An unreachable database is an operational failure with a fixed category.
    absent = write_options(tmp_path, options_document(
        tmp_path, database={"engine": "sqlite", "database": "absent.db", "db_path": str(tmp_path)}),
        name="absent-options.json")
    assert category_of(["compare-invoice", "--supplier", str(compare_case.supplier),
                        "--config", str(absent)]) == "database_missing"

    # A reader failure and a programming error are operational failures too.
    def broken_read(*args, **kwargs):
        raise ValueError("stored DTO shape")

    monkeypatch.setattr(SettlementRepository, "read_reconciliation", broken_read)
    assert category_of(compare_argv(compare_case)) == "book_read_failed"

    def broken_program(*args, **kwargs):
        raise TypeError("unexpected attribute")

    monkeypatch.setattr(SettlementRepository, "read_reconciliation", broken_program)
    assert category_of(compare_argv(compare_case)) == "reader_unavailable"

    # Every category is a fixed code: lower case, digits and underscores only.
    for category in categories:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", category), category
        assert " " not in category


def test_e1d_compare_main_has_no_writer_bootstrap(compare_case, monkeypatch):
    from dao.prog import da_base, da_settlement, task_execution
    from dao.prog.settlement import engine as settlement_engine
    from dao.prog.settlement import publication, service, tariffs
    from dao.prog.settlement.repository import SettlementRepository

    def forbidden(*args, **kwargs):
        pytest.fail("compare-invoice must not bootstrap a writer")

    monkeypatch.setattr(task_execution, "execution", forbidden)
    monkeypatch.setattr(da_base.DaBase, "local_settlement_app", forbidden)
    monkeypatch.setattr(da_settlement, "load_service", forbidden)
    monkeypatch.setattr(SettlementRepository, "ensure_schema", forbidden)
    monkeypatch.setattr(SettlementRepository, "register_inputs", forbidden)
    monkeypatch.setattr(SettlementRepository, "register_prices", forbidden)
    monkeypatch.setattr(SettlementRepository, "commit", forbidden)
    monkeypatch.setattr(SettlementRepository, "read_inputs", forbidden)
    monkeypatch.setattr(SettlementRepository, "read_prices", forbidden)
    monkeypatch.setattr(service.SettlementService, "consolidate", forbidden)
    monkeypatch.setattr(service.SettlementService, "prepare_planning", forbidden)
    monkeypatch.setattr(tariffs, "make_snapshot", forbidden)
    monkeypatch.setattr(settlement_engine, "settle_series", forbidden)
    monkeypatch.setattr(settlement_engine, "close_month", forbidden)
    monkeypatch.setattr(publication, "make_publication", forbidden)
    before = {item.name: item.read_bytes()
              for item in compare_case.directory.iterdir() if item.is_file()}
    assert run_cli(compare_argv(compare_case)) == 0
    after = {item.name: item.read_bytes()
             for item in compare_case.directory.iterdir() if item.is_file()}
    assert after == before
    assert not (compare_case.directory / "fast_plan.json").exists()


def test_e1d_compare_script_bootstrap(compare_case):
    root = Path(__file__).resolve().parents[3]
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "../prog/da_settlement.py", "compare-invoice",
         "--supplier", str(compare_case.supplier), "--config", str(compare_case.options)],
        cwd=root / "dao/webserver", env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "matched"
    assert report["pin"]["meter_id"]
    refused = subprocess.run(
        [sys.executable, "../prog/da_settlement.py", "compare-invoice",
         "--supplier", str(compare_case.supplier), "--config", str(compare_case.options),
         "--publish"],
        cwd=root / "dao/webserver", env=env, capture_output=True, text=True)
    assert refused.returncode == 2
    assert refused.stdout == ""


def test_e1d_compare_errors_are_sanitized(compare_case, tmp_path, capsys, caplog):
    sentinel = "SUPPLIER-SENTINEL-CONTRACT"
    secret = "SECRET-SENTINEL-VALUE"
    # An invalid supplier file leaks neither its content nor a path.
    invalid = write_supplier(tmp_path, supplier_document(contract_id=sentinel, total="not-a-decimal"),
                             name="sentinel.json")
    assert run_cli(compare_argv(compare_case, supplier=invalid)) == 2
    captured = capsys.readouterr()
    assert sentinel not in captured.out + captured.err
    assert str(tmp_path) not in captured.out + captured.err
    assert "not-a-decimal" not in captured.out + captured.err

    # A configuration that carries a secret never leaks it, not even when the
    # route refuses the configuration before it resolves or connects anything.
    directory = tmp_path / "PATH-SENTINEL"
    directory.mkdir()
    (directory / "secrets.json").write_text(json.dumps({"other_key": secret}), encoding="utf-8")
    server = {"engine": "mysql", "server": "127.0.0.1", "port": 3306, "username": "synthetic",
              "password": "!secret db_password", "database": "synthetic_reader"}
    options = write_options(directory, options_document(directory, database=server),
                            name="options.json")
    assert run_cli(["compare-invoice", "--supplier", str(compare_case.supplier),
                    "--config", str(options)]) == 2
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "PATH-SENTINEL" not in captured.out + captured.err
    assert str(directory) not in captured.out + captured.err
    assert secret not in caplog.text
    assert "PATH-SENTINEL" not in caplog.text

    # A secret that is stated in the configuration itself stays out of the output.
    inline = {"engine": "mysql", "server": "127.0.0.1", "port": 3306, "username": "synthetic",
              "password": secret, "database": "synthetic_reader"}
    options = write_options(directory, options_document(directory, meter="another-meter",
                                                         database=inline),
                            name="inline-options.json")
    assert run_cli(["compare-invoice", "--supplier", str(compare_case.supplier),
                    "--config", str(options)]) == 2
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "PATH-SENTINEL" not in captured.out + captured.err

    # A successful report carries no path either.
    assert run_cli(compare_argv(compare_case)) == 0
    captured = capsys.readouterr()
    assert str(tmp_path) not in captured.out + captured.err
    assert secret not in captured.out + captured.err
