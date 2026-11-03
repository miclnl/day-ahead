"""Task results cross real wrappers, CLI and scheduler; all I/O stays synthetic.

The process double runs the actual CLI and models sys.exit(None) as OS success,
not a canned scheduler returncode. No personal configuration is constructed.
"""

import datetime as dt
import json
import logging
import platform
import runpy
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import mip
import pandas as pd
import pytest
import requests

from dao.forecast.weather.service import WeatherStatus
from dao.lib import da_prices, db_connections
from dao.prog import da_base, da_scheduler, day_ahead, task_state, tasks, utils
from dao.prog.config import loader
from dao.prog.da_base import DaBase
from dao.prog.da_scheduler import DaScheduler
from dao.prog.settlement.publication import FinancialPlanBlocked
from dao.tests.settlement.test_optimization_producer import capture as capture
from dao.tests.settlement.test_publication import pricing_config

SCRIPT = Path(day_ahead.__file__).resolve()
REAL_CONFIG_LOAD = loader.ConfigurationLoader.load_and_validate


class FakeDb:
    tzinfo = ZoneInfo("Europe/Amsterdam")

    def __init__(self):
        self.saved = []

    def log_pool_status(self):
        logging.info("synthetic pool status")

    def get_time_border_record(self, code):
        return None

    def savedata(self, frame, tablename="values"):
        self.saved.append(frame.copy())


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("task contract attempted live I/O or an unguarded OS exit")

    for name in ("getaddrinfo", "create_connection"):
        monkeypatch.setattr(socket, name, forbidden)
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(loader.ConfigurationLoader, "load_and_validate", forbidden)
    monkeypatch.setattr(db_connections, "make_db_da", forbidden)
    monkeypatch.setattr(da_base, "make_db_da", forbidden)
    monkeypatch.setattr(da_base, "make_db_ha", forbidden)
    monkeypatch.setattr(day_ahead.os, "_exit", forbidden)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    prog = tmp_path / "prog"
    prog.mkdir()
    logs = tmp_path / "data" / "log"
    logs.mkdir(parents=True)
    monkeypatch.chdir(prog)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level

    app = DaBase.__new__(DaBase)
    app.tasks = DaBase.generate_tasks()
    app.file_name = "synthetic-unused-options"
    app.debug = False
    app.interval = "1hour"
    app.log_level = logging.INFO
    app.ha_context = None
    app.notification_entity = app.last_activity_entity = None
    app.db_da = FakeDb()
    app.prices_options = SimpleNamespace(source_day_ahead="nordpool")
    app.config = SimpleNamespace(
        history=None, boiler=None, battery=[], prices=app.prices_options,
        electric_vehicle=[], heating=None, use_calc_baseload=False,
        grid=SimpleNamespace(max_power=17.0, model_fields_set={"max_power"}),
        machines=[],
    )
    app.prices = SimpleNamespace(get_prices=lambda *a, **k: None)
    monkeypatch.setattr(day_ahead, "DaCalc", lambda file_name: app)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(day_ahead, "FAST_PLAN_FILE", str(tmp_path / "fast_plan.json"))
    yield SimpleNamespace(app=app, logs=logs, prog=prog, root=root)
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


def cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
    return day_ahead.main()


def scheduled(sandbox, monkeypatch, key):
    class CliProcess:
        def __init__(self, cmd, *, cwd, start_new_session, pass_fds, env):
            assert cwd == str(sandbox.prog)
            assert start_new_session is True
            # Model a separate child's fd table/environment: dup shares the
            # open-file-description but child close must not close the parent.
            import os

            from dao.prog import task_execution

            assert pass_fds == (int(env[task_execution.ENV_FD]),)
            child_fd = os.dup(pass_fds[0])
            with monkeypatch.context() as child_env:
                for name, value in env.items():
                    child_env.setenv(name, value)
                child_env.setenv(task_execution.ENV_FD, str(child_fd))
                result = cli(monkeypatch, *cmd[2:])
            # This is the OS meaning of sys.exit(None), including on the RED.
            self.returncode = 0 if result is None else result
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else self.returncode

        def wait(self):
            return self.returncode

    scheduler = DaScheduler.__new__(DaScheduler)
    scheduler.tasks = DaBase.generate_tasks()
    scheduler.PROG_DIR = str(sandbox.prog)
    monkeypatch.setattr(da_scheduler, "Popen", CliProcess)
    monkeypatch.setattr(da_scheduler.time, "sleep", lambda seconds: None)
    scheduler._run_exclusive(key)
    assert not task_state.is_running(key)
    return task_state.last_result(key)


@pytest.mark.parametrize("args,canonical,debug", [
    pytest.param([word], key, False, id=f"spelling-{word}")
    for key, task in tasks.TASKS.items()
    for word in (key, *task["aliases"])
] + [
    pytest.param(task["cmd"][2:], key, "debug" in task["cmd"][2:], id=f"command-{key}")
    for key, task in tasks.TASKS.items()
    if Path(task["cmd"][1]).name == SCRIPT.name
])
def test_every_registered_command_dispatches_once(monkeypatch, args, canonical, debug):
    """Missing/wrong aliases or duplicate dispatch fail without running any task."""
    dispatched = []

    class FakeDaCalc:
        def __init__(self, file_name):
            self.config = object()
            self.debug = False
            self.db_da = FakeDb()

        def run_task_function(self, key):
            dispatched.append((key, self.debug))
            return 0

    monkeypatch.setattr(day_ahead, "DaCalc", FakeDaCalc)

    assert cli(monkeypatch, *args) == 0
    assert dispatched == [(canonical, debug)]


def test_false_task_reaches_scheduler_as_error(sandbox, monkeypatch):
    sandbox.app.prices.get_prices = lambda *a, **k: False

    result = scheduled(sandbox, monkeypatch, "prices")

    assert (result["status"], result["returncode"]) == ("error", 1)
    assert result["source"] == "scheduler"
    assert "get_day_ahead_prices" in next(sandbox.logs.glob("prices_*.log")).read_text()


@pytest.mark.parametrize("legacy", [None, True])
def test_successful_none_remains_success(sandbox, monkeypatch, legacy):
    sandbox.app.prices.get_prices = lambda *a, **k: legacy

    assert sandbox.app.run_task_function("prices") == 0
    result = scheduled(sandbox, monkeypatch, "prices")
    assert (result["status"], result["returncode"]) == ("done", 0)


@pytest.mark.parametrize("status", [0, 1, 7, -3])
@pytest.mark.parametrize("logfile", [True, False])
def test_explicit_integer_status_is_not_discarded(sandbox, status, logfile):
    sandbox.app.prices.get_prices = lambda *a, **k: status
    before, level = sandbox.root.handlers[:], sandbox.root.level

    result = sandbox.app.run_task_function("prices", logfile=logfile)

    assert type(result) is int and result == status
    assert sandbox.root.handlers == before and sandbox.root.level == level


@pytest.mark.parametrize("first", [False, 7])
def test_first_failure_is_not_overwritten(sandbox, monkeypatch, first):
    visited = []
    sandbox.app.prices.get_prices = lambda *a, **k: first
    sandbox.app.meteo = SimpleNamespace(
        get_meteo_data=lambda show: visited.append("meteo"))

    result = cli(monkeypatch, "prices", "meteo")

    assert result == (1 if first is False else 7)
    assert visited == []
    assert not list(sandbox.logs.glob("meteo_*.log"))


def configure_prices(sandbox, monkeypatch, source, payload=None, error=False):
    sandbox.app.prices_options.source_day_ahead = source
    config = SimpleNamespace(
        interval="1hour", prices=SimpleNamespace(entsoe_api_key=None),
        tibber=SimpleNamespace(api_token=SimpleNamespace(resolve=lambda secrets: "synthetic"), api_url=None),
    )
    sandbox.app.prices = da_prices.DaPrices(config, sandbox.app.db_da)

    def fetch(*args, **kwargs):
        if error:
            raise requests.RequestException("synthetic provider outage")
        return payload

    class Response:
        def raise_for_status(self):
            if error:
                raise requests.RequestException("synthetic provider outage")

        def json(self):
            return payload

    monkeypatch.setattr(da_prices, "Prices", lambda: SimpleNamespace(fetch=fetch))
    monkeypatch.setattr(da_prices, "EntsoePandasClient", lambda **kw: SimpleNamespace(query_day_ahead_prices=fetch))
    monkeypatch.setattr(da_prices, "get", lambda *a, **k: Response())
    monkeypatch.setattr(da_prices, "post", lambda *a, **k: Response())


@pytest.mark.parametrize("source,payload,error", [
    ("nordpool", None, True), ("nordpool", None, False),
    ("nordpool", {"areas": {}}, False),
    ("nordpool", {"areas": {"NL": {"values": []}}}, False),
    ("entsoe", None, True), ("entsoe", pd.Series(dtype=float), False),
    ("easyenergy", None, True), ("easyenergy", [], False),
    ("tibber", None, True), ("tibber", {"errors": ["synthetic denial"]}, False),
    ("tibber", {"data": {}}, False),
    ("tibber", {"data": {"viewer": {"homes": [{"currentSubscription": {
        "priceInfo": {"today": [], "tomorrow": []}, "priceInfoRange": {"nodes": []},
    }}]}}}, False),
])
def test_provider_failure_reaches_cli(sandbox, monkeypatch, source, payload, error):
    configure_prices(sandbox, monkeypatch, source, payload, error)

    result = cli(monkeypatch, "prices", "2026-06-15", "2026-06-16")

    assert result == 1
    assert sandbox.app.db_da.saved == []
    assert next(sandbox.logs.glob("prices_*.log")).read_text()


def test_partial_prices_remain_partial_even_if_fetch_task_succeeds(sandbox, monkeypatch):
    at = dt.datetime(2026, 6, 15, tzinfo=ZoneInfo("CET"))
    payload = {"areas": {"NL": {"values": [
        {"start": at, "end": at + dt.timedelta(hours=1), "value": 100.0},
    ]}}}
    configure_prices(sandbox, monkeypatch, "nordpool", payload)

    assert cli(monkeypatch, "prices", "2026-06-15", "2026-06-16") == 0
    assert len(sandbox.app.db_da.saved) == 1
    assert len(sandbox.app.db_da.saved[0]) == 1
    assert "1 of 24 values" in next(sandbox.logs.glob("prices_*.log")).read_text()


@pytest.mark.parametrize("alias", ["prices", "get_prices", "update_prices"])
def test_main_argv_preserves_price_dates_and_debug(sandbox, monkeypatch, alias):
    received = []
    sandbox.app.prices.get_prices = lambda source, **kw: received.append((source, kw))
    argv = ["debug", alias, "2026-06-15", "2026-06-17"]
    original = sys.argv

    assert day_ahead.main(argv) == 0
    assert sandbox.app.debug is True
    assert argv == ["debug", alias, "2026-06-15", "2026-06-17"]
    assert sys.argv is original
    assert received == [("nordpool", {"_start": dt.datetime(2026, 6, 15), "_end": dt.datetime(2026, 6, 17)})]


def test_invalid_price_date_fails_without_fetch(sandbox, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid price date reached provider")

    sandbox.app.prices.get_prices = forbidden
    assert cli(monkeypatch, "prices", "2026-02-30") == 1


def test_single_price_date_retains_one_day_range(sandbox, monkeypatch):
    received = []
    sandbox.app.prices.get_prices = lambda source, **kw: received.append(kw)

    assert cli(monkeypatch, "prices", "2026-06-15") == 0
    assert received == [{"_start": dt.datetime(2026, 6, 15), "_end": dt.datetime(2026, 6, 16)}]


@pytest.mark.parametrize("key", ["no_such_task", "fast_once"])
def test_refused_in_process_task_is_failure(sandbox, key):
    assert sandbox.app.run_task_function(key) == 1


def test_missing_configuration_is_cli_failure(sandbox, monkeypatch):
    sandbox.app.config = None
    assert cli(monkeypatch, "prices") == 1
    assert not list(sandbox.logs.iterdir())


def test_unknown_cli_task_does_not_run_later_task(sandbox, monkeypatch):
    assert cli(monkeypatch, "no_such_task", "prices") == 1
    assert not list(sandbox.logs.iterdir())


def test_exception_propagates_after_logger_cleanup(sandbox, monkeypatch):
    error = RuntimeError("synthetic task exception")

    def fail(*args, **kwargs):
        raise error

    sandbox.app.prices.get_prices = fail
    before, level = sandbox.root.handlers[:], sandbox.root.level
    original = sys.argv
    with pytest.raises(RuntimeError) as caught:
        day_ahead.main(["prices", "meteo"])
    assert caught.value is error
    assert sandbox.root.handlers == before and sandbox.root.level == level
    assert sys.argv is original
    assert "synthetic task exception" in next(sandbox.logs.glob("prices_*.log")).read_text()


@pytest.mark.parametrize("hours,want", [(0, 1), (1, 0)])
def test_weather_result_uses_data_not_truthy_status_object(sandbox, monkeypatch, hours, want):
    at = dt.datetime(2026, 6, 15, tzinfo=dt.UTC)
    status = WeatherStatus(at, hours, {"synthetic": hours}, "synthetic", "primary unavailable", at if hours else None,
                           pd.DataFrame({"time": [int(at.timestamp())] * hours}))
    sandbox.app.meteo = SimpleNamespace(get_meteo_data=lambda show: status)

    assert cli(monkeypatch, "meteo") == want


@pytest.mark.parametrize("method,key", [
    ("calc_optimum", "calc_optimum"),
    ("calc_optimum_met_debug", "calc_optimum_met_debug"),
])
def test_optimizer_wrapper_forwards_failure(sandbox, monkeypatch, method, key):
    optimizer = SimpleNamespace(debug=None, calc_optimum=lambda: False)
    monkeypatch.setitem(sys.modules, "day_ahead", SimpleNamespace(DaCalc=lambda file_name: optimizer))

    assert sandbox.app.run_task_function(key) == 1
    assert optimizer.debug is (method == "calc_optimum_met_debug")


@pytest.mark.parametrize("key", ["train_ml_predictions", "calc_baseloads", "consolidate", "fast_control_simulate", "tibber"])
def test_other_status_wrappers_forward_failure(sandbox, monkeypatch, key):
    sandbox.app.pv_service = lambda: SimpleNamespace(run_training=lambda: False)
    sandbox.app.baseload_service = lambda: SimpleNamespace(fit=lambda: False)
    monkeypatch.setitem(sys.modules, "da_report", SimpleNamespace(Report=lambda *a: SimpleNamespace(
        check_baseload_sensors=lambda: None, consolidate_data=lambda start: False)))
    monkeypatch.setitem(sys.modules, "da_fast", SimpleNamespace(main=lambda argv: False))
    monkeypatch.setattr(da_base, "get_tibber_data", lambda: False)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), key])

    assert sandbox.app.run_task_function(key) == 1


@pytest.mark.parametrize("word,method", [("train", "train_ml_predictions"), ("accuracy", "forecast_accuracy"), ("clean_data", "clean_data")])
def test_legacy_cli_spelling_carries_failure_without_live_task(sandbox, monkeypatch, word, method):
    monkeypatch.setattr(sandbox.app, method, lambda: 7)

    assert cli(monkeypatch, word) == 7


def test_calculation_debug_wrapper_returns_failure(capture, monkeypatch):
    instance = capture["instance"]
    monkeypatch.setattr(instance, "calc_optimum", lambda: False)
    assert instance.calc_optimum_debug() is False
    assert instance.debug is True


def test_invalid_consolidation_date_is_failure(sandbox, monkeypatch):
    monkeypatch.setitem(sys.modules, "da_report", SimpleNamespace(Report=lambda: SimpleNamespace(
        consolidate_data=lambda start: pytest.fail("invalid date reached consolidation"))))

    assert cli(monkeypatch, "consolidate", "2026-02-30") == 1


@pytest.mark.parametrize("status", [False, 7])
def test_settlement_wrapper_forwards_failure(sandbox, monkeypatch, status):
    sandbox.app.config.prices.settlement = SimpleNamespace(mode="shadow")

    def consolidate(start, end, *, publish, reason):
        assert (start, end, publish, reason) == (1, 2, True, "scheduled")
        return status

    service = SimpleNamespace(scheduled_range=lambda: (1, 2), consolidate=consolidate)
    monkeypatch.setattr(sandbox.app, "settlement_service", lambda: service)

    assert cli(monkeypatch, "settlement_consolidate") == (1 if status is False else 7)


@pytest.mark.parametrize("payload,error,start", [
    (None, True, "2026-06-01"), ({"errors": ["synthetic denial"]}, False, "2026-06-01"),
    ({"data": {}}, False, "2026-06-01"), (None, False, "2026-02-30"),
])
def test_handled_tibber_history_failure_reaches_cli(sandbox, monkeypatch, payload, error, start):
    config = SimpleNamespace(
        tibber=SimpleNamespace(api_url=None, api_token=SimpleNamespace(resolve=lambda secrets: "synthetic")),
        prices=SimpleNamespace(last_invoice=dt.date(2026, 6, 1)))

    class Loader:
        def __init__(self, *args):
            self.secrets = {}

        def load_and_validate(self):
            return config

    class Response:
        def raise_for_status(self):
            if error:
                raise requests.RequestException("synthetic history outage")

        def json(self):
            return payload

    monkeypatch.setattr(loader, "ConfigurationLoader", Loader)
    monkeypatch.setattr(db_connections, "make_db_da", lambda *a: sandbox.app.db_da)
    monkeypatch.setattr(utils, "post", lambda *a, **k: Response())

    assert cli(monkeypatch, "tibber", start) == 1
    assert sandbox.app.db_da.saved == []


@pytest.mark.parametrize("reason", ["prices", "quarter-gap", "meteo", "short-meteo", "baseload", "soc-limits",
                                   "no-solution", "no-solution-consumption", "no-solution-retry", "no-solution-final", "strategy"])
def test_fatal_optimizer_abort_is_not_done(capture, sandbox, monkeypatch, reason):
    optimizer = capture["instance"]
    optimizer._solver_threads = 1
    prices = capture["locals"]["price_data"].copy()
    forecast = capture["locals"]["prog_data"][["time", "tijd", "glob_rad", "temp"]].copy()
    start = capture["locals"]["start_dt"]
    if reason == "prices":
        prices = prices.iloc[:0]
    elif reason == "quarter-gap":
        prices.loc[prices.index[-1], "time"] = start + dt.timedelta(days=2)
    elif reason == "meteo":
        forecast = None
    elif reason == "short-meteo":
        forecast = forecast.iloc[:2]
    elif reason == "baseload":
        monkeypatch.setattr(optimizer, "_baseload_for_horizon", lambda *a: [])
    elif reason == "soc-limits":
        optimizer.battery_options[0].entity_min_soc_end_opt = "sensor.synthetic_min"
        optimizer.battery_options[0].entity_max_soc_end_opt = "sensor.synthetic_max"
    elif reason in ("no-solution", "no-solution-consumption"):
        if reason == "no-solution-consumption":
            optimizer.strategy = "minimize consumption"
        monkeypatch.setattr(mip.Model, "optimize", lambda *a, **k: mip.OptimizationStatus.INFEASIBLE)
    elif reason == "no-solution-retry":
        optimizer.strategy = "minimize consumption"
        optimize = mip.Model.optimize
        solves = []

        def exhaust(model):
            solves.append(True)
            if len(solves) == 1:
                return optimize(model)
            monkeypatch.setattr(mip.Model, "num_solutions", property(lambda self: 0))
            return mip.OptimizationStatus.INFEASIBLE

        monkeypatch.setattr(mip.Model, "optimize", exhaust)
    elif reason == "no-solution-final":
        counts = iter((1, 0))

        def no_final_solution(model):
            monkeypatch.setattr(mip.Model, "num_solutions", property(lambda self: next(counts)))
            return mip.OptimizationStatus.INFEASIBLE

        monkeypatch.setattr(mip.Model, "optimize", no_final_solution)
    elif reason == "strategy":
        optimizer.strategy = "synthetic-invalid-strategy"

    class FrozenDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return start if tz is None else start.astimezone(tz)

    monkeypatch.setattr(day_ahead, "dt", SimpleNamespace(datetime=FrozenDatetime, timedelta=dt.timedelta))
    monkeypatch.setattr(day_ahead, "Report", lambda *a: SimpleNamespace(get_price_data=lambda *a, **k: prices.copy()))
    optimizer.db_da = SimpleNamespace(get_prognose_data=lambda **kw: None if forecast is None else forecast.copy())
    monkeypatch.setitem(sys.modules, "day_ahead", SimpleNamespace(DaCalc=lambda file_name: optimizer))

    result = scheduled(sandbox, monkeypatch, "calc_optimum_met_debug")

    assert (result["status"], result["returncode"]) == ("error", 1)
    if reason == "no-solution-retry":
        assert len(solves) == 3  # Existing retry sequence, not a new solve budget.
    content = next(sandbox.logs.glob("calc_debug_*.log")).read_text()
    assert any(message in content for message in ("afgebroken", "kan niet worden gerekend", "Geen oplossing", "geen oplossing", "Kies een strategie", "moet groter zijn"))
    assert "DAO calc afgerond" not in content


def test_financial_gate_exception_remains_failure_after_cleanup(capture, sandbox, monkeypatch):
    optimizer = capture["instance"]
    optimizer.config.prices = pricing_config()
    for name in ("_financial_pricing", "_financial_policy", "_financial_plan_path"):
        delattr(optimizer, name)
    start = capture["locals"]["start_dt"]

    class FrozenDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return start if tz is None else start.astimezone(tz)

    monkeypatch.setattr(day_ahead, "dt", SimpleNamespace(datetime=FrozenDatetime, timedelta=dt.timedelta))
    monkeypatch.setitem(sys.modules, "day_ahead", SimpleNamespace(DaCalc=lambda file_name: optimizer))
    before, level = sandbox.root.handlers[:], sandbox.root.level

    with pytest.raises(FinancialPlanBlocked, match="financial_plan_required"):
        cli(monkeypatch, "debug", "calc")
    assert sandbox.root.handlers == before and sandbox.root.level == level
    assert "financial_plan_required" in next(sandbox.logs.glob("calc_debug_*.log")).read_text()


@pytest.mark.parametrize("machine", ["x86_64", "arm64", "aarch64"])
@pytest.mark.parametrize("status", [0, 7])
def test_real_script_entrypoint_preserves_os_status(sandbox, monkeypatch, machine, status):
    # run_path defines the real DaCalc anew; only its external constructor I/O
    # is replaced, while inherited task/provider wrappers and __main__ run.
    monkeypatch.setattr(DaBase, "__init__", lambda self, **kw: self.__dict__.update(sandbox.app.__dict__))
    sandbox.app.prices.get_prices = lambda *a, **k: status
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "prices"])
    monkeypatch.setattr(platform, "machine", lambda: machine)

    class HardExit(BaseException):
        def __init__(self, code):
            self.code = code

    def hard_exit(code):
        raise HardExit(code)

    monkeypatch.setattr(day_ahead.os, "_exit", hard_exit)
    with pytest.raises((SystemExit, HardExit)) as caught:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert caught.value.code == status
    assert isinstance(caught.value, HardExit) is (machine in ("arm64", "aarch64"))


@pytest.mark.parametrize("argv", [[], ["debug"]])
@pytest.mark.parametrize("configuration,want", [("missing", 1), ("invalid", 1), ("valid", 0)])
def test_no_task_configuration_status_without_publishing_constructor(sandbox, monkeypatch, argv, configuration, want):
    path = sandbox.prog.parent / "data" / "options.json"
    if configuration != "missing":
        path.write_text(json.dumps({"config_version": 3 if configuration == "valid" else 999}))
    (path.parent / "secrets.json").write_text("{}")
    monkeypatch.setattr(DaBase, "_config", None)
    monkeypatch.setattr(DaBase, "_loader", None)
    loaded = []

    def load(instance):
        # Permit the real app loader only for this synthetic sandbox file;
        # the suite's other network/DB/constructor guards remain in force.
        assert instance.config_path.resolve() == path.resolve()
        loaded.append(instance.config_path.resolve())
        return REAL_CONFIG_LOAD(instance)

    def forbidden(*args, **kwargs):
        pytest.fail("no-task CLI constructed a publishing app")

    monkeypatch.setattr(loader.ConfigurationLoader, "load_and_validate", load)
    monkeypatch.setattr(day_ahead, "DaCalc", forbidden)
    assert day_ahead.main(argv) == want
    assert loaded == [path.resolve()]
    assert (DaBase._config is not None) is (configuration == "valid")
    assert not list(sandbox.logs.iterdir())
