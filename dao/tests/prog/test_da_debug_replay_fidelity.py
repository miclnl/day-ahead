"""Missing replay input must stop; only recorded HA failures may use defaults."""

import datetime as dt
import json
import os
import socket
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import freezegun.api
import mip
import pandas as pd
import pytest

from dao.prog import da_base, da_debug, day_ahead
from dao.prog.config.models.base import FlexFloat
from dao.prog.config.versions.v3 import ConfigurationV3

ENTITY = "sensor.synthetic_soc"
TOKEN = "synthetic-error-token-not-a-credential"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("fidelity test attempted live I/O")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    for name in ("HAClient", "make_db_da", "make_db_ha"):
        monkeypatch.setattr(da_base, name, forbidden)


@pytest.fixture
def snapshot():
    # Deliberately a v1 snapshot without any read-error field.
    return {
        "meta": {"schema_version": 1, "captured_at": "2026-01-16T12:00:00"},
        "config": {"config_version": 3},
        "ha_states": {},
    }


@pytest.fixture
def instance():
    # Avoid DaBase construction (filesystem, HA and providers), not its read helpers.
    base = da_base.DaBase.__new__(da_base.DaBase)
    base.config = ConfigurationV3()
    base.loader = SimpleNamespace(secrets={})
    return base


@pytest.mark.parametrize("reader", ["guarded", "flex"])
@pytest.mark.parametrize("after_solve", [False, True])
def test_missing_soc_aborts_instead_of_defaulting(snapshot, instance, reader, after_solve):
    with da_debug.ReplayIO(snapshot) as replay:
        if after_solve:
            model = mip.Model(solver_name=mip.CBC)
            model.verbose = 0
            model.objective = model.add_var(lb=1, ub=1)
            assert model.optimize() == mip.OptimizationStatus.OPTIMAL
            assert replay._model is model
        with pytest.raises(da_debug.SnapshotMiss, match=ENTITY):
            if reader == "guarded":
                instance.get_float(ENTITY, 50.0, "synthetic SoC")
            else:
                FlexFloat(value=ENTITY).resolve(instance.ha_getter, default=50.0)
        assert replay.reads == {ENTITY}


@pytest.mark.parametrize("error_type,category", [
    (RuntimeError, "error"),
    (TimeoutError, "timeout"),
    (ConnectionError, "connection"),
    (type(TOKEN, (Exception,), {}), "error"),
])
def test_recorded_read_failure_replays_fallback(tmp_path, instance, caplog, error_type, category):
    failure = error_type(f"Authorization: Bearer {TOKEN}")

    def fail_read(*, entity_id):
        assert entity_id == ENTITY
        raise failure

    instance._ha_client = SimpleNamespace(get_state=fail_read)
    with da_debug.RecordingIO(tmp_path, debug=True) as recording:
        recording._primary = instance
        # Recording still propagates the original error to production callers.
        with pytest.raises(error_type) as caught:
            instance.get_state(ENTITY)
        assert caught.value is failure
        assert instance.get_float(ENTITY, 50.0) == 50.0

    captured = json.loads(recording.snapshot_path.read_text())
    assert captured.get("ha_read_errors") == {ENTITY: category}
    assert ENTITY not in captured["ha_states"]
    assert TOKEN not in recording.snapshot_path.read_text()
    caplog.clear()
    with da_debug.ReplayIO(recording.snapshot_path) as replay:
        with pytest.raises(da_debug.RecordedReadError) as caught:
            instance.get_state(ENTITY)
        assert caught.value.category == category
        assert instance.get_float(ENTITY, 50.0) == 50.0
        assert FlexFloat(value=ENTITY).resolve(instance.ha_getter, default=20.0) == 20.0
        assert replay.reads == {ENTITY}
    assert TOKEN not in str(caught.value)
    assert TOKEN not in caplog.text


def test_recorded_error_category_is_sanitized_on_replay(snapshot, instance, caplog):
    assert hasattr(da_debug, "RecordedReadError"), "recorded failures need a separate exception"
    snapshot["ha_read_errors"] = {ENTITY: f"Authorization: Bearer {TOKEN}"}
    with da_debug.ReplayIO(snapshot):
        with pytest.raises(da_debug.RecordedReadError) as caught:
            instance.get_state(ENTITY)
        assert caught.value.category == "error"
        assert instance.get_float(ENTITY, 50.0) == 50.0
    assert TOKEN not in str(caught.value)
    assert TOKEN not in caplog.text


@pytest.mark.parametrize("state,expected", [("0", 0.0), ("74.25", 74.25), ("unavailable", 50.0)])
def test_legacy_snapshot_preserves_present_states(snapshot, instance, state, expected):
    snapshot["ha_states"][ENTITY] = state
    with da_debug.ReplayIO(snapshot):
        assert instance.get_float(ENTITY, 50.0) == expected


@pytest.mark.parametrize("first_fails", [False, True])
def test_recording_keeps_latest_read_outcome(tmp_path, instance, first_fails):
    outcomes = [RuntimeError(TOKEN), SimpleNamespace(state="74.25")]
    if not first_fails:
        outcomes.reverse()

    def read(*, entity_id):
        assert entity_id == ENTITY
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    instance._ha_client = SimpleNamespace(get_state=read)
    with da_debug.RecordingIO(tmp_path, debug=True) as recording:
        recording._primary = instance
        for _ in range(2):
            instance.get_float(ENTITY, 50.0)
    captured = json.loads(recording.snapshot_path.read_text())
    assert captured.get("ha_read_errors") == ({} if first_fails else {ENTITY: "error"})
    assert captured["ha_states"] == ({ENTITY: "74.25"} if first_fails else {})
    with da_debug.ReplayIO(recording.snapshot_path):
        assert instance.get_float(ENTITY, 50.0) == (74.25 if first_fails else 50.0)


@pytest.mark.parametrize("context_type", [da_debug.RecordingIO, da_debug.ReplayIO])
@pytest.mark.parametrize("error_type", [RuntimeError, da_debug.SnapshotMiss])
def test_enter_failure_restores_patches(tmp_path, snapshot, monkeypatch, context_type, error_type):
    context = context_type(tmp_path, debug=True) if context_type is da_debug.RecordingIO else context_type(snapshot)
    original_set = da_debug._PatchList.set
    originals = []
    failure = error_type("controlled partial enter failure")

    def fail_after_patch(patches, target, name, value):
        originals.append((target, name, name in vars(target), vars(target).get(name)))
        original_set(patches, target, name, value)
        if name == "get_price_data":
            raise failure

    monkeypatch.setattr(da_debug._PatchList, "set", fail_after_patch)
    try:
        with pytest.raises(error_type) as caught, context:
            pytest.fail("failed enter must not reach the body")
        assert caught.value is failure
        assert originals
        assert context._patches._entries == []
        for target, name, present, value in originals:
            assert (name in vars(target)) == present
            assert vars(target).get(name) is value
    finally:
        # Keep the RED run hermetic even when the production cleanup is broken.
        context._patches.restore()


@pytest.mark.parametrize("error_type", [RuntimeError, da_debug.SnapshotMiss])
def test_enter_failure_stops_started_clock(snapshot, monkeypatch, error_type):
    replay = da_debug.ReplayIO(snapshot)
    original_start = freezegun.api._freeze_time.start
    datetime_class = dt.datetime
    original_get_state = da_base.DaBase.get_state
    original_optimize = mip.Model.optimize
    clock_count = len(freezegun.api.freeze_factories)
    failure = error_type("controlled failure after clock start")

    def fail_after_start(freeze):
        original_start(freeze)
        raise failure

    monkeypatch.setattr(freezegun.api._freeze_time, "start", fail_after_start)
    try:
        with pytest.raises(error_type) as caught, replay:
            pytest.fail("failed enter must not reach the body")
        assert caught.value is failure
        assert replay._freeze is None
        assert dt.datetime is datetime_class
        assert len(freezegun.api.freeze_factories) == clock_count
        assert da_base.DaBase.get_state is original_get_state
        assert mip.Model.optimize is original_optimize
        assert replay._patches._entries == []
    finally:
        replay.__exit__(None, None, None)


def test_clock_stop_failure_still_restores_patches(snapshot, monkeypatch):
    replay = da_debug.ReplayIO(snapshot)
    original_get_state = da_base.DaBase.get_state
    with pytest.raises(RuntimeError, match="controlled clock stop failure"), replay:
        original_stop = replay._freeze.stop

        def fail_after_stop():
            original_stop()
            raise RuntimeError("controlled clock stop failure")

        monkeypatch.setattr(replay._freeze, "stop", fail_after_stop)
    try:
        assert replay._freeze is None
        assert da_base.DaBase.get_state is original_get_state
        assert replay._patches._entries == []
    finally:
        replay._patches.restore()


def test_snapshot_exception_metadata_is_sanitized(tmp_path, instance):
    failure = RuntimeError(f"Authorization: Bearer {TOKEN}")
    with pytest.raises(RuntimeError) as caught, da_debug.RecordingIO(tmp_path, debug=True) as recording:
        recording._primary = instance
        raise failure
    assert caught.value is failure
    captured = json.loads(recording.snapshot_path.read_text())
    assert captured["meta"]["exception"] == "error"
    assert TOKEN not in recording.snapshot_path.read_text()


def test_cli_reports_missing_snapshot_input(tmp_path, snapshot, instance, monkeypatch, capsys):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(snapshot))
    instance.calc_optimum = lambda: instance.get_float(ENTITY, 50.0)
    monkeypatch.setattr(day_ahead, "DaCalc", lambda *args: instance)
    result = da_debug.main(["replay", str(path), "--json"])
    output = json.loads(capsys.readouterr().out)
    assert result == da_debug.EXIT_SNAPSHOT_MISS
    assert output["error"] == "snapshot-miss"
    assert ENTITY in output["message"]


def test_capture_cli_sanitizes_read_failure(tmp_path, monkeypatch, capsys):
    path = tmp_path / "synthetic-options.json"
    path.write_text("{}")

    def fail_read(*, entity_id):
        raise RuntimeError(f"Authorization: Bearer {TOKEN}")

    def initialize(base, file_name=None):
        base.config = ConfigurationV3()
        base.loader = SimpleNamespace(secrets={})
        base.interval = "1hour"
        base._ha_client = SimpleNamespace(get_state=fail_read)

    monkeypatch.setattr(da_base.DaBase, "__init__", initialize)
    monkeypatch.setattr(day_ahead.DaCalc, "calc_optimum", lambda base: base.get_state(ENTITY))
    result = da_debug.main(["capture", "--config", str(path), "--out-dir", str(tmp_path), "--debug", "--json"])
    output = capsys.readouterr().out
    assert result == da_debug.EXIT_SOLVER_FAILURE
    assert json.loads(output)["error"] == "error"
    assert TOKEN not in output


# Clock/TZ mutation belongs to a dedicated process, never a scheduler process.
# Clock tests stop the real producer at preflight, before external inputs.
# Presence tests use the real service with synthetic HA and a DB write spy.
_ISOLATED_SETUP = """
import datetime as dt
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import freezegun.api
import mip

from dao.prog import da_base, da_debug, day_ahead
from dao.prog.config.versions.v3 import ConfigurationV3

data = json.loads(sys.stdin.read())
time.tzset()

def forbidden(*args, **kwargs):
    raise AssertionError("isolated fidelity test attempted live I/O")

socket.socket.connect = socket.socket.connect_ex = forbidden
da_base.HAClient = da_base.make_db_da = da_base.make_db_ha = forbidden

class StartObserved(Exception):
    pass

def calculator():
    base = day_ahead.DaCalc.__new__(day_ahead.DaCalc)
    base.config = ConfigurationV3()
    base.loader = SimpleNamespace(secrets={})
    base.time_zone = "Europe/Amsterdam"
    base.interval, base.interval_s = "15min", 900
    base.debug, base.log_level = False, logging.INFO

    def preflight(**kwargs):
        local = sys._getframe(1).f_locals
        raise StartObserved({
            "start_ts": local["start_ts"],
            "soc": local["_start_soc"], "ev_soc": local["_start_ev_soc"],
            "local": local["start_dt"].isoformat(), "fold": local["start_dt"].fold,
            "now_ts": dt.datetime.now().timestamp(),
            "utc_now_ts": dt.datetime.now(dt.UTC).timestamp(),
            "today": dt.date.today().isoformat(),
            "localtime": time.strftime("%Y-%m-%dT%H:%M", time.localtime()),
        })

    base._financial_preflight = preflight
    return base

def observe(base, *args, **kwargs):
    try:
        base.calc_optimum(*args, **kwargs)
    except StartObserved as exc:
        return exc.args[0]
    raise AssertionError("producer did not reach preflight")

def presence_service(base, data_dir):
    states = {"person.synthetic_home": "home", "person.synthetic_away": "not_home"}
    base.config.baseload_options.absence.entities_presence = list(states)
    base._ha_client = SimpleNamespace(
        get_state=lambda *, entity_id: SimpleNamespace(state=states[entity_id]))
    writes = []
    base.db_da = SimpleNamespace(save_hourly_value=lambda code, moment, value:
                                writes.append([code, value]))
    base.db_ha = None
    base.ha_context = da_base.HAContext(0.0, 0.0, base.time_zone, "NL")
    service = base.baseload_service()
    service.data_dir = Path(data_dir)
    return service, writes
"""


def _isolated_replay(code, data, *, process_zone="UTC"):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    if process_zone is None:
        env.pop("TZ", None)
    else:
        env["TZ"] = process_zone
    child = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_ISOLATED_SETUP) + textwrap.dedent(code)],
        input=json.dumps(data), env=env, capture_output=True, text=True, timeout=60,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    return json.loads(child.stdout)


@pytest.mark.parametrize("recorded_presence", [False, True])
@pytest.mark.parametrize("body_fails", [False, True])
def test_replay_presence_does_not_write(tmp_path, snapshot, recorded_presence, body_fails):
    # Removing the explicit write boundary must expose either the real DB
    # write or the incidental HA read, even on a service made before entry.
    if recorded_presence:
        snapshot["ha_states"] = {"person.synthetic_home": "home", "person.synthetic_away": "not_home"}
    output = _isolated_replay("""
        base = calculator()
        service, writes = presence_service(base, data["data_dir"])
        service.record_presence()
        assert writes == [["presence", 0.5]]
        writes.clear()
        replay = da_debug.ReplayIO(data["snapshot"])
        try:
            with replay:
                service.record_presence()
                assert writes == [], writes
                assert replay.reads == set(), replay.reads
                if data["body_fails"]:
                    raise RuntimeError("controlled presence body failure")
        except RuntimeError as exc:
            assert data["body_fails"] and str(exc) == "controlled presence body failure"
        service.record_presence()
        assert writes == [["presence", 0.5]]
        assert replay._patches._entries == []
        print(json.dumps({"writes_after_cleanup": writes}))
    """, {"snapshot": snapshot, "data_dir": str(tmp_path), "body_fails": body_fails})
    assert output == {"writes_after_cleanup": [["presence", 0.5]]}


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("body_fails", [False, True])
def test_capture_presence_write_boundary_is_debug_only(tmp_path, debug, body_fails):
    # Debug capture must suppress presence, but ordinary recording remains
    # passthrough. Both normal and exceptional exits restore the real service.
    output = _isolated_replay("""
        base = calculator()
        service, writes = presence_service(base, data["out_dir"])
        recording = da_debug.RecordingIO(data["out_dir"], debug=data["debug"])
        try:
            with recording:
                recording._primary = base
                service.record_presence()
                assert writes == ([] if data["debug"] else [["presence", 0.5]]), writes
                if data["body_fails"]:
                    raise RuntimeError("controlled presence body failure")
        except RuntimeError as exc:
            assert data["body_fails"] and str(exc) == "controlled presence body failure"
        snapshot = json.loads(recording.snapshot_path.read_text())
        assert snapshot["ha_states"] == ({} if data["debug"] else {
            "person.synthetic_home": "home", "person.synthetic_away": "not_home"})
        writes.clear()
        service.record_presence()
        assert writes == [["presence", 0.5]]
        assert recording._patches._entries == []
        print(json.dumps({"writes_after_cleanup": writes}))
    """, {"out_dir": str(tmp_path), "debug": debug, "body_fails": body_fails})
    assert output == {"writes_after_cleanup": [["presence", 0.5]]}


@pytest.mark.parametrize("debug", [False, True])
def test_producer_debug_presence_does_not_write(tmp_path, debug):
    # Exercise calc_optimum itself, not a reconstructed debug branch. Stop at
    # the next (solar) boundary, after a real presence service could write.
    output = _isolated_replay("""
        import pandas as pd

        base = calculator()
        service, writes = presence_service(base, data["data_dir"])
        base.debug = data["debug"]
        base.interval, base.interval_s, base.steps_day = "1hour", 3600, 24
        base.file_name = "synthetic-unused-options"
        base.config.baseload = [0.2] * 24
        base.use_calc_baseload = False
        base.battery_options = []
        base.solar = [SimpleNamespace(entity_pv_switch=None, max_power=1.0)]
        base._financial_preflight = lambda **kwargs: None
        base.notify = lambda *args: None
        dates = pd.date_range("2026-01-16T12:00:00", periods=9, freq="h")
        prices = pd.DataFrame({"time": dates, "da_ex": 0.1, "da_cons": 0.2, "da_prod": 0.1})
        forecast = pd.DataFrame({"tijd": dates})
        base.db_da.get_prognose_data = lambda **kwargs: forecast.copy()

        class PresenceObserved(Exception):
            pass

        def stop_at_solar(*args):
            raise PresenceObserved

        base.calc_solar_predictions = stop_at_solar
        report = SimpleNamespace(get_price_data=lambda *args, **kwargs: prices.copy())
        with patch.object(day_ahead, "Report", lambda *args: report), \
             freezegun.freeze_time("2026-01-16T12:00:00"):
            try:
                base.calc_optimum()
            except PresenceObserved:
                pass
            else:
                raise AssertionError("producer did not reach the boundary after presence")
        print(json.dumps({"writes": writes}))
    """, {"data_dir": str(tmp_path), "debug": debug})
    assert output["writes"] == ([] if debug else [["presence", 0.5]])


@pytest.mark.parametrize("channel", ["ha", "db"])
def test_replay_unknown_read_after_first_solve_still_aborts(tmp_path, snapshot, channel):
    # A real solved model is no licence to default an unknown service input;
    # current_regime's ordinary Exception fallbacks must not swallow the miss.
    output = _isolated_replay("""
        base = calculator()
        service, writes = presence_service(base, data["data_dir"])
        try:
            with da_debug.ReplayIO(data["snapshot"]) as replay:
                model = mip.Model(solver_name=mip.CBC)
                model.verbose = 0
                model.objective = model.add_var(lb=1, ub=1)
                assert model.optimize() == mip.OptimizationStatus.OPTIMAL
                assert replay._model is model
                if data["channel"] == "ha":
                    base.config.baseload_options.absence.entity_away = "sensor.synthetic_soc"
                    service.db_da = None
                else:
                    service.db_da = da_base.make_db_da()
                service.current_regime(dt.date(2026, 1, 17))
        except da_debug.SnapshotMiss as exc:
            message = str(exc)
        else:
            raise AssertionError("unknown read after first solve did not abort")
        assert writes == []
        assert replay._patches._entries == []
        print(json.dumps({"message": message, "reads": sorted(replay.reads)}))
    """, {"snapshot": snapshot, "data_dir": str(tmp_path), "channel": channel})
    assert (ENTITY if channel == "ha" else "get_column_data") in output["message"]
    assert output["reads"] == ([ENTITY] if channel == "ha" else [])


@pytest.mark.parametrize("positional", [False, True])
def test_start_overrides_roundtrip(tmp_path, positional):
    # Dropping captured overrides, or treating 0/None as absent, changes the
    # arguments the real producer uses. Both positional and keyword callers win.
    output = _isolated_replay("""
        os.environ["TZ"] = "Europe/Amsterdam"
        time.tzset()
        base = calculator()
        start = dt.datetime(2026, 10, 25, 2, 30, fold=1)
        with da_debug.RecordingIO(data["out_dir"], debug=True) as recording:
            recording._primary = base
            if data["positional"]:
                original = observe(base, start, 0.0, 67.0)
            else:
                original = observe(base, _start_dt=start, _start_soc=0.0, _start_ev_soc=67.0)
        snapshot = json.loads(recording.snapshot_path.read_text())
        os.environ["TZ"] = "UTC"
        time.tzset()
        with da_debug.ReplayIO(snapshot):
            replayed = observe(base)
            partial = observe(base, _start_soc=31.0)
            cleared = observe(base, _start_dt=None, _start_soc=None, _start_ev_soc=None)
            caller_start = dt.datetime(2026, 1, 16, 10, 15, tzinfo=dt.UTC)
            if data["positional"]:
                overridden = observe(base, caller_start, 12.0, 0.0)
            else:
                overridden = observe(base, _start_dt=caller_start, _start_soc=12.0, _start_ev_soc=0.0)
        print(json.dumps({"snapshot": snapshot, "original": original, "replayed": replayed,
                          "partial": partial, "cleared": cleared, "overridden": overridden}))
    """, {"out_dir": str(tmp_path), "positional": positional})
    snapshot = output["snapshot"]
    assert snapshot["meta"]["schema_version"] == 2
    assert snapshot["config"]["config_version"] == 3  # Not the snapshot schema.
    assert snapshot["meta"]["start_utc"] == "2026-10-25T01:30:00+00:00"
    assert snapshot["meta"]["time_zone"] == "Europe/Amsterdam"
    assert snapshot["meta"]["start_args"] == {
        "_start_dt": {"isoformat": "2026-10-25T02:30:00", "fold": 1},
        "_start_soc": 0.0, "_start_ev_soc": 67.0,
    }
    for key in ("start_ts", "soc", "ev_soc", "local", "fold"):
        assert output["replayed"][key] == output["original"][key]
    assert output["partial"]["start_ts"] == output["original"]["start_ts"]
    assert (output["partial"]["soc"], output["partial"]["ev_soc"]) == (31.0, 67.0)
    assert output["cleared"]["start_ts"] == output["original"]["start_ts"]
    assert (output["cleared"]["soc"], output["cleared"]["ev_soc"]) == (None, None)
    assert output["overridden"]["local"] == "2026-01-16T11:15:00"
    assert (output["overridden"]["soc"], output["overridden"]["ev_soc"]) == (12.0, 0.0)


@pytest.mark.parametrize("process_zone", ["UTC", "America/New_York"])
@pytest.mark.parametrize("explicit_start", [False, True])
def test_replay_preserves_both_dst_instants_across_process_timezones(tmp_path, process_zone, explicit_start):
    # UTC identity must come from the start actually used by calc_optimum,
    # not its wrapper's earlier clock sample. The two local 02:30s differ 1h.
    output = _isolated_replay("""
        process_zone = os.environ["TZ"]
        results = []
        for fold, utc in enumerate(("2026-10-25T00:30:00+00:00", "2026-10-25T01:30:00+00:00")):
            os.environ["TZ"] = "Europe/Amsterdam"
            time.tzset()
            real_datetime = dt.datetime
            start = real_datetime.fromisoformat(utc).astimezone(ZoneInfo("Europe/Amsterdam"))

            class CaptureClock(real_datetime):
                calls = 0

                @classmethod
                def now(cls, tz=None):
                    cls.calls += 1
                    # RecordingIO construction + wrapper happen before the
                    # producer's own start selection. Deliberately drift them.
                    stamp = start.timestamp() - (60 if cls.calls <= 2 else 0)
                    return real_datetime.fromtimestamp(stamp, tz)

            base = calculator()
            with patch.object(dt, "datetime", CaptureClock):
                with da_debug.RecordingIO(data["out_dir"], label=str(fold), debug=True) as recording:
                    recording._primary = base
                    original = observe(base, _start_dt=start) if data["explicit_start"] else observe(base)
            snapshot = json.loads(recording.snapshot_path.read_text())
            os.environ["TZ"] = process_zone
            time.tzset()
            with da_debug.ReplayIO(snapshot) as replay:
                assert os.environ["TZ"] == "Europe/Amsterdam"
                replayed = observe(base)
                identity = replay.time_identity
            assert os.environ["TZ"] == process_zone
            results.append({"utc": snapshot["meta"]["start_utc"], "original": original,
                            "replayed": replayed, "identity": identity})
        print(json.dumps(results))
    """, {"out_dir": str(tmp_path), "explicit_start": explicit_start}, process_zone=process_zone)
    assert [row["utc"] for row in output] == ["2026-10-25T00:30:00+00:00", "2026-10-25T01:30:00+00:00"]
    assert output[1]["replayed"]["start_ts"] - output[0]["replayed"]["start_ts"] == 3600
    for fold, row in enumerate(output):
        assert row["identity"]["complete"] is True
        assert row["identity"]["issues"] == []
        assert row["replayed"]["start_ts"] == row["original"]["start_ts"]
        assert row["replayed"]["local"] == "2026-10-25T02:30:00"
        assert row["replayed"]["fold"] == fold
        assert abs(row["replayed"]["now_ts"] - row["replayed"]["start_ts"]) < 2
        assert abs(row["replayed"]["utc_now_ts"] - row["replayed"]["now_ts"]) < 1
        assert row["replayed"]["today"] == "2026-10-25"
        assert row["replayed"]["localtime"] == "2026-10-25T02:30"


@pytest.mark.parametrize("index_kind", ["datetime", "range", "empty_range", "integer"])
def test_frame_timezone_and_index_roundtrip(index_kind):
    # Restoring UTC instead of the column/index zone, inferring object dates,
    # losing ns precision, or replacing RangeIndex with integers is data drift.
    times = pd.to_datetime(["2026-10-25T00:30:00.123456789Z", "2026-10-25T01:30:00.987654321Z"], utc=True)
    frame = pd.DataFrame({
        "tijd": times.tz_convert("Europe/Amsterdam"),
        "other_zone": times.tz_convert("America/New_York"),
        "naive": times.tz_localize(None),
        "time": ["2026-01-01", "2026-01-02"],
        "value": pd.Series([1.0, 2.5], dtype="float32"),
        "count": pd.Series([1, None], dtype="Int64"),
        "flag": pd.Series([True, None], dtype="boolean"),
    })
    if index_kind == "datetime":
        frame.index = pd.DatetimeIndex(times.tz_convert("Europe/Amsterdam"), name="interval")
    elif index_kind in ("range", "empty_range"):
        frame.index = pd.RangeIndex(7, 11, 2, name="row")
        if index_kind == "empty_range":
            frame = frame.iloc[:0]
    else:
        frame.index = pd.Index([7, 11], name="row")
    payload = json.loads(json.dumps(da_debug._dataframe_to_payload(frame)))
    restored = da_debug._dataframe_from_payload(payload)
    pd.testing.assert_frame_equal(restored, frame)
    assert type(restored.index) is type(frame.index)


@pytest.mark.parametrize("captured_at", [None, "2026-10-25T02:30:00", "2026-01-16T12:00:00"])
def test_legacy_snapshot_reports_missing_identity(snapshot, captured_at):
    # Legacy data stays readable, but even a successful solve must not imply
    # an exact time identity (in particular the ambiguous naive autumn hour).
    snapshot["meta"]["captured_at"] = captured_at
    snapshot["ha_states"] = {ENTITY: "74.25"}
    output = _isolated_replay("""
        with da_debug.ReplayIO(data) as replay:
            assert calculator().get_float("sensor.synthetic_soc", 0.0) == 74.25
            model = mip.Model(solver_name=mip.CBC)
            model.verbose = 0
            model.objective = model.add_var(lb=1, ub=1)
            assert model.optimize() == mip.OptimizationStatus.OPTIMAL
        print(json.dumps({"identity": replay.time_identity, "result": replay.build_result()}))
    """, snapshot)
    assert output["identity"]["complete"] is False
    assert {"missing_start_utc", "missing_time_zone", "missing_start_args"} <= set(output["identity"]["issues"])
    assert output["result"]["meta"]["time_identity"] == output["identity"]


@pytest.mark.parametrize("process_zone", ["UTC", None])
@pytest.mark.parametrize("failure_stage", ["clock_patch", "enter", "body", "stop"])
def test_v2_replay_restores_timezone_and_clock_after_failure(snapshot, process_zone, failure_stage):
    # Cleanup must cover TZ and the DST-correct clock methods as well as B1a's
    # existing patches, including a partial enter and a failing clock stop.
    snapshot["meta"].update(schema_version=2, start_utc="2026-10-25T01:30:00+00:00",
                            time_zone="Europe/Amsterdam",
                            start_args={"_start_dt": None, "_start_soc": None, "_start_ev_soc": None})
    output = _isolated_replay("""
        snapshot, stage = data["snapshot"], data["stage"]
        original_tz = os.environ.get("TZ")
        original_tzname = time.tzname
        original_datetime = dt.datetime
        original_methods = dict(vars(freezegun.api.FakeDatetime))
        original_today = vars(freezegun.api.FakeDate)["today"]
        original_localtime = time.localtime
        original_get_state = da_base.DaBase.get_state
        clock_count = len(freezegun.api.freeze_factories)
        replay = da_debug.ReplayIO(snapshot)

        def fail_after_start(clock):
            original_start(clock)
            raise da_debug.SnapshotMiss("controlled enter failure")

        original_set = da_debug._PatchList.set

        def fail_during_clock_patch(patches, target, name, value):
            original_set(patches, target, name, value)
            if stage == "clock_patch" and name == "today":
                raise da_debug.SnapshotMiss("controlled clock patch failure")

        original_start = freezegun.api._freeze_time.start
        try:
            with patch.object(freezegun.api._freeze_time, "start",
                              fail_after_start if stage == "enter" else original_start), \
                 patch.object(da_debug._PatchList, "set", fail_during_clock_patch):
                with replay:
                    assert os.environ["TZ"] == "Europe/Amsterdam"
                    if stage == "body":
                        raise da_debug.SnapshotMiss("controlled body failure")
                    original_stop = replay._freeze.stop

                    def fail_after_stop():
                        original_stop()
                        raise RuntimeError("controlled stop failure")

                    replay._freeze.stop = fail_after_stop
        except (da_debug.SnapshotMiss, RuntimeError) as exc:
            assert "controlled" in str(exc)
        else:
            raise AssertionError("injected failure was not raised")
        assert os.environ.get("TZ") == original_tz
        assert time.tzname == original_tzname
        assert dt.datetime is original_datetime
        assert dict(vars(freezegun.api.FakeDatetime)) == original_methods
        assert vars(freezegun.api.FakeDate)["today"] is original_today
        assert time.localtime is original_localtime
        assert da_base.DaBase.get_state is original_get_state
        assert len(freezegun.api.freeze_factories) == clock_count
        assert replay._freeze is None
        assert replay._patches._entries == []
        print(json.dumps({"restored": True}))
    """, {"snapshot": snapshot, "stage": failure_stage}, process_zone=process_zone)
    assert output == {"restored": True}


@pytest.mark.parametrize("utc,local,fold", [
    ("2026-10-25T00:30:00+00:00", "2026-10-25T02:30:00+02:00", 0),
    ("2026-10-25T01:30:00+00:00", "2026-10-25T02:30:00+01:00", 1),
    ("2026-07-01T22:30:00+00:00", "2026-07-02T00:30:00+02:00", 0),
])
def test_replay_clock_conversions_preserve_fold_and_local_date(snapshot, utc, local, fold):
    # Parsing then converting an aware datetime must not silently reset fold;
    # local date/time must also agree when the UTC calendar day differs.
    snapshot["meta"].update(schema_version=2, start_utc=utc, time_zone="Europe/Amsterdam",
                            start_args={"_start_dt": None, "_start_soc": None, "_start_ev_soc": None})
    output = _isolated_replay("""
        with da_debug.ReplayIO(data) as replay:
            parsed = dt.datetime.fromisoformat(data["meta"]["start_utc"])
            converted = parsed.astimezone(ZoneInfo("Europe/Amsterdam"))
            local = dt.datetime.fromtimestamp(parsed.timestamp())
            print(json.dumps({"converted": converted.isoformat(), "fold": converted.fold,
                              "converted_ts": converted.timestamp(), "parsed_ts": parsed.timestamp(),
                              "local_ts": local.timestamp(), "local_fold": local.fold,
                              "today": dt.date.today().isoformat(),
                              "localtime": time.strftime("%Y-%m-%dT%H:%M", time.localtime())}))
    """, snapshot)
    assert output["converted"] == local
    assert output["fold"] == output["local_fold"] == fold
    assert output["converted_ts"] == output["local_ts"] == output["parsed_ts"]
    assert output["today"] == local[:10]
    assert output["localtime"] == local[:16]


def test_missing_clock_dependency_reports_incomplete_time_identity(snapshot):
    # A valid snapshot alone is insufficient when the replay clock cannot be
    # installed; do not report complete time evidence while using live now().
    snapshot["meta"].update(schema_version=2, start_utc="2026-10-25T01:30:00+00:00",
                            time_zone="Europe/Amsterdam",
                            start_args={"_start_dt": None, "_start_soc": None, "_start_ev_soc": None})
    output = _isolated_replay("""
        import builtins

        original_import = builtins.__import__

        def without_clock(name, *args, **kwargs):
            if name == "freezegun":
                raise ImportError("synthetic unavailable clock dependency")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", without_clock):
            with da_debug.ReplayIO(data) as replay:
                assert replay._freeze is None
        print(json.dumps(replay.time_identity))
    """, snapshot)
    assert output["complete"] is False
    assert "clock_unavailable" in output["issues"]


@pytest.mark.parametrize("field,value,issue", [
    ("start_utc", None, "missing_start_utc"),
    ("start_utc", "2026-10-25T02:30:00", "invalid_start_utc"),
    ("time_zone", None, "missing_time_zone"),
    ("time_zone", "Not/A_Zone", "invalid_time_zone"),
    ("start_args", None, "missing_start_args"),
    ("start_args", {"_start_dt": None}, "missing_start_args"),
])
def test_incomplete_v2_snapshot_reports_missing_identity(snapshot, field, value, issue):
    snapshot["meta"].update(schema_version=2, start_utc="2026-10-25T01:30:00+00:00",
                            time_zone="Europe/Amsterdam",
                            start_args={"_start_dt": None, "_start_soc": None, "_start_ev_soc": None})
    snapshot["meta"][field] = value
    replay = da_debug.ReplayIO(snapshot)
    assert replay.time_identity["complete"] is False
    assert issue in replay.time_identity["issues"]


def test_legacy_frame_payload_remains_readable():
    payload = {
        "data": '{"columns":["time","value"],"index":[0,1],'
                '"data":[["2026-01-16T12:00:00.000",1.5],["2026-01-16T13:00:00.000",2.5]]}',
        "dtypes": {"time": "datetime64[ns]", "value": "float64"},
        "index_name": None, "index_is_datetime": False,
    }
    restored = da_debug._dataframe_from_payload(payload)
    expected = pd.DataFrame({"time": pd.to_datetime(["2026-01-16T12:00:00", "2026-01-16T13:00:00"]),
                             "value": [1.5, 2.5]})
    expected["time"] = expected["time"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(restored, expected)
