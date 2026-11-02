"""Missing replay input must stop; only recorded HA failures may use defaults."""

import datetime as dt
import json
import socket
from types import SimpleNamespace

import freezegun.api
import mip
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
