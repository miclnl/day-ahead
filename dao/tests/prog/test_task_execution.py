"""OS exclusion at real scheduler/CLI boundaries; only our sandbox writes.

Removing the pre-spawn lease, using LOCK_UN in the parent, closing on exec,
expiring a heartbeat, or bypassing the direct CLI guard must break these tests.
Cleanup requests self-exit through files: it never signals a saved pid/pgid.
"""

import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dao.prog import task_execution, task_state

ROOT = Path(__file__).resolve().parents[3]

# Execute the real CLI dispatcher with only its external application replaced.
WRITER = '''
import json, os, socket, sys, time
from pathlib import Path
sys.path.insert(0, REPO)
sys.path.extend([REPO + '/dao', REPO + '/dao/prog', REPO + '/dao/lib'])
from dao.prog import day_ahead
root, mode, key, name = Path(sys.argv[1]), *sys.argv[2:]
def forbidden(*args, **kwargs):
    raise AssertionError("sandbox attempted network I/O")
socket.socket.connect = socket.socket.connect_ex = forbidden
socket.create_connection = socket.getaddrinfo = forbidden
def write_until_stopped():
    (root / (name + '.ready')).write_text(str(os.getpid()))
    with (root / (name + '.writes')).open('a', buffering=1) as stream:
        while not (root / 'stop').exists():
            stream.write('x')
            stream.flush()
            time.sleep(0.01)
    (root / (name + '.exited')).touch()
if mode == 'leaf':
    write_until_stopped()
    sys.exit(0)
class App:
    config = object()
    debug = False
    db_da = type('Db', (), {'log_pool_status': lambda self: None})()
    def run_task_function(self, task):
        (root / (name + '.dispatch')).write_text(task)
        inherited = os.environ.get('DAO_TASK_LEASE_FD')
        (root / (name + '.descriptor')).write_text(inherited or 'missing')
        if mode == 'exec':
            os.execv(sys.executable, [sys.executable, __file__, str(root), 'leaf', key, name])
        if mode == 'descendant':
            if os.fork() == 0:
                os.execv(sys.executable, [sys.executable, __file__, str(root), 'leaf', key, name])
            return 0
        if mode == 'once':
            return 0
        write_until_stopped()
        return 0
day_ahead.DaCalc = lambda path: App()
sys.exit(day_ahead.main([key]))
'''

SCHEDULER = '''
import fcntl, json, os, socket, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, REPO)
sys.path.extend([REPO + '/dao', REPO + '/dao/prog', REPO + '/dao/lib'])
from dao.prog import da_scheduler, task_state
root, mode, writer_mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def forbidden(*args, **kwargs):
    raise AssertionError("sandbox attempted network I/O")
socket.socket.connect = socket.socket.connect_ex = forbidden
socket.create_connection = socket.getaddrinfo = forbidden
obj = da_scheduler.DaScheduler.__new__(da_scheduler.DaScheduler)
obj.tasks = da_scheduler.DaScheduler.generate_tasks()
obj.PROG_DIR = str(root / 'prog')
real_popen = subprocess.Popen
def spawn(cmd, **kwargs):
    with (root / 'data/task_lease/calc_optimum.lock').open('a') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            busy = False
        except BlockingIOError:
            busy = True
    (root / 'handoff.json').write_text(json.dumps({
        'busy_before_spawn': busy, 'fds': kwargs.get('pass_fds', []),
        'env_fd': kwargs.get('env', {}).get('DAO_TASK_LEASE_FD')}))
    proc = real_popen([sys.executable, str(root / 'writer.py'), str(root), writer_mode,
                      'calc_optimum', 'writer'], **kwargs)
    (root / 'spawned').write_text(str(proc.pid))
    return proc
da_scheduler.Popen = spawn
da_scheduler.HEARTBEAT_S = 0.01
def wait_ready():
    deadline = time.monotonic() + 15
    while not (root / 'writer.ready').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('writer readiness timed out')
        time.sleep(0.01)
if mode == 'crash':
    def crash(*args, **kwargs):
        wait_ready()
        handoff = __import__('json').loads((root / 'handoff.json').read_text())
        if handoff['env_fd']:
            try:
                os.fstat(int(handoff['env_fd']))
            except OSError:
                (root / 'parent-closed').touch()
        os._exit(73)  # Only this sandbox scheduler dies; no pid signal.
    task_state.heartbeat = crash
if mode == 'registration-error':
    def fail_registration(proc):
        wait_ready()
        raise RuntimeError('synthetic post-spawn registration failure')
    da_scheduler.task_registry.process_group_id = fail_registration
obj._run_exclusive('calc_optimum')
'''


def wait_for(predicate, description, timeout=20):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out: {description}")
        time.sleep(0.01)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "prog").mkdir()
    (tmp_path / "data" / "task_lease").mkdir(parents=True)
    lease_file(tmp_path).touch()
    monkeypatch.chdir(tmp_path / "prog")
    # Use precisely the same relative paths in this process and its children.
    monkeypatch.setattr(task_state, "STATE_PATH", "../data/task_state.json")
    monkeypatch.setattr(task_state, "LOCK_PATH", "../data/task_state.lock")
    monkeypatch.setattr(task_execution, "LEASE_DIR", "../data/task_lease")
    for name, source in (("writer", WRITER), ("scheduler", SCHEDULER)):
        (tmp_path / f"{name}.py").write_text(f"REPO = {str(ROOT)!r}\n" + source)
    processes = []
    outputs = []

    def start(script, *args):
        output = (tmp_path / f"process-{len(processes)}.log").open("w")
        outputs.append(output)
        proc = subprocess.Popen(
            [sys.executable, str(tmp_path / f"{script}.py"), str(tmp_path), *args],
            cwd=tmp_path / "prog", stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(proc)
        return proc

    yield tmp_path, start
    (tmp_path / "stop").touch()
    try:
        for proc in processes:
            proc.wait(timeout=20)
        # Also confirm self-exit of forked/orphaned writers before cleanup.
        for ready in tmp_path.glob("*.ready"):
            wait_for(ready.with_suffix(".exited").exists, "sandbox writer self-exit")
    finally:
        for output in outputs:
            output.close()


def lease_file(root):
    return root / "data" / "task_lease" / "calc_optimum.lock"


def slot_is_free(root):
    # Independent OS observation, not the product's resource/probe helper.
    with lease_file(root).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True  # close, never explicit LOCK_UN


def age_claim():
    path = Path(task_state.STATE_PATH)
    state = json.loads(path.read_text())
    state["running"]["calc_optimum"]["heartbeat"] = time.time() - 3600
    path.write_text(json.dumps(state))


def assert_writing(root):
    path = root / "writer.writes"
    wait_for(lambda: path.exists() and path.stat().st_size > 0, "writer I/O")
    before = path.stat().st_size
    wait_for(lambda: path.stat().st_size > before, "continued writer I/O")


def test_scheduler_death_does_not_allow_second_writer(sandbox):
    root, start = sandbox
    scheduler = start("scheduler", "crash", "hold")
    assert scheduler.wait(timeout=20) == 73
    assert_writing(root)
    age_claim()
    assert not slot_is_free(root), "scheduler death released the child's OS lease"
    assert task_state.running_tasks()["calc_optimum"]["state"] == "unconfirmed"
    assert task_state.claim("calc_optimum", "replacement") is None
    second = start("writer", "once", "calc_optimum_met_debug", "second")
    assert second.wait(timeout=20) == 1
    assert not (root / "second.dispatch").exists()
    assert_writing(root)


def test_spawn_registration_failure_keeps_exclusion(sandbox):
    root, start = sandbox
    scheduler = start("scheduler", "registration-error", "hold")
    assert scheduler.wait(timeout=20) == 0  # scheduler contains task errors
    assert_writing(root)
    assert task_state.running_tasks().get("calc_optimum", {}).get("state") == "unconfirmed"
    assert not slot_is_free(root)
    assert task_state.last_result("calc_optimum") == {}
    assert task_state.claim("calc_optimum", "replacement") is None


def test_lease_survives_parent_close_and_exec(sandbox):
    root, start = sandbox
    scheduler = start("scheduler", "crash", "exec")
    assert scheduler.wait(timeout=20) == 73
    assert_writing(root)  # I/O after an additional exec, after parent death.
    assert not slot_is_free(root), "lease closed/unlocked across parent close or exec"


def test_scheduler_takes_slot_before_spawn_and_child_reuses_descriptor(sandbox):
    root, start = sandbox
    scheduler = start("scheduler", "crash", "hold")
    assert scheduler.wait(timeout=20) == 73
    handoff = json.loads((root / "handoff.json").read_text())
    assert handoff["busy_before_spawn"] is True
    assert handoff["env_fd"] is not None
    assert handoff["fds"] == [int(handoff["env_fd"])]
    assert (root / "writer.descriptor").read_text() == handoff["env_fd"]
    assert (root / "parent-closed").exists()
    assert not slot_is_free(root)


def test_direct_cli_cannot_bypass_task_lease(sandbox):
    root, start = sandbox
    with lease_file(root).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        direct = start("writer", "once", "optimize_debug", "second")
        assert direct.wait(timeout=20) == 1
        assert not (root / "second.dispatch").exists(), "direct CLI reached task I/O"
    direct = start("writer", "once", "optimize_debug", "third")
    assert direct.wait(timeout=20) == 0
    assert (root / "third.dispatch").read_text() == "calc_optimum_met_debug"


def test_recovery_requires_released_writer_lease(sandbox):
    root, start = sandbox
    scheduler = start("scheduler", "crash", "hold")
    assert scheduler.wait(timeout=20) == 73
    age_claim()
    assert_writing(root)
    inode = lease_file(root).stat().st_ino
    old = task_state.running_tasks().get("calc_optimum", {})
    assert old.get("state") == "unconfirmed", "stale heartbeat silently erased the claim"
    assert task_state.claim("calc_optimum", "replacement") is None
    (root / "stop").touch()
    wait_for(lambda: slot_is_free(root), "last writer descriptor closed")
    replacement = task_state.claim("calc_optimum", "replacement")
    assert replacement is not None and replacement.run_id != old["run_id"]
    assert lease_file(root).stat().st_ino == inode, "stable sidecar was replaced"


def test_writing_descendant_retains_lease_after_leader_and_scheduler_exit(sandbox):
    root, start = sandbox
    scheduler = start("scheduler", "normal", "descendant")
    assert scheduler.wait(timeout=20) == 0
    assert_writing(root)
    assert not slot_is_free(root), "writing descendant lost the leader's lease"
    second = start("writer", "once", "calc_optimum", "second")
    assert second.wait(timeout=20) == 1
    assert not (root / "second.dispatch").exists()
    assert task_state.claim("calc_optimum", "replacement") is None


def test_legacy_unleased_claim_requires_quiescent_upgrade(sandbox):
    root, start = sandbox
    task_state.claim("calc_optimum", "old-scheduler")
    path = Path(task_state.STATE_PATH)
    state = json.loads(path.read_text())
    state["running"]["calc_optimum"].pop("lease_protocol", None)
    path.write_text(json.dumps(state))
    age_claim()
    assert slot_is_free(root)
    assert task_state.claim("calc_optimum", "replacement") is None
    direct = start("writer", "once", "calc_optimum", "second")
    assert direct.wait(timeout=20) == 1
    assert not (root / "second.dispatch").exists()


@pytest.mark.parametrize("operation", ["release", "cancel", "expire", "pickup"])
def test_legacy_pending_claim_cannot_be_migrated_by_bookkeeping(sandbox, operation):
    _, _start = sandbox
    claim = task_state.request("calc_optimum", "old-scheduler")
    path = Path(task_state.STATE_PATH)
    state = json.loads(path.read_text())
    state["running"]["calc_optimum"].pop("lease_protocol")
    path.write_text(json.dumps(state))
    if operation == "release":
        task_state.release(claim, "error")
    elif operation == "cancel":
        task_state.request_cancel(claim)
    elif operation == "expire":
        age_claim()
        task_state.expire_overdue_pending()
    else:
        assert task_state.take_pending(claim) is None
    assert task_state.claim("calc_optimum", "new-scheduler") is None
    assert task_state.running_tasks()["calc_optimum"]["run_id"] == claim.run_id


def test_pending_expiry_cannot_discard_claim_while_resource_has_a_writer(sandbox):
    root, _start = sandbox
    claim = task_state.request("calc_optimum", "dashboard")
    age_claim()
    with lease_file(root).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert task_state.expire_overdue_pending() == []
        entry = task_state.running_tasks()["calc_optimum"]
        assert entry["state"] == "unconfirmed" and entry["run_id"] == claim.run_id
        assert task_state.claim("calc_optimum", "new-scheduler") is None


def test_registered_command_runner_retains_lease_through_child_and_log_io(sandbox, monkeypatch):
    from dao.prog import da_base

    root, _start = sandbox
    obj = da_base.DaBase.__new__(da_base.DaBase)
    obj.tasks = da_base.DaBase.generate_tasks()
    (root / "data" / "log").mkdir()
    real_run = subprocess.run
    observed = []

    def sandbox_run(cmd, **kwargs):
        # These are real OS observations, not just assertions on mock kwargs.
        observed.append(not slot_is_free(root))
        kwargs["start_new_session"] = True
        return real_run(
            [sys.executable, str(root / "writer.py"), str(root), "once", "calc_optimum", "cmd"],
            **kwargs,
        )

    monkeypatch.setattr(da_base, "run", sandbox_run)
    obj.run_task_cmd("calc_optimum")
    assert observed == [True]
    assert (root / "cmd.dispatch").read_text() == "calc_optimum"
    assert (root / "cmd.descriptor").read_text() != "missing"
    assert slot_is_free(root)


def test_superseded_runner_cannot_spawn_after_eventual_lease_acquire(sandbox, monkeypatch):
    from dao.prog import da_scheduler

    _, _start = sandbox
    old = task_state.claim("calc_optimum", "scheduler")
    age_claim()
    replacement = task_state.claim("calc_optimum", "replacement")
    before = Path(task_state.STATE_PATH).read_bytes()
    obj = da_scheduler.DaScheduler.__new__(da_scheduler.DaScheduler)
    obj.tasks = da_scheduler.DaScheduler.generate_tasks()

    def forbidden(*args, **kwargs):
        pytest.fail("superseded claim reached spawn")

    monkeypatch.setattr(da_scheduler, "Popen", forbidden)
    obj._run_claimed(old)
    assert replacement is not None and replacement != old
    assert Path(task_state.STATE_PATH).read_bytes() == before


@pytest.mark.parametrize("entrypoint,key,args,handler", [
    ("da_fast", "fast_once", ["once"], "cmd_once"),
    ("da_fast", "fast_control_simulate", ["simulate"], "cmd_simulate"),
    ("da_settlement", "settlement_consolidate", ["reconcile", "--publish", "--reason", "scheduled", "--scheduled"], "_execute"),
])
def test_other_registered_cli_entrypoints_cannot_bypass_lease(sandbox, monkeypatch, entrypoint, key, args, handler):
    from importlib import import_module

    module = import_module("dao.prog." + entrypoint)

    def forbidden(*args):
        pytest.fail("blocked CLI reached application/task I/O")

    monkeypatch.setattr(module, handler, forbidden)
    if entrypoint == "da_fast":
        monkeypatch.setattr(module, "_configure_logging", lambda level: None)
    lease = task_execution.acquire(key)
    try:
        assert module.main(args) == 1
    finally:
        lease.close()


@pytest.fixture
def publishing_cli(sandbox, monkeypatch):
    """Keep real DaCalc/DaBase constructors and set_last_activity publication.

    Only external config/DB/HA/provider dependencies are synthetic. Unlike the
    inert process fixture, construction here really reaches the publishing
    bootstrap path; the unlocked positive control must prove that fact.
    """
    from dao.prog import da_base, day_ahead

    events = []
    statuses = {}

    def forbidden(*args, **kwargs):
        pytest.fail("publishing CLI attempted live I/O")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    config = SimpleNamespace(
        logging_level="info", time_zone="UTC", interval="1hour",
        homeassistant=SimpleNamespace(
            protocol_api="http", ip_address="synthetic.invalid", ip_port=8123,
            hasstoken=SimpleNamespace(resolve=lambda secrets: "synthetic"),
        ),
        solar=[], prices=None, history=None, boiler=None, battery=[],
        electric_vehicle=[], heating=None, use_calc_baseload=False, machines=[],
        strategy=SimpleNamespace(resolve=lambda getter, default: default), tibber=None,
        notifications=SimpleNamespace(
            notification_entity=None, opstarten=False, berekening=False,
            last_activity_entity="input_datetime.synthetic_last_activity",
        ),
        graphics=None, grid=SimpleNamespace(max_power=17.0, model_fields_set={"max_power"}),
    )
    db = SimpleNamespace(log_pool_status=lambda: None)
    monkeypatch.setattr(da_base.DaBase, "_config", config)
    monkeypatch.setattr(da_base.DaBase, "_loader", SimpleNamespace(secrets={}))
    monkeypatch.setattr(da_base.DaBase, "_configure_root_logging", lambda self: False)
    monkeypatch.setattr(da_base, "make_db_da", lambda *args: db)
    monkeypatch.setattr(da_base, "make_db_ha", lambda *args: db)
    monkeypatch.setattr(da_base, "HAClient", lambda **kwargs: SimpleNamespace(
        get_config=lambda: {"latitude": 1.0, "longitude": 2.0, "time_zone": "UTC", "country": "US"},
    ))
    monkeypatch.setattr(da_base, "Meteo", lambda *args, **kwargs: object())
    monkeypatch.setattr(da_base, "DaPrices", lambda *args, **kwargs: object())

    def descriptor():
        fd = os.environ.get(task_execution.ENV_FD)
        return int(fd) if fd is not None else None

    class PublishingCalc(day_ahead.DaCalc):
        def __init__(self, file_name):
            events.append(("constructor", descriptor()))
            super().__init__(file_name)

        def call_service(self, service, **kwargs):
            events.append(("publication", descriptor(), service, kwargs))

        def run_task_function(self, task):
            events.append(("dispatch", descriptor(), task, self.debug))
            return statuses.get(task, 0)

    monkeypatch.setattr(day_ahead, "DaCalc", PublishingCalc)
    return SimpleNamespace(main=day_ahead.main, events=events, statuses=statuses, calc_type=PublishingCalc)


@pytest.mark.parametrize("argv", [
    ["calc"], ["debug", "calc"], ["2026-06-15", "debug", "calc"], ["optimize_debug"],
])
def test_direct_cli_busy_resource_never_constructs_or_publishes_bootstrap(sandbox, publishing_cli, argv):
    root, _start = sandbox
    with lease_file(root).open("a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert publishing_cli.main(argv) == 1
        assert publishing_cli.events == [], "busy CLI constructed/published before acquiring its lease"

    # Positive control: run the actual constructor chain and its publication.
    assert publishing_cli.main(argv) == 0
    constructor, publication, dispatch = publishing_cli.events
    assert constructor[0] == "constructor"
    assert publication[0] == "publication" and publication[2] == "set_datetime"
    assert publication[3]["entity_id"] == "input_datetime.synthetic_last_activity"
    assert dispatch[0] == "dispatch"


def test_direct_cli_bootstrap_and_first_task_keep_one_lease(sandbox, publishing_cli, monkeypatch):
    acquired = []
    real_acquire = task_execution.acquire

    def observed_acquire(key):
        lease = real_acquire(key)
        acquired.append(lease.fd)
        return lease

    monkeypatch.setattr(task_execution, "acquire", observed_acquire)
    assert publishing_cli.main(["debug", "calc"]) == 0
    constructor, publication, dispatch = publishing_cli.events
    assert acquired == [constructor[1]]
    assert constructor[1] == publication[1] == dispatch[1]
    assert dispatch[2:] == ("calc_optimum_met_debug", True)
    with pytest.raises(OSError):
        os.fstat(acquired[0])


def test_execution_can_hold_two_resources_and_restore_first(sandbox):
    root, _start = sandbox
    with task_execution.execution("calc_optimum") as first:
        with task_execution.execution("meteo") as second:
            assert first.resource == "calc_optimum" and second.resource == "meteo"
            assert first.fd != second.fd
            assert not slot_is_free(root)
        assert os.environ[task_execution.ENV_FD] == str(first.fd)
        assert task_execution._active.get() is first
        assert not slot_is_free(root)
    assert slot_is_free(root)


@pytest.mark.parametrize("first_status,next_task,want_status,want_tasks", [
    (0, "meteo", 0, ["prices", "meteo"]),
    (7, "meteo", 7, ["prices"]),
    (1, "meteo", 1, ["prices"]),
    (0, "unknown-task", 1, ["prices"]),
])
def test_bootstrap_guard_preserves_multitask_first_error_contract(
    sandbox, publishing_cli, first_status, next_task, want_status, want_tasks,
):
    publishing_cli.statuses["prices"] = first_status
    assert publishing_cli.main(["prices", "2026-06-15", "2026-06-16", next_task]) == want_status
    assert [event[2] for event in publishing_cli.events if event[0] == "dispatch"] == want_tasks


def test_multitask_keeps_bootstrap_resource_while_second_task_uses_its_own(
    sandbox, publishing_cli, monkeypatch,
):
    observed = []
    original = publishing_cli.calc_type.run_task_function

    def busy(resource):
        with (Path(task_execution.LEASE_DIR) / f"{resource}.lock").open("a") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False

    def dispatch(self, task):
        observed.append((task, busy("prices"), busy("meteo")))
        return original(self, task)

    monkeypatch.setattr(publishing_cli.calc_type, "run_task_function", dispatch)
    assert publishing_cli.main(["prices", "meteo"]) == 0
    assert observed == [("prices", True, False), ("meteo", True, True)]
    assert not busy("prices") and not busy("meteo")


def test_multitask_guard_preserves_ordered_debug_toggles(sandbox, publishing_cli):
    assert publishing_cli.main(["prices", "debug", "calc", "debug", "meteo"]) == 0
    assert [event[2:] for event in publishing_cli.events if event[0] == "dispatch"] == [
        ("prices", False), ("calc_optimum_met_debug", True), ("meteo", False),
    ]


def test_foreign_inherited_resource_does_not_prevent_second_resource(sandbox, monkeypatch):
    first = task_execution.acquire("prices")
    try:
        monkeypatch.setenv(task_execution.ENV_FD, str(first.fd))
        monkeypatch.setenv(task_execution.ENV_RESOURCE, "prices")
        with task_execution.execution("meteo") as second:
            assert second.resource == "meteo" and second.fd != first.fd
        assert os.environ[task_execution.ENV_FD] == str(first.fd)
        assert os.fstat(first.fd).st_ino == task_execution.lease_path("prices").stat().st_ino
    finally:
        first.close()


@pytest.mark.parametrize("kind", ["wrong-inode", "non-regular", "invalid-fd", "missing-fd"])
def test_requested_resource_inherited_corruption_still_fails_closed(sandbox, monkeypatch, kind):
    root, _start = sandbox
    handle = None
    try:
        if kind == "wrong-inode":
            handle = (root / "wrong-inode").open("a")
            fd = handle.fileno()
        elif kind == "non-regular":
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        else:
            fd = -1
        monkeypatch.setenv(task_execution.ENV_RESOURCE, "calc_optimum")
        if kind == "missing-fd":
            monkeypatch.delenv(task_execution.ENV_FD, raising=False)
        else:
            monkeypatch.setenv(task_execution.ENV_FD, str(fd))
        with pytest.raises(task_execution.LeaseError, match="Ongeldige geërfde writerlease"):
            with task_execution.execution("calc_optimum"):
                pytest.fail("corrupt requested-resource descriptor allowed task I/O")
    finally:
        if handle is not None:
            handle.close()
        elif kind == "non-regular":
            os.close(fd)


@pytest.fixture
def virgin_data_path(tmp_path, monkeypatch):
    """App data is absent; its configured target and all I/O stay in sandbox."""
    from dao.prog import da_base

    prog = tmp_path / "prog"
    prog.mkdir()
    target = tmp_path / "configured-data"
    target.mkdir()
    (target / "options.json").write_text('{"config_version": 3}')
    (target / "secrets.json").write_text("{}")
    monkeypatch.chdir(prog)
    # The production bootstrap will expose this injectable target. Before the
    # fix this patch is deliberately inert, so RED remains a behavior failure.
    monkeypatch.setattr(da_base, "DATA_PATH_TARGET", str(target), raising=False)
    monkeypatch.setattr(task_execution, "LEASE_DIR", "../data/task_lease")
    monkeypatch.setattr(task_state, "STATE_PATH", "../data/task_state.json")
    monkeypatch.setattr(task_state, "LOCK_PATH", "../data/task_state.lock")

    def forbidden(*args, **kwargs):
        pytest.fail("virgin datapath test attempted network I/O")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    assert not os.path.lexists(tmp_path / "data")
    return SimpleNamespace(data=tmp_path / "data", target=target)


def test_lease_acquire_preserves_app_datapath_bootstrap(virgin_data_path):
    from dao.prog.da_base import DaBase

    layout = virgin_data_path
    lease = task_execution.acquire("calc_optimum")
    try:
        assert layout.data.is_symlink(), "lease mkdir replaced the app's datapath bootstrap"
        assert layout.data.resolve() == layout.target
        assert (layout.data / "options.json").read_bytes() == (layout.target / "options.json").read_bytes()
        inode = (layout.data / "task_lease" / "calc_optimum.lock").stat().st_ino
        DaBase.make_data_path()  # Later constructor bootstrap stays idempotent.
        assert layout.data.is_symlink() and layout.data.resolve() == layout.target
        assert (layout.target / "task_lease" / "calc_optimum.lock").stat().st_ino == inode
    finally:
        lease.close()


@pytest.mark.parametrize("entrypoint", ["day_ahead", "da_fast", "da_settlement"])
def test_preconstruction_entrypoints_bootstrap_configured_datapath(virgin_data_path, monkeypatch, entrypoint):
    from importlib import import_module

    from dao.prog.config.loader import ConfigurationLoader
    from dao.prog.da_base import DaBase

    module = import_module("dao.prog." + entrypoint)
    layout = virgin_data_path
    loaded = []

    def application_boundary():
        # This is reached inside the real lease guard, before any app I/O.
        assert layout.data.is_symlink(), "entrypoint's lease preempted the app data link"
        DaBase.make_data_path()
        config = ConfigurationLoader(layout.data / "options.json").load_and_validate()
        loaded.append(config.config_version)
        return 0

    if entrypoint == "day_ahead":
        class App:
            def __init__(self, file_name):
                application_boundary()
                self.config, self.debug = object(), False
                self.db_da = SimpleNamespace(log_pool_status=lambda: None)

            def run_task_function(self, key):
                return 0

        monkeypatch.setattr(module, "DaCalc", App)
        args = ["prices"]
    elif entrypoint == "da_fast":
        monkeypatch.setattr(module, "_configure_logging", lambda level: None)
        monkeypatch.setattr(module, "cmd_once", lambda args: application_boundary())
        args = ["once"]
    else:
        monkeypatch.setattr(module, "_execute", lambda args: application_boundary())
        args = ["reconcile", "--publish", "--reason", "scheduled", "--scheduled"]
    assert module.main(args) == 0
    assert loaded == [3]
    assert layout.data.is_symlink() and layout.data.resolve() == layout.target


def test_no_task_and_constructor_share_app_configuration_loader(sandbox, publishing_cli, monkeypatch):
    from dao.prog import da_base

    root, _start = sandbox
    (root / "data" / "options.json").write_text('{"config_version": 3}')
    (root / "data" / "secrets.json").write_text("{}")
    monkeypatch.setattr(da_base.DaBase, "_config", None)
    monkeypatch.setattr(da_base.DaBase, "_loader", None)
    real_load = da_base.ConfigurationLoader.load_and_validate
    loaded = []

    def load(instance):
        loaded.append(instance.config_path.resolve())
        return real_load(instance)

    monkeypatch.setattr(da_base.ConfigurationLoader, "load_and_validate", load)
    assert publishing_cli.main([]) == 0
    assert publishing_cli.events == []
    config, loader = da_base.DaBase._config, da_base.DaBase._loader
    assert config is not None, "no-task call skipped the app's configuration load"
    assert publishing_cli.main(["prices"]) == 0
    assert da_base.DaBase._config is config and da_base.DaBase._loader is loader
    assert loaded == [(root / "data" / "options.json").resolve()]
