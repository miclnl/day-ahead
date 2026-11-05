"""F2c3: real OS ownership, durable operational mirror, fail-closed lifecycle."""

import hashlib
import json
import os
import selectors
import stat
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError, fields, replace
from importlib import import_module, util
from pathlib import Path
from types import SimpleNamespace

import pytest

from dao.prog import task_execution
from dao.prog.fastctrl.command import Deadline, RevokeRequest
from dao.tests.prog.test_fast_receiver import command, fake

ROOT = Path(__file__).resolve().parents[3]
RESOURCE = "actuator-" + hashlib.sha256(b"sandbox").hexdigest()[:32]
DEADLINE = Deadline(100_000_000_000, 10000)
KEYS = {
    "schema_version", "installation_id", "receiver_id", "receiver_generation",
    "anchor_checkpoint", "owner_session_id", "owner_epoch", "last_reserved_sequence",
    "phase", "pending_command", "last_receipt", "pending_barrier_id", "last_barrier",
}

RECEIPT_IDENTITY_MISMATCHES = [
    ("installation_id", "other-installation"),
    ("battery_name", "accu2"),
    ("owner_session_id", "other-session"),
    ("owner_epoch", 999),
    ("sequence", 999),
    ("command_id", "other-command"),
    ("plan_digest", "f" * 64),
    ("payload_digest", "f" * 64),
]


def ownership():
    assert util.find_spec("dao.prog.fastctrl.ownership") is not None, "F2c3 ownership module missing"
    return import_module("dao.prog.fastctrl.ownership")


def acquire(root):
    assert hasattr(task_execution, "acquire_resource"), "F2c3 acquire_resource missing"
    return task_execution.acquire_resource(RESOURCE, lease_dir=root / "leases")


def initial(**changes):
    return ownership().OwnerState(**(dict(
        schema_version=1, installation_id="sandbox", receiver_id="receiver-a",
        receiver_generation="generation-a", anchor_checkpoint="checkpoint-40",
        owner_session_id="", owner_epoch=0, last_reserved_sequence=40,
        phase="inhibited", pending_command=None, last_receipt=None,
        pending_barrier_id=None, last_barrier=None,
    ) | changes))


def layout(root, *, provision=True):
    module = ownership()
    store = module.OwnerStore(root / "fast_plan.json", "sandbox", lease_dir=root / "leases")
    if provision:
        store.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        store.path.write_text(json.dumps(initial().to_dict()))
        store.path.chmod(0o600)
    gateway = fake(root / "receiver")
    owner = module.OwnerCoordinator(store, receiver=gateway)
    owner.now_ns = lambda: 0
    owner.now_ms = lambda: 1001
    return module, store, gateway, owner


def start(owner, lease, gateway, **changes):
    return owner.start(**(dict(writer=lease, receiver=gateway, installation_id="sandbox",
                              owner_session_id="session-new", requested_not_after_ms=6000,
                              deadline=DEADLINE, now_ms=1001) | changes))


PROCESS = '''
import fcntl, json, os, socket, subprocess, sys
from pathlib import Path
sys.path.insert(0, REPO)
from dao.prog import task_execution as t
root, mode, resource = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def forbidden(*a, **k):
    raise AssertionError('sandbox attempted network')
socket.socket.connect = socket.socket.connect_ex = forbidden
socket.getaddrinfo = socket.create_connection = forbidden
if mode == 'leaf':
    fd = int(os.environ[t.ENV_FD])
    os.fstat(fd)
    print('descendant-held', flush=True)
    sys.stdin.readline()
    os.close(fd)
elif mode == 'handoff':
    lease = t.acquire_resource(resource, lease_dir=root / 'leases')
    child = subprocess.Popen([sys.executable, __file__, str(root), 'leaf', resource],
                             **lease.child_kwargs())
    lease.close()
    print('parent-closed', flush=True)
    child.wait(timeout=20)
else:
    try:
        lease = t.acquire_resource(resource, lease_dir=root / 'leases')
    except t.LeaseBusy:
        print('busy-inhibited', flush=True)
    else:
        if mode == 'hold':
            from dao.tests.prog.test_fast_ownership import layout, start
            _, store, gateway, owner = layout(root)
            binding = start(owner, lease, gateway)
            print('owned:' + str(binding.owner_epoch), flush=True)
        else:
            print('held', flush=True)
        if mode == 'hold':
            sys.stdin.readline()
        lease.close()
'''


@pytest.fixture
def processes(tmp_path):
    script = tmp_path / "owner_process.py"
    script.write_text(f"REPO = {str(ROOT)!r}\n" + PROCESS)
    children = []

    def spawn(mode):
        proc = subprocess.Popen([sys.executable, str(script), str(tmp_path), mode, RESOURCE],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        children.append(proc)
        return proc

    yield spawn
    for proc in children:
        if proc.poll() is None:
            proc.stdin.write("stop\n")
            proc.stdin.flush()
        proc.wait(timeout=25)
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()


def test_01_two_real_processes_cannot_both_own(tmp_path, processes):
    assert hasattr(task_execution, "acquire_resource"), "F2c3 acquire_resource missing"
    first = processes("hold")
    assert first.stdout.readline().strip() == "owned:8"
    inode = (tmp_path / "leases" / (RESOURCE + ".lock")).stat().st_ino
    second = processes("once")
    assert second.stdout.readline().strip() == "busy-inhibited"
    assert second.wait(timeout=20) == 0
    assert (tmp_path / "leases" / (RESOURCE + ".lock")).stat().st_ino == inode


def test_02_resource_survives_parentclose_exec_descendant(tmp_path, processes):
    assert hasattr(task_execution, "acquire_resource"), "F2c3 acquire_resource missing"
    parent = processes("handoff")
    messages = set()
    pending = b""
    end = time.monotonic() + 5
    with selectors.DefaultSelector() as ready:
        ready.register(parent.stdout, selectors.EVENT_READ)
        while len(messages) < 2:
            events = ready.select(max(0, end - time.monotonic()))
            assert events, f"missing causal handoff acknowledgement: {messages}"
            chunk = os.read(parent.stdout.fileno(), 4096)
            assert chunk, f"handoff exited before acknowledgements: {messages}"
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                messages.add(line.decode())
    assert messages == {"parent-closed", "descendant-held"}
    second = processes("once")
    assert second.stdout.readline().strip() == "busy-inhibited"
    assert second.wait(timeout=20) == 0
    parent.stdin.write("stop\n")
    parent.stdin.flush()
    assert parent.wait(timeout=20) == 0
    lease = acquire(tmp_path)
    lease.close()


def test_03_heartbeat_pid_without_flock_is_no_right(tmp_path):
    module, store, gateway, owner = layout(tmp_path)
    (tmp_path / "heartbeat.json").write_text(json.dumps({"pid": os.getpid(), "alive": True}))
    result = start(owner, None, gateway)
    assert isinstance(result, module.RecoveryRequired)
    assert (result.reason, result.phase) == ("owner_lease_missing", "inhibited")
    assert gateway.state.epoch_high_water == 7
    lease = acquire(tmp_path)
    lease.close()
    assert start(owner, lease, gateway).reason == "owner_lease_missing"


def test_04_crash_after_reserve_skips_sequence(tmp_path):
    module, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        assert start(owner, lease, gateway).owner_epoch == 8
        assert owner.reserve_sequence(writer=lease) == 41
        restarted = module.OwnerCoordinator(store, receiver=gateway)
        restarted.now_ns = lambda: 0
        restarted.now_ms = lambda: 1001
        assert start(restarted, lease, gateway, owner_session_id="restart").owner_epoch == 9
        assert restarted.reserve_sequence(writer=lease) == 42
        assert not gateway.effect_log
    finally:
        lease.close()


@pytest.mark.parametrize("operation", ["file_fsync", "replace", "directory_fsync"])
def test_05_persistence_failure_inhibits_before_send(tmp_path, monkeypatch, operation):
    module, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        start(owner, lease, gateway)
        before = store.path.read_bytes()
        real_fsync = os.fsync

        def fail(fd):
            is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
            if is_dir == (operation == "directory_fsync"):
                raise OSError("synthetic fsync failure")
            real_fsync(fd)

        if operation == "replace":
            def fail_replace(*args, **kwargs):
                raise OSError("synthetic replace failure")
            monkeypatch.setattr(module.os, "replace", fail_replace)
        else:
            monkeypatch.setattr(module.os, "fsync", fail)
        with pytest.raises(OSError):
            owner.reserve_sequence(writer=lease)
        assert owner.phase == "inhibited"
        assert not gateway.effect_log
        if operation != "directory_fsync":
            assert store.path.read_bytes() == before
        else:
            # Replace is already visible; never roll back/reuse that reservation.
            assert store.load_strict().last_reserved_sequence == 41
    finally:
        lease.close()


def test_06_missing_mirror_only_initializes_inhibited(tmp_path):
    module, store, gateway, owner = layout(tmp_path, provision=False)
    with pytest.raises(module.RecoveryRequired, match="mirror_missing_or_corrupt"):
        store.load_strict()
    lease = acquire(tmp_path)
    try:
        result = start(owner, lease, gateway)
        assert result.reason == "mirror_missing_or_corrupt"
        state = store.load_strict()
        assert (state.phase, state.owner_epoch, state.last_reserved_sequence, state.owner_session_id) == (
            "inhibited", 0, 0, "")
        assert state.pending_command is state.pending_barrier_id is state.last_barrier is None
        assert gateway.state.epoch_high_water == 7
    finally:
        lease.close()


def test_07_restored_mirror_always_needs_new_barrier_epoch(tmp_path):
    _, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        old = store.path.read_bytes()
        assert start(owner, lease, gateway).owner_epoch == 8
        store.path.write_bytes(old)
        assert start(owner, lease, gateway).owner_epoch == 9
        assert store.load_strict().last_barrier.revoked_through_epoch == 8
        assert not gateway.effect_log
    finally:
        lease.close()


@pytest.mark.parametrize("lost_anchor", [False, True])
def test_08_unknown_queue_or_lost_anchor_inhibits(tmp_path, lost_anchor):
    _, store, gateway, owner = layout(tmp_path)
    if lost_anchor:
        gateway.anchor_path.unlink()
        # The fake does not recover missing anchors; inspect fails closed.
        def inspect(*, deadline):
            gateway.restart()
        gateway.inspect = inspect
    else:
        gateway.state = replace(gateway.state, queue_known=False)
    lease = acquire(tmp_path)
    try:
        result = start(owner, lease, gateway)
        assert result.phase == owner.phase == "inhibited"
        if not lost_anchor:
            assert result.reason == "queue_unknown"
        assert not gateway.effect_log
    finally:
        lease.close()


@pytest.mark.parametrize("raises", [False, True])
def test_09_stop_failure_retains_pending_recovery(tmp_path, raises):
    _, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        start(owner, lease, gateway)
        if raises:
            def fail(*args, **kwargs):
                raise OSError("synthetic revoke failure")
            gateway.revoke_and_stop = fail
        else:
            gateway.safe_state_confirmed = False
        receipt = owner.stop(reason="shutdown")
        assert receipt is None if raises else receipt.status == "unconfirmed"
        state = store.load_strict()
        assert state.phase == "unconfirmed" and state.pending_barrier_id
        assert owner.recovery_required
        assert lease.fd >= 0
    finally:
        lease.close()


def test_10_old_barrier_ack_cannot_release_new_owner(tmp_path):
    _, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        start(owner, lease, gateway)
        old = store.load_strict().last_barrier
        start(owner, lease, gateway, owner_session_id="newer")
        gateway.revoke_and_stop = lambda *args, **kwargs: old
        assert owner.stop(reason="shutdown") == old
        assert store.load_strict().phase == "unconfirmed"
        assert owner.recovery_required
    finally:
        lease.close()


def test_11_operator_stop_during_bind_blocks_bind_and_send(tmp_path):
    _, store, gateway, owner = layout(tmp_path)
    real_bind = gateway.bind_owner

    def racing_bind(**kwargs):
        gateway.revoke_and_stop(RevokeRequest("sandbox", kwargs["expected_epoch"], "operator",
                                             "operator_stop", 1001), deadline=DEADLINE)
        return real_bind(**kwargs)

    gateway.bind_owner = racing_bind
    lease = acquire(tmp_path)
    try:
        assert start(owner, lease, gateway).reason == "bind_refused"
        assert gateway.send(command(owner_epoch=8), deadline=DEADLINE).status == "rejected"
        assert not gateway.effect_log
    finally:
        lease.close()


def test_12_free_os_slot_without_exact_barrier_grants_nothing(tmp_path):
    _, store, gateway, owner = layout(tmp_path)
    gateway.queues_drained = False
    lease = acquire(tmp_path)
    try:
        assert start(owner, lease, gateway).reason == "barrier_unconfirmed"
        assert owner.phase == "inhibited" and owner.recovery_required
        with pytest.raises(ownership().RecoveryRequired, match="bind_refused"):
            owner.reserve_sequence(writer=lease)
        assert not gateway.effect_log
    finally:
        lease.close()


def test_13_schema_is_only_operational_and_roundtrips_pending_dtos(tmp_path):
    module, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        start(owner, lease, gateway)
        cmd = command(owner_session_id="session-new", owner_epoch=8)
        receipt = gateway.send(cmd, deadline=DEADLINE)
        assert receipt.status == "accepted" and len(receipt.helper_results) == 3
        state = replace(store.load_strict(), pending_command=cmd, last_receipt=receipt)
        assert set(state.to_dict()) == KEYS
        assert {field.name for field in fields(state)} == KEYS
        assert module.OwnerState.from_dict(state.to_dict()) == state
        with pytest.raises(FrozenInstanceError):
            state.phase = "released"
    finally:
        lease.close()


@pytest.mark.parametrize("field,value", RECEIPT_IDENTITY_MISMATCHES)
@pytest.mark.parametrize("status", ["accepted", "rejected"])
@pytest.mark.parametrize("terminal", [True, False])
def test_i1_record_receipt_rejects_foreign_terminal_result(
        tmp_path, field, value, status, terminal):
    """A foreign ACK may neither clear pending nor replace this command's history."""
    module, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        start(owner, lease, gateway)
        cmd = command(owner_session_id="session-new", owner_epoch=8,
                      sequence=owner.reserve_sequence(writer=lease))
        owner.record_pending_command(cmd)
        if status == "rejected":
            gateway.helper_ok["setpoint"] = False
        receipt = gateway.send(cmd, deadline=DEADLINE)
        assert receipt.status == status
        foreign = replace(receipt, **{field: value})
        before = store.path.read_bytes()
        before_state = owner.state

        with pytest.raises(module.RecoveryRequired, match="^receipt_identity_mismatch$"):
            owner.record_receipt(foreign, terminal=terminal)

        assert store.path.read_bytes() == before
        assert owner.state == before_state
        assert store.load_strict().pending_command == cmd
    finally:
        lease.close()


@pytest.mark.parametrize("terminal", [True, False])
@pytest.mark.parametrize("recorded", [False, True])
def test_i1_record_receipt_requires_pending_even_for_a_stored_ack(tmp_path, terminal, recorded):
    module, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        start(owner, lease, gateway)
        cmd = command(owner_session_id="session-new", owner_epoch=8)
        receipt = gateway.send(cmd, deadline=DEADLINE)
        assert receipt.status == "accepted"
        if recorded:
            owner.record_pending_command(cmd)
            owner.record_receipt(receipt)
            # Reading terminal history is allowed; recording it again is not.
            assert store.load_strict().last_receipt == receipt
        before = store.path.read_bytes()
        with pytest.raises(module.RecoveryRequired, match="^pending_command_missing$"):
            owner.record_receipt(receipt, terminal=terminal)
        assert store.path.read_bytes() == before
        assert owner.state.pending_command is None
    finally:
        lease.close()


def test_14_store_is_private_nofollow_durable(tmp_path, monkeypatch):
    module, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def sync(fd):
        events.append("dir_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync")
        return real_fsync(fd)

    def replacing(src, dst, **kwargs):
        events.append("replace")
        return real_replace(src, dst, **kwargs)

    try:
        monkeypatch.setattr(module.os, "fsync", sync)
        monkeypatch.setattr(module.os, "replace", replacing)
        start(owner, lease, gateway)
        events.clear()
        owner.reserve_sequence(writer=lease)
        assert events == ["file_fsync", "replace", "dir_fsync"]
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
        target = tmp_path / "untouched.json"
        target.write_text("untouched")
        store.path.unlink()
        store.path.symlink_to(target)
        assert start(owner, lease, gateway).reason == "mirror_missing_or_corrupt"
        assert target.read_text() == "untouched" and store.path.is_symlink()
    finally:
        lease.close()


def runner(root, owner=None):
    from dao.prog.config.models.fastcontrol import FastControlConfig
    from dao.prog.fastctrl.publication import GatePolicy
    from dao.prog.fastctrl.runner import FastControlRunner
    assert "owner" in __import__("inspect").signature(FastControlRunner).parameters, "F2c3 runner owner hook missing"
    return FastControlRunner(SimpleNamespace(), config=FastControlConfig(),
                             plan_path=str(root / "fast_plan.json"), state_path=str(root / "fast_state.json"),
                             gate_policy=GatePolicy(None), owner=owner)


def test_15_runner_hooks_without_owner_are_inert(tmp_path):
    obj = runner(tmp_path)
    obj.start_owner_session()
    obj.stop_owner_session("shutdown")
    assert not (tmp_path / "actuation_owner").exists()
    assert not (tmp_path / "leases").exists()


def test_16_runner_hooks_acquire_revoke_close_after_flush(tmp_path, monkeypatch):
    monkeypatch.setattr("time.time_ns", lambda: 1001_000_000)
    _, store, gateway, owner = layout(tmp_path)
    obj = runner(tmp_path, owner)
    obj.start_owner_session()
    assert owner.phase == "owned"
    with pytest.raises(task_execution.LeaseBusy):
        acquire(tmp_path)
    obj.stop_owner_session("shutdown")
    assert store.load_strict().phase == "released"
    assert not owner.recovery_required
    lease = acquire(tmp_path)
    lease.close()
    events = []
    monkeypatch.setattr(obj, "_persist_state", lambda *a, **kw: events.append("flush"))
    real_stop = obj.stop_owner_session
    def stop(reason):
        events.append(reason)
        real_stop(reason)
    monkeypatch.setattr(obj, "stop_owner_session", stop)
    stopped = threading.Event()
    stopped.set()
    obj.run(stopped)
    assert events == ["flush", "shutdown"]


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("owner_epoch", True),
    ("owner_epoch", -1), ("last_reserved_sequence", True), ("last_reserved_sequence", -1),
    ("phase", "unknown"), ("owner_session_id", None), ("installation_id", ""),
    ("receiver_id", 1), ("receiver_generation", ""), ("anchor_checkpoint", ""),
    ("pending_command", {}), ("last_receipt", {}), ("pending_barrier_id", 1),
    ("last_barrier", {}),
])
def test_state_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match="^[a-z][a-z0-9_]*$"):
        initial(**{field: value})


@pytest.mark.parametrize("phase", ["binding", "owned", "unconfirmed", "revoking"])
def test_active_phase_requires_nonempty_session(phase):
    with pytest.raises(ValueError, match="^[a-z][a-z0-9_]*$"):
        initial(phase=phase)


@pytest.mark.parametrize("change", ["missing", "unknown", "bool", "nested_unknown", "nested_digest"])
def test_strict_from_dict(change):
    module = ownership()
    data = initial(pending_command=command()).to_dict()
    if change == "missing":
        data.pop("phase")
    elif change == "unknown":
        data["price"] = 1
    elif change == "bool":
        data["owner_epoch"] = True
    elif change == "nested_unknown":
        data["pending_command"]["unknown"] = 1
    else:
        data["pending_command"]["plan_digest"] = "A" * 64
    with pytest.raises(ValueError, match="^[a-z][a-z0-9_]*$"):
        module.OwnerState.from_dict(data)


@pytest.mark.parametrize("contents", ["broken", "{}", '{"schema_version":true}', '[]'])
def test_existing_invalid_mirror_is_never_overwritten(tmp_path, contents):
    _, store, gateway, owner = layout(tmp_path)
    store.path.write_text(contents)
    lease = acquire(tmp_path)
    try:
        assert start(owner, lease, gateway).reason == "mirror_missing_or_corrupt"
        assert store.path.read_text() == contents
        assert gateway.state.epoch_high_water == 7
    finally:
        lease.close()


@pytest.mark.parametrize("resource", ["", "../escape", "bad_name", "A", "a" * 65, True])
def test_resource_names_are_strict(tmp_path, resource):
    assert hasattr(task_execution, "acquire_resource"), "F2c3 acquire_resource missing"
    with pytest.raises(task_execution.LeaseError):
        task_execution.acquire_resource(resource, lease_dir=tmp_path)


@pytest.mark.parametrize("when", ["start", "stop"])
def test_late_barrier_ack_never_clears_recovery(tmp_path, when):
    _, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    clock = [0]
    owner.now_ns = lambda: clock[0]
    try:
        if when == "stop":
            start(owner, lease, gateway)
        real_revoke = gateway.revoke_and_stop
        def late(request, *, deadline):
            result = real_revoke(request, deadline=deadline)
            clock[0] = deadline.monotonic_end_ns + 1
            return result
        gateway.revoke_and_stop = late
        if when == "start":
            result = start(owner, lease, gateway)
            assert isinstance(result, ownership().RecoveryRequired), "late ACK granted a binding"
            assert result.reason == "barrier_unconfirmed"
            assert owner.phase == "inhibited"
        else:
            owner.stop(reason="shutdown")
            assert store.load_strict().phase == "unconfirmed"
        assert owner.recovery_required
        assert not gateway.effect_log
    finally:
        lease.close()


def test_unlocked_descriptor_is_not_ownership(tmp_path):
    _, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    lease.close()
    fd = os.open(store.lease_dir / (RESOURCE + ".lock"), os.O_RDWR)
    try:
        forged = task_execution.Lease(RESOURCE, fd)
        assert start(owner, forged, gateway).reason == "owner_lease_missing"
        assert gateway.state.epoch_high_water == 7
    finally:
        os.close(fd)


def test_duplicate_json_keys_are_corrupt_and_preserved(tmp_path):
    _, store, gateway, owner = layout(tmp_path)
    text = store.path.read_text().replace('"schema_version": 1', '"schema_version": 2, "schema_version": 1')
    store.path.write_text(text)
    lease = acquire(tmp_path)
    try:
        result = start(owner, lease, gateway)
        assert isinstance(result, ownership().RecoveryRequired), "duplicate keys granted a binding"
        assert result.reason == "mirror_missing_or_corrupt"
        assert store.path.read_text() == text
    finally:
        lease.close()


@pytest.mark.parametrize("failure", ["busy", "start", "stop", "flush"])
def test_runner_lifecycle_errors_do_not_escape_or_leak_lease(tmp_path, monkeypatch, failure):
    _, store, gateway, owner = layout(tmp_path)
    monkeypatch.setattr("time.time_ns", lambda: 1001_000_000)
    obj = runner(tmp_path, owner)
    lease = acquire(tmp_path) if failure == "busy" else None
    def fail(*args, **kwargs):
        raise OSError("synthetic lifecycle failure")
    try:
        if failure == "start":
            monkeypatch.setattr(owner, "start", fail)
        obj.start_owner_session()
        if failure in ("busy", "start"):
            assert owner.phase == "inhibited"
        if failure == "stop":
            monkeypatch.setattr(owner, "stop", fail)
        if failure == "flush":
            stopped = threading.Event()
            stopped.set()
            monkeypatch.setattr(obj, "_persist_state", fail)
            # Existing flush error semantics remain; lifecycle still closes.
            with pytest.raises(OSError):
                obj.run(stopped)
        else:
            obj.stop_owner_session("shutdown")
    finally:
        if lease is not None:
            lease.close()
    probe = acquire(tmp_path)
    probe.close()


def test_unreadable_mirror_metadata_is_recovery_not_crash(tmp_path, monkeypatch):
    module, store, gateway, owner = layout(tmp_path)
    real_open, real_lstat = os.open, Path.lstat
    def unreadable(path, *args, **kwargs):
        if Path(path) == store.path.parent:
            raise PermissionError("synthetic unreadable mirror directory")
        return real_open(path, *args, **kwargs)
    def inaccessible(path, *args, **kwargs):
        if path == store.path:
            raise PermissionError("synthetic inaccessible mirror metadata")
        return real_lstat(path, *args, **kwargs)
    lease = acquire(tmp_path)
    try:
        monkeypatch.setattr(module.os, "open", unreadable)
        monkeypatch.setattr(Path, "lstat", inaccessible)
        try:
            result = start(owner, lease, gateway)
        except PermissionError as exc:
            result = exc
        assert isinstance(result, module.RecoveryRequired), "unreadable metadata crashed start"
        assert result.reason == "mirror_missing_or_corrupt"
        assert gateway.state.epoch_high_water == 7
    finally:
        lease.close()


@pytest.mark.parametrize("end_ns", [100_000_000_000, 100_000_000_001])
def test_fix1_a_final_owned_write_consumes_deadline(tmp_path, monkeypatch, end_ns):
    module, store, gateway, owner = layout(tmp_path)
    clock = [0]
    owner.now_ns = lambda: clock[0]
    real_write = store.write
    def slow_owned(state, *, writer):
        real_write(state, writer=writer)
        if state.phase == "owned":
            clock[0] = end_ns
    monkeypatch.setattr(store, "write", slow_owned)
    lease = acquire(tmp_path)
    try:
        result = start(owner, lease, gateway)
        assert isinstance(result, module.RecoveryRequired), "owned write released an overdue binding"
        assert result.reason == "deadline_exceeded"
        assert owner.phase == "inhibited" and owner.recovery_required and owner.binding is None
        assert store.load_strict().phase == "owned"  # evidence, never execution rights
        assert not gateway.effect_log
    finally:
        lease.close()


@pytest.mark.parametrize("when", ["bind", "owned_write"])
@pytest.mark.parametrize("current_ms", [6000, 9000])
def test_fix1_a_binding_expiry_uses_current_time(tmp_path, monkeypatch, when, current_ms):
    module, store, gateway, owner = layout(tmp_path)
    clock = [1001]
    owner.now_ms = lambda: clock[0]
    if when == "bind":
        real_bind = gateway.bind_owner
        def expired_bind(**kwargs):
            binding = real_bind(**kwargs)
            clock[0] = current_ms
            return binding
        monkeypatch.setattr(gateway, "bind_owner", expired_bind)
    else:
        real_write = store.write
        def expired_write(state, *, writer):
            real_write(state, writer=writer)
            if state.phase == "owned":
                clock[0] = current_ms
        monkeypatch.setattr(store, "write", expired_write)
    lease = acquire(tmp_path)
    try:
        result = start(owner, lease, gateway)
        assert isinstance(result, module.RecoveryRequired), "start released an already expired binding"
        assert result.reason == "owner_lease_expired"
        assert owner.phase == "inhibited" and owner.recovery_required and owner.binding is None
    finally:
        lease.close()


@pytest.mark.parametrize("operation", ["file_fsync", "replace", "directory_fsync"])
def test_fix1_b_failed_reservation_survives_stop_and_older_reload(tmp_path, monkeypatch, operation):
    _, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        start(owner, lease, gateway)
        older = store.path.read_bytes()
        real_sync = os.fsync
        def failed_sync(fd):
            directory = stat.S_ISDIR(os.fstat(fd).st_mode)
            if directory == (operation == "directory_fsync"):
                raise OSError("synthetic reservation fsync failure")
            real_sync(fd)
        def failed_replace(*args, **kwargs):
            raise OSError("synthetic reservation replace failure")
        with monkeypatch.context() as failure:
            if operation == "replace":
                failure.setattr(os, "replace", failed_replace)
            else:
                failure.setattr(os, "fsync", failed_sync)
            with pytest.raises(OSError):
                owner.reserve_sequence(writer=lease)
        if operation == "directory_fsync":
            assert store.load_strict().last_reserved_sequence == 41
        assert owner.phase == "inhibited" and owner.recovery_required
        owner.stop(reason="shutdown")
        assert store.load_strict().last_reserved_sequence == 41, "stop rolled back attempted reservation"
        store.path.write_bytes(older)
        assert start(owner, lease, gateway, owner_session_id="restart").owner_epoch > 8
        assert store.load_strict().last_reserved_sequence == 41, "reload lowered process high-water"
        assert owner.reserve_sequence(writer=lease) == 42
        assert not gateway.effect_log
    finally:
        lease.close()


def test_fix1_d_public_failure_report_inhibits_and_preserves_reason(tmp_path):
    module, _, gateway, owner = layout(tmp_path)
    assert callable(getattr(owner, "report_failure", None)), "public report_failure missing"
    lease = acquire(tmp_path)
    try:
        start(owner, lease, gateway)
        result = owner.report_failure("owner_stop_failed")
        assert isinstance(result, module.RecoveryRequired) and result.reason == "owner_stop_failed"
        assert owner.phase == "inhibited" and owner.recovery_required and owner.binding is None
    finally:
        lease.close()


@pytest.mark.parametrize("when", ["start", "reserve", "stop"])
def test_fix1_d_mirror_write_is_a_classified_hard_error(tmp_path, monkeypatch, when):
    _, store, gateway, owner = layout(tmp_path)
    lease = acquire(tmp_path)
    try:
        if when != "start":
            start(owner, lease, gateway)
        def failed_write(*args, **kwargs):
            raise OSError("synthetic mirror write failure")
        monkeypatch.setattr(store, "write", failed_write)
        with pytest.raises(OSError) as error:
            if when == "start":
                start(owner, lease, gateway)
            elif when == "reserve":
                owner.reserve_sequence(writer=lease)
            else:
                owner.stop(reason="shutdown")
        assert getattr(error.value, "reason", None) == "mirror_write_failed", "write failure lost its cause code"
        assert owner.phase == "inhibited" and owner.recovery_required
    finally:
        lease.close()


@pytest.mark.parametrize("failure,expected", [
    ("busy", "owner_lease_busy"), ("start", "owner_start_failed"),
    ("stop", "owner_stop_failed"), ("mirror_start", "mirror_write_failed"),
    ("mirror_stop", "mirror_write_failed"),
])
def test_fix1_d_runner_uses_public_failure_reporting(tmp_path, monkeypatch, failure, expected):
    _, store, gateway, owner = layout(tmp_path)
    assert callable(getattr(owner, "report_failure", None)), "runner needs public report_failure"
    reported = []
    report = owner.report_failure
    def reporting(reason):
        reported.append(reason)
        return report(reason)
    monkeypatch.setattr(owner, "report_failure", reporting)
    monkeypatch.setattr("time.time_ns", lambda: 1001_000_000)
    obj = runner(tmp_path, owner)
    blocker = acquire(tmp_path) if failure == "busy" else None
    def fail(*args, **kwargs):
        raise OSError("synthetic hook failure")
    try:
        if failure == "start":
            monkeypatch.setattr(owner, "start", fail)
        elif failure == "mirror_start":
            monkeypatch.setattr(store, "write", fail)
        obj.start_owner_session()
        if failure in ("stop", "mirror_stop"):
            monkeypatch.setattr(owner if failure == "stop" else store,
                                "stop" if failure == "stop" else "write", fail)
            obj.stop_owner_session("shutdown")
        assert reported and reported[-1] == expected
        assert owner.phase == "inhibited" and owner.recovery_required
    finally:
        # Remove injection before orderly cleanup, without changing the assertion.
        monkeypatch.undo()
        obj.stop_owner_session("shutdown")
        if blocker is not None:
            blocker.close()
