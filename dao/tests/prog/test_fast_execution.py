"""F2c4: the mechanical publication boundary, the bounded send and the owner section.

Two real processes (owner and registrator), a temporary SQLite book and IPC
barriers prove the ordering. Nothing here grants a public execute right: the
production v2 route still ends in ``actuation_gates_unproven``.
"""

import hashlib
import inspect
import json
import multiprocessing
import threading
from dataclasses import replace
from importlib import import_module, util
from pathlib import Path
from types import SimpleNamespace

import pytest

from dao.prog import task_execution
from dao.prog.fastctrl.command import ClockSample, Deadline
from dao.prog.fastctrl.publication import (
    GatePolicy,
    PlanRead,
    PublicationError,
    PublicationLockBusy,
    publication_lock,
    read_publication,
)
from dao.prog.fastctrl.receiver import payload_digest
from dao.tests.prog.fakes.fenced_receiver import FencedReceiver
from dao.tests.prog.test_fast_receiver import state as receiver_state
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_fast_contract import ready_plan
from dao.tests.settlement.test_publication import (
    METER,
    connect_publication,
    item,
    pricing_config,
    publish_fixture,
)

CTX = multiprocessing.get_context("fork")
INSTALLATION = "sandbox"
RESOURCE = "actuator-" + hashlib.sha256(b"sandbox").hexdigest()[:32]
MIRROR = hashlib.sha256(b"sandbox").hexdigest() + ".json"
MONO = 15_000_000_000
#: De boek- en plantijdlijn is seconden.
NOW_TS = START_TS + 900
#: De klok en het ontvangende contract zijn milliseconden.
NOW = NOW_TS * 1000
POLICY = GatePolicy(START_TS)
CLOCK = ClockSample(utc_ms=NOW, boottime_ns=MONO, boot_id="sandbox-boot",
                     uncertainty_ms=0, healthy=True)
DEADLINE = Deadline(MONO + 1_000_000_000, NOW + 60_000)


def execution():
    # Deliberate assertion RED, never a collection/import error.
    assert util.find_spec("dao.prog.fastctrl.execution") is not None, "F2c4 execution module missing"
    return import_module("dao.prog.fastctrl.execution")


@pytest.mark.parametrize("boundary", ["owner_section", "send_locked"])
def test_i7_direct_start_stop_and_send_have_no_admission_bypass(artifact, boundary):
    """An independently acquired publication lease grants no owner section."""
    module = execution()
    layout = owner_layout(artifact.root)
    try:
        command = build_command(module, layout, artifact)
        layout.owner.record_pending_command(command)
        before = layout.store.path.read_bytes()
        failure = None
        with publication_lock(artifact.path) as lease:
            try:
                if boundary == "send_locked":
                    module.send_locked(command, lease=lease, owner=layout.owner,
                        receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK, policy=POLICY)
                else:
                    module.owner_section(lease=lease, owner=layout.owner,
                        receiver=layout.gateway, request=request(), deadline=DEADLINE,
                        clock=CLOCK, policy=POLICY)
            except module.ExecutionRefused as exc:
                failure = exc.reason
        assert failure == "owner_section_required"
        assert layout.store.path.read_bytes() == before
        assert layout.gateway.effect_log == []
    finally:
        layout.lease.close()


def attempts():
    module = import_module("dao.prog.fastctrl.publication")
    assert hasattr(module, "publication_attempts"), "F2c4 publication_attempts missing"
    return module


def i7_owner_work(root, clock, control, ready, release):
    """Keep offering real read/send/stop work while the publisher is retrying."""
    from contextlib import contextmanager

    from dao.prog.fastctrl import ownership as own
    from dao.prog.fastctrl import runner as run
    from dao.tests.prog.test_fast_consumer import propose
    from dao.tests.prog.test_fast_runner import FakeHass, make_config

    root = Path(root)
    store = provision_mirror(root, installation_id=METER)
    gateway = FencedReceiver(root / "receiver", initial_state=receiver_state(installation_id=METER), utc_ms=NOW)
    gateway.receiver_max_lease_ms = 60_000
    owner = own.OwnerCoordinator(store, receiver=gateway, now_ns=lambda: clock.value, now_ms=lambda: NOW)
    writer = task_execution.acquire_resource(store.resource, lease_dir=root / "leases")
    acquisitions = []
    real_lock = publication_lock

    @contextmanager
    def observed(path):
        with real_lock(path) as lease:
            acquisitions.append(clock.value)
            yield lease

    run.publication_lock = observed
    if hasattr(own, "publication_lock"):
        own.publication_lock = observed
    run.time.clock_gettime_ns = lambda _: clock.value
    # A DB call from this owner process is always a contract violation.
    import sqlalchemy
    sqlalchemy.create_engine = lambda *a, **k: (_ for _ in ()).throw(AssertionError("owner DB I/O"))

    def start():
        return owner.start(writer=writer, receiver=gateway, installation_id=METER,
                           owner_session_id="i7-session", requested_not_after_ms=NOW + 60_000,
                           deadline=Deadline(clock.value + 1_000_000_000, NOW + 60_000), now_ms=NOW)

    start()  # Lifetime observation at cycle zero: the implementation must defer.
    clock.value = 5_000_000_000
    start()
    runner = run.FastControlRunner(FakeHass(), make_config(), plan_path=str(root / "fast_plan.json"),
                                  state_path=str(root / "fast_state.json"), gate_policy=POLICY, owner=owner)
    reader = run.FastControlRunner(FakeHass(), make_config(), plan_path=str(root / "fast_plan.json"),
                                  state_path=str(root / "read_state.json"), gate_policy=POLICY)
    plan = read_publication(root / "fast_plan.json").plan
    real_send = gateway.send
    first = True

    def send(command, *, deadline):
        nonlocal first
        if first:
            first = False
            ready.set()
            assert release.wait(timeout=30)
        return real_send(command, deadline=deadline)

    gateway.send = send
    control.send("ready")
    try:
        while True:
            operation, at = control.recv()
            clock.value = at
            if operation == "done":
                control.send((acquisitions, len(gateway.effect_log)))
                break
            if operation == "start":
                result = start()
                outcome = getattr(result, "reason", "bound")
            elif operation == "stop":
                result = owner.stop(reason="handoff")
                outcome = None if result is None else result.status
            else:
                reader.tick(NOW_TS)
                receipts = runner._consume_financial_actuation(
                    propose(plan, runner.state), plan, owner=owner, now_ms=NOW)
                outcome = [receipt.status for receipt in receipts]
            control.send((operation, at, outcome))
    finally:
        writer.close()


def i7_publisher_work(url, path, pricing, clock, control, ready, tripwire):
    """Real bounded acquisition, real final basis check/write; no receiver or commit."""
    from contextlib import contextmanager

    from dao.prog.fastctrl import publication as pub
    from dao.prog.settlement import publication as settlement_publication

    install_receiver_tripwire(tripwire)
    engine, repo, adapter = connect_publication(url, path, pricing)
    plan = read_publication(path).plan
    real_lock, real_check = pub.publication_lock, adapter._check_basis_locked
    events, checks, writes = [], [], []
    real_write = settlement_publication._write_verified_ready

    @contextmanager
    def observed(target):
        try:
            with real_lock(target) as lease:
                events.append(("acquired", clock.value))
                yield lease
        except PublicationLockBusy:
            events.append(("busy", clock.value))
            raise

    def wait(seconds):
        control.send(("retry", seconds))
        assert control.recv() == "continue"

    def check(*args, **kwargs):
        checks.append(clock.value)
        return real_check(*args, **kwargs)

    def write(*args, **kwargs):
        writes.append(clock.value)
        return real_write(*args, **kwargs)

    pub.publication_lock, pub.clock_gettime_ns, pub._sleep = observed, lambda _: clock.value, wait
    adapter._check_basis_locked = check
    settlement_publication._write_verified_ready = write
    repo.commit = lambda *a, **k: (_ for _ in ()).throw(AssertionError("repeated settlement commit"))
    assert ready.wait(timeout=30)
    # Candidate is already solved/committed by the system fixture, exactly once.
    adapter.publish(plan, now_ts=NOW_TS)
    control.send(("published", events, checks, writes))
    engine.dispose()


def test_i7_continuous_owner_work_leaves_publication_window(system):
    """Busy admitted send -> continuous owner offers -> publisher wins phase two."""
    from decimal import Decimal

    plan = system.plan
    context = plan.financial_context
    plan = replace(plan, specs=[plan.specs[0], replace(plan.specs[0], name="accu2")],
                   intervals=[replace(row, batteries=[row.battery(0), row.battery(0)]) for row in plan.intervals],
                   financial_context=replace(context, terminal_value=replace(context.terminal_value,
                       reference_efficiency_by_battery=(("accu", Decimal("0.94")), ("accu2", Decimal("0.94"))))))
    publish_fixture(system.adapter, plan, (system.snapshot,))
    clock, ready, release = CTX.Value("q", 0), CTX.Event(), CTX.Event()
    parent, child = CTX.Pipe()
    publisher_parent, publisher_child = CTX.Pipe()
    tripwire = str(system.root / "i7-receiver-tripwire")
    owner = CTX.Process(target=i7_owner_work, args=(str(system.root), clock, child, ready, release))
    publisher = CTX.Process(target=i7_publisher_work,
                            args=(system.url, str(system.path), system.pricing, clock,
                                  publisher_child, ready, tripwire))
    reports = []

    def receive(pipe):
        assert pipe.poll(30), "missing causal IPC acknowledgement"
        return pipe.recv()

    def work(operation, at):
        parent.send((operation, at))
        reports.append(receive(parent))

    owner.start()
    publisher.start()
    try:
        assert receive(parent) == "ready"
        parent.send(("work", 20_000_000_000))
        assert receive(publisher_parent) == ("retry", 0.05)
        release.set()
        reports.append(receive(parent))
        work("work", 20_050_000_000)
        work("work", 21_000_000_000)
        work("work", 22_000_000_000)
        work("stop", 22_000_000_000)
        publisher_parent.send("continue")
        published = receive(publisher_parent)
        for at in (23_000_000_000, 24_000_000_000):
            work("work", at)
        work("start", 25_000_000_000)
        work("work", 30_000_000_000)
        parent.send(("done", 30_000_000_000))
        acquisitions, effects = receive(parent)
        for process in (owner, publisher):
            join(process)
            assert process.exitcode == 0
        assert published == ("published", [("busy", 20_000_000_000), ("acquired", 22_000_000_000)],
                             [22_000_000_000], [22_000_000_000])
        assert all(at % 5_000_000_000 < 1_000_000_000 for at in acquisitions), acquisitions
        assert reports[0][2] == reports[-1][2] == ["accepted", "accepted"]
        assert effects == 12
        assert not Path(tripwire).exists()
    finally:
        release.set()
        for process in (owner, publisher):
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)


def ownership():
    module = import_module("dao.prog.fastctrl.ownership")
    assert hasattr(module.OwnerCoordinator, "record_pending_command"), (
        "F2c4 record_pending_command missing")
    assert hasattr(module.OwnerCoordinator, "record_receipt"), "F2c4 record_receipt missing"
    return module


def mirror_path(root, *, installation_id=INSTALLATION):
    return Path(root) / "actuation_owner" / (
        hashlib.sha256(installation_id.encode()).hexdigest() + ".json")


def receiver_state_file(root):
    return Path(root) / "receiver" / "receiver-state.json"


def read_mirror(root, *, installation_id=INSTALLATION):
    return json.loads(mirror_path(root, installation_id=installation_id).read_text())


def provision_mirror(root, *, installation_id=INSTALLATION):
    """The synthetic trusted mirror; production has no bootstrap path."""
    module = ownership()
    store = module.OwnerStore(Path(root) / "fast_plan.json", installation_id,
                              lease_dir=Path(root) / "leases")
    store.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    store.path.write_text(json.dumps(module.OwnerState(
        1, installation_id, "receiver-a", "generation-a", "checkpoint-40", "", 0, 40,
        "inhibited", None, None, None, None).to_dict()))
    store.path.chmod(0o600)
    return store


def owner_layout(root, *, requested_not_after_ms=None):
    """Bind one synthetic owner against the deterministic fake receiver."""
    module = ownership()
    store = provision_mirror(root)
    gateway = FencedReceiver(Path(root) / "receiver", initial_state=receiver_state(), utc_ms=NOW - 10_000)
    gateway.receiver_max_lease_ms = 70_000
    owner = module.OwnerCoordinator(store, receiver=gateway)
    owner.now_ns = lambda: 0
    owner.now_ms = lambda: gateway.utc_ms
    lease = task_execution.acquire_resource(RESOURCE, lease_dir=Path(root) / "leases")
    owner._bind_lifetime(lease)
    owner.now_ns = lambda: 5_000_000_000
    binding = owner.start(writer=lease, receiver=gateway, installation_id=INSTALLATION,
                          owner_session_id="session-new",
                          requested_not_after_ms=requested_not_after_ms or (NOW + 60_000),
                          deadline=Deadline(6_000_000_000, NOW + 60_000), now_ms=gateway.utc_ms)
    assert binding.owner_epoch == 8, binding
    gateway.utc_ms = NOW
    owner.now_ns = lambda: MONO + gateway.monotonic_ns
    return SimpleNamespace(module=module, store=store, gateway=gateway, owner=owner,
                           lease=lease, binding=binding)


def next_owner_cycle(owner):
    """Advance only the injected BOOTTIME to a separate lifecycle/test section.

    UTC boundary probes control their own independent wire clock. No admission
    predicate or budget is replaced, and the BOOTTIME never goes backwards.
    """
    previous = owner.now_ns
    now = previous()
    target = (now // 5_000_000_000 + 1) * 5_000_000_000
    owner.now_ns = lambda: previous() + target - now
    return target


def final_section(owner):
    """Real admission with the normal one-second cap; wire bounds stay per probe."""
    return owner.final_section(operation="send", deadline=Deadline(
        owner.now_ns() + 1_000_000_000, owner.now_ms() + 3_600_000))


@pytest.mark.parametrize("fault", ["rollback", "boot"])
def test_i7_clock_change_during_transport_cannot_reopen_context(artifact, monkeypatch, fault):
    module = execution()
    layout = owner_layout(artifact.root)
    clock = [MONO]
    layout.owner.now_ns = lambda: clock[0]
    real_send = layout.gateway.send
    calls = []

    def send(command, *, deadline):
        calls.append(command)
        receipt = real_send(command, deadline=deadline)
        if fault == "rollback":
            clock[0] -= 1
        else:
            monkeypatch.setattr(layout.module, "_boot_id", lambda: "another-boot")
        return receipt

    layout.gateway.send = send
    try:
        failure = None
        with final_section(layout.owner) as lease:
            try:
                module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                     request=request(), policy=POLICY, clock=CLOCK, deadline=DEADLINE)
            except module.ExecutionRefused as exc:
                failure = exc.reason
        assert failure == "clock_unhealthy"
        assert len(calls) == 1
        assert layout.owner.binding is None and layout.owner.recovery_required
        assert layout.store.load_strict().pending_command == calls[0]
    finally:
        layout.lease.close()


def i7_receipt_write_clock_fault(monkeypatch, owner, *, fault, fail_restore=False):
    """Fault only after the real receipt write durably cleared the pending command."""
    clock = [owner.now_ns()]
    monkeypatch.setattr(owner, "now_ns", lambda: clock[0])
    real_write = owner.store.write
    committed = []

    def write(state, **kwargs):
        if fail_restore and committed and state.pending_command is not None:
            raise OSError("pending restoration failed")
        pending = owner.state.pending_command
        real_write(state, **kwargs)
        if state.last_receipt is not None and state.pending_command is None:
            committed.append((pending, owner.store.load_strict()))
            if fault == "rollback":
                clock[0] -= 1
            else:
                monkeypatch.setattr(ownership(), "_boot_id", lambda: "changed-after-receipt-write")

    monkeypatch.setattr(owner.store, "write", write)
    return committed


@pytest.mark.parametrize("route", ["direct", "section"])
@pytest.mark.parametrize("fault", ["rollback", "boot"])
def test_i7_fix1_clock_failure_after_receipt_write_retains_pending(artifact, monkeypatch, route, fault):
    module = execution()
    layout = owner_layout(artifact.root)
    committed = i7_receipt_write_clock_fault(monkeypatch, layout.owner, fault=fault)
    try:
        with pytest.raises(module.ExecutionRefused, match="^clock_unhealthy$"):
            with final_section(layout.owner) as lease:
                if route == "direct":
                    command = build_command(module, layout, artifact)
                    layout.owner.record_pending_command(command)
                    module.send_locked(command, lease=lease, owner=layout.owner, receiver=layout.gateway,
                                       deadline=DEADLINE, clock=CLOCK, policy=POLICY)
                else:
                    module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                         request=request(), deadline=DEADLINE, clock=CLOCK, policy=POLICY)
        assert len(committed) == 1
        command, written = committed[0]
        assert written.pending_command is None and written.last_receipt.status == "accepted"
        assert written.last_receipt.command_id == command.command_id
        state = layout.store.load_strict()
        assert state.last_receipt == written.last_receipt
        assert state.pending_command == layout.owner.state.pending_command == command
        assert state.phase == layout.owner.phase == "inhibited"
        assert layout.owner.binding is None and layout.owner.recovery_required
        assert len(layout.gateway.effect_log) == 3
        with pytest.raises(layout.module.RecoveryRequired, match="^clock_unhealthy$"):
            with final_section(layout.owner):
                pytest.fail("post-write clock failure reopened an owner section")
    finally:
        layout.lease.close()


def test_i7_fix1_pending_restore_write_failure_is_not_masked(artifact, monkeypatch):
    module = execution()
    layout = owner_layout(artifact.root)
    committed = i7_receipt_write_clock_fault(monkeypatch, layout.owner, fault="rollback", fail_restore=True)
    try:
        with pytest.raises(layout.module.MirrorWriteFailed, match="pending restoration failed"):
            with final_section(layout.owner) as lease:
                module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                     request=request(), deadline=DEADLINE, clock=CLOCK, policy=POLICY)
        assert len(committed) == 1
        assert layout.store.load_strict().last_receipt == committed[0][1].last_receipt
        assert layout.owner.binding is None and layout.owner.recovery_required
        assert len(layout.gateway.effect_log) == 3
    finally:
        layout.lease.close()


def test_i7_claim_survives_exception_and_new_coordinator(artifact):
    layout = owner_layout(artifact.root)
    try:
        with pytest.raises(RuntimeError, match="body"):
            with final_section(layout.owner):
                raise RuntimeError("body")
        assert layout.owner.section_deadline is layout.owner.transport_deadline is None
        other = layout.module.OwnerCoordinator(layout.store, receiver=layout.gateway,
                                               now_ns=layout.owner.now_ns, now_ms=layout.owner.now_ms)
        result = other.start(writer=layout.lease, receiver=layout.gateway, installation_id=INSTALLATION,
            owner_session_id="other", requested_not_after_ms=NOW + 60_000,
            deadline=DEADLINE, now_ms=NOW)
        assert result.reason == "admission_cycle_used"
        assert layout.gateway.state.epoch_high_water == 8
        with pytest.raises(layout.module.RecoveryRequired, match="admission_cycle_used"):
            with final_section(layout.owner):
                pytest.fail("exception returned its cycle claim")
    finally:
        layout.lease.close()


def test_i7_acquire_recheck_and_shared_budget(artifact, monkeypatch):
    from contextlib import contextmanager

    layout = owner_layout(artifact.root)
    clock = [MONO]
    layout.owner.now_ns = lambda: clock[0]
    real_lock = layout.module.publication_lock
    cost = [200_000_000]

    @contextmanager
    def slow(path):
        with real_lock(path) as lease:
            clock[0] += cost[0]
            yield lease

    monkeypatch.setattr(layout.module, "publication_lock", slow)
    try:
        with final_section(layout.owner) as lease:
            assert layout.owner.section_deadline.monotonic_end_ns == MONO + 1_000_000_000
            assert layout.owner.transport_deadline.remaining_ns(clock[0]) == 550_000_000
            layout.owner.require_final_section(lease, operation="send")
        with pytest.raises(layout.module.RecoveryRequired, match="owner_section_required"):
            layout.owner.require_final_section(lease, operation="send")
        clock[0], cost[0] = 20_000_000_000, 1_000_000_000
        before = layout.store.path.read_bytes()
        with pytest.raises(layout.module.RecoveryRequired, match="admission_window_missed"):
            with final_section(layout.owner):
                pytest.fail("acquire overrun admitted the body")
        assert layout.store.path.read_bytes() == before
        with publication_lock(artifact.path):
            pass
    finally:
        layout.lease.close()


def test_i7_lifetime_checks_consume_the_section_budget(artifact, monkeypatch):
    layout = owner_layout(artifact.root)
    clock = [MONO]
    layout.owner.now_ns = lambda: clock[0]
    real_holds = layout.store.holds

    def holds(writer):
        result = real_holds(writer)
        clock[0] += 200_000_000
        return result

    monkeypatch.setattr(layout.store, "holds", holds)
    try:
        with final_section(layout.owner):
            assert layout.owner.section_deadline.monotonic_end_ns == MONO + 1_000_000_000
            assert layout.owner.transport_deadline.monotonic_end_ns == MONO + 750_000_000
            assert layout.owner.transport_deadline.remaining_ns(clock[0]) == 350_000_000
    finally:
        layout.lease.close()


def test_i7_cycle_claim_belongs_to_lifetime_not_new_store_object(artifact):
    layout = owner_layout(artifact.root)
    owner = layout.owner
    other_store = layout.module.OwnerStore(artifact.root / "other-plan.json", INSTALLATION,
                                           lease_dir=artifact.root / "leases")
    other = layout.module.OwnerCoordinator(other_store, receiver=layout.gateway,
        now_ns=lambda: owner.now_ns(), now_ms=owner.now_ms)
    try:
        other._bind_lifetime(layout.lease)
        next_owner_cycle(owner)
        with final_section(owner):
            pass
        before = layout.store.path.read_bytes()
        result = other.start(writer=layout.lease, receiver=layout.gateway, installation_id=INSTALLATION,
            owner_session_id="other", requested_not_after_ms=NOW + 60_000,
            deadline=Deadline(owner.now_ns() + 1_000_000_000, NOW + 60_000), now_ms=NOW)
        assert isinstance(result, layout.module.RecoveryRequired), "a new store returned the lifetime's cycle claim"
        assert result.reason == "admission_cycle_used"
        assert layout.store.path.read_bytes() == before
    finally:
        layout.lease.close()


def test_i7_invalidation_after_send_cannot_relabel_the_cycle(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    try:
        with final_section(layout.owner) as lease:
            receipt = module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                request=request(), deadline=DEADLINE, clock=CLOCK, policy=POLICY)
            assert receipt.status == "accepted"
            artifact.path.write_text(json.dumps({"format_version": 2, "status": "blocked",
                                                 "created_ts": NOW_TS, "reason": "correction"}))
            assert module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                request=request(), deadline=DEADLINE, clock=CLOCK, policy=POLICY) is None
            assert layout.gateway.state.epoch_high_water == 8
            assert layout.owner.pending_stop_reason == "publication_invalidated"
            assert layout.owner.binding is None
        next_owner_cycle(layout.owner)
        assert layout.owner.stop(reason="publication_invalidated").status == "accepted"
        assert layout.gateway.state.epoch_high_water == 9
        assert len(layout.gateway.effect_log) == 3
    finally:
        layout.lease.close()


def request():
    module = execution()
    return module.CommandRequest(battery_name="accu", setpoint_w=-1200, mode="Aan",
                                 stop_at_ms=None)


def build_command(module, layout, artifact):
    from dao.prog.fastctrl.command import BatteryCommand

    binding = layout.binding
    sequence = layout.owner.reserve_sequence(writer=layout.lease)
    execute_before = min(artifact.plan.financial_context.not_after_ts * 1000,
                         binding.owner_not_after_ms)
    return BatteryCommand(
        installation_id=binding.installation_id, battery_name="accu",
        owner_session_id=binding.owner_session_id, owner_epoch=binding.owner_epoch,
        sequence=sequence,
        command_id=f"{hashlib.sha256(binding.installation_id.encode()).hexdigest()[:32]}/"
                   f"{binding.owner_epoch}/{sequence}",
        plan_digest=module.plan_digest(artifact.plan), issued_at_ms=CLOCK.utc_ms,
        execute_before_ms=execute_before, hold_until_ms=execute_before,
        setpoint_w=-1200, mode="Aan", stop_at_ms=None,
    )


@pytest.fixture
def artifact(tmp_path):
    plan = ready_plan()
    context = plan.financial_context
    plan = replace(plan, financial_context=replace(
        context, basis=replace(context.basis, meter_id=INSTALLATION)))
    path = tmp_path / "fast_plan.json"
    path.write_text(json.dumps(plan.to_dict()))
    return SimpleNamespace(root=tmp_path, path=path, plan=plan)


@pytest.fixture
def system(tmp_path, monkeypatch):
    publication = import_module("dao.prog.settlement.publication")
    monkeypatch.setattr(publication, "time", lambda: NOW_TS)
    from dao.prog.settlement.engine import settle_series
    from dao.prog.settlement.tariffs import make_snapshot
    from dao.prog.settlement.types import SettlementState

    pricing = pricing_config()
    path = tmp_path / "fast_plan.json"
    url = f"sqlite:///{tmp_path / 'book.db'}"
    engine, repo, adapter = connect_publication(url, path, pricing)
    repo.ensure_schema()
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    repo.register_inputs(METER, [item()], [snapshot], 0)
    batch = repo.read_inputs(METER, START_TS, NOW_TS)
    result = settle_series(batch.items, batch.snapshots, SettlementState(START_TS, None, None))
    repo.commit(METER, result, batch, 0, "synthetic-clean-prefix")
    plan = ready_plan(hashes=(snapshot.snapshot_hash,))
    plan = publish_fixture(adapter, plan, (snapshot,))
    yield SimpleNamespace(repo=repo, adapter=adapter, plan=plan, path=path, snapshot=snapshot,
                          pricing=pricing, engine=engine, url=url, root=tmp_path)
    engine.dispose()


# -- the two real processes ------------------------------------------------

def owner_worker(root, mode, barrier, go, registered, results, send_started=None, send_continue=None):
    """The owner process: bind the receiver, then run bounded owner sections."""
    module = execution()
    root = Path(root)
    store = provision_mirror(root, installation_id=METER)
    gateway = FencedReceiver(root / "receiver", initial_state=receiver_state(installation_id=METER),
                             utc_ms=NOW)
    gateway.receiver_max_lease_ms = 60_000
    owner = store_owner(store, gateway)
    lease = task_execution.acquire_resource(store.resource, lease_dir=root / "leases")
    owner.now_ns = lambda: 0
    owner._bind_lifetime(lease)
    owner.now_ns = lambda: 5_000_000_000
    binding = owner.start(writer=lease, receiver=gateway, installation_id=METER,
                          owner_session_id="session-new", requested_not_after_ms=NOW + 60_000,
                          deadline=DEADLINE, now_ms=NOW)
    assert binding.owner_epoch == 8, binding
    owner.now_ns = lambda: MONO
    command_request = module.CommandRequest(battery_name="accu", setpoint_w=-1200,
                                             mode="Aan", stop_at_ms=None)
    if send_started is not None:
        real_send = gateway.send

        def send(command, *, deadline):
            send_started.set()
            assert send_continue.wait(timeout=30)
            return real_send(command, deadline=deadline)

        gateway.send = send

    def section():
        clock = replace(CLOCK, boottime_ns=owner.now_ns())
        deadline = Deadline(owner.now_ns() + 1_000_000_000, NOW + 60_000)
        with owner.final_section(operation="send", deadline=deadline) as lease:
            return module.owner_section(lease=lease, owner=owner, receiver=gateway,
                                        request=command_request, policy=POLICY, clock=clock,
                                        deadline=deadline)

    statuses = []
    barrier.wait(timeout=30)
    if mode == "send-first":
        receipt = section()
        statuses.append(receipt.status if receipt is not None else None)
        results.put(("first-section", statuses[-1]))
        registered.wait(timeout=30)
    else:
        go.wait(timeout=30)
        registered.wait(timeout=30)
    next_owner_cycle(owner)
    receipt = section()
    statuses.append(receipt.status if receipt is not None else None)
    results.put(("sections", statuses, len(gateway.effect_log), owner.phase,
                 store.load_strict().phase))
    lease.close()


def store_owner(store, gateway):
    module = ownership()
    owner = module.OwnerCoordinator(store, receiver=gateway)
    owner.now_ns = lambda: MONO
    owner.now_ms = lambda: NOW
    return owner


def record_attempt(path, kind):
    """Write one acquisition event to the shared evidence file."""
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(kind + "\n")


def registrator_worker(url, path, pricing, barrier, go, registered, results,
                       handshake=None, tripwire=None, attempts=None):
    """The registrator process: a relevant input correction, then a durable block."""
    from contextlib import contextmanager

    from dao.prog.settlement.tariffs import make_snapshot

    if tripwire is not None:
        install_receiver_tripwire(tripwire)
    if handshake is not None:
        import dao.prog.fastctrl.publication as fast_publication

        real_lock = fast_publication.publication_lock

        @contextmanager
        def recording_lock(target):
            # The handshake follows a *real* acquisition attempt that the
            # blockade answered with PublicationLockBusy; it never precedes
            # it. The bounded acquisition behind it continues as before.
            try:
                with real_lock(target) as lease:
                    record_attempt(attempts, "acquired")
                    yield lease
            except fast_publication.PublicationLockBusy:
                record_attempt(attempts, "busy")
                record_attempt(attempts, "handshake")
                handshake.set()
                raise

        fast_publication.publication_lock = recording_lock
    engine, repo, _ = connect_publication(url, path, pricing)
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    barrier.wait(timeout=30)
    assert go.wait(timeout=30)
    revision = repo.register_inputs(METER, [item(amount="2")], [snapshot], 1)
    registered.set()
    results.put(("registered", revision))
    engine.dispose()


#: All entrypoints of the only executable gateway, read-only ones included.
RECEIVER_ENTRYPOINTS = ("enqueue", "deliver", "inspect", "revoke_and_stop", "bind_owner",
                         "send", "query", "advance_clock", "crash", "restart",
                         "restore_files", "snapshot_files", "lose_ack", "step_helper")


def observe_receiver_calls(monkeypatch, gateway):
    """Observe read-only calls as well as writes, keeping the real fake behavior."""
    calls = []

    def observe(name):
        real = getattr(gateway, name)

        def entry(*args, **kwargs):
            calls.append(name)
            return real(*args, **kwargs)

        return entry

    for name in RECEIVER_ENTRYPOINTS:
        monkeypatch.setattr(gateway, name, observe(name))
    return calls


def install_receiver_tripwire(path):
    """Make every receiver entrypoint fail and write it down.

    Unchanged receiver bytes are no proof: a read-only inspect or query leaves
    them untouched. This tripwire records every attempt, also the read-only
    ones, so the absence of the file proves zero receiver calls.
    """
    import dao.tests.prog.fakes.fenced_receiver as fake

    calls = []

    def guard(name):
        def entry(self, *args, **kwargs):
            calls.append(name)
            Path(path).write_text("\n".join(calls), encoding="utf-8")
            raise AssertionError(f"the registration touched the receiver: {name}")

        return entry

    for name in RECEIVER_ENTRYPOINTS:
        setattr(fake.FencedReceiver, name, guard(name))

    def build(self, *args, **kwargs):
        calls.append("__init__")
        Path(path).write_text("\n".join(calls), encoding="utf-8")
        raise AssertionError("the registration built a receiver")

    fake.FencedReceiver.__init__ = build


def join(process):
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("owner/registrator worker did not finish")


def test_01_correction_first_gives_no_send_and_revokes_the_epoch(system):
    barrier, go, registered, results = CTX.Barrier(2), CTX.Event(), CTX.Event(), CTX.Queue()
    owner = CTX.Process(target=owner_worker,
                         args=(str(system.root), "after", barrier, go, registered, results))
    registrar = CTX.Process(target=registrator_worker,
                             args=(system.url, str(system.path), system.pricing, barrier, go,
                                   registered, results))
    owner.start()
    registrar.start()
    go.set()
    for process in (owner, registrar):
        join(process)
        assert process.exitcode == 0
    report = {item[0]: item[1:] for item in (results.get(timeout=10) for _ in range(2))}
    assert report["registered"] == (2,)
    assert report["sections"] == ([None], 0, "released", "released")

    # The plan bytes are the durable invalidation; the book moved to revision two.
    assert read_publication(system.path).status == "blocked"
    assert system.repo.read_book_reference(METER).input_revision == 2
    # The receiver mirror: no command at all and an exactly accepted barrier.
    state = json.loads(receiver_state_file(system.root).read_text())
    assert state["commands"] == []
    assert state["epoch_high_water"] == 9
    receipt = state["last_barrier"]
    assert (receipt["status"], receipt["revoked_through_epoch"],
            receipt["current_epoch"]) == ("accepted", 8, 9)
    assert receipt["queues_fenced"] is True
    assert receipt["safe_state"] == "confirmed"
    # The owner mirror: the epoch is withdrawn, nothing is pending.
    mirror = read_mirror(system.root, installation_id=METER)
    assert (mirror["phase"], mirror["owner_epoch"], mirror["pending_command"]) == ("released", 8, None)
    assert mirror["pending_barrier_id"] is None


def test_02_send_first_then_invalidation_gives_no_new_output(system):
    barrier, go, registered, results = CTX.Barrier(2), CTX.Event(), CTX.Event(), CTX.Queue()
    owner = CTX.Process(target=owner_worker,
                         args=(str(system.root), "send-first", barrier, go, registered, results))
    registrar = CTX.Process(target=registrator_worker,
                             args=(system.url, str(system.path), system.pricing, barrier, go,
                                   registered, results))
    owner.start()
    registrar.start()
    first = results.get(timeout=30)
    assert first[0] == "first-section" and first[1] == "accepted"
    go.set()
    for process in (owner, registrar):
        join(process)
        assert process.exitcode == 0
    report = {item[0]: item[1:] for item in (results.get(timeout=10) for _ in range(2))}
    assert report["registered"] == (2,)
    # Exactly one bounded send happened; the later invalidation adds no new output.
    assert report["sections"] == (["accepted", None], 3, "released", "released")
    assert read_publication(system.path).status == "blocked"
    state = json.loads(receiver_state_file(system.root).read_text())
    assert len(state["commands"]) == 1
    assert state["commands"][0]["phase"] == "accepted"
    assert state["epoch_high_water"] == 9
    mirror = read_mirror(system.root, installation_id=METER)
    assert (mirror["phase"], mirror["pending_command"]) == ("released", None)
    assert mirror["last_receipt"]["status"] == "accepted"


def test_03_registration_never_touches_the_receiver(system):
    # A provisioned receiver whose files must stay byte identical.
    FencedReceiver(system.root / "receiver", initial_state=receiver_state(), utc_ms=NOW)
    before = receiver_state_file(system.root).read_bytes()
    tripwire = system.root / "receiver-calls.txt"
    barrier, go, registered, results = CTX.Barrier(1), CTX.Event(), CTX.Event(), CTX.Queue()
    registrar = CTX.Process(target=registrator_worker,
                             args=(system.url, str(system.path), system.pricing, barrier, go,
                                   registered, results, None, str(tripwire)))
    registrar.start()
    go.set()
    join(registrar)
    assert registrar.exitcode == 0
    assert results.get(timeout=10) == ("registered", 2)
    assert receiver_state_file(system.root).read_bytes() == before
    assert read_publication(system.path).status == "blocked"
    # Not only the bytes: there was not a single receiver call, read-only ones
    # included. The tripwire file only exists after such a call.
    assert not tripwire.exists()


def test_04_the_slot_is_not_released_between_check_and_send(system):
    barrier, go, registered, results = CTX.Barrier(2), CTX.Event(), CTX.Event(), CTX.Queue()
    handshake = CTX.Event()
    attempts = system.root / "acquisition-attempts.txt"
    send_started, send_continue = CTX.Event(), CTX.Event()
    owner = CTX.Process(target=owner_worker,
                         args=(str(system.root), "send-first", barrier, go, registered, results,
                               send_started, send_continue))
    registrar = CTX.Process(target=registrator_worker,
                             args=(system.url, str(system.path), system.pricing, barrier, go,
                                   registered, results, handshake, None, str(attempts)))
    owner.start()
    registrar.start()
    assert send_started.wait(timeout=30), "the owner never reached the send"
    # The publication slot is still held while the receiver call runs.
    with pytest.raises(PublicationLockBusy):
        with publication_lock(system.path):
            pass
    go.set()
    # The registration really tried to acquire the slot while the owner held
    # it: the handshake is written before the owner is allowed to release.
    assert handshake.wait(timeout=30), "the registration never tried to acquire the slot"
    # The handshake follows that real attempt; it never precedes it.
    events = attempts.read_text(encoding="utf-8").split()
    assert "busy" in events, "the registration never made a real acquisition attempt"
    assert events.index("busy") < events.index("handshake"), (
        "the handshake fired before the real acquisition attempt")
    send_continue.set()
    for process in (owner, registrar):
        join(process)
        assert process.exitcode == 0
    report = {item[0]: item[1:] for item in (results.get(timeout=10) for _ in range(3))}
    assert report["sections"][1] == 3
    assert read_publication(system.path).status == "blocked"


# -- the bounded acquisition ------------------------------------------------

def test_05_the_body_runs_exactly_once_after_several_busy_attempts(tmp_path, monkeypatch):
    module = attempts()
    path = tmp_path / "fast_plan.json"
    path.write_text("{}")
    holder_ready, release = threading.Event(), threading.Event()
    real_lock = module.publication_lock

    def holder():
        with real_lock(path):
            holder_ready.set()
            assert release.wait(timeout=30)

    thread = threading.Thread(target=holder)
    thread.start()
    assert holder_ready.wait(timeout=30)
    seen = []

    def counting(target):
        seen.append(target)
        if len(seen) == 3:
            release.set()
        return real_lock(target)

    monkeypatch.setattr(module, "publication_lock", counting)
    bodies = []
    with module.publication_attempts(path, deadline_ns=module.clock_gettime_ns(
            module.CLOCK_BOOTTIME) + 6_000_000_000) as lease:
        bodies.append(lease.path)
    thread.join(timeout=30)
    assert bodies == [path]
    assert len(seen) >= 3


def test_05_the_deadline_refuses_without_mutation(tmp_path):
    module = attempts()
    path = tmp_path / "fast_plan.json"
    path.write_text("{}")
    with module.publication_lock(path):
        before = path.read_bytes()
        with pytest.raises(PublicationLockBusy):
            with module.publication_attempts(path, deadline_ns=module.clock_gettime_ns(
                    module.CLOCK_BOOTTIME) + 150_000_000):
                pass
        assert path.read_bytes() == before


def test_05_a_foreign_error_is_never_retried(tmp_path, monkeypatch):
    module = attempts()
    path = tmp_path / "fast_plan.json"
    path.write_text("{}")
    seen = []

    def failing(target):
        seen.append(target)
        raise PublicationError("synthetic failure")

    monkeypatch.setattr(module, "publication_lock", failing)
    with pytest.raises(PublicationError):
        with module.publication_attempts(path, deadline_ns=module.clock_gettime_ns(
                module.CLOCK_BOOTTIME) + 6_000_000_000):
            pass
    assert len(seen) == 1


def test_05_a_nested_call_is_busy_and_never_deadlocks(tmp_path):
    module = attempts()
    path = tmp_path / "fast_plan.json"
    path.write_text("{}")
    with module.publication_lock(path):
        with pytest.raises(PublicationLockBusy):
            with module.publication_attempts(path, deadline_ns=module.clock_gettime_ns(
                    module.CLOCK_BOOTTIME) + 150_000_000):
                pass


def test_05_the_deadline_that_expires_during_the_acquisition_refuses(tmp_path, monkeypatch):
    """De deadline wordt ná de acquisitie opnieuw getoetst; de body loopt niet."""
    module = attempts()
    path = tmp_path / "fast_plan.json"
    path.write_text("{}")
    real_clock = module.clock_gettime_ns
    telling = {"n": 0}

    def clock(*_arguments):
        telling["n"] += 1
        # De eerste lezing is de acquisitiecontrole; de tweede komt ná de lease.
        if telling["n"] == 1:
            return real_clock(module.CLOCK_BOOTTIME)
        return real_clock(module.CLOCK_BOOTTIME) + 10_000_000_000

    monkeypatch.setattr(module, "clock_gettime_ns", clock)
    bodies = []
    with pytest.raises(PublicationLockBusy):
        with module.publication_attempts(path, deadline_ns=real_clock(
                module.CLOCK_BOOTTIME) + 1_000_000_000) as lease:
            bodies.append(lease.path)
    assert bodies == []
    # De lease is weer gesloten: het slot is vrij en er is geen mutatie.
    with module.publication_lock(path):
        pass
    assert path.read_bytes() == b"{}"


# -- the final validation ---------------------------------------------------

def test_06_validate_current_locked_accepts_the_current_artifact(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    try:
        with publication_lock(artifact.path) as lease:
            plan = module.validate_current_locked(lease=lease, expected_digest=module.plan_digest(
                artifact.plan), clock=CLOCK, policy=POLICY)
        assert plan.created_ts == artifact.plan.created_ts
    finally:
        layout.lease.close()


def test_06_the_clock_is_compared_in_milliseconds(artifact):
    """De klok is milliseconden; de plantijdlijn is seconden: één eenheid."""
    module = execution()
    layout = owner_layout(artifact.root)
    plan = artifact.plan
    context = plan.financial_context
    digest = module.plan_digest(plan)
    try:
        # Een echte millisecondeklok binnen het venster: het plan is geldig.
        with publication_lock(artifact.path) as lease:
            accepted = module.validate_current_locked(
                lease=lease, expected_digest=digest,
                clock=replace(CLOCK, utc_ms=(context.not_after_ts - 1) * 1000), policy=POLICY)
        assert accepted.created_ts == plan.created_ts
        # De leasegrens is exact: op not_after is de lease op.
        with publication_lock(artifact.path) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.validate_current_locked(
                    lease=lease, expected_digest=digest,
                    clock=replace(CLOCK, utc_ms=context.not_after_ts * 1000), policy=POLICY)
        assert error.value.reason == "plan_lease_expired"
        # Eén milliseconde vóór de aanmaak is nog steeds te vroeg.
        with publication_lock(artifact.path) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.validate_current_locked(
                    lease=lease, expected_digest=digest,
                    clock=replace(CLOCK, utc_ms=plan.created_ts * 1000 - 1), policy=POLICY)
        assert error.value.reason == "clock_before_creation"
    finally:
        layout.lease.close()


@pytest.mark.parametrize("fault,reason", [
    ("digest", "plan_digest_mismatch"),
    ("lease", "plan_lease_expired"),
    ("clock", "clock_unhealthy"),
    ("creation", "clock_before_creation"),
    ("blocked", "publication_blocked"),
    ("missing", "publication_missing"),
    ("invalid", "publication_invalid"),
    ("legacy", "financial_plan_required"),
    ("gate", "gate_not_open"),
])
def test_06_validate_current_locked_refuses_with_a_machine_readable_reason(artifact, fault, reason):
    module = execution()
    layout = owner_layout(artifact.root)
    path = artifact.path
    digest = module.plan_digest(artifact.plan)
    clock = CLOCK
    policy = POLICY
    if fault == "digest":
        digest = "b" * 64
    elif fault == "lease":
        # De leasegrens ligt in seconden; de klok in milliseconden.
        clock = replace(CLOCK, utc_ms=artifact.plan.financial_context.not_after_ts * 1000)
    elif fault == "clock":
        clock = replace(CLOCK, healthy=False)
    elif fault == "creation":
        clock = replace(CLOCK, utc_ms=artifact.plan.created_ts * 1000 - 1)
    elif fault == "blocked":
        path.write_text(json.dumps({"format_version": 2, "status": "blocked",
                                    "created_ts": NOW_TS,
                                    "reason": "registered_financial_inputs_changed"}))
    elif fault == "missing":
        path.unlink()
    elif fault == "invalid":
        path.write_text("not-json")
    elif fault == "legacy":
        from dao.prog.fastctrl.plan import FastPlan
        path.write_text(json.dumps(FastPlan(NOW_TS, 900).to_dict()))
    else:
        policy = GatePolicy(NOW_TS + 10_000)
    try:
        with publication_lock(path) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.validate_current_locked(lease=lease, expected_digest=digest,
                                               clock=clock, policy=policy)
        assert error.value.reason == reason
        assert error.value.reason[0].islower()
    finally:
        layout.lease.close()


@pytest.mark.parametrize("fault,reason", [
    ("meter", "meter_identity_missing"),
    ("revision", "input_revision_invalid"),
    ("capture", "clock_before_capture"),
    ("context", "missing_financial_context"),
])
def test_06_a_deviation_the_wire_format_cannot_express_is_refused(artifact, fault, reason,
                                                                  monkeypatch):
    module = execution()
    layout = owner_layout(artifact.root)
    digest = module.plan_digest(artifact.plan)
    doctored = artifact.plan
    basis = doctored.financial_context.basis
    if fault == "meter":
        object.__setattr__(basis, "meter_id", "")
    elif fault == "revision":
        object.__setattr__(basis, "input_revision", 0)
    elif fault == "capture":
        object.__setattr__(basis, "captured_ts", NOW_TS + 1000)
    else:
        object.__setattr__(doctored, "financial_context", None)
    monkeypatch.setattr(module, "read_publication", lambda path: PlanRead("ready", doctored, "ready"))
    try:
        with publication_lock(artifact.path) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.validate_current_locked(lease=lease, expected_digest=digest,
                                               clock=CLOCK, policy=POLICY)
        assert error.value.reason == reason
    finally:
        layout.lease.close()


def test_06_a_changed_plan_is_refused_even_when_it_is_still_valid(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    other = replace(artifact.plan, price_average=artifact.plan.price_average + 0.5)
    artifact.path.write_text(json.dumps(other.to_dict()))
    try:
        with publication_lock(artifact.path) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.validate_current_locked(lease=lease, expected_digest=module.plan_digest(
                    artifact.plan), clock=CLOCK, policy=POLICY)
        assert error.value.reason == "plan_digest_mismatch"
    finally:
        layout.lease.close()


# -- the bounded send --------------------------------------------------------

@pytest.mark.parametrize("boundary,publication", [
    ("owner_section", "ready"), ("owner_section", "blocked"), ("send_locked", "ready"),
])
@pytest.mark.parametrize("adapter,on_owner", [
    ("duck", False), ("duck", True),
    ("labelled_duck", False), ("labelled_duck", True),
    ("generic_ha", False), ("generic_ha", True),
    ("foreign", False), ("foreign", True), ("other_fenced", False),
])
def test_i6_direct_boundary_refuses_duck_receiver(
        artifact, monkeypatch, boundary, publication, adapter, on_owner):
    """Neither duck methods nor equal receiver IDs grant mechanical command rights."""
    from dao.prog.fastctrl.runner import HomeAssistantGateway
    from dao.tests.prog.test_fast_runner import FakeHass

    module = execution()
    layout = owner_layout(artifact.root)
    try:
        command = None
        if boundary == "send_locked":
            command = build_command(module, layout, artifact)
            layout.owner.record_pending_command(command)
        receiver_files = [layout.gateway.state_path, layout.gateway.anchor_path]
        if adapter in ("duck", "labelled_duck"):
            offered = SimpleNamespace(**{name: getattr(layout.gateway, name)
                                         for name in RECEIVER_ENTRYPOINTS})
            if adapter == "labelled_duck":
                offered.actuation_adapter_class = "fenced"
        elif adapter == "generic_ha":
            offered = HomeAssistantGateway(FakeHass())
        elif adapter == "foreign":
            offered = object()
        else:
            offered = FencedReceiver(artifact.root / "other-receiver",
                                      initial_state=layout.gateway.state, utc_ms=NOW)
            assert offered is not layout.owner.receiver
            assert offered.state.receiver_id == layout.binding.receiver_id
            assert offered.state.receiver_generation == layout.binding.receiver_generation
            receiver_files.extend((offered.state_path, offered.anchor_path))
        if on_owner:
            layout.owner.receiver = offered
        if publication == "blocked":
            artifact.path.write_text(json.dumps({"format_version": 2, "status": "blocked",
                                                 "created_ts": NOW_TS, "reason": "correction"}))
        before = {path: path.read_bytes()
                  for path in (artifact.path, layout.store.path, *receiver_files)}
        state_before = layout.owner.state
        calls = []

        def tripwire(name):
            def forbidden(*args, **kwargs):
                calls.append(name)
                raise AssertionError(f"receiver refusal came after a side effect: {name}")
            return forbidden

        for label, target in (("bound", layout.gateway), ("offered", offered)):
            for name in (*RECEIVER_ENTRYPOINTS, "read", "write_number", "select_option",
                         "set_datetime", "publish_state", "switch"):
                if hasattr(target, name):
                    monkeypatch.setattr(target, name, tripwire(f"{label}.{name}"))
        for name in ("reserve_sequence", "record_pending_command", "record_receipt",
                     "stop", "report_failure", "_write"):
            monkeypatch.setattr(layout.owner, name, tripwire(f"owner.{name}"))
        monkeypatch.setattr(layout.store, "write", tripwire("store.write"))
        failure = None
        with publication_lock(artifact.path) as lease:
            try:
                if boundary == "send_locked":
                    module.send_locked(command, lease=lease, owner=layout.owner,
                                       receiver=offered, deadline=DEADLINE, clock=CLOCK, policy=POLICY)
                else:
                    module.owner_section(lease=lease, owner=layout.owner, receiver=offered,
                                         request=request(), deadline=DEADLINE, clock=CLOCK, policy=POLICY)
            except Exception as exc:
                failure = exc
        assert isinstance(failure, module.ExecutionRefused), (failure, calls)
        expected = "receiver_binding_mismatch" if adapter == "other_fenced" else "adapter_unqualified"
        assert failure.reason == expected, calls
        assert calls == []
        assert {path: path.read_bytes() for path in before} == before
        assert layout.owner.state is state_before
        assert layout.owner.binding is layout.binding
        assert layout.owner.phase == "owned" and layout.owner.recovery_required is False
        assert layout.owner._sequence_high_water == (41 if command is not None else 40)
        assert layout.owner.state.pending_command == command
        assert layout.store.load_strict().pending_command == command
        assert layout.gateway.effect_log == []
        if adapter == "other_fenced":
            assert offered.effect_log == []
    finally:
        layout.lease.close()


@pytest.mark.parametrize("boundary", ["owner_section", "send_locked"])
def test_i6_direct_boundary_accepts_own_nominal_receiver(artifact, monkeypatch, boundary):
    module = execution()
    layout = owner_layout(artifact.root)
    try:
        command = None
        if boundary == "send_locked":
            command = build_command(module, layout, artifact)
            layout.owner.record_pending_command(command)
        calls = observe_receiver_calls(monkeypatch, layout.gateway)
        with final_section(layout.owner) as lease:
            if boundary == "send_locked":
                receipt = module.send_locked(command, lease=lease, owner=layout.owner,
                    receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK, policy=POLICY)
            else:
                receipt = module.owner_section(lease=lease, owner=layout.owner,
                    receiver=layout.gateway, request=request(), deadline=DEADLINE, clock=CLOCK, policy=POLICY)
        assert receipt.status == "accepted" and receipt.sequence == 41
        assert calls.count("send") == 1 and "inspect" not in calls
        assert layout.gateway.effect_log == [(receipt.command_id, part, "accu", -1200)
                                            for part in ("setpoint", "mode", "stop_inverter")]
        mirror = layout.store.load_strict()
        assert mirror.pending_command is None and mirror.last_receipt == receipt
        assert mirror.last_reserved_sequence == 41
    finally:
        layout.lease.close()


@pytest.mark.parametrize("fault", ["none", "plan", "binding", "state", "store", "command",
                                    "missing_binding", "missing_state"])
def test_i2_direct_send_rechecks_installation(artifact, monkeypatch, fault):
    """A valid reservation grants no send right after any identity changes."""
    module = execution()
    layout = owner_layout(artifact.root)
    try:
        command = build_command(module, layout, artifact)
        layout.owner.record_pending_command(command)
        before = layout.store.path.read_bytes()
        offered = command
        if fault == "plan":
            context = artifact.plan.financial_context
            other = replace(artifact.plan, financial_context=replace(
                context, basis=replace(context.basis, meter_id="other-installation")))
            artifact.path.write_text(json.dumps(other.to_dict()))
            offered = replace(command, plan_digest=module.plan_digest(other))
        elif fault in ("binding", "state"):
            current = getattr(layout.owner, fault)
            setattr(layout.owner, fault, replace(current, installation_id="other-installation"))
        elif fault == "store":
            layout.store.installation_id = "other-installation"
        elif fault == "command":
            offered = replace(command, installation_id="other-installation")
        elif fault.startswith("missing_"):
            setattr(layout.owner, fault.removeprefix("missing_"), None)
        state_before = layout.owner.state
        calls = observe_receiver_calls(monkeypatch, layout.gateway)
        error = None
        with final_section(layout.owner) as lease:
            # Even the changed plan is valid with the offered digest: this is
            # an installation refusal, not a stale-publication refusal.
            module.validate_current_locked(lease=lease, expected_digest=offered.plan_digest,
                                           clock=CLOCK, policy=POLICY)
            try:
                receipt = module.send_locked(offered, lease=lease, owner=layout.owner,
                                             receiver=layout.gateway, deadline=DEADLINE,
                                             clock=CLOCK, policy=POLICY)
            except Exception as exc:
                error = exc
        if fault == "none":
            assert error is None
            assert receipt.status == "accepted"
            assert receipt.installation_id == INSTALLATION
            assert calls.count("send") == 1 and "inspect" not in calls
            assert layout.store.load_strict().pending_command is None
            assert layout.store.load_strict().last_reserved_sequence == 41
            return
        assert isinstance(error, module.ExecutionRefused), error
        expected = {"missing_binding": "owner_binding_missing",
                    "missing_state": "owner_state_missing"}.get(fault, "installation_mismatch")
        assert error.reason == expected
        assert calls == []
        assert layout.gateway.effect_log == []
        assert layout.store.path.read_bytes() == before
        assert layout.owner.state == state_before
        assert read_mirror(artifact.root)["pending_command"] == command.to_dict()
        assert layout.owner._sequence_high_water == 41
    finally:
        layout.lease.close()


def test_07_send_locked_refuses_a_changed_pending_command(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    changed = replace(command, setpoint_w=command.setpoint_w + 1)
    try:
        with final_section(layout.owner) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.send_locked(changed, lease=lease, owner=layout.owner,
                                   receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK,
                                   policy=POLICY)
        assert error.value.reason == "pending_command_mismatch"
        assert not layout.gateway.effect_log
    finally:
        layout.lease.close()


def test_07_send_locked_refuses_a_missing_binding(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    layout.owner.binding = None
    try:
        with final_section(layout.owner) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.send_locked(command, lease=lease, owner=layout.owner,
                                   receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK,
                                   policy=POLICY)
        assert error.value.reason == "owner_binding_missing"
        assert not layout.gateway.effect_log
    finally:
        layout.lease.close()


def test_07_send_locked_refuses_an_expired_lifetime_lease(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    with final_section(layout.owner) as lease:
        layout.lease.close()
        with pytest.raises(module.ExecutionRefused) as error:
            module.send_locked(command, lease=lease, owner=layout.owner,
                               receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK,
                               policy=POLICY)
    assert error.value.reason == "owner_lease_missing"
    assert not layout.gateway.effect_log


def test_07_a_lost_ack_is_unconfirmed_with_one_retry_and_one_query(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    calls = []
    real_send, real_query = layout.gateway.send, layout.gateway.query

    def send(payload, *, deadline):
        calls.append(("send", payload.command_id, payload.sequence, deadline.monotonic_end_ns))
        return real_send(payload, deadline=deadline)

    def query(**kwargs):
        calls.append(("query", kwargs["command_id"], kwargs["sequence"],
                      kwargs["deadline"].monotonic_end_ns))
        return real_query(**kwargs)

    layout.gateway.send = send
    layout.gateway.query = query
    layout.gateway.lose_ack()
    try:
        with final_section(layout.owner) as lease:
            receipt = module.send_locked(command, lease=lease, owner=layout.owner,
                                        receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK,
                                        policy=POLICY)
        assert receipt.status == "unconfirmed"
        assert [call[0] for call in calls] == ["send", "send", "query"]
        # The retry carries exactly the same bytes and no new budget.
        assert calls[0][1] == calls[1][1] == calls[2][1]
        assert calls[0][2] == calls[1][2] == calls[2][2]
        assert calls[0][3] == calls[1][3] == calls[2][3]
        # The lost acknowledgement leaves one honest effect; the retry with the
        # same bytes never produces a second one.
        assert len(layout.gateway.effect_log) == 3
        assert len({entry[0] for entry in layout.gateway.effect_log}) == 1
        # The uncertain order stays visible in the mirror.
        mirror = read_mirror(artifact.root)
        assert mirror["pending_command"]["command_id"] == command.command_id
        assert mirror["last_receipt"]["status"] == "unconfirmed"
    finally:
        layout.lease.close()


def test_07_an_accepted_send_clears_the_pending_command(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    try:
        with final_section(layout.owner) as lease:
            receipt = module.send_locked(command, lease=lease, owner=layout.owner,
                                        receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK,
                                        policy=POLICY)
        assert receipt.status == "accepted"
        assert len(receipt.helper_results) == 3
        assert [entry[1] for entry in layout.gateway.effect_log] == [
            "setpoint", "mode", "stop_inverter"]
        mirror = read_mirror(artifact.root)
        assert mirror["pending_command"] is None
        assert mirror["last_receipt"]["status"] == "accepted"
    finally:
        layout.lease.close()


@pytest.mark.parametrize("status", ["accepted", "rejected"])
def test_i1_exact_query_proves_the_terminal_outcome(artifact, status):
    """A live reservation plus a duplicate needs the exact terminal query proof."""
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    calls = []
    real_send, real_query = layout.gateway.send, layout.gateway.query

    def send(payload, *, deadline):
        calls.append("send")
        if len(calls) == 1:
            layout.gateway.enqueue(payload)
            return layout.gateway.deliver()
        return real_send(payload, deadline=deadline)

    def query(**kwargs):
        calls.append("query")
        if status == "accepted":
            for part in ("setpoint", "mode", "stop_inverter"):
                layout.gateway.step_helper(part, command=command)
        else:
            layout.gateway.helper_ok["setpoint"] = False
            layout.gateway.step_helper("setpoint", command=command)
        return real_query(**kwargs)

    layout.gateway.send, layout.gateway.query = send, query
    try:
        with final_section(layout.owner) as lease:
            receipt = module.send_locked(command, lease=lease, owner=layout.owner,
                                        receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK,
                                        policy=POLICY)
        assert calls == ["send", "send", "query"]
        # A no-effect query cannot erase the earlier live-reservation observation.
        expected = "accepted" if status == "accepted" else "unconfirmed"
        assert receipt.status == expected
        state = layout.store.load_strict()
        assert state.last_receipt == receipt
        assert state.pending_command == (None if status == "accepted" else command)
        assert len(layout.gateway.effect_log) == (3 if status == "accepted" else 0)
    finally:
        layout.lease.close()


def test_i1_exact_rejection_without_effect_clears_pending(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    layout.gateway.helper_ok["setpoint"] = False
    try:
        with final_section(layout.owner) as lease:
            receipt = module.send_locked(command, lease=lease, owner=layout.owner,
                                        receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK,
                                        policy=POLICY)
        assert (receipt.status, receipt.effect) == ("rejected", "none")
        assert layout.store.load_strict().pending_command is None
        assert layout.store.load_strict().last_receipt == receipt
        assert layout.gateway.effect_log == []
    finally:
        layout.lease.close()


def observed_receipt(status, effect, reason):
    from dao.prog.fastctrl.command import ActuationReceipt

    return ActuationReceipt(installation_id="sandbox", battery_name="accu", owner_epoch=8,
                            sequence=41, command_id="cmd", plan_digest="a" * 64,
                            accepted_ts=NOW if status == "accepted" else None,
                            status=status, helper_results=(),
                            owner_session_id="session-new", payload_digest="b" * 64,
                            reason=reason, observed_at_ms=NOW, effect=effect)


@pytest.mark.parametrize("effect", ["unknown", "partial"])
@pytest.mark.parametrize("order", ["uncertain_first", "rejection_first"])
def test_07_a_mixed_observation_is_never_reported_as_rejected(effect, order):
    """Eén waarneming met effect maakt de uitkomst onzeker, in elke volgorde."""
    module = execution()
    uncertain = observed_receipt("unconfirmed", effect, "effect_unknown")
    refusal = observed_receipt("rejected", "none", "stale_or_duplicate_sequence")
    observations = ([uncertain, refusal] if order == "uncertain_first"
                    else [refusal, uncertain])
    outcome = module._conservative(observations)
    assert outcome.status == "unconfirmed"
    assert outcome.effect == effect


def test_07_a_proven_empty_effect_is_still_rejected():
    module = execution()
    refusal = observed_receipt("rejected", "none", "stale_or_duplicate_sequence")
    empty = observed_receipt("unconfirmed", "none", "no_effect")
    assert module._conservative([refusal]).status == "rejected"
    assert module._conservative([refusal, empty]).status == "rejected"


def test_07_a_late_acknowledgement_stays_history_and_refuses(artifact):
    """Een late ACK is bewijs, geen rechten: historie, geen gewiste pending."""
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    real_send = layout.gateway.send

    def late(payload, *, deadline):
        receipt = real_send(payload, deadline=deadline)
        layout.gateway.advance_clock(2000)
        return receipt

    layout.gateway.send = late
    try:
        with final_section(layout.owner) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.send_locked(command, lease=lease, owner=layout.owner,
                                   receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK,
                                   policy=POLICY)
        assert error.value.reason == "deadline_exceeded"
        mirror = read_mirror(artifact.root)
        # Het ontvangerbewijs blijft als historie staan ...
        assert mirror["last_receipt"]["status"] == "accepted"
        # ... en het pending command blijft staan: geen geaccepteerde commandoupdate.
        assert mirror["pending_command"]["command_id"] == command.command_id
    finally:
        layout.lease.close()


# -- the budget --------------------------------------------------------------

@pytest.mark.parametrize("route", ["direct", "section"])
@pytest.mark.parametrize("finish", ["ack", "write"])
@pytest.mark.parametrize("epsilon", [0, 20])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_i5_short_plan_lease_bounds_ack_deadline(artifact, route, finish, epsilon, offset):
    """Timely real effects do not turn an ACK/write at the lease edge into rights."""
    module = execution()
    artifact.plan = replace(artifact.plan, financial_context=replace(
        artifact.plan.financial_context, not_after_ts=NOW_TS + 1))
    artifact.path.write_text(json.dumps(artifact.plan.to_dict()))
    layout = owner_layout(artifact.root)
    gateway = layout.gateway
    gateway.advance_clock(900)
    layout.owner.now_ns = lambda: MONO + gateway.monotonic_ns
    clock = replace(CLOCK, utc_ms=NOW + 900, boottime_ns=MONO + 900_000_000,
                    uncertainty_ms=epsilon)
    caller = Deadline(clock.boottime_ns + 10_000_000_000, NOW + 60_000)
    commands, deadlines, original = [], [], []
    real_send, real_record = gateway.send, layout.owner.record_receipt

    def send(command, *, deadline):
        commands.append(command)
        deadlines.append(deadline)
        receipt = real_send(command, deadline=deadline)
        original.append(receipt)
        if finish == "ack":
            gateway.advance_clock(100 - epsilon + offset)
        return receipt

    def record(receipt, **kwargs):
        real_record(receipt, **kwargs)
        if finish == "write":
            gateway.advance_clock(100 - epsilon + offset)

    gateway.send, layout.owner.record_receipt = send, record
    try:
        failure, receipt = None, None
        with final_section(layout.owner) as lease:
            try:
                if route == "direct":
                    command = replace(build_command(module, layout, artifact), issued_at_ms=clock.utc_ms)
                    layout.owner.record_pending_command(command)
                    receipt = module.send_locked(command, lease=lease, owner=layout.owner,
                        receiver=gateway, deadline=caller, clock=clock, policy=POLICY)
                else:
                    receipt = module.owner_section(lease=lease, owner=layout.owner,
                        receiver=gateway, request=request(), deadline=caller, clock=clock, policy=POLICY)
            except module.ExecutionRefused as exc:
                failure = exc.reason
        assert failure == (None if offset < 0 else "deadline_exceeded")
        assert len(commands) == len(original) == 1
        command = commands[0]
        assert original[0].status == "accepted"
        assert original[0].accepted_ts == NOW + 900
        assert gateway.effect_log == [(command.command_id, part, "accu", -1200)
                                     for part in ("setpoint", "mode", "stop_inverter")]
        assert deadlines == [Deadline(MONO + (1000 - epsilon) * 1_000_000, NOW + 1000)]
        state = layout.store.load_strict()
        assert state.last_receipt == original[0]
        assert state.pending_command == (None if offset < 0 else command)
        assert receipt == (original[0] if offset < 0 else None)
    finally:
        layout.lease.close()


@pytest.mark.parametrize("bound", ["owner", "caller_wire", "caller_monotone", "command"])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_i5_direct_send_honors_each_short_bound(artifact, bound, offset):
    module = execution()
    layout = owner_layout(artifact.root, requested_not_after_ms=NOW + 100 if bound == "owner" else None)
    layout.owner.now_ns = lambda: MONO + layout.gateway.monotonic_ns
    command = build_command(module, layout, artifact)
    if bound == "command":
        command = replace(command, execute_before_ms=NOW + 100, hold_until_ms=NOW + 100)
    layout.owner.record_pending_command(command)
    caller = Deadline(MONO + (100_000_000 if bound == "caller_monotone" else 10_000_000_000),
                      NOW + (100 if bound == "caller_wire" else 60_000))
    real_send = layout.gateway.send
    seen = []

    def send(payload, *, deadline):
        seen.append(deadline)
        receipt = real_send(payload, deadline=deadline)
        layout.gateway.advance_clock(100 + offset)
        return receipt

    layout.gateway.send = send
    try:
        failure = None
        with final_section(layout.owner) as lease:
            try:
                receipt = module.send_locked(command, lease=lease, owner=layout.owner,
                    receiver=layout.gateway, deadline=caller, clock=CLOCK, policy=POLICY)
            except module.ExecutionRefused as exc:
                failure = exc.reason
        assert failure == (None if offset < 0 else "deadline_exceeded")
        assert seen[0].monotonic_end_ns == MONO + 100_000_000
        assert len(layout.gateway.effect_log) == 3
        assert layout.store.load_strict().last_receipt.status == "accepted"
        assert layout.store.load_strict().pending_command == (None if offset < 0 else command)
        if offset < 0:
            assert receipt.status == "accepted"
    finally:
        layout.lease.close()


@pytest.mark.parametrize("route", ["direct", "section"])
@pytest.mark.parametrize("finish,elapsed,late", [
    ("ack", 749, False), ("ack", 750, True), ("ack", 751, True),
    ("write", 800, False), ("write", 999, False),
    ("write", 1000, True), ("write", 1001, True),
])
def test_i5_transport_and_result_write_keep_their_original_budgets(artifact, route, finish, elapsed, late):
    module = execution()
    layout = owner_layout(artifact.root)
    layout.owner.now_ns = lambda: MONO + layout.gateway.monotonic_ns
    real_send, real_record = layout.gateway.send, layout.owner.record_receipt
    commands = []

    def send(command, *, deadline):
        commands.append(command)
        receipt = real_send(command, deadline=deadline)
        if finish == "ack":
            layout.gateway.advance_clock(elapsed)
        return receipt

    def record(receipt, **kwargs):
        real_record(receipt, **kwargs)
        if finish == "write":
            layout.gateway.advance_clock(elapsed)

    layout.gateway.send, layout.owner.record_receipt = send, record
    caller = Deadline(MONO + 10_000_000_000, NOW + 60_000)
    try:
        failure = None
        with final_section(layout.owner) as lease:
            try:
                if route == "direct":
                    command = build_command(module, layout, artifact)
                    layout.owner.record_pending_command(command)
                    module.send_locked(command, lease=lease, owner=layout.owner,
                        receiver=layout.gateway, deadline=caller, clock=CLOCK, policy=POLICY)
                else:
                    module.owner_section(lease=lease, owner=layout.owner,
                        receiver=layout.gateway, request=request(), deadline=caller, clock=CLOCK, policy=POLICY)
            except module.ExecutionRefused as exc:
                failure = exc.reason
        assert failure == ("deadline_exceeded" if late else None)
        assert len(commands) == 1 and len(layout.gateway.effect_log) == 3
        assert layout.store.load_strict().last_receipt.status == "accepted"
        assert layout.store.load_strict().pending_command == (commands[0] if late else None)
    finally:
        layout.lease.close()


@pytest.mark.parametrize("epsilon,healthy,reason", [
    (99, True, None), (100, True, "budget_exhausted"),
    (101, True, "budget_exhausted"), (10_000, True, "budget_exhausted"),
    (0, False, "clock_unhealthy"),
])
@pytest.mark.parametrize("route", ["direct", "section"])
def test_i5_uncertainty_can_exhaust_the_lease_before_io(artifact, epsilon, healthy, reason, route):
    module = execution()
    layout = owner_layout(artifact.root, requested_not_after_ms=NOW + 100)
    clock = replace(CLOCK, uncertainty_ms=epsilon, healthy=healthy)
    try:
        failure = None
        with final_section(layout.owner) as lease:
            try:
                if route == "direct":
                    command = build_command(module, layout, artifact)
                    layout.owner.record_pending_command(command)
                    module.send_locked(command, lease=lease, owner=layout.owner,
                        receiver=layout.gateway, deadline=DEADLINE, clock=clock, policy=POLICY)
                else:
                    module.owner_section(lease=lease, owner=layout.owner,
                        receiver=layout.gateway, request=request(), deadline=DEADLINE,
                        clock=clock, policy=POLICY)
            except Exception as exc:
                failure = exc
        assert (None if failure is None else type(failure)) == (None if reason is None else module.ExecutionRefused)
        assert (None if failure is None else failure.reason) == reason
        assert len(layout.gateway.effect_log) == (3 if reason is None else 0)
        if route == "section" and reason is not None:
            assert layout.store.load_strict().last_reserved_sequence == 40
    finally:
        layout.lease.close()


@pytest.mark.parametrize("edge", ["plan", "owner", "active", "stop", "interval"])
@pytest.mark.parametrize("field", ["hold", "execute"])
def test_i5_direct_send_rejects_overwide_command_without_rewriting(artifact, monkeypatch, edge, field):
    module = execution()
    plan = artifact.plan
    if edge == "plan":
        plan = replace(plan, financial_context=replace(plan.financial_context, not_after_ts=NOW_TS + 1))
        bound = NOW + 1000
    elif edge in ("active", "stop"):
        first = plan.intervals[0]
        step = replace(first.battery(0), **{f"{'active_from' if edge == 'active' else 'stop'}_ts": NOW_TS + 1})
        plan = replace(plan, intervals=[replace(first, batteries=[step]), *plan.intervals[1:]])
        bound = NOW + 1000
    elif edge == "interval":
        plan = replace(plan, financial_context=replace(plan.financial_context, not_after_ts=NOW_TS + 1800))
        bound = NOW + 900_000
    else:
        bound = NOW + 100
    artifact.plan = plan
    artifact.path.write_text(json.dumps(plan.to_dict()))
    layout = owner_layout(artifact.root, requested_not_after_ms=bound if edge == "owner" else None)
    # Profile tests need an owner that really outlives the tested profile edge.
    if edge == "interval":
        layout.lease.close()
        layout.gateway.receiver_max_lease_ms = 3_600_000
        layout.lease = task_execution.acquire_resource(RESOURCE, lease_dir=artifact.root / "leases")
        layout.owner._bind_lifetime(layout.lease)
        next_owner_cycle(layout.owner)
        layout.binding = layout.owner.start(writer=layout.lease, receiver=layout.gateway,
            installation_id=INSTALLATION, owner_session_id="i5-long", requested_not_after_ms=NOW + 3_600_000,
            deadline=Deadline(layout.owner.now_ns() + 1_000_000_000, NOW + 60_000), now_ms=NOW)
        next_owner_cycle(layout.owner)
    try:
        command = replace(build_command(module, layout, artifact),
                          execute_before_ms=bound + (1 if field == "execute" else 0),
                          hold_until_ms=bound + 1)
        layout.owner.record_pending_command(command)
        before = layout.store.path.read_bytes()
        calls = observe_receiver_calls(monkeypatch, layout.gateway)
        failure = None
        with final_section(layout.owner) as lease:
            try:
                module.send_locked(command, lease=lease, owner=layout.owner,
                    receiver=layout.gateway, deadline=DEADLINE, clock=CLOCK, policy=POLICY)
            except module.ExecutionRefused as exc:
                failure = exc.reason
        assert failure == "command_window_invalid"
        assert calls == []
        assert layout.store.path.read_bytes() == before
        assert layout.owner.state.pending_command == command
        assert layout.gateway.effect_log == []
    finally:
        layout.lease.close()


@pytest.mark.parametrize("edge", ["active", "interval"])
@pytest.mark.parametrize("issued_offset", [-101, -100, -1, 0, 1])
def test_i5_fix1_direct_send_refuses_future_issue_at_profile_edge(tmp_path, monkeypatch, edge, issued_offset):
    """A future issue cannot select a later segment while execution happens now."""
    from dao.tests.prog.test_fast_consumer import build, financial_plan

    module = execution()
    plan = financial_plan(powers=(0.0,))
    boundary_ms = NOW + 900_000
    if edge == "active":
        first = plan.intervals[0]
        step = replace(first.battery(0), active_from_ts=NOW_TS + 1)
        plan = replace(plan, intervals=[replace(first, batteries=[step]), *plan.intervals[1:]])
        boundary_ms = NOW + 1000
    harness = build(tmp_path, plan=plan)
    harness.gateway.advance_clock(boundary_ms - 100 - NOW)
    if edge == "interval":
        next_owner_cycle(harness.owner)
    clock = replace(CLOCK, utc_ms=boundary_ms - 100, boottime_ns=harness.owner.now_ns())
    try:
        command = module._build_command(owner=harness.owner, request=request(), plan=plan,
            sequence=harness.owner.reserve_sequence(writer=harness.lease),
            clock=replace(clock, utc_ms=boundary_ms + issued_offset))
        if issued_offset >= 0:
            assert command.hold_until_ms > boundary_ms
        harness.owner.record_pending_command(command)
        before = harness.store.path.read_bytes()
        state_before = harness.owner.state
        payload_before = command.to_dict()
        calls = observe_receiver_calls(monkeypatch, harness.gateway)
        failure, receipt = None, None
        with final_section(harness.owner) as lease:
            try:
                receipt = module.send_locked(command, lease=lease, owner=harness.owner,
                    receiver=harness.gateway, deadline=Deadline(clock.boottime_ns + 1_000_000_000,
                                                                clock.utc_ms + 60_000),
                    clock=clock, policy=POLICY)
            except module.ExecutionRefused as exc:
                failure = exc.reason
        future = issued_offset > -100
        assert failure == ("command_window_invalid" if future else None)
        assert command.to_dict() == payload_before
        if future:
            assert receipt is None
            assert calls == []
            assert harness.gateway.effect_log == []
            assert harness.store.path.read_bytes() == before
            assert harness.owner.state == state_before
            assert harness.owner.state.pending_command == command
        else:
            assert receipt.status == "accepted"
            assert receipt.accepted_ts == clock.utc_ms
            assert calls.count("send") == 1
            assert len(harness.gateway.effect_log) == 3
            assert harness.store.load_strict().pending_command is None
    finally:
        harness.lease.close()


def test_09_an_exhausted_budget_refuses_and_never_sends(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    expired = Deadline(MONO - 1, NOW + 60_000)
    try:
        with final_section(layout.owner) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.send_locked(command, lease=lease, owner=layout.owner,
                                   receiver=layout.gateway, deadline=expired, clock=CLOCK,
                                   policy=POLICY)
        assert error.value.reason == "budget_exhausted"
        assert not layout.gateway.effect_log
    finally:
        layout.lease.close()


def test_09_a_shorter_owner_lease_shortens_the_deadline(artifact):
    module = execution()
    layout = owner_layout(artifact.root, requested_not_after_ms=NOW + 100)
    seen = []
    real_send = layout.gateway.send

    def send(payload, *, deadline):
        seen.append(deadline.monotonic_end_ns)
        return real_send(payload, deadline=deadline)

    layout.gateway.send = send
    try:
        with final_section(layout.owner) as lease:
            receipt = module.owner_section(lease=lease, owner=layout.owner,
                                           receiver=layout.gateway, request=request(),
                                           policy=POLICY, clock=CLOCK, deadline=DEADLINE)
        assert receipt.status == "accepted"
        # The owner lease ends 100 ms after the clock sample while the section
        # budget is 1000 ms, so the shorter lease bounds the deadline.
        assert seen == [MONO + 100_000_000]
    finally:
        layout.lease.close()


def test_09_an_expired_owner_lease_refuses_the_section(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    expired = replace(CLOCK, utc_ms=layout.binding.owner_not_after_ms)
    try:
        with final_section(layout.owner) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                     request=request(), policy=POLICY, clock=expired,
                                     deadline=DEADLINE)
        assert error.value.reason == "budget_exhausted"
        assert not layout.gateway.effect_log
    finally:
        layout.lease.close()


def test_09_a_result_write_after_the_deadline_restores_the_pending_command(artifact):
    """Na de laatste duurzame write wordt de deadline getoetst; de pending komt terug."""
    module = execution()
    layout = owner_layout(artifact.root)
    real_record = layout.owner.record_receipt
    telling = {"over": False}

    def nu():
        return MONO + 2_000_000_000 if telling["over"] else MONO

    layout.owner.now_ns = nu

    def record(receipt, **kwargs):
        result = real_record(receipt, **kwargs)
        # De duurzame resultaatwrite kost de rest van het budget.
        telling["over"] = True
        return result

    layout.owner.record_receipt = record
    try:
        with final_section(layout.owner) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                     request=request(), policy=POLICY, clock=CLOCK,
                                     deadline=DEADLINE)
        assert error.value.reason == "deadline_exceeded"
        mirror = read_mirror(artifact.root)
        # Het ontvangerbewijs blijft als historie staan ...
        assert mirror["last_receipt"]["status"] == "accepted"
        # ... en het pending command is duurzaam hersteld.
        assert mirror["pending_command"] is not None
        assert mirror["pending_command"]["command_id"] == mirror["last_receipt"]["command_id"]
    finally:
        layout.lease.close()


# -- the owner section -------------------------------------------------------

@pytest.mark.parametrize("meter", ["sandbox", "other-installation", "Sandbox", " sandbox",
                                    "sandbox "])
def test_i2_plan_meter_must_equal_owner_installation(artifact, monkeypatch, meter):
    """Only exact equality permits reservation; aliases must leave no trace."""
    module = execution()
    layout = owner_layout(artifact.root)
    context = artifact.plan.financial_context
    plan = replace(artifact.plan, financial_context=replace(
        context, basis=replace(context.basis, meter_id=meter)))
    artifact.path.write_text(json.dumps(plan.to_dict()))
    before = layout.store.path.read_bytes()
    state_before = layout.owner.state
    calls = observe_receiver_calls(monkeypatch, layout.gateway)
    try:
        error = None
        with final_section(layout.owner) as lease:
            validated = module.validate_current_locked(lease=lease,
                expected_digest=module.plan_digest(plan), clock=CLOCK, policy=POLICY)
            assert validated.financial_context.basis.meter_id == meter
            try:
                receipt = module.owner_section(lease=lease, owner=layout.owner,
                    receiver=layout.gateway, request=request(), policy=POLICY,
                    clock=CLOCK, deadline=DEADLINE)
            except module.ExecutionRefused as exc:
                error = exc
        if meter == INSTALLATION:
            assert error is None
            assert receipt.status == "accepted"
            assert receipt.installation_id == meter
            assert calls.count("send") == 1 and "inspect" not in calls
            assert layout.store.load_strict().last_reserved_sequence == 41
            return
        assert error is not None, "a foreign plan must be refused before reservation"
        assert error.reason == "installation_mismatch"
        assert calls == []
        assert layout.gateway.effect_log == []
        assert layout.store.path.read_bytes() == before
        assert layout.owner.state == state_before
        assert layout.owner.state.pending_command is None
        assert layout.owner._sequence_high_water == 40
    finally:
        layout.lease.close()


@pytest.mark.parametrize("fault", ["binding", "state", "store", "missing_binding", "missing_state"])
def test_i2_owner_section_checks_every_identity_before_reserving(artifact, monkeypatch, fault):
    module = execution()
    layout = owner_layout(artifact.root)
    try:
        if fault in ("binding", "state"):
            current = getattr(layout.owner, fault)
            setattr(layout.owner, fault, replace(current, installation_id="other-installation"))
        elif fault == "store":
            layout.store.installation_id = "other-installation"
        else:
            setattr(layout.owner, fault.removeprefix("missing_"), None)
        before = layout.store.path.read_bytes()
        state_before = layout.owner.state
        calls = observe_receiver_calls(monkeypatch, layout.gateway)
        error = None
        with final_section(layout.owner) as lease:
            try:
                module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                    request=request(), policy=POLICY, clock=CLOCK, deadline=DEADLINE)
            except Exception as exc:
                error = exc
        assert isinstance(error, module.ExecutionRefused), error
        expected = {"missing_binding": "owner_binding_missing",
                    "missing_state": "owner_state_missing"}.get(fault, "installation_mismatch")
        assert error.reason == expected
        assert calls == []
        assert layout.gateway.effect_log == []
        assert layout.store.path.read_bytes() == before
        assert layout.owner.state == state_before
        assert layout.owner._sequence_high_water == 40
    finally:
        layout.lease.close()


def test_i2_build_command_does_not_relabel_a_foreign_plan(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    context = artifact.plan.financial_context
    other = replace(artifact.plan, financial_context=replace(
        context, basis=replace(context.basis, meter_id="other-installation")))
    try:
        command = module._build_command(owner=layout.owner, request=request(), plan=other,
                                        sequence=41, clock=CLOCK)
        assert command.installation_id == "other-installation"
    finally:
        layout.lease.close()


def test_08_the_owner_section_sends_once_and_records_the_mirror(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    try:
        with final_section(layout.owner) as lease:
            receipt = module.owner_section(lease=lease, owner=layout.owner,
                                           receiver=layout.gateway, request=request(),
                                           policy=POLICY, clock=CLOCK, deadline=DEADLINE)
        assert receipt.status == "accepted"
        assert receipt.sequence == 41
        assert receipt.command_id == f"{hashlib.sha256(b'sandbox').hexdigest()[:32]}/8/41"
        mirror = read_mirror(artifact.root)
        assert mirror["pending_command"] is None
        assert mirror["last_receipt"]["status"] == "accepted"
        assert mirror["last_reserved_sequence"] == 41
        assert mirror["phase"] == "owned"
    finally:
        layout.lease.close()


def test_08_a_blocked_plan_revokes_the_epoch_and_gives_no_receipt(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    artifact.path.write_text(json.dumps({"format_version": 2, "status": "blocked",
                                         "created_ts": NOW_TS, "reason": "correction"}))
    try:
        with final_section(layout.owner) as lease:
            receipt = module.owner_section(lease=lease, owner=layout.owner,
                                           receiver=layout.gateway, request=request(),
                                           policy=POLICY, clock=CLOCK, deadline=DEADLINE)
        assert receipt is None
        assert not layout.gateway.effect_log
        mirror = read_mirror(artifact.root)
        assert mirror["phase"] == "released"
        assert mirror["last_barrier"]["status"] == "accepted"
    finally:
        layout.lease.close()


def test_08_the_invalidation_uses_the_remaining_section_budget(artifact):
    """De intrekking gebruikt het resterende sectiebudget, geen eigen venster."""
    module = execution()
    layout = owner_layout(artifact.root)
    artifact.path.write_text(json.dumps({"format_version": 2, "status": "blocked",
                                         "created_ts": NOW_TS, "reason": "correction"}))
    seen = []
    real_revoke = layout.gateway.revoke_and_stop

    def revoke(request, *, deadline):
        seen.append(deadline)
        return real_revoke(request, deadline=deadline)

    layout.gateway.revoke_and_stop = revoke
    short = Deadline(MONO + 300_000_000, NOW + 60_000)
    try:
        with final_section(layout.owner) as lease:
            receipt = module.owner_section(lease=lease, owner=layout.owner,
                                           receiver=layout.gateway, request=request(),
                                           policy=POLICY, clock=CLOCK, deadline=short)
        assert receipt is None
        assert len(seen) == 1
        # Het resterende sectiebudget, nooit een eigen venster van 750 ms.
        assert seen[0].monotonic_end_ns == MONO + 300_000_000
    finally:
        layout.lease.close()


def test_08_without_budget_the_epoch_is_not_withdrawn(artifact):
    """Zonder budget wordt de intrekking niet geprobeerd; de herstelplicht blijft."""
    module = execution()
    layout = owner_layout(artifact.root)
    artifact.path.write_text(json.dumps({"format_version": 2, "status": "blocked",
                                         "created_ts": NOW_TS, "reason": "correction"}))
    seen = []
    layout.gateway.revoke_and_stop = lambda request, *, deadline: seen.append(deadline)
    expired = Deadline(MONO - 1, NOW + 60_000)
    try:
        with final_section(layout.owner) as lease:
            receipt = module.owner_section(lease=lease, owner=layout.owner,
                                           receiver=layout.gateway, request=request(),
                                           policy=POLICY, clock=CLOCK, deadline=expired)
        assert receipt is None
        assert seen == []
        # De herstelplicht blijft staan en is zichtbaar.
        assert layout.owner.recovery_required is True
        assert layout.owner.phase == "inhibited"
        # ... en is ook duurzaam zichtbaar: de spiegel staat op 'inhibited' en
        # niet meer op 'owned'. Het geheugen alleen is geen bewijs.
        state = layout.store.load_strict()
        assert state.phase == "inhibited"
        assert state.owner_epoch == 8
    finally:
        layout.lease.close()


def test_08_the_runner_failure_report_stays_in_memory(artifact):
    """Zonder `durable` blijft de uitkomst in het geheugen; de spiegel blijft zeker."""
    layout = owner_layout(artifact.root)
    try:
        result = layout.owner.report_failure("owner_lease_busy")
        assert result.reason == "owner_lease_busy"
        assert layout.owner.phase == "inhibited"
        assert layout.owner.recovery_required is True
        # De runner-hooks roepen report_failure zonder `durable` aan: hun
        # gedrag blijft exact gelijk en ze kunnen dus nooit een mirror
        # herschrijven of aanmaken.
        state = layout.store.load_strict()
        assert state.phase == "owned"
    finally:
        layout.lease.close()


def test_08_a_durable_failure_never_creates_a_mirror(artifact):
    """Zonder bestaande spiegel wordt er nooit een mirror aangemaakt."""
    module = ownership()
    store = module.OwnerStore(artifact.path, INSTALLATION, lease_dir=artifact.root / "leases")
    owner = module.OwnerCoordinator(store)
    assert not store.path.exists()
    result = owner.report_failure("owner_lease_busy", durable=True)
    assert result.reason == "owner_lease_busy"
    assert owner.phase == "inhibited"
    assert owner.recovery_required is True
    assert not store.path.exists()


def test_08_a_durable_failure_never_raises(artifact):
    """Een schrijffout wordt niet doorgegeven; de herstelplicht blijft staan."""
    layout = owner_layout(artifact.root)
    try:
        # De levensduurlease is weg: de spiegel kan niet meer geschreven worden.
        layout.lease.close()
        result = layout.owner.report_failure("owner_lease_missing", durable=True)
        assert result.reason == "owner_lease_missing"
        assert layout.owner.phase == "inhibited"
        assert layout.owner.recovery_required is True
        # De mislukte write is niet doorgegeven; de spiegel bleef ongewijzigd.
        state = layout.store.load_strict()
        assert state.phase == "owned"
    finally:
        layout.lease.close()


def unconfirmed_receipt(command):
    from dao.prog.fastctrl.command import ActuationReceipt

    return ActuationReceipt(installation_id=command.installation_id, battery_name=command.battery_name,
                            owner_epoch=command.owner_epoch, sequence=command.sequence,
                            command_id=command.command_id, plan_digest=command.plan_digest,
                            accepted_ts=None, status="unconfirmed", helper_results=(),
                            owner_session_id=command.owner_session_id,
                            payload_digest=payload_digest(command), reason="pending_effect",
                            observed_at_ms=command.issued_at_ms, effect="unknown")


def test_11_the_mirror_records_never_clear_the_recovery_duty(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    layout.owner.recovery_required = True
    receipt = unconfirmed_receipt(command)
    layout.owner.record_receipt(receipt)
    state = layout.store.load_strict()
    assert state.pending_command == command
    assert state.last_receipt == receipt
    assert layout.owner.phase == "owned"
    assert layout.owner.recovery_required is True
    layout.lease.close()


def test_11_a_non_terminal_receipt_keeps_the_pending_command(artifact):
    """`terminal=False` bewaart het bewijs als historie; het pending blijft staan."""
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    receipt = layout.gateway.send(command, deadline=DEADLINE)
    assert receipt.status == "accepted"
    layout.owner.record_receipt(receipt, terminal=False)
    state = layout.store.load_strict()
    assert state.pending_command == command
    assert state.last_receipt == receipt
    assert layout.owner.phase == "owned"
    layout.lease.close()


# -- the public entrypoint ---------------------------------------------------

def test_10_the_production_route_never_reaches_the_send(artifact, monkeypatch):
    from dao.prog.config.models.fastcontrol import FastControlConfig
    from dao.prog.fastctrl import runner as runner_module
    from dao.tests.prog.test_fast_runner import FakeHass

    reached = []
    monkeypatch.setattr(runner_module.time, "clock_gettime_ns", lambda _: MONO)
    module = execution()
    monkeypatch.setattr(module, "send_locked", lambda *args, **kwargs: reached.append(kwargs))
    hass = FakeHass()
    runner = runner_module.FastControlRunner(hass, FastControlConfig(),
                                             plan_path=str(artifact.path),
                                             state_path=str(artifact.root / "fast_state.json"),
                                             gate_policy=POLICY)
    assert runner.tick(NOW_TS) is None
    assert runner.last_gate.allowed is False
    assert runner.last_gate.reason == "actuation_gates_unproven"
    assert hass.values == hass.options == hass.switches == hass.published == {}
    assert reached == []
    # The production route never reaches the mechanical boundary: the spy sits
    # on the boundary itself, so a call through any name would show up.
    monkeypatch.setattr(module, "owner_section", lambda *args, **kwargs: reached.append(kwargs))
    assert runner.tick(NOW_TS + 1) is None
    assert runner.last_gate.reason == "actuation_gates_unproven"
    assert reached == []
    # The runner binds the boundary under exactly one alias, so the spy above
    # covers every path into it.
    assert [name for name, member in inspect.getmembers(runner_module)
            if member is module] == ["execution_boundary"]


def test_12_the_regressions_use_no_sleep_as_causal_synchronisation():
    source = Path(__file__).read_text()
    needle = "sleep" + "("
    assert needle not in source
    assert "time" + ".sleep" not in source
