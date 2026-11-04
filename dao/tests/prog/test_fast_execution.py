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
MONO = 1_000_000_000
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


def attempts():
    module = import_module("dao.prog.fastctrl.publication")
    assert hasattr(module, "publication_attempts"), "F2c4 publication_attempts missing"
    return module


def ownership():
    module = import_module("dao.prog.fastctrl.ownership")
    assert hasattr(module.OwnerCoordinator, "record_pending_command"), (
        "F2c4 record_pending_command missing")
    assert hasattr(module.OwnerCoordinator, "record_receipt"), "F2c4 record_receipt missing"
    return module


def mirror_path(root):
    return Path(root) / "actuation_owner" / MIRROR


def receiver_state_file(root):
    return Path(root) / "receiver" / "receiver-state.json"


def read_mirror(root):
    return json.loads(mirror_path(root).read_text())


def provision_mirror(root):
    """The synthetic trusted mirror; production has no bootstrap path."""
    module = ownership()
    store = module.OwnerStore(Path(root) / "fast_plan.json", INSTALLATION,
                              lease_dir=Path(root) / "leases")
    store.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    store.path.write_text(json.dumps(module.OwnerState(
        1, INSTALLATION, "receiver-a", "generation-a", "checkpoint-40", "", 0, 40,
        "inhibited", None, None, None, None).to_dict()))
    store.path.chmod(0o600)
    return store


def owner_layout(root, *, requested_not_after_ms=None):
    """Bind one synthetic owner against the deterministic fake receiver."""
    module = ownership()
    store = provision_mirror(root)
    gateway = FencedReceiver(Path(root) / "receiver", initial_state=receiver_state(), utc_ms=NOW)
    owner = module.OwnerCoordinator(store, receiver=gateway)
    owner.now_ns = lambda: MONO
    owner.now_ms = lambda: NOW
    lease = task_execution.acquire_resource(RESOURCE, lease_dir=Path(root) / "leases")
    binding = owner.start(writer=lease, receiver=gateway, installation_id=INSTALLATION,
                          owner_session_id="session-new",
                          requested_not_after_ms=requested_not_after_ms or (NOW + 60_000),
                          deadline=DEADLINE, now_ms=NOW)
    assert binding.owner_epoch == 8, binding
    return SimpleNamespace(module=module, store=store, gateway=gateway, owner=owner,
                           lease=lease, binding=binding)


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
    store = provision_mirror(root)
    gateway = FencedReceiver(root / "receiver", initial_state=receiver_state(), utc_ms=NOW)
    owner = store_owner(store, gateway)
    lease = task_execution.acquire_resource(RESOURCE, lease_dir=root / "leases")
    binding = owner.start(writer=lease, receiver=gateway, installation_id=INSTALLATION,
                          owner_session_id="session-new", requested_not_after_ms=NOW + 60_000,
                          deadline=DEADLINE, now_ms=NOW)
    assert binding.owner_epoch == 8, binding
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
        module_attempts = attempts()
        with module_attempts.publication_attempts(
                root / "fast_plan.json",
                deadline_ns=module_attempts.publication_deadline_ns()) as lease:
            return module.owner_section(lease=lease, owner=owner, receiver=gateway,
                                        request=command_request, policy=POLICY, clock=CLOCK,
                                        deadline=DEADLINE)

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
    mirror = read_mirror(system.root)
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
    mirror = read_mirror(system.root)
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

def test_07_send_locked_refuses_a_changed_pending_command(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    changed = replace(command, setpoint_w=command.setpoint_w + 1)
    try:
        with publication_lock(artifact.path) as lease:
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
        with publication_lock(artifact.path) as lease:
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
    layout.lease.close()
    with publication_lock(artifact.path) as lease:
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
        with publication_lock(artifact.path) as lease:
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
        with publication_lock(artifact.path) as lease:
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
    telling = {"n": 0}

    def nu():
        telling["n"] += 1
        # Eén controle binnen het budget; de controle na de send erbuiten.
        return MONO if telling["n"] <= 1 else MONO + 2_000_000_000

    layout.owner.now_ns = nu
    try:
        with publication_lock(artifact.path) as lease:
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

def test_09_an_exhausted_budget_refuses_and_never_sends(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    expired = Deadline(MONO - 1, NOW + 60_000)
    try:
        with publication_lock(artifact.path) as lease:
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
        with publication_lock(artifact.path) as lease:
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
    expired = replace(CLOCK, utc_ms=NOW + 6_000)
    try:
        with publication_lock(artifact.path) as lease:
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
        with publication_lock(artifact.path) as lease:
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

def test_08_the_owner_section_sends_once_and_records_the_mirror(artifact):
    module = execution()
    layout = owner_layout(artifact.root)
    try:
        with publication_lock(artifact.path) as lease:
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
        with publication_lock(artifact.path) as lease:
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
        with publication_lock(artifact.path) as lease:
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
        with publication_lock(artifact.path) as lease:
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
                            payload_digest="c" * 64, reason="pending_effect",
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
    receipt = observed_receipt("accepted", "complete", "accepted")
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
    # The runner does not even import the mechanical boundary.
    assert "fastctrl.execution" not in inspect.getsource(runner_module)


def test_12_the_regressions_use_no_sleep_as_causal_synchronisation():
    source = Path(__file__).read_text()
    needle = "sleep" + "("
    assert needle not in source
    assert "time" + ".sleep" not in source
