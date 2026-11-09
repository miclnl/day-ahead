"""F2c5: the financial consumer, the recovery duty and the legacy boundary.

Everything here is synthetic: a deterministic fake receiver, a temporary plan
file and a fake Home Assistant. No network, no device and no public execute
route: the financial consumer is only reachable by calling it directly.
"""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from importlib import import_module
from types import SimpleNamespace

import pytest

from dao.prog import task_execution
from dao.prog.fastctrl.command import ActuationReceipt, Deadline, RevokeRequest
from dao.prog.fastctrl.plan import profile_power_at
from dao.prog.fastctrl.policy import (
    BatteryMeasurement,
    ControllerState,
    FastControlPolicy,
    Measurement,
    PolicyLimits,
)
from dao.prog.fastctrl.publication import GatePolicy
from dao.prog.fastctrl.runner import FastControlRunner
from dao.tests.prog.fakes.fenced_receiver import FencedReceiver
from dao.tests.prog.test_fast_execution import next_owner_cycle, observe_receiver_calls
from dao.tests.prog.test_fast_ownership import RECEIPT_IDENTITY_MISMATCHES
from dao.tests.prog.test_fast_receiver import state as receiver_state
from dao.tests.prog.test_fast_runner import FakeHass, make_config
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_fast_contract import ready_plan

INSTALLATION = "sandbox"
RESOURCE = "actuator-" + hashlib.sha256(b"sandbox").hexdigest()[:32]
MONO = 15_000_000_000
#: The plant timeline is seconds; the clock and the receiver are milliseconds.
NOW_TS = START_TS + 900
NOW = NOW_TS * 1000
POLICY = GatePolicy(START_TS)
DEADLINE = Deadline(MONO + 1_000_000_000, NOW + 60_000)
#: The synthetic owner lease; the fake receiver caps its own lease.
LEASE_MS = 3_600_000
#: The plan power of the synthetic financial plan.
PLAN_W = -2000.0
#: The import price of the synthetic plan; the export price is zero.
PRICE = 0.30
#: The three actuator entities of the synthetic plan.
SETPOINT_ENTITY = "input_number.feedin"
MODE_ENTITY = "input_select.mode"
STOP_ENTITY = "input_datetime.stop"


def actuator():
    # Deliberate assertion RED, never a collection/import error.
    module = import_module("dao.prog.fastctrl.actuator")
    assert hasattr(module, "FinancialActuationAdapter"), (
        "F2c5 nominal adapter base class missing")
    assert hasattr(module, "classify_actuation_adapter"), (
        "F2c5 adapter classification missing")
    assert hasattr(module, "ActuationVerdict"), "F2c5 actuation verdict missing"
    return module


def financial_plan(*, batteries=1, powers=(PLAN_W,), economics_supported=True):
    """A ready-v2 plan with a real profile: the optimizer wrote nothing."""
    plan = ready_plan()
    context = plan.financial_context
    names = ["accu", "accu2"][:batteries]
    specs = [replace(plan.specs[0], name=name, setpoint_entity=SETPOINT_ENTITY,
                     mode_entity=MODE_ENTITY, stop_inverter_entity=STOP_ENTITY,
                     soc_entity="sensor.soc") for name in names]
    terminal = replace(context.terminal_value,
                       reference_efficiency_by_battery=tuple(
                           (name, Decimal("0.94")) for name in names))
    # The lease covers the first two intervals, so a profile transition inside
    # the horizon is reachable without leaving the plan window.
    context = replace(context, basis=replace(context.basis, meter_id=INSTALLATION),
                       terminal_value=terminal, not_after_ts=NOW_TS + 1800,
                      economics_supported=economics_supported)
    intervals = []
    for index, row in enumerate(plan.intervals):
        power = powers[index] if index < len(powers) else powers[-1]
        # A rectangular profile over the first two thirds of the interval.
        mean = power * 600 / 900
        steps = [replace(row.battery(0), ac_power_w=mean, mean_ac_power_w=mean,
                         on_power_w=power, active_from_ts=row.start_ts,
                         stop_ts=row.start_ts + 600) for _ in names]
        intervals.append(replace(row, price_import=PRICE, price_export=0.0,
                                 batteries=steps))
    return replace(plan, specs=specs, intervals=intervals, financial_context=context,
                   price_average=PRICE)


def provision_mirror(root):
    """The synthetic trusted mirror; production has no bootstrap path."""
    module = import_module("dao.prog.fastctrl.ownership")
    store = module.OwnerStore(root / "fast_plan.json", INSTALLATION,
                              lease_dir=root / "leases")
    store.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    store.path.write_text(json.dumps(module.OwnerState(
        1, INSTALLATION, "receiver-a", "generation-a", "checkpoint-40", "", 0, 40,
        "inhibited", None, None, None, None).to_dict()))
    store.path.chmod(0o600)
    return store


def build(tmp_path, *, plan=None, config=None, start_owner=True, lease_ms=LEASE_MS):
    """One synthetic financial session: a plan, an owner and a fake receiver."""
    ownership = import_module("dao.prog.fastctrl.ownership")
    plan = plan or financial_plan()
    path = tmp_path / "fast_plan.json"
    path.write_text(json.dumps(plan.to_dict()))
    store = provision_mirror(tmp_path)
    gateway = FencedReceiver(tmp_path / "receiver", initial_state=receiver_state(),
                              utc_ms=NOW - 10_000 if start_owner else NOW)
    gateway.receiver_max_lease_ms = LEASE_MS + 10_000
    owner = ownership.OwnerCoordinator(
        store, receiver=gateway,
        now_ns=lambda: 0,
        now_ms=lambda: gateway.utc_ms,
    )
    lease = None
    if start_owner:
        lease = task_execution.acquire_resource(RESOURCE, lease_dir=tmp_path / "leases")
        owner._bind_lifetime(lease)
        owner.now_ns = lambda: 5_000_000_000
        binding = owner.start(writer=lease, receiver=gateway, installation_id=INSTALLATION,
                              owner_session_id="session-new",
                              requested_not_after_ms=NOW + lease_ms,
                              deadline=Deadline(6_000_000_000, NOW + lease_ms), now_ms=gateway.utc_ms)
        assert binding.owner_epoch == 8, binding
    gateway.utc_ms = NOW
    owner.now_ns = lambda: MONO + gateway.monotonic_ns
    hass = FakeHass({"sensor.p1_power": "0", "sensor.soc": "50",
                     "sensor.battery_power": "0"})
    runner = FastControlRunner(hass, config or make_config(), plan_path=str(path),
                               state_path=str(tmp_path / "fast_state.json"),
                               gate_policy=POLICY, owner=owner)
    return SimpleNamespace(runner=runner, owner=owner, gateway=gateway, lease=lease,
                           store=store, plan=plan, path=path, root=tmp_path, hass=hass)


def propose(plan, state, *, timestamp=NOW_TS, grid_w=None, limits=None, owner_binding=None):
    """The policy's verdict for one tick."""
    policy = FastControlPolicy(limits or PolicyLimits())
    interval = plan.interval_at(timestamp)
    planned = 0.0
    if interval is not None:
        for index in range(len(plan.specs)):
            planned += profile_power_at(interval.battery(index), timestamp)
    if grid_w is None:
        # The measured grid matches the plan: the economic optimum is the plan
        # itself, so the policy proposes instead of deviating.
        grid_w = -planned
    measurement = Measurement(
        timestamp=timestamp, grid_w=grid_w,
        batteries=[BatteryMeasurement(soc=50.0, power_w=0.0, valid=True)
                   for _ in plan.specs],
        grid_valid=True,
    )
    return policy.decide(plan, measurement, state, owner_binding=owner_binding)


def consume(harness, decision, *, now_ms=NOW):
    return harness.runner._consume_financial_actuation(
        decision, harness.plan, owner=harness.owner, now_ms=now_ms)


def stop(harness, **kwargs):
    next_owner_cycle(harness.owner)
    return harness.owner.stop(**kwargs)


def i5_short_plan():
    plan = financial_plan(batteries=2)
    return replace(plan, financial_context=replace(plan.financial_context, not_after_ts=NOW_TS + 1))


@pytest.mark.parametrize("batteries", [1, 2])
@pytest.mark.parametrize("elapsed", [800, 999, 1000, 1001])
def test_i5_fix1_consumer_result_write_uses_section_budget(tmp_path, batteries, elapsed):
    harness = build(tmp_path, plan=financial_plan(batteries=batteries))
    gateway = harness.gateway
    real_send, real_record = gateway.send, harness.owner.record_receipt
    commands, original, deadlines = [], [], []

    def send(command, *, deadline):
        commands.append(command)
        deadlines.append(deadline)
        receipt = real_send(command, deadline=deadline)
        original.append(receipt)
        return receipt

    def record(receipt, **kwargs):
        real_record(receipt, **kwargs)
        gateway.advance_clock(elapsed)

    gateway.send, harness.owner.record_receipt = send, record
    try:
        decision = propose(harness.plan, harness.runner.state)
        receipts = consume(harness, decision)
        late = elapsed >= 1000
        expected = "deadline_exceeded" if late else ("accepted" if batteries == 1 else "budget_exhausted")
        assert harness.runner.last_actuation.reason == expected
        assert harness.runner.last_actuation.allowed is (not late and batteries == 1)
        assert receipts == (() if late else tuple(original))
        assert len(commands) == len(original) == 1
        command = commands[0]
        assert original[0].status == "accepted"
        assert original[0].accepted_ts == NOW
        assert deadlines == [Deadline(MONO + 750_000_000, NOW + 600_000)]
        assert gateway.effect_log == [(command.command_id, part, "accu", PLAN_W)
                                     for part in ("setpoint", "mode", "stop_inverter")]
        mirror = harness.store.load_strict()
        assert mirror.last_receipt == original[0]
        assert mirror.pending_command == (command if late else None)
        controller = harness.runner.state.battery(0)
        assert controller.execution_status == ("unconfirmed" if late else "accepted")
        assert controller.last_accepted_command == (None if late else command)
        assert controller.pending_command == (command if late else None)
        assert controller.recovery_required is late
        assert controller.held_setpoint_w == (0.0 if late else PLAN_W)
        if batteries == 2:
            second = harness.runner.state.battery(1)
            assert second.pending_command is None
            assert second.last_accepted_command is None
            assert second.recovery_required is False
        if late:
            assert consume(harness, decision, now_ms=gateway.utc_ms) == ()
            assert harness.runner.last_actuation.reason == "recovery_required"
            assert len(commands) == 1 and len(gateway.effect_log) == 3
    finally:
        harness.lease.close()


@pytest.mark.parametrize("next_send", ["battery", "retry"])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_i5_fix1_next_send_keeps_original_transport_cutoff(tmp_path, next_send, offset):
    harness = build(tmp_path, plan=financial_plan(batteries=2 if next_send == "battery" else 1))
    gateway = harness.gateway
    real_send = gateway.send
    commands, deadlines, remaining, original = [], [], [], []

    def send(command, *, deadline):
        commands.append(command)
        deadlines.append(deadline)
        remaining.append(deadline.remaining_ns(harness.owner.now_ns()))
        if next_send == "retry" and len(commands) == 1:
            gateway.enqueue(command)
            receipt = gateway.deliver()
        else:
            if next_send == "retry":
                for part in ("setpoint", "mode", "stop_inverter"):
                    gateway.step_helper(part, command=command)
            receipt = real_send(command, deadline=deadline)
        original.append(receipt)
        gateway.advance_clock(400 if len(commands) == 1 else 350 + offset)
        return receipt

    gateway.send = send
    try:
        receipts = consume(harness, propose(harness.plan, harness.runner.state))
        late = offset >= 0
        assert harness.runner.last_actuation.reason == ("deadline_exceeded" if late else "accepted")
        assert harness.runner.last_actuation.allowed is not late
        assert len(commands) == 2
        assert deadlines == [Deadline(MONO + 750_000_000, NOW + 600_000)] * 2
        assert remaining == [750_000_000, 350_000_000]
        assert original[-1].status == "accepted"
        assert original[-1].accepted_ts == NOW + 400
        expected = (original[:1] if late else original) if next_send == "battery" else (
            [] if late else original[-1:])
        assert receipts == tuple(expected)
        if next_send == "retry":
            assert commands[0] == commands[1]
        assert len(gateway.effect_log) == (6 if next_send == "battery" else 3)
        assert harness.store.load_strict().last_receipt == original[-1]
        assert harness.store.load_strict().pending_command == (commands[-1] if late else None)
        controller = harness.runner.state.battery(1 if next_send == "battery" else 0)
        assert controller.recovery_required is late
    finally:
        harness.lease.close()


@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("finish", ["ack", "write"])
def test_i5_short_plan_lease_bounds_ack_deadline(tmp_path, offset, finish):
    harness = build(tmp_path, plan=i5_short_plan())
    gateway = harness.gateway
    gateway.advance_clock(900)
    real_send, real_record = gateway.send, harness.owner.record_receipt
    commands, original = [], []

    def send(command, *, deadline):
        commands.append(command)
        receipt = real_send(command, deadline=deadline)
        original.append(receipt)
        if finish == "ack" and len(commands) == 1:
            gateway.advance_clock(100 + offset)
        return receipt

    def record(receipt, **kwargs):
        real_record(receipt, **kwargs)
        if finish == "write" and len(commands) == 1:
            gateway.advance_clock(100 + offset)

    gateway.send, harness.owner.record_receipt = send, record
    try:
        decision = propose(harness.plan, harness.runner.state, timestamp=(NOW + 900) / 1000)
        receipts = consume(harness, decision, now_ms=NOW + 900)
        late = offset >= 0
        assert harness.runner.last_actuation.reason == ("deadline_exceeded" if late else "accepted")
        assert harness.runner.last_actuation.allowed is not late
        assert receipts == (() if late else tuple(original))
        assert [command.battery_name for command in commands] == (["accu"] if late else ["accu", "accu2"])
        assert [receipt.status for receipt in original] == ["accepted"] * len(commands)
        assert original[0].accepted_ts == NOW + 900
        assert gateway.effect_log == [(command.command_id, part, command.battery_name, -2000)
                                     for command in commands
                                     for part in ("setpoint", "mode", "stop_inverter")]
        mirror = harness.store.load_strict()
        assert mirror.last_receipt == original[-1]
        assert mirror.pending_command == (commands[0] if late else None)
        first = harness.runner.state.battery(0)
        assert first.execution_status == ("unconfirmed" if late else "accepted")
        assert first.recovery_required is late
        assert first.pending_command == (commands[0] if late else None)
        assert first.last_accepted_command == (None if late else commands[0])
        assert first.held_setpoint_w == (0 if late else -2000)
        second = harness.runner.state.battery(1)
        assert second.last_accepted_command == (None if late else commands[1])
        if late:
            assert second.pending_command is None
            assert consume(harness, decision, now_ms=gateway.utc_ms) == ()
            assert harness.runner.last_actuation.reason == "recovery_required"
            assert len(gateway.effect_log) == 3
    finally:
        harness.lease.close()


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_i5_second_battery_only_has_the_remaining_plan_lease(tmp_path, offset):
    harness = build(tmp_path, plan=i5_short_plan())
    gateway = harness.gateway
    gateway.advance_clock(900)
    real_send = gateway.send
    commands, deadlines, remaining = [], [], []

    def send(command, *, deadline):
        commands.append(command)
        deadlines.append(deadline)
        remaining.append(deadline.remaining_ns(harness.owner.now_ns()))
        receipt = real_send(command, deadline=deadline)
        gateway.advance_clock(60 if len(commands) == 1 else 40 + offset)
        return receipt

    gateway.send = send
    try:
        receipts = consume(harness, propose(harness.plan, harness.runner.state,
                                           timestamp=(NOW + 900) / 1000), now_ms=NOW + 900)
        late = offset >= 0
        assert harness.runner.last_actuation.reason == ("deadline_exceeded" if late else "accepted")
        assert [r.battery_name for r in receipts] == (["accu"] if late else ["accu", "accu2"])
        assert deadlines == [Deadline(MONO + 1_000_000_000, NOW + 1000)] * 2
        assert remaining == [100_000_000, 40_000_000]
        assert len(gateway.effect_log) == 6
        assert harness.store.load_strict().pending_command == (commands[1] if late else None)
        assert harness.runner.state.battery(0).last_accepted_command == commands[0]
        assert harness.runner.state.battery(1).recovery_required is late
    finally:
        harness.lease.close()


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_i5_retry_and_query_share_the_remaining_plan_lease(tmp_path, offset):
    harness = build(tmp_path, plan=i5_short_plan())
    gateway = harness.gateway
    gateway.advance_clock(900)
    gateway.lose_ack()
    real_send, real_query = gateway.send, gateway.query
    commands, calls, deadlines = [], [], []

    def send(command, *, deadline):
        commands.append(command)
        calls.append("send")
        deadlines.append(deadline)
        receipt = real_send(command, deadline=deadline)
        gateway.advance_clock(40 if len(commands) == 1 else 60 + offset)
        return receipt

    def query(**kwargs):
        calls.append("query")
        deadlines.append(kwargs["deadline"])
        return real_query(**kwargs)

    gateway.send, gateway.query = send, query
    try:
        receipts = consume(harness, propose(harness.plan, harness.runner.state,
                                           timestamp=(NOW + 900) / 1000), now_ms=NOW + 900)
        assert harness.runner.last_actuation.reason == ("unconfirmed" if offset < 0 else "deadline_exceeded")
        assert [r.status for r in receipts] == (["unconfirmed"] if offset < 0 else [])
        assert calls == (["send", "send", "query"] if offset < 0 else ["send", "send"])
        assert commands == [commands[0]] * 2
        assert deadlines == [Deadline(MONO + 1_000_000_000, NOW + 1000)] * len(calls)
        assert len(gateway.effect_log) == 3
        assert harness.store.load_strict().pending_command == commands[0]
        assert harness.runner.state.battery(0).recovery_required is True
        assert harness.runner.state.battery(1).last_accepted_command is None
    finally:
        harness.lease.close()


@pytest.mark.parametrize("shift", [-2000, 2000])
def test_i5_utc_adjustment_during_ack_cannot_reopen_boottime_budget(tmp_path, shift):
    harness = build(tmp_path, plan=i5_short_plan())
    gateway = harness.gateway
    gateway.advance_clock(900)
    real_send = gateway.send
    commands = []

    def send(command, *, deadline):
        commands.append(command)
        receipt = real_send(command, deadline=deadline)
        gateway.advance_clock(100)
        # Transport-side UTC changes after the effect; BOOTTIME has still spent 100 ms.
        gateway.utc_ms += shift
        return receipt

    gateway.send = send
    try:
        assert consume(harness, propose(harness.plan, harness.runner.state,
                                        timestamp=(NOW + 900) / 1000), now_ms=NOW + 900) == ()
        assert harness.runner.last_actuation.reason == "deadline_exceeded"
        assert len(commands) == 1
        assert harness.store.load_strict().pending_command == commands[0]
        assert harness.store.load_strict().last_receipt.status == "accepted"
        assert harness.runner.state.battery(0).recovery_required is True
        assert len(gateway.effect_log) == 3
    finally:
        harness.lease.close()


@pytest.mark.parametrize("edge,power", [("active", 0.0), ("stop", 0.0),
                                       ("interval", -100.25), ("interval", 100.25)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_i5_profile_hold_expiry_requires_new_same_power_proposal(tmp_path, edge, power, offset):
    plan = financial_plan(powers=(power, power), economics_supported=False)
    first = plan.intervals[0]
    step = replace(first.battery(0), ac_power_w=power, mean_ac_power_w=power,
                   stop_ts=None)
    boundary_ms = NOW + 900_000
    if edge in ("active", "stop"):
        step = replace(step, **{f"{'active_from' if edge == 'active' else 'stop'}_ts": NOW_TS + 1})
        boundary_ms = NOW + 1000
    plan = replace(plan, intervals=[replace(first, batteries=[step]), *plan.intervals[1:]])
    harness = build(tmp_path, plan=plan)
    try:
        initial = consume(harness, propose(plan, harness.runner.state))[0]
        old = harness.runner.state.battery(0).last_accepted_command
        gateway = harness.gateway
        gateway.advance_clock(boundary_ms + offset - NOW)
        follow = propose(plan, harness.runner.state, timestamp=gateway.utc_ms / 1000,
                         owner_binding=harness.owner.binding)
        assert follow.batteries[0].setpoint_w == power
        assert follow.batteries[0].write is (offset >= 0)
        assert old.execute_before_ms == old.hold_until_ms == boundary_ms
        if offset >= 0 and edge in ("active", "stop"):
            next_owner_cycle(harness.owner)
        receipts = consume(harness, follow, now_ms=gateway.utc_ms)
        if offset < 0:
            assert receipts == ()
            assert harness.runner.last_actuation.reason == "no_proposal"
            assert len(gateway.effect_log) == 3
        else:
            assert len(receipts) == 1 and receipts[0].status == "accepted"
            assert receipts[0].command_id != initial.command_id
            assert receipts[0].sequence > initial.sequence
            assert harness.runner.state.battery(0).last_command_w == round(power)
            assert len(gateway.effect_log) == 6
    finally:
        harness.lease.close()


def test_i5_profile_hold_is_specific_to_each_battery(tmp_path):
    plan = financial_plan(batteries=2, powers=(0.0,))
    first = plan.intervals[0]
    steps = [replace(first.battery(0), active_from_ts=NOW_TS + 1),
             replace(first.battery(1), stop_ts=NOW_TS + 2)]
    plan = replace(plan, intervals=[replace(first, batteries=steps), *plan.intervals[1:]])
    harness = build(tmp_path, plan=plan)
    try:
        assert len(consume(harness, propose(plan, harness.runner.state))) == 2
        assert [battery.last_accepted_command.hold_until_ms for battery in harness.runner.state.batteries] == [
            NOW + 1000, NOW + 2000]
    finally:
        harness.lease.close()


def restart_i4_owner(harness, session="i4-fresh-session"):
    owner = harness.owner
    next_owner_cycle(owner)
    result = owner.start(
        writer=harness.lease, receiver=harness.gateway, installation_id=INSTALLATION,
        owner_session_id=session, requested_not_after_ms=owner.now_ms() + LEASE_MS,
        deadline=Deadline(owner.now_ns() + 750_000_000, owner.now_ms() + 750),
        now_ms=owner.now_ms())
    next_owner_cycle(owner)
    return result


@pytest.mark.parametrize("batteries", [1, 2])
@pytest.mark.parametrize("reload_state", [False, True])
def test_i4_stop_restart_same_power_requires_fresh_command(tmp_path, batteries, reload_state):
    harness = build(tmp_path, plan=financial_plan(batteries=batteries))
    try:
        first = consume(harness, propose(harness.plan, harness.runner.state))
        assert [receipt.status for receipt in first] == ["accepted"] * batteries
        history = harness.runner.state.to_dict()
        assert all(b.pending_command is None for b in harness.runner.state.batteries)
        barrier = stop(harness, reason="operator_stop")
        assert (barrier.status, barrier.safe_state, barrier.queues_fenced) == (
            "accepted", "confirmed", True)
        harness.runner._apply_stop_barrier(barrier, now_ms=NOW)
        # C2.3: clean batteries retain every historical fact after a safe stop.
        assert harness.runner.state.to_dict() == history
        if reload_state:
            from dao.prog.fastctrl.runner import load_state, save_state

            save_state(harness.runner.state, harness.runner.state_path)
            harness.runner.state = load_state(harness.runner.state_path)
        harness.gateway.advance_clock(1000)
        binding = restart_i4_owner(harness)
        assert binding.owner_epoch > first[0].owner_epoch
        follow = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 1,
                         owner_binding=binding)
        assert [b.write for b in follow.batteries] == [True] * batteries
        assert [b.setpoint_w for b in follow.batteries] == [PLAN_W] * batteries
        fresh = consume(harness, follow, now_ms=NOW + 1000)
        assert [receipt.status for receipt in fresh] == ["accepted"] * batteries
        assert {r.command_id for r in first}.isdisjoint(r.command_id for r in fresh)
        assert len(harness.gateway.effect_log) == 6 * batteries
    finally:
        harness.lease.close()


def test_i4_live_acceptance_keeps_noop(tmp_path):
    harness = build(tmp_path)
    try:
        consume(harness, propose(harness.plan, harness.runner.state))
        harness.gateway.advance_clock(60_000)
        follow = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 60,
                         owner_binding=harness.owner.binding)
        assert follow.batteries[0].write is False
        assert consume(harness, follow, now_ms=NOW + 60_000) == ()
        assert harness.runner.last_actuation.reason == "no_proposal"
        assert len(harness.gateway.effect_log) == 3
    finally:
        harness.lease.close()


@pytest.mark.parametrize("case", ["loaded", "missing", "installation", "session", "epoch",
                                  "owner_expired", "hold_before", "hold_at", "hold_after",
                                  "pending", "recovery", "future"])
def test_i4_historical_acceptance_needs_live_context(tmp_path, case):
    harness = build(tmp_path)
    try:
        consume(harness, propose(harness.plan, harness.runner.state))
        controller = harness.runner.state.battery(0)
        command = controller.last_accepted_command
        binding = harness.owner.binding
        timestamp = NOW_TS + 1
        if case == "loaded":
            harness.runner.state = ControllerState.from_dict(harness.runner.state.to_dict())
        elif case == "missing":
            binding = None
        elif case in ("installation", "session", "epoch"):
            field, value = {"installation": ("installation_id", "other"),
                            "session": ("owner_session_id", "other"),
                            "epoch": ("owner_epoch", binding.owner_epoch + 1)}[case]
            binding = replace(binding, **{field: value})
        elif case == "owner_expired":
            binding = replace(binding, owner_not_after_ms=NOW + 1000)
        elif case.startswith("hold_") or case == "future":
            # A short hold inside the same profile isolates expiry from a plan edge.
            from dao.prog.fastctrl.policy import apply_execution_outcome
            from dao.prog.fastctrl.receiver import payload_digest

            command = (replace(command, issued_at_ms=NOW + 1000) if case == "future" else
                       replace(command, execute_before_ms=NOW + 500, hold_until_ms=NOW + 1000))
            receipt = replace(harness.owner.state.last_receipt, accepted_ts=command.issued_at_ms,
                              payload_digest=payload_digest(command))
            assert apply_execution_outcome(controller, outcome="accepted", pending=command,
                                           receipt=receipt)
            delta = {"hold_before": 999, "hold_at": 1000, "hold_after": 1001, "future": 999}[case]
            timestamp = (NOW + delta) / 1000
        elif case == "pending":
            controller.pending_command = command
        elif case == "recovery":
            controller.recovery_required = True
        follow = propose(harness.plan, harness.runner.state, timestamp=timestamp,
                         owner_binding=binding)
        assert follow.batteries[0].write is (case != "hold_before")
        assert harness.runner.state.battery(0).last_accepted_command == command
        if case == "recovery":
            assert controller.recovery_required is True
    finally:
        harness.lease.close()


@pytest.mark.parametrize("path", ["normal", "fallback", "release"])
def test_i4_unknown_zero_is_a_proposal(tmp_path, path):
    harness = build(tmp_path, plan=financial_plan(powers=(0.0,)))
    try:
        policy = FastControlPolicy(PolicyLimits(deadband=5000, min_command_interval=600))
        controller = harness.runner.state.battery(0)
        controller.last_change_ts = NOW_TS
        sample = Measurement(NOW_TS, 0, [BatteryMeasurement(soc=50, power_w=0)])
        if path == "release":
            decision = policy.release(harness.plan, sample, harness.runner.state, "mode_off")
        else:
            decision = policy.decide(harness.plan, replace(sample, grid_valid=path == "normal"),
                                     harness.runner.state)
        assert decision.batteries[0].setpoint_w == 0.0
        assert decision.batteries[0].write is True
    finally:
        harness.lease.close()


@pytest.mark.parametrize("check", ["account", "demand", "benefit"])
@pytest.mark.parametrize("context", ["live", "stopped", "loaded", "missing", "session", "digest", "expired"])
def test_i4_stale_history_books_no_benefit_or_held_fallback(tmp_path, context, check):
    harness = build(tmp_path)
    try:
        policy = FastControlPolicy(PolicyLimits())
        initial = Measurement(NOW_TS, 5000, [BatteryMeasurement(soc=50, power_w=None)])
        policy.account(harness.runner.state, harness.plan, initial, "synthetic",
                       owner_binding=harness.owner.binding)
        receipts = consume(harness, propose(harness.plan, harness.runner.state, grid_w=5000))
        assert receipts[0].status == "accepted"
        binding = harness.owner.binding
        live = context == "live"
        if context == "stopped":
            barrier = stop(harness, reason="operator_stop")
            harness.runner._apply_stop_barrier(barrier, now_ms=NOW)
            binding = harness.owner.binding
        elif context == "loaded":
            harness.runner.state = ControllerState.from_dict(harness.runner.state.to_dict())
        elif context == "missing":
            binding = None
        elif context == "session":
            binding = replace(binding, owner_session_id="other-session")
        elif context == "digest":
            harness.plan = replace(harness.plan, price_average=0.31)
        elif context == "expired":
            binding = replace(binding, owner_not_after_ms=NOW + 60_000)
        sample = replace(initial, timestamp=NOW_TS + 60)
        policy.account(harness.runner.state, harness.plan, sample, "synthetic",
                       owner_binding=binding)
        decision = policy.decide(harness.plan, sample, harness.runner.state,
                                 owner_binding=binding)
        observed = harness.runner._capture_economic_context(harness.plan, sample, mode="active", previous=None)
        harness.runner._book_saving(previous=None, observed=observed, now=sample.timestamp)
        controller = harness.runner.state.battery(0)
        if check == "account":
            assert (controller.daily_deviation_kwh > 0) is live
        elif check == "demand":
            if live:
                assert decision.house_w == sample.grid_w - controller.last_accepted_command.setpoint_w
                assert decision.reason != "execution_unknown"
            else:
                assert decision.reason == "execution_unknown"
                assert decision.batteries[0].setpoint_w == PLAN_W
                assert decision.override is False
        else:
            assert harness.runner.state.saved_today_eur is None
            assert decision.economics_reasons == ("tick_settlement_basis_unavailable",)
    finally:
        harness.lease.close()


@pytest.mark.parametrize("measured_valid", [False, True])
def test_i4_measurements_remain_evidence_after_acceptance_expires(tmp_path, measured_valid):
    harness = build(tmp_path)
    try:
        policy = FastControlPolicy(PolicyLimits())
        sample = Measurement(NOW_TS, 5000, [BatteryMeasurement(soc=50, power_w=-3000,
                                                             valid=measured_valid)])
        policy.account(harness.runner.state, harness.plan, sample, "synthetic")
        consume(harness, propose(harness.plan, harness.runner.state, grid_w=5000))
        barrier = stop(harness, reason="operator_stop")
        harness.runner._apply_stop_barrier(barrier, now_ms=NOW)
        sample = replace(sample, timestamp=NOW_TS + 60)
        policy.account(harness.runner.state, harness.plan, sample, "synthetic")
        controller = harness.runner.state.battery(0)
        assert controller.daily_deviation_kwh == pytest.approx(1 / 60 if measured_valid else 0)
        decision = policy.decide(harness.plan, sample, harness.runner.state)
        if measured_valid:
            assert decision.house_w == 8000
            assert decision.reason != "execution_unknown"
        else:
            assert decision.reason == "execution_unknown"
        assert decision.benefit_eur_h is None
        assert decision.economics_reasons == ("tick_settlement_basis_unavailable",)
    finally:
        harness.lease.close()


@pytest.mark.parametrize("outcome", ["unknown", "unconfirmed", "stopped"])
def test_i4_execution_outcomes_invalidate_runtime_only(tmp_path, outcome):
    from dao.prog.fastctrl.policy import apply_execution_outcome, current_execution_command

    harness = build(tmp_path)
    try:
        consume(harness, propose(harness.plan, harness.runner.state))
        controller = harness.runner.state.battery(0)
        command = controller.last_accepted_command
        assert apply_execution_outcome(controller, outcome=outcome)
        assert current_execution_command(controller, binding=harness.owner.binding, now_ms=NOW,
                                          expected_digest=command.plan_digest) is None
        assert controller._current_accepted_command is None
        assert controller.last_accepted_command == command
        assert controller.last_command_w == controller.held_setpoint_w == PLAN_W
    finally:
        harness.lease.close()


@pytest.mark.parametrize("boundary", ["full", "issued_late", "accepted_late", "hold_expired"])
def test_i4_accounting_requires_the_whole_acceptance_segment(tmp_path, boundary):
    from dao.prog.fastctrl.policy import apply_execution_outcome
    from dao.prog.fastctrl.receiver import payload_digest

    harness = build(tmp_path)
    try:
        state = harness.runner.state
        sample = Measurement(NOW_TS, 5000, [BatteryMeasurement(soc=50, power_w=None)])
        policy = FastControlPolicy(PolicyLimits())
        policy.account(state, harness.plan, sample, "synthetic", owner_binding=harness.owner.binding)
        receipt = consume(harness, propose(harness.plan, state, grid_w=5000))[0]
        controller = state.battery(0)
        command = controller.last_accepted_command
        if boundary == "issued_late":
            command = replace(command, issued_at_ms=NOW + 1000)
        elif boundary == "hold_expired":
            command = replace(command, execute_before_ms=NOW + 1000, hold_until_ms=NOW + 30_000)
        receipt = replace(receipt, payload_digest=payload_digest(command),
                          accepted_ts=NOW + 1000 if boundary.endswith("late") else NOW)
        assert apply_execution_outcome(controller, outcome="accepted", pending=command, receipt=receipt)
        sample = replace(sample, timestamp=NOW_TS + 60)
        policy.account(state, harness.plan, sample, "synthetic", owner_binding=harness.owner.binding)
        decision = policy.decide(harness.plan, sample, state, owner_binding=harness.owner.binding)
        observed = harness.runner._capture_economic_context(harness.plan, sample, mode="active", previous=None)
        harness.runner._book_saving(previous=None, observed=observed, now=sample.timestamp)
        assert (controller.daily_deviation_kwh > 0) is (boundary == "full")
        assert state.saved_today_eur is None
        assert decision.economics_reasons == ("tick_settlement_basis_unavailable",)
    finally:
        harness.lease.close()


def test_i4_booking_keeps_the_other_batterys_proven_segment(tmp_path):
    from dao.prog.fastctrl.policy import apply_execution_outcome

    harness = build(tmp_path, plan=financial_plan(batteries=2))
    try:
        receipts = consume(harness, propose(harness.plan, harness.runner.state, grid_w=20_000))
        assert [receipt.status for receipt in receipts] == ["accepted", "accepted"]
        apply_execution_outcome(harness.runner.state.battery(1), outcome="unknown")
        sample = Measurement(NOW_TS + 60, 20_000,
                             [BatteryMeasurement(soc=50, power_w=-5000) for _ in range(2)])
        decision = FastControlPolicy(PolicyLimits()).decide(
            harness.plan, sample, harness.runner.state, owner_binding=harness.owner.binding)
        assert [battery.override for battery in decision.batteries] == [True, True]
        assert decision.benefit_eur_h is decision.batteries[0].benefit_eur_h is None
        assert harness.runner.state.battery(0)._current_accepted_command is not None
        assert harness.runner.state.battery(1)._current_accepted_command is None
        observed = harness.runner._capture_economic_context(harness.plan, sample, mode="active", previous=None)
        harness.runner._book_saving(previous=None, observed=observed, now=sample.timestamp)
        assert harness.runner.state.saved_today_eur is None
        assert decision.economics_reasons == ("tick_settlement_basis_unavailable",)
    finally:
        harness.lease.close()


@pytest.mark.parametrize("evidence", ["controller", "receipt", "pending"])
def test_i4_new_digest_requires_rebind_before_reserving(tmp_path, evidence):
    harness = build(tmp_path)
    try:
        first = consume(harness, propose(harness.plan, harness.runner.state))[0]
        command = harness.runner.state.battery(0).last_accepted_command
        if evidence == "controller":
            harness.owner.state = replace(harness.owner.state, last_receipt=None)
        else:
            harness.runner.state = ControllerState()
            if evidence == "pending":
                harness.owner.record_pending_command(command)
                harness.owner.state = replace(harness.owner.state, last_receipt=None)
        other = financial_plan(powers=(-1200.0,))
        harness.path.write_text(json.dumps(other.to_dict()))
        before = harness.store.path.read_bytes()
        high_water = harness.owner._sequence_high_water
        next_owner_cycle(harness.owner)
        decision = propose(other, harness.runner.state, owner_binding=harness.owner.binding)
        receipts = harness.runner._consume_financial_actuation(
            decision, other, owner=harness.owner, now_ms=NOW)
        assert harness.runner.last_actuation.reason == "plan_rebind_required"
        assert receipts == ()
        assert harness.store.path.read_bytes() == before
        assert harness.owner._sequence_high_water == high_water
        assert harness.owner.binding is None
        assert harness.owner.recovery_required is True
        assert harness.owner.phase == "inhibited"
        assert len(harness.gateway.effect_log) == 3
        assert harness.gateway.effect_log[0][0] == first.command_id
    finally:
        harness.lease.close()


@pytest.mark.parametrize("action", ["start_failed", "stop_unconfirmed", "stop_no_writer", "blocked"])
def test_i4_lifecycle_withdraws_runtime_without_erasing_history(tmp_path, action):
    from dao.prog.fastctrl.policy import current_execution_command
    from dao.prog.fastctrl.publication import GateVerdict

    harness = build(tmp_path)
    try:
        consume(harness, propose(harness.plan, harness.runner.state))
        controller = harness.runner.state.battery(0)
        command = controller.last_accepted_command
        binding = harness.owner.binding
        if action == "start_failed":
            # The held writer makes a new runner start fail before receiver I/O.
            harness.runner.start_owner_session()
        elif action.startswith("stop_"):
            harness.gateway.safe_state_confirmed = False
            if action == "stop_unconfirmed":
                harness.runner._owner_writer = harness.lease
            harness.runner.stop_owner_session("operator_stop")
        else:
            harness.runner._consume_block(GateVerdict(False, "publication_blocked"), now_ts=NOW_TS)
        # Even the captured old binding cannot revive locally withdrawn runtime evidence.
        assert current_execution_command(controller, binding=binding, now_ms=NOW,
                                          expected_digest=command.plan_digest) is None
        assert controller.last_accepted_command == command
        assert controller.held_setpoint_w == PLAN_W
        assert controller.last_command_w == PLAN_W
        if action != "blocked":
            assert controller.execution_status == "accepted"
    finally:
        harness.lease.close()


@pytest.mark.parametrize("deadband", [0, 150])
@pytest.mark.parametrize("power", [-100.25, 100.25, 0.0])
@pytest.mark.parametrize("fallback", [False, True])
def test_i4_wire_baseline_keeps_fractional_proposal_without_resending(
        tmp_path, deadband, power, fallback):
    harness = build(tmp_path, plan=financial_plan(powers=(power,), economics_supported=False))
    try:
        limits = PolicyLimits(deadband=deadband)
        initial = propose(harness.plan, harness.runner.state, limits=limits)
        assert initial.batteries[0].setpoint_w == power
        receipt = consume(harness, initial)[0]
        controller = harness.runner.state.battery(0)
        assert controller.last_command_w == {-100.25: -100, 100.25: 100, 0.0: 0}[power]
        harness.gateway.advance_clock(1000)
        if fallback:
            sample = Measurement(NOW_TS + 1, 0, [BatteryMeasurement(soc=50)], grid_valid=False)
            follow = FastControlPolicy(limits).release(
                harness.plan, sample, harness.runner.state, "sensor_stale",
                owner_binding=harness.owner.binding)
        else:
            follow = propose(harness.plan, harness.runner.state, limits=limits, timestamp=NOW_TS + 1,
                             owner_binding=harness.owner.binding)
        assert follow.batteries[0].setpoint_w == power
        assert follow.batteries[0].write is False
        assert consume(harness, follow, now_ms=NOW + 1000) == ()
        assert controller.last_accepted_command.command_id == receipt.command_id
        assert len(harness.gateway.effect_log) == 3
    finally:
        harness.lease.close()


def test_i4_contextless_plan_still_reaches_explicit_refusal(tmp_path):
    harness = build(tmp_path)
    try:
        contextless = replace(harness.plan, financial_context=None)
        failure = None
        try:
            decision = propose(contextless, harness.runner.state)
        except ValueError as exc:
            failure = str(exc)
        assert failure is None, f"policy prevented the consumer's explicit refusal: {failure}"
        assert harness.runner._actuate(decision, contextless, "active") == ()
        assert harness.runner.last_actuation.reason == "missing_financial_context"
        assert harness.gateway.effect_log == []
    finally:
        harness.lease.close()


def test_i4_runtime_acceptance_cannot_be_restored_from_json(tmp_path):
    from dao.prog.fastctrl.policy import current_execution_command
    from dao.prog.fastctrl.runner import load_state

    harness = build(tmp_path)
    try:
        consume(harness, propose(harness.plan, harness.runner.state))
        controller = harness.runner.state.battery(0)
        command = controller.last_accepted_command
        payload = harness.runner.state.to_dict()
        assert "_current_accepted_command" not in payload["batteries"][0]
        payload["batteries"][0]["_current_accepted_command"] = command.to_dict()
        (harness.root / "fast_state.json").write_text(json.dumps(payload))
        loaded = load_state(harness.runner.state_path).battery(0)
        assert loaded.execution_status == "accepted"
        assert loaded.last_accepted_command == command
        assert current_execution_command(loaded, binding=harness.owner.binding, now_ms=NOW,
                                          expected_digest=command.plan_digest) is None
        # The predicate is pure and also rejects disagreement with history.
        controller.last_accepted_command = replace(command, command_id="different-history")
        before = controller.to_dict()
        assert current_execution_command(controller, binding=harness.owner.binding, now_ms=NOW,
                                          expected_digest=command.plan_digest) is None
        assert controller.to_dict() == before
    finally:
        harness.lease.close()


@pytest.mark.parametrize("meter", ["sandbox", "other-installation", "Sandbox", " sandbox",
                                    "sandbox "])
def test_i2_plan_meter_must_equal_owner_installation(tmp_path, monkeypatch, meter):
    plan = financial_plan(batteries=2)
    context = plan.financial_context
    plan = replace(plan, financial_context=replace(
        context, basis=replace(context.basis, meter_id=meter)))
    harness = build(tmp_path, plan=plan)
    try:
        decision = propose(harness.plan, harness.runner.state)
        assert [battery.write for battery in decision.batteries] == [True, True]
        before = harness.store.path.read_bytes()
        state_before = harness.owner.state
        calls = observe_receiver_calls(monkeypatch, harness.gateway)
        receipts = consume(harness, decision)
        if meter == INSTALLATION:
            assert [receipt.status for receipt in receipts] == ["accepted", "accepted"]
            assert [receipt.installation_id for receipt in receipts] == [meter, meter]
            assert calls.count("send") == 2 and "inspect" not in calls
            assert harness.runner.last_actuation.allowed is True
            assert harness.store.load_strict().last_reserved_sequence == 42
            return
        assert harness.runner.last_actuation.allowed is False
        assert harness.runner.last_actuation.reason == "installation_mismatch"
        assert receipts == ()
        assert calls == []
        assert harness.gateway.effect_log == []
        assert harness.store.path.read_bytes() == before
        assert harness.owner.state == state_before
        assert harness.owner.state.pending_command is None
        assert harness.owner._sequence_high_water == 40
        for controller in harness.runner.state.batteries:
            assert controller.pending_command is None
            assert controller.last_accepted_command is None
            assert controller.execution_status == "unknown"
    finally:
        harness.lease.close()


@pytest.mark.parametrize("session", ["session-new", "second-session"])
@pytest.mark.parametrize("lease_delta_ms", [-1000, 0, 60_000])
@pytest.mark.parametrize("crash_persist", [None, False, True])
def test_i3_consumed_barrier_cannot_rebind(
        tmp_path, monkeypatch, session, lease_delta_ms, crash_persist):
    """A refused rebind keeps A valid; only a real new barrier fences A out."""
    harness = build(tmp_path)
    gateway = harness.gateway
    real_send = gateway.send
    commands = []

    def reserve(command, *, deadline):
        commands.append(command)
        gateway.enqueue(command)
        return gateway.deliver()

    monkeypatch.setattr(gateway, "send", reserve)
    try:
        receipts = consume(harness, propose(harness.plan, harness.runner.state))
        old = commands[0]
        assert [receipt.status for receipt in receipts] == ["unconfirmed"]
        assert harness.owner.state.pending_command == old
        assert gateway.state.commands[0].phase == "reserved"
        assert gateway.effect_log == []
        binding = harness.owner.binding
        gateway.advance_clock(1)
        if crash_persist is not None:
            gateway.crash(persist=crash_persist)
            assert gateway.restart() == "recovered"
        before = gateway.state
        files = (gateway.state_path.read_bytes(), gateway.anchor_path.read_bytes())
        refused = None
        try:
            gateway.bind_owner(
                installation_id=binding.installation_id, barrier_id=binding.barrier_id,
                expected_epoch=binding.owner_epoch, owner_session_id=session,
                requested_not_after_ms=binding.owner_not_after_ms + lease_delta_ms,
                deadline=DEADLINE)
        except ValueError as exc:
            refused = str(exc)
        assert refused == "barrier_already_bound", "a consumed barrier granted a second binding"
        assert gateway.state == before
        assert gateway.state.owner_not_after_ms == binding.owner_not_after_ms
        assert (gateway.state_path.read_bytes(), gateway.anchor_path.read_bytes()) == files
        assert gateway.effect_log == []

        # Refusal did not revoke A: its still-valid reservation may write.
        partial = gateway.step_helper("setpoint", command=old)
        assert (partial.status, partial.effect) == ("unconfirmed", "partial")
        effects = [(old.command_id, "setpoint", "accu", PLAN_W)]
        assert gateway.effect_log == effects

        request = RevokeRequest(INSTALLATION, 8, "i3-barrier-b", "owner_start", gateway.utc_ms)
        barrier = gateway.revoke_and_stop(request, deadline=DEADLINE)
        assert (barrier.status, barrier.barrier_id, barrier.revoked_through_epoch,
                barrier.current_epoch, barrier.queues_fenced, barrier.safe_state) == (
            "accepted", "i3-barrier-b", 8, 9, True, "confirmed")
        fresh_binding = gateway.bind_owner(
            installation_id=INSTALLATION, barrier_id="i3-barrier-b", expected_epoch=9,
            owner_session_id="second-session", requested_not_after_ms=NOW + LEASE_MS,
            deadline=DEADLINE)
        assert fresh_binding.owner_epoch == 9
        late = gateway.step_helper("mode", command=old)
        assert (late.status, late.reason, late.effect) == ("unconfirmed", "epoch_withdrawn", "partial")
        assert late.helper_results == partial.helper_results
        assert gateway.effect_log == effects

        fresh = replace(old, owner_session_id="second-session", owner_epoch=9,
                        sequence=old.sequence + 1, command_id="i3-command-b")
        assert real_send(fresh, deadline=DEADLINE).status == "accepted"
        assert gateway.effect_log == effects + [
            ("i3-command-b", "setpoint", "accu", PLAN_W),
            ("i3-command-b", "mode", "accu", PLAN_W),
            ("i3-command-b", "stop_inverter", "accu", PLAN_W),
        ]
    finally:
        harness.lease.close()


def test_01_the_first_v2_plan_proposes_a_real_command_and_is_accepted(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    battery = decision.batteries[0]
    # A real proposal: the optimizer wrote nothing, so the plan power is a
    # command and not an already in-force fact.
    assert battery.write is True
    assert battery.setpoint_w == PLAN_W
    receipts = consume(harness, decision)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.status == "accepted"
    assert receipt.battery_name == "accu"
    controller = harness.runner.state.battery(0)
    assert controller.execution_status == "accepted"
    assert controller.pending_command is None
    assert controller.last_accepted_command is not None
    assert controller.last_accepted_command.command_id == receipt.command_id
    assert controller.last_command_w == PLAN_W
    assert controller.last_command_ts == receipt.accepted_ts / 1000.0
    assert controller.held_setpoint_w == PLAN_W
    assert controller.recovery_required is False
    # An accepted command is an in-force change; a proposal is not.
    assert controller.last_change_ts == receipt.accepted_ts / 1000.0
    verdict = harness.runner.last_actuation
    assert verdict.allowed is True
    assert verdict.reason == "accepted"


def test_02_following_the_plan_gives_no_second_command(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    first = consume(harness, decision)
    assert first[0].status == "accepted"
    controller = harness.runner.state.battery(0)
    accepted = (controller.last_command_w, controller.last_command_ts)
    harness.gateway.advance_clock(60_000)
    second_decision = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 60,
                              owner_binding=harness.owner.binding)
    assert second_decision.batteries[0].write is False
    second = consume(harness, second_decision, now_ms=NOW + 60_000)
    assert second == ()
    assert (controller.last_command_w, controller.last_command_ts) == accepted
    assert controller.execution_status == "accepted"
    assert len(harness.gateway.effect_log) == 3


def test_03_a_profile_change_and_a_new_plan_get_new_commands(tmp_path):
    harness = build(tmp_path, plan=financial_plan(powers=(PLAN_W, -1500.0)))
    decision = propose(harness.plan, harness.runner.state)
    first = consume(harness, decision)[0]
    controller = harness.runner.state.battery(0)
    assert controller.last_accepted_command.command_id == first.command_id
    # A profile transition inside the horizon is a new command, never a reuse.
    harness.gateway.advance_clock(900_000)
    transition = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 900,
                         owner_binding=harness.owner.binding)
    second = consume(harness, transition, now_ms=NOW + 900_000)[0]
    assert second.command_id != first.command_id
    assert second.sequence > first.sequence
    assert second.plan_digest == first.plan_digest
    assert controller.last_accepted_command.command_id == second.command_id
    assert controller.last_command_w == -1500.0
    # A new digest requires a real lifecycle barrier and a new binding first.
    other = financial_plan(powers=(-1200.0,))
    harness.path.write_text(json.dumps(other.to_dict()))
    harness.gateway.advance_clock(60_000)
    follow = propose(other, harness.runner.state, timestamp=NOW_TS + 960,
                     owner_binding=harness.owner.binding)
    before = harness.store.path.read_bytes()
    effects = list(harness.gateway.effect_log)
    assert harness.runner._consume_financial_actuation(
        follow, other, owner=harness.owner, now_ms=NOW + 960_000) == ()
    assert harness.runner.last_actuation.reason == "plan_rebind_required"
    assert harness.store.path.read_bytes() == before
    assert harness.gateway.effect_log == effects
    barrier = stop(harness, reason="plan_rebind_required")
    assert barrier.status == "accepted" and barrier.safe_state == "confirmed"
    harness.runner._apply_stop_barrier(barrier, now_ms=NOW + 960_000)
    binding = restart_i4_owner(harness)
    assert binding.owner_epoch > second.owner_epoch
    follow = propose(other, harness.runner.state, timestamp=NOW_TS + 960, owner_binding=binding)
    third = harness.runner._consume_financial_actuation(
        follow, other, owner=harness.owner, now_ms=NOW + 960_000)[0]
    assert third.plan_digest != first.plan_digest
    assert third.command_id != second.command_id
    assert third.sequence > second.sequence
    assert controller.last_accepted_command.command_id == third.command_id
    assert controller.last_command_w == -1200.0


def test_04_two_batteries_share_one_installation_budget(tmp_path, monkeypatch):
    harness = build(tmp_path, plan=financial_plan(batteries=2))
    boundary = import_module("dao.prog.fastctrl.execution")
    seen = []
    real = boundary.owner_section

    def spy(**kwargs):
        seen.append(kwargs["deadline"])
        return real(**kwargs)

    monkeypatch.setattr(boundary, "owner_section", spy)
    decision = propose(harness.plan, harness.runner.state)
    receipts = consume(harness, decision)
    assert [receipt.battery_name for receipt in receipts] == ["accu", "accu2"]
    sequences = [receipt.sequence for receipt in receipts]
    assert sequences[0] < sequences[1]
    # One absolute section window for the whole installation. Execution derives
    # the shorter transport cutoff from the same sample, not from each send.
    assert len({deadline.monotonic_end_ns for deadline in seen}) == 1
    assert len(seen) == 2
    assert seen[0].monotonic_end_ns == (
        harness.owner.now_ns() + boundary.SECTION_BUDGET_NS)
    assert seen[0].monotonic_end_ns != (
        harness.owner.now_ns() + boundary.TRANSPORT_BUDGET_NS)
    state = harness.gateway.state
    assert state.sequence_high_water == sequences[1]


def test_05_no_fictitious_acceptance_after_an_uncertain_outcome(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    harness.gateway.lose_ack()
    receipts = consume(harness, decision)
    assert [receipt.status for receipt in receipts] == ["unconfirmed"]
    controller = harness.runner.state.battery(0)
    assert controller.execution_status == "unconfirmed"
    assert controller.last_accepted_command is None
    assert controller.held_setpoint_w == 0.0
    assert controller.last_command_w == 0.0


def test_06_shadow_and_deadband_give_no_receipt_and_no_state_change(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    controller = harness.runner.state.battery(0)
    before = controller.to_dict()
    shadow = harness.runner._actuate(decision, harness.plan, "shadow")
    assert shadow == ()
    assert harness.runner.last_receipts == []
    assert harness.runner.last_actuation.allowed is False
    assert harness.runner.last_actuation.reason == "shadow"
    assert controller.to_dict() == before
    # A deadband tick has no proposal at all.
    deadband = propose(harness.plan, harness.runner.state)
    deadband.batteries[0].write = False
    assert consume(harness, deadband) == ()
    assert harness.runner.last_actuation.reason == "no_proposal"
    assert controller.to_dict() == before
    assert harness.gateway.effect_log == []


def test_07_an_uncertain_outcome_leaves_a_recovery_duty(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    harness.gateway.lose_ack()
    consume(harness, decision)
    controller = harness.runner.state.battery(0)
    assert controller.execution_status == "unconfirmed"
    assert controller.recovery_required is True
    assert controller.pending_command is not None
    effects = len(harness.gateway.effect_log)
    # No further command in the next tick either, not even a stop.
    harness.gateway.advance_clock(60_000)
    again = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 60)
    assert consume(harness, again, now_ms=NOW + 60_000) == ()
    assert harness.runner.last_actuation.reason == "recovery_required"
    assert len(harness.gateway.effect_log) == effects
    assert controller.pending_command is not None


def test_08_only_an_accepted_segment_books_a_benefit(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state, grid_w=5000.0)
    assert decision.batteries[0].override is True
    # Nothing is proven yet, so the deviation books no euro claim at all.
    assert decision.benefit_eur_h is None
    assert decision.economics_reasons == ("tick_settlement_basis_unavailable",)
    receipts = consume(harness, decision)
    assert receipts[0].status == "accepted"
    harness.gateway.advance_clock(60_000)
    proven = propose(harness.plan, harness.runner.state, grid_w=5000.0,
                     timestamp=NOW_TS + 60, owner_binding=harness.owner.binding)
    assert proven.batteries[0].override is True
    assert proven.benefit_eur_h is None
    assert proven.economics_reasons == ("tick_settlement_basis_unavailable",)


def test_09_an_unaccepted_segment_books_nothing_at_all(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state, grid_w=5000.0)
    harness.gateway.lose_ack()
    consume(harness, decision)
    controller = harness.runner.state.battery(0)
    # The uncertainty is visible and no saving is claimed for the segment.
    assert controller.execution_status == "unconfirmed"
    assert controller.recovery_required is True
    assert controller.last_command_w == 0.0
    assert harness.runner.state.saved_today_eur is None
    harness.gateway.advance_clock(60_000)
    again = propose(harness.plan, harness.runner.state, grid_w=5000.0,
                    timestamp=NOW_TS + 60)
    assert again.benefit_eur_h is None
    assert again.economics_reasons == ("tick_settlement_basis_unavailable",)


def test_10_a_diagnostics_overlap_is_refused_without_any_command(tmp_path):
    config = make_config(diagnostics={"entity setpoint": SETPOINT_ENTITY})
    harness = build(tmp_path, config=config)
    decision = propose(harness.plan, harness.runner.state)
    receipts = consume(harness, decision)
    assert receipts == ()
    verdict = harness.runner.last_actuation
    assert verdict.allowed is False
    assert verdict.reason == "diagnostics_overlap"
    assert harness.gateway.effect_log == []
    controller = harness.runner.state.battery(0)
    assert controller.execution_status == "unknown"
    assert harness.hass.values == harness.hass.options == {}


@pytest.mark.parametrize("entity", [SETPOINT_ENTITY, MODE_ENTITY, STOP_ENTITY])
def test_11_an_overlap_suppresses_the_diagnostics_publication(tmp_path, entity):
    """The diagnostic write may never alias an actuator entity."""
    config = make_config(diagnostics={"entity setpoint": entity})
    harness = build(tmp_path, config=config)
    decision = propose(harness.plan, harness.runner.state)
    assert consume(harness, decision) == ()
    assert harness.runner.last_actuation.reason == "diagnostics_overlap"
    # The publication of this tick is suppressed: not a single Home Assistant
    # write, not even for a diagnostic entity that does not overlap.
    harness.runner._publish(decision, "active", SimpleNamespace(grid_w=0.0, pv_w=None))
    assert harness.hass.values == harness.hass.options == {}
    assert harness.hass.switches == harness.hass.published == {}


@pytest.mark.parametrize("mode", ["off", "shadow", "active"])
def test_i6_public_v2_denies_even_own_nominal_receiver(tmp_path, monkeypatch, mode):
    harness = build(tmp_path, config=make_config(mode=mode))
    try:
        decision = propose(harness.plan, harness.runner.state)
        assert decision.batteries[0].write is True
        assert harness.owner.receiver is harness.gateway
        before = harness.store.path.read_bytes()
        calls = observe_receiver_calls(monkeypatch, harness.gateway)
        boundary = import_module("dao.prog.fastctrl.execution")

        def forbidden(*args, **kwargs):
            pytest.fail("the public v2 gate reached a mechanical boundary")

        monkeypatch.setattr(boundary, "owner_section", forbidden)
        monkeypatch.setattr(boundary, "send_locked", forbidden)
        assert harness.runner.tick(NOW_TS) is None
        assert harness.runner.last_gate.allowed is False
        assert harness.runner.last_gate.reason == "actuation_gates_unproven"
        assert harness.runner.last_actuation is None
        assert harness.runner.last_receipts == []
        assert calls == ["revoke_and_stop"] and harness.gateway.effect_log == []
        assert harness.store.path.read_bytes() != before
        assert harness.store.load_strict().phase == "released"
        assert harness.owner.state.pending_command is None
        assert harness.runner.state.battery(0).pending_command is None
        assert harness.hass.values == harness.hass.options == harness.hass.published == {}
        assert harness.hass.switches == {} and harness.hass.services == []
        assert harness.hass.template_calls == harness.hass.state_calls == 0
    finally:
        harness.lease.close()


def test_12_the_adapter_classification_is_nominal(tmp_path):
    module = actuator()
    from dao.prog.fastctrl.runner import HomeAssistantGateway

    assert module.classify_actuation_adapter(HomeAssistantGateway(FakeHass())) == (
        module.LEGACY_ONLY_ADAPTER)
    assert module.classify_actuation_adapter(object()) == module.LEGACY_ONLY_ADAPTER
    assert module.classify_actuation_adapter(None) == module.LEGACY_ONLY_ADAPTER
    harness = build(tmp_path)
    assert module.classify_actuation_adapter(harness.gateway) == module.FENCED_ADAPTER
    # The generic gateway has no command right at all.
    harness.owner.receiver = HomeAssistantGateway(FakeHass())
    decision = propose(harness.plan, harness.runner.state)
    assert consume(harness, decision) == ()
    assert harness.runner.last_actuation.reason == "adapter_unqualified"
    assert harness.gateway.effect_log == []


def test_13_only_an_exact_barrier_ends_the_recovery_duty(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    harness.gateway.lose_ack()
    consume(harness, decision)
    controller = harness.runner.state.battery(0)
    assert controller.recovery_required is True
    # A late or wrong acknowledgement never lifts the recovery duty.
    wrong = ActuationReceipt(installation_id=INSTALLATION, battery_name="accu",
                             owner_epoch=8, sequence=1, command_id="wrong",
                             plan_digest="a" * 64, accepted_ts=NOW, status="accepted",
                             helper_results=(), owner_session_id="session-new",
                             payload_digest="b" * 64, reason="accepted",
                             observed_at_ms=NOW, effect="complete")
    policy = import_module("dao.prog.fastctrl.policy")
    assert policy.apply_execution_outcome(controller, outcome="accepted", receipt=wrong,
                                          pending=controller.pending_command) is False
    assert controller.recovery_required is True
    assert controller.execution_status == "unconfirmed"
    # Only an exact accepted safe barrier ends the duty.
    barrier = stop(harness, reason="operator_stop")
    assert barrier.status == "accepted"
    harness.runner._apply_stop_barrier(barrier, now_ms=harness.owner.now_ms())
    assert controller.execution_status == "stopped"
    assert controller.recovery_required is False
    assert controller.pending_command is None


def test_14_a_refusal_with_a_pending_command_is_an_uncertain_outcome(tmp_path, monkeypatch):
    """An ExecutionRefused is an execution outcome, never a silent end."""
    harness = build(tmp_path)
    boundary = import_module("dao.prog.fastctrl.execution")

    def refuse(*args, **kwargs):
        raise boundary.ExecutionRefused("deadline_exceeded")

    monkeypatch.setattr(boundary, "send_locked", refuse)
    decision = propose(harness.plan, harness.runner.state)
    receipts = consume(harness, decision)
    assert receipts == ()
    verdict = harness.runner.last_actuation
    assert verdict.allowed is False
    assert verdict.reason == "deadline_exceeded"
    controller = harness.runner.state.battery(0)
    # The owner's own mirror holds the pending command: the effect is uncertain.
    assert harness.owner.state.pending_command is not None
    assert controller.execution_status == "unconfirmed"
    assert controller.recovery_required is True
    assert controller.pending_command is not None
    assert controller.held_setpoint_w == 0.0
    # The next tick refuses: the uncertain command is never replaced.
    harness.gateway.advance_clock(60_000)
    again = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 60)
    assert consume(harness, again, now_ms=NOW + 60_000) == ()
    assert harness.runner.last_actuation.reason == "recovery_required"


def test_15_a_refusal_without_a_pending_command_proves_no_effect(tmp_path, monkeypatch):
    """No pending command in the mirror means no effect is proven."""
    harness = build(tmp_path, lease_ms=600_000)
    boundary = import_module("dao.prog.fastctrl.execution")
    decision = propose(harness.plan, harness.runner.state)
    # The owner lease has expired: the section refuses before it records
    # anything, so no command can have reached the device.
    harness.gateway.advance_clock(600_000)
    receipts = consume(harness, decision, now_ms=NOW + 600_000)
    assert receipts == ()
    verdict = harness.runner.last_actuation
    assert verdict.allowed is False
    assert verdict.reason != "accepted"
    controller = harness.runner.state.battery(0)
    assert harness.owner.state.pending_command is None
    assert controller.execution_status == "unknown"
    assert controller.recovery_required is False
    assert controller.pending_command is None
    assert boundary.ExecutionRefused is not None


def test_16_the_verdict_needs_every_proposed_battery_to_be_accepted(tmp_path, monkeypatch):
    """One accepted battery does not make the installation accepted."""
    harness = build(tmp_path, plan=financial_plan(batteries=2))
    boundary = import_module("dao.prog.fastctrl.execution")
    real = boundary.send_locked
    calls = []

    def refuse_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise boundary.ExecutionRefused("deadline_exceeded")
        return real(*args, **kwargs)

    monkeypatch.setattr(boundary, "send_locked", refuse_second)
    decision = propose(harness.plan, harness.runner.state)
    receipts = consume(harness, decision)
    assert [receipt.status for receipt in receipts] == ["accepted"]
    verdict = harness.runner.last_actuation
    assert verdict.allowed is False
    assert verdict.reason == "deadline_exceeded"
    assert harness.runner.state.battery(0).execution_status == "accepted"
    assert harness.runner.state.battery(1).execution_status == "unconfirmed"


def test_17_the_proposal_is_bound_to_the_fresh_publication(tmp_path):
    """The publication under the lock decides what may be sent."""
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    # A different plan is published while the proposal was made.
    other = financial_plan(powers=(-1200.0,))
    harness.path.write_text(json.dumps(other.to_dict()))
    receipts = consume(harness, decision)
    assert receipts == ()
    verdict = harness.runner.last_actuation
    assert verdict.allowed is False
    assert verdict.reason == "plan_digest_mismatch"
    assert harness.gateway.effect_log == []
    assert harness.runner.state.battery(0).execution_status == "unknown"


def test_18_a_foreign_or_late_barrier_never_ends_the_recovery_duty(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    harness.gateway.lose_ack()
    consume(harness, decision)
    controller = harness.runner.state.battery(0)
    barrier = stop(harness, reason="operator_stop")
    pending = controller.pending_command
    assert pending is not None
    for wrong in (replace(barrier, barrier_id="other-barrier"),
                  replace(barrier, installation_id="other-installation"),
                  replace(barrier, receiver_generation="other-generation"),
                  replace(barrier, accepted_ts=pending.issued_at_ms - 1),
                  replace(barrier, accepted_ts=NOW + 10_000_000)):
        harness.runner._apply_stop_barrier(wrong, now_ms=harness.owner.now_ms())
        assert controller.recovery_required is True
        assert controller.execution_status == "unconfirmed"
        assert controller.pending_command is not None


def test_19_the_exact_barrier_ends_the_duty_whatever_the_status(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    harness.gateway.lose_ack()
    consume(harness, decision)
    controller = harness.runner.state.battery(0)
    # An open duty with an unknown status is still an open duty.
    controller.execution_status = "unknown"
    other = harness.runner.state.battery(1)
    other.execution_status = "accepted"
    barrier = stop(harness, reason="operator_stop")
    harness.runner._apply_stop_barrier(barrier, now_ms=harness.owner.now_ms())
    assert controller.execution_status == "stopped"
    assert controller.recovery_required is False
    assert controller.pending_command is None
    # A battery without an open duty stays untouched.
    assert other.execution_status == "accepted"


def test_20_every_refusal_without_execution_applies_the_unknown_row(tmp_path):
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    receipts = consume(harness, decision)
    assert receipts[0].status == "accepted"
    controller = harness.runner.state.battery(0)
    assert controller.execution_status == "accepted"
    # A refusal that means "nothing was executed" clears the old status.
    denied = harness.runner._consume_financial_actuation(
        decision, harness.plan, owner=None, now_ms=NOW)
    assert denied == ()
    assert harness.runner.last_actuation.reason == "owner_unavailable"
    assert controller.execution_status == "unknown"
    assert controller.recovery_required is False
    assert controller.pending_command is None
    assert controller.last_accepted_command is not None


def test_21_the_pending_command_observation_is_installed_under_the_lock(tmp_path, monkeypatch):
    """The observation exists only while the lease is held, and is restored."""
    harness = build(tmp_path)
    owner = harness.owner
    runner_module = import_module("dao.prog.fastctrl.ownership")
    real_lock = runner_module.publication_lock
    seen = []

    @contextmanager
    def spy(path):
        seen.append("record_pending_command" in owner.__dict__)
        with real_lock(path) as lease:
            yield lease

    monkeypatch.setattr(runner_module, "publication_lock", spy)
    decision = propose(harness.plan, harness.runner.state)
    consume(harness, decision)
    assert seen == [False]
    # The class method is restored exactly: no instance attribute is left.
    assert "record_pending_command" not in owner.__dict__

    # An earlier instance attribute is put back, not removed.
    def sentinel(command):
        return None

    owner.record_pending_command = sentinel
    next_owner_cycle(owner)
    consume(harness, decision)
    assert owner.record_pending_command is sentinel
    del owner.record_pending_command


def test_22_a_receipt_of_another_identity_changes_nothing(tmp_path):
    """A receipt of another session or payload is no proof of this command."""
    from dataclasses import replace as replace_receipt

    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    receipt = consume(harness, decision)[0]
    controller = harness.runner.state.battery(0)
    pending = controller.last_accepted_command
    assert pending is not None
    # Rewind the controller to the moment before the acceptance.
    controller.held_setpoint_w = 0.0
    controller.last_command_w = 0.0
    controller.last_accepted_command = None
    controller.execution_status = "unknown"
    controller.pending_command = pending
    for wrong in (replace_receipt(receipt, owner_session_id="other-session"),
                  replace_receipt(receipt, payload_digest="b" * 64)):
        harness.runner._record_financial_outcome(decision.batteries[0], wrong, pending)
        assert controller.execution_status == "unknown"
        assert controller.last_accepted_command is None
        assert controller.held_setpoint_w == 0.0
        assert controller.last_command_w == 0.0


def changed_two_battery_proposal(harness):
    """Start with real acceptance, then propose the next profile segment."""
    first = consume(harness, propose(harness.plan, harness.runner.state))
    assert [receipt.status for receipt in first] == ["accepted", "accepted"]
    harness.gateway.advance_clock(900_000)
    decision = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 900)
    assert [battery.write for battery in decision.batteries] == [True, True]
    assert [battery.setpoint_w for battery in decision.batteries] == [-1500.0, -1500.0]
    return decision


@pytest.mark.parametrize("field,value", RECEIPT_IDENTITY_MISMATCHES)
@pytest.mark.parametrize("status", ["accepted", "rejected", "unconfirmed"])
@pytest.mark.parametrize("when", ["send", "retry", "query"])
def test_i1_foreign_ack_keeps_pending_and_denies_next_battery(
        tmp_path, monkeypatch, field, value, status, when):
    """Identity must be checked before any retry, history write or next battery."""
    harness = build(tmp_path, plan=financial_plan(batteries=2, powers=(PLAN_W, -1500.0)))
    try:
        decision = changed_two_battery_proposal(harness)
        before = [state.to_dict() for state in harness.runner.state.batteries]
        gateway = harness.gateway
        real_send, real_query = gateway.send, gateway.query
        calls, commands, mirrors = [], [], []
        if status == "rejected":
            gateway.helper_ok["setpoint"] = False
        elif status == "unconfirmed":
            gateway.lose_ack()

        def finish_reserved(command):
            parts = ("setpoint",) if status == "rejected" else (
                "setpoint", "mode", "stop_inverter")
            for part in parts:
                gateway.step_helper(part, command=command)

        def send(command, *, deadline):
            calls.append("send")
            commands.append(command)
            mirrors.append(harness.store.path.read_bytes())
            if len(commands) == 1 and when != "send":
                gateway.enqueue(command)
                return gateway.deliver()
            if when == "retry" and len(commands) == 2:
                finish_reserved(command)
            receipt = real_send(command, deadline=deadline)
            if (when == "send" and len(commands) == 1) or (
                    when == "retry" and len(commands) == 2):
                assert receipt.status == status
                return replace(receipt, **{field: value})
            return receipt

        def query(**kwargs):
            calls.append("query")
            if when == "query":
                finish_reserved(commands[0])
            receipt = real_query(**kwargs)
            assert receipt.status == status
            return replace(receipt, **{field: value})

        monkeypatch.setattr(gateway, "send", send)
        monkeypatch.setattr(gateway, "query", query)
        receipts = consume(harness, decision, now_ms=NOW + 900_000)

        assert harness.runner.last_actuation.allowed is False
        assert harness.runner.last_actuation.reason == "receipt_identity_mismatch"
        assert receipts == ()
        assert calls == {"send": ["send"], "retry": ["send", "send"],
                         "query": ["send", "send", "query"]}[when]
        assert all(command == commands[0] for command in commands)
        assert harness.owner.state.pending_command == commands[0]
        assert harness.store.path.read_bytes() == mirrors[0]
        controller = harness.runner.state.battery(0)
        assert controller.to_dict() == before[0] | {
            "execution_status": "unconfirmed", "recovery_required": True,
            "pending_command": commands[0].to_dict(),
        }
        assert harness.runner.state.battery(1).to_dict() == before[1]
    finally:
        harness.lease.close()


def test_i1_unapplied_acceptance_restores_pending_inside_the_same_lease(tmp_path, monkeypatch):
    """An accepted status without policy acceptance cannot allow the installation."""
    harness = build(tmp_path, plan=financial_plan(batteries=2, powers=(PLAN_W, -1500.0)))
    boundary = import_module("dao.prog.fastctrl.execution")
    real_section, real_write = boundary.owner_section, harness.store.write
    sections, restored = [], []

    def foreign_return(**kwargs):
        sections.append(kwargs["lease"])
        receipt = real_section(**kwargs)
        assert harness.owner.state.pending_command is None
        return replace(receipt, owner_session_id="other-session")

    def write(state, *, writer):
        if (state.pending_command is not None and state.last_receipt is not None
                and state.pending_command.command_id == state.last_receipt.command_id):
            sections[-1]._check()
            restored.append(state.pending_command)
        real_write(state, writer=writer)

    try:
        decision = changed_two_battery_proposal(harness)
        before = [state.to_dict() for state in harness.runner.state.batteries]
        monkeypatch.setattr(boundary, "owner_section", foreign_return)
        monkeypatch.setattr(harness.store, "write", write)
        receipts = consume(harness, decision, now_ms=NOW + 900_000)

        assert harness.runner.last_actuation.allowed is False
        assert harness.runner.last_actuation.reason == "receipt_not_applied"
        assert [receipt.status for receipt in receipts] == ["accepted"]
        assert len(sections) == len(restored) == 1
        pending = restored[0]
        assert pending.battery_name == "accu"
        assert harness.store.load_strict().pending_command == pending
        assert harness.runner.state.battery(0).to_dict() == before[0] | {
            "execution_status": "unconfirmed", "recovery_required": True,
            "pending_command": pending.to_dict(),
        }
        assert decision.batteries[0].actuation_status == "unconfirmed"
        assert harness.runner.state.battery(1).to_dict() == before[1]
        assert [entry[2] for entry in harness.gateway.effect_log[6:]] == ["accu"] * 3
    finally:
        harness.lease.close()


@pytest.mark.parametrize("field,value", RECEIPT_IDENTITY_MISMATCHES)
def test_i1_foreign_acceptance_is_a_pure_policy_noop(tmp_path, field, value):
    from dao.prog.fastctrl.policy import apply_execution_outcome

    harness = build(tmp_path)
    try:
        decision = propose(harness.plan, harness.runner.state)
        receipt = consume(harness, decision)[0]
        controller = harness.runner.state.battery(0)
        before = controller.to_dict()
        assert apply_execution_outcome(
            controller, outcome="accepted", receipt=replace(receipt, **{field: value}),
            pending=controller.last_accepted_command) is False
        assert controller.to_dict() == before
    finally:
        harness.lease.close()


def test_23_one_writer_wins_and_the_second_one_is_refused(tmp_path):
    """The successful first writer and the refused second one, in one test."""
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    receipts = consume(harness, decision)
    assert [receipt.status for receipt in receipts] == ["accepted"]
    assert harness.gateway.effect_log
    ownership = import_module("dao.prog.fastctrl.ownership")
    second = ownership.OwnerCoordinator(harness.store, receiver=harness.gateway)
    with pytest.raises(task_execution.LeaseBusy):
        task_execution.acquire_resource(RESOURCE, lease_dir=tmp_path / "leases")
    refused = harness.runner._consume_financial_actuation(
        decision, harness.plan, owner=second, now_ms=NOW)
    assert refused == ()
    verdict = harness.runner.last_actuation
    assert verdict.allowed is False
    assert verdict.reason == "owner_unavailable"
    assert len(harness.gateway.effect_log) == 3


@pytest.mark.parametrize("outcome", ["confirmed", "rejected", "unconfirmed", "stop", "restart"])
def test_f3b6_policy_is_proposal_and_v2_tick_is_unknown(tmp_path, outcome):
    harness = build(tmp_path)
    try:
        decision = propose(harness.plan, harness.runner.state, grid_w=5000)
        assert decision.benefit_eur_h is None
        assert decision.economics_reasons == ("tick_settlement_basis_unavailable",)
        if outcome == "rejected":
            harness.gateway.helper_ok["setpoint"] = False
        elif outcome == "unconfirmed":
            harness.gateway.lose_ack()
        receipts = consume(harness, decision)
        if outcome in ("confirmed", "stop", "restart"):
            assert receipts[0].status == "accepted"
            assert harness.runner.state.battery(0)._current_accepted_command is not None
        if outcome == "stop":
            barrier = stop(harness, reason="operator_stop")
            harness.runner._apply_stop_barrier(barrier, now_ms=NOW)
        elif outcome == "restart":
            harness.runner.state = ControllerState.from_dict(harness.runner.state.to_dict())
        if outcome != "confirmed":
            assert harness.runner.state.battery(0)._current_accepted_command is None
        sample = Measurement(NOW_TS, 5000, [BatteryMeasurement(soc=50, power_w=0)])
        context = harness.runner._capture_economic_context(
            harness.plan, sample, mode="active", previous=None, decision=decision)
        if outcome == "confirmed":
            command = harness.runner.state.battery(0)._current_accepted_command
            assert context.execution_references == (("accu", command.command_id),)
            assert context.batteries[0].effective_w == command.setpoint_w
        else:
            assert context.execution_references == (("accu", None),)
        assert decision.benefit_eur_h is None
        assert all(b.benefit_eur_h is None for b in decision.batteries)
        assert harness.runner.state.saved_today_eur is None
    finally:
        harness.lease.close()


def test_24_the_real_euro_bookkeeping_follows_the_acceptance(tmp_path):
    """The bookkeeping of the runner books a proven segment and nothing else."""
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state, grid_w=5000.0)
    # Nothing is proven: the real booking books nothing.
    sample = Measurement(NOW_TS, 5000, [BatteryMeasurement(soc=50, power_w=0)])
    observed = harness.runner._capture_economic_context(harness.plan, sample, mode="active", previous=None)
    harness.runner._book_saving(previous=None, observed=observed, now=NOW_TS)
    assert decision.benefit_eur_h is None
    assert harness.runner.state.saved_today_eur is None
    receipts = consume(harness, decision)
    assert receipts[0].status == "accepted"
    harness.gateway.advance_clock(60_000)
    proven = propose(harness.plan, harness.runner.state, grid_w=5000.0,
                     timestamp=NOW_TS + 60, owner_binding=harness.owner.binding)
    assert proven.benefit_eur_h is None
    current = harness.runner._capture_economic_context(
        harness.plan, replace(sample, timestamp=NOW_TS + 60), mode="active", previous=observed)
    harness.runner._book_saving(previous=observed, observed=current, now=NOW_TS + 60)
    assert harness.runner.state.saved_today_eur is None
    assert proven.economics_reasons == ("tick_settlement_basis_unavailable",)


def test_25_the_barrier_is_compared_field_by_field(tmp_path):
    """Only the own, exact and timely confirmed barrier ends the recovery duty."""
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    harness.gateway.lose_ack()
    consume(harness, decision)
    controller = harness.runner.state.battery(0)
    # An open duty without a controller pending command.
    controller.pending_command = None
    controller.recovery_required = True
    barrier = stop(harness, reason="operator_stop")
    # The own barrier, but seen with an earlier clock: its accepted_ts is in the
    # future, so it proves nothing yet.
    harness.runner._apply_stop_barrier(barrier, now_ms=NOW - 1)
    assert controller.recovery_required is True
    assert controller.execution_status == "unconfirmed"
    # A receipt that differs from the owner's own barrier receipt in any field
    # is not the owner's barrier.
    harness.runner._apply_stop_barrier(replace(barrier, accepted_ts=NOW - 1),
                                      now_ms=harness.owner.now_ms())
    assert controller.recovery_required is True
    assert controller.execution_status == "unconfirmed"
    # The own, exact and timely confirmed barrier ends the duty.
    harness.runner._apply_stop_barrier(barrier, now_ms=harness.owner.now_ms())
    assert controller.recovery_required is False
    assert controller.execution_status == "stopped"
    assert controller.pending_command is None


@pytest.mark.parametrize("situation", ["no_owner", "unqualified_adapter", "shadow"])
def test_26_an_overlap_suppresses_every_financial_tick(tmp_path, situation):
    """Every financial tick with an overlap writes nothing to Home Assistant."""
    from dao.prog.fastctrl.runner import HomeAssistantGateway

    config = make_config(diagnostics={"entity setpoint": SETPOINT_ENTITY})
    harness = build(tmp_path, config=config)
    decision = propose(harness.plan, harness.runner.state)
    if situation == "no_owner":
        harness.runner._consume_financial_actuation(
            decision, harness.plan, owner=None, now_ms=NOW)
    elif situation == "unqualified_adapter":
        harness.owner.receiver = HomeAssistantGateway(FakeHass())
        harness.runner._consume_financial_actuation(
            decision, harness.plan, owner=harness.owner, now_ms=NOW)
    else:
        harness.runner._actuate(decision, harness.plan, "shadow")
    # The publication of this tick is suppressed: not a single Home Assistant
    # write, not even for a diagnostic entity that does not overlap.
    harness.runner._publish(decision, "active", SimpleNamespace(grid_w=0.0, pv_w=None))
    assert harness.hass.values == harness.hass.options == {}
    assert harness.hass.switches == harness.hass.published == {}
    assert harness.gateway.effect_log == []
    # The overlap wins from every other refusal of the financial route.
    assert harness.runner.last_actuation.reason == "diagnostics_overlap"


def test_27_the_observation_never_bypasses_the_durable_storage(tmp_path):
    """The wrapper calls the class method; the earlier attribute returns."""
    harness = build(tmp_path)
    owner = harness.owner
    called = []

    def earlier(command):
        called.append(command)

    owner.record_pending_command = earlier
    decision = propose(harness.plan, harness.runner.state)
    harness.gateway.lose_ack()
    consume(harness, decision)
    controller = harness.runner.state.battery(0)
    # The durable storage happened through the class method: the owner's own
    # mirror holds the pending command of the uncertain command.
    assert owner.state.pending_command is not None
    assert owner.state.pending_command.command_id == controller.pending_command.command_id
    # The earlier instance attribute was never called ...
    assert called == []
    # ... and it is present again after the tick.
    assert owner.record_pending_command is earlier
    del owner.record_pending_command


def test_28_the_overlap_suppresses_even_a_contextless_refusal(tmp_path):
    """The overlap suppresses the publication whatever the reason is."""
    config = make_config(diagnostics={"entity setpoint": SETPOINT_ENTITY})
    harness = build(tmp_path, config=config)
    # A contextless v2 plan with an overlapping diagnostic.
    plan = replace(harness.plan, financial_context=None)
    decision = propose(plan, harness.runner.state)
    receipts = harness.runner._actuate(decision, plan, "active")
    assert receipts == ()
    # The refusal reason keeps its precedence: the missing context comes first.
    assert harness.runner.last_actuation.reason == "missing_financial_context"
    # The publication is suppressed anyway: not a single Home Assistant write.
    harness.runner._publish(decision, "active", SimpleNamespace(grid_w=0.0, pv_w=None))
    assert harness.hass.values == harness.hass.options == {}
    assert harness.hass.switches == harness.hass.published == {}
    assert harness.gateway.effect_log == []


def test_29_the_legacy_route_has_no_identity_refusal(tmp_path):
    """The v1 route refuses no plan: the baseline allows every v1 plan.

    A plan without a creation moment keeps the identity ``v1:0`` and reaches
    the helper route, exactly as in the baseline. The refusal of a v2 plan
    without a replay basis stays covered by the actuator test.
    """
    from dao.prog.fastctrl.policy import BatteryDecision, Decision
    from dao.tests.prog.test_fast_runner import T0, F2bHass, make_plan

    path = tmp_path / "fast_plan.json"
    legacy = make_plan(str(path))
    legacy.created_ts = 0  # a plan without a creation moment stays allowed
    hass = F2bHass()
    runner = FastControlRunner(hass, make_config(), plan_path=str(path),
                               state_path=str(tmp_path / "fast_state.json"))
    decision = Decision(T0, [BatteryDecision(0, "accu", -1000, 0, True, True, "override")])
    runner._actuate(decision, legacy, "active")
    # The legacy route is reached: the helper is called and a legacy receipt
    # carries the identity of the plan.
    assert hass.attempts == ["setpoint"]
    assert len(runner.last_receipts) == 1
    assert runner.last_receipts[0].plan_identity == "v1:0"
    assert runner.last_receipts[0].status == "confirmed"
    # The financial route never ran: there is no financial verdict.
    assert runner.last_actuation is None


@pytest.mark.parametrize("phase,allowed", [(0, True), (999_999_999, True),
                                           (1_000_000_000, False),
                                           (2_000_000_000, False), (5_000_000_000, True)])
def test_i7_admission_phase_boundaries(tmp_path, phase, allowed):
    """A valid proposal cannot acquire/send during the publisher's phases."""
    harness = build(tmp_path)
    harness.owner.now_ns = lambda: 20_000_000_000 + phase
    try:
        before = harness.store.path.read_bytes()
        receipts = consume(harness, propose(harness.plan, harness.runner.state))
        assert bool(receipts) is allowed
        if allowed:
            assert receipts[0].status == "accepted"
        else:
            assert harness.runner.last_actuation.reason == "admission_deferred"
            assert harness.store.path.read_bytes() == before
            assert harness.gateway.effect_log == []
    finally:
        harness.lease.close()


def test_i7_one_final_send_or_stop_per_cycle(tmp_path):
    """Two batteries share one claim; another send or lifecycle stop cannot reuse it."""
    harness = build(tmp_path, plan=financial_plan(batteries=2))
    harness.owner.now_ns = lambda: 20_000_000_000
    try:
        decision = propose(harness.plan, harness.runner.state)
        assert [r.status for r in consume(harness, decision)] == ["accepted", "accepted"]
        before = harness.store.path.read_bytes()
        assert consume(harness, decision) == ()
        assert harness.runner.last_actuation.reason == "admission_cycle_used"
        assert harness.store.path.read_bytes() == before
        assert harness.owner.stop(reason="shutdown") is None
        assert harness.owner.binding is None and harness.owner.recovery_required
        assert harness.owner.pending_stop_reason == "shutdown"
        assert harness.store.path.read_bytes() == before
        assert len(harness.gateway.effect_log) == 6
    finally:
        harness.lease.close()


@pytest.mark.parametrize("fault", ["rollback", "boot"])
def test_i7_fix1_consumer_clock_failure_after_receipt_write_keeps_recovery(tmp_path, monkeypatch, fault):
    from dao.tests.prog.test_fast_execution import i7_receipt_write_clock_fault

    harness = build(tmp_path, plan=financial_plan(batteries=2))
    runner, owner = harness.runner, harness.owner
    committed = i7_receipt_write_clock_fault(monkeypatch, owner, fault=fault)
    try:
        decision = propose(harness.plan, runner.state)
        assert consume(harness, decision) == ()
        assert not runner.last_actuation.allowed and runner.last_actuation.reason == "clock_unhealthy"
        assert len(committed) == 1
        command, written = committed[0]
        assert written.pending_command is None and written.last_receipt.status == "accepted"
        first = runner.state.battery(0)
        assert first.execution_status == "unconfirmed"
        assert first.pending_command == command and first.recovery_required
        assert first.last_accepted_command is None and first.held_setpoint_w == 0
        second = runner.state.battery(1)
        assert second.pending_command is second.last_accepted_command is None
        assert harness.gateway.effect_log == [(command.command_id, part, "accu", PLAN_W)
                                             for part in ("setpoint", "mode", "stop_inverter")]
        mirror = harness.store.load_strict()
        assert mirror.pending_command == owner.state.pending_command == command
        assert mirror.last_receipt == written.last_receipt
        assert mirror.phase == owner.phase == "inhibited"
        assert owner.binding is None and owner.recovery_required
        assert consume(harness, decision) == ()
        assert len(harness.gateway.effect_log) == 3 and len(committed) == 1
    finally:
        harness.lease.close()


def test_i7_start_lease_covers_next_send_cycle(tmp_path, monkeypatch):
    """Runner startup asks for the validated plan lease, not a one-second binding."""
    harness = build(tmp_path, start_owner=False)
    clock = [20_000_000_000]
    harness.owner.now_ns = lambda: clock[0]
    seen = []
    real_bind = harness.gateway.bind_owner

    def bind(**kwargs):
        seen.append(kwargs["requested_not_after_ms"])
        return real_bind(**kwargs)

    monkeypatch.setattr(harness.gateway, "bind_owner", bind)
    monkeypatch.setattr("time.time_ns", lambda: NOW * 1_000_000)
    try:
        harness.runner.start_owner_session()
        clock[0] = 25_000_000_000
        harness.runner.start_owner_session()
        assert seen == [harness.plan.financial_context.not_after_ts * 1000]
        assert harness.owner.binding.owner_not_after_ms > NOW + 6000
        clock[0] = 30_000_000_000
        assert consume(harness, propose(harness.plan, harness.runner.state))[0].status == "accepted"
    finally:
        if harness.runner._owner_writer is not None:
            harness.runner._owner_writer.close()


def test_i7_expiry_schedules_stop_and_fresh_start_before_send(tmp_path, monkeypatch):
    from dao.prog.fastctrl import runner as runner_module
    from dao.prog.fastctrl.publication import GateVerdict

    harness = build(tmp_path, lease_ms=20_000)
    runner, owner = harness.runner, harness.owner
    runner._owner_writer = harness.lease
    runner._owner_session_id = "replacement"
    harness.gateway.advance_clock(20_000)
    try:
        decision = propose(harness.plan, runner.state, timestamp=harness.gateway.utc_ms / 1000)
        assert consume(harness, decision, now_ms=harness.gateway.utc_ms) == ()
        assert owner.pending_stop_reason == "owner_lease_expired"
        assert owner.binding is None
        monkeypatch.setattr(runner_module, "following_verdict", lambda *a, **k: GateVerdict(True, "synthetic"))
        monkeypatch.setattr(runner.gateway, "read", lambda ids: {
            "sensor.p1_power": ("2000", harness.gateway.utc_ms / 1000),
            "sensor.soc": ("50", harness.gateway.utc_ms / 1000)})
        runner.tick(harness.gateway.utc_ms / 1000)
        assert harness.gateway.state.epoch_high_water == 9
        assert harness.gateway.effect_log == []
        next_owner_cycle(owner)
        runner.tick(harness.gateway.utc_ms / 1000)
        assert owner.binding.owner_epoch == 10
        assert harness.gateway.effect_log == []  # startup owns this cycle
        next_owner_cycle(owner)
        runner.tick(harness.gateway.utc_ms / 1000)
        assert runner.last_actuation.reason == "accepted"
        assert len(harness.gateway.effect_log) == 3
    finally:
        harness.lease.close()
