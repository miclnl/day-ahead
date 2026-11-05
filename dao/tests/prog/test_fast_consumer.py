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
    FastControlPolicy,
    Measurement,
    PolicyLimits,
)
from dao.prog.fastctrl.publication import GatePolicy
from dao.prog.fastctrl.runner import FastControlRunner
from dao.tests.prog.fakes.fenced_receiver import FencedReceiver
from dao.tests.prog.test_fast_execution import observe_receiver_calls
from dao.tests.prog.test_fast_ownership import RECEIPT_IDENTITY_MISMATCHES
from dao.tests.prog.test_fast_receiver import state as receiver_state
from dao.tests.prog.test_fast_runner import FakeHass, make_config
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_fast_contract import ready_plan

INSTALLATION = "sandbox"
RESOURCE = "actuator-" + hashlib.sha256(b"sandbox").hexdigest()[:32]
MONO = 1_000_000_000
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
                              utc_ms=NOW)
    gateway.receiver_max_lease_ms = LEASE_MS
    owner = ownership.OwnerCoordinator(
        store, receiver=gateway,
        now_ns=lambda: MONO + gateway.monotonic_ns,
        now_ms=lambda: gateway.utc_ms,
    )
    lease = None
    if start_owner:
        lease = task_execution.acquire_resource(RESOURCE, lease_dir=tmp_path / "leases")
        binding = owner.start(writer=lease, receiver=gateway, installation_id=INSTALLATION,
                              owner_session_id="session-new",
                              requested_not_after_ms=NOW + lease_ms,
                              deadline=DEADLINE, now_ms=NOW)
        assert binding.owner_epoch == 8, binding
    hass = FakeHass({"sensor.p1_power": "0", "sensor.soc": "50",
                     "sensor.battery_power": "0"})
    runner = FastControlRunner(hass, config or make_config(), plan_path=str(path),
                               state_path=str(tmp_path / "fast_state.json"),
                               gate_policy=POLICY, owner=owner)
    return SimpleNamespace(runner=runner, owner=owner, gateway=gateway, lease=lease,
                           store=store, plan=plan, path=path, root=tmp_path, hass=hass)


def propose(plan, state, *, timestamp=NOW_TS, grid_w=None, limits=None):
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
    return policy.decide(plan, measurement, state)


def consume(harness, decision, *, now_ms=NOW):
    return harness.runner._consume_financial_actuation(
        decision, harness.plan, owner=harness.owner, now_ms=now_ms)


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
    second_decision = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 60)
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
    transition = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 900)
    second = consume(harness, transition, now_ms=NOW + 900_000)[0]
    assert second.command_id != first.command_id
    assert second.sequence > first.sequence
    assert second.plan_digest == first.plan_digest
    assert controller.last_accepted_command.command_id == second.command_id
    assert controller.last_command_w == -1500.0
    # A new plan identity gets a new digest and a new command.
    other = financial_plan(powers=(-1200.0,))
    harness.path.write_text(json.dumps(other.to_dict()))
    harness.gateway.advance_clock(60_000)
    follow = propose(other, harness.runner.state, timestamp=NOW_TS + 960)
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
    # One absolute transport window for the whole installation.
    assert len({deadline.monotonic_end_ns for deadline in seen}) == 1
    assert len(seen) == 2
    assert seen[0].monotonic_end_ns == (
        harness.owner.now_ns() + boundary.TRANSPORT_BUDGET_NS)
    assert seen[0].monotonic_end_ns != (
        harness.owner.now_ns() + boundary.SECTION_BUDGET_NS)
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
    assert decision.benefit_eur_h == 0.0
    receipts = consume(harness, decision)
    assert receipts[0].status == "accepted"
    harness.gateway.advance_clock(60_000)
    proven = propose(harness.plan, harness.runner.state, grid_w=5000.0,
                     timestamp=NOW_TS + 60)
    assert proven.batteries[0].override is True
    assert proven.benefit_eur_h > 0.0


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
    assert harness.runner.state.saved_today_eur == 0.0
    harness.gateway.advance_clock(60_000)
    again = propose(harness.plan, harness.runner.state, grid_w=5000.0,
                    timestamp=NOW_TS + 60)
    assert again.benefit_eur_h == 0.0


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
    barrier = harness.owner.stop(reason="operator_stop")
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
    barrier = harness.owner.stop(reason="operator_stop")
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
    barrier = harness.owner.stop(reason="operator_stop")
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
    runner_module = import_module("dao.prog.fastctrl.runner")
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


def test_24_the_real_euro_bookkeeping_follows_the_acceptance(tmp_path):
    """The bookkeeping of the runner books a proven segment and nothing else."""
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state, grid_w=5000.0)
    # Nothing is proven: the real booking books nothing.
    harness.runner._book_saving(decision, elapsed_h=1.0)
    assert decision.benefit_eur_h == 0.0
    assert harness.runner.state.saved_today_eur == 0.0
    receipts = consume(harness, decision)
    assert receipts[0].status == "accepted"
    harness.gateway.advance_clock(60_000)
    proven = propose(harness.plan, harness.runner.state, grid_w=5000.0,
                     timestamp=NOW_TS + 60)
    assert proven.benefit_eur_h > 0.0
    harness.runner._book_saving(proven, elapsed_h=1.0)
    assert harness.runner.state.saved_today_eur == pytest.approx(proven.benefit_eur_h)


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
    barrier = harness.owner.stop(reason="operator_stop")
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
