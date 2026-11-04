"""F2c6: the composed failure matrix: the receiving fault through the owner and send boundary.

The five earlier slices prove their own part: the command contract (F2c1), the
receiving state machine (F2c2), the owner and its durable mirror (F2c3), the
slot boundary and the bounded send (F2c4) and the consumer (F2c5). This file
proves the *combination*: the same receiving fault, repeated through the owner
and the send boundary, with the joint invariants.

Every cell of the matrix is one combination of

* a **moment** of the receiving fault: ``before_acceptance`` (the delivery never
  reaches the receiving boundary), ``after_reserve`` (the command is durably
  reserved, no effect yet), ``after_first_helper`` (one helper wrote, a partial
  effect) and ``after_effect`` (the effect is complete, the acknowledgement is
  lost);
* a **recovery action**: ``retry``, ``caller_crash``, ``receiver_crash``,
  ``restore_mirror``, ``correction``, ``operator_stop`` and ``lease_boundary``.

The joint invariant of every cell is:

    old non-zero -> timeout -> new epoch barrier -> old delivery
        => no old effect after the barrier; a late acknowledgement never
           becomes a new accepted status; a new non-zero command needs an
           exact owner binding and a current basis.

Two variants of a recovery action have no delivery at all and therefore no
moment of the receiving fault: the correction *before* the send. That cell
says so explicitly instead of pretending to cover a moment that never
happened. The caller crash *before* the send is a moment of the caller crash
itself: there the caller is a real process that really dies.

Next to the matrix stand the cells of the boundaries and the controls: the
owner lease and the command deadline at one millisecond before, on and after
the bound, the **actual next helper** at the clock boundary of the command
(one millisecond before, on and after), the completed effect that has no next
helper left, the new section after an exact stop, the delayed delivery through
the owner and the send boundary, the delivered duplicate with its reducer
permutations, the transport failure on every step of the bounded send, the
accepted acknowledgement that comes after the local deadline, the negative
controls with a valid starting point and the integrity of the test fake: an
``attach`` of a state that is behind its independent anchor is refused before
any write.

The matrix is synthetic from end to end: a deterministic fake receiver, a
temporary plan file and a temporary mirror. There is no network, no Home
Assistant, no device and no installation. The matrix qualifies no receiver,
driver or hardware -- the fake is a test double -- it does not lift the hard
denial (``following_verdict`` stays ``actuation_gates_unproven`` for every v2
plan), it is no universal single-writer proof for the existing v1
installations, and a software timeout is no hardware guarantee.
"""

import json
import multiprocessing
import sys
from dataclasses import asdict, replace
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import pytest

from dao.prog import task_execution
from dao.prog.fastctrl.command import BatteryCommand, Deadline
from dao.prog.fastctrl.ownership import OwnerCoordinator, OwnerStore, RecoveryRequired
from dao.prog.fastctrl.publication import publication_lock
from dao.prog.fastctrl.receiver import payload_digest
from dao.tests.prog.fakes.fenced_receiver import FencedReceiver
from dao.tests.prog.test_fast_consumer import build, consume, propose
from dao.tests.prog.test_fast_execution import (
    CLOCK,
    DEADLINE,
    INSTALLATION,
    MONO,
    NOW,
    NOW_TS,
    POLICY,
    RESOURCE,
    build_command,
    execution,
    observed_receipt,
    owner_layout,
    ownership,
    provision_mirror,
    read_mirror,
    request,
)
from dao.tests.prog.test_fast_execution import artifact as artifact
from dao.tests.prog.test_fast_receiver import state as receiver_state

#: The process context of the caller crash: the same one as the owner process
#: proof of F2c4, so the durable files are shared without a server.
CTX = multiprocessing.get_context("fork")
#: The exit code of a caller that died on purpose.
CALLER_DEATH = 17

#: The four moments of the receiving fault.
MOMENTS = ("before_acceptance", "after_reserve", "after_first_helper", "after_effect")
#: The moments in which the first delivery is really answered by the receiver.
#: The moment before the acceptance has no answer at all: there the retry is
#: the recovery of a lost transport, not a second delivery of the same section.
DELIVERED_MOMENTS = ("after_reserve", "after_first_helper", "after_effect")
#: The seven recovery actions of the matrix and the cells that cover them.
RECOVERIES = {
    "retry": ("test_01_the_bounded_retry_reaches_the_receiver_and_is_deduplicated",
              "test_01b_the_retransmission_after_a_lost_transport_delivers_the_command_exactly_once"),
    "caller_crash": ("test_02_the_caller_process_dies_and_the_durable_state_survives",
                     "test_02b_the_lost_acknowledgement_keeps_the_evidence_and_fences_the_old_command",
                     "test_03_the_lost_lifetime_lease_refuses_the_send_without_any_effect"),
    "receiver_crash": "test_04_the_receiver_crash_and_restart_never_replay_the_old_command",
    "restore_mirror": "test_05_the_restored_mirror_and_state_inhibit_the_old_command_after_rollback",
    "correction": "test_06_the_correction_after_the_send_withdraws_the_owner_epoch",
    "operator_stop": "test_08_the_operator_stop_withdraws_the_epoch_and_fences_the_old_command",
    "lease_boundary": "test_09_the_lifetime_lease_before_on_and_after_the_new_epoch_barrier",
}
#: The variant that the brief names inside a recovery action. It has no
#: delivery at all, so the four moments do not apply to it. The caller crash
#: before the send is a moment of the caller crash itself: there the process
#: really dies.
VARIANTS = {
    "correction_before_send": "test_07_the_correction_before_the_send_gives_no_send_at_all",
}
#: The cells that are not a recovery action but a boundary, a control or a
#: fault path. The metatest checks that every one of them exists.
CONTROLS = {
    "owner_lease_boundary": "test_09b_the_owner_lease_time_boundary_is_exact",
    "command_deadline_boundary": "test_09c_the_command_deadline_boundary_is_exact",
    "effects_before_the_expiry": "test_09d_the_helper_effects_before_the_expiry_stay_visible",
    "new_section_after_the_stop": "test_09e_a_new_section_after_an_exact_stop_is_allowed",
    "delayed_delivery_through_the_owner":
        "test_09f_a_delayed_delivery_inside_the_window_goes_through_the_owner",
    "delivered_duplicate": "test_14_a_delivered_duplicate_keeps_the_live_reservation_uncertain",
    "reducer_permutations": "test_15_the_reducer_keeps_a_live_reservation_uncertain_in_every_order",
    "transport_failure_per_step": "test_10b_a_transport_failure_on_every_step_is_a_refusal",
    "late_accepted_acknowledgement": "test_11b_a_late_accepted_acknowledgement_stays_history",
    "next_helper_at_the_clock_boundary":
        "test_09g_the_next_helper_at_the_clock_boundary_is_exact",
    "completed_effect_without_a_next_helper":
        "test_09h_the_completed_effect_has_no_next_helper_at_the_clock_boundary",
    "negative_controls_with_a_valid_start":
        "test_17_a_new_command_needs_an_exact_binding_and_a_current_basis",
    "fake_attach_integrity": "test_16_the_attach_refuses_an_inconsistent_state_and_anchor_pair",
}
#: The effect the old command may have caused when the fault hits.
EXPECTED_EFFECTS = {"before_acceptance": 0, "after_reserve": 0,
                    "after_first_helper": 1, "after_effect": 3}
#: The receiver evidence for the old command right after the fault: the status,
#: the reason and the effect. The reserve stays visible as ``pending_effect``.
EVIDENCE = {
    "before_acceptance": None,
    "after_reserve": ("unconfirmed", "pending_effect", "none"),
    "after_first_helper": ("unconfirmed", "pending_effect", "partial"),
    "after_effect": ("unconfirmed", "effect_unknown", "unknown"),
}
#: The evidence after a crash with the extra persistence: the receiver knows
#: that a helper wrote, but not what it wrote, so the effect stays unknown.
EVIDENCE_CRASHED = {
    "before_acceptance": None,
    "after_reserve": ("unconfirmed", "pending_effect", "none"),
    "after_first_helper": ("unconfirmed", "effect_unknown", "unknown"),
    "after_effect": ("unconfirmed", "effect_unknown", "unknown"),
}
#: The late delivery after a barrier that only withdraws the epoch (an operator
#: stop or a correction): the epoch fence answers first.
LATE_FENCED = {
    "before_acceptance": ("rejected", "receiver_inhibited", "none"),
    "after_reserve": ("rejected", "epoch_withdrawn", "none"),
    "after_first_helper": ("unconfirmed", "epoch_withdrawn", "partial"),
    "after_effect": ("unconfirmed", "effect_unknown", "unknown"),
}
#: The late delivery after a new owner session took over: the receiver is bound
#: to the new session, so an old *send* is refused by the session fence.
LATE_REBOUND = {
    "before_acceptance": ("rejected", "wrong_owner_session", "none"),
    "after_reserve": ("rejected", "epoch_withdrawn", "none"),
    "after_first_helper": ("unconfirmed", "epoch_withdrawn", "partial"),
    "after_effect": ("unconfirmed", "effect_unknown", "unknown"),
}
#: The late delivery after a crash with the extra persistence: the effect of the
#: old command stays unknown and the epoch fence answers.
LATE_CRASHED = {
    "before_acceptance": ("rejected", "receiver_inhibited", "none"),
    "after_reserve": ("rejected", "epoch_withdrawn", "none"),
    "after_first_helper": ("unconfirmed", "epoch_withdrawn", "unknown"),
    "after_effect": ("unconfirmed", "effect_unknown", "unknown"),
}
#: The late delivery when the receiver state was rolled back *and* the anchor
#: inhibited the receiver: the record of the old command is gone, so the read
#: back knows nothing and only the effect log keeps the effect.
LATE_INHIBITED = {
    "before_acceptance": ("rejected", "receiver_inhibited", "none"),
    "after_reserve": ("rejected", "epoch_withdrawn", "none"),
    "after_first_helper": ("rejected", "epoch_withdrawn", "none"),
    "after_effect": None,
}
#: The internal bounded retry has no route for the M1 moment: a transport
#: exception ends the bounded send at once, so there is no second attempt
#: inside that send. The M1 cell of the internal retry is therefore not
#: applicable; a retransmission is a separate route and a separate cell.
RETRY_NOT_APPLICABLE = {
    "before_acceptance": "transport_exception_ends_the_bounded_send",
}
#: The transport window the fault burns. The retry cell keeps room for exactly
#: one bounded retry; every other cell burns the whole transport window.
RETRY_WINDOW_MS = 400
FAULT_WINDOW_MS = 1100
#: The deadline of the section: the section budget of one second bounds it.
SECTION_DEADLINE = Deadline(MONO + 10_000_000_000, NOW + 60_000)
#: The machine-readable outcome of a broken transport.
TRANSPORT_OUTCOME = ("refused", "receiver_unavailable")
#: The machine-readable outcome of an expired budget.
TIMEOUT_OUTCOME = ("refused", "deadline_exceeded")
#: The machine-readable outcome of a lost lifetime lease.
LEASE_OUTCOME = ("refused", "owner_lease_missing")
#: The durable evidence of the old command after the caller process died: the
#: number of helpers that wrote and the effect of the command. ``None`` means
#: that the delivery never reached the receiving boundary.
CRASH_RECORD = {
    "before_acceptance": (None, None),
    "after_reserve": (0, "none"),
    "after_first_helper": (1, "partial"),
    "after_effect": (3, "unknown"),
}
#: The answer of the receiver on the old delivery after the new epoch barrier
#: of the recovery: the refusal and the truth about the old effect.
CRASH_LATE = {
    "before_acceptance": ("rejected", "wrong_owner_session", "none"),
    "after_reserve": ("rejected", "wrong_owner_session", "none"),
    "after_first_helper": ("unconfirmed", "invalidated", "partial"),
    "after_effect": ("unconfirmed", "effect_unknown", "unknown"),
}


def layout_at(root, *, requested_not_after_ms=None):
    """One synthetic owner whose clock follows the manual clock of the fake.

    The owner clock follows the fake, so a fault that burns transport time is
    visible in the budget without a wall clock.
    """
    bound = owner_layout(root, requested_not_after_ms=requested_not_after_ms)
    bound.owner.now_ns = lambda: MONO + bound.gateway.monotonic_ns
    bound.owner.now_ms = lambda: bound.gateway.utc_ms
    return bound


@pytest.fixture
def layout(artifact):
    """One synthetic owner against the deterministic fake receiver."""
    bound = layout_at(artifact.root)
    yield bound
    bound.lease.close()


def delayed_transport(gateway, moment, *, window_ms, answer=True, crash=False, persist=True):
    """Break the transport at `moment` and record every send attempt.

    The wrapper performs exactly the part of the delivery that the moment
    allows. The transport then breaks: every later attempt fails, so a retry
    can never become a second effect. With ``answer=False`` the answer of the
    delivery is lost as well: that is the caller crash, and no receipt is ever
    recorded. The manual clock is the only synchronisation: no sleep, no thread.
    """
    attempts = []
    state = {"broken": False}

    def transport_error():
        if crash:
            # The fake's own failure: the receiver is gone.
            return ValueError("fake_receiver_stopped")
        return TimeoutError("transport_timeout")

    def send(command, *, deadline):
        attempts.append((command.command_id, command.sequence, deadline.monotonic_end_ns))
        if state["broken"]:
            gateway.advance_clock(window_ms)
            raise transport_error()
        if moment == "before_acceptance":
            # The command never reaches the receiving boundary: it waits in
            # the fake's own event queue.
            gateway.enqueue(command)
            state["broken"] = True
            gateway.advance_clock(window_ms)
            if crash:
                # The receiver really crashes with the command in flight: the
                # queue is lost and the running flag falls.
                gateway.crash(persist=persist)
            raise transport_error()
        gateway.enqueue(command)
        receipt = gateway.deliver()
        if moment == "after_first_helper":
            receipt = gateway.step_helper("setpoint")
        elif moment == "after_effect":
            gateway.lose_ack()
            for part in ("setpoint", "mode", "stop_inverter"):
                receipt = gateway.step_helper(part)
        state["broken"] = True
        gateway.advance_clock(window_ms)
        if crash:
            gateway.crash(persist=persist)
            raise transport_error()
        if not answer:
            # The acknowledgement is lost: the caller records no receipt.
            raise transport_error()
        return receipt

    gateway.send = send
    return attempts


def delivered_retry(gateway, moment):
    """Deliver the first attempt for real and let the bounded retry arrive too.

    The first delivery performs exactly the part of the delivery that the
    moment allows and answers with the receiver's own receipt. The bounded
    retry then really reaches the fake again with the same bytes: the fake
    refuses the duplicate instead of executing it again. Every attempt is
    recorded with its full payload and both deadline fields.
    """
    real_send = gateway.send
    attempts, observations = [], []

    def send(command, *, deadline):
        attempts.append({
            "command": command.to_dict(),
            "monotonic_end_ns": deadline.monotonic_end_ns,
            "execute_before_ms": deadline.execute_before_ms,
        })
        if len(attempts) == 1:
            gateway.enqueue(command)
            receipt = gateway.deliver()
            if moment == "after_first_helper":
                receipt = gateway.step_helper("setpoint", command=command)
            elif moment == "after_effect":
                gateway.lose_ack()
                for part in ("setpoint", "mode", "stop_inverter"):
                    receipt = gateway.step_helper(part, command=command)
        else:
            # The bounded retry really reaches the receiver.
            receipt = real_send(command, deadline=deadline)
        observations.append([receipt.status, receipt.reason, receipt.effect])
        return receipt

    gateway.send = send
    return attempts, observations


def restore_transport(gateway):
    """Give the fake its own `send` back after the fault injection."""
    gateway.__dict__.pop("send", None)


def exhaust_transport_budget(gateway, owner, deadline):
    """Let the manual clock run past the recorded absolute transport deadline.

    The deadline is the one that the section really handed to the receiver.
    The manual clock is the only synchronisation: no sleep, no thread. The
    returned value is the rest budget of that deadline afterwards: zero means
    the absolute transport deadline is really over.
    """
    remaining = deadline.monotonic_end_ns - owner.now_ns()
    if remaining > 0:
        gateway.advance_clock((remaining // 1_000_000) + 1)
    return deadline.remaining_ns(owner.now_ns())


def faulted_section(artifact, layout, moment, *, window_ms, answer=True, crash=False,
                    persist=True):
    """The old non-zero command through the owner section, delayed at `moment`.

    Returns the send attempts and the outcome of the section: a receipt or a
    machine-readable refusal.
    """
    module = execution()
    gateway = layout.gateway
    attempts = delayed_transport(gateway, moment, window_ms=window_ms, answer=answer,
                                 crash=crash, persist=persist)
    with publication_lock(artifact.path) as lease:
        try:
            receipt = module.owner_section(lease=lease, owner=layout.owner, receiver=gateway,
                                           request=request(), policy=POLICY, clock=CLOCK,
                                           deadline=SECTION_DEADLINE)
        except module.ExecutionRefused as error:
            return attempts, ("refused", error.reason)
    return attempts, ("receipt", receipt.status)


def withdraw_epoch(layout, *, reason="operator_stop"):
    """The new epoch barrier: the epoch is withdrawn and confirmed."""
    barrier = layout.owner.stop(reason=reason)
    assert barrier is not None, "the epoch was not withdrawn"
    assert barrier.status == "accepted", barrier
    assert barrier.queues_fenced is True and barrier.safe_state == "confirmed", barrier
    return barrier


def new_session(layout, *, session_id):
    """A new owner session takes over after a crash or an expired lease.

    The new session withdraws the old epoch and binds the next one: that
    withdrawal is the new epoch barrier of the matrix.
    """
    layout.lease.close()
    lease = task_execution.acquire_resource(RESOURCE, lease_dir=layout.store.lease_dir)
    deadline = Deadline(layout.owner.now_ns() + 1_000_000_000, layout.gateway.utc_ms + 60_000)
    binding = layout.owner.start(writer=lease, receiver=layout.gateway,
                                 installation_id=INSTALLATION, owner_session_id=session_id,
                                 requested_not_after_ms=layout.gateway.utc_ms + 60_000,
                                 deadline=deadline, now_ms=layout.gateway.utc_ms)
    assert binding is not None and binding.owner_epoch > 0, binding
    layout.lease = lease
    # The layout follows the owner: a command that is built here belongs to
    # the binding that is really in force.
    layout.binding = binding
    return binding


def late_delivery(gateway, command, moment):
    """The old delivery arrives after the new epoch barrier.

    The moment decides what is still pending: the reservation, the first
    helper, or the lost acknowledgement. When the crash already discarded the
    in-flight command, the old command is offered again instead.
    """
    if moment == "before_acceptance":
        if gateway.queue:
            return gateway.deliver()
        restore_transport(gateway)
        return gateway.send(command, deadline=SECTION_DEADLINE)
    if moment == "after_effect":
        return gateway.query(installation_id=command.installation_id,
                             owner_epoch=command.owner_epoch, sequence=command.sequence,
                             command_id=command.command_id,
                             payload_digest=payload_digest(command), deadline=SECTION_DEADLINE)
    part = "setpoint" if moment == "after_reserve" else "mode"
    return gateway.step_helper(part, command=command)


def receiver_word(gateway, command):
    """What the receiver still knows about the old command."""
    if not any(item.command_id == command.command_id for item in gateway.state.commands):
        return None
    return gateway.query(installation_id=command.installation_id,
                         owner_epoch=command.owner_epoch, sequence=command.sequence,
                         command_id=command.command_id,
                         payload_digest=payload_digest(command), deadline=SECTION_DEADLINE)


#: The identity fields of a command, of a receipt and of a receiver receipt:
#: the two sides of the chain have to name the same command in the same way.
#: Every field is required: nothing is skipped silently.
IDENTITY_FIELDS = ("installation_id", "battery_name", "owner_session_id", "owner_epoch",
                   "sequence", "command_id", "plan_digest", "payload_digest")


def expected_identity(command):
    """The identity of one expected command, computed from the command itself.

    The payload digest is computed the same way the receiver computes it, so
    the comparison never accepts a digest that was only asserted.
    """
    return {
        "installation_id": command.installation_id,
        "battery_name": command.battery_name,
        "owner_session_id": command.owner_session_id,
        "owner_epoch": command.owner_epoch,
        "sequence": command.sequence,
        "command_id": command.command_id,
        "plan_digest": command.plan_digest,
        "payload_digest": payload_digest(command),
    }


def identity_of(item):
    """The identity fields of a command, a receipt or a durable record.

    A command carries the payload and not its digest: the digest is computed
    here, exactly as the receiver computes it. A receipt carries the digest
    itself. A field that is not there is an error, never a skipped field.
    """
    if type(item) is dict:
        if "payload_digest" in item:
            return {field: item[field] for field in IDENTITY_FIELDS}
        return expected_identity(BatteryCommand.from_dict(item))
    if hasattr(item, "payload_digest"):
        return {field: getattr(item, field) for field in IDENTITY_FIELDS}
    return expected_identity(item)


def assert_identity(item, command, *, what):
    """The item names exactly the expected command, field by field."""
    actual = identity_of(item)
    expected = expected_identity(command)
    assert actual == expected, (what, actual, expected)


def assert_chain_consistent(mirror, receiver_receipt, command=None, *, pending=True,
                             receipt=True):
    """The mirror and the receiver never contradict each other.

    Neither side may claim an acceptance that the other side does not have.
    The mirror never records a rejection either: a rejection needs proof that
    this very command caused no effect. The identity of the durable pending
    command, of the mirror receipt and of the receiver receipt is compared
    field by field with the one expected command. ``pending`` and ``receipt``
    say per cell which parts of the mirror have to be there: a part that must
    be absent is checked as absent.
    """
    mirror_status = mirror["last_receipt"]["status"] if mirror["last_receipt"] else None
    receiver_status = receiver_receipt.status if receiver_receipt else None
    assert mirror_status != "accepted", mirror
    assert receiver_status != "accepted", receiver_receipt
    assert mirror_status != "rejected", mirror
    assert not (mirror_status == "unconfirmed" and receiver_status == "accepted")
    if command is None:
        return
    if pending:
        assert mirror["pending_command"] is not None, mirror
        assert_identity(mirror["pending_command"], command, what="pending command")
    else:
        assert mirror["pending_command"] is None, mirror
    if receipt:
        assert mirror["last_receipt"] is not None, mirror
        assert_identity(mirror["last_receipt"], command, what="mirror receipt")
    else:
        assert mirror["last_receipt"] is None, mirror
    # The receiver receipt names the same command, when the receiver has one.
    if receiver_receipt is not None:
        assert_identity(receiver_receipt, command, what="receiver receipt")


def assert_barrier_exact(mirror, barrier):
    """The last confirmed barrier of the mirror is this barrier, field by field."""
    assert mirror["last_barrier"] is not None, mirror
    for field in ("installation_id", "barrier_id", "revoked_through_epoch", "current_epoch",
                  "status", "queues_fenced", "safe_state", "accepted_ts", "receiver_generation"):
        assert mirror["last_barrier"][field] == getattr(barrier, field), (field, mirror)


def assert_session_barrier(mirror, binding, barrier=None):
    """The mirror holds the exact barrier that the new session needed.

    The barrier of the mirror is compared field by field with the barrier
    evidence of the receiver itself, and the binding of the new session is
    compared with that same barrier: the barrier id, the installation, the
    generation, the epoch and the timestamp.
    """
    assert mirror["last_barrier"] is not None, mirror
    stored = mirror["last_barrier"]
    if barrier is not None:
        assert_barrier_exact(mirror, barrier)
    assert stored["status"] == "accepted", mirror
    assert stored["current_epoch"] == binding.owner_epoch, mirror
    assert stored["revoked_through_epoch"] == binding.owner_epoch - 1, mirror
    assert stored["queues_fenced"] is True, mirror
    assert stored["safe_state"] == "confirmed", mirror
    assert stored["installation_id"] == binding.installation_id, mirror
    assert stored["receiver_generation"] == binding.receiver_generation, mirror
    assert stored["barrier_id"] == binding.barrier_id, mirror
    assert stored["accepted_ts"] is not None, mirror
    assert mirror["owner_epoch"] == binding.owner_epoch, mirror


def assert_pending_stays(mirror, command):
    """The uncertain command stays visible in the mirror of the owner."""
    assert mirror["pending_command"] is not None, mirror
    assert mirror["pending_command"]["command_id"] == command.command_id, mirror


def assert_effect_log(gateway, command, moment):
    """The effect log is the only effect evidence: exactly the effect of the moment.

    Every entry belongs to the one old command: a retry or a late delivery
    never creates a second effect.
    """
    assert len(gateway.effect_log) == EXPECTED_EFFECTS[moment], gateway.effect_log
    assert all(entry[0] == command.command_id for entry in gateway.effect_log), gateway.effect_log


def assert_evidence(gateway, command, moment, expected):
    """The receiver keeps exactly this evidence for the old command."""
    word = receiver_word(gateway, command)
    if expected is None:
        assert word is None, (moment, word)
        return
    assert word is not None, (moment, word)
    assert (word.status, word.reason, word.effect) == expected, (moment, word)


def assert_late(gateway, command, moment, expected):
    """The late delivery after the barrier is refused and never a new effect."""
    late = late_delivery(gateway, command, moment)
    if expected is None:
        assert late is None, (moment, late)
        return
    assert late is not None, (moment, late)
    assert (late.status, late.reason, late.effect) == expected, (moment, late)
    assert late.status != "accepted", late
    assert late.effect != "complete", late


def assert_new_command_needs_binding_and_basis(artifact, layout):
    """A new non-zero command needs an exact owner binding and a current basis.

    The control starts from a state that is valid in itself: a **new** pending
    command under the **current** binding and the **current** basis. Only then
    exactly one precondition is broken: first the binding alone, then the
    basis alone. The old command of the cell is never reused here: it would
    carry more than one invalid precondition at the same time.
    """
    module = execution()
    binding = layout.owner.binding
    assert binding is not None, "the negative control needs a current binding"
    # A fresh and correct clock and deadline: the manual clock of the fake.
    clock = clock_at(layout.gateway, layout.gateway.utc_ms)
    # The valid starting point: a new pending command in the current binding
    # and the current basis.
    fresh = build_command(module, layout, artifact)
    layout.owner.record_pending_command(fresh)
    assert_identity(layout.owner.state.pending_command, fresh, what="new pending command")
    # Only the binding is wrong: the command and the pending command stay
    # valid, so the refusal can only come from the binding.
    layout.owner.binding = replace(binding, owner_epoch=binding.owner_epoch + 1)
    try:
        with publication_lock(artifact.path) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.send_locked(fresh, lease=lease, owner=layout.owner,
                                   receiver=layout.gateway, deadline=SECTION_DEADLINE,
                                   clock=clock, policy=POLICY)
        assert error.value.reason == "owner_binding_mismatch", error.value
    finally:
        layout.owner.binding = binding
    # Only the basis is wrong: the binding, the command and the pending
    # command stay valid, so the refusal can only come from the basis.
    other = replace(artifact.plan, price_average=artifact.plan.price_average + 0.5)
    artifact.path.write_text(json.dumps(other.to_dict()))
    try:
        with publication_lock(artifact.path) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.send_locked(fresh, lease=lease, owner=layout.owner,
                                   receiver=layout.gateway, deadline=SECTION_DEADLINE,
                                   clock=clock, policy=POLICY)
        assert error.value.reason == "plan_digest_mismatch", error.value
    finally:
        artifact.path.write_text(json.dumps(artifact.plan.to_dict()))


# -- the matrix: one recovery action over the four moments --------------------

@pytest.mark.parametrize("moment", DELIVERED_MOMENTS)
def test_01_the_bounded_retry_reaches_the_receiver_and_is_deduplicated(artifact, layout,
                                                                       moment):
    """H1 retry: the bounded retry really arrives and the receiver deduplicates it.

    The first delivery is really performed up to the moment of the fault and
    answers with the receiver's own receipt. The bounded retry then really
    reaches the receiver again, with the full payload and both deadline fields
    of the first attempt, and the receiver refuses the duplicate. The exact
    query reads the command back. The effect log never grows a second time and
    the pending command stays durable.

    The whole sequence happens in **one** `owner_section`: the bounded retry
    and the exact query are the internal ones of that single section, inside
    the transport deadline that the section really derived. After the retry and
    the query the manual clock runs past that recorded absolute transport
    deadline, the rest budget is zero, and only then come the stop barrier and
    the late delivery of the old command.

    The M1 moment is **not applicable** for the internal retry
    (``RETRY_NOT_APPLICABLE``): a transport exception ends the bounded send at
    once, so that moment has no second attempt inside the same send. The
    retransmission of that moment is the separate cell ``test_01b``.
    """
    assert moment != "before_acceptance", (moment, RETRY_NOT_APPLICABLE)
    module = execution()
    attempts, observations = delivered_retry(layout.gateway, moment)
    queries = []
    real_query = layout.gateway.query

    def query(**kwargs):
        receipt = real_query(**kwargs)
        queries.append([receipt.status, receipt.reason, receipt.effect])
        return receipt

    layout.gateway.query = query
    try:
        with publication_lock(artifact.path) as lease:
            receipt = module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                           request=request(), policy=POLICY, clock=CLOCK,
                                           deadline=SECTION_DEADLINE)
    finally:
        layout.gateway.__dict__.pop("send", None)
        layout.gateway.__dict__.pop("query", None)
    # Two real deliveries and one exact read back: the retry really arrived.
    assert len(attempts) == 2, attempts
    assert len(observations) == 2, observations
    assert len(queries) == 1, queries
    # The retry carries the full payload and both deadline fields.
    assert attempts[0]["command"] == attempts[1]["command"], attempts
    assert attempts[0]["monotonic_end_ns"] == attempts[1]["monotonic_end_ns"], attempts
    assert attempts[0]["execute_before_ms"] == attempts[1]["execute_before_ms"], attempts
    # The receiver answers with its own words.
    expected = {
        "after_reserve": (["unconfirmed", "pending_effect", "none"],
                          ["rejected", "stale_or_duplicate_sequence", "none"],
                          ["unconfirmed", "pending_effect", "none"]),
        "after_first_helper": (["unconfirmed", "pending_effect", "partial"],
                               ["unconfirmed", "pending_effect", "partial"],
                               ["unconfirmed", "pending_effect", "partial"]),
        "after_effect": (["unconfirmed", "effect_unknown", "unknown"],
                         ["unconfirmed", "effect_unknown", "unknown"],
                         ["unconfirmed", "effect_unknown", "unknown"]),
    }[moment]
    assert observations[0] == expected[0], observations
    assert observations[1] == expected[1], observations
    assert queries[0] == expected[2], queries
    # The outcome stays uncertain and the pending command stays durable.
    assert receipt.status == "unconfirmed", receipt
    command = layout.owner.state.pending_command
    assert command is not None, "the pending command is not durable"
    mirror = read_mirror(artifact.root)
    assert_pending_stays(mirror, command)
    assert mirror["last_receipt"]["status"] == "unconfirmed", mirror
    # The effect log is the only effect evidence: never a second effect.
    assert_effect_log(layout.gateway, command, moment)
    # The transport deadline that the section really handed to the receiver:
    # it is the one that the section derived, and it is shorter than the
    # deadline of the whole section.
    transport = Deadline(attempts[0]["monotonic_end_ns"], attempts[0]["execute_before_ms"])
    assert transport.monotonic_end_ns < SECTION_DEADLINE.monotonic_end_ns, transport
    assert transport.remaining_ns(layout.owner.now_ns()) > 0, transport
    # The manual clock runs past that absolute deadline: the rest budget is 0.
    assert exhaust_transport_budget(layout.gateway, layout.owner, transport) == 0, transport
    assert transport.remaining_ns(layout.owner.now_ns()) == 0, transport
    # The new epoch barrier, then the late delivery of the old command.
    barrier = withdraw_epoch(layout)
    assert_barrier_exact(read_mirror(artifact.root), barrier)
    assert_late(layout.gateway, command, moment, LATE_FENCED[moment])
    assert_effect_log(layout.gateway, command, moment)
    assert_chain_consistent(read_mirror(artifact.root), receiver_word(layout.gateway, command),
                            command, pending=True, receipt=True)
    # The negative control needs a current binding: the recovery takes a new
    # session, exactly as a real owner would after the barrier.
    new_session(layout, session_id="session-after-the-barrier")
    assert_new_command_needs_binding_and_basis(artifact, layout)


@pytest.mark.parametrize("moment", ["before_acceptance"])
def test_01b_the_retransmission_after_a_lost_transport_delivers_the_command_exactly_once(
        artifact, layout, moment):
    """H1, the retransmission: a separate send really delivers the command once.

    This is **not** the internal bounded retry of one `send_locked`: it is a
    separate transmission of the same bytes in a separate call, with its own
    publication lease and with the section deadline itself. The internal route
    of this moment is not applicable
    (``RETRY_NOT_APPLICABLE["before_acceptance"]``): a transport exception
    ends the bounded send at once.

    The first attempt is lost in the transport: the section refuses with a
    machine-readable reason, no receipt is invented and the pending command
    stays durable. The retransmission sends exactly the same bytes again and
    the receiver really receives it: one reservation, three helpers and one
    acceptance. After a new epoch barrier the old delivery never adds an
    effect.
    """
    assert moment == "before_acceptance", moment
    module = execution()
    command = build_command(module, layout, artifact)
    layout.owner.record_pending_command(command)
    lost = []

    def send(payload, *, deadline):
        lost.append(payload.to_dict())
        raise TimeoutError("transport_timeout")

    layout.gateway.send = send
    try:
        with publication_lock(artifact.path) as lease:
            with pytest.raises(module.ExecutionRefused) as error:
                module.send_locked(command, lease=lease, owner=layout.owner,
                                   receiver=layout.gateway, deadline=SECTION_DEADLINE,
                                   clock=CLOCK, policy=POLICY)
        assert error.value.reason == "receiver_unavailable", error.value
        assert len(lost) == 1, lost
        assert lost[0] == command.to_dict(), lost
        # Nothing was proven: the uncertain command stays durable.
        mirror = read_mirror(artifact.root)
        assert_pending_stays(mirror, command)
        assert mirror["last_receipt"] is None, mirror
        assert not layout.gateway.effect_log, layout.gateway.effect_log
        # The retry: exactly the same bytes and the same pending command.
        layout.gateway.__dict__.pop("send", None)
        with publication_lock(artifact.path) as lease:
            receipt = module.send_locked(command, lease=lease, owner=layout.owner,
                                         receiver=layout.gateway, deadline=SECTION_DEADLINE,
                                         clock=CLOCK, policy=POLICY)
        assert receipt.status == "accepted", receipt
        assert receipt.command_id == command.command_id, receipt
        assert len(layout.gateway.effect_log) == 3, layout.gateway.effect_log
        assert all(entry[0] == command.command_id for entry in layout.gateway.effect_log)
        # The acceptance is a proof: the pending command is gone.
        mirror = read_mirror(artifact.root)
        assert mirror["pending_command"] is None, mirror
        assert mirror["last_receipt"]["status"] == "accepted", mirror
        # The old delivery after a new epoch barrier never adds an effect.
        withdraw_epoch(layout)
        layout.gateway.send(command, deadline=SECTION_DEADLINE)
        assert len(layout.gateway.effect_log) == 3, layout.gateway.effect_log
    finally:
        layout.gateway.__dict__.pop("send", None)


def effect_log_path(root):
    """The durable effect log of the fake: the real writes of the last writer."""
    return Path(root) / "receiver" / "effect-log.jsonl"


def durable_effect_log(root):
    """The real effect writes of the durable effect log, in their order.

    The entries are read from the file, not from an object: the effect
    evidence has to survive the process that caused it. They are never
    reconstructed from the helper receipts or from an expected value.
    """
    path = effect_log_path(root)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(tuple(json.loads(line)))
    return entries


def caller_layout(root):
    """One owner layout whose fake keeps a durable effect log.

    The durable effect log is opt-in fault evidence of the fake: it records
    the real writes of the last writer at the moment they happen, so the
    effect evidence survives the death of the process that owned the fake.
    """
    module = ownership()
    store = provision_mirror(root)
    gateway = FencedReceiver(Path(root) / "receiver", initial_state=receiver_state(),
                             utc_ms=NOW, effect_log_path=effect_log_path(root))
    owner = module.OwnerCoordinator(store, receiver=gateway)
    owner.now_ns = lambda: MONO
    owner.now_ms = lambda: NOW
    lease = task_execution.acquire_resource(RESOURCE, lease_dir=Path(root) / "leases")
    binding = owner.start(writer=lease, receiver=gateway, installation_id=INSTALLATION,
                          owner_session_id="session-new", requested_not_after_ms=NOW + 60_000,
                          deadline=DEADLINE, now_ms=NOW)
    assert binding.owner_epoch == 8, binding
    return SimpleNamespace(module=module, store=store, gateway=gateway, owner=owner,
                           lease=lease, binding=binding)


def receiver_record(root, command_id):
    """The durable record of the receiver for one command.

    The record is read from the file, not from an object: the evidence of a
    crash has to survive the process that caused it.
    """
    path = Path(root) / "receiver" / "receiver-state.json"
    for record in json.loads(path.read_text())["commands"]:
        if record["command_id"] == command_id:
            return record
    return None


def caller_worker(root, cutpoint, moment):
    """The caller process: it dies at the cutpoint, for real.

    The child owns the receiver and the lifetime lease. It dies with
    ``os._exit``: there is no unwinding, no flush of a receipt and no
    cleanup. The parent sees the durable files and nothing else. The child
    never reports anything itself: a report of a dying process is not
    evidence.
    """
    import os

    module = execution()
    root = Path(root)
    store = OwnerStore(root / "fast_plan.json", INSTALLATION, lease_dir=root / "leases")
    # The receiver of the parent: this process takes over the durable files.
    gateway = FencedReceiver(root / "receiver", initial_state=None, utc_ms=NOW, attach=True,
                             effect_log_path=effect_log_path(root))
    owner = OwnerCoordinator(store, receiver=gateway, now_ns=lambda: MONO, now_ms=lambda: NOW)
    lease = task_execution.acquire_resource(RESOURCE, lease_dir=root / "leases")
    binding = owner.start(writer=lease, receiver=gateway, installation_id=INSTALLATION,
                          owner_session_id="session-caller",
                          requested_not_after_ms=NOW + 60_000,
                          deadline=Deadline(MONO + 1_000_000_000, NOW + 60_000), now_ms=NOW)
    assert binding is not None, binding
    if cutpoint == "before_send":
        real_record = owner.record_pending_command

        def record(command):
            real_record(command)
            # The caller dies here: the pending command is durable, the
            # delivery never starts.
            os._exit(CALLER_DEATH)

        owner.record_pending_command = record
    else:

        def send(command, *, deadline):
            gateway.enqueue(command)
            gateway.deliver()
            if moment in ("after_first_helper", "after_effect"):
                gateway.step_helper("setpoint", command=command)
            if moment == "after_effect":
                gateway.lose_ack()
                for part in ("mode", "stop_inverter"):
                    gateway.step_helper(part, command=command)
            # The caller dies here: the delivery happened, the receipt is
            # never written.
            os._exit(CALLER_DEATH)

        gateway.send = send
    with publication_lock(root / "fast_plan.json") as publication:
        module.owner_section(lease=publication, owner=owner, receiver=gateway,
                             request=request(), policy=POLICY, clock=CLOCK,
                             deadline=Deadline(MONO + 1_000_000_000, NOW + 60_000))
    raise AssertionError("the caller did not die at its cutpoint")


def stop_caller(process):
    """End the caller process for sure: no orphan may survive this cell."""
    if not process.is_alive():
        process.join(timeout=5)
        return
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def start_caller(root, cutpoint, moment):
    """Start the caller process and wait for its death.

    Every path ends the process: the normal death, the timeout and the error
    of the cell. There is no orphan and no sleep: the manual clock and the
    durable files are the synchronisation.
    """
    process = CTX.Process(target=caller_worker, args=(str(root), cutpoint, moment))
    process.start()
    try:
        process.join(timeout=60)
        if process.is_alive():
            pytest.fail("the caller process did not die at its cutpoint")
        return process
    finally:
        stop_caller(process)


@pytest.mark.parametrize("moment", MOMENTS)
def test_02_the_caller_process_dies_and_the_durable_state_survives(artifact, moment):
    """H2 callercrash: a real caller process dies, the durable state survives.

    The caller is a separate process: it binds the receiver, reserves the
    sequence, writes the pending command durably and then it dies. The death
    takes the process memory with it, so the receipt is never written and the
    lifetime lease is released by the operating system. The recovery uses a
    new owner object that is built from the durable files only, and it proves
    the lease takeover, the pending command, the effect evidence and the
    exact barrier.

    The effect evidence is the **real** effect log of the fake: the durable
    effect log keeps every real write of the last writer at the moment that
    write happens, so the baseline of the moment (0/0/1/3) survives the death
    of the caller. The baseline is never reconstructed from the helper
    receipts of the receiver.

    After the barrier of the recovery the old command may not write any more:
    for the reserved moment the **first** helper is really offered, for the
    partial moment the **next** helper is really offered, and for the
    completed moment there is no helper left: the exact query gives the
    original receiver evidence. The old effect log never grows.

    The moment decides how far the delivery got before the death: before the
    acceptance nothing was delivered at all, after the reserve the command is
    durably reserved, after the first helper one helper wrote and after the
    effect the whole effect exists and the acknowledgement is lost.
    """
    module = execution()
    cutpoint = "before_send" if moment == "before_acceptance" else "after_send"
    layout = caller_layout(artifact.root)
    # The lease belongs to the caller from now on: the parent gives it up, so
    # the death of the caller really releases it.
    layout.lease.close()
    process = start_caller(artifact.root, cutpoint, moment)
    assert process.exitcode == CALLER_DEATH, process.exitcode
    # The lease is free again: the operating system released it with the dead
    # process.
    lease = task_execution.acquire_resource(RESOURCE, lease_dir=artifact.root / "leases")
    try:
        # The durable mirror of the dead caller: the pending command is there,
        # the receipt is not.
        mirror = read_mirror(artifact.root)
        pending = mirror["pending_command"]
        assert pending is not None, mirror
        assert mirror["last_receipt"] is None, mirror
        assert pending["owner_epoch"] == 9, pending
        assert pending["sequence"] == 41, pending
        # The old command of the dead caller, exactly as the mirror holds it.
        old = pending_command(pending)
        # The full chain of the dead caller: the pending command is there and
        # names the old command on all eight fields, and the mirror receipt is
        # absent, because the caller died before it could write one.
        assert_chain_consistent(mirror, None, old, pending=True, receipt=False)
        # The real effect evidence of the dead caller: the durable effect log.
        baseline = durable_effect_log(artifact.root)
        assert len(baseline) == EXPECTED_EFFECTS[moment], baseline
        assert all(entry[0] == pending["command_id"] for entry in baseline), baseline
        # The receiver's own durable bookkeeping: not the effect evidence.
        record = receiver_record(artifact.root, pending["command_id"])
        helpers, effect = CRASH_RECORD[moment]
        if helpers is None:
            assert record is None, record
        else:
            assert record is not None, record
            assert len(record["helper_results"]) == helpers, record
            assert record["effect"] == effect, record
        # The recovery: a new owner object, built from the durable files only.
        gateway = FencedReceiver(artifact.root / "receiver", initial_state=None, utc_ms=NOW,
                                 attach=True, effect_log_path=effect_log_path(artifact.root))
        # The real effect baseline survived the death of the caller.
        assert gateway.effect_log == baseline, (gateway.effect_log, baseline)
        # The original receiver evidence of the old command, read after the
        # attach and before the barrier of the recovery: this is the reference
        # for the exact history of the query.
        original = gateway.query(installation_id=old.installation_id,
                                 owner_epoch=old.owner_epoch, sequence=old.sequence,
                                 command_id=old.command_id,
                                 payload_digest=payload_digest(old), deadline=SECTION_DEADLINE)
        if original is not None:
            # The original evidence names the old command on all eight fields.
            assert_identity(original, old, what="original query receipt")
        store = OwnerStore(artifact.root / "fast_plan.json", INSTALLATION,
                           lease_dir=artifact.root / "leases")
        owner = OwnerCoordinator(store, receiver=gateway, now_ns=lambda: MONO,
                                 now_ms=lambda: NOW)
        binding = owner.start(writer=lease, receiver=gateway, installation_id=INSTALLATION,
                              owner_session_id="session-after-crash",
                              requested_not_after_ms=NOW + 60_000,
                              deadline=Deadline(MONO + 1_000_000_000, NOW + 60_000), now_ms=NOW)
        assert binding.owner_epoch == 10, binding
        # The barrier of the new session is exact and durable.
        mirror = read_mirror(artifact.root)
        assert mirror["phase"] == "owned", mirror
        assert_session_barrier(mirror, binding, gateway.state.last_barrier)
        # The pending command of the dead caller stays visible and the mirror
        # receipt stays absent: the chain is still the one of the old command.
        assert_chain_consistent(mirror, None, old, pending=True, receipt=False)
        # After the barrier the old command may not write any more.
        if moment == "after_reserve":
            # The first helper is really offered: the fence refuses it.
            helper = gateway.step_helper("setpoint", command=old)
            assert (helper.status, helper.reason, helper.effect) == (
                "rejected", "epoch_withdrawn", "none"), helper
            # The offered helper receipt names the old command on all eight
            # fields.
            assert_identity(helper, old, what="old helper receipt")
            offered = helper
        elif moment == "after_first_helper":
            # The next helper is really offered: the fence refuses it.
            helper = gateway.step_helper("mode", command=old)
            assert (helper.status, helper.reason, helper.effect) == (
                "unconfirmed", "epoch_withdrawn", "partial"), helper
            assert_identity(helper, old, what="old helper receipt")
            offered = helper
        elif moment == "after_effect":
            # No helper is left: the exact query gives the original evidence.
            word = gateway.query(installation_id=old.installation_id,
                                 owner_epoch=old.owner_epoch, sequence=old.sequence,
                                 command_id=old.command_id,
                                 payload_digest=payload_digest(old), deadline=SECTION_DEADLINE)
            assert word is not None, word
            assert (word.status, word.reason, word.effect) == (
                "unconfirmed", "effect_unknown", "unknown"), word
            # The history of the query is exactly the original one: the same
            # identity, the same helper results and the same observation
            # moment. Nothing of the original evidence changed.
            assert original is not None, original
            assert asdict(word) == asdict(original), (word, original)
            assert_identity(word, old, what="query receipt after the recovery")
            offered = word
        else:
            # Nothing was delivered: the old delivery meets the session fence.
            late = gateway.send(old, deadline=SECTION_DEADLINE)
            assert (late.status, late.reason, late.effect) == CRASH_LATE[moment], late
            assert_identity(late, old, what="old late delivery receipt")
            offered = late
        # The chain of the old command after the barrier: the pending command
        # is there, the mirror receipt is absent and the offered receipt names
        # the old command on all eight fields.
        assert_chain_consistent(read_mirror(artifact.root), offered, old,
                                pending=True, receipt=False)
        # The old effect log never grows after the barrier.
        assert gateway.effect_log == baseline, (gateway.effect_log, baseline)
        # A new section is allowed again: a new epoch, a new sequence and a
        # current basis. It adds exactly three new entries on top of the old
        # baseline, and only the new command has them. The command that the
        # section really stored durably before the send is read back from the
        # mirror: it is never derived from the receipt or from expected values.
        recorded = {}
        real_record = owner.record_pending_command

        def record(command):
            real_record(command)
            stored = read_mirror(artifact.root)["pending_command"]
            assert stored is not None, "the new command is not durable"
            recorded["command"] = pending_command(stored)

        owner.record_pending_command = record
        with publication_lock(artifact.path) as publication:
            receipt = module.owner_section(lease=publication, owner=owner, receiver=gateway,
                                           request=request(), policy=POLICY, clock=CLOCK,
                                           deadline=SECTION_DEADLINE)
        owner.record_pending_command = real_record
        new = recorded.get("command")
        assert new is not None, "the new command was not stored durably before the send"
        assert new.owner_epoch == 10, new
        assert new.sequence == 42, new
        assert receipt.status == "accepted", receipt
        assert receipt.owner_epoch == 10, receipt
        assert receipt.sequence == 42, receipt
        # The accepted receipt names the durably stored command on all eight
        # fields.
        assert_identity(receipt, new, what="accepted receipt")
        assert len(gateway.effect_log) == len(baseline) + 3, gateway.effect_log
        assert all(entry[0] == receipt.command_id for entry in gateway.effect_log[-3:])
        assert durable_effect_log(artifact.root) == gateway.effect_log
        mirror = read_mirror(artifact.root)
        # The pending command is gone: the accepted receipt cleared it.
        assert mirror["pending_command"] is None, mirror
        # The durable mirror receipt names the same durably stored command.
        assert mirror["last_receipt"]["command_id"] == receipt.command_id, mirror
        assert_identity(mirror["last_receipt"], new, what="mirror receipt")
        # The receiver knows the same command.
        word = receiver_word(gateway, new)
        assert word is not None, word
        assert_identity(word, new, what="receiver query")
    finally:
        lease.close()


def pending_command(data):
    """The command of the durable mirror, as a command object."""
    from dao.prog.fastctrl.command import BatteryCommand

    return BatteryCommand.from_dict(data)


@pytest.mark.parametrize("moment", MOMENTS)
def test_02b_the_lost_acknowledgement_keeps_the_evidence_and_fences_the_old_command(
        artifact, layout, moment):
    """H2, the transport variant: the acknowledgement is lost, the caller lives.

    The transport breaks after the delivery: the exception is caught and the
    section refuses with a machine-readable reason. The durable mirror of the
    owner is the only evidence that survives: no receipt is recorded and the
    pending command stays visible. A new owner session withdraws the epoch,
    and the late delivery is refused without an effect.
    """
    attempts, outcome = faulted_section(artifact, layout, moment, window_ms=FAULT_WINDOW_MS,
                                        answer=False)
    assert outcome == TRANSPORT_OUTCOME, outcome
    assert len(attempts) == 1, attempts
    command = layout.owner.state.pending_command
    assert command is not None, "the pending command is not durable"
    mirror = read_mirror(artifact.root)
    assert_pending_stays(mirror, command)
    assert mirror["last_receipt"] is None, "a crashed caller records no receipt"
    assert_effect_log(layout.gateway, command, moment)
    assert_evidence(layout.gateway, command, moment, EVIDENCE[moment])
    # A new owner session takes over: that withdrawal is the new epoch barrier.
    binding = new_session(layout, session_id="session-after-crash")
    assert binding.owner_epoch == 9, binding
    assert_session_barrier(read_mirror(artifact.root), binding, layout.gateway.state.last_barrier)
    assert_late(layout.gateway, command, moment, LATE_REBOUND[moment])
    assert_effect_log(layout.gateway, command, moment)
    assert_chain_consistent(read_mirror(artifact.root), receiver_word(layout.gateway, command),
                            command, pending=True, receipt=False)
    assert_new_command_needs_binding_and_basis(artifact, layout)


def test_03_the_lost_lifetime_lease_refuses_the_send_without_any_effect(artifact, layout):
    """H2, the lease variant: the lifetime lease is lost, the process lives.

    This is the B10d writer lease evidence and not a process death: the same
    process closes the lease between `record_pending_command` and
    `send_locked`. The death of a process releases the lease in another way;
    that is proven in `test_02`. Here the lease is gone: the send is refused
    with a machine-readable reason, nothing is ever delivered and there is no
    effect at all.
    """
    real_record = layout.owner.record_pending_command

    def record(command):
        real_record(command)
        # The caller process dies here: the lifetime lease is released.
        layout.lease.close()

    layout.owner.record_pending_command = record
    attempts, outcome = faulted_section(artifact, layout, "before_acceptance",
                                        window_ms=RETRY_WINDOW_MS)
    assert outcome == LEASE_OUTCOME, outcome
    assert attempts == [], "a lost lease never reaches the transport"
    command = layout.owner.state.pending_command
    assert command is not None, "the pending command is not durable"
    mirror = read_mirror(artifact.root)
    assert_pending_stays(mirror, command)
    assert mirror["last_receipt"] is None, "a crashed caller records no receipt"
    assert_effect_log(layout.gateway, command, "before_acceptance")
    assert_evidence(layout.gateway, command, "before_acceptance",
                    EVIDENCE["before_acceptance"])
    # A new owner session takes over: that withdrawal is the new epoch barrier.
    binding = new_session(layout, session_id="session-after-crash")
    assert binding.owner_epoch == 9, binding
    assert_session_barrier(read_mirror(artifact.root), binding, layout.gateway.state.last_barrier)
    assert_late(layout.gateway, command, "before_acceptance",
                LATE_REBOUND["before_acceptance"])
    assert_effect_log(layout.gateway, command, "before_acceptance")
    assert_chain_consistent(read_mirror(artifact.root), receiver_word(layout.gateway, command),
                            command, pending=True, receipt=False)
    # The negative control needs a current lease: the patched recorder, which
    # closes the lease on every pending command, is restored first.
    layout.owner.record_pending_command = real_record
    assert_new_command_needs_binding_and_basis(artifact, layout)


@pytest.mark.parametrize("persist", [True, False])
@pytest.mark.parametrize("moment", MOMENTS)
def test_04_the_receiver_crash_and_restart_never_replay_the_old_command(artifact, layout,
                                                                       moment, persist):
    """H3 receivercrash: the receiver really crashes, with and without extra
    persistence.

    The crash is a transport failure: the machine-readable refusal is the only
    outcome. Before the restart the receiver is not running any more, the
    in-flight queue is gone and the durable boundary is exact: with the extra
    persistence the crash wrote a new checkpoint, without it the last durable
    boundary survives. After the restart the old work is never executed again
    and the late delivery after the new epoch barrier has no effect.
    """
    high_water = (layout.gateway.state.epoch_high_water, layout.gateway.state.sequence_high_water)
    real_crash = layout.gateway.crash
    boundary = {}

    def crash(persist):
        # The state of the receiver just before the crash.
        boundary["queue"] = list(layout.gateway.queue)
        boundary["checkpoint"] = json.loads(
            layout.gateway.state_path.read_text())["anchor_checkpoint"]
        return real_crash(persist)

    layout.gateway.crash = crash
    attempts, outcome = faulted_section(artifact, layout, moment, window_ms=FAULT_WINDOW_MS,
                                        crash=True, persist=persist)
    assert outcome == TRANSPORT_OUTCOME, outcome
    assert len(attempts) == 1, attempts
    # The receiver really crashed: it does not run and the queue is gone.
    assert layout.gateway.running is False
    assert layout.gateway.queue == [], layout.gateway.queue
    assert layout.gateway.work == [], layout.gateway.work
    # The queue before the crash held the command that was in flight.
    if moment == "before_acceptance":
        assert len(boundary["queue"]) == 1, boundary["queue"]
    else:
        assert boundary["queue"] == [], boundary["queue"]
    # The durable boundary of the crash is exact.
    after = json.loads(layout.gateway.state_path.read_text())["anchor_checkpoint"]
    if persist:
        assert after != boundary["checkpoint"], (after, boundary["checkpoint"])
    else:
        assert after == boundary["checkpoint"], (after, boundary["checkpoint"])
    command = layout.owner.state.pending_command
    assert command is not None, "the pending command is not durable"
    mirror = read_mirror(artifact.root)
    assert_pending_stays(mirror, command)
    assert mirror["last_receipt"] is None, "a crashed receiver records no receipt"
    assert_effect_log(layout.gateway, command, moment)
    # The receiver restarts: the durable high waters survive the crash.
    layout.gateway.restart()
    assert layout.gateway.running is True
    assert (layout.gateway.state.epoch_high_water,
            layout.gateway.state.sequence_high_water) >= high_water
    if persist:
        # The crash marked the reserved work as unknown: the receiver knows
        # that a helper wrote, but not what it wrote.
        assert_evidence(layout.gateway, command, moment, EVIDENCE_CRASHED[moment])
    else:
        # Without the final save the last durable boundary survives: the reserve
        # and the helper evidence are exactly as they were written.
        assert_evidence(layout.gateway, command, moment, EVIDENCE[moment])
    barrier = withdraw_epoch(layout)
    assert_barrier_exact(read_mirror(artifact.root), barrier)
    late = late_delivery(layout.gateway, command, moment)
    expected = LATE_CRASHED if persist else LATE_FENCED
    assert (late.status, late.reason, late.effect) == expected[moment], (moment, late)
    assert late.status != "accepted", late
    assert_effect_log(layout.gateway, command, moment)
    assert_chain_consistent(read_mirror(artifact.root), receiver_word(layout.gateway, command),
                            command, pending=True, receipt=False)
    # The negative control needs a current binding: the recovery takes a new
    # session after the restart of the receiver.
    new_session(layout, session_id="session-after-the-restart")
    assert_new_command_needs_binding_and_basis(artifact, layout)


@pytest.mark.parametrize("restore", ["receiver", "mirror", "both"])
@pytest.mark.parametrize("moment", MOMENTS)
def test_05_the_restored_mirror_and_state_inhibit_the_old_command_after_rollback(artifact,
                                                                                  layout, restore, moment):
    """H4 restore mirror: the receiver state, the owner mirror, or both.

    The independent anchor never rolls back. A rolled back receiver state is
    therefore caught and the receiver is inhibited; a rolled back owner mirror
    loses the durable evidence of the pending command. In every case the old
    command can never run again after the new epoch barrier.
    """
    receiver_snapshot = layout.gateway.snapshot_files()
    mirror_snapshot = layout.store.path.read_bytes()
    attempts, outcome = faulted_section(artifact, layout, moment, window_ms=FAULT_WINDOW_MS)
    assert outcome == (TIMEOUT_OUTCOME if moment != "before_acceptance" else TRANSPORT_OUTCOME)
    command = layout.owner.state.pending_command
    assert command is not None, "the pending command is not durable"
    assert_effect_log(layout.gateway, command, moment)
    if restore in ("receiver", "both"):
        # The receiver state is rolled back; the independent anchor is not.
        layout.gateway.restore_files(receiver_snapshot)
        reason = layout.gateway.restart()
        if moment == "before_acceptance":
            # Nothing was durable yet, so there is nothing to roll back.
            assert reason == "recovered"
        else:
            assert reason in ("anchor_checkpoint_mismatch", "rolled_back_state")
            assert layout.gateway.state.inhibited and not layout.gateway.state.queue_known
    if restore in ("mirror", "both"):
        # The owner mirror is rolled back as well: the pending command is gone.
        layout.store.path.write_bytes(mirror_snapshot)
        layout.owner.state = layout.store.load_strict()
        layout.owner.phase = layout.owner.state.phase
    if restore == "mirror":
        # The receiver keeps its evidence, the mirror loses the pending command.
        assert layout.owner.state.pending_command is None
        assert_evidence(layout.gateway, command, moment, EVIDENCE[moment])
    else:
        # The receiver lost the record of the old command: the read back knows
        # nothing, and only the effect log keeps the effect.
        assert_evidence(layout.gateway, command, moment, None)
    barrier = withdraw_epoch(layout)
    assert_barrier_exact(read_mirror(artifact.root), barrier)
    expected = LATE_FENCED if restore == "mirror" else LATE_INHIBITED
    assert_late(layout.gateway, command, moment, expected[moment])
    assert_effect_log(layout.gateway, command, moment)
    mirror = read_mirror(artifact.root)
    if restore == "receiver":
        # The mirror was not rolled back: the uncertain command stays visible.
        assert_pending_stays(mirror, command)
    assert_chain_consistent(mirror, receiver_word(layout.gateway, command), command,
                            pending=(restore == "receiver"),
                            receipt=(restore == "receiver" and moment != "before_acceptance"))
    if layout.gateway.state.inhibited:
        # The receiver is inhibited: there is no current binding, so the
        # negative control has no valid starting point in this cell. The
        # dedicated cell proves both refusals with a valid new command.
        assert layout.owner.binding is None, layout.owner.binding
    else:
        new_session(layout, session_id="session-after-the-restore")
        assert_new_command_needs_binding_and_basis(artifact, layout)


@pytest.mark.parametrize("moment", MOMENTS)
def test_06_the_correction_after_the_send_withdraws_the_owner_epoch(artifact, layout,
                                                                         moment):
    """H5 correction: a registration correction consumes the epoch of the owner.

    The correction blocks the publication, the owner consumes the invalidation
    by withdrawing its epoch, and the old delivery that arrives after that
    barrier has no effect.
    """
    attempts, outcome = faulted_section(artifact, layout, moment, window_ms=FAULT_WINDOW_MS)
    assert outcome == (TIMEOUT_OUTCOME if moment != "before_acceptance" else TRANSPORT_OUTCOME)
    command = layout.owner.state.pending_command
    assert command is not None, "the pending command is not durable"
    assert_effect_log(layout.gateway, command, moment)
    assert_evidence(layout.gateway, command, moment, EVIDENCE[moment])
    # The correction blocks the publication.
    artifact.path.write_text(json.dumps({"format_version": 2, "status": "blocked",
                                          "created_ts": NOW_TS,
                                          "reason": "registered_financial_inputs_changed"}))
    module = execution()
    with publication_lock(artifact.path) as lease:
        assert module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                    request=request(), policy=POLICY, clock=CLOCK,
                                    deadline=SECTION_DEADLINE) is None
    mirror = read_mirror(artifact.root)
    # The invalidation is the new epoch barrier.
    assert mirror["phase"] == "released", mirror
    assert mirror["last_barrier"]["status"] == "accepted", mirror
    assert mirror["last_barrier"]["current_epoch"] == 9, mirror
    assert mirror["last_barrier"]["revoked_through_epoch"] == 8, mirror
    assert mirror["last_barrier"]["queues_fenced"] is True, mirror
    assert mirror["last_barrier"]["safe_state"] == "confirmed", mirror
    assert_pending_stays(mirror, command)
    assert_late(layout.gateway, command, moment, LATE_FENCED[moment])
    assert_effect_log(layout.gateway, command, moment)
    assert_chain_consistent(mirror, receiver_word(layout.gateway, command), command,
                            pending=True, receipt=(moment != "before_acceptance"))
    # A new publication is the current basis again.
    artifact.path.write_text(json.dumps(artifact.plan.to_dict()))
    # The negative control needs a current binding: the recovery takes a new
    # session after the correction.
    new_session(layout, session_id="session-after-the-correction")
    assert_new_command_needs_binding_and_basis(artifact, layout)


def test_07_the_correction_before_the_send_gives_no_send_at_all(artifact, layout):
    """H5 correction, first half: the correction comes before the send.

    The command never leaves the owner: the blocked publication is consumed
    inside the section and the epoch is withdrawn. The four moments do not
    apply here: there is no delivery at all, so there is no moment of the
    receiving fault. The two-process proof of the same ordering stays in
    `test_fast_execution.py::test_01_correction_first_gives_no_send_and_revokes_the_epoch`;
    this cell proves the same ordering inside the composed harness.
    """
    artifact.path.write_text(json.dumps({"format_version": 2, "status": "blocked",
                                          "created_ts": NOW_TS,
                                          "reason": "registered_financial_inputs_changed"}))
    module = execution()
    with publication_lock(artifact.path) as lease:
        assert module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                    request=request(), policy=POLICY, clock=CLOCK,
                                    deadline=SECTION_DEADLINE) is None
    mirror = read_mirror(artifact.root)
    assert mirror["phase"] == "released", mirror
    assert mirror["last_barrier"]["status"] == "accepted", mirror
    assert mirror["last_barrier"]["current_epoch"] == 9, mirror
    assert mirror["pending_command"] is None, "no command was ever built"
    assert mirror["last_receipt"] is None, "no receipt was ever recorded"
    assert not layout.gateway.effect_log, layout.gateway.effect_log
    assert layout.gateway.state.commands == (), layout.gateway.state.commands
    assert layout.gateway.state.epoch_high_water == 9, layout.gateway.state
    # A withdrawn epoch grants no binding: a new non-zero command may not follow.
    assert layout.owner.binding is None, layout.owner.binding
    # And without a current publication there is no basis for a new command.
    with publication_lock(artifact.path) as lease:
        with pytest.raises(module.ExecutionRefused) as error:
            module.validate_current_locked(lease=lease,
                                           expected_digest=module.plan_digest(artifact.plan),
                                           clock=CLOCK, policy=POLICY)
    assert error.value.reason == "publication_blocked", error.value


@pytest.mark.parametrize("moment", MOMENTS)
def test_08_the_operator_stop_withdraws_the_epoch_and_fences_the_old_command(artifact, layout,
                                                                            moment):
    """H6 operatorstop: the operator pulls the epoch in with a stop barrier.

    The barrier confirms that the queues are fenced and the safe state is
    confirmed. The old delivery that arrives after it has no effect, and the
    late acknowledgement never becomes a new accepted status.
    """
    attempts, outcome = faulted_section(artifact, layout, moment, window_ms=FAULT_WINDOW_MS)
    assert outcome == (TIMEOUT_OUTCOME if moment != "before_acceptance" else TRANSPORT_OUTCOME)
    command = layout.owner.state.pending_command
    assert command is not None, "the pending command is not durable"
    assert_effect_log(layout.gateway, command, moment)
    assert_evidence(layout.gateway, command, moment, EVIDENCE[moment])
    barrier = withdraw_epoch(layout, reason="operator_stop")
    assert barrier.revoked_through_epoch == 8 and barrier.current_epoch == 9, barrier
    assert_barrier_exact(read_mirror(artifact.root), barrier)
    assert_late(layout.gateway, command, moment, LATE_FENCED[moment])
    assert_effect_log(layout.gateway, command, moment)
    mirror = read_mirror(artifact.root)
    assert mirror["phase"] == "released", mirror
    assert_pending_stays(mirror, command)
    assert_chain_consistent(mirror, receiver_word(layout.gateway, command), command,
                            pending=True, receipt=(moment != "before_acceptance"))
    # The negative control needs a current binding: the recovery takes a new
    # session after the operator stop.
    new_session(layout, session_id="session-after-the-stop")
    assert_new_command_needs_binding_and_basis(artifact, layout)


@pytest.mark.parametrize("position", ["before", "on", "after"])
@pytest.mark.parametrize("moment", MOMENTS)
def test_09_the_lifetime_lease_before_on_and_after_the_new_epoch_barrier(artifact, layout,
                                                                         moment, position):
    """H7 leasegrens: the lifetime lease is lost before, on or after the barrier.

    The lifetime lease is the B10d writer lease of the owner. Before the
    barrier the old owner cannot withdraw the epoch at all; exactly on the
    barrier the withdrawal reaches the receiver but the mirror can no longer
    record it; after the barrier the withdrawal is recorded and the lease is
    gone all the same. In every position a new owner session takes over, the
    old command is fenced and the late delivery has no effect.
    """
    attempts, outcome = faulted_section(artifact, layout, moment, window_ms=FAULT_WINDOW_MS)
    assert outcome == (TIMEOUT_OUTCOME if moment != "before_acceptance" else TRANSPORT_OUTCOME)
    command = layout.owner.state.pending_command
    assert command is not None, "the pending command is not durable"
    assert_effect_log(layout.gateway, command, moment)
    assert_evidence(layout.gateway, command, moment, EVIDENCE[moment])
    mirror = read_mirror(artifact.root)
    assert_pending_stays(mirror, command)
    if position == "before":
        # The lease is gone before the barrier: the old owner cannot withdraw.
        layout.lease.close()
        assert layout.owner.stop(reason="operator_stop") is None
        assert layout.owner.phase == "inhibited"
        assert layout.owner.recovery_required is True
        # The durable mirror keeps the evidence of the uncertain command.
        mirror = read_mirror(artifact.root)
        assert mirror["phase"] == "owned", mirror
        assert_pending_stays(mirror, command)
    elif position == "on":
        # The lease is gone exactly at the barrier: the withdrawal reaches the
        # receiver, the mirror can no longer record it.
        real_revoke = layout.gateway.revoke_and_stop

        def revoke(request, *, deadline):
            receipt = real_revoke(request, deadline=deadline)
            # The lifetime lease is lost exactly on the barrier.
            layout.lease.close()
            return receipt

        layout.gateway.revoke_and_stop = revoke
        with pytest.raises(RecoveryRequired) as error:
            withdraw_epoch(layout)
        layout.gateway.__dict__.pop("revoke_and_stop", None)
        assert error.value.reason == "owner_lease_missing", error.value
        assert layout.owner.recovery_required is True
        # The durable mirror shows the withdrawal that could not be finished.
        mirror = read_mirror(artifact.root)
        assert mirror["phase"] == "revoking", mirror
        assert_pending_stays(mirror, command)
    else:
        # The barrier is recorded and the lease is gone all the same.
        barrier = withdraw_epoch(layout)
        assert barrier.status == "accepted" and barrier.current_epoch == 9, barrier
        assert_barrier_exact(read_mirror(artifact.root), barrier)
        mirror = read_mirror(artifact.root)
        assert mirror["phase"] == "released", mirror
        assert mirror["last_barrier"]["current_epoch"] == 9, mirror
        layout.lease.close()
    # A new session takes over: that withdrawal is the new epoch barrier.
    binding = new_session(layout, session_id="session-after-lease")
    assert binding.owner_epoch == (10 if position in ("on", "after") else 9), binding
    assert_session_barrier(read_mirror(artifact.root), binding, layout.gateway.state.last_barrier)
    assert_late(layout.gateway, command, moment, LATE_REBOUND[moment])
    assert_effect_log(layout.gateway, command, moment)
    assert_chain_consistent(read_mirror(artifact.root), receiver_word(layout.gateway, command),
                            command, pending=True, receipt=(moment != "before_acceptance"))
    assert_new_command_needs_binding_and_basis(artifact, layout)


# -- the time boundaries of the lease and of the command (fix round 1) --------

#: The owner lease of the boundary cells: the fake caps its own lease at five
#: seconds, so this request is exactly the cap.
LEASE_END_MS = 5000


def clock_at(gateway, utc_ms):
    """The clock sample of the manual clock: the boot time follows the fake.

    The owner's own clock follows the fake as well, so the two never disagree
    about the remaining lease.
    """
    return replace(CLOCK, utc_ms=utc_ms, boottime_ns=MONO + gateway.monotonic_ns)


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_09b_the_owner_lease_time_boundary_is_exact(artifact, offset):
    """I4: the owner lease at one millisecond before, on and after the bound.

    The owner lease is a time bound and not a flock: the owner refuses to
    build a command when the clock reached `owner_not_after_ms`, and the
    receiver refuses a command that arrives later. With the manual clock the
    boundary is exact: one millisecond before the bound the command is built
    and delivered, on the bound and one millisecond later no command is built
    at all and nothing is delivered.
    """
    module = execution()
    bound = layout_at(artifact.root, requested_not_after_ms=NOW + LEASE_END_MS)
    try:
        assert bound.binding.owner_not_after_ms == NOW + LEASE_END_MS, bound.binding
        moment = NOW + LEASE_END_MS + offset
        bound.gateway.advance_clock(LEASE_END_MS + offset)
        clock = clock_at(bound.gateway, moment)
        with publication_lock(artifact.path) as lease:
            if offset < 0:
                receipt = module.owner_section(lease=lease, owner=bound.owner,
                                               receiver=bound.gateway, request=request(),
                                               policy=POLICY, clock=clock,
                                               deadline=SECTION_DEADLINE)
                assert receipt.status == "accepted", receipt
                assert len(bound.gateway.effect_log) == 3, bound.gateway.effect_log
            else:
                with pytest.raises(module.ExecutionRefused) as error:
                    module.owner_section(lease=lease, owner=bound.owner, receiver=bound.gateway,
                                         request=request(), policy=POLICY, clock=clock,
                                         deadline=SECTION_DEADLINE)
                # The lease is over: the section has no budget left, so no
                # command is built, nothing is delivered and no effect exists.
                assert error.value.reason == "budget_exhausted", error.value
                assert not bound.gateway.effect_log, bound.gateway.effect_log
                assert bound.gateway.state.commands == (), bound.gateway.state.commands
                mirror = read_mirror(artifact.root)
                assert mirror["pending_command"] is None, mirror
    finally:
        bound.lease.close()


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_09c_the_command_deadline_boundary_is_exact(artifact, offset):
    """I4: the command deadline at one millisecond before, on and after the bound.

    The command carries its own deadline: `execute_before_ms`. The receiver
    refuses a command that arrives on or after that moment, and it accepts
    the same command one millisecond earlier. The bound is the owner lease
    here, so the plan lease of the publication stays valid.
    """
    module = execution()
    bound = layout_at(artifact.root, requested_not_after_ms=NOW + LEASE_END_MS)
    try:
        command = build_command(module, bound, artifact)
        assert command.execute_before_ms == NOW + LEASE_END_MS, command
        bound.owner.record_pending_command(command)
        moment = command.execute_before_ms + offset
        bound.gateway.advance_clock(LEASE_END_MS + offset)
        clock = clock_at(bound.gateway, moment)
        with publication_lock(artifact.path) as lease:
            receipt = module.send_locked(command, lease=lease, owner=bound.owner,
                                         receiver=bound.gateway, deadline=SECTION_DEADLINE,
                                         clock=clock, policy=POLICY)
        if offset < 0:
            assert receipt.status == "accepted", receipt
            assert len(bound.gateway.effect_log) == 3, bound.gateway.effect_log
        else:
            # On the bound and later the command is expired: no effect at all.
            assert receipt.status == "rejected", receipt
            assert receipt.reason == "command_expired", receipt
            assert receipt.effect == "none", receipt
            assert not bound.gateway.effect_log, bound.gateway.effect_log
    finally:
        bound.lease.close()


def test_09d_the_helper_effects_before_the_expiry_stay_visible(artifact):
    """I4: the effects before the expiry stay visible, no new one after it.

    One helper wrote before the bound of the owner lease. After the expiry the
    old delivery is refused: the effect log keeps exactly the one effect that
    happened before, and no helper writes again.
    """
    module = execution()
    bound = layout_at(artifact.root, requested_not_after_ms=NOW + LEASE_END_MS)
    try:
        command = build_command(module, bound, artifact)
        bound.owner.record_pending_command(command)
        # The delivery before the expiry: the reserve and one helper.
        bound.gateway.enqueue(command)
        reserved = bound.gateway.deliver()
        assert (reserved.status, reserved.reason) == ("unconfirmed", "pending_effect"), reserved
        helper = bound.gateway.step_helper("setpoint", command=command)
        assert (helper.status, helper.effect) == ("unconfirmed", "partial"), helper
        assert len(bound.gateway.effect_log) == 1, bound.gateway.effect_log
        # The clock passes the bound of the owner lease.
        bound.gateway.advance_clock(LEASE_END_MS + 1)
        clock = clock_at(bound.gateway, NOW + LEASE_END_MS + 1)
        with publication_lock(artifact.path) as lease:
            receipt = module.send_locked(command, lease=lease, owner=bound.owner,
                                         receiver=bound.gateway, deadline=SECTION_DEADLINE,
                                         clock=clock, policy=POLICY)
        assert receipt.status != "accepted", receipt
        assert receipt.effect == "partial", receipt
        # The earlier effect is still visible and no new effect was written.
        assert len(bound.gateway.effect_log) == 1, bound.gateway.effect_log
        assert bound.gateway.effect_log[0][1] == "setpoint", bound.gateway.effect_log
    finally:
        bound.lease.close()


#: The window that the transport fault burns in the boundary cells: it is
#: small, so the manual clock can still reach the boundary of the command.
HOLD_WINDOW_MS = 10
#: The machine-readable refusal of the next helper on the clock boundary of
#: the command: the command is expired, so the helper does not write.
HELPER_EXPIRED = {
    "after_reserve": ("rejected", "command_expired", "none"),
    "after_first_helper": ("unconfirmed", "command_expired", "partial"),
}
#: The refusal of the next helper after the new epoch barrier. The effect
#: that was written before the boundary stays visible in the receipt.
HELPER_FENCED = {
    ("after_reserve", -1): ("unconfirmed", "epoch_withdrawn", "partial"),
    ("after_reserve", 0): ("rejected", "epoch_withdrawn", "none"),
    ("after_reserve", 1): ("rejected", "epoch_withdrawn", "none"),
    ("after_first_helper", -1): ("unconfirmed", "epoch_withdrawn", "partial"),
    ("after_first_helper", 0): ("unconfirmed", "epoch_withdrawn", "partial"),
    ("after_first_helper", 1): ("unconfirmed", "epoch_withdrawn", "partial"),
}


@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("moment", ["after_reserve", "after_first_helper"])
def test_09g_the_next_helper_at_the_clock_boundary_is_exact(artifact, moment, offset):
    """I4: the actual next helper at the clock boundary of the command.

    The reserve is real and happens through the owner and the send boundary.
    The execution is held at the moment of the fault: for the reserved moment
    no helper wrote yet, for the partial moment exactly one helper wrote. Then
    the manual clock reaches the boundary of the command and the **actual
    next** helper is offered: the first helper for the reserved moment, the
    next helper for the partial moment.

    One millisecond before the boundary the helper really writes and the
    effect log grows by exactly one entry. On the boundary and one
    millisecond later the helper is refused with a machine-readable reason and
    no entry is added; the effect that was there before stays there. Then the
    ordered recovery follows: the stop barrier and the late delivery of the
    old command, which never writes again.
    """
    bound = layout_at(artifact.root, requested_not_after_ms=NOW + LEASE_END_MS)
    try:
        # The real reserve through the owner and the send boundary; the
        # execution is held at the moment of the fault.
        attempts, outcome = faulted_section(artifact, bound, moment, window_ms=HOLD_WINDOW_MS,
                                            answer=False)
        assert outcome == TRANSPORT_OUTCOME, outcome
        assert len(attempts) == 1, attempts
        command = bound.owner.state.pending_command
        assert command is not None, "the pending command is not durable"
        boundary = command.execute_before_ms
        assert boundary == NOW + LEASE_END_MS, command
        # The baseline of the moment: no helper wrote yet, or exactly one.
        baseline = EXPECTED_EFFECTS[moment]
        assert len(bound.gateway.effect_log) == baseline, bound.gateway.effect_log
        # The full chain of the old command after the fault: the pending
        # command is durable and names the old command on all eight fields,
        # and the mirror receipt is absent, because the transport exception
        # never let the owner record one.
        assert_chain_consistent(read_mirror(artifact.root),
                                receiver_word(bound.gateway, command), command,
                                pending=True, receipt=False)
        # The manual clock reaches the boundary of the command.
        bound.gateway.advance_clock(boundary - bound.gateway.utc_ms + offset)
        assert bound.gateway.utc_ms == boundary + offset, bound.gateway.utc_ms
        # The actual next helper is offered at the boundary.
        part = "setpoint" if moment == "after_reserve" else "mode"
        receipt = bound.gateway.step_helper(part, command=command)
        # The helper receipt at the boundary names the old command on all
        # eight fields.
        assert_identity(receipt, command, what="helper receipt at the clock boundary")
        if offset < 0:
            # One millisecond before the boundary the helper really writes.
            assert (receipt.status, receipt.reason, receipt.effect) == (
                "unconfirmed", "pending_effect", "partial"), receipt
            assert len(bound.gateway.effect_log) == baseline + 1, bound.gateway.effect_log
            assert bound.gateway.effect_log[-1][1] == part, bound.gateway.effect_log
        else:
            # On the boundary and later the command is expired: no new effect.
            assert (receipt.status, receipt.reason, receipt.effect) == HELPER_EXPIRED[moment], (
                receipt, moment)
            assert len(bound.gateway.effect_log) == baseline, bound.gateway.effect_log
        # The ordered recovery: the stop barrier, then the late old delivery.
        barrier = withdraw_epoch(bound, reason="operator_stop")
        assert_barrier_exact(read_mirror(artifact.root), barrier)
        late = bound.gateway.step_helper(part, command=command)
        assert (late.status, late.reason, late.effect) == HELPER_FENCED[(moment, offset)], (
            moment, offset, late)
        assert late.status != "accepted", late
        # The late helper receipt names the old command on all eight fields.
        assert_identity(late, command, what="late helper receipt after the barrier")
        # The full chain after the barrier: the pending command of the old
        # command is still visible, the mirror receipt is still absent and the
        # late helper receipt names the old command.
        assert_chain_consistent(read_mirror(artifact.root), late, command,
                                pending=True, receipt=False)
        # The old effect log never grows after the barrier.
        expected = baseline + (1 if offset < 0 else 0)
        assert len(bound.gateway.effect_log) == expected, bound.gateway.effect_log
    finally:
        bound.lease.close()


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_09h_the_completed_effect_has_no_next_helper_at_the_clock_boundary(artifact, offset):
    """I4, M4: the completed effect has no next helper left.

    The whole effect exists: the three helpers wrote and the acknowledgement
    is lost. There is no next helper to offer, so the helper boundary of the
    command is **not applicable** here. That N/A is proved by the effect log
    itself: the three parts wrote, so there is no fourth part. What can still
    be observed is the query history: the exact query gives the original
    receiver evidence, at the boundary and after the new epoch barrier, and
    the effect log never grows.
    """
    bound = layout_at(artifact.root, requested_not_after_ms=NOW + LEASE_END_MS)
    try:
        attempts, outcome = faulted_section(artifact, bound, "after_effect",
                                            window_ms=HOLD_WINDOW_MS, answer=False)
        assert outcome == TRANSPORT_OUTCOME, outcome
        assert len(attempts) == 1, attempts
        command = bound.owner.state.pending_command
        assert command is not None, "the pending command is not durable"
        boundary = command.execute_before_ms
        assert boundary == NOW + LEASE_END_MS, command
        # The real effect evidence: the three parts wrote, so no next helper.
        assert [entry[1] for entry in bound.gateway.effect_log] == [
            "setpoint", "mode", "stop_inverter"], bound.gateway.effect_log
        assert len(bound.gateway.effect_log) == 3, bound.gateway.effect_log
        # The original receiver evidence, before the boundary.
        original = receiver_word(bound.gateway, command)
        assert original is not None, original
        assert (original.status, original.reason, original.effect) == (
            "unconfirmed", "effect_unknown", "unknown"), original
        # The original evidence names the old command on all eight fields.
        assert_identity(original, command, what="original query receipt")
        # The full chain of the old command after the fault: the pending
        # command is durable and names the old command on all eight fields,
        # and the mirror receipt is absent, because the transport exception
        # never let the owner record one.
        assert_chain_consistent(read_mirror(artifact.root), original, command,
                                pending=True, receipt=False)
        # The manual clock reaches the boundary of the command.
        bound.gateway.advance_clock(boundary - bound.gateway.utc_ms + offset)
        assert bound.gateway.utc_ms == boundary + offset, bound.gateway.utc_ms
        # The query history: the exact query gives the same evidence again.
        readback = receiver_word(bound.gateway, command)
        assert readback is not None, readback
        assert (readback.status, readback.reason, readback.effect) == (
            "unconfirmed", "effect_unknown", "unknown"), readback
        # The readback is the original evidence, field by field: the same
        # identity, the same helper results and the same observation moment.
        assert asdict(readback) == asdict(original), (readback, original)
        assert readback.helper_results == original.helper_results, readback
        assert readback.observed_at_ms == original.observed_at_ms, readback
        assert_identity(readback, command, what="query receipt at the clock boundary")
        # The ordered recovery: the stop barrier, then the late old delivery.
        barrier = withdraw_epoch(bound, reason="operator_stop")
        assert_barrier_exact(read_mirror(artifact.root), barrier)
        late = receiver_word(bound.gateway, command)
        assert late is not None, late
        assert (late.status, late.reason, late.effect) == (
            "unconfirmed", "effect_unknown", "unknown"), late
        # The late query is the original evidence, field by field: the same
        # identity, the same helper results and the same observation moment.
        assert asdict(late) == asdict(original), (late, original)
        assert late.helper_results == original.helper_results, late
        assert late.observed_at_ms == original.observed_at_ms, late
        assert_identity(late, command, what="query receipt after the barrier")
        # The full chain after the barrier: the pending command of the old
        # command is still visible, the mirror receipt is still absent and the
        # late query receipt names the old command.
        assert_chain_consistent(read_mirror(artifact.root), late, command,
                                pending=True, receipt=False)
        # The effect log never grows: there is no helper left to write.
        assert len(bound.gateway.effect_log) == 3, bound.gateway.effect_log
    finally:
        bound.lease.close()


def test_09e_a_new_section_after_an_exact_stop_is_allowed(artifact, layout):
    """I5: after an exact stop and a rebinding a new section is allowed again.

    The old command stays visible in the mirror until the new command replaces
    it. The new section has a new epoch, a new sequence and the current
    publication as its basis: it is accepted, and only the new command has
    effects. The isolated negative controls follow: a foreign binding and a
    changed publication are both refused.
    """
    module = execution()
    attempts, outcome = faulted_section(artifact, layout, "after_first_helper",
                                        window_ms=FAULT_WINDOW_MS)
    assert outcome == TIMEOUT_OUTCOME, outcome
    old = layout.owner.state.pending_command
    assert old is not None, "the pending command is not durable"
    # The exact stop: the epoch is withdrawn and the barrier is confirmed.
    barrier = withdraw_epoch(layout, reason="operator_stop")
    assert_barrier_exact(read_mirror(artifact.root), barrier)
    # A new session takes over: that withdrawal is the new epoch barrier.
    binding = new_session(layout, session_id="session-after-stop")
    assert binding.owner_epoch == 10, binding
    assert_session_barrier(read_mirror(artifact.root), binding, layout.gateway.state.last_barrier)
    # The normal section is allowed again: the transport works again.
    restore_transport(layout.gateway)
    with publication_lock(artifact.path) as lease:
        receipt = module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                       request=request(), policy=POLICY, clock=CLOCK,
                                       deadline=SECTION_DEADLINE)
    assert receipt.status == "accepted", receipt
    assert receipt.owner_epoch == 10, receipt
    assert receipt.sequence > old.sequence, (receipt, old)
    assert receipt.command_id != old.command_id, (receipt, old)
    # The old command keeps its own effect; the new command has its own three.
    old_effects = [entry for entry in layout.gateway.effect_log if entry[0] == old.command_id]
    new_effects = [entry for entry in layout.gateway.effect_log if entry[0] == receipt.command_id]
    assert len(old_effects) == 1, layout.gateway.effect_log
    assert len(new_effects) == 3, layout.gateway.effect_log
    layout.gateway.send(old, deadline=SECTION_DEADLINE)
    assert len(layout.gateway.effect_log) == 4, layout.gateway.effect_log
    # The mirror holds the new command, not the old one.
    mirror = read_mirror(artifact.root)
    assert mirror["pending_command"] is None, mirror
    assert mirror["last_receipt"]["command_id"] == receipt.command_id, mirror
    assert identity_of(mirror["last_receipt"]) == identity_of(receipt), mirror
    # The isolated negative controls: a foreign binding and a changed basis.
    assert_new_command_needs_binding_and_basis(artifact, layout)


def test_09f_a_delayed_delivery_inside_the_window_goes_through_the_owner(artifact, layout):
    """I5, M1: a delivery that is late but inside the window is processed.

    The command is built by the owner and sent through the real send
    boundary. The receiver receives it later, but still inside the window of
    the command: the reservation, the three helpers and the acceptance follow.
    The acceptance carries the moment of the delivery, not the moment of the
    issue.
    """
    module = execution()
    real_send = layout.gateway.send
    delay_ms = 200

    def send(command, *, deadline):
        # The delivery is late, but it stays inside the window.
        layout.gateway.advance_clock(delay_ms)
        return real_send(command, deadline=deadline)

    layout.gateway.send = send
    try:
        with publication_lock(artifact.path) as lease:
            receipt = module.owner_section(lease=lease, owner=layout.owner, receiver=layout.gateway,
                                           request=request(), policy=POLICY, clock=CLOCK,
                                           deadline=SECTION_DEADLINE)
    finally:
        layout.gateway.__dict__.pop("send", None)
    assert receipt.status == "accepted", receipt
    assert len(receipt.helper_results) == 3, receipt
    # The acceptance carries the moment of the delivery, not the issue moment.
    assert receipt.accepted_ts == NOW + delay_ms, receipt
    assert receipt.observed_at_ms == NOW + delay_ms, receipt
    command = layout.owner.state.pending_command
    assert command is None, "the accepted command is not pending any more"
    assert [entry[1] for entry in layout.gateway.effect_log] == [
        "setpoint", "mode", "stop_inverter"]
    # The owner wrote the acceptance durably.
    mirror = read_mirror(artifact.root)
    assert mirror["last_receipt"]["status"] == "accepted", mirror
    assert identity_of(mirror["last_receipt"]) == identity_of(receipt), mirror


# -- the consumer sees the same outcome ----------------------------------------

def test_10_the_transport_failure_is_an_uncertain_outcome_for_the_consumer(tmp_path):
    """The consumer treats the transport failure as any other uncertain refusal.

    The pending command of the owner is the evidence: the row `unconfirmed` is
    applied, the recovery duty stands and the verdict denies.
    """
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    delayed_transport(harness.gateway, "after_first_helper", window_ms=RETRY_WINDOW_MS)
    receipts = consume(harness, decision)
    assert receipts == ()
    verdict = harness.runner.last_actuation
    assert verdict.allowed is False
    assert verdict.reason == "receiver_unavailable"
    controller = harness.runner.state.battery(0)
    assert controller.execution_status == "unconfirmed"
    assert controller.recovery_required is True
    assert controller.pending_command is not None
    assert controller.last_accepted_command is None
    assert controller.last_command_w == 0.0
    # The partial effect is real: the effect log is the only effect evidence.
    assert len(harness.gateway.effect_log) == 1, harness.gateway.effect_log
    # The uncertain command is never replaced in the next tick.
    harness.gateway.advance_clock(60_000)
    again = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 60)
    assert consume(harness, again, now_ms=NOW + 60_000) == ()
    assert harness.runner.last_actuation.reason == "recovery_required"


@pytest.mark.parametrize("failing", ["send", "retry", "query"])
def test_10b_a_transport_failure_on_every_step_is_a_refusal(tmp_path, failing):
    """I8: a transport failure on the first send, on the retry or on the query.

    Every step of the bounded send can fail. The first send really delivers
    the reservation, so the retry and the query are really reached inside one
    budget. The outcome is the same in every case: a machine-readable
    refusal, never an invented receipt, a durable pending command, an
    uncertain controller row with the recovery duty, a denied verdict and no
    next command.
    """
    harness = build(tmp_path)
    gateway = harness.gateway
    real_query = gateway.query
    calls = []

    def send(command, *, deadline):
        calls.append("send")
        index = len(calls)
        if failing == "send" and index == 1:
            raise TimeoutError("transport_timeout")
        if failing == "retry" and index == 2:
            raise TimeoutError("transport_timeout")
        # The delivery really reaches the receiver: the reservation is real.
        gateway.enqueue(command)
        return gateway.deliver()

    def query(**kwargs):
        calls.append("query")
        if failing == "query":
            raise TimeoutError("transport_timeout")
        return real_query(**kwargs)

    gateway.send, gateway.query = send, query
    receipts = consume(harness, propose(harness.plan, harness.runner.state))
    # The failing step is really reached inside one budget.
    assert calls == {"send": ["send"], "retry": ["send", "send"],
                     "query": ["send", "send", "query"]}[failing], calls
    assert receipts == ()
    verdict = harness.runner.last_actuation
    assert verdict.allowed is False
    assert verdict.reason == "receiver_unavailable", verdict
    controller = harness.runner.state.battery(0)
    assert controller.execution_status == "unconfirmed", controller
    assert controller.recovery_required is True, controller
    assert controller.pending_command is not None, controller
    assert controller.last_accepted_command is None, controller
    # No receipt was invented and the pending command stays durable.
    mirror = read_mirror(tmp_path)
    assert mirror["last_receipt"] is None, mirror
    assert mirror["pending_command"] is not None, mirror
    assert mirror["pending_command"]["command_id"] == controller.pending_command.command_id, mirror
    # No effect was written: the reservation is not an effect.
    assert gateway.effect_log == [], gateway.effect_log
    # The next command is refused: the uncertainty stands.
    harness.gateway.advance_clock(60_000)
    again = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 60)
    assert consume(harness, again, now_ms=NOW + 60_000) == ()
    assert harness.runner.last_actuation.reason == "recovery_required"
    assert gateway.effect_log == [], gateway.effect_log


def test_11_the_late_acknowledgement_is_never_a_new_accepted_status(tmp_path):
    """M4: the effect stays durably unknown and the read back is the original.

    The acknowledgement is lost, so the effect of the command stays durably
    unknown. The exact query returns the original receiver evidence again,
    field by field: the identity, the helper results, the observation moment
    and the effect. A read back is never a new acceptance and it never lifts
    the recovery duty of the consumer.
    """
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    harness.gateway.lose_ack()
    original = []
    real_send = harness.gateway.send

    def send(command, *, deadline):
        receipt = real_send(command, deadline=deadline)
        original.append(receipt)
        # The transport burns the rest of the budget after the delivery.
        harness.gateway.advance_clock(FAULT_WINDOW_MS)
        return receipt

    harness.gateway.send = send
    receipts = consume(harness, decision)
    assert receipts == ()
    assert harness.runner.last_actuation.reason == "deadline_exceeded"
    controller = harness.runner.state.battery(0)
    assert controller.execution_status == "unconfirmed"
    assert controller.recovery_required is True
    # The effect is real and durable: the effect log is the effect evidence.
    assert len(harness.gateway.effect_log) == 3, harness.gateway.effect_log
    # The read back gives exactly the original acknowledgement.
    assert len(original) == 1, original
    command = harness.owner.state.pending_command
    assert command is not None
    assert controller.last_accepted_command is None
    readback = harness.gateway.query(installation_id=command.installation_id,
                                     owner_epoch=command.owner_epoch, sequence=command.sequence,
                                     command_id=command.command_id,
                                     payload_digest=payload_digest(command),
                                     deadline=SECTION_DEADLINE)
    assert (readback.status, readback.effect, readback.accepted_ts) == (
        "unconfirmed", "unknown", None)
    # Field by field the read back is the original receiver evidence.
    assert asdict(readback) == asdict(original[0]), (readback, original[0])
    assert readback.helper_results == original[0].helper_results
    assert readback.observed_at_ms == original[0].observed_at_ms
    # The late acknowledgement is history: it never becomes an accepted status.
    assert controller.execution_status == "unconfirmed"
    assert controller.last_accepted_command is None
    assert controller.last_command_w == 0.0
    assert harness.owner.state.last_receipt.status == "unconfirmed"


def test_11b_a_late_accepted_acknowledgement_stays_history(tmp_path):
    """I6: a really accepted acknowledgement that comes after the deadline.

    The delivery takes longer than the budget of the section: the receiver
    completes the effect and answers, but the local deadline is already over.
    That answer is history: it is written durably, it clears no pending
    command and it never becomes an accepted controller status. The recovery
    duty stands and the next tick is refused.
    """
    harness = build(tmp_path)
    decision = propose(harness.plan, harness.runner.state)
    real_send = harness.gateway.send

    def send(command, *, deadline):
        # The delivery takes longer than the local budget of the section.
        harness.gateway.advance_clock(1000)
        return real_send(command, deadline=deadline)

    harness.gateway.send = send
    receipts = consume(harness, decision)
    assert receipts == ()
    verdict = harness.runner.last_actuation
    assert verdict.allowed is False
    assert verdict.reason == "deadline_exceeded", verdict
    controller = harness.runner.state.battery(0)
    # The accepted answer never becomes an accepted controller status.
    assert controller.execution_status == "unconfirmed", controller
    assert controller.recovery_required is True, controller
    assert controller.last_accepted_command is None, controller
    assert controller.last_command_w == 0.0, controller
    assert controller.held_setpoint_w == 0.0, controller
    # The answer is history and the pending command stays durable.
    mirror = read_mirror(tmp_path)
    assert mirror["last_receipt"]["status"] == "accepted", mirror
    assert mirror["last_receipt"]["accepted_ts"] is not None, mirror
    assert mirror["pending_command"] is not None, mirror
    assert mirror["pending_command"]["command_id"] == mirror["last_receipt"]["command_id"], mirror
    # The effect is real: the effect log is the effect evidence.
    assert len(harness.gateway.effect_log) == 3, harness.gateway.effect_log
    # The recovery duty stands: the next tick is refused.
    harness.gateway.advance_clock(60_000)
    again = propose(harness.plan, harness.runner.state, timestamp=NOW_TS + 60)
    assert consume(harness, again, now_ms=NOW + 60_000) == ()
    assert harness.runner.last_actuation.reason == "recovery_required"
    assert len(harness.gateway.effect_log) == 3, harness.gateway.effect_log


def test_12_a_delayed_delivery_inside_the_window_is_processed_correctly(artifact, layout):
    """M1, first half: a delivery that is late but inside the window is correct.

    The reservation, the helpers and the acceptance are all there, and the
    acceptance carries the moment of the delivery, not the moment of the issue.
    """
    module = execution()
    command = build_command(module, layout, artifact)
    layout.gateway.enqueue(command)
    layout.gateway.advance_clock(500)
    receipt = layout.gateway.deliver()
    assert (receipt.status, receipt.reason) == ("unconfirmed", "pending_effect"), receipt
    for part in ("setpoint", "mode", "stop_inverter"):
        receipt = layout.gateway.step_helper(part, command=command)
    assert (receipt.status, receipt.effect) == ("accepted", "complete"), receipt
    # The acceptance carries the moment of the delivery, not the issue moment.
    assert receipt.accepted_ts == layout.gateway.utc_ms
    assert receipt.observed_at_ms == layout.gateway.utc_ms
    assert receipt.accepted_ts != command.issued_at_ms
    assert [entry[1] for entry in layout.gateway.effect_log] == [
        "setpoint", "mode", "stop_inverter"]
    # The owner's mirror records the acceptance: the pending command is gone.
    layout.owner.record_pending_command(command)
    layout.owner.record_receipt(receipt)
    mirror = read_mirror(artifact.root)
    assert mirror["pending_command"] is None
    assert mirror["last_receipt"]["status"] == "accepted"


# -- fix round 1: the delivered duplicate (I1) --------------------------------

def test_14_a_delivered_duplicate_keeps_the_live_reservation_uncertain(tmp_path):
    """I1: a delivered duplicate never ends a live reservation.

    The first delivery really reserves the command at the receiver. The bounded
    retry really reaches the receiver again and is refused as a duplicate. The
    exact query reads the reservation back. All three observations are real
    receiver answers, so the outcome has to stay uncertain: the pending command
    stays durable, the recovery duty stands, and the reservation is still able
    to write one helper before the new epoch barrier.
    """
    harness = build(tmp_path)
    gateway = harness.gateway
    real_send, real_query = gateway.send, gateway.query
    commands, deadlines, observations = [], [], []

    def word(receipt):
        return [receipt.status, receipt.reason, receipt.effect]

    def send(command, *, deadline):
        commands.append(command)
        deadlines.append(deadline)
        if len(commands) == 1:
            # The first delivery really reaches the receiving boundary.
            gateway.enqueue(command)
            receipt = gateway.deliver()
        else:
            # The bounded retry really reaches the receiver again.
            receipt = real_send(command, deadline=deadline)
        observations.append(["send", word(receipt)])
        return receipt

    def query(**kwargs):
        deadlines.append(kwargs["deadline"])
        receipt = real_query(**kwargs)
        observations.append(["query", word(receipt)])
        return receipt

    gateway.send, gateway.query = send, query
    try:
        receipts = consume(harness, propose(harness.plan, harness.runner.state))
        # Three real receiver answers: the reserve, the duplicate refusal and
        # the exact read back of the same command.
        assert observations == [
            ["send", ["unconfirmed", "pending_effect", "none"]],
            ["send", ["rejected", "stale_or_duplicate_sequence", "none"]],
            ["query", ["unconfirmed", "pending_effect", "none"]],
        ], observations
        # The retry carries the full payload and both deadline fields.
        assert len(commands) == 2
        assert commands[0].to_dict() == commands[1].to_dict(), commands
        assert deadlines[0] == deadlines[1], deadlines
        assert deadlines[0].monotonic_end_ns == deadlines[1].monotonic_end_ns
        assert deadlines[0].execute_before_ms == deadlines[1].execute_before_ms
        # The duplicate refusal is no terminal proof: the outcome is uncertain.
        assert receipts[0].status == "unconfirmed", receipts[0]
        assert receipts[0].reason == "pending_effect", receipts[0]
        # The uncertain command stays durable in the owner's own mirror.
        mirror = read_mirror(tmp_path)
        assert mirror["pending_command"]["command_id"] == commands[0].command_id, mirror
        assert mirror["last_receipt"]["status"] == "unconfirmed", mirror
        # The consumer keeps the recovery duty: the effect is not proven absent.
        controller = harness.runner.state.battery(0)
        assert controller.execution_status == "unconfirmed", controller
        assert controller.recovery_required is True, controller
        assert controller.pending_command is not None, controller
        assert controller.last_accepted_command is None, controller
        # The reservation is still live: one helper writes before the barrier.
        late = gateway.step_helper("setpoint", command=commands[0])
        assert (late.status, late.effect) == ("unconfirmed", "partial"), late
        assert len(gateway.effect_log) == 1, gateway.effect_log
        effect_before_barrier = list(gateway.effect_log)
        # The new epoch barrier ends the reservation: no old effect after it.
        barrier = harness.owner.stop(reason="operator_stop")
        assert barrier.status == "accepted", barrier
        gateway.step_helper("mode", command=commands[0])
        assert gateway.effect_log == effect_before_barrier, gateway.effect_log
    finally:
        harness.lease.close()


def test_15_the_reducer_keeps_a_live_reservation_uncertain_in_every_order():
    """I1: the order of the observations never ends a live reservation.

    The three observations of the delivered duplicate are permuted: the
    outcome stays uncertain in every order. The other rules of the reducer stay
    exactly as they were: an acceptance wins, an observed effect makes the
    outcome uncertain and a proven empty effect is still a rejection.
    """
    module = execution()
    reserve = observed_receipt("unconfirmed", "none", "pending_effect")
    duplicate = observed_receipt("rejected", "none", "stale_or_duplicate_sequence")
    readback = observed_receipt("unconfirmed", "none", "pending_effect")
    for order in permutations((reserve, duplicate, readback)):
        outcome = module._conservative(list(order))
        assert outcome.status == "unconfirmed", order
        assert outcome.reason == "pending_effect", order
    # An acceptance is proof and still wins.
    accepted = observed_receipt("accepted", "complete", "accepted")
    assert module._conservative([reserve, duplicate, accepted]).status == "accepted"
    # An observed effect makes the outcome uncertain, in every order.
    partial = observed_receipt("unconfirmed", "partial", "effect_partial")
    unknown = observed_receipt("unconfirmed", "unknown", "effect_unknown")
    for order in permutations((partial, duplicate)):
        outcome = module._conservative(list(order))
        assert outcome.status == "unconfirmed", order
        assert outcome.effect == "partial", order
    for order in permutations((unknown, duplicate)):
        outcome = module._conservative(list(order))
        assert outcome.status == "unconfirmed", order
        assert outcome.effect == "unknown", order
    # A proven empty effect is still a rejection.
    empty = observed_receipt("unconfirmed", "none", "no_effect")
    assert module._conservative([duplicate]).status == "rejected"
    assert module._conservative([duplicate, empty]).status == "rejected"
    assert module._conservative([empty, duplicate]).status == "rejected"


def test_13_the_matrix_declares_its_cells_and_their_moments():
    """The structural check of the matrix: the named cells and their moments.

    This is a structural check of the parametrization, not a proof of
    completeness: it reads which cells exist and which moments they declare.
    It cannot prove that a cell really executed the fault it names; the report
    states the executed fault, the receipt and the assertions per cell.
    """
    module = sys.modules[__name__]
    assert len(RECOVERIES) == 7, RECOVERIES
    assert len(VARIANTS) == 1, VARIANTS
    for recovery, cells in list(RECOVERIES.items()) + list(VARIANTS.items()):
        if type(cells) is str:
            cells = (cells,)
        covered = set()
        for cell in cells:
            function = getattr(module, cell, None)
            assert function is not None, (recovery, cell)
            moments = []
            for mark in getattr(function, "pytestmark", []):
                if mark.name == "parametrize":
                    moments.extend(value for value in mark.args[1] if value in MOMENTS)
            covered.update(moments)
        if recovery in VARIANTS:
            # This variant has no delivery at all: the moment of the receiving
            # fault does not apply to it.
            assert covered == set(), (recovery, covered)
        else:
            # The cells of a recovery action cover the four moments between
            # them: the retry has one cell per kind of first answer.
            assert covered == set(MOMENTS), (recovery, covered)
    # Every boundary, control and fault path exists as a named cell.
    for control, cell in CONTROLS.items():
        assert getattr(module, cell, None) is not None, (control, cell)


# -- the negative controls with a valid starting point (I5) -------------------

def test_17_a_new_command_needs_an_exact_binding_and_a_current_basis(artifact, layout):
    """I5: the negative controls, one broken precondition at a time.

    The controls need a state that is valid in itself, so the positive route
    comes first: a new pending command under the current binding and the
    current basis really goes through and is accepted. Then exactly one
    precondition is broken: first the binding alone, then the basis alone.
    The refusal names that one, and nothing else is wrong with the command.
    """
    module = execution()
    # The positive route: the valid new pending command really goes through.
    first = build_command(module, layout, artifact)
    layout.owner.record_pending_command(first)
    with publication_lock(artifact.path) as lease:
        receipt = module.send_locked(first, lease=lease, owner=layout.owner,
                                    receiver=layout.gateway, deadline=SECTION_DEADLINE,
                                    clock=CLOCK, policy=POLICY)
    assert receipt.status == "accepted", receipt
    assert_identity(receipt, first, what="accepted receipt")
    mirror = read_mirror(artifact.root)
    assert mirror["pending_command"] is None, mirror
    assert_identity(mirror["last_receipt"], first, what="mirror receipt")
    # The two negative controls, each with one broken precondition only.
    assert_new_command_needs_binding_and_basis(artifact, layout)


# -- the integrity of the test fake (N1) ---------------------------------------

def test_16_the_attach_refuses_an_inconsistent_state_and_anchor_pair(artifact, layout):
    """N1: the attach refuses a state that is behind its independent anchor.

    The receiver moves on: the barrier moves the durable boundary of the state
    and of the independent anchor. Then only the state file is put back, so the
    pair is inconsistent: the anchor is ahead of the state. The attach must
    refuse that pair *before* any write, and both files must stay byte-identical
    to what they were before the attempt. The positive route stays: a
    consistent pair attaches without any doubt.
    """
    # The state of the receiver just before it moves on.
    snapshot = layout.gateway.snapshot_files()
    withdraw_epoch(layout)
    # Only the state goes back: the independent anchor stays ahead.
    layout.gateway.restore_files(snapshot)
    state_before = layout.gateway.state_path.read_bytes()
    anchor_before = layout.gateway.anchor_path.read_bytes()
    assert state_before != anchor_before, "the state and the anchor are the same file"
    with pytest.raises(ValueError) as error:
        FencedReceiver(artifact.root / "receiver", initial_state=None, utc_ms=NOW, attach=True)
    assert error.value.args[0] == "fake_attach_inconsistent", error.value
    # The refusal wrote nothing: both files are byte-identical.
    assert layout.gateway.state_path.read_bytes() == state_before
    assert layout.gateway.anchor_path.read_bytes() == anchor_before
    # The positive route: a consistent pair attaches and stays usable.
    clean = FencedReceiver(artifact.root / "receiver-clean", initial_state=receiver_state(),
                            utc_ms=NOW)
    attached = FencedReceiver(artifact.root / "receiver-clean", initial_state=None, utc_ms=NOW,
                              attach=True)
    assert attached.state.anchor_checkpoint == clean.state.anchor_checkpoint, attached.state
    assert attached.state.epoch_high_water == clean.state.epoch_high_water, attached.state
    assert attached.running is True, attached.running
