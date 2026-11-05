"""The mechanical publication boundary: final validation, bounded send, owner section.

Pure module: no database, no home assistant, no solver and no network. It
composes the existing publication lease, the durable owner mirror and the
fenced receiver contract. It is an internal boundary and never an alternative
public execute route: every v2 plan still ends in ``actuation_gates_unproven``
at the public runner entrypoint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .actuator import FENCED_ADAPTER, classify_actuation_adapter
from .command import (
    ActuationReceipt,
    BatteryCommand,
    ClockSample,
    Deadline,
    _integer,
    _reason,
    _string,
    plan_digest,
)
from .ownership import OwnerCoordinator, RecoveryRequired
from .plan import FastPlan
from .publication import GatePolicy, PublicationError, PublicationLease, read_publication
from .receiver import FencedDeviceGateway, payload_digest, receipt_matches_command

__all__ = [
    "CommandRequest",
    "ExecutionRefused",
    "owner_section",
    "plan_digest",
    "send_locked",
    "validate_current_locked",
]

#: Logical budget for one installation section (one second).
SECTION_BUDGET_NS = 1_000_000_000
#: Transport, retry and query share at most this part of the section budget.
TRANSPORT_BUDGET_NS = 750_000_000


class ExecutionRefused(Exception):
    """Explicit refusal with a machine-readable reason; never a silent permission."""

    def __init__(self, reason: str):
        _reason(reason, "execution_reason_invalid")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CommandRequest:
    """One proposed battery command; no plan, price or settlement state."""

    battery_name: str
    setpoint_w: int
    mode: str
    stop_at_ms: int | None

    def __post_init__(self):
        _string(self.battery_name, "request_battery_name_invalid")
        _integer(self.setpoint_w, "request_setpoint_w_invalid", None)
        _string(self.mode, "request_mode_invalid")
        if self.stop_at_ms is not None:
            _integer(self.stop_at_ms, "request_stop_at_ms_invalid", None)


def validate_current_locked(*, lease: PublicationLease, expected_digest: str,
                            clock: ClockSample, policy: GatePolicy) -> FastPlan:
    """Prove under the held lease that the current artefact is still the one.

    This is the final artefact-, time-, basis- and digest check. It never
    replaces `following_verdict`: the public route keeps refusing every v2
    plan with `actuation_gates_unproven`.
    """
    if type(lease) is not PublicationLease:
        raise ExecutionRefused("publication_lease_invalid")
    if type(clock) is not ClockSample:
        raise ExecutionRefused("clock_invalid")
    if type(policy) is not GatePolicy:
        raise ExecutionRefused("policy_invalid")
    if (type(expected_digest) is not str or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)):
        raise ExecutionRefused("expected_digest_invalid")
    try:
        lease._check()
    except PublicationError as exc:
        raise ExecutionRefused("publication_lease_lost") from exc
    read = read_publication(lease.path)
    if read.status == "blocked":
        raise ExecutionRefused("publication_blocked")
    if read.status == "missing":
        raise ExecutionRefused("publication_missing")
    if read.status == "invalid":
        raise ExecutionRefused("publication_invalid")
    plan = read.plan
    if plan is None:
        raise ExecutionRefused("publication_invalid")
    if plan.format_version != 2:
        raise ExecutionRefused("financial_plan_required")
    context = plan.financial_context
    if context is None:
        raise ExecutionRefused("missing_financial_context")
    if not clock.healthy:
        raise ExecutionRefused("clock_unhealthy")
    # The clock is milliseconds; the plan timeline is seconds. Every comparison
    # converts the seconds once and stays in the millisecond domain.
    if clock.utc_ms < plan.created_ts * 1000:
        raise ExecutionRefused("clock_before_creation")
    if clock.utc_ms >= context.not_after_ts * 1000:
        raise ExecutionRefused("plan_lease_expired")
    if clock.utc_ms < context.basis.captured_ts * 1000:
        raise ExecutionRefused("clock_before_capture")
    if not context.basis.meter_id.strip():
        raise ExecutionRefused("meter_identity_missing")
    if context.basis.input_revision <= 0:
        raise ExecutionRefused("input_revision_invalid")
    if policy.required_from_ts is not None and clock.utc_ms < policy.required_from_ts * 1000:
        raise ExecutionRefused("gate_not_open")
    if plan_digest(plan) != expected_digest:
        raise ExecutionRefused("plan_digest_mismatch")
    return plan


def _validate_receiver(*, owner: OwnerCoordinator, receiver: FencedDeviceGateway) -> None:
    """Require the owner's nominal adapter object before any mechanical side effect."""
    if classify_actuation_adapter(receiver) != FENCED_ADAPTER:
        raise ExecutionRefused("adapter_unqualified")
    if receiver is not owner.receiver:
        raise ExecutionRefused("receiver_binding_mismatch")


def _require_section(owner: OwnerCoordinator, lease: PublicationLease) -> None:
    try:
        owner.require_final_section(lease, operation="send")
    except RecoveryRequired as exc:
        raise ExecutionRefused(exc.reason) from exc


def _now(owner: OwnerCoordinator) -> int:
    try:
        return owner._clock()
    except RecoveryRequired as exc:
        raise ExecutionRefused(exc.reason) from exc


def _section_budget(owner: OwnerCoordinator, deadline: Deadline) -> Deadline:
    return Deadline(min(deadline.monotonic_end_ns, owner.section_deadline.monotonic_end_ns),
                    min(deadline.execute_before_ms, owner.section_deadline.execute_before_ms))


def _transport_budget(owner: OwnerCoordinator, section: Deadline) -> Deadline:
    return Deadline(min(section.monotonic_end_ns, owner.transport_deadline.monotonic_end_ns),
                    min(section.execute_before_ms, owner.transport_deadline.execute_before_ms))


def _validate_installation(plan: FastPlan, *, owner: OwnerCoordinator,
                           command: BatteryCommand | None = None) -> None:
    """Bind the freshly validated meter to the startup-proven owner identity."""
    binding = owner.binding
    if binding is None:
        raise ExecutionRefused("owner_binding_missing")
    state = owner.state
    if state is None:
        raise ExecutionRefused("owner_state_missing")
    installation = plan.financial_context.basis.meter_id
    if not (installation == owner.store.installation_id == state.installation_id
            == binding.installation_id):
        raise ExecutionRefused("installation_mismatch")
    if command is not None and command.installation_id != installation:
        raise ExecutionRefused("installation_mismatch")


def _is_reserved(observation: ActuationReceipt) -> bool:
    """The receiver still holds a live reservation of this very command.

    The receiver reports a reserved command as ``unconfirmed`` with the reason
    ``pending_effect``: the command is durably reserved and its helpers may
    still write. Such a reservation is not a terminal outcome.
    """
    return observation.status == "unconfirmed" and observation.reason == "pending_effect"


def _conservative(observations: list[ActuationReceipt]) -> ActuationReceipt:
    """The most conservative observation; the result always comes from the receiver.

    `accepted` needs a receiver proof of acceptance. `rejected` needs proof of
    no effect by this very command: one single observation with an effect is
    enough to make the outcome uncertain. A mixed set is reported with the
    most conservative non-rejection, never with a rejection.

    A live reservation is never terminal either. A refusal of a duplicate says
    nothing about the reservation that the first delivery made, so a rejection
    may not end the command while the receiver still holds that reservation.
    The uncertainty stays until a terminal proof or an exact safe barrier.
    """
    for observation in observations:
        if observation.status == "accepted":
            return observation
    if (all(observation.effect == "none" for observation in observations)
            and not any(_is_reserved(observation) for observation in observations)):
        for observation in observations:
            if observation.status == "rejected":
                return observation
    for observation in observations:
        if observation.status == "unconfirmed":
            return observation
    return observations[-1]


def _bounded_deadline(deadline: Deadline, *, clock: ClockSample, wire_end_ms: int) -> Deadline:
    """Translate a wire bound once, conservatively; an absolute budget only shrinks."""
    if not clock.healthy:
        raise ExecutionRefused("clock_unhealthy")
    execute_before = min(deadline.execute_before_ms, wire_end_ms)
    end = min(deadline.monotonic_end_ns,
              clock.boottime_ns + (execute_before - clock.utc_ms - clock.uncertainty_ms) * 1_000_000)
    if end <= clock.boottime_ns:
        raise ExecutionRefused("budget_exhausted")
    return Deadline(end, execute_before)


def _command_end_ms(plan: FastPlan, *, battery_name: str, at_ms: int,
                    owner_end_ms: int) -> int:
    """The next profile edge ends this battery's hold, even at unchanged power."""
    interval = plan.interval_at(at_ms / 1000)
    index = plan.spec_index(battery_name)
    if interval is None or index is None:
        raise ExecutionRefused("command_window_invalid")
    step = interval.battery(index)
    ends = [plan.financial_context.not_after_ts * 1000, owner_end_ms, interval.end_ts * 1000]
    ends.extend(edge * 1000 for edge in (step.active_from_ts, step.stop_ts)
                if edge is not None and edge * 1000 > at_ms)
    return min(ends)


def _check_result_deadline(*, owner: OwnerCoordinator, command: BatteryCommand,
                           deadline: Deadline) -> None:
    """A result write may clear pending, but an unproven post-write clock may not."""
    try:
        if deadline.remaining_ns(_now(owner)) == 0:
            raise ExecutionRefused("deadline_exceeded")
    except ExecutionRefused:
        # Keep the receipt as history and the command as recovery evidence.
        # A failed restoration remains a hard persistence error, not a refusal.
        owner.record_pending_command(command)
        raise


def send_locked(command: BatteryCommand, *, lease: PublicationLease,
                owner: OwnerCoordinator, receiver: FencedDeviceGateway,
                deadline: Deadline, clock: ClockSample,
                policy: GatePolicy) -> ActuationReceipt:
    """One bounded send under the held publication lease.

    The final validation happens here, never in an earlier caller. At most one
    retry with exactly the same bytes and at most one exact query follow, both
    inside the remaining budget. A retry never creates new budget.
    """
    if type(command) is not BatteryCommand:
        raise ExecutionRefused("command_invalid")
    if type(deadline) is not Deadline:
        raise ExecutionRefused("deadline_invalid")
    _validate_receiver(owner=owner, receiver=receiver)
    _require_section(owner, lease)
    deadline = _section_budget(owner, deadline)
    # An earlier caller check never suffices; the final check runs here.
    plan = validate_current_locked(lease=lease, expected_digest=command.plan_digest,
                                   clock=clock, policy=policy)
    _validate_installation(plan, owner=owner, command=command)
    if not owner.store.holds(owner.writer):
        raise ExecutionRefused("owner_lease_missing")
    try:
        lease._check()
    except PublicationError as exc:
        raise ExecutionRefused("publication_lease_lost") from exc
    binding = owner.binding
    if binding is None:
        raise ExecutionRefused("owner_binding_missing")
    if (binding.installation_id != command.installation_id
            or binding.owner_epoch != command.owner_epoch
            or binding.owner_session_id != command.owner_session_id):
        raise ExecutionRefused("owner_binding_mismatch")
    state = owner.state
    pending = state.pending_command if state is not None else None
    if pending is None:
        raise ExecutionRefused("pending_command_missing")
    if type(pending) is not BatteryCommand or pending.to_dict() != command.to_dict():
        raise ExecutionRefused("pending_command_mismatch")
    if command.issued_at_ms > clock.utc_ms:
        # Future dating cannot select a later profile segment for execution now.
        raise ExecutionRefused("command_window_invalid")
    command_end = _command_end_ms(plan, battery_name=command.battery_name,
                                  at_ms=command.issued_at_ms,
                                  owner_end_ms=binding.owner_not_after_ms)
    if command.hold_until_ms > command_end:
        # Validate the immutable command against the fresh plan and binding;
        # shortening transport time cannot repair an overwide command payload.
        raise ExecutionRefused("command_window_invalid")
    deadline = _bounded_deadline(
        Deadline(min(deadline.monotonic_end_ns, clock.boottime_ns + SECTION_BUDGET_NS),
                 deadline.execute_before_ms),
        clock=clock, wire_end_ms=min(command.execute_before_ms, command_end))
    transport = _transport_budget(owner, deadline)
    def now():
        return _now(owner)
    if transport.remaining_ns(now()) == 0:
        raise ExecutionRefused("budget_exhausted")
    owner._section_used = True
    try:
        receipt = receiver.send(command, deadline=transport)
    except Exception as exc:
        # Only transport failures are masked; identity refusals stay explicit.
        raise ExecutionRefused("receiver_unavailable") from exc
    if not receipt_matches_command(receipt, command):
        raise ExecutionRefused("receipt_identity_mismatch")
    observations = [receipt]
    if receipt.status != "accepted" and transport.remaining_ns(now()) > 0:
        # At most one retry, with exactly the same bytes, id, sequence and deadline.
        try:
            retried = receiver.send(command, deadline=transport)
        except Exception as exc:
            raise ExecutionRefused("receiver_unavailable") from exc
        if not receipt_matches_command(retried, command):
            raise ExecutionRefused("receipt_identity_mismatch")
        observations.append(retried)
        if retried.status != "accepted" and transport.remaining_ns(now()) > 0:
            try:
                queried = receiver.query(installation_id=command.installation_id,
                                         owner_epoch=command.owner_epoch,
                                         sequence=command.sequence,
                                          command_id=command.command_id,
                                          payload_digest=payload_digest(command),
                                         deadline=transport)
            except Exception as exc:
                raise ExecutionRefused("receiver_unavailable") from exc
            if queried is not None:
                if not receipt_matches_command(queried, command):
                    raise ExecutionRefused("receipt_identity_mismatch")
                observations.append(queried)
    outcome = _conservative(observations)
    if outcome.status == "accepted" and transport.remaining_ns(now()) == 0:
        # A late acknowledgement is history only: it grants no right, clears no
        # pending command and never becomes an accepted command update. The
        # proof itself is kept durably as history, without a second budget.
        owner.record_receipt(outcome, terminal=False)
        raise ExecutionRefused("deadline_exceeded")
    owner.record_receipt(outcome)
    _check_result_deadline(owner=owner, command=command, deadline=deadline)
    return outcome


def _build_command(*, owner: OwnerCoordinator, request: CommandRequest, plan: FastPlan,
                   sequence: int, clock: ClockSample) -> BatteryCommand:
    binding = owner.binding
    if binding is None:
        raise ExecutionRefused("owner_binding_missing")
    context = plan.financial_context
    execute_before = _command_end_ms(plan, battery_name=request.battery_name, at_ms=clock.utc_ms,
                                     owner_end_ms=binding.owner_not_after_ms)
    if not clock.utc_ms < execute_before:
        raise ExecutionRefused("command_window_invalid")
    installation = context.basis.meter_id
    # Unique, without a timestamp and without an array index.
    fingerprint = hashlib.sha256(installation.encode("utf-8")).hexdigest()[:32]
    return BatteryCommand(
        installation_id=installation,
        battery_name=request.battery_name,
        owner_session_id=binding.owner_session_id,
        owner_epoch=binding.owner_epoch,
        sequence=sequence,
        command_id=f"{fingerprint}/{binding.owner_epoch}/{sequence}",
        plan_digest=plan_digest(plan),
        issued_at_ms=clock.utc_ms,
        execute_before_ms=execute_before,
        hold_until_ms=execute_before,
        setpoint_w=request.setpoint_w,
        mode=request.mode,
        stop_at_ms=request.stop_at_ms,
    )


def owner_section(*, lease: PublicationLease, owner: OwnerCoordinator,
                  receiver: FencedDeviceGateway, request: CommandRequest,
                  policy: GatePolicy, clock: ClockSample,
                  deadline: Deadline) -> ActuationReceipt | None:
    """Read, validate, reserve, build, record, send and record under one lease.

    The publication lease is taken before the final artefact check and is only
    released after the result is recorded. There is no database transaction and
    no nested acquisition anywhere in this section.
    """
    if type(request) is not CommandRequest:
        raise ExecutionRefused("request_invalid")
    if type(owner) is not OwnerCoordinator:
        raise ExecutionRefused("owner_invalid")
    if type(deadline) is not Deadline:
        raise ExecutionRefused("deadline_invalid")
    _validate_receiver(owner=owner, receiver=receiver)
    _require_section(owner, lease)
    deadline = _section_budget(owner, deadline)
    def now():
        return _now(owner)
    # One absolute deadline for the whole installation.
    end = min(deadline.monotonic_end_ns, clock.boottime_ns + SECTION_BUDGET_NS)
    binding = owner.binding
    section = Deadline(end, deadline.execute_before_ms)
    try:
        lease._check()
    except PublicationError as exc:
        raise ExecutionRefused("publication_lease_lost") from exc
    read = read_publication(lease.path)
    if read.status != "ready" or read.plan is None or read.plan.format_version != 2:
        # The plan is not ready-v2: no send at all. The owner consumes the
        # invalidation by withdrawing its epoch inside this very section, with
        # the remaining section budget and never with a second window.
        try:
            section = _bounded_deadline(section, clock=clock, wire_end_ms=(
                binding.owner_not_after_ms if binding is not None else section.execute_before_ms))
        except ExecutionRefused as exc:
            if exc.reason != "budget_exhausted":
                raise
            owner.report_failure("publication_invalidation_budget_exhausted", durable=True)
            return None
        revoke = _transport_budget(owner, section)
        if revoke.remaining_ns(now()) == 0:
            # Not enough budget for a safe withdrawal: the epoch is not
            # withdrawn, the recovery duty stays and stays visible. The duty
            # is written durably through the existing mirror route.
            owner.report_failure("publication_invalidation_budget_exhausted", durable=True)
            return None
        try:
            owner._stop_in_send_section(lease, reason="publication_invalidated", deadline=revoke)
        except RecoveryRequired as exc:
            raise ExecutionRefused(exc.reason) from exc
        return None
    plan = read.plan
    if section.remaining_ns(now()) == 0:
        raise ExecutionRefused("budget_exhausted")
    plan = validate_current_locked(lease=lease, expected_digest=plan_digest(plan), clock=clock,
                                   policy=policy)
    _validate_installation(plan, owner=owner)
    section = _bounded_deadline(section, clock=clock, wire_end_ms=_command_end_ms(
        plan, battery_name=request.battery_name, at_ms=clock.utc_ms,
        owner_end_ms=owner.binding.owner_not_after_ms))
    transport = _transport_budget(owner, section)
    if section.remaining_ns(now()) == 0:
        raise ExecutionRefused("budget_exhausted")
    # The sequence is reserved durably before the send; gaps are allowed.
    sequence = owner.reserve_sequence(writer=owner.writer)
    command = _build_command(owner=owner, request=request, plan=plan, sequence=sequence,
                             clock=clock)
    if transport.remaining_ns(now()) == 0:
        raise ExecutionRefused("budget_exhausted")
    # The pending command is durable before the send: a crash leaves evidence.
    owner.record_pending_command(command)
    # send_locked records the result durably in the same mirror.
    receipt = send_locked(command, lease=lease, owner=owner, receiver=receiver,
                          deadline=section, clock=clock, policy=policy)
    # The section checks the deadline after the last durable result write. An
    # overrun is never reported as an accepted command update: the pending
    # command is restored durably and the history stays visible.
    _check_result_deadline(owner=owner, command=command, deadline=section)
    return receipt
