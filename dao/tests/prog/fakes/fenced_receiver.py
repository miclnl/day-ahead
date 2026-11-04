"""Manually driven sandbox environment for the pure production receiver rules.

No network, real clock, sleeps, HA or device. The independent anchor is never
part of a state-file backup. Construction requires an explicit provisioned state;
restart is recovery, not provisioning.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

from dao.prog.fastctrl import receiver
from dao.prog.fastctrl.actuator import FinancialActuationAdapter
from dao.prog.fastctrl.command import HelperResult


def _durable_write(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class FencedReceiver(FinancialActuationAdapter):
    """The only executable fenced gateway in F2c2; a test double, not an adapter.

    The nominal qualification is the whole point of this inheritance: it is
    the only way to carry financial command rights in this repository. The fake
    remains a test double; it is not a qualified production adapter.
    """

    def __init__(self, path, *, initial_state=None, utc_ms, clock_uncertainty_ms=0,
                 attach=False, effect_log_path=None):
        """Provision a receiver, or take over the durable files of one.

        With ``attach=True`` this instance takes over the durable files of an
        already provisioned receiver: it reads the state and the independent
        anchor back and never provisions again. That is the documented
        test-only facility for the second process of a caller crash: the
        caller that died owned the durable files, so the recovery needs the
        same receiver. It never clears the queue and never judges the state.

        With ``effect_log_path`` the fake keeps a durable effect log next to
        its state: every real write of the last writer is appended at the
        moment that write happens. That is opt-in test-only fault evidence:
        it lets the effect evidence of the fake survive the death of the
        process that owned the fake. It never invents an entry: only a real
        write of the dummy appends one, and an attached instance reads the
        real entries back.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.state_path = path / "receiver-state.json"
        self.anchor_path = path / "independent-anchor.json"
        self.effect_log_path = Path(effect_log_path) if effect_log_path is not None else None
        if attach:
            if initial_state is not None:
                raise ValueError("fake_attach_takes_no_initial_state")
        elif initial_state is None:
            raise ValueError("fake_initial_state_missing")
        elif self.state_path.exists() or self.anchor_path.exists():
            raise ValueError("fake_already_provisioned")
        self.state = initial_state
        self.utc_ms = utc_ms
        self.monotonic_ns = 0
        self.clock_uncertainty_ms = clock_uncertainty_ms
        self.clock_healthy = True
        self.receiver_max_lease_ms = 5000
        self.queues_drained = True
        self.safe_state_confirmed = True
        self.helper_ok = {"setpoint": True, "mode": True, "stop_inverter": True}
        self.effect_log = []
        self.queue = []
        self.work = []
        self.running = True
        self._lose_ack = False
        self._checkpoint_counter = 0
        if attach:
            self._attach_durable()
        else:
            self._persist()

    def _attach_durable(self):
        """Take over the durable files of an already provisioned receiver.

        The state and the independent anchor have to be one consistent pair.
        The existing pure validation of the receiver decides that: a state
        that is behind its anchor is a rollback, and an unusable state is an
        unknown state. Neither becomes an active receiver here: the attach
        refuses before any write, so both files stay exactly as they were. A
        refused attach never overwrites the independent anchor either.
        """
        try:
            anchor = json.loads(self.anchor_path.read_text())
            loaded = receiver.ReceiverState.from_dict(
                json.loads(self.state_path.read_text()))
            _state, reason = receiver.receiver_recovery_state(
                loaded, anchor_epoch_high_water=anchor["epoch_high_water"],
                anchor_sequence_high_water=anchor["sequence_high_water"],
                anchor_checkpoint=anchor["anchor_checkpoint"])
        except (OSError, KeyError, TypeError, ValueError):
            raise ValueError("fake_attach_inconsistent") from None
        if reason != "recovered":
            # A rollback or an unknown state never becomes an active receiver.
            raise ValueError("fake_attach_inconsistent")
        self.state = loaded
        checkpoint = self.state.anchor_checkpoint
        if checkpoint.startswith("fake-checkpoint-"):
            # The checkpoint counter continues where the other process left it.
            self._checkpoint_counter = max(self._checkpoint_counter,
                                           int(checkpoint.rsplit("-", 1)[1]))
        self.effect_log = self._load_effect_log()
        return anchor

    def _load_effect_log(self):
        """The real effect writes of the durable effect log, in their order."""
        if self.effect_log_path is None or not self.effect_log_path.exists():
            return []
        entries = []
        for line in self.effect_log_path.read_text().splitlines():
            if line.strip():
                entries.append(tuple(json.loads(line)))
        return entries

    def _append_effect(self, entry):
        """Append one real effect write to the durable effect log."""
        if self.effect_log_path is None:
            return
        with self.effect_log_path.open("a") as stream:
            json.dump(list(entry), stream, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    @property
    def receiver_now_upper_ms(self):
        uncertainty = self.clock_uncertainty_ms
        if type(uncertainty) is int and uncertainty >= 0:
            return self.utc_ms + uncertainty
        # Only an observation timestamp for the rejection, NOT an upper bound.
        # The raw invalid/unknown uncertainty goes unchanged to every rule;
        # the clock fence must reject it before any expiry comparison or write.
        return self.utc_ms

    def _require_running(self):
        if not self.running:
            raise ValueError("fake_receiver_stopped")

    def _persist(self):
        self._checkpoint_counter += 1
        self.state = replace(self.state, anchor_checkpoint=f"fake-checkpoint-{self._checkpoint_counter}")
        anchor = dict(epoch_high_water=self.state.epoch_high_water,
                      sequence_high_water=self.state.sequence_high_water,
                      anchor_checkpoint=self.state.anchor_checkpoint)
        # Anchor first: interruption between the files is conservatively blocked.
        _durable_write(self.anchor_path, anchor)
        _durable_write(self.state_path, self.state.to_dict())

    def _apply(self, decision):
        if decision.state != self.state:
            self.state = decision.state
            self._persist()

    def enqueue(self, command):
        self._require_running()
        self.queue.append(command)

    def deliver(self):
        self._require_running()
        command = self.queue.pop(0)
        return self._receive(command).receipt

    def _receive(self, command):
        decision = receiver.classify_send(
            self.state, command, receiver_now_upper_ms=self.receiver_now_upper_ms,
            clock_healthy=self.clock_healthy, clock_uncertainty_ms=self.clock_uncertainty_ms,
        )
        self._apply(decision)
        if decision.status == "reserved":
            self.work.append(command)
        return decision

    def step_helper(self, part, *, command=None):
        self._require_running()
        if command is None:
            command = self.work[0]
        now = self.receiver_now_upper_ms
        allowed, _ = receiver.fence_effect(self.state, command, receiver_now_upper_ms=now,
                                            clock_healthy=self.clock_healthy,
                                            clock_uncertainty_ms=self.clock_uncertainty_ms)
        if not allowed:
            decision = receiver.classify_effect(self.state, command, receiver_now_upper_ms=now, outcome="none",
                                                 clock_healthy=self.clock_healthy,
                                                 clock_uncertainty_ms=self.clock_uncertainty_ms)
        else:
            ok = self.helper_ok[part]
            result = HelperResult(part, f"dummy.{command.battery_name}.{part}", ok,
                                  None if ok else "write_failed", 1)
            # The dummy is the last writer: each part has its own immediate fence.
            # Record order is checked by production code before touching the dummy.
            decision = receiver.classify_helper(self.state, command, receiver_now_upper_ms=now,
                                                clock_healthy=self.clock_healthy,
                                                clock_uncertainty_ms=self.clock_uncertainty_ms,
                                                result=result, acknowledged=not self._lose_ack)
            if ok:
                entry = (command.command_id, part, command.battery_name, command.setpoint_w)
                self.effect_log.append(entry)
                # The durable effect log keeps the real write of the last
                # writer: it survives the death of the process that owns it.
                self._append_effect(entry)
        self._apply(decision)
        if decision.record.phase != "reserved":
            self.work = [item for item in self.work if item.command_id != command.command_id]
            self._lose_ack = False
        return decision.receipt

    def lose_ack(self):
        self._lose_ack = True

    def advance_clock(self, ms):
        if type(ms) is not int:
            raise ValueError("fake_clock_step_invalid")
        self.utc_ms += ms
        self.monotonic_ns += max(0, ms) * 1_000_000
        if ms < 0:
            self.clock_healthy = False

    def crash(self, persist: bool):
        if persist:
            self.state = receiver.receiver_crash_state(self.state, receiver_now_upper_ms=self.receiver_now_upper_ms)
            self._persist()
        # persist=False does not perform a final save. It cannot undo writes
        # already made durable at reservation/helper/barrier boundaries.
        self.queue.clear()
        self.work.clear()
        self.running = False

    def restart(self):
        anchor = json.loads(self.anchor_path.read_text())
        try:
            loaded = receiver.ReceiverState.from_dict(json.loads(self.state_path.read_text()))
        except (OSError, ValueError):
            loaded = None
        self.state, reason = receiver.receiver_recovery_state(
            loaded, anchor_epoch_high_water=anchor["epoch_high_water"],
            anchor_sequence_high_water=anchor["sequence_high_water"],
            anchor_checkpoint=anchor["anchor_checkpoint"],
        )
        self.queue.clear()
        self.work.clear()
        self.running = True
        return reason

    def snapshot_files(self):
        return self.state_path.read_bytes()

    def restore_files(self, snapshot):
        # Deliberately NOT the independent-anchor.json file.
        self.state_path.write_bytes(snapshot)

    def inspect(self, *, deadline):
        self._require_running()
        return receiver.receiver_snapshot(self.state)

    def revoke_and_stop(self, request, *, deadline):
        self._require_running()
        decision = receiver.classify_revoke(
            self.state, request, receiver_now_upper_ms=self.receiver_now_upper_ms,
            queues_drained=self.queues_drained, safe_state_confirmed=self.safe_state_confirmed,
        )
        self._apply(decision)
        return decision.receipt

    def bind_owner(self, *, installation_id, barrier_id, expected_epoch,
                   owner_session_id, requested_not_after_ms, deadline):
        self._require_running()
        decision = receiver.classify_bind(
            self.state, installation_id=installation_id, barrier_id=barrier_id, expected_epoch=expected_epoch,
            owner_session_id=owner_session_id, requested_not_after_ms=requested_not_after_ms,
            receiver_max_lease_ms=self.receiver_max_lease_ms, receiver_now_upper_ms=self.receiver_now_upper_ms,
        )
        self._apply(decision)
        if decision.binding is None:
            raise ValueError(decision.reason)
        return decision.binding

    def send(self, command, *, deadline):
        self._require_running()
        decision = self._receive(command)
        if decision.status != "reserved":
            return decision.receipt
        for part in ("setpoint", "mode", "stop_inverter"):
            result = self.step_helper(part, command=command)
            # Production decides when this work item is terminal. Other delayed
            # callbacks in the environment must not cause a partial to be retried.
            if command not in self.work:
                break
        return result

    def query(self, *, installation_id, owner_epoch, sequence, command_id, payload_digest, deadline):
        self._require_running()
        return receiver.classify_query(self.state, installation_id=installation_id, owner_epoch=owner_epoch,
                                       sequence=sequence, command_id=command_id, payload_digest=payload_digest)
