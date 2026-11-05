"""The process side of the fast control layer.

Reads Home Assistant, feeds the measurements to :mod:`policy`, writes the
resulting setpoint back and persists the controller state so a restart does not
lose the energy budgets.

Everything Home Assistant specific is confined to :class:`HomeAssistantGateway`,
which is injected. The runner therefore needs no live Home Assistant instance in
the tests.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from inspect import getattr_static
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from dao.prog import task_execution
from dao.prog.config.models.fastcontrol import (
    FastBatteryLink,
    FastControlConfig,
    PowerSensor,
)
from dao.prog.config.models.pricing import PricingConfig

from . import execution as execution_boundary
from .actuator import (
    FENCED_ADAPTER,
    LEGACY_ONLY_ADAPTER,
    ActuationVerdict,
    CommandReceipt,
    classify_actuation_adapter,
    execute_battery_command,
)
from .command import (
    ActuationReceipt,
    BarrierReceipt,
    BatteryCommand,
    ClockSample,
    Deadline,
    OwnerBinding,
)
from .ownership import MirrorWriteFailed, OwnerCoordinator, RecoveryRequired
from .plan import FAST_PLAN_FILE, NO_STOP_SENTINEL, FastPlan, load_plan
from .policy import (
    POWER_EPS,
    BatteryDecision,
    BatteryMeasurement,
    ControllerState,
    Decision,
    FastControlPolicy,
    Measurement,
    PolicyLimits,
    apply_accepted_command,
    apply_execution_outcome,
    current_execution_command,
)
from .publication import (
    OWNER_CYCLE_NS,
    FallbackReceipt,
    FallbackRequest,
    FallbackSink,
    GatePolicy,
    GateVerdict,
    PublicationLockBusy,
    canonical_plan_path,
    following_verdict,
    owner_admission_at,
    publication_lock,
    read_publication,
)
from .publication import (
    gate_policy as configured_gate_policy,
)

FAST_STATE_FILE = "../data/fast_state.json"

#: How often the controller state is written to disk, in seconds.
#:
#: The file exists only to survive a restart, and the budgets it holds reset at
#: every plan interval anyway, so losing a few minutes of accounting after a
#: crash is harmless. Writing it every tick would mean thousands of small flash
#: writes per day, which is the wrong thing to do on the eMMC of a Home
#: Assistant Yellow or the SD card of a Green or a Raspberry Pi. The events that
#: really must not be lost -- an override starting or ending, a new plan, a new
#: day -- force an immediate write regardless of this interval.
STATE_SAVE_INTERVAL = 300.0

#: Fraction of a plan interval that must have been measured before the
#: realised energy is stored. A partly covered interval would understate the
#: total and pollute the forecast error statistics.
MEASUREMENT_MIN_COVERAGE = 0.8

#: States that mean "no usable value".
INVALID_STATES = frozenset({"unknown", "unavailable", "none", "", "null"})

#: Sentinel the optimizer writes when the inverter must not stop.
# Imported from .plan to keep a single source of truth.

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ACTIVE = "active"

EVENTS_KIND_MODE = "mode_change"
EVENTS_KIND_OVERRIDE_START = "override_start"
EVENTS_KIND_OVERRIDE_END = "override_end"
EVENTS_KIND_SETPOINT = "setpoint_change"

EVENTS_LIMIT = 200

#: How often to repeat the warning that the plan no longer covers the
#: current moment. The condition is persistent and only the optimiser can
#: fix it, so warning once and then falling silent for days -- which is what
#: it used to do -- leaves the layer parked at 0 W with nothing in the log
#: to say why.
EXPIRED_WARN_EVERY_S = 1800


def _parse_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in INVALID_STATES:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value == value else None


def _boot_id() -> str:
    """The boot identity of this machine; never empty, never a random value."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as handle:
            value = handle.read().strip()
    except OSError:
        value = ""
    return value or "unknown-boot"


class HomeAssistantGateway:
    """Batched reads and guarded writes against the Home Assistant REST API.

    A single templated render fetches every entity the control loop needs, which
    keeps one tick at one HTTP round trip regardless of how many sensors are
    configured. When the template endpoint is unavailable the gateway falls back
    to individual state reads and remembers that for the rest of the run.
    """

    #: The generic gateway is explicitly legacy-only. This is a label, never a
    #: release: it can never become ``FENCED_ADAPTER`` and the class does not
    #: inherit :class:`FinancialActuationAdapter`.
    actuation_adapter_class = LEGACY_ONLY_ADAPTER

    def __init__(self, hass, use_template: bool = True):
        self.hass = hass
        self.use_template = use_template
        self._write_failures: dict[str, int] = {}

    # -- reading ---------------------------------------------------------

    def read(self, entity_ids: Iterable[str]) -> dict[str, tuple[Optional[str], float]]:
        """Return ``{entity_id: (state, last_updated_epoch)}`` for every id."""
        ids = list(dict.fromkeys(e for e in entity_ids if e))
        if not ids:
            return {}
        if self.use_template:
            try:
                return self._read_templated(ids)
            except Exception as exception:  # noqa: BLE001 - degrade, never abort
                logging.warning(
                    f"Fast control: template-API niet bruikbaar ({exception}), "
                    f"overgeschakeld op losse state-verzoeken"
                )
                self.use_template = False
        return self._read_individually(ids)

    def _read_templated(
        self, ids: list[str]
    ) -> dict[str, tuple[Optional[str], float]]:
        entries = ",\n".join(
            "  [{{ states(%s) | tojson }}, {{ (states[%s].last_updated | as_timestamp) "
            "if states[%s] is not none else 0 }}]" % (repr(e), repr(e), repr(e))
            for e in ids
        )
        template = "[\n" + entries + "\n]"
        rendered = self.hass.render_template(template)
        payload = rendered if isinstance(rendered, list) else json.loads(rendered)
        result: dict[str, tuple[Optional[str], float]] = {}
        for entity_id, row in zip(ids, payload):
            state = row[0] if row and row[0] is not None else None
            updated = float(row[1] or 0.0) if len(row) > 1 else 0.0
            result[entity_id] = (state, updated)
        return result

    def _read_individually(
        self, ids: list[str]
    ) -> dict[str, tuple[Optional[str], float]]:
        result: dict[str, tuple[Optional[str], float]] = {}
        for entity_id in ids:
            try:
                state = self.hass.get_state(entity_id)
                updated = getattr(state, "last_updated", None) or getattr(
                    state, "last_changed", None
                )
                result[entity_id] = (
                    state.state,
                    updated.timestamp() if updated else 0.0,
                )
            except Exception as exception:  # noqa: BLE001
                logging.debug(f"Fast control: {entity_id} niet leesbaar: {exception}")
                result[entity_id] = (None, 0.0)
        return result

    # -- writing ---------------------------------------------------------

    def _guarded(self, key: str, action: Callable[[], None], description: str) -> bool:
        try:
            action()
        except Exception as exception:  # noqa: BLE001
            count = self._write_failures.get(key, 0) + 1
            self._write_failures[key] = count
            # Log the first few in full, then only every 20th, so a persistently
            # broken entity cannot flood the log at one line per tick.
            if count <= 3 or count % 20 == 0:
                logging.warning(
                    f"Fast control: {description} mislukt ({count}x): {exception}"
                )
            return False
        self._write_failures.pop(key, None)
        return True

    def write_number(self, entity_id: str, value: float) -> bool:
        return self._guarded(
            f"num:{entity_id}",
            lambda: self.hass.set_value(entity_id, value),
            f"schrijven van {value} naar {entity_id}",
        )

    def select_option(self, entity_id: str, option: str) -> bool:
        return self._guarded(
            f"sel:{entity_id}",
            lambda: self.hass.select_option(entity_id, option),
            f"zetten van {entity_id} op {option}",
        )

    def set_datetime(self, entity_id: str, value: str) -> bool:
        return self._guarded(
            f"dt:{entity_id}",
            lambda: self.hass.call_service(
                "set_datetime", entity_id=entity_id, datetime=value
            ),
            f"zetten van {entity_id} op {value}",
        )

    def publish_state(
        self, entity_id: str, state: str, attributes: Optional[dict] = None
    ) -> bool:
        return self._guarded(
            f"st:{entity_id}",
            lambda: self.hass.set_state(entity_id, state, attributes),
            f"publiceren van status naar {entity_id}",
        )

    def switch(self, entity_id: str, on: bool) -> bool:
        return self._guarded(
            f"sw:{entity_id}",
            lambda: (self.hass.turn_on if on else self.hass.turn_off)(entity_id),
            f"schakelen van {entity_id} naar {'on' if on else 'off'}",
        )


def load_state(path: str = FAST_STATE_FILE) -> ControllerState:
    """Read the persisted controller state, or return a fresh one."""
    try:
        with open(path, "r") as handle:
            return ControllerState.from_dict(json.load(handle))
    except FileNotFoundError:
        return ControllerState()
    except (json.JSONDecodeError, ValueError, TypeError, OSError) as exception:
        logging.warning(
            f"Fast control: status {path} onleesbaar ({exception}), opnieuw begonnen"
        )
        return ControllerState()


def save_state(state: ControllerState, path: str = FAST_STATE_FILE) -> None:
    """Persist the controller state atomically."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", dir=directory, prefix=".fast_state_", suffix=".tmp", delete=False
        )
        try:
            with handle:
                json.dump(state.to_dict(), handle)
            os.replace(handle.name, path)
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise
    except OSError as exception:
        logging.warning(f"Fast control: status kon niet worden opgeslagen: {exception}")


class FastControlRunner:
    """Owns one control loop: read, decide, actuate, persist."""

    def __init__(
        self,
        hass,
        config: Optional[FastControlConfig] = None,
        plan_path: str = FAST_PLAN_FILE,
        state_path: str = FAST_STATE_FILE,
        gateway: Optional[HomeAssistantGateway] = None,
        *,
        gate_policy: GatePolicy | None = None,
        fallback: FallbackSink | None = None,
        owner: OwnerCoordinator | None = None,
    ):
        self.hass = hass
        if gate_policy is None:
            if hass.config is None:
                raise ValueError("Fast control: configuratie is niet geladen")
            missing = object()
            pricing = None if getattr_static(hass.config, "prices", missing) is missing else hass.config.prices
            if pricing is not None:
                # Afwezig is legacy; onleesbaar/ongeldig mag dat nooit worden.
                pricing = PricingConfig.model_validate(pricing.model_dump())
            gate_policy = configured_gate_policy(pricing)
        if type(gate_policy) is not GatePolicy:
            raise TypeError("Expected explicit GatePolicy")
        self.gate_policy = gate_policy
        self.fallback = fallback
        self.owner = owner
        self._owner_writer = None
        self._owner_start_pending = False
        self._owner_release_pending = False
        self._owner_session_id = None
        self._financial_cadence = owner is not None
        self._cadence_last_ns = None
        self._cadence_boot = _boot_id()
        self._tick_cycle = None
        self._shutting_down = False
        self._stop_event = None
        self.last_gate: GateVerdict | None = None
        self.last_fallback: FallbackReceipt | None = None
        self.last_receipts: list[CommandReceipt] = []
        #: The explicit outcome of the last actuation attempt: allowed only for
        #: a canonical accepted receipt.
        self.last_actuation: ActuationVerdict | None = None
        #: True when this tick's diagnostics publication is suppressed because
        #: the financial route refused a diagnostics overlap.
        self._publish_suppressed = False
        self.fallback_pending = False
        self._last_fallback_at: float | None = None
        self.config: FastControlConfig = config or getattr(
            hass.config, "fast_control", None
        ) or FastControlConfig()
        self.plan_path = canonical_plan_path(plan_path)
        self.state_path = state_path
        self.gateway = gateway or HomeAssistantGateway(hass)
        self.state = load_state(state_path)

        self._plan: Optional[FastPlan] = None
        self._plan_mtime: float = 0.0
        self._batch: dict[str, tuple[Optional[str], float]] = {}
        self._links: dict[str, FastBatteryLink] = {
            link.name: link for link in self.config.batteries
        }
        self._time_zone = getattr(hass, "time_zone", None)
        self._tz_info: Optional[datetime.tzinfo] = None
        self._last_mode: Optional[str] = self._initial_mode()
        self._last_overrides: tuple = ()
        self._last_setpoints: tuple = ()
        self._warned_no_grid = False
        self._warned_stale_plan = False
        self._expired_warned_at = 0.0
        self._last_state_save = 0.0
        self._state_signature: Optional[tuple] = None
        self._measure_start: Optional[int] = None
        self._measure_house_kwh = 0.0
        self._measure_pv_kwh = 0.0
        self._measure_seconds = 0.0
        self._measure_pv_seen = False
        self._measure_last_ts: Optional[float] = None

    # -- helpers ---------------------------------------------------------

    def _getter(self, entity_id: str) -> Optional[str]:
        """FlexValue resolver backed by the current batch, with a live fallback."""
        if entity_id in self._batch:
            return self._batch[entity_id][0]
        try:
            return self.hass.get_state(entity_id).state
        except Exception as exception:  # noqa: BLE001 - silent fallback by design
            # Log at warning so a recurring failure is not invisible; the
            # caller (mode/limits/_flex) will fall back to its default.
            logging.warning(
                f"Fast control: live fallback voor {entity_id} faalde: {exception}"
            )
            return None

    def _flex(self, flex, default):
        if flex is None:
            return default
        try:
            value = flex.resolve(self._getter)
        except Exception as exception:  # noqa: BLE001
            logging.warning(f"Fast control: waarde niet oplosbaar ({exception})")
            return default
        return default if value is None else value

    def mode(self) -> str:
        """Current operating mode, resolved live so it can be switched from HA."""
        # Normalize to lower-case so entity payloads (input_select) that use
        # case other than the literal "shadow"/"active"/"off" still map
        # correctly. Without this a value like "Shadow" silently turns into
        # MODE_OFF because the membership test below only checks lowercase.
        raw = str(self._flex(self.config.mode, MODE_OFF)).strip().lower()
        if raw in {"on", "true", "1", "active"}:
            return MODE_ACTIVE
        if raw in {"shadow", "dry-run", "dryrun", "log"}:
            return MODE_SHADOW
        return MODE_OFF

    def _initial_mode(self) -> str:
        # Resolve once at startup so the very first tick has a baseline
        # to compare against (logs a "modus" line if it differs from the
        # next tick's reading). If the mode is backed by an HA entity
        # and the entity is not yet warm we just see MODE_OFF; the first
        # tick re-reads and triggers a mode_change log when it differs.
        return self.mode()

    def limits(self) -> PolicyLimits:
        """Build the policy tunables, resolving anything backed by an entity."""
        config = self.config
        return PolicyLimits(
            storage_value_mode=str(self._flex(config.storage_value_mode, "plan")),
            storage_value_fixed=(
                self._flex(config.storage_value, None)
                if config.storage_value is not None
                else None
            ),
            round_trip_efficiency=config.round_trip_efficiency,
            min_benefit=config.min_benefit,
            deadband=float(config.deadband),
            min_command_interval=float(config.min_command_interval),
            urgent_deviation=float(config.urgent_deviation),
            max_ramp=float(config.max_ramp) if config.max_ramp else None,
            release_deviation=float(config.release_deviation),
            release_time=float(config.release_time),
            energy_budget=config.energy_budget,
            daily_extra_throughput=config.daily_extra_throughput,
            soc_margin=config.soc_margin,
            max_grid_import=(
                float(config.max_grid_import)
                if config.max_grid_import is not None
                else None
            ),
            allow_grid_charge=bool(self._flex(config.allow_grid_charge, False)),
        )

    def plan(self) -> Optional[FastPlan]:
        """The current plan, reloaded whenever the optimizer rewrote the file."""
        try:
            mtime = os.path.getmtime(self.plan_path)
        except OSError:
            self._plan = None
            return None
        if self._plan is None or mtime != self._plan_mtime:
            self._plan = load_plan(self.plan_path)
            self._plan_mtime = mtime
            if self._plan is not None:
                logging.info(
                    f"Fast control: nieuw plan geladen, "
                    f"{len(self._plan.intervals)} intervallen, "
                    f"{len(self._plan.specs)} batterij(en)"
                )
                self._warned_stale_plan = False
        return self._plan

    def _entity_ids(self, plan: FastPlan) -> list[str]:
        ids: list[str] = []
        ids.extend(self.config.grid_power.entity_ids)
        ids.extend(self.config.pv_power.entity_ids)
        for spec in plan.specs:
            if spec.soc_entity:
                ids.append(spec.soc_entity)
            link = self._links.get(spec.name)
            if link is not None:
                ids.extend(link.actual_power.entity_ids)
        for flex in (
            self.config.mode,
            self.config.storage_value_mode,
            self.config.storage_value,
            self.config.allow_grid_charge,
        ):
            if flex is not None and flex.is_entity_id(flex.value):
                ids.append(str(flex.value))
        return [i for i in dict.fromkeys(ids) if i]

    def _power(
        self, sensor: PowerSensor, now: float
    ) -> tuple[Optional[float], bool]:
        """Read a power sensor. Returns (watts, fresh)."""
        if not sensor.configured:
            return None, False
        scale = 1000.0 if sensor.unit == "kW" else 1.0
        if sensor.invert:
            scale = -scale

        if sensor.entity is not None:
            raw, updated = self._batch.get(sensor.entity, (None, 0.0))
            value = _parse_float(raw)
            if value is None:
                return None, False
            fresh = self._fresh(updated, now)
            return value * scale, fresh

        total = 0.0
        newest = 0.0
        seen = False
        for entity_id, sign in (
            (sensor.entity_positive, 1.0),
            (sensor.entity_negative, -1.0),
        ):
            if entity_id is None:
                continue
            raw, updated = self._batch.get(entity_id, (None, 0.0))
            value = _parse_float(raw)
            if value is None:
                return None, False
            total += sign * value
            newest = max(newest, updated)
            seen = True
        if not seen:
            return None, False
        return total * scale, self._fresh(newest, now)

    def _fresh(self, updated: float, now: float) -> bool:
        if not updated:
            # Home Assistant did not report a timestamp; trust the value.
            return True
        return (now - updated) <= self.config.max_sensor_age

    def _day_key(self, now: float) -> str:
        """Local calendar day, used to roll over the daily budgets."""
        if self._tz_info is None and self._time_zone:
            try:
                self._tz_info = ZoneInfo(self._time_zone)
            except Exception:  # noqa: BLE001 - fall back to the system clock
                self._time_zone = None
        return datetime.datetime.fromtimestamp(now, self._tz_info).strftime("%Y-%m-%d")

    # -- one iteration ---------------------------------------------------

    def _invalidate_current_execution(self) -> None:
        """Withdraw runtime evidence without rewriting historical controller facts."""
        for battery in self.state.batteries:
            battery._current_accepted_command = None

    def _owner_binding(self, owner: OwnerCoordinator | None, *, now_ms: int) -> OwnerBinding | None:
        binding = owner.binding if owner is not None else None
        if (binding is None or owner.phase != "owned" or owner.recovery_required
                or now_ms >= binding.owner_not_after_ms or not owner.store.holds(owner.writer)):
            self._invalidate_current_execution()
            return None
        return binding

    def _consume_block(self, verdict: GateVerdict, *, now_ts: int) -> FallbackReceipt:
        """Trek lokale bevoegdheid in; vraag stop aan zonder zelf HA te bedienen."""
        self.last_gate = verdict
        self._plan = None
        self.state.last_decision = None
        self.state.last_tick_ts = 0.0  # Geen uitvoering/benefit over een geblokkeerde periode boeken.
        self._last_overrides = self._last_setpoints = ()
        for battery in self.state.batteries:
            # Alleen lokale eigendom intrekken, geen nulvermogen of fysieke stop claimen.
            battery.override_active = False
            battery.stop_inverter_cleared = False
            battery.held_stop_inverter_cleared = False
            # A blocked period is not an execution. The recovery duty and the
            # pending command stay exactly as they are.
            apply_execution_outcome(battery, outcome="unknown")
        return self._request_fallback(verdict.reason, now_ts=now_ts)

    def _request_fallback(self, reason: str, *, now_ts: int) -> FallbackReceipt:
        """Share the existing sink and monotonic throttle for every failure."""
        self.fallback_pending = True
        if self.owner is not None:
            self._invalidate_current_execution()
            try:
                barrier = self.owner.stop(reason=reason)
                self._apply_stop_barrier(barrier, now_ms=self.owner.now_ms())
                self._finish_owner_release()
                if barrier is not None and not self.owner.recovery_required and self.owner.phase == "released":
                    receipt = FallbackReceipt("acknowledged", barrier.reason)
                elif self.owner.last_admission_reason in (
                        "admission_deferred", "admission_cycle_used", "admission_window_missed"):
                    receipt = FallbackReceipt("requested", "admission_deferred")
                else:
                    receipt = FallbackReceipt("unconfirmed", "barrier_unconfirmed")
            except Exception as exc:
                self.owner.report_failure(getattr(exc, "reason", "owner_stop_failed"))
                receipt = FallbackReceipt("unconfirmed", "barrier_unconfirmed")
            self.last_fallback = receipt
            return receipt
        monotonic_now = time.monotonic()
        if (self._last_fallback_at is not None
                and monotonic_now - self._last_fallback_at < max(1, self.config.interval)):
            return self.last_fallback
        self._last_fallback_at = monotonic_now
        request = FallbackRequest(reason=reason, requested_ts=now_ts)
        receipt = FallbackReceipt("unconfirmed", "no_fallback_sink")
        if self.fallback is not None:
            try:
                receipt = self.fallback.request(request)
                if type(receipt) is not FallbackReceipt:
                    raise ValueError("Expected explicit FallbackReceipt")
                # Ook een handmatig geconstrueerd/veranderd DTO is geen impliciete ack.
                receipt = FallbackReceipt(receipt.status, receipt.reason)
            except Exception as exc:  # noqa: BLE001 - een kapotte sink verleent nooit rechten
                receipt = FallbackReceipt("unconfirmed", f"fallback_failed: {exc}")
        self.last_fallback = receipt
        if receipt.status == "unconfirmed":
            logging.error(f"Fast control: fallback unconfirmed ({reason}): {receipt.reason}; fysieke stop niet bewezen")
        return receipt

    def _cadence_now(self) -> int:
        now = self.owner.now_ns() if self.owner is not None else time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        if (type(now) is not int or now < 0 or _boot_id() != self._cadence_boot
                or (self._cadence_last_ns is not None and now < self._cadence_last_ns)):
            if self.owner is not None:
                self.owner.report_failure("clock_unhealthy")
            raise RecoveryRequired("clock_unhealthy")
        self._cadence_last_ns = now
        return now

    def _financial_regime(self, now: float) -> bool:
        if self.owner is not None or (self.gate_policy.required_from_ts is not None
                                      and now >= self.gate_policy.required_from_ts):
            self._financial_cadence = True
        if not self._financial_cadence:
            # An unlocked preview selects cadence only. Authorization is always
            # read again under the lease, including a v1 -> v2 publication race.
            preview = read_publication(self.plan_path)
            self._financial_cadence = (preview.status == "blocked" or (
                preview.plan is not None and preview.plan.format_version == 2))
        return self._financial_cadence

    def _read_tick(self, now: float):
        financial = self._financial_regime(now)
        while True:
            cycle = None
            if financial:
                cycle, allowed, _ = owner_admission_at(self._cadence_now())
                if not allowed:
                    raise RecoveryRequired("admission_deferred")
            with publication_lock(self.plan_path):
                if financial:
                    current, allowed, _ = owner_admission_at(self._cadence_now())
                    if current != cycle or not allowed:
                        raise RecoveryRequired("admission_window_missed")
                read = read_publication(self.plan_path)
                unexpected = not financial and (read.status == "blocked" or (
                    read.plan is not None and read.plan.format_version == 2))
                if not unexpected:
                    self._tick_cycle = cycle
                    return read, following_verdict(read, policy=self.gate_policy, now_ts=int(now))
            self._financial_cadence = financial = True

    def tick(self, now: Optional[float] = None) -> Optional[Decision]:
        """Run one control cycle. Returns the decision, or None when idle."""
        now = time.time() if now is None else now
        self.last_receipts = []
        # Every tick starts without a suppression: the flag belongs to the tick
        # that refuses a diagnostics overlap.
        self._publish_suppressed = False
        self._tick_cycle = None

        if self.owner is not None and self.owner.pending_stop_reason is not None:
            reason = self.owner.pending_stop_reason
            self._request_fallback(reason, now_ts=int(now))
            if self.owner.pending_stop_reason is not None:
                return None
            if reason in ("plan_rebind_required", "owner_lease_expired") and self._owner_writer is not None:
                self._owner_start_pending = True
        if self._owner_start_pending and not self._shutting_down:
            self.start_owner_session()
            if self._owner_start_pending or self.owner.binding is None:
                return None

        # Geen mtime-/inodescache als autorisatie. Geen sensor-/sink-I/O onder dit slot.
        try:
            read, verdict = self._read_tick(now)
        except RecoveryRequired as exc:
            self.last_gate = GateVerdict(False, exc.reason)
            return None
        except PublicationLockBusy as exc:
            verdict = GateVerdict(False, "publication_busy" if self._financial_cadence
                                  else f"publication_unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001 - lock/readfout is geen permissieve herstart
            verdict = GateVerdict(False, f"publication_unavailable: {exc}")
        self.last_gate = verdict
        if not verdict.allowed:
            self._consume_block(verdict, now_ts=int(now))
            return None
        plan = read.plan
        self._plan = plan
        if plan is None:
            if not self._warned_stale_plan:
                logging.warning(
                    f"Fast control: geen plan gevonden op {self.plan_path}, "
                    f"wacht op de eerstvolgende optimalisatie"
                )
                self._warned_stale_plan = True
            return None

        self._batch = self.gateway.read(self._entity_ids(plan))
        mode = self.mode()
        if mode != self._last_mode:
            logging.info(f"Fast control: modus {mode}")
        if mode == MODE_OFF:
            if plan.format_version == 2 and self.owner is not None:
                self._request_fallback("mode_off", now_ts=int(now))
                return None
            self._release_when_switched_off(plan, now, mode)
            return None

        if not self.config.grid_power.configured:
            if not self._warned_no_grid:
                logging.warning(
                    "Fast control: geen 'grid power' sensor geconfigureerd, "
                    "de snelle laag doet niets"
                )
                self._warned_no_grid = True
            return None

        grid_w, grid_fresh = self._power(self.config.grid_power, now)
        pv_w, _ = self._power(self.config.pv_power, now)

        batteries: list[BatteryMeasurement] = []
        enabled: list[bool] = []
        for spec in plan.specs:
            link = self._links.get(spec.name)
            soc = None
            if spec.soc_entity:
                soc = _parse_float(self._batch.get(spec.soc_entity, (None, 0.0))[0])
            power_w: Optional[float] = None
            power_fresh = True
            if link is not None and link.actual_power.configured:
                power_w, power_fresh = self._power(link.actual_power, now)
            batteries.append(
                BatteryMeasurement(soc=soc, power_w=power_w, valid=power_fresh)
            )
            enabled.append(link.enabled if link is not None else True)

        plan_stale = plan.age(now) > self.config.max_plan_age
        if plan_stale and not self._warned_stale_plan:
            logging.warning(
                f"Fast control: plan is {plan.age(now) / 60:.0f} minuten oud, "
                f"ouder dan de toegestane {self.config.max_plan_age / 60:.0f} minuten; "
                f"de snelle laag volgt vanaf nu het plan"
            )
            self._warned_stale_plan = True

        sensor_ok = grid_w is not None and grid_fresh
        measurement = Measurement(
            timestamp=now,
            grid_w=grid_w if grid_w is not None else 0.0,
            batteries=batteries,
            pv_w=pv_w,
            grid_valid=sensor_ok and not plan_stale,
            # Distinguish "the grid sensor is unreadable" from "the plan is
            # too old"; both used to be reported as sensor_stale, which is
            # misleading when the sensor is fine and only the plan is old.
            invalid_reason="sensor_stale" if not sensor_ok else "plan_stale",
        )

        # Capture the previous tick timestamp BEFORE policy.account() resets
        # it. Without this saved_today_eur would always integrate over
        # elapsed_h = 0 (policy.account sets state.last_tick_ts = now).
        previous_tick_ts = self.state.last_tick_ts
        # In-force setpoints before this tick, preserved as the event baseline
        # across both policy proposals and subsequent helper acceptance.
        previous_setpoints = tuple(
            self.state.battery(index).held_setpoint_w for index in range(len(plan.specs))
        )

        policy = FastControlPolicy(self.limits())
        owner_binding = self._owner_binding(self.owner, now_ms=int(round(now * 1000)))
        policy.account(
            self.state,
            plan,
            measurement,
            self._day_key(now),
            use_measured=mode == MODE_ACTIVE,
            owner_binding=owner_binding,
        )
        decision = policy.decide(plan, measurement, self.state, enabled, owner_binding=owner_binding)
        self._warn_if_plan_expired(decision, plan, now)

        # decision.house_w is the reconstructed site demand excluding the
        # battery, which is exactly the quantity the optimizer forecasts as
        # "hload". Recording it turns the control loop into the measurement
        # the forecast side has been missing.
        self._measure(now, plan, decision.house_w, pv_w,
                      measurement.grid_valid and decision.reason != "execution_unknown")

        elapsed_h = (
            min(now - previous_tick_ts, 600.0) / 3600.0
            if previous_tick_ts
            else 0.0
        )
        self._book_saving(decision, elapsed_h=elapsed_h, plan=plan, owner_binding=owner_binding,
                          previous_tick_ts=previous_tick_ts)
        if mode != MODE_ACTIVE:
            # Nothing was actually written this tick, so this contribution
            # (and therefore today's total) is a projection, not a measured
            # saving.
            self.state.saved_today_is_estimate = True

        self._actuate(decision, plan, mode)
        self._publish(decision, mode, measurement)
        self._record_events(decision, mode, previous_setpoints)
        self.state.refresh_budget_aggregates()
        self._persist_state(now)
        return decision

    def _book_saving(self, decision: Decision, *, elapsed_h: float,
                     plan: FastPlan | None = None, owner_binding: OwnerBinding | None = None,
                     previous_tick_ts: float | None = None) -> None:
        """Book the proven saving of this tick into today's total.

        The euro claim of a decision is only nonzero when the segment was
        really executed: an unproven or ambiguous segment books nothing at
        all, so this addition stays zero for a refused tick.
        """
        benefit = decision.benefit_eur_h
        if plan is not None and plan.format_version == 2:
            digest = execution_boundary.plan_digest(plan)
            # Keep the legacy euro cap, but prove the entire previous-tick gap.
            previous = (decision.timestamp - elapsed_h * 3600
                        if previous_tick_ts is None else previous_tick_ts)
            covered_benefit = 0.0
            for battery in decision.batteries:
                if not battery.override:
                    continue
                controller = self.state.battery(battery.index)
                command = current_execution_command(
                    controller, binding=owner_binding, now_ms=int(round(decision.timestamp * 1000)),
                    expected_digest=digest)
                if (command is None or command.issued_at_ms > int(round(previous * 1000))
                        or controller.last_command_ts > previous):
                    continue
                # Filter existing per-battery claims; do not recalculate value
                # or discard another battery's independently proven segment.
                covered_benefit += max(0.0, battery.benefit_eur_h)
            benefit = min(benefit, covered_benefit)
        self.state.saved_today_eur += benefit * elapsed_h

    def _warn_if_plan_expired(self, decision, plan: FastPlan, now: float) -> None:
        """Keep saying so while the plan does not cover the current moment.

        An expired plan parks every battery at 0 W, which is the safe thing
        to do, but it means the layer does nothing at all until the optimiser
        writes a new plan. That used to be reported once and never again:
        because the event log only records a setpoint *change*, a plan that
        stayed expired for days showed up as a single "plan_expired" entry
        with everything at 0 W and no further trace, in the log or anywhere
        else. This repeats, so the condition stays visible and names the
        cause.
        """
        if decision.reason != "plan_expired":
            self._expired_warned_at = 0.0
            return
        if now - self._expired_warned_at < EXPIRED_WARN_EVERY_S:
            return
        self._expired_warned_at = now
        age_h = plan.age(now) / 3600.0
        created = datetime.datetime.fromtimestamp(plan.created_ts).strftime(
            "%Y-%m-%d %H:%M"
        )
        logging.warning(
            f"Fast control: het plan van {created} ({age_h:.1f} uur oud) dekt "
            f"dit moment niet meer, alle batterijen staan op 0 W. De snelle "
            f"laag doet niets tot de optimalisering een nieuw plan schrijft "
            f"in {self.plan_path}; zoek in het log naar \"Plan voor de snelle "
            f"regellaag\" om te zien waarom dat niet gebeurt."
        )

    def _release_when_switched_off(self, plan: FastPlan, now: float, mode: str) -> None:
        """Undo an active override when the layer is turned off.

        Without this the inverter kept the overridden setpoint, with the
        optimizer's stop moment cleared, until the next optimisation run.
        A shadow-only projection is not a restore obligation. Confirmed stop
        ownership and failed restores remain due even after transitioning off.
        """
        # Preserve the first release of a partially failed active override:
        # helpers may have written even though no command was confirmed (R19).
        active_override = self._last_mode == MODE_ACTIVE and any(
            b.override_active for b in self.state.batteries
        )
        for battery in self.state.batteries:
            if not battery.stop_inverter_cleared and not battery.restore_pending:
                battery.held_stop_inverter_cleared = False
                if battery.last_command_ts == 0 and not active_override:
                    battery.override_active = False
        pending = any(
            b.restore_pending or b.stop_inverter_cleared
            or (b.override_active and b.last_command_ts > 0)
            for b in self.state.batteries
        )
        if not pending and not active_override:
            self._last_mode = mode
            return
        measurement = Measurement(
            timestamp=now,
            grid_w=0.0,
            batteries=[BatteryMeasurement(valid=False) for _ in plan.specs],
            grid_valid=False,
        )
        decision = FastControlPolicy(self.limits()).release(
            plan, measurement, self.state, "mode_off",
            owner_binding=self._owner_binding(self.owner, now_ms=int(round(now * 1000))),
        )
        if self._last_mode == MODE_ACTIVE or any(
            b.stop_inverter_cleared or b.restore_pending for b in self.state.batteries
        ):
            # A failed off-release keeps its confirmed stop status. Retry that
            # restore even after the mode event has already transitioned to off.
            logging.info("Fast control: uitgeschakeld, plan-setpoint hersteld")
            self._actuate(decision, plan, MODE_ACTIVE)
        self._publish(decision, mode, measurement)
        self._record_events(decision, mode)
        self._persist_state(now, force=True)

    # -- measurement for the forecast side -------------------------------

    def _measure(
        self,
        now: float,
        plan: FastPlan,
        house_w: float,
        pv_w: Optional[float],
        usable: bool,
    ) -> None:
        """Integrate the realised house demand and PV over each plan interval.

        The optimizer writes what it expected; this writes what happened, on
        the same time grid and with the same definition. Without it there is
        nothing to compare a forecast against, because Home Assistant's own
        statistics do not know where the battery boundary is.
        """
        interval = plan.interval_at(now)
        if interval is None:
            return
        if self._measure_start is not None and interval.start_ts != self._measure_start:
            self._flush_measurement(plan)
        if self._measure_start != interval.start_ts:
            self._measure_start = interval.start_ts
            self._measure_house_kwh = 0.0
            self._measure_pv_kwh = 0.0
            self._measure_seconds = 0.0
            self._measure_pv_seen = False
            self._measure_last_ts = now
            return

        previous = self._measure_last_ts
        self._measure_last_ts = now
        if previous is None or now <= previous or not usable:
            return
        # Guard against a long gap, for example after the add-on was stopped:
        # integrating across it would invent energy that was never measured.
        elapsed = now - previous
        if elapsed > 10 * max(5, int(self.config.interval)):
            return
        hours = elapsed / 3600.0
        self._measure_house_kwh += house_w * hours / 1000.0
        self._measure_seconds += elapsed
        if pv_w is not None:
            self._measure_pv_kwh += pv_w * hours / 1000.0
            self._measure_pv_seen = True

    def _flush_measurement(self, plan: FastPlan) -> None:
        """Write the finished interval to the values table."""
        start = self._measure_start
        self._measure_start = None
        if start is None or self._measure_seconds <= 0:
            return
        interval = next(
            (i for i in plan.intervals if i.start_ts == start), None
        )
        duration = interval.duration_s if interval else plan.interval_s
        coverage = self._measure_seconds / max(1, duration)
        if coverage < MEASUREMENT_MIN_COVERAGE:
            logging.debug(
                f"Fast control: interval {start} maar {coverage:.0%} gemeten, "
                f"niet opgeslagen"
            )
            return

        rows = [[str(int(start)), "m_house", round(self._measure_house_kwh, 4)]]
        if self._measure_pv_seen:
            rows.append([str(int(start)), "m_pv", round(self._measure_pv_kwh, 4)])
        database = getattr(self.hass, "db_da", None)
        if database is None:
            return
        try:
            database.savedata(
                pd.DataFrame(rows, columns=["time", "code", "value"]),
                tablename="values",
            )
        except Exception as exception:  # noqa: BLE001 - never break the loop
            logging.warning(
                f"Fast control: meting kon niet worden opgeslagen: {exception}"
            )

    def _persist_state(self, now: float, force: bool = False) -> None:
        """Write the controller state, but not on every single tick.

        See :data:`STATE_SAVE_INTERVAL`. Anything that would be painful to lose
        -- an override starting or ending, a new plan, a new day -- changes the
        signature below and is written straight away.
        """
        signature = (
            self.state.day_key,
            self.state.plan_created_ts,
            self.state.interval_start_ts,
            tuple(b.override_active for b in self.state.batteries),
            len(self.state.events),
            self.state.events[-1] if self.state.events else None,
        )
        due = (now - self._last_state_save) >= STATE_SAVE_INTERVAL
        if not (force or due or signature != self._state_signature):
            return
        save_state(self.state, self.state_path)
        self._last_state_save = now
        self._state_signature = signature

    def _actuate(self, decision: Decision, plan: FastPlan, mode: str) -> tuple[CommandReceipt, ...]:
        """Route the tick to the legacy (v1) or the financial (v2) actuator.

        The legacy route stays exactly as it was: the helper path, the legacy
        receipt and the legacy state update. The financial route has its own
        consumer, its own adapter qualification and its own receipts.

        A plan is financial when it is a v2 plan. A v2 plan without a financial
        context is refused by the financial route itself; it never falls
        through to the legacy helpers.
        """
        if plan.format_version == 2:
            return self._actuate_financial(decision, plan, mode)
        return self._actuate_legacy(decision, plan, mode)

    def _actuate_legacy(self, decision: Decision, plan: FastPlan, mode: str) -> tuple[CommandReceipt, ...]:
        self.last_receipts = []
        for battery in decision.batteries:
            battery.actuation_status = None
            if not battery.write:
                continue
            spec = plan.spec(battery.index)
            if spec is None:
                continue
            arrow = "laden" if battery.setpoint_w > 0 else "ontladen"
            message = (
                f"Fast control {spec.name}: {battery.setpoint_w:.0f} W {arrow} "
                f"(plan {battery.plan_w:.0f} W, {battery.reason}, "
                f"{battery.benefit_eur_h:.3f} euro/uur)"
            )
            if mode != MODE_ACTIVE:
                logging.info(f"[schaduw] {message}")
                if mode == MODE_SHADOW:
                    # Project the in-force command, never helper acceptance.
                    controller = self.state.battery(battery.index)
                    controller.held_setpoint_w = battery.setpoint_w
                    if battery.accepted is not None:
                        controller.held_stop_inverter_cleared = battery.accepted.stop_inverter_cleared
                continue
            logging.info(message)
            try:
                receipt = execute_battery_command(self.gateway, plan, battery, decision.timestamp)
            except ValueError:
                # Identity failure occurs before any helpercall; no receipt or
                # implicit acceptance, but keep the same diagnostic/fallback path.
                battery.actuation_status = "unconfirmed"
                self._request_fallback("actuation_identity_invalid", now_ts=int(decision.timestamp))
                continue
            if receipt is None:
                continue
            self.last_receipts.append(receipt)
            battery.actuation_status = receipt.status
            controller = self.state.battery(battery.index)
            if battery.restores:
                controller.restore_pending = receipt.status != "confirmed"
            if receipt.status == "confirmed":
                apply_accepted_command(controller, battery)
            else:
                # Keep accepted facts, held setpoint and decision-owned mode
                # intact so a failed override/restore can be retried next tick.
                self._request_fallback(f"actuation_{receipt.status}", now_ts=int(decision.timestamp))
        return tuple(self.last_receipts)

    def _actuate_financial(self, decision: Decision, plan: FastPlan, mode: str) -> tuple[CommandReceipt, ...]:
        """The financial route: one owner, one writer, never the legacy helpers."""
        self.last_receipts = []
        # The overlap is determined first: the publication of a financial tick
        # with an overlap is suppressed, whatever the refusal reason will be.
        overlap = self._suppress_overlap(plan)
        if plan.financial_context is None:
            # A v2 plan without a financial context is not a financial plan:
            # there is no identity to bind a command to. The reason keeps its
            # precedence over the overlap.
            return self._deny_financial("missing_financial_context", decision)
        if overlap:
            return self._deny_financial("diagnostics_overlap", decision)
        if mode == MODE_SHADOW:
            # Shadow stays a projection: no receipt and no command state change.
            return self._deny_financial("shadow", decision)
        if mode != MODE_ACTIVE:
            return self._deny_financial("mode_inactive", decision)
        return self._consume_financial_actuation(
            decision, plan, owner=self.owner, now_ms=int(round(decision.timestamp * 1000))
        )

    def _suppress_overlap(self, plan: FastPlan) -> bool:
        """Whether the diagnostics of this plan overlap an actuator entity.

        The overlap is determined without side effects apart from the
        suppression of the publication: the diagnostics may never write an
        actuator entity, not even in a tick that is refused for another reason.
        """
        if not self._diagnostics_overlap(plan):
            return False
        self._publish_suppressed = True
        return True

    def _deny_financial(self, reason: str, decision: Decision) -> tuple[ActuationReceipt, ...]:
        """An explicit refusal: no receipt, no command, a readable reason.

        A refusal that means "this tick executed nothing" applies the row
        `unknown`: an old accepted status may not stay behind while nothing
        was executed. A shadow tick and a deadband tick are no-ops: they say
        nothing about an execution at all.
        """
        self.last_receipts = []
        self.last_actuation = ActuationVerdict(False, reason)
        for battery in decision.batteries:
            battery.actuation_status = None
        if reason not in ("shadow", "no_proposal"):
            for battery in decision.batteries:
                apply_execution_outcome(self.state.battery(battery.index), outcome="unknown")
        return ()

    def _diagnostics_overlap(self, plan: FastPlan) -> bool:
        """The diagnostics write Home Assistant entities of their own.

        They may not alias the actuator, mode or stop entity of a battery: the
        diagnostic write would then become a second, unmarked command writer.
        The comparison is on the exact entity name.
        """
        diagnostics = self.config.diagnostics
        published = {
            diagnostics.entity_status,
            diagnostics.entity_active,
            diagnostics.entity_setpoint,
            diagnostics.entity_benefit,
            diagnostics.entity_saved_today,
        }
        published = {entity for entity in published if entity}
        if not published:
            return False
        for spec in plan.specs:
            for entity in (spec.setpoint_entity, spec.mode_entity, spec.stop_inverter_entity):
                if entity and entity in published:
                    return True
        return False

    def _command_request(self, plan: FastPlan, decision: Decision,
                         battery: BatteryDecision) -> execution_boundary.CommandRequest:
        """One canonical command request: no identity, no epoch, no digest."""
        spec = plan.spec(battery.index)
        setpoint_w = int(round(battery.setpoint_w))
        mode = battery.mode
        if mode is None:
            mode = spec.mode_on if abs(setpoint_w) >= POWER_EPS else spec.mode_off
        return execution_boundary.CommandRequest(
            battery_name=battery.name,
            setpoint_w=setpoint_w,
            mode=mode,
            stop_at_ms=self._command_stop_at_ms(plan, decision, battery),
        )

    def _command_stop_at_ms(self, plan: FastPlan, decision: Decision,
                            battery: BatteryDecision) -> Optional[int]:
        """The stop moment this command carries; None means "no stop"."""
        if battery.stop_inverter is not None:
            if battery.stop_inverter == NO_STOP_SENTINEL:
                return None
            try:
                moment = datetime.datetime.fromisoformat(battery.stop_inverter)
            except ValueError:
                return None
            return int(round(moment.timestamp() * 1000))
        interval = plan.interval_at(decision.timestamp)
        if interval is None:
            return None
        step = interval.battery(battery.index)
        if step.stop_ts is None:
            return None
        return int(step.stop_ts) * 1000

    def _consume_financial_actuation(self, decision: Decision, plan: FastPlan, *,
                                     owner: OwnerCoordinator | None,
                                     now_ms: int) -> tuple[ActuationReceipt, ...]:
        """The one financial consumer: the owner writes, the receiver confirms.

        Every refusal is explicit and machine-readable. No epoch, installation
        id or command identity is ever invented: without a bound owner or a
        nominally qualified adapter there simply is no command to give.

        ``decision.timestamp`` is the original proposal time. The caller's
        ``now_ms`` may be that same old tick time, never proof of execution time:
        owner validation uses the owner clock and execution samples it under
        the final publication lease.
        """
        self.last_receipts = []
        # The overlap is determined first: the publication of a financial tick
        # with an overlap is suppressed, whatever the refusal reason will be.
        overlap = self._suppress_overlap(plan)
        if plan.financial_context is None:
            # The reason keeps its precedence over the overlap.
            return self._deny_financial("missing_financial_context", decision)
        if overlap:
            return self._deny_financial("diagnostics_overlap", decision)
        if owner is not None:
            now_ms = owner.now_ms()
        binding = self._owner_binding(owner, now_ms=now_ms)
        if binding is None:
            if owner is not None and owner.binding is not None and now_ms >= owner.binding.owner_not_after_ms:
                owner.pending_stop_reason = "owner_lease_expired"
                owner.report_failure("owner_lease_expired")
            return self._deny_financial("owner_unavailable", decision)
        receiver = owner.receiver
        if receiver is None or classify_actuation_adapter(receiver) != FENCED_ADAPTER:
            return self._deny_financial("adapter_unqualified", decision)
        if any(b.recovery_required for b in self.state.batteries):
            # An ambiguous outcome stops the whole installation: no further
            # command, not even a stop, until an exact safe barrier.
            return self._deny_financial("recovery_required", decision)
        proposals = [b for b in decision.batteries if b.write]
        if not proposals:
            return self._deny_financial("no_proposal", decision)
        if self._shutting_down or (self._stop_event is not None and self._stop_event.is_set()):
            return self._deny_financial("shutdown", decision)
        if self._tick_cycle is not None:
            try:
                cycle, allowed, _ = owner_admission_at(self._cadence_now())
            except RecoveryRequired as exc:
                return self._deny_financial(exc.reason, decision)
            if not allowed or cycle != self._tick_cycle:
                return self._deny_financial("admission_window_missed", decision)
        context = plan.financial_context
        proposal_ms = int(round(decision.timestamp * 1000))
        # One absolute section window for the whole installation. The owner caps
        # transport at 750 ms from section entry, including sampling/acquisition,
        # retries and next batteries; a fresh UTC sample never renews that budget.
        execute_before = min(context.not_after_ts * 1000, binding.owner_not_after_ms)
        receipts: list[ActuationReceipt] = []
        accepted_count = 0
        refusal = ""
        try:
            deadline = Deadline(owner._clock() + execution_boundary.SECTION_BUDGET_NS,
                                execute_before)
            # The publication lease is taken here and released only after the
            # result has been recorded. There is no nested acquisition.
            with owner.final_section(operation="send", deadline=deadline) as lease:
                # Anchor BOOTTIME before reading UTC: latency between the reads
                # must spend budget, never shift a wire deadline into the future.
                sampled_ns = owner._clock()
                clock = ClockSample(utc_ms=owner.now_ms(), boottime_ns=sampled_ns,
                                    boot_id=_boot_id(), uncertainty_ms=0, healthy=True)
                if clock.utc_ms >= binding.owner_not_after_ms:
                    owner.pending_stop_reason = "owner_lease_expired"
                    owner.report_failure("owner_lease_expired")
                    return self._deny_financial("owner_unavailable", decision)
                # The proposal is bound to the fresh publication under the lock:
                # only the published plan may carry this command.
                try:
                    execution_boundary.validate_current_locked(
                        lease=lease, expected_digest=execution_boundary.plan_digest(plan),
                        clock=clock, policy=self.gate_policy)
                    deadline = execution_boundary._bounded_deadline(
                        deadline, clock=clock, wire_end_ms=execute_before)
                except execution_boundary.ExecutionRefused as exception:
                    return self._deny_financial(exception.reason, decision)
                evidence = [owner.state.pending_command, owner.state.last_receipt]
                for controller in self.state.batteries:
                    evidence.extend((controller.pending_command, controller.last_accepted_command))
                digest = execution_boundary.plan_digest(plan)
                if any(item is not None and item.installation_id == binding.installation_id
                       and item.owner_epoch == binding.owner_epoch and item.plan_digest != digest
                       for item in evidence):
                    # The existing lifecycle must withdraw and rebind before a
                    # new digest may use this installation. No reserve/send here.
                    owner.report_failure("plan_rebind_required")
                    owner.pending_stop_reason = "plan_rebind_required"
                    self._invalidate_current_execution()
                    return self._deny_financial("plan_rebind_required", decision)
                if not proposals:
                    return self._deny_financial("no_proposal", decision)
                captured, restore = self._observe_pending_command(owner)
                try:
                    for battery in proposals:
                        if self._stop_event is not None and self._stop_event.is_set():
                            refusal = "shutdown"
                            self._record_refusal(battery, owner)
                            break
                        try:
                            # Fresh issuance cannot relabel a proposal across its
                            # original active/stop/interval edge, even at equal
                            # power. This check is under the final lease and runs
                            # before the mechanical boundary may reserve anything.
                            proposal_end = execution_boundary._command_end_ms(
                                plan, battery_name=battery.name, at_ms=proposal_ms,
                                owner_end_ms=binding.owner_not_after_ms)
                            if (not proposal_ms <= clock.utc_ms < proposal_end
                                    or owner.now_ms() >= proposal_end):
                                raise execution_boundary.ExecutionRefused("command_window_invalid")
                            receipt = execution_boundary.owner_section(
                                lease=lease, owner=owner, receiver=receiver,
                                request=self._command_request(plan, decision, battery),
                                policy=self.gate_policy, clock=clock, deadline=deadline,
                            )
                        except execution_boundary.ExecutionRefused as exception:
                            # A refusal is an execution outcome, never a silent end.
                            refusal = exception.reason
                            self._record_refusal(battery, owner)
                            break
                        if receipt is None:
                            refusal = "no_receipt"
                            self._record_refusal(battery, owner)
                            break
                        receipts.append(receipt)
                        pending = captured.get(battery.name)
                        applied = self._record_financial_outcome(battery, receipt, pending)
                        if receipt.status != "accepted":
                            # An ambiguous or refused outcome stops the installation.
                            break
                        if not applied:
                            # A raw ACK is no acceptance. Preserve the own command
                            # under this same lease and derive uncertainty from it.
                            refusal = "receipt_not_applied"
                            if pending is not None and owner.state.pending_command != pending:
                                owner.record_pending_command(pending)
                            apply_execution_outcome(self.state.battery(battery.index),
                                                    outcome="unconfirmed", pending=pending)
                            battery.actuation_status = "unconfirmed"
                            break
                        accepted_count += 1
                finally:
                    restore()
        except PublicationLockBusy:
            return self._deny_financial("publication_busy", decision)
        except RecoveryRequired as exc:
            return self._deny_financial(exc.reason, decision)

        allowed = bool(proposals) and not refusal and accepted_count == len(proposals)
        if refusal:
            reason = refusal
        elif not receipts:
            reason = "no_receipt"
        elif allowed:
            reason = "accepted"
        else:
            reason = receipts[-1].status
        self.last_receipts = list(receipts)
        self.last_actuation = ActuationVerdict(allowed, reason)
        return tuple(receipts)

    def _observe_pending_command(self, owner: OwnerCoordinator):
        """Observe the exact stored command; restore the attribute exactly.

        The observation exists only while the publication lease is held, so a
        command can never be watched without the lock. The durable storage is
        always the class method: an earlier instance attribute never replaces
        it, so the pending command always lands in the owner's own mirror. An
        earlier instance attribute is put back, not removed.
        """
        had_attribute = "record_pending_command" in owner.__dict__
        original = owner.__dict__.get("record_pending_command")
        captured: dict[str, BatteryCommand] = {}

        def capture(command: BatteryCommand) -> None:
            captured[command.battery_name] = command
            # The durable storage is the class method, never an instance
            # attribute that shadows it.
            OwnerCoordinator.record_pending_command(owner, command)

        def restore() -> None:
            if had_attribute:
                owner.record_pending_command = original
            else:
                del owner.record_pending_command

        owner.record_pending_command = capture
        return captured, restore

    def _pending_in_mirror(self, owner: OwnerCoordinator | None,
                           battery_name: str) -> BatteryCommand | None:
        """The pending command of this battery in the owner's own mirror."""
        state = owner.state if owner is not None else None
        pending = state.pending_command if state is not None else None
        if pending is None or pending.battery_name != battery_name:
            return None
        if state.owner_epoch != pending.owner_epoch:
            return None
        return pending

    def _record_refusal(self, battery: BatteryDecision,
                        owner: OwnerCoordinator | None) -> None:
        """A refusal is an execution outcome: the evidence is the own mirror.

        A pending command for this battery in the current owner epoch means the
        effect is uncertain; without one there is no effect proven.
        """
        controller = self.state.battery(battery.index)
        pending = self._pending_in_mirror(owner, battery.name)
        if pending is None:
            apply_execution_outcome(controller, outcome="unknown")
            return
        apply_execution_outcome(controller, outcome="unconfirmed", pending=pending)

    def _record_financial_outcome(self, battery: BatteryDecision, receipt: ActuationReceipt,
                                  pending: BatteryCommand | None) -> bool:
        """Fold a receipt into controller state; report applied acceptance, no I/O."""
        controller = self.state.battery(battery.index)
        battery.actuation_status = receipt.status
        if receipt.status == "accepted":
            applied = apply_execution_outcome(controller, outcome="accepted",
                                              receipt=receipt, pending=pending)
            if applied:
                # The in-force facts belong to the runner, not to the receipt:
                # the policy never invents an in-force reality. They follow an
                # applied acceptance and nothing else.
                stored = controller.last_accepted_command
                controller.held_setpoint_w = float(stored.setpoint_w)
                controller.held_stop_inverter_cleared = controller.stop_inverter_cleared
                controller.last_change_ts = receipt.accepted_ts / 1000.0
            return applied
        elif receipt.status == "unconfirmed":
            apply_execution_outcome(controller, outcome="unconfirmed", receipt=receipt,
                                    pending=pending)
        else:
            # A proven absence of effect is not an execution.
            apply_execution_outcome(controller, outcome="unknown", receipt=receipt,
                                    pending=pending)
        return False

    def _apply_stop_barrier(self, barrier: BarrierReceipt | None, *, now_ms: int) -> None:
        """Only the owner's own exact safe barrier ends the financial recovery duty.

        The barrier is compared field by field with the owner's own stored
        barrier receipt: a receipt that differs in any field is not the
        owner's barrier. The timeliness of that receipt is judged on its own:
        a future acceptance is not a fact and never lifts the duty, not even
        without a pending command. The epoch coverage is only required when
        there is a pending command that the barrier has to withdraw.
        """
        if type(barrier) is not BarrierReceipt:
            return
        owner = self.owner
        state = owner.state if owner is not None else None
        if state is None or state.phase != "released":
            return
        own = state.last_barrier
        if own is None or type(own) is not BarrierReceipt:
            return
        # The barrier must be the owner's own last barrier, field by field.
        if (barrier.barrier_id != own.barrier_id
                or barrier.installation_id != own.installation_id
                or barrier.receiver_generation != own.receiver_generation
                or barrier.revoked_through_epoch != own.revoked_through_epoch
                or barrier.current_epoch != own.current_epoch
                or barrier.status != own.status
                or barrier.queues_fenced != own.queues_fenced
                or barrier.safe_state != own.safe_state
                or barrier.accepted_ts != own.accepted_ts):
            return
        if barrier.installation_id != state.installation_id:
            return
        if barrier.receiver_generation != state.receiver_generation:
            return
        if (barrier.status != "accepted" or not barrier.queues_fenced
                or barrier.safe_state != "confirmed"):
            return
        if barrier.accepted_ts is None:
            # Without an acceptance moment the barrier proves nothing.
            return
        if barrier.accepted_ts > now_ms:
            # A future timestamp is not a fact: the duty stays.
            return
        self._invalidate_current_execution()
        for battery in self.state.batteries:
            pending = battery.pending_command
            if pending is None and not battery.recovery_required:
                # No open recovery duty: this battery stays untouched.
                continue
            if pending is not None:
                if barrier.accepted_ts < pending.issued_at_ms:
                    # A barrier from before the command does not withdraw it.
                    continue
                if barrier.revoked_through_epoch < pending.owner_epoch:
                    # The barrier does not cover the epoch of the command.
                    continue
            apply_execution_outcome(battery, outcome="stopped")

    def _publish(self, decision: Decision, mode: str, measurement: Measurement) -> None:
        if self._publish_suppressed:
            # A diagnostics overlap refused the financial route: this tick
            # writes nothing at all, not even a diagnostic entity that does
            # not overlap.
            return
        diagnostics = self.config.diagnostics
        attributes = decision.as_attributes()
        attributes["mode"] = mode
        attributes["pv_w"] = (
            round(measurement.pv_w) if measurement.pv_w is not None else None
        )
        attributes["saved_today_eur"] = round(self.state.saved_today_eur, 3)
        attributes["saved_today_is_estimate"] = self.state.saved_today_is_estimate
        attributes["friendly_name"] = "DAO fast control"
        attributes["icon"] = "mdi:speedometer"

        state = "override" if decision.override else decision.reason
        if mode == MODE_SHADOW:
            state = f"shadow:{state}"

        self.state.last_decision = {
            **attributes,
            "ts": decision.timestamp,
            "mode": mode,
            "state": state,
        }

        if diagnostics.entity_status:
            self.gateway.publish_state(diagnostics.entity_status, state, attributes)
        if diagnostics.entity_active:
            self.gateway.switch(diagnostics.entity_active, decision.override)
        if diagnostics.entity_setpoint and decision.batteries:
            self.gateway.write_number(
                diagnostics.entity_setpoint,
                sum(b.setpoint_w for b in decision.batteries),
            )
        if diagnostics.entity_benefit:
            self.gateway.write_number(
                diagnostics.entity_benefit, round(decision.benefit_eur_h, 4)
            )
        if diagnostics.entity_saved_today:
            self.gateway.write_number(
                diagnostics.entity_saved_today, round(self.state.saved_today_eur, 3)
            )

    # -- event ring buffer ----------------------------------------------

    def _record_events(
        self, decision, mode: str, previous_setpoints: Optional[tuple] = None
    ) -> None:
        new_overrides = tuple(b.override for b in decision.batteries)
        new_setpoints = tuple(b.setpoint_w for b in decision.batteries)
        events = self.state.events

        if mode != self._last_mode:
            events.append(self._event_dict(decision, mode, EVENTS_KIND_MODE))
        if self._last_overrides and any(self._last_overrides) and not any(new_overrides):
            events.append(self._event_dict(decision, mode, EVENTS_KIND_OVERRIDE_END))
        elif (not self._last_overrides or not any(self._last_overrides)) and any(new_overrides):
            events.append(self._event_dict(decision, mode, EVENTS_KIND_OVERRIDE_START))

        # A setpoint event marks a real change. On the first tick after a
        # start there is no previous decision yet; compare against what the
        # batteries were last commanded to (persisted across restarts), so a
        # correction in the very first tick is logged and a plain
        # plan-following start is not.
        if self._last_setpoints:
            baseline = self._last_setpoints
        elif previous_setpoints is not None:
            baseline = previous_setpoints
        else:
            baseline = tuple(
                self.state.battery(b.index).held_setpoint_w for b in decision.batteries
            )
        if any(abs(new - old) >= 1.0 for new, old in zip(new_setpoints, baseline)):
            events.append(self._event_dict(decision, mode, EVENTS_KIND_SETPOINT))

        events[:] = events[-EVENTS_LIMIT:]

        self._last_mode = mode
        self._last_overrides = new_overrides
        self._last_setpoints = new_setpoints

    def _event_dict(self, decision, mode: str, kind: str) -> dict:
        battery = decision.batteries[0] if decision.batteries else None
        socs = [b.soc for b in decision.batteries if b.soc is not None]
        soc_pct = round(sum(socs) / len(socs), 1) if socs else None
        return {
            "ts": decision.timestamp,
            "kind": kind,
            "mode": mode,
            "reason": (battery.reason if battery else decision.reason),
            "battery": (battery.name if battery else None),
            "setpoint_w": battery.setpoint_w if battery else 0,
            "plan_w": battery.plan_w if battery else 0,
            "benefit_eur_h": max((b.benefit_eur_h for b in decision.batteries), default=0.0),
            "house_w": round(decision.house_w),
            "soc_pct": soc_pct,
            "price_import": round(decision.price_import, 4),
            "price_export": round(decision.price_export, 4),
        }

    # -- the loop --------------------------------------------------------

    def start_owner_session(self) -> None:
        """Optional fenced-owner lifecycle; never changes the legacy route."""
        self._invalidate_current_execution()
        if (self.owner is None or self._shutting_down
                or (self._stop_event is not None and self._stop_event.is_set())):
            return
        if self._owner_writer is not None and not self._owner_start_pending:
            return
        try:
            store = self.owner.store
            if self._owner_writer is None:
                self._owner_writer = task_execution.acquire_resource(store.resource, lease_dir=store.lease_dir)
                self._owner_session_id = uuid.uuid4().hex
                self._owner_start_pending = True
                self.owner._bind_lifetime(self._owner_writer)
            self.owner.report_failure("owner_lease_missing")
            self.owner._require_receiver(self.owner.receiver)
            with self.owner.final_section(operation="start") as lease:
                # The plan, its lease and startup share this one final claim.
                # Plan I/O consumes the budget that began before acquisition.
                read = read_publication(lease.path)
                if read.plan is None or read.plan.format_version != 2:
                    self.owner.report_failure("financial_plan_required")
                    self._owner_start_pending = False
                    return
                now_ms = self.owner.now_ms()
                clock = ClockSample(now_ms, self.owner._clock(), _boot_id(), 0, True)
                plan = execution_boundary.validate_current_locked(
                    lease=lease, expected_digest=execution_boundary.plan_digest(read.plan),
                    clock=clock, policy=self.gate_policy)
                if plan.financial_context.basis.meter_id != store.installation_id:
                    raise RecoveryRequired("installation_mismatch")
                requested = plan.financial_context.not_after_ts * 1000
                if self._shutting_down or (self._stop_event is not None and self._stop_event.is_set()):
                    self._owner_start_pending = False
                    self.owner.report_failure("shutdown")
                    return
                deadline = Deadline(self.owner.section_deadline.monotonic_end_ns, requested)
                result = self.owner._start_locked(lease=lease, writer=self._owner_writer,
                    receiver=self.owner.receiver, installation_id=store.installation_id,
                    owner_session_id=self._owner_session_id, requested_not_after_ms=requested,
                    deadline=deadline, now_ms=now_ms)
            self._owner_start_pending = isinstance(result, RecoveryRequired) and result.reason in (
                "admission_deferred", "admission_cycle_used", "admission_window_missed", "publication_busy")
            if isinstance(result, RecoveryRequired):
                logging.warning("Fast control: owner inhibited: %s", result.reason)
        except task_execution.LeaseBusy as exception:
            self.owner.report_failure("owner_lease_busy")
            logging.warning("Fast control: owner lease bezet: %s", exception)
        except PublicationLockBusy:
            self._owner_start_pending = True
        except (RecoveryRequired, execution_boundary.ExecutionRefused) as exception:
            self._owner_start_pending = exception.reason in (
                "admission_deferred", "admission_cycle_used", "admission_window_missed", "publication_busy")
            self.owner.report_failure(exception.reason)
            logging.warning("Fast control: owner inhibited: %s", exception.reason)
        except Exception as exception:  # noqa: BLE001 - optional lifecycle never aborts legacy
            self.owner.report_failure(exception.reason if isinstance(exception, (MirrorWriteFailed, RecoveryRequired))
                                      else "owner_start_failed")
            logging.warning("Fast control: owner start mislukt: %s", exception)

    def stop_owner_session(self, reason: str) -> None:
        self._invalidate_current_execution()
        if self.owner is None:
            return None
        self._owner_release_pending = True
        self._owner_start_pending = False
        barrier = None
        try:
            barrier = self.owner.stop(reason=reason)
            # An exact accepted safe barrier ends the financial recovery duty.
            self._apply_stop_barrier(barrier, now_ms=self.owner.now_ms())
        except Exception as exception:  # noqa: BLE001
            self.owner.report_failure(exception.reason if isinstance(exception, (MirrorWriteFailed, RecoveryRequired))
                                      else "owner_stop_failed")
            logging.warning("Fast control: owner stop mislukt: %s", exception)
        finally:
            self._finish_owner_release()
        return barrier

    def _finish_owner_release(self) -> None:
        """Complete a session release, including a barrier deferred to a later tick."""
        if not self._owner_release_pending:
            return
        if not self._shutting_down and (self.owner.phase != "released" or self.owner.recovery_required):
            return
        writer, self._owner_writer = self._owner_writer, None
        self._owner_release_pending = False
        if writer is not None:
            self._close_owner_writer(writer)

    def _close_owner_writer(self, writer) -> None:
        try:
            writer.close()
        except Exception as exception:  # noqa: BLE001
            self.owner.report_failure("owner_stop_failed")
            logging.warning("Fast control: owner lease sluiten mislukt: %s", exception)

    def _shutdown_owner(self) -> None:
        self._shutting_down = True
        self._owner_start_pending = False
        if self.owner is None:
            return
        self._invalidate_current_execution()
        self.owner.binding = None
        self.owner.recovery_required = True
        self.owner.pending_stop_reason = "shutdown"
        try:
            now = self._cadence_now()
            cycle, allowed, next_start = owner_admission_at(now)
            admission = self.owner._admission
            if admission is not None and (cycle < admission.first_cycle or cycle <= admission.claimed_cycle):
                next_start = max(cycle + 1, admission.first_cycle) * OWNER_CYCLE_NS
            elif allowed:
                next_start = now
            if next_start > now:
                # A separate unset event: the already-set stop event would spin.
                threading.Event().wait((next_start - now) / 1e9)
            self.stop_owner_session("shutdown")
            self.last_fallback = FallbackReceipt(
                "acknowledged" if self.owner.phase == "released" and not self.owner.recovery_required else "unconfirmed",
                "shutdown")
        except Exception as exc:
            reason = getattr(exc, "reason", "owner_stop_failed")
            self.owner.report_failure(reason)
            self.last_fallback = FallbackReceipt("unconfirmed", reason)
        finally:
            if self._owner_writer is not None:
                writer, self._owner_writer = self._owner_writer, None
                self._close_owner_writer(writer)

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Run until *stop_event* is set. Never raises."""
        stop_event = stop_event or threading.Event()
        self._stop_event = stop_event
        period = max(5, int(self.config.interval))
        logging.info(f"Fast control gestart, interval {period} s")
        failures = 0
        if not stop_event.is_set():
            self.start_owner_session()
        while not stop_event.is_set():
            started = time.time()
            try:
                self.tick(started)
                failures = 0
            except Exception as exception:  # noqa: BLE001 - the loop must survive
                failures += 1
                if failures <= 3 or failures % 20 == 0:
                    logging.exception(
                        f"Fast control: fout in regellus ({failures}x): {exception}"
                    )
            # Back off after repeated failures so a broken Home Assistant does
            # not turn into a request storm, but cap at 2x period so the layer
            # recovers quickly once HA is responsive again.
            delay = period * min(2, 2 ** max(0, failures - 3)) if failures else period
            # Wait with an absolute upper bound equal to the delay so a stop()
            # call during a long backoff still returns promptly (otherwise we
            # could be stuck waiting up to 8x period before noticing).
            remaining = max(0.5, delay - (time.time() - started))
            if self._financial_regime(time.time()):
                try:
                    now_ns = self._cadence_now()
                    _, allowed, next_start = owner_admission_at(now_ns)
                    pending = self._owner_start_pending or (self.owner is not None
                               and self.owner.pending_stop_reason is not None)
                    deferred = self.last_gate is not None and self.last_gate.reason in (
                        "admission_deferred", "admission_window_missed")
                    target = now_ns if pending or deferred else now_ns + round(remaining * 1e9)
                    _, target_allowed, wake = owner_admission_at(target)
                    if target_allowed:
                        wake = target - target % OWNER_CYCLE_NS
                    if wake <= now_ns:
                        wake = (now_ns // OWNER_CYCLE_NS + 1) * OWNER_CYCLE_NS
                    if not allowed and (pending or deferred):
                        wake = next_start
                    remaining = (wake - now_ns) / 1e9
                except RecoveryRequired:
                    remaining = delay
            if stop_event.wait(remaining):
                break
        # On shutdown, dump any in-flight measurement integration into the
        # values table even if its interval is not yet fully covered. The
        # alternative is losing it on every restart, which biases the
        # forecast error table.
        try:
            plan = self.plan()
        except Exception:  # noqa: BLE001
            plan = None
        try:
            if plan is not None and self._measure_seconds > 0:
                self._flush_measurement(plan)
            # Flush whatever the throttled writer was still holding back.
            self._persist_state(time.time(), force=True)
        finally:
            self._shutdown_owner()
        logging.info("Fast control gestopt")


class FastControlThread(threading.Thread):
    """Runs a :class:`FastControlRunner` alongside the scheduler."""

    def __init__(self, hass, **kwargs):
        super().__init__(name="dao-fast-control", daemon=True)
        self.stop_event = threading.Event()
        self.runner = FastControlRunner(hass, **kwargs)

    def run(self) -> None:
        self.runner.run(self.stop_event)

    def stop(self, timeout: float = 35.0) -> None:
        self.stop_event.set()
        # The run() loop is interrupted as soon as stop_event becomes set
        # via the bounded wait, but join() still needs enough time for the
        # thread to finish its current tick (typically <1s) and run the
        # final force-flush. 35s covers the worst case (period=15s +
        # final-tick time) with a comfortable margin; previous default of
        # 5s would often cut off the flush and leave fast_state.json stale.
        self.join(timeout=timeout)


def start_if_enabled(hass, **kwargs) -> Optional[FastControlThread]:
    """Start the fast control thread when the configuration asks for it.

    Returns None when the layer is off or not usable, after logging why.
    """
    config: FastControlConfig = getattr(hass.config, "fast_control", None)
    if config is None:
        return None
    configured_mode = str(getattr(config.mode, "value", config.mode)).lower()
    if configured_mode == MODE_OFF:
        logging.info("Fast control staat uit")
        return None
    if not config.grid_power.configured:
        logging.warning(
            "Fast control is ingeschakeld maar er is geen 'grid power' sensor "
            "geconfigureerd; de snelle laag start niet"
        )
        return None
    if not getattr(hass.config, "battery", None):
        logging.warning(
            "Fast control is ingeschakeld maar er is geen batterij geconfigureerd; "
            "de snelle laag start niet"
        )
        return None
    thread = FastControlThread(hass, **kwargs)
    thread.start()
    return thread
