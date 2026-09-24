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

import pytz
from typing import Any, Callable, Iterable, Optional

from dao.prog.config.models.fastcontrol import (
    FastBatteryLink,
    FastControlConfig,
    PowerSensor,
)

from .plan import FAST_PLAN_FILE, FastPlan, load_plan
from .policy import (
    BatteryMeasurement,
    ControllerState,
    Decision,
    FastControlPolicy,
    Measurement,
    PolicyLimits,
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
NO_STOP_SENTINEL = "2000-01-01 00:00:00"

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ACTIVE = "active"

EVENTS_KIND_MODE = "mode_change"
EVENTS_KIND_OVERRIDE_START = "override_start"
EVENTS_KIND_OVERRIDE_END = "override_end"
EVENTS_KIND_SETPOINT = "setpoint_change"

EVENTS_LIMIT = 200


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


class HomeAssistantGateway:
    """Batched reads and guarded writes against the Home Assistant REST API.

    A single templated render fetches every entity the control loop needs, which
    keeps one tick at one HTTP round trip regardless of how many sensors are
    configured. When the template endpoint is unavailable the gateway falls back
    to individual state reads and remembers that for the rest of the run.
    """

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
    ):
        self.hass = hass
        self.config: FastControlConfig = config or getattr(
            hass.config, "fast_control", FastControlConfig()
        )
        self.plan_path = plan_path
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
        except Exception:  # noqa: BLE001
            return None

    def _flex(self, flex, default):
        if flex is None:
            return default
        try:
            value = flex.resolve(self._getter)
        except Exception as exception:  # noqa: BLE001
            logging.debug(f"Fast control: waarde niet oplosbaar ({exception})")
            return default
        return default if value is None else value

    def mode(self) -> str:
        """Current operating mode, resolved live so it can be switched from HA."""
        raw = str(self._flex(self.config.mode, MODE_OFF)).strip().lower()
        if raw in {"on", "true", "1", "active"}:
            return MODE_ACTIVE
        if raw in {"shadow", "dry-run", "dryrun", "log"}:
            return MODE_SHADOW
        return MODE_OFF

    def _initial_mode(self) -> str:
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
                self._tz_info = pytz.timezone(self._time_zone)
            except Exception:  # noqa: BLE001 - fall back to the system clock
                self._time_zone = None
        return datetime.datetime.fromtimestamp(now, self._tz_info).strftime("%Y-%m-%d")

    # -- one iteration ---------------------------------------------------

    def tick(self, now: Optional[float] = None) -> Optional[Decision]:
        """Run one control cycle. Returns the decision, or None when idle."""
        now = time.time() if now is None else now

        plan = self.plan()
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

        measurement = Measurement(
            timestamp=now,
            grid_w=grid_w if grid_w is not None else 0.0,
            batteries=batteries,
            pv_w=pv_w,
            grid_valid=grid_w is not None and grid_fresh and not plan_stale,
        )

        policy = FastControlPolicy(self.limits())
        policy.account(
            self.state,
            plan,
            measurement,
            self._day_key(now),
            use_measured=mode == MODE_ACTIVE,
        )
        decision = policy.decide(plan, measurement, self.state, enabled)

        # decision.house_w is the reconstructed site demand excluding the
        # battery, which is exactly the quantity the optimizer forecasts as
        # "hload". Recording it turns the control loop into the measurement
        # the forecast side has been missing.
        self._measure(now, plan, decision.house_w, pv_w, measurement.grid_valid)

        elapsed_h = (
            min(now - self.state.last_tick_ts, 600.0) / 3600.0
            if self.state.last_tick_ts
            else 0.0
        )
        self.state.saved_today_eur += decision.benefit_eur_h * elapsed_h

        self._actuate(decision, plan, mode)
        self._publish(decision, mode, measurement)
        self._record_events(decision, mode)
        self.state.refresh_budget_aggregates()
        self._persist_state(now)
        return decision

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
            import pandas as pd

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

    def _actuate(self, decision: Decision, plan: FastPlan, mode: str) -> None:
        for battery in decision.batteries:
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
                continue
            logging.info(message)
            if spec.setpoint_entity:
                self.gateway.write_number(spec.setpoint_entity, battery.setpoint_w)
            if battery.mode is not None and spec.mode_entity:
                self.gateway.select_option(spec.mode_entity, battery.mode)
            if battery.stop_inverter is not None and spec.stop_inverter_entity:
                self.gateway.set_datetime(
                    spec.stop_inverter_entity,
                    battery.stop_inverter or NO_STOP_SENTINEL,
                )

    def _publish(self, decision: Decision, mode: str, measurement: Measurement) -> None:
        diagnostics = self.config.diagnostics
        attributes = decision.as_attributes()
        attributes["mode"] = mode
        attributes["pv_w"] = (
            round(measurement.pv_w) if measurement.pv_w is not None else None
        )
        attributes["saved_today_eur"] = round(self.state.saved_today_eur, 3)
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

    def _record_events(self, decision, mode: str) -> None:
        new_overrides = tuple(b.override for b in decision.batteries)
        new_setpoints = tuple(b.setpoint_w for b in decision.batteries)
        events = self.state.events

        if mode != self._last_mode:
            events.append(self._event_dict(decision, mode, EVENTS_KIND_MODE))
        if self._last_overrides and any(self._last_overrides) and not any(new_overrides):
            events.append(self._event_dict(decision, mode, EVENTS_KIND_OVERRIDE_END))
        elif (not self._last_overrides or not any(self._last_overrides)) and any(new_overrides):
            events.append(self._event_dict(decision, mode, EVENTS_KIND_OVERRIDE_START))

        if self._last_setpoints and any(
            abs(new - old) >= 1.0 for new, old in zip(new_setpoints, self._last_setpoints)
        ):
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

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Run until *stop_event* is set. Never raises."""
        stop_event = stop_event or threading.Event()
        period = max(5, int(self.config.interval))
        logging.info(f"Fast control gestart, interval {period} s")
        failures = 0
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
            # not turn into a request storm.
            delay = period * min(8, 2 ** max(0, failures - 3)) if failures else period
            stop_event.wait(max(1.0, delay - (time.time() - started)))
        # Flush whatever the throttled writer was still holding back.
        self._persist_state(time.time(), force=True)
        logging.info("Fast control gestopt")


class FastControlThread(threading.Thread):
    """Runs a :class:`FastControlRunner` alongside the scheduler."""

    def __init__(self, hass, **kwargs):
        super().__init__(name="dao-fast-control", daemon=True)
        self.stop_event = threading.Event()
        self.runner = FastControlRunner(hass, **kwargs)

    def run(self) -> None:
        self.runner.run(self.stop_event)

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
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
