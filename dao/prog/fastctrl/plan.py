"""The hand-off artefact between the day-ahead layer and the fast control layer.

After every successful optimisation ``DaCalc.calc_optimum`` serialises the
relevant slice of its solution to ``../data/fast_plan.json``. The fast control
loop reads that file instead of querying the database or the solver, which
keeps the realtime path free of any heavy dependency and makes the contract
between the two layers explicit and inspectable.

Everything in this module is plain data. The only I/O is reading and writing
that one JSON file, and the optional reconstruction of a historical plan from
the ``prognoses`` table for the backtest.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from dao.prog.settlement.types import TerminalValue

# Legacybouwers blijven expliciet v1; alleen de financiële adapter schrijft v2.
PLAN_FORMAT_VERSION = 1

FAST_PLAN_FILE = "../data/fast_plan.json"

# Below this the battery is treated as idle. Mirrors the deadband the optimizer
# itself applies when it publishes the setpoint (day_ahead.py).
IDLE_POWER_W = 20.0

# What the optimizer writes to the "stop inverter" helper when the inverter must
# keep running for the whole interval.
NO_STOP_SENTINEL = "2000-01-01 00:00:00"


def _integer(value, name, *, positive=False):
    if type(value) is not int or not -(2**63) <= value < 2**63 or (positive and value <= 0):
        raise ValueError(f"{name}: expected {'positive ' if positive else ''}integer")
    return value


def _text(value, name, *, nonempty=False):
    if type(value) is not str or (nonempty and not value.strip()):
        raise ValueError(f"{name}: expected {'nonempty ' if nonempty else ''}text")
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: expected a finite number, not coercion/default")
    return value


def _object(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("Expected complete object with known fields")
    return value


def _array(value):
    if type(value) is not list:
        raise ValueError("Expected a JSON array")
    return value


def _decimal_text(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("Expected finite Decimal")
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _read_decimal(value):
    if type(value) is not str:
        raise ValueError("Financial decimals require canonical text")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal text") from exc
    if _decimal_text(result) != value:
        raise ValueError("Noncanonical decimal text")
    return result


@dataclass(frozen=True, slots=True)
class FinancialBasis:
    meter_id: str
    input_revision: int
    snapshot_hashes: tuple[str, ...]
    prefix_start_ts: int
    planning_start_ts: int
    horizon_end_ts: int
    captured_ts: int
    source_watermark_ts: int | None

    def __post_init__(self):
        _text(self.meter_id, "meter_id", nonempty=True)
        if len(self.meter_id) > 128:
            raise ValueError("meter_id: at most 128 characters")
        _integer(self.input_revision, "input_revision", positive=True)
        if type(self.snapshot_hashes) not in (tuple, list):
            raise ValueError("Expected pinned snapshot hashes")
        hashes = tuple(self.snapshot_hashes)
        if (not hashes or any(type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value) for value in hashes)
                or len(hashes) != len(set(hashes))):
            raise ValueError("Expected unique canonical SHA-256 snapshot hashes")
        object.__setattr__(self, "snapshot_hashes", hashes)
        for name in ("prefix_start_ts", "planning_start_ts", "horizon_end_ts", "captured_ts"):
            _integer(getattr(self, name), name)
        if self.source_watermark_ts is not None:
            _integer(self.source_watermark_ts, "source_watermark_ts")
        if not self.prefix_start_ts <= self.planning_start_ts < self.horizon_end_ts:
            raise ValueError("Expected prefix <= planning start < horizon end")


@dataclass(frozen=True, slots=True)
class FinancialContext:
    basis: FinancialBasis
    not_after_ts: int
    price_interval_s: int
    terminal_value: TerminalValue
    economics_supported: bool
    issues: tuple[str, ...]
    replay_basis_hash: str

    def __post_init__(self):
        if type(self.basis) is not FinancialBasis or type(self.terminal_value) is not TerminalValue:
            raise TypeError("Explicit financial basis and existing TerminalValue DTO required")
        _integer(self.not_after_ts, "not_after_ts")
        if not self.basis.planning_start_ts < self.not_after_ts <= self.basis.horizon_end_ts:
            raise ValueError("Lease must end inside the planning horizon")
        if type(self.price_interval_s) is not int or self.price_interval_s not in (900, 3600):
            raise ValueError("Unsupported price interval")
        if type(self.economics_supported) is not bool:
            raise TypeError("economics_supported requires bool")
        if type(self.replay_basis_hash) is not str or not re.fullmatch("[0-9a-f]{64}", self.replay_basis_hash):
            raise ValueError("replay_basis_hash: expected canonical SHA-256")
        if type(self.issues) not in (tuple, list):
            raise TypeError("issues requires a sequence")
        object.__setattr__(self, "issues", tuple(_text(value, "issue", nonempty=True) for value in self.issues))
        _text(self.terminal_value.policy_version, "terminal policy", nonempty=True)
        names = [name for name, _ in self.terminal_value.reference_efficiency_by_battery]
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError("Terminal reference battery names must be unique and explicit")

    def to_dict(self):
        return {
            "basis": {**asdict(self.basis), "snapshot_hashes": list(self.basis.snapshot_hashes)},
            "not_after_ts": self.not_after_ts, "price_interval_s": self.price_interval_s,
            "terminal_value": {
                "policy_version": self.terminal_value.policy_version,
                "eur_per_ac_kwh": _decimal_text(self.terminal_value.eur_per_ac_kwh),
                "reference_efficiency_by_battery": [
                    [name, _decimal_text(value)] for name, value in self.terminal_value.reference_efficiency_by_battery
                ],
            },
            "economics_supported": self.economics_supported, "issues": list(self.issues),
            "replay_basis_hash": self.replay_basis_hash,
        }

    @classmethod
    def from_dict(cls, data):
        if type(data) is dict and "replay_basis_hash" not in data:
            raise ValueError("Missing required replay_basis_hash in strict v2")
        _object(data, ("basis", "not_after_ts", "price_interval_s", "terminal_value", "economics_supported", "issues",
                       "replay_basis_hash"))
        basis = _object(data["basis"], FinancialBasis.__dataclass_fields__)
        _array(basis["snapshot_hashes"])
        terminal = _object(data["terminal_value"], TerminalValue.__dataclass_fields__)
        references = []
        for pair in _array(terminal["reference_efficiency_by_battery"]):
            if len(_array(pair)) != 2:
                raise ValueError("Expected battery name and reference efficiency")
            references.append((_text(pair[0], "battery", nonempty=True), _read_decimal(pair[1])))
        return cls(FinancialBasis(**basis), data["not_after_ts"], data["price_interval_s"],
                   TerminalValue(terminal["policy_version"], _read_decimal(terminal["eur_per_ac_kwh"]),
                                  tuple(references)), data["economics_supported"], _array(data["issues"]),
                   data["replay_basis_hash"])


@dataclass(frozen=True, slots=True)
class BlockedPlan:
    format_version: int
    created_ts: int
    reason: str

    def __post_init__(self):
        if type(self.format_version) is not int or self.format_version != 2:
            raise ValueError("Blocked plans require format_version=2")
        _integer(self.created_ts, "created_ts")
        _text(self.reason, "reason", nonempty=True)

    def to_dict(self):
        return {"format_version": 2, "status": "blocked", "created_ts": self.created_ts, "reason": self.reason}

    @classmethod
    def from_dict(cls, data):
        _object(data, ("format_version", "status", "created_ts", "reason"))
        if data["status"] != "blocked":
            raise ValueError("Expected blocked status")
        return cls(data["format_version"], data["created_ts"], data["reason"])


def _f(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion that never raises."""
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN
        return default
    return result


@dataclass
class BatterySpec:
    """Static properties of one battery, copied from the configuration.

    The fast layer needs these to clamp its corrections. They are snapshotted
    into the plan so the realtime loop never has to re-read and re-validate the
    configuration while it runs.
    """

    name: str
    capacity_kwh: float
    max_charge_w: float
    max_discharge_w: float
    minimum_power_w: float = 0.0
    soc_min: float = 20.0
    soc_max: float = 100.0
    cycle_cost: float = 0.0
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    setpoint_entity: Optional[str] = None
    mode_entity: Optional[str] = None
    mode_on: str = "Aan"
    mode_off: str = "Uit"
    stop_inverter_entity: Optional[str] = None
    soc_entity: Optional[str] = None

    @property
    def round_trip_efficiency(self) -> float:
        """DC round trip efficiency implied by the two configured efficiencies."""
        return max(0.01, self.charge_efficiency * self.discharge_efficiency)

    @property
    def kwh_per_percent(self) -> float:
        """Energy content of one percent of state of charge."""
        return self.capacity_kwh / 100.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "capacity_kwh": self.capacity_kwh,
            "max_charge_w": self.max_charge_w,
            "max_discharge_w": self.max_discharge_w,
            "minimum_power_w": self.minimum_power_w,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "cycle_cost": self.cycle_cost,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency,
            "setpoint_entity": self.setpoint_entity,
            "mode_entity": self.mode_entity,
            "mode_on": self.mode_on,
            "mode_off": self.mode_off,
            "stop_inverter_entity": self.stop_inverter_entity,
            "soc_entity": self.soc_entity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BatterySpec:
        return cls(
            name=str(data.get("name", "battery")),
            capacity_kwh=_f(data.get("capacity_kwh"), 1.0),
            max_charge_w=_f(data.get("max_charge_w")),
            max_discharge_w=_f(data.get("max_discharge_w")),
            minimum_power_w=_f(data.get("minimum_power_w")),
            soc_min=_f(data.get("soc_min"), 20.0),
            soc_max=_f(data.get("soc_max"), 100.0),
            cycle_cost=_f(data.get("cycle_cost")),
            charge_efficiency=_f(data.get("charge_efficiency"), 1.0),
            discharge_efficiency=_f(data.get("discharge_efficiency"), 1.0),
            setpoint_entity=data.get("setpoint_entity"),
            mode_entity=data.get("mode_entity"),
            mode_on=str(data.get("mode_on", "Aan")),
            mode_off=str(data.get("mode_off", "Uit")),
            stop_inverter_entity=data.get("stop_inverter_entity"),
            soc_entity=data.get("soc_entity"),
        )


@dataclass
class BatteryPlanStep:
    """What the plan intends one battery to do during one interval."""

    #: Net AC power, positive is charging from AC, negative is discharging.
    ac_power_w: float = 0.0
    #: State of charge in percent at the start and at the end of the interval.
    soc_begin: float = 0.0
    soc_end: float = 0.0
    #: Operating mode string the optimizer published, only set for the interval
    #: that was actually actuated.
    mode: Optional[str] = None
    #: Inverter stop moment the optimizer published, ``"%Y-%m-%d %H:%M"`` or the
    #: sentinel the optimizer uses to mean "no stop".
    stop_inverter: Optional[str] = None

    def to_dict(self) -> dict:
        data: dict[str, Any] = {
            "ac_power_w": round(self.ac_power_w, 1),
            "soc_begin": round(self.soc_begin, 2),
            "soc_end": round(self.soc_end, 2),
        }
        if self.mode is not None:
            data["mode"] = self.mode
        if self.stop_inverter is not None:
            data["stop_inverter"] = self.stop_inverter
        return data

    @classmethod
    def from_dict(cls, data: dict) -> BatteryPlanStep:
        return cls(
            ac_power_w=_f(data.get("ac_power_w")),
            soc_begin=_f(data.get("soc_begin")),
            soc_end=_f(data.get("soc_end")),
            mode=data.get("mode"),
            stop_inverter=data.get("stop_inverter"),
        )


@dataclass
class PlanInterval:
    """One row of the day-ahead plan."""

    start_ts: int
    end_ts: int
    price_import: float
    price_export: float
    #: Planned net grid power, positive is import.
    grid_w: float = 0.0
    #: Planned house load excluding batteries, positive is consumption.
    house_w: float = 0.0
    #: Planned AC-coupled PV production.
    pv_w: float = 0.0
    batteries: list[BatteryPlanStep] = field(default_factory=list)

    @property
    def duration_s(self) -> int:
        return max(1, self.end_ts - self.start_ts)

    def contains(self, ts: float) -> bool:
        return self.start_ts <= ts < self.end_ts

    def battery(self, index: int) -> BatteryPlanStep:
        if 0 <= index < len(self.batteries):
            return self.batteries[index]
        return BatteryPlanStep()

    @property
    def plan_battery_w(self) -> float:
        """Total planned battery AC power over all batteries."""
        return sum(b.ac_power_w for b in self.batteries)

    def to_dict(self) -> dict:
        return {
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "price_import": round(self.price_import, 6),
            "price_export": round(self.price_export, 6),
            "grid_w": round(self.grid_w, 1),
            "house_w": round(self.house_w, 1),
            "pv_w": round(self.pv_w, 1),
            "batteries": [b.to_dict() for b in self.batteries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanInterval:
        return cls(
            start_ts=int(data.get("start_ts", 0)),
            end_ts=int(data.get("end_ts", 0)),
            price_import=_f(data.get("price_import")),
            price_export=_f(data.get("price_export")),
            grid_w=_f(data.get("grid_w")),
            house_w=_f(data.get("house_w")),
            pv_w=_f(data.get("pv_w")),
            batteries=[
                BatteryPlanStep.from_dict(b) for b in data.get("batteries", []) or []
            ],
        )


@dataclass
class FastPlan:
    """The complete artefact handed from the optimizer to the fast layer."""

    created_ts: int
    interval_s: int
    specs: list[BatterySpec] = field(default_factory=list)
    intervals: list[PlanInterval] = field(default_factory=list)
    strategy: str = ""
    price_average: float = 0.0
    format_version: int = PLAN_FORMAT_VERSION
    financial_context: FinancialContext | None = None

    # -- lookups ---------------------------------------------------------

    def interval_at(self, ts: float) -> Optional[PlanInterval]:
        """The plan interval covering *ts*, or None when *ts* is outside the plan."""
        for interval in self.intervals:
            if interval.contains(ts):
                return interval
        return None

    def index_at(self, ts: float) -> Optional[int]:
        for index, interval in enumerate(self.intervals):
            if interval.contains(ts):
                return index
        return None

    def remaining(self, ts: float) -> list[PlanInterval]:
        """All intervals that end after *ts*, including the current one."""
        return [i for i in self.intervals if i.end_ts > ts]

    def age(self, now_ts: float) -> float:
        """Seconds since the plan was written."""
        return max(0.0, now_ts - self.created_ts)

    def spec(self, index: int) -> Optional[BatterySpec]:
        if 0 <= index < len(self.specs):
            return self.specs[index]
        return None

    def spec_index(self, name: str) -> Optional[int]:
        for index, spec in enumerate(self.specs):
            if spec.name == name:
                return index
        return None

    @property
    def horizon_end_ts(self) -> int:
        return self.intervals[-1].end_ts if self.intervals else self.created_ts

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        if type(self.format_version) is not int:
            raise ValueError("Plan format version requires int without coercion")
        if self.format_version == 2:
            if type(self.financial_context) is not FinancialContext:
                raise ValueError("Ready-v2 requires a complete financial context")
            data = {
                "format_version": 2, "status": "ready", "created_ts": self.created_ts,
                "interval_s": self.interval_s, "strategy": self.strategy, "price_average": self.price_average,
                "batteries": [asdict(spec) for spec in self.specs],
                "intervals": [asdict(interval) for interval in self.intervals],
                "financial_context": self.financial_context.to_dict(),
            }
            # Ook in-memory kandidaten gaan door dezelfde strikte wirevalidatie.
            self.from_dict(data)
            return data
        if type(self.format_version) is not int or self.format_version != 1 or self.financial_context is not None:
            raise ValueError("Legacy plans require explicit v1 without financial context")
        return {
            "format_version": self.format_version,
            "created_ts": self.created_ts,
            "interval_s": self.interval_s,
            "strategy": self.strategy,
            "price_average": round(self.price_average, 6),
            "batteries": [s.to_dict() for s in self.specs],
            "intervals": [i.to_dict() for i in self.intervals],
        }

    @classmethod
    def from_dict(cls, data: dict) -> FastPlan:
        if type(data) is not dict or type(data.get("format_version")) is not int:
            raise ValueError("Expected versioned plan object")
        version = data["format_version"]
        if version == 2:
            return _read_ready_v2(data)
        if version != 1 or "status" in data or "financial_context" in data:
            raise ValueError("Unknown plan format")
        return cls(
            created_ts=int(data.get("created_ts", 0)),
            interval_s=int(data.get("interval_s", 3600)),
            specs=[BatterySpec.from_dict(s) for s in data.get("batteries", []) or []],
            intervals=[
                PlanInterval.from_dict(i) for i in data.get("intervals", []) or []
            ],
            strategy=str(data.get("strategy", "")),
            price_average=_f(data.get("price_average")),
            format_version=version,
        )


def _read_ready_v2(data):
    _object(data, ("format_version", "status", "created_ts", "interval_s", "strategy", "price_average",
                   "batteries", "intervals", "financial_context"))
    if data["status"] != "ready":
        raise ValueError("Expected ready status")
    created = _integer(data["created_ts"], "created_ts")
    step_s = _integer(data["interval_s"], "interval_s", positive=True)
    context = FinancialContext.from_dict(data["financial_context"])
    if not context.basis.captured_ts <= created < context.not_after_ts:
        raise ValueError("Capture/creation/lease ordering is invalid")
    specs = []
    optional_text = ("setpoint_entity", "mode_entity", "stop_inverter_entity", "soc_entity")
    spec_text = ("name", "mode_on", "mode_off")
    for spec in _array(data["batteries"]):
        _object(spec, BatterySpec.__dataclass_fields__)
        for name in spec_text:
            _text(spec[name], name, nonempty=True)
        for name in optional_text:
            if spec[name] is not None:
                _text(spec[name], name, nonempty=True)
        for name in spec.keys() - set(spec_text) - set(optional_text):
            _number(spec[name], name)
        if (spec["capacity_kwh"] <= 0 or min(spec["max_charge_w"], spec["max_discharge_w"],
                                            spec["minimum_power_w"], spec["cycle_cost"]) < 0
                or not 0 <= spec["soc_min"] < spec["soc_max"] <= 100
                or not 0 < spec["charge_efficiency"] <= 1 or not 0 < spec["discharge_efficiency"] <= 1):
            raise ValueError("Invalid battery limits/efficiencies")
        specs.append(BatterySpec(**spec))
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)) or set(names) != {
        name for name, _ in context.terminal_value.reference_efficiency_by_battery
    }:
        raise ValueError("Terminal references must identify exactly the plan batteries")
    intervals, cursor = [], context.basis.planning_start_ts
    for interval in _array(data["intervals"]):
        _object(interval, PlanInterval.__dataclass_fields__)
        start = _integer(interval["start_ts"], "start_ts")
        end = _integer(interval["end_ts"], "end_ts")
        if start != cursor or not 0 < end - start <= step_s:
            raise ValueError("Plan must cover the horizon contiguously without overlap")
        for name in ("price_import", "price_export", "grid_w", "house_w", "pv_w"):
            _number(interval[name], name)
        steps = []
        for step in _array(interval["batteries"]):
            _object(step, ("ac_power_w", "soc_begin", "soc_end"), ("mode", "stop_inverter"))
            for name in ("ac_power_w", "soc_begin", "soc_end"):
                _number(step[name], name)
            if not 0 <= step["soc_begin"] <= 100 or not 0 <= step["soc_end"] <= 100:
                raise ValueError("Invalid planned SOC")
            for name in ("mode", "stop_inverter"):
                if step.get(name) is not None:
                    _text(step[name], name, nonempty=True)
            steps.append(BatteryPlanStep(**step))
        if len(steps) != len(specs):
            raise ValueError("Every interval must specify exactly the declared batteries")
        intervals.append(PlanInterval(**{**interval, "batteries": steps}))
        cursor = end
    if not intervals or cursor != context.basis.horizon_end_ts:
        raise ValueError("Incomplete planning horizon")
    return FastPlan(created, step_s, specs, intervals, _text(data["strategy"], "strategy"),
                    _number(data["price_average"], "price_average"), 2, context)


def write_plan(plan: FastPlan, path: str | Path = FAST_PLAN_FILE) -> None:
    """Legacycompatibiliteit; mag de financiële publicatiegrens nooit herstellen."""
    from dao.prog.fastctrl.publication import (
        PublicationError,
        canonical_plan_path,
        publication_lock,
        write_publication,
    )

    plan_path = canonical_plan_path(path)
    try:
        # Behoud de bestaande v1-eerste-schrijfroute, alleen voor dit vertrouwde pad.
        plan_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise PublicationError("Cannot create legacy plan data directory") from exc
    with publication_lock(plan_path) as lease:
        write_publication(plan, lease=lease)


def load_plan(path: str | Path = FAST_PLAN_FILE) -> Optional[FastPlan]:
    """Compatibilityread; veiligheidconsumers gebruiken read/verdict, niet deze wrapper."""
    from dao.prog.fastctrl.publication import read_publication

    read = read_publication(path)
    return read.plan if read.status == "ready" else None


def build_intervals(
    timestamps: Iterable[int],
    interval_s: int,
    price_import: list[float],
    price_export: list[float],
    grid_w: list[float],
    house_w: list[float],
    pv_w: list[float],
    battery_steps: list[list[BatteryPlanStep]],
) -> list[PlanInterval]:
    """Assemble plan intervals from column-wise series of equal length.

    ``battery_steps`` is indexed ``[battery][interval]``, matching the way the
    optimizer holds its solution.
    """
    intervals: list[PlanInterval] = []
    stamps = list(timestamps)
    for index, start_ts in enumerate(stamps):
        end_ts = (
            stamps[index + 1] if index + 1 < len(stamps) else start_ts + interval_s
        )
        intervals.append(
            PlanInterval(
                start_ts=int(start_ts),
                end_ts=int(end_ts),
                price_import=price_import[index],
                price_export=price_export[index],
                grid_w=grid_w[index],
                house_w=house_w[index],
                pv_w=pv_w[index],
                batteries=[steps[index] for steps in battery_steps],
            )
        )
    return intervals


def build_plan(
    created_ts: int,
    interval_s: int,
    specs: list[BatterySpec],
    start_ts: list[int],
    hour_fraction: list[float],
    price_import: list[float],
    price_export: list[float],
    grid_kwh: list[float],
    pv_kwh: list[float],
    battery_kw: list[list[float]],
    soc: list[list[float]],
    published: Optional[list[dict]] = None,
    strategy: str = "",
    price_average: float = 0.0,
) -> FastPlan:
    """Turn the optimizer's solution into a plan artefact.

    Kept free of solver objects so it can be exercised without the MIP stack.
    Energy quantities arrive per interval in kWh and are converted to average
    power in W; ``battery_kw`` and ``soc`` are indexed ``[battery][interval]``,
    with ``soc`` carrying one extra entry for the end of the horizon.

    ``published`` optionally holds, per battery, the command the optimizer
    actually sent for the first interval, as
    ``{"power_w": float, "mode": str, "stop_inverter": str}``. Handing that to
    the fast layer lets it restore the inverter exactly when it releases an
    override, instead of approximating it from the raw solver value.

    Note the asymmetry between ``start_ts`` and ``hour_fraction``. The
    optimizer starts partway through the current interval, so its first row
    carries the interval boundary as its timestamp but only the *remaining*
    fraction of an hour as its duration. Intervals are therefore laid out on
    whole ``interval_s`` boundaries, which keeps the timeline contiguous and
    keeps ``interval_at(now)`` working from the moment the plan is written,
    while ``hour_fraction`` is used purely to turn energy into average power.
    """
    published = published or []
    battery_count = len(specs)
    intervals: list[PlanInterval] = []

    for index, stamp in enumerate(start_ts):
        fraction = hour_fraction[index] or 1.0
        steps: list[BatteryPlanStep] = []
        for b in range(battery_count):
            if index == 0 and b < len(published):
                steps.append(
                    BatteryPlanStep(
                        ac_power_w=_f(published[b].get("power_w")),
                        soc_begin=_f(soc[b][index]),
                        soc_end=_f(soc[b][index + 1]),
                        mode=published[b].get("mode"),
                        stop_inverter=published[b].get("stop_inverter"),
                    )
                )
            else:
                steps.append(
                    BatteryPlanStep(
                        ac_power_w=1000.0 * _f(battery_kw[b][index]),
                        soc_begin=_f(soc[b][index]),
                        soc_end=_f(soc[b][index + 1]),
                    )
                )
        # grid = house + battery, so the planned house load follows from the
        # planned grid exchange without having to re-add every single device.
        grid_w = 1000.0 * grid_kwh[index] / fraction
        battery_w = sum(step.ac_power_w for step in steps)
        intervals.append(
            PlanInterval(
                start_ts=int(stamp),
                end_ts=int(stamp + interval_s),
                price_import=_f(price_import[index]),
                price_export=_f(price_export[index]),
                grid_w=grid_w,
                house_w=grid_w - battery_w,
                pv_w=1000.0 * _f(pv_kwh[index]) / fraction,
                batteries=steps,
            )
        )

    return FastPlan(
        created_ts=created_ts,
        interval_s=interval_s,
        specs=specs,
        intervals=intervals,
        strategy=strategy,
        price_average=price_average,
    )
