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

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# Bump when the on-disk layout changes in a way older readers cannot handle.
PLAN_FORMAT_VERSION = 1

FAST_PLAN_FILE = "../data/fast_plan.json"

# Below this the battery is treated as idle. Mirrors the deadband the optimizer
# itself applies when it publishes the setpoint (day_ahead.py).
IDLE_POWER_W = 20.0

# What the optimizer writes to the "stop inverter" helper when the inverter must
# keep running for the whole interval.
NO_STOP_SENTINEL = "2000-01-01 00:00:00"


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
    def from_dict(cls, data: dict) -> "BatterySpec":
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
    def from_dict(cls, data: dict) -> "BatteryPlanStep":
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
    def from_dict(cls, data: dict) -> "PlanInterval":
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
    def from_dict(cls, data: dict) -> "FastPlan":
        version = int(data.get("format_version", 0))
        if version > PLAN_FORMAT_VERSION:
            raise ValueError(
                f"fast_plan.json has format version {version}, this build understands "
                f"at most {PLAN_FORMAT_VERSION}"
            )
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


def write_plan(plan: FastPlan, path: str = FAST_PLAN_FILE) -> None:
    """Write the plan atomically so a concurrent reader never sees a partial file."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", dir=directory, prefix=".fast_plan_", suffix=".tmp", delete=False
    )
    try:
        with handle:
            json.dump(plan.to_dict(), handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    logging.debug(
        f"Fast-control plan geschreven: {path}, {len(plan.intervals)} intervallen"
    )


def load_plan(path: str = FAST_PLAN_FILE) -> Optional[FastPlan]:
    """Read the plan, returning None when it is missing or unreadable.

    A missing or corrupt plan is not fatal: the fast layer simply refuses to
    override and leaves the battery on whatever the optimizer last wrote.
    """
    try:
        with open(path, "r") as handle:
            return FastPlan.from_dict(json.load(handle))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, ValueError, TypeError, OSError) as exception:
        logging.warning(f"Fast-control plan {path} kon niet worden gelezen: {exception}")
        return None


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
