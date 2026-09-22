"""Backtest the fast control layer on historical data.

Replays a measured house load against a historical day-ahead plan twice: once
with the plan setpoint held constant across each interval, which is what the
optimizer does today, and once through the production
:class:`~dao.prog.fastctrl.policy.FastControlPolicy`. Because both runs share
the same battery model and the same measurements, the difference is a clean
estimate of what the fast layer is worth on your own house.

Two things make the comparison honest:

* The terminal state of charge difference is priced. Without it the fast run
  could "win" simply by ending the window with an emptier battery.
* Battery wear is priced with the same ``cycle cost`` the optimizer uses, so
  extra cycling shows up as a cost, not as free money.

The historical data comes from the Home Assistant recorder. Note that its
default retention is ten days and that long term statistics are hourly, which
is far too coarse to show peak shaving. Run the backtest on the highest
resolution you have and read the resolution warning the loader prints.
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .plan import BatterySpec, BatteryPlanStep, FastPlan, PlanInterval
from .policy import (
    BatteryMeasurement,
    ControllerState,
    FastControlPolicy,
    Measurement,
    PolicyLimits,
)

#: Default AC/DC conversion efficiency used by the simulation battery model when
#: the plan does not carry the inverter's own stage curve.
DEFAULT_INVERTER_EFFICIENCY = 0.95

#: Days of recorder history read in one go. Keeps the peak memory of a backtest
#: at one chunk of one entity instead of the whole window, which matters on the
#: 4 GB of a Home Assistant Yellow.
CHUNK_DAYS = 2


def _iter_chunks(
    start: datetime.datetime, end: datetime.datetime, days: int
) -> "list[tuple[datetime.datetime, datetime.datetime]]":
    """Split ``[start, end)`` into windows of at most *days* days."""
    span = datetime.timedelta(days=max(1, days))
    chunks = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + span, end)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


@dataclass
class Sample:
    """One measured moment of the house, excluding the battery."""

    timestamp: float
    house_w: float


@dataclass
class BatteryModel:
    """A simple, symmetric battery used to replay both scenarios identically."""

    spec: BatterySpec
    soc: float
    inverter_efficiency: float = DEFAULT_INVERTER_EFFICIENCY

    def step(self, command_w: float, dt_h: float) -> tuple[float, float]:
        """Apply *command_w* for *dt_h* hours.

        Returns the AC power that was actually realised and the DC throughput
        in kWh, both of which the caller needs for the cost accounting.
        """
        spec = self.spec
        power = max(-spec.max_discharge_w, min(spec.max_charge_w, command_w))
        if dt_h <= 0.0:
            return 0.0, 0.0

        if power >= 0.0:
            dc_kwh = power * dt_h * self.inverter_efficiency / 1000.0
            stored_kwh = dc_kwh * spec.charge_efficiency
            room_kwh = max(0.0, (spec.soc_max - self.soc) * spec.kwh_per_percent)
            if stored_kwh > room_kwh:
                scale = room_kwh / stored_kwh if stored_kwh > 0 else 0.0
                power *= scale
                dc_kwh *= scale
                stored_kwh = room_kwh
            self.soc += stored_kwh / spec.kwh_per_percent
            return power, dc_kwh

        ac_kwh = -power * dt_h / 1000.0
        dc_kwh = ac_kwh / self.inverter_efficiency
        drawn_kwh = dc_kwh / spec.discharge_efficiency
        available_kwh = max(0.0, (self.soc - spec.soc_min) * spec.kwh_per_percent)
        if drawn_kwh > available_kwh:
            scale = available_kwh / drawn_kwh if drawn_kwh > 0 else 0.0
            power *= scale
            dc_kwh *= scale
            drawn_kwh = available_kwh
        self.soc -= drawn_kwh / spec.kwh_per_percent
        return power, dc_kwh


@dataclass
class DayMetrics:
    """Per day totals, so a saving can be traced back to a specific event."""

    day: str
    energy_cost: float = 0.0
    wear_cost: float = 0.0
    import_kwh: float = 0.0
    export_kwh: float = 0.0
    throughput_kwh: float = 0.0
    peak_import_w: float = 0.0
    writes: int = 0
    override_s: float = 0.0


@dataclass
class SimResult:
    """Outcome of one replay."""

    label: str
    energy_cost: float = 0.0
    wear_cost: float = 0.0
    import_kwh: float = 0.0
    export_kwh: float = 0.0
    throughput_kwh: float = 0.0
    peak_import_w: float = 0.0
    writes: int = 0
    override_s: float = 0.0
    soc_start: float = 0.0
    soc_end: float = 0.0
    terminal_value: float = 0.0
    capacity_kwh: float = 0.0
    duration_h: float = 0.0
    days: dict[str, DayMetrics] = field(default_factory=dict)
    soc_series: list[tuple[float, float]] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        """Energy plus wear, corrected for the energy left in the battery."""
        return self.energy_cost + self.wear_cost - self.terminal_value

    @property
    def cycles(self) -> float:
        """Equivalent full cycles, DC throughput over twice the capacity."""
        if self.capacity_kwh <= 0:
            return 0.0
        return self.throughput_kwh / (2.0 * self.capacity_kwh)

    @property
    def average_soc(self) -> float:
        if not self.soc_series:
            return 0.0
        return sum(s for _, s in self.soc_series) / len(self.soc_series)


@dataclass
class Comparison:
    """Baseline against fast control."""

    baseline: SimResult
    fast: SimResult

    @property
    def saving(self) -> float:
        return self.baseline.total_cost - self.fast.total_cost

    @property
    def saving_per_day(self) -> float:
        days = max(1.0, self.baseline.duration_h / 24.0)
        return self.saving / days

    @property
    def extra_cycles(self) -> float:
        return self.fast.cycles - self.baseline.cycles

    def report(self) -> str:
        """Human readable summary, ready for the log or the console."""
        base, fast = self.baseline, self.fast
        days = max(1.0, base.duration_h / 24.0)
        lines = [
            "",
            "=" * 72,
            f"Fast control backtest over {base.duration_h / 24.0:.1f} dagen",
            "=" * 72,
            f"{'':30}{'zonder':>12}{'met':>12}{'verschil':>14}",
            "-" * 72,
            _row("Energiekosten (euro)", base.energy_cost, fast.energy_cost),
            _row("Slijtagekosten (euro)", base.wear_cost, fast.wear_cost),
            _row("Restwaarde accu (euro)", -base.terminal_value, -fast.terminal_value),
            _row("Totale kosten (euro)", base.total_cost, fast.total_cost),
            "-" * 72,
            _row("Afname net (kWh)", base.import_kwh, fast.import_kwh),
            _row("Teruglevering (kWh)", base.export_kwh, fast.export_kwh),
            _row("Accu doorzet DC (kWh)", base.throughput_kwh, fast.throughput_kwh),
            _row("Volledige cycli", base.cycles, fast.cycles),
            _row("Piek afname (W)", base.peak_import_w, fast.peak_import_w),
            _row("Gemiddelde SoC (%)", base.average_soc, fast.average_soc),
            _row("Eind SoC (%)", base.soc_end, fast.soc_end),
            "-" * 72,
            f"{'Setpoint-schrijfacties':30}{base.writes:>12.0f}{fast.writes:>12.0f}",
            f"{'Override actief (uur)':30}"
            f"{base.override_s / 3600:>12.1f}{fast.override_s / 3600:>12.1f}",
            "=" * 72,
            f"Besparing: {self.saving:.2f} euro over {days:.1f} dagen "
            f"= {self.saving_per_day:.3f} euro/dag "
            f"= {self.saving_per_day * 365:.0f} euro/jaar",
            f"Extra cycli: {self.extra_cycles:+.2f} over de periode "
            f"({self.extra_cycles / days * 365:+.0f} per jaar)",
        ]
        if fast.throughput_kwh > base.throughput_kwh > 0:
            extra = fast.throughput_kwh - base.throughput_kwh
            lines.append(
                f"Besparing per extra kWh accudoorzet: "
                f"{self.saving / extra:.3f} euro/kWh"
            )
        lines.append("=" * 72)
        lines.append("")
        return "\n".join(lines)

    def daily_table(self) -> str:
        """Per day breakdown so outliers can be inspected."""
        keys = sorted(set(self.baseline.days) | set(self.fast.days))
        lines = [
            f"{'datum':12}{'zonder':>10}{'met':>10}{'besparing':>11}"
            f"{'cycli':>8}{'piek W':>9}{'writes':>8}"
        ]
        lines.append("-" * 68)
        for key in keys:
            base = self.baseline.days.get(key, DayMetrics(key))
            fast = self.fast.days.get(key, DayMetrics(key))
            base_cost = base.energy_cost + base.wear_cost
            fast_cost = fast.energy_cost + fast.wear_cost
            cycles = fast.throughput_kwh / (2.0 * self.fast.capacity_kwh) if (
                self.fast.capacity_kwh
            ) else 0.0
            lines.append(
                f"{key:12}{base_cost:>10.2f}{fast_cost:>10.2f}"
                f"{base_cost - fast_cost:>11.3f}{cycles:>8.2f}"
                f"{fast.peak_import_w:>9.0f}{fast.writes:>8.0f}"
            )
        return "\n".join(lines)


def _row(label: str, left: float, right: float) -> str:
    return f"{label:30}{left:>12.2f}{right:>12.2f}{right - left:>+14.3f}"


def aggregate_spec(specs: Sequence[BatterySpec], name: str = "totaal") -> BatterySpec:
    """Collapse several batteries into one for the simulation.

    The historical ``prognoses`` table only stores site totals, so the backtest
    necessarily works with an aggregated battery. Power limits and capacity add
    up, efficiencies and cycle cost are capacity weighted.
    """
    if len(specs) == 1:
        return specs[0]
    if not specs:
        raise ValueError("geen batterij om te simuleren")
    capacity = sum(s.capacity_kwh for s in specs) or 1.0

    def weighted(attribute: str) -> float:
        return sum(getattr(s, attribute) * s.capacity_kwh for s in specs) / capacity

    return BatterySpec(
        name=name,
        capacity_kwh=capacity,
        max_charge_w=sum(s.max_charge_w for s in specs),
        max_discharge_w=sum(s.max_discharge_w for s in specs),
        minimum_power_w=min(s.minimum_power_w for s in specs),
        soc_min=weighted("soc_min"),
        soc_max=weighted("soc_max"),
        cycle_cost=weighted("cycle_cost"),
        charge_efficiency=weighted("charge_efficiency"),
        discharge_efficiency=weighted("discharge_efficiency"),
    )


def simulate(
    plan: FastPlan,
    samples: Sequence[Sample],
    limits: PolicyLimits,
    soc_start: float,
    label: str,
    use_fast_layer: bool,
    inverter_efficiency: float = DEFAULT_INVERTER_EFFICIENCY,
    timezone: Optional[datetime.tzinfo] = None,
) -> SimResult:
    """Replay *samples* against *plan*, with or without the fast layer."""
    if not plan.specs:
        raise ValueError("het plan bevat geen batterij")
    if len(samples) < 2:
        raise ValueError("te weinig meetpunten om te simuleren")

    spec = plan.specs[0] if len(plan.specs) == 1 else aggregate_spec(plan.specs)
    sim_plan = FastPlan(
        created_ts=plan.created_ts,
        interval_s=plan.interval_s,
        specs=[spec],
        intervals=[_collapse(i) for i in plan.intervals],
        strategy=plan.strategy,
        price_average=plan.price_average,
    )

    battery = BatteryModel(spec, soc_start, inverter_efficiency)
    policy = FastControlPolicy(limits)
    state = ControllerState()
    result = SimResult(
        label=label,
        soc_start=soc_start,
        capacity_kwh=spec.capacity_kwh,
    )

    previous_command = 0.0
    for index in range(len(samples) - 1):
        sample = samples[index]
        dt_s = samples[index + 1].timestamp - sample.timestamp
        if dt_s <= 0:
            continue
        dt_h = dt_s / 3600.0
        interval = sim_plan.interval_at(sample.timestamp)
        if interval is None:
            continue

        plan_w = interval.battery(0).ac_power_w
        if use_fast_layer:
            measurement = Measurement(
                timestamp=sample.timestamp,
                grid_w=sample.house_w + previous_command,
                batteries=[
                    BatteryMeasurement(
                        soc=battery.soc, power_w=previous_command, valid=True
                    )
                ],
                grid_valid=True,
            )
            day_key = _day_key(sample.timestamp, timezone)
            policy.account(state, sim_plan, measurement, day_key)
            decision = policy.decide(sim_plan, measurement, state)
            command = decision.batteries[0].setpoint_w
            wrote = decision.batteries[0].write
            overriding = decision.batteries[0].override
        else:
            command = plan_w
            wrote = abs(command - previous_command) > 1.0
            overriding = False

        actual_w, dc_kwh = battery.step(command, dt_h)
        previous_command = actual_w

        grid_w = sample.house_w + actual_w
        energy_cost = (
            interval.price_import * max(grid_w, 0.0)
            - interval.price_export * max(-grid_w, 0.0)
        ) * dt_h / 1000.0
        wear_cost = spec.cycle_cost * dc_kwh

        day = _day_key(sample.timestamp, timezone)
        metrics = result.days.setdefault(day, DayMetrics(day))
        metrics.energy_cost += energy_cost
        metrics.wear_cost += wear_cost
        metrics.import_kwh += max(grid_w, 0.0) * dt_h / 1000.0
        metrics.export_kwh += max(-grid_w, 0.0) * dt_h / 1000.0
        metrics.throughput_kwh += dc_kwh
        metrics.peak_import_w = max(metrics.peak_import_w, grid_w)
        metrics.writes += 1 if wrote else 0
        metrics.override_s += dt_s if overriding else 0.0

        result.energy_cost += energy_cost
        result.wear_cost += wear_cost
        result.import_kwh += max(grid_w, 0.0) * dt_h / 1000.0
        result.export_kwh += max(-grid_w, 0.0) * dt_h / 1000.0
        result.throughput_kwh += dc_kwh
        result.peak_import_w = max(result.peak_import_w, grid_w)
        result.writes += 1 if wrote else 0
        result.override_s += dt_s if overriding else 0.0
        result.soc_series.append((sample.timestamp, battery.soc))

    result.soc_end = battery.soc
    result.duration_h = (samples[-1].timestamp - samples[0].timestamp) / 3600.0
    # Value what is left in the cells, otherwise ending emptier looks cheaper.
    # Priced at what it would fetch on delivery: average tariff times the
    # efficiency of getting it back out to AC.
    valuation = plan.price_average * max(
        0.01, spec.discharge_efficiency * inverter_efficiency
    )
    result.terminal_value = (
        (result.soc_end - result.soc_start) * spec.kwh_per_percent * valuation
    )
    return result


def compare(
    plan: FastPlan,
    samples: Sequence[Sample],
    limits: PolicyLimits,
    soc_start: float,
    inverter_efficiency: float = DEFAULT_INVERTER_EFFICIENCY,
    timezone: Optional[datetime.tzinfo] = None,
) -> Comparison:
    """Run both scenarios on identical inputs."""
    return Comparison(
        baseline=simulate(
            plan,
            samples,
            limits,
            soc_start,
            "day-ahead",
            use_fast_layer=False,
            inverter_efficiency=inverter_efficiency,
            timezone=timezone,
        ),
        fast=simulate(
            plan,
            samples,
            limits,
            soc_start,
            "day-ahead + fast",
            use_fast_layer=True,
            inverter_efficiency=inverter_efficiency,
            timezone=timezone,
        ),
    )


def _collapse(interval: PlanInterval) -> PlanInterval:
    """Merge the per battery plan steps of one interval into a single step."""
    if len(interval.batteries) == 1:
        return interval
    total_power = sum(b.ac_power_w for b in interval.batteries)
    if interval.batteries:
        soc_begin = sum(b.soc_begin for b in interval.batteries) / len(
            interval.batteries
        )
        soc_end = sum(b.soc_end for b in interval.batteries) / len(interval.batteries)
    else:
        soc_begin = soc_end = 0.0
    return PlanInterval(
        start_ts=interval.start_ts,
        end_ts=interval.end_ts,
        price_import=interval.price_import,
        price_export=interval.price_export,
        grid_w=interval.grid_w,
        house_w=interval.house_w,
        pv_w=interval.pv_w,
        batteries=[
            BatteryPlanStep(
                ac_power_w=total_power, soc_begin=soc_begin, soc_end=soc_end
            )
        ],
    )


def _day_key(timestamp: float, timezone: Optional[datetime.tzinfo]) -> str:
    return datetime.datetime.fromtimestamp(timestamp, timezone).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# synthetic scenarios
# ---------------------------------------------------------------------------


def synthetic_case(
    spec: BatterySpec,
    days: int = 1,
    interval_s: int = 900,
    step_s: int = 60,
    prices: Optional[Sequence[float]] = None,
    export_discount: float = 0.18,
    load_profile: Optional[Callable[[float], float]] = None,
    soc_start: float = 50.0,
    seed: int = 7,
) -> tuple[FastPlan, list[Sample], float]:
    """Build a self-contained scenario with a plausible Dutch price shape.

    Used by the tests and by ``da_fast.py demo`` so the machinery can be
    exercised without a populated Home Assistant database. The house load is a
    smooth baseload plus sharp appliance peaks that the plan cannot know about,
    which is exactly the error the fast layer exists to absorb.
    """
    import random

    rng = random.Random(seed)
    if prices is None:
        # A typical winter day: cheap at night, a solar dip midday, an evening peak.
        prices = [
            0.22, 0.20, 0.19, 0.19, 0.20, 0.24, 0.30, 0.34,
            0.31, 0.26, 0.21, 0.17, 0.14, 0.13, 0.15, 0.20,
            0.28, 0.38, 0.46, 0.44, 0.36, 0.30, 0.26, 0.23,
        ]

    start = int(
        datetime.datetime.combine(
            datetime.date.today() - datetime.timedelta(days=days),
            datetime.time(0, 0),
        ).timestamp()
    )
    steps_per_hour = 3600 // interval_s
    total_intervals = days * 24 * steps_per_hour

    def price_at(index: int) -> float:
        return prices[(index // steps_per_hour) % 24]

    def baseload(timestamp: float) -> float:
        hour = datetime.datetime.fromtimestamp(timestamp).hour
        night = 250.0
        day = 450.0
        evening = 700.0
        if hour < 6:
            return night
        if hour < 17:
            return day
        return evening

    def pv(timestamp: float) -> float:
        hour = (
            datetime.datetime.fromtimestamp(timestamp).hour
            + datetime.datetime.fromtimestamp(timestamp).minute / 60.0
        )
        if not 8.0 <= hour <= 18.0:
            return 0.0
        return 3200.0 * math.sin(math.pi * (hour - 8.0) / 10.0) ** 2

    if load_profile is None:

        def load_profile(timestamp: float) -> float:
            hour = datetime.datetime.fromtimestamp(timestamp).hour
            minute = datetime.datetime.fromtimestamp(timestamp).minute
            extra = 0.0
            # Kettle and coffee in the morning.
            if hour == 7 and 10 <= minute < 16:
                extra += 2200.0
            # Oven during the evening peak, the expensive hour.
            if hour == 18 and 5 <= minute < 45:
                extra += 3200.0
            # Induction hob, short and sharp.
            if hour == 18 and 20 <= minute < 30:
                extra += 2400.0
            # Tumble dryer mid afternoon, cheap hour but PV dependent.
            if hour == 14 and 0 <= minute < 50:
                extra += 1800.0
            return extra + rng.gauss(0.0, 60.0)

    # Build the plan from the *forecast*: baseload and PV only. The appliance
    # peaks are deliberately invisible to it, just as they are in reality.
    intervals: list[PlanInterval] = []
    soc = soc_start
    for index in range(total_intervals):
        interval_start = start + index * interval_s
        price_import = price_at(index)
        price_export = max(0.0, price_import - export_discount)
        forecast_house = baseload(interval_start) - pv(interval_start)
        # Charge when cheap and there is room, discharge when expensive.
        if price_import <= 0.18 and soc < spec.soc_max - 5:
            battery_w = min(spec.max_charge_w, 3000.0)
        elif price_import >= 0.36 and soc > spec.soc_min + 5:
            battery_w = -min(spec.max_discharge_w, max(0.0, forecast_house))
        else:
            battery_w = -max(0.0, min(forecast_house, spec.max_discharge_w)) * (
                1.0 if soc > spec.soc_min + 10 else 0.0
            )
        soc_begin = soc
        dt_h = interval_s / 3600.0
        if battery_w >= 0:
            soc += (
                battery_w * dt_h * spec.charge_efficiency / 1000.0
            ) / spec.kwh_per_percent
        else:
            soc -= (
                -battery_w * dt_h / spec.discharge_efficiency / 1000.0
            ) / spec.kwh_per_percent
        soc = max(spec.soc_min, min(spec.soc_max, soc))
        intervals.append(
            PlanInterval(
                start_ts=interval_start,
                end_ts=interval_start + interval_s,
                price_import=price_import,
                price_export=price_export,
                grid_w=forecast_house + battery_w,
                house_w=forecast_house,
                pv_w=pv(interval_start),
                batteries=[
                    BatteryPlanStep(
                        ac_power_w=battery_w, soc_begin=soc_begin, soc_end=soc
                    )
                ],
            )
        )

    plan = FastPlan(
        created_ts=start,
        interval_s=interval_s,
        specs=[spec],
        intervals=intervals,
        strategy="minimize cost",
        price_average=sum(prices) / len(prices),
    )

    samples = [
        Sample(
            timestamp=start + offset,
            house_w=baseload(start + offset)
            + load_profile(start + offset)
            - pv(start + offset),
        )
        for offset in range(0, total_intervals * interval_s + step_s, step_s)
    ]
    return plan, samples, soc_start


# ---------------------------------------------------------------------------
# historical data
# ---------------------------------------------------------------------------


@dataclass
class HistoryWindow:
    """What the loader produced, plus what it had to compromise on."""

    plan: FastPlan
    samples: list[Sample]
    soc_start: float
    resolution_s: int
    source: str
    warnings: list[str] = field(default_factory=list)


class HistoryLoader:
    """Pulls the backtest inputs out of the two databases DAO already has.

    Prices and the historical plans come from DAO's own tables, the measured
    power comes from the Home Assistant recorder. Both recorder layouts are
    supported: the modern ``states_meta`` join and the legacy ``entity_id``
    column.
    """

    def __init__(self, db_ha, db_da, report, time_zone: Optional[str] = None):
        self.db_ha = db_ha
        self.db_da = db_da
        self.report = report
        self.time_zone = time_zone

    # -- recorder --------------------------------------------------------

    def _series(
        self, entity_id: str, start: datetime.datetime, end: datetime.datetime
    ):
        """Measured series for one entity as a ``(timestamp, value)`` list."""
        import pandas as pd
        from sqlalchemy import Table, and_, select

        metadata = self.db_ha.metadata
        engine = self.db_ha.engine
        states = Table("states", metadata, autoload_with=engine)
        start_ts = start.timestamp()
        end_ts = end.timestamp()

        if "metadata_id" in states.c:
            states_meta = Table("states_meta", metadata, autoload_with=engine)
            query = (
                select(states.c.last_updated_ts, states.c.state)
                .select_from(
                    states.join(
                        states_meta,
                        states.c.metadata_id == states_meta.c.metadata_id,
                    )
                )
                .where(
                    and_(
                        states_meta.c.entity_id == entity_id,
                        states.c.last_updated_ts >= start_ts,
                        states.c.last_updated_ts < end_ts,
                    )
                )
                .order_by(states.c.last_updated_ts)
            )
        else:
            query = (
                select(states.c.last_updated_ts, states.c.state)
                .where(
                    and_(
                        states.c.entity_id == entity_id,
                        states.c.last_updated_ts >= start_ts,
                        states.c.last_updated_ts < end_ts,
                    )
                )
                .order_by(states.c.last_updated_ts)
            )

        with engine.connect() as connection:
            rows = connection.execute(query).fetchall()
        frame = pd.DataFrame(rows, columns=["ts", "value"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame.dropna()

    def _series_on_grid(
        self,
        entity_id: str,
        scale: float,
        stamps: Sequence[int],
        start: datetime.datetime,
        end: datetime.datetime,
        chunk_days: int = CHUNK_DAYS,
    ):
        """Read one entity onto the simulation grid, a few days at a time.

        A P1 meter that updates every second produces roughly a million
        recorder rows per fortnight. Pulling that into one DataFrame is
        pointless -- the grid it ends up on has a few thousand points -- and on
        a Home Assistant Yellow or a Raspberry Pi it is enough to exhaust
        memory. Reading per chunk and downsampling immediately keeps the peak
        footprint at one chunk, at the cost of a few extra queries.

        The last reading of a chunk is carried into the next one so the forward
        fill stays correct across the boundary.
        """
        import pandas as pd

        total = None
        carry = None
        rows_read = 0
        for chunk_start, chunk_end in _iter_chunks(start, end, chunk_days):
            sub = [
                s
                for s in stamps
                if chunk_start.timestamp() <= s < chunk_end.timestamp()
            ]
            frame = self._series(entity_id, chunk_start, chunk_end)
            rows_read += len(frame)
            if carry is not None:
                seed = pd.DataFrame(
                    {"ts": [chunk_start.timestamp()], "value": [carry]}
                )
                frame = pd.concat([seed, frame], ignore_index=True)
            if not frame.empty:
                carry = float(frame["value"].iloc[-1])
            if not sub:
                continue
            series = _on_grid(frame, scale, sub)
            if series is None:
                series = pd.Series(index=sub, dtype="float64")
            total = series if total is None else pd.concat([total, series])

        logging.debug(
            f"Backtest: {entity_id} leverde {rows_read} recorder-rijen, "
            f"teruggebracht tot {len(stamps)} rasterpunten"
        )
        if total is None or total.isna().all():
            return None
        return total.reindex(stamps)

    def _sum_entities(
        self,
        entities: Sequence[tuple[str, float]],
        stamps: Sequence[int],
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> Optional[list[Optional[float]]]:
        """Scale, resample and sum several entities onto the simulation grid."""
        import pandas as pd

        total = None
        for entity_id, scale in entities:
            series = self._series_on_grid(entity_id, scale, stamps, start, end)
            if series is None:
                return None
            total = series if total is None else total.add(series, fill_value=0.0)
        if total is None:
            return None
        return [None if pd.isna(value) else float(value) for value in total]

    def _sample_gap(self, entity_id: str, start: datetime.datetime) -> float:
        """Median seconds between two readings, measured on one sample day."""
        frame = self._series(entity_id, start, start + datetime.timedelta(days=1))
        if frame is None or frame.empty:
            return math.inf
        return _median_gap(frame["ts"].tolist())

    # -- plan ------------------------------------------------------------

    def _plan(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        spec: BatterySpec,
        interval_s: int,
    ) -> FastPlan:
        """Rebuild the historical day-ahead plans from the prognoses table."""
        import pandas as pd

        codes = ["bat_in", "bat_out", "cons", "prod", "soc"]
        frames = {}
        for code in codes:
            frame = self.db_da.get_column_data("prognoses", code, start, end)
            frames[code] = frame.set_index("utc")["value"] if len(frame) else None

        prices = self.report.get_price_data(
            start, end, "1hour" if interval_s >= 3600 else "15min"
        )
        prices["ts"] = prices["time"].apply(_local_epoch)
        price_by_ts = prices.set_index("ts")[["da_cons", "da_prod"]]

        stamps = sorted(
            set().union(*[set(f.index) for f in frames.values() if f is not None])
        )
        hour_fraction = interval_s / 3600.0
        intervals: list[PlanInterval] = []
        for stamp in stamps:
            def value(code: str) -> float:
                series = frames.get(code)
                if series is None or stamp not in series.index:
                    return 0.0
                raw = series.loc[stamp]
                raw = raw.iloc[0] if hasattr(raw, "iloc") else raw
                return 0.0 if pd.isna(raw) else float(raw)

            price_row = _nearest(price_by_ts, stamp, interval_s)
            if price_row is None:
                continue
            battery_w = (value("bat_in") - value("bat_out")) * 1000.0 / hour_fraction
            grid_w = (value("cons") - value("prod")) * 1000.0 / hour_fraction
            soc_value = value("soc")
            intervals.append(
                PlanInterval(
                    start_ts=int(stamp),
                    end_ts=int(stamp + interval_s),
                    price_import=float(price_row[0]),
                    price_export=float(price_row[1]),
                    grid_w=grid_w,
                    house_w=grid_w - battery_w,
                    batteries=[
                        BatteryPlanStep(
                            ac_power_w=battery_w,
                            soc_begin=soc_value,
                            soc_end=soc_value,
                        )
                    ],
                )
            )

        average = (
            sum(i.price_import for i in intervals) / len(intervals) if intervals else 0.0
        )
        return FastPlan(
            created_ts=int(start.timestamp()),
            interval_s=interval_s,
            specs=[spec],
            intervals=intervals,
            price_average=average,
        )

    # -- the public entry point ------------------------------------------

    def load(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        spec: BatterySpec,
        grid_entities: Sequence[tuple[str, float]],
        battery_entities: Sequence[tuple[str, float]],
        soc_entity: Optional[str],
        interval_s: int,
        step_s: int = 60,
    ) -> HistoryWindow:
        """Assemble a backtest window.

        ``grid_entities`` and ``battery_entities`` are ``(entity_id, scale)``
        pairs, where the scale carries both the unit conversion and the sign
        convention, so a positive/negative sensor pair is just two entries.
        """
        warnings: list[str] = []
        stamps = list(range(int(start.timestamp()), int(end.timestamp()), step_s))

        grid_w = self._sum_entities(grid_entities, stamps, start, end)
        if grid_w is None:
            raise RuntimeError(
                "geen netvermogen in de Home Assistant recorder gevonden voor de "
                "opgegeven periode; controleer de entiteit en de bewaartermijn "
                "(purge_keep_days)"
            )

        battery_w = None
        if battery_entities:
            battery_w = self._sum_entities(battery_entities, stamps, start, end)
            if battery_w is None:
                warnings.append(
                    "accuvermogen ontbreekt in de recorder; het plan wordt gebruikt "
                    "als schatting van wat de accu deed"
                )
        else:
            warnings.append(
                "geen accuvermogen-sensor geconfigureerd; het plan wordt gebruikt "
                "als schatting van wat de accu deed"
            )

        plan = self._plan(start, end, spec, interval_s)
        if not plan.intervals:
            raise RuntimeError(
                "geen historische planning in de prognoses-tabel voor deze periode"
            )

        samples: list[Sample] = []
        for position, stamp in enumerate(stamps):
            measured_grid = grid_w[position]
            if measured_grid is None:
                continue
            if battery_w is not None and battery_w[position] is not None:
                measured_battery = battery_w[position]
            else:
                interval = plan.interval_at(stamp)
                measured_battery = interval.battery(0).ac_power_w if interval else 0.0
            samples.append(Sample(stamp, measured_grid - measured_battery))

        soc_start = 50.0
        if soc_entity:
            # Only the opening value is needed, so do not drag in the whole
            # window of state changes.
            soc_frame = self._series(
                soc_entity, start, min(start + datetime.timedelta(days=1), end)
            )
            if soc_frame is not None and not soc_frame.empty:
                soc_start = float(soc_frame.iloc[0]["value"])
            else:
                warnings.append("SoC-historie ontbreekt; gestart op 50%")

        median_gap = (
            self._sample_gap(grid_entities[0][0], start) if grid_entities else math.inf
        )
        if median_gap > 300:
            warnings.append(
                f"de recorder levert gemiddeld maar een meting per "
                f"{median_gap:.0f} s; korte verbruikspieken zijn daarmee "
                f"onzichtbaar en de besparing wordt onderschat"
            )

        return HistoryWindow(
            plan=plan,
            samples=samples,
            soc_start=soc_start,
            resolution_s=step_s,
            source="recorder states",
            warnings=warnings,
        )


def _on_grid(frame, scale: float, stamps: Sequence[int]):
    """Forward fill one sparse event series onto a uniform grid.

    Recorder rows only exist when a sensor changes value, so between two rows
    the previous reading still holds.
    """
    import pandas as pd

    if frame is None or frame.empty:
        return None
    series = pd.Series(frame["value"].to_numpy() * scale, index=frame["ts"].to_numpy())
    series = series.groupby(level=0).last().sort_index()
    return series.reindex(series.index.union(stamps)).ffill().reindex(stamps)


def _sum_on_grid(entries, stamps: Sequence[int]) -> Optional[list[Optional[float]]]:
    """Scale, resample and sum several measurement series onto one grid."""
    import pandas as pd

    total = None
    for _entity_id, scale, frame in entries:
        series = _on_grid(frame, scale, stamps)
        if series is None:
            return None
        total = series if total is None else total.add(series, fill_value=0.0)
    if total is None:
        return None
    return [None if pd.isna(value) else float(value) for value in total]


def _local_epoch(value) -> float:
    """Epoch seconds for a naive datetime that denotes *local* time.

    DAO's price frame holds naive local datetimes. ``pandas.Timestamp`` would
    read a naive value as UTC and silently shift everything by the local offset,
    so convert to a plain ``datetime`` first, which uses the local zone.
    """
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    return value.timestamp()


def _nearest(frame, stamp: float, tolerance: int):
    """Row whose index is closest to *stamp*, within *tolerance* seconds."""
    if frame.empty:
        return None
    differences = (frame.index.values - stamp)
    position = abs(differences).argmin()
    if abs(differences[position]) > tolerance:
        return None
    return frame.iloc[position].values


def _median_gap(stamps: Sequence[float]) -> float:
    if len(stamps) < 3:
        return math.inf
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]) if b > a)
    return gaps[len(gaps) // 2] if gaps else math.inf


def log_comparison(comparison: Comparison, show_daily: bool = True) -> None:
    """Print the summary through the logging machinery DAO already sets up."""
    for line in comparison.report().splitlines():
        logging.info(line)
    if show_daily:
        for line in comparison.daily_table().splitlines():
            logging.info(line)
