"""Command line entry point for the fast control layer.

Normally the layer runs as a thread inside ``da_scheduler.py`` and you never
need this script. It exists for three things:

``run``
    Run the control loop in the foreground, for example while debugging or
    when you prefer it as a separate service.

``once``
    Execute exactly one control cycle and print the decision. The quickest way
    to verify that your sensors are wired up correctly.

``simulate``
    Backtest the layer on your own history and print what it would have saved.

``demo``
    Backtest on a built-in synthetic day, so the machinery can be verified
    without any database at all.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

# Run directly from dao/prog without a PYTHONPATH: put the repository root on
# the path so the "dao.prog.*" imports below resolve. The add-on itself sets
# PYTHONPATH in run.sh, so this is a no-op there.
if __package__ in (None, ""):
    _ROOT = str(Path(__file__).resolve().parents[2])
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from dao.prog.fastctrl.plan import BatterySpec, FAST_PLAN_FILE, load_plan  # noqa: E402
from dao.prog.fastctrl.policy import PolicyLimits  # noqa: E402
from dao.prog.fastctrl.runner import FastControlRunner  # noqa: E402
from dao.prog.fastctrl.simulate import (  # noqa: E402
    HistoryLoader,
    aggregate_spec,
    compare,
    log_comparison,
    synthetic_case,
)


def _configure_logging(level: str = "info") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _make_base(options: str):
    from day_ahead import DaCalc

    instance = DaCalc(options)
    instance.debug = False
    return instance


def cmd_run(args) -> int:
    base = _make_base(args.options)
    runner = FastControlRunner(base, plan_path=args.plan)
    runner.run()
    return 0


def cmd_once(args) -> int:
    base = _make_base(args.options)
    runner = FastControlRunner(base, plan_path=args.plan)
    decision = runner.tick()
    if decision is None:
        logging.warning(
            "Geen beslissing genomen. Controleer of de modus aan staat, of er een "
            "plan is en of de 'grid power' sensor is ingesteld."
        )
        return 1
    print(json.dumps(decision.as_attributes(), indent=2))
    return 0


def _spec_from_config(config) -> BatterySpec:
    """Build a simulation battery from the configured batteries."""
    specs = []
    for battery in config.battery:
        charge_stages = [s.power for s in battery.charge_stages]
        discharge_stages = [s.power for s in battery.discharge_stages]
        specs.append(
            BatterySpec(
                name=battery.name,
                capacity_kwh=float(battery.capacity),
                max_charge_w=float(max(charge_stages)),
                max_discharge_w=float(max(discharge_stages)),
                minimum_power_w=float(battery.minimum_power or 0),
                soc_min=float(battery.lower_limit.value),
                soc_max=float(battery.upper_limit.value),
                cycle_cost=float(battery.cycle_cost),
                charge_efficiency=float(battery.dc_to_bat_efficiency),
                discharge_efficiency=float(battery.bat_to_dc_efficiency),
                soc_entity=battery.entity_actual_level,
            )
        )
    if not specs:
        raise SystemExit("Er is geen batterij geconfigureerd, simuleren heeft geen zin.")
    return aggregate_spec(specs)


def _entity_scales(sensor) -> list[tuple[str, float]]:
    """Turn a PowerSensor into ``(entity_id, scale)`` pairs for the loader."""
    unit = 1000.0 if sensor.unit == "kW" else 1.0
    sign = -1.0 if sensor.invert else 1.0
    if sensor.entity is not None:
        return [(sensor.entity, unit * sign)]
    pairs = []
    if sensor.entity_positive is not None:
        pairs.append((sensor.entity_positive, unit * sign))
    if sensor.entity_negative is not None:
        pairs.append((sensor.entity_negative, -unit * sign))
    return pairs


def _limits_from_config(config) -> PolicyLimits:
    fast = config.fast_control
    return PolicyLimits(
        storage_value_mode=str(fast.storage_value_mode.value),
        storage_value_fixed=(
            float(fast.storage_value.value) if fast.storage_value is not None else None
        ),
        round_trip_efficiency=fast.round_trip_efficiency,
        min_benefit=fast.min_benefit,
        deadband=float(fast.deadband),
        min_command_interval=float(fast.min_command_interval),
        urgent_deviation=float(fast.urgent_deviation),
        max_ramp=float(fast.max_ramp) if fast.max_ramp else None,
        release_deviation=float(fast.release_deviation),
        release_time=float(fast.release_time),
        energy_budget=fast.energy_budget,
        daily_extra_throughput=fast.daily_extra_throughput,
        soc_margin=fast.soc_margin,
        max_grid_import=(
            float(fast.max_grid_import) if fast.max_grid_import is not None else None
        ),
        allow_grid_charge=bool(fast.allow_grid_charge.value),
    )


def cmd_simulate(args) -> int:
    from da_report import Report

    report = Report(args.options)
    config = report.config
    spec = _spec_from_config(config)
    fast = config.fast_control

    grid_entities = _entity_scales(fast.grid_power)
    if not grid_entities:
        raise SystemExit(
            "Stel eerst 'fast control' -> 'grid power' in, anders is er niets te meten."
        )
    battery_entities: list[tuple[str, float]] = []
    for link in fast.batteries:
        battery_entities.extend(_entity_scales(link.actual_power))

    end = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - datetime.timedelta(days=args.days)
    interval_s = 900 if config.interval == "15min" else 3600

    loader = HistoryLoader(report.db_ha, report.db_da, report, config.time_zone)
    window = loader.load(
        start=start,
        end=end,
        spec=spec,
        grid_entities=grid_entities,
        battery_entities=battery_entities,
        soc_entity=spec.soc_entity,
        interval_s=interval_s,
        step_s=args.step,
    )
    for warning in window.warnings:
        logging.warning(f"Backtest: {warning}")

    limits = _limits_from_config(config)
    if args.energy_budget is not None:
        limits.energy_budget = args.energy_budget
    if args.min_benefit is not None:
        limits.min_benefit = args.min_benefit

    comparison = compare(
        window.plan, window.samples, limits, window.soc_start
    )
    log_comparison(comparison, show_daily=not args.no_daily)
    return 0


def cmd_demo(args) -> int:
    spec = BatterySpec(
        name="demo",
        capacity_kwh=args.capacity,
        max_charge_w=args.power,
        max_discharge_w=args.power,
        soc_min=20.0,
        soc_max=95.0,
        cycle_cost=args.cycle_cost,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    plan, samples, soc_start = synthetic_case(spec, days=args.days)
    limits = PolicyLimits(
        energy_budget=args.energy_budget,
        min_benefit=args.min_benefit,
        daily_extra_throughput=args.daily_throughput,
    )
    comparison = compare(plan, samples, limits, soc_start)
    log_comparison(comparison, show_daily=not args.no_daily)
    return 0


def cmd_plan(args) -> int:
    plan = load_plan(args.plan)
    if plan is None:
        logging.error(f"Geen plan gevonden op {args.plan}")
        return 1
    now = datetime.datetime.now().timestamp()
    print(
        f"Plan van {datetime.datetime.fromtimestamp(plan.created_ts)}, "
        f"{plan.age(now) / 60:.0f} minuten oud, {len(plan.intervals)} intervallen"
    )
    print(
        f"{'tijd':>6}{'prijs in':>10}{'prijs uit':>11}{'net W':>9}"
        f"{'huis W':>9}{'accu W':>9}{'SoC':>7}"
    )
    for interval in plan.intervals[: args.rows]:
        moment = datetime.datetime.fromtimestamp(interval.start_ts).strftime("%H:%M")
        battery = interval.battery(0)
        marker = " <" if interval.contains(now) else ""
        print(
            f"{moment:>6}{interval.price_import:>10.4f}{interval.price_export:>11.4f}"
            f"{interval.grid_w:>9.0f}{interval.house_w:>9.0f}"
            f"{battery.ac_power_w:>9.0f}{battery.soc_end:>7.1f}{marker}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="da_fast", description="Fast control layer for the Day Ahead Optimizer"
    )
    parser.add_argument(
        "--options", default="../data/options.json", help="path to options.json"
    )
    parser.add_argument("--plan", default=FAST_PLAN_FILE, help="path to fast_plan.json")
    parser.add_argument("--log-level", default="info")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="run the control loop in the foreground")
    sub.add_parser("once", help="run one control cycle and print the decision")

    plan_parser = sub.add_parser("plan", help="show the current plan")
    plan_parser.add_argument("--rows", type=int, default=24)

    simulate_parser = sub.add_parser("simulate", help="backtest on your own history")
    simulate_parser.add_argument("--days", type=int, default=7)
    simulate_parser.add_argument(
        "--step", type=int, default=60, help="simulation timestep in seconds"
    )
    simulate_parser.add_argument("--energy-budget", type=float, default=None)
    simulate_parser.add_argument("--min-benefit", type=float, default=None)
    simulate_parser.add_argument("--no-daily", action="store_true")

    demo_parser = sub.add_parser("demo", help="backtest on a synthetic day")
    demo_parser.add_argument("--days", type=int, default=3)
    demo_parser.add_argument("--capacity", type=float, default=10.0)
    demo_parser.add_argument("--power", type=float, default=5000.0)
    demo_parser.add_argument("--cycle-cost", type=float, default=0.01)
    demo_parser.add_argument("--energy-budget", type=float, default=0.5)
    demo_parser.add_argument("--min-benefit", type=float, default=0.02)
    demo_parser.add_argument("--daily-throughput", type=float, default=4.0)
    demo_parser.add_argument("--no-daily", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    handlers = {
        "run": cmd_run,
        "once": cmd_once,
        "plan": cmd_plan,
        "simulate": cmd_simulate,
        "demo": cmd_demo,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
