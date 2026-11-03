"""Lokale afrekening: dry-run tenzij --publish; geen HA-client of actuator."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    # Het geregistreerde script draait vanuit dao/webserver, zonder installatie.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dao.prog import task_execution  # noqa: E402
from dao.prog.da_base import DaBase  # noqa: E402
from dao.prog.settlement.tariffs import _canonical_default  # noqa: E402


def load_service(path):
    return DaBase.local_settlement_app(path).settlement_service()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Berekende/geschatte afrekening; geen leveranciersbetaling")
    parser.add_argument("command", choices=("preview", "revalue", "reconcile"))
    parser.add_argument("--config", default="../data/options.json")
    parser.add_argument("--start", type=int, help="UTC epoch, inclusief")
    parser.add_argument("--end", type=int, help="UTC epoch, exclusief")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--scheduled", action="store_true", help="Gebruik de shadow/active scheduleradapter")
    args = parser.parse_args(argv)
    if args.command == "preview" and args.publish:
        parser.error("preview is strikt alleen-lezen")
    if args.publish and (not args.reason or not args.reason.strip()):
        parser.error("--publish vereist --reason")
    if args.scheduled and (args.command != "reconcile" or not args.publish or args.reason != "scheduled"
                           or args.start is not None or args.end is not None):
        parser.error("--scheduled vereist reconcile --publish --reason scheduled zonder bereik")
    try:
        with task_execution.execution("settlement_consolidate"):
            return _execute(args)
    except Exception as exc:
        logging.error(f"Afrekening niet uitgevoerd: {exc}")
        return 1


def _execute(args) -> int:
    if args.scheduled:
        DaBase.local_settlement_app(args.config).consolidate_settlement()
        print(json.dumps({"message": "Scheduleradapter uitgevoerd"}))
        return 0
    service = load_service(args.config)
    lower, upper = service.scheduled_range() if args.start is None or args.end is None else (args.start, args.end)
    start = lower if args.start is None else args.start
    end = upper if args.end is None else args.end
    result = service.consolidate(start, end, publish=args.publish, reason=args.reason or "local-preview")
    print(json.dumps(asdict(result), default=_canonical_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
