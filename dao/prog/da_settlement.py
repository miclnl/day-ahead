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

from dao.prog.settlement.tariffs import _canonical_default  # noqa: E402

#: De handmatige leveranciersvergelijking: uitsluitend lezen, geen taaklease.
_COMPARE_COMMAND = "compare-invoice"
_COMPARE_EXITS = {"matched": 0, "different": 3, "not_comparable": 4}
#: De enige ValueError die de boekreader voor een nooit gepubliceerde generatie opwerpt.
_GENERATION_NOT_PUBLISHED = "Requested financial generation has not been published"
#: De door de boekreader ondersteunde generatie-ruimte (signed 64 bits, exclusief bovengrens).
_MAX_GENERATION = 2 ** 63 - 1


def load_service(path):
    from dao.prog.da_base import DaBase
    return DaBase.local_settlement_app(path).settlement_service()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Berekende/geschatte afrekening; geen leveranciersbetaling")
    parser.add_argument("command", choices=("preview", "revalue", "reconcile", _COMPARE_COMMAND))
    parser.add_argument("--config", default="../data/options.json")
    parser.add_argument("--start", type=int, help="UTC epoch, inclusief")
    parser.add_argument("--end", type=int, help="UTC epoch, exclusief")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--scheduled", action="store_true", help="Gebruik de shadow/active scheduleradapter")
    parser.add_argument("--supplier", help="Genormaliseerd leveranciersbestand; alleen compare-invoice")
    parser.add_argument("--generation", help="Gepinde boekgeneratie; alleen compare-invoice")
    args = parser.parse_args(argv)
    if args.command == _COMPARE_COMMAND:
        # De vergelijkroute staat vóór task_execution: geen lease, geen
        # schedulerclaim, geen taakstatus en geen writerbootstrap.
        return _compare_invoice(args)
    if args.command == "preview" and args.publish:
        parser.error("preview is strikt alleen-lezen")
    if args.publish and (not args.reason or not args.reason.strip()):
        parser.error("--publish vereist --reason")
    if args.scheduled and (args.command != "reconcile" or not args.publish or args.reason != "scheduled"
                           or args.start is not None or args.end is not None):
        parser.error("--scheduled vereist reconcile --publish --reason scheduled zonder bereik")
    if args.supplier is not None:
        parser.error("--supplier is alleen geldig voor compare-invoice")
    if args.generation is not None:
        parser.error("--generation is alleen geldig voor compare-invoice")
    try:
        from dao.prog import task_execution
        with task_execution.execution("settlement_consolidate"):
            return _execute(args)
    except Exception as exc:
        logging.error(f"Afrekening niet uitgevoerd: {exc}")
        return 1


def _execute(args) -> int:
    if args.scheduled:
        from dao.prog.da_base import DaBase
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


def _rejected(category: str) -> int:
    """An invalid argument or file: exit 2, without a financial report."""
    print(json.dumps({"status": "rejected", "category": category}), file=sys.stderr)
    return 2


def _failed(category: str) -> int:
    """A reader or operational failure: exit 1, without a financial report."""
    print(json.dumps({"status": "error", "category": category}), file=sys.stderr)
    return 1


def _compare_invoice(args) -> int:
    """Compare one supplier period with the pinned book; never write anything."""
    from dao.prog.settlement.reconciliation import reconcile
    from dao.prog.settlement.reconciliation_io import (
        ReadOnlyViolation,
        ReconciliationConfigError,
        ReconciliationReaderError,
        SupplierFileError,
        load_supplier,
        open_reconciliation_reader,
    )
    from dao.prog.settlement.repository import ConcurrentSettlementError, ReconciliationReadError

    if args.publish or args.scheduled or args.reason is not None:
        return _rejected("compare_flags_not_allowed")
    if not args.supplier:
        return _rejected("supplier_required")
    if (args.start is None) != (args.end is None):
        return _rejected("bounds_required_together")
    generation = None
    if args.generation is not None:
        try:
            generation = int(args.generation)
        except ValueError:
            return _rejected("generation_invalid")
        if not 0 <= generation < _MAX_GENERATION:
            return _rejected("generation_invalid")
        if generation == 0:
            # Generatie 0 is de sentinel voor "nog nooit gepubliceerd": de
            # lezer zou hem als een leeg boek teruggeven, dus hij wordt hier
            # geweigerd als de ongeldige selectie die hij is.
            return _rejected("generation_not_published")
    try:
        supplier = load_supplier(Path(args.supplier))
    except SupplierFileError as exc:
        return _rejected(exc.category)
    if args.start is not None and (args.start, args.end) != (supplier.start_ts, supplier.end_ts):
        return _rejected("bounds_mismatch")
    try:
        with open_reconciliation_reader(Path(args.config),
                                        expected_meter_id=supplier.meter_id) as reader:
            basis = reader.read_reconciliation(supplier.meter_id, supplier.start_ts,
                                               supplier.end_ts, generation=generation)
    except ReconciliationConfigError as exc:
        return _rejected(exc.category)
    except ReconciliationReaderError as exc:
        return _failed(exc.category)
    except ReconciliationReadError:
        return _failed("book_view_unreadable")
    except ConcurrentSettlementError:
        return _failed("concurrent_settlement")
    except ReadOnlyViolation:
        return _failed("read_only_violation")
    except ValueError as exc:
        # Alleen een aantoonbaar niet-gepubliceerde generatie is een ongeldige
        # selectie.  De aanvraag is al volledig gevalideerd, dus elke andere
        # ValueError uit de reader is een operationele leesfout.
        if generation is not None and str(exc) == _GENERATION_NOT_PUBLISHED:
            return _rejected("generation_not_published")
        return _failed("book_read_failed")
    except Exception:
        # Fail closed: een onverwachte fout is nooit een geslaagde vergelijking
        # en lekt geen interne details.
        return _failed("reader_unavailable")
    try:
        report = reconcile(basis.view, supplier, sources=basis.sources)
    except Exception:
        return _failed("comparison_failed")
    print(json.dumps(asdict(report), default=_canonical_default, sort_keys=True))
    return _COMPARE_EXITS[report.status]


if __name__ == "__main__":
    raise SystemExit(main())
