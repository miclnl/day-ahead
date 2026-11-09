"""Immutable supplier observations and pinned book provenance; no I/O or repricing."""

import json
import re
from dataclasses import asdict, dataclass, fields, is_dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from hashlib import sha256
from typing import Literal
from unicodedata import category
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dao.prog.settlement.tariffs import _canonical_default, quote
from dao.prog.settlement.types import (
    InputInterval,
    Quality,
    SettlementLine,
    SettlementView,
    TariffSnapshot,
    _FrozenDTO,
)


def _canonical_json(value: object) -> str:
    """Use the existing book representation, including lossless Decimal text."""
    return json.dumps(
        asdict(value) if is_dataclass(value) else value, default=_canonical_default,
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )


def _digest(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class _ReconciliationDTO(_FrozenDTO):
    __slots__ = ()

    def __post_init__(self):
        _FrozenDTO.__post_init__(self)
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("_ts") and value is not None:
                # UTC datetime year 1 through year 9999, independent of host TZ.
                if not -62135596800 <= value <= 253402300799:
                    raise ValueError(f"{field.name}: epoch outside supported datetime range")


def _supplier_decimal(value: Decimal | str, places: int, name: str):
    """Check textual resource limits before the shared DTO parser loses spelling."""
    if not isinstance(value, (str, Decimal)):
        raise TypeError(f"{name}: use Decimal from text")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name}: decimal must be finite")
        # Reject extreme exponents before allocating a fixed-point string.
        if value.adjusted() > 63 or value.as_tuple().exponent < -places:
            raise ValueError(f"{name}: decimal outside supplier limits")
        text = format(value, "f")
    else:
        text = value
    if len(text) > 64 or re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", text) is None:
        raise ValueError(f"{name}: expected bounded decimal text without exponent")
    decimal = Decimal(text)
    if len(decimal.as_tuple().digits) > 36 or decimal.as_tuple().exponent < -places:
        raise ValueError(f"{name}: decimal outside supplier precision limits")


class _SupplierDTO(_ReconciliationDTO):
    __slots__ = ()

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and field.name in {
                "ex_vat", "vat", "gross", "import_kwh", "export_kwh", "total",
                "money_quantum", "energy_quantum",
            }:
                places = 9 if field.name in {"import_kwh", "export_kwh"} else 12
                _supplier_decimal(value, places, field.name)
        _ReconciliationDTO.__post_init__(self)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, str):
                limit = 256 if field.name == "evidence_ref" else 128
                if len(value) > limit or any(category(char) == "Cc" for char in value):
                    raise ValueError(f"{field.name}: invalid supplier identifier or label")


@dataclass(frozen=True, slots=True)
class SupplierAmounts(_SupplierDTO):
    ex_vat: Decimal | None
    vat: Decimal | None
    gross: Decimal | None


@dataclass(frozen=True, slots=True)
class SupplierCredit(_SupplierDTO):
    kind: str
    period_start_ts: int
    period_end_ts: int
    amounts: SupplierAmounts


@dataclass(frozen=True, slots=True)
class SupplierRounding(_SupplierDTO):
    method: str
    money_quantum: Decimal | None
    money_mode: str | None
    aggregation: str | None
    total_rule: str | None
    energy_quantum: Decimal | None
    energy_mode: str | None
    evidence_ref: str | None


@dataclass(frozen=True, slots=True)
class SupplierPeriod(_SupplierDTO):
    schema_version: Literal["supplier-period-v1"]
    meter_id: str
    contract_id: str
    timezone: str
    start_ts: int
    end_ts: int
    currency: str
    energy_unit: str
    sign_convention: str
    total_scope: str
    import_kwh: Decimal | None
    export_kwh: Decimal | None
    import_amounts: SupplierAmounts | None
    base_export_amounts: SupplierAmounts | None
    bonus_amounts: SupplierAmounts | None
    credits: tuple[SupplierCredit, ...] | None
    total: Decimal | None
    rounding: SupplierRounding | None

    def __post_init__(self):
        _SupplierDTO.__post_init__(self)
        if self.end_ts - self.start_ts > 370 * 86400:
            raise ValueError("Supplier period exceeds 370 days")
        try:
            ZoneInfo(self.timezone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("Supplier period requires a known IANA timezone") from exc
        if self.credits is not None:
            if len(self.credits) > 64:
                raise ValueError("Supplier period exceeds 64 credits")
            ordered = sorted(self.credits, key=lambda c: (c.kind, c.period_start_ts, c.period_end_ts))
            for previous, current in zip(ordered, ordered[1:], strict=False):
                if previous.kind == current.kind and previous.period_end_ts > current.period_start_ts:
                    raise ValueError("Supplier credit identities must be unique and nonoverlapping")


@dataclass(frozen=True, slots=True)
class BookSourcePin(_ReconciliationDTO):
    meter_id: str
    generation: int
    view_input_revision: int
    current_generation: int
    current_input_revision: int
    dirty_from_ts: int | None
    start_ts: int
    end_ts: int
    view_digest: str
    read_ts: int


@dataclass(frozen=True, slots=True)
class RunSource(_ReconciliationDTO):
    run_id: str
    generation: int
    input_revision: int
    engine_version: str
    input_hash: str
    result_hash: str
    snapshot_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LineSource(_ReconciliationDTO):
    source_key: str
    start_ts: int
    end_ts: int
    input_revision: int
    input_hash: str
    snapshot_hash: str | None
    valid_from_generation: int
    run_id: str
    line_content_hash: str
    input_content_hash: str | None
    item: InputInterval | None
    snapshot: TariffSnapshot | None
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CreditSource(_ReconciliationDTO):
    source_key: str
    kind: str
    period_start_ts: int
    period_end_ts: int
    valid_from_generation: int
    run_id: str
    adjustment_content_hash: str
    contract_id: str | None
    timezone: str | None
    closing_line: SettlementLine | None
    closing_source: LineSource | None
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReconciliationSources(_ReconciliationDTO):
    pin: BookSourcePin
    runs: tuple[RunSource, ...]
    lines: tuple[LineSource, ...]
    credits: tuple[CreditSource, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComponentComparison(_ReconciliationDTO):
    key: str
    unit: Literal["EUR", "kWh"]
    book_value: Decimal | None
    supplier_value: Decimal | None
    comparable_book_value: Decimal | None
    delta: Decimal | None
    status: Literal["matched", "different", "not_comparable"]
    quality: Quality
    reasons: tuple[str, ...]
    source_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReconciliationReport(_ReconciliationDTO):
    schema_version: Literal["reconciliation-report-v1"]
    pin: BookSourcePin
    supplier: SupplierPeriod
    supplier_digest: str
    source_trust: Literal["registered_book_and_caller_supplied_supplier"]
    freshness: Literal["current", "outdated", "unknown"]
    components: tuple[ComponentComparison, ...]
    total: ComponentComparison
    status: Literal["matched", "different", "not_comparable"]
    reasons: tuple[str, ...]
    provenance: ReconciliationSources


@dataclass(frozen=True, slots=True)
class ReconciliationBasis(_ReconciliationDTO):
    view: SettlementView
    sources: ReconciliationSources


def _stored_sum(values):
    """Only add stored finite operands; never reprice an interval."""
    values = tuple(values)
    if any(value is None for value in values):
        return None
    if not values:
        return Decimal(0)
    places = min(value.as_tuple().exponent for value in values)
    digits = max(value.adjusted() for value in values) - places + len(str(len(values))) + 2
    with localcontext() as ctx:
        ctx.prec = max(1, digits)
        return sum(values, Decimal(0))


def _rounded(value, quantum):
    with localcontext() as ctx:
        ctx.prec = max(1, value.adjusted() - quantum.as_tuple().exponent + 3)
        return value.quantize(quantum, rounding=ROUND_HALF_UP)


_ENERGY_QUANTA = tuple(Decimal((0, (1,), -n)) for n in range(10))


def _rounding_reasons(rounding):
    if rounding is None or not rounding.evidence_ref or not rounding.evidence_ref.strip():
        return ('rounding_missing',)
    if (rounding.method != 'period_component_half_up_v1'
            or rounding.money_quantum != Decimal('0.01')
            or rounding.money_mode != 'ROUND_HALF_UP'
            or rounding.aggregation != 'sum_unrounded_then_round_each_component'
            or rounding.total_rule != 'sum_rounded_gross_components'
            or rounding.energy_mode != 'ROUND_HALF_UP'
            or rounding.energy_quantum not in _ENERGY_QUANTA):
        return ('rounding_unsupported',)
    return ()


def _codes(issues):
    # Caller-provided issues are never copied as raw diagnostic text.
    allowed = {'source_missing', 'source_mismatch', 'credit_identity_unproven',
               'source_version_unsupported', 'coverage_incomplete'}
    return tuple(sorted({issue if issue in allowed else 'source_mismatch' for issue in issues}))


def _quality(qualities):
    qualities = tuple(qualities)
    if not qualities or 'incomplete' in qualities:
        return 'incomplete'
    return 'estimated' if 'estimated' in qualities else 'measured'


def _comparison(key, unit, book, supplied, quantum, reasons, quality, keys, *, comparable=None):
    reasons = set(reasons)
    if book is None:
        reasons.add('book_value_missing')
    if supplied is None:
        reasons.add('supplier_missing')
    if quantum is not None and supplied is not None and _rounded(supplied, quantum) != supplied:
        reasons.add('supplier_quantum_mismatch')
    if reasons:
        return ComponentComparison(key, unit, book, supplied, None, None, 'not_comparable',
                                   'incomplete', tuple(sorted(reasons)), tuple(dict.fromkeys(keys)))
    comparable = _rounded(book, quantum) if comparable is None else comparable
    delta = _stored_sum((supplied, comparable.copy_negate()))
    return ComponentComparison(key, unit, book, supplied, comparable, delta,
                               'matched' if delta == 0 else 'different', quality, (),
                               tuple(dict.fromkeys(keys)))


def _line_evidence(line, source, meter, generation, runs, conflicts):
    """Validate exact binding; return energy evidence and money-version reasons."""
    if source is None:
        return ('source_missing',), ()
    reasons = set(_codes(source.issues))
    key = _canonical_json(['settlement-line-v1', meter, source.start_ts,
                           source.valid_from_generation, source.run_id, source.line_content_hash])
    identity = (source.start_ts, source.valid_from_generation)
    if (source.source_key != key or identity in conflicts or line.meter_id != meter
            or source.start_ts != line.start_ts or source.end_ts != line.end_ts
            or source.input_revision != line.input_revision or source.input_hash != line.input_hash
            or source.snapshot_hash != line.snapshot_hash or source.line_content_hash != _digest(line)
            or source.valid_from_generation > generation):
        reasons.add('source_mismatch')
    run = runs.get(source.run_id)
    versions = set()
    if run is None:
        reasons.add('source_missing')
    else:
        if (run.generation != source.valid_from_generation
                or source.input_revision > run.input_revision
                or (source.snapshot_hash is not None and source.snapshot_hash not in run.snapshot_hashes)):
            reasons.add('source_mismatch')
        if run.engine_version != 'settlement-v1':
            versions.add('source_version_unsupported')
    item, snapshot = source.item, source.snapshot
    if item is None:
        reasons.add('source_missing')
    elif (item.meter_id != meter or item.start_ts != line.start_ts or item.end_ts != line.end_ts
          or item.input_revision != line.input_revision or item.input_hash != line.input_hash
          or source.input_content_hash != _digest({k: v for k, v in asdict(item).items()
                                                   if k != 'input_revision'})):
        reasons.add('source_mismatch')
    if snapshot is None:
        versions.add('source_missing')
    else:
        if (snapshot.snapshot_hash != source.snapshot_hash
                or snapshot.snapshot_hash != _digest({k: v for k, v in asdict(snapshot).items()
                                                       if k != 'snapshot_hash'})
                or snapshot.valid_from_ts > line.start_ts
                or (snapshot.valid_to_ts is not None and snapshot.valid_to_ts < line.end_ts)):
            versions.add('source_mismatch')
        if snapshot.rule_version != 'tariffs-v1':
            versions.add('source_version_unsupported')
    return tuple(sorted(reasons)), tuple(sorted(versions))


def _projection(line, source, direction):
    item = source.item
    energy = getattr(item, f'{direction}_kwh')
    if energy is None or item.spot_ex_vat is None or source.snapshot is None:
        return None, None, ('source_missing',)
    # quote owns tariff and interval validation and its existing context36.
    try:
        prices = dict(quote(source.snapshot, line.start_ts, line.end_ts, item.spot_ex_vat).components)
    except ValueError:
        return None, None, ('component_projection_unproven',)
    with localcontext() as ctx:
        ctx.prec = 36
        ex = sum((prices[f'{direction}_{part}_ex_vat'] * energy
                  for part in ('spot', 'tax', 'supplier')), Decimal(0))
        vat = prices[f'{direction}_vat'] * energy
        gross = line.import_cost if direction == 'import' else line.base_export_revenue
        if ex + vat != gross:
            return None, None, ('component_projection_unproven',)
    return ex, vat, ()


def reconcile(view: SettlementView, supplier: SupplierPeriod, *, sources: ReconciliationSources) -> ReconciliationReport:
    """Compare explicit supplier observations against the unmodified, pinned book."""
    pin = sources.pin
    global_reasons = set(_codes(sources.issues))
    if (pin.meter_id != view.meter_id or pin.generation != view.generation
            or pin.view_input_revision != view.input_revision
            or pin.current_input_revision != view.current_input_revision
            or pin.dirty_from_ts != view.dirty_from_ts or pin.view_digest != _digest(view)
            or pin.current_generation < pin.generation):
        global_reasons.add('source_mismatch')
    for mismatch, reason in (
        (supplier.meter_id != view.meter_id, 'meter_mismatch'),
        ((supplier.start_ts, supplier.end_ts) != (pin.start_ts, pin.end_ts), 'period_mismatch'),
        (supplier.currency != 'EUR', 'currency_unsupported'),
        (supplier.energy_unit != 'kWh', 'energy_unit_unsupported'),
        (supplier.sign_convention != 'cost_positive_revenue_positive', 'sign_convention_unsupported'),
        (supplier.total_scope != 'book_close_events_v1', 'total_scope_mismatch'),
    ):
        if mismatch:
            global_reasons.add(reason)
    ordered = sorted(view.lines, key=lambda line: line.start_ts)
    if (not ordered or not view.complete or ordered[0].start_ts != pin.start_ts
            or ordered[-1].end_ts != pin.end_ts
            or any(a.end_ts != b.start_ts for a, b in zip(ordered, ordered[1:], strict=False))):
        global_reasons.add('coverage_incomplete')
    runs = {}
    for run in sources.runs:
        if run.run_id in runs and runs[run.run_id] != run:
            global_reasons.add('source_mismatch')
        runs[run.run_id] = run
    selected_runs = [run for run in sources.runs if run.generation == view.generation]
    if not selected_runs:
        global_reasons.add('source_missing')
    elif len(selected_runs) != 1 or selected_runs[0].input_revision > view.input_revision:
        # A clean no-op may lift the view revision above the stored run revision; those
        # identities stay distinct. Only a run above the view revision is a real conflict.
        global_reasons.add('source_mismatch')
    all_lines = (*sources.lines, *(c.closing_source for c in sources.credits if c.closing_source is not None))
    versions, conflicts = {}, set()
    for source in all_lines:
        identity = (source.start_ts, source.valid_from_generation)
        if identity in versions and versions[identity] != source:
            conflicts.add(identity)
        versions[identity] = source
    contracts = {s.snapshot.contract_id for s in sources.lines if s.snapshot is not None}
    zones = {s.snapshot.timezone for s in sources.lines if s.snapshot is not None}
    if len(contracts) > 1:
        global_reasons.add('contract_scope_unsupported')
    elif contracts and contracts != {supplier.contract_id}:
        global_reasons.add('contract_mismatch')
    if zones and zones != {supplier.timezone}:
        global_reasons.add('timezone_mismatch')
    rounding_reasons = _rounding_reasons(supplier.rounding)
    base_reasons = global_reasons | set(rounding_reasons)
    money_quantum = Decimal('0.01') if not rounding_reasons else None
    energy_quantum = (next(q for q in _ENERGY_QUANTA if q == supplier.rounding.energy_quantum)
                      if not rounding_reasons else None)
    line_data = []
    used = set()
    for line in view.lines:
        candidates = [s for s in sources.lines if (s.start_ts, s.end_ts) == (line.start_ts, line.end_ts)]
        source = candidates[0] if len(candidates) == 1 else None
        energy_reasons, money_reasons = _line_evidence(line, source, view.meter_id, view.generation, runs, conflicts)
        if len(candidates) > 1:
            energy_reasons = ('source_mismatch',)
        if source is not None:
            used.add(source.source_key)
        line_data.append((line, source, energy_reasons, money_reasons))
    if any(s.source_key not in used for s in sources.lines):
        base_reasons.add('source_mismatch')
    keys = tuple(s.source_key for _, s, _, _ in line_data if s is not None)
    line_quality = _quality(
        [line.quality for line in view.lines]
        + [source.item.quality if source and source.item else 'incomplete'
           for _, source, _, _ in line_data]
    )
    components = []
    for direction in ('import', 'export'):
        reasons = set(base_reasons)
        values, qualities = [], []
        for _line, source, energy_reasons, _ in line_data:
            reasons.update(energy_reasons)
            values.append(getattr(source.item, f'{direction}_kwh') if source and source.item else None)
            qualities.append(source.item.quality if source and source.item else 'incomplete')
        components.append(_comparison(f'{direction}_kwh', 'kWh', _stored_sum(values),
                                      getattr(supplier, f'{direction}_kwh'), energy_quantum,
                                      reasons, _quality(qualities), keys))
    for name, field, amounts in (
        ('import', 'import_cost', supplier.import_amounts),
        ('base_export', 'base_export_revenue', supplier.base_export_amounts),
        ('bonus', 'bonus_revenue', supplier.bonus_amounts),
    ):
        money_reasons = set(base_reasons)
        for _, _, energy_issues, money_issues in line_data:
            money_reasons.update((*energy_issues, *money_issues))
        gross = _stored_sum(getattr(line, field) for line in view.lines)
        components.append(_comparison(f'{name}.gross', 'EUR', gross,
                                      amounts.gross if amounts else None, money_quantum,
                                      money_reasons, line_quality, keys))
        ex_values, vat_values = [], []
        detail_reasons = set(money_reasons)
        for line, source, energy_issues, money_issues in line_data:
            if energy_issues or money_issues:
                ex, vat = None, None
            elif name == 'bonus':
                if line.bonus_revenue == 0 and (source.snapshot.bonus is None or not source.snapshot.bonus.enabled):
                    ex, vat = Decimal(0), Decimal(0)
                else:
                    ex, vat = None, None
                    detail_reasons.add('bonus_vat_split_unavailable')
            else:
                ex, vat, issues = _projection(line, source, 'import' if name == 'import' else 'export')
                detail_reasons.update(issues)
            ex_values.append(ex)
            vat_values.append(vat)
        for part, values in (('ex_vat', ex_values), ('vat', vat_values)):
            components.append(_comparison(f'{name}.{part}', 'EUR', _stored_sum(values),
                                          getattr(amounts, part) if amounts else None, money_quantum,
                                          detail_reasons, line_quality, keys))
    # Union by public credit identity; contract/zone proof is checked separately.
    def identity(credit):
        return credit.kind, credit.period_start_ts, credit.period_end_ts

    book_credits = {}
    for adjustment in view.adjustments:
        ident = identity(adjustment)
        if ident in book_credits:
            base_reasons.add('source_mismatch')
        book_credits[ident] = adjustment
    supplier_credits = {identity(c): c for c in supplier.credits or ()}
    credit_set_reasons = set()
    if supplier.credits is None:
        credit_set_reasons.add('supplier_missing')
    elif set(book_credits) != set(supplier_credits):
        credit_set_reasons.add('credit_set_mismatch')
    credit_keys = []
    for ident in sorted(set(book_credits) | set(supplier_credits)):
        adjustment, observation = book_credits.get(ident), supplier_credits.get(ident)
        candidates = [s for s in sources.credits if identity(s) == ident]
        source = candidates[0] if len(candidates) == 1 else None
        reasons = set(base_reasons)
        row_keys = []
        if ident[0] != 'spot_zero':
            reasons.add('credit_kind_unsupported')
        if not pin.start_ts < ident[2] <= pin.end_ts:
            reasons.add('credit_scope_mismatch')
        if adjustment is None:
            reasons.add('credit_identity_unproven')
        if source is None:
            reasons.add('source_missing' if not candidates else 'source_mismatch')
            reasons.add('credit_identity_unproven')
        else:
            row_keys.append(source.source_key)
            reasons.update(_codes(source.issues))
            expected = _canonical_json(['settlement-credit-v1', view.meter_id, *ident,
                                        source.valid_from_generation, source.run_id,
                                        source.adjustment_content_hash])
            run = runs.get(source.run_id)
            if (source.source_key != expected or source.valid_from_generation > view.generation
                    or adjustment is None or source.adjustment_content_hash != _digest(adjustment)):
                reasons.add('source_mismatch')
            if run is None:
                reasons.add('source_missing')
            elif run.generation != source.valid_from_generation:
                reasons.add('source_mismatch')
            elif run.engine_version != 'settlement-v1':
                reasons.add('source_version_unsupported')
            closing, bound = source.closing_line, source.closing_source
            if closing is None or bound is None or run is None:
                reasons.add('credit_identity_unproven')
            else:
                row_keys.append(bound.source_key)
                energy_issues, money_issues = _line_evidence(closing, bound, view.meter_id,
                                                            run.generation, runs, conflicts)
                reasons.update((*energy_issues, *money_issues))
                month = closing.after_state.month
                if (energy_issues or money_issues or closing.end_ts != ident[2]
                        or closing.after_state.asof_ts != closing.end_ts
                        or bound.input_revision > run.input_revision
                        or month is None or (month.period_start_ts, month.period_end_ts) != ident[1:]
                        or month.contract_id != source.contract_id
                        or bound.snapshot is None or bound.snapshot.contract_id != source.contract_id
                        or bound.snapshot.timezone != source.timezone):
                    reasons.add('credit_identity_unproven')
            if source.contract_id is None or source.timezone is None:
                reasons.add('credit_identity_unproven')
            if source.contract_id != supplier.contract_id:
                reasons.add('contract_mismatch')
            if source.timezone != supplier.timezone:
                reasons.add('timezone_mismatch')
        if adjustment is not None:
            with localcontext() as ctx:
                ctx.prec = 36
                if adjustment.credit_ex_vat + adjustment.credit_vat != adjustment.credit_total:
                    reasons.add('source_mismatch')
        credit_keys.extend(row_keys)
        for part, field in (('gross', 'credit_total'), ('ex_vat', 'credit_ex_vat'), ('vat', 'credit_vat')):
            components.append(_comparison(f'credit:{ident[0]}:{ident[1]}:{ident[2]}.{part}', 'EUR',
                                          getattr(adjustment, field) if adjustment else None,
                                          getattr(observation.amounts, part) if observation else None,
                                          money_quantum, reasons,
                                          adjustment.quality if adjustment else 'incomplete', row_keys))
    if any(identity(s) not in book_credits for s in sources.credits):
        credit_set_reasons.add('source_mismatch')
    total_reasons = base_reasons | credit_set_reasons
    if view.freshness == 'unknown' or view.generation == 0:
        total_reasons.add('book_unknown')
    gross_rows = [c for c in components if c.key.endswith('.gross')]
    for row in gross_rows:
        if row.status == 'not_comparable':
            total_reasons.update(row.reasons)
    comparable = None
    if not total_reasons:
        comparable = _stored_sum(row.comparable_book_value if row.key == 'import.gross'
                                  else row.comparable_book_value.copy_negate() for row in gross_rows)
        supplier_sum = _stored_sum(row.supplier_value if row.key == 'import.gross'
                                    else row.supplier_value.copy_negate() for row in gross_rows)
        if supplier.total is not None and supplier.total != supplier_sum:
            total_reasons.add('supplier_total_inconsistent')
    quality = _quality([line_quality] + [c.quality for c in view.adjustments])
    total = _comparison('net_cost', 'EUR', view.net_cost, supplier.total, money_quantum,
                        total_reasons, quality, (*keys, *credit_keys), comparable=comparable)
    report_reasons = set(global_reasons)
    if view.freshness != 'current':
        report_reasons.add('book_outdated' if view.freshness == 'outdated' else 'book_unknown')
    rows = (*components, total)
    if any(c.status == 'not_comparable' for c in rows):
        report_reasons.add('component_not_comparable')
    if any(c.status == 'different' for c in rows):
        report_reasons.add('component_different')
    status = ('not_comparable' if view.freshness != 'current' or global_reasons
              or any(c.status == 'not_comparable' for c in rows) else
              'different' if any(c.status == 'different' for c in rows) else 'matched')
    return ReconciliationReport('reconciliation-report-v1', pin, supplier, _digest(supplier),
                                'registered_book_and_caller_supplied_supplier', view.freshness,
                                tuple(components), total, status, tuple(sorted(report_reasons)), sources)
