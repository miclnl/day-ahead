"""Pure chronologische afrekening zonder database, HA of solver."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, time
from decimal import Decimal, localcontext
from zoneinfo import ZoneInfo

from dao.prog.settlement.tariffs import (
    _check_bonus_boundaries,
    bonus_in_window,
    quote,
    resolve_snapshot,
)
from dao.prog.settlement.types import (
    BonusState,
    InputInterval,
    PriceQuote,
    SettlementLine,
    SettlementResult,
    SettlementState,
    TariffSnapshot,
)

ENGINE_VERSION = "settlement-v1"
ZERO = Decimal("0")


def _anniversary(anchor: date, year: int) -> date:
    try:
        return anchor.replace(year=year)
    except ValueError:  # 29 februari in een niet-schrikkeljaar
        return date(year, 2, 28)


def delivery_year_bounds(anchor: date, day: date) -> tuple[date, date]:
    """Jaargrenzen op maand/dag; de kern toetst daarnaast het echte leveringsbegin."""
    start = _anniversary(anchor, day.year)
    if start > day:
        start = _anniversary(anchor, day.year - 1)
    return start, _anniversary(anchor, start.year + 1)


def remaining_export_cap(cap: Decimal | None, exported: Decimal | None) -> Decimal | None:
    """None blijft onbekend/expliciet onbegrensd; nooit omzetten naar vrije ruimte."""
    if cap is None or exported is None:
        return None
    with localcontext() as context:
        context.prec = 36
        return max(ZERO, cap - exported)


def _bonus_period(snapshot: TariffSnapshot, at: int) -> tuple[int, int] | None:
    rules = snapshot.bonus
    if rules is None or not rules.enabled or snapshot.supply_start is None:
        return None
    zone = ZoneInfo(snapshot.timezone)
    day = datetime.fromtimestamp(at, zone).date()
    start, end = delivery_year_bounds(snapshot.supply_start, day)
    start = max(start, rules.start_date, snapshot.supply_start)
    if day < start:
        return None
    return (int(datetime.combine(start, time.min, zone).timestamp()),
            int(datetime.combine(end, time.min, zone).timestamp()))


def _opening_bonus(snapshot: TariffSnapshot, start: int, state: SettlementState) -> BonusState | None:
    period = _bonus_period(snapshot, start)
    if period is None:
        return None
    previous = state.bonus
    if start == period[0]:
        total = ZERO
    elif (state.asof_ts == start and previous is not None and previous.asof_ts == start
          and (previous.period_start_ts, previous.period_end_ts) == period):
        total = previous.total_export_kwh
    else:
        total = None
    return BonusState(*period, start, total)


def _advance_bonus(
    item: InputInterval, snapshot: TariffSnapshot | None, state: SettlementState,
    before: BonusState | None, issues: list[str],
) -> BonusState | None:
    if snapshot is None:
        return (None if state.bonus is None else
                replace(state.bonus, asof_ts=item.end_ts, total_export_kwh=None))
    period = _bonus_period(snapshot, item.end_ts)
    if period is None:
        return None
    if item.end_ts == period[0]:
        # Het nieuwe jaar begint exact hier; onbekend oud jaar is niet besmettelijk.
        return BonusState(*period, item.end_ts, ZERO)
    if before is None or (before.period_start_ts, before.period_end_ts) != period:
        issues.append("bonus_boundary_requires_split")
        return BonusState(*period, item.end_ts, None)
    total = before.total_export_kwh
    if total is not None and item.export_kwh is not None:
        total += item.export_kwh
    else:
        total = None
    return BonusState(*period, item.end_ts, total)


def _bonus_amounts(
    item: InputInterval, snapshot: TariffSnapshot | None, price: PriceQuote | None,
    before: BonusState | None, issues: list[str],
) -> tuple[Decimal | None, Decimal | None]:
    if snapshot is None:
        return None, None
    rules = snapshot.bonus
    if rules is None or not rules.enabled:
        return ZERO, ZERO
    if ((item.spot_ex_vat is not None and item.spot_ex_vat <= 0)
        or item.export_kwh == 0
        or (item.battery_discharge_kwh is not None and item.battery_discharge_kwh > 0)):
        return ZERO, ZERO
    if "bonus_boundary_requires_split" in issues:
        return None, None
    remaining = remaining_export_cap(rules.yearly_cap_kwh, None if before is None else before.total_export_kwh)
    if remaining == 0:
        return ZERO, ZERO
    if price is None:
        try:
            _check_bonus_boundaries(snapshot, item.start_ts, item.end_ts)
        except ValueError:
            issues.append("bonus_boundary_requires_split")
            return None, None
    moment = datetime.fromtimestamp(item.start_ts, ZoneInfo(snapshot.timezone))
    if (moment.date() < rules.start_date
        or (snapshot.supply_start is not None and moment.date() < snapshot.supply_start)
        or not bonus_in_window(rules.windows, moment)):
        return ZERO, ZERO
    unknown = False
    if snapshot.supply_start is None:
        issues.append("unknown_supply_start")
        unknown = True
    if before is None or before.total_export_kwh is None:
        issues.append("unknown_bonus_history")
        unknown = True
    if rules.cap_basis == "unconfirmed" or rules.vat_percent is None:
        issues.append("unconfirmed_bonus_basis")
        unknown = True
    if item.origin != "verified_pv" or item.battery_discharge_kwh is None:
        issues.append("unknown_bonus_source")
        unknown = True
    if item.end_ts - item.start_ts != 900 or item.start_ts % 900:
        issues.append("bonus_requires_quarter")
        unknown = True
    if unknown or item.export_kwh is None or price is None:
        return None, None
    first = item.export_kwh if rules.yearly_cap_kwh is None else min(item.export_kwh, remaining)
    return first, first * price.bonus_rate


def _settle(
    item: InputInterval, snapshot: TariffSnapshot | None, price: PriceQuote | None,
    state: SettlementState, issues: list[str],
) -> tuple[SettlementLine, SettlementState]:
    with localcontext() as context:
        context.prec = 36
        if item.start_ts > state.asof_ts:
            issues.append("export_gap")
        if item.import_kwh is None:
            issues.append("missing_import")
        if item.export_kwh is None:
            issues.append("missing_export")
        before = None if snapshot is None else _opening_bonus(snapshot, item.start_ts, state)
        after_bonus = _advance_bonus(item, snapshot, state, before, issues)
        bonus_kwh, bonus_revenue = _bonus_amounts(item, snapshot, price, before, issues)
        import_cost = None if price is None or item.import_kwh is None else item.import_kwh * price.import_rate
        export_revenue = None if price is None or item.export_kwh is None else item.export_kwh * price.export_rate
        # Maandaccumulatie/-afsluiting volgt in A5; hier geen verborgen maandcorrectie.
        after = SettlementState(item.end_ts, after_bonus, state.month)
        quality = item.quality
        if issues or any(amount is None for amount in (import_cost, export_revenue, bonus_revenue)):
            quality = "incomplete"
        line = SettlementLine(
            start_ts=item.start_ts, end_ts=item.end_ts, meter_id=item.meter_id,
            input_revision=item.input_revision, input_hash=item.input_hash,
            snapshot_hash=None if snapshot is None else snapshot.snapshot_hash,
            import_cost=import_cost, base_export_revenue=export_revenue,
            bonus_revenue=bonus_revenue, bonus_kwh=bonus_kwh,
            before_state=state, after_state=after, quality=quality, issues=tuple(issues),
        )
    return line, after


def _check_chronology(items: Sequence[InputInterval], state: SettlementState):
    if not items:
        raise ValueError("Settlement requires at least one covered interval")
    end = state.asof_ts
    meter_id = items[0].meter_id
    for item in items:
        if item.start_ts < end:
            raise ValueError("Input must be ordered, non-overlapping and not before the opening state")
        if item.meter_id != meter_id:
            raise ValueError("One settlement series must refer to one explicit meter")
        end = item.end_ts


def settle_interval(
    item: InputInterval, price: PriceQuote, state: SettlementState,
) -> tuple[SettlementLine, SettlementState]:
    """Eén interval; prijzen blijven gescheiden van hoeveelheden en bonusrechten."""
    _check_chronology((item,), state)
    if price.start_ts > item.start_ts or price.end_ts < item.end_ts or price.spot_ex_vat != item.spot_ex_vat:
        raise ValueError("Quote must cover the input interval and match its known spot price")
    resolve_snapshot((price.snapshot,), item.start_ts, item.end_ts)
    return _settle(item, price.snapshot, price, state, [])


def settle_series(
    items: Sequence[InputInterval], snapshots: Sequence[TariffSnapshot], opening: SettlementState,
) -> SettlementResult:
    """Verwerk ongewijzigde invoervolgorde; gaten maken caphistoriek onbekend.

    Onbekende prijzen worden geen nultarief. Revisie 0 uit merge_inputs blijft
    ongeregistreerde invoer; deze pure berekening registreert/publiceert niets.
    """
    _check_chronology(items, opening)
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.valid_from_ts)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.valid_to_ts is None or previous.valid_to_ts > current.valid_from_ts:
            raise ValueError("Tariff snapshots must not overlap")
    lines, state = [], opening
    with localcontext() as context:
        context.prec = 36
        subtotal = ZERO
        complete = True
        for item in items:
            snapshot, price, issues = None, None, []
            try:
                snapshot = resolve_snapshot(snapshots, item.start_ts, item.end_ts)
            except ValueError as exc:
                issues.append(f"unavailable_snapshot: {exc}")
            if item.spot_ex_vat is None:
                issues.append("missing_spot")
            elif snapshot is not None:
                try:
                    price = quote(snapshot, item.start_ts, item.end_ts, item.spot_ex_vat)
                except ValueError as exc:
                    issues.append(f"unavailable_quote: {exc}")
            line, state = _settle(item, snapshot, price, state, issues)
            lines.append(line)
            complete &= line.quality != "incomplete"
            for amount, sign in ((line.import_cost, 1), (line.base_export_revenue, -1), (line.bonus_revenue, -1)):
                if amount is not None:
                    subtotal += sign * amount
        return SettlementResult(
            lines=tuple(lines), adjustments=(), end_state=state, engine_version=ENGINE_VERSION,
            coverage_start_ts=opening.asof_ts, coverage_end_ts=state.asof_ts,
            complete=complete, known_subtotal=subtotal, net_cost=subtotal if complete else None,
        )
