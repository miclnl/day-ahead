"""Pure chronologische afrekening zonder database, HA of solver."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, time
from decimal import Decimal, localcontext
from zoneinfo import ZoneInfo

from dao.prog.settlement.tariffs import (
    _check_bonus_boundaries,
    _effective_value,
    bonus_in_window,
    quote,
    resolve_snapshot,
)
from dao.prog.settlement.types import (
    BonusState,
    InputInterval,
    MonthState,
    PeriodAdjustment,
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


def close_month(state: MonthState) -> PeriodAdjustment | None:
    """Sluit een gedekte periode; de aanroeper bewaakt het bereiken van het einde.

    De positieve negatieve-grondslaggewichten verdelen uitsluitend de credit-btw.
    Deze toerekening blijft een modelraming, geen bevestigde leveranciersnota.
    Ook een bekende nulcredit is een expliciet sluitresultaat. Er wordt niet
    afgerond; presentatie selecteert deze correctie op period_end_ts - 1.
    """
    if not state.complete or state.spot_export_eur is None:
        return None
    with localcontext() as context:
        context.prec = 36
        credit = max(ZERO, -state.spot_export_eur)
        vat = ZERO
        if credit > 0:
            negative_basis = sum((basis for _, basis in state.negative_basis_by_vat), ZERO)
            if negative_basis <= 0:
                return None  # Zonder btwgrondslag geen verzonnen nul-btw.
            vat = sum((credit * basis / negative_basis * rate / 100
                       for rate, basis in state.negative_basis_by_vat), ZERO)
        return PeriodAdjustment(
            kind="spot_zero", period_start_ts=state.period_start_ts, period_end_ts=state.period_end_ts,
            credit_ex_vat=credit, credit_vat=vat, credit_total=credit + vat,
            basis_ex_vat=state.spot_export_eur, quality="estimated",
        )


def _contract_periods(snapshots: Sequence[TariffSnapshot]) -> dict[int, tuple[int | None, int | None]]:
    """Alleen een andere contract-id bewijst een wissel; een nieuwe hash niet."""
    periods = {}
    first = 0
    while first < len(snapshots):
        last = first
        while last + 1 < len(snapshots) and snapshots[last + 1].contract_id == snapshots[first].contract_id:
            last += 1
        start = snapshots[first].valid_from_ts if first > 0 else None
        end = snapshots[last].valid_to_ts if last + 1 < len(snapshots) else None
        for snapshot in snapshots[first:last + 1]:
            periods[snapshot.valid_from_ts] = (start, end)
        first = last + 1
    return periods


def _month_period(
    snapshot: TariffSnapshot, at: int, previous: MonthState | None,
    contract_bounds: tuple[int | None, int | None],
) -> tuple[int, int]:
    zone = ZoneInfo(snapshot.timezone)
    day = datetime.fromtimestamp(at, zone).date()
    first = day.replace(day=1)
    following = date(first.year + 1, 1, 1) if first.month == 12 else date(first.year, first.month + 1, 1)
    start = int(datetime.combine(first, time.min, zone).timestamp())
    end = int(datetime.combine(following, time.min, zone).timestamp())
    contract_start, contract_end = contract_bounds
    if contract_start is not None:
        start = max(start, contract_start)
    if contract_end is not None:
        end = min(end, contract_end)
    if snapshot.supply_start is not None:
        supply_start = int(datetime.combine(snapshot.supply_start, time.min, zone).timestamp())
        if supply_start <= at:
            start = max(start, supply_start)
    if previous is not None:
        if previous.contract_id != snapshot.contract_id:
            start = max(start, snapshot.valid_from_ts)
        elif start <= previous.period_start_ts <= at < previous.period_end_ts:
            # Behoud een eerder bewezen contractdeel, ook bij hervatten met minder snapshots.
            start = previous.period_start_ts
            end = min(end, previous.period_end_ts)
    return start, end


def _advance_month(
    item: InputInterval, snapshot: TariffSnapshot | None, state: SettlementState,
    issues: list[str], contract_bounds: tuple[int | None, int | None],
) -> MonthState | None:
    """Complete bewaakt de gedekte prefix tot asof; alleen het periode-einde sluit."""
    if snapshot is None:
        return None if state.month is None else replace(state.month, spot_export_eur=None, complete=False)
    if snapshot.monthly_export_floor == "none":
        return None
    start, end = _month_period(snapshot, item.start_ts, state.month, contract_bounds)
    if item.end_ts > end:
        issues.append("month_boundary_requires_split")
        start, end = _month_period(snapshot, item.end_ts - 1, None, contract_bounds)
        return MonthState(snapshot.contract_id, start, end, None, (), False)
    previous = state.month
    if item.start_ts == start:
        month = MonthState(snapshot.contract_id, start, end, ZERO, (), True)
    elif (state.asof_ts == item.start_ts and previous is not None
          and previous.contract_id == snapshot.contract_id and previous.period_start_ts == start
          and item.start_ts < previous.period_end_ts):
        month = replace(previous, period_end_ts=end)
    else:
        month = MonthState(snapshot.contract_id, start, end, None, (), False)
    amount = None if item.export_kwh is None or item.spot_ex_vat is None else item.export_kwh * item.spot_ex_vat
    total = None if month.spot_export_eur is None or amount is None else month.spot_export_eur + amount
    groups = dict(month.negative_basis_by_vat)
    if amount is not None and amount < 0:
        day = datetime.fromtimestamp(item.start_ts, ZoneInfo(snapshot.timezone)).date()
        vat = _effective_value(snapshot.vat_production, day)
        groups[vat] = groups.get(vat, ZERO) - amount
    complete = month.complete and total is not None and item.quality != "incomplete"
    if not complete:
        issues.append("incomplete_month")
    return MonthState(snapshot.contract_id, start, end, total, tuple(sorted(groups.items())), complete)


def _settle(
    item: InputInterval, snapshot: TariffSnapshot | None, price: PriceQuote | None,
    state: SettlementState, issues: list[str],
    contract_bounds: tuple[int | None, int | None] = (None, None),
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
        after_month = _advance_month(item, snapshot, state, issues, contract_bounds)
        after = SettlementState(item.end_ts, after_bonus, after_month)
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
    Dezelfde snapshotselectie blijft bij het opsplitsen van een serie gepind:
    aangrenzende contract-id's bepalen de afzonderlijke leveringsperioden.
    Een open maand blijft uitsluitend in de maandstaat, niet in de correcties.
    """
    _check_chronology(items, opening)
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.valid_from_ts)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.valid_to_ts is None or previous.valid_to_ts > current.valid_from_ts:
            raise ValueError("Tariff snapshots must not overlap")
    periods = _contract_periods(ordered)
    lines, adjustments, state = [], [], opening
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
            bounds = (None, None) if snapshot is None else periods[snapshot.valid_from_ts]
            line, state = _settle(item, snapshot, price, state, issues, bounds)
            lines.append(line)
            complete &= line.quality != "incomplete"
            for amount, sign in ((line.import_cost, 1), (line.base_export_revenue, -1), (line.bonus_revenue, -1)):
                if amount is not None:
                    subtotal += sign * amount
            month = state.month
            if month is not None and item.end_ts == month.period_end_ts:
                adjustment = close_month(month)
                if adjustment is None:
                    complete = False
                else:
                    adjustments.append(adjustment)
                    subtotal -= adjustment.credit_total
        return SettlementResult(
            lines=tuple(lines), adjustments=tuple(adjustments), end_state=state, engine_version=ENGINE_VERSION,
            coverage_start_ts=opening.asof_ts, coverage_end_ts=state.asof_ts,
            complete=complete, known_subtotal=subtotal, net_cost=subtotal if complete else None,
        )
