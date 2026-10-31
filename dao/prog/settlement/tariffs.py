"""Pure tariefsnapshots en ongeronde prijzen; geen database, HA of solver."""

import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal, localcontext
from hashlib import sha256
from zoneinfo import ZoneInfo

from dao.prog.config.models.pricing import PricingConfig
from dao.prog.config.models.production_bonus import ProductionBonusConfig
from dao.prog.config.models.settlement import SettlementContractConfig
from dao.prog.settlement.types import TARIFF_FIELDS, BonusRules, PriceQuote, TariffSnapshot

RULE_VERSION = "tariffs-v1"


def bonus_rules_from_config(cfg: ProductionBonusConfig | None) -> BonusRules | None:
    """Kopieer alleen financiële bonusregels, nooit vrije configvelden."""
    if cfg is None:
        return None
    return BonusRules(
        enabled=cfg.enabled, mode=cfg.mode, start_date=cfg.start_date,
        percentage=Decimal(str(cfg.percentage)),
        max_exchange_price=Decimal(str(cfg.max_exchange_price)),
        cap_basis=cfg.cap_basis, vat_percent=cfg.vat_percent,
        yearly_cap_kwh=None if cfg.yearly_cap_kwh is None else Decimal(str(cfg.yearly_cap_kwh)),
        windows=tuple((month, tuple(cfg.windows[str(month)])) for month in range(1, 13)),
    )


def _canonical_default(value):
    if isinstance(value, Decimal):
        if value == 0:
            return "0"
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Unsupported financial JSON type: {type(value).__name__}")


def _midnight(day: date, zone: ZoneInfo) -> int:
    return int(datetime.combine(day, time.min, zone).timestamp())


def make_snapshot(pricing: PricingConfig, contract: SettlementContractConfig) -> TariffSnapshot:
    """Canoniseer de financiële allowlist; contract-id/leveringsbegin blijven gelijk.

    Ook zonder prices.settlement gelden de A1-validatieregels voor het gekozen
    contract. Legacy floats worden via decimale tekst vastgelegd; verloren
    bronprecisie wordt daarmee niet teruggewonnen. Bronversies blijven leeg
    zolang de aanroeper geen bewijsreferenties heeft geregistreerd.
    """
    contract = SettlementContractConfig.model_validate(contract.model_dump())
    financial = {name: getattr(pricing, name) for name in TARIFF_FIELDS
                 if name in pricing.model_fields_set}
    validated = PricingConfig.model_validate({
        **financial, "last_invoice": pricing.last_invoice,
        "tax_refund": pricing.tax_refund, "production_bonus": pricing.production_bonus,
        "settlement": {"contracts": [contract]},
    })
    tables = {}
    for name in TARIFF_FIELDS:
        table = getattr(validated, name)
        if not table:
            raise ValueError(f"{name}: no dated tariff available")
        tables[name] = tuple(sorted((date.fromisoformat(day), Decimal(str(value)))
                                    for day, value in table.items()))
    if contract.product == "legacy" and not validated.tax_refund:
        # Geen saldering: alleen de belastingteruggave vervalt, niet de spotfactor.
        tables["energy_taxes_production"] = tuple(
            (day, Decimal("0")) for day, _ in tables["energy_taxes_production"]
        )
    zone = ZoneInfo(contract.timezone)
    snapshot = TariffSnapshot(
        snapshot_hash="", rule_version=RULE_VERSION, contract_id=contract.id,
        status=contract.status, valid_from_ts=_midnight(contract.valid_from, zone),
        valid_to_ts=None if contract.valid_to is None else _midnight(contract.valid_to, zone),
        supply_start=contract.supply_start, timezone=contract.timezone,
        product=contract.product, regime=contract.regime,
        price_interval_seconds=contract.price_interval_seconds,
        meter_interval_seconds=contract.meter_interval_seconds,
        hour_price_rule=contract.hour_price_rule, monthly_export_floor=contract.monthly_export_floor,
        **tables, bonus=bonus_rules_from_config(validated.production_bonus), source_versions=(),
    )
    content = asdict(snapshot)
    del content["snapshot_hash"]
    canonical = json.dumps(content, default=_canonical_default, sort_keys=True, separators=(",", ":"))
    return replace(snapshot, snapshot_hash=sha256(canonical.encode("utf-8")).hexdigest())


def _check_interval(start_ts: int, end_ts: int):
    if type(start_ts) is not int or type(end_ts) is not int:
        raise TypeError("Interval boundaries must be integer UTC epochs")
    if end_ts <= start_ts:
        raise ValueError("Intervals must have positive duration and are half-open")


def _effective_value(table, day: date) -> Decimal:
    for effective, value in reversed(table):
        if effective <= day:
            return value
    raise ValueError("No tariff available on the requested local date")


def _check_snapshot_interval(snapshot: TariffSnapshot, start_ts: int, end_ts: int):
    _check_interval(start_ts, end_ts)
    if start_ts < snapshot.valid_from_ts or (snapshot.valid_to_ts is not None and end_ts > snapshot.valid_to_ts):
        raise ValueError("Interval is not covered by one snapshot; split at the contract boundary")
    zone = ZoneInfo(snapshot.timezone)
    day = datetime.fromtimestamp(start_ts, zone).date()
    for name in TARIFF_FIELDS:
        table = getattr(snapshot, name)
        _effective_value(table, day)
        if any(start_ts < _midnight(effective, zone) < end_ts for effective, _ in table):
            raise ValueError("Interval crosses a tariff change; split before resolving/quoting")


def resolve_snapshot(
    snapshots: Sequence[TariffSnapshot], start_ts: int, end_ts: int,
) -> TariffSnapshot:
    """Eén volledige dekking, nooit alleen de snapshot van de eerste datum."""
    _check_interval(start_ts, end_ts)
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.valid_from_ts)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.valid_to_ts is None or previous.valid_to_ts > current.valid_from_ts:
            raise ValueError("Tariff snapshots must not overlap")
    for snapshot in ordered:
        if snapshot.valid_from_ts <= start_ts and (snapshot.valid_to_ts is None or end_ts <= snapshot.valid_to_ts):
            _check_snapshot_interval(snapshot, start_ts, end_ts)
            return snapshot
    raise ValueError("No single snapshot covers the complete interval; missing terms or split required")


def bonus_in_window(windows: Sequence[tuple[int, tuple[int, int]]], moment: datetime) -> bool:
    """Halfopen lokaal bonusvenster, gedeeld met de legacy-adapter en kern."""
    start, end = dict(windows)[moment.month]
    return start <= moment.hour < end


def bonus_rate_at(
    moment: datetime, spot_ex_vat: Decimal, *, enabled: bool, start_date: date,
    percentage: Decimal, max_exchange_price: Decimal, cap_basis: str,
    vat_percent: Decimal | None, windows: Sequence[tuple[int, tuple[int, int]]],
) -> Decimal:
    """Eén tariefregel, ook voor legacy-invoer zonder nieuwe DTO-precisie-eisen."""
    if not enabled or moment.date() < start_date or spot_ex_vat <= 0:
        return Decimal("0")
    if not bonus_in_window(windows, moment) or cap_basis == "unconfirmed" or vat_percent is None:
        return Decimal("0")
    with localcontext() as ctx:
        ctx.prec = 36
        vat_factor = 1 + vat_percent / 100
        if cap_basis == "inc_vat":
            return min(spot_ex_vat * vat_factor, max_exchange_price) * percentage / 100
        return min(spot_ex_vat, max_exchange_price) * percentage / 100 * vat_factor


def potential_bonus_rate(bonus: BonusRules | None, moment: datetime, spot_ex_vat: Decimal) -> Decimal:
    """Potentieel tarief; geen bewijs van bron, resterende cap of uitbetaling."""
    if bonus is None:
        return Decimal("0")
    return bonus_rate_at(
        moment, spot_ex_vat, enabled=bonus.enabled, start_date=bonus.start_date,
        percentage=bonus.percentage, max_exchange_price=bonus.max_exchange_price,
        cap_basis=bonus.cap_basis, vat_percent=bonus.vat_percent, windows=bonus.windows,
    )


def _check_bonus_boundaries(snapshot: TariffSnapshot, start_ts: int, end_ts: int):
    bonus = snapshot.bonus
    if bonus is None or not bonus.enabled:
        return
    zone = ZoneInfo(snapshot.timezone)
    for boundary in (bonus.start_date, snapshot.supply_start):
        if boundary is not None and start_ts < _midnight(boundary, zone) < end_ts:
            raise ValueError("Interval crosses bonus/supply start; split before quoting")
    day = datetime.fromtimestamp(start_ts, zone).date()
    last_day = datetime.fromtimestamp(end_ts - 1, zone).date()
    windows = dict(bonus.windows)
    while day <= last_day:
        for hour in windows[day.month]:
            boundary_day = day + timedelta(days=1) if hour == 24 else day
            boundary = int(datetime.combine(boundary_day, time(hour % 24), zone).timestamp())
            if start_ts < boundary < end_ts:
                raise ValueError("Interval crosses a bonus window; split before quoting")
        day += timedelta(days=1)


def quote(snapshot: TariffSnapshot, start_ts: int, end_ts: int, spot_ex_vat: Decimal) -> PriceQuote:
    """Ongeronde basistarieven en losse potentiële bonus voor één prijsinterval."""
    _check_snapshot_interval(snapshot, start_ts, end_ts)
    if not isinstance(spot_ex_vat, Decimal):
        raise TypeError("Spot price must be Decimal from text, not missing or a float")
    if not spot_ex_vat.is_finite() or spot_ex_vat.as_tuple().exponent < -12:
        raise ValueError("Spot price must be finite with at most 12 decimal places")
    _check_bonus_boundaries(snapshot, start_ts, end_ts)
    moment = datetime.fromtimestamp(start_ts, ZoneInfo(snapshot.timezone))
    components, rates = [], []
    with localcontext() as ctx:
        ctx.prec = 36
        for direction, suffix in (("import", "consumption"), ("export", "production")):
            spot = spot_ex_vat * _effective_value(getattr(snapshot, f"multiplier_{suffix}"), moment.date())
            tax = _effective_value(getattr(snapshot, f"energy_taxes_{suffix}"), moment.date())
            supplier = _effective_value(getattr(snapshot, f"cost_supplier_{suffix}"), moment.date())
            base = spot + tax + supplier
            vat = base * _effective_value(getattr(snapshot, f"vat_{suffix}"), moment.date()) / 100
            components.extend(((f"{direction}_spot_ex_vat", spot), (f"{direction}_tax_ex_vat", tax),
                               (f"{direction}_supplier_ex_vat", supplier), (f"{direction}_vat", vat)))
            rates.append(base + vat)
        bonus_rate = Decimal("0")
        if snapshot.supply_start is not None and moment.date() >= snapshot.supply_start:
            bonus_rate = potential_bonus_rate(snapshot.bonus, moment, spot_ex_vat)
    return PriceQuote(
        start_ts=start_ts, end_ts=end_ts, snapshot=snapshot, spot_ex_vat=spot_ex_vat,
        import_rate=rates[0], export_rate=rates[1], components=tuple(components), bonus_rate=bonus_rate,
    )
