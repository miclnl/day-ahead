"""Immutable financial DTOs using UTC epochs and unrounded Decimal amounts.

Energy inputs have at most nine decimal places, tariffs at most twelve. The
arithmetic consumers use precision 36 and round only displayed totals to cents
with ROUND_HALF_UP. None denotes missing information, never an implicit zero.
"""

from dataclasses import dataclass, fields
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import cache
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


Quality = Literal["measured", "estimated", "incomplete"]
Origin = Literal["verified_pv", "unknown"]
TariffTable = tuple[tuple[date, Decimal], ...]

TARIFF_FIELDS = (
    "energy_taxes_consumption", "energy_taxes_production",
    "cost_supplier_consumption", "cost_supplier_production",
    "vat_consumption", "vat_production",
    "multiplier_consumption", "multiplier_production",
)
_ENERGY_FIELDS = {
    "import_kwh", "export_kwh", "battery_discharge_kwh", "total_export_kwh",
    "bonus_kwh", "yearly_cap_kwh",
}
_INPUT_RATE_FIELDS = {
    "spot_ex_vat", "price_ex_vat", "percentage", "max_exchange_price", "vat_percent",
}


@cache
def _field_types(cls):
    return get_type_hints(cls)


def _immutable_value(value, annotation, name):
    """Validate declared DTO types and detach every nested sequence from callers."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        for option in args:
            try:
                return _immutable_value(value, option, name)
            except (TypeError, ValueError):
                continue
        raise ValueError(f"{name}: value does not match its declared optional type")
    if origin is Literal:
        if not any(type(value) is type(option) and value == option for option in args):
            raise ValueError(f"{name}: unsupported value")
        return value
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise TypeError(f"{name}: expected an immutable sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_immutable_value(item, args[0], name) for item in value)
        if len(value) != len(args):
            raise ValueError(f"{name}: unexpected tuple length")
        return tuple(_immutable_value(item, kind, name) for item, kind in zip(value, args))
    if annotation is Decimal:
        if isinstance(value, str):
            try:
                value = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f"{name}: expected decimal text") from exc
        if not isinstance(value, Decimal):
            raise TypeError(f"{name}: use Decimal from text, not a float")
        if not value.is_finite():
            raise ValueError(f"{name}: decimal must be finite")
        return value
    if annotation in (str, int, bool, date, type(None)):
        if type(value) is not annotation:
            raise TypeError(f"{name}: expected {annotation.__name__}")
    elif not isinstance(value, annotation):
        raise TypeError(f"{name}: expected {annotation.__name__}")
    return value


def _precision(value: Decimal, places: int, name: str):
    if value.as_tuple().exponent < -places:
        raise ValueError(f"{name}: at most {places} decimal places are permitted")


class _FrozenDTO:
    __slots__ = ()

    def __post_init__(self):
        annotations = _field_types(type(self))
        for field in fields(self):
            value = _immutable_value(getattr(self, field.name), annotations[field.name], field.name)
            object.__setattr__(self, field.name, value)
            if value is not None and field.name in _ENERGY_FIELDS:
                _precision(value, 9, field.name)
                if value < 0:
                    raise ValueError(f"{field.name}: energy cannot be negative")
            if value is not None and field.name in _INPUT_RATE_FIELDS:
                _precision(value, 12, field.name)
            if value is not None and field.name in {"generation", "input_revision", "current_input_revision"}:
                if value < 0:
                    raise ValueError(f"{field.name}: revision/generation cannot be negative")
        for start_name, end_name in (
            ("start_ts", "end_ts"), ("period_start_ts", "period_end_ts"),
            ("valid_from_ts", "valid_to_ts"), ("coverage_start_ts", "coverage_end_ts"),
        ):
            if hasattr(self, start_name) and hasattr(self, end_name):
                start, end = getattr(self, start_name), getattr(self, end_name)
                if end is not None and end <= start:
                    raise ValueError(f"{end_name} must be after {start_name}; intervals are half-open")


@dataclass(frozen=True, slots=True)
class BonusRules(_FrozenDTO):
    """Financial bonus allowlist, separate from mutable/legacy configuration."""

    enabled: bool
    mode: Literal["pv only", "all export"]
    start_date: date
    percentage: Decimal
    max_exchange_price: Decimal
    cap_basis: Literal["ex_vat", "inc_vat", "unconfirmed"]
    vat_percent: Decimal | None
    yearly_cap_kwh: Decimal | None
    windows: tuple[tuple[int, tuple[int, int]], ...]

    def __post_init__(self):
        _FrozenDTO.__post_init__(self)
        if not 0 <= self.percentage <= 100 or self.max_exchange_price <= 0:
            raise ValueError("Bonus percentage/ceiling is outside its permitted range")
        if self.vat_percent is not None and not 0 <= self.vat_percent <= 100:
            raise ValueError("Bonus VAT percentage must be between 0 and 100")
        if self.yearly_cap_kwh is not None and self.yearly_cap_kwh <= 0:
            raise ValueError("Bonus cap must be positive or unknown/unbounded by explicit rules")
        if len(self.windows) != 12 or {month for month, _ in self.windows} != set(range(1, 13)):
            raise ValueError("Bonus windows must specify months 1..12 exactly once")
        if any(not 0 <= start < end <= 24 for _, (start, end) in self.windows):
            raise ValueError("Bonus windows require 0 <= start < end <= 24")


@dataclass(frozen=True, slots=True)
class TariffSnapshot(_FrozenDTO):
    """Financial allowlist; A2 supplies the SHA-256 canonical-JSON identity.

    A new tariff hash does not change contract_id or supply_start. Source
    versions are non-secret evidence identifiers, not complete settings.
    """

    snapshot_hash: str
    rule_version: str
    contract_id: str
    status: Literal["draft", "confirmed"]
    valid_from_ts: int
    valid_to_ts: int | None
    supply_start: date | None
    timezone: str
    product: Literal["legacy", "budget_dynamic_2027"]
    regime: Literal["legacy_interval", "no_net_metering"]
    price_interval_seconds: Literal[900, 3600]
    meter_interval_seconds: Literal[900]
    hour_price_rule: Literal["native", "mean_quarters", "unconfirmed"]
    monthly_export_floor: Literal["none", "spot_zero"]
    energy_taxes_consumption: TariffTable
    energy_taxes_production: TariffTable
    cost_supplier_consumption: TariffTable
    cost_supplier_production: TariffTable
    vat_consumption: TariffTable
    vat_production: TariffTable
    multiplier_consumption: TariffTable
    multiplier_production: TariffTable
    bonus: BonusRules | None
    source_versions: tuple[tuple[str, str], ...]

    def __post_init__(self):
        _FrozenDTO.__post_init__(self)
        try:
            ZoneInfo(self.timezone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("Snapshot requires a known IANA timezone") from exc
        for name in TARIFF_FIELDS:
            table = getattr(self, name)
            dates = [effective for effective, _ in table]
            if not dates or dates != sorted(set(dates)):
                raise ValueError(f"{name}: tariff dates must be nonempty, unique and ordered")
            for _, value in table:
                _precision(value, 12, name)


@dataclass(frozen=True, slots=True)
class MeterSelection(_FrozenDTO):
    """Selected sources; absent battery sensors do not prove no battery."""

    meter_id: str
    import_sensors: tuple[str, ...]
    export_sensors: tuple[str, ...]
    battery_sources: tuple[str, ...]
    no_battery_evidence: str | None

    def __post_init__(self):
        _FrozenDTO.__post_init__(self)
        if self.no_battery_evidence is not None and not self.no_battery_evidence.strip():
            raise ValueError("No-battery evidence must identify an explicit installation record")
        if self.battery_sources and self.no_battery_evidence is not None:
            raise ValueError("Battery sources conflict with explicit no-battery evidence")


@dataclass(frozen=True, slots=True)
class MeterInterval(_FrozenDTO):
    """Separate, nonnegative metered energies on a half-open UTC interval."""

    start_ts: int
    end_ts: int
    import_kwh: Decimal | None
    export_kwh: Decimal | None
    battery_discharge_kwh: Decimal | None
    origin: Origin
    quality: Quality
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InputInterval(MeterInterval):
    """Registered input identity retains revision even when old content returns."""

    spot_ex_vat: Decimal | None
    spot_source: str | None
    input_hash: str
    meter_id: str
    input_revision: int


@dataclass(frozen=True, slots=True)
class SpotInterval(_FrozenDTO):
    """Market-price source interval, independent of meter and solver resolution."""

    start_ts: int
    end_ts: int
    price_ex_vat: Decimal | None
    source: str
    quality: Quality


@dataclass(frozen=True, slots=True)
class RegisteredSpot(_FrozenDTO):
    """Immutable bronprijs met eigen rijrevisie binnen het globale boek."""

    spot: SpotInterval
    input_revision: int
    content_hash: str

    def __post_init__(self):
        _FrozenDTO.__post_init__(self)
        if self.input_revision <= 0:
            raise ValueError("Registered prices require a positive row revision")


@dataclass(frozen=True, slots=True)
class PriceBatch(_FrozenDTO):
    """Gepinde raw prijzen; het vraagbereik is geen gesloten meetdekking."""

    meter_id: str
    input_revision: int
    start_ts: int
    end_ts: int
    items: tuple[RegisteredSpot, ...]
    snapshots: tuple[TariffSnapshot, ...]
    captured_ts: int
    source_watermark_ts: int | None

    def __post_init__(self):
        _FrozenDTO.__post_init__(self)
        if not self.meter_id.strip() or len(self.meter_id) > 128:
            raise ValueError("Price batch requires an explicit meter identifier")
        if any(item.input_revision > self.input_revision for item in self.items):
            raise ValueError("Price row revision exceeds the pinned book revision")


@dataclass(frozen=True, slots=True)
class ResolvedSpot(_FrozenDTO):
    """Pure geselecteerde contractprijs, zonder meet- of publicatierechten."""

    price_ex_vat: Decimal | None
    source: str | None
    quality: Quality


@dataclass(frozen=True, slots=True)
class PriceQuote(_FrozenDTO):
    """Known rates and full immutable rules; missing spot means no quote."""

    start_ts: int
    end_ts: int
    snapshot: TariffSnapshot
    spot_ex_vat: Decimal
    import_rate: Decimal
    export_rate: Decimal
    components: tuple[tuple[str, Decimal], ...]
    bonus_rate: Decimal


@dataclass(frozen=True, slots=True)
class BonusState(_FrozenDTO):
    """Chronological total export; None means an unknown cap position."""

    period_start_ts: int
    period_end_ts: int
    asof_ts: int
    total_export_kwh: Decimal | None


@dataclass(frozen=True, slots=True)
class MonthState(_FrozenDTO):
    """Contract-local month as UTC bounds; complete means sufficient coverage."""

    contract_id: str
    period_start_ts: int
    period_end_ts: int
    spot_export_eur: Decimal | None
    negative_basis_by_vat: tuple[tuple[Decimal, Decimal], ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class SettlementState(_FrozenDTO):
    """Opening or closing chronological bonus/month state at an exact epoch."""

    asof_ts: int
    bonus: BonusState | None
    month: MonthState | None


@dataclass(frozen=True, slots=True)
class SettlementLine(_FrozenDTO):
    """Unrounded components; an unavailable quote may have no snapshot hash."""

    start_ts: int
    end_ts: int
    meter_id: str
    input_revision: int
    input_hash: str
    snapshot_hash: str | None
    import_cost: Decimal | None
    base_export_revenue: Decimal | None
    bonus_revenue: Decimal | None
    bonus_kwh: Decimal | None
    before_state: SettlementState
    after_state: SettlementState
    quality: Quality
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PeriodAdjustment(_FrozenDTO):
    """Separate period credit and its basis, not a fictional interval rate."""

    kind: str
    period_start_ts: int
    period_end_ts: int
    credit_ex_vat: Decimal
    credit_vat: Decimal
    credit_total: Decimal
    basis_ex_vat: Decimal
    quality: Quality


@dataclass(frozen=True, slots=True)
class SettlementResult(_FrozenDTO):
    """Covered calculation; known subtotal does not turn missing amounts into zero."""

    lines: tuple[SettlementLine, ...]
    adjustments: tuple[PeriodAdjustment, ...]
    end_state: SettlementState
    engine_version: str
    coverage_start_ts: int
    coverage_end_ts: int
    complete: bool
    known_subtotal: Decimal
    net_cost: Decimal | None


@dataclass(frozen=True, slots=True)
class InputBatch(_FrozenDTO):
    """Pinned registered revision, including immutable contract snapshots."""

    input_revision: int
    items: tuple[InputInterval, ...]
    snapshots: tuple[TariffSnapshot, ...]
    captured_ts: int
    source_watermark_ts: int | None


@dataclass(frozen=True, slots=True)
class SettlementView(_FrozenDTO):
    """Pinned financial generation and explicit registered-input freshness.

    Current means consistent with registered input, not a guarantee that every
    external correction has already been collected. Complete is not invoice
    finality.
    """

    meter_id: str
    generation: int
    input_revision: int
    current_input_revision: int
    freshness: Literal["current", "outdated", "unknown"]
    lines: tuple[SettlementLine, ...]
    adjustments: tuple[PeriodAdjustment, ...]
    known_subtotal: Decimal
    net_cost: Decimal | None
    coverage_start_ts: int
    coverage_end_ts: int
    complete: bool
    captured_ts: int
    source_watermark_ts: int | None
    dirty_from_ts: int | None


@dataclass(frozen=True, slots=True)
class TerminalValue(_FrozenDTO):
    """Policy-labelled EUR per AC-deliverable kWh and battery reference efficiency."""

    policy_version: str
    eur_per_ac_kwh: Decimal
    reference_efficiency_by_battery: tuple[tuple[str, Decimal], ...]

    def __post_init__(self):
        _FrozenDTO.__post_init__(self)
        if self.eur_per_ac_kwh < 0:
            raise ValueError("Terminal value cannot be negative")
        for _, efficiency in self.reference_efficiency_by_battery:
            _precision(efficiency, 12, "reference efficiency")
            if not 0 < efficiency <= 1:
                raise ValueError("Reference efficiency must be greater than zero and at most one")
