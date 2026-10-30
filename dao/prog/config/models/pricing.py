"""
Pricing configuration models.
"""

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .base import SecretStr
from .production_bonus import ProductionBonusConfig
from .settlement import SettlementConfig, reject_alias_conflicts


SETTLEMENT_TARIFF_FIELDS = (
    "energy_taxes_consumption", "energy_taxes_production",
    "cost_supplier_consumption", "cost_supplier_production",
    "vat_consumption", "vat_production",
    "multiplier_consumption", "multiplier_production",
)


class PricingConfig(BaseModel):
    """Day-ahead pricing and tariff configuration."""

    source_day_ahead: Literal["nordpool", "entsoe", "tibber"] = Field(
        default="nordpool",
        alias="source day ahead",
        description="Source for day-ahead prices",
        json_schema_extra={
            "x-help": "Data source for day-ahead electricity market prices. 'nordpool' for Nordic/Baltic, 'entsoe' for European markets, 'tibber' if using Tibber integration.",
            "x-ui-section": "Prices",
        },
    )
    entsoe_api_key: Optional[SecretStr] = Field(
        default=None,
        alias="entsoe-api-key",
        description="ENTSO-E API key (can use !secret)",
        json_schema_extra={
            "x-help": "API key for ENTSO-E Transparency Platform. Required if source_day_ahead='entsoe'. Get free key at transparency.entsoe.eu. Use !secret for security.",
            "x-ui-section": "Prices",
            "x-validation-hint": "Required for entsoe source, use !secret",
            "x-docs-url": "https://transparency.entsoe.eu/",
            "x-ui-rules": {
                "effect": "SHOW",
                "condition": {
                    "scope": "#/properties/source_day_ahead",
                    "schema": {"const": "entsoe"},
                },
            },
        },
    )

    # Date-based tariff configurations (date string -> value)
    energy_taxes_consumption: dict[str, float] = Field(
        alias="energy taxes consumption",
        description="Energy taxes for consumption by date (YYYY-MM-DD -> euro/kWh ex VAT)",
        json_schema_extra={
            "x-help": "Energy taxes on consumption (excluding VAT) indexed by effective date. Format: {'2024-01-01': 0.05}. Use date when tariff changes.",
            "x-unit": "€/kWh",
            "x-ui-section": "Taxes",
            "x-validation-hint": "Dict with YYYY-MM-DD keys, float values (ex VAT)",
        },
    )
    energy_taxes_production: dict[str, float] = Field(
        alias="energy taxes production",
        description="Energy taxes for production by date (YYYY-MM-DD -> euro/kWh ex VAT)",
        json_schema_extra={
            "x-help": "Energy taxes on feed-in/production (excluding VAT) indexed by effective date. Often zero or negative. Format: {'2024-01-01': 0.0}.",
            "x-unit": "€/kWh",
            "x-ui-section": "Taxes",
            "x-validation-hint": "Dict with YYYY-MM-DD keys, float values (ex VAT)",
        },
    )
    cost_supplier_consumption: dict[str, float] = Field(
        alias="cost supplier consumption",
        description="Supplier costs for consumption by date (YYYY-MM-DD -> euro/kWh ex VAT)",
        json_schema_extra={
            "x-help": "Supplier markup/fees for consumption (excluding VAT) indexed by effective date. Fixed part of electricity cost. Format: {'2024-01-01': 0.02}.",
            "x-unit": "€/kWh",
            "x-ui-section": "Cost",
            "x-validation-hint": "Dict with YYYY-MM-DD keys, float values (ex VAT)",
        },
    )
    cost_supplier_production: dict[str, float] = Field(
        alias="cost supplier production",
        description="Supplier costs for production by date (YYYY-MM-DD -> euro/kWh ex VAT)",
        json_schema_extra={
            "x-help": "Supplier fees for feed-in/production (excluding VAT) indexed by effective date. May be negative (credit). Format: {'2024-01-01': -0.02}.",
            "x-unit": "€/kWh",
            "x-ui-section": "Cost",
            "x-validation-hint": "Dict with YYYY-MM-DD keys, float values (ex VAT)",
        },
    )
    vat_consumption: dict[str, float] = Field(
        alias="vat consumption",
        description="VAT percentage for consumption by date (YYYY-MM-DD -> %)",
        json_schema_extra={
            "x-help": "VAT percentage on consumption indexed by effective date. Format: {'2024-01-01': 21}. Applied to market price + taxes + supplier costs.",
            "x-unit": "%",
            "x-ui-section": "Taxes",
            "x-validation-hint": "Dict with YYYY-MM-DD keys, integer 0-100 values",
        },
    )
    vat_production: dict[str, float] = Field(
        alias="vat production",
        description="VAT percentage for production by date (YYYY-MM-DD -> %)",
        json_schema_extra={
            "x-help": "VAT percentage on feed-in/production indexed by effective date. Format: {'2024-01-01': 21}. Often same as consumption VAT.",
            "x-unit": "%",
            "x-ui-section": "Taxes",
            "x-validation-hint": "Dict with YYYY-MM-DD keys, integer 0-100 values",
        },
    )
    multiplier_consumption: Optional[dict[str, float]] = Field(
        default={"2000-01-01": 1.0},
        alias="multiplier consumption",
        description="Multiplier for consumption by date (YYYY-MM-DD -> x.xx)",
        json_schema_extra={
            "x-help": "Multiplier on consumption day-ahead price indexed by effective date. Format: {'2024-01-01': 0.94}.",
            "x-unit": "-",
            "x-ui-section": "Cost",
            "x-validation-hint": "Dict with YYYY-MM-DD keys, float -100.0 - +100.0 values",
        },
    )
    multiplier_production: Optional[dict[str, float]] = Field(
        default={"2000-01-01": 1.0},
        alias="multiplier production",
        description="Multiplier for production by date (YYYY-MM-DD -> x.xx)",
        json_schema_extra={
            "x-help": "Multiplier on feed-in/production day-ahead price indexed by effective date. Format: {'2024-01-01': 0.94}.",
            "x-unit": "-",
            "x-ui-section": "Cost",
            "x-validation-hint": "Dict with YYYY-MM-DD keys, float -100.0 - +100.0 values",
        },
    )
    # Invoice settings
    last_invoice: date = Field(
        alias="last invoice",
        description="Date of last invoice (YYYY-MM-DD)",
        json_schema_extra={
            "x-help": "Date of last electricity invoice. Used for calculating costs since last billing period. Format: YYYY-MM-DD. Update after receiving invoices.",
            "x-ui-section": "Prices",
            "x-validation-hint": "Must be YYYY-MM-DD format",
        },
    )
    tax_refund: bool = Field(
        default=True,
        alias="tax refund",
        description="Whether tax refund applies",
        json_schema_extra={
            "x-help": "Enable tax refund calculation if eligible. Some regions/users get energy tax refunds for solar production.",
            "x-ui-section": "Taxes",
        },
    )
    production_bonus: Optional[ProductionBonusConfig] = Field(
        default=None,
        alias="production bonus",
        description="Supplier bonus on grid feed-in (e.g. Budget Thuis ZonneBonus)",
        json_schema_extra={
            "x-help": "Leave out, or set enabled to false, when your supplier pays no bonus. "
            "See the Production bonus section for the fields and their defaults.",
            "x-ui-section": "Prices",
        },
    )

    settlement: SettlementConfig | None = Field(
        default=None,
        description="Optional explicit contract-based electricity settlement",
        json_schema_extra={
            "x-help": "Omit to preserve legacy behavior. The settlement mode defaults to off; configuring draft terms does not activate them or establish 2027 readiness.",
            "x-ui-section": "Settlement",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def validate_settlement_input(cls, data):
        data = reject_alias_conflicts(cls, data)
        if not isinstance(data, dict) or data.get("settlement") is None:
            return data
        for name in SETTLEMENT_TARIFF_FIELDS:
            table = data.get(cls.model_fields[name].alias, data.get(name))
            if not isinstance(table, dict):
                continue
            for effective, raw in table.items():
                try:
                    if date.fromisoformat(effective).isoformat() != effective:
                        raise ValueError("Noncanonical date")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name}: settlement tariff dates must use YYYY-MM-DD") from exc
                try:
                    value = Decimal(str(raw))
                except InvalidOperation as exc:
                    raise ValueError(f"{name}: settlement tariff must be a finite decimal") from exc
                if not value.is_finite() or value.as_tuple().exponent < -12:
                    raise ValueError(f"{name}: settlement tariff must be finite with at most 12 decimals")
        return data

    @model_validator(mode="after")
    def validate_settlement_terms(self):
        if self.settlement is None:
            return self
        for name in SETTLEMENT_TARIFF_FIELDS:
            table = getattr(self, name)
            if table and any(not Decimal(str(value)).is_finite() for value in table.values()):
                raise ValueError(f"{name}: settlement tariff must remain finite after pricing conversion")
        for contract in self.settlement.contracts:
            bonus = self.production_bonus
            if (contract.product == "budget_dynamic_2027" and bonus is not None
                    and bonus.enabled and bonus.mode == "all export"):
                raise ValueError("Budget settlement does not permit bonus mode all export")
            if contract.status != "confirmed":
                continue
            for name in SETTLEMENT_TARIFF_FIELDS:
                if name not in self.model_fields_set:
                    raise ValueError(f"{name}: confirmed settlement requires an explicit dated component")
                table = getattr(self, name)
                if not table or not any(date.fromisoformat(key) <= contract.valid_from for key in table):
                    raise ValueError(f"{name}: no tariff available at contract start")
                if contract.product == "budget_dynamic_2027" and name in {
                    "energy_taxes_production", "cost_supplier_production", "multiplier_production",
                }:
                    expected = 1 if name == "multiplier_production" else 0
                    applicable = max(key for key in table if date.fromisoformat(key) <= contract.valid_from)
                    for key, value in table.items():
                        in_period = (date.fromisoformat(key) >= contract.valid_from
                                     and (contract.valid_to is None or date.fromisoformat(key) < contract.valid_to))
                        if (key == applicable or in_period) and value != expected:
                            raise ValueError(f"{name}: Budget export component must be {expected}")
            if bonus is not None and bonus.enabled:
                if bonus.cap_basis == "unconfirmed" or bonus.vat_percent is None:
                    raise ValueError("Confirmed bonus requires explicit cap basis and vat percent")
                for name, places in (("percentage", 12), ("max_exchange_price", 12), ("yearly_cap_kwh", 9)):
                    raw = getattr(bonus, name)
                    if raw is None:
                        continue
                    value = Decimal(str(raw))
                    if not value.is_finite() or value.as_tuple().exponent < -places:
                        raise ValueError(f"{name}: confirmed bonus requires a finite value with at most {places} decimals")
        return self

    @field_validator(
        "energy_taxes_consumption",
        "energy_taxes_production",
        "cost_supplier_consumption",
        "cost_supplier_production",
        "vat_consumption",
        "vat_production",
        "multiplier_consumption",
        "multiplier_production",
    )
    @classmethod
    def validate_dated_dict(cls, v: Optional[dict[str, float]], info) -> Optional[dict[str, float]]:
        """Every key must be an ISO date; the dict is returned sorted by date.

        utils.get_value_from_dict() looks up "the tariff in effect on this
        date" with a binary search over the keys, which silently returns the
        wrong tariff if the dict is not sorted, and an unparsable date breaks
        that lookup with a confusing string-comparison error instead of a
        clear configuration error at load time.
        """
        if v is None:
            return v
        if not v:
            raise ValueError(f"{info.field_name} must have at least one dated entry")
        for key in v:
            try:
                date.fromisoformat(key)
            except ValueError as ex:
                raise ValueError(
                    f"{info.field_name}: key {key!r} is not a valid date (YYYY-MM-DD)"
                ) from ex
        return dict(sorted(v.items()))

    @field_validator("vat_consumption", "vat_production")
    @classmethod
    def validate_vat_percentages(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate VAT percentages are between 0 and 100."""
        for date_key, percentage in v.items():
            if not (0 <= percentage <= 100):
                raise ValueError(
                    f"VAT percentage must be between 0 and 100, got {percentage} for date {date_key}"
                )
        return v

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "x-ui-group": "Pricing",
            "x-icon": "currency-eur",
            "x-order": 11,
            "x-help": """# Pricing & Tariff Configuration

Configure electricity market prices and tariff components for accurate cost optimization.

## Price Components

Total electricity cost consists of:
1. **Market price**: Day-ahead spot price (nordpool/entsoe/tibber)
2. **Energy taxes**: Government energy taxes
3. **Supplier costs**: Your supplier's markup/fees
4. **VAT**: Value-added tax on sum of above
5. **Multiplier**: For calculation of productionprice in Belgium 

Price = (market*multiplier + taxes + supplier) × (1 + VAT)

## Date-Based Tariffs

All tariff fields use date-indexed dictionaries:
```json
{
  "2024-01-01": 0.05,
  "2024-07-01": 0.06
}
```

System uses tariff active on optimization date.

## Data Sources

- **nordpool**: Nord Pool (Nordic/Baltic markets)
- **entsoe**: ENTSO-E Transparency Platform (all European markets)
- **tibber**: Tibber API (if using Tibber as supplier)

## Tips

- All prices are EXCLUDING VAT (VAT applied separately)
- Update tariffs when your supplier changes prices
- Use !secret for API keys
- Production costs often lower than consumption (or negative for feed-in credit)
- Keep last_invoice updated for accurate cost tracking
- Check your electricity bill for exact tariff components
""",
            "x-docs-url": "https://github.com/miclnl/day-ahead/wiki/Pricing-Configuration",
        },
    )
