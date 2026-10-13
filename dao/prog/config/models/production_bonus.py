"""Bonus on grid feed-in, modelled after the Budget Thuis ZonneBonus (2027)."""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Budget Thuis ZonneBonus: [beginuur, einduur] per maand, begin inclusief, eind exclusief.
DEFAULT_WINDOWS: dict[str, list[int]] = {
    "1": [9, 17], "2": [8, 18], "3": [7, 19], "4": [7, 21],
    "5": [6, 21], "6": [5, 22], "7": [6, 22], "8": [6, 21],
    "9": [7, 20], "10": [8, 19], "11": [8, 17], "12": [9, 17],
}

SECTION = "Production bonus"


class ProductionBonusConfig(BaseModel):
    """A supplier bonus on top of the exchange price for grid feed-in."""

    enabled: bool = Field(
        default=False,
        description="Switch the production bonus on",
        json_schema_extra={
            "x-help": "The single switch. With false (or without this section) nothing changes: no bonus in the prices, the reports or the optimizer.",
            "x-ui-section": SECTION,
        },
    )
    mode: Literal["pv only", "all export"] = Field(
        default="pv only",
        description="Which feed-in the optimizer counts for the bonus",
        json_schema_extra={
            "x-help": "'pv only': the optimizer counts the bonus only on export up to the PV production of that interval, so discharging the battery into the grid earns no bonus and there is no incentive to do so. Budget Thuis forbids using a home battery to collect more bonus (art. 8 of the action conditions). 'all export': every kWh exported in the window earns the bonus; the optimizer will then shift battery discharge into the window. Reports always count the bonus on all metered export, which is what the supplier pays.",
            "x-ui-section": SECTION,
        },
    )
    start_date: date = Field(
        default=date(2027, 1, 1),
        alias="start date",
        description="First day the bonus applies (YYYY-MM-DD)",
        json_schema_extra={
            "x-help": "Before this date the bonus is 0. Set it to today to try the feature before 2027.",
            "x-ui-section": SECTION,
        },
    )
    percentage: float = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Bonus as a percentage of the exchange price",
        json_schema_extra={
            "x-help": "30 means 30% on top of the exchange price.",
            "x-unit": "%",
            "x-ui-section": SECTION,
        },
    )
    max_exchange_price: float = Field(
        default=0.20,
        gt=0,
        alias="max exchange price",
        description="Exchange price above which the bonus no longer grows (euro/kWh ex VAT)",
        json_schema_extra={
            "x-help": "The bonus is percentage x min(exchange price, this value), so 30% of 0.20 gives at most 0.06 euro/kWh. The bonus is computed on the raw exchange price (ex VAT) and then gets the same 'vat production' as the rest of the feed-in price. If your supplier quotes the exchange price including VAT, enter 0.20 / 1.21 here.",
            "x-unit": "€/kWh",
            "x-ui-section": SECTION,
        },
    )
    windows: dict[str, list[int]] = Field(
        default=DEFAULT_WINDOWS,
        description="Per month (1-12) the [start, end] hours of the bonus window",
        json_schema_extra={
            "x-help": "An interval earns the bonus when it starts at or after the start hour and before the end hour, local time. Default: the Budget Thuis table.",
            "x-ui-section": SECTION,
        },
    )
    yearly_cap_kwh: Optional[float] = Field(
        default=5000.0,
        alias="yearly cap kwh",
        description="Feed-in per delivery year that earns the bonus; null for no cap",
        json_schema_extra={
            "x-help": "Budget Thuis pays the bonus over the first 5000 kWh per delivery year. DAO+ sums the consolidated feed-in since the start of the delivery year and stops counting the bonus once it is used up. Needs 'delivery year start'.",
            "x-unit": "kWh",
            "x-ui-section": SECTION,
        },
    )
    delivery_year_start: Optional[date] = Field(
        default=None,
        alias="delivery year start",
        description="Start date of the supplier contract; its month and day mark each new delivery year",
        json_schema_extra={
            "x-help": "A delivery year runs 12 months from the start of the contract. Only month and day are used. The first bonus year runs from 'start date' to the next anniversary. Without this date the yearly cap is not tracked.",
            "x-ui-section": SECTION,
        },
    )

    @field_validator("windows")
    @classmethod
    def validate_windows(cls, v: dict[str, list[int]]) -> dict[str, list[int]]:
        expected = {str(month) for month in range(1, 13)}
        if set(v.keys()) != expected:
            raise ValueError(f"windows: keys must be the months 1..12, got {sorted(v.keys())}")
        for month, window in v.items():
            if len(window) != 2:
                raise ValueError(
                    f"windows[{month}]: expected two integers [start, end], got {window}"
                )
            start, end = window
            if not (0 <= start < end <= 24):
                raise ValueError(
                    f"windows[{month}]: need 0 <= start < end <= 24, got {window}"
                )
            if not (isinstance(start, int) and isinstance(end, int)) or isinstance(
                window[0], bool
            ) or isinstance(window[1], bool):
                # strings or floats fall through here
                raise ValueError(
                    f"windows[{month}]: start and end must be valid integers, got {window}"
                )
        return v

    @field_validator("yearly_cap_kwh")
    @classmethod
    def validate_yearly_cap(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("yearly cap kwh must be greater than 0, or null for no cap")
        return v

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "x-ui-group": "Pricing",
            "x-icon": "currency-eur",
            "x-order": 12,
            "x-help": """# Production bonus

A bonus your supplier pays on top of the exchange price for feed-in, e.g. the
Budget Thuis ZonneBonus from 1 January 2027: 30% extra inside a monthly daytime
window, only when the exchange price is positive, at most 0.06 euro/kWh, over
the first 5000 kWh per delivery year.

All defaults are the Budget Thuis values. A Budget Thuis customer sets
`enabled: true` and `delivery year start`; everything else can stay.

The bonus shows up as the column `da_bonus` next to `da_prod` (which stays the
feed-in price without bonus), in the `profit` of the reports and prognoses, and
as a dashed line in the price graph. The yearly cap is evaluated once per
calculation at its start moment; a report over a long past period applies the
bonus to the whole period based on the cap state at its start.
""",
        },
    )
