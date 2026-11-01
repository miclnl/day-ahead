"""Explicit contract terms for optional, versioned electricity settlement."""

from datetime import date
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SECTION = "Settlement"


def reject_alias_conflicts(model: type[BaseModel], data):
    """Reject contradictory aliases and collapse equal duplicates without mutation."""
    if not isinstance(data, dict):
        return data
    result = dict(data)
    for name, field in model.model_fields.items():
        alias = field.alias
        if alias and alias != name and alias in result and name in result:
            if result[alias] != result[name]:
                raise ValueError(f"Conflicting alias values for {alias!r} and {name!r}")
            del result[name]
    return result


class _SettlementConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def validate_aliases(cls, data):
        return reject_alias_conflicts(cls, data)


class SettlementContractConfig(_SettlementConfigBase):
    """A dated contract version; draft terms are not an active financial basis."""

    id: str = Field(
        min_length=1, description="Stable supplier contract identifier",
        json_schema_extra={"x-help": "Keep this identifier on tariff changes; a new tariff snapshot does not restart the delivery year.", "x-ui-section": SECTION},
    )
    status: Literal["draft", "confirmed"] = Field(
        default="draft", description="Whether the applicable contract terms are confirmed",
        json_schema_extra={"x-help": "Draft terms are for shadow/what-if only. Confirmed terms require explicit financial components at the contract start.", "x-ui-section": SECTION},
    )
    valid_from: date = Field(
        alias="valid from", description="First local date of this contract version, inclusive",
        json_schema_extra={"x-help": "Use YYYY-MM-DD in the contract timezone. No tariff value for this date is guessed.", "x-ui-section": SECTION},
    )
    valid_to: date | None = Field(
        default=None, alias="valid to", description="Last local boundary of this version, exclusive",
        json_schema_extra={"x-help": "Null leaves the version open-ended. Adjacent versions may share this exclusive boundary but may not overlap.", "x-ui-section": SECTION},
    )
    supply_start: date | None = Field(
        default=None, alias="supply start", description="Actual first delivery date of the supplier contract",
        json_schema_extra={"x-help": "Required for confirmed terms. Never inferred from the invoice date or the legacy bonus delivery-year anchor.", "x-ui-section": SECTION},
    )
    timezone: str = Field(
        description="IANA timezone for contract dates and calendar rules",
        json_schema_extra={"x-help": "Budget dynamic 2027 uses Europe/Amsterdam. Interval identities themselves use UTC epoch seconds.", "x-ui-section": SECTION},
    )
    product: Literal["legacy", "budget_dynamic_2027"] = Field(
        description="Supplier product whose explicit rules apply",
        json_schema_extra={"x-help": "The Budget profile requires hourly spot prices, quarter-hour metering, no export tax/adder and an export spot multiplier of one.", "x-ui-section": SECTION},
    )
    regime: Literal["legacy_interval", "no_net_metering"] = Field(
        description="Electricity settlement regime for this version",
        json_schema_extra={"x-help": "Use no_net_metering for the Budget 2027 profile. This does not globally change legacy tax-refund settings.", "x-ui-section": SECTION},
    )
    price_interval_seconds: Literal[900, 3600] = Field(
        alias="price interval seconds", description="Duration of a contractual spot-price interval",
        json_schema_extra={"x-help": "Independent of metering and solver steps. The current Budget contract profile requires 3600 seconds.", "x-ui-section": SECTION},
    )
    meter_interval_seconds: Literal[900] = Field(
        alias="meter interval seconds", description="Required metering interval in seconds",
        json_schema_extra={"x-help": "Quarter-hour metering is 900 seconds; hourly statistics cannot prove four measured quarters.", "x-ui-section": SECTION},
    )
    hour_price_rule: Literal["native", "mean_quarters", "unconfirmed"] = Field(
        default="unconfirmed", alias="hour price rule", description="Confirmed construction of an hourly contract price",
        json_schema_extra={"x-help": "native uses supplied hourly prices; mean_quarters explicitly confirms duration-weighted averaging of four market quarters. unconfirmed cannot support a confirmed hourly contract.", "x-ui-section": SECTION},
    )
    monthly_export_floor: Literal["none", "spot_zero"] = Field(
        alias="monthly export floor", description="Separate monthly export spot-price adjustment rule",
        json_schema_extra={"x-help": "spot_zero floors the covered monthly ex-VAT spot export proceeds, not individual intervals, import costs or bonus revenue. It is a model adjustment, not an official invoice.", "x-ui-section": SECTION},
    )

    @field_validator("valid_from", "valid_to", "supply_start", mode="before")
    @classmethod
    def validate_local_dates(cls, value):
        if value is None or type(value) is date:
            return value
        if not isinstance(value, str) or len(value) != 10:
            raise ValueError("Contract boundaries must be local dates in YYYY-MM-DD format")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Contract boundaries must be local dates in YYYY-MM-DD format") from exc

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("Contract timezone must be a known IANA timezone") from exc
        return value

    @model_validator(mode="after")
    def validate_confirmed_terms(self):
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid to must be after valid from (exclusive boundary)")
        if self.status != "confirmed":
            return self
        if self.supply_start is None or self.supply_start > self.valid_from:
            raise ValueError("Confirmed terms require supply start on or before valid from")
        if self.price_interval_seconds == 3600 and self.hour_price_rule == "unconfirmed":
            raise ValueError("A confirmed hour contract requires native or mean_quarters hour price rule")
        if self.product == "budget_dynamic_2027":
            if self.valid_from < date(2027, 1, 1):
                raise ValueError("budget_dynamic_2027 cannot apply before 2027-01-01")
            if self.timezone != "Europe/Amsterdam" or self.regime != "no_net_metering":
                raise ValueError("budget_dynamic_2027 requires Europe/Amsterdam and no_net_metering")
            if self.price_interval_seconds != 3600:
                raise ValueError("budget_dynamic_2027 requires 3600-second contract spot prices")
        return self

    model_config = ConfigDict(json_schema_extra={
        "x-ui-group": "Pricing", "x-icon": "currency-eur", "x-order": 14,
        "x-help": "Explicit, half-open contract versions. Confirm applicability against supplier evidence before confirmation; synthetic test values are not customer settings.",
    })


class SettlementConfig(_SettlementConfigBase):
    """Optional settlement configuration; omission preserves the legacy calculation."""

    mode: Literal["off", "shadow", "active"] = Field(
        default="off", description="Settlement calculation mode",
        json_schema_extra={"x-help": "off is the default; shadow is for non-actuating comparisons. active requires a meter and confirmed contracts, but configuration alone does not implement or authorize live activation.", "x-ui-section": SECTION},
    )
    meter_id: str | None = Field(
        default=None, alias="meter id", min_length=1, description="Stable metering-installation identifier",
        json_schema_extra={"x-help": "Used with interval start and registered input revision to identify input. An input content hash is not a revision identifier.", "x-ui-section": SECTION},
    )
    contracts: list[SettlementContractConfig] = Field(
        default=[], description="Non-overlapping contract versions",
        json_schema_extra={"x-help": "Each version has explicit validity and terms. Drafts cannot be used in active mode; tariff changes may keep the same contract identifier.", "x-ui-section": SECTION},
    )
    planning_source_policy: Literal["aggregate_cell_min_1wh_v1"] | None = Field(
        default=None, alias="planning source policy",
        description="Planning-only gross cell discharge domain policy",
        json_schema_extra={"x-help": "Null selects no policy. aggregate_cell_min_1wh_v1 requires zero or at least 1 Wh gross cell discharge for the whole battery group per applicable full bonus quarter. This can make planning infeasible; it is not a hardware minimum or permission for activation.", "x-ui-section": SECTION},
    )

    @model_validator(mode="after")
    def validate_contracts(self):
        ordered = sorted(self.contracts, key=lambda contract: contract.valid_from)
        for previous, current in zip(ordered, ordered[1:]):
            if previous.valid_to is None or previous.valid_to > current.valid_from:
                raise ValueError("Settlement contract versions must not overlap")
        if self.mode == "active":
            if not self.meter_id or not self.contracts:
                raise ValueError("Active settlement requires meter id and contracts")
            if any(contract.status == "draft" for contract in self.contracts):
                raise ValueError("A draft contract cannot be used in active settlement")
        return self

    model_config = ConfigDict(json_schema_extra={
        "x-ui-group": "Pricing", "x-icon": "currency-eur", "x-order": 13,
        "x-help": "Optional contract-based electricity settlement, default off. Leaving out prices.settlement preserves legacy behavior and does not establish readiness for 2027. Complete means sufficient model input coverage, never official invoice finality.",
    })
