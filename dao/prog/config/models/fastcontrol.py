"""
Fast control (realtime feedback layer) configuration models.

The day-ahead optimizer produces a plan on a 15-minute or hourly grid, based on a
*forecast* of the house load and PV production. Between two optimizer runs the
battery setpoint is frozen, so every forecast error lands one-to-one on the grid
meter. The fast control layer closes that gap: it samples the P1 meter every few
seconds and corrects the battery setpoint whenever correcting is economically
worthwhile, while staying inside an energy trust region around the planned state
of charge.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator

from .base import DAOConfigBaseModel, EntityId, FlexBool, FlexEnum, FlexFloat


class PowerSensor(BaseModel):
    """A power measurement read from Home Assistant.

    Supports the two shapes commonly produced by P1 / inverter integrations: a
    single signed sensor, or a separate positive-only pair.
    """

    entity: Optional[EntityId] = Field(
        default=None,
        description="HA entity with the signed power value",
        json_schema_extra={
            "x-help": "Home Assistant entity holding the power as a single signed number. "
            "For the grid meter the convention is positive = import from grid. For a "
            "battery the convention is positive = charging. Use 'invert' when your "
            "integration uses the opposite sign.",
            "x-ui-widget-filter": "sensor,input_number,number",
        },
    )
    entity_positive: Optional[EntityId] = Field(
        default=None,
        alias="entity positive",
        description="HA entity with the positive-only part of the power value",
        json_schema_extra={
            "x-help": "Alternative to 'entity': the sensor that only reports the positive "
            "direction (grid import, or battery charging). Combine with 'entity negative'.",
            "x-ui-widget-filter": "sensor,input_number,number",
        },
    )
    entity_negative: Optional[EntityId] = Field(
        default=None,
        alias="entity negative",
        description="HA entity with the negative-only part of the power value",
        json_schema_extra={
            "x-help": "Alternative to 'entity': the sensor that only reports the negative "
            "direction (grid export, or battery discharging), as a positive number. "
            "Combine with 'entity positive'.",
            "x-ui-widget-filter": "sensor,input_number,number",
        },
    )
    unit: Literal["W", "kW"] = Field(
        default="W",
        description="Unit of the sensor value",
        json_schema_extra={
            "x-help": "Unit reported by the sensor. kW values are converted to W internally.",
            "x-unit": "W or kW",
        },
    )
    invert: bool = Field(
        default=False,
        description="Flip the sign of the measured value",
        json_schema_extra={
            "x-help": "Enable when your integration reports the opposite sign, for example "
            "a grid sensor that is positive while exporting."
        },
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="after")
    def validate_source(self) -> "PowerSensor":
        """A sensor is either signed or a positive/negative pair, never both."""
        signed = self.entity is not None
        split = self.entity_positive is not None or self.entity_negative is not None
        if signed and split:
            raise ValueError(
                "use either 'entity' (signed) or 'entity positive'/'entity negative', not both"
            )
        return self

    @property
    def configured(self) -> bool:
        """True when at least one source entity is set."""
        return (
            self.entity is not None
            or self.entity_positive is not None
            or self.entity_negative is not None
        )

    @property
    def entity_ids(self) -> list[str]:
        """All entity ids this sensor reads, in a stable order."""
        return [
            e
            for e in (self.entity, self.entity_positive, self.entity_negative)
            if e is not None
        ]


class FastBatteryLink(BaseModel):
    """Links a battery from the `battery` section to its realtime measurements."""

    name: str = Field(
        description="Battery name, must match a name in the battery section",
        json_schema_extra={
            "x-help": "Exact name of the battery as configured under 'battery'. The fast "
            "control layer reuses that battery's stages, capacity, efficiency and "
            "setpoint entity."
        },
    )
    enabled: bool = Field(
        default=True,
        description="Allow the fast control layer to steer this battery",
        json_schema_extra={
            "x-help": "Disable to keep this battery strictly on the day-ahead plan while "
            "other batteries are steered in realtime."
        },
    )
    priority: int = Field(
        default=0,
        description="Dispatch order, lowest number is corrected first",
        json_schema_extra={
            "x-help": "With multiple batteries the correction is handed to the battery with "
            "the lowest priority number first; the remainder spills over to the next one."
        },
    )
    actual_power: PowerSensor = Field(
        default_factory=PowerSensor,
        alias="actual power",
        description="Measured AC power of this battery, positive is charging",
        json_schema_extra={
            "x-help": "Strongly recommended. Without it the controller assumes the inverter "
            "follows its last command exactly, which makes it blind to derating, "
            "standby losses and manual overrides.",
            "x-ui-section": "Measurements",
        },
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class FastControlDiagnostics(BaseModel):
    """Entities the fast control layer writes its own state to."""

    entity_status: Optional[str] = Field(
        default="sensor.dao_fast_control",
        alias="entity status",
        description="Entity written with the controller status and attributes",
        json_schema_extra={
            "x-help": "Written through the Home Assistant states API, so the entity does not "
            "have to exist up front. Its attributes carry the full decision trace: "
            "deviation, storage value, benefit, budget use and the reason for the "
            "current decision. Leave empty to disable.",
            "x-ui-section": "Diagnostics",
        },
    )
    entity_active: Optional[EntityId] = Field(
        default=None,
        alias="entity active",
        description="Optional input_boolean set to on while an override is active",
        json_schema_extra={
            "x-help": "Optional helper (input_boolean) that mirrors whether the fast layer is "
            "currently deviating from the day-ahead plan.",
            "x-ui-section": "Diagnostics",
            "x-ui-widget-filter": "input_boolean",
        },
    )
    entity_setpoint: Optional[EntityId] = Field(
        default=None,
        alias="entity setpoint",
        description="Optional input_number receiving the corrected battery setpoint in W",
        json_schema_extra={
            "x-help": "Optional helper (input_number) that records the setpoint the fast "
            "layer computed, in watts. Useful in shadow mode to compare the fast "
            "layer against the plan without touching the inverter.",
            "x-ui-section": "Diagnostics",
            "x-ui-widget-filter": "input_number",
        },
    )
    entity_benefit: Optional[EntityId] = Field(
        default=None,
        alias="entity benefit",
        description="Optional input_number receiving the estimated benefit in euro/hour",
        json_schema_extra={
            "x-help": "Optional helper (input_number) with the instantaneous estimated "
            "saving rate of the current correction, in euro per hour.",
            "x-ui-section": "Diagnostics",
            "x-ui-widget-filter": "input_number",
        },
    )
    entity_saved_today: Optional[EntityId] = Field(
        default=None,
        alias="entity saved today",
        description="Optional input_number with the cumulative saving of today in euro",
        json_schema_extra={
            "x-help": "Optional helper (input_number) accumulating the estimated saving since "
            "midnight, in euro. Resets at midnight.",
            "x-ui-section": "Diagnostics",
            "x-ui-widget-filter": "input_number",
        },
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class FastControlConfig(DAOConfigBaseModel):
    """Realtime feedback layer on top of the day-ahead plan."""

    mode: FlexEnum = Field(
        default=FlexEnum(value="off", enum_values=["off", "shadow", "active"]),
        description="Operating mode of the fast control layer",
        json_schema_extra={
            "x-help": "**off** - the layer does not run at all.\n\n"
            "**shadow** - the layer runs, logs every decision and writes its "
            "diagnostics, but never touches the inverter. Always start here and "
            "compare the logged setpoints against reality for a few days.\n\n"
            "**active** - the layer writes the corrected setpoint to the battery.\n\n"
            "May also be a Home Assistant entity (input_select) so you can switch "
            "modes without restarting the add-on.",
            "x-ui-section": "Main",
            "x-order": 1,
            "x-validation-hint": "'off', 'shadow' or 'active', or an HA entity id",
            "x-enum-values": ["off", "shadow", "active"],
            "x-ui-widget-filter": "input_select,select,sensor",
        },
    )
    interval: int = Field(
        default=15,
        ge=5,
        le=300,
        description="Control loop period in seconds",
        json_schema_extra={
            "x-help": "How often the P1 meter is sampled and the setpoint recomputed. 10-20 "
            "seconds is a good balance: fast enough to catch an oven or a kettle, "
            "slow enough to stay well inside Home Assistant's API budget. Every "
            "cycle costs one templated API call.",
            "x-unit": "s",
            "x-ui-section": "Main",
            "x-order": 2,
        },
    )
    grid_power: PowerSensor = Field(
        default_factory=PowerSensor,
        alias="grid power",
        description="Realtime P1 power measurement, positive is import",
        json_schema_extra={
            "x-help": "The single most important input. Point this at your P1 reader's "
            "instantaneous power sensor. Without it the fast control layer cannot "
            "run.",
            "x-ui-section": "Measurements",
            "x-order": 10,
        },
    )
    pv_power: PowerSensor = Field(
        default_factory=PowerSensor,
        alias="pv power",
        description="Optional realtime PV power, used for diagnostics only",
        json_schema_extra={
            "x-help": "Optional. The control law does not need it because PV is already "
            "contained in the P1 measurement, but logging it makes the decision "
            "trace far easier to interpret.",
            "x-ui-section": "Measurements",
            "x-order": 11,
        },
    )
    max_sensor_age: int = Field(
        default=120,
        ge=10,
        alias="max sensor age",
        description="Maximum age of a measurement in seconds before it is considered stale",
        json_schema_extra={
            "x-help": "When the grid or battery measurement has not updated within this "
            "window the controller falls back to the day-ahead plan. Protects "
            "against a frozen P1 integration silently driving the battery.",
            "x-unit": "s",
            "x-ui-section": "Safety",
            "x-order": 20,
        },
    )
    max_plan_age: int = Field(
        default=5400,
        ge=300,
        alias="max plan age",
        description="Maximum age of the day-ahead plan in seconds before overriding stops",
        json_schema_extra={
            "x-help": "If the optimizer has not produced a fresh plan within this window "
            "the fast layer stops overriding. Keep it slightly above your longest "
            "gap between calc_optimum runs (default allows one missed hourly run).",
            "x-unit": "s",
            "x-ui-section": "Safety",
            "x-order": 21,
        },
    )
    batteries: list[FastBatteryLink] = Field(
        default_factory=list,
        description="Per battery realtime measurement links",
        json_schema_extra={
            "x-help": "Leave empty to steer every configured battery with default settings. "
            "Add entries to attach a measured power sensor or to exclude a battery.",
            "x-ui-section": "Batteries",
            "x-order": 30,
        },
    )

    # --- economics -------------------------------------------------------
    storage_value_mode: FlexEnum = Field(
        default=FlexEnum(
            value="plan", enum_values=["plan", "average", "fixed"]
        ),
        alias="storage value mode",
        description="How the marginal value of stored energy is determined",
        json_schema_extra={
            "x-help": "The controller compares the current grid price against the marginal "
            "value of a kWh sitting in the battery.\n\n"
            "**plan** - derived from the remaining day-ahead plan: the cheaper of "
            "(best future use of the energy) and (cheapest future refill). This is "
            "the recommended setting.\n\n"
            "**average** - the average consumption price over the remaining horizon, "
            "the same valuation the day-ahead optimizer itself uses.\n\n"
            "**fixed** - use the 'storage value' field, optionally backed by an HA "
            "entity so you can tune it live.",
            "x-ui-section": "Economics",
            "x-order": 40,
            "x-enum-values": ["plan", "average", "fixed"],
            "x-ui-widget-filter": "input_select,select,sensor",
        },
    )
    storage_value: Optional[FlexFloat] = Field(
        default=None,
        alias="storage value",
        description="Marginal value of stored energy in euro/kWh when mode is 'fixed'",
        json_schema_extra={
            "x-help": "Only used when 'storage value mode' is 'fixed'. Discharging happens "
            "when the import price exceeds this value, charging from surplus when "
            "this value exceeds the export price. A sensible starting point is your "
            "typical evening import price times the round trip efficiency.",
            "x-unit": "euro/kWh",
            "x-ui-section": "Economics",
            "x-order": 41,
        },
    )
    round_trip_efficiency: float = Field(
        default=0.90,
        gt=0.1,
        le=1.0,
        alias="round trip efficiency",
        description="AC to AC round trip efficiency used to value stored energy",
        json_schema_extra={
            "x-help": "Used to discount the value of stored energy. Should match the real "
            "AC-to-AC efficiency of your inverter plus battery, typically 0.85-0.92.",
            "x-unit": "ratio",
            "x-ui-section": "Economics",
            "x-order": 42,
        },
    )
    min_benefit: float = Field(
        default=0.02,
        ge=0.0,
        alias="min benefit",
        description="Minimum estimated benefit in euro/hour before the plan is overridden",
        json_schema_extra={
            "x-help": "An override is only issued when the estimated saving rate exceeds "
            "this threshold. Raise it to make the layer more conservative and to "
            "ignore small, short deviations. 0.02 euro/hour corresponds to roughly "
            "a 100 W correction at a 0.20 euro/kWh spread.",
            "x-unit": "euro/h",
            "x-ui-section": "Economics",
            "x-order": 43,
        },
    )

    # --- stability and wear ---------------------------------------------
    deadband: int = Field(
        default=150,
        ge=0,
        description="Minimum setpoint change in W before a new command is sent",
        json_schema_extra={
            "x-help": "Suppresses command chatter caused by measurement noise. Set it above "
            "the noise level of your P1 sensor, typically 100-250 W.",
            "x-unit": "W",
            "x-ui-section": "Stability",
            "x-order": 50,
        },
    )
    min_command_interval: int = Field(
        default=60,
        ge=0,
        alias="min command interval",
        description="Minimum time in seconds between two setpoint writes",
        json_schema_extra={
            "x-help": "Protects the inverter and its Modbus/cloud link against excessive "
            "write rates, and keeps the battery from hunting. Check your inverter's "
            "documented minimum setpoint interval.",
            "x-unit": "s",
            "x-ui-section": "Stability",
            "x-order": 51,
        },
    )
    urgent_deviation: int = Field(
        default=1500,
        ge=0,
        alias="urgent deviation",
        description="Deviation in W that is allowed to bypass the minimum command interval",
        json_schema_extra={
            "x-help": "A large step, such as an oven or an EV charger starting, may be acted "
            "on immediately instead of waiting out the minimum command interval. "
            "Set to 0 to always respect the interval.",
            "x-unit": "W",
            "x-ui-section": "Stability",
            "x-order": 52,
        },
    )
    max_ramp: Optional[int] = Field(
        default=None,
        ge=0,
        alias="max ramp",
        description="Maximum setpoint change in W per command, empty means unlimited",
        json_schema_extra={
            "x-help": "Optional slew rate limit. Most hybrid inverters ramp internally and do "
            "not need this. Set it if your inverter trips on large setpoint steps.",
            "x-unit": "W",
            "x-ui-section": "Stability",
            "x-order": 53,
        },
    )
    release_deviation: int = Field(
        default=100,
        ge=0,
        alias="release deviation",
        description="Deviation in W below which the controller returns to the plan",
        json_schema_extra={
            "x-help": "Together with 'release time' this forms the hysteresis that ends an "
            "override. Keep it below the deadband to avoid immediate re-engagement.",
            "x-unit": "W",
            "x-ui-section": "Stability",
            "x-order": 54,
        },
    )
    release_time: int = Field(
        default=120,
        ge=0,
        alias="release time",
        description="Time in seconds the deviation must stay small before returning to the plan",
        json_schema_extra={
            "x-help": "Prevents the controller from dropping an override during a short dip, "
            "for example between two heating elements of an oven cycling.",
            "x-unit": "s",
            "x-ui-section": "Stability",
            "x-order": 55,
        },
    )

    # --- budgets ---------------------------------------------------------
    energy_budget: float = Field(
        default=0.5,
        ge=0.0,
        alias="energy budget",
        description="Allowed energy deviation from the planned SoC trajectory, in kWh",
        json_schema_extra={
            "x-help": "The trust region around the day-ahead plan. The fast layer may move "
            "at most this much energy more or less than the plan intended within a "
            "single plan interval; after that it snaps back. This is what keeps a "
            "wrong storage value from emptying the battery before the evening peak. "
            "0.3-1.0 kWh works well for a 10-30 kWh battery.",
            "x-unit": "kWh",
            "x-ui-section": "Budgets",
            "x-order": 60,
        },
    )
    daily_extra_throughput: float = Field(
        default=4.0,
        ge=0.0,
        alias="daily extra throughput",
        description="Maximum extra battery throughput per day caused by the fast layer, in kWh",
        json_schema_extra={
            "x-help": "Hard cap on the additional wear the fast layer is allowed to cause. "
            "Counted as the integral of the absolute difference between the actual "
            "and the planned battery power. Once exhausted the layer follows the "
            "plan for the rest of the day. Set to 0 to disable the cap.",
            "x-unit": "kWh/day",
            "x-ui-section": "Budgets",
            "x-order": 61,
        },
    )
    soc_margin: float = Field(
        default=2.0,
        ge=0.0,
        le=50.0,
        alias="soc margin",
        description="Extra SoC margin in percent kept away from the configured battery limits",
        json_schema_extra={
            "x-help": "The fast layer stays this far inside the battery's 'lower limit' and "
            "'upper limit'. The day-ahead plan may use the full range; the realtime "
            "corrections may not, so a failed plan refresh can never leave the "
            "battery stranded at a limit.",
            "x-unit": "%",
            "x-ui-section": "Budgets",
            "x-order": 62,
        },
    )
    max_grid_import: Optional[int] = Field(
        default=None,
        ge=0,
        alias="max grid import",
        description="Optional peak shaving limit on grid import in W",
        json_schema_extra={
            "x-help": "When set, the fast layer also acts as a peak shaver: it discharges "
            "whatever is needed to keep grid import below this value, regardless of "
            "price, as long as the battery has energy and the SoC margin allows it. "
            "Useful for capacity tariffs or a marginal main fuse. Leave empty to "
            "disable.",
            "x-unit": "W",
            "x-ui-section": "Budgets",
            "x-order": 63,
        },
    )
    allow_grid_charge: FlexBool = Field(
        default=FlexBool(value=False),
        alias="allow grid charge",
        description="Allow the fast layer to increase charging beyond the plan from the grid",
        json_schema_extra={
            "x-help": "When disabled (default) the fast layer will charge more than planned "
            "only from a genuine surplus, so it never turns a planned export into a "
            "grid purchase. Enabling it lets the layer also buy during very cheap or "
            "negative price intervals, which is profitable but increases cycling.",
            "x-ui-section": "Budgets",
            "x-order": 64,
            "x-ui-widget-filter": "input_boolean,switch",
        },
    )

    diagnostics: FastControlDiagnostics = Field(
        default_factory=FastControlDiagnostics,
        description="Entities written with the controller state",
        json_schema_extra={"x-ui-section": "Diagnostics", "x-order": 70},
    )

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "x-ui-group": "Energy",
            "x-icon": "speedometer",
            "x-order": 13,
            "x-help": """# Fast Control

A realtime feedback layer that sits on top of the day-ahead plan.

## The problem it solves

The optimizer plans on a 15-minute or hourly grid using a *forecast* of the house
load and the PV production. Between two runs the battery setpoint is frozen.
Every watt of forecast error therefore flows straight through the meter:

- An oven starting at 18:03 is imported at the full consumption price, even
  though the battery is standing by with cheap energy.
- A cloud passing over the array turns a planned self-consumption into a grid
  purchase.
- A load that finishes early turns a planned discharge into an export at the
  much lower feed-in price.

Each of those costs you the spread between the import and the export price.

## How it works

Every few seconds the layer reads the P1 meter, reconstructs the true house load
by subtracting the measured battery power, and compares it with the plan. It
then solves a one-dimensional cost minimisation over the grid setpoint, using
the current import price, the current export price and the marginal value of a
kWh in the battery. The result is clamped by the inverter limits, an energy
trust region around the planned state of charge, and a daily wear budget.

In the normal price regime the solution is simply "cover the deviation from the
battery". During negative prices or a price spike the same formula automatically
produces the opposite behaviour, without any special casing.

## Getting started

1. Configure `grid power` with your P1 instantaneous power sensor.
2. Configure `actual power` for each battery if you have a sensor for it.
3. Leave `mode` on **shadow** for a few days and inspect
   `sensor.dao_fast_control` and the log.
4. Run the backtest to quantify the expected saving:
   `python3 da_fast.py simulate --days 14`
5. Switch `mode` to **active**.

## Tuning

- Too much switching: raise `deadband`, `min command interval` and `min benefit`.
- Too little effect: lower `min benefit`, raise `energy budget`.
- Battery empty before the evening peak: lower `energy budget`, or switch
  `storage value mode` to `fixed` with a high value.
""",
            "x-docs-url": "https://github.com/miclnl/day-ahead/wiki/Fast-Control",
        },
    )

    @property
    def enabled_batteries(self) -> list[FastBatteryLink]:
        """Configured battery links that are enabled, in dispatch order."""
        return sorted(
            [b for b in self.batteries if b.enabled], key=lambda b: (b.priority, b.name)
        )
