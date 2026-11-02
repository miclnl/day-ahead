"""
Reporting configuration models.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import EntityId


class ReportConfig(BaseModel):
    """Reporting and sensor entity configuration."""

    entities_grid_consumption: list[EntityId] = Field(
        default_factory=list,
        alias="entities grid consumption",
        description="HA entities for grid consumption",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities measuring grid consumption (import). Used for actual vs optimized comparison and reporting.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    entities_grid_production: list[EntityId] = Field(
        default_factory=list,
        alias="entities grid production",
        description="HA entities for grid production",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities measuring grid production (export/feed-in). Used for production reporting and solar surplus calculation.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    entities_solar_production_ac: list[EntityId] = Field(
        default_factory=list,
        alias="entities solar production ac",
        description="HA entities for AC solar production",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities for AC-coupled solar production. After inverter, directly to AC side. Separate from DC-coupled solar.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    entities_solar_production_dc: list[EntityId] = Field(
        default_factory=list,
        alias="entities solar production dc",
        description="HA entities for DC solar production",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities for DC-coupled solar production. Before inverter, directly to battery DC bus. Separate from AC-coupled solar.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    entities_ev_consumption: list[EntityId] = Field(
        default_factory=list,
        alias="entities ev consumption",
        description="HA entities for EV consumption",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities measuring EV charging consumption. Used for EV-specific energy reporting and cost allocation.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    entities_wp_consumption: list[EntityId] = Field(
        default_factory=list,
        alias="entities wp consumption",
        description="HA entities for heat pump (warmtepomp) consumption",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities measuring heat pump consumption. Used for heating-specific energy reporting and COP calculation.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    entities_boiler_consumption: list[EntityId] = Field(
        default_factory=list,
        alias="entities boiler consumption",
        description="HA entities for boiler consumption",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities measuring hot water boiler consumption. Used for boiler-specific energy reporting.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    entities_battery_consumption: list[EntityId] = Field(
        default_factory=list,
        alias="entities battery consumption",
        description="HA entities for battery consumption",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities measuring battery charging (consumption). Used for battery efficiency and cycling cost calculation.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    entities_battery_production: list[EntityId] = Field(
        default_factory=list,
        alias="entities battery production",
        description="HA entities for battery production",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities measuring battery discharging (production). Used for battery efficiency and round-trip calculation.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    entities_machine_consumption: list[EntityId] = Field(
        default_factory=list,
        alias="entities machine consumption",
        description="HA entities for machine consumption",
        json_schema_extra={
            "x-help": "List of Home Assistant sensor entities measuring appliance/machine consumption (washing machine, dishwasher, etc.). Used for machine-specific reporting.",
            "x-unit": "kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    co2_intensity_sensor: Optional[EntityId] = Field(
        default=None,
        alias="co2 intensity sensor",
        description="HA entity for CO2 intensity",
        json_schema_extra={
            "x-help": "Optional: One Home Assistant sensor for grid CO2 intensity (gCO2/kWh). "
            "Accepts 'co2 intensity sensor', 'co2_intensity_sensor' or the legacy key "
            "'entity co2-intensity', as an entity ID string or a list containing exactly "
            "one entity ID. Normalized to a single typed entity ID; null or [] means absent. "
            "Multiple sensors, invalid entity IDs and conflicting aliases are rejected. "
            "Equivalent aliases are accepted. Saving the raw configuration preserves its "
            "key spelling and value form. Used to calculate and report carbon footprint "
            "of electricity usage.",
            "x-unit": "gCO2/kWh",
            "x-ui-section": "General",
            "x-ui-widget-filter": "sensor",
        },
    )
    sensors: Optional[dict] = Field(
        default=None,
        description="Additional sensors configuration",
        json_schema_extra={
            "x-help": "Optional: Additional custom sensor configurations. Advanced use for extending reporting capabilities beyond standard entities.",
            "x-ui-section": "General",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_co2_sensor(cls, data):
        """Consume all CO2 spellings without changing the raw configuration."""
        if not isinstance(data, dict):
            return data
        aliases = ("co2 intensity sensor", "co2_intensity_sensor", "entity co2-intensity")
        normalized = {}
        for key in aliases:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, list):
                if not value:
                    value = None
                elif len(value) == 1 and isinstance(value[0], str):
                    value = value[0]
                else:
                    raise ValueError("CO2 intensity must contain exactly one sensor entity ID")
            if value is not None and not isinstance(value, str):
                raise ValueError("CO2 intensity must be an entity ID string, singleton list or null")
            normalized[key] = value
        if not normalized:
            return data
        if len(set(normalized.values())) > 1:
            raise ValueError(f"Conflicting CO2 intensity sensor aliases: {', '.join(normalized)}")
        result = dict(data)
        for key in normalized:
            del result[key]
        result["co2 intensity sensor"] = next(iter(normalized.values()))
        return result

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "x-ui-group": "Reporting",
            "x-icon": "chart-bar",
            "x-order": 17,
            "x-help": """# Reporting & Sensors Configuration

Configure Home Assistant sensors for comprehensive energy reporting.

## Purpose

Connect actual energy measurements to Day Ahead Optimizer:
- Compare actual vs optimized usage
- Track device-specific consumption
- Calculate efficiency metrics
- Generate cost reports
- Monitor carbon footprint

## Entity Lists

The `entities ...` fields accept lists of HA sensors:
```json
"entities_grid_consumption": [
  "sensor.grid_import_meter",
  "sensor.backup_meter"
]
```

CO2 intensity uses a single typed entity ID, not a multi-sensor list. The keys
`co2 intensity sensor`, `co2_intensity_sensor` and legacy `entity co2-intensity`
accept a string or a one-sensor list. `null` and `[]` disable CO2 reporting.
Multiple sensors, invalid entity IDs and conflicting aliases are rejected.

## Key Measurements

### Grid
- **Consumption**: Total import from grid
- **Production**: Total export/feed-in to grid

### Solar
- **AC**: After inverter, to AC distribution
- **DC**: Before inverter, directly to battery

### Devices
- **EV**: Charging consumption
- **Heat Pump**: Heating/cooling consumption
- **Boiler**: Hot water heating consumption
- **Battery**: Charge/discharge tracking
- **Machines**: Appliance consumption

## Tips

- Use energy sensors (kWh cumulative)
- Multiple sensors per category are summed
- Essential for accurate cost reporting
- CO2 tracking optional but insightful
- Update when adding new monitoring devices
""",
            "x-docs-url": "https://github.com/miclnl/day-ahead/wiki/Reports",
        },
    )
