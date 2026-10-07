"""Weather source configuration: forecast fallback and observation source."""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class WeatherConfig(BaseModel):
    """How the forecast and its fallback, and the observation source, are chosen."""

    fallback: Optional[Literal["openmeteo"]] = Field(
        default="openmeteo",
        description="Weather source used to fill gaps in the primary forecast",
        json_schema_extra={
            "x-help": "Meteoserver is the primary forecast source when a key is "
            "configured. When it is unreachable, or its horizon ends before the "
            "optimizer's, Open-Meteo fills the remaining hours. Leave empty to "
            "never fall back -- a gap then stays a gap.",
            "x-ui-section": "Weather",
            "x-order": 130,
        },
    )
    openmeteo_model: str = Field(
        default="knmi_seamless",
        alias="openmeteo model",
        description="Open-Meteo forecast model",
        json_schema_extra={
            "x-help": "knmi_seamless blends KNMI's short-range model with a global "
            "one for the days beyond it. See open-meteo.com/en/docs for the full "
            "model list.",
            "x-ui-section": "Weather",
            "x-order": 131,
        },
    )
    observations: Literal["auto", "knmi", "openmeteo", "off"] = Field(
        default="auto",
        description="Source used to backfill measured weather (gr/temp/winds)",
        json_schema_extra={
            "x-help": "auto uses KNMI in the Netherlands and Belgium and "
            "Open-Meteo's archive elsewhere. Measured weather is what the "
            "accuracy report and the PV calibration compare a forecast against.",
            "x-ui-section": "Weather",
            "x-order": 132,
        },
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)
