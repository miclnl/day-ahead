"""Baseload estimation configuration."""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from dao.prog.config.models.base import EntityId


class AbsenceConfig(BaseModel):
    """When the household is away, detected from consumption or told directly."""

    detect: bool = Field(
        default=True,
        description="Label past away days from consumption history",
        json_schema_extra={
            "x-help": "Without any extra sensors, a day counts as away when its active "
            "energy (the day's total minus standby) falls well below what recent "
            "days led the estimator to expect. Past away days feed a separate "
            "away profile, so absences are forecast rather than papered over "
            "with the ordinary weekday pattern.",
            "x-ui-section": "Baseload",
            "x-order": 120,
        },
    )
    threshold: float = Field(
        default=0.4,
        ge=0.1,
        le=0.9,
        description="Fraction of expected consumption below which a day counts as away",
        json_schema_extra={
            "x-help": "Once at least ten days have both a consumption label and an "
            "`entities presence` reading, this is recalibrated automatically "
            "against reality; the configured value only applies until then.",
            "x-ui-section": "Baseload",
            "x-order": 121,
        },
    )
    entities_presence: list[EntityId] = Field(
        default_factory=list,
        alias="entities presence",
        description="Person-tracker entities, used both to detect the regime and to "
        "calibrate the threshold",
        json_schema_extra={"x-ui-section": "Baseload", "x-order": 122},
    )
    entity_away: Optional[EntityId] = Field(
        default=None,
        alias="entity away",
        description="An entity whose state alone decides the household is away",
        json_schema_extra={
            "x-help": "Takes priority over every other signal. For "
            "`alarm_control_panel`, `away state: armed_away` is the usual choice.",
            "x-ui-section": "Baseload",
            "x-order": 123,
        },
    )
    away_state: str = Field(
        default="on",
        alias="away state",
        description="The state of `entity away` that means the household is away",
        json_schema_extra={"x-ui-section": "Baseload", "x-order": 124},
    )
    entity_calendar: Optional[EntityId] = Field(
        default=None,
        alias="entity calendar",
        description="A calendar entity whose events mark away periods",
        json_schema_extra={"x-ui-section": "Baseload", "x-order": 125},
    )
    calendar_keywords: list[str] = Field(
        default_factory=lambda: ["vakantie", "weg", "afwezig", "holiday"],
        alias="calendar keywords",
        description="Case-insensitive words in an event's title that mark it as away",
        json_schema_extra={"x-ui-section": "Baseload", "x-order": 126},
    )
    away_after_hours: int = Field(
        default=3,
        ge=1,
        alias="away after hours",
        description="Consecutive hours with nobody present marks the rest of today away",
        json_schema_extra={"x-ui-section": "Baseload", "x-order": 127},
    )
    assume_next_day_after_hours: int = Field(
        default=24,
        ge=1,
        alias="assume next day after hours",
        description="Consecutive hours with nobody present marks tomorrow away too",
        json_schema_extra={"x-ui-section": "Baseload", "x-order": 128},
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class BaseloadOptionsConfig(BaseModel):
    """How the daily baseload profile is estimated from history."""

    model: Literal["profile", "ml", "auto"] = Field(
        default="profile",
        description="Which model forecasts the baseload",
        json_schema_extra={
            "x-help": "**profile** - the twenty-four values per weekday estimated from "
            "history. Predictable, needs little data, cannot react to the "
            "weather.\n\n"
            "**ml** - an XGBoost model on hour, weekday, season, temperature, sun "
            "elevation and the away flag. Needs roughly four months of history "
            "before it beats the profile.\n\n"
            "**auto** - backtests both on every `calc_baseloads` run over the last "
            "`backtest days` days and keeps whichever had the lower error. Below "
            "`ml min days` of history this is the same as `profile`.",
            "x-ui-section": "Baseload",
            "x-order": 105,
        },
    )
    ml_min_days: int = Field(
        default=120,
        ge=30,
        alias="ml min days",
        description="History needed before 'auto' will consider the ML model",
        json_schema_extra={
            "x-help": "An XGBoost model on a few weeks of history mostly memorises those "
            "weeks. Four months is roughly where it starts to beat the profile on "
            "a backtest instead of only on its own training data.",
            "x-unit": "days",
            "x-ui-section": "Baseload",
            "x-order": 106,
        },
    )
    backtest_days: int = Field(
        default=28,
        ge=7,
        alias="backtest days",
        description="Window 'auto' compares the two models over",
        json_schema_extra={
            "x-help": "Both models forecast each day in this window using only data from "
            "before that day, and the one with the lower mean absolute error "
            "wins. Longer is a more reliable comparison but a slower "
            "`calc_baseloads` run.",
            "x-unit": "days",
            "x-ui-section": "Baseload",
            "x-order": 107,
        },
    )
    aggregate: Literal["median", "mean", "trimmed"] = Field(
        default="mean",
        description="Statistic used to combine the observations of one hour",
        json_schema_extra={
            "x-help": "The profile rests on roughly eight observations per weekday and "
            "hour, so the choice matters.\n\n"
            "**mean** - recommended. The optimizer plans an energy balance, and "
            "only the mean adds up to the energy actually used over the day; the "
            "median of each hour separately does not, and systematically "
            "under-plans a household with occasional heavy hours.\n\n"
            "**median** - the robust alternative. A single odd day cannot move an "
            "hour, at the cost of a profile whose daily total is too low.\n\n"
            "**trimmed** - drops the extremes and averages the rest, a middle "
            "ground between the two.",
            "x-ui-section": "Baseload",
            "x-order": 110,
        },
    )
    trim_fraction: float = Field(
        default=0.2,
        ge=0.0,
        lt=0.5,
        alias="trim fraction",
        description="Fraction dropped from each tail when aggregate is 'trimmed'",
        json_schema_extra={
            "x-help": "Only used with the 'trimmed' aggregate. 0.2 drops the highest and "
            "the lowest fifth of the observations before averaging.",
            "x-ui-section": "Baseload",
            "x-order": 111,
        },
    )
    remove_outliers: bool = Field(
        default=True,
        alias="remove outliers",
        description="Reject implausible observations before aggregating",
        json_schema_extra={
            "x-help": "Rejects observations far outside the interquartile range of their "
            "own hour. Catches parties, guests and recorder gaps. Switched off "
            "automatically for hours with fewer than five observations.",
            "x-ui-section": "Baseload",
            "x-order": 112,
        },
    )
    outlier_factor: float = Field(
        default=2.0,
        gt=0.0,
        alias="outlier factor",
        description="Interquartile range multiplier for outlier rejection",
        json_schema_extra={
            "x-help": "Higher is more permissive. The textbook value is 1.5; the default "
            "of 2.0 is deliberately wider because household consumption is "
            "genuinely skewed and the aim is to remove the exceptional day, not "
            "the merely busy one.",
            "x-ui-section": "Baseload",
            "x-order": 113,
        },
    )
    half_life_days: Optional[float] = Field(
        default=28.0,
        gt=0.0,
        alias="half life days",
        description="Half-life in days of the recency weighting, empty to disable",
        json_schema_extra={
            "x-help": "Observations of four weeks ago count half as heavily as those of "
            "today. Without this a new freezer or a departed housemate takes the "
            "whole calculation period to work through. Leave empty to weight "
            "every observation equally.",
            "x-unit": "days",
            "x-ui-section": "Baseload",
            "x-order": 114,
        },
    )
    holidays: Literal["sunday", "saturday", "ignore"] = Field(
        default="sunday",
        description="Which profile public holidays are folded into",
        json_schema_extra={
            "x-help": "A public holiday has the consumption pattern of a weekend day, not "
            "of the weekday it happens to fall on. Folding it into the Sunday "
            "profile keeps Christmas from contaminating every Thursday for two "
            "months. Covers New Year, King's Day, Easter Monday, Ascension, "
            "Whit Monday and both Christmas days.",
            "x-ui-section": "Baseload",
            "x-order": 115,
        },
    )
    clip_negative: bool = Field(
        default=True,
        alias="clip negative",
        description="Never let an estimated hour go below zero",
        json_schema_extra={
            "x-help": "A negative baseload is physically impossible. It occurs when the "
            "grid meter has a recorder gap while the solar meter does not, and "
            "it lets the optimizer plan with energy that never existed.",
            "x-ui-section": "Baseload",
            "x-order": 116,
        },
    )
    min_samples: int = Field(
        default=3,
        ge=1,
        alias="min samples",
        description="Observations needed before an hour is trusted on its own",
        json_schema_extra={
            "x-help": "Hours with fewer observations left after filtering borrow the "
            "estimate for the same hour pooled over all weekdays, rather than "
            "resting on one or two measurements.",
            "x-ui-section": "Baseload",
            "x-order": 117,
        },
    )
    absence: AbsenceConfig = Field(
        default_factory=AbsenceConfig,
        description="Away-day detection and anticipation",
        json_schema_extra={"x-ui-section": "Baseload", "x-order": 118},
    )

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "x-ui-group": "DAO",
            "x-icon": "chart-bell-curve",
            "x-order": 19,
            "x-help": """# Baseload estimation

The baseload is the entire consumption forecast the optimizer works with:
twenty-four values per weekday, in kilowatt hours, with everything the
optimizer schedules itself already subtracted.

It is estimated from the last `baseload calc periode` days, which at the
default of 56 days means about eight observations per weekday and hour. That is
very few, so how those eight are combined matters more than it looks.

## What the defaults do

- Public holidays are counted as Sundays instead of contaminating a weekday.
- Observations far outside the spread of their own hour are rejected.
- Recent weeks weigh more heavily, with a half-life of four weeks.
- The median is used rather than the average, so a single odd day cannot move
  an hour.
- Hours left with too few observations borrow from the same hour on other days.
- Nothing is ever negative.

## When to change something

- **Very regular household, want maximum responsiveness**: `aggregate: mean`
  and a shorter `half life days`.
- **Irregular household, a lot of variation**: keep the median and raise
  `baseload calc periode` to 84 days.
- **You want the old behaviour back**: `aggregate: mean`,
  `remove outliers: false`, `half life days` empty, `holidays: ignore`.

## Checking whether it helps

Run the forecast accuracy report, `<url>/api/run/forecast_accuracy`, and look
at the bias per hour. A consistently negative bias in the evening means the
optimizer reserves too little energy for the peak, which no amount of realtime
correction can repair afterwards.
""",
        },
    )
