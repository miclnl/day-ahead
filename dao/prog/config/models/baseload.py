"""Baseload estimation configuration."""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class BaseloadOptionsConfig(BaseModel):
    """How the daily baseload profile is estimated from history."""

    aggregate: Literal["median", "mean", "trimmed"] = Field(
        default="median",
        description="Statistic used to combine the observations of one hour",
        json_schema_extra={
            "x-help": "The profile rests on roughly eight observations per weekday and "
            "hour, so the choice matters.\n\n"
            "**median** - robust, a single odd day cannot move it. Recommended.\n\n"
            "**trimmed** - drops the extremes and averages the rest, a middle "
            "ground.\n\n"
            "**mean** - the old behaviour. One party or one recorder gap shifts "
            "the hour by an eighth of the excursion, for two months.",
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
