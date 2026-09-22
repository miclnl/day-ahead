"""Robust estimation of the daily baseload profile.

The baseload is the whole consumption forecast the optimizer works with: 24
values per weekday, in kWh. It is estimated from roughly eight observations per
(weekday, hour) cell, which is very little. The plain arithmetic mean that was
used before is the worst possible estimator on that few samples -- one party,
one week of holiday or one recorder gap shifts a cell by an eighth of the
excursion, and stays in the window for two months.

Everything here is pure: lists in, lists out, no database, no configuration
object, no I/O. That keeps the statistics testable on their own and keeps the
reporting module free of statistics.

Four things are layered on top of a plain average, in this order:

1. **Outlier rejection** per cell, on the interquartile range.
2. **Recency weighting**, so a change in the household propagates in weeks
   rather than in two months.
3. **A robust location estimate** -- weighted median by default.
4. **A pooled fallback** for cells with too few observations left, so a thin
   cell borrows from the same hour on other days instead of inventing a value.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from dateutil import easter

#: Cells with fewer observations than this after filtering fall back to the
#: pooled estimate for the same hour across all weekdays.
DEFAULT_MIN_SAMPLES = 3

#: Below this many observations outlier rejection is switched off: with four
#: points the interquartile range is meaningless and would throw away real
#: variation.
MIN_SAMPLES_FOR_OUTLIERS = 5

PROFILE_FORMAT_VERSION = 1


@dataclass
class BaseloadOptions:
    """Tunables, mirrored from the configuration."""

    aggregate: str = "median"
    trim_fraction: float = 0.2
    remove_outliers: bool = True
    outlier_factor: float = 2.0
    half_life_days: Optional[float] = 28.0
    holidays: str = "sunday"
    clip_negative: bool = True
    min_samples: int = DEFAULT_MIN_SAMPLES


@dataclass
class Sample:
    """One observed hour."""

    #: Days between the observation and the moment of calculation.
    age_days: float
    value: float


@dataclass
class BaseloadProfile:
    """A 24 value profile plus how well founded each value is."""

    values: list[float] = field(default_factory=lambda: [0.0] * 24)
    samples: list[int] = field(default_factory=lambda: [0] * 24)
    pooled: list[bool] = field(default_factory=lambda: [False] * 24)

    @property
    def total(self) -> float:
        return sum(self.values)


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------


def dutch_holidays(year: int) -> set:
    """The public holidays that change household behaviour.

    Deliberately the set that makes a weekday look like a Sunday. Not an
    exhaustive legal list: Good Friday and Liberation Day are working days for
    most people and are left out on purpose.
    """
    pasen = easter.easter(year)
    return {
        datetime.date(year, 1, 1),
        datetime.date(year, 4, 27),
        datetime.date(year, 12, 25),
        datetime.date(year, 12, 26),
        pasen + datetime.timedelta(days=1),  # tweede paasdag
        pasen + datetime.timedelta(days=39),  # hemelvaart
        pasen + datetime.timedelta(days=50),  # tweede pinksterdag
    }


def is_holiday(day: datetime.date) -> bool:
    return day in dutch_holidays(day.year)


def effective_weekday(day: datetime.date, mode: str = "sunday") -> int:
    """Weekday a day should be treated as, 0 = Monday.

    A public holiday has the consumption pattern of a weekend day, not of the
    weekday it happens to fall on. Folding it into the Sunday profile keeps it
    from contaminating, say, every Thursday for two months.
    """
    weekday = day.weekday()
    if mode == "ignore" or not is_holiday(day):
        return weekday
    return 5 if mode == "saturday" else 6


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolated quantile of an already sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def outlier_mask(values: Sequence[float], factor: float = 2.0) -> list[bool]:
    """Keep mask based on the interquartile range.

    A wider factor than the textbook 1.5 because household consumption is
    genuinely skewed: the aim is to remove the party and the recorder gap, not
    the slightly busier evening.
    """
    if len(values) < MIN_SAMPLES_FOR_OUTLIERS:
        return [True] * len(values)
    ordered = sorted(values)
    q1 = quantile(ordered, 0.25)
    q3 = quantile(ordered, 0.75)
    spread = q3 - q1
    if spread <= 0:
        return [True] * len(values)
    low = q1 - factor * spread
    high = q3 + factor * spread
    mask = [low <= v <= high for v in values]
    # Never throw away everything; if the filter is that aggressive the cell
    # is simply noisy and the robust estimator should handle it instead.
    return mask if any(mask) else [True] * len(values)


def recency_weights(
    ages: Sequence[float], half_life_days: Optional[float]
) -> list[float]:
    """Exponential decay, so recent weeks count more than old ones."""
    if not half_life_days or half_life_days <= 0:
        return [1.0] * len(ages)
    return [0.5 ** (max(0.0, age) / half_life_days) for age in ages]


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Value at which half of the total weight is reached."""
    if not values:
        return 0.0
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    if total <= 0:
        return float(pairs[len(pairs) // 2][0])
    running = 0.0
    half = total / 2.0
    for value, weight in pairs:
        running += weight
        if running >= half:
            return float(value)
    return float(pairs[-1][0])


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return sum(values) / len(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def trimmed(values: Sequence[float], weights: Sequence[float], fraction: float):
    """Drop *fraction* of the observations from each tail, by value."""
    count = len(values)
    drop = int(count * max(0.0, min(0.49, fraction)))
    if drop <= 0 or count - 2 * drop < 1:
        return list(values), list(weights)
    order = sorted(range(count), key=lambda i: values[i])
    keep = set(order[drop : count - drop])
    return (
        [values[i] for i in range(count) if i in keep],
        [weights[i] for i in range(count) if i in keep],
    )


def estimate_cell(
    samples: Sequence[Sample], options: BaseloadOptions
) -> tuple[Optional[float], int]:
    """Robust estimate for one (weekday, hour) cell.

    Returns the estimate and the number of observations it rests on, or
    ``(None, n)`` when too little is left to be trusted.
    """
    if not samples:
        return None, 0

    values = [s.value for s in samples]
    ages = [s.age_days for s in samples]

    if options.remove_outliers:
        mask = outlier_mask(values, options.outlier_factor)
        values = [v for v, keep in zip(values, mask) if keep]
        ages = [a for a, keep in zip(ages, mask) if keep]

    if len(values) < max(1, options.min_samples):
        return None, len(values)

    weights = recency_weights(ages, options.half_life_days)

    if options.aggregate == "mean":
        estimate = weighted_mean(values, weights)
    elif options.aggregate == "trimmed":
        kept_values, kept_weights = trimmed(values, weights, options.trim_fraction)
        estimate = weighted_mean(kept_values, kept_weights)
    else:
        estimate = weighted_median(values, weights)

    if options.clip_negative:
        # A negative baseload is physically impossible. It happens when the
        # grid meter has a recorder gap while the PV meter does not, and it
        # lets the solver harvest energy that never existed.
        estimate = max(0.0, estimate)
    return estimate, len(values)


def build_profile(
    cells: dict,
    pooled_cells: Optional[dict] = None,
    options: Optional[BaseloadOptions] = None,
) -> BaseloadProfile:
    """Turn per hour observations into a 24 value profile.

    ``cells`` maps hour of day to a list of :class:`Sample`. ``pooled_cells``
    is the same structure but gathered over all weekdays, used for hours where
    this weekday alone has too little to say.
    """
    options = options or BaseloadOptions()
    profile = BaseloadProfile()
    for hour in range(24):
        estimate, count = estimate_cell(cells.get(hour, []), options)
        pooled = False
        if estimate is None and pooled_cells is not None:
            estimate, _ = estimate_cell(pooled_cells.get(hour, []), options)
            pooled = estimate is not None
        if estimate is None:
            estimate = 0.0
        profile.values[hour] = round(estimate, 3)
        profile.samples[hour] = count
        profile.pooled[hour] = pooled
    return profile


# ---------------------------------------------------------------------------
# file format
# ---------------------------------------------------------------------------


def profile_to_dict(
    profile: BaseloadProfile, weekday: int, period_days: int, options: BaseloadOptions
) -> dict:
    return {
        "version": PROFILE_FORMAT_VERSION,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": weekday,
        "period_days": period_days,
        "aggregate": options.aggregate,
        "baseload": profile.values,
        "samples": profile.samples,
        "pooled": profile.pooled,
    }


def profile_from_file(payload) -> list[float]:
    """Read either the bare list of the old format or the new dictionary."""
    if isinstance(payload, dict):
        values = payload.get("baseload")
    else:
        values = payload
    if not isinstance(values, list) or len(values) != 24:
        raise ValueError(
            f"baseload-profiel moet 24 waarden bevatten, gevonden: "
            f"{len(values) if isinstance(values, list) else type(values).__name__}"
        )
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError) as exception:
            raise ValueError(f"ongeldige baseload-waarde: {value!r}") from exception
        if number != number:
            raise ValueError("baseload-profiel bevat NaN")
        result.append(max(0.0, number))
    return result


def profile_age_days(payload) -> Optional[float]:
    """How old the stored profile is, or None for the old format."""
    if not isinstance(payload, dict):
        return None
    created = payload.get("created")
    if not created:
        return None
    try:
        moment = datetime.datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return (datetime.datetime.now() - moment).total_seconds() / 86400.0


def iter_samples(
    rows: Iterable, reference: datetime.datetime, holidays: str = "sunday"
) -> dict:
    """Group ``(datetime, value)`` rows into ``{weekday: {hour: [Sample]}}``."""
    grouped: dict = {}
    for moment, value in rows:
        if value is None or value != value:
            continue
        weekday = effective_weekday(moment.date(), holidays)
        age = (reference - moment).total_seconds() / 86400.0
        grouped.setdefault(weekday, {}).setdefault(moment.hour, []).append(
            Sample(age_days=age, value=float(value))
        )
    return grouped
