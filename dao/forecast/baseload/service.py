"""Baseload forecasting service.

Fits the seven weekday profiles from Home Assistant history and serves them
to the optimizer. When no profile has been fitted yet it migrates the old
per-weekday files once, then falls back to the static configuration, and
only raises when neither exists.
"""

from __future__ import annotations

import datetime
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from dao.forecast.baseload.profile import (
    BaseloadOptions,
    BaseloadProfile,
    build_profile,
    effective_weekday,
    iter_samples,
    quantile,
)
from dao.forecast.baseload.store import (
    ProfileSet,
    load_profile_set,
    migrate_legacy_files,
    profile_age_days,
    save_profile_set,
)
from dao.forecast.history import (
    HistoryReader,
    baseload_from_components,
    component_caps,
    component_groups,
)
from dao.prog.utils import interpolate

#: Beyond this many days a saved profile set is still used, but flagged: the
#: household may have changed since the last fit.
MAX_PROFILE_AGE_DAYS = 14.0


@dataclass
class Regime:
    """Whether the target day is expected to be a normal or an away day.

    Extended in a later task with the signals (entity, calendar, presence,
    consumption) that decide it; here it is just the outcome.
    """

    away: bool = False
    reason: str = "none"
    switch_hour: Optional[int] = None


class BaseloadUnavailable(RuntimeError):
    """Neither a fitted profile set nor a static baseload is configured."""


def options_from_config(config) -> BaseloadOptions:
    """Translate the configuration's baseload options into the estimator's."""
    options = config.baseload_options
    return BaseloadOptions(
        aggregate=options.aggregate,
        trim_fraction=options.trim_fraction,
        remove_outliers=options.remove_outliers,
        outlier_factor=options.outlier_factor,
        half_life_days=options.half_life_days,
        holidays=options.holidays,
        clip_negative=options.clip_negative,
        min_samples=options.min_samples,
    )


def _standby_from_home(home: dict[int, BaseloadProfile]) -> BaseloadProfile:
    """A rough standby profile from the tenth percentile of the seven weekdays.

    Used when the household is away but no away days have been labelled yet,
    so an away regime does not fall all the way back to the static
    configuration just because it has never been observed.
    """
    profile = BaseloadProfile()
    for hour in range(24):
        values = sorted(p.values[hour] for p in home.values())
        if values:
            profile.values[hour] = round(quantile(values, 0.10), 3)
            profile.samples[hour] = len(values)
    return profile


def _localize(moment: datetime.datetime, tz: ZoneInfo) -> datetime.datetime:
    """A tz-aware version of ``moment``: converted if aware, localized if not."""
    if moment.tzinfo is not None:
        return moment.astimezone(tz)
    return moment.replace(tzinfo=tz)


class BaseloadService:
    """Fits and serves the household's baseload profile."""

    def __init__(
        self,
        config,
        db_da,
        db_ha,
        data_dir: Path,
        tz: str,
        now: Optional[Callable[[], datetime.datetime]] = None,
    ) -> None:
        self.config = config
        self.db_da = db_da
        self.db_ha = db_ha
        self.data_dir = Path(data_dir)
        self.tz = tz
        self._zone = ZoneInfo(tz)
        self._now = now or (lambda: datetime.datetime.now(tz=self._zone))

    def profile_set(self) -> Optional[ProfileSet]:
        """The saved profile set, migrating the legacy files once when absent."""
        loaded = load_profile_set(self.data_dir)
        if loaded is not None:
            return loaded
        legacy_dir = self.data_dir.parent.parent / "baseload"
        migrated = migrate_legacy_files(legacy_dir, self._now())
        if migrated is not None:
            save_profile_set(migrated, self.data_dir)
        return migrated

    def fit(self) -> ProfileSet:
        """Recompute the seven weekday profiles from history and save them."""
        now = self._now()
        options = options_from_config(self.config)
        period_days = self.config.baseload_calc_periode
        tot = now.replace(hour=0, minute=0, second=0, microsecond=0)
        vanaf = tot - datetime.timedelta(days=period_days)

        reader = HistoryReader(self.db_ha, self.tz)
        groups = component_groups(self.config.report)
        caps = component_caps(self.config)
        frame = reader.read_components(groups, vanaf, tot, caps)
        base = baseload_from_components(frame).dropna()

        rows = list(zip(base.index, base.values, strict=True))
        grouped = iter_samples(rows, now, options.holidays)
        pooled: dict[int, list] = {}
        for cells in grouped.values():
            for hour, samples in cells.items():
                pooled.setdefault(hour, []).extend(samples)

        home: dict[int, BaseloadProfile] = {}
        for weekday in range(7):
            profile = build_profile(grouped.get(weekday, {}), pooled, options)
            thin = [hour for hour in range(24) if profile.pooled[hour]]
            logging.info(
                f"baseload weekdag {weekday}: totaal {profile.total:.2f} kWh, "
                f"mediaan aantal metingen per uur {sorted(profile.samples)[12]}"
                + (
                    f", {len(thin)} uur uit de gepoolde schatting"
                    if thin
                    else ""
                )
            )
            home[weekday] = profile

        profile_set = ProfileSet(
            created=now,
            period_days=period_days,
            aggregate=options.aggregate,
            home=home,
        )
        save_profile_set(profile_set, self.data_dir)
        return profile_set

    def forecast(
        self,
        start: datetime.datetime,
        hours: int,
        regime: Optional[Regime] = None,
    ) -> pd.Series:
        """Hourly kWh for ``hours`` hours from ``start``, home or away."""
        regime = regime or Regime()
        options = options_from_config(self.config)
        start = _localize(start, self._zone)
        floored = start.replace(minute=0, second=0, microsecond=0)
        index = pd.date_range(floored, periods=hours, freq="h", tz=self.tz)

        profile_set = self.profile_set()
        if profile_set is not None:
            age = profile_age_days(profile_set, self._now())
            if age > MAX_PROFILE_AGE_DAYS:
                logging.warning(
                    f"Baseload: profiel is {age:.1f} dagen oud, de schatting "
                    f"kan verouderd zijn"
                )
            away_profile = None
            if regime.away:
                away_profile = profile_set.away or _standby_from_home(
                    profile_set.home
                )
            values = []
            for timestamp in index:
                if away_profile is not None:
                    day_profile = away_profile
                else:
                    weekday = effective_weekday(timestamp.date(), options.holidays)
                    day_profile = profile_set.home.get(weekday, BaseloadProfile())
                values.append(day_profile.values[timestamp.hour])
            return pd.Series(values, index=index, name="baseload")

        static = self.config.baseload
        if static:
            logging.warning(
                "Baseload: geen profiel beschikbaar, statische baseload uit "
                "de configuratie gebruikt"
            )
            values = [static[timestamp.hour] for timestamp in index]
            return pd.Series(values, index=index, name="baseload")

        raise BaseloadUnavailable(
            "Baseload: geen profiel en geen statische baseload geconfigureerd"
        )

    def forecast_for_optimizer(
        self, start_interval: datetime.datetime, intervals: int, interval: str
    ) -> list[float]:
        """The baseload the optimizer needs for its horizon, in its own step."""
        if interval == "1hour":
            series = self.forecast(start_interval, intervals)
            return list(series.values[:intervals])

        if interval == "15min":
            start = _localize(start_interval, self._zone)
            floored = start.replace(minute=0, second=0, microsecond=0)
            offset_quarters = int((start - floored).total_seconds() // 900)
            hours_needed = max(2, math.ceil((intervals + offset_quarters) / 4) + 1)
            series = self.forecast(floored, hours_needed)
            frame = pd.DataFrame(
                {
                    "tijd": [timestamp.tz_localize(None) for timestamp in series.index],
                    "baseload": series.values,
                }
            )
            quarters = interpolate(frame, "baseload", quantity=True)
            values = list(quarters["baseload"].values)
            return values[offset_quarters : offset_quarters + intervals]

        raise ValueError(f"onbekend interval: {interval!r}")
