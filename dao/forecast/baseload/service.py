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
import statistics
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from dao.forecast.baseload.absence import (  # noqa: F401 (Regime re-exported)
    Regime,
    RegimeSignals,
    active_energy_per_day,
    calibrate_threshold,
    determine_regime,
    label_days,
    standby_kwh,
)
from dao.forecast.baseload.profile import (
    BaseloadOptions,
    BaseloadProfile,
    Sample,
    build_profile,
    effective_weekday,
    iter_samples,
    quantile,
    standby_profile,
)
from dao.forecast.baseload.store import (
    STATUS_FILE,
    ProfileSet,
    load_profile_set,
    migrate_legacy_files,
    profile_age_days,
    read_json,
    save_profile_set,
    write_json,
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


def _presence_series_from_frame(frame, tz: ZoneInfo) -> Optional[pd.Series]:
    """``get_column_data("values", "presence", ...)``'s frame as a tz-aware series."""
    if frame is None or len(frame) == 0:
        return None
    index = pd.to_datetime(frame["utc"], unit="s", utc=True).dt.tz_convert(tz)
    return pd.Series(frame["value"].astype(float).values, index=index).sort_index()


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
        ha=None,
    ) -> None:
        self.config = config
        self.db_da = db_da
        self.db_ha = db_ha
        self.data_dir = Path(data_dir)
        self.tz = tz
        self.ha = ha
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

    def _read_status(self) -> dict:
        return read_json(self.data_dir / STATUS_FILE) or {}

    def _write_status(self, **updates) -> None:
        status = self._read_status()
        status.update(updates)
        write_json(self.data_dir / STATUS_FILE, status)

    def _calibrate_threshold(
        self, base: pd.Series, current_threshold: float
    ) -> Optional[float]:
        """Recalibrates the away threshold from presence history, once there
        is enough of it; ``None`` when there is not (yet)."""
        if self.db_da is None:
            return None
        try:
            frame = self.db_da.get_column_data("values", "presence")
        except Exception as ex:  # noqa: BLE001 - calibration is best-effort
            logging.debug(f"Afwezigheid: kon presence-historie niet lezen: {ex}")
            return None
        presence = _presence_series_from_frame(frame, self._zone)
        if presence is None:
            return None
        presence_daily = presence.groupby(presence.index.date).mean()

        standby = standby_kwh(base)
        active = active_energy_per_day(base, standby)
        fractions: dict = {}
        history: list[float] = []
        for day, value in active.items():
            if len(history) >= 7:
                reference = statistics.median(history[-28:])
                if reference:
                    fractions[day] = value / reference
            history.append(value)
        if not fractions:
            return None

        calibrated = calibrate_threshold(pd.Series(fractions), presence_daily)
        if calibrated is not None and calibrated != current_threshold:
            overlap = len(
                pd.Series(fractions).index.intersection(presence_daily.index)
            )
            logging.info(
                f"Afwezigheid: drempel gekalibreerd op {calibrated:.2f} uit "
                f"{overlap} dagen"
            )
        return calibrated

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

        away_profile = None
        away_source = None
        standby_value = None

        absence_config = self.config.baseload_options.absence
        if absence_config.detect:
            status = self._read_status()
            threshold = status.get("threshold", absence_config.threshold)

            labels = label_days(base, threshold)
            away_dates = {day for day, is_away in labels.items() if is_away}

            if self.db_da is not None:
                for day, is_away in labels.items():
                    self.db_da.save_daily_value("away", day, 1.0 if is_away else 0.0)

            standby_value = standby_kwh(base)

            if len(away_dates) >= 3:
                away_cells: dict[int, list] = {}
                for moment, value in rows:
                    if value != value or moment.date() not in away_dates:
                        continue
                    away_cells.setdefault(moment.hour, []).append(
                        Sample(
                            age_days=(now - moment).total_seconds() / 86400.0,
                            value=float(value),
                        )
                    )
                away_options = replace(options, min_samples=1)
                away_profile = build_profile(away_cells, None, away_options)
                away_source = "labels"
            else:
                all_cells: dict[int, list] = {}
                for cells in grouped.values():
                    for hour, samples in cells.items():
                        all_cells.setdefault(hour, []).extend(samples)
                away_profile = standby_profile(all_cells)
                away_source = "standby"

            calibrated = self._calibrate_threshold(base, threshold)
            if calibrated is not None:
                threshold = calibrated

            self._write_status(threshold=threshold, away_days_labelled=len(away_dates))

        profile_set = ProfileSet(
            created=now,
            period_days=period_days,
            aggregate=options.aggregate,
            home=home,
            away=away_profile,
            away_source=away_source,
            standby_kwh=standby_value,
        )
        save_profile_set(profile_set, self.data_dir)
        return profile_set

    def forecast(
        self,
        start: datetime.datetime,
        hours: int,
        regime: Optional[Regime] = None,
    ) -> pd.Series:
        """Hourly kWh for ``hours`` hours from ``start``, home or away.

        Without an explicit ``regime`` the regime is decided per target day
        via :meth:`current_regime`, so a horizon that crosses a departure or
        an arrival switches profile on the right day instead of using one
        regime for the whole call.
        """
        options = options_from_config(self.config)
        start = _localize(start, self._zone)
        floored = start.replace(minute=0, second=0, microsecond=0)
        index = pd.date_range(floored, periods=hours, freq="h", tz=self.tz)

        regime_cache: dict = {}

        def regime_for(day) -> Regime:
            if regime is not None:
                return regime
            if day not in regime_cache:
                regime_cache[day] = self.current_regime(day)
            return regime_cache[day]

        profile_set = self.profile_set()
        if profile_set is not None:
            age = profile_age_days(profile_set, self._now())
            if age > MAX_PROFILE_AGE_DAYS:
                logging.warning(
                    f"Baseload: profiel is {age:.1f} dagen oud, de schatting "
                    f"kan verouderd zijn"
                )
            values = []
            for timestamp in index:
                day = timestamp.date()
                day_regime = regime_for(day)
                if day_regime.away:
                    day_profile = profile_set.away or _standby_from_home(
                        profile_set.home
                    )
                else:
                    weekday = effective_weekday(day, options.holidays)
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

    def current_regime(self, day: datetime.date) -> Regime:
        """Home or away for ``day``, gathering signals from HA and history.

        Writes the decision to ``status.json`` alongside the threshold and
        away-day count :meth:`fit` last wrote there.
        """
        now = self._now()
        absence_config = self.config.baseload_options.absence

        entity_state = None
        if absence_config.entity_away and self.ha is not None:
            try:
                entity_state = self.ha.get_state(absence_config.entity_away).state
            except Exception as ex:  # noqa: BLE001 - a signal, not the calculation
                logging.warning(
                    f"Kon status van {absence_config.entity_away} niet ophalen: {ex}"
                )

        calendar_events = []
        if absence_config.entity_calendar and self.ha is not None:
            try:
                calendar_events = self.ha.get_calendar_events(
                    absence_config.entity_calendar,
                    now,
                    now + datetime.timedelta(days=2),
                )
            except Exception as ex:  # noqa: BLE001 - a signal, not the calculation
                logging.warning(
                    f"Kon kalender {absence_config.entity_calendar} niet ophalen: {ex}"
                )

        presence = None
        if absence_config.entities_presence and self.db_da is not None:
            try:
                frame = self.db_da.get_column_data(
                    "values",
                    "presence",
                    start=now - datetime.timedelta(hours=48),
                    end=now,
                )
                presence = _presence_series_from_frame(frame, self._zone)
            except Exception as ex:  # noqa: BLE001 - a signal, not the calculation
                logging.warning(f"Kon presence-historie niet lezen: {ex}")

        status = self._read_status()
        threshold = status.get("threshold", absence_config.threshold)
        standby = 0.0
        consumption_today = None
        home_profile_today = None

        if day == now.date():
            profile_set = self.profile_set()
            if profile_set is not None:
                options = options_from_config(self.config)
                weekday = effective_weekday(day, options.holidays)
                home_profile = profile_set.home.get(weekday)
                if home_profile is not None:
                    home_profile_today = home_profile.values
                standby = profile_set.standby_kwh or 0.0
            if self.db_ha is not None:
                try:
                    reader = HistoryReader(self.db_ha, self.tz)
                    groups = component_groups(self.config.report)
                    caps = component_caps(self.config)
                    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    frame = reader.read_components(groups, midnight, now, caps)
                    consumption_today = baseload_from_components(frame).tolist()
                except Exception as ex:  # noqa: BLE001 - a signal, not the calculation
                    logging.warning(f"Kon verbruik van vandaag niet lezen: {ex}")

        signals = RegimeSignals(
            now=now,
            entity_state=entity_state,
            away_state=absence_config.away_state,
            calendar_events=calendar_events,
            keywords=absence_config.calendar_keywords,
            presence=presence,
            away_after_hours=absence_config.away_after_hours,
            assume_next_day_after_hours=absence_config.assume_next_day_after_hours,
            consumption_today=consumption_today,
            home_profile_today=home_profile_today,
            standby=standby,
            threshold=threshold,
        )
        regime = determine_regime(day, signals)

        self._write_status(
            regime="away" if regime.away else "home",
            reason=regime.reason,
            switch_hour=regime.switch_hour,
            decided_at=now.isoformat(),
        )
        return regime

    def record_presence(self) -> None:
        """Fraction of ``entities presence`` currently home, written hourly.

        A no-op without configured presence entities: nothing to measure,
        nothing written.
        """
        absence_config = self.config.baseload_options.absence
        entities = absence_config.entities_presence
        if not entities or self.ha is None or self.db_da is None:
            return
        home_count = 0
        for entity_id in entities:
            try:
                state = self.ha.get_state(entity_id).state
            except Exception as ex:  # noqa: BLE001 - one entity should not abort the rest
                logging.warning(f"Kon status van {entity_id} niet ophalen: {ex}")
                continue
            if state == "home":
                home_count += 1
        fraction = home_count / len(entities)
        moment = self._now().replace(minute=0, second=0, microsecond=0)
        self.db_da.save_hourly_value("presence", moment, fraction)
