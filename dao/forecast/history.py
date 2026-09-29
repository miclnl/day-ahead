"""Hourly energy per component group, read from the Home Assistant recorder.

The recorder keeps long-term statistics per sensor in ``statistics`` (one row
per hour) with the sensor's kind in ``statistics_meta``. An energy sensor has
``has_sum = 1`` and a reset-corrected running total in the ``sum`` column; a
power sensor has ``mean_type != 0`` and an hourly mean in ``mean``. The old
code read the ``state`` column, which is NULL for power sensors, so a power
sensor in the configuration silently contributed nothing. This module reads
the right column for the kind, converts units once, turns glitches into NaN
and never invents a zero for an hour that has no measurement.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlalchemy import Table, select

from dao.lib.db_manager import DBmanagerObj

#: Column names of the component frame, in the order of the baseload formula.
COMPONENT_COLUMNS = (
    "grid_in",
    "grid_out",
    "pv_ac",
    "ev",
    "wp",
    "boiler",
    "machines",
    "bat_in",
    "bat_out",
)

#: Groups whose absence in an hour makes the baseload of that hour unknown.
REQUIRED_COMPONENTS = ("grid_in", "grid_out", "pv_ac", "bat_in", "bat_out")

#: Groups that may be missing for an hour; they are then counted as zero.
OPTIONAL_COMPONENTS = ("ev", "wp", "boiler", "machines")

#: Multiply a statistic value by this to get kWh. Energy sensors report a
#: running total, power sensors an hourly mean, so W x 1 h = Wh.
_UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "Wh": ("energy", 0.001),
    "kWh": ("energy", 1.0),
    "MWh": ("energy", 1000.0),
    "W": ("power", 0.001),
    "kW": ("power", 1.0),
}

_REPORT_ATTRIBUTES = {
    "grid_in": "entities_grid_consumption",
    "grid_out": "entities_grid_production",
    "pv_ac": "entities_solar_production_ac",
    "ev": "entities_ev_consumption",
    "wp": "entities_wp_consumption",
    "boiler": "entities_boiler_consumption",
    "machines": "entities_machine_consumption",
    "bat_in": "entities_battery_consumption",
    "bat_out": "entities_battery_production",
}


class UnsupportedSensorError(ValueError):
    """A configured meter is neither an energy nor a power sensor."""


@dataclass(frozen=True)
class SensorMeta:
    statistic_id: str
    metadata_id: int
    unit: str
    kind: Literal["energy", "power"]
    factor_to_kwh: float


def _hour_index(start: datetime.datetime, end: datetime.datetime, tz: str) -> pd.DatetimeIndex:
    """Every hour start in ``[start, end)`` as a tz-aware index in ``tz``."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(tz)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(tz)
    # Build in UTC so the autumn DST hour is two distinct hours, then present
    # in the configured zone.
    utc = pd.date_range(
        start_ts.tz_convert("UTC").floor("h"),
        end_ts.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    return utc.tz_convert(tz)


class HistoryReader:
    """Read hourly energy per sensor group from the recorder database."""

    def __init__(self, db_ha: DBmanagerObj, tz: str) -> None:
        self.db_ha = db_ha
        self.tz = tz
        self._meta_table: Table | None = None
        self._stats_table: Table | None = None

    # ------------------------------------------------------------------ meta
    def _tables(self) -> tuple[Table, Table]:
        if self._meta_table is None or self._stats_table is None:
            self._meta_table = Table(
                "statistics_meta", self.db_ha.metadata, autoload_with=self.db_ha.engine
            )
            self._stats_table = Table(
                "statistics", self.db_ha.metadata, autoload_with=self.db_ha.engine
            )
        return self._meta_table, self._stats_table

    def sensor_meta(self, sensors: Sequence[str]) -> dict[str, SensorMeta]:
        """Kind, unit and conversion factor per sensor.

        Raises :class:`UnsupportedSensorError` for a sensor that is not in the
        statistics tables or whose unit is not an energy or power unit, so a
        misconfigured meter is a clear error rather than a silent zero.
        """
        if not sensors:
            return {}
        meta, _ = self._tables()
        query = select(
            meta.c.id,
            meta.c.statistic_id,
            meta.c.unit_of_measurement,
            meta.c.has_sum,
            meta.c.mean_type,
            meta.c.has_mean,
        ).where(meta.c.statistic_id.in_(list(sensors)))
        with self.db_ha.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        found = {row["statistic_id"]: row for row in rows}
        result: dict[str, SensorMeta] = {}
        for sensor in sensors:
            row = found.get(sensor)
            if row is None:
                raise UnsupportedSensorError(
                    f"{sensor}: geen statistieken gevonden in de Home Assistant database"
                )
            unit = row["unit_of_measurement"] or ""
            expected_kind, factor = _UNIT_FACTORS.get(unit, (None, 0.0))
            has_sum = bool(row["has_sum"])
            has_mean = bool(row["mean_type"]) or bool(row["has_mean"])
            if has_sum and expected_kind == "energy":
                kind = "energy"
            elif has_mean and expected_kind == "power":
                kind = "power"
            else:
                raise UnsupportedSensorError(
                    f"{sensor}: eenheid {unit!r} is geen energie of vermogen"
                )
            result[sensor] = SensorMeta(
                statistic_id=sensor,
                metadata_id=int(row["id"]),
                unit=unit,
                kind=kind,
                factor_to_kwh=factor,
            )
        return result

    # ---------------------------------------------------------------- energy
    def _read_sensor(
        self,
        meta: SensorMeta,
        index: pd.DatetimeIndex,
        cap_kwh: float | None,
    ) -> pd.Series:
        _, stats = self._tables()
        start_ts = int(index[0].timestamp())
        end_ts = int(index[-1].timestamp()) + 3600
        query = (
            select(stats.c.start_ts, stats.c.sum, stats.c.mean)
            .where(stats.c.metadata_id == meta.metadata_id)
            # One hour earlier is not needed: the delta of hour t is
            # sum[t+1h] - sum[t], so the row after the window is what we need.
            .where(stats.c.start_ts >= start_ts)
            .where(stats.c.start_ts <= end_ts)
            .order_by(stats.c.start_ts)
        )
        with self.db_ha.engine.connect() as connection:
            frame = pd.read_sql(query, connection)
        if frame.empty:
            return pd.Series(float("nan"), index=index, dtype="float64")
        moments = pd.to_datetime(frame["start_ts"].astype("int64"), unit="s", utc=True)
        if meta.kind == "energy":
            totals = pd.Series(frame["sum"].astype("float64").values, index=moments)
            # Consecutive rows only: a gap in the recorder must not become
            # one big delta assigned to the hour before the gap.
            following = totals.shift(-1)
            gap = pd.Series(moments.values, index=moments).shift(-1) - pd.Series(
                moments.values, index=moments
            )
            values = (following - totals).where(gap == pd.Timedelta(hours=1))
        else:
            values = pd.Series(frame["mean"].astype("float64").values, index=moments)
            logging.info(
                f"{meta.statistic_id} is een vermogenssensor; het uurgemiddelde "
                f"wordt als energie gebruikt"
            )
        values = values * meta.factor_to_kwh
        values.index = values.index.tz_convert(self.tz)
        values = values.reindex(index)
        out_of_range = values < 0
        if cap_kwh is not None:
            out_of_range |= values > cap_kwh
        count = int(out_of_range.sum())
        if count:
            upper = "-" if cap_kwh is None else f"{cap_kwh:g}"
            logging.warning(
                f"{meta.statistic_id}: {count} uurwaarden buiten bereik "
                f"(0..{upper} kWh) genegeerd"
            )
            values = values.mask(out_of_range)
        return values

    def read_energy(
        self,
        sensors: Sequence[str],
        start: datetime.datetime,
        end: datetime.datetime,
        *,
        cap_kwh: float | None = None,
    ) -> pd.Series:
        """kWh per hour for the sum of ``sensors`` over ``[start, end)``.

        The result is indexed by tz-aware hour start and holds ``NaN`` for
        hours in which any of the sensors has no measurement.
        """
        index = _hour_index(start, end, self.tz)
        metas = self.sensor_meta(sensors)
        total: pd.Series | None = None
        for sensor in sensors:
            series = self._read_sensor(metas[sensor], index, cap_kwh)
            total = series if total is None else total + series
        if total is None:
            total = pd.Series(float("nan"), index=index, dtype="float64")
        total.name = "kwh"
        return total

    def read_components(
        self,
        groups: dict[str, list[str]],
        start: datetime.datetime,
        end: datetime.datetime,
        caps: dict[str, float | None] | None = None,
    ) -> pd.DataFrame:
        """One column of hourly kWh per group; an empty group is a column of zeros."""
        index = _hour_index(start, end, self.tz)
        caps = caps or {}
        columns: dict[str, pd.Series] = {}
        for name, sensors in groups.items():
            if sensors:
                columns[name] = self.read_energy(
                    sensors, start, end, cap_kwh=caps.get(name)
                ).reindex(index)
            else:
                columns[name] = pd.Series(0.0, index=index, dtype="float64")
        return pd.DataFrame(columns, index=index)


# ---------------------------------------------------------------- helpers
def component_groups(report) -> dict[str, list[str]]:
    """The configured sensor list per component group, from ``config.report``."""
    return {
        column: list(getattr(report, attribute, None) or [])
        for column, attribute in _REPORT_ATTRIBUTES.items()
    }


def component_caps(config) -> dict[str, float | None]:
    """Physical upper bound in kWh per hour per group, or None for no bound."""
    pv_capacity = sum(
        (installation.total_capacity or 0.0) for installation in (config.solar or [])
    )
    battery_kw = 0.0
    for battery in config.battery or []:
        battery_kw += (
            max(
                float(getattr(battery, "dc_to_bat_max_power", 0.0) or 0.0),
                float(getattr(battery, "bat_to_dc_max_power", 0.0) or 0.0),
            )
            / 1000.0
        )
    grid_kw = getattr(config.grid, "max_power", None)
    return {
        "grid_in": grid_kw,
        "grid_out": grid_kw,
        "pv_ac": 1.2 * pv_capacity if pv_capacity > 0 else None,
        "ev": None,
        "wp": None,
        "boiler": None,
        "machines": None,
        "bat_in": battery_kw or None,
        "bat_out": battery_kw or None,
    }


def baseload_from_components(frame: pd.DataFrame) -> pd.Series:
    """grid_in - grid_out + pv_ac - ev - wp - boiler - machines - bat_in + bat_out.

    A missing required component makes the hour NaN. A missing optional
    device meter counts as zero, which leaves that device's consumption
    inside the baseload for that hour; the count is logged so it is visible.
    """
    filled = frame.copy()
    for column in OPTIONAL_COMPONENTS:
        if column not in filled:
            filled[column] = 0.0
            continue
        missing = int(filled[column].isna().sum())
        if missing:
            logging.info(
                f"Baseload: {missing} uren zonder meting van {column}, als 0 geteld"
            )
            filled[column] = filled[column].fillna(0.0)
    for column in REQUIRED_COMPONENTS:
        if column not in filled:
            filled[column] = 0.0
    result = (
        filled["grid_in"]
        - filled["grid_out"]
        + filled["pv_ac"]
        - filled["ev"]
        - filled["wp"]
        - filled["boiler"]
        - filled["machines"]
        - filled["bat_in"]
        + filled["bat_out"]
    )
    result.name = "baseload"
    return result
