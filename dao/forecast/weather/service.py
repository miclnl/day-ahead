"""Fetch the forecast the household actually gets: Meteoserver when a key
is configured, Open-Meteo otherwise, and Open-Meteo again to fill in
whatever the primary source left short -- an outage, or a horizon that
simply ends too soon. One call also refreshes the measured weather that the
accuracy report and the PV calibration compare a forecast against.
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from dao.forecast.baseload.store import write_json
from dao.forecast.weather.meteoserver import fetch_meteoserver
from dao.forecast.weather.observations import update_observations
from dao.forecast.weather.openmeteo import fetch_openmeteo
from dao.forecast.weather.schema import WEATHER_COLUMNS, validate_weather_frame

#: Codes archived per source. The first four are what the accuracy report
#: scores; ``winds`` is archived because the PV model needs it, not because
#: anything reports on it -- Faiman's cell temperature is
#: ``temp + poa/(u0 + u1*wind)``, and an installation without local
#: observations has nothing but this archive to calibrate and backtest
#: against. ``neersl`` stays out: nothing downstream reads it.
_ARCHIVED_WEATHER_CODES = ("gr", "dni", "dhi", "temp", "winds")

STATUS_FILE = "status.json"


@dataclass
class WeatherStatus:
    """What the last fetch produced, for the log and the dashboard."""

    fetched_at: datetime.datetime
    hours_total: int
    hours_by_source: dict
    primary: str
    primary_error: Optional[str]
    horizon_end: Optional[datetime.datetime]
    frame: pd.DataFrame  # not serialised: too large and not useful as JSON

    def to_dict(self) -> dict:
        return {
            "fetched_at": self.fetched_at.isoformat(),
            "hours_total": self.hours_total,
            "hours_by_source": dict(self.hours_by_source),
            "primary": self.primary,
            "primary_error": self.primary_error,
            "horizon_end": self.horizon_end.isoformat() if self.horizon_end else None,
        }


def _forecast_days_for(horizon_hours: int) -> int:
    """Open-Meteo's ``forecast_days`` covering ``horizon_hours`` from now,
    with one day of slack since "now" is rarely exactly midnight."""
    return max(1, math.ceil(horizon_hours / 24) + 1)


class WeatherService:
    """Fetches, archives and observes the weather for one installation."""

    def __init__(
        self,
        config,
        db_da,
        latitude: float,
        longitude: float,
        secrets: dict,
        data_dir: Path,
        country: str,
        now=None,
        session=requests,
    ) -> None:
        self.config = config
        self.db_da = db_da
        self.latitude = latitude
        self.longitude = longitude
        self.secrets = secrets or {}
        self.data_dir = Path(data_dir)
        self.country = country
        self._now = now or (lambda: datetime.datetime.now(datetime.UTC))
        self.session = session

    def _meteoserver_key(self) -> Optional[str]:
        key = getattr(self.config, "meteoserver_key", None)
        if not key:
            return None
        return key.resolve(self.secrets)

    def fetch(self, horizon_hours: int = 72) -> tuple[pd.DataFrame, WeatherStatus]:
        now = self._now()
        weather_config = self.config.weather
        key = self._meteoserver_key()
        primary = "meteoserver" if key else "openmeteo"
        forecast_days = _forecast_days_for(horizon_hours)

        if primary == "meteoserver":
            primary_frame = fetch_meteoserver(
                key,
                self.config.meteoserver_model,
                self.config.meteoserver_attempts,
                self.latitude,
                self.longitude,
                session=self.session,
            )
        else:
            primary_frame = fetch_openmeteo(
                self.latitude,
                self.longitude,
                model=weather_config.openmeteo_model,
                forecast_days=forecast_days,
                session=self.session,
            )

        primary_error = None
        if len(primary_frame) == 0:
            primary_error = f"{primary} niet bereikbaar"

        floored = now.replace(minute=0, second=0, microsecond=0)
        needed = {
            int((floored + datetime.timedelta(hours=h)).timestamp())
            for h in range(horizon_hours)
        }
        have = set(int(t) for t in primary_frame["time"]) if len(primary_frame) else set()
        missing = needed - have

        frame = primary_frame
        if missing and weather_config.fallback == "openmeteo" and primary == "meteoserver":
            fallback_frame = fetch_openmeteo(
                self.latitude,
                self.longitude,
                model=weather_config.openmeteo_model,
                forecast_days=forecast_days,
                session=self.session,
            )
            fallback_rows = fallback_frame[fallback_frame["time"].isin(missing)]
            if len(fallback_rows):
                if not have:
                    reason = "Meteoserver niet bereikbaar"
                else:
                    end_dt = datetime.datetime.fromtimestamp(
                        max(have), tz=datetime.UTC
                    )
                    reason = f"horizon Meteoserver eindigt {end_dt.isoformat()}"
                logging.info(
                    f"Weer: {len(fallback_rows)} uren aangevuld uit Open-Meteo "
                    f"({reason})"
                )
                frame = pd.concat([primary_frame, fallback_rows], ignore_index=True)

        if len(frame):
            frame = validate_weather_frame(frame)
        else:
            logging.error("Geen weerdata van enige bron ontvangen")

        hours_by_source = (
            frame["source"].value_counts().to_dict() if len(frame) else {}
        )
        horizon_end = (
            datetime.datetime.fromtimestamp(int(frame["time"].max()), tz=datetime.UTC)
            if len(frame)
            else None
        )

        status = WeatherStatus(
            fetched_at=now,
            hours_total=len(frame),
            hours_by_source=hours_by_source,
            primary=primary,
            primary_error=primary_error,
            horizon_end=horizon_end,
            frame=frame,
        )
        return frame, status

    def update(self, horizon_hours: int = 72) -> WeatherStatus:
        """Fetch, archive to prognoses/forecasts, refresh observations, and
        write ``status.json``."""
        frame, status = self.fetch(horizon_hours)

        if len(frame) and self.db_da is not None:
            records = []
            for column in WEATHER_COLUMNS:
                if column not in frame.columns:
                    continue
                values = frame[["time", column]].dropna(subset=[column])
                for _, row in values.iterrows():
                    records.append((int(row["time"]), column, float(row[column])))
            if records:
                save_frame = pd.DataFrame(records, columns=["time", "code", "value"])
                self.db_da.savedata(save_frame, tablename="prognoses")

            issued_ts = int(self._now().timestamp())
            for source_name, group in frame.groupby("source"):
                forecast_rows = []
                for _, row in group.iterrows():
                    for code in _ARCHIVED_WEATHER_CODES:
                        value = row.get(code)
                        if value == value:  # skip NaN
                            forecast_rows.append((int(row["time"]), code, float(value)))
                if forecast_rows:
                    self.db_da.save_forecasts(
                        forecast_rows, issued_ts=issued_ts, source=source_name
                    )

        try:
            update_observations(
                self.db_da,
                self.latitude,
                self.longitude,
                self.country,
                self.config.weather.observations,
                now=self._now(),
            )
        except Exception as ex:  # noqa: BLE001 - the forecast is already stored
            # Refreshing the measured weather is a second network call to a
            # different provider, and it runs after the forecast has been
            # archived. An outage there must not throw away the status file
            # and the graph of a forecast that was fetched perfectly well;
            # the observations are only used for accuracy and calibration,
            # both of which tolerate a day's gap.
            logging.warning(
                f"Weer: bijwerken van de waarnemingen mislukt ({ex}); de "
                f"prognose is wel opgeslagen"
            )

        write_json(self.data_dir / STATUS_FILE, status.to_dict())
        return status
