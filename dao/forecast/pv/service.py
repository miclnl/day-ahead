"""Forecast and calibrate PV production for every configured installation.

The forecast path always runs the physical model, with whichever
parameters are current: calibrated when a fit exists and calibration is
switched on, the plain configuration otherwise. Calibration itself is a
periodic background step (``run_training``), not on the forecast path, so
a slow or failing fit never holds up the optimizer.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from dao.forecast.baseload.store import read_json, write_json
from dao.forecast.evaluate import Selection, backtest
from dao.forecast.history import HistoryReader
from dao.forecast.pv.calibrate import CalibrationResult, calibrate
from dao.forecast.pv.physical import params_from_config, simulate, weather_for_pv
from dao.forecast.pv.select import MLCandidate, PhysicalCandidate, select_pv_model
from dao.forecast.pv.store import (
    calibration_path,
    load_calibration,
    save_calibration,
    selection_path,
)


def _parse_iso(value, fallback: datetime.datetime) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return fallback

#: Beyond this many days a stored calibration is still used, but flagged:
#: the installation may have changed (shading, cleaning, a fault) since.
MAX_CALIBRATION_AGE_DAYS = 90.0

#: Weather codes the physical model and its calibration need.
_WEATHER_CODES = ("gr", "dni", "dhi", "temp", "winds")


class PVService:
    """Forecasts and calibrates PV production for one installation set."""

    def __init__(
        self,
        config,
        db_da,
        db_ha,
        latitude: float,
        longitude: float,
        data_dir: Path,
        tz: str,
        interval: str,
        now=None,
    ) -> None:
        self.config = config
        self.db_da = db_da
        self.db_ha = db_ha
        self.latitude = latitude
        self.longitude = longitude
        self.data_dir = Path(data_dir)
        self.tz = tz
        self.interval = interval
        self._now = now or (
            lambda: datetime.datetime.now(tz=datetime.UTC)
        )

    def installations(self) -> list:
        """Every AC installation (``config.solar``) and DC installation on a
        battery (``config.battery[].solar``), in that order."""
        result = list(self.config.solar or [])
        for battery in self.config.battery or []:
            result.extend(battery.solar or [])
        return result

    def params_for(self, installation) -> tuple:
        """The parameters to forecast with, and where they came from."""
        config_params = params_from_config(installation)
        if installation.calibration == "off":
            return config_params, "config"

        path = calibration_path(self.data_dir, installation.name)
        result = load_calibration(path)
        if result is None:
            return config_params, "config"

        age_days = (self._now() - result.created).total_seconds() / 86400.0
        if age_days > MAX_CALIBRATION_AGE_DAYS:
            logging.warning(
                f"PV: kalibratie van {installation.name} is {age_days:.0f} dagen "
                f"oud, mogelijk niet meer representatief"
            )
        return result.params, "calibrated"

    def _interval_s(self, interval: str) -> int:
        return 900 if interval == "15min" else 3600

    def selection_for(self, installation) -> Selection:
        """The stored model choice for ``installation``, or the configured one."""
        payload = read_json(selection_path(self.data_dir, installation.name))
        if payload:
            return Selection(
                model=payload.get("model", "physical"),
                scores=payload.get("scores", {}),
                decided_at=_parse_iso(payload.get("decided_at"), self._now()),
                reason=payload.get("reason", ""),
            )
        return select_pv_model(installation.effective_model, None)

    def _forecast_ml(self, installation, start, end, interval) -> Optional[pd.DataFrame]:
        """The ML model's forecast, or ``None`` to fall back to physical."""
        from dao.prog.solar_predictor import SolarPredictor

        try:
            predictor = SolarPredictor()
            frame = predictor.predict_solar_device(installation, start, end)
        except FileNotFoundError:
            logging.warning(
                f"PV: ML-model gekozen voor {installation.name} maar niet "
                f"aanwezig, fysisch model gebruikt"
            )
            return None
        except Exception as ex:  # noqa: BLE001 - the physical model is the fallback
            logging.warning(
                f"PV: ML-voorspelling voor {installation.name} mislukt ({ex}), "
                f"fysisch model gebruikt"
            )
            return None

        if frame is None or len(frame) == 0:
            logging.warning(
                f"PV: ML-model gaf niets terug voor {installation.name}, "
                f"fysisch model gebruikt"
            )
            return None

        tijd = frame["date_time"] if "date_time" in frame.columns else frame["tijd"]
        return pd.DataFrame(
            {"tijd": tijd.to_numpy(), "prediction": frame["prediction"].to_numpy()}
        ).reset_index(drop=True)

    def forecast(
        self,
        installation,
        start: datetime.datetime,
        end: datetime.datetime,
        interval: str,
        model: Optional[str] = None,
    ) -> pd.DataFrame:
        """Forecast production for ``installation`` between ``start`` and ``end``.

        Columns ``tijd`` (tz-aware) and ``prediction`` (kWh per interval),
        one row per step of ``interval``. ``model`` overrides the stored
        selection, which is how the solar report asks for the ML column
        specifically. An empty weather forecast becomes zeros rather than
        an exception: no data beats no plan.
        """
        chosen = model or self.selection_for(installation).model
        if chosen == "ml":
            predicted = self._forecast_ml(installation, start, end, interval)
            if predicted is not None:
                return predicted

        interval_s = self._interval_s(interval)
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        prog = self.db_da.get_prognose_fields(
            list(_WEATHER_CODES), start_ts, end_ts, interval
        )

        if prog is None or len(prog) == 0 or prog["gr"].dropna().empty:
            logging.error(
                f"PV: geen weerprognose beschikbaar voor {installation.name}, "
                f"voorspelling op 0 gezet"
            )
            index = pd.date_range(
                start, end, freq=pd.Timedelta(seconds=interval_s), inclusive="left"
            )
            return pd.DataFrame({"tijd": index, "prediction": [0.0] * len(index)})

        weather = weather_for_pv(prog, self.tz)
        params, _source = self.params_for(installation)
        production = simulate(params, weather, self.latitude, self.longitude, interval_s)
        return pd.DataFrame(
            {"tijd": production.index, "prediction": production.to_numpy()}
        ).reset_index(drop=True)

    def forecast_from_weather(self, installation, weather: pd.DataFrame) -> pd.Series:
        """Forecast production from a weather frame already in pv layout
        (``ghi``/``dni``/``dhi``/``temp``/``wind``), for the solar report."""
        params, _source = self.params_for(installation)
        interval_s = self._interval_s(self.interval)
        return simulate(params, weather, self.latitude, self.longitude, interval_s)

    def _values_weather(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        """``gr``/``dni``/``dhi``/``temp``/``winds`` from measured ``values``."""
        columns: dict[str, pd.Series] = {}
        for code in _WEATHER_CODES:
            frame = self.db_da.get_column_data("values", code, start=start, end=end)
            if frame is None or len(frame) == 0:
                continue
            columns[code] = pd.Series(
                frame["value"].astype(float).to_numpy(),
                index=frame["utc"].astype("int64").to_numpy(),
            )
        if "gr" not in columns or columns["gr"].empty:
            return None
        frame = pd.DataFrame(columns)
        frame.index.name = "time"
        return frame.reset_index()

    def _forecasts_archive_weather(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        """``gr``/``dni``/``dhi``/``temp``/``winds`` from the forecast archive,
        at lead bucket 0 or 1 -- the closest thing to a measurement once
        ``values`` has nothing."""
        from sqlalchemy import Table, and_, select
        from sqlalchemy.exc import NoSuchTableError

        try:
            forecasts = Table(
                "forecasts", self.db_da.metadata, autoload_with=self.db_da.engine
            )
            variabel = Table(
                "variabel", self.db_da.metadata, autoload_with=self.db_da.engine
            )
        except NoSuchTableError:
            return None

        query = select(
            forecasts.c.target_time.label("time"),
            variabel.c.code,
            forecasts.c.value,
        ).where(
            and_(
                forecasts.c.variabel == variabel.c.id,
                variabel.c.code.in_(_WEATHER_CODES),
                forecasts.c.lead_bucket.in_([0, 1]),
                forecasts.c.target_time >= int(start.timestamp()),
                forecasts.c.target_time < int(end.timestamp()),
            )
        )
        with self.db_da.engine.connect() as connection:
            rows = connection.execute(query).fetchall()
        if not rows:
            return None
        long_frame = pd.DataFrame(rows, columns=["time", "code", "value"])
        pivot = long_frame.pivot_table(
            index="time", columns="code", values="value", aggfunc="first"
        )
        if "gr" not in pivot.columns or pivot["gr"].dropna().empty:
            return None
        return pivot.reset_index()

    def _calibration_weather(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> pd.DataFrame:
        frame = self._values_weather(start, end)
        if frame is None:
            logging.warning(
                "PV-kalibratie: geen straling in 'values', val terug op het "
                "prognose-archief (lead bucket 0/1)"
            )
            frame = self._forecasts_archive_weather(start, end)
        if frame is None:
            return pd.DataFrame(columns=["time", *_WEATHER_CODES])
        for code in _WEATHER_CODES:
            if code not in frame.columns:
                frame[code] = float("nan")
        return weather_for_pv(frame, self.tz)

    def calibrate_installation(self, installation) -> Optional[CalibrationResult]:
        """Fit and, if accepted, save the physical model for one installation."""
        now = self._now()
        start = now - datetime.timedelta(days=365)
        cap = 1.2 * installation.total_capacity if installation.total_capacity else None

        reader = HistoryReader(self.db_ha, self.tz)
        production = reader.read_energy(
            list(installation.entities_sensors), start, now, cap_kwh=cap
        )

        weather = self._calibration_weather(start, now)
        if weather.empty:
            logging.warning(
                f"PV-kalibratie {installation.name}: geen bruikbare weerdata, "
                f"kalibratie overgeslagen"
            )
            return None

        config_params = params_from_config(installation)
        result = calibrate(
            config_params,
            production,
            weather,
            installation.calibration,
            self.latitude,
            self.longitude,
            interval_s=3600,
            now=now,
        )
        if result is not None:
            save_calibration(result, calibration_path(self.data_dir, installation.name))
        return result

    def run_training(self) -> None:
        """Calibrate every installation, then train the ML model for those
        configured for it.

        Calibration always runs first: an installation's ml/auto model
        trains on features that include the physical model's own output,
        so that output should be current before training starts.
        """
        installations = self.installations()
        for installation in installations:
            result = self.calibrate_installation(installation)
            if result is not None:
                logging.info(
                    f"PV-kalibratie {installation.name}: geaccepteerd "
                    f"(holdout-MAE {result.holdout_mae_fit:.3f} kWh, "
                    f"configuratie {result.holdout_mae_config:.3f} kWh)"
                )
            else:
                logging.info(
                    f"PV-kalibratie {installation.name}: geen nieuwe kalibratie"
                )

        ml_installations = [
            installation
            for installation in installations
            if installation.effective_model in ("ml", "auto")
            and installation.entities_sensors
        ]

        if ml_installations:
            from dao.prog.solar_predictor import SolarPredictor

            solar_predictor = SolarPredictor()
            start = (self._now() - datetime.timedelta(days=3 * 365)).replace(tzinfo=None)
            for installation in ml_installations:
                try:
                    solar_predictor.train_solar_option(installation, start)
                except Exception as ex:  # noqa: BLE001 - one bad model must not
                    # abort every other installation's training in the same run.
                    logging.warning(
                        f"ML-training van {installation.name} mislukt: {ex}"
                    )

        for installation in installations:
            self._select_and_store(installation)

    def _select_and_store(self, installation) -> None:
        """Backtest when configured for ``auto``, then record the choice."""
        configured = installation.effective_model
        result = None
        if configured == "auto":
            try:
                result = self._backtest_installation(installation)
            except Exception as ex:  # noqa: BLE001 - physical is the fallback
                logging.warning(
                    f"PV-backtest van {installation.name} mislukt: {ex}"
                )
                result = None

        selection = select_pv_model(configured, result)
        logging.info(
            f"PV-model {installation.name}: {selection.model} ({selection.reason})"
        )
        write_json(
            selection_path(self.data_dir, installation.name), selection.to_dict()
        )

    def _backtest_installation(self, installation, days: int = 28):
        """Compare the physical and ML model over the last ``days`` days."""
        now = self._now()
        start = now - datetime.timedelta(days=days + 1)

        production = HistoryReader(self.db_ha, self.tz).read_energy(
            list(installation.entities_sensors),
            start,
            now,
            cap_kwh=1.2 * installation.total_capacity
            if installation.total_capacity
            else None,
        )

        archive = self._archive_weather(start, now)
        perfect_weather = archive is None
        weather = archive if archive is not None else self._calibration_weather(start, now)
        if weather is None or len(weather) == 0:
            return None

        params, _source = self.params_for(installation)
        candidates = [
            PhysicalCandidate(params, self.latitude, self.longitude, 3600),
            MLCandidate(
                lambda inst, frame: self.forecast_from_weather(inst, frame),
                installation,
            ),
        ]

        def context_for_day(day):
            day_start = pd.Timestamp(
                datetime.datetime.combine(day, datetime.time.min)
            ).tz_localize(self.tz)
            day_end = day_start + pd.Timedelta(days=1)
            window = weather[(weather.index >= day_start) & (weather.index < day_end)]
            return {"weather": window}

        return backtest(
            "pv",
            candidates,
            production,
            days,
            end=now.date(),
            context_for_day=context_for_day,
            perfect_weather=perfect_weather,
        )

    def _archive_weather(self, start, end) -> Optional[pd.DataFrame]:
        """Archived forecasts at lead bucket 12/24, in pv layout.

        ``None`` when the archive does not cover the window: the caller
        then falls back to observations and marks the result
        ``perfect_weather``, because a model scored against the weather
        that actually happened flatters itself.
        """
        try:
            rows = self.db_da.forecast_rows(
                ["gr", "dni", "dhi", "temp"],
                [12, 24],
                int(start.timestamp()),
                int(end.timestamp()),
            )
        except Exception as ex:  # noqa: BLE001 - observations cover it
            logging.debug(f"PV-backtest: archief niet leesbaar: {ex}")
            return None
        if rows is None or len(rows) == 0:
            return None

        pivot = (
            rows.pivot_table(
                index="target_time", columns="code", values="value", aggfunc="first"
            )
            .reset_index()
            .rename(columns={"target_time": "time"})
        )
        expected_days = (end.date() - start.date()).days
        covered = pd.to_datetime(pivot["time"], unit="s", utc=True).dt.date.nunique()
        if covered < expected_days:
            return None
        for code in _WEATHER_CODES:
            if code not in pivot.columns:
                pivot[code] = float("nan")
        return weather_for_pv(pivot, self.tz)
