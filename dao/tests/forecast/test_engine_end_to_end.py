"""The whole forecast engine on one synthetic installation.

This is the test that guards the removal of the old forecasting code: it
drives every service end to end (weather, baseload, PV, accuracy) against
a SQLite recorder and a stubbed weather API, and asserts on the artefacts
they leave behind. If a deletion elsewhere breaks a path that production
actually uses, this fails.
"""

from __future__ import annotations

import datetime as dt
import json
import types

import numpy as np
import pandas as pd
import pvlib
import pytest

from dao.forecast.baseload.service import BaseloadService
from dao.forecast.evaluate import archive_accuracy
from dao.forecast.pv.physical import Plane, PVParams, simulate
from dao.forecast.pv.service import PVService
from dao.forecast.pv.store import calibration_path, load_calibration
from dao.forecast.weather.service import WeatherService
from dao.lib.db_manager import DBmanagerObj, forecasts_table
from dao.prog.config.models.devices.solar import SolarConfig
from dao.prog.config.models.weather import WeatherConfig
from dao.tests.forecast.conftest import HOUR, TZ, RecorderHelper

LAT, LON = 52.1, 5.2
NOW = dt.datetime(2026, 6, 21, 0, 0, tzinfo=dt.UTC)
HISTORY_DAYS = 90
VACATION_DAYS = 10

#: The installation the synthetic production is generated from; calibration
#: has to find its way back to roughly this from the measurements alone.
TRUE_PARAMS = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=2.6)])


def clear_sky(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Clear-sky irradiance, scaled by a per-day cloud factor."""
    rng = np.random.default_rng(11)
    midpoints = index + pd.Timedelta(minutes=30)
    solpos = pvlib.solarposition.get_solarposition(midpoints, LAT, LON)
    zenith = np.asarray(solpos["apparent_zenith"])
    airmass = pvlib.atmosphere.get_relative_airmass(zenith)
    airmass = np.where(np.isnan(airmass), 10.0, airmass)
    absolute = pvlib.atmosphere.get_absolute_airmass(airmass)
    turbidity = np.asarray(pvlib.clearsky.lookup_linke_turbidity(midpoints, LAT, LON))
    dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(midpoints))
    sky = pvlib.clearsky.ineichen(zenith, absolute, turbidity, dni_extra=dni_extra)

    days = len(index) // 24 + 1
    cloud = np.repeat(rng.uniform(0.45, 1.0, size=days), 24)[: len(index)]
    return pd.DataFrame(
        {
            "ghi": sky["ghi"] * cloud,
            "dni": sky["dni"] * cloud,
            "dhi": sky["dhi"] * cloud,
            "temp": 16.0,
            "wind": 3.0,
        },
        index=index,
    )


def make_installation() -> SolarConfig:
    return SolarConfig(
        name="Roof South",
        tilt=35,
        orientation=0,
        capacity=3.6,          # deliberately wrong: calibration must find 2.6
        model="physical",
        calibration="scale",
        **{"entities sensors": ["sensor.test_pv_energy"]},
    )


def make_config(installation: SolarConfig):
    """A configuration object with exactly what the services read."""
    return types.SimpleNamespace(
        report=types.SimpleNamespace(
            entities_grid_consumption=["sensor.test_grid_in"],
            entities_grid_production=["sensor.test_grid_out"],
            entities_solar_production_ac=["sensor.test_pv_energy"],
            entities_ev_consumption=[],
            entities_wp_consumption=[],
            entities_boiler_consumption=[],
            entities_machine_consumption=["sensor.test_machine"],
            entities_battery_consumption=[],
            entities_battery_production=[],
        ),
        solar=[installation],
        battery=[],
        grid=types.SimpleNamespace(max_power=17.0),
        baseload_calc_periode=HISTORY_DAYS,
        baseload=None,
        baseload_options=types.SimpleNamespace(
            aggregate="mean",
            trim_fraction=0.2,
            remove_outliers=True,
            outlier_factor=2.0,
            half_life_days=28.0,
            holidays="sunday",
            clip_negative=True,
            min_samples=3,
            model="profile",
            ml_min_days=120,
            backtest_days=7,
            absence=types.SimpleNamespace(
                detect=True,
                threshold=0.4,
                entities_presence=[],
                entity_away=None,
                away_state="on",
                entity_calendar=None,
                calendar_keywords=["vakantie"],
                away_after_hours=3,
                assume_next_day_after_hours=24,
            ),
        ),
        meteoserver_key=None,  # Open-Meteo is then the primary source
        meteoserver_model="harmonie",
        meteoserver_attempts=1,
        weather=WeatherConfig(observations="off"),
    )


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _StubSession:
    """Answers Open-Meteo with a generated 72-hour forecast."""

    def __init__(self, payload):
        self.payload = payload
        self.calls: list = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        return _StubResponse(self.payload)


def openmeteo_payload(start: dt.datetime, hours: int) -> dict:
    # parse_openmeteo shifts back one hour, so the first label is start + 1h.
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    weather = clear_sky(index)
    return {
        "hourly": {
            "time": [
                (moment + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
                for moment in index
            ],
            "shortwave_radiation": [float(v) for v in weather["ghi"]],
            "direct_normal_irradiance": [float(v) for v in weather["dni"]],
            "diffuse_radiation": [float(v) for v in weather["dhi"]],
            "temperature_2m": [16.0] * hours,
            "wind_speed_10m": [3.0] * hours,
            "precipitation": [0.0] * hours,
        }
    }


@pytest.fixture
def engine_env(tmp_path):
    """A recorder, a day_ahead database, and 90 days of synthetic history."""
    from sqlalchemy import (
        BigInteger,
        Column,
        Float,
        ForeignKey,
        Integer,
        String,
        Table,
        UniqueConstraint,
        insert,
    )

    # --- day_ahead database -------------------------------------------------
    da_dir = tmp_path / "da"
    da_dir.mkdir()
    db_da = DBmanagerObj(
        db_dialect="sqlite", db_name="day_ahead.db", db_path=str(da_dir)
    )
    metadata = db_da.metadata
    variabel = Table(
        "variabel",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("code", String(10), unique=True, nullable=False),
        Column("name", String(50), unique=True, nullable=False),
        Column("dim", String(10), nullable=False),
        Column("aggregate", String(3), nullable=False, default="avg"),
    )
    for name in ("values", "prognoses"):
        Table(
            name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
            Column("time", BigInteger, nullable=False),
            Column("value", Float),
            UniqueConstraint("variabel", "time"),
        )
    forecasts_table(metadata)
    metadata.create_all(db_da.engine)
    with db_da.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {"id": 4, "code": "gr", "name": "Globale straling", "dim": "J/cm2"},
                {"id": 5, "code": "temp", "name": "Temperatuur", "dim": "C"},
                {"id": 11, "code": "base", "name": "Basislast", "dim": "kWh"},
                {"id": 15, "code": "pv_ac", "name": "Zonne energie AC", "dim": "kWh"},
                {"id": 23, "code": "winds", "name": "Windsnelheid", "dim": "m/s"},
                {"id": 24, "code": "neersl", "name": "Neerslag", "dim": "mm"},
                {"id": 27, "code": "hload", "name": "Geplande huisvraag", "dim": "kWh"},
                {"id": 28, "code": "dni", "name": "Directe straling", "dim": "J/cm2"},
                {"id": 29, "code": "dhi", "name": "Diffuse straling", "dim": "J/cm2"},
                {"id": 30, "code": "away", "name": "Afwezig", "dim": "-"},
                {"id": 31, "code": "presence", "name": "Aanwezigheid", "dim": "-"},
            ],
        )

    # --- recorder database --------------------------------------------------
    ha_dir = tmp_path / "ha"
    ha_dir.mkdir()
    db_ha = DBmanagerObj(
        db_dialect="sqlite", db_name="homeassistant.db", db_path=str(ha_dir)
    )
    ha_meta = db_ha.metadata
    meta = Table(
        "statistics_meta",
        ha_meta,
        Column("id", Integer, primary_key=True),
        Column("statistic_id", String(255)),
        Column("source", String(32)),
        Column("unit_of_measurement", String(255)),
        Column("has_mean", Integer, nullable=True),
        Column("has_sum", Integer),
        Column("name", String(255), nullable=True),
        Column("mean_type", Integer),
        Column("unit_class", String(255), nullable=True),
    )
    stats = Table(
        "statistics",
        ha_meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("created_ts", Float),
        Column("metadata_id", Integer),
        Column("start_ts", Float),
        Column("mean", Float, nullable=True),
        Column("min", Float, nullable=True),
        Column("max", Float, nullable=True),
        Column("last_reset_ts", Float, nullable=True),
        Column("state", Float, nullable=True),
        Column("sum", Float, nullable=True),
        Column("mean_weight", Float, nullable=True),
    )
    ha_meta.create_all(db_ha.engine, tables=[meta, stats])
    helper = RecorderHelper(db_ha, meta, stats)

    # --- synthetic history --------------------------------------------------
    start = NOW - dt.timedelta(days=HISTORY_DAYS)
    index = pd.date_range(start, periods=HISTORY_DAYS * 24, freq="h", tz="UTC")
    weather = clear_sky(index)
    pv_hourly = simulate(TRUE_PARAMS, weather, LAT, LON, 3600).to_numpy()

    vacation_start = start + dt.timedelta(days=40)
    vacation_end = vacation_start + dt.timedelta(days=VACATION_DAYS)

    grid_in, grid_out, pv_energy, machine = {}, {}, {}, {}
    totals = {"grid_in": 0.0, "grid_out": 0.0, "pv": 0.0, "machine": 0.0}
    for i, moment in enumerate(index):
        ts = int(moment.timestamp())
        grid_in[ts] = totals["grid_in"]
        grid_out[ts] = totals["grid_out"]
        pv_energy[ts] = totals["pv"]
        machine[ts] = totals["machine"]

        away = vacation_start <= moment.to_pydatetime() < vacation_end
        hour = moment.tz_convert(TZ).hour
        if away:
            household = 0.10
            machine_use = 0.0
        else:
            household = 0.8 if 17 <= hour < 21 else (0.35 if 7 <= hour < 23 else 0.12)
            machine_use = 0.3 if hour == 10 else 0.0

        production = float(pv_hourly[i])
        demand = household + machine_use
        totals["pv"] += production
        totals["machine"] += machine_use
        # Self-consumption first, the surplus goes to the grid.
        totals["grid_in"] += max(0.0, demand - production)
        totals["grid_out"] += max(0.0, production - demand)

    last = int(index[-1].timestamp()) + HOUR
    for series, total_key in (
        (grid_in, "grid_in"),
        (grid_out, "grid_out"),
        (pv_energy, "pv"),
        (machine, "machine"),
    ):
        series[last] = totals[total_key]

    helper.add_energy("sensor.test_grid_in", "kWh", grid_in)
    helper.add_energy("sensor.test_grid_out", "kWh", grid_out)
    helper.add_energy("sensor.test_pv_energy", "kWh", pv_energy)
    helper.add_energy("sensor.test_machine", "kWh", machine)

    # Measured weather, which the PV calibration fits against.
    rows = []
    for moment, row in weather.iterrows():
        ts = int(moment.timestamp())
        rows.append((ts, "gr", float(row["ghi"]) * 0.36))
        rows.append((ts, "temp", float(row["temp"])))
        rows.append((ts, "winds", float(row["wind"])))
    db_da.savedata(pd.DataFrame(rows, columns=["time", "code", "value"]))

    # An archive as it stands after a week of running: forecasts made 12 and
    # 24 hours ahead of targets that have since happened. save_forecasts
    # deliberately drops a target already in the past, so a single run can
    # never produce this -- without it the accuracy report has nothing to
    # score and every assertion on it is vacuous.
    archive_rows = []
    archive_start = NOW - dt.timedelta(days=7)
    for moment, row in weather.iterrows():
        if moment.to_pydatetime() < archive_start:
            continue
        ts = int(moment.timestamp())
        for lead, error in ((12, 1.06), (24, 0.91)):
            archive_rows.append(
                {
                    "variabel": 4,  # gr
                    "target_time": ts,
                    "lead_bucket": lead,
                    "issued_time": ts - lead * HOUR,
                    "value": float(row["ghi"]) * 0.36 * error,
                    "source": "meteoserver",
                }
            )
    with db_da.engine.begin() as connection:
        connection.execute(insert(Table("forecasts", metadata)), archive_rows)

    installation = make_installation()
    config = make_config(installation)
    data_root = tmp_path / "forecast"
    session = _StubSession(openmeteo_payload(NOW, 72))

    return types.SimpleNamespace(
        config=config,
        installation=installation,
        db_da=db_da,
        db_ha=db_ha,
        data_root=data_root,
        session=session,
        vacation_start=vacation_start,
        vacation_end=vacation_end,
    )


@pytest.mark.slow
def test_engine_runs_all_tasks_on_synthetic_data(engine_env):
    env = engine_env
    now = NOW

    # --- weather ------------------------------------------------------------
    weather_service = WeatherService(
        env.config,
        env.db_da,
        LAT,
        LON,
        secrets={},
        data_dir=env.data_root / "weather",
        country="NL",
        now=lambda: now,
        session=env.session,
    )
    status = weather_service.update(horizon_hours=72)
    assert status.hours_total >= 72

    prognoses = env.db_da.get_prognose_fields(
        ["gr", "dni", "dhi", "temp"],
        int(now.timestamp()),
        int((now + dt.timedelta(hours=72)).timestamp()),
    )
    assert len(prognoses) >= 72
    for code in ("gr", "dni", "dhi", "temp"):
        assert prognoses[code].notna().any(), code

    # --- baseload -----------------------------------------------------------
    baseload = BaseloadService(
        env.config,
        env.db_da,
        env.db_ha,
        env.data_root / "baseload",
        TZ,
        now=lambda: now.astimezone(dt.UTC),
        latitude=LAT,
        longitude=LON,
    )
    profile_set = baseload.fit()

    assert len(profile_set.home) == 7
    assert profile_set.away_source == "labels"

    away_rows = env.db_da.get_column_data(
        "values", "away", start=env.vacation_start, end=env.vacation_end
    )
    assert len(away_rows[away_rows["value"] >= 0.5]) >= VACATION_DAYS - 2

    saved = json.loads((env.data_root / "baseload" / "profile.json").read_text())
    assert len(saved["home"]) == 7

    horizon = baseload.forecast_for_optimizer(now, 48, "1hour")
    assert len(horizon) == 48
    assert all(np.isfinite(value) for value in horizon)

    # --- PV -----------------------------------------------------------------
    pv = PVService(
        env.config,
        env.db_da,
        env.db_ha,
        LAT,
        LON,
        env.data_root / "pv",
        TZ,
        "1hour",
        now=lambda: now,
    )
    pv.run_training()

    path = calibration_path(env.data_root / "pv", env.installation.name)
    assert path.exists(), "calibration artefact was not written"
    calibration = load_calibration(path)
    assert calibration.params.total_kwp() == pytest.approx(
        TRUE_PARAMS.total_kwp(), rel=0.10
    )

    selection = json.loads(
        (env.data_root / "pv" / "Roof_South.selection.json").read_text()
    )
    assert selection["model"] == "physical"

    forecast = pv.forecast(
        env.installation, now, now + dt.timedelta(hours=48), "1hour"
    )
    assert len(forecast) == 48
    assert (forecast["prediction"] >= 0).all()

    # --- accuracy -----------------------------------------------------------
    report = archive_accuracy(
        env.config, env.db_da, env.db_ha, TZ, days=(7, 28), now=now
    )
    assert json.dumps(report.to_dict())
    # The weather service archived its own forecast, so gr has pairs to
    # score against the measured series written above. "gr" in components
    # is unconditionally true -- every component gets a key, scored or not
    # -- so the only assertion worth making is that it actually scored
    # something.
    gr_window = report.components["gr"].windows[7]
    assert gr_window["pairs"] > 0
    assert gr_window["by_lead"]
