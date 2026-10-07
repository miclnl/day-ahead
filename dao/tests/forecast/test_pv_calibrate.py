"""Tests for PV calibration: fitting against measured production."""

from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd
import pvlib
import pytest

from dao.forecast.pv.calibrate import CalibrationResult, calibrate
from dao.forecast.pv.physical import Plane, PVParams, clearsky_irradiance, simulate
from dao.forecast.pv.store import calibration_path, load_calibration, save_calibration

TZ = "Europe/Amsterdam"
LAT, LON = 52.1, 5.2


def clear_sky_weather(times: pd.DatetimeIndex, interval_s: int) -> pd.DataFrame:
    sun_times = times + pd.Timedelta(seconds=interval_s / 2)
    solpos = pvlib.solarposition.get_solarposition(sun_times, LAT, LON)
    zenith = np.asarray(solpos["apparent_zenith"])
    airmass = pvlib.atmosphere.get_relative_airmass(zenith)
    airmass = np.where(np.isnan(airmass), 10.0, airmass)
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass)
    turbidity = np.asarray(pvlib.clearsky.lookup_linke_turbidity(sun_times, LAT, LON))
    dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(sun_times))
    clearsky = clearsky_irradiance(zenith, airmass_abs, turbidity, dni_extra)
    return pd.DataFrame(
        {"ghi": clearsky["ghi"], "dni": clearsky["dni"], "dhi": clearsky["dhi"]},
        index=times,
    )


def synthetic_production(true_params: PVParams, days: int = 120):
    """Clear-sky weather scaled by a daily cloud factor, plus 3% noise."""
    rng = np.random.default_rng(3)
    start = pd.Timestamp("2026-03-01", tz=TZ)
    times = pd.date_range(start, periods=days * 24, freq="h")
    weather = clear_sky_weather(times, 3600)
    cloud_daily = rng.uniform(0.4, 1.0, size=days)
    cloud_factor = np.repeat(cloud_daily, 24)
    for column in ("ghi", "dni", "dhi"):
        weather[column] = weather[column] * cloud_factor
    weather["temp"] = 15.0
    weather["wind"] = 3.0

    clean = simulate(true_params, weather, LAT, LON, 3600)
    noise = rng.normal(0, 0.03, size=len(clean))
    production = clean * (1 + noise)
    production.name = "production"
    return production, weather


@pytest.mark.slow
def test_scale_mode_recovers_scale_factor():
    true_params = PVParams(planes=[Plane(tilt=45, azimuth=238, pdc0_kw=2.4)])
    config_params = PVParams(planes=[Plane(tilt=45, azimuth=238, pdc0_kw=3.6)])
    production, weather = synthetic_production(true_params)

    result = calibrate(config_params, production, weather, "scale", LAT, LON)

    assert result is not None
    assert result.params.total_kwp() == pytest.approx(2.4, rel=0.05)
    assert 0.97 <= result.ratio <= 1.03


@pytest.mark.slow
def test_planes_mode_recovers_two_planes():
    true_params = PVParams(
        planes=[
            Plane(tilt=30, azimuth=120, pdc0_kw=1.0),
            Plane(tilt=25, azimuth=250, pdc0_kw=1.2),
        ]
    )
    # Generic starting guess for both planes -- the fit has to separate them
    # from a symmetric start using nothing but the data.
    config_params = PVParams(
        planes=[
            Plane(tilt=45, azimuth=200, pdc0_kw=1.8),
            Plane(tilt=45, azimuth=280, pdc0_kw=1.8),
        ]
    )
    production, weather = synthetic_production(true_params)

    result = calibrate(config_params, production, weather, "planes", LAT, LON)

    assert result is not None
    fitted_azimuths = sorted(plane.azimuth for plane in result.params.planes)
    true_azimuths = sorted([120, 250])
    for fitted, true in zip(fitted_azimuths, true_azimuths, strict=True):
        assert abs(fitted - true) <= 15
    assert result.params.total_kwp() == pytest.approx(2.2, rel=0.10)


@pytest.mark.parametrize(
    ("tilt", "azimuth", "what"),
    [
        (0.0, 180.0, "een plat dak"),
        (35.0, 20.0, "een vlak op het noordnoordoosten"),
    ],
)
def test_planes_mode_starts_inside_its_bounds(tilt, azimuth, what, caplog):
    """The spec bounds the fit to tilt 5-70 and azimuth 60-300 but starts it
    from the configuration, and SolarConfig allows tilt 0 and any
    orientation. least_squares then raises "Initial guess is outside of
    provided bounds" before a single residual is computed."""
    true_params = PVParams(planes=[Plane(tilt=tilt, azimuth=azimuth, pdc0_kw=2.4)])
    config_params = PVParams(planes=[Plane(tilt=tilt, azimuth=azimuth, pdc0_kw=3.6)])
    production, weather = synthetic_production(true_params, days=70)

    with caplog.at_level(logging.WARNING):
        calibrate(config_params, production, weather, "planes", LAT, LON)

    assert any("buiten de grenzen" in message for message in caplog.messages), what


def test_rejects_when_holdout_not_better(caplog):
    params = PVParams(planes=[Plane(tilt=45, azimuth=238, pdc0_kw=3.6)])
    _, weather = synthetic_production(params, days=90)
    exact_production = simulate(params, weather, LAT, LON, 3600)

    with caplog.at_level("INFO"):
        result = calibrate(params, exact_production, weather, "scale", LAT, LON)

    assert result is None
    assert any("niet beter" in message for message in caplog.messages)


def test_needs_sixty_days():
    params = PVParams(planes=[Plane(tilt=45, azimuth=238, pdc0_kw=3.6)])
    production, weather = synthetic_production(params, days=30)

    result = calibrate(params, production, weather, "scale", LAT, LON)

    assert result is None


def test_mode_off_returns_none():
    params = PVParams(planes=[Plane(tilt=45, azimuth=238, pdc0_kw=3.6)])
    production, weather = synthetic_production(params, days=120)

    result = calibrate(params, production, weather, "off", LAT, LON)

    assert result is None


def test_round_trip_store(tmp_path):
    params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=3.0)])
    now = dt.datetime(2026, 6, 1, tzinfo=dt.UTC)
    result = CalibrationResult(
        params=params,
        mode="scale",
        window_start=dt.datetime(2026, 3, 1, tzinfo=dt.UTC),
        window_end=dt.datetime(2026, 6, 1, tzinfo=dt.UTC),
        n_hours=1000,
        holdout_mae_fit=0.05,
        holdout_mae_config=0.08,
        ratio=1.02,
        created=now,
    )
    path = calibration_path(tmp_path, "Roof South")

    save_calibration(result, path)
    loaded = load_calibration(path)

    assert loaded.params == params
    assert loaded.mode == "scale"
    assert loaded.holdout_mae_fit == pytest.approx(0.05)
    assert loaded.holdout_mae_config == pytest.approx(0.08)
    assert loaded.ratio == pytest.approx(1.02)
    assert loaded.created == now

    payload = json.loads(path.read_text())
    assert payload["planes"][0]["orientation"] == pytest.approx(0.0)
