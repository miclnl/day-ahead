"""Tests for the pvlib-based physical PV model."""

from __future__ import annotations

import logging
import types

import numpy as np
import pandas as pd
import pvlib
import pytest

from dao.forecast.pv.physical import (
    YIELD_TO_KWP,
    Plane,
    PVParams,
    azimuth_to_dao_orientation,
    dao_orientation_to_azimuth,
    params_from_config,
    simulate,
)

TZ = "Europe/Amsterdam"
LAT, LON = 52.1, 5.2


def test_orientation_conversion_round_trip():
    assert dao_orientation_to_azimuth(58) == pytest.approx(238)
    assert azimuth_to_dao_orientation(238) == pytest.approx(58)
    assert dao_orientation_to_azimuth(-90) == pytest.approx(90)


def _flat_config(**overrides):
    defaults = dict(
        strings=[],
        tilt=45,
        orientation=58,
        capacity=3.6,
        yield_factor=None,
        max_power=None,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_params_from_flat_config_uses_capacity():
    params = params_from_config(_flat_config())
    assert params.planes == [Plane(tilt=45, azimuth=238, pdc0_kw=3.6)]
    assert params.ac_max_kw is None


def test_params_from_yield_without_capacity():
    params = params_from_config(_flat_config(capacity=None, yield_factor=0.00782))
    assert params.planes[0].pdc0_kw == pytest.approx(0.00782 * YIELD_TO_KWP)
    assert params.planes[0].pdc0_kw == pytest.approx(2.8152)


def test_params_from_strings():
    config = types.SimpleNamespace(
        strings=[
            types.SimpleNamespace(tilt=40, orientation=0, capacity=2.0, yield_factor=None),
            types.SimpleNamespace(tilt=40, orientation=90, capacity=1.5, yield_factor=None),
        ],
        tilt=None,
        orientation=None,
        capacity=None,
        yield_factor=None,
        max_power=2.4,
    )
    params = params_from_config(config)
    assert len(params.planes) == 2
    assert params.ac_max_kw == 2.4


def clear_sky_weather(times: pd.DatetimeIndex, interval_s: int) -> pd.DataFrame:
    """A weather frame filled with pvlib's own clear-sky irradiance."""
    sun_times = times + pd.Timedelta(seconds=interval_s / 2)
    solpos = pvlib.solarposition.get_solarposition(sun_times, LAT, LON)
    zenith = np.asarray(solpos["apparent_zenith"])
    airmass = pvlib.atmosphere.get_relative_airmass(zenith)
    airmass = np.where(np.isnan(airmass), 10.0, airmass)
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass)
    turbidity = np.asarray(pvlib.clearsky.lookup_linke_turbidity(sun_times, LAT, LON))
    dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(sun_times))
    clearsky = pvlib.clearsky.ineichen(zenith, airmass_abs, turbidity, dni_extra=dni_extra)
    return pd.DataFrame(
        {
            "ghi": clearsky["ghi"],
            "dni": clearsky["dni"],
            "dhi": clearsky["dhi"],
            "temp": 20.0,
            "wind": 2.0,
        },
        index=times,
    )


@pytest.fixture
def clear_day_weather():
    times = pd.date_range("2026-06-21 00:00", periods=24, freq="h", tz=TZ)
    return clear_sky_weather(times, 3600)


def test_clear_day_reference(clear_day_weather):
    params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=3.0)])
    result = simulate(params, clear_day_weather, LAT, LON, 3600)

    assert 17.0 <= result.sum() <= 22.0
    midday = result.loc["2026-06-21 12:00:00+02:00"]
    assert 2.0 <= midday <= 2.7
    assert result.loc["2026-06-21 02:00:00+02:00"] == 0.0


def test_ac_cap_applies_per_quarter():
    times = pd.date_range("2026-06-21 08:00", periods=8, freq="15min", tz=TZ)
    weather = clear_sky_weather(times, 900)
    weather["ghi"] = 1200.0  # a bright, cloudless midsummer value
    weather["dni"] = 900.0
    weather["dhi"] = 150.0

    params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=10.0)], ac_max_kw=1.0)
    result = simulate(params, weather, LAT, LON, 900)

    assert (result <= 0.25 + 1e-9).all()


def test_dni_dhi_used_when_present_else_erbs():
    times = pd.date_range("2026-06-21 10:00", periods=4, freq="h", tz=TZ)
    base = clear_sky_weather(times, 3600)

    with_components = base.copy()

    without_components = base.copy()
    without_components["dni"] = float("nan")
    without_components["dhi"] = float("nan")
    # Deliberately wrong dni/dhi so the erbs-derived result differs measurably
    # from using the (correct) supplied components.
    with_components["dni"] = with_components["dni"] * 0.5
    with_components["dhi"] = with_components["dhi"] * 0.5

    params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=3.0)])
    result_supplied = simulate(params, with_components, LAT, LON, 3600)
    result_erbs = simulate(params, without_components, LAT, LON, 3600)

    assert not np.allclose(result_supplied.to_numpy(), result_erbs.to_numpy())


def test_nan_ghi_gives_zero_not_nan():
    times = pd.date_range("2026-06-21 12:00", periods=1, freq="h", tz=TZ)
    weather = pd.DataFrame(
        {
            "ghi": [float("nan")],
            "dni": [float("nan")],
            "dhi": [float("nan")],
            "temp": [20.0],
            "wind": [2.0],
        },
        index=times,
    )
    params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=3.0)])
    result = simulate(params, weather, LAT, LON, 3600)
    assert result.iloc[0] == 0.0
    assert not np.isnan(result.iloc[0])


def test_missing_wind_and_temperature_do_not_nan_the_whole_day(caplog):
    """Faiman's cell temperature is temp + poa/(u0 + u1*wind); one NaN in
    either turns a whole day of production into NaN. No weather source is
    guaranteed to carry both, and the forecast archive did not even store
    wind, so the model fills them and says so."""
    times = pd.date_range("2026-06-21 00:00", periods=24, freq="h", tz=TZ)
    weather = clear_sky_weather(times, 3600)
    weather["temp"] = float("nan")
    weather["wind"] = float("nan")

    params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=3.0)])
    with caplog.at_level(logging.WARNING):
        result = simulate(params, weather, LAT, LON, 3600)

    assert not result.isna().any()
    assert 15.0 <= result.sum() <= 24.0
    messages = " ".join(record.message for record in caplog.records)
    assert "temperatuur" in messages
    assert "wind" in messages


def test_clearsky_cap_limits_glitch_input():
    times = pd.date_range("2026-06-21 12:00", periods=1, freq="h", tz=TZ)
    glitched = clear_sky_weather(times, 3600)
    reference = glitched.copy()
    glitched["ghi"] = 5000.0
    glitched["dni"] = 5000.0
    glitched["dhi"] = 5000.0

    params = PVParams(planes=[Plane(tilt=35, azimuth=180, pdc0_kw=3.0)])
    glitched_result = simulate(params, glitched, LAT, LON, 3600)
    clear_result = simulate(params, reference, LAT, LON, 3600)

    assert glitched_result.iloc[0] <= 1.1 * clear_result.iloc[0] + 1e-9
