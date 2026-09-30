"""Physical PV production model, built on pvlib.

The chain runs twice per call: once against the forecast weather, once
against a clear-sky reference for the same moments and location, so an
occasional irradiance glitch (a weather source reporting several times the
physically possible value) cannot inflate the plan by that same factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pvlib

from dao.forecast.weather.schema import jcm2h_to_wm2

#: W/m^2 of full sun a "yield" factor of 1.0 is taken to correspond to, so a
#: flat installation given only a yield (no capacity) still has a pdc0.
YIELD_TO_KWP = 360.0

#: Real production is allowed to exceed the clear-sky reference by this
#: much before it is capped -- pvlib's own models are not perfectly exact
#: even on a genuinely clear day.
_CLEARSKY_HEADROOM = 1.1

#: Airmass is undefined (NaN) once the sun is far enough below the horizon;
#: pvlib's own examples use this as the fallback for that case.
_AIRMASS_FALLBACK = 10.0


@dataclass(frozen=True)
class Plane:
    """One tilted, oriented surface of PV panels."""

    tilt: float
    azimuth: float  # pvlib convention: 0 = N, 90 = E, 180 = S, 270 = W
    pdc0_kw: float


@dataclass
class PVParams:
    """Everything :func:`simulate` needs to know about one installation."""

    planes: list = field(default_factory=list)
    ac_max_kw: Optional[float] = None
    gamma_pdc: float = -0.004
    losses_pct: float = 14.0
    albedo: float = 0.2

    def total_kwp(self) -> float:
        return sum(plane.pdc0_kw for plane in self.planes)


def dao_orientation_to_azimuth(orientation: float) -> float:
    """DAO orientation (0 south, -90 east, 90 west) -> pvlib azimuth."""
    return (orientation + 180) % 360


def azimuth_to_dao_orientation(azimuth: float) -> float:
    """The inverse of :func:`dao_orientation_to_azimuth`."""
    return ((azimuth - 180) + 180) % 360 - 180


def _pdc0_kw(capacity: Optional[float], yield_factor: Optional[float]) -> float:
    if capacity is not None:
        return capacity
    return (yield_factor or 0.0) * YIELD_TO_KWP


def params_from_config(installation) -> PVParams:
    """Build :class:`PVParams` from a ``SolarConfig`` installation.

    ``strings`` becomes one plane each; a flat installation becomes a
    single plane. Either way pdc0 is the configured capacity, or the old
    "yield" factor scaled by :data:`YIELD_TO_KWP` when capacity is absent.
    """
    if installation.strings:
        planes = [
            Plane(
                tilt=string.tilt,
                azimuth=dao_orientation_to_azimuth(string.orientation),
                pdc0_kw=_pdc0_kw(string.capacity, string.yield_factor),
            )
            for string in installation.strings
        ]
    else:
        planes = [
            Plane(
                tilt=installation.tilt,
                azimuth=dao_orientation_to_azimuth(installation.orientation),
                pdc0_kw=_pdc0_kw(installation.capacity, installation.yield_factor),
            )
        ]
    return PVParams(planes=planes, ac_max_kw=installation.max_power)


def weather_for_pv(prog: pd.DataFrame, tz: str) -> pd.DataFrame:
    """``get_prognose_fields``'s output, reshaped for :func:`simulate`.

    Converts ``gr``/``dni``/``dhi`` from J/cm^2 per hour back to W/m^2 (the
    unit every pvlib call below expects) and renames to pvlib's own
    ``ghi``/``dni``/``dhi``/``temp``/``wind``. A code the source never
    supplied (Meteoserver has no dni/dhi) stays ``NaN`` rather than being
    invented.
    """
    index = pd.DatetimeIndex(
        pd.to_datetime(prog["time"], unit="s", utc=True).dt.tz_convert(tz).to_numpy(),
        name="time",
    )
    result = pd.DataFrame(index=index)
    for source_column, target_column, convert in (
        ("gr", "ghi", True),
        ("dni", "dni", True),
        ("dhi", "dhi", True),
        ("temp", "temp", False),
        ("winds", "wind", False),
    ):
        if source_column in prog.columns:
            values = prog[source_column].astype(float).to_numpy()
            result[target_column] = jcm2h_to_wm2(values) if convert else values
        else:
            result[target_column] = float("nan")
    return result


def _decompose_missing(ghi, zenith, times, dni, dhi):
    """Fill dni/dhi from Erbs decomposition wherever either is missing."""
    dni = np.array(dni, dtype=float)
    dhi = np.array(dhi, dtype=float)
    missing = np.isnan(dni) | np.isnan(dhi)
    if missing.any():
        erbs = pvlib.irradiance.erbs(ghi[missing], zenith[missing], times[missing])
        dni[missing] = np.asarray(erbs["dni"])
        dhi[missing] = np.asarray(erbs["dhi"])
    return dni, dhi


def _ac_kw(params: PVParams, ghi, dni, dhi, temp, wind, zenith, azimuth, dni_extra, airmass):
    """AC kW at each moment for one irradiance scenario."""
    total_dc_kw = np.zeros(len(zenith), dtype=float)
    for plane in params.planes:
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=plane.tilt,
            surface_azimuth=plane.azimuth,
            solar_zenith=zenith,
            solar_azimuth=azimuth,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            dni_extra=dni_extra,
            airmass=airmass,
            model="perez",
            albedo=params.albedo,
        )
        poa_global = np.nan_to_num(np.asarray(poa["poa_global"], dtype=float), nan=0.0)
        tcell = pvlib.temperature.faiman(poa_global, temp, wind)
        dc_w = pvlib.pvsystem.pvwatts_dc(
            poa_global, tcell, plane.pdc0_kw * 1000.0, params.gamma_pdc
        )
        total_dc_kw += np.asarray(dc_w, dtype=float) / 1000.0

    ac_kw = total_dc_kw * (1 - params.losses_pct / 100.0)
    if params.ac_max_kw is not None:
        ac_kw = np.minimum(ac_kw, params.ac_max_kw)
    return np.maximum(ac_kw, 0.0)


def simulate(
    params: PVParams,
    weather: pd.DataFrame,
    latitude: float,
    longitude: float,
    interval_s: int,
) -> pd.Series:
    """kWh produced in each interval of ``weather``'s index.

    Solar position and irradiance decomposition are evaluated at the
    interval's midpoint, not its start, so a quarter-hour step is not
    systematically biased toward the sun's position half a step early.
    """
    times = weather.index
    sun_times = times + pd.Timedelta(seconds=interval_s / 2)

    solpos = pvlib.solarposition.get_solarposition(sun_times, latitude, longitude)
    zenith = np.asarray(solpos["apparent_zenith"], dtype=float)
    azimuth = np.asarray(solpos["azimuth"], dtype=float)

    ghi_raw = weather["ghi"].astype(float).to_numpy()
    ghi_missing = np.isnan(ghi_raw)
    if ghi_missing.any():
        logging.debug(
            f"PV: {int(ghi_missing.sum())} uren zonder globale straling, "
            f"als 0 kWh behandeld"
        )
    ghi = np.where(ghi_missing, 0.0, ghi_raw)

    dni, dhi = _decompose_missing(
        ghi, zenith, sun_times, weather["dni"].astype(float).to_numpy(),
        weather["dhi"].astype(float).to_numpy(),
    )

    dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(sun_times), dtype=float)
    airmass = pvlib.atmosphere.get_relative_airmass(zenith)
    airmass = np.where(np.isnan(airmass), _AIRMASS_FALLBACK, airmass)

    temp = weather["temp"].astype(float).to_numpy()
    wind = weather["wind"].astype(float).to_numpy()

    ac_kw = _ac_kw(params, ghi, dni, dhi, temp, wind, zenith, azimuth, dni_extra, airmass)
    ac_kw = np.where(ghi_missing, 0.0, ac_kw)

    # Clear-sky reference: the same chain, fed pvlib's own clear-sky
    # irradiance for these moments and this location instead of measured.
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass)
    linke_turbidity = np.asarray(
        pvlib.clearsky.lookup_linke_turbidity(sun_times, latitude, longitude),
        dtype=float,
    )
    clearsky = pvlib.clearsky.ineichen(
        zenith, airmass_absolute, linke_turbidity, dni_extra=dni_extra
    )
    clear_ac_kw = _ac_kw(
        params,
        np.asarray(clearsky["ghi"], dtype=float),
        np.asarray(clearsky["dni"], dtype=float),
        np.asarray(clearsky["dhi"], dtype=float),
        temp,
        wind,
        zenith,
        azimuth,
        dni_extra,
        airmass,
    )
    ac_kw = np.minimum(ac_kw, _CLEARSKY_HEADROOM * clear_ac_kw)

    energy_kwh = ac_kw * (interval_s / 3600.0)
    return pd.Series(energy_kwh, index=times, name="pv")
