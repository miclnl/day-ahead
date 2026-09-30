"""Calibrate the physical PV model against measured production.

A configured installation's tilt, orientation and capacity are estimates: a
datasheet capacity is rarely quite what the inverter delivers, and a guessed
tilt is rarely exact. This fits the physical model's own parameters against
measured production, accepting the fit only when it predicts a held-out
slice of that history better than the plain configuration did -- a fit that
merely explains its own training window is not worth replacing a working
configuration for.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import least_squares

from dao.forecast.pv.physical import Plane, PVParams, azimuth_to_dao_orientation, simulate

#: Below this many distinct calendar days in the window, a fit is not
#: trusted: too little variation in sun angle and weather to separate the
#: parameters from noise.
MIN_CALIBRATION_DAYS = 60

#: The trailing fraction of the window held out from fitting, used only to
#: judge whether the fit is worth keeping.
_HOLDOUT_FRACTION = 0.2

_SCALE_BOUNDS = (0.3, 2.0)
_AC_FACTOR_BOUNDS_WITH_CONFIG = (0.3, 2.0)
_AC_FACTOR_BOUNDS_WITHOUT_CONFIG = (0.2, 2.0)
_PLANE_TILT_BOUNDS = (5.0, 70.0)
_PLANE_AZIMUTH_BOUNDS = (60.0, 300.0)
_PLANE_PDC0_FACTOR_BOUNDS = (0.2, 2.0)

#: "planes" mode fits each plane's own tilt/azimuth/capacity separately;
#: beyond this many planes there is not enough signal per plane left in a
#: season of data, so calibration falls back to "scale".
MAX_PLANES_FOR_PLANES_MODE = 2


@dataclass
class CalibrationResult:
    """The fitted parameters and how well they did, for the artefact store."""

    params: PVParams
    mode: str
    window_start: datetime.datetime
    window_end: datetime.datetime
    n_hours: int
    holdout_mae_fit: float
    holdout_mae_config: float
    ratio: float
    created: datetime.datetime

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "planes": [
                {
                    "tilt": plane.tilt,
                    "azimuth": plane.azimuth,
                    "orientation": azimuth_to_dao_orientation(plane.azimuth),
                    "pdc0_kw": plane.pdc0_kw,
                }
                for plane in self.params.planes
            ],
            "ac_max_kw": self.params.ac_max_kw,
            "gamma_pdc": self.params.gamma_pdc,
            "losses_pct": self.params.losses_pct,
            "albedo": self.params.albedo,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "n_hours": self.n_hours,
            "holdout_mae_fit": self.holdout_mae_fit,
            "holdout_mae_config": self.holdout_mae_config,
            "ratio": self.ratio,
            "created": self.created.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CalibrationResult:
        planes = [
            Plane(tilt=p["tilt"], azimuth=p["azimuth"], pdc0_kw=p["pdc0_kw"])
            for p in d["planes"]
        ]
        params = PVParams(
            planes=planes,
            ac_max_kw=d.get("ac_max_kw"),
            gamma_pdc=d.get("gamma_pdc", -0.004),
            losses_pct=d.get("losses_pct", 14.0),
            albedo=d.get("albedo", 0.2),
        )
        return cls(
            params=params,
            mode=d["mode"],
            window_start=datetime.datetime.fromisoformat(d["window_start"]),
            window_end=datetime.datetime.fromisoformat(d["window_end"]),
            n_hours=d["n_hours"],
            holdout_mae_fit=d["holdout_mae_fit"],
            holdout_mae_config=d["holdout_mae_config"],
            ratio=d["ratio"],
            created=datetime.datetime.fromisoformat(d["created"]),
        )


def _ac_reference(config_params: PVParams):
    """The value an AC-ceiling factor multiplies, and its fit bounds."""
    if config_params.ac_max_kw is not None:
        return config_params.ac_max_kw, _AC_FACTOR_BOUNDS_WITH_CONFIG
    return config_params.total_kwp(), _AC_FACTOR_BOUNDS_WITHOUT_CONFIG


def _params_for_scale(config_params: PVParams, x) -> PVParams:
    scale, ac_factor = x
    reference, _ = _ac_reference(config_params)
    planes = [
        Plane(tilt=plane.tilt, azimuth=plane.azimuth, pdc0_kw=plane.pdc0_kw * scale)
        for plane in config_params.planes
    ]
    return PVParams(
        planes=planes,
        ac_max_kw=ac_factor * reference,
        gamma_pdc=config_params.gamma_pdc,
        losses_pct=config_params.losses_pct,
        albedo=config_params.albedo,
    )


def _params_for_planes(config_params: PVParams, x) -> PVParams:
    n = len(config_params.planes)
    planes = []
    for i, plane in enumerate(config_params.planes):
        tilt, azimuth, pdc0_factor = x[i * 3], x[i * 3 + 1], x[i * 3 + 2]
        planes.append(Plane(tilt=tilt, azimuth=azimuth, pdc0_kw=pdc0_factor * plane.pdc0_kw))
    ac_factor = x[n * 3]
    reference, _ = _ac_reference(config_params)
    return PVParams(
        planes=planes,
        ac_max_kw=ac_factor * reference,
        gamma_pdc=config_params.gamma_pdc,
        losses_pct=config_params.losses_pct,
        albedo=config_params.albedo,
    )


def _bounds_for_scale(config_params: PVParams):
    _, ac_bounds = _ac_reference(config_params)
    lower = [_SCALE_BOUNDS[0], ac_bounds[0]]
    upper = [_SCALE_BOUNDS[1], ac_bounds[1]]
    x0 = [1.0, 1.0]
    return x0, lower, upper


def _bounds_for_planes(config_params: PVParams):
    _, ac_bounds = _ac_reference(config_params)
    lower: list = []
    upper: list = []
    x0: list = []
    for plane in config_params.planes:
        lower += [_PLANE_TILT_BOUNDS[0], _PLANE_AZIMUTH_BOUNDS[0], _PLANE_PDC0_FACTOR_BOUNDS[0]]
        upper += [_PLANE_TILT_BOUNDS[1], _PLANE_AZIMUTH_BOUNDS[1], _PLANE_PDC0_FACTOR_BOUNDS[1]]
        x0 += [plane.tilt, plane.azimuth, 1.0]
    lower.append(ac_bounds[0])
    upper.append(ac_bounds[1])
    x0.append(1.0)
    return x0, lower, upper


def calibrate(
    config_params: PVParams,
    production: pd.Series,
    weather: pd.DataFrame,
    mode: str,
    latitude: float,
    longitude: float,
    interval_s: int = 3600,
    now: Optional[datetime.datetime] = None,
) -> Optional[CalibrationResult]:
    """Fit the physical model to measured production, or say why not.

    Returns ``None`` when calibration is switched off, when there is not
    enough history in the window, or when the fit does not beat the plain
    configuration on the held-out slice of it.
    """
    if mode == "off":
        return None

    if mode == "planes" and len(config_params.planes) > MAX_PLANES_FOR_PLANES_MODE:
        logging.warning(
            f"PV-kalibratie: {len(config_params.planes)} vlakken is meer dan de "
            f"{MAX_PLANES_FOR_PLANES_MODE} die 'planes' aankan, terug naar 'scale'"
        )
        mode = "scale"

    now = now or datetime.datetime.now(datetime.UTC)

    joined = weather.join(production.rename("production"), how="inner")
    solpos = pvlib.solarposition.get_solarposition(joined.index, latitude, longitude)
    joined = joined.assign(zenith=np.asarray(solpos["apparent_zenith"]))
    joined = joined[
        (joined["zenith"] < 90)
        & np.isfinite(joined["production"])
        & np.isfinite(joined["ghi"])
    ].sort_index()

    distinct_days = pd.Series(joined.index.date).nunique()
    if distinct_days < MIN_CALIBRATION_DAYS:
        logging.warning(
            f"PV-kalibratie: {distinct_days} dagen bruikbare data, minder dan de "
            f"vereiste {MIN_CALIBRATION_DAYS}; kalibratie overgeslagen"
        )
        return None

    weather_columns = list(weather.columns)
    n_holdout = max(1, int(len(joined) * _HOLDOUT_FRACTION))
    fit_rows = joined.iloc[:-n_holdout]
    holdout_rows = joined.iloc[-n_holdout:]
    fit_weather = fit_rows[weather_columns]
    fit_actual = fit_rows["production"].to_numpy()

    if mode == "planes":
        x0, lower, upper = _bounds_for_planes(config_params)
        build = _params_for_planes
    else:
        x0, lower, upper = _bounds_for_scale(config_params)
        build = _params_for_scale

    def residuals(x):
        params = build(config_params, x)
        predicted = simulate(params, fit_weather, latitude, longitude, interval_s)
        return predicted.to_numpy() - fit_actual

    fit = least_squares(
        residuals, x0=x0, bounds=(lower, upper), diff_step=1e-3, x_scale="jac"
    )
    fitted_params = build(config_params, fit.x)

    holdout_weather = holdout_rows[weather_columns]
    holdout_actual = holdout_rows["production"].to_numpy()
    fit_predicted = simulate(
        fitted_params, holdout_weather, latitude, longitude, interval_s
    ).to_numpy()
    config_predicted = simulate(
        config_params, holdout_weather, latitude, longitude, interval_s
    ).to_numpy()

    holdout_mae_fit = float(np.mean(np.abs(fit_predicted - holdout_actual)))
    holdout_mae_config = float(np.mean(np.abs(config_predicted - holdout_actual)))

    if not (holdout_mae_fit < holdout_mae_config):
        logging.info(
            f"PV-kalibratie: fit niet beter dan configuratie op holdout "
            f"({holdout_mae_fit:.3f} >= {holdout_mae_config:.3f} kWh MAE), "
            f"configuratie blijft gehandhaafd"
        )
        return None

    whole_predicted = simulate(
        fitted_params, joined[weather_columns], latitude, longitude, interval_s
    )
    total_actual = float(joined["production"].sum())
    ratio = float(whole_predicted.sum() / total_actual) if total_actual else 1.0

    return CalibrationResult(
        params=fitted_params,
        mode=mode,
        window_start=joined.index[0].to_pydatetime(),
        window_end=joined.index[-1].to_pydatetime(),
        n_hours=len(joined),
        holdout_mae_fit=holdout_mae_fit,
        holdout_mae_config=holdout_mae_config,
        ratio=ratio,
        created=now,
    )
