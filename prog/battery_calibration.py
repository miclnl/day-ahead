import datetime as dt
import json
import math
from typing import Dict, List, Tuple, Optional

import pandas as pd


def _safe_get_series(db, code: str, start: dt.datetime, end: dt.datetime) -> Optional[pd.Series]:
    try:
        df = db.get_column_data("values", code, start, end)
        if df is None or df.empty:
            return None
        s = pd.to_datetime(df["time"]).astype("datetime64[ns]")
        out = pd.Series(pd.to_numeric(df["value"], errors="coerce"), index=pd.to_datetime(df["time"]))
        return out.sort_index()
    except Exception:
        return None


def _resample_hourly(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    if s.index.tz is not None:
        s = s.tz_convert(None)
    # som per uur (kWh) of gemiddelde voor %
    return s.resample("1h").mean() if s.max() <= 100.0 else s.resample("1h").sum()


def calibrate_from_db(db, config, days: int = 14) -> Dict[str, Dict[str, Dict[float, float]]]:
    """Calibreer laad/ontlaad efficiency vs vermogen (kW) uit sqlite data.
    Verwacht dat volgende variabel-codes in de DB bestaan (als codes in 'values.variabel'):
      - entity calculated soc (config: battery[]."entity calculated soc")
      - entity from battery (kWh/h), entity from pv (kWh/h), entity from ac (kWh/h)
    Retourneert: { battery_name: { 'charge': {kW: eff%}, 'discharge': {kW: eff%} } }
    """
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    results: Dict[str, Dict[str, Dict[float, float]]] = {}

    for b_idx, batt in enumerate(config.get(["battery"], None, [])):
        name = batt.get("name", f"Battery{b_idx}")
        soc_code = batt.get("entity calculated soc") or batt.get("entity actual level") or "soc"
        from_bat_code = batt.get("entity from battery", "from_battery")
        from_pv_code = batt.get("entity from pv", "from_pv")
        from_ac_code = batt.get("entity from ac", "from_ac")

        s_soc = _safe_get_series(db, soc_code, start, end)
        s_from_bat = _safe_get_series(db, from_bat_code, start, end)
        s_from_pv = _safe_get_series(db, from_pv_code, start, end)
        s_from_ac = _safe_get_series(db, from_ac_code, start, end)
        if s_soc is None or (s_from_bat is None and s_from_pv is None and s_from_ac is None):
            continue

        s_soc_h = _resample_hourly(s_soc)
        s_bat_h = _resample_hourly(s_from_bat) if s_from_bat is not None else pd.Series(dtype=float)
        s_pv_h = _resample_hourly(s_from_pv) if s_from_pv is not None else pd.Series(dtype=float)
        s_ac_h = _resample_hourly(s_from_ac) if s_from_ac is not None else pd.Series(dtype=float)

        df = pd.DataFrame({
            "soc": s_soc_h,
            "from_bat": s_bat_h,
            "from_pv": s_pv_h,
            "from_ac": s_ac_h,
        }).fillna(0.0)

        if df.empty or len(df) < 24:
            continue

        capacity_kwh = float(batt.get("capacity", 10.0))

        # Compute hour-to-hour SOC delta (% → kWh)
        df["soc_shift"] = df["soc"].shift(1)
        df = df.dropna()
        df["d_soc_%"] = df["soc"] - df["soc_shift"]
        df["d_soc_kwh"] = df["d_soc_%"] * capacity_kwh / 100.0

        # Energy in and out
        df["e_out_kwh"] = df["from_bat"]  # AC energie uit batterij
        df["e_in_kwh"] = df["from_pv"] + df["from_ac"]  # DC naar batterij (benadering)

        # Estimate power (kW) as energy per uur
        df["p_discharge_kw"] = df["e_out_kwh"].clip(lower=0)
        df["p_charge_kw"] = df["e_in_kwh"].clip(lower=0)

        # Compute instantaneous efficiency estimates
        # Discharge: e_out = d_soc_dc * eff_discharge  ⇒ eff_discharge ≈ e_out_kwh / (-d_soc_kwh) voor d_soc<0
        disch = df[df["d_soc_kwh"] < -0.05].copy()
        disch["eff"] = disch.apply(lambda r: (r["e_out_kwh"] / (-r["d_soc_kwh"])) if r["d_soc_kwh"] < 0 else float("nan"), axis=1)
        disch = disch[(disch["eff"].replace([float("inf"), -float("inf")], pd.NA).notna()) & (disch["eff"] > 0) & (disch["eff"] < 1.2)]

        # Charge: d_soc = e_in * eff_charge ⇒ eff_charge ≈ d_soc_kwh / e_in_kwh voor d_soc>0
        ch = df[df["d_soc_kwh"] > 0.05].copy()
        ch["eff"] = ch.apply(lambda r: (r["d_soc_kwh"] / r["e_in_kwh"]) if r["e_in_kwh"] > 0 else float("nan"), axis=1)
        ch = ch[(ch["eff"].replace([float("inf"), -float("inf")], pd.NA).notna()) & (ch["eff"] > 0) & (ch["eff"] < 1.2)]

        # Bin per vermogen (0.5kW stappen)
        def bin_and_median(s: pd.DataFrame, p_col: str) -> Dict[float, float]:
            if s is None or s.empty:
                return {}
            s = s.copy()
            s["bin"] = (s[p_col] / 0.5).round() * 0.5
            grp = s.groupby("bin")["eff"].median()
            out: Dict[float, float] = {}
            for k, v in grp.items():
                if pd.isna(v):
                    continue
                # Clip naar [0.6, 1.0]
                out[float(k)] = float(max(0.6, min(1.0, v))) * 100.0
            return out

        charge_map = bin_and_median(ch, "p_charge_kw")
        discharge_map = bin_and_median(disch, "p_discharge_kw")

        if not charge_map and not discharge_map:
            continue

        results[name] = {
            "charge": charge_map,
            "discharge": discharge_map,
        }

    return results


def write_calibration_json(path: str, calibration: Dict[str, Dict[str, Dict[float, float]]]) -> None:
    try:
        with open(path, "w") as f:
            json.dump(calibration, f, indent=2)
    except Exception:
        pass


