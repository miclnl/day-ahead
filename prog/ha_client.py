import os
import json
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import urllib.request
import urllib.error
import urllib.parse


def _build_request(url: str, token: Optional[str]) -> urllib.request.Request:
    req = urllib.request.Request(url)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def _http_get_json(url: str, token: Optional[str], timeout: int = 5) -> Any:
    req = _build_request(url, token)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8"))


def get_base_and_token(config) -> Tuple[Optional[str], Optional[str]]:
    # Prefer Supervisor proxy if available (inside add-on)
    sup_token = os.getenv("SUPERVISOR_TOKEN")
    if sup_token:
        return "http://supervisor/core/api", sup_token

    # Fallback to config-provided URL/token
    try:
        url = config.get(["homeassistant", "url"], None)
        token = config.get(["homeassistant", "token"], None)
        if url and token:
            return url.rstrip("/"), token
    except Exception:
        pass
    return None, None


def get_core_config(config) -> Optional[Dict[str, Any]]:
    base, token = get_base_and_token(config)
    if not base:
        return None
    try:
        return _http_get_json(f"{base}/api/config", token, timeout=5)
    except Exception:
        return None


def get_states(config) -> List[Dict[str, Any]]:
    base, token = get_base_and_token(config)
    if not base:
        return []
    try:
        data = _http_get_json(f"{base}/api/states", token, timeout=10)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def suggest_pv_energy_entities(states: List[Dict[str, Any]]) -> List[str]:
    suggestions: List[str] = []
    for st in states:
        try:
            entity_id = st.get("entity_id")
            attrs = st.get("attributes", {})
            device_class = attrs.get("device_class")
            state_class = attrs.get("state_class")
            unit = attrs.get("unit_of_measurement", "")
            name = (attrs.get("friendly_name") or "").lower()
            eid = (entity_id or "").lower()
            # Prefer energy totals (kWh or Wh, total increasing)
            if device_class == "energy" and state_class in ("total", "total_increasing"):
                if "kwh" in unit.lower() or "wh" in unit.lower():
                    if any(k in eid for k in ["solar", "pv"]) or any(k in name for k in ["solar", "pv", "zonne"]):
                        suggestions.append(entity_id)
        except Exception:
            continue
    # de-duplicate
    return list(dict.fromkeys(suggestions))[:10]


def _iso(dt_obj: dt.datetime) -> str:
    return dt_obj.replace(microsecond=0).isoformat()


def get_statistics_max_daily(config, entity_id: str, days: int = 365) -> Optional[float]:
    """Gebruik HA statistics API om max dagelijkse energie (kWh) te bepalen.
    Return kWh (float) of None.
    """
    base, token = get_base_and_token(config)
    if not base:
        return None
    start = dt.datetime.now() - dt.timedelta(days=days)
    # Statistics API: /api/statistics/period?start_time=...&period=day&statistic_id=ENTITY
    url = f"{base}/api/statistics/period?start_time={_iso(start)}&period=day&statistic_id={entity_id}"
    try:
        data = _http_get_json(url, token, timeout=15)
        # Verwacht list[list of samples], kies entity data
        if isinstance(data, list) and data:
            series = data[0] if isinstance(data[0], list) else data
            vals: List[float] = []
            for row in series:
                # Home Assistant kan 'sum' of 'change' leveren afhankelijk van sensor
                v = row.get("sum") or row.get("change")
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if vals:
                return max(vals)
    except Exception:
        pass
    return None


# ---- Service calls ----
def _http_post_json(url: str, token: Optional[str], body: Dict[str, Any], timeout: int = 5) -> Any:
    data = json.dumps(body).encode('utf-8')
    req = _build_request(url, token)
    req.method = 'POST'
    with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
        res = resp.read()
        try:
            return json.loads(res.decode('utf-8'))
        except Exception:
            return None


def call_service(config, domain: str, service: str, service_data: Dict[str, Any]) -> bool:
    base, token = get_base_and_token(config)
    if not base:
        return False
    try:
        url = f"{base}/api/services/{urllib.parse.quote(domain)}/{urllib.parse.quote(service)}"
        _http_post_json(url, token, service_data, timeout=10)
        return True
    except Exception:
        return False


def turn_switch(config, entity_id: str, on: bool) -> bool:
    domain = 'switch'
    service = 'turn_on' if on else 'turn_off'
    return call_service(config, domain, service, {"entity_id": entity_id})



