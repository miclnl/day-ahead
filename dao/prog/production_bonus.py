"""Bonus op teruglevering (bijv. de ZonneBonus van Budget Thuis): alleen de regels.

Pure functies: geen database, geen Home Assistant. Elke functie behalve
add_to_model() accepteert cfg=None en antwoordt dan "geen bonus", zodat
aanroepers geen eigen guard nodig hebben.

add_to_model is uitsluitend legacy: geen fallback voor nieuwe settlement,
geen bewijs van bruto-celuitsluiting of chronologische cap in het nieuwe regime.
"""

import datetime
from decimal import Decimal

from dao.prog.config.models.production_bonus import DEFAULT_WINDOWS
from dao.prog.settlement.engine import delivery_year_bounds, remaining_export_cap
from dao.prog.settlement.tariffs import bonus_in_window, bonus_rate_at

__all__ = [
    "DEFAULT_WINDOWS",
    "in_window",
    "bonus_per_kwh",
    "bonus_period_start",
    "remaining_cap_kwh",
    "describe",
    "add_to_model",
]


def in_window(cfg, moment: datetime.datetime) -> bool:
    """True als het interval dat op `moment` begint in het venster van moment.month ligt."""
    if cfg is None or not cfg.enabled:
        return False
    return bonus_in_window(_windows(cfg), moment)


def _windows(cfg):
    return tuple((int(month), tuple(window)) for month, window in cfg.windows.items())


def bonus_per_kwh(
    cfg,
    moment: datetime.datetime,
    exchange_price: float,
    vat_pct: float,
) -> float:
    """Legacycompatibiliteit en floatgrens rond de expliciete Decimal-bonusregel.

    Onbevestigde oude config behoudt de ex-btw-cap en meegegeven export-btw.
    Expliciete nieuwe grondslagen gebruiken alleen hun eigen bonus-btw.
    """
    if cfg is None or not cfg.enabled:
        return 0.0
    legacy = cfg.cap_basis == "unconfirmed"
    # Alleen de adapter bevestigt de oude ex-btw-conventie; nieuwe snapshots niet.
    return float(bonus_rate_at(
        moment, Decimal(str(exchange_price)), enabled=cfg.enabled, start_date=cfg.start_date,
        percentage=Decimal(str(cfg.percentage)), max_exchange_price=Decimal(str(cfg.max_exchange_price)),
        cap_basis="ex_vat" if legacy else cfg.cap_basis,
        vat_percent=Decimal(str(vat_pct)) if legacy else cfg.vat_percent, windows=_windows(cfg),
    ))


def bonus_period_start(cfg, moment: datetime.datetime) -> datetime.date | None:
    """Eerste dag van het huidige bonusjaar, of None als er geen limiet geldt."""
    if cfg is None or not cfg.enabled or cfg.yearly_cap_kwh is None or cfg.delivery_year_start is None:
        return None
    # Compatibiliteit: deze oude helper gebruikt alleen maand/dag van het anker.
    anniversary, _ = delivery_year_bounds(cfg.delivery_year_start, moment.date())
    return max(anniversary, cfg.start_date)


def remaining_cap_kwh(cfg, exported_kwh: float) -> float | None:
    """Resterende kWh die nog bonus krijgt, of None als er geen limiet geldt."""
    if cfg is None or cfg.yearly_cap_kwh is None:
        return None
    return float(remaining_export_cap(Decimal(str(cfg.yearly_cap_kwh)), Decimal(str(exported_kwh))))


def describe(cfg, moment: datetime.datetime, remaining: float | None) -> str:
    """Een logregel met modus, venster, percentage en limietstatus."""
    if cfg is None or not cfg.enabled:
        return "Bonus teruglevering: uit"
    start, end = cfg.windows[str(moment.month)]
    max_bonus = cfg.percentage / 100 * cfg.max_exchange_price
    if cfg.yearly_cap_kwh is None:
        cap = "geen limiet"
    elif remaining is None:
        cap = f"limiet {cfg.yearly_cap_kwh:.0f} kWh niet bijgehouden"
    else:
        cap = f"resterend {remaining:.0f} kWh van {cfg.yearly_cap_kwh:.0f} kWh"
    return (
        f"Bonus teruglevering: modus '{cfg.mode}', venster "
        f"{start:02d}:00-{end:02d}:00, {cfg.percentage:g}% van de beursprijs tot "
        f"{cfg.max_exchange_price:.2f} euro/kWh (max {max_bonus:.3f} euro/kWh ex btw), "
        f"{cap}"
    )


def add_to_model(model, cfg, c_t, pb, pv_kwh, max_kwh, remaining_kwh):
    """Voegt per interval de variabele c_t_bonus (kWh teruglevering met bonus)
    en de bijbehorende beperkingen toe. Geeft (c_t_bonus, bonus_revenue) terug."""
    from mip import CONTINUOUS, xsum  # alleen hier: de rest van de module blijft solver-vrij

    intervals = range(len(c_t))
    c_t_bonus = [
        model.add_var(var_type=CONTINUOUS, lb=0, ub=max_kwh[u] if pb[u] > 0 else 0)
        for u in intervals
    ]
    for u in intervals:
        model += c_t_bonus[u] <= c_t[u]
        if pb[u] > 0 and cfg.mode == "pv only":
            # alleen teruglevering tot de PV-productie van dit interval telt
            model += c_t_bonus[u] <= pv_kwh[u]
    if remaining_kwh is not None:
        model += xsum(c_t_bonus[u] for u in intervals) <= remaining_kwh
    bonus_revenue = xsum(c_t_bonus[u] * pb[u] for u in intervals)
    return c_t_bonus, bonus_revenue
