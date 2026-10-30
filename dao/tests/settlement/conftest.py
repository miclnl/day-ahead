"""Factories with explicit synthetic terms; never operator configuration."""

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from dao.prog.settlement.types import (
    BonusRules, BonusState, MeterInterval, MonthState, SettlementState, TariffSnapshot,
)


START_TS = int(datetime(2027, 1, 1, tzinfo=ZoneInfo("Europe/Amsterdam")).timestamp())
MONTH_END_TS = int(datetime(2027, 2, 1, tzinfo=ZoneInfo("Europe/Amsterdam")).timestamp())
YEAR_END_TS = int(datetime(2028, 1, 1, tzinfo=ZoneInfo("Europe/Amsterdam")).timestamp())


@pytest.fixture
def bonus_rules():
    return BonusRules(
        enabled=True, mode="pv only", start_date=date(2027, 1, 1),
        percentage=Decimal("30"), max_exchange_price=Decimal("0.20"),
        cap_basis="ex_vat", vat_percent=Decimal("0"), yearly_cap_kwh=Decimal("5000"),
        windows=((1, (9, 17)), (2, (8, 18)), (3, (7, 19)), (4, (7, 21)),
                 (5, (6, 21)), (6, (5, 22)), (7, (6, 22)), (8, (6, 21)),
                 (9, (7, 20)), (10, (8, 19)), (11, (8, 17)), (12, (9, 17))),
    )


@pytest.fixture
def snapshot():
    def build(**overrides):
        zero = ((date(2027, 1, 1), Decimal("0")),)
        one = ((date(2027, 1, 1), Decimal("1")),)
        return TariffSnapshot(**{
            "snapshot_hash": "a" * 64, "rule_version": "synthetic-v1",
            "contract_id": "synthetic-2027", "status": "confirmed",
            "valid_from_ts": START_TS, "valid_to_ts": None,
            "supply_start": date(2026, 8, 10) if overrides.get("bonus") else date(2027, 1, 1),
            "timezone": "Europe/Amsterdam", "product": "budget_dynamic_2027",
            "regime": "no_net_metering", "price_interval_seconds": 3600,
            "meter_interval_seconds": 900, "hour_price_rule": "native",
            "monthly_export_floor": "spot_zero",
            "energy_taxes_consumption": zero, "energy_taxes_production": zero,
            "cost_supplier_consumption": zero, "cost_supplier_production": zero,
            "vat_consumption": zero, "vat_production": zero,
            "multiplier_consumption": one, "multiplier_production": one,
            "bonus": None, "source_versions": (("synthetic", "v1"),),
            **overrides,
        })
    return build


@pytest.fixture
def meter():
    def build(**overrides):
        return MeterInterval(**{
            "start_ts": START_TS, "end_ts": START_TS + 900,
            "import_kwh": Decimal("1"), "export_kwh": Decimal("2"),
            "battery_discharge_kwh": Decimal("0"), "origin": "verified_pv",
            "quality": "measured", "sources": ("synthetic-meter", "no-battery-proof"),
            **overrides,
        })
    return build


@pytest.fixture
def opening():
    def build(**overrides):
        return SettlementState(**{
            "asof_ts": START_TS,
            "bonus": BonusState(START_TS, YEAR_END_TS, START_TS, Decimal("0")),
            "month": MonthState("synthetic-2027", START_TS, MONTH_END_TS,
                                Decimal("0"), (), True),
            **overrides,
        })
    return build
