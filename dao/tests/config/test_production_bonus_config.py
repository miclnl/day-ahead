"""prices."production bonus": the parameters of a supplier bonus on feed-in.

The defaults are the Budget Thuis ZonneBonus values, so a Budget Thuis customer
only sets "enabled": true. A malformed window table must fail at load time:
get_price_data() and calc_optimum() index it blindly by month.
"""

import datetime

import pytest
from pydantic import ValidationError

from dao.prog.config.models.pricing import PricingConfig
from dao.prog.config.models.production_bonus import DEFAULT_WINDOWS, ProductionBonusConfig

PRICING_BASE = {
    "source day ahead": "nordpool",
    "energy taxes consumption": {"2024-01-01": 0.1},
    "energy taxes production": {"2024-01-01": 0.0},
    "cost supplier consumption": {"2024-01-01": 0.02},
    "cost supplier production": {"2024-01-01": 0.0},
    "vat consumption": {"2024-01-01": 21},
    "vat production": {"2024-01-01": 21},
    "last invoice": "2024-01-01",
}


def test_defaults_are_the_budget_thuis_values():
    cfg = ProductionBonusConfig()
    assert cfg.enabled is False
    assert cfg.mode == "pv only"
    assert cfg.start_date == datetime.date(2027, 1, 1)
    assert cfg.percentage == 30.0
    assert cfg.max_exchange_price == 0.20
    assert cfg.yearly_cap_kwh == 5000.0
    assert cfg.delivery_year_start is None
    assert cfg.windows == DEFAULT_WINDOWS
    assert DEFAULT_WINDOWS == {
        "1": [9, 17], "2": [8, 18], "3": [7, 19], "4": [7, 21],
        "5": [6, 21], "6": [5, 22], "7": [6, 22], "8": [6, 21],
        "9": [7, 20], "10": [8, 19], "11": [8, 17], "12": [9, 17],
    }


def test_aliases_with_spaces_are_accepted():
    cfg = ProductionBonusConfig(
        **{
            "enabled": True,
            "mode": "all export",
            "start date": "2027-01-01",
            "max exchange price": 0.15,
            "yearly cap kwh": 4000,
            "delivery year start": "2025-08-10",
        }
    )
    assert cfg.enabled is True
    assert cfg.mode == "all export"
    assert cfg.max_exchange_price == 0.15
    assert cfg.yearly_cap_kwh == 4000.0
    assert cfg.delivery_year_start == datetime.date(2025, 8, 10)


def test_python_names_are_accepted_too():
    cfg = ProductionBonusConfig(enabled=True, delivery_year_start=datetime.date(2025, 8, 10))
    assert cfg.delivery_year_start == datetime.date(2025, 8, 10)


def test_yearly_cap_may_be_null():
    assert ProductionBonusConfig(**{"yearly cap kwh": None}).yearly_cap_kwh is None


@pytest.mark.parametrize(
    "windows, message",
    [
        ({**DEFAULT_WINDOWS, "13": [9, 17]}, "months 1..12"),
        ({k: v for k, v in DEFAULT_WINDOWS.items() if k != "6"}, "months 1..12"),
        ({**DEFAULT_WINDOWS, "1": [17, 9]}, "0 <= start < end <= 24"),
        ({**DEFAULT_WINDOWS, "1": [9, 25]}, "0 <= start < end <= 24"),
        ({**DEFAULT_WINDOWS, "1": [9]}, "two integers"),
        ({**DEFAULT_WINDOWS, "1": ["09:00", "17:00"]}, "valid integer"),
    ],
)
def test_a_malformed_window_table_is_rejected(windows, message):
    with pytest.raises(ValidationError, match=message):
        ProductionBonusConfig(windows=windows)


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValidationError):
        ProductionBonusConfig(mode="battery too")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"percentage": 130}, "less than or equal to 100"),
        ({"percentage": -1}, "greater than or equal to 0"),
        ({"max exchange price": 0}, "greater than 0"),
        ({"yearly cap kwh": -1}, "greater than 0"),
    ],
)
def test_out_of_range_numbers_are_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        ProductionBonusConfig(**overrides)


def test_pricing_without_the_key_has_no_bonus():
    assert PricingConfig(**PRICING_BASE).production_bonus is None


def test_pricing_with_the_key_builds_the_model():
    pricing = PricingConfig(**{**PRICING_BASE, "production bonus": {"enabled": True}})
    assert isinstance(pricing.production_bonus, ProductionBonusConfig)
    assert pricing.production_bonus.enabled is True
    assert pricing.production_bonus.windows == DEFAULT_WINDOWS
