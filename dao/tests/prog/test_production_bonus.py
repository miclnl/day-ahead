"""production_bonus: the rules of a supplier bonus on feed-in, without database or HA.

Values are the Budget Thuis ZonneBonus action conditions of 7 October 2026:
30% of the exchange price inside a monthly window, over at most 0.20 euro/kWh,
never on a price <= 0, over the first 5000 kWh per delivery year.
"""

import datetime

import pytest

from dao.prog import production_bonus
from dao.prog.config.models.production_bonus import ProductionBonusConfig

CFG = ProductionBonusConfig(enabled=True)
JUNE_NOON = datetime.datetime(2027, 6, 1, 12, 0)


class TestInWindow:
    @pytest.mark.parametrize(
        "moment, expected",
        [
            (datetime.datetime(2027, 1, 15, 8, 59), False),
            (datetime.datetime(2027, 1, 15, 9, 0), True),
            (datetime.datetime(2027, 1, 15, 16, 45), True),  # last quarter of the window
            (datetime.datetime(2027, 1, 15, 16, 59), True),
            (datetime.datetime(2027, 1, 15, 17, 0), False),  # end hour is exclusive
            (datetime.datetime(2027, 6, 15, 4, 59), False),
            (datetime.datetime(2027, 6, 15, 5, 0), True),
            (datetime.datetime(2027, 6, 15, 21, 59), True),
            (datetime.datetime(2027, 6, 15, 22, 0), False),
        ],
    )
    def test_window_boundaries_per_month(self, moment, expected):
        assert production_bonus.in_window(CFG, moment) is expected

    def test_no_config_is_never_in_the_window(self):
        assert production_bonus.in_window(None, JUNE_NOON) is False


class TestBonusPerKwh:
    @pytest.mark.parametrize(
        "price, expected",
        [
            (0.10, 0.03),
            (0.20, 0.06),
            (0.25, 0.06),  # capped at 30% of 0.20
            (0.0, 0.0),
            (-0.05, 0.0),  # never negative
        ],
    )
    def test_percentage_and_cap_without_vat(self, price, expected):
        assert production_bonus.bonus_per_kwh(CFG, JUNE_NOON, price, 0.0) == pytest.approx(expected)

    def test_vat_is_applied_like_the_rest_of_the_feed_in_price(self):
        assert production_bonus.bonus_per_kwh(CFG, JUNE_NOON, 0.10, 21.0) == pytest.approx(0.03 * 1.21)

    def test_outside_the_window_is_zero(self):
        assert production_bonus.bonus_per_kwh(CFG, datetime.datetime(2027, 6, 1, 23, 0), 0.10, 0.0) == 0.0

    def test_before_the_start_date_is_zero(self):
        assert production_bonus.bonus_per_kwh(CFG, datetime.datetime(2026, 12, 31, 12, 0), 0.10, 0.0) == 0.0

    def test_on_the_start_date_it_applies(self):
        assert production_bonus.bonus_per_kwh(CFG, datetime.datetime(2027, 1, 1, 12, 0), 0.10, 0.0) == pytest.approx(0.03)

    def test_disabled_or_absent_config_is_zero(self):
        assert production_bonus.bonus_per_kwh(ProductionBonusConfig(enabled=False), JUNE_NOON, 0.10, 0.0) == 0.0
        assert production_bonus.bonus_per_kwh(None, JUNE_NOON, 0.10, 0.0) == 0.0

    def test_other_parameters_are_honoured(self):
        cfg = ProductionBonusConfig(enabled=True, percentage=10, **{"max exchange price": 0.50})
        assert production_bonus.bonus_per_kwh(cfg, JUNE_NOON, 0.40, 0.0) == pytest.approx(0.04)


class TestBonusPeriodStart:
    ANCHOR = datetime.date(2025, 8, 10)

    def _cfg(self, anchor=ANCHOR, **overrides):
        return ProductionBonusConfig(enabled=True, delivery_year_start=anchor, **overrides)

    def test_first_bonus_year_starts_on_the_start_date(self):
        # delivery year 2026-08-10 .. 2027-08-09 already running on 2027-01-01
        assert production_bonus.bonus_period_start(self._cfg(), datetime.datetime(2027, 3, 1)) == datetime.date(2027, 1, 1)

    def test_after_the_anniversary_the_period_starts_there(self):
        assert production_bonus.bonus_period_start(self._cfg(), datetime.datetime(2027, 9, 1)) == datetime.date(2027, 8, 10)
        assert production_bonus.bonus_period_start(self._cfg(), datetime.datetime(2028, 1, 15)) == datetime.date(2027, 8, 10)

    def test_on_the_anniversary_itself_the_new_period_starts(self):
        assert production_bonus.bonus_period_start(self._cfg(), datetime.datetime(2027, 8, 10, 0, 0)) == datetime.date(2027, 8, 10)

    def test_only_month_and_day_of_the_anchor_matter(self):
        # a contract that starts later in the year than today
        assert production_bonus.bonus_period_start(self._cfg(anchor=datetime.date(2028, 3, 1)), datetime.datetime(2027, 6, 1)) == datetime.date(2027, 3, 1)

    def test_a_29_february_anchor_falls_back_to_28_february(self):
        assert production_bonus.bonus_period_start(self._cfg(anchor=datetime.date(2024, 2, 29)), datetime.datetime(2027, 3, 1)) == datetime.date(2027, 2, 28)

    def test_none_when_the_cap_cannot_be_tracked(self):
        moment = datetime.datetime(2027, 3, 1)
        assert production_bonus.bonus_period_start(None, moment) is None
        assert production_bonus.bonus_period_start(ProductionBonusConfig(enabled=True), moment) is None  # no anchor
        assert production_bonus.bonus_period_start(self._cfg(**{"yearly cap kwh": None}), moment) is None
        assert production_bonus.bonus_period_start(ProductionBonusConfig(enabled=False, delivery_year_start=self.ANCHOR), moment) is None


class TestRemainingCapKwh:
    def test_cap_minus_exported_never_below_zero(self):
        assert production_bonus.remaining_cap_kwh(CFG, 4000.0) == pytest.approx(1000.0)
        assert production_bonus.remaining_cap_kwh(CFG, 5000.0) == 0.0
        assert production_bonus.remaining_cap_kwh(CFG, 6000.0) == 0.0

    def test_none_without_a_cap(self):
        assert production_bonus.remaining_cap_kwh(ProductionBonusConfig(enabled=True, **{"yearly cap kwh": None}), 1.0) is None
        assert production_bonus.remaining_cap_kwh(None, 1.0) is None


class TestDescribe:
    def test_mentions_mode_window_percentage_and_remaining(self):
        text = production_bonus.describe(CFG, datetime.datetime(2027, 1, 15, 12, 0), 1234.0)
        assert "pv only" in text
        assert "09:00-17:00" in text
        assert "30%" in text
        assert "0.060" in text
        assert "resterend 1234 kWh" in text

    def test_no_cap_and_untracked_cap_are_spelled_out(self):
        no_cap = ProductionBonusConfig(enabled=True, **{"yearly cap kwh": None})
        assert "geen limiet" in production_bonus.describe(no_cap, JUNE_NOON, None)
        assert "niet bijgehouden" in production_bonus.describe(CFG, JUNE_NOON, None)

    def test_disabled_is_short(self):
        assert production_bonus.describe(None, JUNE_NOON, None) == "Bonus teruglevering: uit"
