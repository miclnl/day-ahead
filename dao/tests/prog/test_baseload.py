"""Tests for the robust baseload estimation.

The estimator works from roughly eight observations per weekday and hour, so
these tests are mostly about what happens when one of those eight is wrong.
"""

import datetime

import pytest

from dao.prog.baseload import (
    BaseloadOptions,
    Sample,
    build_profile,
    dutch_holidays,
    effective_weekday,
    estimate_cell,
    is_holiday,
    iter_samples,
    outlier_mask,
    profile_age_days,
    profile_from_file,
    profile_to_dict,
    quantile,
    recency_weights,
    trimmed,
    weighted_mean,
    weighted_median,
)


def cell(*values, ages=None):
    ages = ages if ages is not None else [0.0] * len(values)
    return [Sample(age_days=a, value=v) for a, v in zip(ages, values)]


class TestQuantile:
    def test_endpoints_and_middle(self):
        data = [1.0, 2.0, 3.0, 4.0]
        assert quantile(data, 0.0) == 1.0
        assert quantile(data, 1.0) == 4.0
        assert quantile(data, 0.5) == pytest.approx(2.5)

    def test_single_value(self):
        assert quantile([7.0], 0.25) == 7.0

    def test_empty(self):
        assert quantile([], 0.5) == 0.0


class TestOutlierMask:
    def test_a_party_is_rejected(self):
        values = [0.3, 0.32, 0.29, 0.31, 0.30, 0.33, 4.0]
        mask = outlier_mask(values, 2.0)
        assert mask[-1] is False
        assert all(mask[:-1])

    def test_normal_variation_is_kept(self):
        values = [0.3, 0.5, 0.4, 0.6, 0.35, 0.55]
        assert all(outlier_mask(values, 2.0))

    def test_too_few_samples_disables_the_filter(self):
        values = [0.3, 0.3, 99.0]
        assert all(outlier_mask(values, 2.0))

    def test_identical_values_are_all_kept(self):
        assert all(outlier_mask([1.0] * 8, 2.0))

    def test_it_never_rejects_everything(self):
        values = [1.0, 1.0, 1.0, 1.0, 50.0, 100.0, 200.0]
        assert any(outlier_mask(values, 0.01))


class TestWeights:
    def test_recency_halves_at_the_half_life(self):
        weights = recency_weights([0.0, 28.0, 56.0], 28.0)
        assert weights == pytest.approx([1.0, 0.5, 0.25])

    def test_disabled_weighting_is_uniform(self):
        assert recency_weights([0.0, 100.0], None) == [1.0, 1.0]
        assert recency_weights([0.0, 100.0], 0) == [1.0, 1.0]

    def test_weighted_median_follows_the_weight(self):
        # The high value carries almost all of the weight.
        assert weighted_median([1.0, 10.0], [0.01, 1.0]) == 10.0
        assert weighted_median([1.0, 10.0], [1.0, 0.01]) == 1.0

    def test_weighted_median_of_a_symmetric_set(self):
        assert weighted_median([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == 2.0

    def test_weighted_mean(self):
        assert weighted_mean([1.0, 3.0], [1.0, 3.0]) == pytest.approx(2.5)

    def test_weighted_mean_without_weight_falls_back(self):
        assert weighted_mean([2.0, 4.0], [0.0, 0.0]) == pytest.approx(3.0)

    def test_trimmed_drops_both_tails(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        kept, _ = trimmed(values, [1.0] * 5, 0.2)
        assert 100.0 not in kept
        assert 1.0 not in kept

    def test_trimmed_keeps_everything_when_the_set_is_small(self):
        kept, _ = trimmed([1.0, 2.0], [1.0, 1.0], 0.2)
        assert kept == [1.0, 2.0]


class TestEstimateCell:
    OPTIONS = BaseloadOptions(half_life_days=None)

    def test_median_is_unmoved_by_one_bad_day(self):
        """The whole point of the change.

        Eight quiet evenings and one party. The mean is dragged up by almost
        half a kWh and stays there for two months; the median is not.
        """
        samples = cell(0.30, 0.31, 0.29, 0.32, 0.30, 0.31, 0.30, 4.00)
        median, _ = estimate_cell(samples, self.OPTIONS)
        mean, _ = estimate_cell(
            samples,
            BaseloadOptions(
                aggregate="mean", remove_outliers=False, half_life_days=None
            ),
        )
        assert median == pytest.approx(0.30, abs=0.02)
        assert mean > 0.7

    def test_outlier_rejection_also_rescues_the_mean(self):
        samples = cell(0.30, 0.31, 0.29, 0.32, 0.30, 0.31, 0.30, 4.00)
        value, count = estimate_cell(
            samples, BaseloadOptions(aggregate="mean", half_life_days=None)
        )
        assert value == pytest.approx(0.30, abs=0.02)
        assert count == 7

    def test_recency_weighting_tracks_a_step_change(self):
        """A housemate leaves: the recent weeks must win."""
        old = cell(1.0, 1.0, 1.0, 1.0, ages=[56, 49, 42, 35])
        new = cell(0.5, 0.5, 0.5, 0.5, ages=[21, 14, 7, 0])
        weighted, _ = estimate_cell(
            old + new, BaseloadOptions(remove_outliers=False, half_life_days=14.0)
        )
        flat, _ = estimate_cell(
            old + new, BaseloadOptions(remove_outliers=False, half_life_days=None)
        )
        assert weighted == 0.5
        assert flat in (0.5, 1.0)

    def test_negative_values_are_clipped(self):
        # A recorder gap in the grid meter with the PV meter intact.
        samples = cell(-2.0, -2.1, -1.9, -2.0, -2.0)
        value, _ = estimate_cell(samples, self.OPTIONS)
        assert value == 0.0

    def test_clipping_can_be_switched_off(self):
        samples = cell(-2.0, -2.0, -2.0)
        value, _ = estimate_cell(
            samples, BaseloadOptions(clip_negative=False, half_life_days=None)
        )
        assert value < 0

    def test_too_few_samples_returns_none(self):
        value, count = estimate_cell(cell(1.0, 1.0), BaseloadOptions(min_samples=3))
        assert value is None
        assert count == 2

    def test_an_empty_cell_returns_none(self):
        assert estimate_cell([], self.OPTIONS) == (None, 0)

    def test_trimmed_aggregate(self):
        samples = cell(1.0, 2.0, 3.0, 4.0, 100.0)
        value, _ = estimate_cell(
            samples,
            BaseloadOptions(
                aggregate="trimmed", remove_outliers=False, half_life_days=None
            ),
        )
        assert value == pytest.approx(3.0)


class TestBuildProfile:
    def test_a_full_profile_is_produced(self):
        cells = {h: cell(*(0.1 * h + i * 0.01 for i in range(8))) for h in range(24)}
        profile = build_profile(cells, None, BaseloadOptions())
        assert len(profile.values) == 24
        assert all(v >= 0 for v in profile.values)
        assert profile.samples[0] == 8

    def test_a_thin_hour_borrows_from_the_pool(self):
        cells = {5: cell(1.0)}  # one observation, below min_samples
        pooled = {5: cell(*(2.0 for _ in range(10)))}
        profile = build_profile(cells, pooled, BaseloadOptions(min_samples=3))
        assert profile.values[5] == pytest.approx(2.0)
        assert profile.pooled[5] is True

    def test_without_a_pool_a_thin_hour_becomes_zero(self):
        profile = build_profile({5: cell(1.0)}, None, BaseloadOptions(min_samples=3))
        assert profile.values[5] == 0.0
        assert profile.pooled[5] is False

    def test_an_empty_input_yields_a_flat_zero_profile(self):
        profile = build_profile({}, None, BaseloadOptions())
        assert profile.values == [0.0] * 24
        assert profile.total == 0.0


class TestHolidays:
    def test_the_fixed_dates(self):
        holidays = dutch_holidays(2026)
        assert datetime.date(2026, 1, 1) in holidays
        assert datetime.date(2026, 4, 27) in holidays
        assert datetime.date(2026, 12, 25) in holidays
        assert datetime.date(2026, 12, 26) in holidays

    def test_the_easter_derived_dates(self):
        # Easter 2026 is 5 April, so Easter Monday is the 6th.
        holidays = dutch_holidays(2026)
        assert datetime.date(2026, 4, 6) in holidays
        assert datetime.date(2026, 5, 14) in holidays  # hemelvaart
        assert datetime.date(2026, 5, 25) in holidays  # tweede pinksterdag

    def test_an_ordinary_day_is_not_a_holiday(self):
        assert not is_holiday(datetime.date(2026, 3, 17))

    def test_a_holiday_is_folded_into_sunday(self):
        christmas = datetime.date(2026, 12, 25)  # a Friday
        assert christmas.weekday() == 4
        assert effective_weekday(christmas, "sunday") == 6

    def test_it_can_be_folded_into_saturday_instead(self):
        assert effective_weekday(datetime.date(2026, 12, 25), "saturday") == 5

    def test_it_can_be_ignored(self):
        assert effective_weekday(datetime.date(2026, 12, 25), "ignore") == 4

    def test_ordinary_days_are_untouched(self):
        monday = datetime.date(2026, 3, 16)
        assert effective_weekday(monday, "sunday") == 0


class TestIterSamples:
    def test_rows_are_grouped_by_weekday_and_hour(self):
        reference = datetime.datetime(2026, 3, 20, 0, 0)
        rows = [
            (datetime.datetime(2026, 3, 16, 9, 0), 1.0),  # Monday
            (datetime.datetime(2026, 3, 17, 9, 0), 2.0),  # Tuesday
            (datetime.datetime(2026, 3, 16, 10, 0), 3.0),
        ]
        grouped = iter_samples(rows, reference)
        assert grouped[0][9][0].value == 1.0
        assert grouped[1][9][0].value == 2.0
        assert grouped[0][10][0].value == 3.0

    def test_the_age_is_measured_from_the_reference(self):
        reference = datetime.datetime(2026, 3, 20, 0, 0)
        rows = [(datetime.datetime(2026, 3, 13, 0, 0), 1.0)]
        grouped = iter_samples(rows, reference)
        assert grouped[4][0][0].age_days == pytest.approx(7.0)

    def test_none_and_nan_are_dropped(self):
        reference = datetime.datetime(2026, 3, 20)
        rows = [
            (datetime.datetime(2026, 3, 16, 9, 0), None),
            (datetime.datetime(2026, 3, 16, 9, 0), float("nan")),
            (datetime.datetime(2026, 3, 16, 9, 0), 1.0),
        ]
        grouped = iter_samples(rows, reference)
        assert len(grouped[0][9]) == 1

    def test_a_holiday_lands_in_the_sunday_bucket(self):
        reference = datetime.datetime(2026, 12, 31)
        rows = [(datetime.datetime(2026, 12, 25, 9, 0), 5.0)]
        grouped = iter_samples(rows, reference, "sunday")
        assert 6 in grouped and 4 not in grouped


class TestFileFormat:
    def test_the_new_format_roundtrips(self):
        profile = build_profile(
            {h: cell(1.0, 1.0, 1.0, 1.0) for h in range(24)},
            None,
            BaseloadOptions(),
        )
        payload = profile_to_dict(profile, 3, 56, BaseloadOptions())
        assert payload["weekday"] == 3
        assert payload["period_days"] == 56
        assert profile_from_file(payload) == profile.values

    def test_the_old_bare_list_is_still_accepted(self):
        values = [0.5] * 24
        assert profile_from_file(values) == values

    def test_a_wrong_length_is_refused(self):
        with pytest.raises(ValueError, match="24 waarden"):
            profile_from_file([1.0, 2.0])

    def test_nan_is_refused(self):
        with pytest.raises(ValueError, match="NaN"):
            profile_from_file([float("nan")] * 24)

    def test_garbage_is_refused(self):
        with pytest.raises(ValueError, match="ongeldige"):
            profile_from_file(["x"] * 24)

    def test_negative_values_are_clipped_on_read(self):
        assert profile_from_file([-1.0] * 24) == [0.0] * 24

    def test_the_age_is_readable(self):
        payload = profile_to_dict(build_profile({}, None), 0, 56, BaseloadOptions())
        assert profile_age_days(payload) == pytest.approx(0.0, abs=0.01)

    def test_the_old_format_has_no_age(self):
        assert profile_age_days([0.0] * 24) is None

    def test_a_corrupt_timestamp_has_no_age(self):
        assert profile_age_days({"created": "not a date"}) is None
