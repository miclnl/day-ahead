"""Localising hour buckets across a DST transition, twice a year.

Both call sites in da_report build one row per hour LABEL -- the API data
from an SQL GROUP BY on the hour string, the solar report from a loop that
adds an hour at a time to local midnight. Neither produces a repeated 02:00
on the autumn DST day, so ambiguous="infer" still raises; a fixed policy
(ambiguous=False) is required. And neither skips the 02:00 that does not
exist on the spring day, so nonexistent="shift_forward" is required too.

Without both, every report covering one of those two days raised outright.
These tests exercise the production helper, not a copy of its expression:
the two call sites used to carry the policy separately and the solar
report's copy was missing it entirely.
"""

import pandas as pd
import pytest

from dao.prog.da_report import localize_hour_buckets

TZ = "Europe/Amsterdam"

#: The hour that exists twice (autumn) and the hour that does not exist
#: (spring), in the first year after this code was written.
AUTUMN_DST_DAY = "2026-10-25"
SPRING_DST_DAY = "2026-03-29"


def hour_labels(day: str) -> pd.DatetimeIndex:
    """Midnight to 04:00 of ``day``, one row per hour label."""
    return pd.date_range(f"{day} 00:00", f"{day} 04:00", freq="h")


def test_an_ordinary_day_localizes_normally():
    result = localize_hour_buckets(hour_labels("2026-06-15"), TZ)

    assert result.tz is not None
    assert list(result.hour) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("day", [AUTUMN_DST_DAY, SPRING_DST_DAY])
def test_a_datetime_index_survives_both_dst_days(day):
    """The shape the solar report hands it: result.index."""
    result = localize_hour_buckets(hour_labels(day), TZ)

    assert len(result) == 5
    assert not result.isna().any()
    assert result.tz is not None


@pytest.mark.parametrize("day", [AUTUMN_DST_DAY, SPRING_DST_DAY])
def test_a_series_survives_both_dst_days(day):
    """The shape get_api_data hands it: df["time"]."""
    result = localize_hour_buckets(pd.Series(hour_labels(day)), TZ)

    assert len(result) == 5
    assert not result.isna().any()
    assert result.dt.tz is not None


def test_the_spring_hour_that_does_not_exist_shifts_forward():
    result = localize_hour_buckets(hour_labels(SPRING_DST_DAY), TZ)

    assert result[2] == result[3]


def test_ambiguous_infer_alone_would_still_raise_on_this_data_shape():
    """Regression guard: confirms *why* ambiguous=False is needed instead of
    the more commonly recommended ambiguous="infer". If a future pandas
    version makes infer() succeed here, that is fine too, but a change back
    to "infer" must not silently reintroduce the crash for this data shape."""
    with pytest.raises(ValueError):
        hour_labels(AUTUMN_DST_DAY).tz_localize(TZ, ambiguous="infer")
