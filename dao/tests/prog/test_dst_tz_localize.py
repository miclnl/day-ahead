"""get_api_data's df["time"].dt.tz_localize(...) across a DST transition.

Every row here is one bucket per hour LABEL (from an SQL GROUP BY on the
hour string), not one row per actual wall-clock hour. On the autumn DST day
there is therefore no repeated 02:00 in the series for pandas to infer the
right UTC offset from, so ambiguous="infer" still raises; a fixed policy
(ambiguous=False) is required. Without any policy at all this endpoint
raised outright on both DST days, twice a year.
"""

import datetime

import pandas as pd
import pytest


def _localize(series: pd.Series) -> pd.Series:
    """The exact expression used in Report.get_api_data()."""
    return series.dt.tz_localize(
        "Europe/Amsterdam", ambiguous=False, nonexistent="shift_forward"
    )


def test_an_ordinary_day_localizes_normally():
    series = pd.Series(pd.date_range("2026-06-15 00:00", "2026-06-15 04:00", freq="h"))
    result = _localize(series)
    assert str(result.iloc[0].tzinfo) is not None
    assert list(result.dt.hour) == [0, 1, 2, 3, 4]


def test_the_autumn_dst_day_does_not_raise():
    """2026-10-25: the local hour 02:00-03:00 occurs twice in reality, but
    this series only has one row labelled "02:00"."""
    series = pd.Series(pd.date_range("2026-10-25 00:00", "2026-10-25 04:00", freq="h"))

    result = _localize(series)

    assert len(result) == 5
    assert not result.isna().any()


def test_the_spring_dst_day_does_not_raise():
    """2026-03-29: the local hour 02:00-03:00 does not exist at all."""
    series = pd.Series(pd.date_range("2026-03-29 00:00", "2026-03-29 04:00", freq="h"))

    result = _localize(series)

    assert len(result) == 5
    assert not result.isna().any()
    # Shifted forward into the one moment that does exist.
    assert result.iloc[2] == result.iloc[3]


def test_ambiguous_infer_alone_would_still_raise_on_this_data_shape():
    """Regression guard: confirms *why* ambiguous=False is needed instead of
    the more commonly recommended ambiguous="infer". If a future pandas
    version makes infer() succeed here, that is fine too, but a change back
    to "infer" must not silently reintroduce the crash for this data shape."""
    series = pd.Series(pd.date_range("2026-10-25 00:00", "2026-10-25 04:00", freq="h"))
    with pytest.raises(ValueError):
        series.dt.tz_localize("Europe/Amsterdam", ambiguous="infer")
