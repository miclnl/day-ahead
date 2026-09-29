"""Report.generate_df: the bucket skeleton every report is built on.

Rewritten from a row-by-row .loc[shape[0]] append (O(n^2), 8760 rows for an
hourly "dit jaar" report) to building the frame once from a list of tuples.
These tests pin down the exact shape and values so that rewrite could not
silently change behaviour.
"""

import datetime

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from dao.prog.da_report import Report  # noqa: E402


def test_hourly_buckets():
    start = datetime.datetime(2026, 3, 15, 10, 0)
    result = Report.generate_df(start, start + datetime.timedelta(hours=3), "uur")

    assert list(result.columns) == ["uur", "tijd", "tot", "datasoort"]
    assert len(result) == 3
    assert list(result["uur"]) == [" 10:00", " 11:00", " 12:00"]  # str(datetime)[10:16]
    assert list(result["tijd"]) == [
        datetime.datetime(2026, 3, 15, 10, 0),
        datetime.datetime(2026, 3, 15, 11, 0),
        datetime.datetime(2026, 3, 15, 12, 0),
    ]
    assert list(result["tot"]) == [
        datetime.datetime(2026, 3, 15, 11, 0),
        datetime.datetime(2026, 3, 15, 12, 0),
        datetime.datetime(2026, 3, 15, 13, 0),
    ]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert list(result.index) == list(pd.to_datetime(result["tijd"]))


def test_daily_buckets():
    start = datetime.datetime(2026, 3, 15)
    result = Report.generate_df(start, start + datetime.timedelta(days=2), "dag")

    assert list(result["dag"]) == ["2026-03-15", "2026-03-16"]
    assert list(result["tijd"]) == [
        datetime.datetime(2026, 3, 15),
        datetime.datetime(2026, 3, 16),
    ]


def test_monthly_buckets_use_the_first_of_the_month():
    start = datetime.datetime(2026, 1, 15)
    result = Report.generate_df(
        start, start + datetime.timedelta(days=95), "maand", get_interval="maand"
    )

    assert list(result["maand"]) == ["2026-01", "2026-02", "2026-03", "2026-04"]
    # old_moment is pinned to the 1st of the month even though the period
    # itself starts on the 15th.
    assert result["tijd"].iloc[0] == datetime.datetime(2026, 1, 1)
    assert result["tijd"].iloc[1] == datetime.datetime(2026, 2, 1)


def test_an_empty_range_yields_an_empty_frame_with_the_right_columns():
    start = datetime.datetime(2026, 3, 15)
    result = Report.generate_df(start, start, "uur")

    assert len(result) == 0
    assert list(result.columns) == ["uur", "tijd", "tot", "datasoort"]


def test_a_column_argument_adds_a_zero_filled_column():
    start = datetime.datetime(2026, 3, 15)
    result = Report.generate_df(start, start + datetime.timedelta(hours=2), "uur", column="cons")

    assert list(result["cons"]) == [0.0, 0.0]


def test_datasoort_switches_from_recorded_to_expected_around_now(monkeypatch):
    fixed_now = datetime.datetime(2026, 3, 15, 12, 30)

    class FrozenDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(datetime, "datetime", FrozenDatetime)
    start = datetime.datetime(2026, 3, 15, 11, 0)
    result = Report.generate_df(start, start + datetime.timedelta(hours=3), "uur")

    # Only the bucket ending exactly at 12:00 (<= the rounded-down now) is
    # "recorded"; both later buckets are "expected".
    assert list(result["datasoort"]) == ["recorded", "expected", "expected"]
