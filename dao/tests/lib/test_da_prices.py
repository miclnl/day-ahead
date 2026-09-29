"""DaPrices.get_prices: the Nordpool path with the library replaced.

get_prices used to read sys.argv to decide whether a backfill was requested,
which broke under gunicorn, and its completeness check dereferenced
end_date=None exactly when Nordpool had returned partial data.
"""

import datetime
import math
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")

from dao.lib import da_prices  # noqa: E402


class FakeDb:
    def __init__(self, present=None):
        self.present = present
        self.saved = []

    def get_time_border_record(self, code, latest=True, table_name="values"):
        return self.present

    def savedata(self, df, tablename="values"):
        self.saved.append(df)


def _values(day: datetime.date, count: int, missing_hour: int | None = None):
    tz = datetime.timezone(datetime.timedelta(hours=2))
    start = datetime.datetime(day.year, day.month, day.day, tzinfo=tz)
    rows = []
    for hour in range(count):
        value = float("inf") if hour == missing_hour else 100.0 + hour
        rows.append({"start": start + datetime.timedelta(hours=hour), "value": value})
    return rows


@pytest.fixture
def prices(monkeypatch):
    config = SimpleNamespace(interval="1hour", prices=None, tibber=None)
    db = FakeDb()
    instance = da_prices.DaPrices(config, db, country="NL", secrets={})
    calls = []

    class FakeNordpool:
        payload = {"areas": {"NL": {"values": _values(datetime.date(2026, 6, 15), 24)}}}

        def fetch(self, areas=None, end_date=None, resolution=60):
            calls.append({"areas": areas, "end_date": end_date, "resolution": resolution})
            return FakeNordpool.payload

    monkeypatch.setattr(da_prices, "Prices", FakeNordpool)
    return instance, db, calls, FakeNordpool


def test_prices_are_stored_in_euro_per_kwh(prices):
    instance, db, calls, _ = prices
    instance.get_prices("nordpool")
    assert calls[0]["end_date"] is None and calls[0]["resolution"] == 60
    assert len(db.saved) == 1
    frame = db.saved[0]
    assert len(frame) == 24
    assert frame["value"].iloc[0] == pytest.approx(0.100)
    assert frame["code"].unique().tolist() == ["da"]


def test_command_line_arguments_do_not_influence_the_fetch(prices, monkeypatch):
    instance, db, calls, _ = prices
    monkeypatch.setattr(sys, "argv", ["day_ahead.py", "debug", "prices", "junk"])
    instance.get_prices("nordpool")
    assert calls[0]["end_date"] is None
    assert len(db.saved) == 1


def test_an_explicit_range_is_a_backfill(prices):
    instance, db, calls, _ = prices
    start = datetime.datetime(2026, 6, 15)
    instance.get_prices("nordpool", _start=start, _end=start + datetime.timedelta(days=1))
    assert calls[0]["end_date"] == start
    assert len(db.saved) == 1


def test_already_present_prices_are_not_fetched_again(prices):
    instance, db, calls, _ = prices
    db.present = datetime.datetime(2100, 1, 1, 23)
    instance.get_prices("nordpool")
    assert calls == [] and db.saved == []


def test_partial_data_is_stored_with_a_warning(prices, caplog):
    instance, db, calls, fake = prices
    fake.payload = {"areas": {"NL": {"values": _values(datetime.date(2026, 6, 15), 10)}}}
    instance.get_prices("nordpool")
    assert len(db.saved) == 1 and len(db.saved[0]) == 10
    assert "incomplete" in caplog.text


def test_infinite_values_are_skipped(prices):
    instance, db, calls, fake = prices
    fake.payload = {"areas": {"NL": {"values": _values(datetime.date(2026, 6, 15), 24, missing_hour=5)}}}
    instance.get_prices("nordpool")
    frame = db.saved[0]
    assert len(frame) == 23
    assert all(math.isfinite(v) for v in frame["value"])


def test_a_failing_fetch_is_logged_not_raised(prices, caplog):
    instance, db, calls, fake = prices

    def boom(self, **kwargs):
        raise RuntimeError("nordpool down")

    fake.fetch = boom
    instance.get_prices("nordpool")
    assert db.saved == []
    assert "nordpool down" in caplog.text


def test_an_unexpected_payload_is_logged_not_raised(prices, caplog):
    instance, db, calls, fake = prices
    fake.payload = {"areas": {}}
    instance.get_prices("nordpool")
    assert db.saved == []
    assert "Onverwacht antwoord" in caplog.text
