"""DaPrices.get_prices: the Nordpool path with the library replaced.

get_prices used to read sys.argv to decide whether a backfill was requested,
which broke under gunicorn, and its completeness check dereferenced
end_date=None exactly when Nordpool had returned partial data.
"""

import datetime
import math
import socket
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from dao.lib import da_prices  # noqa: E402


class FakeDb:
    def __init__(self, present=None, zone="Europe/Amsterdam"):
        self.present = present
        self.saved = []
        self.zone = zone

    @property
    def tzinfo(self):
        """The installation zone, independent of the market delivery calendar.

        get_prices compares a timestamp from the database against the
        requested range, and used to localize both as a hard-coded "CET" --
        wrong for anyone outside it.
        """
        return ZoneInfo(self.zone)

    def get_time_border_record(self, code, latest=True, table_name="values"):
        return self.present

    def savedata(self, df, tablename="values"):
        self.saved.append(df)


def _values(
    day: datetime.date,
    count: int,
    missing_hour: int | None = None,
    *,
    resolution=60,
):
    # The provider's CET/CEST calendar never follows the installation zone.
    tz = ZoneInfo("CET")
    start = datetime.datetime(day.year, day.month, day.day, tzinfo=tz).astimezone(
        datetime.UTC
    )
    rows = []
    for index in range(count):
        value = float("inf") if index == missing_hour else 100.0 + index
        at = start + datetime.timedelta(minutes=index * resolution)
        rows.append({
            "start": at,
            "end": at + datetime.timedelta(minutes=resolution),
            "value": value,
        })
    return rows


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Network access is forbidden in price ingestion tests")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def clock(monkeypatch):
    class FrozenDatetime(datetime.datetime):
        instant = datetime.datetime(2026, 6, 15, 7, tzinfo=datetime.UTC)

        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls.instant.replace(tzinfo=None)
            return cls.instant.astimezone(tz)

    monkeypatch.setattr(da_prices, "datetime", SimpleNamespace(
        datetime=FrozenDatetime,
        timedelta=datetime.timedelta,
        time=datetime.time,
    ))
    return FrozenDatetime


@pytest.fixture
def prices(monkeypatch, clock):
    config = SimpleNamespace(interval="1hour", prices=None, tibber=None)
    db = FakeDb()
    instance = da_prices.DaPrices(config, db, country="NL", secrets={})
    calls = []

    class FakeNordpool:
        payload = {"areas": {"NL": {"values": _values(datetime.date(2026, 6, 15), 24)}}}
        payloads = None

        def fetch(self, areas=None, end_date=None, resolution=60):
            calls.append({"areas": areas, "end_date": end_date, "resolution": resolution})
            if FakeNordpool.payloads is not None:
                day = end_date.date() if end_date is not None else datetime.date(2026, 6, 16)
                payload = FakeNordpool.payloads.get(day)
                if isinstance(payload, Exception):
                    raise payload
                return payload
            return FakeNordpool.payload

    monkeypatch.setattr(da_prices, "Prices", FakeNordpool)
    return instance, db, calls, FakeNordpool


def test_prices_are_stored_in_euro_per_kwh(prices):
    instance, db, calls, _ = prices
    instance.get_prices("nordpool")
    assert calls[0]["end_date"].date() == datetime.date(2026, 6, 15)
    assert calls[0]["resolution"] == 60
    assert len(db.saved) == 1
    frame = db.saved[0]
    assert len(frame) == 24
    assert frame["value"].iloc[0] == pytest.approx(0.100)
    assert frame["code"].unique().tolist() == ["da"]


def test_command_line_arguments_do_not_influence_the_fetch(prices, monkeypatch):
    instance, db, calls, _ = prices
    monkeypatch.setattr(sys, "argv", ["day_ahead.py", "debug", "prices", "junk"])
    instance.get_prices("nordpool")
    assert [call["end_date"].date() for call in calls] == [datetime.date(2026, 6, 15)]
    assert len(db.saved) == 1


def test_an_explicit_range_is_a_backfill(prices):
    instance, db, calls, _ = prices
    start = datetime.datetime(2026, 6, 15)
    instance.get_prices("nordpool", _start=start, _end=start + datetime.timedelta(days=1))
    assert calls[0]["end_date"].date() == start.date()
    assert len(db.saved) == 1


def test_nordpool_fetches_each_requested_day(prices):
    instance, db, calls, fake = prices
    days = [datetime.date(2026, 6, day) for day in (15, 16, 17)]
    fake.payloads = {
        day: {"areas": {"NL": {"values": _values(day, 24)}}} for day in days
    }

    instance.get_prices("nordpool", datetime.datetime(2026, 6, 15), datetime.datetime(2026, 6, 18))

    assert [call["end_date"].date() for call in calls] == days
    assert all(call["areas"] == ["NL"] and call["resolution"] == 60 for call in calls)
    frame = pd.concat(db.saved, ignore_index=True)
    assert len(frame) == frame["time"].nunique() == 72
    assert int(frame["time"].iloc[0]) == 1781474400  # 2026-06-14 22:00 UTC
    assert int(frame["time"].iloc[-1]) == 1781730000  # 2026-06-17 21:00 UTC


@pytest.mark.parametrize(("utc_hour", "zone", "day_count", "source_days", "first_epoch"), [
    (9, "Europe/Amsterdam", 1, (15,), 1781474400),
    (10, "Europe/Amsterdam", 2, (15, 16), 1781474400),
    (14, "America/New_York", 1, (15, 16), 1781496000),
    (16, "America/New_York", 2, (15, 16, 17), 1781496000),
    (8, "Europe/Helsinki", 1, (14, 15), 1781470800),
    (9, "Europe/Helsinki", 2, (14, 15, 16), 1781470800),
])
def test_default_fetches_today_and_after_local_noon_tomorrow(prices, clock, utc_hour, zone, day_count, source_days, first_epoch):
    instance, db, calls, fake = prices
    clock.instant = datetime.datetime(2026, 6, 15, utc_hour, tzinfo=datetime.UTC)
    db.zone = zone
    days = [datetime.date(2026, 6, day) for day in source_days]
    fake.payloads = {
        day: {"areas": {"NL": {"values": _values(day, 24)}}} for day in days
    }

    instance.get_prices("nordpool")

    assert [call["end_date"].date() for call in calls] == days
    epochs = pd.concat(db.saved)["time"].astype("int64")
    assert len(epochs) == epochs.nunique() == day_count * 24
    assert epochs.iloc[0] == first_epoch
    assert epochs.iloc[-1] == first_epoch + day_count * 86400 - 3600


@pytest.mark.parametrize(("month", "days", "counts"), [
    (3, (28, 29, 30), (24, 23, 24)),
    (10, (24, 25, 26), (24, 25, 24)),
])
@pytest.mark.parametrize(("interval", "resolution"), [("1hour", 60), ("15min", 15)])
def test_nordpool_range_preserves_dst_quarters(prices, caplog, month, days, counts, interval, resolution):
    instance, db, calls, fake = prices
    instance.interval = interval
    dates = [datetime.date(2026, month, day) for day in days]
    expected = [hours * (60 // resolution) for hours in counts]
    fake.payloads = {
        day: {"areas": {"NL": {"values": _values(day, count, resolution=resolution)}}}
        for day, count in zip(dates, expected, strict=True)
    }
    start = datetime.datetime(2026, month, days[0])
    end = datetime.datetime(2026, month, days[-1] + 1)

    instance.get_prices("nordpool", start, end)

    assert [call["end_date"].date() for call in calls] == dates
    assert all(call["resolution"] == resolution for call in calls)
    frame = pd.concat(db.saved, ignore_index=True)
    epochs = frame["time"].astype("int64")
    local_days = pd.to_datetime(epochs, unit="s", utc=True).dt.tz_convert(db.tzinfo).dt.date
    assert frame.groupby(local_days).size().tolist() == expected
    assert epochs.nunique() == sum(expected)
    assert epochs.diff().dropna().eq(resolution * 60).all()
    assert "incomplete" not in caplog.text


@pytest.mark.parametrize(("day", "hours", "first_utc", "end_utc"), [
    ("2026-06-15", 24, "2026-06-14T21:00:00+00:00", "2026-06-15T21:00:00+00:00"),
    ("2026-03-29", 23, "2026-03-28T22:00:00+00:00", "2026-03-29T21:00:00+00:00"),
    ("2026-10-25", 25, "2026-10-24T21:00:00+00:00", "2026-10-25T22:00:00+00:00"),
])
@pytest.mark.parametrize(("interval", "resolution"), [("1hour", 60), ("15min", 15)])
def test_nordpool_uses_source_days_not_installation_days(prices, caplog, day, hours, first_utc, end_utc, interval, resolution):
    instance, db, calls, fake = prices
    db.zone = "Europe/Helsinki"
    instance.interval = interval
    day = datetime.date.fromisoformat(day)
    source_days = [day - datetime.timedelta(days=1), day]
    slots = 60 // resolution
    fake.payloads = {
        date: {"areas": {"NL": {"values": _values(date, count * slots, resolution=resolution)}}}
        for date, count in zip(source_days, (24, hours), strict=True)
    }
    start = datetime.datetime.combine(day, datetime.time.min)
    end = datetime.datetime.combine(day + datetime.timedelta(days=1), datetime.time.min)

    instance.get_prices("nordpool", start, end)

    epochs = pd.concat(db.saved, ignore_index=True)["time"].astype("int64")
    assert len(epochs) == epochs.nunique() == hours * slots
    assert [call["end_date"].date() for call in calls] == source_days
    assert epochs.iloc[0] == int(datetime.datetime.fromisoformat(first_utc).timestamp())
    assert epochs.iloc[-1] == int(datetime.datetime.fromisoformat(end_utc).timestamp()) - resolution * 60
    assert epochs.diff().dropna().eq(resolution * 60).all()
    assert "incomplete" not in caplog.text


@pytest.mark.parametrize(("zone", "first_utc", "end_utc", "source_days"), [
    ("Europe/Amsterdam", "2026-06-14T22:30:00+00:00", "2026-06-15T00:00:00+00:00", (15,)),
    ("America/New_York", "2026-06-15T04:30:00+00:00", "2026-06-15T06:00:00+00:00", (15,)),
    ("Europe/Helsinki", "2026-06-14T21:30:00+00:00", "2026-06-14T23:00:00+00:00", (14, 15)),
])
def test_naive_range_uses_the_installation_zone(prices, caplog, zone, first_utc, end_utc, source_days):
    instance, db, calls, fake = prices
    instance.interval = "15min"
    db.zone = zone
    days = [datetime.date(2026, 6, day) for day in source_days]
    fake.payloads = {
        day: {"areas": {"NL": {"values": _values(day, 96, resolution=15)}}} for day in days
    }

    instance.get_prices("nordpool", datetime.datetime(2026, 6, 15, 0, 30), datetime.datetime(2026, 6, 15, 2))

    epochs = pd.concat(db.saved)["time"].astype("int64").tolist()
    assert len(epochs) == 6
    assert epochs[0] == int(datetime.datetime.fromisoformat(first_utc).timestamp())
    assert epochs[-1] == int(datetime.datetime.fromisoformat(end_utc).timestamp()) - 900
    assert [call["end_date"].date() for call in calls] == days
    assert "incomplete" not in caplog.text


@pytest.mark.parametrize("zone", ["Europe/Amsterdam", "Europe/Helsinki", "America/New_York"])
def test_aware_range_keeps_instants_and_filters_half_open_boundaries(prices, zone):
    instance, db, calls, fake = prices
    db.zone = zone
    days = [datetime.date(2026, 6, day) for day in (15, 16)]
    fake.payloads = {
        day: {"areas": {"NL": {"values": _values(day, 24)}}} for day in days
    }
    start = datetime.datetime(2026, 6, 14, 22, 30, tzinfo=datetime.UTC)
    end = datetime.datetime(2026, 6, 16, 0, tzinfo=datetime.UTC)

    instance.get_prices("nordpool", start, end)

    assert [call["end_date"].date() for call in calls] == days
    epochs = pd.concat(db.saved)["time"].astype("int64").tolist()
    assert len(epochs) == 25
    assert epochs[0] == int(datetime.datetime(2026, 6, 14, 23, tzinfo=datetime.UTC).timestamp())
    assert epochs[-1] == int(datetime.datetime(2026, 6, 15, 23, tzinfo=datetime.UTC).timestamp())
    assert all(start.timestamp() <= at < end.timestamp() for at in epochs)


def test_aware_fold_boundaries_keep_both_occurrences(prices):
    instance, db, calls, fake = prices
    instance.interval = "15min"
    fake.payload = {"areas": {"NL": {"values": _values(datetime.date(2026, 10, 25), 100, resolution=15)}}}
    start = datetime.datetime(2026, 10, 25, 2, 15, tzinfo=db.tzinfo, fold=0)
    end = datetime.datetime(2026, 10, 25, 2, 45, tzinfo=db.tzinfo, fold=1)

    instance.get_prices("nordpool", start, end)

    epochs = pd.concat(db.saved)["time"].astype("int64").tolist()
    assert epochs == [1792887300, 1792888200, 1792889100, 1792890000, 1792890900, 1792891800]
    assert len(calls) == 1


def test_empty_range_does_not_fetch_or_store(prices):
    instance, db, calls, _ = prices
    start = datetime.datetime(2026, 6, 15, 12)

    instance.get_prices("nordpool", start, start)

    assert calls == [] and db.saved == []


@pytest.mark.parametrize(("interval", "resolution", "count"), [("1hour", 60, 24), ("15min", 15, 96)])
def test_partial_day_does_not_hide_missing_prices(prices, caplog, interval, resolution, count):
    instance, db, calls, fake = prices
    instance.interval = interval
    start = datetime.datetime(2026, 6, 15)
    end = datetime.datetime(2026, 6, 16)
    values = _values(start.date(), count, resolution=resolution)
    missing = values.pop(5)
    db.present = values[-1]["start"].astimezone(db.tzinfo).replace(tzinfo=None)
    fake.payload = {"areas": {"NL": {"values": values}}}

    instance.get_prices("nordpool", start, end)

    frame = pd.concat(db.saved)
    assert len(frame) == count - 1
    assert str(int(missing["start"].timestamp())) not in frame["time"].tolist()
    assert frame["value"].ne(0).all()
    assert "incomplete" in caplog.text

    revised = _values(start.date(), count, resolution=resolution)
    revised[6]["value"] = 200.0
    fake.payload = {"areas": {"NL": {"values": revised}}}
    instance.get_prices("nordpool", start, end)

    assert len(calls) == 2
    assert len(db.saved[-1]) == count
    assert db.saved[-1]["value"].iloc[6] == pytest.approx(0.2)


@pytest.mark.parametrize("unavailable", [None, {"areas": {}}, {"areas": {"NL": {"values": []}}}, RuntimeError("nordpool down")])
def test_missing_day_remains_visible_and_other_requested_days_are_fetched(prices, caplog, unavailable):
    instance, db, calls, fake = prices
    days = [datetime.date(2026, 6, day) for day in (15, 16, 17)]
    fake.payloads = {
        day: {"areas": {"NL": {"values": _values(day, 24)}}} for day in days
    }
    fake.payloads[days[1]] = unavailable

    instance.get_prices("nordpool", datetime.datetime(2026, 6, 15), datetime.datetime(2026, 6, 18))

    assert [call["end_date"].date() for call in calls] == days
    frame = pd.concat(db.saved)
    assert len(frame) == 48 and frame["value"].ne(0).all()
    assert "2026-06-16" in caplog.text


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


@pytest.mark.parametrize("invalid", [None, math.nan, math.inf, -math.inf])
def test_infinite_values_are_skipped(prices, invalid):
    instance, db, calls, fake = prices
    values = _values(datetime.date(2026, 6, 15), 24)
    values[5]["value"] = invalid
    fake.payload = {"areas": {"NL": {"values": values}}}
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


class TestEntsoe:
    """The ENTSO-E branch melts a pandas Series of hourly prices into
    (time, code, value) rows; this used to be a df_db.loc[shape[0]] append
    per row."""

    def test_prices_are_stored_in_euro_per_kwh(self, monkeypatch):
        config = SimpleNamespace(
            interval="1hour",
            prices=SimpleNamespace(entsoe_api_key=None),
            tibber=None,
        )
        db = FakeDb()
        instance = da_prices.DaPrices(config, db, country="NL", secrets={})

        index = pd.date_range("2026-06-15", periods=24, freq="h", tz="CET")
        series = pd.Series([100.0 + h for h in range(24)], index=index)

        class FakeEntsoeClient:
            def __init__(self, api_key=None):
                pass

            def query_day_ahead_prices(self, country, start, end):
                return series

        monkeypatch.setattr(da_prices, "EntsoePandasClient", FakeEntsoeClient)

        start = datetime.datetime(2026, 6, 15)
        instance.get_prices(
            "entsoe", _start=start, _end=start + datetime.timedelta(days=1)
        )

        assert len(db.saved) == 1
        frame = db.saved[0]
        assert len(frame) == 24
        assert frame["code"].unique().tolist() == ["da"]
        assert frame["value"].iloc[0] == pytest.approx(0.100)


class TestEasyEnergy:
    """The EasyEnergy branch parses a JSON list of {Timestamp, TariffReturn}
    records into (time, code, value) rows."""

    def test_prices_are_stored(self, monkeypatch):
        config = SimpleNamespace(interval="1hour", prices=None, tibber=None)
        db = FakeDb()
        instance = da_prices.DaPrices(config, db, country="NL", secrets={})

        payload = [
            {
                "Timestamp": f"2026-06-15T{hour:02d}:00:00+02:00",
                "TariffReturn": 0.10 + hour * 0.01,
            }
            for hour in range(24)
        ]

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        monkeypatch.setattr(da_prices, "get", lambda *a, **kw: FakeResponse())

        start = datetime.datetime(2026, 6, 15)
        instance.get_prices(
            "easyenergy", _start=start, _end=start + datetime.timedelta(days=1)
        )

        assert len(db.saved) == 1
        frame = db.saved[0]
        assert len(frame) == 24
        assert frame["code"].unique().tolist() == ["da"]
        assert frame["value"].iloc[0] == pytest.approx(0.10)
        assert frame["value"].iloc[-1] == pytest.approx(0.33)


class TestTibber:
    """The Tibber branch concatenates today/tomorrow/range price nodes into
    (time, code, value) rows."""

    def test_prices_from_all_three_node_lists_are_stored(self, monkeypatch):
        config = SimpleNamespace(
            interval="1hour",
            prices=None,
            tibber=SimpleNamespace(
                api_token=SimpleNamespace(resolve=lambda secrets: "tok"),
                api_url=None,
            ),
        )
        db = FakeDb()
        instance = da_prices.DaPrices(config, db, country="NL", secrets={})

        def node(hour, energy):
            return {
                "startsAt": f"2026-06-15T{hour:02d}:00:00.000+02:00",
                "energy": energy,
            }

        payload = {
            "data": {
                "viewer": {
                    "homes": [
                        {
                            "currentSubscription": {
                                "priceInfo": {
                                    "today": [node(0, 0.10), node(1, 0.11)],
                                    "tomorrow": [node(2, 0.12)],
                                },
                                "priceInfoRange": {"nodes": [node(3, 0.13)]},
                            }
                        }
                    ]
                }
            }
        }

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        monkeypatch.setattr(da_prices, "post", lambda *a, **kw: FakeResponse())

        start = datetime.datetime(2026, 6, 15)
        instance.get_prices(
            "tibber", _start=start, _end=start + datetime.timedelta(days=1)
        )

        assert len(db.saved) == 1
        frame = db.saved[0]
        assert len(frame) == 4
        assert frame["code"].unique().tolist() == ["da"]
        assert sorted(frame["value"].tolist()) == pytest.approx(
            [0.10, 0.11, 0.12, 0.13]
        )
