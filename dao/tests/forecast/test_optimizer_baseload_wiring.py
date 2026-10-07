"""The optimizer's baseload horizon goes through one extracted helper."""

from __future__ import annotations

import datetime as dt

import dao.prog.day_ahead as day_ahead

TZ = "Europe/Amsterdam"


class _StubBaseloadService:
    """Records every call and answers with a fixed, repeating profile."""

    def __init__(self, values):
        self.values = values
        self.calls: list[tuple] = []

    def forecast_for_optimizer(self, start_interval, intervals, interval):
        self.calls.append((start_interval, intervals, interval))
        return list(self.values[:intervals])


def _bare_da_calc(*, use_calc_baseload, interval="1hour", steps_day=24, baseload=None):
    """A DaCalc with only the attributes ``_baseload_for_horizon`` touches."""
    instance = day_ahead.DaCalc.__new__(day_ahead.DaCalc)
    instance.use_calc_baseload = use_calc_baseload
    instance.time_zone = TZ
    instance.interval = interval
    instance.steps_day = steps_day
    instance.config = type("Config", (), {"baseload": baseload})()
    return instance


def test_calc_optimum_baseload_block_uses_service(monkeypatch):
    instance = _bare_da_calc(use_calc_baseload=True, interval="1hour")
    stub = _StubBaseloadService([0.4] * 48)
    monkeypatch.setattr(instance, "baseload_service", lambda: stub)

    start = dt.datetime(2026, 3, 2, 13, 0)  # naive local, as calc_optimum builds it
    intervals = 10
    result = instance._baseload_for_horizon(start, intervals)

    assert len(result) == intervals
    assert result == [0.4] * intervals
    assert len(stub.calls) == 1
    called_start, called_intervals, called_interval = stub.calls[0]
    assert called_start.tzinfo is not None
    assert called_start.replace(tzinfo=None) == start
    assert called_intervals == intervals
    assert called_interval == "1hour"


def test_baseload_for_horizon_passes_through_an_already_aware_start(monkeypatch):
    instance = _bare_da_calc(use_calc_baseload=True, interval="15min")
    stub = _StubBaseloadService([0.1] * 40)
    monkeypatch.setattr(instance, "baseload_service", lambda: stub)

    from zoneinfo import ZoneInfo

    start = dt.datetime(2026, 3, 2, 13, 0, tzinfo=ZoneInfo(TZ))
    instance._baseload_for_horizon(start, 8)

    called_start, called_intervals, called_interval = stub.calls[0]
    assert called_start is start
    assert called_interval == "15min"


def test_baseload_for_horizon_uses_static_list_when_not_calculated():
    instance = _bare_da_calc(
        use_calc_baseload=False,
        interval="1hour",
        steps_day=24,
        baseload=[float(h) for h in range(24)],
    )
    start = dt.datetime(2026, 3, 2, 22, 0)  # late evening, needs padding
    result = instance._baseload_for_horizon(start, 5)
    # Starting at hour 22, the static list only has hours 22 and 23 left;
    # the last value repeats for the rest, exactly as before this change.
    assert result == [22.0, 23.0, 23.0, 23.0, 23.0]
