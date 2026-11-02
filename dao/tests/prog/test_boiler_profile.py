"""A boiler start is usable only when its full energy fits the real UTC grid."""

import datetime as dt
from importlib.util import find_spec
from zoneinfo import ZoneInfo

import pytest


def allocate(**kwargs):
    assert find_spec("dao.prog.boiler_profile") is not None, "full boiler profile helper is missing"
    from dao.prog.boiler_profile import boiler_profile

    return boiler_profile(**kwargs)


@pytest.mark.parametrize("intervals,start,required,expected", [
    pytest.param(tuple((u * 900, (u + 1) * 900) for u in range(12)), 0,
                 0.7798611111111111, (0.075,) * 10 + (0.0298611111111111, 0), id="long-quarter"),
    pytest.param(((0, 3600), (3600, 7200), (7200, 10800)), 0,
                 0.7, (0.3, 0.3, 0.1), id="hours"),
    pytest.param(((0, 900), (900, 1800), (1800, 2700), (2700, 3600)), 1,
                 0.1, (0, 0.075, 0.025, 0), id="later-start"),
    pytest.param(((0, 900), (900, 1800)), 1, 0, (0, 0), id="zero-demand"),
])
def test_boiler_profile_preserves_total_and_partial_tail(intervals, start, required, expected):
    """Truncating, padding or shifting a profile loses energy or invents a tail."""
    result = allocate(intervals=intervals, start_index=start, required_kwh=required, power_w=300)
    assert isinstance(result, tuple)
    assert result == pytest.approx(expected)
    assert sum(result) == pytest.approx(required)
    assert all(0 <= value <= 300 * (end - begin) / 3_600_000
               for value, (begin, end) in zip(result, intervals, strict=True))


def test_boiler_profile_uses_actual_first_interval():
    """A 450-second first slot holds 0.0375 kWh, not a nominal quarter's 0.075."""
    result = allocate(intervals=((450, 900), (900, 1800), (1800, 2700)),
                      start_index=0, required_kwh=0.1, power_w=300)
    assert result == pytest.approx((0.0375, 0.0625, 0))


def test_boiler_profile_accepts_exact_horizon_end():
    """Decimal-exact capacity needs no extra nominal interval or epsilon."""
    result = allocate(intervals=tuple((u * 900, (u + 1) * 900) for u in range(12)),
                      start_index=0, required_kwh=0.9, power_w=300)
    assert result == (0.075,) * 12
    assert sum(result) == pytest.approx(0.9)


@pytest.mark.parametrize("required", [0.900001, 0.9000000000000001])
def test_boiler_profile_rejects_incomplete_horizon(required):
    """Neither an epsilon nor repeating the last interval may hide a shortfall."""
    assert allocate(intervals=tuple((u * 900, (u + 1) * 900) for u in range(12)),
                    start_index=0, required_kwh=required, power_w=300) is None


@pytest.mark.parametrize("day,offsets", [
    ("2027-03-28", (3600, 7200, 7200, 7200)),
    ("2027-10-31", (7200, 3600, 3600, 3600)),
])
def test_boiler_profile_uses_utc_across_dst(day, offsets):
    """Skipped or repeated wall-clock hours must not change UTC capacity."""
    start = int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.UTC).timestamp())
    intervals = tuple((start + u * 3600, start + (u + 1) * 3600) for u in range(4))
    assert tuple(int(dt.datetime.fromtimestamp(a, ZoneInfo("Europe/Amsterdam"))
                     .utcoffset().total_seconds()) for a, _ in intervals) == offsets
    assert allocate(intervals=intervals, start_index=1, required_kwh=0.7,
                    power_w=300) == pytest.approx((0, 0.3, 0.3, 0.1))


@pytest.mark.parametrize("changes", [
    {"intervals": ()},
    {"intervals": 1},
    {"intervals": ((0,),)},
    {"intervals": ((0, 0),)},
    {"intervals": ((900, 0),)},
    {"intervals": ((0, 900), (901, 1800))},
    {"intervals": ((0, 900), (899, 1800))},
    {"intervals": ((900, 1800), (0, 900))},
    {"intervals": ((0, float("nan")),)},
    {"intervals": ((0, float("inf")),)},
    {"intervals": ((0, 900.5),)},
    {"start_index": -1},
    {"start_index": 1},
    {"start_index": 0.5},
    {"required_kwh": -0.1},
    {"required_kwh": float("nan")},
    {"required_kwh": float("inf")},
    {"required_kwh": -float("inf")},
    {"power_w": 0},
    {"power_w": -300},
    {"power_w": float("nan")},
    {"power_w": float("inf")},
    {"power_w": -float("inf")},
])
@pytest.mark.parametrize("required", [0, 0.1], ids=["zero-demand", "heat-demand"])
def test_boiler_profile_rejects_invalid_input(changes, required):
    """Bad grids/inputs are errors even when demand is zero; never normalize them."""
    kwargs = dict(intervals=((0, 900),), start_index=0, required_kwh=required, power_w=300)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        allocate(**kwargs)
