"""Exact component accounting and partitioned presentation of one calculation."""

from dataclasses import asdict
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from zoneinfo import ZoneInfo

import pytest

from dao.prog.settlement import engine
from dao.prog.settlement.types import BonusState, InputInterval

ZONE = ZoneInfo("Europe/Amsterdam")


def ts(day, hour=0):
    return int(datetime.fromisoformat(day).replace(hour=hour, tzinfo=ZONE).timestamp())


def row(meter, start, spot="0.10", **overrides):
    return InputInterval(
        **asdict(meter(**{"start_ts": start, "end_ts": start + 900,
                         "import_kwh": Decimal("0"), "export_kwh": Decimal("0"), **overrides})),
        spot_ex_vat=None if spot is None else Decimal(spot),
        spot_source=None if spot is None else "synthetic-hour-price",
        input_hash="b" * 64, meter_id="synthetic-grid", input_revision=0,
    )


def test_negative_import_prices_remain_negative(meter, snapshot, opening):
    start = ts("2027-02-01") - 900
    state = opening(asof_ts=start, bonus=None)
    item = row(meter, start, "-0.10", import_kwh=Decimal("2"), export_kwh=Decimal("1"))
    result = engine.settle_series([item], [snapshot()], state)
    assert result.lines[0].import_cost == Decimal("-0.20")
    assert result.lines[0].base_export_revenue == Decimal("-0.10")
    assert result.adjustments[0].credit_total == Decimal("0.10")
    assert result.net_cost == Decimal("-0.20")


def test_bonus_is_added_once_not_folded_into_base_export(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    state = opening(asof_ts=start,
                    bonus=BonusState(ts("2027-01-01"), ts("2027-08-10"), start, Decimal("0")))
    item = row(meter, start, import_kwh=Decimal("3"), export_kwh=Decimal("2"))
    result = engine.settle_series([item], [snapshot(bonus=bonus_rules)], state)
    assert result.lines[0].import_cost == Decimal("0.30")
    assert result.lines[0].base_export_revenue == Decimal("0.20")
    assert result.lines[0].bonus_revenue == Decimal("0.06")
    assert result.end_state.month.spot_export_eur == Decimal("0.20")
    assert result.net_cost == Decimal("0.04")
    assert result.adjustments == ()


@pytest.mark.parametrize("missing,want_subtotal", [("import", "-0.13"), ("source", "0.10")])
def test_unknown_component_keeps_subtotal_not_final_zero(meter, snapshot, bonus_rules, opening, missing, want_subtotal):
    start = ts("2027-01-01", 12)
    state = opening(asof_ts=start,
                    bonus=BonusState(ts("2027-01-01"), ts("2027-08-10"), start, Decimal("0")))
    item = row(meter, start, import_kwh=None if missing == "import" else Decimal("2"),
               export_kwh=Decimal("1"), origin="unknown" if missing == "source" else "verified_pv")
    result = engine.settle_series([item], [snapshot(bonus=bonus_rules)], state)
    assert result.known_subtotal == Decimal(want_subtotal)
    assert result.net_cost is None and result.complete is False
    if missing == "import":
        assert result.lines[0].import_cost is None
        assert result.lines[0].bonus_revenue == Decimal("0.03")
    else:
        assert result.lines[0].bonus_revenue is None


def test_aggregate_before_rounding_half_up(meter, snapshot, opening):
    end = ts("2027-02-01")
    state = opening(asof_ts=end - 1800, bonus=None)
    items = [row(meter, end - 1800 + i * 900, "0.005", import_kwh=Decimal("1")) for i in range(2)]
    result = engine.settle_series(items, [snapshot()], state)
    assert [line.import_cost for line in result.lines] == [Decimal("0.005"), Decimal("0.005")]
    assert len(result.adjustments) == 1 and result.adjustments[0].credit_total == Decimal("0")
    assert result.net_cost == Decimal("0.010")
    assert result.net_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) == Decimal("0.01")


def test_floor_products_and_vat_use_precision36(meter, snapshot, opening):
    start = ts("2027-02-01") - 900
    state = opening(asof_ts=start, bonus=None)
    item = row(meter, start, "-0.123456789012", export_kwh=Decimal("0.123456789"))
    terms = snapshot(vat_production=((date(2027, 1, 1), Decimal("21")),))
    with localcontext() as context:
        context.prec = 6
        result = engine.settle_series([item], [terms], state)
    assert result.end_state.month.spot_export_eur == Decimal("-0.015241578751672002468")
    adjustment = result.adjustments[0]
    assert adjustment.credit_ex_vat == Decimal("0.015241578751672002468")
    assert adjustment.credit_vat == Decimal("0.00320073153785112051828")
    assert adjustment.credit_total == Decimal("0.01844231028952312298628")
    assert result.lines[0].base_export_revenue == Decimal("-0.01844231028952312298628")
    assert result.net_cost == Decimal("0")


def test_query_partition_same_generation(meter, snapshot, opening):
    start, end = ts("2027-01-01"), ts("2027-02-01")
    items = []
    for at in range(start, end, 900):
        spot = "-0.20" if at < start + 3600 else "0.05" if at >= end - 3600 else "0"
        exported = Decimal("1") if at in {start, end - 3600} else Decimal("0")
        items.append(row(meter, at, spot, export_kwh=exported))
    terms = [snapshot()]
    full_month = engine.settle_series(items, terms, opening())
    assert full_month.net_cost == Decimal("0")
    assert len(full_month.adjustments) == 1
    assert full_month.adjustments[0].credit_total == Decimal("0.15")
    cuts = (start, start + 900, ts("2027-01-16"), end - 900, end)
    parts, state = [], opening()
    for left, right in zip(cuts, cuts[1:], strict=False):
        part = engine.settle_series([item for item in items if left <= item.start_ts < right], terms, state)
        parts.append(part)
        state = part.end_state
    assert full_month.net_cost == sum(part.net_cost for part in parts)
    assert tuple(line for part in parts for line in part.lines) == full_month.lines
    assert tuple(adjustment for part in parts for adjustment in part.adjustments) == full_month.adjustments
    assert state == full_month.end_state
    assert [len(part.adjustments) for part in parts] == [0, 0, 0, 1]
    # Deelrapporten selecteren uit één bevroren resultaat, niet opnieuw een credit berekenen.
    presented = []
    for left, right in zip(cuts, cuts[1:], strict=False):
        lines = [line for line in full_month.lines if left <= line.start_ts < right]
        adjustments = [adjustment for adjustment in full_month.adjustments
                       if left <= adjustment.period_end_ts - 1 < right]
        presented.append(sum(line.import_cost - line.base_export_revenue - line.bonus_revenue for line in lines)
                         - sum(adjustment.credit_total for adjustment in adjustments))
    assert presented == [Decimal("0.20"), Decimal("0"), Decimal("-0.05"), Decimal("-0.15")]
    assert sum(presented) == full_month.net_cost


def test_floor_none_keeps_negative_export_without_credit(meter, snapshot, opening):
    start = ts("2027-02-01") - 900
    result = engine.settle_series([row(meter, start, "-0.20", export_kwh=Decimal("1"))],
                                 [snapshot(monthly_export_floor="none")], opening(asof_ts=start, bonus=None))
    assert result.adjustments == ()
    assert result.end_state.month is None
    assert result.net_cost == Decimal("0.20")
