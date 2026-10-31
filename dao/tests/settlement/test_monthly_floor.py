"""Monthly spot floors, with explicit known prefixes and hand-derived credits."""

from dataclasses import asdict, replace
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from zoneinfo import ZoneInfo

import pytest

from dao.prog.settlement import engine
from dao.prog.settlement.tariffs import quote
from dao.prog.settlement.types import BonusState, InputInterval, MonthState

ZONE = ZoneInfo("Europe/Amsterdam")


def ts(day, hour=0):
    return int(datetime.fromisoformat(day).replace(hour=hour, tzinfo=ZONE).timestamp())


def row(meter, start, spot, **overrides):
    return InputInterval(
        **asdict(meter(**{"start_ts": start, "end_ts": start + 900,
                         "import_kwh": Decimal("0"), "export_kwh": Decimal("0"), **overrides})),
        spot_ex_vat=None if spot is None else Decimal(spot),
        spot_source=None if spot is None else "synthetic-hour-price",
        input_hash="b" * 64, meter_id="synthetic-grid", input_revision=0,
    )


def tail(meter, opening, spots):
    """Known zero-export month prefix; one exported kWh in each remaining hour."""
    end = ts("2027-02-01")
    start = end - 3600 * len(spots)
    state = opening(asof_ts=start, bonus=None)
    rows = [row(meter, start + i * 900, spot,
                export_kwh=Decimal("1") if i % 4 == 0 else Decimal("0"))
            for i, spot in enumerate(spot for spot in spots for _ in range(4))]
    return rows, state


@pytest.mark.parametrize("spots,credit,net", [
    (("-0.10", "0.20"), "0", "-0.10"),
    (("-0.20", "0.05"), "0.15", "0"),
])
def test_floor_not_per_hour(meter, snapshot, opening, spots, credit, net):
    rows, state = tail(meter, opening, spots)
    result = engine.settle_series(rows, [snapshot()], state)
    assert len(result.adjustments) == 1
    adjustment = result.adjustments[0]
    assert adjustment.credit_ex_vat == Decimal(credit)
    assert adjustment.credit_vat == Decimal("0")
    assert adjustment.credit_total == Decimal(credit)
    assert adjustment.basis_ex_vat == Decimal("0.10" if credit == "0" else "-0.15")
    assert adjustment.kind == "spot_zero"
    assert (adjustment.period_start_ts, adjustment.period_end_ts) == (ts("2027-01-01"), ts("2027-02-01"))
    assert result.lines[0].base_export_revenue == Decimal(spots[0])
    assert result.lines[4].base_export_revenue == Decimal(spots[1])
    assert result.net_cost == result.known_subtotal == Decimal(net)
    assert result.complete is True


def test_zero_exports_no_division(meter, snapshot, opening):
    rows, state = tail(meter, opening, ("-0.20", "0.05"))
    rows = [replace(item, export_kwh=Decimal("0")) for item in rows]
    result = engine.settle_series(rows, [snapshot()], state)
    assert result.end_state.month.spot_export_eur == Decimal("0")
    assert result.end_state.month.negative_basis_by_vat == ()
    assert len(result.adjustments) == 1  # Bekende nulsluiting, niet ontbrekend.
    adjustment = result.adjustments[0]
    assert (adjustment.credit_ex_vat, adjustment.credit_vat, adjustment.credit_total) == (
        Decimal("0"), Decimal("0"), Decimal("0"))
    assert result.net_cost == Decimal("0")


@pytest.mark.parametrize("missing", ["spot", "export", "prefix", "gap", "quality"])
def test_unknown_month_no_final_credit(meter, snapshot, opening, missing):
    rows, state = tail(meter, opening, ("-0.20", "0.05"))
    if missing == "spot":
        rows[0] = replace(rows[0], spot_ex_vat=None, spot_source=None)
    elif missing == "export":
        rows[0] = replace(rows[0], export_kwh=None)
    elif missing == "prefix":
        state = replace(state, month=replace(state.month, spot_export_eur=None, complete=False))
    elif missing == "gap":
        del rows[1]
    else:
        rows[0] = replace(rows[0], quality="incomplete")
    result = engine.settle_series(rows, [snapshot()], state)
    assert result.adjustments == ()
    assert result.end_state.month.complete is False
    assert engine.close_month(result.end_state.month) is None
    assert result.complete is False and result.net_cost is None
    assert result.known_subtotal == Decimal("-0.05" if missing in {"spot", "export"} else "0.15")


def test_close_month_rejects_unknown_basis_even_with_complete_flag(opening):
    assert engine.close_month(replace(opening().month, spot_export_eur=None)) is None


def test_contract_revision_does_not_reset_floor(meter, snapshot, opening):
    rows, state = tail(meter, opening, ("-0.20", "0.05"))
    boundary = rows[4].start_ts
    first = snapshot(valid_to_ts=boundary)
    revised = replace(first, snapshot_hash="c" * 64, valid_from_ts=boundary, valid_to_ts=None)
    result = engine.settle_series(rows, [first, revised], state)
    assert result.lines[3].after_state.month.spot_export_eur == Decimal("-0.20")
    assert result.end_state.month.spot_export_eur == Decimal("-0.15")
    assert len(result.adjustments) == 1
    assert result.adjustments[0].credit_ex_vat == Decimal("0.15")
    assert result.adjustments[0].period_start_ts == ts("2027-01-01")


@pytest.mark.parametrize("supply_start", [date(2027, 1, 1), date(2027, 1, 16)])
def test_contract_change_splits_period_not_just_snapshot(meter, snapshot, opening, supply_start):
    boundary = ts("2027-01-16")
    first = snapshot(valid_to_ts=boundary)
    second = snapshot(snapshot_hash="c" * 64, contract_id="other-contract",
                      valid_from_ts=boundary, supply_start=supply_start)
    rows = [row(meter, boundary - 900, "-0.20", export_kwh=Decimal("1")),
            row(meter, boundary, "0.05", export_kwh=Decimal("1"))]
    state = opening(asof_ts=boundary - 900, bonus=None)
    result = engine.settle_series(rows, [first, second], state)
    assert len(result.adjustments) == 1
    adjustment = result.adjustments[0]
    assert adjustment.credit_ex_vat == Decimal("0.20")
    assert (adjustment.period_start_ts, adjustment.period_end_ts) == (ts("2027-01-01"), boundary)
    assert result.lines[0].after_state.month.period_end_ts == boundary
    assert result.end_state.month == MonthState(
        "other-contract", boundary, ts("2027-02-01"), Decimal("0.05"), (), True)
    assert result.net_cost == Decimal("-0.05")
    first_part = engine.settle_series(rows[:1], [first, second], state)
    second_part = engine.settle_series(rows[1:], [first, second], first_part.end_state)
    assert first_part.lines + second_part.lines == result.lines
    assert first_part.adjustments + second_part.adjustments == result.adjustments
    assert second_part.end_state == result.end_state
    assert first_part.net_cost + second_part.net_cost == result.net_cost


def test_vat_groups_and_rounding(meter, snapshot, opening):
    rows, state = tail(meter, opening, ("-0.20", "-0.10", "0.15"))
    boundary = rows[4].start_ts
    first = snapshot(valid_to_ts=boundary, vat_production=((date(2027, 1, 1), Decimal("21")),))
    revised = replace(first, snapshot_hash="c" * 64, valid_from_ts=boundary, valid_to_ts=None,
                      vat_production=((date(2027, 1, 1), Decimal("9")),))
    with localcontext() as context:
        context.prec = 6
        result = engine.settle_series(rows, [first, revised], state)
        adjustment = engine.close_month(result.end_state.month)
    assert result.end_state.month.negative_basis_by_vat == (
        (Decimal("9"), Decimal("0.10")), (Decimal("21"), Decimal("0.20")))
    assert adjustment == result.adjustments[0]
    assert adjustment.credit_ex_vat == Decimal("0.15")
    # Credit 0,15: 1/3 naar 9% en 2/3 naar 21% -> 0,0045 + 0,021.
    assert adjustment.credit_vat == Decimal("0.0255")
    assert adjustment.credit_total == Decimal("0.1755")
    assert adjustment.quality == "estimated"  # Expliciete modelconventie, geen leveranciersbewijs.
    assert adjustment.credit_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) == Decimal("0.18")
    assert result.net_cost == Decimal("0.0120")


def test_floor_basis_excludes_import_adders_bonus_and_multiplier(meter, snapshot, bonus_rules, opening):
    rows, state = tail(meter, opening, ("-0.20", "0.05"))
    rows[4] = replace(rows[4], import_kwh=Decimal("2"))
    all_day_bonus = replace(bonus_rules, windows=tuple((month, (0, 24)) for month in range(1, 13)))
    # Synthetisch niet-Budget-profiel om onafhankelijke componenten te onderscheiden.
    terms = snapshot(
        product="legacy", regime="legacy_interval", bonus=all_day_bonus,
        cost_supplier_consumption=((date(2027, 1, 1), Decimal("0.03")),),
        energy_taxes_consumption=((date(2027, 1, 1), Decimal("0.07")),),
        cost_supplier_production=((date(2027, 1, 1), Decimal("0.02")),),
        energy_taxes_production=((date(2027, 1, 1), Decimal("0.04")),),
        multiplier_production=((date(2027, 1, 1), Decimal("0.9")),),
    )
    # De fixture levert een bekende bonusprefix op de juiste leveringsjaargrenzen.
    state = replace(state, bonus=BonusState(ts("2027-01-01"), ts("2027-08-10"), state.asof_ts, Decimal("0")))
    result = engine.settle_series(rows, [terms], state)
    assert result.end_state.month.spot_export_eur == Decimal("-0.15")
    assert result.adjustments[0].credit_ex_vat == Decimal("0.15")
    assert result.lines[0].base_export_revenue == Decimal("-0.12")
    assert result.lines[4].base_export_revenue == Decimal("0.105")
    assert result.lines[4].import_cost == Decimal("0.30")
    assert result.lines[4].bonus_revenue == Decimal("0.015")
    assert result.net_cost == Decimal("0.15")


def test_open_month_keeps_provision_out_of_booked_adjustments(meter, snapshot, opening):
    rows, state = tail(meter, opening, ("-0.20", "0.05"))
    result = engine.settle_series(rows[:4], [snapshot()], state)
    assert result.adjustments == ()
    assert result.end_state.month.spot_export_eur == Decimal("-0.20")
    assert result.end_state.month.complete is True
    assert result.net_cost == Decimal("0.20")  # Nog geen gesloten credit/uitbetaling.


def test_settle_interval_accumulates_same_month_state(meter, snapshot, opening):
    start = ts("2027-02-01") - 900
    state = opening(asof_ts=start, bonus=None,
                    month=replace(opening().month, spot_export_eur=Decimal("-0.20"),
                                  negative_basis_by_vat=((Decimal("0"), Decimal("0.20")),)))
    item = row(meter, start, "0.05", export_kwh=Decimal("1"))
    terms = snapshot()
    line, end = engine.settle_interval(item, quote(terms, start, start + 900, Decimal("0.05")), state)
    assert line.after_state == end
    assert end.month.spot_export_eur == Decimal("-0.15")
    assert engine.close_month(end.month).credit_total == Decimal("0.15")


def test_month_boundary_resets_unknown_old_prefix(meter, snapshot, opening):
    boundary = ts("2027-02-01")
    state = opening(asof_ts=boundary - 900, bonus=None,
                    month=replace(opening().month, spot_export_eur=None, complete=False))
    rows = [row(meter, boundary - 900, "-0.20", export_kwh=Decimal("1")),
            row(meter, boundary, "0.05", export_kwh=Decimal("1"))]
    result = engine.settle_series(rows, [snapshot()], state)
    assert result.adjustments == ()
    assert result.end_state.month == MonthState(
        "synthetic-2027", boundary, ts("2027-03-01"), Decimal("0.05"), (), True)
    assert result.net_cost is None  # Onvolledige oude periode blijft zichtbaar.


@pytest.mark.parametrize("month,following,quarters", [
    ("2027-03-01", "2027-04-01", 2972), ("2027-10-01", "2027-11-01", 2980),
])
def test_month_bounds_use_local_dst_calendar(meter, snapshot, opening, month, following, quarters):
    start, end = ts(month), ts(following)
    state = opening(asof_ts=start, bonus=None, month=None)
    result = engine.settle_series([row(meter, start, "0")], [snapshot()], state)
    assert result.end_state.month == MonthState("synthetic-2027", start, end, Decimal("0"), (), True)
    assert end - start == quarters * 900


def test_unsplit_month_boundary_does_not_fabricate_export(meter, snapshot, opening):
    boundary = ts("2027-02-01")
    state = opening(asof_ts=boundary - 900, bonus=None)
    item = row(meter, boundary - 900, "-0.20", end_ts=boundary + 900, export_kwh=Decimal("1"))
    result = engine.settle_series([item], [snapshot()], state)
    assert len(result.lines) == 1
    assert "month_boundary_requires_split" in result.lines[0].issues
    assert result.end_state.month.period_start_ts == boundary
    assert result.end_state.month.spot_export_eur is None
    assert result.end_state.month.complete is False
    assert result.adjustments == () and result.net_cost is None


def test_absent_midmonth_prefix_is_not_zero(meter, snapshot, opening):
    start = ts("2027-02-01") - 900
    state = opening(asof_ts=start, bonus=None, month=None)
    result = engine.settle_series([row(meter, start, "-0.20", export_kwh=Decimal("1"))], [snapshot()], state)
    assert result.end_state.month.spot_export_eur is None
    assert result.end_state.month.complete is False
    assert result.adjustments == () and result.net_cost is None
    assert result.known_subtotal == Decimal("0.20")


@pytest.mark.parametrize("unknown", [False, True])
def test_split_series_at_every_interval_preserves_month_and_amounts(meter, snapshot, opening, unknown):
    rows, state = tail(meter, opening, ("-0.20", "0.05"))
    if unknown:
        rows[1] = replace(rows[1], export_kwh=None)
    first = snapshot(valid_to_ts=rows[4].start_ts)
    second = replace(first, snapshot_hash="c" * 64, valid_from_ts=rows[4].start_ts, valid_to_ts=None)
    terms = [first, second]
    whole = engine.settle_series(rows, terms, state)
    for split in range(1, len(rows)):
        left = engine.settle_series(rows[:split], terms, state)
        right = engine.settle_series(rows[split:], terms, left.end_state)
        assert left.lines + right.lines == whole.lines
        assert left.adjustments + right.adjustments == whole.adjustments
        assert right.end_state == whole.end_state
        assert left.known_subtotal + right.known_subtotal == whole.known_subtotal
        if not unknown:
            assert left.net_cost + right.net_cost == whole.net_cost
