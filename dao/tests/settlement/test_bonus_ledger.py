"""Chronological rights with hand-derived amounts, never an optimal allocation."""

from dataclasses import asdict, replace
from datetime import date, datetime
from decimal import Decimal, localcontext
from zoneinfo import ZoneInfo

import pytest

from dao.prog.settlement.engine import settle_interval, settle_series
from dao.prog.settlement.metering import merge_inputs
from dao.prog.settlement.tariffs import quote
from dao.prog.settlement.types import BonusState, InputInterval, SpotInterval

ZONE = ZoneInfo("Europe/Amsterdam")


def ts(day, hour=0, minute=0):
    return int(datetime.fromisoformat(day).replace(hour=hour, minute=minute, tzinfo=ZONE).timestamp())


def item(meter, start, *, spot="0.10", **overrides):
    return InputInterval(
        **asdict(meter(**{"start_ts": start, "end_ts": start + 900,
                         "import_kwh": Decimal("0"), "export_kwh": Decimal("1"), **overrides})),
        spot_ex_vat=None if spot is None else Decimal(spot),
        spot_source=None if spot is None else "synthetic-hour-price",
        input_hash="b" * 64, meter_id="synthetic-grid", input_revision=0,
    )


def ledger(opening, start, total="0", *, period_start=None, period_end=None):
    return opening(
        asof_ts=start, month=None,
        bonus=BonusState(
            ts("2027-01-01") if period_start is None else period_start,
            ts("2027-08-10") if period_end is None else period_end,
            start, None if total is None else Decimal(total),
        ),
    )


def contract(snapshot, bonus_rules, **overrides):
    return snapshot(bonus=bonus_rules, monthly_export_floor="none", **overrides)


def one(row, terms, state):
    return settle_interval(row, quote(terms, row.start_ts, row.end_ts, row.spot_ex_vat), state)


def test_all_export_consumes_cap(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 8, 45)
    rows = [item(meter, start), item(meter, start + 900)]
    result = settle_series(rows, [contract(snapshot, bonus_rules)], ledger(opening, start, "4999"))
    assert [line.bonus_kwh for line in result.lines] == [Decimal("0"), Decimal("0")]
    assert result.end_state.bonus.total_export_kwh == Decimal("5001")


def test_early_cheap_bonus_cannot_be_reassigned(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    terms = contract(snapshot, replace(bonus_rules, percentage=Decimal("100")))
    rows = [item(meter, start, spot="0.01"), item(meter, start + 900, spot="0.05")]
    result = settle_series(rows, [terms], ledger(opening, start, "4999"))
    assert [line.bonus_kwh for line in result.lines] == [Decimal("1"), Decimal("0")]
    assert [line.bonus_revenue for line in result.lines] == [Decimal("0.01"), Decimal("0")]


def test_partial_crossing(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    line, end = one(item(meter, start, export_kwh=Decimal("0.5")),
                    contract(snapshot, bonus_rules), ledger(opening, start, "4999.8"))
    assert line.bonus_kwh == Decimal("0.2")
    assert line.bonus_revenue == Decimal("0.006")
    assert end.bonus.total_export_kwh == Decimal("5000.3")


def test_pv_equals_load_battery_discharge_no_bonus(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    mixed_battery_line, end = one(
        item(meter, start, battery_discharge_kwh=Decimal("1")),
        contract(snapshot, bonus_rules), ledger(opening, start),
    )
    assert mixed_battery_line.bonus_revenue == Decimal("0")
    assert mixed_battery_line.bonus_kwh == Decimal("0")
    assert mixed_battery_line.base_export_revenue == Decimal("0.10")
    assert end.bonus.total_export_kwh == Decimal("1")


@pytest.mark.parametrize("source", [{"origin": "unknown"}, {"battery_discharge_kwh": None}])
def test_unknown_source_is_not_confirmed_zero(meter, snapshot, bonus_rules, opening, source):
    start = ts("2027-01-01", 12)
    unknown_source_line, end = one(item(meter, start, **source),
                                   contract(snapshot, bonus_rules), ledger(opening, start))
    assert unknown_source_line.bonus_revenue is None
    assert unknown_source_line.bonus_kwh is None
    assert unknown_source_line.base_export_revenue == Decimal("0.10")
    assert unknown_source_line.quality == "incomplete"
    assert unknown_source_line.issues
    assert end.bonus.total_export_kwh == Decimal("1")


@pytest.mark.parametrize("absent_state", [False, True])
def test_unknown_history_is_not_unlimited(meter, snapshot, bonus_rules, opening, absent_state):
    start = ts("2027-01-01", 12)
    state = opening(asof_ts=start, bonus=None, month=None) if absent_state else ledger(opening, start, None)
    unknown_history_line, end = one(item(meter, start), contract(snapshot, bonus_rules), state)
    assert unknown_history_line.bonus_revenue is None
    assert unknown_history_line.bonus_kwh is None
    assert end.bonus.total_export_kwh is None


def test_anniversary_resets_unknown_old_year(meter, snapshot, bonus_rules, opening):
    boundary = ts("2027-08-10")
    windows = tuple((month, (0, 24)) for month in range(1, 13))
    terms = contract(snapshot, replace(bonus_rules, windows=windows))
    rows = [item(meter, boundary - 900), item(meter, boundary)]
    result = settle_series(rows, [terms], ledger(opening, boundary - 900, None))
    assert result.lines[0].bonus_revenue is None
    assert result.lines[0].after_state.bonus == BonusState(boundary, ts("2028-08-10"), boundary, Decimal("0"))
    assert result.lines[1].bonus_revenue == Decimal("0.03")
    assert result.end_state.bonus.total_export_kwh == Decimal("1")


def test_supply_start_in_future(meter, snapshot, bonus_rules, opening):
    start = ts("2027-06-01", 12)
    terms = contract(snapshot, bonus_rules, supply_start=date(2028, 3, 1))
    result = settle_series([item(meter, start)], [terms], opening(asof_ts=start, bonus=None, month=None))
    assert result.lines[0].bonus_revenue == Decimal("0")
    assert result.lines[0].bonus_kwh == Decimal("0")
    assert result.end_state.bonus is None


def test_no_export_before_bonus_start_and_supply_start(meter, snapshot, bonus_rules, opening):
    boundary = ts("2027-01-01")
    zero = ((date(2026, 1, 1), Decimal("0")),)
    unit = ((date(2026, 1, 1), Decimal("1")),)
    terms = contract(
        snapshot, bonus_rules, valid_from_ts=ts("2026-01-01"),
        energy_taxes_consumption=zero, energy_taxes_production=zero,
        cost_supplier_consumption=zero, cost_supplier_production=zero,
        vat_consumption=zero, vat_production=zero,
        multiplier_consumption=unit, multiplier_production=unit,
    )
    rows = [item(meter, boundary - 900, export_kwh=Decimal("5000")), item(meter, boundary)]
    result = settle_series(rows, [terms], opening(asof_ts=boundary - 900, bonus=None, month=None))
    assert [line.bonus_revenue for line in result.lines] == [Decimal("0"), Decimal("0")]
    assert result.lines[0].after_state.bonus.total_export_kwh == Decimal("0")
    assert result.end_state.bonus.total_export_kwh == Decimal("1")


def test_supply_start_is_a_real_zero_opening(meter, snapshot, bonus_rules, opening):
    start = ts("2027-03-01")
    terms = contract(snapshot, bonus_rules, supply_start=date(2027, 3, 1))
    result = settle_series([item(meter, start)], [terms], opening(asof_ts=start, bonus=None, month=None))
    assert result.end_state.bonus == BonusState(start, ts("2028-03-01"), start + 900, Decimal("1"))


@pytest.mark.parametrize("spot", ["0", "-0.10"])
def test_nonpositive_export_consumes_cap(meter, snapshot, bonus_rules, opening, spot):
    start = ts("2027-01-01", 12)
    rows = [item(meter, start, spot=spot), item(meter, start + 900)]
    result = settle_series(rows, [contract(snapshot, bonus_rules)], ledger(opening, start, "4999"))
    assert [line.bonus_kwh for line in result.lines] == [Decimal("0"), Decimal("0")]
    assert result.lines[0].base_export_revenue == Decimal(spot)


def test_battery_and_unknown_origin_export_also_consume_cap(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    rows = [item(meter, start, battery_discharge_kwh=Decimal("0.001")),
            item(meter, start + 900, origin="unknown"), item(meter, start + 1800)]
    result = settle_series(rows, [contract(snapshot, bonus_rules)], ledger(opening, start, "4998"))
    assert [line.bonus_kwh for line in result.lines] == [Decimal("0"), None, Decimal("0")]
    assert result.end_state.bonus.total_export_kwh == Decimal("5001")


def test_missing_spot_still_consumes_cap(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    rows = [item(meter, start, spot=None), item(meter, start + 900)]
    result = settle_series(rows, [contract(snapshot, bonus_rules)], ledger(opening, start, "4999"))
    missing, later = result.lines
    assert missing.import_cost is None and missing.base_export_revenue is None
    assert missing.bonus_revenue is None and missing.quality == "incomplete"
    assert missing.snapshot_hash == "a" * 64
    assert later.bonus_kwh == Decimal("0")
    assert result.end_state.bonus.total_export_kwh == Decimal("5001")
    assert result.complete is False and result.net_cost is None
    assert result.known_subtotal == Decimal("-0.10")


def test_missing_export_makes_later_cap_unknown(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    rows = [item(meter, start, export_kwh=None), item(meter, start + 900)]
    result = settle_series(rows, [contract(snapshot, bonus_rules)], ledger(opening, start))
    assert result.lines[0].base_export_revenue is None
    assert [line.bonus_revenue for line in result.lines] == [None, None]
    assert result.end_state.bonus.total_export_kwh is None


def test_gap_is_explicit_and_destroys_known_cap(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    rows = [item(meter, start), item(meter, start + 1800)]
    result = settle_series(rows, [contract(snapshot, bonus_rules)], ledger(opening, start))
    assert result.lines[0].bonus_revenue == Decimal("0.03")
    assert result.lines[1].bonus_revenue is None
    assert "export_gap" in result.lines[1].issues
    assert result.end_state.bonus.total_export_kwh is None
    assert result.complete is False


def test_gap_before_anniversary_does_not_poison_new_zero(meter, snapshot, bonus_rules, opening):
    boundary = ts("2027-08-10")
    terms = contract(snapshot, bonus_rules)
    result = settle_series([item(meter, boundary)], [terms], ledger(opening, boundary - 3600, None))
    assert result.end_state.bonus.total_export_kwh == Decimal("1")
    assert "export_gap" in result.lines[0].issues


def test_missing_new_year_prefix_is_not_reset_to_zero(meter, snapshot, bonus_rules, opening):
    boundary = ts("2027-08-10")
    start = boundary + 12 * 3600
    result = settle_series([item(meter, start)], [contract(snapshot, bonus_rules)],
                           ledger(opening, boundary - 900, None))
    assert result.lines[0].bonus_revenue is None
    assert result.end_state.bonus.period_start_ts == boundary
    assert result.end_state.bonus.total_export_kwh is None


def test_leap_day_anniversary_is_local_midnight(meter, snapshot, bonus_rules, opening):
    boundary = ts("2027-02-28")
    terms = contract(snapshot, bonus_rules, supply_start=date(2024, 2, 29))
    state = ledger(opening, boundary - 900, None, period_end=boundary)
    result = settle_series([item(meter, boundary - 900), item(meter, boundary)], [terms], state)
    assert result.end_state.bonus == BonusState(boundary, ts("2028-02-29"), boundary + 900, Decimal("1"))


def test_tariff_revision_does_not_reset_export(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    first = contract(snapshot, bonus_rules, valid_to_ts=start + 900)
    second = replace(first, snapshot_hash="c" * 64, valid_from_ts=start + 900, valid_to_ts=None)
    rows = [item(meter, start), item(meter, start + 900)]
    result = settle_series(rows, [first, second], ledger(opening, start, "4999"))
    assert [line.bonus_kwh for line in result.lines] == [Decimal("1"), Decimal("0")]
    assert result.end_state.bonus.total_export_kwh == Decimal("5001")


def test_hour_energy_is_cap_history_not_quarter_bonus(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    hourly = item(meter, start, end_ts=start + 3600, export_kwh=Decimal("2"), quality="estimated")
    result = settle_series([hourly], [contract(snapshot, bonus_rules)], ledger(opening, start))
    assert len(result.lines) == 1
    assert result.lines[0].base_export_revenue == Decimal("0.20")
    assert result.lines[0].bonus_revenue is None
    assert result.end_state.bonus.total_export_kwh == Decimal("2")


def test_unsplittable_year_boundary_does_not_invent_new_year_export(meter, snapshot, bonus_rules, opening):
    boundary = ts("2027-08-10")
    terms = contract(snapshot, replace(bonus_rules, windows=tuple((month, (0, 24)) for month in range(1, 13))))
    row = item(meter, boundary - 900, end_ts=boundary + 900, export_kwh=Decimal("2"))
    result = settle_series([row], [terms], ledger(opening, boundary - 900, "4999"))
    # A2 also rejects this unsplit local window boundary: no substitute quote.
    assert result.lines[0].base_export_revenue is None
    assert result.lines[0].bonus_revenue is None
    assert result.end_state.bonus.period_start_ts == boundary
    assert result.end_state.bonus.total_export_kwh is None


@pytest.mark.parametrize("spot", ["0.10", None])
def test_unsplit_window_is_unknown_even_when_start_is_outside(meter, snapshot, bonus_rules, opening, spot):
    start = ts("2027-01-01", 8, 45)
    row = item(meter, start, spot=spot, end_ts=start + 1800)
    result = settle_series([row], [contract(snapshot, bonus_rules)], ledger(opening, start))
    assert result.lines[0].bonus_kwh is None
    assert result.lines[0].bonus_revenue is None
    assert result.end_state.bonus.total_export_kwh == Decimal("1")


def test_unsplit_anniversary_is_unknown_even_when_old_cap_is_exhausted(meter, snapshot, bonus_rules, opening):
    boundary = ts("2027-08-10")
    terms = contract(snapshot, replace(bonus_rules, windows=tuple((month, (0, 24)) for month in range(1, 13))))
    row = item(meter, boundary - 900, end_ts=boundary + 900)
    result = settle_series([row], [terms], ledger(opening, boundary - 900, "5000"))
    assert result.lines[0].bonus_kwh is None
    assert result.lines[0].bonus_revenue is None
    assert result.end_state.bonus.total_export_kwh is None


@pytest.mark.parametrize("boundary_kind", ["bonus", "supply"])
def test_unsplit_activation_is_not_confirmed_zero(meter, snapshot, bonus_rules, opening, boundary_kind):
    boundary = ts("2027-03-01")
    terms = contract(snapshot, replace(bonus_rules, start_date=date(2027, 3, 1))
                     if boundary_kind == "bonus" else bonus_rules,
                     supply_start=date(2027, 3, 1) if boundary_kind == "supply" else date(2026, 8, 10))
    row = item(meter, boundary - 900, end_ts=boundary + 3600 * 12)
    result = settle_series([row], [terms], ledger(opening, boundary - 900))
    assert result.lines[0].bonus_kwh is None
    assert result.lines[0].bonus_revenue is None
    assert result.end_state.bonus.total_export_kwh is None


def test_missing_terms_produce_incomplete_not_zero_price(meter, opening):
    start = ts("2027-01-01", 12)
    result = settle_series([item(meter, start)], [], ledger(opening, start))
    assert result.lines[0].snapshot_hash is None
    assert result.lines[0].import_cost is None
    assert result.lines[0].base_export_revenue is None
    assert result.lines[0].bonus_revenue is None
    assert result.lines[0].issues
    assert result.end_state.bonus.total_export_kwh is None
    assert result.complete is False and result.net_cost is None


def test_unknown_supply_start_does_not_get_unlimited_bonus(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    terms = contract(snapshot, bonus_rules, supply_start=None)
    result = settle_series([item(meter, start)], [terms], ledger(opening, start))
    assert result.lines[0].bonus_revenue is None
    assert result.lines[0].base_export_revenue == Decimal("0.10")


def test_unconfirmed_bonus_basis_is_unknown_not_zero(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    terms = contract(snapshot, replace(bonus_rules, cap_basis="unconfirmed", vat_percent=None))
    result = settle_series([item(meter, start)], [terms], ledger(opening, start))
    assert result.lines[0].bonus_revenue is None
    assert result.complete is False


def test_no_bonus_and_zero_export_do_not_require_source_proof(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    row = item(meter, start, export_kwh=Decimal("0"), origin="unknown", battery_discharge_kwh=None)
    for terms in (snapshot(monthly_export_floor="none"), contract(snapshot, bonus_rules)):
        result = settle_series([row], [terms], ledger(opening, start, None))
        assert result.lines[0].bonus_kwh == Decimal("0")
        assert result.lines[0].bonus_revenue == Decimal("0")


def test_amount_signs_separate_import_export_and_bonus(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    rows = [item(meter, start, import_kwh=Decimal("2")),
            item(meter, start + 900, spot="-0.10", import_kwh=Decimal("2"))]
    result = settle_series(rows, [contract(snapshot, bonus_rules)], ledger(opening, start))
    assert result.lines[0].import_cost == Decimal("0.20")
    assert result.lines[0].base_export_revenue == Decimal("0.10")
    assert result.lines[0].bonus_revenue == Decimal("0.03")
    assert result.lines[1].import_cost == Decimal("-0.20")
    assert result.lines[1].base_export_revenue == Decimal("-0.10")
    assert result.known_subtotal == result.net_cost == Decimal("-0.03")
    assert result.adjustments == () and result.complete is True


def test_amount_arithmetic_uses_precision36_not_callers_context(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    row = item(meter, start, spot="0.123456789012", export_kwh=Decimal("0.123456789"))
    with localcontext() as context:
        context.prec = 6
        result = settle_series([row], [contract(snapshot, bonus_rules)], ledger(opening, start))
    # 123456789 * 123456789012 = 15241578751672002468; then 30% bonus.
    assert result.lines[0].base_export_revenue == Decimal("0.015241578751672002468")
    assert result.lines[0].bonus_revenue == Decimal("0.0045724736255016007404")
    assert result.net_cost == Decimal("-0.0198140523771736032084")


def test_merge_revision_zero_and_meter_identity_survive_engine(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    terms = contract(snapshot, bonus_rules)
    rows = merge_inputs(
        [meter(start_ts=start, end_ts=start + 900)],
        [SpotInterval(start, start + 3600, Decimal("0.10"), "synthetic-price", "measured")],
        [terms], meter_id="explicit-meter",
    )
    result = settle_series(rows, [terms], ledger(opening, start))
    line = result.lines[0]
    assert line.meter_id == "explicit-meter" and line.input_revision == 0
    assert line.input_hash == rows[0].input_hash and line.snapshot_hash == terms.snapshot_hash
    assert line.bonus_revenue == Decimal("0.06")


@pytest.mark.parametrize("case", ["reversed", "overlap", "future_opening", "mixed_meters"])
def test_invalid_chronology_is_rejected_not_sorted(meter, snapshot, bonus_rules, opening, case):
    start = ts("2027-01-01", 12)
    rows = [item(meter, start), item(meter, start + 900)]
    state = ledger(opening, start)
    if case == "reversed":
        rows.reverse()
    elif case == "overlap":
        rows[1] = replace(rows[1], start_ts=start + 300)
    elif case == "future_opening":
        state = ledger(opening, start + 900)
    else:
        rows[1] = replace(rows[1], meter_id="different-meter")
    with pytest.raises(ValueError):
        settle_series(rows, [contract(snapshot, bonus_rules)], state)


def test_quote_must_cover_the_input_and_match_its_spot(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    row = item(meter, start)
    price = quote(contract(snapshot, bonus_rules), start, start + 900, Decimal("0.10"))
    for wrong in (replace(price, start_ts=start + 1), replace(price, spot_ex_vat=Decimal("0.20"))):
        with pytest.raises(ValueError):
            settle_interval(row, wrong, ledger(opening, start))


def test_hour_quote_can_cover_a_measured_quarter(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    terms = contract(snapshot, bonus_rules)
    row = item(meter, start + 900)
    price = quote(terms, start, start + 3600, Decimal("0.10"))
    line, end = settle_interval(row, price, ledger(opening, start + 900))
    assert line.bonus_revenue == Decimal("0.03")
    assert end.bonus.total_export_kwh == Decimal("1")


def test_missing_import_preserves_known_export_components(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    result = settle_series([item(meter, start, import_kwh=None)],
                           [contract(snapshot, bonus_rules)], ledger(opening, start))
    assert result.lines[0].import_cost is None
    assert result.lines[0].base_export_revenue == Decimal("0.10")
    assert result.lines[0].bonus_revenue == Decimal("0.03")
    assert result.known_subtotal == Decimal("-0.13")
    assert result.net_cost is None and result.complete is False


def test_stale_or_wrong_year_opening_is_not_current_history(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    state = ledger(opening, start)
    for bonus in (replace(state.bonus, asof_ts=start - 900),
                  replace(state.bonus, period_end_ts=ts("2028-01-01"))):
        result = settle_series([item(meter, start)], [contract(snapshot, bonus_rules)], replace(state, bonus=bonus))
        assert result.lines[0].bonus_revenue is None
        assert result.end_state.bonus.total_export_kwh is None


def test_empty_coverage_and_conflicting_snapshots_are_rejected(meter, snapshot, bonus_rules, opening):
    start = ts("2027-01-01", 12)
    terms = contract(snapshot, bonus_rules)
    with pytest.raises(ValueError, match="interval"):
        settle_series([], [terms], ledger(opening, start))
    with pytest.raises(ValueError, match="overlap"):
        settle_series([item(meter, start)], [terms, terms], ledger(opening, start))


@pytest.mark.parametrize("scenario", ["cap", "missing", "reset"])
def test_split_series_at_every_interval_preserves_lines_and_state(meter, snapshot, bonus_rules, opening, scenario):
    start = ts("2027-01-01", 8, 45) if scenario != "reset" else ts("2027-08-09", 23, 45)
    rows = [item(meter, start + i * 900, spot=spot)
            for i, spot in enumerate(("0.10", "0.20", "-0.10", "0.05", "0.10"))]
    state = ledger(opening, start, "4998.8")
    if scenario == "missing":
        rows[1] = replace(rows[1], spot_ex_vat=None, spot_source=None)
        rows[2] = replace(rows[2], export_kwh=None)
        rows[3] = replace(rows[3], start_ts=rows[3].start_ts + 900, end_ts=rows[3].end_ts + 900)
        rows[4] = replace(rows[4], start_ts=rows[4].start_ts + 900, end_ts=rows[4].end_ts + 900)
    rules = (replace(bonus_rules, windows=tuple((month, (0, 24)) for month in range(1, 13)))
             if scenario == "reset" else bonus_rules)
    terms = [contract(snapshot, rules)]
    whole = settle_series(rows, terms, state)
    for split in range(1, len(rows)):
        first = settle_series(rows[:split], terms, state)
        second = settle_series(rows[split:], terms, first.end_state)
        assert first.lines + second.lines == whole.lines
        assert second.end_state == whole.end_state
        assert first.known_subtotal + second.known_subtotal == whole.known_subtotal
