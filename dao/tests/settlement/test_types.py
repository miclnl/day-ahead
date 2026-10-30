"""Immutable financial boundaries: unknown quantities are not zero."""

from dataclasses import FrozenInstanceError, asdict, replace
from datetime import date, datetime
from decimal import Decimal

import pytest

from dao.prog.settlement.types import (
    BonusState, InputBatch, InputInterval, MeterSelection,
    PeriodAdjustment, PriceQuote, SettlementLine, SettlementResult,
    SettlementView, SpotInterval, TerminalValue,
)


def input_item(meter, **overrides):
    return InputInterval(**{
        **asdict(meter()), "spot_ex_vat": Decimal("0.1"), "spot_source": "synthetic",
        "input_hash": "b" * 64, "meter_id": "synthetic-meter", "input_revision": 1,
        **overrides,
    })


@pytest.mark.parametrize("overrides", [
    {"import_kwh": Decimal("NaN")}, {"export_kwh": Decimal("Infinity")},
    {"battery_discharge_kwh": Decimal("-Infinity")}, {"import_kwh": Decimal("-0.1")},
    {"export_kwh": Decimal("-1")}, {"battery_discharge_kwh": Decimal("-1")},
    {"import_kwh": Decimal("0.1234567891")}, {"import_kwh": 0.1},
    {"start_ts": datetime(2027, 1, 1)}, {"start_ts": 1.5}, {"start_ts": True},
    {"start_ts": 2, "end_ts": 2}, {"start_ts": 3, "end_ts": 2},
    {"origin": "assumed_pv"}, {"quality": "perfect"},
])
def test_nonfinite_negative_energy_and_naive_boundaries_rejected(meter, overrides):
    with pytest.raises((ValueError, TypeError)):
        meter(**overrides)


def test_unknown_energy_and_spot_stay_unknown(meter):
    item = input_item(meter, import_kwh=None, export_kwh=None,
                      battery_discharge_kwh=None, origin="unknown", quality="incomplete",
                      spot_ex_vat=None, spot_source=None)
    assert item.import_kwh is None
    assert item.export_kwh is None
    assert item.battery_discharge_kwh is None
    assert item.spot_ex_vat is None
    assert BonusState(item.start_ts, item.end_ts, item.start_ts, None).total_export_kwh is None


def test_nested_snapshot_and_meter_contents_are_detached_and_immutable(snapshot, meter, bonus_rules):
    tables = [[date(2027, 1, 1), Decimal("0.1")]]
    versions = [["synthetic", "v1"]]
    windows = [[month, list(window)] for month, window in bonus_rules.windows]
    frozen = snapshot(energy_taxes_consumption=tables, source_versions=versions,
                      bonus=replace(bonus_rules, windows=windows))
    sources = ["synthetic"]
    reading = meter(sources=sources)
    tables[0][1] = Decimal("99")
    versions[0][1] = "changed"
    windows[0][1][0] = 0
    sources.append("changed")
    assert frozen.energy_taxes_consumption == ((date(2027, 1, 1), Decimal("0.1")),)
    assert frozen.source_versions == (("synthetic", "v1"),)
    assert frozen.bonus.windows[0] == (1, (9, 17))
    assert reading.sources == ("synthetic",)
    with pytest.raises(FrozenInstanceError):
        frozen.contract_id = "changed"
    with pytest.raises(TypeError):
        frozen.bonus.windows[0][1][0] = 0


def test_snapshot_is_a_financial_allowlist(snapshot):
    with pytest.raises(TypeError):
        snapshot(options={})
    with pytest.raises(TypeError):
        snapshot(entsoe_api_key=None)


def test_snapshot_tariffs_reject_nonfinite_or_overprecise_values(snapshot):
    for value in (Decimal("NaN"), Decimal("0.1234567890123"), 0.1):
        with pytest.raises((ValueError, TypeError)):
            snapshot(cost_supplier_consumption=((date(2027, 1, 1), value),))


def test_price_quote_keeps_full_snapshot_and_known_rates(snapshot, meter, bonus_rules):
    reading = meter()
    frozen = snapshot(bonus=bonus_rules)
    quote = PriceQuote(reading.start_ts, reading.end_ts, frozen, Decimal("-0.1"),
                       Decimal("-0.1"), Decimal("-0.1"), (("spot", Decimal("-0.1")),),
                       Decimal("0"))
    assert quote.snapshot is frozen
    assert quote.snapshot.supply_start == date(2026, 8, 10)
    assert quote.snapshot.bonus.yearly_cap_kwh == Decimal("5000")
    for field in ("spot_ex_vat", "import_rate", "export_rate", "bonus_rate"):
        with pytest.raises((ValueError, TypeError)):
            replace(quote, **{field: None})
    with pytest.raises(ValueError):
        replace(quote, spot_ex_vat=Decimal("0.1234567890123"))


def test_computed_quote_rates_are_not_rounded_to_input_precision(snapshot, meter):
    reading = meter()
    # 0,123456789012 * 1,21 heeft veertien decimalen: geen invoertarief.
    quote = PriceQuote(reading.start_ts, reading.end_ts, snapshot(),
                       Decimal("0.123456789012"), Decimal("0.14938271470452"),
                       Decimal("0.14938271470452"), (), Decimal("0.044814814411356"))
    assert quote.import_rate == Decimal("0.14938271470452")
    assert quote.bonus_rate == Decimal("0.044814814411356")


def test_unknown_line_totals_and_month_basis_are_explicit(meter, opening):
    reading, state = meter(), opening()
    line = SettlementLine(
        reading.start_ts, reading.end_ts, "synthetic-meter", 1, "b" * 64, None,
        Decimal("0.123456789012345678901234567890123456"), None, None, None,
        state, state, "incomplete", ("missing_export",),
    )
    result = SettlementResult((line,), (), state, "synthetic-v1", reading.start_ts,
                              reading.end_ts, False, line.import_cost, None)
    assert result.known_subtotal == Decimal("0.123456789012345678901234567890123456")
    assert result.net_cost is None
    assert result.lines[0].bonus_kwh is None
    assert replace(state.month, spot_export_eur=None, complete=False).spot_export_eur is None
    basis = [[Decimal("21"), Decimal("0.10")]]
    month = replace(state.month, negative_basis_by_vat=basis)
    basis[0][1] = Decimal("1")
    assert month.negative_basis_by_vat == ((Decimal("21"), Decimal("0.10")),)


def test_registered_revision_not_content_hash_identifies_input(meter, snapshot):
    original = input_item(meter)
    restored = replace(original, input_revision=3)
    assert restored.input_hash == original.input_hash
    assert (restored.meter_id, restored.start_ts, restored.input_revision) != (
        original.meter_id, original.start_ts, original.input_revision)
    batch = InputBatch(3, (restored,), (snapshot(),), restored.end_ts, None)
    assert batch.input_revision == 3
    assert batch.items[0].input_revision == 3
    with pytest.raises(FrozenInstanceError):
        batch.input_revision = 1


def test_view_carries_generation_revision_and_freshness(meter):
    reading = meter()
    view = SettlementView(
        meter_id="synthetic-meter", generation=4, input_revision=3,
        current_input_revision=5, freshness="outdated", lines=(), adjustments=(),
        known_subtotal=Decimal("0"), net_cost=None, coverage_start_ts=reading.start_ts,
        coverage_end_ts=reading.end_ts, complete=False, captured_ts=reading.end_ts,
        source_watermark_ts=None, dirty_from_ts=reading.start_ts,
    )
    assert (view.generation, view.input_revision, view.current_input_revision) == (4, 3, 5)
    assert view.freshness == "outdated"
    assert view.net_cost is None


def test_no_battery_requires_explicit_evidence():
    selection = MeterSelection("synthetic-meter", ("import",), ("export",), (), None)
    assert selection.no_battery_evidence is None
    proven = replace(selection, no_battery_evidence="synthetic-installation-record")
    assert proven.battery_sources == ()
    with pytest.raises(ValueError):
        replace(proven, battery_sources=("battery-discharge",))


def test_spot_adjustment_and_terminal_value_have_explicit_units(meter):
    reading = meter()
    spot = SpotInterval(reading.start_ts, reading.start_ts + 3600,
                        Decimal("-0.1"), "synthetic", "measured")
    adjustment = PeriodAdjustment("spot_zero", reading.start_ts, spot.end_ts,
                                  Decimal("0.10"), Decimal("0.021"), Decimal("0.121"),
                                  Decimal("-0.10"), "estimated")
    terminal = TerminalValue("export_mean_v1", Decimal("0.1"), (("battery", Decimal("0.9")),))
    assert spot.end_ts - spot.start_ts == 3600
    assert adjustment.credit_total == Decimal("0.121")
    assert terminal.reference_efficiency_by_battery == (("battery", Decimal("0.9")),)
