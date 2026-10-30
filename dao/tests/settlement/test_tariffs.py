"""Pure tariff quotes use explicit synthetic terms and hand-derived rates."""

from dataclasses import FrozenInstanceError, asdict, replace
from datetime import date, datetime
from decimal import Decimal, localcontext
import json
from zoneinfo import ZoneInfo

import pytest

from dao.prog.config.models.pricing import PricingConfig
from dao.prog.config.models.settlement import SettlementContractConfig
from dao.prog.settlement.tariffs import make_snapshot, quote, resolve_snapshot
from dao.prog.settlement.types import PriceQuote, TARIFF_FIELDS, TariffSnapshot


ZONE = ZoneInfo("Europe/Amsterdam")


def epoch(day, hour=0, minute=0):
    return int(datetime.combine(date.fromisoformat(day), datetime.min.time(), ZONE)
               .replace(hour=hour, minute=minute).timestamp())


@pytest.fixture
def contract():
    def build(**overrides):
        return SettlementContractConfig(**{
            "id": "synthetic-contract", "status": "confirmed",
            "valid_from": date(2027, 1, 1), "valid_to": None,
            "supply_start": date(2026, 8, 10), "timezone": "Europe/Amsterdam",
            "product": "budget_dynamic_2027", "regime": "no_net_metering",
            "price_interval_seconds": 3600, "meter_interval_seconds": 900,
            "hour_price_rule": "native", "monthly_export_floor": "spot_zero",
            **overrides,
        })
    return build


@pytest.fixture
def pricing():
    def build(**overrides):
        return PricingConfig(**{
            "energy_taxes_consumption": {"2026-01-01": "0.09157"},
            "energy_taxes_production": {"2026-01-01": "0.09157", "2027-01-01": "0"},
            "cost_supplier_consumption": {"2026-01-01": "0.01682"},
            "cost_supplier_production": {"2026-01-01": "0.01682", "2027-01-01": "0"},
            "vat_consumption": {"2026-01-01": "21"},
            "vat_production": {"2026-01-01": "21"},
            "multiplier_consumption": {"2026-01-01": "1"},
            "multiplier_production": {"2026-01-01": "1"},
            "last_invoice": date(2026, 8, 10), **overrides,
        })
    return build


def test_legacy_december_new_january(pricing, contract):
    cfg = pricing()
    legacy = make_snapshot(cfg, contract(
        valid_from=date(2026, 8, 10), valid_to=date(2027, 1, 1),
        product="legacy", regime="legacy_interval", monthly_export_floor="none",
    ))
    current = make_snapshot(cfg, contract())
    december_ts, january_ts = epoch("2026-12-31", 12), epoch("2027-01-01", 12)
    december = quote(resolve_snapshot([current, legacy], december_ts, december_ts + 900),
                     december_ts, december_ts + 900, Decimal("0.10"))
    january = quote(resolve_snapshot([legacy, current], january_ts, january_ts + 900),
                    january_ts, january_ts + 900, Decimal("0.10"))
    assert isinstance(december, PriceQuote)
    assert december.export_rate == Decimal("0.2521519")
    assert january.export_rate == Decimal("0.121")
    assert december.import_rate == january.import_rate == Decimal("0.2521519")
    assert january.bonus_rate == Decimal("0")
    assert legacy.contract_id == current.contract_id == "synthetic-contract"
    assert legacy.supply_start == current.supply_start == date(2026, 8, 10)


def test_export_multiplier_without_refund(pricing, contract):
    cfg = pricing(tax_refund=False, multiplier_production={"2026-01-01": "0.9"})
    snap = make_snapshot(cfg, contract(
        product="legacy", regime="legacy_interval", valid_from=date(2026, 8, 10),
    ))
    start = epoch("2026-12-31", 12)
    non_refund_multiplier_09 = quote(snap, start, start + 900, Decimal("0.10"))
    assert non_refund_multiplier_09.export_rate == Decimal("0.1292522")
    assert dict(non_refund_multiplier_09.components)["export_tax_ex_vat"] == Decimal("0")
    assert cfg.energy_taxes_production["2026-01-01"] == 0.09157


@pytest.mark.parametrize("basis,spot,expected", [
    ("ex_vat", "0.10", "0.0363"), ("inc_vat", "0.10", "0.0363"),
    ("ex_vat", "0.17", "0.06171"), ("inc_vat", "0.17", "0.06"),
    ("ex_vat", "0.25", "0.0726"), ("inc_vat", "0.25", "0.06"),
    ("ex_vat", "0", "0"), ("inc_vat", "-0.05", "0"),
])
def test_bonus_vat_basis(snapshot, bonus_rules, basis, spot, expected):
    rules = replace(bonus_rules, cap_basis=basis, vat_percent=Decimal("21"))
    # Export VAT is deliberately different from the separate bonus VAT.
    snap = snapshot(bonus=rules, vat_production=((date(2027, 1, 1), Decimal("0")),))
    start = epoch("2027-01-01", 12)
    result = quote(snap, start, start + 900, Decimal(spot))
    assert result.bonus_rate == Decimal(expected)
    assert result.export_rate == Decimal(spot)  # bonus is never part of the base rate


def test_snapshot_excludes_secrets_and_is_immutable(pricing, contract):
    cfg = pricing(source_day_ahead="entsoe", entsoe_api_key="synthetic-private",
                  production_bonus={"enabled": True, "cap basis": "ex_vat", "vat percent": "0",
                                    "secret_extra": "not-financial"})
    snap = make_snapshot(cfg, contract())
    assert isinstance(snap, TariffSnapshot)
    snapshot_json = json.dumps(asdict(snap), default=str)
    assert "entsoe" not in snapshot_json and "secret" not in snapshot_json
    assert "synthetic-private" not in snapshot_json and "not-financial" not in snapshot_json
    cfg.cost_supplier_consumption["2026-01-01"] = 99.0
    cfg.production_bonus.windows["1"][0] = 0
    assert snap.cost_supplier_consumption == ((date(2026, 1, 1), Decimal("0.01682")),)
    assert dict(snap.bonus.windows)[1] == (9, 17)
    with pytest.raises(FrozenInstanceError):
        snap.contract_id = "changed"
    with pytest.raises(TypeError):
        snap.bonus.windows[0][1][0] = 0


def test_snapshot_hash_is_canonical_and_changes_only_with_financial_input(pricing, contract):
    first = make_snapshot(pricing(), contract())
    with localcontext() as ctx:
        ctx.prec = 6
        reordered = make_snapshot(pricing(
            energy_taxes_production={"2027-01-01": "0.000", "2026-01-01": "0.0915700"},
            source_day_ahead="entsoe", entsoe_api_key="different-private",
            last_invoice=date(2026, 9, 1),
        ), contract())
    assert first == reordered
    assert len(first.snapshot_hash) == 64
    assert int(first.snapshot_hash, 16) >= 0
    changed = make_snapshot(pricing(cost_supplier_consumption={"2026-01-01": "0.01683"}), contract())
    assert changed.snapshot_hash != first.snapshot_hash
    assert changed.contract_id == first.contract_id
    assert changed.supply_start == first.supply_start


def test_hour_quote_repeated_for_four_quarters(snapshot):
    snap = snapshot(vat_production=((date(2027, 1, 1), Decimal("21")),))
    start = epoch("2027-01-01", 12)
    with localcontext() as ctx:
        ctx.prec = 36
        hourly = quote(snap, start, start + 3600, Decimal("0.123456789012"))
        quarters = [quote(snap, start + offset, start + offset + 900, Decimal("0.123456789012"))
                    for offset in (0, 900, 1800, 2700)]
        quarter_export_amounts = [item.export_rate * Decimal("0.000000001") for item in quarters]
        hourly_export_amount = hourly.export_rate * Decimal("0.000000004")
        assert sum(quarter_export_amounts) == hourly_export_amount
        assert hourly_export_amount == Decimal("0.00000000059753085881808")
    assert all(item.export_rate == Decimal("0.14938271470452") for item in quarters)
    assert all(item.snapshot is snap for item in quarters)


def test_quote_uses_local_decimal_context_without_rounding(snapshot):
    snap = snapshot(vat_production=((date(2027, 1, 1), Decimal("21")),))
    start = epoch("2027-01-01")  # still the previous UTC date
    with localcontext() as ctx:
        ctx.prec = 6
        result = quote(snap, start, start + 900, Decimal("0.123456789012"))
        assert result.export_rate == Decimal("0.14938271470452")
        assert ctx.prec == 6


@pytest.mark.parametrize("supplier,expected", [("0.01682", "0.1413522"),
                                               ("-0.01682", "0.1006478")])
def test_supplier_export_component_keeps_its_sign(snapshot, supplier, expected):
    snap = snapshot(product="legacy", cost_supplier_production=((date(2027, 1, 1), Decimal(supplier)),),
                    vat_production=((date(2027, 1, 1), Decimal("21")),))
    start = epoch("2027-01-01", 12)
    result = quote(snap, start, start + 900, Decimal("0.10"))
    assert result.export_rate == Decimal(expected)
    components = dict(result.components)
    assert components["export_supplier_ex_vat"] == Decimal(supplier)
    assert sum(value for name, value in result.components if name.startswith("export_")) == result.export_rate


def test_negative_spot_has_no_interval_floor(snapshot):
    start = epoch("2027-01-01", 12)
    result = quote(snapshot(), start, start + 900, Decimal("-0.10"))
    assert result.import_rate == result.export_rate == Decimal("-0.10")


def test_resolve_snapshot_rejects_contract_boundary_until_split(snapshot):
    boundary = epoch("2027-02-01")
    old = snapshot(valid_to_ts=boundary)
    new = snapshot(snapshot_hash="b" * 64, valid_from_ts=boundary)
    assert resolve_snapshot([new, old], boundary - 900, boundary) is old
    assert resolve_snapshot([old, new], boundary, boundary + 900) is new
    with pytest.raises(ValueError):
        resolve_snapshot([old, new], boundary - 900, boundary + 900)
    with pytest.raises(ValueError):
        quote(old, boundary - 900, boundary + 900, Decimal("0.10"))


@pytest.mark.parametrize("field", TARIFF_FIELDS)
def test_tariff_change_boundary_requires_split(snapshot, field):
    boundary = epoch("2027-02-01")
    snap = snapshot(**{field: ((date(2027, 1, 1), Decimal("0")),
                              (date(2027, 2, 1), Decimal("1")))})
    with pytest.raises(ValueError):
        resolve_snapshot([snap], boundary - 900, boundary + 900)
    with pytest.raises(ValueError):
        quote(snap, boundary - 900, boundary + 900, Decimal("0.10"))


def test_dated_tariff_is_selected_on_each_local_date(snapshot):
    boundary = epoch("2027-02-01")
    snap = snapshot(cost_supplier_production=((date(2027, 1, 1), Decimal("0.01")),
                                              (date(2027, 2, 1), Decimal("-0.02"))))
    assert quote(snap, boundary - 900, boundary, Decimal("0.10")).export_rate == Decimal("0.11")
    assert quote(snap, boundary, boundary + 900, Decimal("0.10")).export_rate == Decimal("0.08")


def test_missing_snapshot_gaps_and_overlaps_are_not_silently_resolved(snapshot):
    start = epoch("2027-01-01")
    for snapshots in ([], [snapshot(valid_from_ts=start + 900)], [snapshot(), snapshot()]):
        with pytest.raises(ValueError):
            resolve_snapshot(snapshots, start, start + 900)


def test_future_tariff_does_not_wrap_or_fill_missing_history(snapshot):
    snap = snapshot(vat_production=((date(2027, 1, 2), Decimal("21")),))
    start = epoch("2027-01-01", 12)
    with pytest.raises(ValueError):
        quote(snap, start, start + 900, Decimal("0.10"))


@pytest.mark.parametrize("start,end", [(1, 1), (2, 1), (True, 900), (0.0, 900)])
def test_interval_requires_integer_epochs_and_positive_duration(snapshot, start, end):
    for calculate in (lambda: resolve_snapshot([snapshot()], start, end),
                      lambda: quote(snapshot(), start, end, Decimal("0.10"))):
        with pytest.raises((TypeError, ValueError)):
            calculate()


@pytest.mark.parametrize("spot", [None, 0.1, Decimal("NaN"), Decimal("0.1234567890123")])
def test_quote_rejects_unknown_float_nonfinite_and_overprecise_spot(snapshot, spot):
    start = epoch("2027-01-01", 12)
    with pytest.raises((TypeError, ValueError)):
        quote(snapshot(), start, start + 900, spot)


@pytest.mark.parametrize("rules_override", [{"cap_basis": "unconfirmed"}, {"vat_percent": None}])
def test_unknown_bonus_basis_does_not_invent_a_positive_rate(snapshot, bonus_rules, rules_override):
    start = epoch("2027-01-01", 12)
    snap = snapshot(status="draft", bonus=replace(bonus_rules, **rules_override))
    assert quote(snap, start, start + 900, Decimal("0.10")).bonus_rate == Decimal("0")


@pytest.mark.parametrize("hour,expected", [(8, "0"), (9, "0.03"), (16, "0.03"), (17, "0")])
def test_bonus_windows_use_contract_timezone(snapshot, bonus_rules, hour, expected):
    start = epoch("2027-01-01", hour)
    assert quote(snapshot(bonus=bonus_rules), start, start + 900, Decimal("0.10")).bonus_rate == Decimal(expected)


def test_bonus_cannot_precede_actual_supply_or_its_start_date(snapshot, bonus_rules):
    start = epoch("2027-01-01", 12)
    for snap in (snapshot(bonus=bonus_rules, supply_start=date(2027, 1, 2)),
                 snapshot(bonus=bonus_rules, supply_start=None),
                 snapshot(bonus=replace(bonus_rules, start_date=date(2027, 1, 2)))):
        assert quote(snap, start, start + 900, Decimal("0.10")).bonus_rate == Decimal("0")


def test_quote_crossing_bonus_window_requires_split(snapshot, bonus_rules):
    snap = snapshot(bonus=bonus_rules)
    for boundary in (epoch("2027-01-01", 9), epoch("2027-01-01", 17)):
        with pytest.raises(ValueError):
            quote(snap, boundary - 900, boundary + 900, Decimal("0.10"))


def test_make_snapshot_revalidates_confirmed_terms_without_settlement_section(pricing, contract):
    with pytest.raises(ValueError):
        make_snapshot(pricing(cost_supplier_production={"2026-01-01": "0.01682"}), contract())
    with pytest.raises(ValueError):
        make_snapshot(pricing(vat_production={"2027-01-02": "21"}), contract())
