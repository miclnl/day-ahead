"""Explicit settlement terms must not silently activate guessed tariffs."""

from copy import deepcopy
from datetime import date
from decimal import Decimal

import pytest

from dao.prog.config.models.pricing import PricingConfig
from dao.prog.config.models.production_bonus import ProductionBonusConfig
from dao.prog.config.models.settlement import SettlementConfig, SettlementContractConfig
from dao.prog.config.versions.v0 import ConfigurationV0


TARIFF_FIELDS = (
    "energy taxes consumption", "energy taxes production",
    "cost supplier consumption", "cost supplier production",
    "vat consumption", "vat production",
    "multiplier consumption", "multiplier production",
)


def contract(**overrides):
    """Synthetic terms, not the operator's supplier settings."""
    return {
        "id": "synthetic-2027", "status": "confirmed",
        "valid from": "2027-01-01", "valid to": None,
        "supply start": "2026-08-10", "timezone": "Europe/Amsterdam",
        "product": "budget_dynamic_2027", "regime": "no_net_metering",
        "price interval seconds": 3600, "meter interval seconds": 900,
        "hour price rule": "native", "monthly export floor": "spot_zero",
        **overrides,
    }


def pricing(**overrides):
    return {
        **{name: {"2027-01-01": "0"} for name in TARIFF_FIELDS},
        "multiplier consumption": {"2027-01-01": "1"},
        "multiplier production": {"2027-01-01": "1"},
        "last invoice": "2026-08-10",
        **overrides,
    }


def configured(terms=None, mode="active", **overrides):
    return PricingConfig.model_validate(pricing(**{
        "settlement": {"mode": mode, "meter id": "synthetic-meter",
                       "contracts": [contract() if terms is None else terms]},
        **overrides,
    }))


def test_default_off_preserves_legacy():
    parsed_without_settlement = ConfigurationV0.model_validate({"prices": pricing()})
    assert parsed_without_settlement.prices.settlement is None
    assert SettlementConfig().mode == "off"
    assert PricingConfig.model_validate(pricing(settlement={})).settlement.mode == "off"
    assert ProductionBonusConfig().cap_basis == "unconfirmed"
    assert ProductionBonusConfig().vat_percent is None


def test_policy_none_alias_and_invalid():
    """The planner domain is explicit and cannot silently activate with settlement."""
    policy = "aggregate_cell_min_1wh_v1"
    assert SettlementConfig().planning_source_policy is None
    assert SettlementConfig(planning_source_policy=None).planning_source_policy is None
    for name in ("planning source policy", "planning_source_policy"):
        parsed = SettlementConfig.model_validate({name: policy})
        assert parsed.planning_source_policy == policy
        assert parsed.mode == "off"
    data = {"planning source policy": policy, "planning_source_policy": policy}
    original = deepcopy(data)
    assert SettlementConfig.model_validate(data).planning_source_policy == policy
    assert data == original
    for bad in ({"planning source policy": "aggregate_cell_min_2wh_v1"},
                {"planning source policy": "typo"},
                {"planning source policy": policy, "planning_source_policy": None}):
        with pytest.raises(ValueError):
            SettlementConfig.model_validate(bad)
    for mode in ("active", "shadow"):
        assert configured(mode=mode).settlement.planning_source_policy is None


@pytest.mark.parametrize("missing", ["supply start", "hour price rule"])
def test_confirmed_rejects_missing_terms(missing):
    terms = contract(**{missing: None if missing == "supply start" else "unconfirmed"})
    bonus = {"enabled": True, "cap basis": "ex_vat", "vat percent": "0"}
    with pytest.raises(ValueError):
        configured(terms, **{"production bonus": bonus})


@pytest.mark.parametrize("field", TARIFF_FIELDS)
def test_confirmed_requires_each_dated_component_at_start(field):
    with pytest.raises(ValueError, match="contract start"):
        configured(**{field: {"2027-01-02": "0"}})


@pytest.mark.parametrize("field", ["multiplier consumption", "multiplier production"])
def test_confirmed_does_not_use_implicit_multiplier_defaults(field):
    data = pricing(settlement={"mode": "active", "meter id": "synthetic-meter",
                               "contracts": [contract()]})
    del data[field]
    with pytest.raises(ValueError, match="explicit"):
        PricingConfig.model_validate(data)


@pytest.mark.parametrize("overrides", [
    {"price interval seconds": 900},
    {"timezone": "UTC"},
    {"regime": "legacy_interval"},
])
def test_budget_profile_requires_hour_spot_and_no_export_adders(overrides):
    with pytest.raises(ValueError):
        configured(contract(**overrides))


@pytest.mark.parametrize("field,value", [
    ("energy taxes production", "0.01"),
    ("cost supplier production", "0.01"),
    ("multiplier production", "0.99"),
])
def test_budget_rejects_export_adders_also_after_contract_start(field, value):
    for effective_date in ("2027-01-01", "2027-06-01"):
        with pytest.raises(ValueError):
            configured(**{field: {"2027-01-01": "0" if "multiplier" not in field else "1",
                                  effective_date: value}})


def test_budget_ignores_export_adders_outside_the_contract_period():
    cfg = configured(contract(**{"valid to": "2028-01-01"}), **{
        "energy taxes production": {"2026-01-01": "0.1", "2027-01-01": "0",
                                    "2028-01-01": "0.1"},
    })
    assert cfg.settlement.contracts[0].id == "synthetic-2027"


@pytest.mark.parametrize("bonus_overrides", [
    {"cap basis": "unconfirmed"}, {"vat percent": None}, {"mode": "all export"},
])
def test_confirmed_bonus_requires_explicit_basis_and_vat(bonus_overrides):
    with pytest.raises(ValueError):
        configured(**{"production bonus": {"enabled": True, "cap basis": "ex_vat",
                                            "vat percent": "0", **bonus_overrides}})


@pytest.mark.parametrize("field,value", [
    ("max exchange price", "Infinity"), ("yearly cap kwh", "Infinity"),
    ("yearly cap kwh", "NaN"), ("max exchange price", "0.1234567890123"),
    ("percentage", "30.1234567890123"), ("yearly cap kwh", "1.1234567891"),
])
def test_confirmed_bonus_rejects_nonfinite_and_overprecise_terms(field, value):
    with pytest.raises(ValueError):
        configured(**{"production bonus": {"enabled": True, "cap basis": "ex_vat",
                                            "vat percent": "0", field: value}})


@pytest.mark.parametrize("field,value", [
    ("max exchange price", "0.200000000000000001"),
    ("percentage", "30.000000000000001"),
    ("yearly cap kwh", "5000.0000000000001"),
    ("max_exchange_price", "0.200000000000000001"),
    ("yearly_cap_kwh", "5000.0000000000001"),
])
def test_confirmed_bonus_rejects_precision_lost_by_float_conversion(field, value):
    with pytest.raises(ValueError, match="at most"):
        configured(**{"production bonus": {"enabled": True, "cap basis": "ex_vat",
                                            "vat percent": "0", field: value}})


@pytest.mark.parametrize("field,value,expected", [
    ("max exchange price", "0.200000000000000001", 0.2),
    ("percentage", "30.000000000000001", 30.0),
    ("yearly cap kwh", "5000.0000000000001", 5000.0),
])
@pytest.mark.parametrize("route", ["legacy", "draft", "disabled", "disabled_string"])
def test_original_bonus_precision_check_preserves_unconfirmed_routes(field, value, expected, route):
    bonus = {"enabled": True, "cap basis": "ex_vat", "vat percent": "0", field: value}
    if route == "legacy":
        cfg = PricingConfig.model_validate(pricing(**{"production bonus": bonus}))
    elif route == "draft":
        cfg = configured(contract(status="draft"), mode="shadow", **{"production bonus": bonus})
    else:
        bonus["enabled"] = False if route == "disabled" else "false"
        cfg = configured(**{"production bonus": bonus})
        assert cfg.production_bonus.enabled is False
    assert getattr(cfg.production_bonus, field.replace(" ", "_")) == expected


def test_confirmed_bonus_does_not_infer_supply_start_from_invoice_or_bonus_anchor():
    with pytest.raises(ValueError):
        configured(contract(**{"supply start": None}), **{"production bonus": {
            "enabled": True, "cap basis": "ex_vat", "vat percent": "0",
            "delivery year start": "2026-08-10",
        }})


def test_alias_conflict_and_overlapping_contracts():
    with pytest.raises(ValueError, match="alias"):
        SettlementContractConfig.model_validate(contract(valid_from="2028-01-01"))
    with pytest.raises(ValueError, match="alias"):
        SettlementConfig.model_validate({"meter id": "one", "meter_id": "two"})
    with pytest.raises(ValueError, match="alias"):
        ProductionBonusConfig.model_validate({"cap basis": "ex_vat", "cap_basis": "inc_vat"})
    with pytest.raises(ValueError, match="alias"):
        PricingConfig.model_validate(pricing(**{"vat production": {"2027-01-01": 0},
                                               "vat_production": {"2027-01-01": 21}}))
    with pytest.raises(ValueError, match="overlap"):
        SettlementConfig.model_validate({"contracts": [contract(), contract(**{
            "id": "next", "valid from": "2027-02-01",
        })]})


def test_equal_aliases_are_safe_and_do_not_mutate_the_document():
    data = contract(valid_from="2027-01-01")
    original = deepcopy(data)
    parsed = SettlementContractConfig.model_validate(data)
    assert parsed.valid_from == date(2027, 1, 1)
    assert data == original


def test_adjacent_contracts_are_halfopen_and_keep_contract_identity():
    cfg = SettlementConfig.model_validate({"contracts": [
        contract(**{"valid to": "2027-06-01"}),
        contract(**{"valid from": "2027-06-01"}),
    ]})
    assert [item.id for item in cfg.contracts] == ["synthetic-2027", "synthetic-2027"]


def test_draft_is_not_active():
    terms = contract(status="draft", **{"supply start": None, "hour price rule": "unconfirmed"})
    draft_config = configured(terms, mode="shadow").settlement
    assert draft_config.mode == "shadow"
    assert draft_config.contracts[0].status == "draft"
    with pytest.raises(ValueError, match="draft"):
        configured(terms)


@pytest.mark.parametrize("rule", ["native", "mean_quarters"])
def test_confirmed_hour_price_rule_is_explicit(rule):
    cfg = configured(contract(**{"hour price rule": rule}))
    assert cfg.settlement.contracts[0].hour_price_rule == rule


def test_active_requires_a_meter_and_contract():
    for data in ({"mode": "active"}, {"mode": "active", "contracts": [contract()]}):
        with pytest.raises(ValueError):
            SettlementConfig.model_validate(data)


@pytest.mark.parametrize("overrides", [
    {"valid to": "2027-01-01"}, {"supply start": "2027-01-02"},
    {"valid from": "2027-01-01T00:00:00"}, {"timezone": "Not/AZone"},
])
def test_invalid_contract_boundaries_rejected(overrides):
    with pytest.raises(ValueError):
        SettlementContractConfig.model_validate(contract(**overrides))


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "0.1234567890123", "1e1000"])
def test_confirmed_rejects_nonfinite_or_overprecise_tariffs(value):
    with pytest.raises(ValueError):
        configured(**{"cost supplier consumption": {"2027-01-01": value}})


def test_settlement_requires_unambiguous_iso_tariff_dates():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        configured(**{"cost supplier consumption": {"20270101": "0.1"}})


def test_bonus_metadata_and_decimal_vat_are_available_to_docs():
    bonus = ProductionBonusConfig.model_validate({"cap basis": "inc_vat", "vat percent": "21"})
    assert bonus.vat_percent == Decimal("21")
    for model in (SettlementConfig, SettlementContractConfig, ProductionBonusConfig):
        for field in model.model_fields.values():
            assert field.description
            assert field.json_schema_extra["x-help"]
            assert field.json_schema_extra["x-ui-section"]
