"""Explicit settlement handles must not disappear behind frozen dataclasses."""

import json

import mip

from dao.prog import da_debug
from dao.tests.settlement import conftest as settlement_fixtures
from dao.tests.settlement.test_optimization_audit import financial_case, run_audit


def test_debug_registers_source_and_cap_without_var_equality(monkeypatch):
    snapshot = settlement_fixtures.snapshot.__wrapped__()
    bonus_rules = settlement_fixtures.bonus_rules.__wrapped__()
    model, terms = financial_case(snapshot, bonus_rules)
    result = run_audit(model, terms)
    _, unknown = financial_case(snapshot, bonus_rules, total=None)
    monkeypatch.setattr(mip.Var, "__eq__", lambda *args: (_ for _ in ()).throw(AssertionError("Var equality")))
    registry = da_debug.build_var_registry({"financial_terms": terms, "cost": terms.grid_import[0],
                                           "c_t_bonus": terms.bonus.eligible_kwh})
    for field, name in (("eligible_kwh", "settlement_eligible"), ("first_kwh", "settlement_first"),
                        ("cumulative_export_kwh", "settlement_C"), ("capped_export_kwh", "settlement_Q"),
                        ("cap_saturated", "settlement_cap_saturated"),
                        ("capped_opening_kwh", "settlement_Q_open"),
                        ("opening_cap_saturated", "settlement_opening_cap_saturated")):
        assert registry.get(getattr(terms.bonus, field)[0].idx) == (name, (0,))
        assert da_debug.SHAPES[name][0] == ("u",)
    assert registry.get(terms.bonus.source.discharge_on[0].idx) == ("settlement_source", (0,))
    assert da_debug.SHAPES["dc_from_bat"][2] == "gross discharge power on the DC bus"
    payload = da_debug._build_result_dict(model, [], {}, financial_terms=terms, planning_audit=result)
    assert {"objective_value", "status", "cbc_log", "note"} <= payload.keys()
    diag = payload["settlement"]
    assert diag["policy"] == "aggregate_cell_min_1wh_v1"
    assert diag["required"] == [True]
    assert diag["cell_unit"] == "kWh"
    assert diag["bus_unit"] == "kW"
    assert diag["minimum_cell_kwh"] == "0.001"
    assert diag["parity_complete"]
    assert "cbc_integer_native_readback_unavailable" in diag["evidence"]["limitations"]
    assert "planning_source_policy=aggregate_cell_min_1wh_v1" in diag["issues"]
    assert "synthetic-meter" not in json.dumps(diag)
    # Unknown cap has None helpers, not fake Vars or a fabricated state.
    assert da_debug.build_var_registry({"financial_terms": unknown})
