"""Explicit settlement handles must not disappear behind frozen dataclasses."""

import json
from decimal import Decimal

import mip

from dao.prog import da_debug, day_ahead
from dao.tests.settlement import conftest as settlement_fixtures
from dao.tests.settlement.test_optimization_audit import (
    financial_case,
    override_solution,
    run_audit,
)


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


def test_debug_presents_finalization_and_only_canonical_money(monkeypatch):
    """The initial evidence and raw eligible residue cannot masquerade as final earnings."""
    snapshot = settlement_fixtures.snapshot.__wrapped__()
    bonus_rules = settlement_fixtures.bonus_rules.__wrapped__()
    model, terms = financial_case(snapshot, bonus_rules)
    cell = terms.bonus.source.cell_discharge.bus_power_vars[0][0]
    raw_get, optimize = model.solver.var_get_x, model.optimize
    override_solution(monkeypatch, model, {cell.idx: 1e-13})

    def resolve():
        monkeypatch.setattr(model.solver, "var_get_x", raw_get)
        status = optimize()
        override_solution(monkeypatch, model, {terms.bonus.eligible_kwh[0].idx: 1.0000000005})
        return status

    monkeypatch.setattr(model, "optimize", resolve)
    assert hasattr(day_ahead, "finalize_day_ahead_finance")
    selected = day_ahead.finalize_day_ahead_finance(terms,
        cbc_log="Welcome to the CBC MILP Solver Version: 2.10.12")
    result = day_ahead.finish_day_ahead_finance(selected, org_l=(0.0,), org_t=(-1.0,))
    assert terms.bonus.eligible_kwh[0].x == 1.0000000005
    payload = da_debug._build_result_dict(model, [], {}, financial_terms=selected.terms,
        planning_audit=result, finalized_solution=selected)
    diag = payload["settlement"]
    final = diag["finalization"]
    assert final["model_digest"] == selected.model_digest
    assert final["solution_digest"] == selected.solution_digest
    assert final["evidence"]["cbc_version"] is None
    refinement = final["refinement"]
    assert refinement["policy_version"] == "fixed_choice_refine_v1"
    assert refinement["initial_evidence"]["cbc_version"] == "2.10.12"
    assert refinement["final_evidence"] == final["evidence"]
    assert Decimal(refinement["objective_delta"]) == 0
    fix = next(item for item in refinement["fixes"] if item["var_index"] == cell.idx)
    assert fix["original_value"] == "1E-13"
    assert fix["fixed_value"] == "0"
    assert fix["reason"] == "source_zero"
    assert diag["bonus_kwh"] == ["1.000000000"]
    assert Decimal(diag["bonus_revenue_eur"][0]) == Decimal("0.03")
    assert Decimal(diag["energy_cost_eur"]) == Decimal("-0.13")
    assert any(issue.startswith("planning_auxiliary_residual:") for issue in diag["issues"])
    assert "refined_original_optimality_unproven" in diag["issues"]
    assert "synthetic-meter" not in json.dumps(diag)
