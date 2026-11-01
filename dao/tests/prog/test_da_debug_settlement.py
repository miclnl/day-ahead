"""Explicit settlement handles must not disappear behind frozen dataclasses."""

import json
from decimal import Decimal
from types import SimpleNamespace

import mip
import pytest

from dao.prog import da_debug, day_ahead
from dao.tests.settlement import conftest as settlement_fixtures
from dao.tests.settlement.test_optimization_audit import (
    financial_case,
    override_solution,
    run_audit,
)
from dao.tests.settlement.test_optimization_periods import audited_period_case, period_audit


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


def test_debug_registers_physical_cell_direction_separately():
    model = mip.Model(solver_name=mip.CBC)
    charge = tuple(tuple(model.add_var(ub=3) for _ in range(3)) for _ in range(2))
    discharge = tuple(tuple(model.add_var(ub=7) for _ in range(3)) for _ in range(2))
    direction = day_ahead._bind_battery_cell_direction(model, dc_to_bat=charge, dc_from_bat=discharge,
        installation_battery_count=2, interval_count=3)
    registry = da_debug.build_var_registry({"cell_charge_on": direction})
    for b, row in enumerate(direction):
        for i, var in enumerate(row):
            assert registry.get(var.idx) == ("cell_charge_on", (b, i))
    assert da_debug.SHAPES["cell_charge_on"][:2] == (("b", "u"), "bool")


def test_debug_keeps_net_economics_separate_and_unknown_amounts_null():
    snapshot = settlement_fixtures.snapshot.__wrapped__()
    bonus_rules = settlement_fixtures.bonus_rules.__wrapped__()
    model, terms = financial_case(snapshot, bonus_rules)
    audit = run_audit(model, terms)
    economics = {"net_model_advantage_eur": None, "battery_wear_eur": Decimal("0.0225"),
                 "estimated": True, "economics_supported": False,
                 "issues": ("planning_economics_incomplete",), "constraints": None}
    payload = da_debug._build_result_dict(model, [], {}, financial_terms=terms, planning_audit=audit,
                                         planning_economics=economics)
    assert payload["settlement"]["savings_energy_eur"] == str(audit.savings_energy_eur)
    shown = payload["settlement"]["economics"]
    assert shown["net_model_advantage_eur"] is None
    assert shown["battery_wear_eur"] == "0.0225"
    assert shown["estimated"] is True and shown["economics_supported"] is False
    assert "NaN" not in json.dumps(payload)


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
    assert refinement["policy_version"] == "fixed_choice_refine_v2"
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


@pytest.mark.parametrize("mode,engine", [("closed", "0.1186"), ("unknown", None), ("open", "0")])
def test_period_debug_keeps_model_credit_separate_from_engine_credit(mode, engine):
    """Omitted or conservative planner credit must not masquerade as a paid engine amount."""
    snapshot = settlement_fixtures.snapshot.__wrapped__()
    model, terms, periods, selected, audit = audited_period_case(snapshot, mode=mode)
    result = period_audit(terms, periods, audit)
    legacy = da_debug._build_result_dict(model, [], {}, financial_terms=terms, planning_audit=audit,
                                         finalized_solution=selected)
    payload = da_debug._build_result_dict(model, [], {}, financial_terms=terms, planning_audit=audit,
        finalized_solution=selected, period_terms=periods, period_audit=result)
    diag = payload["settlement"].pop("periods")
    assert payload == legacy
    assert diag["policy"] == "vat_lower_bound_v1"
    assert diag["exact"] is False
    assert (None if diag["engine_closed_credit_eur"] is None else Decimal(diag["engine_closed_credit_eur"])) == (
        None if engine is None else Decimal(engine))
    assert Decimal(diag["model_credit_eur"]) == pytest.approx(
        Decimal("0.109") if mode == "closed" else Decimal("0"), abs=Decimal("0.000001"))
    assert diag["credit_unit"] == "EUR gross"
    assert diag["estimated"] is True
    assert diag["economics_supported"] is False
    if mode == "closed":
        assert Decimal(diag["selected_trajectory_credit_difference_eur"]) == pytest.approx(
            Decimal("0.0096"), abs=Decimal("0.000001"))
        assert Decimal(diag["baseline_engine_closed_credit_eur"]) == Decimal("0.0605")
        assert diag["interval_bound_proven"] is True
    else:
        assert diag["selected_trajectory_credit_difference_eur"] is None
        assert diag["interval_bound_proven"] is False
    assert "conservative_objective_not_global_original_optimum" in diag["limitations"]
    assert "terminal_export_mean_v1_heuristic" in diag["limitations"]
    assert "synthetic-meter" not in json.dumps(diag)


def test_period_debug_flows_through_both_existing_capture_replay_callsites(tmp_path):
    """Real capture/replay serializers must retain the separately audited period result."""
    snapshot = settlement_fixtures.snapshot.__wrapped__()
    model, terms, periods, selected, audit = audited_period_case(snapshot, mode="unknown")
    result = period_audit(terms, periods, audit)
    primary = SimpleNamespace(_debug_financial_terms=terms, _debug_planning_audit=audit,
        _debug_finalized_solution=selected, _debug_period_terms=periods, _debug_period_audit=result)
    recording = da_debug.RecordingIO(out_dir=tmp_path)
    recording._primary, recording._model = primary, model
    recording._write_result("synthetic")
    captured = json.loads(recording.result_path.read_text())
    replay = da_debug.ReplayIO.__new__(da_debug.ReplayIO)
    replay._primary, replay._model, replay._cbc_log_chunks = primary, model, []
    replay._recorded_solver_threads, replay._solver_threads_override, replay._source = 1, None, "synthetic"
    assert replay.build_result()["settlement"]["periods"] == captured["settlement"]["periods"]
    assert captured["settlement"]["periods"]["engine_closed_credit_eur"] is None
