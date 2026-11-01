"""CBC must enforce cap chronology and both source directions without objective help."""

from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import mip
import pytest

from dao.prog.settlement.optimization import make_cell_discharge_input
from dao.prog.settlement.tariffs import quote
from dao.prog.settlement.types import BonusState, SettlementState

D = Decimal
POLICY = "aggregate_cell_min_1wh_v1"
ZONE = ZoneInfo("Europe/Amsterdam")


def ts(day="2027-01-01", hour=12, minute=0):
    return int(datetime.fromisoformat(day).replace(hour=hour, minute=minute, tzinfo=ZONE).timestamp())


def setup(snapshot, bonus_rules, *, cells=("0",), exports=("1",), total="0", starts=None,
          durations=None, bounds=None, policy=POLICY, spots=None, origins=None, rules=None,
          efficiencies=None):
    """Use fixed flows as constraints, not bounds that reveal future chosen cap use."""
    from dao.prog.settlement.optimization import build_bonus_terms
    n = len(exports)
    starts = starts or tuple(ts() + 900 * i for i in range(n))
    durations = durations or (900,) * n
    spots = spots or ("0.10",) * n
    rules = rules or replace(bonus_rules, windows=tuple((m, (0, 24)) for m in range(1, 13)))
    snap = snapshot(bonus=rules, supply_start=date(2027, 1, 1))
    quotes = tuple(quote(snap, start, start + duration, D(spot))
                   for start, duration, spot in zip(starts, durations, spots, strict=True))
    year = datetime.fromtimestamp(starts[0], ZONE).year
    opening = SettlementState(starts[0], BonusState(ts(f"{year}-01-01", 0),
        ts(f"{year + 1}-01-01", 0), starts[0], None if total is None else D(total)), None)
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    grid = tuple(model.add_var(ub=2, name=f"export_{i}") for i in range(n))
    for var, amount in zip(grid, exports, strict=True):
        if amount is not None:
            model += var == float(D(amount))
    if cells and isinstance(cells[0], str):
        cells = (cells,)
    efficiencies = efficiencies or (D("1"),) * len(cells)
    matrix = tuple(tuple(model.add_var(ub=float((bounds or (D("2"),) * n)[i] * efficiencies[b]) * 4,
                                        name=f"cell_{b}_{i}") for i in range(n))
                   for b in range(len(cells)))
    hours = tuple(D(str(duration / 3600)) for duration in durations)
    for b, (row, energy) in enumerate(zip(matrix, cells, strict=True)):
        for i, value in enumerate(energy):
            if value is not None:
                model += row[i] == float(D(value) * efficiencies[b] / hours[i])
    cell = make_cell_discharge_input(model, matrix, meter_id="synthetic-meter",
        installation_battery_count=len(cells), battery_indices=tuple(range(len(cells))),
        no_battery_evidence=None if cells else "synthetic battery-free installation",
        hour_fractions=hours, discharge_efficiencies=efficiencies)
    terms = build_bonus_terms(model, quotes, grid, cell, (D("2"),) * n, opening,
                              origins=origins or ("verified_pv",) * n, source_policy=policy)
    return model, terms


@pytest.mark.parametrize("objective", ["min", "max", "constant"])
@pytest.mark.parametrize("energy,expected", [("0", 1), ("0.001", 0), ("0.0015", 0)])
def test_exact_eligibility_independent_of_objective(snapshot, bonus_rules, objective, energy, expected):
    model, terms = setup(snapshot, bonus_rules, cells=(energy,))
    expression = terms.eligible_kwh[0] if objective != "constant" else mip.xsum([])
    model.objective = mip.maximize(expression) if objective == "max" else mip.minimize(expression)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.eligible_kwh[0].x == pytest.approx(expected)
    assert terms.revenue.x == pytest.approx(0.03 if expected else 0)
    assert terms.first_kwh[0].x == pytest.approx(1)


@pytest.mark.parametrize("cells,feasible", [
    (("0.0005",), False), (("0.0015",), True),
    ((("0.0004",), ("0.0006",)), True),
    ((("0.0002",), ("0.0003",)), False),
])
def test_gap_and_aggregate_not_lattice(snapshot, bonus_rules, cells, feasible):
    model, _ = setup(snapshot, bonus_rules, cells=cells)
    assert model.optimize() == (mip.OptimizationStatus.OPTIMAL if feasible
                               else mip.OptimizationStatus.INFEASIBLE)


def test_aggregate_source_uses_each_original_efficiency(snapshot, bonus_rules):
    model, terms = setup(snapshot, bonus_rules, cells=(("0.0004",), ("0.0006",)),
                         efficiencies=(D("0.8"), D("1")))
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.source.cell_discharge.total_kwh[0].x == pytest.approx(0.001)
    assert terms.eligible_kwh[0].x == 0


@pytest.mark.parametrize("upper", ["0", "0.0008", "0.001"])
@pytest.mark.parametrize("forced", [False, True])
def test_small_upper_bound_zero_only(snapshot, bonus_rules, upper, forced):
    energy = ("0.0004" if upper == "0.0008" else "0.001") if forced else "0"
    model, terms = setup(snapshot, bonus_rules, cells=(energy,), bounds=(D(upper),))
    feasible = not forced or upper == "0.001"
    assert model.optimize() == (mip.OptimizationStatus.OPTIMAL if feasible
                               else mip.OptimizationStatus.INFEASIBLE)
    if D(upper) < D("0.001"):
        assert terms.source.discharge_on == (None,)


@pytest.mark.parametrize("policy,code", [(None, "source_domain_policy_required"),
                                        ("typo", "source_domain_policy_invalid")])
def test_missing_or_invalid_policy(snapshot, bonus_rules, policy, code):
    with pytest.raises(ValueError) as rejected:
        setup(snapshot, bonus_rules, policy=policy)
    assert rejected.value.code == code


def test_nonrequired_none_policy_preserves_continuous_domain(snapshot, bonus_rules):
    model, terms = setup(snapshot, bonus_rules, cells=("0.0005",), spots=("-0.10",), policy=None)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.source.required == (False,)
    assert terms.source.discharge_on == (None,)
    assert terms.eligible_kwh[0].x == 0


def test_required_not_decided_by_solved_first(snapshot, bonus_rules):
    model, terms = setup(snapshot, bonus_rules, cells=("0", "0.0005"),
                         exports=("1", "1"), total="4999")
    assert terms.source.required == (True, True)
    assert model.optimize() == mip.OptimizationStatus.INFEASIBLE


def test_forged_zero_proxy_rejected(snapshot, bonus_rules):
    from dao.prog.settlement.optimization import build_bonus_terms
    model, terms = setup(snapshot, bonus_rules, cells=("0.001",))
    forged = replace(terms.source.cell_discharge, total_kwh=(mip.xsum([]),))
    with pytest.raises(ValueError) as rejected:
        build_bonus_terms(model, terms.quotes, terms.grid_export, forged,
            terms.export_bounds_kwh, terms.opening, origins=terms.origins, source_policy=POLICY)
    assert rejected.value.code == "cell_conversion_invalid"


@pytest.mark.parametrize("early", ["window", "cell", "negative"])
def test_all_export_consumes_cap_before_later_high_price(snapshot, bonus_rules, early):
    starts = (ts(hour=8, minute=45), ts(hour=9)) if early == "window" else (ts(), ts() + 900)
    model, terms = setup(snapshot, bonus_rules, cells=("0.001" if early == "cell" else "0", "0"),
        exports=("1", "1"), total="4999", starts=starts,
        spots=("-0.10" if early == "negative" else "0.01", "0.20"),
        rules=bonus_rules if early == "window" else None)
    model.objective = mip.maximize(terms.revenue)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.cumulative_export_kwh[1].x == pytest.approx(5001)
    # Raw CBC feasibility is not the strict postsolve source certificate.
    assert terms.first_kwh[1].x == pytest.approx(0, abs=model.infeas_tol)
    assert terms.eligible_kwh[1].x == pytest.approx(0, abs=model.infeas_tol)


def test_cap_crossing_and_reset(snapshot, bonus_rules):
    model, terms = setup(snapshot, bonus_rules, cells=("0", "0"), exports=("1", "1"),
        total="4999.5", starts=(ts("2027-12-31", 23, 45), ts("2028-01-01", 0)))
    model.objective = mip.minimize(mip.xsum(terms.eligible_kwh))
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert [var.x for var in terms.first_kwh] == pytest.approx([0.5, 1])
    assert [var.x for var in terms.cumulative_export_kwh] == pytest.approx([5000.5, 1])


@pytest.mark.parametrize("known", [True, False])
def test_partial_unknown_prefix_recovers_only_at_explicit_reset(snapshot, bonus_rules, known):
    model, terms = setup(snapshot, bonus_rules, cells=("0", "0", "0"), exports=("1", "1", "1"),
        total="4998" if known else None, starts=(ts("2027-12-31", 23, 35),
            ts("2027-12-31", 23, 45), ts("2028-01-01", 0)), durations=(600, 900, 900))
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.source.required == (False, known, True)
    assert terms.cap_known == (known, known, True)
    assert terms.eligible_kwh[0].x == 0
    assert terms.eligible_kwh[1].x == pytest.approx(1 if known else 0)
    assert terms.eligible_kwh[2].x == pytest.approx(1)


def test_hourly_grid_has_no_bonus_or_source_minimum(snapshot, bonus_rules):
    model, terms = setup(snapshot, bonus_rules, cells=("0.0005",), durations=(3600,), policy=None)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.source.required == (False,)
    assert terms.eligible_kwh[0].x == 0
    assert "optimizer_bonus_disabled_hourly_grid" in terms.issues


@pytest.mark.parametrize("cell,expected", [("0", 1), ("1", 0)])
def test_dc_pv_pass_through_and_house_load(snapshot, bonus_rules, cell, expected):
    model, terms = setup(snapshot, bonus_rules, cells=(cell,))
    # PV export with no cell use, or PV=house and export supplied by the cell.
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.eligible_kwh[0].x == expected


@pytest.mark.parametrize("case", ["forged_rate", "mismatched_duration", "crosses_contract",
                                  "foreign_grid", "negative_grid"])
def test_invalid_quote_or_grid_rejected_with_explicit_code(snapshot, bonus_rules, case):
    from dao.prog.settlement.optimization import build_bonus_terms
    model, terms = setup(snapshot, bonus_rules)
    prices, grid = terms.quotes, terms.grid_export
    expected = "planning_quote_invalid"
    if case == "forged_rate":
        prices = (replace(prices[0], bonus_rate=D("0.19")),)
    elif case == "mismatched_duration":
        prices = (replace(prices[0], end_ts=prices[0].start_ts + 7200),)
        expected = "planning_grid_invalid"
    elif case == "crosses_contract":
        invalid = replace(prices[0].snapshot, valid_to_ts=prices[0].start_ts + 300)
        prices = (replace(prices[0], snapshot=invalid),)
    elif case == "foreign_grid":
        other = mip.Model(solver_name=mip.CBC)
        grid = (other.add_var(ub=2),)
        expected = "planning_grid_invalid"
    else:
        grid[0].lb = -1
        expected = "planning_grid_invalid"
    with pytest.raises(ValueError) as rejected:
        build_bonus_terms(model, prices, grid, terms.source.cell_discharge,
            terms.export_bounds_kwh, terms.opening, origins=terms.origins, source_policy=POLICY)
    assert rejected.value.code == expected


def test_cap_change_preserves_uncapped_export(snapshot, bonus_rules):
    from dao.prog.settlement.optimization import build_bonus_terms
    model, old = setup(snapshot, bonus_rules, cells=("0", "0"), exports=("2", "1"), total="4999")
    changed = replace(old.quotes[1].snapshot, snapshot_hash="b" * 64,
                      bonus=replace(old.quotes[1].snapshot.bonus, yearly_cap_kwh=D("6000")))
    # A separate financial terms graph is safe here: both track identical exports.
    prices = (old.quotes[0], quote(changed, old.quotes[1].start_ts,
                                  old.quotes[1].end_ts, D("0.10")))
    terms = build_bonus_terms(model, prices, old.grid_export, old.source.cell_discharge,
        old.export_bounds_kwh, old.opening, origins=old.origins, source_policy=POLICY)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.cumulative_export_kwh[1].x == pytest.approx(5002)
    assert [v.x for v in terms.first_kwh] == pytest.approx([1, 1])
