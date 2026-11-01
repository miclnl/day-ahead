"""Conservative gross month credits and explicit terminal units, using real CBC."""

import math
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
from decimal import Decimal, localcontext
from fractions import Fraction
from zoneinfo import ZoneInfo

import mip
import pytest

from dao.prog import day_ahead
from dao.prog.fastctrl.plan import FinancialBasis
from dao.prog.settlement import optimization_audit
from dao.prog.settlement.engine import settle_series
from dao.prog.settlement.optimization import OptimizationInputError
from dao.prog.settlement.service import PlanningContext
from dao.prog.settlement.tariffs import quote
from dao.prog.settlement.types import (
    InputInterval,
    MonthState,
    PriceBatch,
    RegisteredSpot,
    SettlementState,
    SpotInterval,
    TerminalValue,
)

D = Decimal
ZONE = ZoneInfo("Europe/Amsterdam")
OMITTED = "monthly_credit_omitted_open_or_unknown"
MIXED = "monthly_credit_conservative_mixed_vat"


def ts(day, hour=0, minute=0):
    return int(datetime.fromisoformat(day).replace(hour=hour, minute=minute, tzinfo=ZONE).timestamp())


def prefix(start, *, revenue="0", groups=(), complete=True):
    return SettlementState(start, None, MonthState(
        "synthetic-2027", ts("2027-01-01"), ts("2027-02-01"),
        None if revenue is None else D(revenue), tuple((D(r), D(n)) for r, n in groups), complete,
    ))


def model_grid(prices, amounts, bounds=None):
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    bounds = (D("1"),) * len(prices) if bounds is None else bounds
    grid = tuple(model.add_var(ub=float(bound), name=f"export_{i}")
                 for i, bound in enumerate(bounds))
    for var, amount in zip(grid, amounts, strict=True):
        if amount is not None:
            model += var == float(D(amount))
    return model, grid, bounds


def build(prices, amounts, opening, snapshots, *, bounds=None):
    from dao.prog.settlement.optimization import build_month_floor_terms

    model, grid, bounds = model_grid(prices, amounts, bounds)
    terms = build_month_floor_terms(model, prices, grid, bounds, opening, snapshots=snapshots)
    return model, terms


def solve(model, terms, objective="constant"):
    expression = (terms.credit + terms.credit_upper
                  + mip.xsum(period.credit_ex_vat for period in terms.periods))
    if objective == "constant":
        expression = mip.xsum([])
    model.objective = mip.maximize(expression) if objective == "max" else mip.minimize(expression)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL


def replay(prices, amounts, opening, snapshots):
    items = tuple(InputInterval(
        start_ts=price.start_ts, end_ts=price.end_ts, import_kwh=D("0"),
        export_kwh=D(amount), battery_discharge_kwh=D("0"), origin="verified_pv",
        quality="estimated", sources=("synthetic-meter",), spot_ex_vat=price.spot_ex_vat,
        spot_source="synthetic-price", input_hash="b" * 64, meter_id="synthetic-meter",
        input_revision=0,
    ) for price, amount in zip(prices, amounts, strict=True))
    return settle_series(items, snapshots, opening)


@pytest.mark.parametrize("objective", ["min", "max", "constant"])
def test_uniform_vat_gross_floor_is_exact(snapshot, objective):
    """Neither objective direction may inflate C or apply an hourly floor."""
    end = ts("2027-02-01")
    snap = snapshot(vat_production=((date(2027, 1, 1), D("21")),))
    prices = (quote(snap, end - 1800, end - 900, D("-0.20")),
              quote(snap, end - 900, end, D("0.05")))
    opening = prefix(prices[0].start_ts)
    model, terms = build(prices, ("1", "1"), opening, (snap,))
    solve(model, terms, objective)
    period, = terms.periods
    assert period.credit_ex_vat.x == pytest.approx(0.15)
    assert period.credit.x == terms.credit.x == pytest.approx(0.1815)
    assert period.credit_upper.x == terms.credit_upper.x == pytest.approx(0.1815)
    assert period.vat_bounds_percent == (D("21"), D("21"))
    assert period.exact is terms.exact is True
    assert terms.policy_version == "vat_lower_bound_v1"
    assert terms.issues == period.issues == ()
    assert period.interval_indices == (0, 1)
    assert (period.period_start_ts, period.period_end_ts, period.contract_id) == (
        ts("2027-01-01"), end, "synthetic-2027")
    assert replay(prices, ("1", "1"), opening, (snap,)).adjustments[0].credit_total == D("0.1815")
    for expression in (period.credit_ex_vat, period.credit, period.credit_upper,
                       terms.credit, terms.credit_upper):
        assert isinstance(expression, mip.LinExpr)
    with pytest.raises(FrozenInstanceError):
        period.exact = False
    with pytest.raises(FrozenInstanceError):
        terms.exact = False
    assert not hasattr(terms, "__dict__") and not hasattr(period, "__dict__")


@pytest.mark.parametrize("objective", ["min", "max", "constant"])
@pytest.mark.parametrize("x,lower,engine,upper", [
    ("0", "0.0545", "0.0605", "0.0605"),
    ("0.5", "0.1090", "0.1186", "0.1210"),
    ("1", "0.1635", "0.1755", "0.1815"),
])
def test_mixed_vat_prefix_credit_envelope(snapshot, objective, x, lower, engine, upper):
    """Prefix VAT and all possible horizon VAT remain in the pre-solve envelope."""
    end = ts("2027-02-01")
    snap = snapshot(vat_production=((date(2027, 1, 1), D("9")),),
                    vat_consumption=((date(2027, 1, 1), D("50")),))
    prices = (quote(snap, end - 1800, end - 900, D("-0.10")),
              quote(snap, end - 900, end, D("0.15")))
    opening = prefix(prices[0].start_ts, revenue="-0.20", groups=(("21", "0.20"),))
    model, terms = build(prices, (x, "1"), opening, (snap,))
    solve(model, terms, objective)
    period, = terms.periods
    assert period.credit_ex_vat.x == pytest.approx(float({"0": D("0.05"), "0.5": D("0.10"), "1": D("0.15")}[x]))
    assert period.credit.x == terms.credit.x == pytest.approx(float(D(lower)))
    assert period.credit_upper.x == terms.credit_upper.x == pytest.approx(float(D(upper)))
    assert period.vat_bounds_percent == (D("9"), D("21"))
    assert period.exact is terms.exact is False
    assert period.issues == terms.issues == (MIXED,)
    actual = replay(prices, (x, "1"), opening, (snap,)).adjustments[0].credit_total
    assert actual == D(engine)
    assert D(lower) <= actual <= D(upper)


@pytest.mark.parametrize("case,exact,issues", [
    ("open", False, (OMITTED,)), ("unknown", False, (OMITTED,)),
    ("missing", False, (OMITTED,)), ("disabled", True, ()),
    ("structural_zero", True, ()), ("mixed_solved_zero", False, (MIXED,)),
])
@pytest.mark.parametrize("objective", ["min", "max", "constant"])
def test_floor_open_unknown_disabled_and_zero(snapshot, case, exact, issues, objective):
    """Omitted credit is not an exact zero or an upper-bound proof."""
    end = ts("2027-02-01") if case != "open" else ts("2027-01-16")
    snap = snapshot(monthly_export_floor="none" if case == "disabled" else "spot_zero",
                    vat_production=((date(2027, 1, 1), D("9")),))
    prices = (quote(snap, end - 1800, end - 900, D("-0.10")),
              quote(snap, end - 900, end, D("0.30")))
    opening = prefix(prices[0].start_ts, revenue="-0.20", groups=(("21", "0.20"),))
    if case in ("unknown", "disabled"):
        opening = prefix(prices[0].start_ts, revenue=None, complete=False)
    elif case == "missing":
        opening = replace(opening, month=None)
    elif case == "structural_zero":
        opening = prefix(prices[0].start_ts, revenue="1", groups=(("21", "0.20"),))
    model, terms = build(prices, ("0", "1"), opening, (snap,))
    solve(model, terms, objective)
    period, = terms.periods
    assert period.credit_ex_vat.x == period.credit.x == period.credit_upper.x == pytest.approx(0)
    assert terms.credit.x == terms.credit_upper.x == pytest.approx(0)
    assert period.exact is terms.exact is exact
    assert period.issues == terms.issues == issues
    if case != "mixed_solved_zero":
        assert period.vat_bounds_percent is None
    else:
        assert period.vat_bounds_percent == (D("9"), D("21"))


@pytest.mark.parametrize("case", ["insufficient", "duplicate", "negative_vat", "negative_basis",
                                  "wrong_contract", "wrong_month", "wrong_end"])
def test_invalid_negative_basis_prefix_rejected(snapshot, case):
    """Inconsistent known prefixes must not be repaired or turned into missing history."""
    end = ts("2027-02-01")
    snap = snapshot()
    prices = (quote(snap, end - 900, end, D("-0.10")),)
    opening = prefix(end - 900, revenue="-0.20", groups=(("21", "0.20"),))
    if case == "insufficient":
        opening = prefix(end - 900, revenue="-0.20", groups=(("21", "0.10"),))
    elif case == "duplicate":
        opening = prefix(end - 900, revenue="-0.20", groups=(("21", "0.10"), ("21", "0.10")))
    elif case == "negative_vat":
        opening = prefix(end - 900, revenue="-0.20", groups=(("-1", "0.20"),))
    elif case == "negative_basis":
        opening = prefix(end - 900, revenue="-0.20", groups=(("21", "-0.20"),))
    elif case == "wrong_contract":
        opening = replace(opening, month=replace(opening.month, contract_id="wrong-contract"))
    elif case == "wrong_end":
        opening = replace(opening, month=replace(opening.month, period_end_ts=ts("2027-03-01")))
    else:
        opening = replace(opening, month=replace(opening.month,
                          period_start_ts=ts("2026-12-01"), period_end_ts=ts("2027-01-01")))
    with pytest.raises(OptimizationInputError, match="planning_month_state_invalid"):
        build(prices, ("1",), opening, (snap,))


@pytest.mark.parametrize("case", ["closed_open", "contract_change", "tariff_revision", "unknown_new_month"])
def test_month_credit_calendar_segments_do_not_leak(snapshot, case):
    """Only true contract/month boundaries reset coverage and the spot prefix."""
    boundary = ts("2027-01-16") if case == "contract_change" else ts("2027-02-01")
    first = snapshot()
    opening = prefix(boundary - 900)
    amounts = ("1", "1")
    if case == "tariff_revision":
        boundary -= 900
        first = snapshot(valid_to_ts=boundary, vat_production=((date(2027, 1, 1), D("21")),))
        second = snapshot(snapshot_hash="c" * 64, valid_from_ts=boundary,
                          vat_production=((date(2027, 1, 1), D("9")),))
        prices = (quote(first, boundary - 900, boundary, D("-0.20")),
                  quote(second, boundary, boundary + 900, D("0.05")))
        opening = prefix(boundary - 900)
        snapshots = (first, second)
        expected = (0.1815,)
        exact = (True,)
        indices = ((0, 1),)
    elif case == "contract_change":
        first = snapshot(valid_to_ts=boundary)
        second = snapshot(snapshot_hash="c" * 64, contract_id="next-contract",
                          valid_from_ts=boundary, supply_start=date(2027, 1, 16))
        prices = (quote(first, boundary - 900, boundary, D("-0.20")),
                  quote(second, boundary, boundary + 900, D("-0.10")))
        snapshots = (first, second)
        expected, exact, indices = (0.20, 0), (True, False), ((0,), (1,))
    else:
        following = ts("2027-03-01") if case == "unknown_new_month" else boundary + 900
        prices = (quote(first, boundary - 900, boundary, D("-0.20")),
                  quote(first, boundary, following, D("-0.10")))
        if case == "unknown_new_month":
            # One whole-month interval isolates calendar coverage from source/bonus grids.
            opening = prefix(boundary - 900, revenue=None, complete=False)
            expected, exact = (0, 0.10), (False, True)
        else:
            expected, exact = (0.20, 0), (True, False)
        snapshots, indices = (first,), ((0,), (1,))
    model, terms = build(prices, amounts, opening, snapshots)
    solve(model, terms, "max")
    assert tuple(period.credit.x for period in terms.periods) == pytest.approx(expected)
    assert tuple(period.exact for period in terms.periods) == exact
    assert tuple(period.interval_indices for period in terms.periods) == indices
    assert terms.credit.x == terms.credit_upper.x == pytest.approx(sum(expected))
    assert terms.exact is all(exact)
    result = replay(prices, amounts, opening, snapshots)
    assert sum((a.credit_total for a in result.adjustments), D("0")) == D(str(sum(expected)))
    for period in terms.periods:
        for i in period.interval_indices:
            month = result.lines[i].after_state.month
            assert (period.contract_id, period.period_start_ts, period.period_end_ts) == (
                month.contract_id, month.period_start_ts, month.period_end_ts)


@pytest.mark.parametrize("changed_contract", [True, False])
def test_horizon_end_contract_change_uses_pinned_adjacent_snapshot(snapshot, changed_contract):
    """A next snapshot outside the quote horizon distinguishes closure from revision."""
    end = ts("2027-01-16")
    first = snapshot(valid_to_ts=end)
    adjacent = snapshot(snapshot_hash="c" * 64, valid_from_ts=end,
                        contract_id="next-contract" if changed_contract else first.contract_id)
    snapshots = (adjacent, first)  # The engine also accepts pinned snapshots out of order.
    prices = (quote(first, end - 900, end, D("-0.20")),)
    opening = prefix(end - 900)
    model, terms = build(prices, ("1",), opening, snapshots)
    solve(model, terms, "max")
    assert terms.credit.x == terms.credit_upper.x == pytest.approx(0.20 if changed_contract else 0)
    assert terms.exact is changed_contract
    result = replay(prices, ("1",), opening, snapshots)
    if changed_contract:
        assert result.adjustments[0].credit_total == D("0.20")
        assert terms.periods[0].period_end_ts == end
    else:
        assert result.adjustments == ()
        assert terms.periods[0].period_end_ts == ts("2027-02-01")
        assert terms.issues == (OMITTED,)


@pytest.mark.parametrize("month,following,quarters", [
    ("2027-03-01", "2027-04-01", 2972), ("2027-10-01", "2027-11-01", 2980),
])
def test_month_credit_uses_local_dst_calendar(snapshot, month, following, quarters):
    snap = snapshot()
    start, end = ts(month), ts(following)
    prices = (quote(snap, start, end, D("-0.20")),)
    opening = SettlementState(start, None, None)
    model, terms = build(prices, ("1",), opening, (snap,))
    solve(model, terms)
    period, = terms.periods
    assert (period.period_start_ts, period.period_end_ts) == (start, end)
    assert end - start == quarters * 900
    assert period.credit.x == pytest.approx(0.20) and terms.exact is True


@pytest.mark.parametrize("case,code", [
    ("quote_snapshot_mismatch", "planning_quote_invalid"),
    ("forged_quote", "planning_quote_invalid"), ("missing_snapshot", "planning_quote_invalid"),
    ("overlap", "planning_quote_invalid"), ("duplicate_grid", "planning_grid_invalid"),
    ("foreign_grid", "planning_grid_invalid"), ("negative_grid", "planning_grid_invalid"),
    ("unbounded_grid", "planning_grid_invalid"), ("underbound", "planning_grid_invalid"),
    ("nonfinite_bound", "planning_grid_invalid"), ("gap", "planning_grid_invalid"),
    ("wrong_asof", "planning_grid_invalid"), ("month_crossing", "planning_grid_invalid"),
])
def test_period_quote_and_grid_binding_rejected(snapshot, case, code):
    from dao.prog.settlement.optimization import build_month_floor_terms

    end = ts("2027-02-01")
    snap = snapshot()
    prices = (quote(snap, end - 1800, end - 900, D("-0.20")),
              quote(snap, end - 900, end, D("0.05")))
    opening, snapshots = prefix(prices[0].start_ts), (snap,)
    model, grid, bounds = model_grid(prices, ("1", "1"))
    if case == "quote_snapshot_mismatch":
        snapshots = (replace(snap, snapshot_hash="c" * 64),)
    elif case == "forged_quote":
        prices = (replace(prices[0], export_rate=D("10")), prices[1])
    elif case == "missing_snapshot":
        snapshots = ()
    elif case == "overlap":
        snapshots = (snap, replace(snap, snapshot_hash="c" * 64))
    elif case == "duplicate_grid":
        grid = (grid[0], grid[0])
    elif case == "foreign_grid":
        foreign = mip.Model(solver_name=mip.CBC)
        grid = (foreign.add_var(ub=1), grid[1])
    elif case == "negative_grid":
        grid[0].lb = -1
    elif case == "unbounded_grid":
        grid[0].ub = math.inf
    elif case == "underbound":
        bounds = (D("0.5"), bounds[1])
    elif case == "nonfinite_bound":
        bounds = (D("Infinity"), bounds[1])
    elif case == "gap":
        prices = (quote(snap, end - 2700, end - 1800, D("-0.20")), prices[1])
        opening = prefix(prices[0].start_ts)
    elif case == "wrong_asof":
        opening = replace(opening, asof_ts=opening.asof_ts - 1)
    else:
        prices = (prices[0], quote(snap, end - 900, end + 900, D("0.05")))
    with pytest.raises(OptimizationInputError) as rejected:
        build_month_floor_terms(model, prices, grid, bounds, opening, snapshots=snapshots)
    assert rejected.value.code == code


def test_period_snapshots_argument_is_required(snapshot):
    from dao.prog.settlement.optimization import build_month_floor_terms

    end = ts("2027-02-01")
    prices = (quote(snapshot(), end - 900, end, D("-0.20")),)
    model, grid, bounds = model_grid(prices, ("1",))
    with pytest.raises(TypeError, match="snapshots"):
        build_month_floor_terms(model, prices, grid, bounds, prefix(end - 900))


def test_zero_weight_vat_groups_and_nonnegative_spots_do_not_widen_envelope(snapshot):
    end = ts("2027-02-01")
    snap = snapshot(vat_production=((date(2027, 1, 1), D("9")),))
    prices = (quote(snap, end - 1800, end - 900, D("-0.10")),
              quote(snap, end - 900, end, D("0.05")))
    opening = prefix(end - 1800, revenue="-0.20", groups=(("21", "0.20"), ("50", "0")))
    model, terms = build(prices, ("0", "1"), opening, (snap,), bounds=(D("0"), D("1")))
    solve(model, terms)
    assert terms.periods[0].vat_bounds_percent == (D("21"), D("21"))
    assert terms.credit.x == pytest.approx(0.1815) and terms.exact is True


def test_negative_horizon_production_vat_is_rejected(snapshot):
    end = ts("2027-02-01")
    snap = snapshot(vat_production=((date(2027, 1, 1), D("-1")),))
    prices = (quote(snap, end - 900, end, D("-0.20")),)
    with pytest.raises(OptimizationInputError, match="planning_month_state_invalid"):
        build(prices, ("1",), prefix(end - 900), (snap,))


def test_month_credit_bound_is_outward_for_decimal_and_solver_coefficients(snapshot):
    end = ts("2027-02-01")
    prices = (quote(snapshot(), end - 900, end, D("-0.123456789012")),)
    bound = D("0.333333333")
    model, terms = build(prices, ("0.333333333",), prefix(end - 900),
                         (prices[0].snapshot,), bounds=(bound,))
    period, = terms.periods
    credit_var, = period.credit_ex_vat.expr
    exact = Fraction(bound) * Fraction(D("0.123456789012"))
    solver = Fraction.from_float(float(bound)) * Fraction.from_float(float(D("0.123456789012")))
    assert Fraction.from_float(credit_var.ub) >= max(exact, solver)
    solve(model, terms, "max")
    assert period.credit.x == pytest.approx(float(exact))


def test_terminal_value_is_duration_weighted_and_context_independent(snapshot):
    """A simple mean or caller Decimal precision must not alter terminal value."""
    from dao.prog.settlement.optimization import terminal_value

    start = ts("2027-01-16", 12)
    snap = snapshot()
    prices = (quote(snap, start, start + 900, D("0.10")),
              quote(snap, start + 900, start + 3600, D("0.30")))
    with localcontext() as context:
        context.prec = 3
        result = terminal_value(prices, (0.8, 0.9))
    assert isinstance(result, TerminalValue)
    assert result.policy_version == "export_mean_v1"
    assert result.eur_per_ac_kwh == D("0.25")
    assert result.reference_efficiency_by_battery == (("0", D("0.8")), ("1", D("0.9")))


@pytest.mark.parametrize("spots,expected", [(('-0.20', '-0.10'), '0'), (('-0.20', '0.10'), '0')])
def test_terminal_negative_mean_is_zero_without_hourly_clipping(snapshot, spots, expected):
    from dao.prog.settlement.optimization import terminal_value

    start = ts("2027-01-16", 12)
    snap = snapshot()
    prices = tuple(quote(snap, start + i * 900, start + (i + 1) * 900, D(spot))
                   for i, spot in enumerate(spots))
    assert terminal_value(prices, (1.0,)).eur_per_ac_kwh == D(expected)


def test_terminal_value_uses_bonus_free_base_export_rate(snapshot, bonus_rules):
    from dao.prog.settlement.optimization import terminal_value

    start = ts("2027-01-16", 12)
    snap = snapshot(bonus=bonus_rules, cost_supplier_production=((date(2027, 1, 1), D("0.02")),),
                    vat_production=((date(2027, 1, 1), D("21")),))
    price = quote(snap, start, start + 900, D("0.10"))
    assert price.bonus_rate > 0
    assert terminal_value((price,), (0.8,)).eur_per_ac_kwh == D("0.1452")
    without_bonus = quote(replace(snap, bonus=None), start, start + 900, D("0.10"))
    assert terminal_value((price,), (0.8,)) == terminal_value((without_bonus,), (0.8,))


def test_terminal_reference_converts_cell_to_ac_once_and_equal_end_energy_has_equal_value(snapshot):
    """The helper must not pre-discount an AC rate by the stored reference efficiency."""
    from dao.prog.settlement.optimization import terminal_value

    start = ts("2027-01-16", 12)
    prices = (quote(snapshot(), start, start + 900, D("0.20")),)
    result = terminal_value(prices, (0.8,))
    assert result.eur_per_ac_kwh == D("0.20")
    assert result.reference_efficiency_by_battery == (("0", D("0.8")),)
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    end = tuple(model.add_var(ub=2) for _ in range(2))
    model += end[0] == 1
    model += end[1] == 1
    rate = float(result.eur_per_ac_kwh * result.reference_efficiency_by_battery[0][1])
    values = tuple(var * rate for var in end)
    model.objective = mip.minimize(values[0] - values[1])
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert [value.x for value in values] == pytest.approx([0.16, 0.16])


@pytest.mark.parametrize("efficiency", [0.0, -0.1, 1.1, math.inf, math.nan])
def test_terminal_invalid_reference_efficiency_is_rejected(snapshot, efficiency):
    from dao.prog.settlement.optimization import terminal_value

    start = ts("2027-01-16", 12)
    with pytest.raises(ValueError):
        terminal_value((quote(snapshot(), start, start + 900, D("0.20")),), (efficiency,))


def financial_period_case(snapshot, *, amounts=("0.5", "1"), spots=("-0.10", "0.15"),
                          revenue="-0.20", groups=(("21", "0.20"),), mode="closed",
                          vat="9", snapshots=None):
    """Build the actual finance binding before any solve, with pinned hourly prices."""
    end = ts("2027-01-16") if mode == "open" else ts("2027-02-01")
    snap = snapshot(vat_production=((date(2027, 1, 1), D(vat)),),
                    monthly_export_floor="none" if mode == "disabled" else "spot_zero")
    if snapshots is not None:
        snap = snapshots[0]
    else:
        snapshots = (snap,)
    prices = tuple(quote(snap, end - (len(spots)-i)*3600, end - (len(spots)-i-1)*3600, D(spot))
                   for i, spot in enumerate(spots))
    first = prices[0].start_ts
    opening = prefix(first, revenue=None if mode == "unknown" else revenue,
                     groups=groups, complete=mode != "unknown")
    if mode == "missing":
        opening = replace(opening, month=None)
    return bind_financial_period_case(prices, amounts, opening, snapshots)


def bind_financial_period_case(prices, amounts, opening, snapshots):
    from dao.prog.settlement.optimization import build_month_floor_terms

    first, end = prices[0].start_ts, prices[-1].end_ts
    hashes = tuple(s.snapshot_hash for s in snapshots)
    basis = FinancialBasis("synthetic-meter", 1, hashes, ts("2027-01-01"),
                           first, end, first, first)
    raw = tuple(RegisteredSpot(SpotInterval(p.start_ts, p.end_ts, p.spot_ex_vat,
                                           "synthetic-price", "estimated"), 1, "b" * 64)
                for p in prices)
    batch = PriceBatch("synthetic-meter", 1, first, end, raw, tuple(snapshots), first, first)
    context = PlanningContext(basis, opening, prices, batch)
    model, exports, bounds = model_grid(prices, amounts)
    terms = day_ahead.bind_day_ahead_finance(model, context=context,
        intervals=tuple((p.start_ts, p.end_ts) for p in prices), meter_id="synthetic-meter",
        dc_from_bat=(), installation_battery_count=0, battery_indices=(),
        no_battery_evidence="synthetic inventory",
        hour_fractions=tuple(D(p.end_ts-p.start_ts)/D(3600) for p in prices),
        discharge_efficiencies=(), grid_import=tuple(model.add_var(ub=0) for _ in prices),
        grid_export=exports, export_bounds_kwh=bounds, origins=("verified_pv",)*len(prices),
        source_policy=None)
    periods = build_month_floor_terms(model, terms.quotes, terms.bonus.grid_export,
        terms.bonus.export_bounds_kwh, terms.bonus.opening, snapshots=context.price_batch.snapshots)
    return model, terms, periods


def audited_period_case(snapshot, **kwargs):
    model, terms, periods = financial_period_case(snapshot, **kwargs)
    model.objective = terms.energy_cost - periods.credit
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    selected = day_ahead.finalize_day_ahead_finance(terms, cbc_log="")
    audit = day_ahead.finish_day_ahead_finance(selected, org_l=(0.0, 0.0), org_t=(0.0, -1.0))
    return model, terms, periods, selected, audit


def test_net_economics_uses_each_canonical_credit_not_planner_credit(snapshot):
    """Mixed-VAT planner credit cannot be reused for either actual engine ledger."""
    from dao.prog.settlement.optimization_economics import planning_cost_breakdown
    _, _, periods, _, audit = audited_period_case(snapshot)
    assert periods.credit.x == pytest.approx(0.109)
    assert audit.forecast_result.adjustments[0].credit_total == D("0.1186")
    assert audit.baseline_result.adjustments[0].credit_total == D("0.0605")
    result = planning_cost_breakdown(forecast=audit.forecast_result, baseline=audit.baseline_result,
        battery_wear_eur=D("0"), battery_terminal_cost_eur=D("0"),
        preference_cost_eur=D("0"), other_terminal_cost_eur=D("0"))
    assert result["forecast_invoice_energy_eur"] == D("-0.2276")
    assert result["baseline_invoice_energy_eur"] == D("-0.2240")
    assert result["net_model_advantage_eur"] == D("0.0036")


def period_audit(terms, periods, audit):
    assert hasattr(optimization_audit, "audit_period_solution")
    return optimization_audit.audit_period_solution(terms, periods, planning_audit=audit)


def test_period_cost_binding_uses_lower_gross_credit_once(snapshot):
    """Subtracting the engine/upper credit or counting the lower credit twice is wrong."""
    model, terms, periods = financial_period_case(snapshot)
    cost = model.add_var(lb=-10, ub=10)
    day_ahead.bind_day_ahead_cost(model, cost=cost, legacy_energy_cost=mip.xsum([]) + 7,
        other_cost=mip.xsum([]) + 0.03, financial_terms=terms, period_terms=periods)
    model.objective = cost
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.energy_cost.x == pytest.approx(-0.109)
    assert periods.credit.x == pytest.approx(0.109)
    assert periods.credit_upper.x == pytest.approx(0.121)
    assert cost.x == pytest.approx(-0.188)


@pytest.mark.parametrize("direction,export,credit,cost_value", [
    ("min", 1, 0.0545, -0.0255), ("max", 0.5, 0, 0.0145),
])
def test_period_cost_free_export_crosses_zero_revenue(snapshot, direction, export, credit, cost_value):
    """The two-sided max remains valid when the objective chooses either side of R=0."""
    model, terms, periods = financial_period_case(snapshot, amounts=(None, "0"),
        spots=("-0.10", "0"), revenue="0.05", groups=())
    cost = model.add_var(lb=-10, ub=10)
    day_ahead.bind_day_ahead_cost(model, cost=cost, legacy_energy_cost=mip.xsum([]),
        other_cost=-0.08 * terms.bonus.grid_export[0], financial_terms=terms, period_terms=periods)
    model.objective = mip.minimize(cost) if direction == "min" else mip.maximize(cost)
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    assert terms.bonus.grid_export[0].x == pytest.approx(export)
    assert periods.credit.x == pytest.approx(credit)
    assert cost.x == pytest.approx(cost_value)


def test_period_cost_without_financial_binding_is_rejected(snapshot):
    model, _, periods = financial_period_case(snapshot)
    with pytest.raises(OptimizationInputError, match="planning_grid_invalid"):
        day_ahead.bind_day_ahead_cost(model, cost=model.add_var(lb=-10),
            legacy_energy_cost=mip.xsum([]), other_cost=mip.xsum([]),
            financial_terms=None, period_terms=periods)


def test_period_audit_accepts_conservatism_not_wrong_model(snapshot, monkeypatch):
    """The EUR 0.0096 gap is conservative credit, never A8 parity tolerance."""
    from dao.prog.settlement import optimization

    model, terms, periods, selected, audit = audited_period_case(snapshot)
    before = (day_ahead.model_digest(model), day_ahead.solution_digest(model))
    monkeypatch.setattr(optimization, "build_month_floor_terms", lambda *a, **k: pytest.fail("self audit"))
    monkeypatch.setattr(model, "optimize", lambda *a, **k: pytest.fail("post-selection solve"))
    result = period_audit(terms, periods, audit)
    assert result.model_credit_eur == pytest.approx(D("0.109"), abs=D("0.000001"))
    assert result.engine_closed_credit_eur == D("0.1186")
    assert result.exact is False and result.issues == (MIXED,)
    assert audit.parity_complete and audit.parity_error_eur <= D("0.000001")
    assert audit.model_energy_cost_eur == pytest.approx(D("-0.109"), abs=D("0.000001"))
    assert before == (selected.model_digest, selected.solution_digest)
    assert before == (day_ahead.model_digest(model), day_ahead.solution_digest(model))
    with pytest.raises(FrozenInstanceError):
        result.exact = True
    assert not hasattr(result, "__dict__")


@pytest.mark.parametrize("case", [
    "lower", "upper", "ex_vat", "total", "total_upper", "vat", "exact", "issues", "policy",
    "duplicate", "missing_period", "wrong_period", "wrong_indices", "engine_low", "engine_high",
    "engine_within_envelope_not_canonical", "missing_engine", "duplicate_engine", "wrong_engine_period",
    "baseline_forecast_credit", "forecast_binding", "export_bound", "quote_binding", "foreign_expression",
])
def test_period_audit_rejects_wrong_model_and_replay(snapshot, case):
    model, terms, periods, _, audit = audited_period_case(snapshot)
    period, = periods.periods
    if case in ("lower", "upper", "ex_vat"):
        field = {"lower": "credit", "upper": "credit_upper", "ex_vat": "credit_ex_vat"}[case]
        periods = replace(periods, periods=(replace(period, **{field: getattr(period, field) + 0.01}),))
    elif case in ("total", "total_upper"):
        field = "credit" if case == "total" else "credit_upper"
        periods = replace(periods, **{field: getattr(periods, field) + 0.01})
    elif case == "vat":
        periods = replace(periods, periods=(replace(period, vat_bounds_percent=(D("9"), D("50"))),))
    elif case == "exact":
        periods = replace(periods, exact=True, periods=(replace(period, exact=True),))
    elif case == "issues":
        periods = replace(periods, issues=(), periods=(replace(period, issues=()),))
    elif case == "policy":
        periods = replace(periods, policy_version="unapproved")
    elif case == "duplicate":
        periods = replace(periods, periods=(period, period))
    elif case == "missing_period":
        periods = replace(periods, periods=())
    elif case == "wrong_period":
        periods = replace(periods, periods=(replace(period, contract_id="wrong"),))
    elif case == "wrong_indices":
        periods = replace(periods, periods=(replace(period, interval_indices=(1, 0)),))
    elif case.startswith("engine_"):
        amount = {"engine_low": "0.108", "engine_high": "0.122",
                  "engine_within_envelope_not_canonical": "0.110"}[case]
        adj, = audit.forecast_result.adjustments
        adj = replace(adj, credit_total=D(amount), credit_vat=D(amount)-adj.credit_ex_vat)
        audit = replace(audit, forecast_result=replace(audit.forecast_result, adjustments=(adj,)))
    elif case in ("missing_engine", "duplicate_engine", "wrong_engine_period"):
        adj, = audit.forecast_result.adjustments
        adjustments = () if case == "missing_engine" else (adj, adj) if case == "duplicate_engine" else (
            replace(adj, period_start_ts=adj.period_start_ts+1),)
        audit = replace(audit, forecast_result=replace(audit.forecast_result, adjustments=adjustments))
    elif case == "baseline_forecast_credit":
        audit = replace(audit, baseline_result=replace(audit.baseline_result,
            adjustments=audit.forecast_result.adjustments))
    elif case == "forecast_binding":
        audit = replace(audit, forecast=(replace(audit.forecast[0], export_kwh=D("0")), audit.forecast[1]))
    elif case == "export_bound":
        terms = replace(terms, bonus=replace(terms.bonus, export_bounds_kwh=(D("0.1"), D("1"))))
    elif case == "quote_binding":
        terms = replace(terms, quotes=(replace(terms.quotes[0], spot_ex_vat=D("-0.2")), terms.quotes[1]))
    else:
        foreign = mip.Model(solver_name=mip.CBC)
        periods = replace(periods, credit=foreign.add_var(ub=1) * 1)
    with pytest.raises(optimization_audit.PlanningSolutionRejected, match="planning_period_parity_failed"):
        period_audit(terms, periods, audit)


def test_baseline_has_its_own_vat_weight(snapshot):
    """Forecast and baseline must not share the forecast's weighted credit."""
    _, terms, periods, _, audit = audited_period_case(snapshot)
    result = period_audit(terms, periods, audit)
    assert result.engine_closed_credit_eur == D("0.1186")
    assert audit.baseline_result.adjustments[0].credit_total == D("0.0605")
    assert audit.forecast_result.adjustments[0].credit_vat == D("0.0186")
    assert audit.baseline_result.adjustments[0].credit_vat == D("0.0105")
    assert audit.savings_energy_eur == D("-0.0545")
    assert audit.baseline_result.lines[0].before_state is terms.bonus.opening
    assert audit.forecast_result.lines[0].before_state is terms.bonus.opening


@pytest.mark.parametrize("mode,spot,revenue,vat,exact,engine", [
    ("open", "-0.10", "-0.20", "9", False, "0"),
    ("unknown", "-0.10", "-0.20", "9", False, None),
    ("missing", "-0.10", "-0.20", "9", False, None),
    ("disabled", "-0.10", "-0.20", "9", True, "0"),
    ("closed", "0", "0", "9", True, "0"),
    ("closed", "-0.10", "-0.20", "0", False, "0.292"),
    ("closed", "0", "-0.20", "9", True, "0.242"),
])
def test_period_audit_preserves_unknown_open_disabled_and_zero(snapshot, mode, spot, revenue, vat, exact, engine):
    _, terms, periods, _, audit = audited_period_case(snapshot, mode=mode, spots=(spot, "0"),
                                                    revenue=revenue, vat=vat)
    result = period_audit(terms, periods, audit)
    assert result.exact is exact
    assert result.engine_closed_credit_eur == (None if engine is None else D(engine))
    if mode in ("open", "unknown", "missing"):
        assert result.model_credit_eur == 0
        assert result.issues == (OMITTED,)


@pytest.mark.parametrize("spots,amounts,vat,exact,lower,engine", [
    (("-0.10", "0.15"), ("0.5", "1"), "21", True, "0.121", "0.121"),
    (("-0.10", "0.30"), ("0", "1"), "9", False, "0", "0"),
])
def test_period_audit_exactness_is_presolved_not_reclassified(snapshot, spots, amounts, vat, exact, lower, engine):
    _, terms, periods, _, audit = audited_period_case(snapshot, spots=spots, amounts=amounts, vat=vat)
    result = period_audit(terms, periods, audit)
    assert result.model_credit_eur == pytest.approx(D(lower), abs=D("0.000001"))
    assert result.engine_closed_credit_eur == D(engine)
    assert result.exact is exact


@pytest.mark.parametrize("changed_contract", [True, False])
def test_period_audit_calendar_uses_all_pinned_snapshots(snapshot, changed_contract):
    """The same quote and prefix close only with the pinned adjacent contract change."""
    end = ts("2027-01-16")
    first = snapshot(valid_to_ts=end)
    adjacent = snapshot(snapshot_hash="c"*64, valid_from_ts=end,
                        contract_id="next-contract" if changed_contract else first.contract_id)
    model, terms, periods, _, audit = audited_period_case(snapshot, mode="open",
        revenue="0", groups=(), spots=("-0.20", "0"), snapshots=(first, adjacent))
    result = period_audit(terms, periods, audit)
    assert result.model_credit_eur == pytest.approx(D("0.1") if changed_contract else D("0"))
    assert result.engine_closed_credit_eur == (D("0.1") if changed_contract else D("0"))
    assert result.exact is changed_contract
    assert day_ahead.model_digest(model)
    if changed_contract:
        # A coherent reduced context still cannot certify terms built with a
        # different calendar. Do not trust the builder's own period end.
        context = replace(terms.context, basis=replace(terms.context.basis, snapshot_hashes=(first.snapshot_hash,)),
                          price_batch=replace(terms.context.price_batch, snapshots=(first,)))
        changed = replace(terms, context=context)
        with pytest.raises(optimization_audit.PlanningSolutionRejected, match="planning_period_parity_failed"):
            period_audit(changed, periods, audit)


@pytest.mark.parametrize("unknown", [True, False])
def test_period_audit_closed_then_open_month_never_double_counts(snapshot, unknown):
    end = ts("2027-02-01")
    snap = snapshot(vat_production=((date(2027, 1, 1), D("21")),))
    prices = (quote(snap, end-3600, end, D("-0.20")), quote(snap, end, end+3600, D("-0.10")))
    model, terms, periods = bind_financial_period_case(prices, ("1", "1"),
        prefix(end-3600, revenue=None if unknown else "0", complete=not unknown), (snap,))
    model.objective = terms.energy_cost - periods.credit
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    selected = day_ahead.finalize_day_ahead_finance(terms, cbc_log="")
    audit = day_ahead.finish_day_ahead_finance(selected, org_l=(0.0, 0.0), org_t=(-1.0, -1.0))
    result = period_audit(terms, periods, audit)
    assert result.model_credit_eur == pytest.approx(D("0") if unknown else D("0.242"))
    assert result.engine_closed_credit_eur == (None if unknown else D("0.242"))
    assert periods.periods[1].vat_bounds_percent is None
    assert result.issues == (OMITTED,)
    assert result.exact is False


@pytest.mark.parametrize("spot,changed,accepted", [("-2000", 1, True), ("-2000", 2, False), ("-3000", 1, False)])
def test_period_audit_normalization_budget_applies_per_component_and_horizon(snapshot, monkeypatch,
                                                                           spot, changed, accepted):
    """Two individually sub-budget normalization errors may not exceed the sum budget."""
    from dao.tests.settlement.test_optimization_audit import override_solution

    boundary, end = ts("2027-01-16", 23), ts("2027-01-17")
    first = snapshot(valid_to_ts=boundary)
    second = snapshot(snapshot_hash="c"*64, contract_id="next-contract", valid_from_ts=boundary, valid_to_ts=end)
    adjacent = snapshot(snapshot_hash="d"*64, contract_id="following-contract", valid_from_ts=end)
    prices = (quote(first, boundary-3600, boundary, D(spot)), quote(second, boundary, end, D(spot)))
    model, terms, periods = bind_financial_period_case(prices, ("0.5", "0.5"), prefix(boundary-3600),
                                                      (first, second, adjacent))
    model.objective = terms.energy_cost - periods.credit
    assert model.optimize() == mip.OptimizationStatus.OPTIMAL
    selected = day_ahead.finalize_day_ahead_finance(terms, cbc_log="")
    audit = day_ahead.finish_day_ahead_finance(selected, org_l=(0.0, 0.0), org_t=(-0.5, -0.5))
    # Controlled raw read perturbations, not claims about this CBC solution.
    # Each still normalizes to exactly the original 0.500000000 kWh.
    override_solution(monkeypatch, model, {var.idx: 0.50000000049 for var in terms.bonus.grid_export[:changed]})
    if accepted:
        result = period_audit(terms, periods, audit)
        assert result.engine_closed_credit_eur == D("2000")
    else:
        with pytest.raises(optimization_audit.PlanningSolutionRejected, match="planning_period_parity_failed"):
            period_audit(terms, periods, audit)
