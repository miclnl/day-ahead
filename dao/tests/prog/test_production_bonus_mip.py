"""production_bonus.add_to_model(): the MIP building block that calc_optimum() uses.

calc_optimum() needs Home Assistant, so the model around the block is built by
hand: two intervals, the first inside the bonus window, PV only in the first,
a battery that may discharge in either, and a base export price that makes the
battery prefer the first interval.
"""

import pytest

pytest.importorskip("mip")
from mip import CONTINUOUS, Model, maximize, xsum  # noqa: E402

from dao.prog import production_bonus  # noqa: E402
from dao.prog.config.models.production_bonus import ProductionBonusConfig  # noqa: E402

PV_KWH = [2.0, 0.0]
BATTERY_KWH = 3.0
PB = [0.05, 0.0]
PT = [0.10, 0.09]
MAX_KWH = [20.0, 20.0]


def solve(mode, remaining):
    cfg = ProductionBonusConfig(enabled=True, mode=mode)
    model = Model()
    model.verbose = 0
    c_t = [model.add_var(var_type=CONTINUOUS, lb=0, ub=MAX_KWH[u]) for u in range(2)]
    bat_out = [model.add_var(var_type=CONTINUOUS, lb=0, ub=BATTERY_KWH) for u in range(2)]
    model += xsum(bat_out) <= BATTERY_KWH
    for u in range(2):
        model += c_t[u] == PV_KWH[u] + bat_out[u]

    c_t_bonus, bonus_revenue = production_bonus.add_to_model(
        model, cfg, c_t, PB, PV_KWH, MAX_KWH, remaining
    )

    model.objective = maximize(xsum(c_t[u] * PT[u] for u in range(2)) + bonus_revenue)
    model.optimize()
    return [v.x for v in c_t], [v.x for v in c_t_bonus]


def test_pv_only_gives_no_bonus_to_battery_export():
    c_t, c_t_bonus = solve("pv only", None)
    assert c_t[0] == pytest.approx(5.0)  # PV 2 + battery 3 exported in the window
    assert c_t_bonus[0] == pytest.approx(2.0)  # only the PV part earns the bonus


def test_all_export_gives_the_bonus_to_everything_in_the_window():
    c_t, c_t_bonus = solve("all export", None)
    assert c_t[0] == pytest.approx(5.0)
    assert c_t_bonus[0] == pytest.approx(5.0)


def test_the_remaining_cap_limits_the_bonus_kwh():
    _, c_t_bonus = solve("all export", 1.0)
    assert sum(c_t_bonus) == pytest.approx(1.0)


def test_an_exhausted_cap_leaves_no_bonus():
    _, c_t_bonus = solve("all export", 0.0)
    assert sum(c_t_bonus) == pytest.approx(0.0)


def test_outside_the_window_the_variable_is_fixed_at_zero():
    c_t, c_t_bonus = solve("all export", None)
    assert c_t[1] == pytest.approx(0.0)
    assert c_t_bonus[1] == pytest.approx(0.0)
