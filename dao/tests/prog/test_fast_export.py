"""Integration test for the optimizer's hand-off to the fast control layer.

Exercises ``DaCalc.export_fast_plan`` the way ``calc_optimum`` calls it, with
stand-ins for the solver variables, so a change to either side of the contract
breaks a test instead of silently producing a plan the realtime loop cannot
use.
"""

import datetime
import json

import pytest

pytest.importorskip("mip", reason="the optimizer stack is not installed")
pytest.importorskip("matplotlib", reason="the optimizer stack is not installed")

from day_ahead import DaCalc  # noqa: E402
from dao.prog.fastctrl.plan import load_plan  # noqa: E402


class Var:
    """Stand-in for a mip variable: all the exporter reads is ``.x``."""

    def __init__(self, value):
        self.x = float(value)


class Battery:
    """Stand-in for a validated BatteryConfig."""

    def __init__(self):
        self.name = "Accu1"
        self.capacity = 28.0
        self.minimum_power = 1000
        self.entity_set_power_feedin = "input_number.feedin_grid"
        self.entity_set_operating_mode = "input_select.ess_operating_mode"
        self.entity_set_operating_mode_on = "Aan"
        self.entity_set_operating_mode_off = "Uit"
        self.entity_stop_inverter = "input_datetime.stop_victron"
        self.entity_actual_level = "sensor.ess_battery_soc"


@pytest.fixture
def exporter():
    """A DaCalc with only the attributes the exporter touches."""
    instance = DaCalc.__new__(DaCalc)
    instance.debug = False
    instance.interval_s = 3600
    instance.strategy = "minimize cost"
    instance.battery_options = [Battery()]
    return instance


def call(exporter, path, steps=4, **overrides):
    start = datetime.datetime(2026, 1, 15, 18, 0, 0)
    kwargs = dict(
        tijd=[start + datetime.timedelta(hours=i) for i in range(steps)],
        hour_fraction=[1.0] * steps,
        pl=[0.32, 0.46, 0.38, 0.28][:steps],
        pt=[0.08, 0.14, 0.11, 0.06][:steps],
        c_l=[Var(v) for v in (0.0, 0.0, 0.4, 2.5)][:steps],
        c_t=[Var(v) for v in (0.0, 0.0, 0.0, 0.0)][:steps],
        solar_hour_sum_opt=[0.0] * steps,
        ac_to_dc=[[Var(v) for v in (0.0, 0.0, 0.0, 2.0)][:steps]],
        ac_from_dc=[[Var(v) for v in (1.2, 3.0, 0.5, 0.0)][:steps]],
        soc=[[Var(v) for v in (80.0, 75.0, 64.0, 62.0, 69.0)][: steps + 1]],
        max_charge_power=[7.2],
        max_discharge_power=[7.2],
        kwh_cycle_cost=[0.011],
        lower_limit=[21],
        upper_limit=[98],
        eff_dc_to_bat=[0.93],
        eff_bat_to_dc=[0.94],
        published_battery=[
            {
                "power_w": -1200.0,
                "mode": "Aan",
                "stop_inverter": "2000-01-01 00:00:00",
            }
        ],
        p_avg=0.36,
        U=steps,
        B=1,
        path=path,
    )
    kwargs.update(overrides)
    exporter.export_fast_plan(**kwargs)
    return load_plan(path)


class TestExport:
    def test_a_readable_plan_is_produced(self, exporter, tmp_path):
        plan = call(exporter, str(tmp_path / "fast_plan.json"))
        assert plan is not None
        assert len(plan.intervals) == 4
        assert plan.interval_s == 3600
        assert plan.strategy == "minimize cost"
        assert plan.price_average == pytest.approx(0.36)

    def test_the_battery_specification_is_carried_over(self, exporter, tmp_path):
        plan = call(exporter, str(tmp_path / "fast_plan.json"))
        spec = plan.specs[0]
        assert spec.name == "Accu1"
        assert spec.capacity_kwh == 28.0
        assert spec.max_charge_w == 7200.0
        assert spec.max_discharge_w == 7200.0
        assert spec.minimum_power_w == 1000.0
        assert spec.soc_min == 21.0
        assert spec.soc_max == 98.0
        assert spec.cycle_cost == pytest.approx(0.011)
        assert spec.setpoint_entity == "input_number.feedin_grid"
        assert spec.mode_entity == "input_select.ess_operating_mode"
        assert spec.stop_inverter_entity == "input_datetime.stop_victron"
        assert spec.soc_entity == "sensor.ess_battery_soc"

    def test_the_published_command_is_used_for_the_first_interval(
        self, exporter, tmp_path
    ):
        plan = call(exporter, str(tmp_path / "fast_plan.json"))
        first = plan.intervals[0].battery(0)
        # The solver said -1200 W; the published value is what was actuated.
        assert first.ac_power_w == -1200.0
        assert first.mode == "Aan"
        assert first.stop_inverter == "2000-01-01 00:00:00"

    def test_later_intervals_come_from_the_solver(self, exporter, tmp_path):
        plan = call(exporter, str(tmp_path / "fast_plan.json"))
        assert plan.intervals[1].battery(0).ac_power_w == pytest.approx(-3000.0)
        assert plan.intervals[3].battery(0).ac_power_w == pytest.approx(2000.0)

    def test_the_prices_and_the_energy_balance_line_up(self, exporter, tmp_path):
        plan = call(exporter, str(tmp_path / "fast_plan.json"))
        assert plan.intervals[1].price_import == pytest.approx(0.46)
        assert plan.intervals[1].price_export == pytest.approx(0.14)
        for interval in plan.intervals:
            assert interval.grid_w == pytest.approx(
                interval.house_w + interval.plan_battery_w
            )

    def test_the_state_of_charge_trajectory_is_complete(self, exporter, tmp_path):
        plan = call(exporter, str(tmp_path / "fast_plan.json"))
        assert plan.intervals[0].battery(0).soc_begin == 80.0
        assert plan.intervals[-1].battery(0).soc_end == 69.0

    def test_the_timeline_is_contiguous(self, exporter, tmp_path):
        plan = call(exporter, str(tmp_path / "fast_plan.json"))
        for first, second in zip(plan.intervals, plan.intervals[1:]):
            assert first.end_ts == second.start_ts
        assert plan.interval_at(plan.intervals[2].start_ts + 10) is plan.intervals[2]

    def test_a_partial_first_interval_keeps_the_timeline_contiguous(
        self, exporter, tmp_path
    ):
        """calc_optimum starts mid-interval; the plan must still cover it.

        Otherwise the fast layer finds no interval for the current moment and
        does nothing until the next optimizer run.
        """
        plan = call(
            exporter,
            str(tmp_path / "fast_plan.json"),
            hour_fraction=[0.25, 1.0, 1.0, 1.0],
        )
        assert plan.intervals[0].duration_s == 3600
        assert plan.intervals[0].end_ts == plan.intervals[1].start_ts
        now = plan.intervals[0].start_ts + 3000
        assert plan.interval_at(now) is plan.intervals[0]
        # Power is still scaled to the remaining quarter of an hour.
        assert plan.intervals[0].battery(0).ac_power_w == -1200.0
        assert plan.intervals[1].battery(0).ac_power_w == pytest.approx(-3000.0)

    def test_debug_mode_writes_nothing(self, exporter, tmp_path):
        exporter.debug = True
        path = tmp_path / "fast_plan.json"
        call(exporter, str(path))
        assert not path.exists()

    def test_an_empty_horizon_writes_nothing(self, exporter, tmp_path):
        path = tmp_path / "fast_plan.json"
        exporter.export_fast_plan(
            tijd=[],
            hour_fraction=[],
            pl=[],
            pt=[],
            c_l=[],
            c_t=[],
            solar_hour_sum_opt=[],
            ac_to_dc=[[]],
            ac_from_dc=[[]],
            soc=[[]],
            max_charge_power=[7.2],
            max_discharge_power=[7.2],
            kwh_cycle_cost=[0.011],
            lower_limit=[21],
            upper_limit=[98],
            eff_dc_to_bat=[0.93],
            eff_bat_to_dc=[0.94],
            published_battery=[],
            p_avg=0.0,
            U=0,
            B=1,
            path=str(path),
        )
        assert not path.exists()

    def test_the_file_is_valid_json(self, exporter, tmp_path):
        path = tmp_path / "fast_plan.json"
        call(exporter, str(path))
        payload = json.loads(path.read_text())
        assert payload["batteries"][0]["name"] == "Accu1"
        assert len(payload["intervals"]) == 4


class TestSignatureContract:
    """The optimizer calls the exporter by keyword; keep the two in step."""

    def test_calc_optimum_passes_exactly_the_declared_parameters(self):
        import ast
        import inspect

        source = inspect.getsource(DaCalc)
        tree = ast.parse(source.lstrip())
        called = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "export_fast_plan"
            ):
                called = {kw.arg for kw in node.keywords}
                assert not node.args, "call the exporter by keyword only"
        assert called is not None, "calc_optimum no longer exports the plan"

        declared = set(inspect.signature(DaCalc.export_fast_plan).parameters)
        declared.discard("self")
        optional = {"path"}
        assert called == declared - optional
