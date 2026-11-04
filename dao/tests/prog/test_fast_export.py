"""Integration test for the optimizer's hand-off to the fast control layer.

Exercises ``DaCalc.export_fast_plan`` the way ``calc_optimum`` calls it, with
stand-ins for the solver variables, so a change to either side of the contract
breaks a test instead of silently producing a plan the realtime loop cannot
use.
"""

import json
import logging
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("mip", reason="the optimizer stack is not installed")
pytest.importorskip("matplotlib", reason="the optimizer stack is not installed")

from day_ahead import DaCalc  # noqa: E402

from dao.prog.config.models.fastcontrol import FastControlConfig  # noqa: E402
from dao.prog.fastctrl.plan import load_plan  # noqa: E402
from dao.prog.fastctrl.publication import PublicationError, read_publication  # noqa: E402
from dao.prog.settlement.publication import FinancialPlanBlocked  # noqa: E402
from dao.tests.settlement.test_publication import NOW, START_TS, item, replay_for_plan  # noqa: E402
from dao.tests.settlement.test_publication import system as system  # noqa: E402


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
def exporter(tmp_path, monkeypatch):
    """A DaCalc with only the attributes the exporter touches."""
    instance = DaCalc.__new__(DaCalc)
    instance.debug = False
    instance.interval_s = 3600
    instance.strategy = "minimize cost"
    instance.battery_options = [Battery()]
    instance.config = SimpleNamespace(prices=None, fast_control=FastControlConfig())
    monkeypatch.setattr("day_ahead.FAST_PLAN_FILE", str(tmp_path / "fast_plan.json"))
    return instance


def call(exporter, path, steps=4, **overrides):
    # The epoch axis the optimizer carries alongside its display column.
    start = 1768496400  # 2026-01-15 18:00 CET
    kwargs = dict(
        interval_ts=[start + 3600 * i for i in range(steps)],
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

    def test_the_boundaries_are_the_epochs_the_optimizer_handed_over(
        self, exporter, tmp_path
    ):
        """No zone conversion happens here, in either direction.

        The optimizer's own time axis is a column of naive pandas Timestamps,
        and ``pandas.Timestamp.timestamp()`` reads a naive wall clock as UTC
        where ``datetime.timestamp()`` reads it as local time. The exporter
        therefore takes the epoch the database already stores and passes it
        through untouched.
        """
        epochs = [1768496400 + 3600 * i for i in range(4)]
        plan = call(exporter, str(tmp_path / "fast_plan.json"), interval_ts=epochs)
        assert [i.start_ts for i in plan.intervals] == epochs

    def test_the_plan_covers_the_moment_it_was_written(self, exporter, tmp_path):
        """Regression: the fast layer sat at 0 W for a fortnight.

        Converting the optimizer's naive Timestamps put every boundary a full
        UTC offset into the future, so ``interval_at(now)`` never matched,
        every decision came back as ``plan_expired`` and the setpoint stayed
        at 0 W. Because an event is only recorded when the setpoint changes,
        the whole outage showed up as a single stale row in the web UI and
        nothing else.
        """
        now = int(time.time())
        boundary = now - now % 3600
        plan = call(
            exporter,
            str(tmp_path / "fast_plan.json"),
            interval_ts=[boundary + 3600 * i for i in range(4)],
        )
        assert plan.interval_at(now) is plan.intervals[0]

    def test_a_plan_that_misses_the_current_moment_is_reported(
        self, exporter, tmp_path, caplog
    ):
        """The condition that stayed invisible for two weeks must be loud.

        Writing it anyway is deliberate: the fast layer copes with a horizon
        it cannot use, and withholding the plan would take the optimizer's
        own schedule away as well.
        """
        now = int(time.time())
        ahead = now - now % 3600 + 2 * 3600
        path = tmp_path / "fast_plan.json"
        with caplog.at_level(logging.ERROR):
            plan = call(
                exporter,
                str(path),
                interval_ts=[ahead + 3600 * i for i in range(4)],
            )
        assert plan is not None and path.exists()
        assert "dekt het huidige moment niet" in caplog.text

    def test_debug_mode_writes_nothing(self, exporter, tmp_path):
        exporter.debug = True
        path = tmp_path / "fast_plan.json"
        call(exporter, str(path))
        assert not path.exists()

    def test_an_empty_horizon_writes_nothing(self, exporter, tmp_path):
        path = tmp_path / "fast_plan.json"
        exporter.export_fast_plan(
            interval_ts=[],
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


@pytest.fixture
def financial_exporter(exporter, system, monkeypatch):
    import day_ahead

    exporter.config.prices = system.pricing
    exporter.db_da = SimpleNamespace(engine=system.engine)
    exporter.interval_s = 900
    monkeypatch.setattr(day_ahead, "FAST_PLAN_FILE", str(system.path))
    monkeypatch.setattr(day_ahead.time, "time", lambda: NOW)
    context = system.plan.financial_context
    context = replace(context, terminal_value=replace(
        context.terminal_value, reference_efficiency_by_battery=(("Accu1", context.terminal_value.reference_efficiency_by_battery[0][1]),),
    ))
    from dao.prog.settlement.replay import write_replay_basis

    replay = replay_for_plan(replace(system.plan, financial_context=context), (system.snapshot,))
    digest = write_replay_basis(replay, plan_path=system.path)
    context = replace(context, replay_basis_hash=digest)
    return exporter, context


def financial_call(exporter, system, context):
    return call(exporter, str(system.path), steps=3,
                 interval_ts=[NOW, NOW + 900, NOW + 1800], hour_fraction=[0.25] * 3,
                 pl=[0.1] * 3, pt=[0.1] * 3, p_avg=0.1,
                financial_context=context)


def test_explicit_v2_context_is_published_through_real_adapter(financial_exporter, system):
    exporter, context = financial_exporter
    plan = financial_call(exporter, system, context)
    assert plan.format_version == 2
    assert plan.financial_context == context


def test_f2a_export_uses_final_solver_power_not_local_command(financial_exporter, system):
    from dao.prog.fastctrl import plan as plans

    exporter, context = financial_exporter
    context = replace(context, basis=replace(context.basis, planning_start_ts=NOW + 300))
    from dao.prog.settlement.replay import write_replay_basis

    evidence = replace(system.plan, financial_context=context,
                       intervals=[replace(system.plan.intervals[0], start_ts=NOW + 300), *system.plan.intervals[1:]])
    replay = replay_for_plan(evidence, (system.snapshot,))
    context = replace(context, replay_basis_hash=write_replay_basis(replay, plan_path=system.path))
    plan = call(exporter, str(system.path), steps=3,
                interval_ts=[NOW, NOW + 900, NOW + 1800],
                hour_fraction=[1 / 6, 0.25, 0.25],
                c_l=[Var(0.2), Var(0), Var(0.4)],
                financial_context=context,
                published_battery=[{"power_w": -6000, "stop_inverter": "invalid-local-text"}])
    first = plan.intervals[0]
    assert first.start_ts == NOW + 300 and first.end_ts == NOW + 900
    assert first.battery(0).ac_power_w == -1200.0
    assert first.battery(0).mean_ac_power_w == -1200.0
    assert first.battery(0).on_power_w == -1200.0
    assert plans.profile_energy_kwh(first.battery(0), first.start_ts, first.end_ts) == pytest.approx(-0.2)
    assert first.grid_w == pytest.approx(1200) and first.house_w == pytest.approx(2400)


def test_f2a_simultaneous_solver_directions_have_own_refusal(financial_exporter, system):
    exporter, context = financial_exporter
    with pytest.raises((ValueError, FinancialPlanBlocked), match="profile_simultaneous_charge_discharge"):
        call(exporter, str(system.path), steps=3,
             interval_ts=[NOW, NOW + 900, NOW + 1800], hour_fraction=[0.25] * 3,
             financial_context=context, ac_to_dc=[[Var(1), Var(0), Var(0)]])


def test_limited_v2_export_keeps_unknown_economics_and_base_tick_prices(financial_exporter, system):
    exporter, context = financial_exporter
    context = replace(context, economics_supported=False, issues=("bonus_prefix_unknown",))
    plan = financial_call(exporter, system, context)
    assert plan is not None and plan.format_version == 2
    assert plan.financial_context.economics_supported is False
    assert plan.financial_context.issues == ("bonus_prefix_unknown",)
    assert [i.price_import for i in plan.intervals] == [0.1] * 3
    assert [i.price_export for i in plan.intervals] == [0.1] * 3
    assert plan.price_average == 0.1


def test_changed_basis_before_export_is_blocked_not_relabelled(financial_exporter, system):
    exporter, context = financial_exporter
    system.repo.register_inputs("synthetic-grid", [item(NOW)], [system.snapshot], 1)
    with pytest.raises(FinancialPlanBlocked, match="input_revision_changed"):
        financial_call(exporter, system, context)
    assert read_publication(system.path).status == "blocked"
    assert load_plan(system.path) is None


def test_even_a_fresh_basis_cannot_authorize_hardware_output(financial_exporter, system):
    exporter, context = financial_exporter
    with pytest.raises(FinancialPlanBlocked, match="actuation_gates_unproven"):
        exporter._check_financial_output(basis=context.basis, now_ts=NOW)
    assert read_publication(system.path).status == "blocked"


@pytest.mark.parametrize("artifact", ["ready", "blocked"])
def test_omitting_context_cannot_escape_to_v1(financial_exporter, system, artifact):
    exporter, _ = financial_exporter
    if artifact == "blocked":
        system.adapter.block(now_ts=NOW, reason="correction")
    with pytest.raises((FinancialPlanBlocked, PublicationError)):
        call(exporter, str(system.path))
    assert load_plan(system.path) is None


def test_legacy_export_cannot_publish_a_mixed_contract_horizon(exporter, tmp_path, monkeypatch):
    import day_ahead
    from sqlalchemy import create_engine

    from dao.tests.settlement.test_publication import pricing_config

    path = tmp_path / "fast_plan.json"
    engine = create_engine("sqlite://")
    exporter.db_da = SimpleNamespace(engine=engine)
    exporter.config.prices = pricing_config()
    monkeypatch.setattr(day_ahead, "FAST_PLAN_FILE", str(path))
    monkeypatch.setattr(day_ahead.time, "time", lambda: START_TS - 1)
    with pytest.raises(FinancialPlanBlocked):
        call(exporter, str(path), interval_ts=[START_TS - 3600, START_TS, START_TS + 3600, START_TS + 7200])
    assert read_publication(path).status == "blocked"
    engine.dispose()


def test_financial_debug_export_preserves_existing_plan(financial_exporter, system):
    exporter, context = financial_exporter
    exporter.debug = True
    before = system.path.read_bytes()
    financial_call(exporter, system, context)
    assert system.path.read_bytes() == before


@pytest.mark.parametrize("crosses_during_build", [False, True])
def test_legacy_export_rechecks_wall_clock_after_building_the_payload(exporter, tmp_path, monkeypatch, crosses_during_build):
    import day_ahead
    from sqlalchemy import create_engine

    from dao.tests.settlement.test_publication import pricing_config

    path = tmp_path / "fast_plan.json"
    gate = 1798758000
    clock = [gate - 1]
    engine = create_engine("sqlite://")
    exporter.config.prices = pricing_config()
    exporter.db_da = SimpleNamespace(engine=engine)
    monkeypatch.setattr(day_ahead.time, "time", lambda: clock[0])
    class CrossingVar:
        @property
        def x(self):
            if crosses_during_build:
                clock[0] = gate
            return 0.0
    def export():
        return call(exporter, str(path), interval_ts=[gate - 14400, gate - 10800, gate - 7200, gate - 3600],
                    c_l=[CrossingVar(), Var(0), Var(0), Var(0)])
    if crosses_during_build:
        with pytest.raises(FinancialPlanBlocked):
            export()
        assert read_publication(path).status == "blocked"
    else:
        assert export().format_version == 1  # Half-open horizon ends exactly at the boundary.
    engine.dispose()


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

    def test_the_hand_off_takes_the_epoch_axis_not_the_display_column(self):
        """``tijd`` must never reach the exporter.

        It holds naive pandas Timestamps, which convert to epoch as if their
        wall clock were UTC. That moved every boundary a full UTC offset into
        the future, and the fast layer reported ``plan_expired`` on every tick
        for a fortnight. The optimizer carries the database's own epoch
        alongside it; that is what the exporter gets.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(DaCalc).lstrip())
        passed = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "export_fast_plan"
            ):
                passed = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
        assert passed is not None, "calc_optimum no longer exports the plan"
        assert passed["interval_ts"].startswith("interval_ts")
        assert "tijd" not in "".join(passed.values())
