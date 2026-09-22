"""Tests for the plan artefact and its on-disk contract."""

import json

import pytest

from dao.prog.fastctrl.plan import (
    PLAN_FORMAT_VERSION,
    BatteryPlanStep,
    BatterySpec,
    FastPlan,
    PlanInterval,
    build_intervals,
    load_plan,
    write_plan,
)

T0 = 1_700_000_000


def make_plan(count: int = 4, interval_s: int = 900) -> FastPlan:
    return FastPlan(
        created_ts=T0,
        interval_s=interval_s,
        specs=[
            BatterySpec(
                name="accu",
                capacity_kwh=12.5,
                max_charge_w=5000.0,
                max_discharge_w=5000.0,
                soc_min=21.0,
                soc_max=98.0,
                cycle_cost=0.012,
                charge_efficiency=0.93,
                discharge_efficiency=0.94,
                setpoint_entity="input_number.feedin",
                soc_entity="sensor.soc",
            )
        ],
        intervals=[
            PlanInterval(
                start_ts=T0 + i * interval_s,
                end_ts=T0 + (i + 1) * interval_s,
                price_import=0.25 + i * 0.01,
                price_export=0.07,
                grid_w=800.0 * i,
                house_w=500.0,
                pv_w=0.0,
                batteries=[
                    BatteryPlanStep(
                        ac_power_w=-500.0 * i,
                        soc_begin=60.0 - i,
                        soc_end=59.0 - i,
                        mode="Aan" if i == 0 else None,
                        stop_inverter="2000-01-01 00:00:00" if i == 0 else None,
                    )
                ],
            )
            for i in range(count)
        ],
        strategy="minimize cost",
        price_average=0.27,
    )


class TestSerialisation:
    def test_roundtrip_is_lossless(self, tmp_path):
        original = make_plan()
        path = str(tmp_path / "fast_plan.json")
        write_plan(original, path)
        restored = load_plan(path)
        assert restored is not None
        assert restored.to_dict() == original.to_dict()

    def test_the_file_is_valid_json_with_a_version(self, tmp_path):
        path = tmp_path / "fast_plan.json"
        write_plan(make_plan(), str(path))
        payload = json.loads(path.read_text())
        assert payload["format_version"] == PLAN_FORMAT_VERSION
        assert len(payload["intervals"]) == 4
        assert payload["batteries"][0]["name"] == "accu"

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert load_plan(str(tmp_path / "absent.json")) is None

    def test_a_corrupt_file_is_not_an_error(self, tmp_path):
        path = tmp_path / "fast_plan.json"
        path.write_text("{ this is not json")
        assert load_plan(str(path)) is None

    def test_a_newer_format_is_refused(self, tmp_path):
        path = tmp_path / "fast_plan.json"
        path.write_text(json.dumps({"format_version": PLAN_FORMAT_VERSION + 1}))
        assert load_plan(str(path)) is None

    def test_writing_is_atomic(self, tmp_path):
        """A reader must never observe a half written file."""
        path = tmp_path / "fast_plan.json"
        write_plan(make_plan(2), str(path))
        write_plan(make_plan(8), str(path))
        restored = load_plan(str(path))
        assert restored is not None and len(restored.intervals) == 8
        assert not list(tmp_path.glob(".fast_plan_*"))

    def test_unknown_fields_do_not_break_reading(self, tmp_path):
        path = tmp_path / "fast_plan.json"
        payload = make_plan(2).to_dict()
        payload["something_new"] = 42
        payload["intervals"][0]["extra"] = "ignored"
        path.write_text(json.dumps(payload))
        assert load_plan(str(path)) is not None


class TestLookups:
    def test_interval_at_uses_a_half_open_range(self):
        plan = make_plan()
        assert plan.interval_at(T0) is plan.intervals[0]
        assert plan.interval_at(T0 + 899) is plan.intervals[0]
        assert plan.interval_at(T0 + 900) is plan.intervals[1]
        assert plan.interval_at(T0 - 1) is None
        assert plan.interval_at(T0 + 4 * 900) is None

    def test_remaining_includes_the_current_interval(self):
        plan = make_plan()
        assert len(plan.remaining(T0 + 100)) == 4
        assert len(plan.remaining(T0 + 901)) == 3
        assert plan.remaining(T0 + 99999) == []

    def test_age(self):
        plan = make_plan()
        assert plan.age(T0 + 600) == 600
        assert plan.age(T0 - 600) == 0

    def test_battery_returns_a_blank_step_when_out_of_range(self):
        plan = make_plan()
        assert plan.intervals[0].battery(9).ac_power_w == 0.0

    def test_spec_index(self):
        plan = make_plan()
        assert plan.spec_index("accu") == 0
        assert plan.spec_index("nope") is None


class TestSpec:
    def test_derived_properties(self):
        spec = BatterySpec(
            name="a",
            capacity_kwh=20.0,
            max_charge_w=1.0,
            max_discharge_w=1.0,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
        )
        assert spec.kwh_per_percent == pytest.approx(0.2)
        assert spec.round_trip_efficiency == pytest.approx(0.81)

    def test_nan_and_garbage_survive_reading(self):
        spec = BatterySpec.from_dict(
            {"name": "x", "capacity_kwh": "not a number", "max_charge_w": None}
        )
        assert spec.capacity_kwh == 1.0
        assert spec.max_charge_w == 0.0


class TestBuildIntervals:
    def test_columns_are_zipped_into_intervals(self):
        stamps = [T0, T0 + 900, T0 + 1800]
        intervals = build_intervals(
            timestamps=stamps,
            interval_s=900,
            price_import=[0.1, 0.2, 0.3],
            price_export=[0.0, 0.1, 0.2],
            grid_w=[100.0, 200.0, 300.0],
            house_w=[100.0, 200.0, 300.0],
            pv_w=[0.0, 0.0, 0.0],
            battery_steps=[[BatteryPlanStep() for _ in range(3)]],
        )
        assert [i.start_ts for i in intervals] == stamps
        assert intervals[-1].end_ts == T0 + 2700
        assert intervals[1].price_import == 0.2


class TestBuildPlan:
    """``build_plan`` is what the optimizer hands over, so its unit
    conversions and its grid/house bookkeeping have to be exactly right."""

    SPECS = [
        BatterySpec(name="accu", capacity_kwh=10.0, max_charge_w=5000.0, max_discharge_w=5000.0)
    ]

    def build(self, **overrides):
        from dao.prog.fastctrl.plan import build_plan

        kwargs = dict(
            created_ts=T0,
            interval_s=3600,
            specs=self.SPECS,
            start_ts=[T0, T0 + 3600],
            hour_fraction=[1.0, 1.0],
            price_import=[0.30, 0.40],
            price_export=[0.10, 0.20],
            grid_kwh=[1.5, -0.5],
            pv_kwh=[0.0, 2.0],
            battery_kw=[[1.0, -2.0]],
            soc=[[50.0, 59.0, 40.0]],
            strategy="minimize cost",
            price_average=0.35,
        )
        kwargs.update(overrides)
        return build_plan(**kwargs)

    def test_energy_is_converted_to_average_power(self):
        plan = self.build()
        assert plan.intervals[0].grid_w == pytest.approx(1500.0)
        assert plan.intervals[1].grid_w == pytest.approx(-500.0)
        assert plan.intervals[1].pv_w == pytest.approx(2000.0)
        assert plan.intervals[0].battery(0).ac_power_w == pytest.approx(1000.0)

    def test_the_house_load_closes_the_energy_balance(self):
        plan = self.build()
        for interval in plan.intervals:
            assert interval.grid_w == pytest.approx(
                interval.house_w + interval.plan_battery_w
            )

    def test_a_partial_first_interval_scales_the_power_but_keeps_the_slot(self):
        """The optimizer starts partway through the interval.

        Its first row carries the interval boundary as timestamp but only the
        remaining fraction of an hour as duration. The plan must still cover
        the whole slot, otherwise the fast layer would find no interval for
        "now" and sit idle until the next optimizer run.
        """
        plan = self.build(hour_fraction=[0.25, 1.0], grid_kwh=[0.5, -0.5])
        assert plan.intervals[0].duration_s == 3600
        assert plan.intervals[0].end_ts == plan.intervals[1].start_ts
        # 0.5 kWh in a quarter of an hour is 2000 W.
        assert plan.intervals[0].grid_w == pytest.approx(2000.0)
        # A moment well past the partial part still resolves to interval 0.
        assert plan.interval_at(T0 + 3000) is plan.intervals[0]

    def test_quarter_hour_intervals(self):
        plan = self.build(
            interval_s=900,
            start_ts=[T0, T0 + 900],
            hour_fraction=[0.25, 0.25],
            grid_kwh=[0.25, 0.25],
        )
        assert plan.interval_s == 900
        assert plan.intervals[0].duration_s == 900
        assert plan.intervals[0].grid_w == pytest.approx(1000.0)

    def test_the_published_command_wins_for_the_first_interval(self):
        plan = self.build(
            published=[
                {
                    "power_w": 1234.0,
                    "mode": "Aan",
                    "stop_inverter": "2024-01-01 12:34",
                }
            ]
        )
        first = plan.intervals[0].battery(0)
        assert first.ac_power_w == 1234.0
        assert first.mode == "Aan"
        assert first.stop_inverter == "2024-01-01 12:34"
        # Later intervals keep the raw solver value and carry no command.
        second = plan.intervals[1].battery(0)
        assert second.ac_power_w == pytest.approx(-2000.0)
        assert second.mode is None

    def test_the_state_of_charge_trajectory_is_carried(self):
        plan = self.build()
        assert plan.intervals[0].battery(0).soc_begin == 50.0
        assert plan.intervals[0].battery(0).soc_end == 59.0
        assert plan.intervals[1].battery(0).soc_end == 40.0

    def test_several_batteries(self):
        specs = [
            BatterySpec(name="a", capacity_kwh=10.0, max_charge_w=1.0, max_discharge_w=1.0),
            BatterySpec(name="b", capacity_kwh=5.0, max_charge_w=1.0, max_discharge_w=1.0),
        ]
        plan = self.build(
            specs=specs,
            battery_kw=[[1.0, -2.0], [0.5, -0.5]],
            soc=[[50.0, 59.0, 40.0], [30.0, 35.0, 30.0]],
        )
        assert plan.intervals[0].plan_battery_w == pytest.approx(1500.0)
        assert len(plan.intervals[0].batteries) == 2
        assert plan.specs[1].name == "b"

    def test_it_survives_a_json_roundtrip(self, tmp_path):
        from dao.prog.fastctrl.plan import load_plan, write_plan

        plan = self.build()
        path = str(tmp_path / "p.json")
        write_plan(plan, path)
        assert load_plan(path).to_dict() == plan.to_dict()
