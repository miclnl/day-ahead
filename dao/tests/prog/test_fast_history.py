"""Tests for reading the backtest inputs out of the databases.

Builds a real SQLite Home Assistant recorder, in both the modern
``states_meta`` layout and the legacy ``entity_id`` layout, so the queries are
exercised against actual SQL rather than a mock.
"""

import datetime

import pytest

pytest.importorskip("pandas")

from sqlalchemy import (  # noqa: E402
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
)

from dao.prog.fastctrl.plan import BatterySpec  # noqa: E402
from dao.prog.fastctrl.simulate import (  # noqa: E402
    HistoryLoader,
    _median_gap,
    _on_grid,
    _sum_on_grid,
)

T0 = 1_700_000_000


class FakeDb:
    """The two attributes HistoryLoader uses from a DBmanagerObj."""

    def __init__(self, engine, metadata):
        self.engine = engine
        self.metadata = metadata


def modern_recorder(rows):
    """``states`` joined to ``states_meta``, the layout used since 2023."""
    engine = create_engine("sqlite://")
    metadata = MetaData()
    states_meta = Table(
        "states_meta",
        metadata,
        Column("metadata_id", Integer, primary_key=True),
        Column("entity_id", String),
    )
    states = Table(
        "states",
        metadata,
        Column("state_id", Integer, primary_key=True),
        Column("metadata_id", Integer),
        Column("state", String),
        Column("last_updated_ts", Float),
    )
    metadata.create_all(engine)

    entities = sorted({entity for entity, _, _ in rows})
    ids = {entity: index + 1 for index, entity in enumerate(entities)}
    with engine.begin() as connection:
        for entity, index in ids.items():
            connection.execute(
                insert(states_meta).values(metadata_id=index, entity_id=entity)
            )
        for entity, timestamp, value in rows:
            connection.execute(
                insert(states).values(
                    metadata_id=ids[entity],
                    state=str(value),
                    last_updated_ts=float(timestamp),
                )
            )
    return FakeDb(engine, MetaData())


def legacy_recorder(rows):
    """``states`` with an inline ``entity_id`` column, pre 2023."""
    engine = create_engine("sqlite://")
    metadata = MetaData()
    states = Table(
        "states",
        metadata,
        Column("state_id", Integer, primary_key=True),
        Column("entity_id", String),
        Column("state", String),
        Column("last_updated_ts", Float),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        for entity, timestamp, value in rows:
            connection.execute(
                insert(states).values(
                    entity_id=entity, state=str(value), last_updated_ts=float(timestamp)
                )
            )
    return FakeDb(engine, MetaData())


class FakeDaDb:
    """Serves the prognoses columns the plan reconstruction asks for."""

    def __init__(self, series):
        self.series = series

    def get_column_data(self, tablename, column_name, start=None, end=None, **kwargs):
        import pandas as pd

        rows = self.series.get(column_name, {})
        return pd.DataFrame(
            {
                "utc": list(rows.keys()),
                "value": list(rows.values()),
            }
        )


class FakeReport:
    def __init__(self, prices):
        self.prices = prices

    def get_price_data(self, start, end, interval="1hour"):
        import pandas as pd

        return pd.DataFrame(
            {
                "time": [datetime.datetime.fromtimestamp(t) for t in self.prices],
                "da_cons": [v[0] for v in self.prices.values()],
                "da_prod": [v[1] for v in self.prices.values()],
            }
        )


SPEC = BatterySpec(
    name="accu",
    capacity_kwh=10.0,
    max_charge_w=5000.0,
    max_discharge_w=5000.0,
    soc_entity="sensor.soc",
)


class TestSeries:
    ROWS = [
        ("sensor.p1", T0 + 0, 1000),
        ("sensor.p1", T0 + 60, 2500),
        ("sensor.p1", T0 + 120, -800),
        ("sensor.other", T0 + 0, 42),
    ]

    @pytest.mark.parametrize("builder", [modern_recorder, legacy_recorder])
    def test_both_recorder_layouts_are_supported(self, builder):
        loader = HistoryLoader(builder(self.ROWS), None, None)
        frame = loader._series(
            "sensor.p1",
            datetime.datetime.fromtimestamp(T0),
            datetime.datetime.fromtimestamp(T0 + 1000),
        )
        assert list(frame["value"]) == [1000.0, 2500.0, -800.0]

    def test_other_entities_are_not_included(self):
        loader = HistoryLoader(modern_recorder(self.ROWS), None, None)
        frame = loader._series(
            "sensor.p1",
            datetime.datetime.fromtimestamp(T0),
            datetime.datetime.fromtimestamp(T0 + 1000),
        )
        assert len(frame) == 3

    def test_the_window_is_respected(self):
        loader = HistoryLoader(modern_recorder(self.ROWS), None, None)
        frame = loader._series(
            "sensor.p1",
            datetime.datetime.fromtimestamp(T0 + 30),
            datetime.datetime.fromtimestamp(T0 + 100),
        )
        assert list(frame["value"]) == [2500.0]

    def test_unusable_states_are_dropped(self):
        rows = self.ROWS + [
            ("sensor.p1", T0 + 180, "unavailable"),
            ("sensor.p1", T0 + 240, "unknown"),
            ("sensor.p1", T0 + 300, 500),
        ]
        loader = HistoryLoader(modern_recorder(rows), None, None)
        frame = loader._series(
            "sensor.p1",
            datetime.datetime.fromtimestamp(T0),
            datetime.datetime.fromtimestamp(T0 + 1000),
        )
        assert list(frame["value"]) == [1000.0, 2500.0, -800.0, 500.0]

    def test_an_absent_entity_yields_an_empty_frame(self):
        loader = HistoryLoader(modern_recorder(self.ROWS), None, None)
        frame = loader._series(
            "sensor.nope",
            datetime.datetime.fromtimestamp(T0),
            datetime.datetime.fromtimestamp(T0 + 1000),
        )
        assert frame.empty


class TestResampling:
    def test_event_rows_are_forward_filled(self):
        import pandas as pd

        frame = pd.DataFrame({"ts": [T0, T0 + 120], "value": [1000.0, 2000.0]})
        stamps = [T0 + offset for offset in (0, 60, 120, 180)]
        values = _on_grid(frame, 1.0, stamps)
        assert list(values) == [1000.0, 1000.0, 2000.0, 2000.0]

    def test_the_scale_carries_the_unit_and_the_sign(self):
        import pandas as pd

        frame = pd.DataFrame({"ts": [T0], "value": [1.5]})
        values = _on_grid(frame, -1000.0, [T0, T0 + 60])
        assert list(values) == [-1500.0, -1500.0]

    def test_a_positive_negative_pair_is_summed_per_entity(self):
        import pandas as pd

        imported = pd.DataFrame({"ts": [T0], "value": [3000.0]})
        exported = pd.DataFrame({"ts": [T0 + 60], "value": [500.0]})
        values = _sum_on_grid(
            [("a", 1.0, imported), ("b", -1.0, exported)],
            [T0, T0 + 60, T0 + 120],
        )
        assert values == [3000.0, 2500.0, 2500.0]

    def test_a_missing_series_makes_the_whole_sum_unusable(self):
        import pandas as pd

        good = pd.DataFrame({"ts": [T0], "value": [1.0]})
        assert _sum_on_grid([("a", 1.0, good), ("b", 1.0, None)], [T0]) is None

    def test_the_median_gap_detects_a_coarse_recorder(self):
        assert _median_gap([T0 + i * 300 for i in range(20)]) == 300
        assert _median_gap([T0, T0 + 10]) == float("inf")


class TestPlanReconstruction:
    def build_loader(self):
        stamps = [T0 + i * 3600 for i in range(3)]
        da_db = FakeDaDb(
            {
                "bat_in": {stamps[0]: 0.0, stamps[1]: 2.0, stamps[2]: 0.0},
                "bat_out": {stamps[0]: 1.5, stamps[1]: 0.0, stamps[2]: 0.5},
                "cons": {stamps[0]: 0.2, stamps[1]: 2.4, stamps[2]: 0.0},
                "prod": {stamps[0]: 0.0, stamps[1]: 0.0, stamps[2]: 0.3},
                "soc": {stamps[0]: 80.0, stamps[1]: 65.0, stamps[2]: 84.0},
            }
        )
        report = FakeReport(
            {
                stamps[0]: (0.40, 0.12),
                stamps[1]: (0.15, 0.03),
                stamps[2]: (0.22, 0.06),
            }
        )
        return HistoryLoader(None, da_db, report), stamps

    def test_energy_rows_become_average_power(self):
        loader, stamps = self.build_loader()
        plan = loader._plan(
            datetime.datetime.fromtimestamp(stamps[0]),
            datetime.datetime.fromtimestamp(stamps[-1] + 3600),
            SPEC,
            3600,
        )
        assert len(plan.intervals) == 3
        # 1.5 kWh out over an hour is 1500 W of discharge.
        assert plan.intervals[0].battery(0).ac_power_w == pytest.approx(-1500.0)
        assert plan.intervals[1].battery(0).ac_power_w == pytest.approx(2000.0)
        assert plan.intervals[0].grid_w == pytest.approx(200.0)
        assert plan.intervals[2].grid_w == pytest.approx(-300.0)

    def test_the_prices_are_matched_to_the_interval(self):
        loader, _ = self.build_loader()
        plan = loader._plan(
            datetime.datetime.fromtimestamp(T0),
            datetime.datetime.fromtimestamp(T0 + 3 * 3600),
            SPEC,
            3600,
        )
        assert plan.intervals[0].price_import == pytest.approx(0.40)
        assert plan.intervals[1].price_export == pytest.approx(0.03)
        assert plan.price_average == pytest.approx((0.40 + 0.15 + 0.22) / 3)

    def test_the_energy_balance_closes(self):
        loader, _ = self.build_loader()
        plan = loader._plan(
            datetime.datetime.fromtimestamp(T0),
            datetime.datetime.fromtimestamp(T0 + 3 * 3600),
            SPEC,
            3600,
        )
        for interval in plan.intervals:
            assert interval.grid_w == pytest.approx(
                interval.house_w + interval.plan_battery_w
            )

    def test_the_reconstructed_plan_can_drive_the_backtest(self):
        from dao.prog.fastctrl.policy import PolicyLimits
        from dao.prog.fastctrl.simulate import Sample, compare

        loader, stamps = self.build_loader()
        plan = loader._plan(
            datetime.datetime.fromtimestamp(stamps[0]),
            datetime.datetime.fromtimestamp(stamps[-1] + 3600),
            SPEC,
            3600,
        )
        samples = [
            Sample(stamps[0] + offset, 1500.0 if offset % 900 else 4000.0)
            for offset in range(0, 3 * 3600, 60)
        ]
        result = compare(plan, samples, PolicyLimits(), 70.0)
        assert result.baseline.duration_h == pytest.approx(3.0, abs=0.1)
        assert result.fast.import_kwh <= result.baseline.import_kwh


class TestChunkedReading:
    """Recorder history must not be pulled into memory in one piece.

    A P1 meter that updates every second produces on the order of a million
    rows a fortnight. The grid it ends up on has a few thousand points, so
    reading it whole is both pointless and, on a 4 GB Home Assistant Yellow,
    enough to run out of memory.
    """

    def build(self, days=6, period_s=30):
        rows = []
        start = T0
        for step in range(int(days * 86400 / period_s)):
            rows.append(("sensor.p1", start + step * period_s, 1000 + step % 7))
        return modern_recorder(rows), start

    def test_the_result_matches_an_unchunked_read(self):
        db, start = self.build(days=4)
        loader = HistoryLoader(db, None, None)
        begin = datetime.datetime.fromtimestamp(start)
        end = begin + datetime.timedelta(days=4)
        stamps = list(range(start, start + 4 * 86400, 600))

        chunked = loader._series_on_grid(
            "sensor.p1", 1.0, stamps, begin, end, chunk_days=1
        )
        whole = loader._series_on_grid(
            "sensor.p1", 1.0, stamps, begin, end, chunk_days=365
        )
        assert list(chunked.fillna(-1)) == list(whole.fillna(-1))

    def test_the_value_is_carried_across_a_chunk_boundary(self):
        """A sensor that does not change for days must not become NaN."""
        db = modern_recorder([("sensor.p1", T0, 1234.0)])
        loader = HistoryLoader(db, None, None)
        begin = datetime.datetime.fromtimestamp(T0)
        end = begin + datetime.timedelta(days=5)
        stamps = list(range(T0, T0 + 5 * 86400, 3600))
        series = loader._series_on_grid(
            "sensor.p1", 1.0, stamps, begin, end, chunk_days=1
        )
        assert series.notna().all()
        assert set(series.unique()) == {1234.0}

    def test_each_chunk_only_reads_its_own_window(self, monkeypatch):
        db, start = self.build(days=6)
        loader = HistoryLoader(db, None, None)
        sizes = []
        original = loader._series

        def spy(entity_id, chunk_start, chunk_end):
            frame = original(entity_id, chunk_start, chunk_end)
            sizes.append(len(frame))
            return frame

        monkeypatch.setattr(loader, "_series", spy)
        begin = datetime.datetime.fromtimestamp(start)
        end = begin + datetime.timedelta(days=6)
        stamps = list(range(start, start + 6 * 86400, 600))
        loader._series_on_grid("sensor.p1", 1.0, stamps, begin, end, chunk_days=2)

        assert len(sizes) == 3
        # Two days at one row per 30 s is 5760 rows, never the full 17k.
        assert max(sizes) < 6200

    def test_the_scale_is_applied(self):
        db = modern_recorder([("sensor.p1", T0, 1.5)])
        loader = HistoryLoader(db, None, None)
        begin = datetime.datetime.fromtimestamp(T0)
        series = loader._series_on_grid(
            "sensor.p1", -1000.0, [T0, T0 + 60], begin, begin + datetime.timedelta(hours=1)
        )
        assert list(series) == [-1500.0, -1500.0]

    def test_an_absent_entity_yields_none(self):
        db = modern_recorder([("sensor.other", T0, 1.0)])
        loader = HistoryLoader(db, None, None)
        begin = datetime.datetime.fromtimestamp(T0)
        assert (
            loader._series_on_grid(
                "sensor.p1", 1.0, [T0], begin, begin + datetime.timedelta(hours=1)
            )
            is None
        )

    def test_the_sample_gap_is_measured_on_one_day_only(self, monkeypatch):
        db, start = self.build(days=6, period_s=300)
        loader = HistoryLoader(db, None, None)
        windows = []
        original = loader._series
        monkeypatch.setattr(
            loader,
            "_series",
            lambda e, s, x: windows.append((s, x)) or original(e, s, x),
        )
        gap = loader._sample_gap("sensor.p1", datetime.datetime.fromtimestamp(start))
        assert gap == 300
        assert len(windows) == 1
        assert (windows[0][1] - windows[0][0]).days == 1

    def test_summing_two_entities_stays_chunked(self):
        rows = [("sensor.a", T0 + i * 60, 100.0) for i in range(2000)]
        rows += [("sensor.b", T0 + i * 60, 40.0) for i in range(2000)]
        db = modern_recorder(rows)
        loader = HistoryLoader(db, None, None)
        begin = datetime.datetime.fromtimestamp(T0)
        end = begin + datetime.timedelta(days=2)
        stamps = list(range(T0, T0 + 3600, 600))
        total = loader._sum_entities(
            [("sensor.a", 1.0), ("sensor.b", -1.0)], stamps, begin, end
        )
        assert total == [60.0] * len(stamps)

    def test_one_missing_entity_makes_the_sum_unusable(self):
        db = modern_recorder([("sensor.a", T0, 100.0)])
        loader = HistoryLoader(db, None, None)
        begin = datetime.datetime.fromtimestamp(T0)
        end = begin + datetime.timedelta(hours=2)
        assert (
            loader._sum_entities(
                [("sensor.a", 1.0), ("sensor.absent", 1.0)], [T0], begin, end
            )
            is None
        )
