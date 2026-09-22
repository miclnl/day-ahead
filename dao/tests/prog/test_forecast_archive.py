"""Tests for the forecast archive and the accuracy reporting.

Runs against a real SQLite database created by the same CheckDB code that runs
at add-on startup, so the schema under test is the schema that ships.
"""

import datetime

import pytest

pytest.importorskip("pandas")

from dao.lib.db_manager import (  # noqa: E402
    LEAD_BUCKETS,
    DBmanagerObj,
    lead_bucket,
)

HOUR = 3600
T0 = 1_700_000_000 // HOUR * HOUR  # aligned on a whole hour


@pytest.fixture
def db(tmp_path):
    """A day_ahead database with the production schema."""
    from sqlalchemy import (
        BigInteger,
        Column,
        Float,
        ForeignKey,
        Index,
        Integer,
        String,
        Table,
        UniqueConstraint,
        insert,
    )

    manager = DBmanagerObj(
        db_dialect="sqlite", db_name="day_ahead.db", db_path=str(tmp_path)
    )
    metadata = manager.metadata
    variabel = Table(
        "variabel",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("code", String(10), unique=True, nullable=False),
        Column("name", String(50), unique=True, nullable=False),
        Column("dim", String(10), nullable=False),
        Column("aggregate", String(3), nullable=False, default="avg"),
    )
    for name in ("values", "prognoses"):
        Table(
            name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
            Column("time", BigInteger, nullable=False),
            Column("value", Float),
            UniqueConstraint("variabel", "time"),
        )
    forecasts = Table(
        "forecasts",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
        Column("target_time", BigInteger, nullable=False),
        Column("lead_bucket", Integer, nullable=False),
        Column("issued_time", BigInteger, nullable=False),
        Column("value", Float),
        UniqueConstraint("variabel", "target_time", "lead_bucket"),
    )
    Index("ix_forecasts_target", forecasts.c.target_time)
    metadata.create_all(manager.engine)

    with manager.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {"id": 11, "code": "base", "name": "Basislast", "dim": "kWh",
                 "aggregate": "sum"},
                {"id": 25, "code": "m_house", "name": "Gemeten huisvraag",
                 "dim": "kWh", "aggregate": "sum"},
                {"id": 27, "code": "hload", "name": "Geplande huisvraag",
                 "dim": "kWh", "aggregate": "sum"},
            ],
        )
    return manager


def put_realised(db, code, rows):
    from sqlalchemy import Table, insert, select

    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    values = Table("values", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        ident = connection.execute(
            select(variabel.c.id).where(variabel.c.code == code)
        ).scalar_one()
        connection.execute(
            insert(values),
            [
                {"variabel": ident, "time": t, "value": v}
                for t, v in rows
            ],
        )


def count_forecasts(db):
    from sqlalchemy import Table, func, select

    table = Table("forecasts", db.metadata, autoload_with=db.engine)
    with db.engine.connect() as connection:
        return connection.execute(select(func.count()).select_from(table)).scalar()


class TestLeadBucket:
    def test_it_picks_the_largest_bucket_at_or_below_the_lead(self):
        assert lead_bucket(0.0) == 0
        assert lead_bucket(0.9) == 0
        assert lead_bucket(1.0) == 1
        assert lead_bucket(3.99) == 1
        assert lead_bucket(4.0) == 4
        assert lead_bucket(100.0) == LEAD_BUCKETS[-1]

    def test_a_negative_lead_still_returns_the_lowest_bucket(self):
        assert lead_bucket(-5.0) == LEAD_BUCKETS[0]


class TestSaveForecasts:
    def test_rows_are_stored_with_their_lead_bucket(self, db):
        issued = T0
        rows = [(T0 + h * HOUR, "hload", 1.0 + h) for h in (0, 2, 8, 20, 40)]
        assert db.save_forecasts(rows, issued) == 5

        from sqlalchemy import Table, select

        table = Table("forecasts", db.metadata, autoload_with=db.engine)
        with db.engine.connect() as connection:
            stored = connection.execute(
                select(table.c.target_time, table.c.lead_bucket, table.c.value)
                .order_by(table.c.target_time)
            ).all()
        assert [r.lead_bucket for r in stored] == [0, 1, 4, 12, 24]

    def test_targets_in_the_past_are_dropped(self, db):
        # A "forecast" for a moment that has already happened says nothing
        # about forecast skill.
        rows = [(T0 - HOUR, "hload", 1.0), (T0 + HOUR, "hload", 2.0)]
        assert db.save_forecasts(rows, T0) == 1

    def test_unknown_codes_are_skipped_silently(self, db):
        rows = [(T0 + HOUR, "does_not_exist", 1.0), (T0 + HOUR, "hload", 2.0)]
        assert db.save_forecasts(rows, T0) == 1

    def test_nan_and_garbage_are_skipped(self, db):
        rows = [
            (T0 + HOUR, "hload", float("nan")),
            (T0 + HOUR, "hload", None),
            (T0 + 2 * HOUR, "hload", "x"),
            (T0 + 3 * HOUR, "hload", 1.0),
        ]
        assert db.save_forecasts(rows, T0) == 1

    def test_the_table_size_is_bounded_by_the_bucket_count(self, db):
        """The whole point of the design: re-running does not grow the table.

        Fifty optimizer passes over the same targets must collapse onto at most
        one row per (variable, target, lead bucket).
        """
        target = T0 + 30 * HOUR
        for run in range(50):
            db.save_forecasts([(target, "hload", float(run))], T0 + run * HOUR)
        assert count_forecasts(db) == len(LEAD_BUCKETS)

    def test_a_later_pass_overwrites_within_the_same_bucket(self, db):
        target = T0 + 2 * HOUR
        db.save_forecasts([(target, "hload", 1.0)], T0)          # lead 2h -> bucket 1
        db.save_forecasts([(target, "hload", 9.0)], T0 + 1800)   # lead 1.5h -> bucket 1

        from sqlalchemy import Table, select

        table = Table("forecasts", db.metadata, autoload_with=db.engine)
        with db.engine.connect() as connection:
            rows = connection.execute(select(table.c.value, table.c.lead_bucket)).all()
        assert len(rows) == 1
        assert rows[0].value == 9.0

    def test_an_empty_input_is_a_no_op(self, db):
        assert db.save_forecasts([], T0) == 0
        assert count_forecasts(db) == 0


class TestPrune:
    def test_old_targets_are_removed(self, db):
        db.save_forecasts([(T0 + h * HOUR, "hload", 1.0) for h in range(10)], T0)
        assert count_forecasts(db) == 10
        removed = db.prune_forecasts(T0 + 5 * HOUR)
        assert removed == 5
        assert count_forecasts(db) == 5


class TestAccuracy:
    def seed(self, db):
        """Forecast is 0.5 kWh too high at every lead, realised is 2.0."""
        targets = [T0 + h * HOUR for h in range(1, 25)]
        put_realised(db, "m_house", [(t, 2.0) for t in targets])
        for issue_offset in (0, 20 * HOUR):
            db.save_forecasts(
                [(t, "hload", 2.5) for t in targets], T0 + issue_offset
            )
        return targets

    def test_bias_and_mae_are_computed_per_bucket(self, db):
        targets = self.seed(db)
        rows = db.forecast_accuracy(
            "hload", "values", "m_house", targets[0], targets[-1] + HOUR
        )
        assert rows
        for row in rows:
            assert row["bias"] == pytest.approx(0.5)
            assert row["mae"] == pytest.approx(0.5)
            assert row["mse"] == pytest.approx(0.25)
            assert row["scale"] == pytest.approx(2.0)
            assert row["n"] > 0

    def test_several_buckets_are_present(self, db):
        targets = self.seed(db)
        rows = db.forecast_accuracy(
            "hload", "values", "m_house", targets[0], targets[-1] + HOUR
        )
        assert len(rows) >= 2
        assert sorted(r["lead_bucket"] for r in rows) == [
            r["lead_bucket"] for r in rows
        ]

    def test_an_empty_window_yields_nothing(self, db):
        self.seed(db)
        assert db.forecast_accuracy("hload", "values", "m_house", 1, 2) == []

    def test_only_matching_targets_are_joined(self, db):
        """Realised values without a forecast, and vice versa, are ignored."""
        targets = [T0 + h * HOUR for h in range(1, 5)]
        put_realised(db, "m_house", [(targets[0], 2.0), (targets[1], 2.0)])
        db.save_forecasts([(t, "hload", 3.0) for t in targets], T0)
        rows = db.forecast_accuracy(
            "hload", "values", "m_house", targets[0], targets[-1] + HOUR
        )
        assert sum(r["n"] for r in rows) == 2

    def test_bias_by_hour_finds_a_time_of_day_pattern(self, db):
        """The evening is underforecast, the rest of the day is spot on.

        This is the pattern that matters: it makes the optimizer reserve too
        little for the evening peak, and the realtime layer cannot repair it.
        """
        targets = []
        realised = []
        forecast = []
        for day in range(6):
            for hour in range(24):
                target = T0 + (day * 24 + hour) * HOUR
                local_hour = datetime.datetime.fromtimestamp(target).hour
                actual = 3.0 if 17 <= local_hour <= 20 else 1.0
                targets.append(target)
                realised.append((target, actual))
                forecast.append((target, "hload", 1.0))
        put_realised(db, "m_house", realised)
        db.save_forecasts(forecast, T0)

        rows = db.forecast_bias_by_hour(
            "hload", "values", "m_house", targets[0], targets[-1] + HOUR
        )
        by_hour = {r["uur"]: r["bias"] for r in rows}
        assert by_hour["18:00"] == pytest.approx(-2.0)
        assert by_hour["03:00"] == pytest.approx(0.0)

    def test_bias_by_hour_can_be_limited_to_one_bucket(self, db):
        targets = self.seed(db)
        all_rows = db.forecast_bias_by_hour(
            "hload", "values", "m_house", targets[0], targets[-1] + HOUR
        )
        one = db.forecast_bias_by_hour(
            "hload", "values", "m_house", targets[0], targets[-1] + HOUR, bucket=0
        )
        assert sum(r["n"] for r in one) < sum(r["n"] for r in all_rows)


class TestVariabelIds:
    def test_codes_are_resolved_and_cached(self, db):
        assert db.variabel_ids(["hload", "m_house"]) == {"hload": 27, "m_house": 25}
        # Second call must be served from the cache, not the database.
        db.engine.dispose()
        assert db.variabel_ids(["hload"]) == {"hload": 27}

    def test_unknown_codes_are_absent_from_the_result(self, db):
        assert db.variabel_ids(["nope"]) == {}
