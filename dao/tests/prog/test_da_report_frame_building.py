"""Report.recalc_df_ha, aggregate_balance_df, calc_grid_columns and
get_price_data used to build their result frame with one
df.loc[df.shape[0]] = row append per source row, which is O(n^2) -- a
full year of hourly report data is 8760 such appends. Rewritten to
collect plain tuples and build the frame once; these tests pin the
resulting values down.
"""

import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("pandas")

from dao.lib.db_manager import DBmanagerObj  # noqa: E402
from dao.prog.da_report import Report  # noqa: E402

HOUR = 3600


def make_report():
    return Report.__new__(Report)


class TestRecalcDfHa:
    def test_melts_hourly_rows_and_keeps_a_running_sum_per_bucket(self):
        report = make_report()
        t0 = datetime.datetime(2026, 6, 1, 10, 0)
        org = pd.DataFrame(
            {
                "tijd": [t0, t0 + datetime.timedelta(hours=1)],
                "consumption": [1.0, 2.0],
                "production": [0.5, 0.25],
                "da_cons": [0.20, 0.22],
                "da_prod": [0.10, 0.11],
                "datasoort": ["recorded", "expected"],
            }
        )

        result = report.recalc_df_ha(org, "dag")

        assert list(result.columns) == [
            "dag", "vanaf", "tot", "consumption", "production", "cost", "profit",
            "datasoort",
        ]
        assert len(result) == 1  # both hours fall in the same day bucket
        row = result.iloc[0]
        assert row.consumption == pytest.approx(3.0)
        assert row.production == pytest.approx(0.75)
        assert row.cost == pytest.approx(1.0 * 0.20 + 2.0 * 0.22)
        assert row.profit == pytest.approx(0.5 * 0.10 + 0.25 * 0.11)

    def test_an_empty_input_returns_an_empty_frame_with_the_right_columns(self):
        report = make_report()
        org = pd.DataFrame(
            columns=["tijd", "consumption", "production", "da_cons", "da_prod", "datasoort"]
        )

        result = report.recalc_df_ha(org, "uur")

        assert len(result) == 0
        assert "consumption" in result.columns


class TestAggregateBalanceDf:
    def test_melts_hourly_rows_into_labelled_buckets(self):
        report = make_report()
        t0 = datetime.datetime(2026, 6, 1, 10, 0)
        df = pd.DataFrame(
            {
                "tijd": [t0],
                "datasoort": ["recorded"],
                "cons": [1.0],
                "prod": [0.5],
                "bat_out": [0.1],
                "bat_in": [0.2],
                "pv_ac": [0.6],
                "ev": [0.0],
                "wp": [0.0],
                "boil": [0.0],
                "base": [0.4],
            }
        )

        result = report.aggregate_balance_df(df, "uur")

        # str(datetime)[10:16] keeps the space before a single-digit-safe
        # hour string ("2026-06-01 10:00:00"[10:16] == " 10:00"); pinning
        # this existing quirk down, not asserting it is desirable.
        assert list(result["uur"]) == [" 10:00"]
        assert result["cons"].iloc[0] == pytest.approx(1.0)
        assert result["vanaf"].iloc[0] == t0
        assert result["tot"].iloc[0] == t0 + datetime.timedelta(hours=1)


class TestCalcGridColumns:
    def test_computes_the_derived_net_columns_per_bucket(self):
        report = make_report()
        t0 = datetime.datetime(2026, 6, 1, 10, 0)
        df = pd.DataFrame(
            {
                "vanaf": [t0, t0 + datetime.timedelta(hours=1)],
                "consumption": [2.0, 3.0],
                "production": [0.5, 1.0],
                "cost": [0.40, 0.60],
                "profit": [0.05, 0.10],
            }
        )

        result = report.calc_grid_columns(df, "uur", "grafiek")

        assert list(result.columns) == [
            "Uur", "Verbruik", "Productie", "Netto verbr.", "Kosten",
            "Opbrengst", "Netto kosten", "Tarief verbr.", "Tarief prod.",
        ]
        assert list(result["Verbruik"]) == [2.0, 3.0]
        assert list(result["Netto verbr."]) == [1.5, 2.0]
        assert list(result["Netto kosten"]) == pytest.approx([0.35, 0.50])

    def test_an_empty_input_returns_an_empty_frame_with_the_right_columns(self):
        report = make_report()
        df = pd.DataFrame(columns=["vanaf", "consumption", "production", "cost", "profit"])

        result = report.calc_grid_columns(df, "uur", "grafiek")

        assert len(result) == 0
        assert "Verbruik" in result.columns


@pytest.fixture
def db(tmp_path):
    from sqlalchemy import (
        BigInteger,
        Column,
        Float,
        ForeignKey,
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
    Table(
        "values",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
        Column("time", BigInteger, nullable=False),
        Column("value", Float),
        UniqueConstraint("variabel", "time"),
    )
    metadata.create_all(manager.engine)
    with manager.engine.begin() as connection:
        connection.execute(
            insert(variabel), [{"id": 1, "code": "da", "name": "Day ahead", "dim": "eur"}]
        )
    return manager


def make_report_with_prices(db):
    report = Report.__new__(Report)
    report.db_da = db
    report.prices_options = SimpleNamespace(tax_refund=True)
    report.ol_l_def = {"2026-01-01": 0.0}
    report.ol_t_def = {"2026-01-01": 0.0}
    report.taxes_l_def = {"2026-01-01": 0.10}
    report.taxes_t_def = {"2026-01-01": 0.05}
    report.btw_l_def = {"2026-01-01": 21.0}
    report.btw_t_def = {"2026-01-01": 21.0}
    report.multiplier_l_def = {"2026-01-01": 1.0}
    report.multiplier_t_def = {"2026-01-01": 1.0}
    return report


class TestGetPriceData:
    def test_builds_one_row_per_hour_with_taxes_and_btw_applied(self, db):
        from sqlalchemy import Table, insert

        variabel = Table("variabel", db.metadata, autoload_with=db.engine)
        values = Table("values", db.metadata, autoload_with=db.engine)
        t0 = int(datetime.datetime(2026, 6, 1, 10, 0).timestamp())
        with db.engine.begin() as connection:
            connection.execute(
                insert(values),
                [
                    {"variabel": 1, "time": t0, "value": 0.10},
                    {"variabel": 1, "time": t0 + HOUR, "value": 0.20},
                ],
            )

        report = make_report_with_prices(db)
        result = report.get_price_data(
            datetime.datetime(2026, 6, 1, 10, 0),
            datetime.datetime(2026, 6, 1, 12, 0),
        )

        assert list(result.columns) == ["time", "da_ex", "da_cons", "da_prod", "datasoort"]
        assert len(result) == 2
        assert list(result["da_ex"]) == [0.10, 0.20]
        expected_cons_0 = (0.10 * 1.0 + 0.10 + 0.0) * (1 + 21.0 / 100)
        assert result["da_cons"].iloc[0] == pytest.approx(expected_cons_0)
