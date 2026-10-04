"""DaBase.save_df and DaBase.calc_solar_predictions (DAO branch) used to
build their result frame with df.loc[df.shape[0]] = row inside a nested
loop -- one append per (interval, column) pair for save_df, one per
interval for calc_solar_predictions. Rewritten to collect plain tuples and
build the frame once; these tests pin the resulting values down.
"""

import datetime

import pandas as pd
import pytest

pytest.importorskip("pandas")

from dao.lib.db_manager import DBmanagerObj  # noqa: E402
from dao.prog.da_base import DaBase  # noqa: E402

HOUR = 3600


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
    for name in ("prognoses", "values"):
        Table(
            name,
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
            insert(variabel),
            [
                {"id": 1, "code": "pl", "name": "Plan load", "dim": "kWh"},
                {"id": 2, "code": "pv", "name": "Plan pv", "dim": "kWh"},
                {"id": 3, "code": "gr", "name": "Globale straling", "dim": "J/cm2"},
                {"id": 4, "code": "temp", "name": "Temperatuur", "dim": "C"},
            ],
        )
    return manager


def stored(db, tablename="values"):
    from sqlalchemy import Table, select

    values = Table(tablename, db.metadata, autoload_with=db.engine)
    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    with db.engine.connect() as connection:
        rows = connection.execute(
            select(variabel.c.code, values.c.time, values.c.value)
            .select_from(values.join(variabel, values.c.variabel == variabel.c.id))
            .order_by(values.c.time, variabel.c.code)
        )
        return [(code, int(time), value) for code, time, value in rows]


def make_da_base(db):
    base = DaBase.__new__(DaBase)
    base.db_da = db
    base.time_zone = "UTC"
    # save_df converts through db_da.epoch, so the database layer has to
    # carry the same zone. In production DaBase.__init__ pushes Home
    # Assistant's zone into the managers for exactly this reason.
    db.TARGET_TIMEZONE = "UTC"
    return base


class TestSaveDf:
    def test_melts_each_column_into_its_own_code_row(self, db):
        base = make_da_base(db)
        tijd = [
            datetime.datetime(2026, 6, 1, 10, 0),
            datetime.datetime(2026, 6, 1, 11, 0),
        ]
        df = pd.DataFrame(
            {"tijd": tijd, "pl": [1.5, 2.5], "pv": [0.5, 0.75]}
        )

        base.save_df("values", tijd, df)

        t0 = int(datetime.datetime(2026, 6, 1, 10, tzinfo=datetime.timezone.utc).timestamp())
        t1 = int(datetime.datetime(2026, 6, 1, 11, tzinfo=datetime.timezone.utc).timestamp())
        assert stored(db) == [
            ("pl", t0, 1.5),
            ("pv", t0, 0.5),
            ("pl", t1, 2.5),
            ("pv", t1, 0.75),
        ]

    def test_shorter_tijd_list_truncates_the_dataframe(self, db):
        """The original loop bounded on min(len(tijd), len(df)); a caller
        passing fewer timestamps than rows must not raise or save extras."""
        base = make_da_base(db)
        tijd = [datetime.datetime(2026, 6, 1, 10, 0)]
        df = pd.DataFrame(
            {"tijd": [tijd[0], datetime.datetime(2026, 6, 1, 11, 0)], "pl": [1.5, 2.5]}
        )

        base.save_df("values", tijd, df)

        t0 = int(datetime.datetime(2026, 6, 1, 10, tzinfo=datetime.timezone.utc).timestamp())
        assert stored(db) == [("pl", t0, 1.5)]


class TestCalcSolarPredictionsDaoBranch:
    def test_builds_one_row_per_prognose_interval(self, db, monkeypatch):
        from types import SimpleNamespace

        base = make_da_base(db)
        base.interval = "1hour"
        base.interval_s = 3600

        vanaf = datetime.datetime(2026, 6, 1, 10, 0)
        tot = datetime.datetime(2026, 6, 1, 12, 0)
        solar_option = SimpleNamespace(name="Test Panel")

        class StubPVService:
            def __init__(self):
                self.calls = []

            def forecast(self, installation, start, end, interval):
                self.calls.append((installation, start, end, interval))
                tijd = pd.date_range(start, end, freq="h", inclusive="left")
                return pd.DataFrame(
                    {"tijd": tijd, "prediction": [1.0, 2.0][: len(tijd)]}
                )

        stub = StubPVService()
        monkeypatch.setattr(base, "pv_service", lambda: stub)

        result = base.calc_solar_predictions(
            solar_option, vanaf, tot, interval="1hour"
        )

        assert list(result.columns) == ["tijd", "prediction"]
        assert list(result["prediction"]) == [1.0, 2.0]
        assert len(result) == 2
        assert stub.calls == [(solar_option, vanaf, tot, "1hour")]
