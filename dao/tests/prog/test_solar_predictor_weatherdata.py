"""import_weatherdata: building the save frame.

Used to append three rows (temp, gr, winds) per source row with
save_df.loc[save_df.shape[0]] = [...], which is O(n^2) -- three years of
hourly KNMI data is roughly 26000 source rows, ~78000 such appends. Rewritten
to collect a plain list of tuples and build the DataFrame once.

get_and_save_knmi_data's own tests moved to
dao/tests/forecast/test_observations.py: that method is now a thin delegate
to update_observations, so the real coverage belongs on that function.
"""

import datetime as dt

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from dao.lib.db_manager import DBmanagerObj  # noqa: E402
from dao.prog.solar_predictor import SolarPredictor  # noqa: E402

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
            insert(variabel),
            [
                {"id": 1, "code": "temp", "name": "Temperatuur", "dim": "C"},
                {"id": 2, "code": "gr", "name": "Globale straling", "dim": "J/cm2"},
                {"id": 3, "code": "winds", "name": "Windsnelheid", "dim": "m/s"},
            ],
        )
    return manager


def make_predictor(db):
    predictor = SolarPredictor.__new__(SolarPredictor)
    predictor.db_da = db
    predictor.time_zone = "UTC"
    predictor.knmi_station = 275
    return predictor


def stored(db):
    from sqlalchemy import Table, select

    values = Table("values", db.metadata, autoload_with=db.engine)
    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    with db.engine.connect() as connection:
        rows = connection.execute(
            select(variabel.c.code, values.c.time, values.c.value)
            .select_from(values.join(variabel, values.c.variabel == variabel.c.id))
            .order_by(values.c.time, variabel.c.code)
        )
        return [(code, int(time), value) for code, time, value in rows]


def test_import_weatherdata_saves_three_codes_per_source_row(db, tmp_path):
    csv_path = tmp_path / "knmi_export.txt"
    csv_path.write_text(
        "# some header comment\n"
        "# another comment line\n"
        "# STN,YYYYMMDD,HH,   FH,    T,    Q\n"
        "275,20220101,    1,   50,  119,    0\n"
        "275,20220101,    2,   60,  117,    5\n"
    )
    predictor = make_predictor(db)

    predictor.import_weatherdata(str(csv_path))

    t0 = int(dt.datetime(2022, 1, 1, 0, tzinfo=dt.timezone.utc).timestamp())
    t1 = int(dt.datetime(2022, 1, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    assert stored(db) == [
        ("gr", t0, 0.0),
        ("temp", t0, 11.9),
        ("winds", t0, 5.0),
        ("gr", t1, 5.0),
        ("temp", t1, 11.7),
        ("winds", t1, 6.0),
    ]
    # The source file is consumed and removed.
    assert not csv_path.exists()
