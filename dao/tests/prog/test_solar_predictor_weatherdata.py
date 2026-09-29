"""import_weatherdata / get_and_save_knmi_data: building the save frame.

Both used to append three rows (temp, gr, winds) per source row with
save_df.loc[save_df.shape[0]] = [...], which is O(n^2) -- three years of
hourly KNMI data is roughly 26000 source rows, ~78000 such appends. Rewritten
to collect a plain list of tuples and build the DataFrame once.
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


def test_get_and_save_knmi_data_saves_three_codes_per_row(db, monkeypatch):
    # knmi-py returns a naive DatetimeIndex (UTC-implied); the code itself
    # localizes it to self.time_zone.
    knmi_frame = pd.DataFrame(
        {"T": [100, 110], "Q": [50, 60], "FH": [30, 40]},
        index=pd.to_datetime(["2026-03-15 10:00:00", "2026-03-15 11:00:00"]),
    )

    def fake_get_hour_data_dataframe(stations, start, end, variables):
        return knmi_frame

    monkeypatch.setattr(
        "dao.prog.solar_predictor.knmi.get_hour_data_dataframe",
        fake_get_hour_data_dataframe,
    )
    predictor = make_predictor(db)

    predictor.get_and_save_knmi_data(
        dt.datetime(2026, 3, 15), dt.datetime(2026, 3, 16)
    )

    t0 = int(pd.Timestamp("2026-03-15 10:00:00", tz="UTC").timestamp())
    t1 = int(pd.Timestamp("2026-03-15 11:00:00", tz="UTC").timestamp())
    assert stored(db) == [
        ("gr", t0, 50.0),
        ("temp", t0, 10.0),
        ("winds", t0, 3.0),
        ("gr", t1, 60.0),
        ("temp", t1, 11.0),
        ("winds", t1, 4.0),
    ]


def test_get_and_save_knmi_data_only_saves_requested_variables(db, monkeypatch):
    """variables=["Q"] means only "gr" ends up in the frame; the code must
    not try to save temp/winds that were never fetched."""
    knmi_frame = pd.DataFrame(
        {"Q": [50]},
        index=pd.to_datetime(["2026-03-15 10:00:00"]),
    )
    monkeypatch.setattr(
        "dao.prog.solar_predictor.knmi.get_hour_data_dataframe",
        lambda stations, start, end, variables: knmi_frame,
    )
    predictor = make_predictor(db)

    predictor.get_and_save_knmi_data(
        dt.datetime(2026, 3, 15), dt.datetime(2026, 3, 16), variables=["Q"]
    )

    codes = [code for code, _, _ in stored(db)]
    assert codes == ["gr"]


def test_get_and_save_knmi_data_is_a_no_op_when_nothing_is_returned(db, monkeypatch):
    monkeypatch.setattr(
        "dao.prog.solar_predictor.knmi.get_hour_data_dataframe",
        lambda stations, start, end, variables: pd.DataFrame(),
    )
    predictor = make_predictor(db)

    predictor.get_and_save_knmi_data(dt.datetime(2026, 3, 15), dt.datetime(2026, 3, 16))

    assert stored(db) == []
