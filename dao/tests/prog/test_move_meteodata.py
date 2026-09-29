"""CheckDB.move_meteodata_to_prognoses: the one-time values->prognoses move.

savedata() upserts, so moving a measurement into "prognoses" as-is would
silently replace any forecast already stored there for the same moment,
destroying the forecast-vs-measured comparison the forecast archive exists
for. Only rows "prognoses" does not have yet must move.
"""

import pytest

pytest.importorskip("pandas")

from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer, String, Table, UniqueConstraint, insert, select

from dao.lib.db_manager import DBmanagerObj
from dao.prog.check_db import CheckDB

HOUR = 3600
T0 = 1_800_000_000 // HOUR * HOUR


@pytest.fixture
def checker(tmp_path):
    instance = CheckDB.__new__(CheckDB)
    instance.db_da = DBmanagerObj(
        db_dialect="sqlite", db_name="day_ahead.db", db_path=str(tmp_path)
    )
    instance.engine = instance.db_da.engine
    metadata = instance.db_da.metadata
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
    metadata.create_all(instance.engine)
    with instance.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {"id": 4, "code": "gr", "name": "Globale straling", "dim": "J/cm2"},
                {"id": 5, "code": "temp", "name": "Temperatuur", "dim": "C"},
                {"id": 6, "code": "solar_rad", "name": "PV radiation", "dim": "J/cm2"},
                {"id": 23, "code": "winds", "name": "Windsnelheid", "dim": "m/s"},
                {"id": 24, "code": "neersl", "name": "Neerslag", "dim": "mm"},
            ],
        )
    return instance


def put(checker, table, variabel_id, rows):
    t = Table(table, checker.db_da.metadata, autoload_with=checker.engine)
    with checker.engine.begin() as connection:
        connection.execute(
            insert(t), [{"variabel": variabel_id, "time": time, "value": value} for time, value in rows]
        )


def stored(checker, table, variabel_id):
    t = Table(table, checker.db_da.metadata, autoload_with=checker.engine)
    with checker.engine.connect() as connection:
        rows = connection.execute(
            select(t.c.time, t.c.value).where(t.c.variabel == variabel_id).order_by(t.c.time)
        )
        return [(int(time), value) for time, value in rows]


def test_measurements_move_from_values_to_prognoses(checker):
    put(checker, "values", 4, [(T0, 100.0), (T0 + HOUR, 200.0)])

    checker.move_meteodata_to_prognoses()

    assert stored(checker, "values", 4) == []
    assert stored(checker, "prognoses", 4) == [(T0, 100.0), (T0 + HOUR, 200.0)]


def test_an_existing_forecast_is_not_overwritten_by_the_measurement(checker):
    """The real bug: a forecast for T0 already exists (it was made ahead of
    time and has not been superseded yet); the measurement for that same
    moment must not clobber it."""
    put(checker, "values", 5, [(T0, 12.0), (T0 + HOUR, 13.0)])
    put(checker, "prognoses", 5, [(T0, 99.0)])  # the pre-existing forecast

    checker.move_meteodata_to_prognoses()

    assert stored(checker, "prognoses", 5) == [(T0, 99.0), (T0 + HOUR, 13.0)]
    assert stored(checker, "values", 5) == []  # still fully removed from values


def test_solar_rad_is_removed_from_values_without_touching_prognoses(checker):
    """variabel id 6 (solar_rad) is only ever deleted from "values", never
    copied: it has no equivalent in "prognoses"."""
    put(checker, "values", 6, [(T0, 5.0)])

    checker.move_meteodata_to_prognoses()

    assert stored(checker, "values", 6) == []
    assert stored(checker, "prognoses", 6) == []


def test_a_variable_with_nothing_in_values_is_a_no_op(checker):
    checker.move_meteodata_to_prognoses()
    assert stored(checker, "prognoses", 4) == []
