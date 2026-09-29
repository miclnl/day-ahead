"""Shared fixtures for the forecast package tests.

Everything here is synthetic: invented entity ids, invented meter readings,
an in-memory recorder database. Nothing refers to a real installation.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("pandas")

from sqlalchemy import (  # noqa: E402
    BigInteger,
    Column,
    Float,
    Integer,
    String,
    Table,
    insert,
)

from dao.lib.db_manager import DBmanagerObj  # noqa: E402

TZ = "Europe/Amsterdam"
HOUR = 3600
#: 2026-03-02 00:00 local (CET) as epoch, aligned on a whole hour.
T0 = int(dt.datetime(2026, 3, 2, 0, 0, tzinfo=ZoneInfo(TZ)).timestamp())


class RecorderHelper:
    """Insert statistics rows into the fake recorder database."""

    def __init__(self, manager: DBmanagerObj, meta: Table, stats: Table):
        self.manager = manager
        self.meta = meta
        self.stats = stats
        self._next_meta_id = 1

    def _add_meta(self, statistic_id: str, unit: str, *, has_sum: int, mean_type: int) -> int:
        meta_id = self._next_meta_id
        self._next_meta_id += 1
        with self.manager.engine.begin() as connection:
            connection.execute(
                insert(self.meta).values(
                    id=meta_id,
                    statistic_id=statistic_id,
                    source="recorder",
                    unit_of_measurement=unit,
                    has_mean=None,
                    has_sum=has_sum,
                    name=None,
                    mean_type=mean_type,
                    unit_class=None,
                )
            )
        return meta_id

    def add_energy(self, statistic_id: str, unit: str, sums: dict[int, float]) -> int:
        """A cumulative energy sensor: ``sums`` maps start_ts -> reset-corrected sum."""
        meta_id = self._add_meta(statistic_id, unit, has_sum=1, mean_type=0)
        rows = [
            {
                "metadata_id": meta_id,
                "start_ts": float(start_ts),
                "created_ts": float(start_ts + HOUR),
                "state": value,
                "sum": value,
                "mean": None,
            }
            for start_ts, value in sorted(sums.items())
        ]
        with self.manager.engine.begin() as connection:
            connection.execute(insert(self.stats), rows)
        return meta_id

    def add_energy_with_state(
        self, statistic_id: str, unit: str, rows: dict[int, tuple[float, float]]
    ) -> int:
        """Like add_energy, but ``rows`` maps start_ts -> (state, sum) so a meter reset can be modelled."""
        meta_id = self._add_meta(statistic_id, unit, has_sum=1, mean_type=0)
        payload = [
            {
                "metadata_id": meta_id,
                "start_ts": float(start_ts),
                "created_ts": float(start_ts + HOUR),
                "state": state,
                "sum": total,
                "mean": None,
            }
            for start_ts, (state, total) in sorted(rows.items())
        ]
        with self.manager.engine.begin() as connection:
            connection.execute(insert(self.stats), payload)
        return meta_id

    def add_power(self, statistic_id: str, unit: str, means: dict[int, float]) -> int:
        """A power sensor: ``means`` maps start_ts -> hourly mean."""
        meta_id = self._add_meta(statistic_id, unit, has_sum=0, mean_type=1)
        rows = [
            {
                "metadata_id": meta_id,
                "start_ts": float(start_ts),
                "created_ts": float(start_ts + HOUR),
                "state": None,
                "sum": None,
                "mean": value,
            }
            for start_ts, value in sorted(means.items())
        ]
        with self.manager.engine.begin() as connection:
            connection.execute(insert(self.stats), rows)
        return meta_id

    def add_other(self, statistic_id: str, unit: str = "%") -> int:
        """A sensor that is neither energy nor power (state of charge, temperature)."""
        return self._add_meta(statistic_id, unit, has_sum=0, mean_type=1)


@pytest.fixture
def ha_db(tmp_path):
    """A recorder-like database with ``statistics`` and ``statistics_meta``.

    Returns ``(DBmanagerObj, RecorderHelper)``.
    """
    manager = DBmanagerObj(
        db_dialect="sqlite", db_name="homeassistant.db", db_path=str(tmp_path)
    )
    metadata = manager.metadata
    meta = Table(
        "statistics_meta",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("statistic_id", String(255)),
        Column("source", String(32)),
        Column("unit_of_measurement", String(255)),
        Column("has_mean", Integer, nullable=True),
        Column("has_sum", Integer),
        Column("name", String(255), nullable=True),
        Column("mean_type", Integer),
        Column("unit_class", String(255), nullable=True),
    )
    stats = Table(
        "statistics",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("created_ts", Float),
        Column("metadata_id", Integer),
        Column("start_ts", Float),
        Column("mean", Float, nullable=True),
        Column("min", Float, nullable=True),
        Column("max", Float, nullable=True),
        Column("last_reset_ts", Float, nullable=True),
        Column("state", Float, nullable=True),
        Column("sum", Float, nullable=True),
        Column("mean_weight", Float, nullable=True),
    )
    metadata.create_all(manager.engine, tables=[meta, stats])
    return manager, RecorderHelper(manager, meta, stats)


@pytest.fixture
def da_db(tmp_path):
    """A day_ahead database with variabel/values/prognoses in the production shape."""
    from sqlalchemy import ForeignKey, Index, UniqueConstraint

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
    tables = [variabel]
    for name in ("values", "prognoses"):
        tables.append(
            Table(
                name,
                metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
                Column("time", BigInteger, nullable=False),
                Column("value", Float),
                UniqueConstraint("variabel", "time"),
                Index(f"ix_{name}_time", "time"),
            )
        )
    metadata.create_all(manager.engine, tables=tables)
    codes = [
        (1, "cons", "Verbruik", "kWh", "sum"),
        (2, "prod", "Productie", "kWh", "sum"),
        (3, "da", "Tarief", "euro/kWh", "avg"),
        (4, "gr", "Globale straling", "J/cm2", "avg"),
        (5, "temp", "Temperatuur", "°C", "avg"),
        (11, "base", "Basislast", "kWh", "sum"),
        (15, "pv_ac", "Zonne energie AC", "kWh", "sum"),
        (17, "pv_dc", "Zonne energie DC", "kWh", "sum"),
        (23, "winds", "Windsnelheid", "m/s", "avg"),
        (24, "neersl", "Neerslag", "mm", "sum"),
        (25, "m_house", "Gemeten huisvraag", "kWh", "sum"),
        (26, "m_pv", "Gemeten pv productie", "kWh", "sum"),
        (27, "hload", "Geplande huisvraag", "kWh", "sum"),
        (28, "dni", "Directe straling", "J/cm2", "avg"),
        (29, "dhi", "Diffuse straling", "J/cm2", "avg"),
        (30, "away", "Afwezig", "-", "avg"),
        (31, "presence", "Aanwezigheid", "-", "avg"),
    ]
    with manager.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {"id": i, "code": c, "name": n, "dim": d, "aggregate": a}
                for i, c, n, d, a in codes
            ],
        )
    return manager
