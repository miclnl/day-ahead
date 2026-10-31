"""DaBase.production_bonus_remaining_kwh(): the yearly cap against the consolidated feed-in."""

import datetime

import pytest

pytest.importorskip("pandas")

from dao.lib.db_manager import DBmanagerObj  # noqa: E402
from dao.prog.config.models.production_bonus import ProductionBonusConfig  # noqa: E402
from dao.prog.da_report import Report  # noqa: E402

ANCHOR = datetime.date(2025, 8, 10)
MOMENT = datetime.datetime(2027, 2, 1, 0, 0)


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
                {"id": 1, "code": "da", "name": "Day ahead", "dim": "eur"},
                {"id": 2, "code": "prod", "name": "Productie", "dim": "kWh"},
            ],
        )
    return manager


def insert_prod(db, rows):
    from sqlalchemy import Table, insert

    values = Table("values", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        connection.execute(
            insert(values),
            [{"variabel": 2, "time": int(moment.timestamp()), "value": kwh} for moment, kwh in rows],
        )


def make_report(db, cfg):
    report = Report.__new__(Report)
    report.db_da = db
    report.production_bonus = cfg
    return report


def test_sparse_legacy_exports_do_not_prove_cap_position(db):
    insert_prod(
        db,
        [
            (datetime.datetime(2026, 12, 20, 12), 1000.0),  # before the start date: does not count
            (datetime.datetime(2027, 1, 5, 10), 100.0),
            (datetime.datetime(2027, 1, 6, 10), 200.0),
        ],
    )
    report = make_report(db, ProductionBonusConfig(enabled=True, delivery_year_start=ANCHOR))
    assert report.production_bonus_remaining_kwh(MOMENT) == 0.0


def test_future_legacy_exports_do_not_prove_complete_prefix(db):
    insert_prod(db, [(datetime.datetime(2027, 3, 1, 10), 100.0)])
    report = make_report(db, ProductionBonusConfig(enabled=True, delivery_year_start=ANCHOR))
    assert report.production_bonus_remaining_kwh(MOMENT) == 0.0


def test_unknown_state_means_no_positive_bonus_and_one_warning(db, caplog):
    report = make_report(db, ProductionBonusConfig(enabled=True))
    assert report.production_bonus_remaining_kwh(MOMENT) == 0.0
    assert report.production_bonus_remaining_kwh(MOMENT) == 0.0
    assert caplog.text.count("afrekenstaat") == 1


def test_disabled_absent_or_uncapped_config_means_no_cap_and_no_query(db):
    def never(*args, **kwargs):
        raise AssertionError("no database query expected")

    db.get_column_data = never
    assert make_report(db, None).production_bonus_remaining_kwh(MOMENT) is None
    disabled = ProductionBonusConfig(enabled=False, delivery_year_start=ANCHOR)
    assert make_report(db, disabled).production_bonus_remaining_kwh(MOMENT) is None
    uncapped = ProductionBonusConfig(enabled=True, delivery_year_start=ANCHOR, **{"yearly cap kwh": None})
    assert make_report(db, uncapped).production_bonus_remaining_kwh(MOMENT) is None


def test_a_database_error_disables_positive_bonus_and_warns(db, caplog):
    report = make_report(db, ProductionBonusConfig(enabled=True, delivery_year_start=ANCHOR))

    def broken(*args, **kwargs):
        raise RuntimeError("database weg")

    report.settlement_service = broken
    assert report.production_bonus_remaining_kwh(MOMENT) == 0.0
    assert "database weg" in caplog.text


def test_compatibility_wrapper_uses_settlement_state(db):
    from decimal import Decimal
    from types import SimpleNamespace

    from dao.prog.settlement.types import BonusState, SettlementState
    report = make_report(db, ProductionBonusConfig(enabled=True, delivery_year_start=ANCHOR))
    at = db.epoch(MOMENT)
    state = SettlementState(at, BonusState(at - 86400, at + 86400, at, Decimal("4999")), None)
    report.settlement_service = lambda: SimpleNamespace(state_at=lambda moment: state)
    assert report.production_bonus_remaining_kwh(MOMENT) == 1.0
