"""E1f: the real chain from recorder to report to CLI; synthetic values only.

Every fixture in this file is a real, temporary database and a real adapter.
No hand-made book replaces the registered one, no supplier value replaces the
book, and no test claims anything about a server, a supplier contract or a
production period.
"""

import json
import logging
import socket
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlalchemy import text

from dao.prog.settlement.publication import make_publication
from dao.prog.settlement.service import (
    DatabasePriceAdapter,
    HAMeteringAdapter,
    SettlementService,
)
from dao.prog.settlement.tariffs import make_snapshot
from dao.prog.settlement.types import MeterSelection
from dao.tests.prog.test_consolidate import db as db
from dao.tests.settlement.conftest import MONTH_END_TS, START_TS
from dao.tests.settlement.test_metering import put_counter, short_points
from dao.tests.settlement.test_metering import recorder as recorder
from dao.tests.settlement.test_publication import pricing_config

METER = "synthetic-grid"
#: The one explicit installation selection of this chain: two grid sensors and
#: an explicit no-battery statement.  Nothing is inferred from a default.
SELECTION = MeterSelection(METER, ("sensor.import",), ("sensor.export",),
                            (), "synthetic-installation-record")
ZONE = ZoneInfo("Europe/Amsterdam")
QUARTER = 900
HOUR = 3600
#: The two real Amsterdam DST days of 2027: 92 and 100 quarters.
SPRING_DAY = (datetime(2027, 3, 28, tzinfo=ZONE), datetime(2027, 3, 29, tzinfo=ZONE))
AUTUMN_DAY = (datetime(2027, 10, 31, tzinfo=ZONE), datetime(2027, 11, 1, tzinfo=ZONE))
#: The DA price row of the temporary price database.
DA_VARIABLE = 5
#: The documentation of this chain; its examples are validated, not trusted.
DOCS_PATH = Path(__file__).resolve().parents[2] / "DOCS.md"


def put_price(db, at, value, *, variabel=DA_VARIABLE):
    """One real price row in the DA price table of the temporary database."""
    with db.engine.begin() as connection:
        connection.execute(text(
            'INSERT OR IGNORE INTO variabel (id,code,name,dim,aggregate) '
            "VALUES (:variabel,'da','Prijs','eur','avg')"), {"variabel": variabel})
        connection.execute(text(
            'INSERT INTO "values" (variabel,time,value) VALUES (:variabel,:at,:value)'),
            {"variabel": variabel, "at": at, "value": value})


def put_prices(db, start, end, value):
    """One real price row per contract hour of the half-open window."""
    for at in range(start // HOUR * HOUR, (end + HOUR - 1) // HOUR * HOUR, HOUR):
        put_price(db, at, value)


def put_grid(recorder, start, end, *, increment="0.1", initial="10"):
    """Real recorder buckets for both grid directions of the selection."""
    points = short_points(start, end, increment=increment, initial=initial)
    for sensor in ("sensor.import", "sensor.export"):
        put_counter(recorder, sensor, points)


def put_steps(recorder, sensor, start, end, steps):
    """Cumulative recorder sums with explicit kWh steps at quarter starts.

    The HA convention counts a bucket from its start to its end; the delta of a
    quarter is therefore the difference between its last and its preceding
    bucket.  ``steps`` maps the start of a quarter to the kWh added in exactly
    that quarter; every other quarter adds nothing.
    """
    moments = sorted(steps)
    points = []
    for at in range(start - 300, end - 300 + 1, 300):
        total = Decimal("10") + sum((Decimal(steps[moment]) for moment in moments
                                     if moment <= at), Decimal("0"))
        points.append((at, total))
    put_counter(recorder, sensor, points)
    return points


def make_recorder_db(path):
    """A second real recorder database with the same HA-shaped tables."""
    from sqlalchemy import Column, Float, Integer, String, Table

    from dao.lib.db_manager import DBmanagerObj

    Path(path).mkdir(parents=True, exist_ok=True)
    db = DBmanagerObj("sqlite", "recorder.db", db_path=str(path))
    Table(
        "statistics_meta", db.metadata,
        Column("id", Integer, primary_key=True),
        Column("statistic_id", String),
        Column("unit_of_measurement", String),
        Column("has_sum", Integer),
    )
    for name in ("statistics", "statistics_short_term"):
        Table(
            name, db.metadata, Column("id", Integer, primary_key=True),
            Column("metadata_id", Integer), Column("start_ts", Float),
            Column("state", Float), Column("sum", Float),
            Column("last_reset_ts", Float),
        )
    db.metadata.create_all(db.engine)
    return db


def make_price_db(path):
    """A second real DA database with the variabel/values tables of the chain."""
    from sqlalchemy import (
        BigInteger,
        Column,
        Float,
        ForeignKey,
        Integer,
        String,
        Table,
        UniqueConstraint,
    )

    from dao.lib.db_manager import DBmanagerObj

    Path(path).mkdir(parents=True, exist_ok=True)
    db = DBmanagerObj("sqlite", "day_ahead.db", db_path=str(path))
    metadata = db.metadata
    variabel = Table(
        "variabel", metadata,
        Column("id", Integer, primary_key=True),
        Column("code", String(10), unique=True, nullable=False),
        Column("name", String(50), unique=True, nullable=False),
        Column("dim", String(10), nullable=False),
        Column("aggregate", String(3), nullable=False, default="avg"),
    )
    Table(
        "values", metadata,
        Column("id", Integer, primary_key=True),
        Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
        Column("time", BigInteger, nullable=False),
        Column("value", Float),
        UniqueConstraint("variabel", "time"),
    )
    metadata.create_all(db.engine)
    with db.engine.begin() as connection:
        connection.execute(variabel.insert(), [
            {"id": 1, "code": "cons", "name": "Verbruik", "dim": "kWh"},
            {"id": 2, "code": "prod", "name": "Productie", "dim": "kWh"},
            {"id": 3, "code": "cost", "name": "Kosten", "dim": "eur"},
            {"id": 4, "code": "profit", "name": "Opbrengst", "dim": "eur"},
            {"id": DA_VARIABLE, "code": "da", "name": "Prijs", "dim": "eur"},
        ])
    return db


def options_document(tmp_path, *, pricing, meter=METER, database=None):
    """The operator configuration of this chain; explicit meter, no secrets."""
    prices = pricing.model_dump(mode="json", by_alias=True)
    prices["settlement"]["meter id"] = meter
    return {
        "database da": database or {"engine": "sqlite", "database": "day_ahead.db",
                                    "db_path": str(tmp_path)},
        "database ha": {"engine": "sqlite", "database": "recorder.db",
                        "db_path": str(tmp_path)},
        "time_zone": "Europe/Amsterdam",
        "prices": prices,
    }


def supplier_document(**changes):
    """One normalised supplier period; every amount is synthetic text."""
    document = {
        "schema_version": "supplier-period-v1",
        "meter_id": METER,
        "contract_id": "synthetic-2027",
        "timezone": "Europe/Amsterdam",
        "start_ts": START_TS,
        "end_ts": START_TS + 2 * QUARTER,
        "currency": "EUR",
        "energy_unit": "kWh",
        "sign_convention": "cost_positive_revenue_positive",
        "total_scope": "book_close_events_v1",
        "import_kwh": "0.6",
        "export_kwh": "0.6",
        "import_amounts": {"ex_vat": "0.06", "vat": "0", "gross": "0.06"},
        "base_export_amounts": {"ex_vat": "0.06", "vat": "0", "gross": "0.06"},
        "bonus_amounts": {"ex_vat": "0", "vat": "0", "gross": "0"},
        "credits": [],
        "total": "0",
        "rounding": {
            "method": "period_component_half_up_v1",
            "money_quantum": "0.01",
            "money_mode": "ROUND_HALF_UP",
            "aggregation": "sum_unrounded_then_round_each_component",
            "total_rule": "sum_rounded_gross_components",
            "energy_quantum": "0.001",
            "energy_mode": "ROUND_HALF_UP",
            "evidence_ref": "synthetic-rounding-v1",
        },
    }
    document.update(changes)
    return document


def write_supplier(tmp_path, document=None, *, name="supplier.json"):
    path = Path(tmp_path) / name
    path.write_text(json.dumps(supplier_document() if document is None else document),
                    encoding="utf-8")
    return path


def build_chain(tmp_path, recorder, price_db, monkeypatch, pricing, *, clock):
    """One real service, repository, publication and reader on one database."""
    for module in ("service", "repository", "publication"):
        monkeypatch.setattr(f"dao.prog.settlement.{module}.time", lambda: clock[0])
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    engine = price_db.engine
    plan_path = Path(tmp_path) / "fast_plan.json"
    repo, publication = make_publication(engine, plan_path=plan_path, pricing=pricing,
                                         max_plan_age=600)
    repo.ensure_schema()
    service = SettlementService(
        repo, HAMeteringAdapter(recorder, SELECTION),
        DatabasePriceAdapter(price_db, (snapshot,)), (snapshot,), publication=publication)
    config = Path(tmp_path) / "options.json"
    config.write_text(json.dumps(options_document(tmp_path, pricing=pricing)),
                      encoding="utf-8")
    (Path(tmp_path) / "secrets.json").write_text("{}", encoding="utf-8")
    return SimpleNamespace(recorder=recorder, price_db=price_db, repo=repo,
                           publication=publication, service=service, pricing=pricing,
                           snapshot=snapshot, config=config, plan_path=plan_path,
                           clock=clock, tmp_path=Path(tmp_path), engine=engine)


def clock_after(ts, *, margin=3600):
    """One explicit clock after the closed data window of a single chain."""
    return [ts + margin]


@pytest.fixture
def price_db(db):
    """The real DA price database of this chain: the existing DA fixture."""
    return db


@pytest.fixture
def clock():
    """One explicit clock for every module of the chain."""
    return [START_TS + 90 * 86400]


@pytest.fixture
def chain(tmp_path, recorder, price_db, monkeypatch, clock):
    """The real service, repository, publication and reader on one database."""
    chain = build_chain(tmp_path, recorder, price_db, monkeypatch, pricing_config(),
                        clock=clock)
    yield chain
    chain.engine.dispose()


def make_report(chain):
    """A report bound to the very same service and database as the chain."""
    from dao.prog.da_report import Report

    instance = Report.__new__(Report)
    instance.config = SimpleNamespace(prices=chain.pricing)
    instance.prices_options = chain.pricing
    instance.time_zone = "Europe/Amsterdam"
    instance.db_da = SimpleNamespace(
        epoch=lambda value: int(value.replace(tzinfo=ZONE).timestamp())
        if value.tzinfo is None else int(value.timestamp()),
        engine=chain.price_db.engine, metadata=chain.price_db.metadata)
    instance.settlement_service = lambda: chain.service
    instance.periodes = {"chain": {"vanaf": START_TS, "tot": START_TS + 2 * QUARTER,
                                   "interval": "uur"}}
    return instance


def docs_example(marker):
    """One marked JSON example out of the documentation; never a real invoice."""
    text = DOCS_PATH.read_text(encoding="utf-8")
    start, stop = f"<!-- e1f:{marker} -->", f"<!-- /e1f:{marker} -->"
    assert start in text and stop in text, marker
    block = text.split(start, 1)[1].split(stop, 1)[0]
    return json.loads(block.split("```json", 1)[1].split("```", 1)[0])


def row_counts(chain):
    """The stored row counts of the registered book; no file or object identity."""
    from sqlalchemy import func, select

    from dao.prog.settlement.repository import _METADATA

    counts = {}
    with chain.price_db.engine.connect() as connection:
        for name in ("interval", "run", "adjustment", "input"):
            table = _METADATA.tables[f"settlement_{name}"]
            counts[name] = connection.scalar(select(func.count()).select_from(table))
    return counts


def legacy_rows(chain):
    """The legacy cost/profit/prognosis rows that the settlement route must not touch."""
    with chain.price_db.engine.connect() as connection:
        values = connection.execute(text(
            'SELECT variabel,time,value FROM "values" WHERE variabel IN (3,4) '
            "ORDER BY variabel,time")).all()
        prognoses = connection.execute(text(
            "SELECT variabel,time,value FROM prognoses ORDER BY variabel,time")).all()
    return values, prognoses


def restate(recorder, sensor, points):
    """Correct stored recorder buckets; the correction enters through the service."""
    with recorder.engine.begin() as connection:
        for at, total in points:
            connection.execute(text(
                'UPDATE "statistics_short_term" SET sum=:total, state=:total '
                "WHERE metadata_id=(SELECT id FROM statistics_meta WHERE statistic_id=:sensor) "
                "AND start_ts=:at"), {"total": float(total), "at": at, "sensor": sensor})


def drop_bucket(recorder, sensor, at):
    """Remove one stored recorder bucket; the source itself is then missing."""
    with recorder.engine.begin() as connection:
        connection.execute(text(
            'DELETE FROM "statistics_short_term" '
            "WHERE metadata_id=(SELECT id FROM statistics_meta WHERE statistic_id=:sensor) "
            "AND start_ts=:at"), {"at": at, "sensor": sensor})


def test_e1f_real_adapters_book_report_api_cli(chain, capsys):
    """Recorder and price database feed the book; report and CLI read it back."""
    from dao.prog import da_settlement

    put_grid(chain.recorder, START_TS, START_TS + 2 * QUARTER)
    put_price(chain.price_db, START_TS, 0.1)
    view = chain.service.consolidate(START_TS, START_TS + 2 * QUARTER, publish=True,
                                     reason="e1f-chain")
    assert view.generation == 1 and view.freshness == "current"
    assert [line.import_cost for line in view.lines] == [Decimal("0.03")] * 2
    assert [line.base_export_revenue for line in view.lines] == [Decimal("0.03")] * 2
    assert [line.bonus_revenue for line in view.lines] == [Decimal("0")] * 2
    assert [line.quality for line in view.lines] == ["estimated"] * 2
    assert view.net_cost == Decimal("0")
    assert view.complete is True

    # The same book, read through the shared historical reader.
    basis = chain.repo.read_reconciliation(METER, START_TS, START_TS + 2 * QUARTER)
    assert basis.view.net_cost == Decimal("0")
    assert [source.item.import_kwh for source in basis.sources.lines] == [Decimal("0.3")] * 2
    assert [source.item.export_kwh for source in basis.sources.lines] == [Decimal("0.3")] * 2
    assert [source.item.quality for source in basis.sources.lines] == ["estimated"] * 2
    assert not any(source.issues for source in basis.sources.lines)

    # The report keeps the very same amounts, pins and proven energy.
    report = make_report(chain)
    frame = report.get_settlement_data(START_TS, START_TS + 2 * QUARTER)
    assert list(frame.import_cost) == [Decimal("0.03")] * 2
    assert list(frame.base_export_revenue) == [Decimal("0.03")] * 2
    assert list(frame.net_cost) == [Decimal("0")] * 2
    assert set(frame.generation) == {1}
    assert set(frame.quality) == {"estimated"}
    grid = report.get_grid_data("chain")
    # Both quarters fall in one UTC hour, so the hour row aggregates them.
    assert list(grid.cost) == [Decimal("0.06")]
    assert list(grid.profit) == [Decimal("0.06")]
    assert list(grid.consumption) == [Decimal("0.6")]
    assert list(grid.production) == [Decimal("0.6")]
    assert list(grid.netto_cost) == [Decimal("0")]
    assert set(grid.generation) == {1}
    assert set(grid.quality) == {"estimated"}
    payload = json.loads(report.get_api_data("grid", "chain"))
    assert [row["cost"] for row in payload["data"]] == ["0.06"]
    assert [row["consumption"] for row in payload["data"]] == ["0.6"]

    # The historical lookup costs exactly one extra read per grid call.
    historical_reads = []
    original_read = chain.repo.read_reconciliation

    def counted_read(*args, **kwargs):
        historical_reads.append((args, kwargs))
        return original_read(*args, **kwargs)

    chain.repo.read_reconciliation = counted_read
    grid_calls = []
    original_grid = report._book_grid

    def counted_grid(*args, **kwargs):
        grid_calls.append((args, kwargs))
        return original_grid(*args, **kwargs)

    report._book_grid = counted_grid
    report.get_grid_data("chain")
    assert len(grid_calls) == 1
    # One extra read_reconciliation per _book_grid call: the historical lookup
    # that proves the kWh.  No caching is added for it.
    assert len(historical_reads) == len(grid_calls)
    assert len(historical_reads) == 1

    # The reports that settle every hour separately call _book_grid per hour,
    # so they pay one extra historical read per hour of the grid.
    hourly_reads = []
    hourly_grids = []

    def counted_hourly_read(*args, **kwargs):
        hourly_reads.append((args, kwargs))
        return original_read(*args, **kwargs)

    def counted_hourly_grid(*args, **kwargs):
        hourly_grids.append((args, kwargs))
        return original_grid(*args, **kwargs)

    chain.repo.read_reconciliation = counted_hourly_read
    report._book_grid = counted_hourly_grid
    hours = 2
    hourly = pd.DataFrame({
        "tijd": [pd.Timestamp(START_TS + index * HOUR, unit="s", tz="UTC")
                 .tz_convert(ZONE).tz_localize(None) for index in range(hours)],
        "utc": [START_TS + index * HOUR for index in range(hours)],
        "consumption": [0.3] * hours,
        "production": [0.3] * hours,
        "da_cons": [0.1] * hours,
        "da_prod": [0.1] * hours,
        "datasoort": ["recorded"] * hours,
    })
    report.recalc_df_ha(hourly, "uur")
    assert len(hourly_grids) == hours
    assert len(hourly_reads) == hours
    chain.repo.read_reconciliation = counted_read
    report._book_grid = counted_grid

    # The manual comparison route reaches the same book through its own reader.
    supplier = write_supplier(chain.tmp_path)
    exit_code = da_settlement.main([
        "compare-invoice", "--supplier", str(supplier), "--config", str(chain.config),
        "--start", str(START_TS), "--end", str(START_TS + 2 * QUARTER)])
    assert exit_code == 0
    report_json = json.loads(capsys.readouterr().out)
    assert report_json["status"] == "matched"
    assert report_json["freshness"] == "current"
    assert report_json["pin"]["generation"] == 1
    assert report_json["total"]["delta"] == "0"
    assert {component["quality"] for component in report_json["components"]} == {"estimated"}
    keys = {component["key"] for component in report_json["components"]}
    assert "import.gross" in keys and "base_export.gross" in keys
    assert "bonus.gross" in keys and "import_kwh" in keys and "export_kwh" in keys
    assert "net_cost" not in keys

    # The documented examples are validated against the real a/d schema.  The
    # files are written by this test from the documentation itself: no real
    # invoice, no supplier data and no network access are involved.
    from dao.prog.settlement.reconciliation_io import load_supplier

    documented = docs_example("supplier-period-example")
    path = chain.tmp_path / "documented-supplier.json"
    path.write_text(json.dumps(documented), encoding="utf-8")
    period = load_supplier(path)
    assert period.schema_version == "supplier-period-v1"
    assert period.meter_id == METER and period.contract_id == "synthetic-2027"
    assert period.timezone == "Europe/Amsterdam"
    assert (period.start_ts, period.end_ts) == (START_TS, START_TS + 2 * QUARTER)
    assert period.currency == "EUR" and period.energy_unit == "kWh"
    assert period.sign_convention == "cost_positive_revenue_positive"
    assert period.total_scope == "book_close_events_v1"
    assert period.import_kwh == Decimal("0.6") and period.export_kwh == Decimal("0.6")
    assert period.import_amounts.ex_vat == Decimal("0.06")
    assert period.import_amounts.vat == Decimal("0")
    assert period.import_amounts.gross == Decimal("0.06")
    assert period.base_export_amounts.gross == Decimal("0.06")
    assert period.bonus_amounts.gross == Decimal("0")
    assert period.credits == ()
    assert period.total == Decimal("0")
    assert period.rounding.method == "period_component_half_up_v1"
    assert period.rounding.money_quantum == Decimal("0.01")
    assert period.rounding.money_mode == "ROUND_HALF_UP"
    assert period.rounding.aggregation == "sum_unrounded_then_round_each_component"
    assert period.rounding.total_rule == "sum_rounded_gross_components"
    assert period.rounding.energy_quantum == Decimal("0.001")
    assert period.rounding.energy_mode == "ROUND_HALF_UP"
    assert period.rounding.evidence_ref == "synthetic-rounding-v1"

    # Every nullable field may be null, but it must be there explicitly.
    empty = docs_example("supplier-period-null-example")
    assert set(empty) == set(documented)
    path = chain.tmp_path / "documented-null-supplier.json"
    path.write_text(json.dumps(empty), encoding="utf-8")
    empty_period = load_supplier(path)
    assert empty_period.import_kwh is None and empty_period.export_kwh is None
    assert empty_period.import_amounts is None
    assert empty_period.base_export_amounts is None
    assert empty_period.bonus_amounts is None
    assert empty_period.credits is None
    assert empty_period.total is None and empty_period.rounding is None

    # Explicit nulls inside the amounts, the credit and the rounding survive.
    partial = docs_example("supplier-period-partial-null-example")
    path = chain.tmp_path / "documented-partial-supplier.json"
    path.write_text(json.dumps(partial), encoding="utf-8")
    partial_period = load_supplier(path)
    assert partial_period.import_amounts.ex_vat is None
    assert partial_period.import_amounts.vat is None
    assert partial_period.import_amounts.gross == Decimal("0.06")
    assert len(partial_period.credits) == 1
    credit = partial_period.credits[0]
    assert credit.kind == "spot_zero"
    assert (credit.period_start_ts, credit.period_end_ts) == (START_TS, MONTH_END_TS)
    assert credit.amounts.ex_vat is None and credit.amounts.vat is None
    assert credit.amounts.gross == Decimal("0.20")
    assert partial_period.rounding.evidence_ref is None
    assert partial_period.rounding.money_quantum == Decimal("0.01")


def test_e1f_partition_noop_and_late_corrections(chain):
    """Partial reports partition the book; late corrections publish a new generation."""
    from decimal import localcontext

    from dao.prog.fastctrl.publication import read_publication

    put_grid(chain.recorder, START_TS, START_TS + 2 * QUARTER)
    put_price(chain.price_db, START_TS, 0.1)
    with chain.price_db.engine.begin() as connection:
        connection.execute(text(
            'INSERT INTO "values" (variabel,time,value) VALUES (3,:at,1.5),(4,:at,2.5)'),
            {"at": START_TS})
        connection.execute(text("CREATE TABLE prognoses (id INTEGER PRIMARY KEY, "
                                "variabel INTEGER, time BIGINT, value FLOAT)"))
        connection.execute(text("INSERT INTO prognoses (variabel,time,value) "
                                "VALUES (4,:at,0.25)"), {"at": START_TS})
    legacy_before = legacy_rows(chain)

    view = chain.service.consolidate(START_TS, START_TS + 2 * QUARTER, publish=True,
                                     reason="e1f-first")
    assert view.generation == 1
    counts_before = row_counts(chain)
    assert counts_before["interval"] == 2 and counts_before["run"] == 1

    # Every source key and every credit belongs to exactly one part; the parts
    # add up to the whole scope without a tolerance and without a residue.
    whole = chain.repo.read_reconciliation(METER, START_TS, START_TS + 2 * QUARTER)
    first = chain.repo.read_reconciliation(METER, START_TS, START_TS + QUARTER)
    second = chain.repo.read_reconciliation(METER, START_TS + QUARTER, START_TS + 2 * QUARTER)
    whole_keys = {source.source_key for source in whole.sources.lines}
    first_keys = {source.source_key for source in first.sources.lines}
    second_keys = {source.source_key for source in second.sources.lines}
    assert first_keys and second_keys
    assert not (first_keys & second_keys)
    assert first_keys | second_keys == whole_keys
    first_credits = {credit.source_key for credit in first.sources.credits}
    second_credits = {credit.source_key for credit in second.sources.credits}
    assert not (first_credits & second_credits)
    assert first_credits | second_credits == {credit.source_key for credit in whole.sources.credits}
    with localcontext() as context:
        context.prec = 36
        assert (first.view.net_cost + second.view.net_cost) == whole.view.net_cost
        assert (first.view.known_subtotal + second.view.known_subtotal) == whole.view.known_subtotal

    # A repeated consolidation of identical sources changes nothing.
    repeated = chain.service.consolidate(START_TS, START_TS + 2 * QUARTER, publish=True,
                                         reason="e1f-noop")
    assert repeated.generation == view.generation
    assert row_counts(chain) == counts_before
    assert chain.repo.read(METER, START_TS, START_TS + 2 * QUARTER).lines == view.lines

    # A late recorder correction of the second quarter enters through the service.
    # The 300 s buckets of the second quarter carry three deltas; three 0.2
    # deltas make 0.6 kWh where the archived quarter held 0.3 kWh.
    restate(chain.recorder, "sensor.import", [
        (START_TS + 900, Decimal("10.5")),
        (START_TS + 1200, Decimal("10.7")),
        (START_TS + 1500, Decimal("10.9")),
    ])
    chain.clock[0] = START_TS + 2 * 86400
    corrected = chain.service.consolidate(START_TS, START_TS + 2 * QUARTER, publish=True,
                                          reason="e1f-recorder-correction")
    assert corrected.generation == view.generation + 1
    assert [line.import_cost for line in corrected.lines] == [Decimal("0.03"), Decimal("0.06")]
    assert sum(line.import_cost for line in corrected.lines) == Decimal("0.09")
    assert sum(line.base_export_revenue for line in corrected.lines) == Decimal("0.06")
    assert corrected.net_cost == Decimal("0.03")
    old = chain.repo.read(METER, START_TS, START_TS + 2 * QUARTER, view.generation)
    assert old.freshness == "outdated"
    assert [line.import_cost for line in old.lines] == [Decimal("0.03")] * 2
    assert old.net_cost == Decimal("0")
    blocked = read_publication(chain.plan_path)
    assert blocked.status == "blocked" and blocked.plan is None

    # A late price correction follows the same route and a new generation.
    with chain.price_db.engine.begin() as connection:
        connection.execute(text('UPDATE "values" SET value=0.2 WHERE variabel=5 AND time=:at'),
                           {"at": START_TS})
    chain.clock[0] = START_TS + 3 * 86400
    repriced = chain.service.consolidate(START_TS, START_TS + 2 * QUARTER, publish=True,
                                          reason="e1f-price-correction")
    assert repriced.generation == corrected.generation + 1
    assert [line.import_cost for line in repriced.lines] == [Decimal("0.06"), Decimal("0.12")]
    assert repriced.net_cost == Decimal("0.06")
    assert chain.repo.read(METER, START_TS, START_TS + 2 * QUARTER,
                            corrected.generation).freshness == "outdated"

    # A late contract correction registers a new tariff snapshot and a new generation.
    changed = chain.pricing.model_copy(update={"cost_supplier_consumption": {"2020-01-01": "0.01"}})
    chain.service.snapshots = (make_snapshot(changed, changed.settlement.contracts[0]),)
    chain.clock[0] = START_TS + 4 * 86400
    recontracted = chain.service.consolidate(START_TS, START_TS + 2 * QUARTER, publish=True,
                                             reason="e1f-contract-correction")
    assert recontracted.generation == repriced.generation + 1
    # The corrected contract adds 0.01 EUR ex-VAT supplier cost to consumption
    # only: 0.3 kWh and 0.6 kWh at 0.21 EUR, while the export rate stays 0.20.
    assert [line.import_cost for line in recontracted.lines] == [Decimal("0.063"), Decimal("0.126")]
    assert sum(line.base_export_revenue for line in recontracted.lines) == Decimal("0.12")
    assert recontracted.net_cost == Decimal("0.069")

    # The legacy cost/profit/prognosis rows are untouched by every correction.
    assert legacy_rows(chain) == legacy_before


def build_producer(chain, monkeypatch, *, start, quarters=12, baseload=0.4):
    """One offline DaCalc bound to the real service, repository and publication.

    The harness follows the existing producer-capture pattern, but the service,
    the repository and the publication are the real ones of this chain: the
    producer binds them before ``calc_optimum`` and never replaces them.
    """
    import pandas as pd

    import dao.prog.day_ahead as da
    from dao.prog.config.models.base import FlexFloat
    from dao.prog.config.models.devices.battery import BatteryConfig
    from dao.prog.config.models.production_bonus import ProductionBonusConfig

    found = {"ha_writes": [], "prepare_calls": [], "ready_publications": [],
             "forecast_frames": []}

    def forbidden(*args, **kwargs):
        raise AssertionError("offline producer attempted I/O")

    def ha_write(*args, **kwargs):
        found["ha_writes"].append((args, kwargs))

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(da, "FAST_PLAN_FILE", str(chain.plan_path))
    monkeypatch.setattr(da.time, "time", lambda: chain.clock[0])

    instance = da.DaCalc.__new__(da.DaCalc)
    instance.debug = True
    instance.file_name = "synthetic-e1-chain"
    instance.interval, instance.interval_s, instance.steps_day = "15min", 900, 96
    instance.interval_name, instance.time_zone = "kwartier", "Europe/Amsterdam"
    instance.log_level = logging.INFO
    instance.config = SimpleNamespace(
        prices=chain.pricing, baseload=[baseload] * 24, max_gap=FlexFloat(value=0.005),
        fast_control=SimpleNamespace(max_plan_age=600))
    instance.battery_options = [BatteryConfig.model_validate({
        "name": f"synthetic-{b}", "entity actual level": f"sensor.synthetic_{b}",
        "capacity": 100,
        "charge stages": [{"power": 2000, "efficiency": 1}, {"power": 4000, "efficiency": 1}],
        "discharge stages": [{"power": 2000, "efficiency": 1},
                             {"power": 4000, "efficiency": 1}],
        "bat_to_dc efficiency": efficiency, "dc_to_bat efficiency": 1, "cycle cost": 0.01,
        "solar": [{"name": f"pv-{b}", "max power": 4, "tilt": 30, "orientation": 0,
                   "capacity": 4}],
    }) for b, efficiency in enumerate((0.8, 1.0))]
    instance.ev_options, instance.machines, instance.solar = [], [], []
    instance.boiler_options = instance.heating_options = None
    instance.salderen = False
    instance.grid_max_power = 17.0
    instance.production_bonus = ProductionBonusConfig(enabled=False)
    instance.strategy = "minimize cost"
    instance._solver_threads = 1
    instance.use_calc_baseload = False
    instance.baseload_service = lambda: SimpleNamespace(record_presence=lambda: None)
    instance.get_float = lambda entity, *args: 50.0
    instance.ha_getter = forbidden
    instance.notify = lambda *args: None
    for name in ("set_value", "call_service", "turn_on", "turn_off", "set_entity_state",
                 "set_entity_option", "select_option"):
        monkeypatch.setattr(instance, name, ha_write)
    instance.save_df = lambda **kwargs: found["forecast_frames"].append(kwargs)
    instance.get_str = lambda entity, *args: "off"
    instance.grid = SimpleNamespace()

    grid_start = start // instance.interval_s * instance.interval_s
    stamps = [grid_start + instance.interval_s * i for i in range(quarters)]
    dates = [datetime.fromtimestamp(at) for at in stamps]
    #: Synthetic weather rows for the modelled horizon; no external forecast is read.
    forecast = pd.DataFrame({"time": stamps, "tijd": dates,
                             "glob_rad": [0.0] * quarters, "temp": [5.0] * quarters})

    def report(*args):
        return SimpleNamespace(get_price_data=lambda *a, **k: pd.DataFrame(),
                               get_heatpump_run_hours=lambda entity: 1)

    monkeypatch.setattr(da, "Report", report)
    instance.calc_solar_predictions = lambda *args: pd.DataFrame(
        {"prediction": [0.5] * quarters})
    instance.db_da = SimpleNamespace(
        get_prognose_data=lambda **kwargs: forecast.copy())

    export = da.DaCalc.export_fast_plan.__get__(instance)

    def export_plan(**kwargs):
        found["export"] = kwargs
        found["export_locals"] = dict(sys._getframe(1).f_locals)
        return export(**kwargs, path=str(chain.plan_path))

    monkeypatch.setattr(instance, "export_fast_plan", export_plan)

    prepare = chain.service.prepare_planning

    def prepare_planning(lower, upper):
        found["prepare_calls"].append((lower, upper))
        context = prepare(lower, upper)
        found["context"] = context
        return context

    monkeypatch.setattr(chain.service, "prepare_planning", prepare_planning)
    instance._settlement_service = chain.service
    instance._settlement_repository = chain.repo
    instance._settlement_publication = chain.publication

    real_publish = chain.publication.publish

    def publish(*args, **kwargs):
        found["ready_publications"].append((args, kwargs))
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(chain.publication, "publish", publish)

    instance._debug_capture_vars = True
    instance._debug_record_start = lambda *args: setattr(instance, "debug", False)
    found["instance"] = instance
    found["stamps"] = stamps
    return found

def test_e1f_producer_publication_archive_replay(tmp_path, recorder, price_db, monkeypatch):
    """The real producer publishes; the archive and the replay bind to that plan."""
    from dao.prog.fastctrl.command import plan_digest
    from dao.prog.fastctrl.publication import read_publication
    from dao.prog.fastctrl.simulate import Sample, simulate
    from dao.prog.settlement.publication import FinancialPlanBlocked
    from dao.prog.settlement.replay import read_replay_basis, replay_archive_path
    from dao.tests.prog.test_fast_simulate import modeled_case
    from dao.tests.settlement.test_fast_financial_replay import f3b4_limits, f3b4_source

    chain = build_chain(tmp_path, recorder, price_db, monkeypatch, pricing_config(),
                        clock=[START_TS + 60])
    try:
        put_prices(price_db, START_TS, START_TS + 3 * HOUR, 0.1)
        producer = build_producer(chain, monkeypatch, start=START_TS)
        instance = producer["instance"]
        with pytest.raises(FinancialPlanBlocked, match="actuation_gates_unproven"):
            instance.calc_optimum(_start_dt=datetime.fromtimestamp(START_TS))
        # The publication is real; only the actuation is refused.
        assert producer["ha_writes"] == []
        assert producer["ready_publications"]
        read = read_publication(chain.plan_path)
        assert read.status == "ready", read.reason
        plan = read.plan
        context = plan.financial_context
        observed = producer["context"]
        assert producer["prepare_calls"] == [(observed.basis.planning_start_ts,
                                              observed.basis.horizon_end_ts)]
        assert context.basis == observed.basis
        assert context.economics_supported is True

        # The archive holds the very basis, opening and prices of this producer.
        replay = read_replay_basis(chain.plan_path, context.replay_basis_hash,
                                   financial_basis=context.basis,
                                   terminal_value=context.terminal_value)
        assert replay.status == "known"
        assert replay.basis.basis == context.basis
        assert replay.basis.opening == observed.state
        assert replay.basis.price_batch == observed.price_batch
        assert replay.basis.quotes == observed.quotes
        assert replay.basis.content_hash == context.replay_basis_hash
        archive = replay_archive_path(chain.plan_path, context.replay_basis_hash)
        assert archive.exists()
        assert archive.name == context.replay_basis_hash

        # The replay is a synthetic replay assumption, never a supplier oracle.
        samples = [Sample(at, 400.0) for at in
                   range(observed.basis.planning_start_ts,
                         observed.basis.horizon_end_ts + 1, QUARTER)]
        scenario = modeled_case("synthetic-e1-chain", hold=QUARTER)
        evidence = f3b4_source(plan, replay.basis, samples, hold=QUARTER)
        result = simulate(plan, samples, f3b4_limits(),
                          {spec.name: 50.0 for spec in plan.specs},
                          "e1f-replay", False,
                          inverter_efficiency={spec.name: 1.0 for spec in plan.specs},
                          timezone=UTC, replay_basis=replay.basis, execution=scenario,
                          source_evidence=evidence)
        assert result.execution_kind == "modeled"
        assert result.coverage_status == "complete"
        # The source binding points at the very plan and archive of this producer.
        assert result.source_plan_digest == plan_digest(plan)
        assert result.execution_plan_digest == result.source_plan_digest
        assert result.opening_state == observed.state
        assert result.replay_terminal_value == context.terminal_value
        assert not result.settlement_comparison_pending
    finally:
        chain.engine.dispose()


def bonus_pricing(*, supply_start="2027-01-01", cap="5000", start_date="2027-01-01",
                  floor="none"):
    """Explicit production-bonus terms; every value is synthetic."""
    from dao.prog.config.models.pricing import PricingConfig
    from dao.prog.config.models.production_bonus import ProductionBonusConfig

    base = pricing_config()
    data = base.model_dump(by_alias=True)
    data["production bonus"] = ProductionBonusConfig(
        enabled=True, mode="pv only", start_date=start_date, percentage=30,
        max_exchange_price=0.20, windows={str(month): [0, 24] for month in range(1, 13)},
        yearly_cap_kwh=cap, delivery_year_start=supply_start, cap_basis="ex_vat",
        vat_percent=0)
    data["settlement"]["contracts"][0]["supply start"] = supply_start
    data["settlement"]["contracts"][0]["monthly export floor"] = floor
    return PricingConfig.model_validate(data)


def write_meter_csv(path, rows):
    """A normalised meter file with explicit bounds; the real CSV adapter reads it."""
    lines = ["start_ts,end_ts,import_kwh,export_kwh,battery_discharge_kwh,unit,quality,"
             "source,origin"]
    for start, end, imported, exported, quality, origin in rows:
        lines.append(f"{start},{end},{imported},{exported},0,kWh,{quality},"
                     f"synthetic-explicit-bounds,{origin}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


class CsvMetering:
    """The real normalised-file meter adapter behind one explicit file."""

    meter_id = METER

    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, start, end):
        from dao.prog.settlement.metering import read_meter_csv

        items = read_meter_csv(self.path, METER)
        return tuple(item for item in items if start <= item.start_ts and item.end_ts <= end)


def register_chain(chain, readings, opening, *, start, end, generation):
    """The real registration route: prices, inputs, settlement and one commit."""
    from dao.prog.settlement.engine import settle_series
    from dao.prog.settlement.metering import merge_inputs

    lower = start // HOUR * HOUR
    upper = (end + HOUR - 1) // HOUR * HOUR
    spots = chain.service.price_adapter(lower, upper)
    revision = chain.repo.read_book_reference(METER).input_revision
    # The tariff snapshot enters the book first; only then can the closed
    # price batch and the closed inputs be pinned against it.
    revision = chain.repo.register_inputs(METER, (), chain.service.snapshots, revision)
    revision = chain.repo.register_prices(METER, spots, revision)
    prices = chain.repo.read_prices(METER, lower, upper)
    items = merge_inputs(readings, spots, chain.service.snapshots, meter_id=METER)
    revision = chain.repo.register_inputs(METER, items, prices.snapshots, revision,
                                         price_batch=prices, closed_before_ts=end)
    batch = chain.repo.read_inputs(METER, lower, upper)
    result = settle_series(batch.items, batch.snapshots, opening)
    chain.repo.commit(METER, result, batch, generation, "e1f-registration")
    return revision


def test_e1f_negative_cap_month_year_dst_and_gaps(tmp_path, monkeypatch):
    """Negative price, bonus cap, month credit, delivery year and both DST days."""
    from dao.prog.settlement.types import BonusState, SettlementState

    # --- a negative price enters the book through the real adapters -----------
    negative = build_chain(tmp_path / "negative", make_recorder_db(tmp_path / "negative"),
                           make_price_db(tmp_path / "negative"), monkeypatch,
                           pricing_config(), clock=clock_after(START_TS + 2 * QUARTER))
    try:
        put_grid(negative.recorder, START_TS, START_TS + 2 * QUARTER)
        put_prices(negative.price_db, START_TS, START_TS + 2 * QUARTER, -0.1)
        view = negative.service.consolidate(START_TS, START_TS + 2 * QUARTER, publish=True,
                                            reason="e1f-negative")
        assert [line.import_cost for line in view.lines] == [Decimal("-0.03")] * 2
        assert [line.base_export_revenue for line in view.lines] == [Decimal("-0.03")] * 2
        assert view.net_cost == Decimal("0") and view.complete is True
        basis = negative.repo.read_reconciliation(METER, START_TS, START_TS + 2 * QUARTER)
        assert [source.item.spot_ex_vat for source in basis.sources.lines] == [Decimal("-0.1")] * 2
        assert not any(source.issues for source in basis.sources.lines)
    finally:
        negative.engine.dispose()

    # --- the two real Amsterdam DST days -------------------------------------
    # The synthetic contract starts on the spring day, so each day is a
    # self-contained book window: the day proves its own quarter count and UTC
    # placement instead of a hole before it.
    dst = build_chain(tmp_path / "dst", make_recorder_db(tmp_path / "dst"),
                      make_price_db(tmp_path / "dst"), monkeypatch,
                      pricing_config(start="2027-03-28"),
                      clock=clock_after(int(AUTUMN_DAY[1].timestamp())))
    try:
        for (first, second), quarters, kwh, amount in (
                (SPRING_DAY, 92, "27.6", "2.76"), (AUTUMN_DAY, 100, "30", "3")):
            start, end = int(first.timestamp()), int(second.timestamp())
            put_grid(dst.recorder, start, end)
            put_prices(dst.price_db, start, end, 0.1)
            view = dst.service.consolidate(start, end, publish=True, reason="e1f-dst")
            assert len(view.lines) == quarters
            assert sum(line.import_cost for line in view.lines) == Decimal(amount)
            assert sum(line.base_export_revenue for line in view.lines) == Decimal(amount)
            assert view.net_cost == Decimal("0")
            assert view.complete is True
            basis = dst.repo.read_reconciliation(METER, start, end)
            assert sum(source.item.import_kwh for source in basis.sources.lines) == Decimal(kwh)
            assert sum(source.item.export_kwh for source in basis.sources.lines) == Decimal(kwh)
            assert not any(source.issues for source in basis.sources.lines)
        # The repeated local hour keeps two distinct UTC instants.
        autumn = dst.repo.read_reconciliation(
            METER, int(AUTUMN_DAY[0].timestamp()), int(AUTUMN_DAY[1].timestamp()))
        hours = sorted({source.start_ts // HOUR * HOUR for source in autumn.sources.lines})
        assert len(hours) == 25
        assert hours[2] - hours[1] == HOUR
    finally:
        dst.engine.dispose()

    # --- an explicit recorder gap stays a gap, never a zero ------------------
    gap = build_chain(tmp_path / "gap", make_recorder_db(tmp_path / "gap"),
                      make_price_db(tmp_path / "gap"), monkeypatch,
                      pricing_config(start="2027-10-31"),
                      clock=clock_after(int(AUTUMN_DAY[1].timestamp())))
    try:
        start, end = int(AUTUMN_DAY[0].timestamp()), int(AUTUMN_DAY[1].timestamp())
        put_grid(gap.recorder, start, end)
        put_prices(gap.price_db, start, end, 0.1)
        drop_bucket(gap.recorder, "sensor.import", start + 900)
        view = gap.service.consolidate(start, end, publish=True, reason="e1f-gap")
        assert view.complete is False
        assert view.net_cost is None
        missing = [line for line in view.lines if line.import_cost is None]
        assert len(missing) == 1 and missing[0].quality == "incomplete"
        assert "missing_import" in missing[0].issues
        basis = gap.repo.read_reconciliation(METER, start, end)
        assert sum(source.item.import_kwh or Decimal("0")
                   for source in basis.sources.lines) == Decimal("29.7")
        assert any(source.item.import_kwh is None for source in basis.sources.lines)
    finally:
        gap.engine.dispose()

    # --- the monthly floor credit closes the month exactly once --------------
    monthly = build_chain(tmp_path / "month", make_recorder_db(tmp_path / "month"),
                          make_price_db(tmp_path / "month"), monkeypatch,
                          bonus_pricing(floor="spot_zero"), clock=clock_after(MONTH_END_TS))
    try:
        put_prices(monthly.price_db, START_TS, MONTH_END_TS, -0.1)
        # Zero import over the whole month and 1 kWh export in each of the two
        # quarters that end at the month boundary: -0.20 EUR monthly proceeds.
        put_steps(monthly.recorder, "sensor.import", START_TS, MONTH_END_TS, {})
        put_steps(monthly.recorder, "sensor.export", START_TS, MONTH_END_TS,
                  {MONTH_END_TS - 2 * QUARTER: "1", MONTH_END_TS - QUARTER: "1"})
        view = monthly.service.consolidate(START_TS, MONTH_END_TS, publish=True,
                                           reason="e1f-month")
        assert len(view.adjustments) == 1
        credit = view.adjustments[0]
        assert credit.kind == "spot_zero"
        assert credit.period_start_ts == START_TS and credit.period_end_ts == MONTH_END_TS
        assert credit.credit_ex_vat == Decimal("0.20")
        assert credit.credit_vat == Decimal("0") and credit.credit_total == Decimal("0.20")
        assert credit.basis_ex_vat == Decimal("-0.20")
        assert credit.quality == "estimated"
        assert view.net_cost == Decimal("0")
        basis = monthly.repo.read_reconciliation(METER, START_TS, MONTH_END_TS)
        assert len(basis.sources.credits) == 1
        assert basis.sources.credits[0].source_key
        assert not basis.sources.credits[0].issues
    finally:
        monthly.engine.dispose()

    # --- the bonus cap: 4999 kWh leaves one kWh, 5000 kWh leaves none --------
    cap = build_chain(tmp_path / "cap", make_recorder_db(tmp_path / "cap"),
                      make_price_db(tmp_path / "cap"), monkeypatch, bonus_pricing(),
                      clock=clock_after(MONTH_END_TS))
    try:
        start = MONTH_END_TS - 2 * QUARTER
        put_prices(cap.price_db, start, MONTH_END_TS, 0.1)
        opening = SettlementState(
            start, BonusState(START_TS, int(datetime(2028, 1, 1, tzinfo=ZONE).timestamp()),
                              start, Decimal("0")), None)
        rows = [(start, start + QUARTER, "0", "4999", "measured", "verified_pv"),
                (start + QUARTER, MONTH_END_TS, "0", "2", "measured", "verified_pv")]
        path = write_meter_csv(tmp_path / "cap" / "cap.csv", rows)
        readings = CsvMetering(path)(start, MONTH_END_TS)
        register_chain(cap, readings, opening, start=start, end=MONTH_END_TS, generation=0)
        view = cap.repo.read(METER, start, MONTH_END_TS)
        assert [line.bonus_kwh for line in view.lines] == [Decimal("4999"), Decimal("1")]
        assert view.lines[-1].bonus_revenue == Decimal("0.03")

        # The corrected first quarter exhausts the cap before the second one.
        rows = [(start, start + QUARTER, "0", "5000", "measured", "verified_pv"),
                (start + QUARTER, MONTH_END_TS, "0", "2", "measured", "verified_pv")]
        path = write_meter_csv(tmp_path / "cap" / "cap.csv", rows)
        readings = CsvMetering(path)(start, MONTH_END_TS)
        register_chain(cap, readings, opening, start=start, end=MONTH_END_TS, generation=1)
        corrected = cap.repo.read(METER, start, MONTH_END_TS)
        assert [line.bonus_kwh for line in corrected.lines] == [Decimal("5000"), Decimal("0")]
        assert corrected.lines[-1].bonus_revenue == Decimal("0")
        # The first publication stays readable as the historical generation.
        assert cap.repo.read(METER, start, MONTH_END_TS, 1).lines[0].bonus_kwh == Decimal("4999")
    finally:
        cap.engine.dispose()

    # --- the delivery year resets on the supply-start anniversary -------------
    year = build_chain(tmp_path / "year", make_recorder_db(tmp_path / "year"),
                       make_price_db(tmp_path / "year"), monkeypatch, bonus_pricing(),
                       clock=clock_after(int(datetime(2028, 1, 1, tzinfo=ZONE).timestamp())
                                         + QUARTER))
    try:
        boundary = int(datetime(2028, 1, 1, tzinfo=ZONE).timestamp())
        start = boundary - 4 * QUARTER
        put_prices(year.price_db, start, boundary + QUARTER, 0.1)
        opening = SettlementState(start, BonusState(START_TS, boundary, start, Decimal("0")),
                                  None)
        rows = [(start, start + QUARTER, "0", "5000", "measured", "verified_pv"),
                (start + QUARTER, start + 2 * QUARTER, "0", "1", "measured", "verified_pv"),
                (start + 2 * QUARTER, start + 3 * QUARTER, "0", "1", "measured", "verified_pv"),
                (start + 3 * QUARTER, boundary, "0", "1", "measured", "verified_pv"),
                (boundary, boundary + QUARTER, "0", "1", "measured", "verified_pv")]
        path = write_meter_csv(tmp_path / "year" / "year.csv", rows)
        readings = CsvMetering(path)(start, boundary + QUARTER)
        register_chain(year, readings, opening, start=start, end=boundary + QUARTER,
                       generation=0)
        view = year.repo.read(METER, start, boundary + QUARTER)
        # The cap is exhausted inside the old delivery year and starts over in
        # the new one, at the supply-start anniversary of 1 January.
        assert [line.bonus_kwh for line in view.lines] == [
            Decimal("5000"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("1")]
        assert view.lines[-1].bonus_revenue == Decimal("0.03")
        basis = year.repo.read_reconciliation(METER, start, boundary + QUARTER)
        assert not any(source.issues for source in basis.sources.lines)
    finally:
        year.engine.dispose()
