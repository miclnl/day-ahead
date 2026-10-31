"""Raw prices share the book CAS, never the closed-meter watermark or money rows."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import datetime
from decimal import Decimal, localcontext
from hashlib import sha256
from threading import Barrier, Event
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import event, text

from dao.prog.settlement.repository import ConcurrentSettlementError, SettlementRepository
from dao.prog.settlement.types import SpotInterval
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_repository import (
    METER,
    publish,
    rows,
    table,
)
from dao.tests.settlement.test_repository import (
    calculate as calculate,
)
from dao.tests.settlement.test_repository import (
    inputs as inputs,
)
from dao.tests.settlement.test_repository import (
    repo as repo,
)
from dao.tests.settlement.test_repository import (
    terms as terms,
)


def price(at=START_TS, value="0.1", *, step=3600, source="synthetic-hour-price", quality="measured"):
    return SpotInterval(at, at + step, value, source, quality)


@pytest.fixture(autouse=True)
def closed_clock(monkeypatch):
    monkeypatch.setattr("dao.prog.settlement.repository.time", lambda: START_TS + 86400)


def frozen(repo):
    return {name: rows(repo, name) for name in ("book", "spot", "input", "run", "interval", "adjustment")}


def test_future_prices_are_immutable_noop_and_aba_without_meter_or_money_progress(repo, terms, inputs, calculate):
    snapshots = [terms()]
    publish(repo, inputs(snapshots, imports=("1",)), snapshots, calculate)
    original = frozen(repo)
    future = price(START_TS + 3600, "-0.20")
    assert repo.register_prices(METER, [future], 1) == 2
    book = rows(repo, "book")[0]
    assert book["generation"] == 1 and book["dirty_from_ts"] is None
    assert book["source_watermark_ts"] == START_TS + 900
    assert {name: rows(repo, name) for name in ("input", "run", "interval", "adjustment")} == {
        name: original[name] for name in ("input", "run", "interval", "adjustment")}
    batch = repo.read_prices(METER, START_TS + 4500, START_TS + 5400, expected_input_revision=2)
    assert (batch.start_ts, batch.end_ts, batch.source_watermark_ts) == (
        START_TS + 4500, START_TS + 5400, START_TS + 900)
    assert batch.snapshots == tuple(snapshots) and batch.items[0].spot == future
    assert batch.items[0].input_revision == 2
    with localcontext() as context:
        context.prec = 6
        assert repo.register_prices(METER, [replace(future, price_ex_vat="-0.200000000000")], 2) == 2
    assert repo.register_prices(METER, [replace(future, price_ex_vat="0.3")], 2) == 3
    assert repo.register_prices(METER, [future], 3) == 4
    selected = repo.read_prices(METER, START_TS + 3600, START_TS + 7200)
    assert selected.items[0].input_revision == 4
    assert [row["input_revision"] for row in rows(repo, "spot")] == [2, 3, 4]
    canonical = json.dumps({"start_ts": future.start_ts, "end_ts": future.end_ts,
                            "price_ex_vat": "-0.2", "source": future.source, "quality": "measured"},
                           sort_keys=True, separators=(",", ":"))
    assert selected.items[0].content_hash == sha256(canonical.encode()).hexdigest()
    assert rows(repo, "spot")[0]["price_ex_vat"] == "-0.2"
    assert repo.read(METER, START_TS, START_TS + 900).input_revision == 4
    with pytest.raises(ConcurrentSettlementError):
        repo.read_prices(METER, START_TS, START_TS + 7200, expected_input_revision=2)


def test_empty_read_is_write_free_and_revision_zero(repo):
    batch = repo.read_prices(METER, START_TS + 900, START_TS + 1800, expected_input_revision=0)
    assert batch.items == batch.snapshots == () and batch.input_revision == 0
    assert batch.source_watermark_ts is None and rows(repo, "book") == []
    with pytest.raises(ConcurrentSettlementError):
        repo.read_inputs(METER, START_TS, START_TS + 900, expected_input_revision=1)


def test_both_resolutions_full_hour_envelope_withdrawal_and_omission(repo):
    hour = price(value="0.9")
    quarters = [price(START_TS + i * 900, str(i), step=900) for i in range(4)]
    assert repo.register_prices(METER, [hour, *quarters], 0) == 1
    batch = repo.read_prices(METER, START_TS + 900, START_TS + 1800)
    assert [(row.spot.end_ts - row.spot.start_ts, row.spot.start_ts) for row in batch.items] == [
        (900, START_TS + i * 900) for i in range(4)] + [(3600, START_TS)]
    assert repo.register_prices(METER, [price(value=None, quality="incomplete")], 1) == 2
    latest = repo.read_prices(METER, START_TS, START_TS + 3600)
    assert len(latest.items) == 5 and latest.items[-1].spot.price_ex_vat is None
    assert [row.input_revision for row in latest.items] == [1, 1, 1, 1, 2]
    assert repo.register_prices("other-grid", [hour], 0) == 1
    assert repo.read_prices(METER, START_TS, START_TS + 3600).input_revision == 2


@pytest.mark.parametrize("case", ["duration", "alignment", "source", "duplicate", "conflict", "revision"])
def test_invalid_price_registration_is_atomic(repo, case):
    supplied, revision = [price()], 0
    if case == "duration":
        supplied = [replace(price(), end_ts=START_TS + 1800)]
    elif case == "alignment":
        supplied = [price(START_TS + 1)]
    elif case == "source":
        supplied = [price(source=" ")]
    elif case in ("duplicate", "conflict"):
        supplied.append(price(value="0.2" if case == "conflict" else "0.1"))
    else:
        revision = True
    with pytest.raises((ValueError, TypeError)):
        repo.register_prices(METER, supplied, revision)
    assert rows(repo, "spot") == [] and rows(repo, "input") == []


def test_price_dtos_reject_false_revisions_and_keep_nested_data_immutable():
    from dao.prog.settlement.types import PriceBatch, RegisteredSpot, ResolvedSpot

    registered = RegisteredSpot(price(), 1, "a" * 64)
    batch = PriceBatch(METER, 2, START_TS, START_TS + 900, [registered], [], START_TS, None)
    assert batch.items == (registered,) and batch.snapshots == ()
    with pytest.raises(FrozenInstanceError):
        batch.input_revision = 3
    for revision in (0, -1, True):
        with pytest.raises((TypeError, ValueError)):
            RegisteredSpot(price(), revision, "a" * 64)
    with pytest.raises(ValueError):
        replace(batch, input_revision=0)
    for value in ("NaN", "Infinity", "0.0000000000001"):
        with pytest.raises(ValueError):
            ResolvedSpot(value, "source", "measured")


@pytest.mark.parametrize("month,day,next_month,next_day,hours", [(3, 28, 3, 29, 23), (10, 31, 11, 1, 25)])
def test_price_reads_preserve_real_dst_hours(repo, month, day, next_month, next_day, hours):
    zone = ZoneInfo("Europe/Amsterdam")
    start = int(datetime(2027, month, day, tzinfo=zone).timestamp())
    end = int(datetime(2027, next_month, next_day, tzinfo=zone).timestamp())
    supplied = [price(at, str(i)) for i, at in enumerate(range(start, end, 3600))]
    repo.register_prices(METER, supplied, 0)
    selected = repo.read_prices(METER, start, end)
    assert len(selected.items) == hours and tuple(row.spot for row in selected.items) == tuple(supplied)


@pytest.mark.parametrize("reader", ["read_prices", "read_inputs"])
@pytest.mark.parametrize("change", ["price", "contract"])
def test_read_pins_one_book_reference_even_when_writer_interleaves(repo, terms, inputs, reader, change):
    snap = terms()
    repo.register_inputs(METER, inputs([snap]), [snap], 0)
    repo.register_prices(METER, [price()], 1)
    if repo.engine.dialect.name == "sqlite":
        with repo.engine.connect() as connection:
            connection.execute(text("PRAGMA journal_mode=WAL"))
    barrier = Barrier(2)
    fired = False

    def pause(connection, cursor, statement, parameters, context, executemany):
        nonlocal fired
        if not fired and statement.startswith("SELECT settlement_book.meter_id"):
            fired = True
            barrier.wait(timeout=10)
            barrier.wait(timeout=10)

    def writer():
        barrier.wait(timeout=10)
        try:
            other = SettlementRepository(repo.engine)
            if change == "price":
                return other.register_prices(METER, [price(value="0.2")], 2)
            return other.register_inputs(METER, (), [terms(contract_id="other-contract")], 2)
        finally:
            barrier.wait(timeout=10)

    event.listen(repo.engine, "after_cursor_execute", pause)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(writer)
            batch = getattr(repo, reader)(METER, START_TS, START_TS + 1800, expected_input_revision=2)
            assert future.result(timeout=10) == 3
    finally:
        event.remove(repo.engine, "after_cursor_execute", pause)
    assert fired and batch.input_revision == 2 and batch.snapshots == (snap,)
    if reader == "read_prices":
        assert batch.items[0].spot.price_ex_vat == Decimal("0.1")
    else:
        assert batch.items[0].spot_ex_vat == Decimal("0.1")
    with pytest.raises(ConcurrentSettlementError):
        getattr(repo, reader)(METER, START_TS, START_TS + 1800, expected_input_revision=2)


@pytest.mark.parametrize("first", ["price", "commit"])
def test_two_connections_price_commit_order_never_publishes_money_after_price_correction(
    repo, terms, inputs, calculate, first,
):
    snap = terms()
    repo.register_inputs(METER, inputs([snap]), [snap], 0)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    result = calculate(batch)
    barrier, done = Barrier(2), Event()

    def writer(role):
        other = SettlementRepository(repo.engine)
        barrier.wait(timeout=10)
        if role != first:
            assert done.wait(timeout=10)
        try:
            if role == "price":
                return other.register_prices(METER, [price(value="0.2")], 1)
            try:
                return other.commit(METER, result, batch, 0, "ordered-price-commit")
            except ConcurrentSettlementError:
                return "conflict"
        finally:
            if role == first:
                done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(writer, ("price", "commit")))
    assert outcomes == [2, "conflict" if first == "price" else 1]
    assert rows(repo, "book")[0]["dirty_from_ts"] == START_TS
    assert len(rows(repo, "run")) == (0 if first == "price" else 1)


def test_price_and_closed_writers_share_one_cas_winner(repo, terms, inputs):
    snap = terms()
    repo.register_inputs(METER, (), [snap], 0)
    batch = repo.read_prices(METER, START_TS, START_TS + 900)
    unknown = tuple(replace(item, spot_source=None) for item in inputs([snap], imports=("1",), spot=None))
    barrier = Barrier(2)

    def writer(kind):
        barrier.wait(timeout=10)
        other = SettlementRepository(repo.engine)
        try:
            if kind == "price":
                return other.register_prices(METER, [price()], 1)
            return other.register_inputs(METER, unknown, [snap], 1, price_batch=batch,
                                         closed_before_ts=START_TS + 900)
        except ConcurrentSettlementError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(writer, ("price", "closed")))
    assert sorted(outcomes, key=str) == [2, "conflict"]
    assert repo.read_book_reference(METER).input_revision == 2


def test_price_correction_between_context_reads_and_commit_rejects_old_batch(repo, terms, inputs, calculate):
    snap = terms()
    _, batch, result = publish(repo, inputs([snap]), [snap], calculate)
    repo.register_prices(METER, [price(value="0.2")], 1)
    before = frozen(repo)
    with pytest.raises(ConcurrentSettlementError):
        repo.read_prices(METER, START_TS, START_TS + 1800, expected_input_revision=batch.input_revision)
    with pytest.raises(ConcurrentSettlementError):
        repo.commit(METER, result, batch, 1, "stale-price-context")
    assert frozen(repo) == before


def test_seventh_table_added_to_populated_six_without_rewriting_history(repo, terms, inputs, calculate):
    publish(repo, inputs([terms()]), [terms()], calculate)
    # Simuleer de bestaande zes-tabellen-installatie, uitsluitend in deze tijdelijke DB.
    before = {name: rows(repo, name) for name in ("book", "input", "run", "interval", "adjustment", "tariff")}
    table(repo, "spot").drop(repo.engine)
    repo.ensure_schema()
    repo.ensure_schema()
    assert {name: rows(repo, name) for name in before} == before and rows(repo, "spot") == []


def test_raw_price_comparison_is_bulk_not_one_query_per_closed_quarter(repo, terms, inputs):
    from contextlib import nullcontext

    changes, reads = [], []

    class Guard:
        def hold(self):
            return nullcontext()

        def before_revision(self, change):
            changes.append(change)

    snap = terms(hour_price_rule="mean_quarters")
    repo.register_inputs(METER, inputs([snap], imports=("1",) * 96, spot=None), [snap], 0)

    def observe(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("SELECT"):
            reads.append(statement)

    event.listen(repo.engine, "before_cursor_execute", observe)
    try:
        SettlementRepository(repo.engine, publication=Guard()).register_prices(
            METER, [price(START_TS + i * 900, step=900) for i in range(96)], 1)
    finally:
        event.remove(repo.engine, "before_cursor_execute", observe)
    assert len(reads) <= 6
    assert (START_TS, START_TS + 3600) in changes[0].changed_prices
    assert changes[0].changed_intervals == changes[0].changed_contracts == ()
    assert rows(repo, "book")[0]["dirty_from_ts"] == START_TS


def test_changed_prices_include_full_selected_window_when_snapshot_starts_inside_it(repo, terms):
    from contextlib import nullcontext

    changes = []

    class Guard:
        def hold(self):
            return nullcontext()

        def before_revision(self, change):
            changes.append(change)

    snap = terms(hour_price_rule="mean_quarters", valid_from_ts=START_TS + 900)
    repo.register_inputs(METER, (), [snap], 0)
    writer = SettlementRepository(repo.engine, publication=Guard())
    writer.register_prices(METER, [price(step=900)], 1)
    assert (START_TS, START_TS + 3600) in changes[-1].changed_prices
