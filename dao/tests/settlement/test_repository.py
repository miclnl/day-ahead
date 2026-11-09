"""Real transactions must preserve revisions, exact money and pinned generations."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from hashlib import sha256
from threading import Barrier

import pytest
from sqlalchemy import MetaData, Table, create_engine, event, func, select, text
from sqlalchemy.pool import NullPool

from dao.prog.settlement.engine import settle_series
from dao.prog.settlement.metering import merge_inputs
from dao.prog.settlement.tariffs import _canonical_default
from dao.prog.settlement.types import SpotInterval
from dao.tests.settlement.conftest import MONTH_END_TS, START_TS

METER = "synthetic-grid"


def e1c_read(repo, start=START_TS, end=START_TS + 1800, generation=None):
    return repo.read_reconciliation(METER, start, end, generation=generation)


def e1c_credit_history(repo, terms, inputs, opening, *, bonus_rules=None, meter=None):
    """Real publications retaining an unchanged adjustment across two runs."""
    from datetime import date

    start = MONTH_END_TS - 1800
    options = dict(rule_version="tariffs-v1", monthly_export_floor="spot_zero")
    if bonus_rules is not None:
        options.update(bonus=replace(bonus_rules, windows=tuple((m, (0, 24)) for m in range(1, 13))),
                       supply_start=date(2027, 1, 1), price_interval_seconds=900)
    first = terms(**options)

    def compute(batch):
        initial = opening(asof_ts=start)
        initial = replace(initial, month=replace(initial.month, contract_id=batch.snapshots[0].contract_id),
                          bonus=None if bonus_rules is None else replace(initial.bonus, asof_ts=start))
        return settle_series(batch.items, batch.snapshots, initial)

    exports = ("1", "1") if bonus_rules is None else ("4999", "2")
    spot = "-0.10" if bonus_rules is None else "0.10"
    def items(snapshot, exports, first_spot):
        if bonus_rules is None:
            return inputs([snapshot], start=start, imports=("0", "0"), exports=exports, spot=spot)
        readings = [meter(start_ts=start + i * 900, end_ts=start + (i + 1) * 900,
                          import_kwh=Decimal("0"), export_kwh=Decimal(export))
                    for i, export in enumerate(exports)]
        prices = [SpotInterval(start + i * 900, start + (i + 1) * 900, Decimal(price),
                               "synthetic-quarter-price", "measured")
                  for i, price in enumerate((first_spot, "0.10"))]
        return merge_inputs(readings, prices, [snapshot], meter_id=METER)

    publish(repo, items(first, exports, spot), [first], compute)
    old = e1c_read(repo, start, MONTH_END_TS)
    second = terms(**options, contract_id="new-contract") if bonus_rules is None else first
    exports = exports if bonus_rules is None else ("5000", "2")
    publish(repo, items(second, exports, spot if bonus_rules is None else "0.09998"),
            [second], compute, revision=1, generation=1)
    return old, e1c_read(repo, start, MONTH_END_TS)


@pytest.mark.parametrize("change", ["meter", "price_snapshot"])
def test_e1c_old_generation_uses_exact_inputs_and_snapshot(repo, terms, inputs, calculate, monkeypatch, change):
    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots), snapshots, calculate)
    original = e1c_read(repo)
    new = snapshots if change == "meter" else [terms(rule_version="tariffs-v1", source_versions=(("new", "2"),))]
    publish(repo, inputs(new, imports=("1", "2") if change == "meter" else ("1", "1"),
                         spot="0.10" if change == "meter" else "0.20"), new, calculate, revision=1, generation=1)
    def forbidden(*args, **kwargs):
        pytest.fail("current input/price lookup on historical read")
    monkeypatch.setattr(repo, "read_inputs", forbidden)
    monkeypatch.setattr(repo, "read_prices", forbidden)
    historical, current = e1c_read(repo, generation=1), e1c_read(repo)
    assert historical.view.net_cost == Decimal("0.20")
    assert historical.view.freshness == "outdated"
    assert historical.sources.lines == original.sources.lines
    assert sum(s.item.import_kwh for s in historical.sources.lines) == 2
    assert all(s.item.spot_ex_vat == Decimal("0.10") and s.snapshot == snapshots[0]
               for s in historical.sources.lines)
    assert current.view.net_cost == Decimal("0.30" if change == "meter" else "0.40")
    assert current.view.freshness == "current"
    assert historical.sources.pin.view_digest != current.sources.pin.view_digest


def test_e1c_mixed_runs_and_noop_revision(repo, terms, inputs, calculate):
    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots), snapshots, calculate)
    publish(repo, inputs(snapshots, imports=("1", "2")), snapshots, calculate, revision=1, generation=1)
    basis = e1c_read(repo)
    assert [s.valid_from_generation for s in basis.sources.lines] == [1, 2]
    assert [s.input_revision for s in basis.sources.lines] == [1, 2]
    assert {r.generation for r in basis.sources.runs} == {1, 2}
    repo.register_prices(METER, [SpotInterval(START_TS + 3600, START_TS + 7200, Decimal("0.3"),
                                            "future", "measured")], 2)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    assert repo.commit(METER, calculate(batch), batch, 2, "noop") == 2
    new = e1c_read(repo)
    assert new.view.input_revision == 3 and new.view.freshness == "current"
    assert new.sources.lines == basis.sources.lines
    assert new.sources.runs == basis.sources.runs
    assert all(s.input_revision <= next(r.input_revision for r in new.sources.runs if r.run_id == s.run_id)
               for s in new.sources.lines)


def test_e1c_credit_uses_producer_closing_line(repo, terms, inputs, opening):
    from dao.prog.settlement.reconciliation import reconcile
    from dao.tests.settlement.test_reconciliation import e1b_rounding, supplier

    old, new = e1c_credit_history(repo, terms, inputs, opening)
    assert old.view.net_cost == 0
    assert sum(line.base_export_revenue for line in old.view.lines) == Decimal("-0.20")
    assert old.view.adjustments[0].credit_total == Decimal("0.20")
    assert old.view.adjustments[0].quality == "estimated"
    credit = new.sources.credits[0]
    assert credit == old.sources.credits[0]
    assert credit.contract_id == "synthetic-2027"
    assert credit.closing_line == old.view.lines[-1]
    assert credit.closing_source == old.sources.lines[-1]
    assert new.sources.lines[-1].snapshot.contract_id == "new-contract"
    assert credit.closing_source.source_key != new.sources.lines[-1].source_key
    assert not credit.issues
    assert {r.generation for r in new.sources.runs} == {1, 2}
    report = reconcile(new.view, supplier(meter_id=METER, contract_id="new-contract",
                                         start_ts=MONTH_END_TS - 1800, end_ts=MONTH_END_TS,
                                         rounding=e1b_rounding()), sources=new.sources)
    credit_components = [c for c in report.components if c.key.startswith("credit:")]
    assert credit_components and all(c.status == "not_comparable" and c.delta is None for c in credit_components)
    assert any("contract_mismatch" in c.reasons for c in credit_components)


def test_e1c_same_input_different_state_source_keys(repo, terms, inputs, opening, bonus_rules, meter):
    from dao.prog.settlement.reconciliation import _canonical_json, reconcile
    from dao.tests.settlement.test_reconciliation import e1b_rounding, supplier

    old, new = e1c_credit_history(repo, terms, inputs, opening, bonus_rules=bonus_rules, meter=meter)
    a, b = old.sources.lines[-1], new.sources.lines[-1]
    assert old.view.lines[-1].bonus_revenue == Decimal("0.03")
    assert new.view.lines[-1].bonus_revenue == 0
    assert old.view.adjustments == new.view.adjustments
    assert old.view.adjustments[0].basis_ex_vat == Decimal("500.1")
    assert old.view.lines[0].base_export_revenue == new.view.lines[0].base_export_revenue == Decimal("499.9")
    assert (a.input_hash, a.input_revision, a.snapshot_hash) == (b.input_hash, b.input_revision, b.snapshot_hash)
    assert a.source_key != b.source_key and a.line_content_hash != b.line_content_hash
    assert a.valid_from_generation == 1 and b.valid_from_generation == 2
    assert old.sources.credits[0].closing_source == a
    assert new.sources.credits[0].closing_source == a
    assert json.loads(a.source_key) == ["settlement-line-v1", METER, a.start_ts, 1, a.run_id, a.line_content_hash]
    # Same primary version in both roles cannot become two alternate contents.
    conflicting = replace(a, line_content_hash="conflicting-content", source_key=_canonical_json(
        ["settlement-line-v1", METER, a.start_ts, 1, a.run_id, "conflicting-content"]))
    sources = replace(old.sources, credits=(replace(old.sources.credits[0], closing_source=conflicting),))
    report = reconcile(old.view, supplier(meter_id=METER, contract_id="synthetic-2027",
                                         start_ts=MONTH_END_TS - 1800, end_ts=MONTH_END_TS,
                                         rounding=e1b_rounding()), sources=sources)
    assert any("source_mismatch" in component.reasons for component in report.components)
    assert report.total.delta is None


def e1c_corrupt(repo, callback, *, break_foreign_keys=False):
    """Deliberate fixture corruption only; never weaken production FK handling."""
    with repo.engine.connect() as connection:
        dialect = repo.engine.dialect.name
        commands = {
            "sqlite": ("PRAGMA foreign_keys=OFF", "PRAGMA foreign_keys=ON"),
            "mysql": ("SET SESSION FOREIGN_KEY_CHECKS=0", "SET SESSION FOREIGN_KEY_CHECKS=1"),
            "mariadb": ("SET SESSION FOREIGN_KEY_CHECKS=0", "SET SESSION FOREIGN_KEY_CHECKS=1"),
            "postgresql": ("SET session_replication_role=replica", "SET session_replication_role=origin"),
        }
        # Dedicated synthetic server cells need permission for this explicit corruption.
        # Setup failure is a failure, not a skip or an unverified fallback.
        if break_foreign_keys:
            connection.exec_driver_sql(commands[dialect][0])
        try:
            callback(connection)
            connection.commit()
        finally:
            connection.rollback()
            if break_foreign_keys:
                connection.exec_driver_sql(commands[dialect][1])
                connection.commit()


def test_e1c_credit_scope_tombstone_and_unknown(repo, terms, inputs, opening):
    from dao.prog.settlement import repository as r

    empty = e1c_read(repo)
    assert empty.view.freshness == "unknown" and empty.view.net_cost is None
    assert empty.sources.lines == empty.sources.credits == empty.sources.runs == ()
    old, new = e1c_credit_history(repo, terms, inputs, opening)
    left = e1c_read(repo, MONTH_END_TS - 1800, MONTH_END_TS - 900)
    right = e1c_read(repo, MONTH_END_TS - 900, MONTH_END_TS)
    after = e1c_read(repo, MONTH_END_TS, MONTH_END_TS + 900)
    assert left.sources.credits == after.sources.credits == ()
    assert right.sources.credits == new.sources.credits and len(right.sources.credits) == 1
    e1c_corrupt(repo, lambda c: c.execute(r._ADJUSTMENT.update().values(tombstone=True, adjustment_json=None)))
    assert e1c_read(repo, MONTH_END_TS - 1800, MONTH_END_TS).sources.credits == ()


@pytest.mark.parametrize("case", ["input_missing", "snapshot_missing", "input_json", "input_hash",
                                  "input_column", "input_meter", "input_end", "input_revision",
                                  "snapshot_json", "snapshot_hash", "snapshot_coverage", "line_hash",
                                  "line_column", "run_missing", "run_meter", "run_revision", "run_membership"])
def test_e1c_missing_vs_mismatched_enrichment(repo, terms, inputs, calculate, case):
    from dao.prog.settlement import repository as r

    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots), snapshots, calculate)
    publish(repo, inputs(snapshots, imports=("1", "2")), snapshots, calculate, revision=1, generation=1)
    before = repo.read(METER, START_TS, START_TS + 1800)
    clean = e1c_read(repo)
    assert all(not s.issues and s.item is not None for s in clean.sources.lines)
    source = rows(repo, "input")[0]
    def corrupt(c):
        target = r._INPUT.c.start_ts == START_TS
        if case == "input_missing":
            c.execute(r._INPUT.delete().where(target))
        elif case == "snapshot_missing":
            c.execute(r._TARIFF.delete())
        elif case.startswith("input_"):
            data = json.loads(source["input_json"])
            fields = {"input_meter": ("meter_id", "wrong"), "input_end": ("end_ts", START_TS + 901),
                      "input_revision": ("input_revision", 9)}
            if case in fields:
                key, value = fields[case]
                data[key] = value
            values = {"input_json": "{" if case == "input_json" else json.dumps(data)}
            if case == "input_hash":
                values = {"content_hash": "bad"}
            if case == "input_column":
                values = {"input_hash": "bad"}
            c.execute(r._INPUT.update().where(target).values(**values))
        elif case.startswith("snapshot_"):
            data = asdict(snapshots[0])
            if case == "snapshot_coverage":
                data["valid_from_ts"] = START_TS + 900
                del data["snapshot_hash"]
                digest = r._hash(r._json(data))
                data["snapshot_hash"] = digest
                c.execute(r._TARIFF.insert().values(snapshot_hash=digest, snapshot_json=r._json(data)))
                for row in c.execute(select(r._INTERVAL)).mappings().all():
                    line = replace(r._load(r.SettlementLine, row["line_json"]), snapshot_hash=digest)
                    c.execute(r._INTERVAL.update().where(
                        r._INTERVAL.c.start_ts == row["start_ts"],
                        r._INTERVAL.c.valid_from_generation == row["valid_from_generation"],
                    ).values(snapshot_hash=digest, line_json=r._json(line), content_hash=r._hash(r._json(line))))
                c.execute(r._RUN.update().values(snapshot_hashes_json=r._json([digest])))
            else:
                c.execute(r._TARIFF.update().values(snapshot_json="{" if case == "snapshot_json" else
                                                   r._json({**data, "snapshot_hash": "bad"})))
        elif case.startswith("line_"):
            c.execute(r._INTERVAL.update().where(r._INTERVAL.c.start_ts == START_TS).values(
                **({"content_hash": "bad"} if case == "line_hash" else {"import_cost": "999"})))
        elif case == "run_missing":
            c.execute(r._RUN.delete().where(r._RUN.c.generation == 1))
        else:
            values = {"run_meter": {"meter_id": "wrong"}, "run_revision": {"input_revision": 1},
                      "run_membership": {"snapshot_hashes_json": "[]"}}[case]
            c.execute(r._RUN.update().where(r._RUN.c.generation == (2 if case == "run_revision" else 1)).values(**values))
    e1c_corrupt(repo, corrupt, break_foreign_keys=case in ("input_missing", "snapshot_missing", "run_missing"))
    basis = e1c_read(repo)
    if case == "snapshot_coverage":
        assert basis.view == repo.read(METER, START_TS, START_TS + 1800)
        assert basis.view.net_cost == before.net_cost
        assert not basis.sources.lines[1].issues
    else:
        assert basis.view == before
    affected = basis.sources.lines[-1] if case == "run_revision" else basis.sources.lines[0]
    expected = "source_missing" if case in ("input_missing", "snapshot_missing", "run_missing") else "source_mismatch"
    assert affected.issues == (expected,)
    if case.startswith("input_"):
        assert affected.item is None
    if case.startswith("snapshot_"):
        assert affected.snapshot is None


@pytest.mark.parametrize("case", ["line_json", "line_nonfinite", "line_invariant", "line_shape", "line_float",
                                  "adjustment_json", "adjustment_invariant", "selected_run_missing",
                                  "selected_run_json", "header_json", "historical_line", "incomplete"])
def test_e1c_unreadable_view_vs_historical_enrichment(repo, terms, inputs, opening, case):
    from dao.prog.settlement import repository as r

    old, new = e1c_credit_history(repo, terms, inputs, opening)
    if case == "incomplete":
        basis = e1c_read(repo, MONTH_END_TS - 2700, MONTH_END_TS)
        assert not basis.view.complete and basis.view.net_cost is None
        return
    def corrupt(c):
        if case == "selected_run_missing":
            c.execute(r._RUN.delete().where(r._RUN.c.generation == 2))
        elif case == "selected_run_json":
            c.execute(r._RUN.update().where(r._RUN.c.generation == 2).values(snapshot_hashes_json="{}"))
        elif case == "header_json":
            c.execute(r._BOOK.update().values(snapshot_hashes_json="{"))
        elif case.startswith("adjustment"):
            data = asdict(new.view.adjustments[0])
            data["period_end_ts"] = data["period_start_ts"]
            c.execute(r._ADJUSTMENT.update().values(adjustment_json="{" if case == "adjustment_json" else r._json(data)))
        else:
            data = asdict(new.view.lines[-1])
            if case == "line_nonfinite":
                data["import_cost"] = "NaN"
            if case == "line_invariant":
                data["end_ts"] = data["start_ts"]
            if case == "line_float":
                data["import_cost"] = 1.0
            if case == "line_shape":
                data["after_state"] = []
            c.execute(r._INTERVAL.update().where(r._INTERVAL.c.start_ts == MONTH_END_TS - 900,
                                                r._INTERVAL.c.valid_from_generation == (1 if case == "historical_line" else 2))
                      .values(line_json="{" if case in ("line_json", "historical_line") else r._json(data)))
    e1c_corrupt(repo, corrupt, break_foreign_keys=case == "selected_run_missing")
    if case == "historical_line":
        basis = e1c_read(repo, MONTH_END_TS - 1800, MONTH_END_TS)
        assert basis.view == new.view
        assert basis.sources.credits[0].closing_line is None
        assert basis.sources.credits[0].closing_source is None
        assert "credit_identity_unproven" in basis.sources.credits[0].issues
    else:
        with pytest.raises(r.ReconciliationReadError, match="^book_view_unreadable$"):
            e1c_read(repo, MONTH_END_TS - 1800, MONTH_END_TS)


@pytest.mark.parametrize("field", ["dirty_from_ts", "source_watermark_ts"])
@pytest.mark.parametrize("epoch", [-62135596801, 253402300800, -62135596800, 253402300799, None])
def test_e1c_header_epoch_decode_boundary(repo, terms, inputs, calculate, field, epoch):
    from dao.prog.settlement import repository as r

    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots), snapshots, calculate)
    e1c_corrupt(repo, lambda c: c.execute(r._BOOK.update().values(**{field: epoch})))
    if epoch is not None and epoch in (-62135596801, 253402300800):
        with pytest.raises(r.ReconciliationReadError, match="^book_view_unreadable$"):
            e1c_read(repo)
    else:
        basis = e1c_read(repo)
        assert basis.view.net_cost == Decimal("0.20")
        if field == "dirty_from_ts":
            assert basis.sources.pin.dirty_from_ts == epoch


@pytest.mark.parametrize("kind", ["interval", "adjustment"])
def test_e1c_duplicate_visible_versions(repo, terms, inputs, opening, kind):
    from dao.prog.settlement import repository as r

    e1c_credit_history(repo, terms, inputs, opening)
    t = r._INTERVAL if kind == "interval" else r._ADJUSTMENT
    def corrupt(c):
        if kind == "interval":
            c.execute(t.update().values(valid_to_generation=None))
        else:
            row = dict(c.execute(select(t)).mappings().first())
            row["valid_from_generation"] = 2
            row["content_hash"] = "conflicting-content"
            c.execute(t.insert().values(**row))
    e1c_corrupt(repo, corrupt)
    with pytest.raises(r.ReconciliationReadError, match="^book_view_unreadable$"):
        e1c_read(repo, MONTH_END_TS - 1800, MONTH_END_TS)


@pytest.mark.parametrize("field", ["generation", "input_revision", "dirty_from_ts", "snapshot_hashes_json",
                                   "source_watermark_ts", "registration", "publication"])
def test_e1c_header_race_or_consistent_snapshot(repo, terms, inputs, calculate, field):
    from dao.prog.settlement import repository as r

    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots), snapshots, calculate)
    if repo.engine.dialect.name == "sqlite":
        with repo.engine.connect() as c:
            c.exec_driver_sql("PRAGMA journal_mode=WAL")
    gate = Barrier(2)
    fired = False
    def pause(c, cursor, statement, parameters, context, many):
        nonlocal fired
        if statement.startswith("SELECT") and "FROM settlement_interval" in statement and not fired:
            fired = True
            gate.wait(timeout=20)
            gate.wait(timeout=20)
    def writer():
        gate.wait(timeout=20)
        if field in ("registration", "publication"):
            repo.register_inputs(METER, inputs(snapshots, imports=("1", "2")), snapshots, 1)
            if field == "publication":
                batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
                repo.commit(METER, calculate(batch), batch, 1, "race-publication")
            gate.wait(timeout=20)
            return
        values = {"generation": 2, "input_revision": 2, "dirty_from_ts": START_TS,
                  "snapshot_hashes_json": "[]", "source_watermark_ts": START_TS + 3600}
        with repo.engine.begin() as c:
            c.execute(r._BOOK.update().values(**{field: values[field]}))
        gate.wait(timeout=20)
    event.listen(repo.engine, "before_cursor_execute", pause)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(writer)
            try:
                basis = e1c_read(repo)
            except r.ConcurrentSettlementError:
                pass
            else:
                assert basis.sources.pin.current_generation == 1
                assert basis.sources.pin.current_input_revision == 1
                assert basis.view.net_cost == Decimal("0.20")
                assert sum(s.item.import_kwh for s in basis.sources.lines) == 2
            future.result(timeout=20)
        assert fired
    finally:
        event.remove(repo.engine, "before_cursor_execute", pause)


def test_e1c_read_is_select_only_and_preserves_book(repo, terms, inputs, calculate, monkeypatch):
    from dao.prog.settlement import engine, tariffs
    from dao.prog.settlement.reconciliation import _digest

    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots), snapshots, calculate)
    expected = repo.read(METER, START_TS, START_TS + 1800)
    before = {name: rows(repo, name) for name in counts(repo)}
    def forbidden(*args, **kwargs):
        pytest.fail("producer called from reader")
    for name in ("read_inputs", "read_prices", "register_inputs", "commit", "ensure_schema"):
        monkeypatch.setattr(repo, name, forbidden)
    for obj, name in ((engine, "settle_series"), (engine, "close_month"), (tariffs, "make_snapshot")):
        monkeypatch.setattr(obj, name, forbidden)
    connections = set()
    def select_only(c, cursor, statement, parameters, context, many):
        assert statement.lstrip().startswith("SELECT"), statement
        connections.add(id(c))
    event.listen(repo.engine, "before_cursor_execute", select_only)
    try:
        basis = e1c_read(repo)
    finally:
        event.remove(repo.engine, "before_cursor_execute", select_only)
    assert len(connections) == 1
    assert basis.view == expected and basis.sources.pin.view_digest == _digest(expected)
    assert basis.sources.pin.read_ts >= basis.view.captured_ts
    assert not basis.sources.issues and all(not s.issues for s in basis.sources.lines)
    assert before == {name: rows(repo, name) for name in before}


def test_e1c_reconciliation_read_does_not_query_per_book_line(repo, terms, inputs, calculate):
    """De historische read haalt zijn bronbewijs in bulk op, niet per boekregel.

    De lezer mag niet per regel een vaste aantal statements uitvoeren: dat
    maakt de kosten van een groot boekvenster onbegrensd. De grens hieronder
    is ruim boven het aantal vaste statements van de read en ruim onder het
    oude gedrag (drie statements per boekregel).
    """
    count = 48
    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots, imports=("1",) * count, exports=("0",) * count),
           snapshots, calculate)
    statements = []

    def counted(connection, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(repo.engine, "before_cursor_execute", counted)
    try:
        basis = repo.read_reconciliation(METER, START_TS, START_TS + 900 * count)
    finally:
        event.remove(repo.engine, "before_cursor_execute", counted)
    assert len(basis.view.lines) == count
    assert not basis.sources.issues and all(not source.issues for source in basis.sources.lines)
    assert len(statements) <= count // 4 + 12, (
        f"{len(statements)} statements voor {count} boekregels")


@pytest.fixture(params=["sqlite"] + [name for name in ("POSTGRESQL", "MARIADB")
                                   if os.getenv(f"SETTLEMENT_TEST_{name}_URL")])
def repo(tmp_path, request):
    from dao.prog.settlement.repository import SettlementRepository
    from dao.tests.settlement.test_schema import local_server_url

    server = request.param != "sqlite"
    url = local_server_url(request.param) if server else f"sqlite:///{tmp_path / 'settlement.db'}"
    engine = create_engine(url, poolclass=NullPool)

    if not server:
        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")
    else:
        from sqlalchemy import inspect

        assert not inspect(engine).get_table_names(), "Use an empty, dedicated A6 test database"

    repository = SettlementRepository(engine)
    repository.ensure_schema()
    repository.ensure_schema()
    yield repository
    if server:
        from dao.prog.settlement.repository import _METADATA

        _METADATA.drop_all(engine)  # Uitsluitend de zes tabellen die deze fixture zelf maakte.
    engine.dispose()


@pytest.fixture
def terms(snapshot):
    def build(**overrides):
        item = snapshot(**{"monthly_export_floor": "none", **overrides})
        content = asdict(item)
        del content["snapshot_hash"]
        canonical = json.dumps(content, default=_canonical_default, sort_keys=True,
                               separators=(",", ":"))
        return replace(item, snapshot_hash=sha256(canonical.encode()).hexdigest())

    return build


@pytest.fixture
def inputs(meter):
    def build(snapshots, *, start=START_TS, imports=("1", "1"), exports=None, spot="0.10",
              sources=("synthetic-meter", "no-battery-proof")):
        exports = ("0",) * len(imports) if exports is None else exports
        rows = [meter(start_ts=start + i * 900, end_ts=start + (i + 1) * 900,
                      import_kwh=None if imported is None else Decimal(imported),
                      export_kwh=Decimal(exported), sources=sources)
                for i, (imported, exported) in enumerate(zip(imports, exports, strict=True))]
        prices = [SpotInterval(at, at + 3600, None if spot is None else Decimal(spot),
                               "synthetic-hour-price", "measured")
                  for at in range(start // 3600 * 3600, rows[-1].end_ts, 3600)]
        return merge_inputs(rows, prices, snapshots, meter_id=METER)

    return build


@pytest.fixture
def calculate(opening):
    def build(batch):
        return settle_series(batch.items, batch.snapshots,
                             opening(asof_ts=batch.items[0].start_ts, bonus=None))

    return build


def table(repo, name):
    return Table(f"settlement_{name}", MetaData(), autoload_with=repo.engine)


def rows(repo, name):
    with repo.engine.connect() as connection:
        return connection.execute(select(table(repo, name))).mappings().all()


def counts(repo):
    with repo.engine.connect() as connection:
        return {name: connection.scalar(select(func.count()).select_from(table(repo, name)))
                for name in ("tariff", "input", "run", "interval", "adjustment", "book")}


def publish(repo, items, snapshots, calculate, *, revision=0, generation=0):
    repo.register_inputs(METER, items, snapshots, revision)
    batch = repo.read_inputs(METER, items[0].start_ts, items[-1].end_ts)
    result = calculate(batch)
    return repo.commit(METER, result, batch, generation, "synthetic-test"), batch, result


def test_commit_same_input_is_noop(repo, terms, inputs, calculate):
    snapshots = [terms()]
    items = inputs(snapshots)
    first, _, _ = publish(repo, items, snapshots, calculate)
    original = counts(repo)
    second, batch, _ = publish(repo, items, snapshots, calculate, revision=1, generation=first)
    assert first == second == 1
    assert counts(repo) == original
    assert original["run"] == 1 and original["interval"] == 2
    assert batch.input_revision == 1
    assert [item.input_revision for item in batch.items] == [1, 1]
    assert [item.input_revision for item in items] == [0, 0]
    assert repo.read(METER, START_TS, START_TS + 1800).net_cost == Decimal("0.20")


def test_conflict_rolls_back_all_rows(repo, terms, inputs, calculate):
    from dao.prog.settlement.repository import ConcurrentSettlementError

    snapshots = [terms()]
    publish(repo, inputs(snapshots), snapshots, calculate)
    repo.register_inputs(METER, inputs(snapshots, imports=("2", "1")), snapshots, 1)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    before = {name: rows(repo, name) for name in ("book", "run", "interval", "adjustment")}
    with pytest.raises(ConcurrentSettlementError):
        repo.commit(METER, calculate(batch), batch, 0, "stale-generation")
    assert before == {name: rows(repo, name) for name in before}
    assert repo.read(METER, START_TS, START_TS + 1800).net_cost == Decimal("0.20")


def test_crash_before_pointer_publish(repo, terms, inputs, calculate):
    snapshots = [terms()]
    publish(repo, inputs(snapshots), snapshots, calculate)
    repo.register_inputs(METER, inputs(snapshots, imports=("2", "1")), snapshots, 1)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    before = {name: rows(repo, name) for name in ("book", "run", "interval", "adjustment")}
    def crash(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE settlement_book SET generation="):
            raise RuntimeError("injected-before-pointer-publish")

    event.listen(repo.engine, "before_cursor_execute", crash)
    try:
        with pytest.raises(RuntimeError, match="injected-before-pointer-publish"):
            repo.commit(METER, calculate(batch), batch, 1, "crash-test")
    finally:
        event.remove(repo.engine, "before_cursor_execute", crash)
    assert before == {name: rows(repo, name) for name in before}
    assert repo.read(METER, START_TS, START_TS + 1800).generation == 1


def test_pinned_old_generation(repo, terms, inputs, calculate):
    snapshots = [terms()]
    publish(repo, inputs(snapshots), snapshots, calculate)
    publish(repo, inputs(snapshots, imports=("2", "1")), snapshots, calculate,
            revision=1, generation=1)
    old = repo.read(METER, START_TS, START_TS + 1800, generation=1)
    current = repo.read(METER, START_TS, START_TS + 1800)
    assert old.generation == 1 and old.net_cost == Decimal("0.20")
    assert current.generation == 2 and current.net_cost == Decimal("0.30")
    assert old.input_revision == 1 and old.current_input_revision == 2
    assert old.freshness == "outdated" and current.freshness == "current"


def test_only_changed_intervals_get_versions(repo, terms, inputs, calculate):
    snapshots = [terms()]
    publish(repo, inputs(snapshots), snapshots, calculate)
    publish(repo, inputs(snapshots, imports=("2", "1")), snapshots, calculate,
            revision=1, generation=1)
    versions = sorted(rows(repo, "interval"), key=lambda row: (row["start_ts"], row["valid_from_generation"]))
    assert [(row["start_ts"] - START_TS, row["valid_from_generation"], row["valid_to_generation"],
             row["input_revision"]) for row in versions] == [(0, 1, 2, 1), (0, 2, None, 2), (900, 1, None, 1)]
    assert len(rows(repo, "input")) == 3
    assert versions[0]["import_cost"] == "0.1"


def test_new_input_revision_rejects_old_result(repo, terms, inputs, calculate):
    from dao.prog.settlement.repository import ConcurrentSettlementError

    snapshots = [terms()]
    _, batch, result = publish(repo, inputs(snapshots), snapshots, calculate)
    # Alleen bronbewijs verandert: de financiële generatie blijft nog 1.
    repo.register_inputs(METER, inputs(snapshots, sources=("corrected-source",)), snapshots, 1)
    before = counts(repo)
    with pytest.raises(ConcurrentSettlementError):
        repo.commit(METER, result, batch, 1, "stale-input-revision")
    assert counts(repo) == before
    stale = repo.read(METER, START_TS, START_TS + 1800)
    assert stale.generation == 1 and stale.current_input_revision == 2
    assert stale.freshness == "outdated" and stale.dirty_from_ts == START_TS


def test_restore_previous_content_still_has_new_revision(repo, terms, inputs, calculate):
    snapshots = [terms()]
    original = inputs(snapshots)
    publish(repo, original, snapshots, calculate)
    publish(repo, inputs(snapshots, imports=("2", "1")), snapshots, calculate,
            revision=1, generation=1)
    generation, batch, _ = publish(repo, original, snapshots, calculate, revision=2, generation=2)
    assert generation == 3 and batch.input_revision == 3
    assert batch.items[0].input_hash == original[0].input_hash
    assert [item.input_revision for item in batch.items] == [3, 1]
    assert sorted(row["input_revision"] for row in rows(repo, "input") if row["start_ts"] == START_TS) == [1, 2, 3]


def test_two_real_connections_publish_exactly_one_generation4(repo, terms, inputs, calculate):
    from dao.prog.settlement.repository import ConcurrentSettlementError, SettlementRepository

    snapshots = [terms()]
    for index in range(3):
        publish(repo, inputs(snapshots, imports=(str(index + 1), "1")), snapshots, calculate,
                revision=index, generation=index)
    repo.register_inputs(METER, inputs(snapshots, imports=("4", "1")), snapshots, 3)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    result = calculate(batch)
    barrier = Barrier(2)

    def writer():
        barrier.wait(timeout=10)
        try:
            return SettlementRepository(repo.engine).commit(METER, result, batch, 3, "concurrent")
        except ConcurrentSettlementError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: writer(), range(2)))
    assert sorted(outcomes, key=str) == [4, "conflict"]
    assert len(rows(repo, "run")) == 4
    assert repo.read(METER, START_TS, START_TS + 1800).net_cost == Decimal("0.50")


def test_concurrent_first_registrations_initialize_one_book(repo, terms, inputs):
    from dao.prog.settlement.repository import ConcurrentSettlementError, SettlementRepository

    snapshots = [terms()]
    items = inputs(snapshots)
    barrier = Barrier(2)

    def writer():
        barrier.wait(timeout=10)
        try:
            return SettlementRepository(repo.engine).register_inputs(METER, items, snapshots, 0)
        except ConcurrentSettlementError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: writer(), range(2)))
    assert sorted(outcomes, key=str) == [1, "conflict"]
    assert len(rows(repo, "book")) == 1 and len(rows(repo, "input")) == 2


def test_different_meter_writers_can_register_one_shared_tariff(repo, terms, inputs):
    """Separate server row locks must not race on the global immutable tariff PK."""
    from dao.prog.settlement.repository import SettlementRepository

    if repo.engine.dialect.name == "sqlite":
        pytest.skip("SQLite serializes all writers; shared-tariff race requires server row locks")
    snapshots = [terms()]
    original = inputs(snapshots)
    barrier = Barrier(2)

    def both_read_absent(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("SELECT settlement_tariff.snapshot_hash \n"):
            barrier.wait(timeout=10)

    def writer(meter_id):
        items = tuple(replace(item, meter_id=meter_id) for item in original)
        return SettlementRepository(repo.engine).register_inputs(meter_id, items, snapshots, 0)

    event.listen(repo.engine, "after_cursor_execute", both_read_absent)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(writer, ("first-grid", "second-grid")))
    finally:
        event.remove(repo.engine, "after_cursor_execute", both_read_absent)
    assert outcomes == [1, 1]
    assert len(rows(repo, "tariff")) == 1 and len(rows(repo, "input")) == 4


def test_contract_selection_is_explicit_pinned_and_revisioned(repo, terms, inputs, calculate):
    original = terms()
    revised = terms(cost_supplier_consumption=((original.cost_supplier_consumption[0][0], Decimal("0.20")),))
    items = inputs([original])
    _, old_batch, old_result = publish(repo, items, [original], calculate)
    assert repo.register_inputs(METER, items, [revised], 1) == 2
    new_batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    assert old_batch.snapshots == (original,) and new_batch.snapshots == (revised,)
    assert [item.input_revision for item in new_batch.items] == [1, 1]
    assert repo.commit(METER, calculate(new_batch), new_batch, 1, "approved-contract-correction") == 2
    assert old_result.net_cost == Decimal("0.20")
    assert repo.read(METER, START_TS, START_TS + 1800).net_cost == Decimal("0.60")
    assert repo.read(METER, START_TS, START_TS + 1800, 1).net_cost == Decimal("0.20")


@pytest.mark.parametrize("partial_window", [None, (0, 900), (900, 1800)],
                         ids=["full", "prefix-first", "tail-first"])
def test_contract_roundtrip_revalidation_preserves_financial_generation(repo, terms, inputs, calculate,
                                                                      partial_window):
    """A restored contract is current only after full CAS-checked revalidation."""
    from dao.prog.settlement.repository import ConcurrentSettlementError

    original = terms()
    revised = terms(cost_supplier_consumption=((original.cost_supplier_consumption[0][0], Decimal("0.20")),))
    items = inputs([original])
    generation, old_batch, old_result = publish(repo, items, [original], calculate)
    original_view = repo.read(METER, START_TS, START_TS + 1800)
    frozen = {name: rows(repo, name) for name in ("input", "run", "interval", "adjustment")}
    assert repo.register_inputs(METER, items, [revised], 1) == 2
    assert repo.register_inputs(METER, items, [original], 2) == 3
    restored = repo.read_inputs(METER, START_TS, START_TS + 1800)
    assert restored.input_revision == 3 and restored.snapshots == (original,)
    stale = repo.read(METER, START_TS, START_TS + 1800)
    assert stale.input_revision == 1 and stale.current_input_revision == 3
    assert stale.dirty_from_ts == START_TS and stale.freshness == "outdated"

    if partial_window is not None:
        left, right = partial_window
        partial = repo.read_inputs(METER, START_TS + left, START_TS + right)
        partial_result = settle_series(partial.items, partial.snapshots, old_result.lines[left // 900].before_state)
        assert repo.commit(METER, partial_result, partial, generation, "partial-revalidation") == generation
        still_dirty = repo.read(METER, START_TS, START_TS + 1800)
        assert still_dirty.dirty_from_ts == START_TS and still_dirty.freshness == "outdated"
        assert still_dirty.input_revision == 1 and still_dirty.current_input_revision == 3
        assert frozen == {name: rows(repo, name) for name in frozen}

    assert repo.commit(METER, calculate(restored), restored, generation, "full-revalidation") == generation
    current = repo.read(METER, START_TS, START_TS + 1800)
    assert current.generation == generation == 1
    assert current.dirty_from_ts is None and current.freshness == "current"
    assert current.input_revision == current.current_input_revision == 3
    assert current.net_cost == Decimal("0.20") and current.complete
    assert [line.input_revision for line in current.lines] == [1, 1]
    assert frozen == {name: rows(repo, name) for name in frozen}
    assert rows(repo, "run")[0]["input_revision"] == 1
    assert old_batch.input_revision == original_view.input_revision == 1

    validated_book = rows(repo, "book")
    with pytest.raises(ConcurrentSettlementError):
        repo.commit(METER, old_result, old_batch, generation, "stale-before-revalidation")
    assert rows(repo, "book") == validated_book
    assert frozen == {name: rows(repo, name) for name in frozen}

    # Na een echte volgende financiële generatie blijft de oude auditmetadata gepind.
    assert repo.register_inputs(METER, items, [revised], 3) == 4
    next_batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    assert repo.commit(METER, calculate(next_batch), next_batch, generation, "changed-contract") == 2
    pinned = repo.read(METER, START_TS, START_TS + 1800, generation=1)
    assert pinned.input_revision == 1 and pinned.current_input_revision == 4
    assert pinned.freshness == "outdated" and pinned.dirty_from_ts is None
    assert pinned.lines == original_view.lines and pinned.adjustments == original_view.adjustments
    assert pinned.captured_ts == original_view.captured_ts
    assert pinned.source_watermark_ts == original_view.source_watermark_ts
    assert pinned.net_cost == original_view.net_cost == Decimal("0.20")
    assert next(row for row in rows(repo, "run") if row["generation"] == 1) == frozen["run"][0]
    latest = repo.read(METER, START_TS, START_TS + 1800)
    assert latest.input_revision == 4 and latest.freshness == "current" and latest.net_cost == Decimal("0.60")


def test_draft_is_never_silently_confirmed(repo, terms, inputs):
    draft = terms(status="draft")
    with pytest.raises(ValueError, match="confirmed"):
        repo.register_inputs(METER, inputs([draft]), [draft], 0)
    assert len(rows(repo, "tariff")) == 0 and len(rows(repo, "input")) == 0


def test_snapshot_hash_cannot_alias_different_financial_content(repo, terms, inputs):
    original = terms()
    forged = replace(original, contract_id="different-contract-with-same-hash")
    with pytest.raises(ValueError, match="hash"):
        repo.register_inputs(METER, inputs([original]), [forged], 0)
    assert len(rows(repo, "tariff")) == 0


def test_unregistered_line_cannot_be_published(repo, terms, inputs, calculate):
    snapshots = [terms()]
    repo.register_inputs(METER, inputs(snapshots), snapshots, 0)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    result = calculate(batch)
    forged = replace(result, lines=(replace(result.lines[0], input_revision=0), *result.lines[1:]))
    with pytest.raises(ValueError, match="(revision|registered)"):
        repo.commit(METER, forged, batch, 0, "unregistered-preview")
    assert len(rows(repo, "run")) == 0 and len(rows(repo, "interval")) == 0


def test_forged_batch_is_not_a_registered_input_reference(repo, terms, inputs, calculate):
    snapshots = [terms()]
    repo.register_inputs(METER, inputs(snapshots), snapshots, 0)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    forged = replace(batch, items=(replace(batch.items[0], import_kwh=Decimal("100")), *batch.items[1:]))
    with pytest.raises(ValueError, match="registered"):
        repo.commit(METER, calculate(forged), forged, 0, "forged-batch")
    assert len(rows(repo, "run")) == 0


def test_register_same_hash_with_changed_raw_content_is_not_a_noop(repo, terms, inputs):
    snapshots = [terms()]
    original = inputs(snapshots)
    assert repo.register_inputs(METER, original, snapshots, 0) == 1
    changed = (replace(original[0], import_kwh=Decimal("2")), original[1])
    assert repo.register_inputs(METER, changed, snapshots, 1) == 2
    assert repo.read_inputs(METER, START_TS, START_TS + 1800).items[0].import_kwh == Decimal("2")


def test_canonical_decimals_do_not_create_false_revisions(repo, terms, inputs):
    snapshots = [terms()]
    original = inputs(snapshots)
    assert repo.register_inputs(METER, original, snapshots, 0) == 1
    equal = tuple(replace(item, import_kwh=Decimal("1.000000000"), spot_ex_vat=Decimal("0.100000000000"))
                  for item in original)
    with localcontext() as context:
        context.prec = 6
        assert repo.register_inputs(METER, equal, snapshots, 1) == 1


def test_dirty_timestamp_and_source_watermark_survive_until_covered(repo, terms, inputs, calculate):
    snapshots = [terms()]
    publish(repo, inputs(snapshots, imports=("1", "1", "1")), snapshots, calculate)
    repo.register_inputs(METER, inputs(snapshots, imports=("1", "2", "1")), snapshots, 1)
    before = repo.read(METER, START_TS, START_TS + 2700)
    assert before.dirty_from_ts == START_TS + 900 and before.source_watermark_ts == START_TS + 2700
    partial = repo.read_inputs(METER, START_TS + 900, START_TS + 1800)
    assert partial.source_watermark_ts == START_TS + 2700
    repo.commit(METER, calculate(partial), partial, 1, "partial-not-a-full-tail")
    assert repo.read(METER, START_TS, START_TS + 2700).freshness == "outdated"
    full = repo.read_inputs(METER, START_TS, START_TS + 2700)
    # Zelfs als de resterende regels al gelijk zijn, mag pas volledige dekking dirty wissen.
    repo.commit(METER, calculate(full), full, 2, "complete-tail")
    current = repo.read(METER, START_TS, START_TS + 2700)
    assert current.dirty_from_ts is None and current.freshness == "current"


def test_zero_epoch_watermark_is_not_treated_as_missing(repo, terms, inputs, calculate):
    snapshots = [terms(valid_from_ts=-900)]
    original = inputs(snapshots, start=-900, imports=("1",))
    publish(repo, original, snapshots, calculate)
    # Oude prijsvoorwaarden zijn hier bewust onbekend; dat mag een geregistreerde
    # bronwatermerkwaarde 0 niet veranderen in een ontbrekend watermerk.
    revised = [replace(original[0], end_ts=-450)]
    publish(repo, revised, snapshots, calculate, revision=1, generation=1)
    view = repo.read(METER, -900, 0)
    assert view.source_watermark_ts == 0
    assert view.dirty_from_ts == -900 and view.freshness == "outdated"


@pytest.mark.parametrize("replacement_spot,want_credit", [("0.10", Decimal("0")), (None, None)])
def test_obsolete_adjustment_does_not_leak_into_new_generation(repo, terms, inputs, calculate,
                                                            replacement_spot, want_credit):
    snapshots = [terms(monthly_export_floor="spot_zero")]
    start = MONTH_END_TS - 900
    original = inputs(snapshots, start=start, imports=("0",), exports=("1",), spot="-0.20")
    publish(repo, original, snapshots, calculate)
    assert repo.read(METER, start, MONTH_END_TS).adjustments[0].credit_total == Decimal("0.20")
    revised = inputs(snapshots, start=start, imports=("0",), exports=("1",), spot=replacement_spot)
    publish(repo, revised, snapshots, calculate, revision=1, generation=1)
    old = repo.read(METER, start, MONTH_END_TS, 1)
    current = repo.read(METER, start, MONTH_END_TS)
    assert old.net_cost == Decimal("0") and old.adjustments[0].credit_total == Decimal("0.20")
    if want_credit is None:
        assert current.adjustments == () and current.net_cost is None
    else:
        assert current.adjustments[0].credit_total == want_credit and current.net_cost == Decimal("-0.10")
    assert len(rows(repo, "adjustment")) == 2


def test_read_partition_uses_period_end_minus_one_and_exact_decimal_sum(repo, terms, inputs, calculate):
    snapshots = [terms(monthly_export_floor="spot_zero")]
    start = MONTH_END_TS - 1800
    items = inputs(snapshots, start=start, imports=("1", "1"), exports=("1", "1"), spot="-0.123456789012")
    publish(repo, items, snapshots, calculate)
    full = repo.read(METER, start, MONTH_END_TS)
    left = repo.read(METER, start, start + 900, full.generation)
    right = repo.read(METER, start + 900, MONTH_END_TS, full.generation)
    assert left.adjustments == () and len(right.adjustments) == 1
    assert full.net_cost == Decimal("-0.246913578024")
    assert left.net_cost + right.net_cost == full.net_cost


def test_decimal_components_and_nested_states_roundtrip_without_context_rounding(repo, terms, inputs, calculate):
    from datetime import date

    snapshots = [terms(monthly_export_floor="spot_zero",
                       vat_production=((date(2027, 1, 1), Decimal("21")),))]
    start = MONTH_END_TS - 900
    items = inputs(snapshots, start=start, imports=("0",), exports=("0.123456789",), spot="-0.123456789012")
    _, batch, result = publish(repo, items, snapshots, calculate)
    with localcontext() as context:
        context.prec = 6
        view = repo.read(METER, start, MONTH_END_TS)
        restored = repo.read_inputs(METER, start, MONTH_END_TS)
    assert view.lines == result.lines and restored.items == batch.items and restored.snapshots == batch.snapshots
    assert view.lines[0].base_export_revenue == Decimal("-0.01844231028952312298628")
    assert view.adjustments[0].credit_vat == Decimal("0.00320073153785112051828")
    assert view.net_cost == Decimal("0")


def test_large_batch_does_not_copy_unchanged_financial_rows(repo, terms, inputs, calculate):
    snapshots = [terms()]
    original = inputs(snapshots, imports=("1",) * 501)
    publish(repo, original, snapshots, calculate)
    revised = list(original)
    revised[250] = replace(revised[250], import_kwh=Decimal("2"))
    publish(repo, revised, snapshots, calculate, revision=1, generation=1)
    assert len(rows(repo, "input")) == 502 and len(rows(repo, "interval")) == 502
    assert repo.read(METER, START_TS, START_TS + 501 * 900).net_cost == Decimal("50.20")
    assert repo.read(METER, START_TS, START_TS + 501 * 900, 1).net_cost == Decimal("50.10")


def test_database_rejects_a_positive_but_unregistered_financial_fk(repo, terms, inputs, calculate):
    from sqlalchemy.exc import IntegrityError

    snapshots = [terms()]
    publish(repo, inputs(snapshots), snapshots, calculate)
    financial = dict(rows(repo, "interval")[0])
    financial.update(valid_from_generation=2, input_revision=999)
    with pytest.raises(IntegrityError), repo.engine.begin() as connection:
        connection.execute(table(repo, "interval").insert(), financial)
    assert repo.read(METER, START_TS, START_TS + 1800).generation == 1


def test_reader_pins_generation_once_when_publication_interleaves(repo, terms, inputs, calculate):
    snapshots = [terms()]
    publish(repo, inputs(snapshots), snapshots, calculate)
    repo.register_inputs(METER, inputs(snapshots, imports=("2", "1")), snapshots, 1)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    result = calculate(batch)
    fired = False

    def interleave(connection, cursor, statement, parameters, context, executemany):
        nonlocal fired
        if not fired and statement.lstrip().startswith("SELECT") and "FROM settlement_interval" in statement:
            fired = True
            repo.commit(METER, result, batch, 1, "interleaved-writer")

    # WAL laat de onafhankelijke schrijver door terwijl de lezer zijn gepinde boek verwerkt.
    if repo.engine.dialect.name == "sqlite":
        with repo.engine.connect() as connection:
            connection.execute(text("PRAGMA journal_mode=WAL"))
    event.listen(repo.engine, "before_cursor_execute", interleave)
    try:
        view = repo.read(METER, START_TS, START_TS + 1800)
    finally:
        event.remove(repo.engine, "before_cursor_execute", interleave)
    assert fired and view.generation == 1 and view.net_cost == Decimal("0.20")
    assert repo.read(METER, START_TS, START_TS + 1800).generation == 2


def test_unknown_and_missing_coverage_never_become_zero(repo, terms, inputs, calculate):
    empty = repo.read(METER, START_TS, START_TS + 1800)
    assert empty.generation == 0 and empty.net_cost is None and empty.freshness == "unknown"
    assert repo.read_inputs(METER, START_TS, START_TS + 1800).input_revision == 0
    assert len(rows(repo, "book")) == 0  # Een leesactie initialiseert niets.
    snapshots = [terms()]
    publish(repo, inputs(snapshots, imports=(None, "1")), snapshots, calculate)
    view = repo.read(METER, START_TS, START_TS + 1800)
    assert view.known_subtotal == Decimal("0.10") and view.net_cost is None and not view.complete
    assert repo.read(METER, START_TS, START_TS + 2700).net_cost is None


@pytest.mark.parametrize("case", ["other-meter", "overlap", "duplicate", "overlapping-terms"])
def test_invalid_registration_is_atomic(repo, terms, inputs, case):
    snapshots = [terms()]
    items = inputs(snapshots)
    if case == "other-meter":
        items = (replace(items[0], meter_id="another-grid"), items[1])
    elif case == "overlap":
        items = (replace(items[0], end_ts=items[1].end_ts), items[1])
    elif case == "duplicate":
        items = (items[0], items[0])
    else:
        snapshots.append(terms(contract_id="overlapping-contract"))
    with pytest.raises(ValueError):
        repo.register_inputs(METER, items, snapshots, 0)
    assert len(rows(repo, "input")) == 0 and len(rows(repo, "tariff")) == 0


def test_changed_end_cannot_overlap_previously_registered_input(repo, terms, inputs):
    snapshots = [terms()]
    original = inputs(snapshots)
    repo.register_inputs(METER, original, snapshots, 0)
    with pytest.raises(ValueError, match="overlap"):
        repo.register_inputs(METER, [replace(original[0], end_ts=START_TS + 1800)], snapshots, 1)
    assert repo.read_inputs(METER, START_TS, START_TS + 1800).input_revision == 1


def test_multiple_meters_do_not_share_revisions_or_money(repo, terms, inputs, calculate):
    snapshots = [terms()]
    items = inputs(snapshots)
    publish(repo, items, snapshots, calculate)
    other = tuple(replace(item, meter_id="second-grid") for item in items)
    assert repo.register_inputs("second-grid", other, snapshots, 0) == 1
    batch = repo.read_inputs("second-grid", START_TS, START_TS + 1800)
    assert repo.commit("second-grid", calculate(batch), batch, 0, "other-meter") == 1
    assert repo.read("second-grid", START_TS, START_TS + 1800).net_cost == Decimal("0.20")
    assert repo.read(METER, START_TS, START_TS + 1800).generation == 1


def test_book_reference_is_read_only_and_uses_current_validation_revision(repo, terms, inputs, calculate):
    from dao.prog.settlement.repository import BookReference

    assert repo.read_book_reference(METER) == BookReference(0, (), None)
    assert not rows(repo, "book")
    original = terms()
    revised = terms(contract_id="changed-contract")
    items = inputs([original])
    publish(repo, items, [original], calculate)
    repo.register_inputs(METER, items, [revised], 1)
    repo.register_inputs(METER, items, [original], 2)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    repo.commit(METER, calculate(batch), batch, 1, "full-ABA-revalidation")
    assert repo.read_book_reference(METER) == BookReference(3, (original.snapshot_hash,), None)
    assert rows(repo, "run")[0]["input_revision"] == 1


def test_optional_registration_guard_observes_real_changes_and_rolls_back(repo, terms, inputs):
    from contextlib import contextmanager

    from dao.prog.settlement.repository import SettlementRepository

    changes = []
    class Guard:
        held = False
        fail = False

        @contextmanager
        def hold(self):
            assert not self.held
            self.held = True
            try:
                yield
            finally:
                self.held = False

        def before_revision(self, change):
            assert self.held
            changes.append(change)
            if self.fail:
                raise RuntimeError("invalidation failed")

    guard = Guard()
    writer = SettlementRepository(repo.engine, publication=guard)
    snapshots = [terms()]
    original = inputs(snapshots, imports=("1",))
    assert writer.register_inputs(METER, original, snapshots, 0) == 1
    assert changes[-1].previous_revision == 0 and changes[-1].next_revision == 1
    assert changes[-1].changed_intervals == ((START_TS, START_TS + 900),)
    assert changes[-1].changed_prices == ((START_TS, START_TS + 3600),)
    assert changes[-1].changed_contracts == ((START_TS, None),)
    changes.clear()
    assert writer.register_inputs(METER, original, snapshots, 1) == 1
    assert changes == []
    assert writer.register_inputs(METER, inputs(snapshots, start=START_TS + 900, imports=("1",)),
                                  snapshots, 1) == 2
    assert changes[-1].changed_prices == ()  # Dezelfde reeds gepinde uurprijs.
    assert changes[-1].changed_contracts == ()
    frozen = {name: rows(repo, name) for name in ("book", "input", "tariff")}
    guard.fail = True
    with pytest.raises(RuntimeError, match="invalidation failed"):
        writer.register_inputs(METER, inputs(snapshots, imports=("2",)), snapshots, 2)
    assert frozen == {name: rows(repo, name) for name in frozen}
    assert not guard.held


def test_guard_distinguishes_meter_quality_from_price_quality(repo, terms, inputs):
    from contextlib import nullcontext

    from dao.prog.settlement.repository import SettlementRepository

    changes = []
    class Guard:
        def hold(self):
            return nullcontext()

        def before_revision(self, change):
            changes.append(change)

    writer = SettlementRepository(repo.engine, publication=Guard())
    snapshots = [terms()]
    original = inputs(snapshots, imports=("1",))
    writer.register_inputs(METER, original, snapshots, 0)
    # End/meetkwaliteit veranderen de bronkwaliteit van de uurprijs niet.
    revised = [replace(original[0], end_ts=START_TS + 450, quality="estimated")]
    writer.register_inputs(METER, revised, snapshots, 1)
    assert changes[-1].changed_intervals == ((START_TS, START_TS + 900),)
    assert changes[-1].changed_prices == ()
    priced = [replace(revised[0], spot_source=revised[0].spot_source.replace('"measured"', '"estimated"'))]
    writer.register_inputs(METER, priced, snapshots, 2)
    assert changes[-1].changed_prices == ((START_TS, START_TS + 3600),)


def test_guard_price_comparison_does_not_do_one_sql_read_per_new_meter_row(repo, terms, inputs):
    """Historical registration must not expand its short file lease into N+1 DB reads."""
    from contextlib import nullcontext

    from dao.prog.settlement.repository import SettlementRepository

    changes, reads = [], []
    class Guard:
        def hold(self):
            return nullcontext()

        def before_revision(self, change):
            changes.append(change)

    snapshots = [terms()]
    items = inputs(snapshots, imports=("1",) * 96)
    def observe(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("SELECT") and "settlement_input" in statement:
            reads.append(statement)
    event.listen(repo.engine, "before_cursor_execute", observe)
    try:
        SettlementRepository(repo.engine, publication=Guard()).register_inputs(METER, items, snapshots, 0)
    finally:
        event.remove(repo.engine, "before_cursor_execute", observe)
    assert len(reads) <= 3
    assert changes[0].changed_prices == tuple((START_TS + hour * 3600, START_TS + (hour + 1) * 3600)
                                             for hour in range(24))
