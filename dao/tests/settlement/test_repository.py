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
