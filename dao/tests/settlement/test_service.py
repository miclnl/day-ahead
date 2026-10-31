"""Real service, guarded SQLite book and synthetic, independently priced sources."""

from dataclasses import replace
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from dao.prog.settlement.publication import FinancialPlanBlocked, make_publication
from dao.prog.settlement.tariffs import make_snapshot
from dao.prog.settlement.types import MeterInterval, SpotInterval
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_publication import METER, pricing_config
from dao.tests.settlement.test_repository import terms as terms


@pytest.fixture
def system(tmp_path, monkeypatch):
    from dao.prog.settlement.service import SettlementService

    clock = [START_TS + 86400]
    for module in ("service", "repository", "publication"):
        monkeypatch.setattr(f"dao.prog.settlement.{module}.time", lambda: clock[0])
    pricing = pricing_config()
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    engine = create_engine(f"sqlite:///{tmp_path / 'service.db'}")
    path = tmp_path / "fast_plan.json"
    repo, publication = make_publication(engine, plan_path=path, pricing=pricing, max_plan_age=600)
    repo.ensure_schema()
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE "values" (time INTEGER, cost TEXT, profit TEXT)'))
        connection.execute(text('INSERT INTO "values" VALUES (1, \'2\', \'3\')'))

    class Meter:
        meter_id = METER

        def __init__(self):
            self.items = [MeterInterval(START_TS + i * 900, START_TS + (i + 1) * 900,
                                       Decimal("1"), Decimal("0"), Decimal("0"),
                                       "verified_pv", "measured", ("synthetic-meter",)) for i in range(8)]
            self.calls = []

        def __call__(self, start, end):
            self.calls.append((start, end))
            return tuple(item for item in self.items if start <= item.start_ts and item.end_ts <= end)

    meter = Meter()
    spots = [SpotInterval(START_TS + i * 3600, START_TS + (i + 1) * 3600,
                          Decimal("0.10") if i == 0 else Decimal("0.40"),
                          "synthetic-hour", "measured") for i in range(4)]

    def prices(start, end):
        return tuple(item for item in spots if item.start_ts < end and start < item.end_ts)

    service = SettlementService(repo, meter, prices, (snapshot,), publication=publication)
    yield SimpleNamespace(service=service, repo=repo, publication=publication, snapshot=snapshot,
                          meter=meter, spots=spots, prices=prices, pricing=pricing,
                          path=path, engine=engine, clock=clock)
    engine.dispose()


def contents(system):
    with system.engine.connect() as connection:
        return tuple(connection.execute(text("SELECT name FROM sqlite_master ORDER BY name")).all()), tuple(
            (name, tuple(connection.execute(text(f'SELECT * FROM "{name}"')).all()))
            for name in ("values", "settlement_spot", "settlement_input", "settlement_tariff", "settlement_book",
                         "settlement_run", "settlement_interval", "settlement_adjustment"))


def test_preview_never_invalidates_or_creates_lock(system):
    before = contents(system)
    result = system.service.preview(START_TS, START_TS + 1800)
    assert result.net_cost == Decimal("0.20")
    assert all(line.input_revision == 0 for line in result.lines)
    assert contents(system) == before
    assert not system.path.exists() and not system.path.with_suffix(".json.lock").exists()


def test_shadow_engine_never_writes_legacy_or_actuation(system):
    before = contents(system)[1][0]
    view = system.service.consolidate(START_TS, START_TS + 1800, publish=True, reason="shadow-test")
    assert view.net_cost == Decimal("0.20") and view.generation == 1
    assert contents(system)[1][0] == before
    assert system.path.exists()  # Guard invalidation, never a gateway/sink.
    assert system.repo.read_inputs(METER, START_TS, START_TS + 1800).source_watermark_ts == START_TS + 1800


def test_consolidate_defaults_to_read_only(system):
    before = contents(system)
    system.service.consolidate(START_TS, START_TS + 1800, reason="what-if")
    assert contents(system) == before and not system.path.exists()


def test_state_tail_not_counted_twice(system, terms):
    system.meter.items = [replace(item, export_kwh="1") for item in system.meter.items]
    # Use explicit all-day synthetic bonus terms to observe cumulative export.
    from dao.prog.settlement.types import BonusRules
    bonus = BonusRules(True, "pv only", date(2027, 1, 1), "30", "0.2", "ex_vat", "0", "5000",
                       tuple((month, (0, 24)) for month in range(1, 13)))
    system.service.snapshots = (terms(bonus=bonus, supply_start=date(2027, 1, 1)),)
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="prefix")
    revision = system.repo.read_book_reference(METER).input_revision
    prices = system.repo.read_prices(METER, START_TS, START_TS + 1800)
    from dao.prog.settlement.metering import merge_inputs
    inputs = merge_inputs(system.meter.items[1:2], tuple(row.spot for row in prices.items),
                          prices.snapshots, meter_id=METER)
    system.repo.register_inputs(METER, inputs, prices.snapshots, revision,
                                price_batch=prices, closed_before_ts=START_TS + 1800)
    state = system.service.state_at(START_TS + 1800)
    assert state.bonus.total_export_kwh == Decimal("2")
    assert system.service.state_at(START_TS + 2700).bonus.total_export_kwh is None


def test_late_revision_replays_dependent_tail(system, terms):
    from dao.prog.settlement.types import BonusRules
    bonus = BonusRules(True, "pv only", date(2027, 1, 1), "30", "0.2", "ex_vat", "0", "5000",
                       tuple((month, (0, 24)) for month in range(1, 13)))
    system.service.snapshots = (terms(bonus=bonus, supply_start=date(2027, 1, 1)),)
    system.meter.items[0] = replace(system.meter.items[0], export_kwh="4999")
    system.meter.items[1] = replace(system.meter.items[1], export_kwh="2")
    old = system.service.consolidate(START_TS, START_TS + 1800, publish=True, reason="first")
    assert old.lines[1].bonus_revenue == Decimal("0.03")
    system.meter.items[0] = replace(system.meter.items[0], export_kwh="5000")
    new = system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="late")
    tail = system.repo.read(METER, START_TS, START_TS + 1800)
    assert tail.lines[1].bonus_revenue == Decimal("0")
    assert tail.dirty_from_ts is None and new.generation == old.generation + 1
    assert system.repo.read(METER, START_TS, START_TS + 1800, old.generation).lines == old.lines


def test_prepare_planning_pins_registered_revision(system):
    context = system.service.prepare_planning(START_TS + 1800, START_TS + 7200)
    assert context.basis.input_revision == context.price_batch.input_revision > 0
    assert context.state.asof_ts == START_TS + 1800
    assert context.basis.source_watermark_ts == START_TS + 1800
    assert [q.import_rate for q in context.quotes] == [Decimal("0.10")] * 2 + [Decimal("0.40")] * 4
    assert {q.snapshot.snapshot_hash for q in context.quotes} == set(context.basis.snapshot_hashes)
    assert system.repo.read(METER, START_TS + 1800, START_TS + 7200).lines == ()
    assert system.repo.read_inputs(METER, START_TS + 1800, START_TS + 7200).items == ()


def test_correction_between_prepare_and_publish_blocks(system):
    context = system.service.prepare_planning(START_TS + 1800, START_TS + 7200)
    system.repo.register_prices(METER, [replace(system.spots[1], price_ex_vat="0.3")], context.basis.input_revision)
    with pytest.raises(FinancialPlanBlocked, match="revision"):
        system.publication.check_basis(context.basis, now_ts=system.clock[0])


def test_stale_input_revision_not_used_for_current_plan(system):
    first = system.service.prepare_planning(START_TS + 1800, START_TS + 7200)
    system.meter.items[0] = replace(system.meter.items[0], import_kwh="2")
    second = system.service.prepare_planning(START_TS + 1800, START_TS + 7200)
    assert second.basis.input_revision > first.basis.input_revision
    assert first.price_batch.input_revision == first.basis.input_revision
    with pytest.raises(FinancialPlanBlocked):
        system.publication.check_basis(first.basis, now_ts=system.clock[0])


@pytest.mark.parametrize("retention", ["empty", "grid_missing"])
def test_horizon_prices_never_push_closed_watermark_and_archived_repricing_recovers(system, retention):
    system.service.prepare_planning(START_TS + 1800, START_TS + 7200)
    # A surviving no-battery observation does not restore expired grid counters.
    system.meter.items = ([] if retention == "empty" else [replace(item, import_kwh=None, export_kwh=None,
                                                                 quality="incomplete") for item in system.meter.items])
    system.spots[0] = replace(system.spots[0], price_ex_vat="0.20")
    current = system.service.prepare_planning(START_TS + 1800, START_TS + 7200)
    closed = system.repo.read(METER, START_TS, START_TS + 1800)
    assert closed.net_cost == Decimal("0.40") and closed.freshness == "current"
    assert current.basis.source_watermark_ts == START_TS + 1800
    assert system.repo.read_inputs(METER, START_TS + 1800, START_TS + 7200).items == ()


def test_missing_horizon_price_blocks_not_zero(system):
    system.spots[:] = system.spots[:1]
    with pytest.raises(FinancialPlanBlocked, match="price"):
        system.service.prepare_planning(START_TS + 1800, START_TS + 7200)


def test_withdrawn_horizon_price_never_reuses_old_registered_value(system):
    system.service.prepare_planning(START_TS + 1800, START_TS + 7200)
    system.spots[:] = system.spots[:1]
    with pytest.raises(FinancialPlanBlocked, match="price"):
        system.service.prepare_planning(START_TS + 1800, START_TS + 7200)


def test_state_reader_does_not_clean_dirty_published_prefix(system):
    system.service.consolidate(START_TS, START_TS + 1800, publish=True, reason="first")
    revision = system.repo.read_book_reference(METER).input_revision
    system.repo.register_prices(METER, [replace(system.spots[0], price_ex_vat="0.20")], revision)
    from dao.prog.settlement.repository import ConcurrentSettlementError
    before = contents(system)
    with pytest.raises(ConcurrentSettlementError):
        system.service.state_at(START_TS + 1800)
    assert contents(system) == before


def test_configured_unconfirmed_regime_blocks_financial_plan(system):
    system.service.snapshots = (replace(system.snapshot, status="draft"),)
    with pytest.raises(FinancialPlanBlocked):
        system.service.prepare_planning(START_TS + 1800, START_TS + 7200)


def test_pricebatch_revision_conflict_aborts_context(system, monkeypatch):
    from dao.prog.settlement.repository import ConcurrentSettlementError
    original = system.repo.read_prices
    changed = []
    def read(*args, **kwargs):
        batch = original(*args, **kwargs)
        if not changed:
            changed.append(True)
            system.repo.register_prices(METER, [replace(system.spots[1], price_ex_vat="0.31")], batch.input_revision)
        return batch
    monkeypatch.setattr(system.repo, "read_prices", read)
    with pytest.raises(ConcurrentSettlementError):
        system.service.prepare_planning(START_TS + 1800, START_TS + 7200)


def test_all_application_registrations_share_guard(system, monkeypatch):
    import dao.prog.day_ahead as day_ahead
    from dao.prog.config.models.fastcontrol import FastControlConfig
    planner = day_ahead.DaCalc.__new__(day_ahead.DaCalc)
    planner.config = SimpleNamespace(prices=system.pricing, fast_control=FastControlConfig(), report=None)
    planner.db_da = SimpleNamespace(engine=system.engine)
    planner.db_ha = SimpleNamespace()
    monkeypatch.setattr(day_ahead, "FAST_PLAN_FILE", str(system.path))
    adapter = planner._financial_publication()
    service = planner.settlement_service()
    assert service.publication is adapter
    assert service.repository is planner.settlement_components()[0]
    service.repository.register_prices(METER, [system.spots[0]], 0)
    from dao.prog.fastctrl.publication import read_publication
    assert read_publication(system.path).status == "blocked"


def test_shadow_registration_invalidates_without_ha_writes(system):
    from dao.prog.da_base import DaBase
    app = DaBase.__new__(DaBase)
    app.config = SimpleNamespace(prices=pricing_config(mode="shadow"))
    app.settlement_service = lambda: system.service
    before = contents(system)[1][0]
    app.consolidate_settlement()
    from dao.prog.fastctrl.publication import read_publication
    assert read_publication(system.path).status == "blocked"
    assert contents(system)[1][0] == before


def test_off_scheduler_does_not_even_construct_service(system):
    from dao.prog.da_base import DaBase
    app = DaBase.__new__(DaBase)
    app.config = SimpleNamespace(prices=pricing_config(mode="off"))
    def forbidden():
        raise AssertionError("off must not collect or register")
    app.settlement_service = forbidden
    app.consolidate_settlement()
    assert not system.path.exists()


def test_context_revalidates_contract_aba_without_new_financial_generation(system):
    original = system.service.prepare_planning(START_TS + 1800, START_TS + 7200)
    before = system.repo.read(METER, START_TS, START_TS + 1800)
    from dao.prog.settlement.tariffs import make_snapshot
    changed_pricing = system.pricing.model_copy(update={"cost_supplier_consumption": {"2020-01-01": "0.2"}})
    changed = make_snapshot(changed_pricing, changed_pricing.settlement.contracts[0])
    revision = system.repo.register_inputs(METER, (), (changed,), original.basis.input_revision)
    system.repo.register_inputs(METER, (), (system.snapshot,), revision)
    current = system.service.prepare_planning(START_TS + 1800, START_TS + 7200)
    after = system.repo.read(METER, START_TS, START_TS + 1800)
    assert after.generation == before.generation and after.lines == before.lines
    assert after.input_revision == current.basis.input_revision == current.price_batch.input_revision
    assert after.input_revision > original.basis.input_revision and after.freshness == "current"
    with pytest.raises(FinancialPlanBlocked, match="revision"):
        system.publication.check_basis(original.basis, now_ts=system.clock[0])


def test_known_dirty_before_48h_is_replayed_from_archive(system):
    system.service.consolidate(START_TS, START_TS + 1800, publish=True, reason="archive")
    system.clock[0] = START_TS + 4 * 86400
    recent = system.clock[0] - 900
    system.meter.items.append(replace(system.meter.items[0], start_ts=recent, end_ts=recent + 900))
    system.spots.append(replace(system.spots[0], start_ts=recent // 3600 * 3600, end_ts=system.clock[0]))
    system.service.consolidate(recent, recent + 900, publish=True, reason="recent")
    revision = system.repo.read_book_reference(METER).input_revision
    system.repo.register_prices(METER, (replace(system.spots[0], price_ex_vat="0.20"),), revision)
    system.meter.items = []
    calls = len(system.meter.calls)
    start, end = system.service.scheduled_range()
    assert start > START_TS + 1800
    system.service.consolidate(start, end, publish=True, reason="scheduled")
    closed = system.repo.read(METER, START_TS, START_TS + 1800)
    assert closed.net_cost == Decimal("0.40") and closed.dirty_from_ts is None
    assert all(left >= start for left, _ in system.meter.calls[calls:])


def test_archived_repricing_does_not_upgrade_missing_measurement_quality(system):
    system.meter.items[0] = replace(system.meter.items[0], quality="incomplete")
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="incomplete")
    system.meter.items = []
    system.spots[0] = replace(system.spots[0], price_ex_vat="0.20")
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="retention")
    assert system.repo.read_inputs(METER, START_TS, START_TS + 900).items[0].quality == "incomplete"


def test_planning_before_bonus_start_has_no_invalid_future_prefix(system, terms):
    from dao.prog.settlement.types import BonusRules
    bonus = BonusRules(True, "pv only", date(2027, 2, 1), "30", "0.2", "ex_vat", "0", "5000",
                       tuple((month, (0, 24)) for month in range(1, 13)))
    system.service.snapshots = (terms(bonus=bonus, supply_start=date(2027, 1, 1)),)
    state = system.service.state_at(START_TS + 900)
    assert state.bonus is None and state.asof_ts == START_TS + 900


@pytest.mark.parametrize("missing", ["import_kwh", "export_kwh"])
def test_single_recorder_component_retention_preserves_archive_and_applies_updates(system, missing):
    original = replace(system.meter.items[0], import_kwh="2", export_kwh="3", quality="estimated", sources=(
        "ha:statistics_short_term:sensor.import:kWh:300s:bucket_end_estimate",
        "ha:statistics_short_term:sensor.export:kWh:300s:bucket_end_estimate"))
    system.meter.items = [original]
    before = system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="archive-components")
    other = "export_kwh" if missing == "import_kwh" else "import_kwh"
    system.meter.items = [replace(original, **{missing: None, other: "4"}, quality="incomplete",
                                  sources=(*original.sources, "issue:missing-or-invalid-buckets:sensor.expired"))]
    view = system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="single-retention")
    item = system.repo.read_inputs(system.meter.meter_id, START_TS, START_TS + 900).items[0]
    assert getattr(item, missing) == getattr(original, missing)
    assert getattr(item, other) == Decimal("4")
    assert item.quality == view.lines[0].quality == "incomplete"
    assert view.lines[0].import_cost == (Decimal("0.2") if missing == "import_kwh" else Decimal("0.4"))
    assert view.lines[0].base_export_revenue == (Decimal("0.4") if missing == "import_kwh" else Decimal("0.3"))
    assert system.repo.read(system.meter.meter_id, START_TS, START_TS + 900, before.generation).lines == before.lines


@pytest.mark.parametrize("withdrawn", ["import", "export", "both"])
def test_explicit_csv_correction_can_withdraw_archived_grid_components(system, tmp_path, withdrawn):
    from dao.prog.settlement.metering import read_meter_csv

    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="before-withdrawal")
    imported = "" if withdrawn in ("import", "both") else "4"
    exported = "" if withdrawn in ("export", "both") else "5"
    path = tmp_path / "corrected.csv"
    path.write_text("start_ts,end_ts,import_kwh,export_kwh,battery_discharge_kwh,unit,quality,source\n"
                    f"{START_TS},{START_TS + 900},{imported},{exported},0,kWh,incomplete,explicit-correction\n")
    system.meter.items = list(read_meter_csv(path, system.meter.meter_id))
    view = system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="explicit-withdrawal")
    item = system.repo.read_inputs(system.meter.meter_id, START_TS, START_TS + 900).items[0]
    assert item.import_kwh == (None if not imported else Decimal("4"))
    assert item.export_kwh == (None if not exported else Decimal("5"))
    assert view.freshness == "current" and item.quality == "incomplete"
