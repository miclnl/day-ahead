"""Local file/SQLite barriers prove fail-closed publication, not physical stopping."""

import importlib
import json
import multiprocessing
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.pool import NullPool

from dao.prog.config.models.pricing import PricingConfig
from dao.prog.fastctrl.plan import FastPlan, load_plan, write_plan
from dao.prog.settlement.engine import settle_series
from dao.prog.settlement.tariffs import _canonical_default, make_snapshot, quote
from dao.prog.settlement.types import (
    InputInterval,
    PriceBatch,
    RegisteredSpot,
    SettlementState,
    SpotInterval,
)
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_fast_contract import ready_plan

NOW = START_TS + 900
METER = "synthetic-grid"
CTX = multiprocessing.get_context("fork")


def pricing_config(*, mode="active", status="confirmed", start="2027-01-01", end=None):
    return PricingConfig.model_validate({
        **{name: {"2020-01-01": value} for name, value in (
            ("energy_taxes_consumption", "0"), ("energy_taxes_production", "0"),
            ("cost_supplier_consumption", "0"), ("cost_supplier_production", "0"),
            ("vat_consumption", "0"), ("vat_production", "0"),
            ("multiplier_consumption", "1"), ("multiplier_production", "1"))},
        "last_invoice": "2026-01-01", "settlement": {
            "mode": mode, "meter_id": METER, "contracts": [{
                "id": "synthetic-2027", "status": status, "valid_from": start,
                "valid_to": end, "supply_start": "2027-01-01", "timezone": "Europe/Amsterdam",
                "product": "budget_dynamic_2027", "regime": "no_net_metering",
                "price_interval_seconds": 3600, "meter_interval_seconds": 900,
                "hour_price_rule": "native", "monthly_export_floor": "none",
            }],
        },
    })


def item(at=START_TS, *, amount="1", spot="0.1", source="known-hour"):
    return InputInterval(at, at + 900, Decimal(amount), Decimal("0"), Decimal("0"),
                         "verified_pv", "measured", ("synthetic",),
                         None if spot is None else Decimal(spot), source, "synthetic-hash", METER, 0)


def connect_publication(url, path, pricing):
    from dao.prog.settlement.publication import make_publication

    engine = create_engine(url, poolclass=NullPool)
    repo, adapter = make_publication(engine, plan_path=Path(path), pricing=pricing, max_plan_age=600)
    return engine, repo, adapter


def replay_for_plan(plan, snapshots):
    """Explicit synthetic evidence; never reconstruct production evidence from a plan."""
    from dao.prog.settlement.replay import ReplayBasis

    basis = plan.financial_context.basis
    quotes = []
    for interval in plan.intervals:
        start = interval.start_ts
        while start < interval.end_ts:
            snapshot = next(s for s in snapshots if s.valid_from_ts <= start
                            and (s.valid_to_ts is None or start < s.valid_to_ts))
            end = min(interval.end_ts, snapshot.valid_to_ts or interval.end_ts)
            quotes.append(quote(snapshot, start, end, Decimal("0.1")))
            start = end
    quotes = tuple(quotes)
    spots = tuple(RegisteredSpot(SpotInterval(i.start_ts, i.end_ts, Decimal("0.1"),
                        "synthetic-replay", "estimated"), basis.input_revision, "c" * 64)
                  for i in plan.intervals)
    prices = PriceBatch(basis.meter_id, basis.input_revision, basis.planning_start_ts,
                       basis.horizon_end_ts, spots, snapshots, basis.captured_ts, basis.source_watermark_ts)
    return ReplayBasis(basis, SettlementState(basis.planning_start_ts, None, None),
                       quotes, prices, plan.financial_context.terminal_value)


def publish_fixture(adapter, plan, snapshots):
    replay = replay_for_plan(plan, snapshots)
    plan = replace(plan, financial_context=replace(plan.financial_context, replay_basis_hash=replay.content_hash))
    adapter.publish(plan, now_ts=plan.created_ts, replay_basis=replay)
    return plan


@pytest.fixture
def local():
    return importlib.import_module("dao.prog.fastctrl.publication")


@pytest.fixture
def system(tmp_path, monkeypatch):
    publication = importlib.import_module("dao.prog.settlement.publication")
    monkeypatch.setattr(publication, "time", lambda: NOW)
    pricing = pricing_config()
    path = tmp_path / "fast_plan.json"
    url = f"sqlite:///{tmp_path / 'book.db'}"
    engine, repo, adapter = connect_publication(url, path, pricing)
    repo.ensure_schema()
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    repo.register_inputs(METER, [item()], [snapshot], 0)
    batch = repo.read_inputs(METER, START_TS, NOW)
    result = settle_series(batch.items, batch.snapshots, SettlementState(START_TS, None, None))
    repo.commit(METER, result, batch, 0, "synthetic-clean-prefix")
    plan = ready_plan(hashes=(snapshot.snapshot_hash,))
    plan = publish_fixture(adapter, plan, (snapshot,))
    yield SimpleNamespace(repo=repo, adapter=adapter, plan=plan, path=path, snapshot=snapshot,
                          pricing=pricing, engine=engine, url=url)
    engine.dispose()


def reference_rows(repo):
    from dao.prog.settlement.repository import _BOOK, _INPUT, _TARIFF

    with repo.engine.connect() as connection:
        return [connection.execute(select(table)).all() for table in (_BOOK, _INPUT, _TARIFF)]


@pytest.mark.parametrize("mode,status", [("off", "draft"), ("shadow", "confirmed"), ("active", "confirmed")])
def test_policy_uses_explicit_local_contract_date_not_mode(local, mode, status):
    pricing = pricing_config(mode=mode, status=status, start="2027-07-01")
    assert local.gate_policy(pricing).required_from_ts == 1814392800  # Amsterdam summer midnight.
    assert local.gate_policy(None).required_from_ts is None
    assert local.gate_policy(PricingConfig.model_validate({
        **pricing.model_dump(), "settlement": None,
    })).required_from_ts is None


def test_ready_roundtrip_never_opens_unproven_actuation_gates(system, local):
    read = local.read_publication(system.path)
    assert read.status == "ready" and read.plan.financial_context == system.plan.financial_context
    verdict = local.following_verdict(read, policy=system.adapter.policy(), now_ts=NOW)
    assert not verdict.allowed and verdict.reason == "actuation_gates_unproven"
    # Ook zonder geconfigureerde gate geeft v2 geen legacyrechten.
    assert not local.following_verdict(read, policy=local.GatePolicy(None), now_ts=NOW).allowed
    system.adapter.block(now_ts=NOW, reason="fallback_requested")
    read = local.read_publication(system.path)
    assert (read.status, read.plan, read.reason) == ("blocked", None, "fallback_requested")
    assert load_plan(system.path) is None
    assert set(json.loads(system.path.read_text())) == {"format_version", "status", "created_ts", "reason"}


def test_legacy_rights_end_exactly_at_the_configured_boundary(tmp_path, local):
    path = tmp_path / "fast_plan.json"
    legacy = FastPlan(START_TS - 1, 900, intervals=system_intervals(START_TS - 900))
    write_plan(legacy, path)
    read = local.read_publication(path)
    assert read.status == "ready"
    policy = local.GatePolicy(START_TS)
    assert local.following_verdict(read, policy=policy, now_ts=START_TS - 1).allowed
    assert not local.following_verdict(read, policy=policy, now_ts=START_TS).allowed


def system_intervals(start):
    from dao.prog.fastctrl.plan import PlanInterval

    return [PlanInterval(start, start + 900, 0.1, 0.1), PlanInterval(start + 900, start + 1800, 0.1, 0.1)]


@pytest.mark.parametrize("content", ["null", "[]", "NaN", "Infinity", '{"format_version":99}',
                                       '{"format_version":true}', '{"format_version":1,"created_ts":NaN}',
                                       '{"format_version":2,"status":"ready"}',
                                       '{"format_version":2,"status":"blocked","created_ts":1,"reason":"x","intervals":[]}'])
def test_invalid_artifact_is_not_legacy(tmp_path, local, content):
    path = tmp_path / "fast_plan.json"
    path.write_text(content)
    read = local.read_publication(path)
    assert read.status == "invalid" and read.plan is None
    assert not local.following_verdict(read, policy=local.GatePolicy(None), now_ts=NOW).allowed
    assert load_plan(path) is None


def test_missing_and_same_mtime_replacement_are_read_fresh(tmp_path, system, local):
    assert local.read_publication(tmp_path / "missing").status == "missing"
    old_stat = system.path.stat()
    assert local.read_publication(system.path).status == "ready"
    other = system.path.with_name("replacement")
    other.write_text("null")
    os.utime(other, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    os.replace(other, system.path)
    assert local.read_publication(system.path).status == "invalid"


def test_lease_is_nonconstructible_nonreentrant_and_expires(tmp_path, local):
    path = tmp_path / "fast_plan.json"
    with pytest.raises(TypeError):
        local.PublicationLease()
    with local.publication_lock(path) as lease:
        with pytest.raises(local.PublicationLockBusy):
            with local.publication_lock(path):
                pass
        local.invalidate_plan(lease=lease, now_ts=NOW, reason="test")
        with pytest.raises((AttributeError, TypeError)):
            lease.path = tmp_path / "other"
    with pytest.raises(local.PublicationError):
        local.invalidate_plan(lease=lease, now_ts=NOW, reason="expired")
    assert local.read_publication(path).reason == "test"
    assert Path(str(path) + ".lock").exists()


def test_threads_cannot_share_a_lease_or_acquire_the_same_path(tmp_path, local):
    path = tmp_path / "fast_plan.json"
    barrier = Barrier(2)
    with local.publication_lock(path) as lease:
        def contender():
            barrier.wait(timeout=10)
            with pytest.raises(local.PublicationError):
                local.invalidate_plan(lease=lease, now_ts=NOW, reason="wrong-thread")
            with pytest.raises(local.PublicationLockBusy):
                with local.publication_lock(path):
                    pass
            return "denied"

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(contender)
            barrier.wait(timeout=10)
            assert future.result(timeout=10) == "denied"
    assert not path.exists()


def test_canonical_parent_aliases_share_one_lock_and_leaf_symlinks_are_refused(tmp_path, local):
    directory = tmp_path / "data"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    path = directory / "fast_plan.json"
    assert local.canonical_plan_path(alias / path.name) == path
    with local.publication_lock(path):
        with pytest.raises(local.PublicationLockBusy):
            with local.publication_lock(alias / path.name):
                pass
    path.symlink_to(tmp_path / "target")
    assert local.read_publication(path).status == "invalid"
    with pytest.raises(local.PublicationError):
        with local.publication_lock(path):
            pass
    path.unlink()
    lockpath = Path(str(path) + ".lock")
    lockpath.unlink()  # Testfixture alleen; productcode verwijdert dit bestand nooit.
    lockpath.symlink_to(tmp_path / "target-lock")
    with pytest.raises(local.PublicationError):
        with local.publication_lock(path):
            pass


def test_v1_and_raw_local_writer_cannot_restore_a_financial_artifact(system, local):
    legacy = FastPlan(NOW, 900, intervals=system_intervals(NOW))
    for expected in ("ready", "blocked"):
        if expected == "blocked":
            system.adapter.block(now_ts=NOW, reason="stay-blocked")
        before = system.path.read_bytes()
        with pytest.raises(local.PublicationError):
            write_plan(legacy, system.path)
        with local.publication_lock(system.path) as lease:
            with pytest.raises(local.PublicationError):
                local.write_publication(system.plan, lease=lease)
        assert system.path.read_bytes() == before


def test_noop_and_new_meter_tail_preserve_issued_plan_but_reject_old_candidate(system, local):
    from dao.prog.settlement.publication import FinancialPlanBlocked

    before = system.path.read_bytes()
    assert system.repo.register_inputs(METER, [item()], [system.snapshot], 1) == 1
    assert system.path.read_bytes() == before
    assert system.repo.register_inputs(METER, [item(NOW)], [system.snapshot], 1) == 2
    assert system.path.read_bytes() == before
    with pytest.raises(FinancialPlanBlocked):
        system.adapter.publish(system.plan, now_ts=NOW)
    assert local.read_publication(system.path).status == "blocked"


@pytest.mark.parametrize("change", ["prefix", "spot", "spot-source", "new-price", "contract", "unknown"])
def test_relevant_changes_are_durably_blocked_before_registration(system, local, change):
    inputs, snapshots = [item()], [system.snapshot]
    if change == "prefix":
        inputs = [item(amount="2")]
    elif change == "spot":
        inputs = [item(NOW, spot="0.2")]
    elif change == "spot-source":
        inputs = [item(NOW, source="corrected-hour-source")]
    elif change == "new-price":
        # Geen eerder gepinde prijs voor deze tweede horizon-uurdekking.
        plan = system.plan
        context = replace(plan.financial_context, basis=replace(plan.financial_context.basis,
                                                               horizon_end_ts=START_TS + 7200))
        from dao.prog.fastctrl.plan import BatteryPlanStep, PlanInterval
        plan = replace(plan, financial_context=context, intervals=[*plan.intervals, *[
            PlanInterval(at, at + 900, 0.1, 0.1, batteries=[BatteryPlanStep(0, 50, 50)])
            for at in range(START_TS + 3600, START_TS + 7200, 900)]])
        publish_fixture(system.adapter, plan, (system.snapshot,))
        inputs = [item(START_TS + 3600)]
    elif change == "contract":
        pricing = system.pricing.model_copy(deep=True)
        pricing.cost_supplier_consumption = {"2020-01-01": 0.2}
        snapshots = [make_snapshot(pricing, pricing.settlement.contracts[0])]
    else:
        system.path.write_text("not-json")
        inputs = [item(NOW)]
    observations = []
    def before_insert(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO settlement_input") or statement.startswith("INSERT INTO settlement_tariff"):
            observations.append(local.read_publication(system.path).status)
    event.listen(system.engine, "before_cursor_execute", before_insert)
    try:
        assert system.repo.register_inputs(METER, inputs, snapshots, 1) == 2
    finally:
        event.remove(system.engine, "before_cursor_execute", before_insert)
    assert observations and set(observations) == {"blocked"}
    assert local.read_publication(system.path).status == "blocked"
    assert system.repo.read_book_reference(METER).input_revision == 2


def test_aba_and_explicit_revalidation_use_book_revision_not_immutable_line_revision(system, local):
    from dao.prog.settlement.publication import FinancialPlanBlocked

    changed = system.pricing.model_copy(deep=True)
    changed.cost_supplier_consumption = {"2020-01-01": 0.2}
    other = make_snapshot(changed, changed.settlement.contracts[0])
    system.repo.register_inputs(METER, [item()], [other], 1)
    system.repo.register_inputs(METER, [item()], [system.snapshot], 2)
    with pytest.raises(FinancialPlanBlocked):
        system.adapter.check_basis(system.plan.financial_context.basis, now_ts=NOW)
    batch = system.repo.read_inputs(METER, START_TS, NOW)
    result = settle_series(batch.items, batch.snapshots, SettlementState(START_TS, None, None))
    assert system.repo.commit(METER, result, batch, 1, "ABA-revalidation") == 1
    basis = replace(system.plan.financial_context.basis, input_revision=3)
    system.adapter.check_basis(basis, now_ts=NOW)
    publish_fixture(system.adapter, replace(system.plan, financial_context=replace(system.plan.financial_context, basis=basis)),
                    (system.snapshot,))
    assert local.read_publication(system.path).plan.financial_context.basis.input_revision == 3


@pytest.mark.parametrize("fault", ["file", "directory"])
def test_invalidation_fsync_failure_cannot_commit_input_or_tariff(system, local, monkeypatch, fault):
    before = reference_rows(system.repo)
    original = os.fsync
    def broken(fd):
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if directory == (fault == "directory"):
            raise OSError("injected-fsync-failure")
        return original(fd)
    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(local.PublicationError, match="fsync"):
        system.repo.register_inputs(METER, [item(amount="2")], [system.snapshot], 1)
    assert reference_rows(system.repo) == before
    assert not list(system.path.parent.glob(".fast_plan_*.tmp"))
    assert local.read_publication(system.path).status == ("ready" if fault == "file" else "blocked")


def test_blocked_retry_requires_successful_directory_fsync_before_registration(system, local, monkeypatch):
    before = reference_rows(system.repo)
    pricing = system.pricing.model_copy(deep=True)
    pricing.cost_supplier_consumption = {"2020-01-01": 0.2}
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    original = os.fsync

    def broken(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected-retry-directory-fsync-failure")
        return original(fd)

    monkeypatch.setattr(os, "fsync", broken)
    for _ in range(3):
        with pytest.raises(local.PublicationError, match="fsync"):
            system.repo.register_inputs(METER, [item(amount="2")], [snapshot], 1)
        # Replace is al zichtbaar, maar geen enkele mislukte poging mag DB-inhoud publiceren.
        assert local.read_publication(system.path).status == "blocked"
        assert system.repo.read_book_reference(METER).input_revision == 1
        assert reference_rows(system.repo) == before
        assert not list(system.path.parent.glob(".fast_plan_*.tmp"))

    durable = []
    def restored(fd):
        original(fd)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            durable.append(True)

    def require_durable_before_insert(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith(("INSERT INTO settlement_input", "INSERT INTO settlement_tariff",
                                 "UPDATE settlement_book SET input_revision=?, snapshot_hashes_json=")):
            assert durable == [True]

    monkeypatch.setattr(os, "fsync", restored)
    event.listen(system.engine, "before_cursor_execute", require_durable_before_insert)
    try:
        assert system.repo.register_inputs(METER, [item(amount="2")], [snapshot], 1) == 2
    finally:
        event.remove(system.engine, "before_cursor_execute", require_durable_before_insert)
    assert durable == [True]
    reference = system.repo.read_book_reference(METER)
    assert reference.input_revision == 2 and reference.snapshot_hashes == (snapshot.snapshot_hash,)
    assert system.repo.read_inputs(METER, START_TS, NOW).items[0].import_kwh == Decimal("2")
    assert local.read_publication(system.path).status == "blocked"


@pytest.mark.parametrize("fault", ["revision", "hashes", "meter", "future-capture", "dirty"])
def test_basis_rejections_replace_old_ready_with_blocked(system, local, fault):
    from dao.prog.settlement.publication import FinancialPlanBlocked

    basis = system.plan.financial_context.basis
    if fault == "revision":
        basis = replace(basis, input_revision=2)
    elif fault == "hashes":
        basis = replace(basis, snapshot_hashes=("f" * 64,))
    elif fault == "meter":
        basis = replace(basis, meter_id="another-meter")
    elif fault == "future-capture":
        basis = replace(basis, captured_ts=NOW + 1)
    else:
        system.repo.register_inputs(METER, [item(amount="2")], [system.snapshot], 1)
        basis = replace(basis, input_revision=2)
    with pytest.raises(FinancialPlanBlocked):
        system.adapter.check_basis(basis, now_ts=NOW)
    assert local.read_publication(system.path).status == "blocked"


@pytest.mark.parametrize("fault", ["expired", "future-created", "too-long", "legacy", "rate"])
def test_publish_revalidates_payload_and_lease(system, local, fault):
    from dao.prog.settlement.publication import FinancialPlanBlocked

    system.adapter.check_basis(system.plan.financial_context.basis, now_ts=NOW)
    plan = system.plan
    now = NOW
    if fault == "expired":
        now += 600
    elif fault == "future-created":
        plan = replace(plan, created_ts=NOW + 1)
    elif fault == "too-long":
        plan = replace(plan, financial_context=replace(plan.financial_context, not_after_ts=NOW + 601))
    elif fault == "legacy":
        plan = FastPlan(NOW, 900, intervals=system_intervals(NOW))
    else:
        plan = replace(plan, intervals=[replace(plan.intervals[0], price_export=float("nan")), *plan.intervals[1:]])
    with pytest.raises(FinancialPlanBlocked):
        system.adapter.publish(plan, now_ts=now)
    assert local.read_publication(system.path).status == "blocked"


def test_limited_publication_preserves_unknown_economics_without_execution_rights(system, local):
    context = replace(system.plan.financial_context, economics_supported=False,
                      issues=("bonus_prefix_unknown", "month_prefix_unknown"))
    system.adapter.publish(replace(system.plan, financial_context=context), now_ts=NOW)
    read = local.read_publication(system.path)
    assert read.status == "ready", read.reason
    assert read.plan.financial_context.economics_supported is False
    assert read.plan.financial_context.issues == ("bonus_prefix_unknown", "month_prefix_unknown")
    verdict = local.following_verdict(read, policy=system.adapter.policy(), now_ts=NOW)
    assert not verdict.allowed and verdict.reason == "actuation_gates_unproven"


@pytest.mark.parametrize("mode,status", [("off", "confirmed"), ("shadow", "confirmed"), ("off", "draft")])
def test_inactive_or_draft_config_cannot_publish(system, local, mode, status):
    from dao.prog.settlement.publication import FinancialPlanBlocked, SettlementPublication

    adapter = SettlementPublication(system.repo, plan_path=system.path,
                                    pricing=pricing_config(mode=mode, status=status), max_plan_age=600)
    with pytest.raises(FinancialPlanBlocked):
        adapter.check_basis(system.plan.financial_context.basis, now_ts=NOW)
    assert local.read_publication(system.path).status == "blocked"


def test_guard_requires_its_own_live_hold(system, local):
    from dao.prog.settlement.repository import InputChange

    guard = system.adapter.registration_guard()
    change = InputChange(METER, 1, 2, ((START_TS, NOW),), (), ())
    with pytest.raises(local.PublicationError):
        guard.before_revision(change)
    with guard.hold():
        guard.before_revision(change)
    assert local.read_publication(system.path).status == "blocked"


@pytest.mark.parametrize("future", [False, True])
def test_v1_registration_invalidation_depends_on_gate_date_not_mode(system, local, future):
    from dao.prog.settlement.publication import make_publication

    system.path.unlink()  # Alleen testsetup simuleert een oude deployment.
    legacy = FastPlan(NOW, 900, intervals=system_intervals(NOW))
    write_plan(legacy, system.path)
    pricing = pricing_config(mode="shadow", start="2027-07-01" if future else "2027-01-01")
    writer, _ = make_publication(system.engine, plan_path=system.path, pricing=pricing, max_plan_age=600)
    writer.register_inputs(METER, [item(amount="2")], [system.snapshot], 1)
    assert local.read_publication(system.path).status == ("ready" if future else "blocked")


@pytest.mark.parametrize("phase", ["before-invalidation", "after-invalidation", "before-commit"])
def test_registration_exception_rolls_back_without_restoring_ready(system, local, monkeypatch, phase):
    import dao.prog.settlement.publication as publication

    before = reference_rows(system.repo)
    if phase in ("before-invalidation", "after-invalidation"):
        original = publication.invalidate_plan
        def broken(**kwargs):
            if phase == "before-invalidation":
                raise RuntimeError("injected-before-invalidation")
            original(**kwargs)
            raise RuntimeError("injected-after-invalidation")
        monkeypatch.setattr(publication, "invalidate_plan", broken)
    else:
        def broken(connection, cursor, statement, parameters, context, executemany):
            if statement.startswith("UPDATE settlement_book SET input_revision=?, snapshot_hashes_json="):
                raise RuntimeError("injected-before-commit")
        event.listen(system.engine, "after_cursor_execute", broken)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            system.repo.register_inputs(METER, [item(amount="2")], [system.snapshot], 1)
    finally:
        if phase == "before-commit":
            event.remove(system.engine, "after_cursor_execute", broken)
    assert reference_rows(system.repo) == before
    assert local.read_publication(system.path).status == ("ready" if phase == "before-invalidation" else "blocked")


def test_empty_used_prefix_does_not_treat_a_meter_overlap_as_price_change(system, local):
    plan = replace(system.plan, financial_context=replace(system.plan.financial_context, basis=replace(
        system.plan.financial_context.basis, prefix_start_ts=NOW,
    )))
    publish_fixture(system.adapter, plan, (system.snapshot,))
    before = system.path.read_bytes()
    # Geen verbruikte prefix; de reeds gepinde uurprijs en contractselectie blijven gelijk.
    system.repo.register_inputs(METER, [replace(item(), end_ts=NOW + 450)], [system.snapshot], 1)
    assert system.path.read_bytes() == before
    assert local.read_publication(system.path).status == "ready"


def test_reference_read_error_cannot_leave_an_old_plan_valid(system, local):
    from dao.prog.settlement.publication import FinancialPlanBlocked

    def broken(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("SELECT settlement_book.input_revision,"):
            raise RuntimeError("injected-reference-read-failure")
    event.listen(system.engine, "before_cursor_execute", broken)
    try:
        with pytest.raises(FinancialPlanBlocked, match="reference-read"):
            system.adapter.publish(system.plan, now_ts=NOW)
    finally:
        event.remove(system.engine, "before_cursor_execute", broken)
    assert local.read_publication(system.path).status == "blocked"


def test_no_lease_renewal_at_expiry_or_when_clock_precedes_creation(system, local):
    read = local.read_publication(system.path)
    before = system.path.read_bytes()
    for now, reason in ((NOW + 600, "plan_lease_expired"), (NOW - 1, "clock_before_creation")):
        verdict = local.following_verdict(read, policy=system.adapter.policy(), now_ts=now)
        assert not verdict.allowed and verdict.reason == reason
        assert local.read_publication(system.path).plan.financial_context.not_after_ts == NOW + 600
    assert system.path.read_bytes() == before


def test_contract_horizon_gap_and_price_resolution_mismatch_are_blocked(system, local):
    from dao.prog.settlement.publication import FinancialPlanBlocked

    plan = system.plan
    context = replace(plan.financial_context, price_interval_s=900)
    with pytest.raises(FinancialPlanBlocked, match="price_interval"):
        system.adapter.publish(replace(plan, financial_context=context), now_ts=NOW)
    basis = replace(plan.financial_context.basis, prefix_start_ts=START_TS - 1, planning_start_ts=START_TS - 1)
    with pytest.raises(FinancialPlanBlocked, match="contract_horizon"):
        system.adapter.check_basis(basis, now_ts=NOW)
    assert local.read_publication(system.path).status == "blocked"


def test_two_threads_register_under_the_same_file_before_sqlite_lock(system, local):
    from dao.prog.settlement.publication import make_publication
    from dao.prog.settlement.repository import ConcurrentSettlementError

    barrier = Barrier(2)
    def writer(at, amount):
        repo, _ = make_publication(system.engine, plan_path=system.path, pricing=system.pricing, max_plan_age=600)
        barrier.wait(timeout=10)
        try:
            return repo.register_inputs(METER, [item(at, amount=amount)], [system.snapshot], 1)
        except (local.PublicationLockBusy, ConcurrentSettlementError):
            return "retry"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(writer, START_TS, "2"), executor.submit(writer, NOW, "1")]
        results = [future.result(timeout=15) for future in futures]
    assert sorted(results, key=str) == [2, "retry"]
    revision = system.repo.read_book_reference(METER).input_revision
    assert revision == 2
    if results[0] == "retry":
        system.repo.register_inputs(METER, [item(amount="2")], [system.snapshot], revision)
    else:
        system.repo.register_inputs(METER, [item(NOW)], [system.snapshot], revision)
    batch = system.repo.read_inputs(METER, START_TS, NOW + 900)
    assert batch.input_revision == 3
    assert [(row.start_ts, row.import_kwh) for row in batch.items] == [(START_TS, Decimal("2")), (NOW, Decimal("1"))]
    assert local.read_publication(system.path).status == "blocked"


def test_factory_construction_has_no_database_or_file_io(tmp_path):
    from dao.prog.settlement.publication import make_publication

    path = tmp_path / "not-created" / "fast_plan.json"
    engine = create_engine("sqlite://", poolclass=NullPool)
    @event.listens_for(engine, "connect")
    def forbidden(*args):
        raise AssertionError("factory attempted a database connection")
    repo, adapter = make_publication(engine, plan_path=path, pricing=pricing_config(), max_plan_age=600)
    assert repo.engine is engine and adapter.policy().required_from_ts == START_TS
    assert not path.parent.exists()


def test_adapter_freezes_the_bootstrap_path_before_cwd_changes(system, local, monkeypatch):
    from dao.prog.settlement.publication import SettlementPublication

    monkeypatch.chdir(system.path.parent)
    adapter = SettlementPublication(system.repo, plan_path=Path("fast_plan.json"),
                                    pricing=system.pricing, max_plan_age=600)
    other = system.path.parent / "other-cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    adapter.block(now_ts=NOW, reason="same-bootstrap-path")
    assert local.read_publication(system.path).status == "blocked"
    assert not (other / "fast_plan.json").exists()
    assert not (other / "fast_plan.json.lock").exists()


def test_pinned_source_versions_are_not_replaced_by_config_generated_hashes(system, local):
    from dao.prog.fastctrl.plan import FinancialBasis

    # Bronversies zijn onderdeel van de A2-hash, niet van options.json.
    selected = replace(system.snapshot, source_versions=(("synthetic", "corrected-v2"),))
    content = asdict(selected)
    del content["snapshot_hash"]
    selected = replace(selected, snapshot_hash=sha256(json.dumps(
        content, default=_canonical_default, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest())
    system.repo.register_inputs(METER, [item()], [selected], 1)
    batch = system.repo.read_inputs(METER, START_TS, NOW)
    result = settle_series(batch.items, batch.snapshots, SettlementState(START_TS, None, None))
    system.repo.commit(METER, result, batch, 1, "explicit-source-version-revalidation")
    basis = FinancialBasis(METER, 2, (selected.snapshot_hash,), START_TS, NOW, START_TS + 3600, NOW, NOW)
    plan = replace(system.plan, financial_context=replace(system.plan.financial_context, basis=basis))
    publish_fixture(system.adapter, plan, (selected,))
    assert local.read_publication(system.path).plan.financial_context.basis.snapshot_hashes == (selected.snapshot_hash,)


def test_next_contract_boundary_caps_publication_lease(system, local):
    from dao.prog.fastctrl.plan import BatteryPlanStep, FinancialBasis, PlanInterval
    from dao.prog.settlement.publication import FinancialPlanBlocked, SettlementPublication

    pricing = pricing_config(end="2027-01-02")
    second = {**pricing.settlement.contracts[0].model_dump(), "valid_from": "2027-01-02", "valid_to": None}
    pricing = PricingConfig.model_validate({**pricing.model_dump(), "settlement": {
        **pricing.settlement.model_dump(), "contracts": [*pricing.settlement.contracts, second],
    }})
    snapshots = tuple(make_snapshot(pricing, contract) for contract in pricing.settlement.contracts)
    system.repo.register_inputs(METER, [item()], snapshots, 1)
    batch = system.repo.read_inputs(METER, START_TS, NOW)
    result = settle_series(batch.items, batch.snapshots, SettlementState(START_TS, None, None))
    system.repo.commit(METER, result, batch, 1, "boundary-contracts-revalidation")
    adapter = SettlementPublication(system.repo, plan_path=system.path, pricing=pricing, max_plan_age=600)
    boundary = START_TS + 86400
    created = boundary - 300
    basis = FinancialBasis(METER, 2, tuple(snapshot.snapshot_hash for snapshot in snapshots),
                           START_TS, created, created + 1800, created, NOW)
    context = replace(system.plan.financial_context, basis=basis, not_after_ts=created + 600)
    plan = replace(system.plan, created_ts=created, financial_context=context, intervals=[
        PlanInterval(at, at + 900, 0.1, 0.1, batteries=[BatteryPlanStep(0, 50, 50)])
        for at in (created, created + 900)
    ])
    adapter.check_basis(basis, now_ts=created)  # Horizon is wel gedekt door beide confirmed-contracten.
    with pytest.raises(FinancialPlanBlocked, match="lease"):
        adapter.publish(plan, now_ts=created)
    publish_fixture(adapter, replace(plan, financial_context=replace(context, not_after_ts=boundary)), snapshots)
    read = local.read_publication(system.path)
    assert read.plan.financial_context.not_after_ts == boundary
    assert not local.following_verdict(read, policy=adapter.policy(), now_ts=boundary).allowed


def join(process):
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("barrier worker did not finish")


def ordering_worker(url, path, pricing, candidate, barrier, first_done, role, result):
    from dao.prog.settlement.publication import FinancialPlanBlocked

    engine, repo, adapter = connect_publication(url, path, pricing)
    barrier.wait(timeout=10)
    if role.endswith("second"):
        assert first_done.wait(timeout=10)
    try:
        if role.startswith("correct"):
            snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
            outcome = repo.register_inputs(METER, [item(amount="2")], [snapshot], 1)
        else:
            try:
                adapter.publish(candidate, now_ts=NOW)
                outcome = "published"
            except FinancialPlanBlocked:
                outcome = "blocked"
        result.put((role, outcome))
    finally:
        if role.endswith("first"):
            first_done.set()
        engine.dispose()


@pytest.mark.parametrize("first", ["correct", "publish"])
def test_two_process_publication_orders_have_no_stale_ready(system, local, first):
    barrier, done, results = CTX.Barrier(2), CTX.Event(), CTX.Queue()
    roles = [first + "-first", ("publish" if first == "correct" else "correct") + "-second"]
    processes = [CTX.Process(target=ordering_worker, args=(system.url, str(system.path), system.pricing,
                 system.plan, barrier, done, role, results)) for role in roles]
    for process in processes:
        process.start()
    for process in processes:
        join(process)
        assert process.exitcode == 0
    outcomes = dict(results.get(timeout=5) for _ in roles)
    assert outcomes["correct-" + ("first" if first == "correct" else "second")] == 2
    assert outcomes["publish-" + ("first" if first == "publish" else "second")] == (
        "published" if first == "publish" else "blocked"
    )
    assert system.repo.read_book_reference(METER).input_revision == 2
    assert local.read_publication(system.path).status == "blocked"


def registration_worker(url, path, pricing, barrier, retried, ready_to_retry, results, at, amount):
    from dao.prog.fastctrl.publication import PublicationLockBusy
    from dao.prog.settlement.repository import ConcurrentSettlementError

    engine, repo, _ = connect_publication(url, path, pricing)
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    barrier.wait(timeout=10)
    try:
        repo.register_inputs(METER, [item(at, amount=amount)], [snapshot], 1)
        results.put("first")
    except (PublicationLockBusy, ConcurrentSettlementError):
        results.put("retry")
        ready_to_retry.set()
        assert retried.wait(timeout=10)
        revision = repo.read_book_reference(METER).input_revision
        assert repo.register_inputs(METER, [item(at, amount=amount)], [snapshot], revision) == 3
    engine.dispose()


def test_two_process_registrations_retry_without_losing_a_revision(system, local):
    barrier, retried, ready, results = CTX.Barrier(2), CTX.Event(), CTX.Event(), CTX.Queue()
    processes = [CTX.Process(target=registration_worker, args=(system.url, str(system.path), system.pricing,
                 barrier, retried, ready, results, at, amount)) for at, amount in ((START_TS, "2"), (NOW, "1"))]
    for process in processes:
        process.start()
    try:
        assert ready.wait(timeout=10)
        assert {results.get(timeout=10), results.get(timeout=10)} == {"first", "retry"}
    finally:
        retried.set()
        for process in processes:
            join(process)
    assert all(process.exitcode == 0 for process in processes)
    batch = system.repo.read_inputs(METER, START_TS, NOW + 900)
    assert batch.input_revision == 3
    assert [(row.start_ts, row.import_kwh) for row in batch.items] == [(START_TS, Decimal("2")), (NOW, Decimal("1"))]
    assert local.read_publication(system.path).status == "blocked"


def lock_worker(path, lease, barrier, done, results):
    from dao.prog.fastctrl.publication import PublicationError, invalidate_plan, publication_lock

    # Een over fork geërfde lease is geen bewijs van een in dit proces gehouden slot.
    try:
        invalidate_plan(lease=lease, now_ts=NOW, reason="inherited")
    except PublicationError:
        results.put("foreign-lease-denied")
    with publication_lock(path):
        barrier.wait(timeout=10)
        assert done.wait(timeout=10)


def busy_writer(url, path, pricing, barrier, done, results):
    from dao.prog.fastctrl.publication import PublicationLockBusy

    engine, repo, _ = connect_publication(url, path, pricing)
    barrier.wait(timeout=10)
    try:
        snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
        # Ook _ensure_book voor een nog niet bestaand boek mag niet vóór dit slot komen.
        repo.register_inputs("other-grid", [replace(item(amount="2"), meter_id="other-grid")], [snapshot], 0)
        results.put("unexpected-write")
    except PublicationLockBusy:
        results.put("busy")
    finally:
        done.set()
        engine.dispose()


def test_process_lockbusy_does_not_initialize_or_change_database(system, local):
    before, file_before = reference_rows(system.repo), system.path.read_bytes()
    with local.publication_lock(system.path) as expired:
        pass
    barrier, done, results = CTX.Barrier(2), CTX.Event(), CTX.Queue()
    holder = CTX.Process(target=lock_worker, args=(str(system.path), expired, barrier, done, results))
    writer = CTX.Process(target=busy_writer, args=(system.url, str(system.path), system.pricing, barrier, done, results))
    holder.start()
    writer.start()
    for process in (holder, writer):
        join(process)
        assert process.exitcode == 0
    assert {results.get(timeout=5), results.get(timeout=5)} == {"foreign-lease-denied", "busy"}
    assert reference_rows(system.repo) == before and system.path.read_bytes() == file_before


def crash_worker(url, path, pricing, phase, barrier):
    import dao.prog.settlement.publication as publication

    engine, repo, _ = connect_publication(url, path, pricing)
    original = publication.invalidate_plan
    def invalidation(**kwargs):
        if phase == "before-invalidation":
            os._exit(41)
        original(**kwargs)
        if phase == "after-invalidation":
            os._exit(41)
    publication.invalidate_plan = invalidation
    if phase == "before-commit":
        @event.listens_for(engine, "after_cursor_execute")
        def crash(connection, cursor, statement, parameters, context, executemany):
            if statement.startswith("UPDATE settlement_book SET input_revision=?, snapshot_hashes_json="):
                os._exit(41)
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    barrier.wait(timeout=10)
    repo.register_inputs(METER, [item(amount="2")], [snapshot], 1)
    os._exit(41 if phase == "after-commit" else 42)


@pytest.mark.parametrize("phase,revision,status", [
    ("before-invalidation", 1, "ready"), ("after-invalidation", 1, "blocked"),
    ("before-commit", 1, "blocked"), ("after-commit", 2, "blocked"),
])
def test_process_crash_windows_preserve_conservative_file_db_order(system, local, phase, revision, status):
    before = reference_rows(system.repo)
    barrier = CTX.Barrier(2)
    process = CTX.Process(target=crash_worker, args=(system.url, str(system.path), system.pricing, phase, barrier))
    process.start()
    barrier.wait(timeout=10)
    join(process)
    assert process.exitcode == 41
    assert system.repo.read_book_reference(METER).input_revision == revision
    assert local.read_publication(system.path).status == status
    if revision == 1:
        assert reference_rows(system.repo) == before
    else:
        assert system.repo.read_inputs(METER, START_TS, NOW).items[0].import_kwh == Decimal("2")


def rename_crash_worker(url, path, pricing, candidate, phase, barrier):
    engine, _, adapter = connect_publication(url, path, pricing)
    original = os.replace
    def rename(source, target):
        if phase == "before":
            os._exit(43)
        original(source, target)
        os._exit(43)
    os.replace = rename
    barrier.wait(timeout=10)
    adapter.publish(candidate, now_ts=NOW)
    engine.dispose()


@pytest.mark.parametrize("phase,status", [("before", "blocked"), ("after", "ready")])
def test_ready_rename_crash_never_exposes_half_json(system, local, phase, status):
    system.adapter.block(now_ts=NOW, reason="awaiting-verified-plan")
    before = reference_rows(system.repo)
    barrier = CTX.Barrier(2)
    process = CTX.Process(target=rename_crash_worker, args=(system.url, str(system.path), system.pricing,
                          system.plan, phase, barrier))
    process.start()
    barrier.wait(timeout=10)
    join(process)
    assert process.exitcode == 43
    read = local.read_publication(system.path)
    assert read.status == status
    assert reference_rows(system.repo) == before
    assert not local.following_verdict(read, policy=system.adapter.policy(), now_ts=NOW).allowed


def test_closed_prefix_future_prices_late_energy_correction_cleans_without_future_financial_rows(
    system, local, monkeypatch,
):
    from dao.tests.settlement.test_price_repository import price
    from dao.tests.settlement.test_repository import rows

    monkeypatch.setattr("dao.prog.settlement.repository.time", lambda: NOW)
    money = {name: rows(system.repo, name) for name in ("run", "interval", "adjustment")}
    assert system.repo.register_prices(METER, [price(START_TS + 3600), price(START_TS + 7200)], 1) == 2
    assert rows(system.repo, "book")[0]["source_watermark_ts"] == NOW
    assert {name: rows(system.repo, name) for name in money} == money
    assert rows(system.repo, "book")[0]["dirty_from_ts"] is None
    prices = system.repo.read_prices(METER, START_TS, NOW, expected_input_revision=2)
    assert system.repo.register_inputs(METER, [item(amount="2")], [system.snapshot], 2,
                                       price_batch=prices, closed_before_ts=NOW) == 3
    assert local.read_publication(system.path).status == "blocked"
    batch = system.repo.read_inputs(METER, START_TS, NOW, expected_input_revision=3)
    result = settle_series(batch.items, batch.snapshots, SettlementState(START_TS, None, None))
    assert system.repo.commit(METER, result, batch, 1, "closed-only-correction") == 2
    current = system.repo.read(METER, START_TS, NOW)
    assert current.net_cost == Decimal("0.2") and current.dirty_from_ts is None
    assert current.source_watermark_ts == NOW and current.input_revision == 3
    basis = replace(system.plan.financial_context.basis, input_revision=3)
    system.adapter.check_basis(basis, now_ts=NOW)
    for name in ("input", "interval"):
        assert all(row["end_ts"] <= NOW for row in rows(system.repo, name))
    assert rows(system.repo, "adjustment") == []


def raw_ready(system, *, rule="native"):
    from dao.prog.settlement.metering import reprice_inputs
    from dao.tests.settlement.test_price_repository import price

    revision, snapshot = 1, system.snapshot
    if rule != "native":
        pricing = system.pricing.model_copy(deep=True)
        pricing.settlement.contracts[0].hour_price_rule = rule
        snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
        revision = system.repo.register_inputs(METER, (), [snapshot], revision)
    supplied = ([price()] if rule == "native" else [price(START_TS + i * 900, step=900) for i in range(4)])
    revision = system.repo.register_prices(METER, supplied, revision)
    prices = system.repo.read_prices(METER, START_TS, NOW)
    batch = system.repo.read_inputs(METER, START_TS, NOW)
    revision = system.repo.register_inputs(METER, reprice_inputs(batch, prices), [snapshot], revision,
                                           price_batch=prices, closed_before_ts=NOW)
    batch = system.repo.read_inputs(METER, START_TS, NOW)
    result = settle_series(batch.items, batch.snapshots, SettlementState(START_TS, None, None))
    generation = system.repo.read(METER, START_TS, NOW).generation
    system.repo.commit(METER, result, batch, generation, "raw-ready-prefix")
    basis = replace(system.plan.financial_context.basis, input_revision=revision,
                    snapshot_hashes=(snapshot.snapshot_hash,))
    plan = replace(system.plan, financial_context=replace(system.plan.financial_context, basis=basis))
    plan = publish_fixture(system.adapter, plan, (snapshot,))
    return snapshot, plan


@pytest.mark.parametrize("new_hour", [False, True])
def test_equal_registered_hour_price_new_measurement_does_not_invalidate_prognosis(system, local, monkeypatch, new_hour):
    from dao.prog.fastctrl.plan import BatteryPlanStep, PlanInterval
    from dao.prog.settlement.metering import merge_inputs
    from dao.prog.settlement.types import MeterInterval
    from dao.tests.settlement.test_price_repository import price

    at = START_TS + 3600 if new_hour else NOW
    monkeypatch.setattr("dao.prog.settlement.repository.time", lambda: at + 900)
    snapshot, plan = raw_ready(system)
    if new_hour:
        revision = plan.financial_context.basis.input_revision
        system.repo.register_prices(METER, [price(at)], revision)
        basis = replace(plan.financial_context.basis, input_revision=revision + 1,
                        horizon_end_ts=START_TS + 7200)
        plan = replace(plan, financial_context=replace(plan.financial_context, basis=basis), intervals=[
            *plan.intervals, *[PlanInterval(start, start + 900, 0.1, 0.1,
                                           batteries=[BatteryPlanStep(0, 50, 50)])
                              for start in range(at, START_TS + 7200, 900)],
        ])
        plan = publish_fixture(system.adapter, plan, (snapshot,))
    before = system.path.read_bytes()
    revision = plan.financial_context.basis.input_revision
    assert system.repo.register_prices(METER, [price(value="0.100000000000")], revision) == revision
    assert system.path.read_bytes() == before
    measurement = MeterInterval(at, at + 900, "1", "0", "0", "verified_pv", "measured", ("new-quarter",))
    candidate = merge_inputs([measurement], [price(at // 3600 * 3600)], [snapshot], meter_id=METER)
    prices = system.repo.read_prices(METER, at, at + 900)
    system.repo.register_inputs(METER, candidate, [snapshot], revision, price_batch=prices,
                                closed_before_ts=at + 900)
    assert system.path.read_bytes() == before and local.read_publication(system.path).status == "ready"
    from dao.prog.settlement.publication import FinancialPlanBlocked
    with pytest.raises(FinancialPlanBlocked):
        system.adapter.publish(plan, now_ts=NOW)


def test_first_market_quarter_correction_invalidates_plan_starting_later_same_hour(system, local, monkeypatch):
    from dao.tests.settlement.test_price_repository import price

    monkeypatch.setattr("dao.prog.settlement.repository.time", lambda: NOW)
    snapshot, plan = raw_ready(system, rule="mean_quarters")
    basis = replace(plan.financial_context.basis, prefix_start_ts=NOW)
    plan = replace(plan, financial_context=replace(plan.financial_context, basis=basis))
    publish_fixture(system.adapter, plan, (snapshot,))
    observations = []

    def observe(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO settlement_spot"):
            observations.append(local.read_publication(system.path).status)

    event.listen(system.engine, "before_cursor_execute", observe)
    try:
        system.repo.register_prices(METER, [price(value="0.5", step=900)], basis.input_revision)
    finally:
        event.remove(system.engine, "before_cursor_execute", observe)
    assert observations == ["blocked"] and local.read_publication(system.path).status == "blocked"


def test_raw_price_aba_rejects_old_planning_token_after_no_money_version_revalidation(system, local, monkeypatch):
    from dao.prog.settlement.publication import FinancialPlanBlocked
    from dao.tests.settlement.test_price_repository import price
    from dao.tests.settlement.test_repository import rows

    monkeypatch.setattr("dao.prog.settlement.repository.time", lambda: NOW)
    _, plan = raw_ready(system)
    revision = plan.financial_context.basis.input_revision
    history = {name: rows(system.repo, name) for name in ("input", "run", "interval", "adjustment")}
    generation = rows(system.repo, "book")[0]["generation"]
    system.repo.register_prices(METER, [price(value="0.2")], revision)
    system.repo.register_prices(METER, [price()], revision + 1)
    assert local.read_publication(system.path).status == "blocked"
    batch = system.repo.read_inputs(METER, START_TS, NOW)
    result = settle_series(batch.items, batch.snapshots, SettlementState(START_TS, None, None))
    assert system.repo.commit(METER, result, batch, generation, "raw-price-ABA") == generation
    assert {name: rows(system.repo, name) for name in history} == history
    with pytest.raises(FinancialPlanBlocked):
        system.adapter.publish(plan, now_ts=NOW)
    basis = replace(plan.financial_context.basis, input_revision=revision + 2)
    publish_fixture(system.adapter, replace(plan, financial_context=replace(plan.financial_context, basis=basis)),
                    (system.snapshot,))
    assert local.read_publication(system.path).plan.financial_context.basis.input_revision == revision + 2


@pytest.mark.parametrize("fault", ["file", "directory"])
def test_price_invalidation_fsync_failure_rolls_back_raw_revision_dirty_and_money(system, local, monkeypatch, fault):
    from dao.tests.settlement.test_price_repository import frozen, price

    before = frozen(system.repo)
    original = os.fsync

    def broken(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) == (fault == "directory"):
            raise OSError("injected-price-fsync-failure")
        return original(fd)

    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(local.PublicationError, match="fsync"):
        system.repo.register_prices(METER, [price(value="0.2")], 1)
    assert frozen(system.repo) == before
    assert local.read_publication(system.path).status == ("ready" if fault == "file" else "blocked")


def price_crash_worker(url, path, pricing, phase, barrier):
    import dao.prog.settlement.publication as publication
    from dao.tests.settlement.test_price_repository import price

    engine, repo, _ = connect_publication(url, path, pricing)
    original = publication.invalidate_plan

    def invalidation(**kwargs):
        if phase == "before-invalidation":
            os._exit(44)
        original(**kwargs)
        if phase == "after-invalidation":
            os._exit(44)

    publication.invalidate_plan = invalidation
    if phase == "before-commit":
        @event.listens_for(engine, "after_cursor_execute")
        def crash(connection, cursor, statement, parameters, context, executemany):
            if statement.startswith("UPDATE settlement_book SET input_revision=?, dirty_from_ts="):
                os._exit(44)
    barrier.wait(timeout=10)
    repo.register_prices(METER, [price(value="0.2")], 1)
    os._exit(44 if phase == "after-commit" else 45)


@pytest.mark.parametrize("phase,revision,status", [
    ("before-invalidation", 1, "ready"), ("after-invalidation", 1, "blocked"),
    ("before-commit", 1, "blocked"), ("after-commit", 2, "blocked"),
])
def test_price_process_crash_keeps_conservative_file_db_order_and_recoverable_kwh(system, local, phase, revision, status):
    from dao.tests.settlement.test_price_repository import frozen
    from dao.tests.settlement.test_repository import rows

    before = frozen(system.repo)
    barrier = CTX.Barrier(2)
    process = CTX.Process(target=price_crash_worker, args=(system.url, str(system.path), system.pricing, phase, barrier))
    process.start()
    barrier.wait(timeout=10)
    join(process)
    assert process.exitcode == 44
    assert system.repo.read_book_reference(METER).input_revision == revision
    assert local.read_publication(system.path).status == status
    if revision == 1:
        assert frozen(system.repo) == before
    else:
        assert rows(system.repo, "book")[0]["dirty_from_ts"] == START_TS
        assert rows(system.repo, "book")[0]["source_watermark_ts"] == NOW
        assert rows(system.repo, "input") == before["input"]
        assert rows(system.repo, "run") == before["run"]
