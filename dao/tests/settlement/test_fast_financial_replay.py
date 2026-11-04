"""Pinned replay evidence, never a capability or a historical savings claim."""

import importlib
import importlib.util
import json
import logging
import os
import stat
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from hashlib import sha256

import pytest

from dao.prog.fastctrl.policy import PolicyLimits
from dao.prog.fastctrl.publication import following_verdict, read_publication
from dao.prog.fastctrl.simulate import Comparison, HistoryLoader, Sample, log_comparison, simulate
from dao.prog.settlement.publication import FinancialPlanBlocked
from dao.prog.settlement.tariffs import make_snapshot, quote
from dao.prog.settlement.types import SettlementState
from dao.tests.settlement.test_fast_contract import ready_plan
from dao.tests.settlement.test_publication import NOW, START_TS, pricing_config, replay_for_plan
from dao.tests.settlement.test_publication import system as system

D = Decimal


def replay_api():
    assert importlib.util.find_spec("dao.prog.settlement.replay") is not None, "F1c ReplayBasis module missing"
    return importlib.import_module("dao.prog.settlement.replay")


def synthetic_replay():
    pricing = pricing_config()
    snapshot = make_snapshot(pricing, pricing.settlement.contracts[0])
    plan = ready_plan(hashes=(snapshot.snapshot_hash,))
    return plan, replay_for_plan(plan, (snapshot,))


def bind_replay(plan, replay):
    return replace(plan, financial_context=replace(plan.financial_context, replay_basis_hash=replay.content_hash))


def test_archive_is_canonical_content_addressed_immutable_and_opening_sensitive(tmp_path):
    api = replay_api()
    plan, replay = synthetic_replay()
    canonical = replay.canonical_bytes()
    with localcontext() as context:
        context.prec = 6
        assert replay.canonical_bytes() == canonical
    assert b"0.123456789012345678901234567890123456" in canonical
    digest = sha256(canonical).hexdigest()
    assert replay.content_hash == digest
    plan_path = tmp_path / "fast_plan.json"
    assert api.write_replay_basis(replay, plan_path=plan_path) == digest
    path = api.replay_archive_path(plan_path, digest)
    assert path.name == digest and path.read_bytes() == canonical
    assert api.write_replay_basis(replay, plan_path=plan_path) == digest
    read = api.read_replay_basis(plan_path, digest, financial_basis=replay.basis,
                                 terminal_value=replay.terminal_value)
    assert read.status == "known" and read.basis == replay
    assert read.basis.opening == SettlementState(NOW, None, None)
    assert read.basis.price_batch.items[0].spot.source == "synthetic-replay"
    assert read.basis.snapshots[0].snapshot_hash == plan.financial_context.basis.snapshot_hashes[0]
    with pytest.raises(FrozenInstanceError):
        replay.opening = SettlementState(NOW + 1, None, None)
    independent = replace(replay, opening=SettlementState(NOW, None, None))
    assert independent.opening is not replay.opening
    assert independent.canonical_bytes() == canonical
    # Same financial selection, distinct exact opening (not an alias of a shared mutable state).
    from dao.prog.settlement.types import BonusState
    other = replace(replay, opening=SettlementState(NOW, BonusState(START_TS, START_TS + 86400, NOW, D("7")), None))
    assert other.content_hash != digest
    assert api.write_replay_basis(other, plan_path=plan_path) != digest
    later_plan = ready_plan(hashes=replay.basis.snapshot_hashes, start=NOW + 60)
    later_plan.intervals[-1].end_ts = START_TS + 3600
    later = replay_for_plan(later_plan, replay.snapshots)
    assert later.basis.input_revision == replay.basis.input_revision
    assert later.opening.asof_ts == NOW + 60
    assert later.content_hash != digest
    assert api.write_replay_basis(later, plan_path=plan_path) != digest


@pytest.mark.parametrize("fault", ["missing", "corrupt", "hash", "basis", "terminal", "invalid-plan-path"])
def test_unknown_archive_never_falls_back_or_returns_financial_evidence(tmp_path, fault):
    api = replay_api()
    _, replay = synthetic_replay()
    plan_path = tmp_path / "fast_plan.json"
    digest = api.write_replay_basis(replay, plan_path=plan_path)
    path = api.replay_archive_path(plan_path, digest)
    expected_basis, terminal = replay.basis, replay.terminal_value
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(b"broken archive")
    elif fault == "hash":
        path.write_bytes(replay.canonical_bytes() + b" ")
    elif fault == "basis":
        expected_basis = replace(expected_basis, input_revision=2)
    elif fault == "terminal":
        terminal = replace(terminal, eur_per_ac_kwh=D("0.5"))
    else:
        plan_path.symlink_to(tmp_path / "not-a-plan")
    read = api.read_replay_basis(plan_path, digest, financial_basis=expected_basis, terminal_value=terminal)
    assert read.status == "unknown" and read.basis is None
    assert read.reason.startswith("replay_")


def test_archive_roundtrip_rejects_self_consistent_but_noncanonical_bytes(tmp_path):
    api = replay_api()
    _, replay = synthetic_replay()
    raw = json.dumps(json.loads(replay.canonical_bytes()), indent=2).encode()
    digest = sha256(raw).hexdigest()
    api.replay_archive_path(tmp_path / "plan", digest).write_bytes(raw)
    read = api.read_replay_basis(tmp_path / "plan", digest)
    assert read.status == "unknown" and read.basis is None


def test_ready_publication_has_durable_archive_before_ready_replace(system, monkeypatch):
    api = replay_api()
    import dao.prog.settlement.publication as publication

    replay = replay_for_plan(system.plan, (system.snapshot,))
    plan = bind_replay(system.plan, replay)
    ready = publication._write_verified_ready
    observed = []

    def check_before_ready(candidate, *, lease):
        digest = candidate.financial_context.replay_basis_hash
        raw = api.replay_archive_path(system.path, digest).read_bytes()
        assert sha256(raw).hexdigest() == digest
        read = api.read_replay_basis(system.path, digest)
        assert read.status == "known"
        assert read.basis.terminal_value == candidate.financial_context.terminal_value
        observed.append(True)
        return ready(candidate, lease=lease)

    monkeypatch.setattr(publication, "_write_verified_ready", check_before_ready)
    system.adapter.publish(plan, now_ts=NOW, replay_basis=replay)
    assert observed == [True]
    read = read_publication(system.path)
    assert read.status == "ready"
    assert not following_verdict(read, policy=system.adapter.policy(), now_ts=NOW).allowed


@pytest.mark.parametrize("fault", ["file-fsync", "directory-fsync", "verification"])
def test_archive_failure_blocks_instead_of_issuing_a_dangling_ready(system, monkeypatch, fault):
    api = replay_api()
    replay = replay_for_plan(system.plan, (system.snapshot,))
    # New archive; ready from fixture may not be retained on refusal.
    replay = replace(replay, opening=replace(replay.opening, month=None),
                     price_batch=replace(replay.price_batch, items=tuple(reversed(replay.price_batch.items))))
    plan = bind_replay(system.plan, replay)
    original = os.fsync
    failed = []

    def fsync(fd):
        target = os.readlink(f"/proc/self/fd/{fd}")
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        is_archive = ".replay_basis_" in target or target.endswith(replay.content_hash)
        if not failed and ((fault == "file-fsync" and is_archive)
                           or (fault == "directory-fsync" and directory)):
            failed.append(True)
            raise OSError("synthetic_archive_fsync_failure")
        return original(fd)

    if fault == "verification":
        def unreadable(*args, **kwargs):
            return api.ReplayRead("unknown", None, "replay_archive_unverifiable")
        monkeypatch.setattr(api, "read_replay_basis", unreadable)
    else:
        monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(FinancialPlanBlocked, match="replay_"):
        system.adapter.publish(plan, now_ts=NOW, replay_basis=replay)
    read = read_publication(system.path)
    assert read.status == "blocked" and read.plan is None


def test_missing_archive_without_supplied_basis_cannot_republish(system):
    api = replay_api()
    path = api.replay_archive_path(system.path, system.plan.financial_context.replay_basis_hash)
    path.unlink()
    with pytest.raises(FinancialPlanBlocked, match="replay_"):
        system.adapter.publish(system.plan, now_ts=NOW)
    assert read_publication(system.path).status == "blocked"


def test_later_tariff_revision_cannot_change_archived_evidence(system):
    api = replay_api()
    digest = system.plan.financial_context.replay_basis_hash
    path = api.replay_archive_path(system.path, digest)
    before = path.read_bytes()
    pricing = system.pricing.model_copy(deep=True)
    pricing.cost_supplier_consumption = {"2020-01-01": 0.7}
    revised = make_snapshot(pricing, pricing.settlement.contracts[0])
    system.repo.register_inputs(system.plan.financial_context.basis.meter_id, (), (revised,), 1)
    assert path.read_bytes() == before and sha256(before).hexdigest() == digest
    read = api.read_replay_basis(system.path, digest, financial_basis=system.plan.financial_context.basis)
    assert read.status == "known"
    assert read.basis.quotes[0].import_rate == D("0.1")
    assert read.basis.snapshots == (system.snapshot,)


def test_simulate_uses_explicit_archived_quotes_opening_terminal_without_io(monkeypatch):
    api = replay_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    plan.intervals = [replace(i, price_import=999, price_export=999) for i in plan.intervals]
    plan.price_average = 999

    def forbidden(*args, **kwargs):
        pytest.fail("explicit ReplayBasis attempted DB/solver/optimizer/archive I/O")

    import mip

    import dao.prog.day_ahead as day_ahead
    from dao.prog.settlement.repository import SettlementRepository
    monkeypatch.setattr(SettlementRepository, "read_prices", forbidden)
    monkeypatch.setattr(SettlementRepository, "read_inputs", forbidden)
    monkeypatch.setattr(mip.Model, "optimize", forbidden)
    monkeypatch.setattr(day_ahead.DaCalc, "calc_optimum", forbidden)
    monkeypatch.setattr(HistoryLoader, "_series", forbidden)
    monkeypatch.setattr(api, "read_replay_basis", forbidden)
    samples = [Sample(NOW, 1000), Sample(NOW + 900, 1000)]
    result = simulate(plan, samples, PolicyLimits(), 50, "sandbox", False, replay_basis=replay)
    assert result.label.startswith("reconstruction_base_tariffs")
    assert result.energy_cost == pytest.approx(0.025)  # 0.25 kWh * archived EUR 0.10, not 999.
    assert result.opening_state is replay.opening
    assert result.replay_terminal_value is replay.terminal_value
    assert result.soc_start == 50  # Settlement opening is not battery SoC.
    assert simulate(plan, samples, PolicyLimits(), 50, "legacy", False).energy_cost == pytest.approx(249.75)


def test_simulate_terminal_uses_archived_ac_reference_once_not_new_tariff_or_inverter():
    replay_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    step = plan.intervals[0].batteries[0]
    step.ac_power_w = step.mean_ac_power_w = step.on_power_w = 1000
    result = simulate(plan, [Sample(NOW, 0), Sample(NOW + 900, 0)], PolicyLimits(), 50,
                      "sandbox", False, inverter_efficiency=1, replay_basis=replay)
    # 0.25 cell kWh added, then valued with exactly archived EUR/kWh * archived eta 0.94.
    assert result.terminal_value == pytest.approx(float(D("0.25") * D("0.94") *
                            D("0.123456789012345678901234567890123456")))


@pytest.mark.parametrize("fault", ["hash", "basis", "terminal"])
def test_simulate_mismatch_is_explicit_unknown_without_a_euro_claim(fault):
    replay_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    context = plan.financial_context
    if fault == "hash":
        context = replace(context, replay_basis_hash="f" * 64)
    elif fault == "basis":
        context = replace(context, basis=replace(context.basis, input_revision=2))
    else:
        context = replace(context, terminal_value=replace(context.terminal_value, eur_per_ac_kwh=D("0.7")))
    plan = replace(plan, financial_context=context)
    result = simulate(plan, [Sample(NOW, 1000), Sample(NOW + 900, 1000)], PolicyLimits(),
                      50, "sandbox", False, replay_basis=replay)
    assert result.label.startswith("unknown_reconstruction:")
    assert result.energy_cost is result.wear_cost is result.terminal_value is result.total_cost is None


def test_replay_basis_rejects_a_different_financial_selection_or_opening():
    replay_api()
    _, replay = synthetic_replay()
    for changes in ({"basis": replace(replay.basis, input_revision=2)},
                    {"opening": SettlementState(NOW + 1, None, None)},
                    {"quotes": replay.quotes[1:]},
                    {"price_batch": replace(replay.price_batch, snapshots=())}):
        with pytest.raises((TypeError, ValueError)):
            replace(replay, **changes)


@pytest.mark.parametrize("fault", [
    "truncated", "missing-middle", "extra-interval", "duplicate-interval",
    "reordered", "gap", "empty", "fewer-quotes",
])
def test_fix1_replay_requires_complete_bidirectional_plan_quote_coverage(fault):
    plan, replay = synthetic_replay()
    if fault == "truncated":
        plan.intervals = plan.intervals[:1]
    elif fault == "missing-middle":
        plan.intervals = [plan.intervals[0], plan.intervals[2]]
    elif fault == "extra-interval":
        plan.intervals = [*plan.intervals, plan.intervals[-1]]
    elif fault == "duplicate-interval":
        plan.intervals = [plan.intervals[0], plan.intervals[0], plan.intervals[2]]
    elif fault == "reordered":
        plan.intervals.reverse()
    elif fault == "gap":
        plan.intervals[1] = replace(plan.intervals[1], start_ts=NOW + 901)
    elif fault == "empty":
        plan.intervals = []
    else:
        # A valid archive with two quotes covering the same full horizon,
        # against a plan with three intervals: no exact one-to-one pairing.
        replay = replace(replay, quotes=(replace(replay.quotes[0], end_ts=NOW + 1800), replay.quotes[2]))
    plan = bind_replay(plan, replay)
    result = simulate(plan, [Sample(NOW, 1000), Sample(NOW + 900, 1000)], PolicyLimits(),
                      50, "sandbox", True, replay_basis=replay)
    assert result.label.startswith("unknown_reconstruction:"), result
    assert result.energy_cost is result.wear_cost is result.terminal_value is result.total_cost is None


def test_fix1_valid_replay_policy_uses_average_of_exactly_paired_quotes(monkeypatch):
    from dao.prog.fastctrl.policy import FastControlPolicy

    plan, replay = synthetic_replay()
    rates = (D("0.1"), D("0.2"), D("0.6"))
    quotes = tuple(quote(q.snapshot, q.start_ts, q.end_ts, rate)
                   for q, rate in zip(replay.quotes, rates, strict=True))
    items = tuple(replace(item, spot=replace(item.spot, price_ex_vat=rate))
                  for item, rate in zip(replay.price_batch.items, rates, strict=True))
    replay = replace(replay, quotes=quotes, price_batch=replace(replay.price_batch, items=items))
    plan = bind_replay(plan, replay)
    plan.price_average = 999
    observed = []
    decide = FastControlPolicy.decide

    def capture_policy(self, actual_plan, *args, **kwargs):
        observed.append((actual_plan.price_average, [i.price_import for i in actual_plan.intervals]))
        return decide(self, actual_plan, *args, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "decide", capture_policy)
    samples = [Sample(NOW + i * 900, 1000) for i in range(4)]
    result = simulate(plan, samples, PolicyLimits(), 50, "sandbox", True, replay_basis=replay)
    assert result.label.startswith("reconstruction_base_tariffs:")
    assert observed == [(0.3, [0.1, 0.2, 0.6])] * 3
    assert plan.price_average == 999  # The mutable caller plan is not relabelled.


def comparison_with_unknown_result(unknown_side):
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    samples = [Sample(NOW, 1000), Sample(NOW + 900, 1000)]
    known = simulate(plan, samples, PolicyLimits(), 50, "legacy", False)
    plan = replace(plan, financial_context=replace(plan.financial_context, replay_basis_hash="f" * 64))
    unknown = simulate(plan, samples, PolicyLimits(), 50, "sandbox", False, replay_basis=replay)
    assert unknown.label.startswith("unknown_reconstruction:") and unknown.total_cost is None
    return Comparison(unknown if unknown_side != "fast" else known,
                      unknown if unknown_side != "baseline" else known)


@pytest.mark.parametrize("unknown_side", ["baseline", "fast", "both"])
@pytest.mark.parametrize("consumer", ["saving", "saving_per_day", "report", "daily_table"])
def test_fix1_comparison_consumers_label_unknown_without_money_or_typeerror(unknown_side, consumer):
    comparison = comparison_with_unknown_result(unknown_side)
    output = getattr(comparison, consumer)
    if callable(output):
        output = output()
    assert isinstance(output, str) and output.startswith("unknown_reconstruction:")
    assert "euro" not in output.lower() and "besparing" not in output.lower()


@pytest.mark.parametrize("unknown_side", ["baseline", "fast", "both"])
def test_fix1_log_comparison_labels_unknown_without_money_or_typeerror(unknown_side, caplog):
    comparison = comparison_with_unknown_result(unknown_side)
    with caplog.at_level(logging.INFO):
        log_comparison(comparison)
    assert "unknown_reconstruction:" in caplog.text
    assert "euro" not in caplog.text.lower() and "besparing" not in caplog.text.lower()


@pytest.mark.parametrize("with_replay", [False, True])
def test_fix1_no_batteries_keeps_specific_configuration_error_with_or_without_replay(with_replay):
    plan, replay = synthetic_replay()
    plan = replace(bind_replay(plan, replay), specs=[])
    kwargs = {"replay_basis": replay} if with_replay else {}
    with pytest.raises(ValueError, match="^het plan bevat geen batterij$"):
        simulate(plan, [Sample(NOW, 1000), Sample(NOW + 900, 1000)], PolicyLimits(),
                 50, "sandbox", False, **kwargs)
