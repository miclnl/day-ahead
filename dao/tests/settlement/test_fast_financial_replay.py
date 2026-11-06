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
from dao.tests.prog.test_fast_simulate import modeled_case
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


def _f3a8_profile_replay():
    from dao.prog.fastctrl.plan import BatteryPlanStep, FastPlan

    plan, replay = synthetic_replay()
    plan.specs = [replace(plan.specs[0], charge_efficiency=.8, discharge_efficiency=.6, cycle_cost=.03),
                  replace(plan.specs[0], name="B", capacity_kwh=20, discharge_efficiency=.5, cycle_cost=.09)]
    terminal = replace(replay.terminal_value,
                       reference_efficiency_by_battery=(("accu", D("0.7")), ("B", D("0.9"))))
    replay = replace(replay, terminal_value=terminal)
    plan.financial_context = replace(plan.financial_context, terminal_value=terminal,
                                     not_after_ts=plan.horizon_end_ts)
    for index, row in enumerate(plan.intervals):
        row.price_import = row.price_export = 999
        row.batteries = [
            BatteryPlanStep(1000 if index == 0 else 0, 40 - index, 39 - index,
                mean_ac_power_w=1000 if index == 0 else 0, on_power_w=1500 if index == 0 else 0,
                active_from_ts=row.start_ts, stop_ts=NOW + 600 if index == 0 else None),
            BatteryPlanStep(-1000 if index == 0 else 0, 80 - index, 79 - index,
                mean_ac_power_w=-1000 if index == 0 else 0, on_power_w=-1500 if index == 0 else 0,
                active_from_ts=NOW + 300 if index == 0 else row.start_ts, stop_ts=None),
        ]
    plan.price_average = 999
    return FastPlan.from_dict(bind_replay(plan, replay).to_dict()), replay


def test_f3a8_public_model_profile_and_terminal_are_independent_per_scenario(monkeypatch):
    from copy import deepcopy

    from dao.prog.fastctrl.command import plan_digest
    from dao.prog.fastctrl.policy import FastControlPolicy
    from dao.prog.fastctrl.receiver import receipt_matches_command
    from dao.prog.fastctrl.replay_execution import ProposalEvent, ReceiptEvent
    from dao.prog.fastctrl.simulate import compare

    plan, replay = _f3a8_profile_replay()
    samples = [Sample(NOW, 0), Sample(NOW + 900, 0)]
    socs, efficiencies = {"accu": 40, "B": 80}, {"accu": .5, "B": 1}
    baseline = modeled_case("f3a8-baseline", hold=900)
    fast = replace(modeled_case("f3a8-fast", hold=900),
                   execution=replace(baseline.execution, prepare_delay_s=.25))
    limits = PolicyLimits(min_benefit=1e9)
    before = deepcopy((plan, replay, samples, socs, efficiencies, baseline, fast, limits))
    policy_plans = []
    decide = FastControlPolicy.decide

    def observe(self, actual_plan, *args, **kwargs):
        policy_plans.append(deepcopy(actual_plan))
        return decide(self, actual_plan, *args, **kwargs)

    monkeypatch.setattr(FastControlPolicy, "decide", observe)
    comparison = compare(plan, samples, limits, socs, inverter_efficiency=efficiencies,
                         replay_basis=replay, baseline_execution=baseline, fast_execution=fast)
    assert len(policy_plans) == 3
    for actual in policy_plans:
        assert actual.format_version == 2 and actual.financial_context == plan.financial_context
        assert actual.specs == plan.specs
        assert [row.batteries for row in actual.intervals] == [row.batteries for row in plan.intervals]
        assert actual.price_average == pytest.approx(.1)
        assert [(row.start_ts, row.end_ts) for row in actual.intervals] == [(q.start_ts, q.end_ts) for q in replay.quotes]
        assert all(row.price_import == row.price_export == .1 for row in actual.intervals)

    command_ids = []
    for result, scenario, delay in ((comparison.baseline, baseline, 0), (comparison.fast, fast, .25)):
        assert result.execution_kind == "modeled" and result.coverage_status == "complete"
        assert result.soc_start_by_battery == (("accu", 40), ("B", 80))
        assert result.opening_state is replay.opening and result.replay_terminal_value is replay.terminal_value
        assert dict(result.assumptions)["scenario_id"] == scenario.scenario_id
        assert json.loads(dict(result.assumptions)["execution_model"])["prepare_delay_s"] == delay
        assert result.source_plan_digest == plan_digest(plan)
        assert result.execution_plan_digest == plan_digest(policy_plans[0]) != result.source_plan_digest
        sends = [e for e in result.trace if isinstance(e, ProposalEvent) and e.command is not None]
        receipts = [e for e in result.trace if isinstance(e, ReceiptEvent)]
        command_ids.append({event.command.command_id for event in sends})
        assert len(sends) == len(receipts) == result.writes == 4
        assert [(e.command.battery_name, e.command.setpoint_w, e.at_ts - NOW) for e in sends
                if e.command.setpoint_w] == [("accu", 1500, delay), ("B", -1500, 300 + delay)]
        for event, receipt in zip(sends, receipts, strict=True):
            assert event.command.plan_digest == result.execution_plan_digest
            assert event.at_ts - event.timing.anchor_ts == delay
            assert event.command.issued_at_ms == round(event.timing.anchor_ts * 1000)
            assert receipt_matches_command(receipt.receipt, event.command)
            assert receipt.receipt.status == receipt.application.outcome == "accepted"
            assert receipt.at_ts == receipt.application.at_ts == event.at_ts
        spans = [(s.start_ts - NOW, s.end_ts - NOW, tuple(b.modeled_ac_w for b in s.batteries))
                 for s in result.segments]
        assert spans == ([(0, 300, (1500, 0)), (300, 600, (1500, -1500)), (600, 900, (0, -1500))]
            if delay == 0 else [(0, .25, (0, 0)), (.25, 300, (1500, 0)), (300, 300.25, (1500, 0)),
                (300.25, 600, (1500, -1500)), (600, 600.25, (0, -1500)), (600.25, 900, (0, -1500))])
        fraction = (600 - delay) / 600
        for index, charge, discharge, dc, cell in ((0, .25, 0, .125, .1), (1, 0, .25, .25, -.5)):
            batteries = [s.batteries[index] for s in result.segments]
            for field, expected in (("charge_ac_kwh", charge), ("discharge_ac_kwh", discharge),
                                    ("dc_throughput_kwh", dc), ("cell_delta_kwh", cell)):
                assert sum(getattr(b, field) for b in batteries) == pytest.approx(expected * fraction)
            assert all(b.charge_ac_kwh * b.discharge_ac_kwh == 0 for b in batteries)
        assert result.soc_end_by_battery == (("accu", pytest.approx(40 + fraction)),
                                             ("B", pytest.approx(80 - 2.5 * fraction)))
        assert result.import_kwh == pytest.approx(.125) and result.export_kwh == pytest.approx(.125)
        assert result.throughput_kwh == pytest.approx(.375 * fraction)
        assert result.wear_cost == pytest.approx((.03 * .125 + .09 * .25) * fraction)
        # .1*.7 - .5*.9 = -.38 AC-reference kWh; exactly one archived conversion.
        assert result.terminal_value == pytest.approx(float(D("-0.38") * replay.terminal_value.eur_per_ac_kwh) * fraction)
        assert result.energy_cost == pytest.approx(0, abs=1e-12)
        assert result.total_cost == pytest.approx(result.wear_cost - result.terminal_value)
        assert result.settlement_comparison_pending
    assert command_ids[0].isdisjoint(command_ids[1])
    assert comparison.baseline.soc_end_by_battery != comparison.fast.soc_end_by_battery
    assert comparison.baseline.terminal_value != comparison.fast.terminal_value
    assert comparison.saving == comparison.saving_per_day == "unknown_reconstruction: settlement_comparison_pending"
    for output in (comparison.report(), comparison.daily_table()):
        assert output.startswith(comparison.saving) and "Besparing:" not in output
    changed = compare(plan, samples, limits, socs, inverter_efficiency=efficiencies, replay_basis=replay,
                      baseline_execution=replace(baseline, execution=replace(baseline.execution, prepare_delay_s=.125)),
                      fast_execution=fast)
    assert changed.fast == comparison.fast
    assert changed.baseline.soc_end_by_battery != comparison.baseline.soc_end_by_battery
    assert (plan, replay, samples, socs, efficiencies, baseline, fast, limits) == before


def test_f3a8_replay_has_no_io_or_test_imports(monkeypatch):
    import ast
    import builtins
    import sys
    from copy import deepcopy
    from pathlib import Path
    from unittest.mock import Mock

    import mip
    import requests
    from sqlalchemy.engine import Engine

    import dao.prog.day_ahead as day_ahead
    from dao.prog import da_fast
    from dao.prog.da_base import DaBase
    from dao.prog.fastctrl import ownership, publication, replay_execution
    from dao.prog.fastctrl.simulate import compare
    from dao.prog.settlement.publication import SettlementPublication
    from dao.prog.settlement.repository import SettlementRepository
    from dao.tests.prog.test_fast_simulate import _f3a8_recorded_timeline

    simulation = importlib.import_module("dao.prog.fastctrl.simulate")
    archive = replay_api()

    def test_dependency(name):
        return any(part in ("tests", "fakes") or part.startswith(("test_", "fake_"))
                   for part in name.split("."))

    for module in (simulation, replay_execution, da_fast):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [f"{node.module or ''}.{alias.name}" for alias in node.names]
            else:
                continue
            assert not any(test_dependency(name) for name in names), (module.__name__, node.lineno, names)

    plan, replay = _f3a8_profile_replay()
    samples = [Sample(NOW, 0), Sample(NOW + 900, 0)]
    socs, efficiencies = {"accu": 40, "B": 80}, {"accu": .5, "B": 1}
    baseline, fast = modeled_case("pure-base", hold=900), modeled_case("pure-fast", hold=900)
    recorded_plan, recorded_samples, recorded = _f3a8_recorded_timeline()
    limits = PolicyLimits(min_benefit=1e9)
    inputs = (plan, replay, samples, socs, efficiencies, baseline, fast, limits,
              recorded_plan, recorded_samples, recorded)
    before, archive_bytes = deepcopy(inputs), replay.canonical_bytes()
    forbidden_calls = []

    def forbid(boundary):
        def fail(*args, **kwargs):
            forbidden_calls.append(boundary)
            pytest.fail(f"replay crossed {boundary}")
        return Mock(side_effect=fail)

    for target, names in (
        (archive, ("read_replay_basis", "write_replay_basis")),
        (mip.Model, ("optimize",)), (day_ahead.DaCalc, ("calc_optimum",)),
        (SettlementRepository, ("__init__", "read_inputs", "read_prices", "commit")),
        (Engine, ("connect", "raw_connection")),
        (HistoryLoader, ("load", "_series")),
        (ownership.OwnerStore, ("__init__",)), (ownership.OwnerCoordinator, ("__init__",)),
        (DaBase, ("__init__", "get_state", "call_service", "set_state")),
        (requests.Session, ("request",)),
        (publication, ("read_publication", "write_publication", "publication_lock", "publication_attempts")),
        (SettlementPublication, ("publish", "block")),
    ):
        for name in names:
            monkeypatch.setattr(target, name, forbid(f"{target.__name__}.{name}"))
    real_import, real_import_module = builtins.__import__, importlib.import_module

    def check_import(caller, names):
        if caller.startswith(("dao.prog.", "prog.")) and any(test_dependency(name) for name in names):
            forbidden_calls.append((caller, names))
            pytest.fail(f"product imported test/fake code: {caller}: {names}")

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        check_import((globals or {}).get("__name__", ""), (name, *(fromlist or ())))
        return real_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name, package=None):
        check_import(sys._getframe(1).f_globals.get("__name__", ""), (name,))
        return real_import_module(name, package)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    comparison = compare(plan, samples, limits, socs, inverter_efficiency=efficiencies,
                         replay_basis=replay, baseline_execution=baseline, fast_execution=fast)
    result = simulate(recorded_plan, recorded_samples, limits, socs, "pure recorded", True,
                      inverter_efficiency=efficiencies, execution=recorded)
    assert comparison.baseline.coverage_status == comparison.fast.coverage_status == "complete"
    assert comparison.baseline.throughput_kwh == comparison.fast.throughput_kwh == pytest.approx(.375)
    assert comparison.baseline.trace and comparison.fast.trace
    assert result.execution_kind == "recorded" and result.writes == 3 and result.trace
    assert result.coverage_status == "partial" and result.soc_end_by_battery[0] == ("accu", None)
    assert result.soc_end_by_battery[1][1] is not None
    assert forbidden_calls == []
    assert inputs == before and replay.canonical_bytes() == archive_bytes


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
    from dao.prog.fastctrl.simulate import compare

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
    result = simulate(plan, samples, PolicyLimits(), 50, "sandbox", False, replay_basis=replay,
                      execution=modeled_case("archived-quotes", hold=900))
    assert result.label.startswith("reconstruction_base_tariffs")
    assert result.energy_cost == pytest.approx(0.025)  # 0.25 kWh * archived EUR 0.10, not 999.
    assert result.opening_state is replay.opening
    assert result.replay_terminal_value is replay.terminal_value
    assert result.soc_start == 50  # Settlement opening is not battery SoC.
    missing_basis = simulate(plan, samples, PolicyLimits(), 50, "missing basis", False,
                             execution=modeled_case("missing-basis", hold=900))
    assert missing_basis.energy_cost is missing_basis.total_cost is None
    assert missing_basis.import_kwh == pytest.approx(.25)
    assert "replay_basis_required" in missing_basis.reasons
    comparison = compare(plan, samples, PolicyLimits(), 50, replay_basis=replay,
                         baseline_execution=modeled_case("archived-baseline", hold=900),
                         fast_execution=modeled_case("archived-fast", hold=900))
    assert comparison.baseline.energy_cost == pytest.approx(.025)
    for side in (comparison.baseline, comparison.fast):
        assert side.opening_state is replay.opening
        assert side.replay_terminal_value is replay.terminal_value
    assert comparison.saving == "unknown_reconstruction: settlement_comparison_pending"


def test_simulate_terminal_uses_archived_ac_reference_once_not_new_tariff_or_inverter():
    replay_api()
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    # Keep the full 900s physical oracle within an explicitly valid context.
    plan.financial_context = replace(plan.financial_context, not_after_ts=plan.horizon_end_ts)
    step = plan.intervals[0].batteries[0]
    step.ac_power_w = step.mean_ac_power_w = step.on_power_w = 1000
    result = simulate(plan, [Sample(NOW, 0), Sample(NOW + 900, 0)], PolicyLimits(), 50,
                      "sandbox", False, inverter_efficiency=1, replay_basis=replay,
                      execution=modeled_case("archived-terminal", hold=900))
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
                      50, "sandbox", False, replay_basis=replay,
                      execution=modeled_case("mismatch", hold=900))
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
                      50, "sandbox", True, replay_basis=replay,
                      execution=modeled_case("quote-coverage", hold=900))
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
    result = simulate(plan, samples, PolicyLimits(), 50, "sandbox", True, replay_basis=replay,
                      execution=modeled_case("quote-projection", hold=900))
    assert result.label.startswith("reconstruction_base_tariffs:")
    assert observed == [(0.3, [0.1, 0.2, 0.6])] * 3
    assert plan.price_average == 999  # The mutable caller plan is not relabelled.


def comparison_with_unknown_result(unknown_side):
    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    samples = [Sample(NOW, 1000), Sample(NOW + 900, 1000)]
    known = simulate(plan, samples, PolicyLimits(), 50, "known reconstruction", False,
                     replay_basis=replay, execution=modeled_case("known", hold=900))
    assert known.total_cost is not None
    plan = replace(plan, financial_context=replace(plan.financial_context, replay_basis_hash="f" * 64))
    unknown = simulate(plan, samples, PolicyLimits(), 50, "sandbox", False, replay_basis=replay,
                       execution=modeled_case("unknown", hold=900))
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


def test_f3a6_archived_terminal_per_battery_once():
    from dao.prog.fastctrl.plan import BatteryPlanStep, FastPlan
    from dao.tests.prog.test_fast_simulate import _f3a5_case, _f3a6_result

    plan, replay = synthetic_replay()
    plan.specs = [replace(plan.specs[0], charge_efficiency=.8, discharge_efficiency=.6),
                  replace(plan.specs[0], name="B", capacity_kwh=20, discharge_efficiency=.5)]
    terminal = replace(replay.terminal_value,
                       reference_efficiency_by_battery=(("accu", D("0.7")), ("B", D("0.9"))))
    replay = replace(replay, terminal_value=terminal)
    plan.financial_context = replace(plan.financial_context, terminal_value=terminal)
    plan = bind_replay(plan, replay)
    for row in plan.intervals:
        row.batteries = [BatteryPlanStep(1500, 40, 40, mean_ac_power_w=1500, on_power_w=1500,
                                        active_from_ts=row.start_ts, stop_ts=None),
                         BatteryPlanStep(-1500, 80, 80, mean_ac_power_w=-1500, on_power_w=-1500,
                                         active_from_ts=row.start_ts, stop_ts=None)]
    plan = FastPlan.from_dict(plan.to_dict())
    kwargs = dict(case=_f3a5_case(), soc={"accu": 40, "B": 80},
                  efficiencies={"accu": .5, "B": 1}, replay_basis=replay)
    result = _f3a6_result(plan, ((NOW, 0), (NOW + 600, 0)), **kwargs)
    # A stores .1 cell kWh; B draws .5. Each gets only its OWN archived cell→AC eta.
    expected = float(D("-0.38") * D("0.123456789012345678901234567890123456"))
    assert result.terminal_value == expected
    assert result.soc_end_by_battery == (("accu", 41), ("B", 77.5))
    assert result.throughput_kwh == pytest.approx(.375)
    assert result.opening_state is replay.opening
    assert result.replay_terminal_value is terminal
    with localcontext() as context:
        context.prec = 6
        repeated = _f3a6_result(plan, ((NOW, 0), (NOW + 600, 0)), **kwargs)
    assert repeated.terminal_value == expected
    row = plan.intervals[0]
    row.batteries = [replace(step, ac_power_w=1000 if step.on_power_w > 0 else -1000,
                            mean_ac_power_w=1000 if step.on_power_w > 0 else -1000,
                            on_power_w=2 * step.on_power_w, stop_ts=NOW + 300)
                     for step in row.batteries]
    alternative = _f3a6_result(FastPlan.from_dict(plan.to_dict()), ((NOW, 0), (NOW + 600, 0)), **kwargs)
    assert alternative.soc_end_by_battery == result.soc_end_by_battery
    assert alternative.terminal_value == result.terminal_value


@pytest.mark.parametrize("fault", [None, "missing", "hash"])
def test_f3a6_v2_comparison_waits_for_settlement(fault, caplog):
    from dao.tests.prog.test_fast_simulate import _f3a5_case, _f3a6_result

    plan, replay = synthetic_replay()
    plan = bind_replay(plan, replay)
    plan.price_average = 999
    plan.intervals = [replace(row, price_import=999, price_export=999) for row in plan.intervals]
    if fault == "hash":
        plan.financial_context = replace(plan.financial_context, replay_basis_hash="f" * 64)
    result = _f3a6_result(plan, ((NOW, 1000), (NOW + 900, 1000)), case=_f3a5_case(),
                          replay_basis=None if fault == "missing" else replay)
    assert result.coverage_status == "complete" and result.import_kwh == .25
    assert result.settlement_comparison_pending is True
    if fault is None:
        assert result.label == "reconstruction_base_tariffs: f3a6"
        assert result.energy_cost == pytest.approx(.025)
        assert result.opening_state is replay.opening
        assert result.replay_terminal_value is replay.terminal_value
        assert result.terminal_value == 0
    else:
        assert result.label.startswith("unknown_reconstruction:")
        assert ("replay_basis_required" if fault == "missing" else "replay_basis_mismatch") in result.reasons
        assert result.energy_cost is result.wear_cost is result.terminal_value is result.total_cost is None
    comparison = Comparison(result, result)
    assert comparison.saving == comparison.saving_per_day == "unknown_reconstruction: settlement_comparison_pending"
    assert comparison.report().startswith(comparison.saving)
    assert comparison.daily_table().startswith(comparison.saving)
    with caplog.at_level(logging.INFO):
        log_comparison(comparison)
    assert "settlement_comparison_pending" in caplog.text
    assert "Besparing:" not in caplog.text and "euro" not in caplog.text
