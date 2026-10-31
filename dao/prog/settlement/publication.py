"""DB-/configadapter voor één lokale financiële publicatiegrens.

Config is een processtartsnapshot. Herconfigureren/downgraden vereist een
gecontroleerde herstart/operationele stop; dit synchroniseert geen options.json.
De referentiefacade haalt onder het korte slot uitsluitend de actuele boekrij op.
"""

from contextlib import contextmanager
from pathlib import Path
from threading import local
from time import time

from sqlalchemy.engine import Engine

from dao.prog.config.models.pricing import PricingConfig
from dao.prog.fastctrl.plan import FastPlan, FinancialBasis, _integer
from dao.prog.fastctrl.publication import (
    GatePolicy,
    PublicationError,
    PublicationLease,
    _write_verified_ready,
    gate_policy,
    invalidate_plan,
    publication_lock,
    read_publication,
)
from dao.prog.settlement.repository import InputChange, InputPublicationGuard, SettlementRepository
from dao.prog.settlement.tariffs import make_snapshot


class FinancialPlanBlocked(Exception):
    """Kandidaat geweigerd; duurzaam blocked, geen herlabeling van oud token."""


def _overlaps(start, end, left, right):
    if type(start) is not int or (end is not None and (type(end) is not int or end <= start)):
        raise ValueError("Undecidable change range")
    return left < right and start < right and (end is None or left < end)


class _RegistrationGuard:
    def __init__(self, adapter):
        self._adapter = adapter
        self._held = local()

    @contextmanager
    def hold(self):
        with publication_lock(self._adapter._plan_path) as lease:
            self._held.lease = lease
            try:
                yield
            finally:
                del self._held.lease

    def before_revision(self, change: InputChange) -> None:
        lease = getattr(self._held, "lease", None)
        if type(lease) is not PublicationLease:
            raise PublicationError("Registration callback requires its own live hold()")
        lease._check()
        self._adapter._before_revision_locked(change, lease=lease)


class SettlementPublication:
    def __init__(self, repository: SettlementRepository, *, plan_path: Path,
                 pricing: PricingConfig, max_plan_age: int):
        _integer(max_plan_age, "max_plan_age", positive=True)
        self._repository = repository
        # Geen filesystem-/DB-I/O bij constructie. Canonicalisatie gebeurt bij slotname.
        self._plan_path = Path(plan_path).absolute()
        self._pricing = PricingConfig.model_validate(pricing.model_dump())
        self._policy = gate_policy(self._pricing)
        contracts = () if self._pricing.settlement is None else self._pricing.settlement.contracts
        self._snapshots = tuple(make_snapshot(self._pricing, contract)
                                for contract in sorted(contracts, key=lambda contract: contract.valid_from))
        self._max_plan_age = max_plan_age

    def registration_guard(self) -> InputPublicationGuard:
        return _RegistrationGuard(self)

    def policy(self) -> GatePolicy:
        return self._policy

    def _before_revision_locked(self, change, *, lease):
        now_ts = int(time())
        read = read_publication(lease.path)
        if read.status == "blocked":
            # Zichtbaar blocked kan van een mislukte directory-fsync komen; bewijs duurzaamheid opnieuw.
            invalidate_plan(lease=lease, now_ts=now_ts, reason=read.reason)
            return
        reason = None
        if read.status in ("missing", "invalid"):
            reason = "unknown_previous_publication"
        elif read.plan.format_version == 1:
            if self._policy.required_from_ts is not None and now_ts >= self._policy.required_from_ts:
                reason = "financial_plan_required"
        else:
            try:
                basis = read.plan.financial_context.basis
                if (type(change) is not InputChange or type(change.meter_id) is not str or not change.meter_id.strip()
                        or type(change.previous_revision) is not int or type(change.next_revision) is not int
                        or not 0 <= change.previous_revision < change.next_revision
                        or change.next_revision != change.previous_revision + 1):
                    raise ValueError("Undecidable input revision change")
                if basis.meter_id != change.meter_id:
                    return  # Een expliciet ander boek verandert deze grondslag niet.
                if basis.input_revision > change.previous_revision:
                    raise ValueError("Plan references an unregistered future revision")
                prefix = basis.prefix_start_ts, basis.planning_start_ts
                used = basis.prefix_start_ts, basis.horizon_end_ts
                if (any(_overlaps(start, end, *prefix) for start, end in change.changed_intervals)
                        or any(_overlaps(start, end, *used) for start, end in change.changed_prices)
                        or any(_overlaps(start, end, *used) for start, end in change.changed_contracts)):
                    reason = "registered_financial_inputs_changed"
                elif not (change.changed_intervals or change.changed_prices or change.changed_contracts):
                    reason = "undecidable_input_change"
            except (TypeError, ValueError, AttributeError):
                reason = "undecidable_input_change"
        if reason is not None:
            # Moet file- én directory-fsync afgerond hebben vóór de DB-inserts/revisie.
            invalidate_plan(lease=lease, now_ts=now_ts, reason=reason)

    def _check_basis_locked(self, basis: FinancialBasis, *, lease: PublicationLease, now_ts: int) -> None:
        lease._check()
        _integer(now_ts, "now_ts")
        if type(basis) is not FinancialBasis:
            raise ValueError("Expected explicit financial basis")
        settlement = self._pricing.settlement
        if settlement is None or settlement.mode != "active":
            raise ValueError("settlement_not_active")
        if basis.meter_id != settlement.meter_id:
            raise ValueError("meter_identity_mismatch")
        if basis.captured_ts > now_ts:
            raise ValueError("clock_before_capture")
        reference = self._repository.read_book_reference(basis.meter_id)
        if reference.input_revision <= 0 or reference.input_revision != basis.input_revision:
            raise ValueError("input_revision_changed")
        if reference.snapshot_hashes != basis.snapshot_hashes:
            raise ValueError("snapshot_selection_changed")
        # De canonieke selectie komt uitsluitend uit het boek/de berekening.
        # A2-bronversies zitten ook in die hashes; niet vervangen door confighashes.
        if reference.dirty_from_ts is not None and reference.dirty_from_ts < basis.planning_start_ts:
            raise ValueError("dirty_financial_prefix")
        cursor = basis.planning_start_ts
        for snapshot in self._snapshots:
            if snapshot.valid_to_ts is not None and snapshot.valid_to_ts <= cursor:
                continue
            if snapshot.valid_from_ts > cursor:
                break
            if snapshot.status != "confirmed":
                raise ValueError("contract_unconfirmed")
            cursor = basis.horizon_end_ts if snapshot.valid_to_ts is None else min(snapshot.valid_to_ts, basis.horizon_end_ts)
            if cursor == basis.horizon_end_ts:
                return
        raise ValueError("contract_horizon_uncovered")

    @staticmethod
    def _refuse(*, lease, now_ts, error):
        reason = str(error).strip() or type(error).__name__
        # Een schrijffout gaat als PublicationError door, niet als succesvolle blokkade.
        invalidate_plan(lease=lease, now_ts=now_ts, reason=reason)
        raise FinancialPlanBlocked(reason) from error

    def check_basis(self, basis: FinancialBasis, *, now_ts: int) -> None:
        _integer(now_ts, "now_ts")
        with publication_lock(self._plan_path) as lease:
            try:
                self._check_basis_locked(basis, lease=lease, now_ts=now_ts)
            except Exception as exc:
                self._refuse(lease=lease, now_ts=now_ts, error=exc)

    def publish(self, plan: FastPlan, *, now_ts: int) -> None:
        _integer(now_ts, "now_ts")
        with publication_lock(self._plan_path) as lease:
            try:
                if type(plan) is not FastPlan or plan.format_version != 2:
                    raise ValueError("financial_publication_requires_complete_ready_v2")
                # Loskoppelen van de mutable optimizerpayload vóór finale validatie/schrijven.
                candidate = FastPlan.from_dict(plan.to_dict())
                context = candidate.financial_context
                self._check_basis_locked(context.basis, lease=lease, now_ts=now_ts)
                if not context.economics_supported:
                    raise ValueError("economics_unsupported")
                if candidate.created_ts > now_ts:
                    raise ValueError("clock_before_creation")
                boundaries = [boundary for snapshot in self._snapshots
                              for boundary in (snapshot.valid_from_ts, snapshot.valid_to_ts)
                              if boundary is not None and boundary > candidate.created_ts]
                upper_bound = min(candidate.horizon_end_ts, candidate.created_ts + self._max_plan_age,
                                  min(boundaries, default=candidate.horizon_end_ts))
                if not now_ts < context.not_after_ts <= upper_bound:
                    raise ValueError("plan_lease_expired_or_out_of_bounds")
                if any(snapshot.price_interval_seconds != context.price_interval_s for snapshot in self._snapshots
                       if _overlaps(snapshot.valid_from_ts, snapshot.valid_to_ts,
                                    context.basis.planning_start_ts, context.basis.horizon_end_ts)):
                    raise ValueError("contract_price_interval_mismatch")
            except Exception as exc:
                self._refuse(lease=lease, now_ts=now_ts, error=exc)
            try:
                _write_verified_ready(candidate, lease=lease)
            except PublicationError:
                invalidate_plan(lease=lease, now_ts=now_ts, reason="ready_publication_failed")
                raise

    def block(self, *, now_ts: int, reason: str) -> None:
        with publication_lock(self._plan_path) as lease:
            invalidate_plan(lease=lease, now_ts=now_ts, reason=reason)


def make_publication(engine: Engine, *, plan_path: Path, pricing: PricingConfig,
                     max_plan_age: int) -> tuple[SettlementRepository, SettlementPublication]:
    """Twee facades, één engine; alleen de bewaakte writer verlaat deze factory."""
    reference_reader = SettlementRepository(engine)
    adapter = SettlementPublication(reference_reader, plan_path=plan_path, pricing=pricing, max_plan_age=max_plan_age)
    return SettlementRepository(engine, publication=adapter.registration_guard()), adapter
