"""Strict wire data must never turn missing financial evidence into authority."""

import copy
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import pytest

from dao.prog.fastctrl import plan as plans
from dao.prog.fastctrl.plan import BatteryPlanStep, BatterySpec, FastPlan, PlanInterval
from dao.prog.settlement.types import TerminalValue
from dao.tests.settlement.conftest import START_TS


def ready_plan(*, hashes=("a" * 64,), revision=1, start=START_TS + 900):
    basis = plans.FinancialBasis("synthetic-grid", revision, hashes, START_TS,
                                 start, START_TS + 3600, start, start)
    context = plans.FinancialContext(
        basis, start + 600, 3600,
        TerminalValue("synthetic-explicit-v1", Decimal("0.123456789012345678901234567890123456"),
                      (("accu", Decimal("0.94")),)), True, (),
    )
    return FastPlan(
        created_ts=start, interval_s=900,
        specs=[BatterySpec("accu", 10.0, 5000.0, 5000.0)],
        intervals=[PlanInterval(at, at + 900, 0.1, 0.1,
                                batteries=[BatteryPlanStep(0.0, 50.0, 50.0)])
                   for at in range(start, START_TS + 3600, 900)],
        strategy="minimize cost", price_average=0.1, format_version=2,
        financial_context=context,
    )


def test_ready_roundtrip_retains_explicit_terminal_decimal_and_detaches_basis():
    plan = ready_plan()
    payload = plan.to_dict()
    assert payload["format_version"] == 2 and payload["status"] == "ready"
    assert payload["financial_context"]["terminal_value"]["eur_per_ac_kwh"] == (
        "0.123456789012345678901234567890123456"
    )
    with localcontext() as context:
        context.prec = 6
        restored = FastPlan.from_dict(json.loads(json.dumps(payload)))
    assert restored.financial_context == plan.financial_context
    assert restored.to_dict() == payload
    with pytest.raises(FrozenInstanceError):
        restored.financial_context.basis.input_revision = 9
    hashes = ["a" * 64]
    basis = plans.FinancialBasis("m", 1, hashes, 0, 1, 2, 1, None)
    hashes.clear()
    assert basis.snapshot_hashes == ("a" * 64,)


def test_blocked_is_only_a_durable_fallback_request():
    blocked = plans.BlockedPlan(2, START_TS, "input_revision_changed")
    assert blocked.to_dict() == {
        "format_version": 2, "status": "blocked", "created_ts": START_TS,
        "reason": "input_revision_changed",
    }


@pytest.mark.parametrize("value", [0, True, "1", -1])
def test_basis_requires_a_strict_positive_revision(value):
    with pytest.raises((ValueError, TypeError)):
        plans.FinancialBasis("m", value, ("a" * 64,), 0, 1, 2, 1, None)


@pytest.mark.parametrize("hashes", [("A" * 64,), ("a" * 64, "a" * 64), ("not-a-hash",), ()])
def test_basis_requires_canonical_unique_pinned_hashes(hashes):
    with pytest.raises((ValueError, TypeError)):
        plans.FinancialBasis("m", 1, hashes, 0, 1, 2, 1, None)


@pytest.mark.parametrize("field,value", [
    (("format_version",), True), (("format_version",), 3),
    (("status",), "unknown"), (("status",), None),
    (("created_ts",), True), (("interval_s",), "900"),
    (("batteries",), {}), (("intervals",), None),
    (("financial_context",), None),
    (("financial_context", "basis", "meter_id"), ""),
    (("financial_context", "basis", "input_revision"), True),
    (("financial_context", "basis", "planning_start_ts"), START_TS - 1),
    (("financial_context", "price_interval_s"), 123),
    (("financial_context", "economics_supported"), 1),
    (("financial_context", "issues"), {}),
    (("financial_context", "terminal_value", "eur_per_ac_kwh"), 0.12),
    (("financial_context", "terminal_value", "eur_per_ac_kwh"), "NaN"),
    (("financial_context", "terminal_value", "eur_per_ac_kwh"), "0.1200"),
    (("financial_context", "terminal_value", "policy_version"), ""),
    (("intervals", 0, "price_import"), None),
    (("intervals", 0, "price_export"), "0.1"),
    (("intervals", 0, "grid_w"), float("inf")),
    (("intervals", 0, "house_w"), float("nan")),
    (("intervals", 0, "batteries"), []),
    (("intervals", 1, "start_ts"), START_TS + 1801),
    (("batteries", 0, "charge_efficiency"), 0),
    (("batteries", 0, "capacity_kwh"), True),
])
def test_malformed_v2_cannot_use_legacy_coercion_or_defaults(field, value):
    payload = copy.deepcopy(ready_plan().to_dict())
    target = payload
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    with pytest.raises((ValueError, TypeError)):
        FastPlan.from_dict(payload)


@pytest.mark.parametrize("field", ["status", "financial_context", "price_average", "interval_s"])
def test_partial_ready_is_rejected(field):
    payload = ready_plan().to_dict()
    del payload[field]
    with pytest.raises((ValueError, TypeError)):
        FastPlan.from_dict(payload)


@pytest.mark.parametrize("version", [2.0, "2", True])
def test_serialization_cannot_coerce_a_financial_format_version(version):
    plan = ready_plan()
    plan.format_version = version
    with pytest.raises((ValueError, TypeError)):
        plan.to_dict()


def test_financial_protocol_import_does_not_load_database_pandas_or_solver():
    result = subprocess.run([sys.executable, "-c", (
        "import sys; import dao.prog.fastctrl.publication; "
        "assert not any(x in sys.modules for x in ('sqlalchemy', 'pandas', 'mip'))"
    )], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
