#!/usr/bin/env python3
"""Run the tests that belong to a change, instead of the whole suite.

The full suite takes about forty minutes. Running it after every edit is
waste: most edits touch one module, and the tests that cover that module
run in seconds. This script picks the tests that belong to the changed
sources and runs those.

    python scripts/verify_changed.py                  # uncommitted work, tier "file"
    python scripts/verify_changed.py --base c0fa963   # everything since a base
    python scripts/verify_changed.py --tier area      # the whole test area
    python scripts/verify_changed.py --list           # only show the selection

The tiers, cheapest first:

    file   the test files that cover the changed module   (seconds)
    area   the whole test directory of the changed area   (minutes)
    full   everything                                     (about forty minutes)

The selection is a proposal, not a proof. It is derived from the imports
in the test tree, so a change whose effect reaches another area shows up
in the full suite instead. "full" stays the gate before a review verdict
and before a release, not the price of every edit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = ROOT / "dao" / "tests"

# Where the tests for a source area live. Used when no test file mentions
# the changed module directly.
AREA_TESTS = {
    "dao/prog/settlement": "dao/tests/settlement",
    "dao/prog/fastctrl": "dao/tests/prog",
    "dao/prog/config": "dao/tests/config",
    "dao/prog/forecast": "dao/tests/forecast",
    "dao/webserver": "dao/tests/webserver",
    "dao/lib": "dao/tests/lib",
    "dao/prog": "dao/tests/prog",
}

# Modules that many test files import, so the import based selection says
# little. These name the tests that actually cover them.
COVERED_BY = {
    "dao/prog/day_ahead.py": (
        "dao/tests/settlement/test_optimization_producer.py",
        "dao/tests/settlement/test_optimization_refinement.py",
        "dao/tests/settlement/test_optimization_winter.py",
        "dao/tests/settlement/test_optimization_audit.py",
        "dao/tests/settlement/test_optimization_economics.py",
        "dao/tests/settlement/test_fast_financial_replay.py",
        "dao/tests/prog/test_optimizer_physical_acceptance.py",
        "dao/tests/prog/test_optimizer_inactive_devices.py",
        "dao/tests/prog/test_da_debug_settlement.py",
        "dao/tests/prog/test_heatpump_output.py",
    ),
    "dao/prog/da_base.py": (
        "dao/tests/prog/test_dao.py",
        "dao/tests/prog/test_da_base_dataframe_building.py",
        "dao/tests/prog/test_da_base_retry.py",
    ),
    "dao/prog/da_debug.py": (
        "dao/tests/prog/test_da_debug_settlement.py",
        "dao/tests/prog/test_da_debug_replay_fidelity.py",
        "dao/tests/prog/test_da_debug_ha_client.py",
    ),
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def changed_files(base: str | None) -> list[str]:
    """Everything that differs from base, plus the uncommitted work."""
    files: set[str] = set()
    if base:
        files.update(git("diff", "--name-only", f"{base}...HEAD").split())
    files.update(git("diff", "--name-only", "HEAD").split())
    files.update(git("diff", "--name-only", "--cached").split())
    return sorted(f for f in files if f)


def module_name(path: str) -> str:
    """dao/prog/day_ahead.py -> dao.prog.day_ahead"""
    stem = path[:-3] if path.endswith(".py") else path
    return stem.replace("/", ".")


def test_files() -> list[Path]:
    return sorted(p for p in TEST_ROOT.rglob("test_*.py"))


def mentions(test: Path, needles: list[str]) -> bool:
    text = test.read_text(encoding="utf-8", errors="ignore")
    return any(needle in text for needle in needles)


def needles_for(path: str) -> list[str]:
    """Import forms a test can use for the module: dotted, or bare."""
    stem = Path(path).stem
    return [module_name(path), f"import {stem}", f"from {stem} import"]


# Files outside a package that still have one obvious test area.
FILE_AREAS = {
    "dao/config.yaml": "dao/tests/config",
}


def area_of(path: str) -> str | None:
    if path in FILE_AREAS:
        return FILE_AREAS[path]
    return next((a for a in sorted(AREA_TESTS, key=len, reverse=True) if path.startswith(a + "/")), None)


def in_area(test: Path, directory: str) -> bool:
    return test.relative_to(ROOT).as_posix().startswith(directory)


def select_file_tier(paths: list[str], all_tests: list[Path]) -> tuple[list[Path], list[str]]:
    """Test files that cover the changed modules."""
    chosen: set[Path] = set()
    notes: list[str] = []
    for path in paths:
        if path.startswith("dao/tests/"):
            chosen.add(ROOT / path)
            continue
        if not path.endswith(".py") or not path.startswith("dao/"):
            # A changelog, a config file or a template carries no import of
            # its own; the area tier covers it.
            continue
        if path in COVERED_BY:
            for name in COVERED_BY[path]:
                chosen.add(ROOT / name)
            continue
        needles = needles_for(path)
        direct = [t for t in all_tests if mentions(t, needles)]
        if direct:
            chosen.update(direct)
        else:
            notes.append(f"{path} -> geen test noemt de module; gebruik --tier area")
    return sorted(chosen), notes


def select_area_tier(paths: list[str]) -> tuple[list[Path], list[str]]:
    """The whole test directory of every changed area."""
    chosen: set[Path] = set()
    notes: list[str] = []
    for path in paths:
        if path.startswith("dao/tests/"):
            chosen.add(ROOT / path)
            continue
        if not path.startswith("dao/"):
            continue
        area = area_of(path)
        if area is None:
            notes.append(f"{path} -> geen gebied gevonden")
            continue
        directory = AREA_TESTS[area]
        chosen.update(p for p in test_files() if in_area(p, directory))
    return sorted(chosen), notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="base commit; without it only uncommitted work is considered")
    parser.add_argument("--tier", choices=("file", "area", "full"), default="file")
    parser.add_argument("--list", action="store_true", help="show the selection without running it")
    parser.add_argument("--no-slow", action="store_true", help="skip the tests marked slow")
    args = parser.parse_args()

    paths = changed_files(args.base)
    if not paths:
        print("No changed files; nothing to select.")
        return 0
    print("Changed:")
    for path in paths:
        print(f"  {path}")

    if args.tier == "full":
        chosen = sorted(test_files())
        notes: list[str] = []
    else:
        chosen, notes = (select_file_tier(paths, test_files()) if args.tier == "file"
                         else select_area_tier(paths))

    print(f"\nTier {args.tier} selects {len(chosen)} test files:")
    for test in chosen:
        print(f"  {test.relative_to(ROOT)}")
    for note in notes:
        print(f"\nNote: {note}")
    if not chosen:
        print("\nNothing selected; run the full suite by hand.")
        return 1

    command = [sys.executable, "-m", "pytest", "-q", *[str(t.relative_to(ROOT)) for t in chosen]]
    if args.no_slow:
        command += ["-m", "not slow"]
    print("\nCommand:\n  " + " ".join(command))
    if args.list:
        return 0
    print()
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
