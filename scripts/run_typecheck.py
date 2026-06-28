#!/usr/bin/env python3
"""Strict type-check gate for the graduated module set.

mypy on the whole tree still reports ~300 historical findings in the legacy
modules. Blocking CI on all of them at once is not actionable. Instead we hold a
strict zero-error line on modules that have been cleaned, and grow the set over
time. CI runs THIS script; it fails if any graduated module regresses.

To graduate a module: clean `python -m mypy <mod>.py`, add it to STRICT_MODULES
here, and (optionally) tighten its section in mypy.ini.

Usage:
    python scripts/run_typecheck.py          # check the strict set
    python scripts/run_typecheck.py --all     # also report (non-gating) legacy total
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Modules held to zero mypy errors. Grow this list as legacy modules are fixed.
STRICT_MODULES = [
    "cli_io",
    "model_client",
    "benchmarking",
    "rescore",
    "decisions",
    "logging_setup",
    "redaction",
    "trace_evaluation",
    "evidence_registry",
    "quality_gate",
    "campaigns",
    "run_eval",
    "run_records",
    "authenticity",
    "api_server",
    "eval_cli",
    "local_env",
    "acceptance_pack",
    "archive_registry",
    "baseline_registry",
]


def _run_mypy(targets: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(ROOT / "mypy.ini"), *targets],
        cwd=ROOT, capture_output=True, text=True,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def main(argv: list[str]) -> int:
    files = [f"{m}.py" for m in STRICT_MODULES]
    print(f"type-checking {len(files)} graduated module(s): {', '.join(STRICT_MODULES)}")
    rc, out = _run_mypy(files)
    # Count only errors attributable to the strict files themselves.
    own_errors = [
        ln for ln in out.splitlines()
        if ": error:" in ln and any(ln.startswith(f) for f in files)
    ]
    if own_errors:
        print(f"\nFAIL — {len(own_errors)} type error(s) in the strict set:\n")
        for ln in own_errors:
            print(f"  {ln}")
        return 1
    print("ok — strict module set is type-clean")

    if "--all" in argv:
        all_py = [str(p.name) for p in sorted(ROOT.glob("*.py"))]
        _, all_out = _run_mypy(all_py)
        total = sum(1 for ln in all_out.splitlines() if ": error:" in ln)
        print(f"\n(non-gating) whole-tree mypy errors remaining: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
