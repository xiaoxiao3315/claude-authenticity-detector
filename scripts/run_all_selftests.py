#!/usr/bin/env python3
"""Aggregate offline self-test runner — one command, every module's --self-test.

Cross-platform (pure Python, no shell). Runs each module's offline self-test in
a subprocess and reports a single pass/fail summary. Exits non-zero if ANY
self-test fails, so CI can gate on it.

Why a discovery-based runner: the previous PowerShell check_all.ps1 hard-coded a
subset of modules and silently missed newly-added ones. Here the module list is
explicit and reviewed, but the runner is the single source of truth that both CI
and check_all.ps1 call — add a module here once and every entry point picks it up.

Usage:
    python scripts/run_all_selftests.py          # run all, summary at end
    python scripts/run_all_selftests.py -v        # stream each module's output
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Modules exposing `python <mod>.py --self-test`. Keep alphabetical.
SELFTEST_FLAG_MODULES = [
    "api_server",
    "archive_registry",
    "audit_export",
    "authenticity",
    "baseline_registry",
    "benchmarking",
    "campaigns",
    "cli_io",
    "compatibility",
    "decisions",
    "evidence_registry",
    "job_runtime",
    "judge_calibration",
    "logging_setup",
    "model_client",
    "quality_gate",
    "rescore",
    "run_eval",
    "trace_evaluation",
    "validate_run_records",
]

# Modules whose self-test is a subcommand rather than a flag.
SUBCOMMAND_SELFTESTS = [
    ("eval_cli", ["self-test"]),
]


def _commands() -> list[tuple[str, list[str]]]:
    cmds: list[tuple[str, list[str]]] = []
    for mod in SELFTEST_FLAG_MODULES:
        cmds.append((mod, [sys.executable, f"{mod}.py", "--self-test"]))
    for mod, extra in SUBCOMMAND_SELFTESTS:
        cmds.append((mod, [sys.executable, f"{mod}.py", *extra]))
    return cmds


def main(argv: list[str]) -> int:
    verbose = "-v" in argv or "--verbose" in argv
    cmds = _commands()
    # Force UTF-8 so emoji/CJK in self-test output don't crash on Windows cp1252.
    env_overrides = {"PYTHONIOENCODING": "utf-8"}
    import os
    env = {**os.environ, **env_overrides}

    failures: list[tuple[str, int, str]] = []
    started = time.time()
    print(f"running {len(cmds)} self-tests...\n")
    for name, cmd in cmds:
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, env=env,
                              capture_output=not verbose, text=True)
        dt = (time.time() - t0) * 1000
        if proc.returncode == 0:
            print(f"  PASS  {name:<22} ({dt:5.0f} ms)")
        else:
            print(f"  FAIL  {name:<22} (exit {proc.returncode})")
            tail = ""
            if not verbose and proc.stdout:
                tail = "\n".join(proc.stdout.strip().splitlines()[-8:])
            if not verbose and proc.stderr:
                tail += "\n" + "\n".join(proc.stderr.strip().splitlines()[-8:])
            failures.append((name, proc.returncode, tail.strip()))

    total_ms = (time.time() - started) * 1000
    print()
    if failures:
        print(f"{len(failures)} of {len(cmds)} self-tests FAILED ({total_ms:.0f} ms):")
        for name, rc, tail in failures:
            print(f"\n--- {name} (exit {rc}) ---")
            if tail:
                print(tail)
        return 1
    print(f"all {len(cmds)} self-tests passed ({total_ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
