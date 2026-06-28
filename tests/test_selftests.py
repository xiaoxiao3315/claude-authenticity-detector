"""Pytest bridge over the modules' built-in offline self-tests.

Each module already ships a `--self-test` that asserts its pure helpers. This
file imports those self-test functions and runs them under pytest so that:
  1. coverage.py can measure what they exercise (subprocess runs can't be
     measured easily), and
  2. they live in a standard test runner alongside finer-grained unit tests
     and the end-to-end suite.

Modules without an importable self-test function (their self-test logic lives
in __main__/CLI) are exercised via subprocess against their --self-test flag.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# module name -> attribute name of its self-test function (callable, returns/raises)
IMPORTABLE_SELFTESTS = {
    "cli_io": "_self_test",
    "model_client": "_self_test",
    "benchmarking": "_self_test",
    "campaigns": "_self_test",
    "decisions": "_self_test",
    "rescore": "_self_test",
    "run_eval": "_self_test",
    "api_server": "_self_test",
    "baseline_registry": "_self_test",
    "judge_calibration": "_self_test",
    "authenticity": "_self_test",
    "job_runtime": "_self_test",
    "quality_gate": "self_test",
    "compatibility": "self_test",
    "trace_evaluation": "run_self_test",
    "audit_export": "self_test",
    "archive_registry": "self_test",
    "evidence_registry": "self_test",
}

# Modules whose self-test only exists behind a CLI flag/subcommand.
SUBPROCESS_SELFTESTS = {
    "validate_run_records": ["validate_run_records.py", "--self-test"],
    "eval_cli": ["eval_cli.py", "self-test"],
}


@pytest.mark.parametrize("module_name, func_name", sorted(IMPORTABLE_SELFTESTS.items()))
def test_module_self_test(module_name: str, func_name: str) -> None:
    """Import the module and run its self-test function; any assert inside fails the test."""
    import importlib

    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name, None)
    assert callable(func), f"{module_name}.{func_name} is not callable"
    result = func()
    # self-test functions either return 0 / None on success or raise on failure.
    assert result in (0, None), f"{module_name}.{func_name}() returned {result!r}"


@pytest.mark.parametrize("module_name, argv", sorted(SUBPROCESS_SELFTESTS.items()))
def test_cli_self_test(module_name: str, argv: list[str]) -> None:
    """Run CLI-only self-tests in a subprocess; non-zero exit fails the test."""
    proc = subprocess.run(
        [sys.executable, *argv],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.returncode == 0, f"{module_name} self-test failed:\n{proc.stdout}\n{proc.stderr}"
