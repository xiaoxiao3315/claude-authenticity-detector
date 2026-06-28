"""eval_cli main() argparse-dispatch coverage (R19).

main() builds the full subparser tree and dispatches to a handler. It reads
sys.argv and returns the handler's exit code. We invoke it with crafted argv
for the read-only / dry-run subcommands, covering the ~280-line dispatcher and
each subparser's argument wiring without a live network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_cli as E  # noqa: E402


def _providers_file(tmp_path: Path) -> Path:
    data = {
        "tested_model": {"provider_id": "tested", "base_url": "https://gw.x/v1",
                         "model": "claude-opus-4-6", "api_key_env": "TESTED_KEY",
                         "protocol": "anthropic_messages", "auth_type": "x-api-key"},
        "judge_model": {"provider_id": "judge", "base_url": "https://gw.x/v1",
                        "model": "gpt-5.5", "api_key_env": "JUDGE_KEY",
                        "protocol": "openai_chat", "auth_type": "bearer"},
    }
    p = tmp_path / "providers.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["eval_cli.py", *argv])
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)
    return E.main()


@pytest.fixture(autouse=True)
def _no_secrets(monkeypatch):
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)


# ---------------------------------------------------------------------------
# self-test subcommand
# ---------------------------------------------------------------------------
def test_main_self_test(monkeypatch, capsys):
    rc = _run_main(monkeypatch, ["self-test"])
    assert rc == 0
    assert "self-test ok" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# fingerprint subcommand (config-only)
# ---------------------------------------------------------------------------
def test_main_fingerprint(monkeypatch, capsys, tmp_path):
    rc = _run_main(monkeypatch, ["fingerprint", "--providers", str(_providers_file(tmp_path)),
                                 "--provider", "tested_model"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["provider_id"] == "tested"


# ---------------------------------------------------------------------------
# dry-run probe subcommands
# ---------------------------------------------------------------------------
def test_main_error_envelope_dry(monkeypatch, capsys, tmp_path):
    rc = _run_main(monkeypatch, ["error-envelope", "--providers", str(_providers_file(tmp_path))])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["evidence_status"] == "dry_run_reference_only"


def test_main_sse_fingerprint_dry(monkeypatch, capsys, tmp_path):
    rc = _run_main(monkeypatch, ["sse-fingerprint", "--providers", str(_providers_file(tmp_path))])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["evidence_status"] == "dry_run_reference_only"


# ---------------------------------------------------------------------------
# inspect requires --latest or --job-id (parser.error path)
# ---------------------------------------------------------------------------
def test_main_inspect_requires_target(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["inspect", "--runs-dir", str(tmp_path / "runs")])


def test_main_no_command_errors(monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, [])


def test_main_unknown_command_errors(monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["bogus-command"])


# ---------------------------------------------------------------------------
# baseline-inspect / baseline-versions via main()
# ---------------------------------------------------------------------------
def _baseline(tmp_path: Path, bid="OFFICIAL-X"):
    bdir = tmp_path / "baselines" / bid
    bdir.mkdir(parents=True)
    (bdir / "baseline.json").write_text(json.dumps({
        "baseline_id": bid, "schema_version": "claude_baseline_v1", "sample_count": 5,
        "protocol": {}, "behavior": {},
    }), encoding="utf-8")
    return tmp_path / "baselines"


def test_main_baseline_inspect(monkeypatch, capsys, tmp_path):
    bdir = _baseline(tmp_path)
    rc = _run_main(monkeypatch, ["baseline-inspect", "--baseline-id", "OFFICIAL-X",
                                 "--baselines-dir", str(bdir)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["baseline_id"] == "OFFICIAL-X"


def test_main_baseline_versions_legacy(monkeypatch, capsys, tmp_path):
    bdir = _baseline(tmp_path)
    rc = _run_main(monkeypatch, ["baseline-versions", "--baseline-id", "OFFICIAL-X",
                                 "--baselines-dir", str(bdir)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["baseline_id"] == "OFFICIAL-X"


# ---------------------------------------------------------------------------
# run subcommand (dry) via main() — exercises the run subparser wiring
# ---------------------------------------------------------------------------
def test_main_run_dry(monkeypatch, capsys, tmp_path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"tasks": [
        {"id": "t1", "category": "C", "prompt": "hi", "scoring_type": "manual", "difficulty": "easy"},
    ]}), encoding="utf-8")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "job_id_prefix": "MAINJOB", "tasks_file": str(tasks),
        "providers_file": str(_providers_file(tmp_path)),
        "benchmark_mode": "custom", "live_provider": False, "max_tasks": 1,
    }), encoding="utf-8")
    rc = _run_main(monkeypatch, ["run", "--job", str(job),
                                 "--runs-dir", str(tmp_path / "runs"),
                                 "--run-id", "main_run_1", "--skip-trace-evaluation"])
    assert rc in (0, 2)
    assert (tmp_path / "runs" / "main_run_1" / "state.json").exists()
