"""run_job orchestration integration (R17) — the full eval loop, dry-run.

run_job is the biggest uncovered block in eval_cli: it loads a job, selects
tasks, calls the tested + judge models per task, rule/judge-scores, writes
records, and runs the quality gate. With live=False every model call uses the
no-network dry completion, so the whole loop runs offline. We drive it with
job/tasks/providers fixtures (absolute paths) and assert the run artifacts.
"""
from __future__ import annotations

import argparse
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


def _tasks_file(tmp_path: Path) -> Path:
    tasks = {"tasks": [
        {"id": "t1", "category": "Reasoning", "prompt": "What is 2+2?",
         "scoring_type": "json_exact", "expected_json": {"answer": 4}, "difficulty": "easy"},
        {"id": "t2", "category": "QA", "prompt": "Name a color.",
         "scoring_type": "manual", "difficulty": "easy"},
    ]}
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(tasks), encoding="utf-8")
    return p


def _job_file(tmp_path: Path, **over) -> Path:
    job = {
        "job_id_prefix": "TESTJOB",
        "tasks_file": str(_tasks_file(tmp_path)),
        "providers_file": str(_providers_file(tmp_path)),
        "benchmark_mode": "custom",
        "live_provider": False,
        "max_tasks": 2,
    }
    job.update(over)
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job), encoding="utf-8")
    return p


def _ns(**kw):
    base = dict(job=None, providers=None, live=False, run_id=None, runs_dir=None,
                tested_max_tokens=None, judge_max_tokens=None, timeout=None,
                retries=None, require_go=False, campaign_id=None, campaigns_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _no_secrets(monkeypatch):
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)


# placeholder-r17


def test_run_job_dry_run_end_to_end(tmp_path, capsys):
    runs = tmp_path / "runs"
    args = _ns(job=str(_job_file(tmp_path)), runs_dir=str(runs), run_id="run_test_1", live=False)
    rc = E.run_job(args)
    assert rc in (0, 2)  # 0 normally; 2 only if require_go and decision != GO
    out = json.loads(capsys.readouterr().out)
    assert out["job_id"] == "run_test_1"
    assert out["run_dir"]
    run_dir = runs / "run_test_1"
    # core artifacts written by the loop
    assert (run_dir / "state.json").exists()
    assert (run_dir / "run_records.jsonl").exists()
    assert (run_dir / "providers.redacted.json").exists()
    # per-task tested responses written
    assert (run_dir / "responses" / "tested" / "t1.txt").exists()
    # records cover both tasks
    records = [json.loads(x) for x in
               (run_dir / "run_records.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert len(records) == 2
    task_ids = {r["task"]["id"] for r in records}
    assert task_ids == {"t1", "t2"}


def test_run_job_redacted_providers_no_key(tmp_path, capsys):
    runs = tmp_path / "runs"
    args = _ns(job=str(_job_file(tmp_path)), runs_dir=str(runs), run_id="run_test_2", live=False)
    E.run_job(args)
    redacted = (runs / "run_test_2" / "providers.redacted.json").read_text(encoding="utf-8")
    # the redacted snapshot reports env var names, never raw secret values
    assert "TESTED_KEY" in redacted
    assert "sk-" not in redacted


def test_run_job_zero_tasks_raises(tmp_path):
    runs = tmp_path / "runs"
    # max_tasks 0 with an empty task set -> zero selected
    empty_tasks = tmp_path / "empty.json"
    empty_tasks.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    job = _job_file(tmp_path, tasks_file=str(empty_tasks))
    args = _ns(job=str(job), runs_dir=str(runs), run_id="run_empty", live=False)
    with pytest.raises(ValueError, match="zero tasks"):
        E.run_job(args)


def test_run_job_state_completed_or_partial(tmp_path, capsys):
    runs = tmp_path / "runs"
    args = _ns(job=str(_job_file(tmp_path)), runs_dir=str(runs), run_id="run_test_3", live=False)
    E.run_job(args)
    state = json.loads((runs / "run_test_3" / "state.json").read_text(encoding="utf-8"))
    # the gate may degrade to 'partial' under dry-run, but the loop ran to a terminal state
    assert state["status"] in ("completed", "partial")
    assert state.get("completed_at")
    assert "final_decision" in state


# ---------------------------------------------------------------------------
# export_job: run -> export acceptance pack -> the pack verifies (round-trip)
# ---------------------------------------------------------------------------
def test_export_job_produces_verifiable_pack(tmp_path, capsys):
    import acceptance_pack as AP
    runs = tmp_path / "runs"
    E.run_job(_ns(job=str(_job_file(tmp_path)), runs_dir=str(runs), run_id="run_exp", live=False))
    capsys.readouterr()
    export_args = argparse.Namespace(runs_dir=str(runs), latest=False, job_id="run_exp",
                                     include_raw=False)
    rc = E.export_job(export_args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    pack = Path(out["artifact"])
    assert pack.exists() and pack.name == "acceptance_pack.zip"
    # the produced pack passes the independent integrity verifier
    result = AP.verify_acceptance_pack(pack)
    assert result["verified"] is True, result


def test_export_job_include_raw(tmp_path, capsys):
    import acceptance_pack as AP
    runs = tmp_path / "runs"
    E.run_job(_ns(job=str(_job_file(tmp_path)), runs_dir=str(runs), run_id="run_exp2", live=False))
    capsys.readouterr()
    export_args = argparse.Namespace(runs_dir=str(runs), latest=False, job_id="run_exp2",
                                     include_raw=True)
    assert E.export_job(export_args) == 0
    out = json.loads(capsys.readouterr().out)
    assert AP.verify_acceptance_pack(Path(out["artifact"]))["verified"] is True


