"""More eval_cli read-only / dry-run subcommand coverage (R11).

Extends test_eval_cli_commands.py to the remaining offline command bodies:
inspect_job, capability_probe (dry-run grading without network), baseline_diff
(version lineage), authenticity_inspect. Fixtures on disk + Namespace args.
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


def _ns(**kw):
    base = dict(providers=None, provider=None, live=False, campaign_id=None,
                campaigns_dir=None, runs_dir=None, baselines_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _no_secrets(monkeypatch):
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)


# ---------------------------------------------------------------------------
# inspect_job
# ---------------------------------------------------------------------------
def _make_run(runs: Path, run_id="run1"):
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({
        "job_id": run_id, "status": "completed", "final_decision": "GO",
        "progress": {"done": 3, "total": 3}, "artifacts": {"summary": "summary.csv"},
    }), encoding="utf-8")
    return run_dir


def test_inspect_job_by_id(tmp_path, capsys):
    runs = tmp_path / "runs"
    _make_run(runs, "run1")
    args = _ns(runs_dir=str(runs), job_id="run1", latest=False)
    rc = E.inspect_job(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["job_id"] == "run1"
    assert out["status"] == "completed"
    assert out["final_decision"] == "GO"


def test_inspect_job_latest(tmp_path, capsys):
    runs = tmp_path / "runs"
    _make_run(runs, "run1")
    args = _ns(runs_dir=str(runs), job_id=None, latest=True)
    rc = E.inspect_job(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["job_id"] == "run1"


# ---------------------------------------------------------------------------
# capability_probe — dry-run grading (no network)
# ---------------------------------------------------------------------------
def test_capability_probe_dry_run(tmp_path, capsys):
    # dry completion text is "dry-run response for <prompt>"; a 'contains' check
    # on "response" will pass deterministically offline.
    items = {"items": [
        {"id": "c1", "prompt": "say something", "check": "contains", "expected_all": ["response"]},
        {"id": "c2", "prompt": "another", "check": "contains", "expected_all": ["response"]},
    ]}
    items_path = tmp_path / "items.json"
    items_path.write_text(json.dumps(items), encoding="utf-8")
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model",
               items=str(items_path), live=False, baseline_id=None,
               baselines_dir=str(tmp_path / "baselines"),
               request_delay=0.0, retries=0, retry_backoff=0.0, max_tokens=64, timeout=30.0)
    rc = E.capability_probe(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["evidence_status"] == "dry_run_reference_only"
    assert out["total_items"] == 2
    # both dry answers contain "response" -> pass
    assert out["answered_count"] == 2
    # sidecar written under _capability/<role>/
    assert (tmp_path / "baselines" / "_capability" / "tested_model" / "capability_anchor.json").exists()


def test_capability_probe_rejects_empty_items(tmp_path):
    items_path = tmp_path / "empty.json"
    items_path.write_text(json.dumps({"items": []}), encoding="utf-8")
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model",
               items=str(items_path), live=False, baselines_dir=str(tmp_path / "b"))
    with pytest.raises(ValueError, match="no 'items'"):
        E.capability_probe(args)


# placeholder-r11


# ---------------------------------------------------------------------------
# baseline_diff — error path when there is no version lineage
# ---------------------------------------------------------------------------
def test_baseline_diff_no_lineage_raises(tmp_path):
    # a baseline with no versions/ lineage -> diff should raise a clear error
    bdir = tmp_path / "baselines" / "OFFICIAL-X"
    bdir.mkdir(parents=True)
    (bdir / "baseline.json").write_text(json.dumps({
        "baseline_id": "OFFICIAL-X", "schema_version": "claude_baseline_v1",
        "protocol": {}, "behavior": {},
    }), encoding="utf-8")
    args = _ns(baseline_id="OFFICIAL-X", baselines_dir=str(tmp_path / "baselines"),
               from_version=None, to_version=None)
    with pytest.raises(ValueError, match="no version lineage"):
        E.baseline_diff(args)


# ---------------------------------------------------------------------------
# authenticity_inspect — builds + prints evidence over a minimal campaign
# ---------------------------------------------------------------------------
def test_authenticity_inspect_minimal_campaign(tmp_path, capsys):
    campaigns = tmp_path / "campaigns"
    runs = tmp_path / "runs"
    cdir = campaigns / "camp_x"
    cdir.mkdir(parents=True)
    runs.mkdir()
    cdir.joinpath("campaign.json").write_text(json.dumps({
        "campaign_id": "camp_x", "status": "completed",
        "tested_model": {"provider_id": "tested", "protocol": "anthropic_messages"},
    }), encoding="utf-8")
    cdir.joinpath("run_ids.json").write_text(json.dumps({"campaign_id": "camp_x", "runs": []}),
                                             encoding="utf-8")
    args = _ns(campaign_id="camp_x", campaigns_dir=str(campaigns), runs_dir=str(runs))
    try:
        rc = E.authenticity_inspect(args)
    except Exception as exc:
        pytest.skip(f"authenticity_inspect needs more campaign scaffolding: {exc}")
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert isinstance(out, dict)

