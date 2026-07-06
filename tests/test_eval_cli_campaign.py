"""run_campaign orchestration + campaign management commands (R18).

run_campaign creates a campaign and runs N rounds, each delegating to run_job
in-process. With live=False the whole thing runs offline (dry completions).
We drive a 1-round dry campaign end to end and then exercise the read-only
campaign management commands against the produced campaign.
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
        {"id": "t1", "category": "Reasoning", "prompt": "2+2?",
         "scoring_type": "json_exact", "expected_json": {"answer": 4}, "difficulty": "easy"},
    ]}
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(tasks), encoding="utf-8")
    return p


def _job_file(tmp_path: Path) -> Path:
    job = {
        "job_id_prefix": "CAMPJOB",
        "tasks_file": str(_tasks_file(tmp_path)),
        "providers_file": str(_providers_file(tmp_path)),
        "benchmark_mode": "custom",
        "live_provider": False,
        "max_tasks": 1,
    }
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job), encoding="utf-8")
    return p


def _campaign_ns(tmp_path, **over):
    base = dict(
        job=str(_job_file(tmp_path)), providers=None, live=False, resume=False,
        repeat=1, campaign_id="camp_test",
        campaigns_dir=str(tmp_path / "campaigns"), runs_dir=str(tmp_path / "runs"),
        timeout=None, tested_max_tokens=None, judge_max_tokens=None,
        max_concurrency=None, retries=None, retry_backoff=None,
        skip_trace_evaluation=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _no_secrets(monkeypatch):
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)


# placeholder-r18


def test_run_campaign_dry_one_round(tmp_path, capsys):
    args = _campaign_ns(tmp_path)
    rc = E.run_campaign(args)
    assert rc == 0
    # run_campaign prints run_job's result then the campaign summary; assert on
    # the produced artifacts rather than parsing the (multi-object) stdout.
    capsys.readouterr()
    camp_dir = tmp_path / "campaigns" / "camp_test"
    assert (camp_dir / "campaign.json").exists()
    assert (camp_dir / "run_ids.json").exists()
    assert (camp_dir / "summary.json").exists()
    campaign = json.loads((camp_dir / "campaign.json").read_text(encoding="utf-8"))
    assert campaign["campaign_id"] == "camp_test"
    assert campaign["status"] in ("completed", "partial")
    # one round recorded in the run index
    run_index = json.loads((camp_dir / "run_ids.json").read_text(encoding="utf-8"))
    assert len(run_index["runs"]) == 1


def test_run_campaign_requires_repeat_when_new(tmp_path):
    args = _campaign_ns(tmp_path, repeat=None)
    with pytest.raises(ValueError, match="repeat is required"):
        E.run_campaign(args)


def test_run_campaign_repeat_must_be_positive(tmp_path):
    args = _campaign_ns(tmp_path, repeat=0)
    with pytest.raises(ValueError, match="repeat must be >= 1"):
        E.run_campaign(args)


def test_run_campaign_resume_requires_id(tmp_path):
    args = _campaign_ns(tmp_path, resume=True, campaign_id=None)
    with pytest.raises(ValueError, match="resume requires"):
        E.run_campaign(args)


# ---------------------------------------------------------------------------
# campaign management read-only commands over the produced campaign
# ---------------------------------------------------------------------------
def _mgmt_ns(tmp_path, **over):
    base = dict(campaign_id="camp_test", campaigns_dir=str(tmp_path / "campaigns"),
                runs_dir=str(tmp_path / "runs"))
    base.update(over)
    return argparse.Namespace(**base)


def test_campaign_status_after_run(tmp_path, capsys):
    E.run_campaign(_campaign_ns(tmp_path))
    capsys.readouterr()  # drain
    rc = E.campaign_status(_mgmt_ns(tmp_path))
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out.get("campaign_id") == "camp_test" or "status" in out


def test_campaign_inspect_after_run(tmp_path, capsys):
    E.run_campaign(_campaign_ns(tmp_path))
    capsys.readouterr()
    rc = E.campaign_inspect(_mgmt_ns(tmp_path))
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert isinstance(out, dict)


def test_campaign_list_after_run(tmp_path, capsys):
    E.run_campaign(_campaign_ns(tmp_path))
    capsys.readouterr()
    args = argparse.Namespace(campaigns_dir=str(tmp_path / "campaigns"),
                              runs_dir=str(tmp_path / "runs"))
    rc = E.campaign_list(args)
    assert rc == 0
    flat = capsys.readouterr().out
    assert "camp_test" in flat


# ---------------------------------------------------------------------------
# export_campaign: run -> export -> the produced pack verifies (round-trip)
# ---------------------------------------------------------------------------
def test_export_campaign_verifies(tmp_path, capsys):
    import campaigns as C
    import acceptance_pack as AP
    E.run_campaign(_campaign_ns(tmp_path))
    capsys.readouterr()
    camp_dir = tmp_path / "campaigns" / "camp_test"
    runs_dir = tmp_path / "runs"
    zip_path = C.export_campaign(camp_dir, runs_dir)
    assert zip_path.exists() and zip_path.name == "acceptance_pack.zip"
    result = AP.verify_acceptance_pack(zip_path)
    assert result["verified"] is True, result


def test_export_campaign_include_raw(tmp_path, capsys):
    import campaigns as C
    import acceptance_pack as AP
    E.run_campaign(_campaign_ns(tmp_path))
    capsys.readouterr()
    camp_dir = tmp_path / "campaigns" / "camp_test"
    zip_path = C.export_campaign(camp_dir, tmp_path / "runs", include_raw=True)
    assert AP.verify_acceptance_pack(zip_path)["verified"] is True


# ---------------------------------------------------------------------------
# campaign_leaderboard over a produced campaign
# ---------------------------------------------------------------------------
def test_campaign_leaderboard_after_run(tmp_path, capsys):
    import campaigns as C
    E.run_campaign(_campaign_ns(tmp_path))
    capsys.readouterr()
    board = C.campaign_leaderboard(tmp_path / "campaigns", tmp_path / "runs",
                                   include_dry_run=True, persist_refresh=False)
    assert isinstance(board, dict)
    # the produced campaign should surface somewhere in the payload
    assert "camp_test" in json.dumps(board, ensure_ascii=False)


# ---------------------------------------------------------------------------
# campaign_retest
# ---------------------------------------------------------------------------
def _retest_ns(tmp_path, **over):
    base = dict(campaign_id="camp_test", campaigns_dir=str(tmp_path / "campaigns"),
                runs_dir=str(tmp_path / "runs"), new_campaign_id=None,
                job=str(_job_file(tmp_path)), providers=None, repeat=1, live=False,
                force=False, dry_run=True, timeout=None, tested_max_tokens=None,
                judge_max_tokens=None, max_concurrency=None, retries=None,
                retry_backoff=None, skip_trace_evaluation=True)
    base.update(over)
    return argparse.Namespace(**base)


def test_campaign_retest_skips_non_retest_without_force(tmp_path, capsys):
    E.run_campaign(_campaign_ns(tmp_path))
    capsys.readouterr()
    rc = E.campaign_retest(_retest_ns(tmp_path, force=False))
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # a non-RETEST campaign is skipped unless --force
    assert out["status"] == "skipped"
    assert "RETEST" in out["reason"]


def test_campaign_retest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="campaign not found"):
        E.campaign_retest(_retest_ns(tmp_path, campaign_id="ghost"))


def test_campaign_retest_force_runs_new_campaign(tmp_path, capsys):
    E.run_campaign(_campaign_ns(tmp_path))
    capsys.readouterr()
    rc = E.campaign_retest(_retest_ns(tmp_path, force=True, new_campaign_id="camp_retest"))
    assert rc in (0, 1)
    # a new campaign dir was created by the forced retest
    assert (tmp_path / "campaigns" / "camp_retest" / "campaign.json").exists()

