"""Integration tests for api_server.py report/route logic.

api_server.py serves observability over runs. The pure helpers are covered in
test_api_server_helpers.py; here we exercise the data-assembly functions that
read on-disk run state — summarize_run, list_jobs/latest_job, artifact_listing,
provider_model_name, latest_quality_gate, sanitized_config — with tmp fixtures.
No HTTP server is started; we call the functions the Handler delegates to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import api_server as S  # noqa: E402


def _run(tmp_path: Path, run_id="run1", records=None, status="completed") -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({
        "job_id": run_id, "status": status, "final_decision": "GO",
        "started_at": "2026-06-28T00:00:00Z", "progress": {"done": 3, "total": 3},
        "models": {"tested_model": {"provider_id": "tested", "model": "claude-opus-4-6"}},
    }), encoding="utf-8")
    recs = records if records is not None else [{
        "task": {"id": "t1", "category": "C", "enterprise_dimension": "D1"},
        "provider": {"id": "tested", "model_returned": "claude-opus-4-6"},
        "telemetry": {"ok": True, "first_content_token_ms": 800, "error": None},
        "scoring": {"final_score": {"score": 9.0}},
    }]
    with (run_dir / "run_records.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return run_dir


# ---------------------------------------------------------------------------
# summarize_run
# ---------------------------------------------------------------------------
def test_summarize_run_basic(tmp_path):
    run_dir = _run(tmp_path)
    out = S.summarize_run(run_dir)
    assert out["metrics"]["sample_count"] == 1
    assert out["metrics"]["ok_count"] == 1
    assert out["metrics"]["success_rate"] == 1.0
    assert out["metrics"]["average_score_0_10"] == 9.0
    assert out["state"]["final_decision"] == "GO"
    assert len(out["samples"]) == 1


def test_summarize_run_redacts_sample_error(tmp_path):
    recs = [{
        "task": {"id": "t1"}, "provider": {"id": "tested"},
        "telemetry": {"ok": False, "error": "auth failed key sk-LEAKED1234567890ABC"},
        "scoring": {},
    }]
    run_dir = _run(tmp_path, records=recs)
    out = S.summarize_run(run_dir)
    assert out["metrics"]["ok_count"] == 0
    assert "sk-LEAKED1234567890ABC" not in json.dumps(out, ensure_ascii=False)


def test_summarize_run_mixed_outcomes(tmp_path):
    recs = [
        {"task": {"id": "t1"}, "provider": {"id": "tested"},
         "telemetry": {"ok": True, "total_ms": 500}, "scoring": {"final_score": {"score": 8.0}}},
        {"task": {"id": "t2"}, "provider": {"id": "tested"},
         "telemetry": {"ok": False, "error": "boom"}, "scoring": {}},
    ]
    run_dir = _run(tmp_path, records=recs)
    out = S.summarize_run(run_dir)
    assert out["metrics"]["sample_count"] == 2
    assert out["metrics"]["failure_count"] == 1
    assert out["metrics"]["success_rate"] == 0.5


# placeholder-r8


# ---------------------------------------------------------------------------
# list_jobs / latest_job (monkeypatch RUNS_DIR)
# ---------------------------------------------------------------------------
def test_list_jobs_and_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RUNS_DIR", tmp_path)
    _run(tmp_path, run_id="run_a")
    _run(tmp_path, run_id="run_b")
    jobs = S.list_jobs()
    assert len(jobs) == 2
    ids = {j["job_id"] for j in jobs}
    assert ids == {"run_a", "run_b"}
    assert all(j["status"] == "completed" for j in jobs)
    latest = S.latest_job()
    assert latest is not None and latest["job_id"] in ids


def test_list_jobs_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RUNS_DIR", tmp_path / "nonexistent")
    assert S.list_jobs() == []
    assert S.latest_job() is None


def test_list_jobs_skips_non_run_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RUNS_DIR", tmp_path)
    _run(tmp_path, run_id="real_run")
    (tmp_path / "stray_dir").mkdir()        # no state.json -> skipped
    (tmp_path / "loose_file.txt").write_text("x", encoding="utf-8")
    jobs = S.list_jobs()
    assert [j["job_id"] for j in jobs] == ["real_run"]


# ---------------------------------------------------------------------------
# provider_model_name / latest_quality_gate
# ---------------------------------------------------------------------------
def test_provider_model_name_from_state():
    state = {"models": {"tested_model": {"provider_id": "tested", "model": "claude-opus-4-6"}}}
    assert S.provider_model_name(state, [], "tested") == "claude-opus-4-6"


def test_provider_model_name_from_records():
    records = [{"provider": {"id": "tested", "claimed_model": "claude-x"}}]
    assert S.provider_model_name({}, records, "tested") == "claude-x"


def test_provider_model_name_fallback_to_id():
    assert S.provider_model_name({}, [], "tested") == "tested"


def test_latest_quality_gate_none_when_absent(tmp_path):
    run_dir = _run(tmp_path)
    assert S.latest_quality_gate(run_dir) is None


def test_latest_quality_gate_reads_record(tmp_path):
    run_dir = _run(tmp_path)
    gate_dir = run_dir / "quality_gates" / "gate_1"
    gate_dir.mkdir(parents=True)
    (gate_dir / "quality_gate_records.jsonl").write_text(
        json.dumps({"decision": "GO", "blockers": []}) + "\n", encoding="utf-8")
    gate = S.latest_quality_gate(run_dir)
    assert gate is not None
    assert gate["gate_id"] == "gate_1"


# ---------------------------------------------------------------------------
# artifact_listing — includes acceptance pack verification
# ---------------------------------------------------------------------------
def test_artifact_listing_lists_files(tmp_path):
    (tmp_path / "summary.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "scores.json").write_text("{}", encoding="utf-8")
    (tmp_path / "subdir").mkdir()           # dirs excluded
    rows = S.artifact_listing(tmp_path)
    names = {r["name"] for r in rows}
    assert names == {"summary.csv", "scores.json"}
    assert all("bytes" in r for r in rows)


def test_artifact_listing_verifies_acceptance_pack(tmp_path):
    import zipfile, hashlib
    pack = tmp_path / "acceptance_pack.zip"
    manifest = b'{"v":1}'
    with zipfile.ZipFile(pack, "w") as zf:
        zf.writestr("acceptance_manifest.json", manifest)
        digest = hashlib.sha256(manifest).hexdigest()
        zf.writestr("checksums.sha256", f"{digest}  acceptance_manifest.json".encode("utf-8"))
    rows = S.artifact_listing(tmp_path)
    pack_row = next(r for r in rows if r["name"] == "acceptance_pack.zip")
    assert pack_row["verification"]["verified"] is True


def test_artifact_listing_missing_root(tmp_path):
    assert S.artifact_listing(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# sanitized_config — never leaks key values, reports presence
# ---------------------------------------------------------------------------
def test_sanitized_config_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "PROVIDERS_LOCAL", tmp_path / "nope.json")
    monkeypatch.setattr(S, "load_local_env", lambda *a, **k: {}, raising=False)
    out = S.sanitized_config()
    assert out["exists"] is False


def test_sanitized_config_reads_and_redacts(tmp_path, monkeypatch):
    cfg = {
        "tested_model": {"provider_id": "tested", "base_url": "https://gw.x", "model": "m",
                         "protocol": "anthropic_messages", "auth_type": "x-api-key",
                         "api_key_env": "TESTED_KEY",
                         "extra_body": {"secret": "should-be-hidden", "top_p": 0.9}},
        "judge_model": {"provider_id": "judge", "api_key_env": "JUDGE_KEY"},
    }
    p = tmp_path / "providers.local.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(S, "PROVIDERS_LOCAL", p)
    monkeypatch.setattr(S, "load_local_env", lambda *a, **k: {}, raising=False)
    monkeypatch.delenv("TESTED_KEY", raising=False)
    out = S.sanitized_config()
    assert out["exists"] is True
    tested = out["providers"]["tested_model"]
    assert tested["api_key_env"] == "TESTED_KEY"
    assert tested["api_key_present"] is False
    # extra_body secret redacted, non-secret kept
    assert tested["extra_body"]["secret"] == "[REDACTED]"
    assert tested["extra_body"]["top_p"] == 0.9
    assert "should-be-hidden" not in json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------------------
# probe_config_role — the no-network early returns
# ---------------------------------------------------------------------------
def test_probe_config_role_rejects_bad_role():
    with pytest.raises(ValueError, match="role must be"):
        S.probe_config_role("bogus")


def test_probe_config_role_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "PROVIDERS_LOCAL", tmp_path / "nope.json")
    monkeypatch.setattr(S, "load_local_env", lambda *a, **k: {}, raising=False)
    with pytest.raises(FileNotFoundError):
        S.probe_config_role("tested_model")


def test_probe_config_role_no_key_returns_present_false(tmp_path, monkeypatch):
    cfg = {"tested_model": {"provider_id": "tested", "base_url": "https://gw.x",
                            "api_key_env": "TESTED_KEY", "auth_type": "x-api-key"},
           "judge_model": {"provider_id": "judge", "api_key_env": "JUDGE_KEY"}}
    p = tmp_path / "providers.local.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(S, "PROVIDERS_LOCAL", p)
    monkeypatch.setattr(S, "load_local_env", lambda *a, **k: {}, raising=False)
    monkeypatch.delenv("TESTED_KEY", raising=False)
    out = S.probe_config_role("tested_model")
    assert out["api_key_present"] is False
    assert "missing environment variable" in out["error"]


def test_probe_config_role_success_offline(tmp_path, monkeypatch):
    # drive the full success path by stubbing http_json (no network)
    cfg = {"tested_model": {"provider_id": "tested", "base_url": "https://gw.x",
                            "api_key_env": "TESTED_KEY", "auth_type": "x-api-key",
                            "protocol": "anthropic_messages", "model": "claude-opus-4-6"},
           "judge_model": {"provider_id": "judge", "api_key_env": "JUDGE_KEY"}}
    p = tmp_path / "providers.local.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(S, "PROVIDERS_LOCAL", p)
    monkeypatch.setattr(S, "load_local_env", lambda *a, **k: {}, raising=False)
    monkeypatch.setenv("TESTED_KEY", "sk-test")

    def fake_http_json(method, url, *, headers, payload=None, timeout=30.0):
        return 200, {"data": [{"id": "claude-opus-4-6"}, {"id": "text-embedding-3"}]}, 12.5
    monkeypatch.setattr(S, "http_json", fake_http_json)

    out = S.probe_config_role("tested_model", include_reasoning=False)
    assert out["api_key_present"] is True
    assert out["models_ok"] is True
    assert out["model_count"] == 2
    assert "claude-opus-4-6" in out["text_models"]
    assert "text-embedding-3" not in out["text_models"]  # embedding excluded


def test_probe_config_role_models_http_error(tmp_path, monkeypatch):
    import urllib.error, io
    cfg = {"tested_model": {"provider_id": "tested", "base_url": "https://gw.x",
                            "api_key_env": "TESTED_KEY", "auth_type": "bearer", "model": "m"},
           "judge_model": {"provider_id": "judge", "api_key_env": "JUDGE_KEY"}}
    p = tmp_path / "providers.local.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(S, "PROVIDERS_LOCAL", p)
    monkeypatch.setattr(S, "load_local_env", lambda *a, **k: {}, raising=False)
    monkeypatch.setenv("TESTED_KEY", "sk-test")

    def boom(method, url, *, headers, payload=None, timeout=30.0):
        raise urllib.error.HTTPError(url, 401, "unauthorized", {}, io.BytesIO(b"bad key"))
    monkeypatch.setattr(S, "http_json", boom)

    out = S.probe_config_role("tested_model", include_reasoning=False)
    assert out["models_ok"] is False
    assert out["models_status"] == 401


# ---------------------------------------------------------------------------
# probe_reasoning_efforts — offline via stubbed http_json
# ---------------------------------------------------------------------------
def test_probe_reasoning_efforts_non_openai_skips():
    out = S.probe_reasoning_efforts({"protocol": "anthropic_messages"}, "sk", "claude-opus-4-6")
    assert out["supported"] == []
    assert "only supports openai_chat" in out["skipped"]


def test_probe_reasoning_efforts_offline(monkeypatch):
    item = {"protocol": "openai_chat", "base_url": "https://gw.x", "auth_type": "bearer"}

    def fake_http_json(method, url, *, headers, payload=None, timeout=90.0):
        return 200, {"model": "gpt-5.5", "choices": [{"message": {"content": "OK"}}],
                     "usage": {"completion_tokens": 1}}, 30.0
    monkeypatch.setattr(S, "http_json", fake_http_json)

    out = S.probe_reasoning_efforts(item, "sk", "gpt-5.5")
    assert out["probe_model"] == "gpt-5.5"
    assert len(out["supported"]) == len(S.REASONING_PROBE_VALUES)
    assert out["rejected"] == []


def test_probe_reasoning_efforts_rejects(monkeypatch):
    import urllib.error, io
    item = {"protocol": "openai_chat", "base_url": "https://gw.x", "auth_type": "bearer"}

    def boom(method, url, *, headers, payload=None, timeout=90.0):
        raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(b"effort not supported"))
    monkeypatch.setattr(S, "http_json", boom)

    out = S.probe_reasoning_efforts(item, "sk", "gpt-5.5")
    assert out["supported"] == []
    assert len(out["rejected"]) == len(S.REASONING_PROBE_VALUES)
    assert all(r["status"] == 400 for r in out["rejected"])

