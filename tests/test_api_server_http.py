"""HTTP integration tests for api_server — real server, real requests.

Starts the actual BaseHTTPRequestHandler on an ephemeral port (against a temp
runs/campaigns dir) and hits GET endpoints over the loopback socket. This is
the only way to cover the do_GET routing, error mapping, and the send_json
redaction wrapper — the part that actually faces the network. We also assert
no secret leaks through the wire and that unknown/bad routes map correctly.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import api_server as S  # noqa: E402


@pytest.fixture
def server(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    campaigns = tmp_path / "campaigns"
    runs.mkdir()
    campaigns.mkdir()
    monkeypatch.setattr(S, "RUNS_DIR", runs)
    monkeypatch.setattr(S, "CAMPAIGNS_DIR", campaigns)
    monkeypatch.setattr(S, "PROVIDERS_LOCAL", tmp_path / "providers.local.json")
    monkeypatch.setattr(S, "load_local_env", lambda *a, **k: {}, raising=False)

    httpd = HTTPServer(("127.0.0.1", 0), S.Handler)
    httpd.config_write_enabled = False
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    yield {"base": f"http://127.0.0.1:{port}", "runs": runs, "campaigns": campaigns,
           "tmp": tmp_path, "httpd": httpd}
    httpd.shutdown()
    thread.join(timeout=5)


def _get(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _get_status(base, path):
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _make_run(runs: Path, run_id="run1"):
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({
        "job_id": run_id, "status": "completed", "final_decision": "GO",
        "started_at": "2026-06-28T00:00:00Z",
    }), encoding="utf-8")
    (run_dir / "run_records.jsonl").write_text(json.dumps({
        "task": {"id": "t1"}, "provider": {"id": "tested"},
        "telemetry": {"ok": True}, "scoring": {"final_score": {"score": 9.0}},
    }) + "\n", encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# jobs endpoints
# ---------------------------------------------------------------------------
def test_jobs_empty(server):
    status, body = _get(server["base"], "/api/jobs")
    assert status == 200
    assert body == {"jobs": []}


def test_jobs_lists_runs(server):
    _make_run(server["runs"], "run_a")
    _make_run(server["runs"], "run_b")
    status, body = _get(server["base"], "/api/jobs")
    assert status == 200
    ids = {j["job_id"] for j in body["jobs"]}
    assert ids == {"run_a", "run_b"}


def test_jobs_latest_empty(server):
    status, body = _get(server["base"], "/api/jobs/latest")
    assert status == 200
    assert body == {}


def test_jobs_latest_returns_run(server):
    _make_run(server["runs"], "run1")
    status, body = _get(server["base"], "/api/jobs/latest")
    assert status == 200
    assert body["job_id"] == "run1"


# placeholder-r10


# ---------------------------------------------------------------------------
# config endpoint — reflects providers.local.json, never leaks key values
# ---------------------------------------------------------------------------
def test_config_no_file(server):
    status, body = _get(server["base"], "/api/config")
    assert status == 200
    assert body["exists"] is False


def test_config_reads_and_redacts(server, monkeypatch):
    cfg = {
        "tested_model": {"provider_id": "tested", "base_url": "https://gw.x", "model": "m",
                         "protocol": "anthropic_messages", "auth_type": "x-api-key",
                         "api_key_env": "TESTED_KEY",
                         "extra_body": {"secret": "DONOTLEAK1234567890", "top_p": 0.9}},
        "judge_model": {"provider_id": "judge", "api_key_env": "JUDGE_KEY"},
    }
    server["tmp"].joinpath("providers.local.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.delenv("TESTED_KEY", raising=False)
    status, body = _get(server["base"], "/api/config")
    assert status == 200
    assert body["exists"] is True
    tested = body["providers"]["tested_model"]
    assert tested["api_key_present"] is False
    # secret must not appear anywhere in the wire response
    assert "DONOTLEAK1234567890" not in json.dumps(body, ensure_ascii=False)


# ---------------------------------------------------------------------------
# leaderboard / campaigns
# ---------------------------------------------------------------------------
def test_leaderboard_empty(server):
    status, body = _get(server["base"], "/api/leaderboard")
    assert status == 200
    assert isinstance(body, dict)


def test_leaderboard_limit_clamped(server):
    # absurd limit is clamped, not an error
    status, body = _get(server["base"], "/api/leaderboard?limit=99999")
    assert status == 200


def test_leaderboard_bad_limit_defaults(server):
    status, body = _get(server["base"], "/api/leaderboard?limit=notanumber")
    assert status == 200


def test_campaigns_empty(server):
    status, body = _get(server["base"], "/api/campaigns")
    assert status == 200
    assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# run detail
# ---------------------------------------------------------------------------
def test_job_detail(server):
    _make_run(server["runs"], "run1")
    status, body = _get(server["base"], "/api/jobs/run1")
    # handle_job_get returns a summary payload for the run
    assert status == 200
    assert isinstance(body, dict)


def test_job_detail_missing_is_404(server):
    status, raw = _get_status(server["base"], "/api/jobs/ghost_run")
    assert status == 404


def test_job_detail_traversal_rejected(server):
    # safe_run_dir guards path traversal -> not a 200 with data
    status, raw = _get_status(server["base"], "/api/jobs/..%2f..%2fetc")
    assert status in (400, 404)


# ---------------------------------------------------------------------------
# routing / error mapping
# ---------------------------------------------------------------------------
def test_unknown_api_endpoint_404(server):
    status, raw = _get_status(server["base"], "/api/does-not-exist")
    assert status == 404
    assert b"unknown api endpoint" in raw


def test_campaign_missing_404(server):
    status, raw = _get_status(server["base"], "/api/campaigns/nope/summary")
    assert status == 404


def test_post_config_forbidden_when_disabled(server):
    req = urllib.request.Request(f"{server['base']}/api/config", data=b"{}",
                                 headers={"content-type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        assert False, "expected 403"
    except urllib.error.HTTPError as exc:
        assert exc.code == 403

