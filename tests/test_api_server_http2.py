"""api_server POST + remaining-route HTTP integration (R15).

Extends the HTTP harness to the write path and the campaign/static routes:
POST /api/config (forbidden vs enabled), campaign summary/runs sub-routes,
static serving + traversal guard.
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
def wserver(tmp_path, monkeypatch):
    """Server with config writes ENABLED and all on-disk paths redirected to tmp."""
    runs = tmp_path / "runs"
    campaigns = tmp_path / "campaigns"
    web = tmp_path / "web"
    runs.mkdir(); campaigns.mkdir(); web.mkdir()
    (web / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(S, "RUNS_DIR", runs)
    monkeypatch.setattr(S, "CAMPAIGNS_DIR", campaigns)
    monkeypatch.setattr(S, "WEB_DIR", web)
    monkeypatch.setattr(S, "PROVIDERS_LOCAL", tmp_path / "providers.local.json")
    monkeypatch.setattr(S, "LOCAL_SECRETS", tmp_path / "local_secrets.env")
    monkeypatch.setattr(S, "load_local_env", lambda *a, **k: {}, raising=False)

    httpd = HTTPServer(("127.0.0.1", 0), S.Handler)
    httpd.config_write_enabled = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield {"base": f"http://127.0.0.1:{httpd.server_address[1]}",
           "runs": runs, "campaigns": campaigns, "web": web, "tmp": tmp_path}
    httpd.shutdown()
    thread.join(timeout=5)


def _post(base, path, payload):
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(payload).encode("utf-8"),
                                 headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _get(base, path):
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# ---------------------------------------------------------------------------
# POST /api/config — write path enabled
# ---------------------------------------------------------------------------
def test_post_config_writes_and_redacts(wserver, monkeypatch):
    monkeypatch.delenv("TESTED_MODEL_API_KEY", raising=False)
    payload = {"providers": {
        "tested_model": {"provider_id": "tested", "base_url": "https://gw.x", "model": "claude-opus-4-6",
                         "protocol": "anthropic_messages", "auth_type": "x-api-key",
                         "reasoning_effort": "", "api_key": "sk-SECRETVALUE1234567890"},
        "judge_model": {"provider_id": "judge", "base_url": "https://gw.x", "model": "gpt-5.5",
                        "protocol": "openai_chat", "auth_type": "bearer",
                        "reasoning_effort": "high", "api_key": "sk-JUDGEKEY1234567890"},
    }}
    status, body = _post(wserver["base"], "/api/config", payload)
    assert status == 200
    # providers.local.json written, secret stored in env file (not the json)
    saved = json.loads((wserver["tmp"] / "providers.local.json").read_text(encoding="utf-8"))
    assert saved["tested_model"]["api_key_env"] == "TESTED_MODEL_API_KEY"
    assert "sk-SECRETVALUE1234567890" not in json.dumps(saved)
    # the response is the sanitized config — no raw key over the wire
    assert "sk-SECRETVALUE1234567890" not in json.dumps(body)
    # env file got the secret
    env_text = (wserver["tmp"] / "local_secrets.env").read_text(encoding="utf-8")
    assert "TESTED_MODEL_API_KEY=sk-SECRETVALUE1234567890" in env_text


def test_post_config_rejects_missing_providers(wserver):
    status, raw = _post(wserver["base"], "/api/config", {"nope": 1})
    assert status == 400


def test_post_config_rejects_bad_protocol(wserver):
    payload = {"providers": {
        "tested_model": {"provider_id": "t", "base_url": "u", "model": "m",
                         "protocol": "grpc", "auth_type": "bearer"},
        "judge_model": {"provider_id": "j", "base_url": "u", "model": "m",
                        "protocol": "openai_chat", "auth_type": "bearer"},
    }}
    status, raw = _post(wserver["base"], "/api/config", payload)
    assert status == 400


# placeholder-r15


# ---------------------------------------------------------------------------
# campaign sub-routes
# ---------------------------------------------------------------------------
def _make_campaign(campaigns: Path, runs: Path, cid="camp_x"):
    cdir = campaigns / cid
    cdir.mkdir(parents=True)
    cdir.joinpath("campaign.json").write_text(json.dumps({
        "campaign_id": cid, "status": "completed",
        "tested_model": {"provider_id": "tested", "protocol": "anthropic_messages"},
    }), encoding="utf-8")
    cdir.joinpath("run_ids.json").write_text(json.dumps({"campaign_id": cid, "runs": []}),
                                             encoding="utf-8")
    return cdir


def test_campaign_summary_route(wserver):
    _make_campaign(wserver["campaigns"], wserver["runs"], "camp_x")
    status, raw = _get(wserver["base"], "/api/campaigns/camp_x/summary")
    assert status == 200
    body = json.loads(raw)
    assert body.get("campaign_id") == "camp_x"


def test_campaign_runs_route(wserver):
    _make_campaign(wserver["campaigns"], wserver["runs"], "camp_y")
    status, raw = _get(wserver["base"], "/api/campaigns/camp_y/runs")
    assert status == 200
    body = json.loads(raw)
    assert "runs" in body


def test_campaign_missing_404(wserver):
    status, raw = _get(wserver["base"], "/api/campaigns/ghost/summary")
    assert status == 404


def test_campaigns_list_and_latest(wserver):
    _make_campaign(wserver["campaigns"], wserver["runs"], "camp_z")
    status, raw = _get(wserver["base"], "/api/campaigns")
    assert status == 200
    status2, raw2 = _get(wserver["base"], "/api/campaigns/latest")
    assert status2 == 200


# ---------------------------------------------------------------------------
# static serving + traversal guard
# ---------------------------------------------------------------------------
def test_serve_index_html(wserver):
    status, raw = _get(wserver["base"], "/")
    assert status == 200
    assert b"ok" in raw


def test_serve_static_traversal_rejected(wserver):
    # a path escaping WEB_DIR must not return file contents
    status, raw = _get(wserver["base"], "/../../etc/passwd")
    assert status in (400, 404)


def test_serve_unknown_static_404(wserver):
    status, raw = _get(wserver["base"], "/nope.js")
    assert status == 404


# ---------------------------------------------------------------------------
# config GET reflects what POST wrote
# ---------------------------------------------------------------------------
def test_config_get_after_post(wserver, monkeypatch):
    monkeypatch.delenv("TESTED_MODEL_API_KEY", raising=False)
    payload = {"providers": {
        "tested_model": {"provider_id": "tested", "base_url": "https://gw.x", "model": "m",
                         "protocol": "anthropic_messages", "auth_type": "x-api-key",
                         "reasoning_effort": ""},
        "judge_model": {"provider_id": "judge", "base_url": "https://gw.x", "model": "m",
                        "protocol": "openai_chat", "auth_type": "bearer", "reasoning_effort": ""},
    }}
    _post(wserver["base"], "/api/config", payload)
    status, raw = _get(wserver["base"], "/api/config")
    assert status == 200
    body = json.loads(raw)
    assert body["exists"] is True
    assert body["providers"]["tested_model"]["model"] == "m"

