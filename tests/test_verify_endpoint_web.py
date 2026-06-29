"""HTTP integration tests for the web authenticity-verify endpoint.

Covers the R-001 ban-avoidance gates on POST /api/authenticity/verify:
dry-run by default, live needs risk_ack + the server switch, request_delay is
floored, and no key value ever appears in the wire response. dry-run does NOT
hit the network (verify_core short-circuits without live), so these run offline.
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


def _make_server(tmp_path, monkeypatch, *, live_enabled: bool):
    runs = tmp_path / "runs"; runs.mkdir()
    campaigns = tmp_path / "campaigns"; campaigns.mkdir()
    monkeypatch.setattr(S, "RUNS_DIR", runs)
    monkeypatch.setattr(S, "CAMPAIGNS_DIR", campaigns)

    httpd = HTTPServer(("127.0.0.1", 0), S.Handler)
    httpd.config_write_enabled = False
    httpd.authenticity_live_enabled = live_enabled
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    return httpd, thread, f"http://127.0.0.1:{port}"


@pytest.fixture
def dry_server(tmp_path, monkeypatch):
    httpd, thread, base = _make_server(tmp_path, monkeypatch, live_enabled=False)
    yield base
    httpd.shutdown(); thread.join(timeout=5)


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    httpd, thread, base = _make_server(tmp_path, monkeypatch, live_enabled=True)
    yield base
    httpd.shutdown(); thread.join(timeout=5)


def _post(base, body):
    req = urllib.request.Request(f"{base}/api/authenticity/verify",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# baseline_id that ships with the repo (live_observed, real official opus-4-6).
BASELINE = "OFFICIAL-CLAUDE-OPUS46"
SUSPECT = {"base_url": "https://gw.example.com", "model": "claude-opus-4-6",
           "protocol": "anthropic_messages", "auth_type": "x-api-key",
           "baseline_id": BASELINE}


def test_dry_run_default_no_live(dry_server):
    status, body = _post(dry_server, dict(SUSPECT))
    assert status == 200
    assert body["live"] is False
    # dry-run can't conclude authenticity
    assert body["verdict"]["verdict"] == "insufficient_evidence"
    assert "dry-run" in (body["note"] or "")


def test_live_requires_risk_ack(live_server):
    # live=true but no risk_ack -> 400
    status, body = _post(live_server, {**SUSPECT, "live": True, "api_key": "sk-throwaway"})
    assert status == 400
    assert "风险" in body["error"] or "risk" in body["error"].lower()


def test_live_requires_server_switch(dry_server):
    # risk_ack ok, but server NOT started with --enable-live-verify -> 403
    status, body = _post(dry_server, {**SUSPECT, "live": True, "risk_ack": True, "api_key": "sk-throwaway"})
    assert status == 403
    assert "enable-live-verify" in body["error"]


def test_missing_required_fields(dry_server):
    status, body = _post(dry_server, {"protocol": "anthropic_messages"})
    assert status == 400


def test_no_key_leak_in_response(dry_server):
    secret = "sk-DONOTLEAK-9988776655"
    status, body = _post(dry_server, {**SUSPECT, "api_key": secret})
    assert status == 200
    assert secret not in json.dumps(body, ensure_ascii=False)


def test_request_delay_floored(monkeypatch):
    # unit-level: run_web_verify floors request_delay to >= WEB_VERIFY_MIN_DELAY.
    captured = {}

    def fake_verify_core(model, baseline, **kw):
        captured.update(kw)
        return {"verdict": "insufficient_evidence", "confidence": 0.0}

    monkeypatch.setattr(S.eval_cli, "verify_core", fake_verify_core)
    monkeypatch.setattr(S.eval_cli, "render_verdict_report", lambda v, **k: "report")
    S.run_web_verify({**SUSPECT, "request_delay": 0.1, "api_key": "sk-x"}, live=True)
    assert captured["request_delay"] >= S.WEB_VERIFY_MIN_DELAY
    # the dangerous probes must be OFF on the web path
    assert captured["with_needle"] is False
    assert captured["with_error_envelope"] is False
    assert captured["with_sse"] is False
