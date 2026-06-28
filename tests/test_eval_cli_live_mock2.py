"""More eval_cli --live coverage via MockTransport (R14).

Drives the remaining live probe bodies offline:
- sse_fingerprint --live: a streaming SSE mock the iter_lines parser consumes.
- needle --live: an anthropic-JSON mock that echoes the planted AUTH_CANARY so
  needle_recall scores a genuine context_ok (no truncation).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx
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
    base = dict(providers=None, provider="tested_model", live=True, campaign_id=None,
                campaigns_dir=None, runs_dir=None, baselines_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _patch_client(monkeypatch, handler):
    real_client = httpx.Client

    class _Factory:
        def __init__(self, *a, **k):
            self._c = real_client(transport=httpx.MockTransport(handler))
        def __enter__(self):
            return self._c
        def __exit__(self, *a):
            self._c.close()

    monkeypatch.setattr(E.httpx, "Client", _Factory)
    monkeypatch.setenv("TESTED_KEY", "sk-test")
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)


# ---------------------------------------------------------------------------
# sse_fingerprint --live (streaming SSE)
# ---------------------------------------------------------------------------
def test_sse_fingerprint_live(tmp_path, capsys, monkeypatch):
    body = (
        b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
        b"event: content_block_start\ndata: {\"type\":\"content_block_start\"}\n\n"
        b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n"
        b"event: message_delta\ndata: {\"type\":\"message_delta\"}\n\n"
        b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    )

    def handler(request):
        def gen():
            for i in range(0, len(body), 20):
                yield body[i:i + 20]
        return httpx.Response(200, content=gen(),
                              headers={"content-type": "text/event-stream"})

    _patch_client(monkeypatch, handler)
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model", live=True,
               baselines_dir=str(tmp_path / "b"))
    rc = E.sse_fingerprint(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["probe"] == "sse_event_order"
    assert out["evidence_status"] == "live_observed"
    assert out["http_status"] == 200
    assert "message_start" in out["event_sequence"]


# ---------------------------------------------------------------------------
# needle --live (echoes planted canary -> context_ok)
# ---------------------------------------------------------------------------
def test_needle_live_context_ok(tmp_path, capsys, monkeypatch):
    def handler(request):
        # extract the planted AUTH_CANARY from the prompt and echo it back,
        # simulating a genuine long-context model that recalls the needle.
        text = request.content.decode("utf-8", errors="replace")
        m = re.search(r"AUTH_CANARY=[0-9a-f]+", text)
        echoed = m.group(0) if m else "no canary"
        return httpx.Response(200, json={
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": echoed}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50000, "output_tokens": 8},
        })

    _patch_client(monkeypatch, handler)
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model", live=True,
               target_tokens=2000, seed=5, baseline_id=None,
               baselines_dir=str(tmp_path / "b"), timeout=30.0)
    rc = E.needle(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["probe"] == "needle_recall"
    assert out["evidence_status"] == "live_observed"
    assert out["http_status"] == 200
    # canary echoed -> recall scores full -> context_ok (genuine long context)
    assert out["verdict"] == "context_ok"


def test_needle_live_http_error(tmp_path, capsys, monkeypatch):
    def handler(request):
        return httpx.Response(429, text="Upstream rate limit exceeded")

    _patch_client(monkeypatch, handler)
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model", live=True,
               target_tokens=2000, seed=5, baseline_id=None,
               baselines_dir=str(tmp_path / "b"), timeout=30.0)
    rc = E.needle(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # a 429 is a legit error, not a silent-truncation fake-1M
    assert out["verdict"] in ("insufficient_or_legit_error", "context_ok")
    assert out["http_status"] in (429, 0)


# ---------------------------------------------------------------------------
# judge_calibrate --live (judge model graded against the golden set)
# ---------------------------------------------------------------------------
def test_judge_calibrate_live(tmp_path, capsys, monkeypatch):
    # tiny authored golden set with one GO and one NO-GO case
    golden = {
        "schema_version": "judge_golden_v1",
        "description": "test",
        "cases": [
            {"id": "g1", "task": {"id": "t1", "prompt": "2+2?", "scoring_type": "exact"},
             "candidate_answer": "4", "expected_decision": "GO"},
            {"id": "g2", "task": {"id": "t2", "prompt": "2+2?", "scoring_type": "exact"},
             "candidate_answer": "5", "expected_decision": "NO-GO"},
        ],
    }
    gp = tmp_path / "golden.json"
    gp.write_text(json.dumps(golden), encoding="utf-8")

    def handler(request):
        # judge is openai_chat; grade by whether the candidate answer is "4"
        text = request.content.decode("utf-8", errors="replace")
        decision = "GO" if '"4"' in text or "answer\": \"4" in text or "4" in text and "5" not in text else "NO-GO"
        verdict = json.dumps({"score_0_10": 9 if decision == "GO" else 1,
                              "format_ok": True, "decision": decision,
                              "reason": "graded", "missing_key_points": []})
        return httpx.Response(200, json={
            "model": "gpt-5.5",
            "choices": [{"message": {"content": verdict}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        })

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("JUDGE_KEY", "sk-judge")
    args = _ns(providers=str(_providers_file(tmp_path)), provider="judge_model", live=True,
               golden_set=str(gp), baselines_dir=str(tmp_path / "b"),
               out_dir=str(tmp_path / "cal"), judge_max_tokens=256,
               retries=0, retry_backoff=0.0, request_delay=0.0, timeout=30.0,
               min_scored=2, write=True, report=False)
    rc = E.judge_calibrate(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "verdict" in out
    assert "result" in out
    assert out["result"]["scored_cases"] >= 1
    assert out["result"]["accuracy"] is not None
    # the written calibration file exists
    assert (tmp_path / "cal" / "last_calibration.json").exists()

