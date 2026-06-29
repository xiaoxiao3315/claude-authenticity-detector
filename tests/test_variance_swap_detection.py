"""Low-frequency swap detection — the consistency-variance probe (Task #7).

Drives _run_variance_probe through an httpx.MockTransport fake gateway that
answers a deterministic anchor correctly MOST of the time but wrong a fraction
of the time (simulating an upstream that silently routes ~25% of requests to a
weaker model). Asserts the probe + score_consistency_variance catch it, and
that a fully-consistent gateway scores clean.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_cli as E  # noqa: E402


ANCHOR = {"id": "cap_add", "prompt": "What is 19 + 23? Answer with the number only.",
          "check": "exact", "expected_any": ["42"]}


def _model() -> "E.ModelConfig":
    return E.ModelConfig(provider_id="suspect", base_url="https://gw.x/v1",
                         model="claude-opus-4-6", api_key_env="K",
                         protocol="anthropic_messages", auth_type="x-api-key")


def _anthropic_response(text: str) -> httpx.Response:
    return httpx.Response(200, json={
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 3},
        "content": [{"type": "text", "text": text}],
        "model": "claude-opus-4-6",
    })


def _run(tmp_path, answers, monkeypatch):
    """Drive the variance probe; `answers` is an iterable of reply strings."""
    it = iter(answers)
    real_client = httpx.Client  # capture before patching to avoid recursion
    monkeypatch.setenv("K", "sk-throwaway-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return _anthropic_response(next(it))

    monkeypatch.setattr(E.httpx, "Client",
                        lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    return E._run_variance_probe(
        ANCHOR, _model(), live=True, events_file=tmp_path / "ev.jsonl",
        repeats=16, request_delay=0.0, retries=1, retry_backoff=0.0,
        max_tokens=64, timeout=30.0,
    )


def test_consistent_gateway_scores_clean(tmp_path, monkeypatch):
    reps = _run(tmp_path, ["42"] * 16, monkeypatch)
    score = E.score_consistency_variance(reps)
    assert score["score"] == 10.0, score
    assert score["failures"] == 0


def test_low_frequency_swap_is_caught(tmp_path, monkeypatch):
    # 12 correct + 4 wrong (25% "swapped to a weaker model") interleaved
    answers = ["42", "42", "42", "41", "42", "42", "42", "41",
               "42", "42", "42", "41", "42", "42", "42", "41"]
    reps = _run(tmp_path, answers, monkeypatch)
    score = E.score_consistency_variance(reps)
    assert score["failures"] == 4, score
    assert score["score"] == 0.0, score          # statistically significant -> swap
    assert score["suspected_swap"] is True


def test_rate_limited_repeats_are_advisory(tmp_path, monkeypatch):
    # if most repeats 429, they're excluded -> too few answered -> advisory, not convict
    real_client = httpx.Client
    monkeypatch.setenv("K", "sk-throwaway-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error"}})

    monkeypatch.setattr(E.httpx, "Client",
                        lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    reps = E._run_variance_probe(
        ANCHOR, _model(), live=True, events_file=tmp_path / "ev.jsonl",
        repeats=16, request_delay=0.0, retries=1, retry_backoff=0.0,
        max_tokens=64, timeout=30.0,
    )
    # circuit breaker: a gateway that fails every request aborts after fail_fast
    # consecutive failures (default 4) instead of grinding all 16 — R-002 lesson.
    assert len(reps) == 4, reps
    assert all(r["ok"] is False for r in reps), reps
    score = E.score_consistency_variance(reps)
    assert score["score"] is None and score.get("advisory") is True, score
    # a variance_circuit_break event was logged
    import json
    evs = [json.loads(l) for l in open(tmp_path / "ev.jsonl", encoding="utf-8") if l.strip()]
    assert any(e.get("type") == "variance_circuit_break" for e in evs), evs


def test_circuit_break_does_not_trip_on_intermittent_failures(tmp_path, monkeypatch):
    # fails are NOT consecutive (fail, ok, fail, ok…) -> breaker never trips, runs full.
    real_client = httpx.Client
    monkeypatch.setenv("K", "sk-throwaway-test")
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["i"] += 1
        if state["i"] % 2 == 0:  # every other call fails — never 4 in a row
            return httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error"}})
        return _anthropic_response("42")

    monkeypatch.setattr(E.httpx, "Client",
                        lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    reps = E._run_variance_probe(
        ANCHOR, _model(), live=True, events_file=tmp_path / "ev2.jsonl",
        repeats=12, request_delay=0.0, retries=1, retry_backoff=0.0,
        max_tokens=64, timeout=30.0,
    )
    assert len(reps) == 12, reps  # ran the full set, no early abort

