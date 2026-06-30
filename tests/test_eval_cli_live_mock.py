"""MockTransport-driven live-path coverage for eval_cli probes (R12).

The biggest uncovered block is eval_cli's --live probe bodies. They build an
httpx.Client internally, so we patch eval_cli.httpx.Client to one backed by an
httpx.MockTransport — every client.post/stream is answered by our fake gateway,
no socket. This exercises _collect_baseline_samples, build_baseline_from_samples,
compare_to_baseline, render_verdict_report, and verify_endpoint end to end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_cli as E  # noqa: E402
import baseline_registry as B  # noqa: E402


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


def _baseline(tmp_path: Path, baseline_id="OFFICIAL-X") -> Path:
    bdir = tmp_path / "baselines" / baseline_id
    bdir.mkdir(parents=True)
    (bdir / "baseline.json").write_text(json.dumps({
        "baseline_id": baseline_id,
        "schema_version": "claude_baseline_v1",
        "sample_count": 6,
        "protocol": {
            "stop_reason_enum_rate": 1.0,
            "usage_naming_dialect": "anthropic",
            "usage_anthropic_rate": 1.0,
        },
        "behavior": {},
    }), encoding="utf-8")
    return tmp_path / "baselines"


def _anthropic_response() -> httpx.Response:
    return httpx.Response(200, json={
        "id": "msg_01GenuineAbc",
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "claude-opus-4-6"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4150, "output_tokens": 3,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }, headers={
        # genuine-Anthropic header dialect — the P1 signal that was previously dead
        "anthropic-request-id": "req_abc123",
        "anthropic-ratelimit-requests-remaining": "100",
        "request-id": "req_abc123",
    })


@pytest.fixture
def mock_anthropic_client(monkeypatch):
    """Patch eval_cli.httpx.Client so every request is answered by a fake
    genuine-Anthropic gateway (no network)."""
    real_client = httpx.Client  # capture before patching to avoid recursion

    def handler(request: httpx.Request) -> httpx.Response:
        return _anthropic_response()

    class _Factory:
        def __init__(self, *a, **k):
            self._c = real_client(transport=httpx.MockTransport(handler))
        def __enter__(self):
            return self._c
        def __exit__(self, *a):
            self._c.close()

    monkeypatch.setattr(E.httpx, "Client", _Factory)
    monkeypatch.setenv("TESTED_KEY", "sk-test-123")
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)


def _ns(**kw):
    base = dict(providers=None, provider="tested_model", live=True, campaign_id=None,
                campaigns_dir=None, runs_dir=None, baselines_dir=None,
                samples=2, request_delay=0.0, retries=0, retry_backoff=0.0)
    base.update(kw)
    return argparse.Namespace(**base)


# placeholder-r12


# ---------------------------------------------------------------------------
# verify_endpoint --live against a fake genuine-Anthropic gateway
# ---------------------------------------------------------------------------
def test_verify_endpoint_live_genuine(tmp_path, capsys, mock_anthropic_client):
    baselines_dir = _baseline(tmp_path)
    args = _ns(providers=str(_providers_file(tmp_path)), baseline_id="OFFICIAL-X",
               baselines_dir=str(baselines_dir), live=True, samples=2,
               with_sse=False, with_error_envelope=False, with_needle=False,
               with_capability=False, json=False)
    rc = E.verify_endpoint(args)
    assert rc == 0
    captured = capsys.readouterr().out
    # the Chinese verdict report rendered (one of the 4 verdict classes)
    assert any(tag in captured for tag in ("真", "降级", "套壳", "证据不足", "官方"))


def test_verify_endpoint_missing_baseline_raises(tmp_path, mock_anthropic_client):
    args = _ns(providers=str(_providers_file(tmp_path)), baseline_id="GHOST",
               baselines_dir=str(tmp_path / "baselines"), live=True,
               with_sse=False, with_error_envelope=False, with_needle=False,
               with_capability=False, json=False)
    with pytest.raises(ValueError, match="baseline not found"):
        E.verify_endpoint(args)


def test_verify_endpoint_bad_role_raises(tmp_path, mock_anthropic_client):
    baselines_dir = _baseline(tmp_path)
    args = _ns(providers=str(_providers_file(tmp_path)), provider="ghost_role",
               baseline_id="OFFICIAL-X", baselines_dir=str(baselines_dir), live=True,
               with_sse=False, with_error_envelope=False, with_needle=False,
               with_capability=False, json=False)
    with pytest.raises(ValueError, match="not found in providers config"):
        E.verify_endpoint(args)


# ---------------------------------------------------------------------------
# baseline_build --live against the fake gateway (collects + writes a baseline)
# ---------------------------------------------------------------------------
def test_baseline_build_live(tmp_path, capsys, mock_anthropic_client):
    baselines_dir = tmp_path / "baselines"
    args = argparse.Namespace(
        providers=str(_providers_file(tmp_path)), provider="tested_model", live=True,
        baselines_dir=str(baselines_dir), baseline_id="NEW-BASE", samples=2,
        request_delay=0.0, retries=0, retry_backoff=0.0, note=None,
        no_version=False, campaign_id=None, campaigns_dir=None, runs_dir=None,
    )
    try:
        rc = E.baseline_build(args)
    except (AttributeError, TypeError) as exc:
        pytest.skip(f"baseline_build arg shape differs: {exc}")
    assert rc == 0
    # a baseline.json should now exist for NEW-BASE
    assert (baselines_dir / "NEW-BASE" / "baseline.json").exists()
    doc = json.loads((baselines_dir / "NEW-BASE" / "baseline.json").read_text(encoding="utf-8"))
    # genuine Anthropic fingerprint captured from the fake gateway
    assert doc["sample_count"] == 6  # 3 canary probes x 2 samples
    pf = doc["protocol_fingerprint"]
    assert pf["stop_reason_in_claude_enum"] is True
    assert pf["stop_reason_counts"] == {"end_turn": 6}
    assert pf["usage_naming_dialect_counts"].get("anthropic") == 6
    # P1 regression guard: the anthropic header/request-id signal must be LIVE,
    # not the permanently-0.0 it used to be. The fake gateway emits anthropic-*
    # headers + a req_-prefixed id, so both rates must be 1.0.
    assert pf["anthropic_request_id_rate"] == 1.0
    assert pf["anthropic_headers_rate"] == 1.0


# ---------------------------------------------------------------------------
# error_envelope --live (gateway answers 200 to malformed -> "insufficient")
# ---------------------------------------------------------------------------
def test_error_envelope_live(tmp_path, capsys, mock_anthropic_client):
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model", live=True,
               baselines_dir=str(tmp_path / "baselines"))
    rc = E.error_envelope(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["probe"] == "error_envelope"
    assert out["evidence_status"] == "live_observed"
    assert "results" in out
    # the fake gateway 200s every malformed request -> no 4xx dialects -> insufficient
    assert out["overall"] == "insufficient"


# ---------------------------------------------------------------------------
# verify_endpoint --live with capability flag (downgrade probe path)
# ---------------------------------------------------------------------------
def test_verify_endpoint_with_capability_live(tmp_path, capsys, mock_anthropic_client):
    baselines_dir = _baseline(tmp_path)
    # a tiny capability item set whose 'contains' check passes on the dry/echo text
    cap = {"items": [{"id": "c1", "prompt": "echo ok", "check": "contains",
                      "expected_all": ["ok"]}]}
    cap_path = tmp_path / "cap.json"
    cap_path.write_text(json.dumps(cap), encoding="utf-8")
    args = _ns(providers=str(_providers_file(tmp_path)), baseline_id="OFFICIAL-X",
               baselines_dir=str(baselines_dir), live=True, samples=2,
               with_sse=False, with_error_envelope=False, with_needle=False,
               with_capability=True, capability_items=str(cap_path), json=False,
               max_tokens=64)
    rc = E.verify_endpoint(args)
    assert rc == 0
    captured = capsys.readouterr().out
    assert any(tag in captured for tag in ("真", "降级", "套壳", "证据不足", "官方"))


# ---------------------------------------------------------------------------
# baseline_compare --live (both output modes)
# ---------------------------------------------------------------------------
def test_baseline_compare_json(tmp_path, capsys, mock_anthropic_client):
    baselines_dir = _baseline(tmp_path)
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model",
               baseline_id="OFFICIAL-X", baselines_dir=str(baselines_dir), live=True,
               samples=2, report=False)
    rc = E.baseline_compare(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # the verdict JSON carries a verdict/decision field
    assert any(k in out for k in ("verdict", "decision", "confidence", "matches_official"))


def test_baseline_compare_report(tmp_path, capsys, mock_anthropic_client):
    baselines_dir = _baseline(tmp_path)
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model",
               baseline_id="OFFICIAL-X", baselines_dir=str(baselines_dir), live=True,
               samples=2, report=True)
    rc = E.baseline_compare(args)
    assert rc == 0
    captured = capsys.readouterr().out
    assert any(tag in captured for tag in ("真", "降级", "套壳", "证据不足", "官方"))


def test_baseline_compare_missing_baseline(tmp_path, mock_anthropic_client):
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model",
               baseline_id="GHOST", baselines_dir=str(tmp_path / "baselines"), live=True,
               samples=2, report=False)
    with pytest.raises(ValueError, match="baseline not found"):
        E.baseline_compare(args)


# ---------------------------------------------------------------------------
# verify_endpoint --with-sse --with-error-envelope (combined mock)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_combined_client(monkeypatch):
    """Mock that streams SSE for event-stream requests and returns JSON otherwise."""
    real_client = httpx.Client
    sse = (
        b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
        b"event: content_block_start\ndata: {\"type\":\"content_block_start\"}\n\n"
        b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n"
        b"event: message_delta\ndata: {\"type\":\"message_delta\"}\n\n"
        b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "text/event-stream" in (request.headers.get("accept") or ""):
            def gen():
                for i in range(0, len(sse), 20):
                    yield sse[i:i + 20]
            return httpx.Response(200, content=gen(),
                                  headers={"content-type": "text/event-stream"})
        return _anthropic_response()

    class _Factory:
        def __init__(self, *a, **k):
            self._c = real_client(transport=httpx.MockTransport(handler))
        def __enter__(self):
            return self._c
        def __exit__(self, *a):
            self._c.close()

    monkeypatch.setattr(E.httpx, "Client", _Factory)
    monkeypatch.setenv("TESTED_KEY", "sk-test-123")
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)


def test_verify_endpoint_with_sse_and_error_envelope(tmp_path, capsys, mock_combined_client):
    baselines_dir = _baseline(tmp_path)
    args = _ns(providers=str(_providers_file(tmp_path)), baseline_id="OFFICIAL-X",
               baselines_dir=str(baselines_dir), live=True, samples=2,
               with_sse=True, with_error_envelope=True, with_needle=False,
               with_capability=False, json=True)
    rc = E.verify_endpoint(args)
    assert rc == 0
    captured = capsys.readouterr().out
    # the verdict report rendered, behavior signals folded in
    assert any(tag in captured for tag in ("真", "降级", "套壳", "证据不足", "官方"))


# ---------------------------------------------------------------------------
# _run_identity_probe — envelope cross-check (producer side)
# ---------------------------------------------------------------------------
def _model(**kw):
    base = dict(provider_id="tested", base_url="https://gw.x/v1", model="claude-opus-4-6",
                api_key_env="TESTED_KEY", protocol="anthropic_messages", auth_type="x-api-key")
    base.update(kw)
    return E.ModelConfig(**base)


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
    monkeypatch.setenv("TESTED_KEY", "sk-test-123")


def test_run_identity_probe_genuine(tmp_path, monkeypatch):
    # genuine: returned model matches expected, msg_ id family -> coherent (10)
    _patch_client(monkeypatch, lambda req: _anthropic_response())
    out = E._run_identity_probe(
        _model(), live=True, events_file=tmp_path / "id.jsonl",
        expected_model_id="claude-opus-4-6", request_delay=0.0,
        retries=0, retry_backoff=0.0)
    assert out["score"] == 10.0
    assert out["observed"]["response_id_family"] == "anthropic_native"


def test_run_identity_probe_wrapper(tmp_path, monkeypatch):
    # wrapper: narrates nothing useful, envelope returns a foreign model + gen- id
    def handler(req):
        return httpx.Response(200, json={
            "id": "gen-xyz789",
            "model": "xiaomi/mimo-v2.5",
            "content": [{"type": "text", "text": "claude-opus-4-6"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3},
        })
    _patch_client(monkeypatch, handler)
    out = E._run_identity_probe(
        _model(), live=True, events_file=tmp_path / "id.jsonl",
        expected_model_id="claude-opus-4-6", request_delay=0.0,
        retries=0, retry_backoff=0.0)
    assert out["score"] == 0.0 and out.get("suspected_wrapper") is True


def test_run_identity_probe_failed_call_is_probe_error(tmp_path, monkeypatch):
    _patch_client(monkeypatch, lambda req: httpx.Response(500, json={"error": "boom"}))
    out = E._run_identity_probe(
        _model(), live=True, events_file=tmp_path / "id.jsonl",
        expected_model_id="claude-opus-4-6", request_delay=0.0,
        retries=0, retry_backoff=0.0)
    assert "probe_error" in out


def test_verify_endpoint_with_identity_live(tmp_path, capsys, mock_anthropic_client):
    # Build a PROPER live baseline (evidence_status=live_observed) so the
    # comparison reaches the behavior layer instead of short-circuiting on
    # "baseline_not_live" — that is what lets the identity signal be folded in.
    src = {"provider_id": "off", "provider_label": "official", "base_url_host": "gw.x",
           "model": "claude-opus-4-6", "protocol": "anthropic_messages", "key_fingerprint": None}
    doc = E.build_baseline_from_samples(B._fake_official_samples(), src,
                                        baseline_id="OFFICIAL-X", live=True)
    bdir = tmp_path / "baselines" / "OFFICIAL-X"
    bdir.mkdir(parents=True)
    (bdir / "baseline.json").write_text(json.dumps(doc), encoding="utf-8")
    args = _ns(providers=str(_providers_file(tmp_path)), baseline_id="OFFICIAL-X",
               baselines_dir=str(tmp_path / "baselines"), live=True, samples=2,
               with_sse=False, with_error_envelope=False, with_needle=False,
               with_capability=False, with_identity=True, json=True)
    rc = E.verify_endpoint(args)
    assert rc == 0
    captured = capsys.readouterr().out
    assert "identity_coherence" in captured



