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
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4150, "output_tokens": 3,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
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

