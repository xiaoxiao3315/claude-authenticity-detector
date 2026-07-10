"""Regression tests for the two bugs ai3 reported (2026-07-09), using the
no-credential repro methods it suggested.

Both would have FAILED against the pre-fix code:
  1. quickcheck's in-memory suspect + enabled sub-probes -> probe_error, because
     the SSE/error-envelope/needle probes re-read a providers file by role.
  2. the threaded web path bound each request's key into one process-global env
     var, so concurrent requests could observe/erase each other's key.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_cli as E  # noqa: E402
from model_client import ModelConfig  # noqa: E402


def _mixed_response(request: httpx.Request) -> httpx.Response:
    """A fake gateway that answers every non-stream request with a genuine-looking
    Anthropic envelope. Malformed/oversized requests still get a 4xx so the
    error-envelope probe has something to classify."""
    return httpx.Response(200, json={
        "id": "msg_01Fake",
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "42"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 2,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }, headers={"anthropic-request-id": "req_x", "request-id": "req_x"})


@pytest.fixture
def mock_client(monkeypatch):
    real_client = httpx.Client

    class _Factory:
        def __init__(self, *a, **k):
            self._c = real_client(transport=httpx.MockTransport(_mixed_response))
        def __enter__(self):
            return self._c
        def __exit__(self, *a):
            self._c.close()

    monkeypatch.setattr(E.httpx, "Client", _Factory)
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)


def _in_memory_suspect() -> ModelConfig:
    # exactly what quickcheck builds: role-less, not in any providers file,
    # with the key carried on the instance (secret_override), no env var needed.
    return ModelConfig(
        provider_id="quickcheck_suspect", base_url="https://gw.example/v1",
        model="claude-opus-4-6", api_key_env="NO_SUCH_ENV",
        protocol="anthropic_messages", auth_type="x-api-key",
        secret_override="sk-quickcheck")


import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402


def test_subprobe_runs_with_injected_model(mock_client):
    # WITH fix: the sub-probe uses args._model (the in-memory quickcheck suspect),
    # so it runs to completion even with providers=None and a role that is in no
    # providers file. This is the exact path that used to degrade to probe_error.
    args = argparse.Namespace(providers=None, provider="quickcheck_suspect",
                              live=True, _model=_in_memory_suspect())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = E.error_envelope(args)
    assert rc == 0
    assert buf.getvalue().strip(), "probe produced no output"


def test_subprobe_without_injection_fails_on_missing_role():
    # RED-LIGHT: reproduce the pre-fix wiring. Without _model, the probe falls
    # back to loading a providers file by role; quickcheck_suspect isn't a known
    # role there, so it raises — exactly what became probe_error before the fix.
    args = argparse.Namespace(providers=None, provider="quickcheck_suspect", live=True)
    with pytest.raises((ValueError, KeyError)):
        E.error_envelope(args)


# ---------------------------------------------------------------------------
# ai3 #2: web key-isolation race
# ---------------------------------------------------------------------------
import os  # noqa: E402
import api_server  # noqa: E402
from model_client import auth_value  # noqa: E402


def test_request_key_never_touches_global_env(monkeypatch):
    # RED-LIGHT: the old web path did os.environ[WEB_VERIFY_API_KEY] = key for the
    # duration of each verify. Under ThreadingHTTPServer that global slot is shared,
    # so concurrent requests race. The fix binds the key to model.secret_override
    # and must never write the shared env var.
    monkeypatch.delenv(api_server.WEB_VERIFY_KEY_ENV, raising=False)

    seen_env = []

    def fake_verify_core(model, baseline, **kw):
        # what a real call would do to authenticate — and what the probe sees:
        seen_env.append(os.environ.get(api_server.WEB_VERIFY_KEY_ENV))
        # the key must arrive via the model instance, not the process env
        assert auth_value(model) == "sk-req-A"
        return {"verdict": "consistent_with_baseline", "confidence": 1.0,
                "evidence_chain": []}

    monkeypatch.setattr(api_server.eval_cli, "verify_core", fake_verify_core)
    monkeypatch.setattr(api_server.eval_cli, "render_verdict_report", lambda *a, **k: "ok")

    model = api_server.eval_cli.ModelConfig(
        provider_id="web_suspect", base_url="https://gw.example/v1",
        model="claude-opus-4-6", api_key_env=api_server.WEB_VERIFY_KEY_ENV,
        protocol="anthropic_messages", auth_type="x-api-key",
        secret_override="sk-req-A")

    api_server._invoke_verify_core(
        model, {"baseline_id": "B"}, "B", Path("/tmp"),
        live=True, api_key="sk-req-A", req_delay=0.0, with_capability=False)

    # the shared env var was never populated during the call, and is absent after
    assert seen_env == [None], seen_env
    assert api_server.WEB_VERIFY_KEY_ENV not in os.environ


def test_two_requests_keep_isolated_keys():
    # Two concurrent requests build their own ModelConfig; each key rides on its
    # own instance, so there is no shared slot to clobber.
    a = api_server.eval_cli.ModelConfig(
        provider_id="web_suspect", base_url="https://x/v1", model="m",
        api_key_env=api_server.WEB_VERIFY_KEY_ENV, protocol="anthropic_messages",
        auth_type="x-api-key", secret_override="sk-A")
    b = api_server.eval_cli.ModelConfig(
        provider_id="web_suspect", base_url="https://x/v1", model="m",
        api_key_env=api_server.WEB_VERIFY_KEY_ENV, protocol="anthropic_messages",
        auth_type="x-api-key", secret_override="sk-B")
    results = {}

    def worker(name, model):
        results[name] = auth_value(model)

    ta = threading.Thread(target=worker, args=("A", a))
    tb = threading.Thread(target=worker, args=("B", b))
    ta.start(); tb.start(); ta.join(); tb.join()
    assert results == {"A": "sk-A", "B": "sk-B"}

