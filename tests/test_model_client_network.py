"""Network-layer tests for model_client.call_model + call_model_with_retries.

Before this file, the HTTP machinery had no test double. The module's own
``_self_test`` covers auth headers, header-allowlisting, the dry path, and the
``retryable_call_failure`` *classifier* in isolation — but never the live
request/response path or the retry *loop* itself. Those ran only against a real
gateway, so a regression in response shaping, Set-Cookie stripping, retry
counting, or backoff escalation would ship unnoticed.

We drive the real ``call_model`` through an ``httpx.MockTransport`` (no socket),
scripting status codes / bodies / headers per call, and assert the resulting
``Completion`` and the retry loop's behavior. ``time.sleep`` is patched so the
backoff schedule is asserted without wall-clock cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model_client as M  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _openai_model(**over):
    kw = dict(
        provider_id="tested_model",
        base_url="https://gw.example",
        model="gpt-x",
        api_key_env="TESTED_MODEL_API_KEY",
        protocol="openai_chat",
        auth_type="bearer",
    )
    kw.update(over)
    return M.ModelConfig(**kw)


def _anthropic_model(**over):
    kw = dict(
        provider_id="tested_model",
        base_url="https://gw.example",
        model="claude-opus-4-6",
        api_key_env="TESTED_MODEL_API_KEY",
        protocol="anthropic_messages",
        auth_type="x-api-key",
    )
    kw.update(over)
    return M.ModelConfig(**kw)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def events_file(tmp_path) -> Path:
    return tmp_path / "events.jsonl"


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    # auth_value reads this env var; set it so the live path doesn't raise.
    monkeypatch.setenv("TESTED_MODEL_API_KEY", "sk-test-123")
    # Don't let a real local_secrets.env override it.
    monkeypatch.setattr(M, "load_local_env", lambda: None)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Record backoff sleeps instead of actually sleeping."""
    slept: list[float] = []
    monkeypatch.setattr(M.time, "sleep", lambda s: slept.append(s))
    return slept


def _call(model, client, events_file, **over):
    kw = dict(
        client=client,
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        temperature=0.0,
        live=True,
        events_file=events_file,
    )
    kw.update(over)
    return M.call_model(**kw)


# placeholder-D3


# ---------------------------------------------------------------------------
# request shaping: URL, auth header dialect, payload
# ---------------------------------------------------------------------------
def test_openai_request_shape(events_file):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={
            "model": "gpt-x",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        })

    out = _call(_openai_model(), _client(handler), events_file)
    assert seen["url"] == "https://gw.example/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test-123"
    assert seen["body"]["model"] == "gpt-x"
    assert seen["body"]["max_tokens"] == 32
    assert seen["body"]["temperature"] == 0.0
    assert out.metrics.ok
    assert out.text == "hello"
    assert out.metrics.input_tokens == 11
    assert out.metrics.output_tokens == 3
    assert out.metrics.stop_reason == "stop"
    assert out.metrics.server_model == "gpt-x"


def test_anthropic_request_shape_splits_system(events_file):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["xkey"] = request.headers.get("x-api-key")
        seen["aver"] = request.headers.get("anthropic-version")
        seen["body"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "hi there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 5,
                      "cache_creation_input_tokens": 2, "cache_read_input_tokens": 1},
        })

    msgs = [{"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}]
    out = _call(_anthropic_model(), _client(handler), events_file, messages=msgs)
    assert seen["url"] == "https://gw.example/v1/messages"
    assert seen["xkey"] == "sk-test-123"
    assert seen["aver"] == "2023-06-01"
    # system message is pulled out of messages and into the top-level system field
    assert seen["body"]["system"] == "be terse"
    assert [m["role"] for m in seen["body"]["messages"]] == ["user"]
    assert out.metrics.ok and out.text == "hi there"
    assert out.metrics.stop_reason == "end_turn"
    assert out.metrics.input_tokens == 20
    assert out.metrics.cache_creation_input_tokens == 2
    assert out.metrics.cache_read_input_tokens == 1


def test_temperature_omitted_for_deprecating_model(events_file):
    # claude-opus-4-8 rejects `temperature` (HTTP 400 confirmed live 2026-07-09),
    # so call_model must drop it at the egress point even though the caller
    # passed temperature=0.0. Covers both protocol dialects.
    for maker in (_openai_model, _anthropic_model):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = __import__("json").loads(request.content)
            return httpx.Response(200, json={
                "model": "claude-opus-4-8",
                "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                "content": [{"type": "text", "text": "x"}],
                "stop_reason": "end_turn",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "input_tokens": 1, "output_tokens": 1},
            })

        out = _call(maker(model="claude-opus-4-8"), _client(handler), events_file, temperature=0.0)
        assert "temperature" not in seen["body"], seen["body"]
        assert out.metrics.ok


def test_temperature_kept_for_supporting_model(events_file):
    # A model that still accepts temperature must keep it (regression guard so the
    # deprecation gate does not strip temperature from everything).
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={
            "model": "gpt-x",
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    _call(_openai_model(model="gpt-5.5"), _client(handler), events_file, temperature=0.0)
    assert seen["body"]["temperature"] == 0.0


def test_model_supports_temperature_predicate():
    assert M.model_supports_temperature("claude-opus-4-6") is True
    assert M.model_supports_temperature("gpt-5.5") is True
    assert M.model_supports_temperature("claude-opus-4-8") is False
    assert M.model_supports_temperature("claude-opus-4-8-20260115") is False


def test_secret_override_used_without_env(monkeypatch):
    # A per-instance secret_override must be used directly, without touching
    # os.environ — this is what makes concurrent web requests key-isolated.
    monkeypatch.setattr(M, "load_local_env", lambda: None)
    monkeypatch.delenv("TESTED_MODEL_API_KEY", raising=False)
    model = _openai_model(secret_override="sk-request-scoped")
    assert M.auth_value(model) == "sk-request-scoped"
    # env var is absent, yet no RuntimeError: proves it never read os.environ


def test_secret_override_is_isolated_between_instances(monkeypatch):
    # Two models with different overrides don't interfere (no shared global).
    monkeypatch.setattr(M, "load_local_env", lambda: None)
    monkeypatch.delenv("TESTED_MODEL_API_KEY", raising=False)
    a = _openai_model(secret_override="sk-key-A")
    b = _openai_model(secret_override="sk-key-B")
    assert M.auth_value(a) == "sk-key-A"
    assert M.auth_value(b) == "sk-key-B"


def test_env_used_when_no_secret_override(monkeypatch):
    # Without an override, auth_value still falls back to the env var (CLI path).
    monkeypatch.setattr(M, "load_local_env", lambda: None)
    monkeypatch.setenv("TESTED_MODEL_API_KEY", "sk-from-env")
    assert M.auth_value(_openai_model()) == "sk-from-env"


def test_unsupported_protocol_raises(events_file):
    def handler(request):  # never called
        raise AssertionError("should not reach network")

    with pytest.raises(ValueError, match="unsupported protocol"):
        _call(_openai_model(protocol="grpc_weird"), _client(handler), events_file)


# placeholder-D3b


# ---------------------------------------------------------------------------
# error handling on a single call (no retries here)
# ---------------------------------------------------------------------------
def test_http_error_returns_failed_completion(events_file):
    def handler(request):
        return httpx.Response(429, text="Upstream rate limit exceeded")

    out = _call(_openai_model(), _client(handler), events_file)
    assert out.metrics.ok is False
    assert out.text == ""
    assert "HTTP 429" in (out.metrics.error or "")
    assert out.metrics.total_ms is not None


def test_transport_exception_returns_failed_completion(events_file):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    out = _call(_openai_model(), _client(handler), events_file)
    assert out.metrics.ok is False
    # The exception type name is recorded so retryable_call_failure can classify it.
    assert "ConnectError" in (out.metrics.error or "")


def test_bad_json_body_returns_failed_completion(events_file):
    def handler(request):
        return httpx.Response(200, text="this is not json", headers={"content-type": "text/plain"})

    out = _call(_openai_model(), _client(handler), events_file)
    assert out.metrics.ok is False
    assert "parse failed" in (out.metrics.error or "")


def test_set_cookie_is_stripped_from_recorded_headers(events_file):
    def handler(request):
        return httpx.Response(
            200,
            json={"model": "gpt-x", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            headers={"Set-Cookie": "session=secret-value; HttpOnly",
                     "x-request-id": "req_42",
                     "server": "nginx"},
        )

    out = _call(_openai_model(), _client(handler), events_file)
    assert out.metrics.ok
    # Read the response_completed event and confirm no cookie/secret leaked.
    lines = events_file.read_text(encoding="utf-8").splitlines()
    import json as _json
    completed = [_json.loads(x) for x in lines if x]
    rec = [e for e in completed if e.get("type") == "response_completed"][-1]
    assert "set-cookie" not in rec["response_headers"]
    assert "session=secret-value" not in _json.dumps(rec, ensure_ascii=False)
    assert rec["response_headers"].get("x-request-id") == "req_42"
    assert rec["request_id"] == "req_42"


# placeholder-D3c


# ---------------------------------------------------------------------------
# the retry LOOP itself (call_model_with_retries)
# ---------------------------------------------------------------------------
def _ok_response():
    return httpx.Response(200, json={
        "model": "gpt-x",
        "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })


def _sequenced_client(responses):
    """A client whose handler returns/raises items from `responses` in order.

    Each item is either an httpx.Response or an Exception instance (raised).
    Records the number of calls made on the returned client via .call_count.
    """
    state = {"i": 0}

    def handler(request):
        i = state["i"]
        state["i"] += 1
        item = responses[min(i, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    client = _client(handler)
    client._d3_state = state  # type: ignore[attr-defined]
    return client


def _retry(model, client, events_file, **over):
    kw = dict(
        client=client, model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32, temperature=0.0, live=True, events_file=events_file,
        retries=2, retry_backoff=0.5,
    )
    kw.update(over)
    return M.call_model_with_retries(**kw)


def test_retry_succeeds_after_two_429s(events_file, _no_sleep):
    client = _sequenced_client([
        httpx.Response(429, text="slow down"),
        httpx.Response(429, text="slow down"),
        _ok_response(),
    ])
    out = _retry(_openai_model(), client, events_file, retries=2, retry_backoff=0.5)
    assert out.metrics.ok
    assert out.text == "done"
    assert client._d3_state["i"] == 3            # 1 initial + 2 retries
    assert out.metrics.attempts == 3
    assert out.metrics.retry_count == 2
    # backoff escalates 0.5 * 2^0, 0.5 * 2^1 = [0.5, 1.0]
    assert _no_sleep == [0.5, 1.0]


def test_retry_exhausts_and_returns_last_failure(events_file, _no_sleep):
    client = _sequenced_client([httpx.Response(503, text="unavailable")])
    out = _retry(_openai_model(), client, events_file, retries=2, retry_backoff=0.25)
    assert out.metrics.ok is False
    assert "HTTP 503" in (out.metrics.error or "")
    assert client._d3_state["i"] == 3            # initial + 2 retries, all fail
    assert out.metrics.attempts == 3
    assert out.metrics.retry_count == 2
    assert _no_sleep == [0.25, 0.5]              # last attempt does NOT sleep after


def test_400_is_not_retried(events_file, _no_sleep):
    client = _sequenced_client([httpx.Response(400, text="bad request")])
    out = _retry(_openai_model(), client, events_file, retries=3, retry_backoff=0.5)
    assert out.metrics.ok is False
    assert "HTTP 400" in (out.metrics.error or "")
    assert client._d3_state["i"] == 1            # no retry on a hard 4xx
    assert out.metrics.attempts == 1
    assert _no_sleep == []                        # never slept


def test_transport_timeout_is_retried_then_succeeds(events_file, _no_sleep):
    client = _sequenced_client([
        httpx.ReadTimeout("read timed out"),
        _ok_response(),
    ])
    out = _retry(_openai_model(), client, events_file, retries=2, retry_backoff=0.1)
    assert out.metrics.ok
    assert client._d3_state["i"] == 2
    assert out.metrics.retry_count == 1
    assert _no_sleep == [0.1]


def test_zero_retries_makes_one_attempt(events_file, _no_sleep):
    client = _sequenced_client([httpx.Response(429, text="slow")])
    out = _retry(_openai_model(), client, events_file, retries=0, retry_backoff=1.0)
    assert out.metrics.ok is False
    assert client._d3_state["i"] == 1
    assert out.metrics.attempts == 1
    assert _no_sleep == []


def test_dry_run_path_skips_network_and_retries(events_file, _no_sleep):
    def handler(request):
        raise AssertionError("dry run must not hit the network")

    out = _retry(_openai_model(), _client(handler), events_file, live=False)
    assert out.metrics.ok
    assert out.text.startswith("dry-run response")
    assert _no_sleep == []


def test_retry_emits_request_retry_events(events_file, _no_sleep):
    client = _sequenced_client([
        httpx.Response(429, text="slow"),
        _ok_response(),
    ])
    _retry(_openai_model(), client, events_file, retries=2, retry_backoff=0.5)
    import json as _json
    events = [_json.loads(x) for x in events_file.read_text(encoding="utf-8").splitlines() if x]
    retries = [e for e in events if e.get("type") == "request_retry"]
    assert len(retries) == 1
    assert retries[0]["attempt"] == 1
    assert retries[0]["max_attempts"] == 3
    assert retries[0]["sleep_seconds"] == 0.5


def test_retry_emits_warning_log(events_file, _no_sleep, caplog):
    import logging
    from logging_setup import setup_logging
    setup_logging(level="WARNING", force=True)
    # The eval logger has propagate=False, so attach caplog's handler to it
    # directly rather than relying on root propagation.
    eval_logger = logging.getLogger("eval")
    eval_logger.addHandler(caplog.handler)
    try:
        client = _sequenced_client([httpx.Response(429, text="slow"), _ok_response()])
        with caplog.at_level(logging.WARNING, logger="eval"):
            _retry(_openai_model(), client, events_file, retries=2, retry_backoff=0.5)
    finally:
        eval_logger.removeHandler(caplog.handler)
    # one retry happened -> one WARNING about retrying, carrying structured extras
    retry_records = [r for r in caplog.records if "retrying" in r.getMessage()]
    assert len(retry_records) == 1
    assert getattr(retry_records[0], "provider_id", None) == "tested_model"
    assert getattr(retry_records[0], "attempt", None) == 1




