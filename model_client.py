"""HTTP model-call layer for the eval toolchain.

Extracted from eval_cli.py: the request/response machinery for talking to an
OpenAI-compatible or Anthropic-messages endpoint, plus the dataclasses that
describe a model, a single call's metrics, and a completion.

Pure transport + shaping — no command/CLI logic. Depends only on cli_io
(now_iso/append_jsonl), redaction (redact_text), local_env (load_local_env),
and httpx. Protocol/auth-type *validation* stays in eval_cli (config layer);
this module trusts the ModelConfig it is handed.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from cli_io import append_jsonl, now_iso
from local_env import load_local_env
from redaction import redact_text

SAFE_RESPONSE_HEADER_NAMES = {
    "request-id",
    "x-request-id",
    "x-correlation-id",
    "openai-request-id",
    "anthropic-request-id",
    "cf-ray",
    "server",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
}

RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class ModelConfig:
    provider_id: str
    base_url: str
    model: str
    api_key_env: str
    protocol: str
    auth_type: str = "bearer"
    provider_channel: str = "gateway"
    provider_display_name: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallMetrics:
    ok: bool
    error: str | None = None
    first_event_ms: float | None = None
    first_content_token_ms: float | None = None
    total_ms: float | None = None
    event_count: int = 0
    content_event_count: int = 0
    content_chars: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    server_model: str | None = None
    stop_reason: str | None = None
    attempts: int = 1
    retry_count: int = 0


@dataclass
class Completion:
    text: str
    metrics: CallMetrics
    raw: dict[str, Any] | None = None


def auth_value(model: ModelConfig) -> str:
    load_local_env()
    value = os.environ.get(model.api_key_env)
    if not value:
        raise RuntimeError(f"missing environment variable {model.api_key_env!r} for {model.provider_id}")
    return value


def auth_headers(model: ModelConfig, secret: str) -> dict[str, str]:
    if model.auth_type == "bearer":
        return {"Authorization": f"Bearer {secret}"}
    if model.auth_type == "x-api-key":
        return {"x-api-key": secret}
    raise ValueError(f"unsupported auth_type: {model.auth_type}")


def safe_response_headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in dict(headers or {}).items():
        normalized = str(key).lower()
        if normalized in SAFE_RESPONSE_HEADER_NAMES:
            out[normalized] = redact_text(str(value), max_chars=160) or ""
    return dict(sorted(out.items()))


def response_request_id(headers: Any) -> str | None:
    safe = safe_response_headers(headers)
    for key in ("request-id", "x-request-id", "openai-request-id", "anthropic-request-id", "x-correlation-id", "cf-ray"):
        if safe.get(key):
            return safe[key]
    return None


def apply_extra_body(payload: dict[str, Any], model: ModelConfig) -> None:
    for key, value in model.extra_body.items():
        if key in payload:
            raise ValueError(f"{model.provider_id}.extra_body cannot override core request field: {key}")
        payload[key] = value


def dry_completion(model: ModelConfig, messages: list[dict[str, str]], max_tokens: int) -> Completion:
    user_text = " ".join(message.get("content", "") for message in messages if message.get("role") == "user")
    if model.provider_id.startswith("judge"):
        text = json.dumps(
            {
                "score_0_10": 8.0,
                "format_ok": True,
                "decision": "REVIEW",
                "reason": "dry-run judge result",
                "missing_key_points": [],
            },
            ensure_ascii=False,
        )
    else:
        text = f"dry-run response for {user_text[:120]}"
    metrics = CallMetrics(
        ok=True,
        first_event_ms=1,
        first_content_token_ms=1,
        total_ms=1,
        event_count=1,
        content_event_count=1,
        content_chars=len(text),
        input_tokens=max(1, len(user_text) // 4),
        output_tokens=max(1, min(max_tokens, len(text) // 4)),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        server_model=model.model,
        stop_reason="dry_run",
    )
    return Completion(text=text, metrics=metrics, raw={"dry_run": True})


def call_model(
    *,
    client: httpx.Client | None,
    model: ModelConfig,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    live: bool,
    events_file: Path,
) -> Completion:
    if not live:
        result = dry_completion(model, messages, max_tokens)
        append_jsonl(events_file, {"at": now_iso(), "type": "dry_completion", "provider_id": model.provider_id})
        return result

    secret = auth_value(model)
    assert client is not None, "live call_model requires an httpx client"
    payload: dict[str, Any]
    headers: dict[str, str]
    url: str
    if model.protocol == "openai_chat":
        url = f"{model.base_url}/v1/chat/completions"
        headers = {**auth_headers(model, secret), "content-type": "application/json"}
        payload = {"model": model.model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature
        apply_extra_body(payload, model)
    elif model.protocol == "anthropic_messages":
        url = f"{model.base_url}/v1/messages"
        headers = {
            **auth_headers(model, secret),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        system_messages = [message["content"] for message in messages if message.get("role") == "system"]
        user_messages = [message for message in messages if message.get("role") != "system"]
        payload = {"model": model.model, "messages": user_messages, "max_tokens": max_tokens}
        if system_messages:
            payload["system"] = "\n\n".join(system_messages)
        if temperature is not None:
            payload["temperature"] = temperature
        apply_extra_body(payload, model)
    else:
        raise ValueError(f"unsupported protocol: {model.protocol}")

    append_jsonl(
        events_file,
        {
            "at": now_iso(),
            "type": "request_started",
            "provider_id": model.provider_id,
            "protocol": model.protocol,
            "auth_type": model.auth_type,
            "model": model.model,
            "url_path": url.replace(model.base_url, ""),
            "extra_body_keys": sorted(model.extra_body),
        },
    )
    started = time.perf_counter()
    try:
        response = client.post(url, headers=headers, json=payload)
    except Exception as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        metrics = CallMetrics(ok=False, error=redact_text(f"{type(exc).__name__}: {exc}", max_chars=500), total_ms=elapsed)
        append_jsonl(events_file, {"at": now_iso(), "type": "request_failed", "error": metrics.error})
        return Completion(text="", metrics=metrics)

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    if response.status_code != 200:
        body = response.text[:1000]
        metrics = CallMetrics(ok=False, error=redact_text(f"HTTP {response.status_code}: {body}", max_chars=500), total_ms=elapsed)
        append_jsonl(events_file, {"at": now_iso(), "type": "http_error", "status": response.status_code, "body_preview": body[:300]})
        return Completion(text="", metrics=metrics)

    try:
        data = response.json()
    except Exception as exc:
        metrics = CallMetrics(ok=False, error=redact_text(f"{type(exc).__name__}: response JSON parse failed", max_chars=500), total_ms=elapsed)
        append_jsonl(events_file, {"at": now_iso(), "type": "response_parse_failed", "error": metrics.error})
        return Completion(text="", metrics=metrics)
    text = ""
    usage: dict[str, Any] = {}
    stop_reason: str | None = None
    returned_model = data.get("model") if isinstance(data, dict) else None
    if model.protocol == "openai_chat":
        choices = data.get("choices") or []
        if choices:
            first_choice = choices[0]
            text = str((first_choice.get("message") or {}).get("content") or first_choice.get("text") or "")
            stop_reason = first_choice.get("finish_reason")
        usage = data.get("usage") or {}
    else:
        blocks = data.get("content") or []
        text = "".join(str(block.get("text") or "") for block in blocks if isinstance(block, dict))
        usage = data.get("usage") or {}
        stop_reason = data.get("stop_reason")

    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    metrics = CallMetrics(
        ok=True,
        first_event_ms=elapsed,
        first_content_token_ms=elapsed if text else None,
        total_ms=elapsed,
        event_count=1,
        content_event_count=1 if text else 0,
        content_chars=len(text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        server_model=str(returned_model or model.model),
        stop_reason=str(stop_reason or "stop"),
    )
    append_jsonl(
        events_file,
        {
            "at": now_iso(),
            "type": "response_completed",
            "provider_id": model.provider_id,
            "status": response.status_code,
            "request_id": response_request_id(response.headers),
            "response_headers": safe_response_headers(response.headers),
            "total_ms": elapsed,
            "content_chars": len(text),
            "model_returned": metrics.server_model,
        },
    )
    return Completion(text=text, metrics=metrics, raw=data)


def retryable_call_failure(metrics: CallMetrics) -> bool:
    error = str(metrics.error or "")
    if not error:
        return False
    match = re.search(r"HTTP\s+(\d+)", error)
    if match:
        return int(match.group(1)) in RETRYABLE_HTTP_STATUS
    lowered = error.lower()
    retryable_tokens = (
        "timeout",
        "timed out",
        "connect",
        "connection",
        "readerror",
        "read error",
        "ssl",
        "temporar",
        "server disconnected",
        "remote protocol",
        "network",
    )
    return any(token in lowered for token in retryable_tokens)


def call_model_with_retries(
    *,
    client: httpx.Client | None,
    model: ModelConfig,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    live: bool,
    events_file: Path,
    retries: int,
    retry_backoff: float,
) -> Completion:
    if not live:
        return call_model(
            client=client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            live=live,
            events_file=events_file,
        )

    retries = max(0, int(retries or 0))
    retry_backoff = max(0.0, float(retry_backoff or 0.0))
    total_attempts = retries + 1
    overall_started = time.perf_counter()
    for attempt in range(1, total_attempts + 1):
        result = call_model(
            client=client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            live=live,
            events_file=events_file,
        )
        result.metrics.attempts = attempt
        result.metrics.retry_count = attempt - 1
        final_attempt = attempt >= total_attempts
        if result.metrics.ok or final_attempt or not retryable_call_failure(result.metrics):
            if attempt > 1:
                elapsed = round((time.perf_counter() - overall_started) * 1000, 2)
                result.metrics.total_ms = elapsed
                result.metrics.first_event_ms = elapsed
                if result.metrics.content_chars:
                    result.metrics.first_content_token_ms = elapsed
            return result

        sleep_seconds = min(60.0, retry_backoff * (2 ** (attempt - 1)))
        append_jsonl(
            events_file,
            {
                "at": now_iso(),
                "type": "request_retry",
                "provider_id": model.provider_id,
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "max_attempts": total_attempts,
                "sleep_seconds": round(sleep_seconds, 3),
                "error": redact_text(result.metrics.error, max_chars=500),
            },
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    raise RuntimeError("unreachable retry loop state")


def _self_test() -> int:
    """Offline checks: no network. Covers auth, header safety, dry path, retry rules."""
    m = ModelConfig(provider_id="tested", base_url="https://x", model="claude-opus-4",
                    api_key_env="X_KEY", protocol="anthropic_messages", auth_type="x-api-key")
    # auth_headers dialects.
    assert auth_headers(m, "sek") == {"x-api-key": "sek"}
    mb = ModelConfig(provider_id="p", base_url="https://x", model="gpt", api_key_env="K",
                     protocol="openai_chat", auth_type="bearer")
    assert auth_headers(mb, "tok") == {"Authorization": "Bearer tok"}
    try:
        auth_headers(ModelConfig(provider_id="p", base_url="u", model="m", api_key_env="K",
                                 protocol="openai_chat", auth_type="weird"), "s")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # safe_response_headers keeps only allowlisted names, lowercased + sorted.
    safe = safe_response_headers({"X-Request-Id": "r1", "Set-Cookie": "secret=1", "Server": "nginx"})
    assert safe == {"server": "nginx", "x-request-id": "r1"}, safe
    assert "set-cookie" not in safe
    assert response_request_id({"anthropic-request-id": "req_9"}) == "req_9"
    assert response_request_id({"x-foo": "bar"}) is None

    # apply_extra_body merges, but refuses to clobber core fields.
    me = ModelConfig(provider_id="p", base_url="u", model="m", api_key_env="K",
                     protocol="openai_chat", extra_body={"top_p": 0.9})
    pay = {"model": "m"}
    apply_extra_body(pay, me)
    assert pay["top_p"] == 0.9
    try:
        apply_extra_body({"top_p": 1}, me)
        raise AssertionError("expected ValueError on core-field clobber")
    except ValueError:
        pass

    # dry_completion: judge providers emit gradeable JSON; others echo.
    import tempfile
    jm = ModelConfig(provider_id="judge_main", base_url="u", model="m", api_key_env="K",
                     protocol="openai_chat")
    dj = dry_completion(jm, [{"role": "user", "content": "grade this"}], 256)
    assert dj.metrics.ok and json.loads(dj.text)["decision"] == "REVIEW"
    dr = dry_completion(m, [{"role": "user", "content": "hello"}], 64)
    assert dr.text.startswith("dry-run response") and dr.metrics.stop_reason == "dry_run"

    # call_model on the dry path needs no network and logs a dry_completion event.
    with tempfile.TemporaryDirectory() as tmp:
        ev = Path(tmp) / "events.jsonl"
        out = call_model(client=None, model=m, messages=[{"role": "user", "content": "hi"}],
                         max_tokens=32, temperature=0.0, live=False, events_file=ev)
        assert out.metrics.ok and ev.exists()

    # retryable_call_failure: transient vs hard errors.
    assert retryable_call_failure(CallMetrics(ok=False, error="HTTP 429: slow down")) is True
    assert retryable_call_failure(CallMetrics(ok=False, error="HTTP 400: bad request")) is False
    assert retryable_call_failure(CallMetrics(ok=False, error="Connection timed out")) is True
    assert retryable_call_failure(CallMetrics(ok=True, error=None)) is False

    print("model_client self-test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
