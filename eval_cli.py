from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    sys.stderr.write("httpx is required: pip install -r requirements.txt\n")
    raise SystemExit(1)

from benchmarking import (
    SCORE_FORMULA_VERSION,
    calculate_benchmark_scores,
    enrich_task_metadata,
    index_run,
    load_benchmark_modes,
    select_benchmark_tasks,
)
from campaigns import (
    campaign_dir,
    campaign_list_payload,
    export_campaign,
    load_campaign,
    load_run_index,
    load_summary,
    safe_campaign_id,
    summarize_campaign,
    write_json as write_campaign_json,
)
from local_env import load_local_env
from quality_gate import run_quality_gate
from run_records import stable_json_hash, text_hash
from validate_run_records import validate_records


ROOT = Path(__file__).resolve().parent
DEFAULT_JOB = "smoke_10"
DEFAULT_PROVIDERS = Path("configs/providers.local.json")
DEFAULT_CAMPAIGNS_DIR = Path("campaigns")
ALLOWED_PROTOCOLS = {"openai_chat", "anthropic_messages"}
ALLOWED_AUTH_TYPES = {"bearer", "x-api-key"}


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


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def utcish_job_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def resolve_path(path_value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def resolve_job(job: str | Path) -> Path:
    raw = Path(job)
    candidates = []
    if raw.suffix:
        candidates.append(raw)
    else:
        candidates.append(Path("configs/jobs") / f"{raw}.json")
        candidates.append(Path("configs/jobs") / str(raw))
    for candidate in candidates:
        path = resolve_path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"job config not found: {job}")


def base_url_host(base_url: str) -> str | None:
    parsed = urlparse(base_url)
    return parsed.netloc or parsed.path or None


def load_model_config(raw: dict[str, Any], label: str) -> ModelConfig:
    protocol = str(raw.get("protocol") or "").strip()
    if protocol not in ALLOWED_PROTOCOLS:
        raise ValueError(f"{label}.protocol must be one of: {', '.join(sorted(ALLOWED_PROTOCOLS))}")
    auth_type = str(raw.get("auth_type") or "bearer").strip()
    if auth_type not in ALLOWED_AUTH_TYPES:
        raise ValueError(f"{label}.auth_type must be one of: {', '.join(sorted(ALLOWED_AUTH_TYPES))}")
    return ModelConfig(
        provider_id=str(raw["provider_id"]),
        base_url=str(raw["base_url"]).rstrip("/"),
        model=str(raw["model"]),
        api_key_env=str(raw["api_key_env"]),
        protocol=protocol,
        auth_type=auth_type,
        provider_channel=str(raw.get("provider_channel") or "gateway"),
        provider_display_name=str(raw.get("provider_display_name") or raw["provider_id"]),
    )


def load_two_model_config(path: Path) -> dict[str, ModelConfig]:
    load_local_env()
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {
        "tested_model": load_model_config(data["tested_model"], "tested_model"),
        "judge_model": load_model_config(data["judge_model"], "judge_model"),
    }


def sanitized_models(models: dict[str, ModelConfig]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, model in models.items():
        out[label] = {
            "provider_id": model.provider_id,
            "base_url_host": base_url_host(model.base_url),
            "model": model.model,
            "api_key_env": model.api_key_env,
            "api_key_present": bool(os.environ.get(model.api_key_env)),
            "protocol": model.protocol,
            "auth_type": model.auth_type,
        }
    return out


def key_fingerprint(env_name: str) -> str | None:
    value = os.environ.get(env_name)
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def campaign_model_identity(model: ModelConfig) -> dict[str, Any]:
    return {
        "provider_id": model.provider_id,
        "base_url_host": base_url_host(model.base_url),
        "model": model.model,
        "api_key_env": model.api_key_env,
        "key_fingerprint": key_fingerprint(model.api_key_env),
        "protocol": model.protocol,
        "auth_type": model.auth_type,
        "provider_channel": model.provider_channel,
        "provider_display_name": model.provider_display_name or model.provider_id,
    }


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def policy_metadata(path: Path) -> dict[str, Any]:
    data = read_json(path) if path.exists() else {}
    policies = data.get("policies") if isinstance(data.get("policies"), list) else []
    first = policies[0] if policies and isinstance(policies[0], dict) else {}
    return {
        "policy_version": data.get("policy_version") or "unknown",
        "policy_id": first.get("policy_id") or "unknown",
    }


def benchmark_metadata(job: dict[str, Any], benchmark_config: dict[str, Any], benchmark_mode: str) -> dict[str, Any]:
    benchmark_version = str(benchmark_config.get("version") or job.get("benchmark_version") or "custom")
    return {
        "benchmark_config_version": benchmark_version,
        "benchmark_version": f"{benchmark_version}:{benchmark_mode}",
        "benchmark_mode": benchmark_mode,
        "score_formula_version": str(benchmark_config.get("score_formula_version") or SCORE_FORMULA_VERSION),
    }


def campaign_id_for(model: ModelConfig) -> str:
    model_part = re.sub(r"[^A-Za-z0-9]+", "-", model.model).strip("-").upper() or "MODEL"
    return f"CMP-{model_part}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, list):
        raise ValueError(f"{path} must contain a tasks array")
    return [task for task in tasks if isinstance(task, dict)]


def task_metadata(task: dict[str, Any]) -> dict[str, Any]:
    task = enrich_task_metadata(task)
    return {
        "id": task.get("id"),
        "category": task.get("category"),
        "enterprise_dimension": task.get("enterprise_dimension"),
        "difficulty": task.get("difficulty"),
        "scoring_type": task.get("scoring_type"),
        "risk_tags": task.get("risk_tags") or [],
        "point_value": task.get("point_value"),
        "scoring_confidence": task.get("scoring_confidence"),
        "dimension_weight_group": task.get("dimension_weight_group"),
    }


def select_tasks(job: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks_file = resolve_path(job.get("tasks_file", "tasks/enterprise_v0_2.json"))
    tasks = load_tasks(tasks_file)
    benchmark_config: dict[str, Any] = {}
    benchmark_mode = str(job.get("benchmark_mode") or "custom")
    if job.get("benchmarks_file") and job.get("benchmark_mode"):
        benchmark_config = load_benchmark_modes(resolve_path(job["benchmarks_file"]))
        tasks = select_benchmark_tasks(tasks, benchmark_mode, benchmark_config)
    else:
        tasks = [enrich_task_metadata(task) for task in tasks]
    max_tasks = int(job.get("max_tasks") or 0)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    return tasks, benchmark_config


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
    client: httpx.Client,
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
    payload: dict[str, Any]
    headers: dict[str, str]
    url: str
    if model.protocol == "openai_chat":
        url = f"{model.base_url}/v1/chat/completions"
        headers = {**auth_headers(model, secret), "content-type": "application/json"}
        payload = {"model": model.model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature
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
        },
    )
    started = time.perf_counter()
    try:
        response = client.post(url, headers=headers, json=payload)
    except Exception as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        metrics = CallMetrics(ok=False, error=f"{type(exc).__name__}: {exc}", total_ms=elapsed)
        append_jsonl(events_file, {"at": now_iso(), "type": "request_failed", "error": metrics.error})
        return Completion(text="", metrics=metrics)

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    if response.status_code != 200:
        body = response.text[:1000]
        metrics = CallMetrics(ok=False, error=f"HTTP {response.status_code}: {body}", total_ms=elapsed)
        append_jsonl(events_file, {"at": now_iso(), "type": "http_error", "status": response.status_code, "body_preview": body[:300]})
        return Completion(text="", metrics=metrics)

    try:
        data = response.json()
    except Exception as exc:
        metrics = CallMetrics(ok=False, error=f"{type(exc).__name__}: response JSON parse failed", total_ms=elapsed)
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
            "total_ms": elapsed,
            "content_chars": len(text),
            "model_returned": metrics.server_model,
        },
    )
    return Completion(text=text, metrics=metrics, raw=data)


RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


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
    client: httpx.Client,
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
                "error": (result.metrics.error or "")[:500],
            },
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    raise RuntimeError("unreachable retry loop state")


def expected_context(task: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("expected_key_points", "expected_json", "expected_output", "rubric"):
        if key in task:
            out[key] = task[key]
    return out


def judge_messages(task: dict[str, Any], response_text: str) -> list[dict[str, str]]:
    prompt = {
        "task_id": task.get("id"),
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
        "scoring_type": task.get("scoring_type"),
        "prompt": task.get("prompt"),
        "expected": expected_context(task),
        "answer": response_text,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an acceptance judge for private LLM evaluation. "
                "Return only a JSON object with keys: score_0_10, format_ok, "
                "decision, reason, missing_key_points. decision must be GO, REVIEW, or NO-GO."
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(text)
        if not match:
            return None, "judge did not return a JSON object"
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return None, f"judge JSON parse failed: {exc}"
    if not isinstance(value, dict):
        return None, "judge JSON is not an object"
    return value, None


def normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(10.0, score))


def json_exact_rule_score(task: dict[str, Any], response_text: str) -> dict[str, Any] | None:
    if task.get("scoring_type") != "json_exact" or "expected_json" not in task:
        return None
    try:
        actual = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return {"score": 0.0, "format_ok": False, "details": f"invalid JSON: {exc}"}
    expected = task.get("expected_json")
    ok = actual == expected
    return {
        "score": 10.0 if ok else 2.0,
        "format_ok": ok,
        "details": "exact JSON match" if ok else "JSON did not match expected object",
    }


def final_score_from_judge(
    *,
    tested: Completion,
    judge: Completion | None,
    judge_payload: dict[str, Any] | None,
    judge_error: str | None,
    rule_score: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not tested.metrics.ok:
        return {"score": 0.0, "format_ok": False, "details": tested.metrics.error}, None
    if judge_payload is not None:
        score = normalize_score(judge_payload.get("score_0_10"))
        if score is not None:
            return (
                {
                    "score": score,
                    "format_ok": bool(judge_payload.get("format_ok", True)),
                    "details": str(judge_payload.get("reason") or ""),
                    "decision": str(judge_payload.get("decision") or "REVIEW"),
                },
                judge_payload,
            )
    if rule_score is not None:
        return rule_score, {"error": judge_error or "judge unavailable; used rule score"}
    fallback = 0.0 if judge_error else 5.0
    return {"score": fallback, "format_ok": None, "details": judge_error or "no judge score"}, {"error": judge_error or "no judge score"}


def run_record(
    *,
    run_id: str,
    timestamp: str,
    benchmark_mode: str,
    task: dict[str, Any],
    tested_model: ModelConfig,
    tested: Completion,
    final_score: dict[str, Any],
    judge_score: dict[str, Any] | None,
    response_file: Path,
    events_file: Path,
    max_tokens: int,
    temperature: float | None,
) -> dict[str, Any]:
    metadata = task_metadata(task)
    request_fingerprint = {
        "api_style": tested_model.protocol,
        "model": tested_model.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt_hash": text_hash(str(task.get("prompt") or "")),
        "messages_count": 1,
    }
    metrics = tested.metrics
    return {
        "schema_version": "run_record_v1",
        "record_id": f"{run_id}:{tested_model.provider_id}:{metadata.get('id')}",
        "run": {
            "run_id": run_id,
            "timestamp": timestamp,
            "benchmark_mode": benchmark_mode,
            "formula_version": SCORE_FORMULA_VERSION,
            "runner": "cli",
            "status": "completed" if metrics.ok else "failed",
        },
        "task": {
            "id": metadata.get("id"),
            "category": metadata.get("category"),
            "enterprise_dimension": metadata.get("enterprise_dimension"),
            "difficulty": metadata.get("difficulty"),
            "scoring_type": metadata.get("scoring_type"),
            "risk_tags": metadata.get("risk_tags") or [],
            "point_value": metadata.get("point_value"),
            "scoring_confidence": metadata.get("scoring_confidence"),
        },
        "provider": {
            "id": tested_model.provider_id,
            "api_style": tested_model.protocol,
            "base_url_host": base_url_host(tested_model.base_url),
            "auth_env_name": tested_model.api_key_env,
            "model_requested": tested_model.model,
            "model_returned": metrics.server_model,
            "provider_channel": tested_model.provider_channel,
            "provider_display_name": tested_model.provider_display_name or tested_model.provider_id,
            "claimed_model": tested_model.model,
            "baseline_model": tested_model.model,
            "leaderboard_group": "gateway_candidate",
        },
        "request": {
            "request_hash": stable_json_hash(request_fingerprint),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system_present": False,
            "messages_count": 1,
            "prompt_hash": request_fingerprint["prompt_hash"],
        },
        "response": {
            "response_file": str(response_file),
            "events_file": str(events_file),
            "content_chars": metrics.content_chars,
            "normalized_text_hash": text_hash(tested.text or ""),
        },
        "telemetry": {
            "ok": metrics.ok,
            "error": metrics.error,
            "first_event_ms": metrics.first_event_ms,
            "first_content_token_ms": metrics.first_content_token_ms,
            "total_ms": metrics.total_ms,
            "event_count": metrics.event_count,
            "content_event_count": metrics.content_event_count,
            "stop_reason": metrics.stop_reason,
            "attempts": metrics.attempts,
            "retry_count": metrics.retry_count,
        },
        "usage": {
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "cache_creation_input_tokens": metrics.cache_creation_input_tokens,
            "cache_read_input_tokens": metrics.cache_read_input_tokens,
        },
        "scoring": {
            "rule_score": final_score,
            "judge_score": judge_score,
            "final_score": final_score,
            "judge_provider": (judge_score or {}).get("provider"),
            "judge_model_requested": (judge_score or {}).get("model_requested"),
            "judge_model_returned": (judge_score or {}).get("model_returned"),
        },
        "trace": {"tool_calls": [], "raw_event_types": []},
        "artifacts": {"response_file": str(response_file), "events_file": str(events_file)},
    }


def summary_row(record: dict[str, Any], response_file: Path) -> dict[str, Any]:
    task = record["task"]
    provider = record["provider"]
    telemetry = record["telemetry"]
    usage = record["usage"]
    scoring = record["scoring"]
    final_score = scoring.get("final_score") or {}
    judge_score = scoring.get("judge_score") or {}
    return {
        "run_id": record["run"]["run_id"],
        "timestamp": record["run"]["timestamp"],
        "provider": provider.get("id"),
        "task_id": task.get("id"),
        "category": task.get("category"),
        "enterprise_dimension": task.get("enterprise_dimension"),
        "difficulty": task.get("difficulty"),
        "scoring_type": task.get("scoring_type"),
        "risk_tags": ";".join(task.get("risk_tags") or []),
        "point_value": task.get("point_value"),
        "scoring_confidence": task.get("scoring_confidence"),
        "ok": telemetry.get("ok"),
        "error": telemetry.get("error"),
        "quality_0_10": final_score.get("score"),
        "format_ok": final_score.get("format_ok"),
        "judge_error": judge_score.get("error") if isinstance(judge_score, dict) else None,
        "judge_provider": scoring.get("judge_provider"),
        "judge_score_0_10": judge_score.get("score_0_10") if isinstance(judge_score, dict) else None,
        "judge_format_ok": judge_score.get("format_ok") if isinstance(judge_score, dict) else None,
        "model_requested": provider.get("model_requested"),
        "model_returned": provider.get("model_returned"),
        "stop_reason": telemetry.get("stop_reason"),
        "first_content_token_ms": telemetry.get("first_content_token_ms"),
        "total_ms": telemetry.get("total_ms"),
        "attempts": telemetry.get("attempts"),
        "retry_count": telemetry.get("retry_count"),
        "input_tokens": usage.get("input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "response_file": str(response_file),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["run_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_state(run_dir: Path, state: dict[str, Any]) -> None:
    write_json(run_dir / "state.json", state)


def latest_run_dir(runs_dir: Path) -> Path:
    candidates = [path for path in runs_dir.iterdir() if path.is_dir() and (path / "state.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"no job runs found under {runs_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def run_job(args: argparse.Namespace) -> int:
    job_path = resolve_job(args.job)
    job = read_json(job_path)
    models_path = resolve_path(args.providers or job.get("providers_file") or DEFAULT_PROVIDERS)
    models = load_two_model_config(models_path)
    tasks, benchmark_config = select_tasks(job)
    if not tasks:
        raise ValueError("job selected zero tasks")

    live = bool(args.live)
    if not live and bool(job.get("live_provider")):
        live = True
    run_id = args.run_id or utcish_job_id(str(job.get("job_id_prefix") or "JOB"))
    runs_dir = resolve_path(args.runs_dir or job.get("runs_dir") or "runs")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    benchmark_mode = str(job.get("benchmark_mode") or "custom")
    timestamp = now_iso()
    temperature = job.get("temperature")
    if temperature is not None:
        temperature = float(temperature)
    tested_max_tokens = int(args.tested_max_tokens or job.get("tested_max_tokens") or 768)
    judge_max_tokens = int(args.judge_max_tokens or job.get("judge_max_tokens") or 512)
    timeout = float(args.timeout or job.get("timeout") or 120)
    retries_raw = getattr(args, "retries", None)
    if retries_raw is None:
        retries_raw = job.get("request_retries")
    if retries_raw is None:
        retries_raw = 2 if live else 0
    backoff_raw = getattr(args, "retry_backoff", None)
    if backoff_raw is None:
        backoff_raw = job.get("retry_backoff_seconds")
    if backoff_raw is None:
        backoff_raw = 2.0
    request_retries = int(retries_raw)
    retry_backoff = float(backoff_raw)
    request_retries = max(0, request_retries)
    retry_backoff = max(0.0, retry_backoff)

    state = {
        "job_id": run_id,
        "status": "running",
        "label": job.get("label"),
        "started_at": timestamp,
        "completed_at": None,
        "progress": {"completed": 0, "total": len(tasks)},
        "current_task": None,
        "live_provider": live,
        "models": sanitized_models(models),
        "retry": {"retries": request_retries if live else 0, "backoff_seconds": retry_backoff if live else 0},
        "final_decision": None,
        "artifacts": {},
    }
    write_state(run_dir, state)
    write_json(run_dir / "job_config.snapshot.json", {**job, "job_path": str(job_path)})
    write_json(run_dir / "providers.redacted.json", sanitized_models(models))
    append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "job_started", "job_id": run_id, "live_provider": live})

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 20.0), read=timeout, write=min(timeout, 30.0), pool=10.0)
    with httpx.Client(timeout=client_timeout, follow_redirects=True) as client:
        for index, task in enumerate(tasks, start=1):
            task_id = str(task.get("id") or f"task_{index}")
            state["current_task"] = task_id
            write_state(run_dir, state)
            append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "task_started", "task_id": task_id, "index": index})

            tested_events = run_dir / "events" / models["tested_model"].provider_id / f"{task_id}.jsonl"
            tested = call_model_with_retries(
                client=client,
                model=models["tested_model"],
                messages=[{"role": "user", "content": str(task.get("prompt") or "")}],
                max_tokens=tested_max_tokens,
                temperature=temperature,
                live=live,
                events_file=tested_events,
                retries=request_retries,
                retry_backoff=retry_backoff,
            )
            response_file = run_dir / "responses" / models["tested_model"].provider_id / f"{task_id}.txt"
            response_file.parent.mkdir(parents=True, exist_ok=True)
            response_file.write_text(tested.text, encoding="utf-8")

            rule_score = json_exact_rule_score(task, tested.text)
            judge_completion: Completion | None = None
            judge_payload: dict[str, Any] | None = None
            judge_parse_error: str | None = None
            if tested.metrics.ok:
                judge_events = run_dir / "events" / models["judge_model"].provider_id / f"{task_id}.jsonl"
                judge_completion = call_model_with_retries(
                    client=client,
                    model=models["judge_model"],
                    messages=judge_messages(task, tested.text),
                    max_tokens=judge_max_tokens,
                    temperature=0,
                    live=live,
                    events_file=judge_events,
                    retries=request_retries,
                    retry_backoff=retry_backoff,
                )
                judge_response_file = run_dir / "judge_responses" / models["judge_model"].provider_id / f"{task_id}.txt"
                judge_response_file.parent.mkdir(parents=True, exist_ok=True)
                judge_response_file.write_text(judge_completion.text, encoding="utf-8")
                if judge_completion.metrics.ok:
                    judge_payload, judge_parse_error = parse_judge_json(judge_completion.text)
                else:
                    judge_parse_error = judge_completion.metrics.error
            else:
                judge_parse_error = "tested model call failed"

            final_score, judge_score = final_score_from_judge(
                tested=tested,
                judge=judge_completion,
                judge_payload=judge_payload,
                judge_error=judge_parse_error,
                rule_score=rule_score,
            )
            if judge_score is not None and "error" not in judge_score:
                judge_score = {
                    **judge_score,
                    "provider": models["judge_model"].provider_id,
                    "model_requested": models["judge_model"].model,
                    "model_returned": judge_completion.metrics.server_model if judge_completion else None,
                }

            record = run_record(
                run_id=run_id,
                timestamp=timestamp,
                benchmark_mode=benchmark_mode,
                task=task,
                tested_model=models["tested_model"],
                tested=tested,
                final_score=final_score,
                judge_score=judge_score,
                response_file=response_file,
                events_file=tested_events,
                max_tokens=tested_max_tokens,
                temperature=temperature,
            )
            append_jsonl(run_dir / "run_records.jsonl", record)
            records.append(record)
            rows.append(summary_row(record, response_file))
            results.append(
                {
                    "task": task_metadata(task),
                    "tested_model": models["tested_model"].provider_id,
                    "judge_model": models["judge_model"].provider_id,
                    "tested_ok": tested.metrics.ok,
                    "score": final_score,
                    "judge_score": judge_score,
                    "response_file": str(response_file),
                }
            )
            state["progress"]["completed"] = index
            append_jsonl(
                run_dir / "events.jsonl",
                {"at": now_iso(), "type": "task_completed", "task_id": task_id, "score": final_score.get("score"), "ok": tested.metrics.ok},
            )
            write_state(run_dir, state)

    write_json(run_dir / "results.json", results)
    write_summary_csv(run_dir / "summary.csv", rows)
    benchmark_scores = calculate_benchmark_scores(rows, benchmark_mode, SCORE_FORMULA_VERSION)
    write_json(run_dir / "benchmark_scores.json", benchmark_scores)
    try:
        index_run(runs_dir / "runs_index.sqlite", run_id, run_dir, rows, benchmark_scores)
    except Exception as exc:
        append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "index_warning", "error": str(exc)})

    validation_errors = validate_records(records)
    write_json(run_dir / "validation.json", {"error_count": len(validation_errors), "errors": validation_errors})
    gate_result: dict[str, Any] | None = None
    policy_path = resolve_path(job.get("quality_gate_policy") or "quality_gate.policy.json")
    try:
        gate_result = run_quality_gate(
            runs_dir=runs_dir,
            run_id=run_id,
            policy_path=policy_path,
            provider_id=models["tested_model"].provider_id,
            gate_label="two_model_headless",
        )
    except Exception as exc:
        append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "quality_gate_warning", "error": str(exc)})

    decision = "REVIEW"
    if gate_result and gate_result.get("records"):
        decision = str(gate_result["records"][0].get("decision") or "REVIEW")
    state["status"] = "completed" if not validation_errors else "partial"
    state["completed_at"] = now_iso()
    state["current_task"] = None
    state["final_decision"] = decision
    state["validation_errors"] = len(validation_errors)
    write_state(run_dir, state)
    append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "job_completed", "job_id": run_id, "decision": decision})
    print(json.dumps({"job_id": run_id, "status": state["status"], "decision": decision, "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
    return 0


def inspect_job(args: argparse.Namespace) -> int:
    runs_dir = resolve_path(args.runs_dir or "runs")
    run_dir = latest_run_dir(runs_dir) if args.latest else runs_dir / str(args.job_id)
    state = read_json(run_dir / "state.json")
    summary = {
        "job_id": state.get("job_id") or run_dir.name,
        "status": state.get("status"),
        "progress": state.get("progress"),
        "final_decision": state.get("final_decision"),
        "run_dir": str(run_dir),
        "artifacts": state.get("artifacts") or {},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def export_job(args: argparse.Namespace) -> int:
    runs_dir = resolve_path(args.runs_dir or "runs")
    run_dir = latest_run_dir(runs_dir) if args.latest else runs_dir / str(args.job_id)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    zip_path = artifacts_dir / "acceptance_pack.zip"
    include_names = [
        "state.json",
        "events.jsonl",
        "run_records.jsonl",
        "results.json",
        "summary.csv",
        "benchmark_scores.json",
        "validation.json",
        "job_config.snapshot.json",
        "providers.redacted.json",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include_names:
            path = run_dir / name
            if path.exists():
                zf.write(path, arcname=name)
        for folder in ("responses", "judge_responses", "quality_gates"):
            root = run_dir / folder
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(run_dir)))
    state = read_json(run_dir / "state.json")
    state.setdefault("artifacts", {})["acceptance_pack"] = str(zip_path)
    write_state(run_dir, state)
    print(json.dumps({"job_id": run_dir.name, "artifact": str(zip_path)}, ensure_ascii=False, indent=2))
    return 0


def campaign_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    campaigns_dir = resolve_path(getattr(args, "campaigns_dir", None) or DEFAULT_CAMPAIGNS_DIR)
    runs_dir = resolve_path(getattr(args, "runs_dir", None) or "runs")
    return campaigns_dir, runs_dir


def active_run_refs(run_index: dict[str, Any]) -> list[dict[str, Any]]:
    refs = run_index.get("runs") if isinstance(run_index.get("runs"), list) else []
    return [ref for ref in refs if ref.get("status") != "replaced"]


def run_state_for(runs_dir: Path, run_id: str) -> dict[str, Any]:
    path = runs_dir / run_id / "state.json"
    if not path.exists():
        return {}
    return read_json(path)


def run_ref_completed(run_ref: dict[str, Any], runs_dir: Path) -> bool:
    run_id = str(run_ref.get("run_id") or "")
    state = run_state_for(runs_dir, run_id)
    status = state.get("status") or run_ref.get("status")
    return status == "completed"


def latest_active_run_ref(run_index: dict[str, Any], round_index: int) -> dict[str, Any] | None:
    refs = [ref for ref in active_run_refs(run_index) if int(ref.get("round") or 0) == round_index]
    return refs[-1] if refs else None


def next_campaign_run_id(campaign_id: str, round_index: int, run_index: dict[str, Any], runs_dir: Path) -> tuple[str, int]:
    refs = run_index.get("runs") if isinstance(run_index.get("runs"), list) else []
    attempt = max([int(ref.get("attempt") or 1) for ref in refs if int(ref.get("round") or 0) == round_index] or [0]) + 1
    while True:
        run_id = f"{campaign_id}-R{round_index:02d}" if attempt == 1 else f"{campaign_id}-R{round_index:02d}-A{attempt:02d}"
        if not (runs_dir / run_id).exists():
            return run_id, attempt
        attempt += 1


def run_campaign(args: argparse.Namespace) -> int:
    resume = bool(getattr(args, "resume", False))
    if resume and not args.campaign_id:
        raise ValueError("--resume requires --campaign-id")
    if not resume and args.repeat is None:
        raise ValueError("--repeat is required when creating a campaign")
    if args.repeat is not None and int(args.repeat) < 1:
        raise ValueError("--repeat must be >= 1")
    job_path = resolve_job(args.job)
    job = read_json(job_path)
    models_path = resolve_path(args.providers or job.get("providers_file") or DEFAULT_PROVIDERS)
    models = load_two_model_config(models_path)
    _, benchmark_config = select_tasks(job)
    benchmark_mode = str(job.get("benchmark_mode") or "custom")
    benchmark_meta = benchmark_metadata(job, benchmark_config, benchmark_mode)
    policy_path = resolve_path(job.get("quality_gate_policy") or "quality_gate.policy.json")
    policy_meta = policy_metadata(policy_path)
    campaigns_dir, runs_dir = campaign_paths(args)
    campaign_id = safe_campaign_id(args.campaign_id or campaign_id_for(models["tested_model"]))
    out_dir = campaign_dir(campaigns_dir, campaign_id)
    if resume:
        if not out_dir.exists():
            raise FileNotFoundError(f"campaign not found: {campaign_id}")
        campaign_doc = load_campaign(out_dir)
        run_index = load_run_index(out_dir)
        live = campaign_doc.get("live_provider") is True
        repeat = int(args.repeat or campaign_doc.get("repeat") or 0)
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        campaign_doc = {}
        run_index = {"campaign_id": campaign_id, "runs": []}
        live = bool(args.live)
        if not live and bool(job.get("live_provider")):
            live = True
        repeat = int(args.repeat)
    tested_identity = campaign_model_identity(models["tested_model"])
    judge_identity = campaign_model_identity(models["judge_model"])
    config_hash = stable_json_hash(
        {
            "job": {**job, "job_path": str(job_path)},
            "providers": {
                "tested_model": tested_identity,
                "judge_model": judge_identity,
            },
            "benchmark": benchmark_meta,
            "quality_gate": policy_meta,
            "live_provider": live,
        }
    )
    if resume:
        existing_hash = campaign_doc.get("config_hash")
        if existing_hash and existing_hash != config_hash:
            raise ValueError("campaign config hash changed; refusing resume")
        if repeat != int(campaign_doc.get("repeat") or repeat):
            raise ValueError("--repeat must match the existing campaign when using --resume")
        campaign_doc["status"] = "running"
        campaign_doc["completed_at"] = None
        campaign_doc["resumed_at"] = now_iso()
    else:
        campaign_doc = {
            "schema_version": "campaign_v1",
            "campaign_id": campaign_id,
            "job": str(args.job),
            "job_path": str(job_path),
            "repeat": repeat,
            "status": "running",
            "created_at": now_iso(),
            "completed_at": None,
            "live_provider": live,
            "tested_model": tested_identity,
            "judge_model": judge_identity,
            "benchmark_config_version": benchmark_meta["benchmark_config_version"],
            "benchmark_version": benchmark_meta["benchmark_version"],
            "benchmark_mode": benchmark_meta["benchmark_mode"],
            "score_formula_version": benchmark_meta["score_formula_version"],
            "quality_gate_version": policy_meta["policy_version"],
            "quality_gate_policy_id": policy_meta["policy_id"],
            "code_git_commit": git_commit(),
            "config_hash": config_hash,
            "artifacts": {},
        }
    write_campaign_json(out_dir / "campaign.json", campaign_doc)
    write_campaign_json(out_dir / "run_ids.json", run_index)

    exit_code = 0
    skipped_rounds = 0
    for round_index in range(1, repeat + 1):
        existing_ref = latest_active_run_ref(run_index, round_index)
        if existing_ref and run_ref_completed(existing_ref, runs_dir):
            skipped_rounds += 1
            continue
        replaced_run_id = None
        if existing_ref:
            existing_ref["status"] = "replaced"
            existing_ref["replaced_at"] = now_iso()
            replaced_run_id = existing_ref.get("run_id")
        run_id, attempt = next_campaign_run_id(campaign_id, round_index, run_index, runs_dir)
        run_ref = {
            "round": round_index,
            "attempt": attempt,
            "run_id": run_id,
            "status": "running",
            "started_at": now_iso(),
            "completed_at": None,
        }
        if replaced_run_id:
            run_ref["replaces_run_id"] = replaced_run_id
        run_index["runs"].append(run_ref)
        write_campaign_json(out_dir / "run_ids.json", run_index)
        summarize_campaign(out_dir, runs_dir)
        child_args = argparse.Namespace(
            job=args.job,
            providers=args.providers,
            runs_dir=runs_dir,
            run_id=run_id,
            live=live,
            timeout=args.timeout,
            tested_max_tokens=args.tested_max_tokens,
            judge_max_tokens=args.judge_max_tokens,
            retries=getattr(args, "retries", None),
            retry_backoff=getattr(args, "retry_backoff", None),
        )
        try:
            run_job(child_args)
            state = read_json(runs_dir / run_id / "state.json")
            run_ref["status"] = state.get("status") or "completed"
            run_ref["started_at"] = state.get("started_at") or run_ref["started_at"]
            run_ref["completed_at"] = state.get("completed_at") or now_iso()
            run_ref["final_decision"] = state.get("final_decision")
        except Exception as exc:
            run_ref["status"] = "failed"
            run_ref["completed_at"] = now_iso()
            run_ref["error"] = str(exc)
            exit_code = 1
        write_campaign_json(out_dir / "run_ids.json", run_index)
        summarize_campaign(out_dir, runs_dir)
        if exit_code:
            break

    campaign_doc["completed_at"] = now_iso()
    if exit_code:
        campaign_doc["status"] = "failed"
    elif active_run_refs(run_index) and all(run_ref_completed(run, runs_dir) for run in active_run_refs(run_index)):
        campaign_doc["status"] = "completed"
    else:
        campaign_doc["status"] = "partial"
    write_campaign_json(out_dir / "campaign.json", campaign_doc)
    summary = summarize_campaign(out_dir, runs_dir)
    print(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "status": campaign_doc["status"],
                "campaign_dir": str(out_dir),
                "summary": {
                    "total_runs": summary["metrics"]["total_runs"],
                    "total_cases": summary["metrics"]["total_cases"],
                    "overall_decision": summary["decisions"]["overall_decision"],
                    "skipped_completed_rounds": skipped_rounds,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code


def campaign_list(args: argparse.Namespace) -> int:
    campaigns_dir, runs_dir = campaign_paths(args)
    print(json.dumps(campaign_list_payload(campaigns_dir, runs_dir), ensure_ascii=False, indent=2))
    return 0


def campaign_status(args: argparse.Namespace) -> int:
    campaigns_dir, runs_dir = campaign_paths(args)
    out_dir = campaign_dir(campaigns_dir, args.campaign_id)
    campaign_doc = load_campaign(out_dir)
    summary = load_summary(out_dir) or summarize_campaign(out_dir, runs_dir)
    payload = {
        "campaign_id": campaign_doc.get("campaign_id"),
        "status": campaign_doc.get("status"),
        "created_at": campaign_doc.get("created_at"),
        "completed_at": campaign_doc.get("completed_at"),
        "live_provider": campaign_doc.get("live_provider"),
        "repeat": campaign_doc.get("repeat"),
        "metrics": summary.get("metrics"),
        "decisions": summary.get("decisions"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def campaign_inspect(args: argparse.Namespace) -> int:
    campaigns_dir, runs_dir = campaign_paths(args)
    out_dir = campaign_dir(campaigns_dir, args.campaign_id)
    summary = load_summary(out_dir) or summarize_campaign(out_dir, runs_dir)
    payload = {
        "campaign": load_campaign(out_dir),
        "run_ids": load_run_index(out_dir),
        "summary": summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def campaign_export(args: argparse.Namespace) -> int:
    campaigns_dir, runs_dir = campaign_paths(args)
    out_dir = campaign_dir(campaigns_dir, args.campaign_id)
    summarize_campaign(out_dir, runs_dir)
    zip_path = export_campaign(out_dir, runs_dir)
    print(json.dumps({"campaign_id": args.campaign_id, "artifact": str(zip_path)}, ensure_ascii=False, indent=2))
    return 0


def model_ids_from_payload(data: Any) -> list[str]:
    raw: Any
    if isinstance(data, dict):
        raw = data.get("data") or data.get("models") or data.get("items") or []
    elif isinstance(data, list):
        raw = data
    else:
        raw = []
    ids: list[str] = []
    for item in raw:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            value = item.get("id") or item.get("name") or item.get("model")
            if value:
                ids.append(str(value))
    return ids


def probe_models_endpoint(client: httpx.Client, model: ModelConfig, auth_type: str) -> dict[str, Any]:
    secret = auth_value(model)
    temp = ModelConfig(
        provider_id=model.provider_id,
        base_url=model.base_url,
        model=model.model,
        api_key_env=model.api_key_env,
        protocol=model.protocol,
        auth_type=auth_type,
    )
    try:
        response = client.get(f"{model.base_url}/v1/models", headers=auth_headers(temp, secret))
    except Exception as exc:
        return {"auth_type": auth_type, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        data = response.json()
    except Exception:
        data = {"body": response.text[:400]}
    ids = model_ids_from_payload(data)
    return {
        "auth_type": auth_type,
        "ok": response.status_code == 200,
        "status": response.status_code,
        "model_count": len(ids),
        "models": ids,
        "body_preview": None if ids else json.dumps(data, ensure_ascii=True)[:300],
    }


def probe_single_call(
    client: httpx.Client,
    base_model: ModelConfig,
    *,
    model_name: str,
    protocol: str,
    auth_type: str,
    timeout_label: str,
) -> dict[str, Any]:
    temp = ModelConfig(
        provider_id=base_model.provider_id,
        base_url=base_model.base_url,
        model=model_name,
        api_key_env=base_model.api_key_env,
        protocol=protocol,
        auth_type=auth_type,
        provider_channel=base_model.provider_channel,
        provider_display_name=base_model.provider_display_name,
    )
    events_file = ROOT / "_probe_events.tmp.jsonl"
    if events_file.exists():
        try:
            events_file.unlink()
        except OSError:
            pass
    completion = call_model(
        client=client,
        model=temp,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        max_tokens=4,
        temperature=0,
        live=True,
        events_file=events_file,
    )
    if events_file.exists():
        try:
            events_file.unlink()
        except OSError:
            pass
    return {
        "model": model_name,
        "protocol": protocol,
        "auth_type": auth_type,
        "ok": completion.metrics.ok,
        "status": "ok" if completion.metrics.ok else "failed",
        "returned_model": completion.metrics.server_model,
        "content_preview": completion.text[:80] if completion.metrics.ok else None,
        "error_preview": completion.metrics.error[:240] if completion.metrics.error else None,
        "total_ms": completion.metrics.total_ms,
        "timeout": timeout_label,
    }


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def probe(args: argparse.Namespace) -> int:
    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    configs = load_two_model_config(models_path)
    timeout = float(args.timeout or 20)
    client_timeout = httpx.Timeout(timeout, connect=min(8.0, timeout), read=timeout, write=min(8.0, timeout), pool=5.0)
    default_aliases = [
        "opus4.8",
        "opus-4.8",
        "claude-opus-4.8",
        "claude-opus-4-8",
        "claude-opus-4",
        "claude-sonnet-4",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    requested = args.model or []
    report: dict[str, Any] = {"providers_file": str(models_path), "roles": {}}
    with httpx.Client(timeout=client_timeout, follow_redirects=True) as client:
        for role, base_model in configs.items():
            role_report: dict[str, Any] = {
                "provider_id": base_model.provider_id,
                "base_url_host": base_url_host(base_model.base_url),
                "api_key_env": base_model.api_key_env,
                "models_endpoints": [],
                "successes": [],
                "failures_sample": [],
            }
            endpoint_models: list[str] = []
            for auth_type in sorted(ALLOWED_AUTH_TYPES):
                endpoint_result = probe_models_endpoint(client, base_model, auth_type)
                role_report["models_endpoints"].append({k: v for k, v in endpoint_result.items() if k != "models"})
                endpoint_models.extend(endpoint_result.get("models") or [])

            candidates = dedupe(requested + endpoint_models + default_aliases)
            if args.max_models:
                candidates = candidates[: int(args.max_models)]
            probes: list[dict[str, Any]] = []
            for model_name in candidates:
                for protocol in sorted(ALLOWED_PROTOCOLS):
                    for auth_type in sorted(ALLOWED_AUTH_TYPES):
                        result = probe_single_call(
                            client,
                            base_model,
                            model_name=model_name,
                            protocol=protocol,
                            auth_type=auth_type,
                            timeout_label=f"{timeout}s",
                        )
                        probes.append(result)
                        if result.get("ok"):
                            role_report["successes"].append(result)
                if args.stop_after_success and role_report["successes"]:
                    break

            role_report["failure_count"] = len([item for item in probes if not item.get("ok")])
            role_report["failures_sample"] = [item for item in probes if not item.get("ok")][: int(args.failure_sample)]
            report["roles"][role] = role_report
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="Two-model headless eval CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run a configured two-model job")
    run_parser.add_argument("--job", default=DEFAULT_JOB, help="job name or path")
    run_parser.add_argument("--providers", type=Path, help="providers config path")
    run_parser.add_argument("--runs-dir", type=Path, help="runs directory")
    run_parser.add_argument("--run-id", help="explicit job id")
    run_parser.add_argument("--live", action="store_true", help="call configured live providers")
    run_parser.add_argument("--timeout", type=float, help="per-request timeout seconds")
    run_parser.add_argument("--tested-max-tokens", type=int)
    run_parser.add_argument("--judge-max-tokens", type=int)
    run_parser.add_argument("--retries", type=int, help="retry transient live provider failures this many times")
    run_parser.add_argument("--retry-backoff", type=float, help="initial retry backoff seconds for live provider failures")
    run_parser.set_defaults(func=run_job)

    inspect_parser = sub.add_parser("inspect", help="inspect a job state")
    inspect_parser.add_argument("--latest", action="store_true")
    inspect_parser.add_argument("--job-id")
    inspect_parser.add_argument("--runs-dir", type=Path)
    inspect_parser.set_defaults(func=inspect_job)

    export_parser = sub.add_parser("export", help="export an acceptance pack")
    export_parser.add_argument("--latest", action="store_true")
    export_parser.add_argument("--job-id")
    export_parser.add_argument("--runs-dir", type=Path)
    export_parser.set_defaults(func=export_job)

    campaign_parser = sub.add_parser("campaign", help="run a repeated campaign")
    campaign_parser.add_argument("--job", default=DEFAULT_JOB, help="job name or path")
    campaign_parser.add_argument("--providers", type=Path, help="providers config path")
    campaign_parser.add_argument("--runs-dir", type=Path, help="runs directory")
    campaign_parser.add_argument("--campaigns-dir", type=Path, help="campaigns directory")
    campaign_parser.add_argument("--campaign-id", help="explicit campaign id")
    campaign_parser.add_argument("--repeat", type=int)
    campaign_parser.add_argument("--resume", action="store_true", help="resume an existing campaign by rerunning incomplete rounds")
    campaign_parser.add_argument("--live", action="store_true", help="call configured live providers")
    campaign_parser.add_argument("--timeout", type=float, help="per-request timeout seconds")
    campaign_parser.add_argument("--tested-max-tokens", type=int)
    campaign_parser.add_argument("--judge-max-tokens", type=int)
    campaign_parser.add_argument("--retries", type=int, help="retry transient live provider failures this many times")
    campaign_parser.add_argument("--retry-backoff", type=float, help="initial retry backoff seconds for live provider failures")
    campaign_parser.set_defaults(func=run_campaign)

    campaign_list_parser = sub.add_parser("campaign-list", help="list campaigns")
    campaign_list_parser.add_argument("--campaigns-dir", type=Path)
    campaign_list_parser.add_argument("--runs-dir", type=Path)
    campaign_list_parser.set_defaults(func=campaign_list)

    campaign_status_parser = sub.add_parser("campaign-status", help="inspect campaign status")
    campaign_status_parser.add_argument("--campaign-id", required=True)
    campaign_status_parser.add_argument("--campaigns-dir", type=Path)
    campaign_status_parser.add_argument("--runs-dir", type=Path)
    campaign_status_parser.set_defaults(func=campaign_status)

    campaign_inspect_parser = sub.add_parser("campaign-inspect", help="inspect a campaign summary")
    campaign_inspect_parser.add_argument("--campaign-id", required=True)
    campaign_inspect_parser.add_argument("--campaigns-dir", type=Path)
    campaign_inspect_parser.add_argument("--runs-dir", type=Path)
    campaign_inspect_parser.set_defaults(func=campaign_inspect)

    campaign_export_parser = sub.add_parser("campaign-export", help="export a campaign acceptance pack")
    campaign_export_parser.add_argument("--campaign-id", required=True)
    campaign_export_parser.add_argument("--campaigns-dir", type=Path)
    campaign_export_parser.add_argument("--runs-dir", type=Path)
    campaign_export_parser.set_defaults(func=campaign_export)

    probe_parser = sub.add_parser("probe", help="probe model/protocol/auth combinations")
    probe_parser.add_argument("--providers", type=Path, help="providers config path")
    probe_parser.add_argument("--model", action="append", default=[], help="candidate model id; repeatable")
    probe_parser.add_argument("--max-models", type=int, default=20)
    probe_parser.add_argument("--timeout", type=float, default=20)
    probe_parser.add_argument("--failure-sample", type=int, default=12)
    probe_parser.add_argument("--stop-after-success", action="store_true")
    probe_parser.set_defaults(func=probe)

    args = parser.parse_args()
    if args.command in {"inspect", "export"} and not args.latest and not args.job_id:
        parser.error(f"{args.command} requires --latest or --job-id")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
