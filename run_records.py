from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from redaction import redact_text


RUN_RECORD_SCHEMA_VERSION = "run_record_v1"
ALLOWED_RUN_STATUSES = {"completed", "failed", "stopped", "partial"}
SCHEMA_PATH = Path(__file__).with_name("run_record.schema.json")


def _as_dict(value: Any) -> dict[str, Any]:
    """Narrow Any -> dict (empty when not a dict) for the type checker."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Narrow Any -> list (empty when not a list) for the type checker."""
    return value if isinstance(value, list) else []


def as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    data: dict[str, Any] = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            attr = getattr(value, key)
        except Exception:
            continue
        if callable(attr):
            continue
        data[key] = attr
    return data


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def base_url_host(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    return parsed.netloc or parsed.path or None


def extract_raw_event_types(events_file: str | Path | None, limit: int = 5000) -> list[str]:
    if not events_file:
        return []
    path = Path(events_file)
    if not path.exists():
        return []

    seen: set[str] = set()
    ordered: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                if index >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    dtype = "non_json_event"
                else:
                    dtype = str(data.get("type") or data.get("event") or "unknown")
                if dtype not in seen:
                    seen.add(dtype)
                    ordered.append(dtype)
    except OSError:
        return []
    return ordered


def build_run_record(
    *,
    run_id: str,
    timestamp: str,
    benchmark_mode: str,
    formula_version: str,
    runner: str,
    status: str,
    task: dict[str, Any],
    provider: Any,
    metrics: Any,
    final_score: dict[str, Any] | None,
    response_text: str,
    response_file: str | Path,
    events_file: str | Path,
    max_tokens: int,
    temperature: float | None,
    system_prompt: str | None,
    rule_score: dict[str, Any] | None = None,
    judge_score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider_data = as_plain_dict(provider)
    metrics_data = as_plain_dict(metrics)
    task_id = str(task.get("id") or "unknown_task")
    provider_id = str(provider_data.get("id") or "unknown_provider")
    model_requested = provider_data.get("model")
    model_returned = metrics_data.get("server_model")
    claimed_model = provider_data.get("claimed_model") or model_requested
    baseline_model = provider_data.get("baseline_model") or claimed_model
    provider_channel = str(provider_data.get("provider_channel") or "unknown").strip().lower()
    leaderboard_group = provider_data.get("leaderboard_group")
    if not leaderboard_group:
        if provider_channel in {"official", "direct"}:
            leaderboard_group = "official_baseline"
        elif provider_channel == "gateway":
            leaderboard_group = "gateway_candidate"
        elif provider_channel == "byo":
            leaderboard_group = "imported"
        else:
            leaderboard_group = "unknown"
    status_value = status if status in ALLOWED_RUN_STATUSES else "failed"

    request_fingerprint = {
        "api_style": "anthropic_messages",
        "model": model_requested,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system_hash": text_hash(system_prompt or "") if system_prompt else None,
        "prompt_hash": text_hash(str(task.get("prompt") or "")),
        "messages_count": 1,
    }

    final_score = final_score or {}
    rule_score = rule_score if rule_score is not None else final_score
    response_file_str = str(response_file)
    events_file_str = str(events_file)

    return {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "record_id": f"{run_id}:{provider_id}:{task_id}",
        "run": {
            "run_id": run_id,
            "timestamp": timestamp,
            "benchmark_mode": benchmark_mode,
            "formula_version": formula_version,
            "runner": runner,
            "status": status_value,
        },
        "task": {
            "id": task.get("id"),
            "category": task.get("category"),
            "enterprise_dimension": task.get("enterprise_dimension"),
            "difficulty": task.get("difficulty"),
            "scoring_type": task.get("scoring_type"),
            "risk_tags": task.get("risk_tags") or [],
            "point_value": task.get("point_value"),
            "scoring_confidence": task.get("scoring_confidence"),
        },
        "provider": {
            "id": provider_id,
            "api_style": "anthropic_messages",
            "base_url_host": base_url_host(str(provider_data.get("base_url") or "")),
            "auth_env_name": provider_data.get("auth_env"),
            "model_requested": model_requested,
            "model_returned": model_returned,
            "provider_channel": provider_channel,
            "provider_display_name": provider_data.get("provider_display_name") or provider_id,
            "claimed_model": claimed_model,
            "baseline_model": baseline_model,
            "leaderboard_group": leaderboard_group,
        },
        "request": {
            "request_hash": stable_json_hash(request_fingerprint),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system_present": bool(system_prompt),
            "messages_count": 1,
            "prompt_hash": request_fingerprint["prompt_hash"],
        },
        "response": {
            "response_file": response_file_str,
            "events_file": events_file_str,
            "content_chars": metrics_data.get("content_chars"),
            "normalized_text_hash": text_hash(response_text or ""),
        },
        "telemetry": {
            "ok": metrics_data.get("ok"),
            "error": redact_text(metrics_data.get("error"), max_chars=500),
            "first_event_ms": metrics_data.get("first_event_ms"),
            "first_content_token_ms": metrics_data.get("first_content_token_ms"),
            "total_ms": metrics_data.get("total_ms"),
            "event_count": metrics_data.get("event_count"),
            "content_event_count": metrics_data.get("content_event_count"),
            "stop_reason": metrics_data.get("stop_reason"),
        },
        "usage": {
            "input_tokens": metrics_data.get("input_tokens"),
            "output_tokens": metrics_data.get("output_tokens"),
            "cache_creation_input_tokens": metrics_data.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": metrics_data.get("cache_read_input_tokens"),
        },
        "scoring": {
            "rule_score": rule_score,
            "judge_score": judge_score,
            "final_score": final_score,
            "judge_provider": (judge_score or {}).get("provider") if isinstance(judge_score, dict) else None,
            "judge_model_requested": (judge_score or {}).get("model_requested") if isinstance(judge_score, dict) else None,
            "judge_model_returned": (judge_score or {}).get("model_returned") if isinstance(judge_score, dict) else None,
        },
        "trace": {
            "tool_calls": [],
            "raw_event_types": extract_raw_event_types(events_file_str),
        },
        "artifacts": {
            "response_file": response_file_str,
            "events_file": events_file_str,
        },
    }


def append_run_record_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


@lru_cache(maxsize=1)
def load_run_record_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    if not isinstance(schema, dict):
        raise ValueError(f"{SCHEMA_PATH} must contain a JSON object")
    return schema


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def _schema_type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(value, item) for item in expected)
    if isinstance(expected, str):
        return _type_matches(value, expected)
    return True


def _validate_schema_subset(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path} must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(str(item) for item in schema["enum"])
        errors.append(f"{path} must be one of: {allowed}")
    if "type" in schema and not _schema_type_matches(value, schema["type"]):
        errors.append(f"{path} must match type {schema['type']}")
        return errors
    if isinstance(value, str) and "minLength" in schema and len(value) < int(schema["minLength"]):
        errors.append(f"{path} must have length >= {schema['minLength']}")
    if isinstance(value, dict):
        required = _as_list(schema.get("required"))
        for key in required:
            if key not in value:
                errors.append(f"missing {path}.{key}" if path != "$" else f"missing top-level key: {key}")
        properties = _as_dict(schema.get("properties"))
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, dict):
                child_path = f"{path}.{key}" if path != "$" else key
                errors.extend(_validate_schema_subset(value[key], child_schema, child_path))
    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(_validate_schema_subset(item, item_schema, f"{path}[{index}]"))
    return errors


def validate_run_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = _validate_schema_subset(record, load_run_record_schema(), "$")
    required_top = [
        "schema_version",
        "record_id",
        "run",
        "task",
        "provider",
        "request",
        "response",
        "telemetry",
        "usage",
        "scoring",
        "trace",
        "artifacts",
    ]
    for key in required_top:
        if key not in record:
            errors.append(f"missing top-level key: {key}")

    if record.get("schema_version") != RUN_RECORD_SCHEMA_VERSION:
        errors.append("schema_version must be run_record_v1")
    if not isinstance(record.get("record_id"), str) or not record.get("record_id"):
        errors.append("record_id must be a non-empty string")

    nested_required = {
        "run": ["run_id", "timestamp", "benchmark_mode", "formula_version", "runner", "status"],
        "task": ["id", "category", "enterprise_dimension", "difficulty", "scoring_type", "risk_tags", "point_value", "scoring_confidence"],
        "provider": ["id", "api_style", "base_url_host", "model_requested", "model_returned"],
        "request": ["request_hash", "max_tokens", "temperature", "system_present", "messages_count"],
        "response": ["response_file", "events_file", "content_chars", "normalized_text_hash"],
        "telemetry": ["ok", "error", "first_event_ms", "first_content_token_ms", "total_ms", "event_count", "content_event_count", "stop_reason"],
        "usage": ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"],
        "scoring": ["rule_score", "judge_score", "final_score", "judge_provider", "judge_model_requested", "judge_model_returned"],
        "trace": ["tool_calls", "raw_event_types"],
        "artifacts": ["response_file", "events_file"],
    }
    for parent, keys in nested_required.items():
        value = record.get(parent)
        if not isinstance(value, dict):
            errors.append(f"{parent} must be an object")
            continue
        for key in keys:
            if key not in value:
                errors.append(f"missing {parent}.{key}")

    run = _as_dict(record.get("run"))
    if run.get("status") not in ALLOWED_RUN_STATUSES:
        errors.append("run.status must be completed, failed, stopped, or partial")
    provider = _as_dict(record.get("provider"))
    if provider.get("api_style") not in {"anthropic_messages", "openai_chat"}:
        errors.append("provider.api_style must be anthropic_messages or openai_chat")
    trace = _as_dict(record.get("trace"))
    if not isinstance(trace.get("tool_calls"), list):
        errors.append("trace.tool_calls must be an array")
    if not isinstance(trace.get("raw_event_types"), list):
        errors.append("trace.raw_event_types must be an array")
    return errors
