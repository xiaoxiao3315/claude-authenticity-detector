from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import httpx

import evidence_registry as registry
from run_eval import Provider, auth_header, iter_sse_events, load_json, load_providers
from run_records import stable_json_hash


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
STATUS_RANK = {PASS: 0, WARN: 1, FAIL: 2}
COMPATIBILITY_RECORD_VERSION = "compatibility_record_v1"


@dataclass
class ProbeMetrics:
    ok: bool = False
    error: str | None = None
    http_status: int | None = None
    first_event_ms: float | None = None
    first_content_token_ms: float | None = None
    total_ms: float | None = None
    event_count: int = 0
    content_event_count: int = 0
    thinking_event_count: int = 0
    tool_use_event_count: int = 0
    input_json_delta_count: int = 0
    content_chars: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    server_model: str | None = None
    stop_reason: str | None = None
    event_types: list[str] = field(default_factory=list)


def base_url_host(base_url: str) -> str:
    parsed = urlparse(base_url)
    return parsed.netloc or parsed.path


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # tolerate a corrupt/partial line (e.g. a truncated events file) the
            # same way the other readers do, rather than crashing the whole probe.
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "timestamp",
        "case_id",
        "category",
        "provider_id",
        "status",
        "failed_checks",
        "warned_checks",
        "model_requested",
        "model_returned",
        "ok",
        "error",
        "first_event_ms",
        "first_content_token_ms",
        "total_ms",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "response_file",
        "events_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_suite(path: Path) -> dict[str, Any]:
    data = load_json(path)
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("compatibility suite must contain a non-empty cases array")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each compatibility case must be an object")
        if not case.get("id"):
            raise ValueError("compatibility case missing id")
        if not isinstance(case.get("request"), dict):
            raise ValueError(f"compatibility case {case.get('id')} missing request")
    return data


def compatibility_case_count(suite_path: Path) -> int:
    return len(load_suite(suite_path).get("cases") or [])


def worst_status(statuses: Iterable[str]) -> str:
    current = PASS
    for status in statuses:
        if STATUS_RANK.get(status, STATUS_RANK[FAIL]) > STATUS_RANK[current]:
            current = status
    return current


def check(name: str, status: str, details: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "details": details,
        "evidence": evidence or {},
    }


def build_payload(provider: Provider, suite: dict[str, Any], case: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    max_tokens = int(request.get("max_tokens") or case.get("max_tokens") or suite.get("default_max_tokens") or 512)
    temperature = request.get("temperature", case.get("temperature", suite.get("default_temperature", 0)))
    payload: dict[str, Any] = {
        "model": provider.model,
        "max_tokens": max_tokens,
        "stream": True,
    }
    messages = request.get("messages")
    if messages is None:
        prompt = str(request.get("prompt") or "")
        messages = [{"role": "user", "content": prompt}]
    payload["messages"] = messages
    if request.get("system") is not None:
        payload["system"] = request.get("system")
    if isinstance(request.get("tools"), list):
        payload["tools"] = request.get("tools")
    if request.get("tool_choice") is not None:
        payload["tool_choice"] = request.get("tool_choice")
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def update_usage(metrics: ProbeMetrics, usage: dict[str, Any]) -> None:
    for attr, key in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_creation_input_tokens", "cache_creation_input_tokens"),
        ("cache_read_input_tokens", "cache_read_input_tokens"),
    ):
        if key in usage and usage[key] is not None:
            try:
                setattr(metrics, attr, int(usage[key]))
            except (TypeError, ValueError):
                pass


def call_streaming_messages(
    client: httpx.Client,
    provider: Provider,
    payload: dict[str, Any],
    events_path: Path,
) -> tuple[ProbeMetrics, str]:
    metrics = ProbeMetrics(ok=False)
    auth_name, auth_value = auth_header(provider)
    headers = {
        auth_name: auth_value,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "accept": "text/event-stream",
    }
    url = f"{provider.base_url}/v1/messages"
    t_send = time.perf_counter()
    first_event_t: float | None = None
    first_content_t: float | None = None
    response_parts: list[str] = []
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", encoding="utf-8") as events_file:
        try:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                metrics.http_status = resp.status_code
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", errors="replace")[:1000]
                    metrics.error = f"HTTP {resp.status_code}: {body}"
                    metrics.total_ms = (time.perf_counter() - t_send) * 1000
                    return metrics, ""
                for event_name, data_str in iter_sse_events(resp.iter_raw()):
                    now = time.perf_counter()
                    metrics.event_count += 1
                    if first_event_t is None:
                        first_event_t = now
                        metrics.first_event_ms = (now - t_send) * 1000
                    if data_str:
                        events_file.write(data_str + "\n")
                        events_file.flush()
                    try:
                        data = json.loads(data_str) if data_str else {}
                    except json.JSONDecodeError:
                        continue
                    dtype = data.get("type") or event_name
                    metrics.event_types.append(str(dtype))
                    if dtype == "message_start":
                        message = data.get("message") or {}
                        metrics.server_model = message.get("model")
                        update_usage(metrics, message.get("usage") or {})
                    elif dtype == "content_block_start":
                        block = data.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            metrics.tool_use_event_count += 1
                    elif dtype == "content_block_delta":
                        delta = data.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            text = delta.get("text") or ""
                            if text:
                                response_parts.append(text)
                                metrics.content_event_count += 1
                                metrics.content_chars += len(text)
                                if first_content_t is None:
                                    first_content_t = now
                                    metrics.first_content_token_ms = (now - t_send) * 1000
                        elif delta.get("type") == "thinking_delta":
                            metrics.thinking_event_count += 1
                        elif delta.get("type") == "input_json_delta":
                            metrics.input_json_delta_count += 1
                    elif dtype == "message_delta":
                        delta = data.get("delta") or {}
                        if "stop_reason" in delta:
                            metrics.stop_reason = delta.get("stop_reason")
                        update_usage(metrics, data.get("usage") or {})
            metrics.total_ms = (time.perf_counter() - t_send) * 1000
            if not response_parts and metrics.tool_use_event_count <= 0:
                metrics.error = "no assistant text produced"
                return metrics, ""
            metrics.ok = True
            return metrics, "".join(response_parts)
        except httpx.HTTPError as exc:
            metrics.error = f"{type(exc).__name__}: {exc}"
            metrics.total_ms = (time.perf_counter() - t_send) * 1000
            return metrics, "".join(response_parts)


def usage_value(metrics: ProbeMetrics, attr: str) -> int | None:
    value = getattr(metrics, attr)
    return value if isinstance(value, int) else None


def total_input(metrics: ProbeMetrics) -> int | None:
    pieces = [
        usage_value(metrics, "input_tokens"),
        usage_value(metrics, "cache_creation_input_tokens"),
        usage_value(metrics, "cache_read_input_tokens"),
    ]
    if all(piece is None for piece in pieces):
        return None
    return sum(piece or 0 for piece in pieces)


def index_of(event_types: list[str], event_type: str) -> int | None:
    try:
        return event_types.index(event_type)
    except ValueError:
        return None


def evaluate_common(provider: Provider, metrics: ProbeMetrics, *, require_text: bool = True) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if metrics.http_status == 200:
        checks.append(check("http_status", PASS, "provider returned HTTP 200", {"http_status": metrics.http_status}))
    else:
        checks.append(check("http_status", FAIL, metrics.error or "provider did not return HTTP 200", {"http_status": metrics.http_status}))
    if not require_text:
        checks.append(
            check(
                "displayable_text",
                PASS,
                "displayable text is optional for this compatibility case",
                {"content_chars": metrics.content_chars},
            )
        )
    elif metrics.content_chars > 0:
        checks.append(check("displayable_text", PASS, "response included displayable text", {"content_chars": metrics.content_chars}))
    else:
        checks.append(check("displayable_text", FAIL, metrics.error or "response did not include displayable text", {"content_chars": metrics.content_chars}))
    if not metrics.server_model:
        checks.append(check("model_identity", WARN, "model_returned is missing", {"model_requested": provider.model, "model_returned": metrics.server_model}))
    elif metrics.server_model != provider.model:
        checks.append(check("model_identity", FAIL, "model_returned differs from model_requested", {"model_requested": provider.model, "model_returned": metrics.server_model}))
    else:
        checks.append(check("model_identity", PASS, "model_returned matches model_requested", {"model_requested": provider.model, "model_returned": metrics.server_model}))
    return checks


def is_forced_tool_choice_mode_unsupported(case: dict[str, Any], metrics: ProbeMetrics) -> bool:
    if str(case.get("id") or "") != "tool_call_probe":
        return False
    if str(case.get("category") or "") != "tool_call":
        return False
    error_text = str(metrics.error or "").lower()
    if metrics.http_status != 400:
        return False
    return "tool_choice" in error_text and ("thinking" in error_text or "not support" in error_text or "unsupported" in error_text)


def evaluate_forced_tool_choice_mode_unsupported(provider: Provider, metrics: ProbeMetrics) -> list[dict[str, Any]]:
    checks = [
        check(
            "http_status",
            WARN,
            "tool_call_probe returned HTTP 400 because forced tool_choice is not supported in this provider/model mode",
            {"http_status": metrics.http_status, "error": metrics.error},
        ),
        check(
            "tool_choice_support",
            WARN,
            "forced tool_choice is unsupported, so tool-call compatibility was not fully validated",
            {"tool_choice_supported": False},
        ),
    ]
    if not metrics.server_model:
        checks.append(check("model_identity", WARN, "model_returned is missing", {"model_requested": provider.model, "model_returned": metrics.server_model}))
    elif metrics.server_model != provider.model:
        checks.append(check("model_identity", FAIL, "model_returned differs from model_requested", {"model_requested": provider.model, "model_returned": metrics.server_model}))
    else:
        checks.append(check("model_identity", PASS, "model_returned matches model_requested", {"model_requested": provider.model, "model_returned": metrics.server_model}))
    return checks


def evaluate_expected_substring(case: dict[str, Any], response_text: str) -> list[dict[str, Any]]:
    expected = case.get("expected_substring")
    if not expected:
        return []
    if str(expected).lower() in response_text.lower():
        return [check("expected_substring", PASS, "expected marker found in response")]
    return [check("expected_substring", WARN, "expected marker missing from response", {"expected": expected})]


def evaluate_usage(primary: ProbeMetrics, secondary: ProbeMetrics | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required = ["input_tokens", "output_tokens"]
    missing_required = [name for name in required if usage_value(primary, name) is None]
    if missing_required:
        checks.append(check("usage_required_fields", FAIL, "required usage fields missing", {"missing": missing_required}))
    else:
        checks.append(check("usage_required_fields", PASS, "input_tokens and output_tokens are present"))
    cache_missing = [
        name
        for name in ("cache_creation_input_tokens", "cache_read_input_tokens")
        if usage_value(primary, name) is None
    ]
    if cache_missing:
        checks.append(check("usage_cache_fields", WARN, "cache usage fields are missing", {"missing": cache_missing}))
    else:
        checks.append(check("usage_cache_fields", PASS, "cache usage fields are present"))
    if secondary:
        short_total = total_input(primary)
        long_total = total_input(secondary)
        if short_total is None or long_total is None:
            checks.append(check("usage_prompt_scale", FAIL, "cannot compare input token scale", {"short_total_input": short_total, "long_total_input": long_total}))
        elif long_total > short_total:
            checks.append(check("usage_prompt_scale", PASS, "longer prompt reports more input tokens", {"short_total_input": short_total, "long_total_input": long_total}))
        else:
            checks.append(check("usage_prompt_scale", FAIL, "longer prompt did not report more input tokens", {"short_total_input": short_total, "long_total_input": long_total}))
    return checks


def evaluate_sse(metrics: ProbeMetrics) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    event_types = metrics.event_types
    required = ["message_start", "content_block_delta", "message_delta", "message_stop"]
    missing = [event for event in required if event not in event_types]
    if missing:
        checks.append(check("sse_required_events", FAIL, "required SSE events missing", {"missing": missing, "event_types": event_types}))
    else:
        checks.append(check("sse_required_events", PASS, "required SSE events present", {"event_types": event_types}))
    positions = {event: index_of(event_types, event) for event in required}
    p_start = positions["message_start"]
    p_delta = positions["content_block_delta"]
    p_stop = positions["message_stop"]
    if p_start is not None and p_delta is not None and p_stop is not None and p_start < p_delta < p_stop:
        checks.append(check("sse_event_order", PASS, "SSE event order is plausible", positions))
    else:
        checks.append(check("sse_event_order", FAIL, "SSE event order is invalid or incomplete", positions))
    if metrics.first_event_ms is not None and metrics.first_content_token_ms is not None:
        checks.append(check("sse_latency", PASS, "first event and first text latency were captured", {"first_event_ms": metrics.first_event_ms, "first_content_token_ms": metrics.first_content_token_ms}))
    else:
        checks.append(check("sse_latency", FAIL, "first event or first text latency is missing", {"first_event_ms": metrics.first_event_ms, "first_content_token_ms": metrics.first_content_token_ms}))
    return checks


def type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if value is None:
        return "null"
    return type(value).__name__


def evaluate_json(case: dict[str, Any], response_text: str) -> list[dict[str, Any]]:
    expected = case.get("expected_json")
    if expected is None:
        return []
    text = response_text.strip()
    checks: list[dict[str, Any]] = []
    if text.startswith("```") or text.endswith("```"):
        checks.append(check("json_no_markdown", FAIL, "JSON response is wrapped in markdown"))
    else:
        checks.append(check("json_no_markdown", PASS, "JSON response is not markdown wrapped"))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        checks.append(check("json_parse", FAIL, f"JSON parse failed: {exc}"))
        return checks
    checks.append(check("json_parse", PASS, "response parsed as JSON"))
    if not isinstance(parsed, dict):
        checks.append(check("json_schema", FAIL, "parsed JSON is not an object", {"actual_type": type_name(parsed)}))
        return checks
    missing = [key for key in expected if key not in parsed]
    extra = [key for key in parsed if key not in expected]
    type_mismatches = [
        {"key": key, "expected_type": type_name(value), "actual_type": type_name(parsed.get(key))}
        for key, value in expected.items()
        if key in parsed and type(parsed.get(key)) is not type(value)
    ]
    value_mismatches = [
        {"key": key, "expected": value, "actual": parsed.get(key)}
        for key, value in expected.items()
        if key in parsed and parsed.get(key) != value
    ]
    if missing or extra or type_mismatches or value_mismatches:
        checks.append(
            check(
                "json_schema",
                FAIL,
                "JSON object does not match expected schema/value contract",
                {
                    "missing": missing,
                    "extra": extra,
                    "type_mismatches": type_mismatches,
                    "value_mismatches": value_mismatches,
                },
            )
        )
    else:
        checks.append(check("json_schema", PASS, "JSON object matches expected schema/value contract"))
    return checks


def request_has_cache_control(value: Any) -> bool:
    if isinstance(value, dict):
        if "cache_control" in value:
            return True
        return any(request_has_cache_control(item) for item in value.values())
    if isinstance(value, list):
        return any(request_has_cache_control(item) for item in value)
    return False


def evaluate_cache(case: dict[str, Any], primary: ProbeMetrics, secondary: ProbeMetrics | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    cache_sent = request_has_cache_control(case.get("request"))
    checks.append(
        check(
            "cache_control_sent",
            PASS if cache_sent else FAIL,
            "cache_control was included in the probe request" if cache_sent else "cache_control was not included in the probe request",
        )
    )
    cache_values = {
        "cache_creation_input_tokens": primary.cache_creation_input_tokens,
        "cache_read_input_tokens": primary.cache_read_input_tokens,
    }
    if any(value is not None for value in cache_values.values()):
        checks.append(check("cache_usage_observed", PASS, "cache usage fields were reported", cache_values))
    else:
        checks.append(check("cache_usage_observed", WARN, "cache probe ran, but provider did not report cache usage fields", cache_values))
    if secondary:
        secondary_values = {
            "cache_creation_input_tokens": secondary.cache_creation_input_tokens,
            "cache_read_input_tokens": secondary.cache_read_input_tokens,
        }
        if any(value is not None for value in secondary_values.values()):
            checks.append(check("cache_repeat_usage_observed", PASS, "repeat cache probe reported cache usage fields", secondary_values))
        else:
            checks.append(check("cache_repeat_usage_observed", WARN, "repeat cache probe did not report cache usage fields", secondary_values))
    return checks


def evaluate_tool_call(metrics: ProbeMetrics) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if metrics.tool_use_event_count > 0:
        checks.append(check("tool_use_event", PASS, "tool_use content block was observed", {"tool_use_event_count": metrics.tool_use_event_count}))
    else:
        checks.append(check("tool_use_event", FAIL, "tool_use content block was not observed", {"tool_use_event_count": metrics.tool_use_event_count}))
    if metrics.input_json_delta_count > 0:
        checks.append(check("tool_input_delta", PASS, "tool input JSON delta was observed", {"input_json_delta_count": metrics.input_json_delta_count}))
    else:
        checks.append(check("tool_input_delta", WARN, "tool_use was requested, but no input_json_delta event was observed", {"input_json_delta_count": metrics.input_json_delta_count}))
    return checks


def evaluate_case(
    *,
    suite: dict[str, Any],
    case: dict[str, Any],
    provider: Provider,
    primary_metrics: ProbeMetrics,
    primary_response_text: str,
    secondary_metrics: ProbeMetrics | None = None,
) -> list[dict[str, Any]]:
    category = str(case.get("category") or "")
    if is_forced_tool_choice_mode_unsupported(case, primary_metrics):
        return evaluate_forced_tool_choice_mode_unsupported(provider, primary_metrics)
    checks = evaluate_common(provider, primary_metrics, require_text=category != "tool_call")
    if category == "messages":
        checks.extend(evaluate_expected_substring(case, primary_response_text))
    if category == "usage":
        checks.extend(evaluate_usage(primary_metrics, secondary_metrics))
    if category == "sse":
        checks.extend(evaluate_sse(primary_metrics))
    if category == "json":
        checks.extend(evaluate_json(case, primary_response_text))
    if category == "cache":
        checks.extend(evaluate_cache(case, primary_metrics, secondary_metrics))
    if category == "tool_call":
        checks.extend(evaluate_tool_call(primary_metrics))
    return checks


def wait_for_resume(job_control: dict[str, Any] | None, progress_callback: Callable[[dict[str, Any]], None] | None, completed: int, total: int) -> bool:
    if not job_control or not job_control.get("pause_requested"):
        return False
    resume_event = job_control.get("resume_event")
    if progress_callback:
        progress_callback({"event": "run_paused", "completed_tasks": completed, "total_tasks": total})
    while job_control.get("pause_requested") and not job_control.get("stop_requested"):
        if resume_event:
            resume_event.wait(0.5)
        else:
            time.sleep(0.5)
    if job_control.get("stop_requested"):
        return True
    if progress_callback:
        progress_callback({"event": "run_resumed", "completed_tasks": completed, "total_tasks": total})
    return False


def run_probe(
    *,
    client: httpx.Client,
    provider: Provider,
    suite: dict[str, Any],
    case: dict[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = str(case["id"])
    provider_id = provider.id
    request = case["request"]
    payload = build_payload(provider, suite, case, request)
    events_path = run_dir / "events" / provider_id / f"{case_id}.jsonl"
    response_path = run_dir / "responses" / provider_id / f"{case_id}.txt"
    metrics, response_text = call_streaming_messages(client, provider, payload, events_path)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(response_text, encoding="utf-8")

    secondary_result: dict[str, Any] | None = None
    secondary_metrics: ProbeMetrics | None = None
    if isinstance(case.get("secondary_request"), dict):
        secondary_payload = build_payload(provider, suite, case, case["secondary_request"])
        secondary_events_path = run_dir / "events" / provider_id / f"{case_id}__secondary.jsonl"
        secondary_response_path = run_dir / "responses" / provider_id / f"{case_id}__secondary.txt"
        secondary_metrics, secondary_response = call_streaming_messages(client, provider, secondary_payload, secondary_events_path)
        secondary_response_path.parent.mkdir(parents=True, exist_ok=True)
        secondary_response_path.write_text(secondary_response, encoding="utf-8")
        secondary_result = {
            "request_hash": stable_json_hash(secondary_payload),
            "metrics": asdict(secondary_metrics),
            "response_file": str(secondary_response_path),
            "events_file": str(secondary_events_path),
            "content_chars": len(secondary_response),
        }

    checks = evaluate_case(
        suite=suite,
        case=case,
        provider=provider,
        primary_metrics=metrics,
        primary_response_text=response_text,
        secondary_metrics=secondary_metrics,
    )
    status = worst_status(check_item["status"] for check_item in checks)
    failed = [item["name"] for item in checks if item["status"] == FAIL]
    warned = [item["name"] for item in checks if item["status"] == WARN]
    evidence = {
        "http_status": metrics.http_status,
        "event_types": metrics.event_types,
        "event_count": metrics.event_count,
        "content_event_count": metrics.content_event_count,
        "thinking_event_count": metrics.thinking_event_count,
        "tool_use_event_count": metrics.tool_use_event_count,
        "input_json_delta_count": metrics.input_json_delta_count,
        "content_chars": metrics.content_chars,
        "stop_reason": metrics.stop_reason,
        "base_url_host": base_url_host(provider.base_url),
        "cache_semantics_checked": case.get("category") == "cache",
        "tool_use_checked": case.get("category") == "tool_call",
    }
    if secondary_result:
        evidence["secondary_probe"] = secondary_result
    record = {
        "schema_version": COMPATIBILITY_RECORD_VERSION,
        "run_id": run_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "case_id": case_id,
        "category": case.get("category"),
        "provider_id": provider_id,
        "status": status,
        "checks": checks,
        "evidence": evidence,
        "request": {
            "request_hash": stable_json_hash(payload),
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "system_present": "system" in payload,
            "messages_count": len(payload.get("messages") or []),
        },
        "usage": {
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "cache_creation_input_tokens": metrics.cache_creation_input_tokens,
            "cache_read_input_tokens": metrics.cache_read_input_tokens,
        },
        "latency": {
            "first_event_ms": metrics.first_event_ms,
            "first_content_token_ms": metrics.first_content_token_ms,
            "total_ms": metrics.total_ms,
        },
        "model_requested": provider.model,
        "model_returned": metrics.server_model,
        "response_file": str(response_path),
        "events_file": str(events_path),
        "artifacts": {
            "response_file": str(response_path),
            "events_file": str(events_path),
        },
        "error": metrics.error,
    }
    summary_row = {
        "run_id": run_dir.name,
        "timestamp": record["timestamp"],
        "case_id": case_id,
        "category": case.get("category"),
        "provider_id": provider_id,
        "status": status,
        "failed_checks": ";".join(failed),
        "warned_checks": ";".join(warned),
        "model_requested": provider.model,
        "model_returned": metrics.server_model,
        "ok": metrics.ok,
        "error": metrics.error,
        "first_event_ms": metrics.first_event_ms,
        "first_content_token_ms": metrics.first_content_token_ms,
        "total_ms": metrics.total_ms,
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "cache_creation_input_tokens": metrics.cache_creation_input_tokens,
        "cache_read_input_tokens": metrics.cache_read_input_tokens,
        "response_file": str(response_path),
        "events_file": str(events_path),
    }
    return record, summary_row


def run_compatibility_suite(
    *,
    runs_dir: Path,
    provider: Provider,
    suite_path: Path,
    timeout: float = 120.0,
    run_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    job_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    cases = list(suite.get("cases") or [])
    compat_run_id = run_id or registry.unique_artifact_id("compat")
    run_dir = runs_dir / compat_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    stopped = False
    stop_reason = None
    if progress_callback:
        progress_callback({"event": "run_started", "total_tasks": len(cases), "benchmark_mode": "compatibility", "current_provider": provider.id})
    with httpx.Client(timeout=timeout, http2=False) as client:
        for index, case in enumerate(cases):
            if job_control and job_control.get("stop_requested"):
                stopped = True
                stop_reason = "user_stop_requested"
                break
            if wait_for_resume(job_control, progress_callback, index, len(cases)):
                stopped = True
                stop_reason = "user_stop_requested"
                break
            case_id = str(case["id"])
            if progress_callback:
                progress_callback({"event": "task_started", "current_task_id": case_id, "current_provider": provider.id, "phase": "compat_calling_provider", "completed_tasks": index})
            record, row = run_probe(client=client, provider=provider, suite=suite, case=case, run_dir=run_dir)
            if progress_callback:
                progress_callback({"event": "task_phase", "current_task_id": case_id, "current_provider": provider.id, "phase": "compat_checking_evidence", "completed_tasks": index})
            records.append(record)
            summary_rows.append(row)
            if progress_callback:
                progress_callback({"event": "task_completed", "current_task_id": case_id, "current_provider": provider.id, "phase": "compat_completed", "completed_tasks": index + 1, "ok": record["status"] != FAIL, "error": record.get("error")})
    records_path = run_dir / "compatibility_records.jsonl"
    summary_path = run_dir / "compatibility_summary.csv"
    manifest_path = run_dir / "compatibility_manifest.json"
    write_jsonl(records_path, records)
    write_summary_csv(summary_path, summary_rows)
    statuses = [record.get("status", FAIL) for record in records]
    suite_status = worst_status(statuses) if records else (FAIL if not stopped else WARN)
    status_counts = {PASS: 0, WARN: 0, FAIL: 0}
    for status in statuses:
        status_counts[status] = status_counts.get(status, 0) + 1
    completed_at = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "run_id": compat_run_id,
        "suite_version": suite.get("suite_version") or "compatibility_suite_v1",
        "status": "stopped" if stopped else "completed",
        "suite_status": suite_status,
        "created_at": created_at,
        "completed_at": completed_at,
        "provider_id": provider.id,
        "base_url_host": base_url_host(provider.base_url),
        "model_requested": provider.model,
        "case_count": len(cases),
        "record_count": len(records),
        "status_counts": status_counts,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "records_file": str(records_path.resolve()),
        "summary_file": str(summary_path.resolve()),
        "manifest_file": str(manifest_path.resolve()),
    }
    write_json(manifest_path, manifest)
    if progress_callback:
        progress_callback({"event": "run_stopped" if stopped else "run_completed", "run_id": compat_run_id, "completed_tasks": len(records), "total_tasks": len(cases), "stop_reason": stop_reason})
    return {"run_id": compat_run_id, "record_count": len(records), "suite_status": suite_status, "manifest": manifest}


def list_compatibility_runs(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    runs: list[dict[str, Any]] = []
    for child in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        manifest_path = child / "compatibility_manifest.json"
        if not child.is_dir() or not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        runs.append(manifest)
    return runs


def read_compatibility_run(runs_dir: Path, run_id: str) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    manifest_path = run_dir / "compatibility_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"compatibility run not found: {run_id}")
    return {
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "summary": read_csv_rows(run_dir / "compatibility_summary.csv"),
        "records": read_jsonl(run_dir / "compatibility_records.jsonl"),
    }


def self_test() -> None:
    provider = Provider(id="fake_provider", base_url="https://example.invalid/anthropic", model="fake-model", auth_type="bearer", auth_env="FAKE_KEY")
    suite: dict[str, Any] = {
        "suite_version": "compatibility_suite_v1",
        "cases": [
            {"id": "messages_basic", "category": "messages", "request": {"prompt": "x"}, "expected_substring": "ok"},
            {"id": "usage_integrity", "category": "usage", "request": {"prompt": "x"}},
            {"id": "sse_health", "category": "sse", "request": {"prompt": "x"}},
            {"id": "json_schema", "category": "json", "request": {"prompt": "x"}, "expected_json": {"status": "ok", "count": 3}},
            {"id": "cache_control_probe", "category": "cache", "request": {"messages": [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}]}},
            {"id": "tool_call_probe", "category": "tool_call", "request": {"prompt": "x", "tools": [{"name": "t", "input_schema": {"type": "object"}}]}},
        ],
    }
    good = ProbeMetrics(ok=True, http_status=200, content_chars=20, content_event_count=2, event_count=5, first_event_ms=10, first_content_token_ms=20, total_ms=50, input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0, server_model="fake-model", stop_reason="end_turn", event_types=["message_start", "content_block_delta", "message_delta", "message_stop"])
    long = ProbeMetrics(ok=True, http_status=200, content_chars=20, content_event_count=2, event_count=5, first_event_ms=10, first_content_token_ms=20, total_ms=50, input_tokens=30, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0, server_model="fake-model", stop_reason="end_turn", event_types=["message_start", "content_block_delta", "message_delta", "message_stop"])
    tool = ProbeMetrics(ok=True, http_status=200, content_chars=0, tool_use_event_count=1, input_json_delta_count=1, event_count=4, server_model="fake-model", event_types=["message_start", "content_block_start", "content_block_delta", "message_delta", "message_stop"])
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][0], provider=provider, primary_metrics=good, primary_response_text="ok")) == PASS
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][1], provider=provider, primary_metrics=good, primary_response_text="ok", secondary_metrics=long)) == PASS
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][2], provider=provider, primary_metrics=good, primary_response_text="ok")) == PASS
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][3], provider=provider, primary_metrics=good, primary_response_text='{"status":"ok","count":3}')) == PASS
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][4], provider=provider, primary_metrics=good, primary_response_text="ok", secondary_metrics=long)) == PASS
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][5], provider=provider, primary_metrics=tool, primary_response_text="")) == PASS
    http_fail = ProbeMetrics(ok=False, http_status=400, error="HTTP 400: bad", server_model=None)
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][0], provider=provider, primary_metrics=http_fail, primary_response_text="")) == FAIL
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][5], provider=provider, primary_metrics=http_fail, primary_response_text="")) == FAIL
    tool_choice_unsupported = ProbeMetrics(ok=False, http_status=400, error='HTTP 400: {"error":{"message":"Thinking mode does not support this tool_choice","type":"invalid_request_error"}}', server_model=None)
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][5], provider=provider, primary_metrics=tool_choice_unsupported, primary_response_text="")) == WARN
    thinking_only = ProbeMetrics(ok=False, http_status=200, error="no assistant text produced", thinking_event_count=10, server_model="fake-model", event_types=["message_start", "content_block_delta", "message_delta", "message_stop"])
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][2], provider=provider, primary_metrics=thinking_only, primary_response_text="")) == FAIL
    usage_missing = ProbeMetrics(ok=True, http_status=200, content_chars=5, server_model="fake-model")
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][1], provider=provider, primary_metrics=usage_missing, primary_response_text="ok")) == FAIL
    mismatch = ProbeMetrics(ok=True, http_status=200, content_chars=5, server_model="other-model")
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][0], provider=provider, primary_metrics=mismatch, primary_response_text="ok")) == FAIL
    assert worst_status(item["status"] for item in evaluate_case(suite=suite, case=suite["cases"][3], provider=provider, primary_metrics=good, primary_response_text="```json\n{}\n```")) == FAIL
    print("compatibility self-test ok")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="Run provider compatibility suite")
    parser.add_argument("--providers", type=Path, default=Path("providers.local.json"))
    parser.add_argument("--provider-id", default=None)
    parser.add_argument("--suite", type=Path, default=Path("compatibility_suite.json"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    providers = load_providers(args.providers)
    if args.provider_id:
        providers = [provider for provider in providers if provider.id == args.provider_id]
        if not providers:
            raise SystemExit(f"unknown provider id: {args.provider_id}")
    if len(providers) != 1:
        raise SystemExit("compatibility CLI expects exactly one provider; use --provider-id")
    result = run_compatibility_suite(
        runs_dir=args.out,
        provider=providers[0],
        suite_path=args.suite,
        timeout=args.timeout,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
