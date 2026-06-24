from __future__ import annotations

import argparse
import csv
import json
import tempfile
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import evidence_registry as registry


TRACE_EVAL_RECORD_VERSION = "trace_eval_record_v1"
TRACE_EVAL_POLICY_VERSION = "trace_eval_policy_v1"
DEFAULT_POLICY_ID = "trace_health_v1"

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
NOT_APPLICABLE = "NOT_APPLICABLE"

ProgressFn = Callable[[dict[str, Any]], None]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def numeric(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "ok"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def load_policy(policy_path: Path, policy_id: str | None = None) -> dict[str, Any]:
    wanted = policy_id or DEFAULT_POLICY_ID
    if not policy_path.exists():
        return {
            "policy_id": wanted,
            "policy_version": TRACE_EVAL_POLICY_VERSION,
            "thresholds": {
                "first_content_token_ms_warn": 15000,
                "thinking_delta_count_warn": 100,
            },
        }
    data = read_json(policy_path)
    if data.get("policy_id") == wanted:
        policy = dict(data)
    else:
        policies = data.get("policies") or []
        policy = next((dict(item) for item in policies if item.get("policy_id") == wanted), None)
        if policy is None:
            raise ValueError(f"trace policy not found: {wanted}")
    policy.setdefault("policy_id", wanted)
    policy.setdefault("policy_version", data.get("policy_version") or TRACE_EVAL_POLICY_VERSION)
    policy.setdefault("thresholds", {})
    policy["thresholds"].setdefault("first_content_token_ms_warn", 15000)
    policy["thresholds"].setdefault("thinking_delta_count_warn", 100)
    return policy


def check(name: str, status: str, details: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "details": details,
        "evidence": evidence or {},
    }


def worst_status(statuses: list[str]) -> str:
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return PASS


def resolve_artifact_path(path_value: Any, *, root_dir: Path, run_dir: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    raw = Path(str(path_value))
    candidates = [raw] if raw.is_absolute() else [run_dir / raw, root_dir / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def source_from_run_record(record: dict[str, Any], run_id: str, run_dir: Path, root_dir: Path) -> dict[str, Any]:
    run = record.get("run") if isinstance(record.get("run"), dict) else {}
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
    telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    events_file = artifacts.get("events_file") or response.get("events_file")
    response_file = artifacts.get("response_file") or response.get("response_file")
    resolved_events = resolve_artifact_path(events_file, root_dir=root_dir, run_dir=run_dir)
    resolved_response = resolve_artifact_path(response_file, root_dir=root_dir, run_dir=run_dir)
    return {
        "source_run_id": run.get("run_id") or run_id,
        "source_record_id": record.get("record_id") or f"{run_id}:{provider.get('id')}:{task.get('id')}",
        "provider_id": provider.get("id"),
        "task_id": task.get("id"),
        "run_status": run.get("status"),
        "task": task,
        "provider": provider,
        "telemetry": telemetry,
        "trace": trace,
        "source_response_file": str(resolved_response) if resolved_response else str(response_file or ""),
        "source_events_file": str(resolved_events) if resolved_events else str(events_file or ""),
        "events_path": resolved_events,
        "raw_event_types": trace.get("raw_event_types") if isinstance(trace.get("raw_event_types"), list) else [],
    }


def load_trace_sources(
    *,
    runs_dir: Path,
    run_id: str,
    provider_id: str | None = None,
    task_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    run_dir = runs_dir / run_id
    records_path = run_dir / "run_records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"run_records.jsonl not found for run: {run_id}")
    wanted_tasks = set(task_ids or [])
    root_dir = runs_dir.parent
    sources = [
        source_from_run_record(record, run_id, run_dir, root_dir)
        for record in read_jsonl(records_path)
    ]
    out: list[dict[str, Any]] = []
    for source in sources:
        if provider_id and source.get("provider_id") != provider_id:
            continue
        if wanted_tasks and source.get("task_id") not in wanted_tasks:
            continue
        out.append(source)
    return out


def trace_source_count(
    runs_dir: Path,
    run_id: str,
    provider_id: str | None = None,
    task_ids: list[str] | None = None,
) -> int:
    return len(load_trace_sources(runs_dir=runs_dir, run_id=run_id, provider_id=provider_id, task_ids=task_ids))


def parse_events(events_path: Path) -> dict[str, Any]:
    event_types: list[str] = []
    positions: dict[str, int] = {}
    block_types: list[str] = []
    delta_types: list[str] = []
    text_delta_count = 0
    thinking_delta_count = 0
    tool_event_count = 0
    invalid_json_count = 0
    stop_reason = None
    usage = None
    with events_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_count += 1
                event_types.append("non_json_event")
                positions.setdefault("non_json_event", index)
                continue
            if not isinstance(event, dict):
                event_types.append("non_object_event")
                positions.setdefault("non_object_event", index)
                continue
            event_type = str(event.get("type") or event.get("event") or "unknown")
            event_types.append(event_type)
            positions.setdefault(event_type, index)
            if event_type == "content_block_start":
                block = event.get("content_block") if isinstance(event.get("content_block"), dict) else {}
                block_type = str(block.get("type") or "unknown")
                block_types.append(block_type)
                if block_type == "tool_use":
                    tool_event_count += 1
            elif event_type == "content_block_delta":
                delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
                delta_type = str(delta.get("type") or "unknown")
                delta_types.append(delta_type)
                if delta_type == "text_delta":
                    text_delta_count += 1
                elif delta_type == "thinking_delta":
                    thinking_delta_count += 1
                elif "tool" in delta_type or "input_json" in delta_type:
                    tool_event_count += 1
            elif event_type == "message_delta":
                delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
                if delta.get("stop_reason") not in (None, ""):
                    stop_reason = delta.get("stop_reason")
                if isinstance(event.get("usage"), dict):
                    usage = event.get("usage")
    return {
        "event_types": event_types,
        "unique_event_types": list(dict.fromkeys(event_types)),
        "positions": positions,
        "block_types": block_types,
        "delta_types": delta_types,
        "text_delta_count": text_delta_count,
        "thinking_delta_count": thinking_delta_count,
        "tool_event_count": tool_event_count,
        "invalid_json_count": invalid_json_count,
        "event_count": len(event_types),
        "message_delta_stop_reason": stop_reason,
        "message_delta_usage": usage,
    }


def validate_tool_calls(tool_calls: Any) -> tuple[str, str, dict[str, Any]]:
    if not tool_calls:
        return NOT_APPLICABLE, "no tool calls recorded", {"tool_call_count": 0}
    if not isinstance(tool_calls, list):
        return FAIL, "trace.tool_calls must be an array", {"actual_type": type(tool_calls).__name__}
    invalid: list[dict[str, Any]] = []
    for index, item in enumerate(tool_calls):
        if not isinstance(item, dict):
            invalid.append({"index": index, "error": "tool call must be object"})
            continue
        name = item.get("name") or item.get("tool_name")
        arguments = item.get("arguments") if "arguments" in item else item.get("tool_input")
        if not name:
            invalid.append({"index": index, "error": "missing name/tool_name"})
        if arguments is not None and not isinstance(arguments, (dict, list, str)):
            invalid.append({"index": index, "error": "arguments/tool_input has unsupported type"})
    if invalid:
        return FAIL, "trace.tool_calls contains invalid entries", {"invalid": invalid, "tool_call_count": len(tool_calls)}
    return PASS, "trace.tool_calls structure is valid", {"tool_call_count": len(tool_calls)}


def evaluate_source(source: dict[str, Any], policy: dict[str, Any], trace_eval_id: str) -> dict[str, Any]:
    thresholds = policy.get("thresholds") or {}
    first_content_warn = numeric(thresholds.get("first_content_token_ms_warn"), 15000) or 15000
    thinking_warn_count = int(numeric(thresholds.get("thinking_delta_count_warn"), 100) or 100)
    telemetry = source.get("telemetry") if isinstance(source.get("telemetry"), dict) else {}
    trace = source.get("trace") if isinstance(source.get("trace"), dict) else {}
    checks: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "raw_event_types": source.get("raw_event_types") or [],
        "telemetry": {
            "ok": telemetry.get("ok"),
            "error": telemetry.get("error"),
            "first_content_token_ms": telemetry.get("first_content_token_ms"),
            "stop_reason": telemetry.get("stop_reason"),
        },
    }

    events_path = source.get("events_path")
    parsed_events = None
    if not events_path or not Path(events_path).exists():
        checks.append(check("events_file_present", WARN, "events file is missing", {"events_file": source.get("source_events_file")}))
        evidence["missing_events"] = True
    else:
        checks.append(check("events_file_present", PASS, "events file is present", {"events_file": str(events_path)}))
        parsed_events = parse_events(Path(events_path))
        evidence.update(parsed_events)
        required = ["message_start", "message_delta", "message_stop"]
        missing = [event for event in required if event not in parsed_events["positions"]]
        if missing:
            checks.append(check("required_sse_events", FAIL, "required SSE events are missing", {"missing": missing, "event_types": parsed_events["unique_event_types"]}))
        else:
            checks.append(check("required_sse_events", PASS, "required SSE events are present", {"event_types": parsed_events["unique_event_types"]}))
        positions = parsed_events["positions"]
        if (
            "message_start" in positions
            and "message_delta" in positions
            and "message_stop" in positions
            and positions["message_start"] < positions["message_delta"] < positions["message_stop"]
        ):
            checks.append(check("event_order", PASS, "SSE event order is plausible", positions))
        else:
            checks.append(check("event_order", FAIL, "SSE event order is invalid or incomplete", positions))
        if parsed_events["text_delta_count"] <= 0:
            if parsed_events["thinking_delta_count"] > 0:
                checks.append(check("visible_text_path", FAIL, "trace has thinking deltas but no visible text deltas", {"thinking_delta_count": parsed_events["thinking_delta_count"]}))
                evidence["thinking_only"] = True
            else:
                checks.append(check("visible_text_path", FAIL, "trace has no visible text deltas", {"text_delta_count": 0}))
        elif parsed_events["thinking_delta_count"] > thinking_warn_count:
            checks.append(check("visible_text_path", WARN, "thinking path is long before/alongside visible text", {"thinking_delta_count": parsed_events["thinking_delta_count"], "threshold": thinking_warn_count}))
        else:
            checks.append(check("visible_text_path", PASS, "trace contains visible text deltas", {"text_delta_count": parsed_events["text_delta_count"]}))
        if "message_stop" not in positions:
            checks.append(check("terminal_state", FAIL, "message_stop is missing"))
        elif telemetry.get("error") or boolish(telemetry.get("ok")) is False:
            checks.append(check("terminal_state", FAIL, "run telemetry indicates an error", {"ok": telemetry.get("ok"), "error": telemetry.get("error")}))
        else:
            checks.append(check("terminal_state", PASS, "trace terminal state is complete"))
        if parsed_events["invalid_json_count"]:
            checks.append(check("event_json_validity", FAIL, "events file contains invalid JSON rows", {"invalid_json_count": parsed_events["invalid_json_count"]}))

    stop_reason = telemetry.get("stop_reason") or (parsed_events or {}).get("message_delta_stop_reason")
    if stop_reason == "max_tokens":
        checks.append(check("max_tokens_stop", WARN, "trace stopped because max_tokens was reached", {"stop_reason": stop_reason}))
    else:
        checks.append(check("max_tokens_stop", PASS, "trace did not stop at max_tokens", {"stop_reason": stop_reason}))
    first_content_ms = numeric(telemetry.get("first_content_token_ms"))
    if first_content_ms is None:
        checks.append(check("latency_trace", WARN, "first visible token latency is missing", {"first_content_token_ms": None}))
    elif first_content_ms > first_content_warn:
        checks.append(check("latency_trace", WARN, "first visible token latency exceeds threshold", {"first_content_token_ms": first_content_ms, "threshold": first_content_warn}))
    else:
        checks.append(check("latency_trace", PASS, "first visible token latency is within threshold", {"first_content_token_ms": first_content_ms, "threshold": first_content_warn}))
    tool_status, tool_details, tool_evidence = validate_tool_calls(trace.get("tool_calls"))
    checks.append(check("tool_trace_container", tool_status, tool_details, tool_evidence))
    status = worst_status([item["status"] for item in checks])
    return {
        "schema_version": TRACE_EVAL_RECORD_VERSION,
        "trace_eval_id": trace_eval_id,
        "source_run_id": source.get("source_run_id"),
        "source_record_id": source.get("source_record_id"),
        "provider_id": source.get("provider_id"),
        "task_id": source.get("task_id"),
        "status": status,
        "checks": checks,
        "evidence": evidence,
        "source_response_file": source.get("source_response_file"),
        "source_events_file": source.get("source_events_file"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }


def provider_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("provider_id") or "unknown")].append(record)
    out: dict[str, Any] = {}
    for provider_id, rows in grouped.items():
        total = len(rows)
        fail_count = sum(1 for row in rows if row.get("status") == FAIL)
        warn_count = sum(1 for row in rows if row.get("status") == WARN)
        pass_count = sum(1 for row in rows if row.get("status") == PASS)
        missing_events_count = sum(1 for row in rows if (row.get("evidence") or {}).get("missing_events"))
        thinking_only_count = sum(1 for row in rows if (row.get("evidence") or {}).get("thinking_only"))
        max_tokens_count = sum(
            1
            for row in rows
            for item in row.get("checks") or []
            if item.get("name") == "max_tokens_stop" and item.get("status") == WARN
        )
        out[provider_id] = {
            "provider_id": provider_id,
            "record_count": total,
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
            "trace_fail_rate": ratio(fail_count, total),
            "trace_warn_rate": ratio(warn_count, total),
            "thinking_only_count": thinking_only_count,
            "missing_events_count": missing_events_count,
            "max_tokens_count": max_tokens_count,
            "status": FAIL if fail_count else WARN if warn_count else PASS,
        }
    return out


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "trace_eval_id",
        "source_run_id",
        "source_record_id",
        "provider_id",
        "task_id",
        "status",
        "failed_checks",
        "warned_checks",
        "source_response_file",
        "source_events_file",
        "evaluated_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            checks = record.get("checks") or []
            row = {
                "trace_eval_id": record.get("trace_eval_id"),
                "source_run_id": record.get("source_run_id"),
                "source_record_id": record.get("source_record_id"),
                "provider_id": record.get("provider_id"),
                "task_id": record.get("task_id"),
                "status": record.get("status"),
                "failed_checks": ";".join(item.get("name") for item in checks if item.get("status") == FAIL),
                "warned_checks": ";".join(item.get("name") for item in checks if item.get("status") == WARN),
                "source_response_file": record.get("source_response_file"),
                "source_events_file": record.get("source_events_file"),
                "evaluated_at": record.get("evaluated_at"),
            }
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def wait_for_resume(job_control: dict[str, Any] | None, progress_callback: ProgressFn | None, completed: int, total: int) -> bool:
    if not job_control or not job_control.get("pause_requested"):
        return False
    if progress_callback:
        progress_callback({"event": "run_paused", "completed_tasks": completed, "total_tasks": total})
    resume_event = job_control.get("resume_event")
    while job_control.get("pause_requested") and not job_control.get("stop_requested"):
        if isinstance(resume_event, threading.Event):
            resume_event.wait(0.25)
        else:
            break
    if job_control.get("stop_requested"):
        return True
    if progress_callback:
        progress_callback({"event": "run_resumed", "completed_tasks": completed, "total_tasks": total})
    return False


def run_trace_evaluation(
    *,
    runs_dir: Path,
    run_id: str,
    policy_path: Path,
    provider_id: str | None = None,
    task_ids: list[str] | None = None,
    policy_id: str | None = None,
    trace_eval_label: str | None = None,
    progress_callback: ProgressFn | None = None,
    job_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_id}")
    created_at = datetime.now().isoformat(timespec="seconds")
    trace_eval_id = registry.unique_artifact_id("trace_eval")
    policy = load_policy(policy_path, policy_id)
    sources = load_trace_sources(runs_dir=runs_dir, run_id=run_id, provider_id=provider_id, task_ids=task_ids)
    eval_dir = run_dir / "trace_evaluations" / trace_eval_id
    eval_dir.mkdir(parents=True, exist_ok=False)
    records_path = eval_dir / "trace_eval_records.jsonl"
    records_path.write_text("", encoding="utf-8")
    total = len(sources)
    if progress_callback:
        progress_callback({"event": "run_started", "total_tasks": total, "benchmark_mode": "trace_evaluation", "current_provider": provider_id})
    records: list[dict[str, Any]] = []
    stopped = False
    for index, source in enumerate(sources):
        if job_control and job_control.get("stop_requested"):
            stopped = True
            break
        if wait_for_resume(job_control, progress_callback, index, total):
            stopped = True
            break
        if progress_callback:
            progress_callback(
                {
                    "event": "task_started",
                    "current_task_id": source.get("task_id"),
                    "current_provider": source.get("provider_id"),
                    "phase": "trace_evaluating",
                    "completed_tasks": index,
                }
            )
        try:
            record = evaluate_source(source, policy, trace_eval_id)
        except Exception as exc:
            record = {
                "schema_version": TRACE_EVAL_RECORD_VERSION,
                "trace_eval_id": trace_eval_id,
                "source_run_id": source.get("source_run_id"),
                "source_record_id": source.get("source_record_id"),
                "provider_id": source.get("provider_id"),
                "task_id": source.get("task_id"),
                "status": FAIL,
                "checks": [
                    check(
                        "trace_evaluation_error",
                        FAIL,
                        f"{type(exc).__name__}: {exc}",
                    )
                ],
                "evidence": {},
                "source_response_file": source.get("source_response_file"),
                "source_events_file": source.get("source_events_file"),
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            }
        records.append(record)
        append_jsonl(records_path, record)
        if progress_callback:
            progress_callback(
                {
                    "event": "task_completed",
                    "current_task_id": source.get("task_id"),
                    "current_provider": source.get("provider_id"),
                    "phase": "trace_evaluating",
                    "completed_tasks": index + 1,
                    "ok": record.get("status") != FAIL,
                    "error": None if record.get("status") != FAIL else "trace evaluation failed",
                }
            )
    if progress_callback:
        progress_callback({"event": "task_phase", "phase": "trace_writing", "completed_tasks": len(records), "total_tasks": total})
    summary_path = eval_dir / "trace_eval_summary.csv"
    manifest_path = eval_dir / "trace_eval_manifest.json"
    write_summary(summary_path, records)
    metrics = provider_metrics(records)
    fail_count = sum(1 for record in records if record.get("status") == FAIL)
    warn_count = sum(1 for record in records if record.get("status") == WARN)
    manifest = {
        "schema_version": "trace_eval_manifest_v1",
        "trace_eval_id": trace_eval_id,
        "source_run_id": run_id,
        "status": "stopped" if stopped else "completed",
        "created_at": created_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "trace_eval_label": trace_eval_label,
        "policy_id": policy.get("policy_id"),
        "policy_version": policy.get("policy_version"),
        "filters": {"provider_id": provider_id, "task_ids": task_ids or []},
        "record_count": len(records),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "pass_count": sum(1 for record in records if record.get("status") == PASS),
        "provider_metrics": metrics,
        "records_file": str(records_path),
        "summary_file": str(summary_path),
        "manifest_file": str(manifest_path),
        "stopped": stopped,
    }
    write_json(manifest_path, manifest)
    if stopped and progress_callback:
        progress_callback({"event": "run_stopped", "run_id": run_id, "trace_eval_id": trace_eval_id, "completed_tasks": len(records), "total_tasks": total, "stop_reason": "user_stop_requested"})
    return {
        "trace_eval_id": trace_eval_id,
        "source_run_id": run_id,
        "record_count": len(records),
        "records": records,
        "summary": read_csv_rows(summary_path),
        "manifest": manifest,
        "stopped": stopped,
    }


def list_trace_evaluations(run_dir: Path) -> list[dict[str, Any]]:
    evals_dir = run_dir / "trace_evaluations"
    if not evals_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted((p for p in evals_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        manifest_path = child / "trace_eval_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if isinstance(manifest, dict):
            out.append(manifest)
    return out


def read_trace_evaluation(run_dir: Path, trace_eval_id: str) -> dict[str, Any]:
    eval_dir = run_dir / "trace_evaluations" / trace_eval_id
    manifest_path = eval_dir / "trace_eval_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"trace evaluation not found: {trace_eval_id}")
    return {
        "manifest": read_json(manifest_path),
        "summary": read_csv_rows(eval_dir / "trace_eval_summary.csv"),
        "records": read_jsonl(eval_dir / "trace_eval_records.jsonl"),
    }


def write_fixture_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def fixture_record(run_id: str, task_id: str, provider_id: str, events_file: str, *, stop_reason: str = "end_turn", tool_calls: Any = None, first_ms: int | None = 100) -> dict[str, Any]:
    return {
        "schema_version": "run_record_v1",
        "record_id": f"{run_id}:{provider_id}:{task_id}",
        "run": {"run_id": run_id, "timestamp": "2026-06-20T00:00:00", "benchmark_mode": "fixture", "formula_version": "score_formula_v1", "runner": "cli", "status": "completed"},
        "task": {"id": task_id, "category": "fixture", "enterprise_dimension": "fixture", "difficulty": "easy", "scoring_type": "manual_rubric", "risk_tags": [], "point_value": 100, "scoring_confidence": 0.5},
        "provider": {"id": provider_id, "api_style": "anthropic_messages", "base_url_host": None, "model_requested": "model-a", "model_returned": "model-a"},
        "request": {"request_hash": task_id, "max_tokens": 128, "temperature": 0, "system_present": False, "messages_count": 1},
        "response": {"response_file": "", "events_file": events_file, "content_chars": 10, "normalized_text_hash": task_id},
        "telemetry": {"ok": True, "error": None, "first_event_ms": 10, "first_content_token_ms": first_ms, "total_ms": 200, "event_count": 4, "content_event_count": 1, "stop_reason": stop_reason},
        "usage": {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        "scoring": {"rule_score": None, "judge_score": None, "final_score": None, "judge_provider": None, "judge_model_requested": None, "judge_model_returned": None},
        "trace": {"tool_calls": tool_calls if tool_calls is not None else [], "raw_event_types": []},
        "artifacts": {"response_file": "", "events_file": events_file},
    }


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        runs_dir = root / "runs"
        run_dir = runs_dir / "trace_fixture"
        events_dir = run_dir / "events" / "provider_a"
        good = [
            {"type": "message_start"},
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
        no_stop = good[:-1]
        thinking_only = [
            {"type": "message_start"},
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
        write_fixture_events(events_dir / "good.jsonl", good)
        write_fixture_events(events_dir / "no_stop.jsonl", no_stop)
        write_fixture_events(events_dir / "thinking_only.jsonl", thinking_only)
        records = [
            fixture_record("trace_fixture", "good", "provider_a", "events/provider_a/good.jsonl"),
            fixture_record("trace_fixture", "missing_events", "provider_a", ""),
            fixture_record("trace_fixture", "no_stop", "provider_a", "events/provider_a/no_stop.jsonl"),
            fixture_record("trace_fixture", "thinking_only", "provider_a", "events/provider_a/thinking_only.jsonl"),
            fixture_record("trace_fixture", "max_tokens", "provider_a", "events/provider_a/good.jsonl", stop_reason="max_tokens"),
            fixture_record("trace_fixture", "bad_tool", "provider_a", "events/provider_a/good.jsonl", tool_calls=[{"arguments": {}}]),
        ]
        write_jsonl(run_dir / "run_records.jsonl", records)
        policy_path = root / "trace_evaluation.policy.json"
        write_json(policy_path, {"policy_version": TRACE_EVAL_POLICY_VERSION, "policies": [{"policy_id": DEFAULT_POLICY_ID, "thresholds": {}}]})
        result = run_trace_evaluation(runs_dir=runs_dir, run_id="trace_fixture", policy_path=policy_path)
        by_task = {record["task_id"]: record for record in result["records"]}
        assert by_task["good"]["status"] == PASS
        assert by_task["missing_events"]["status"] == WARN
        assert by_task["no_stop"]["status"] == FAIL
        assert by_task["thinking_only"]["status"] == FAIL
        assert by_task["max_tokens"]["status"] == WARN
        assert by_task["bad_tool"]["status"] == FAIL
        provider = result["manifest"]["provider_metrics"]["provider_a"]
        assert provider["record_count"] == 6
        assert provider["missing_events_count"] == 1
        assert provider["thinking_only_count"] == 1
        assert provider["max_tokens_count"] == 1


def main() -> int:
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="Run offline trace evaluation over run_records/events")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--policy", type=Path, default=Path("trace_evaluation.policy.json"))
    parser.add_argument("--policy-id", default=DEFAULT_POLICY_ID)
    parser.add_argument("--provider-id", default=None)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--label", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        print("trace evaluation self-test ok")
        return 0
    if not args.run_id:
        parser.error("--run-id is required unless --self-test is used")
    result = run_trace_evaluation(
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        policy_path=args.policy,
        provider_id=args.provider_id,
        task_ids=args.task_id or None,
        policy_id=args.policy_id,
        trace_eval_label=args.label,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
