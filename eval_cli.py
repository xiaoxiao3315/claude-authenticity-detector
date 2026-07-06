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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

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
    outcomes_from_summary,
    safe_campaign_id,
    summarize_campaign,
    write_json as write_campaign_json,
)
from authenticity import build_config_protocol_fingerprint, load_or_build_authenticity, write_authenticity_evidence, numeric
from baseline_registry import (
    DEFAULT_BASELINES_DIR,
    aggregate_capability,
    build_baseline_from_samples,
    classify_error_envelope,
    classify_sse_event_order,
    compare_to_baseline,
    derive_token_windows,
    diff_baselines,
    evaluate_silent_truncation,
    key_fingerprint,
    load_baseline,
    load_baseline_version,
    list_baseline_versions,
    make_sample,
    render_verdict_report,
    score_capability_item,
    score_capability_vs_baseline,
    score_consistency_variance,
    score_identity_coherence,
    score_needle_recall,
    write_baseline,
    write_baseline_version,
)
from local_env import load_local_env
from judge_calibration import (
    classify_judge,
    compute_calibration,
    load_golden_set,
    normalize_decision,
    render_calibration_report,
)
from quality_gate import run_quality_gate
from run_eval import iter_sse_events
from run_records import extract_raw_event_types, stable_json_hash, text_hash
from trace_evaluation import run_trace_evaluation
from validate_run_records import validate_records
from redaction import redact_raw_fragments, redact_text
from model_client import (
    CallMetrics,
    Completion,
    ModelConfig,
    apply_extra_body,
    auth_headers,
    auth_value,
    call_model,
    call_model_with_retries,
    dry_completion,
    response_request_id,
    retryable_call_failure,
    safe_response_headers,
)
from cli_io import (
    append_jsonl,
    base_url_host,
    now_iso,
    read_json,
    read_jsonl,
    resolve_job,
    resolve_path,
    utcish_job_id,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_JOB = "smoke_10"
DEFAULT_PROVIDERS = Path("configs/providers.local.json")
DEFAULT_CAMPAIGNS_DIR = Path("campaigns")
ALLOWED_PROTOCOLS = {"openai_chat", "anthropic_messages"}
ALLOWED_AUTH_TYPES = {"bearer", "x-api-key"}


# --- model dataclasses + HTTP call layer extracted to model_client (imported above) ---


# --- low-level IO + path helpers extracted to cli_io (imported above) ---


SENSITIVE_CONFIG_KEY_TOKENS = ("authorization", "credential", "key", "password", "secret", "token")


def _as_dict(value: Any) -> dict[str, Any]:
    """Narrow Any -> dict (empty when not a dict) for the type checker."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Narrow Any -> list (empty when not a list) for the type checker."""
    return value if isinstance(value, list) else []


def sanitize_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in SENSITIVE_CONFIG_KEY_TOKENS):
                out[str(key)] = "[REDACTED]"
            else:
                out[str(key)] = sanitize_config_value(item)
        return out
    if isinstance(value, list):
        return [sanitize_config_value(item) for item in value]
    return value


def load_extra_body(raw: dict[str, Any], label: str) -> dict[str, Any]:
    value = raw.get("extra_body") or {}
    if not isinstance(value, dict):
        raise ValueError(f"{label}.extra_body must be a JSON object when provided")
    try:
        json.dumps(value, ensure_ascii=False)
    except TypeError as exc:
        raise ValueError(f"{label}.extra_body must be JSON serializable") from exc
    return dict(value)


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
        extra_body=load_extra_body(raw, label),
    )


def load_two_model_config(path: Path) -> dict[str, ModelConfig]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    models = {
        "tested_model": load_model_config(data["tested_model"], "tested_model"),
        "judge_model": load_model_config(data["judge_model"], "judge_model"),
    }
    # also load any extra provider roles (e.g. suspect_model, official_baseline)
    # so authenticity verification can target arbitrary endpoints.
    for label, raw in data.items():
        if label in models or not isinstance(raw, dict):
            continue
        if "base_url" in raw and "model" in raw and "api_key_env" in raw:
            models[label] = load_model_config(raw, label)
    return models


def _optional_arg(args: argparse.Namespace, name: str) -> str | None:
    value = getattr(args, name, None)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


MODEL_OVERRIDE_FIELDS = (
    "provider_id",
    "base_url",
    "model",
    "api_key_env",
    "protocol",
    "auth_type",
    "display_name",
    "reasoning_effort",
)


def model_override_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        f"{prefix}_{field}": getattr(args, f"{prefix}_{field}", None)
        for prefix in ("tested", "judge")
        for field in MODEL_OVERRIDE_FIELDS
    }


def apply_model_overrides(models: dict[str, ModelConfig], args: argparse.Namespace) -> dict[str, ModelConfig]:
    out = dict(models)
    for prefix, label in (("tested", "tested_model"), ("judge", "judge_model")):
        current = out[label]
        provider_id = _optional_arg(args, f"{prefix}_provider_id")
        base_url = _optional_arg(args, f"{prefix}_base_url")
        model_name = _optional_arg(args, f"{prefix}_model")
        api_key_env = _optional_arg(args, f"{prefix}_api_key_env")
        protocol = _optional_arg(args, f"{prefix}_protocol")
        auth_type = _optional_arg(args, f"{prefix}_auth_type")
        display_name = _optional_arg(args, f"{prefix}_display_name")
        reasoning_effort = _optional_arg(args, f"{prefix}_reasoning_effort")

        if protocol and protocol not in ALLOWED_PROTOCOLS:
            raise ValueError(f"--{prefix}-protocol must be one of: {', '.join(sorted(ALLOWED_PROTOCOLS))}")
        if auth_type and auth_type not in ALLOWED_AUTH_TYPES:
            raise ValueError(f"--{prefix}-auth-type must be one of: {', '.join(sorted(ALLOWED_AUTH_TYPES))}")
        if reasoning_effort and reasoning_effort not in {"default", "none", "low", "medium", "high", "xhigh"}:
            raise ValueError(f"--{prefix}-reasoning-effort must be one of: default, none, low, medium, high, xhigh")

        if any([provider_id, base_url, model_name, api_key_env, protocol, auth_type, display_name, reasoning_effort]):
            extra_body = dict(current.extra_body)
            if reasoning_effort == "default":
                extra_body.pop("reasoning_effort", None)
            elif reasoning_effort:
                extra_body["reasoning_effort"] = reasoning_effort
            out[label] = replace(
                current,
                provider_id=provider_id or current.provider_id,
                base_url=(base_url.rstrip("/") if base_url else current.base_url),
                model=model_name or current.model,
                api_key_env=api_key_env or current.api_key_env,
                protocol=protocol or current.protocol,
                auth_type=auth_type or current.auth_type,
                provider_display_name=display_name or provider_id or current.provider_display_name,
                extra_body=extra_body,
            )
    return out


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
        if model.extra_body:
            out[label]["extra_body"] = sanitize_config_value(model.extra_body)
    return out


def key_fingerprint_from_env(env_name: str) -> str | None:
    value = os.environ.get(env_name)
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def campaign_model_identity(model: ModelConfig, *, include_key_fingerprint: bool = False) -> dict[str, Any]:
    identity = {
        "provider_id": model.provider_id,
        "base_url_host": base_url_host(model.base_url),
        "model": model.model,
        "api_key_env": model.api_key_env,
        "key_fingerprint": key_fingerprint_from_env(model.api_key_env) if include_key_fingerprint else None,
        "protocol": model.protocol,
        "auth_type": model.auth_type,
        "provider_channel": model.provider_channel,
        "provider_display_name": model.provider_display_name or model.provider_id,
    }
    if model.extra_body:
        identity["extra_body"] = sanitize_config_value(model.extra_body)
    return identity


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


def benchmark_mode_settings(benchmark_config: dict[str, Any], benchmark_mode: str) -> dict[str, Any]:
    modes = _as_dict(benchmark_config.get("modes"))
    return _as_dict(modes.get(benchmark_mode))


def configured_max_concurrency(args: argparse.Namespace, job: dict[str, Any], benchmark_config: dict[str, Any], benchmark_mode: str) -> int:
    raw = getattr(args, "max_concurrency", None)
    if raw is None:
        raw = job.get("max_concurrency")
    if raw is None:
        raw = benchmark_mode_settings(benchmark_config, benchmark_mode).get("max_concurrency")
    try:
        value = int(raw or 1)
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


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
    raw_redaction_values: list[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not tested.metrics.ok:
        return {"score": 0.0, "format_ok": False, "details": redact_text(tested.metrics.error, max_chars=500)}, None
    if judge_payload is not None:
        score = normalize_score(judge_payload.get("score_0_10"))
        if score is not None:
            reason = redact_raw_fragments(judge_payload.get("reason") or "", raw_redaction_values, max_chars=500) or ""
            decision_value = str(judge_payload.get("decision") or "REVIEW").strip().upper()
            decision = decision_value if decision_value in {"GO", "REVIEW", "NO-GO"} else "REVIEW"
            format_ok = bool(judge_payload.get("format_ok", True))
            return (
                {
                    "score": score,
                    "format_ok": format_ok,
                    "details": reason,
                    "decision": decision,
                },
                {
                    "score_0_10": score,
                    "format_ok": format_ok,
                    "decision": decision,
                    "reason": reason,
                },
            )
    if rule_score is not None:
        return rule_score, {"error": redact_text(judge_error or "judge unavailable; used rule score", max_chars=500)}
    fallback = 0.0 if judge_error else 5.0
    redacted_error = redact_text(judge_error or "no judge score", max_chars=500)
    return {"score": fallback, "format_ok": None, "details": redacted_error}, {"error": redacted_error}


def _self_test_judge_payload_sanitization() -> None:
    secret_prompt = "SECRET_PROMPT_ALPHA_BETA_1234567890"
    secret_answer = "SECRET_RESPONSE_TEXT_GAMMA_DELTA_1234567890"
    final_score, judge_score = final_score_from_judge(
        tested=Completion(text=secret_answer, metrics=CallMetrics(ok=True)),
        judge=None,
        judge_payload={
            "score_0_10": 8,
            "format_ok": True,
            "decision": secret_prompt,
            "reason": f"Prompt was {secret_prompt}; answer was {secret_answer}",
            "extra_raw": secret_answer,
        },
        judge_error=None,
        rule_score=None,
        raw_redaction_values=[secret_prompt, secret_answer],
    )
    blob = json.dumps({"final_score": final_score, "judge_score": judge_score}, ensure_ascii=False)
    assert secret_prompt not in blob
    assert secret_answer not in blob
    assert "extra_raw" not in blob
    assert '"decision": "REVIEW"' in blob
    assert "[REDACTED_RAW]" in blob


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
        "extra_body": sanitize_config_value(tested_model.extra_body),
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
            "extra_body": sanitize_config_value(tested_model.extra_body),
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
            "error": redact_text(metrics.error, max_chars=500),
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
        "trace": {"tool_calls": [], "raw_event_types": extract_raw_event_types(events_file)},
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
        "error": redact_text(telemetry.get("error"), max_chars=500),
        "quality_0_10": final_score.get("score"),
        "format_ok": final_score.get("format_ok"),
        "judge_error": redact_text(judge_score.get("error"), max_chars=500) if isinstance(judge_score, dict) else None,
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
    models = apply_model_overrides(load_two_model_config(models_path), args)
    tasks, benchmark_config = select_tasks(job)
    if not tasks:
        raise ValueError("job selected zero tasks")

    live = bool(args.live)
    if not live and bool(job.get("live_provider")):
        live = True
    if live:
        load_local_env()
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
    max_concurrency = min(configured_max_concurrency(args, job, benchmark_config, benchmark_mode), len(tasks))

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
        "execution": {"max_concurrency": max_concurrency},
        "final_decision": None,
        "artifacts": {},
    }
    write_state(run_dir, state)
    write_json(run_dir / "job_config.snapshot.json", {**job, "job_path": str(job_path)})
    write_json(run_dir / "providers.redacted.json", sanitized_models(models))
    append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "job_started", "job_id": run_id, "live_provider": live})

    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 20.0), read=timeout, write=min(timeout, 30.0), pool=10.0)
    task_items = list(enumerate(tasks, start=1))

    def evaluate_task(index: int, task: dict[str, Any]) -> dict[str, Any]:
        task_id = str(task.get("id") or f"task_{index}")
        with httpx.Client(timeout=client_timeout, follow_redirects=True) as client:
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
                raw_redaction_values=[task.get("prompt"), tested.text],
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
            return {
                "index": index,
                "task_id": task_id,
                "record": record,
                "row": summary_row(record, response_file),
                "result": {
                    "task": task_metadata(task),
                    "tested_model": models["tested_model"].provider_id,
                    "judge_model": models["judge_model"].provider_id,
                    "tested_ok": tested.metrics.ok,
                    "score": final_score,
                    "judge_score": judge_score,
                    "response_file": str(response_file),
                },
                "score": final_score.get("score"),
                "ok": tested.metrics.ok,
            }

    outcomes: list[dict[str, Any]] = []
    completed_count = 0
    if max_concurrency == 1:
        for index, task in task_items:
            task_id = str(task.get("id") or f"task_{index}")
            state["current_task"] = task_id
            write_state(run_dir, state)
            append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "task_started", "task_id": task_id, "index": index})
            outcome = evaluate_task(index, task)
            outcomes.append(outcome)
            completed_count += 1
            state["progress"]["completed"] = completed_count
            append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "task_completed", "task_id": task_id, "score": outcome["score"], "ok": outcome["ok"]})
            write_state(run_dir, state)
    else:
        state["current_task"] = f"{max_concurrency} concurrent tasks"
        write_state(run_dir, state)
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = []
            for index, task in task_items:
                task_id = str(task.get("id") or f"task_{index}")
                append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "task_started", "task_id": task_id, "index": index})
                futures.append(executor.submit(evaluate_task, index, task))
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                completed_count += 1
                state["progress"]["completed"] = completed_count
                state["current_task"] = outcome["task_id"]
                append_jsonl(
                    run_dir / "events.jsonl",
                    {"at": now_iso(), "type": "task_completed", "task_id": outcome["task_id"], "score": outcome["score"], "ok": outcome["ok"]},
                )
                write_state(run_dir, state)

    outcomes.sort(key=lambda item: int(item["index"]))
    records = [outcome["record"] for outcome in outcomes]
    rows = [outcome["row"] for outcome in outcomes]
    results = [outcome["result"] for outcome in outcomes]
    write_jsonl(run_dir / "run_records.jsonl", records)

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
    trace_eval_id: str | None = None
    trace_evaluation_enabled = bool(job.get("trace_evaluation", True)) and not bool(getattr(args, "skip_trace_evaluation", False))
    if trace_evaluation_enabled:
        trace_policy_path = resolve_path(job.get("trace_evaluation_policy") or "trace_evaluation.policy.json")
        try:
            trace_result = run_trace_evaluation(
                runs_dir=runs_dir,
                run_id=run_id,
                policy_path=trace_policy_path,
                provider_id=models["tested_model"].provider_id,
                trace_eval_label="two_model_headless",
            )
            trace_eval_id = str(trace_result.get("trace_eval_id") or "")
            if trace_eval_id:
                state.setdefault("artifacts", {})["trace_eval_id"] = trace_eval_id
                write_state(run_dir, state)
        except Exception as exc:
            validation_errors.append(f"trace evaluation failed: {redact_text(exc, max_chars=500)}")
            append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "trace_evaluation_warning", "error": redact_text(exc, max_chars=500)})
    try:
        gate_result = run_quality_gate(
            runs_dir=runs_dir,
            run_id=run_id,
            policy_path=policy_path,
            provider_id=models["tested_model"].provider_id,
            trace_eval_id=trace_eval_id,
            gate_label="two_model_headless",
        )
    except Exception as exc:
        validation_errors.append(f"quality gate failed: {redact_text(exc, max_chars=500)}")
        append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "quality_gate_warning", "error": redact_text(exc, max_chars=500)})

    decision = "REVIEW"
    if gate_result and gate_result.get("records"):
        decision = str(gate_result["records"][0].get("decision") or "REVIEW")
    write_json(run_dir / "validation.json", {"error_count": len(validation_errors), "errors": validation_errors})
    state["status"] = "completed" if not validation_errors else "partial"
    state["completed_at"] = now_iso()
    state["current_task"] = None
    state["final_decision"] = decision
    state["validation_errors"] = len(validation_errors)
    write_state(run_dir, state)
    append_jsonl(run_dir / "events.jsonl", {"at": now_iso(), "type": "job_completed", "job_id": run_id, "decision": decision})
    print(json.dumps({"job_id": run_id, "status": state["status"], "decision": decision, "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
    if bool(getattr(args, "require_go", False)) and decision != "GO":
        return 2
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
    include_raw = bool(getattr(args, "include_raw", False))
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    zip_path = artifacts_dir / "acceptance_pack.zip"
    include_names = [
        "state.json",
        "run_records.jsonl",
        "results.json",
        "summary.csv",
        "benchmark_scores.json",
        "validation.json",
        "job_config.snapshot.json",
        "providers.redacted.json",
        "authenticity_summary.json",
    ]
    if include_raw:
        include_names.append("events.jsonl")
    checksums: dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include_names:
            path = run_dir / name
            if path.exists():
                data = path.read_bytes()
                zf.writestr(name, data)
                checksums[name] = hashlib.sha256(data).hexdigest()
        folders = ["quality_gates", "trace_evaluations", "baseline_comparisons", "protocol_fingerprints"]
        if include_raw:
            folders.extend(["events", "responses", "judge_responses"])
        for folder in folders:
            root = run_dir / folder
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    arcname = path.relative_to(run_dir).as_posix()
                    data = path.read_bytes()
                    zf.writestr(arcname, data)
                    checksums[arcname] = hashlib.sha256(data).hexdigest()
        manifest = {
            "schema_version": "acceptance_pack_manifest_v1",
            "pack_type": "run",
            "job_id": run_dir.name,
            "generated_at": now_iso(),
            "include_raw": include_raw,
            "entry_count_without_manifest": len(checksums),
            "raw_entry_policy": "included by explicit request" if include_raw else "excluded by default",
        }
        manifest_data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        zf.writestr("acceptance_manifest.json", manifest_data)
        checksums["acceptance_manifest.json"] = hashlib.sha256(manifest_data).hexdigest()
        checksum_lines = [f"{digest}  {name}" for name, digest in sorted(checksums.items())]
        zf.writestr("checksums.sha256", ("\n".join(checksum_lines) + "\n").encode("utf-8"))
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
    refs = _as_list(run_index.get("runs"))
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
    refs = _as_list(run_index.get("runs"))
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
    models = apply_model_overrides(load_two_model_config(models_path), args)
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
    if live:
        load_local_env()
    tested_identity = campaign_model_identity(models["tested_model"], include_key_fingerprint=live)
    judge_identity = campaign_model_identity(models["judge_model"], include_key_fingerprint=live)
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
        campaign_doc["resumed_code_git_commit"] = git_commit()
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
            max_concurrency=getattr(args, "max_concurrency", None),
            retries=getattr(args, "retries", None),
            retry_backoff=getattr(args, "retry_backoff", None),
            require_go=False,
            skip_trace_evaluation=getattr(args, "skip_trace_evaluation", False),
            **model_override_kwargs(args),
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
            run_ref["error"] = redact_text(exc, max_chars=500)
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
                    "overall_outcome": summary.get("outcomes", {}).get("overall_outcome"),
                    "next_action": summary.get("outcomes", {}).get("next_action"),
                    "skipped_completed_rounds": skipped_rounds,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if bool(getattr(args, "require_go", False)) and summary["decisions"]["overall_decision"] != "GO":
        return exit_code or 2
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
        "outcomes": outcomes_from_summary(summary),
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
    write_authenticity_evidence(out_dir, runs_dir, persist=True)
    zip_path = export_campaign(out_dir, runs_dir, include_raw=bool(getattr(args, "include_raw", False)))
    print(json.dumps({"campaign_id": args.campaign_id, "artifact": str(zip_path)}, ensure_ascii=False, indent=2))
    return 0


def campaign_retest(args: argparse.Namespace) -> int:
    campaigns_dir, runs_dir = campaign_paths(args)
    source_id = safe_campaign_id(args.campaign_id)
    source_dir = campaign_dir(campaigns_dir, source_id)
    if not source_dir.exists():
        raise FileNotFoundError(f"campaign not found: {source_id}")
    source_campaign = load_campaign(source_dir)
    source_summary = load_summary(source_dir) or summarize_campaign(source_dir, runs_dir)
    outcomes = outcomes_from_summary(source_summary)
    overall_outcome = str(outcomes.get("overall_outcome") or "PENDING")
    if overall_outcome != "RETEST" and not bool(getattr(args, "force", False)):
        print(
            json.dumps(
                {
                    "campaign_id": source_id,
                    "status": "skipped",
                    "overall_outcome": overall_outcome,
                    "reason": "campaign-retest only reruns RETEST campaigns unless --force is provided",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    new_campaign_id = safe_campaign_id(
        args.new_campaign_id or f"{source_id}-RETEST-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    retest_repeat = int(args.repeat or 1)
    if retest_repeat < 1:
        raise ValueError("--repeat must be >= 1")
    live = bool(args.live or (source_campaign.get("live_provider") is True and not bool(getattr(args, "dry_run", False))))
    child_args = argparse.Namespace(
        job=args.job or source_campaign.get("job") or DEFAULT_JOB,
        providers=args.providers,
        runs_dir=runs_dir,
        campaigns_dir=campaigns_dir,
        campaign_id=new_campaign_id,
        repeat=retest_repeat,
        resume=False,
        live=live,
        timeout=args.timeout,
        tested_max_tokens=args.tested_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        max_concurrency=getattr(args, "max_concurrency", None) or 1,
        retries=getattr(args, "retries", None),
        retry_backoff=getattr(args, "retry_backoff", None),
        require_go=False,
        skip_trace_evaluation=getattr(args, "skip_trace_evaluation", False),
        **model_override_kwargs(args),
    )
    print(
        json.dumps(
            {
                "source_campaign_id": source_id,
                "retest_campaign_id": new_campaign_id,
                "source_overall_outcome": overall_outcome,
                "source_next_action": outcomes.get("next_action"),
                "source_next_action_reason": outcomes.get("next_action_reason"),
                "live_provider": live,
                "repeat": retest_repeat,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return run_campaign(child_args)


def baseline_campaign_path(args: argparse.Namespace, campaigns_dir: Path) -> Path | None:
    baseline_id = str(getattr(args, "baseline_campaign_id", "") or "").strip()
    if not baseline_id:
        return None
    path = campaign_dir(campaigns_dir, baseline_id)
    if not path.exists():
        raise FileNotFoundError(f"baseline campaign not found: {baseline_id}")
    return path


def authenticity(args: argparse.Namespace) -> int:
    campaigns_dir, runs_dir = campaign_paths(args)
    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    models = apply_model_overrides(load_two_model_config(models_path), args)
    campaign_id = safe_campaign_id(args.campaign_id or f"CMP-AUTH-{campaign_id_for(models['tested_model'])[4:]}")
    out_dir = campaign_dir(campaigns_dir, campaign_id)
    exit_code = 0
    if not out_dir.exists():
        child_args = argparse.Namespace(
            job=args.job,
            providers=args.providers,
            runs_dir=runs_dir,
            campaigns_dir=campaigns_dir,
            campaign_id=campaign_id,
            repeat=int(args.repeat or 1),
            resume=False,
            live=bool(args.live),
            timeout=args.timeout,
            tested_max_tokens=args.tested_max_tokens,
            judge_max_tokens=args.judge_max_tokens,
            max_concurrency=getattr(args, "max_concurrency", None),
            retries=getattr(args, "retries", None),
            retry_backoff=getattr(args, "retry_backoff", None),
            require_go=False,
            skip_trace_evaluation=getattr(args, "skip_trace_evaluation", False),
            **model_override_kwargs(args),
        )
        exit_code = run_campaign(child_args)
    else:
        summarize_campaign(out_dir, runs_dir)

    baseline_dir = baseline_campaign_path(args, campaigns_dir)
    evidence = write_authenticity_evidence(
        out_dir,
        runs_dir,
        baseline_campaign_dir=baseline_dir,
        baseline_provider=str(args.baseline_provider or "official_baseline"),
        gateway_provider=str(args.gateway_provider or models["tested_model"].provider_id),
        persist=True,
    )
    print(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "status": evidence.get("status"),
                "authenticity_summary": str(out_dir / "authenticity_summary.json"),
                "decisions": evidence.get("decisions"),
                "metrics": evidence.get("metrics"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code


BASELINE_CANARY_PROBES = [
    {"id": "canary_mixed", "text": "Hello 世界 🌍 def f(x): return x*2 — café naïve Ω≈3.14"},
    {"id": "canary_zh", "text": "请用一句话总结：人工智能正在改变软件开发的方式。"},
    {"id": "canary_code", "text": "```python\nfor i in range(10):\n    print(i, i**2)\n```"},
]


def _raw_protocol_observation(completion: Completion, model: ModelConfig) -> dict[str, Any]:
    """Extract RAW protocol values from the upstream response body + headers.

    Reads completion.raw (set in call_model BEFORE the L639/640 stop/model
    fallback would rewrite them), so the baseline captures the source's true
    shape, not the normalized CallMetrics. Also inspects the response headers
    (completion.response_headers, allowlisted+lowercased in call_model) for the
    Anthropic header dialect — a genuine api.anthropic.com / faithful gateway
    emits anthropic-request-id (a `req_`-prefixed id) and anthropic-* headers;
    a thin OpenAI-style wrapper typically does not.
    """
    data = _as_dict(completion.raw)
    usage = _as_dict(data.get("usage"))
    raw_usage_keys = list(usage.keys())
    if model.protocol == "openai_chat":
        choices = data.get("choices") or []
        raw_stop = (choices[0].get("finish_reason") if choices else None)
    else:
        raw_stop = data.get("stop_reason")
    headers = completion.response_headers if isinstance(completion.response_headers, dict) else {}
    request_id = headers.get("anthropic-request-id") or ""
    has_anthropic_request_id = bool(request_id) and str(request_id).startswith("req_")
    has_anthropic_headers = any(
        key.startswith("anthropic-") for key in headers
    )
    # Envelope identity facts (for the identity-coherence probe): the upstream
    # `model` field and the response `id` are emitted by the serving infra, not
    # the narrated text, so they are forge-resistant. openai_chat puts the id at
    # the top level too; both dialects expose `model`.
    returned_model_field = data.get("model")
    response_id = data.get("id")
    return {
        "raw_stop_reason": raw_stop,
        "raw_usage_keys": raw_usage_keys,
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "has_anthropic_request_id": has_anthropic_request_id,
        "has_anthropic_headers": has_anthropic_headers,
        "returned_model_field": returned_model_field,
        "response_id": response_id,
    }


def _collect_baseline_samples(
    model: ModelConfig,
    *,
    samples_per_probe: int,
    live: bool,
    events_file: Path,
    request_delay: float = 0.0,
    retries: int = 1,
    retry_backoff: float = 0.5,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    timeout = httpx.Timeout(120.0)
    first = True
    with httpx.Client(timeout=timeout) as client:
        for probe in BASELINE_CANARY_PROBES:
            for _ in range(max(1, samples_per_probe)):
                if request_delay > 0 and not first:
                    time.sleep(request_delay)
                first = False
                messages = [{"role": "user", "content": probe["text"]}]
                completion = call_model_with_retries(
                    client=client,
                    model=model,
                    messages=messages,
                    max_tokens=64,
                    temperature=0,
                    live=live,
                    events_file=events_file,
                    retries=retries,
                    retry_backoff=retry_backoff,
                )
                obs = _raw_protocol_observation(completion, model)
                collected.append(make_sample(
                    protocol=model.protocol,
                    raw_stop_reason=obs["raw_stop_reason"],
                    raw_usage_keys=obs["raw_usage_keys"],
                    input_tokens=obs["input_tokens"],
                    output_tokens=obs["output_tokens"],
                    total_ms=completion.metrics.total_ms,
                    has_anthropic_request_id=obs["has_anthropic_request_id"],
                    has_anthropic_headers=obs["has_anthropic_headers"],
                    probe_id=probe["id"],
                    live=live,
                    ok=bool(completion.metrics.ok),
                ))
    return collected


def baseline_build(args: argparse.Namespace) -> int:
    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    role = str(args.provider or "tested_model")
    live = bool(getattr(args, "live", False))
    if live:
        load_local_env()
    models = apply_model_overrides(load_two_model_config(models_path), args)
    if role not in models:
        raise ValueError("--provider must be tested_model or judge_model")
    model = models[role]
    baselines_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR)
    baseline_id = str(getattr(args, "baseline_id", None) or utcish_job_id("BASE"))
    samples_per_probe = max(1, int(getattr(args, "samples", 2) or 2))

    out_dir = baselines_dir / baseline_id.replace("/", "_").replace("\\", "_")
    events_file = out_dir / "collection_events.jsonl"
    samples = _collect_baseline_samples(
        model,
        samples_per_probe=samples_per_probe,
        live=live,
        events_file=events_file,
    )
    secret = auth_value(model) if live else None
    source = {
        "provider_id": model.provider_id,
        "provider_label": role,
        "base_url_host": base_url_host(model.base_url),
        "model": model.model,
        "protocol": model.protocol,
        "key_fingerprint": key_fingerprint(secret) if secret else None,
    }
    doc = build_baseline_from_samples(
        samples, source, baseline_id=baseline_id, live=live,
        collected_window={"samples_per_probe": samples_per_probe, "probe_count": len(BASELINE_CANARY_PROBES)},
    )
    if getattr(args, "no_version", False):
        path = write_baseline(baselines_dir, baseline_id, doc)
        print(json.dumps({"baseline_id": baseline_id, "path": str(path), "evidence_status": doc["evidence_status"], "sample_count": doc["sample_count"], "versioned": False}, ensure_ascii=False, indent=2))
        return 0
    result = write_baseline_version(
        baselines_dir, baseline_id, doc,
        now=now_iso(), note=getattr(args, "note", None),
    )
    summary = {
        "baseline_id": baseline_id,
        "version": result["version"],
        "dedup": result["dedup"],
        "evidence_status": doc["evidence_status"],
        "sample_count": doc["sample_count"],
        "pointer_path": result["pointer_path"],
    }
    if result.get("regressed"):
        # fingerprint matched an OLDER version, not the tip — worth surfacing.
        summary["regressed_to_prior_version"] = True
    if result.get("drift") and result["drift"].get("changed"):
        summary["drift_from_parent"] = result["drift"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def baseline_inspect(args: argparse.Namespace) -> int:
    baselines_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR)
    doc = load_baseline(baselines_dir, args.baseline_id)
    if doc is None:
        raise ValueError(f"baseline not found: {args.baseline_id}")
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0


def baseline_versions(args: argparse.Namespace) -> int:
    baselines_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR)
    versions = list_baseline_versions(baselines_dir, args.baseline_id)
    if not versions:
        # An un-versioned (legacy) baseline still exists as a single file.
        if load_baseline(baselines_dir, args.baseline_id) is not None:
            print(json.dumps({"baseline_id": args.baseline_id, "versions": [], "note": "legacy single-file baseline (no lineage yet); rebuild to start versioning"}, ensure_ascii=False, indent=2))
            return 0
        raise ValueError(f"baseline not found: {args.baseline_id}")
    rows = [
        {
            "version": v.get("version"),
            "created_at": v.get("created_at"),
            "last_seen": v.get("last_seen"),
            "observed_count": v.get("observed_count"),
            "evidence_status": v.get("evidence_status"),
            "sample_count": v.get("sample_count"),
            "model": v.get("model"),
            "content_hash": v.get("content_hash"),
            "drift_changed": bool((v.get("drift_from_parent") or {}).get("changed")),
            "note": v.get("note"),
        }
        for v in versions
    ]
    print(json.dumps({"baseline_id": args.baseline_id, "version_count": len(rows), "versions": rows}, ensure_ascii=False, indent=2))
    return 0


def baseline_diff(args: argparse.Namespace) -> int:
    baselines_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR)

    def _resolve(ref: str) -> dict[str, Any] | None:
        # "latest" / None -> the current pointer; otherwise a version label (v0003).
        if ref in (None, "", "latest"):
            return load_baseline(baselines_dir, args.baseline_id)
        return load_baseline_version(baselines_dir, args.baseline_id, ref)

    versions = list_baseline_versions(baselines_dir, args.baseline_id)
    from_ref = getattr(args, "from_version", None)
    to_ref = getattr(args, "to_version", None) or "latest"
    if from_ref is None:
        # default: diff the previous version against the latest
        if len(versions) >= 2:
            from_ref = versions[-2]["version"]
        elif len(versions) == 1:
            from_ref = versions[-1]["version"]
        else:
            raise ValueError("no version lineage to diff; rebuild the baseline first")

    old = _resolve(from_ref)
    new = _resolve(to_ref)
    if old is None:
        raise ValueError(f"version not found: {from_ref}")
    if new is None:
        raise ValueError(f"version not found: {to_ref}")
    drift = diff_baselines(old, new)
    print(json.dumps({"baseline_id": args.baseline_id, "from": from_ref, "to": to_ref, "drift": drift}, ensure_ascii=False, indent=2))
    return 0


def baseline_derive_windows(args: argparse.Namespace) -> int:
    baselines_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR)
    doc = load_baseline(baselines_dir, args.baseline_id)
    if doc is None:
        raise ValueError(f"baseline not found: {args.baseline_id}")
    derived = derive_token_windows(
        doc,
        long_probe=str(getattr(args, "long_probe", None) or "canary_mixed"),
        short_probe=str(getattr(args, "short_probe", None) or "canary_zh"),
    )
    if derived.get("ok") and getattr(args, "write", False):
        out_path = baselines_dir / args.baseline_id.replace("/", "_").replace("\\", "_") / "token_probe_windows.json"
        from baseline_registry import write_json as _wj
        _wj(out_path, derived)
        derived["written_to"] = str(out_path)
    print(json.dumps(derived, ensure_ascii=False, indent=2))
    return 0 if derived.get("ok") else 1


def baseline_compare(args: argparse.Namespace) -> int:
    baselines_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR)
    baseline = load_baseline(baselines_dir, args.baseline_id)
    if baseline is None:
        raise ValueError(f"baseline not found: {args.baseline_id}")
    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    role = str(args.provider or "tested_model")
    live = bool(getattr(args, "live", False))
    if live:
        load_local_env()
    models = apply_model_overrides(load_two_model_config(models_path), args)
    if role not in models:
        raise ValueError("--provider must be tested_model or judge_model")
    model = models[role]
    out_dir = baselines_dir / "_compare" / role
    events_file = out_dir / "compare_events.jsonl"
    samples = _collect_baseline_samples(
        model, samples_per_probe=max(1, int(getattr(args, "samples", 2) or 2)),
        live=live, events_file=events_file,
    )
    observed = build_baseline_from_samples(
        samples,
        {"provider_id": model.provider_id, "provider_label": role,
         "base_url_host": base_url_host(model.base_url), "model": model.model, "protocol": model.protocol},
        baseline_id=f"observed_{role}", live=live,
    )
    verdict = compare_to_baseline(observed, baseline)
    if getattr(args, "report", False):
        print(render_verdict_report(verdict, baseline=baseline))
    else:
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


def judge_calibrate(args: argparse.Namespace) -> int:
    """Calibrate the judge model against an authored golden-set.

    For each golden case, send (task, candidate_answer) to the judge model,
    parse its JSON verdict, and feed the observed decision/score into the
    offline calibration metrics. --live makes REAL judge calls (cost). Without
    --live it runs dry (stub judge), useful only to smoke the wiring.
    """
    golden_path = resolve_path(args.golden_set)
    golden = load_golden_set(golden_path)
    cases = golden["cases"]

    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    role = str(args.provider or "judge_model")
    live = bool(getattr(args, "live", False))
    if live:
        load_local_env()
    models = apply_model_overrides(load_two_model_config(models_path), args)
    if role not in models:
        raise ValueError(f"--provider role '{role}' not found in providers config")
    judge_model = models[role]
    judge_max_tokens = int(getattr(args, "judge_max_tokens", None) or 512)
    retries = int(getattr(args, "retries", 0) or 0)
    retry_backoff = float(getattr(args, "retry_backoff", 0.0) or 0.0)
    request_delay = float(getattr(args, "request_delay", 0.0) or 0.0)

    out_dir = resolve_path(getattr(args, "out_dir", None) or (DEFAULT_BASELINES_DIR / "_judge_calibration"))
    events_file = out_dir / "judge_events.jsonl"
    client_timeout = httpx.Timeout(float(getattr(args, "timeout", 120.0) or 120.0))

    observations: list[dict[str, Any]] = []
    with httpx.Client(timeout=client_timeout, follow_redirects=True) as client:
        for i, case in enumerate(cases):
            if live and request_delay and i > 0:
                time.sleep(request_delay)
            task = case.get("task") or {}
            candidate = str(case.get("candidate_answer") or "")
            completion = call_model_with_retries(
                client=client,
                model=judge_model,
                messages=judge_messages(task, candidate),
                max_tokens=judge_max_tokens,
                temperature=0,
                live=live,
                events_file=events_file,
                retries=retries,
                retry_backoff=retry_backoff,
            )
            if not completion.metrics.ok:
                observations.append({"id": case["id"], "observed_decision": None, "ok": False,
                                     "error": redact_text(completion.metrics.error, max_chars=300)})
                continue
            payload, parse_error = parse_judge_json(completion.text)
            if payload is None:
                observations.append({"id": case["id"], "observed_decision": None, "ok": False,
                                     "error": redact_text(parse_error or "judge JSON parse failed", max_chars=300)})
                continue
            observations.append({
                "id": case["id"],
                "observed_decision": normalize_decision(payload.get("decision")),
                "observed_score": payload.get("score_0_10"),
                "ok": True,
            })

    result = compute_calibration(cases, observations)
    result["judge_provider"] = judge_model.provider_id
    result["judge_model_requested"] = judge_model.model
    result["live"] = live
    verdict = classify_judge(result, min_scored=int(getattr(args, "min_scored", 4) or 4))

    if getattr(args, "write", False):
        from baseline_registry import write_json as _wj
        _wj(out_dir / "last_calibration.json", {"result": result, "verdict": verdict})

    if getattr(args, "report", False):
        print(render_calibration_report(result, verdict))
    else:
        print(json.dumps({"verdict": verdict, "result": result}, ensure_ascii=False, indent=2))
    return 0


def _run_capability_items(
    items: list[dict[str, Any]], model: ModelConfig, *,
    live: bool, events_file: Path, request_delay: float, retries: int,
    retry_backoff: float, max_tokens: int, timeout: float,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Send each capability item to the model and grade it deterministically.

    `progress`, when given, is called with a dict per item so a caller (the web
    SSE endpoint) can stream live progress. None = no-op (CLI path unchanged).
    """
    results: list[dict[str, Any]] = []
    client_timeout = httpx.Timeout(float(timeout or 120.0))
    total = len(items)
    with httpx.Client(timeout=client_timeout, follow_redirects=True) as client:
        for i, item in enumerate(items):
            if live and request_delay and i > 0:
                time.sleep(request_delay)
            if progress is not None:
                progress({"stage": "capability", "done": i, "total": total,
                          "label": f"能力探针 {i + 1}/{total}"})
            completion = call_model_with_retries(
                client=client, model=model,
                messages=[{"role": "user", "content": str(item.get("prompt") or "")}],
                max_tokens=max_tokens, temperature=0, live=live,
                events_file=events_file, retries=retries, retry_backoff=retry_backoff,
            )
            if not completion.metrics.ok:
                results.append({"id": item.get("id"), "passed": None, "ok": False,
                                "detail": redact_text(completion.metrics.error, max_chars=200)})
                continue
            graded = score_capability_item(item, completion.text)
            results.append({"id": item.get("id"), "passed": graded["passed"], "ok": True,
                            "detail": graded["detail"]})
    if progress is not None:
        progress({"stage": "capability", "done": total, "total": total,
                  "label": f"能力探针完成 {total}/{total}"})
    return results


def _run_variance_probe(
    item: dict[str, Any], model: ModelConfig, *,
    live: bool, events_file: Path, repeats: int, request_delay: float,
    retries: int, retry_backoff: float, max_tokens: int, timeout: float,
    progress: Any = None, fail_fast: int = 4,
) -> list[dict[str, Any]]:
    """Repeat ONE deterministic anchor `repeats` times at temp=0 to surface
    low-frequency swapping (a fraction of requests routed to a weaker model shows
    up as occasional wrong/varying answers). Returns per-repeat
    [{passed, answer_norm, ok}] for score_consistency_variance.

    `progress`, when given, is called per repeat for live SSE updates.

    Circuit breaker (R-001/R-002 lesson): if `fail_fast` consecutive repeats fail
    (e.g. the gateway rate-limits rapid identical requests, as aigocode did),
    abort the remaining repeats instead of grinding through 12×retries of doomed
    calls — that both wastes quota and looks like an abuse pattern to upstream
    rate-limiters. score_consistency_variance then sees too-few-answered ->
    advisory, NOT a false conviction.
    """
    out: list[dict[str, Any]] = []
    client_timeout = httpx.Timeout(float(timeout or 120.0))
    prompt = str(item.get("prompt") or "")
    consecutive_fail = 0
    aborted = False
    with httpx.Client(timeout=client_timeout, follow_redirects=True) as client:
        for i in range(max(1, repeats)):
            if live and request_delay and i > 0:
                time.sleep(request_delay)
            if progress is not None:
                progress({"stage": "variance", "done": i, "total": repeats,
                          "label": f"一致性复检 {i + 1}/{repeats}"})
            completion = call_model_with_retries(
                client=client, model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0, live=live,
                events_file=events_file, retries=retries, retry_backoff=retry_backoff,
            )
            if not completion.metrics.ok:
                out.append({"passed": None, "answer_norm": None, "ok": False})
                consecutive_fail += 1
                if live and consecutive_fail >= max(1, fail_fast):
                    append_jsonl(events_file, {"at": now_iso(), "type": "variance_circuit_break",
                                               "consecutive_failures": consecutive_fail,
                                               "completed": i + 1, "planned": repeats})
                    aborted = True
                    break
                continue
            consecutive_fail = 0
            graded = score_capability_item(item, completion.text)
            # normalized answer for determinism check: collapse whitespace + lowercase
            answer_norm = " ".join((completion.text or "").split()).lower()[:120]
            out.append({"passed": bool(graded["passed"]), "answer_norm": answer_norm, "ok": True})
    if progress is not None:
        label = (f"一致性复检中止（连续失败，已跑 {len(out)} 次）" if aborted
                 else f"一致性复检完成 {repeats}/{repeats}")
        progress({"stage": "variance", "done": len(out) if aborted else repeats,
                  "total": repeats, "label": label, "aborted": aborted})
    return out


# A neutral identity question. We do NOT grade the answer text for correctness
# (a wrapper can recite "I am Claude Opus 4.8"); we only need a live request so we
# can read the ENVELOPE (returned model field + response id) and cross-check it
# against the narration. Asked plainly so a genuine model answers naturally.
IDENTITY_PROBE_PROMPT = (
    "请直接回答你的精确模型标识符（如 claude-opus-4-8 这种完整版本号），只回标识符本身。"
)


def _run_identity_probe(
    model: ModelConfig, *, live: bool, events_file: Path,
    expected_model_id: str | None, request_delay: float, retries: int,
    retry_backoff: float, timeout: float = 60.0, attempts: int = 3,
    progress: Any = None,
) -> dict[str, Any]:
    """One live request whose ENVELOPE (returned model field + response id) is
    cross-checked against the model's self-narrated identity.

    The narration is forgeable; the envelope is not. score_identity_coherence
    flags the mismatch. Returns the scored dict (compare_to_baseline contract) or
    a probe_error dict so a crash is surfaced as an incomplete check, never a
    silent clean pass.

    The envelope can be read from ANY successful response, so on a transient
    failure we retry the whole probe up to `attempts` times (outer loop, on top of
    call_model_with_retries' inner retries). Otherwise one unlucky timeout on the
    single probe call would waste the entire signal — exactly what happened to
    two gateways in the 2026-06-30 sweep, dropping them to an incomplete 0.5 verdict.
    """
    if progress is not None:
        progress({"stage": "identity", "done": 0, "total": 1, "label": "身份一致性探针…"})
    client_timeout = httpx.Timeout(float(timeout or 60.0))
    last_err = "no attempt made"
    with httpx.Client(timeout=client_timeout, follow_redirects=True) as client:
        for attempt in range(max(1, attempts)):
            if live and request_delay and attempt > 0:
                time.sleep(request_delay)
            completion = call_model_with_retries(
                client=client, model=model,
                messages=[{"role": "user", "content": IDENTITY_PROBE_PROMPT}],
                max_tokens=64, temperature=0, live=live,
                events_file=events_file, retries=retries, retry_backoff=retry_backoff,
            )
            if completion.metrics.ok:
                obs = _raw_protocol_observation(completion, model)
                narrated = " ".join((completion.text or "").split())[:120] or None
                scored = score_identity_coherence(
                    narrated_model_id=narrated,
                    returned_model_field=obs.get("returned_model_field"),
                    response_id=obs.get("response_id"),
                    expected_model_id=expected_model_id,
                )
                if progress is not None:
                    progress({"stage": "identity", "done": 1, "total": 1, "label": "身份一致性探针完成"})
                return scored
            last_err = redact_text(completion.metrics.error, max_chars=200) or "request failed"
    return {"probe_error": last_err}


def capability_probe(args: argparse.Namespace) -> int:
    """Probe a provider's capability pass-rate on hard anchors (downgrade detector).

    Runs the objective anchor set, grades deterministically (no judge), and
    writes capability_anchor.json into the baseline dir. With --baseline-id it
    becomes the trusted baseline's pass-rate; for a suspect, compare via
    verify-endpoint --with-capability. --live makes REAL calls (cost).
    """
    items_path = resolve_path(args.items)
    items_doc = read_json(items_path)
    items = items_doc.get("items") if isinstance(items_doc, dict) else items_doc
    if not isinstance(items, list) or not items:
        raise ValueError(f"capability item set has no 'items': {items_path}")

    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    role = str(args.provider or "tested_model")
    live = bool(getattr(args, "live", False))
    if live:
        load_local_env()
    models = apply_model_overrides(load_two_model_config(models_path), args)
    if role not in models:
        raise ValueError(f"--provider role '{role}' not found in providers config")
    model = models[role]

    baselines_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR)
    out_id = getattr(args, "baseline_id", None)
    out_dir = (baselines_dir / out_id.replace("/", "_").replace("\\", "_")) if out_id else (baselines_dir / "_capability" / role)
    events_file = out_dir / "capability_events.jsonl"

    results = _run_capability_items(
        items, model, live=live, events_file=events_file,
        request_delay=float(getattr(args, "request_delay", 0.0) or 0.0),
        retries=int(getattr(args, "retries", 1) or 0),
        retry_backoff=float(getattr(args, "retry_backoff", 0.5) or 0.0),
        max_tokens=int(getattr(args, "max_tokens", 256) or 256),
        timeout=float(getattr(args, "timeout", 120.0) or 120.0),
    )
    agg = aggregate_capability(results)
    doc = {
        "schema_version": "capability_anchor_v1",
        "evidence_status": "live_observed" if live else "dry_run_reference_only",
        "provider_id": model.provider_id,
        "model": model.model,
        "items_source": str(items_path.name),
        **agg,
        "per_item": results,
    }
    from baseline_registry import write_json as _wj
    _wj(out_dir / "capability_anchor.json", doc)

    summary = {k: doc[k] for k in ("capability_anchor_pass_rate", "answered_count",
                                   "passed_count", "failed_request_count", "total_items",
                                   "evidence_status")}
    summary["written_to"] = str(out_dir / "capability_anchor.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def verify_core(
    model: ModelConfig,
    baseline: dict[str, Any],
    *,
    role: str,
    baselines_dir: Path,
    baseline_id: str,
    live: bool,
    samples_per_probe: int = 5,
    request_delay: float = 0.0,
    retries: int = 1,
    retry_backoff: float = 0.5,
    with_sse: bool = False,
    with_error_envelope: bool = False,
    with_needle: bool = False,
    needle_tokens: int = 120000,
    with_capability: bool = False,
    capability_items: Path | None = None,
    with_variance: bool = False,
    variance_repeats: int = 12,
    with_identity: bool = False,
    providers_path: Path | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Pure verification core, no argparse/print/stdout — returns the verdict dict.

    Shared by the CLI (`verify_endpoint`) and the web endpoint
    (`api_server.run_web_verify`) so both paths run identical detection logic.
    Collects the suspect's fingerprint, folds in behavior signals when `live`,
    and returns `compare_to_baseline(...)`.

    The SSE / error-envelope / needle sub-probes reconstruct an argparse
    Namespace and re-read the providers file, so they need `providers_path` +
    `role`; the web path keeps them off (only tokenizer + capability), so those
    args may be None there.

    `progress`, when given, is called with a dict at each stage so the web SSE
    endpoint can stream live progress. None = no-op (CLI path unchanged).
    """
    def _emit(ev: dict[str, Any]) -> None:
        if progress is not None:
            progress(ev)

    out_dir = baselines_dir / "_verify" / role
    events_file = out_dir / "verify_events.jsonl"
    _emit({"stage": "sampling", "label": "采集协议指纹（stop_reason / usage / 头）…"})
    samples = _collect_baseline_samples(
        model, samples_per_probe=max(1, int(samples_per_probe or 5)),
        live=live, events_file=events_file,
        request_delay=float(request_delay or 0.0),
        retries=int(retries or 1),
        retry_backoff=float(retry_backoff or 0.5),
    )
    observed = build_baseline_from_samples(
        samples,
        {"provider_id": model.provider_id, "provider_label": role,
         "base_url_host": base_url_host(model.base_url), "model": model.model, "protocol": model.protocol},
        baseline_id=f"verify_{role}", live=live,
    )
    _emit({"stage": "protocol_done", "label": "协议指纹采集完成"})

    # gather behavior signals (the hard-to-fake layer) when live
    behavior: dict[str, Any] = {}
    if live:
        # tokenizer delta from observed probe windows vs baseline delta window
        bwin = (baseline.get("behavior") or {}).get("tokenizer_probe_windows") or {}
        owin = (observed.get("behavior") or {}).get("tokenizer_probe_windows") or {}
        def _mean(w, p):
            d = w.get(p)
            return d.get("mean") if isinstance(d, dict) else None
        b_long, b_short = _mean(bwin, "canary_mixed"), _mean(bwin, "canary_zh")
        o_long, o_short = _mean(owin, "canary_mixed"), _mean(owin, "canary_zh")
        def _std(w, p):
            d = w.get(p)
            return d.get("stdev") if isinstance(d, dict) else None
        if None not in (b_long, b_short, o_long, o_short):
            base_delta = b_long - b_short
            obs_delta = o_long - o_short
            # dynamic tolerance: absorb the actual per-probe noise on BOTH sides
            # (short canaries have a few-token jitter that differencing amplifies),
            # floored generously so small samples don't false-positive.
            noise = sum(filter(None, [_std(bwin, "canary_mixed"), _std(bwin, "canary_zh"),
                                      _std(owin, "canary_mixed"), _std(owin, "canary_zh")]))
            tol = max(0.15 * abs(base_delta), 3.0 * (noise or 1.0), 25.0)
            # #4: tokenizer differencing is noisy at small N -> demote to advisory
            # (score None: shown in evidence, but does NOT vote or penalize the verdict)
            def _n(w, p):
                d = w.get(p)
                return d.get("count") if isinstance(d, dict) else None
            min_n = min(filter(lambda x: x is not None,
                               [_n(owin, "canary_mixed"), _n(owin, "canary_zh")] or [None]), default=0)
            within = abs(obs_delta - base_delta) <= tol
            if min_n is not None and min_n < 3:
                score = None  # advisory only — too few samples to trust the delta
                detail = f"tokenizer delta advisory (only {min_n} samples/probe; need >=3 to vote)"
            else:
                score = 10.0 if within else 0.0
                detail = "tokenizer delta within baseline window (corroborating)" if within else "tokenizer delta off baseline (noisy / corroborating only)"
            behavior["tokenizer"] = {
                "score": score,
                "observed": {"obs_delta": round(obs_delta, 1), "base_delta": round(base_delta, 1), "tol": round(tol, 1), "min_samples": min_n},
                "details": detail,
                "suspected_tokenizer": None if score in (10.0, None) else "unknown",
            }
        # SSE event-order probe
        if with_sse:
            try:
                sse_args = argparse.Namespace(providers=providers_path, provider=role, live=True)
                # reuse the classifier by collecting one stream inline
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    sse_fingerprint(sse_args)
                behavior["sse"] = json.loads(buf.getvalue())
            except Exception as exc:
                # A requested probe that crashed must NOT vanish silently — that
                # would let a verify run look complete while a signal is missing.
                # Record it so compare_to_baseline counts it as an incomplete check.
                behavior["sse"] = {"probe_error": f"{type(exc).__name__}: {exc}"}
        # error-envelope probe (#2): malformed request -> classify error dialect
        if with_error_envelope:
            try:
                ee_args = argparse.Namespace(providers=providers_path, provider=role, live=True)
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    error_envelope(ee_args)
                ee = json.loads(buf.getvalue())
                results = ee.get("results") or []
                # pick a dialect from any 4xx variant
                dialects = [r.get("error_envelope_dialect") for r in results if (r.get("http_status") or 0) >= 400]
                if dialects:
                    behavior["error_envelope"] = {"error_envelope_dialect": dialects[0]}
                else:
                    behavior["error_envelope"] = {"probe_error": "no 4xx response to classify"}
            except Exception as exc:
                behavior["error_envelope"] = {"probe_error": f"{type(exc).__name__}: {exc}"}
        # needle long-context probe (#1): ~120K context, recall + silent-truncation
        if with_needle:
            try:
                nd_args = argparse.Namespace(providers=providers_path, provider=role, live=True,
                                             target_tokens=int(needle_tokens or 120000),
                                             seed=7, baseline_id=baseline_id,
                                             baselines_dir=baselines_dir, timeout=300.0)
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    needle(nd_args)
                nd = json.loads(buf.getvalue())
                behavior["needle"] = nd
            except Exception as exc:
                behavior["needle"] = {"probe_error": f"{type(exc).__name__}: {exc}"}
        # capability-anchor probe: the dedicated silent-DOWNGRADE detector.
        # Runs the hard anchors against the suspect, compares pass-rate to the
        # baseline's stored capability_anchor.json (built earlier via capability-probe).
        if with_capability:
            try:
                cap_items_path = resolve_path(capability_items
                                              or (ROOT / "judge_golden" / "capability_anchors_v1.json"))
                cap_items_doc = read_json(cap_items_path)
                cap_items = _as_list(cap_items_doc.get("items") if isinstance(cap_items_doc, dict) else cap_items_doc)
                base_cap_path = baselines_dir / str(baseline_id).replace("/", "_").replace("\\", "_") / "capability_anchor.json"
                base_rate = None
                if base_cap_path.exists():
                    base_rate = numeric(read_json(base_cap_path).get("capability_anchor_pass_rate"))
                cap_results = _run_capability_items(
                    cap_items, model, live=True,
                    events_file=out_dir / "capability_events.jsonl",
                    request_delay=float(request_delay or 0.0),
                    retries=int(retries or 1),
                    retry_backoff=float(retry_backoff or 0.5),
                    max_tokens=256, timeout=120.0,
                    progress=progress,
                )
                cap_agg = aggregate_capability(cap_results)
                cap_score = score_capability_vs_baseline(
                    cap_agg.get("capability_anchor_pass_rate"), base_rate,
                    answered_count=cap_agg.get("answered_count", 0),
                )
                behavior["capability"] = {**cap_score, "answered": cap_agg.get("answered_count"),
                                          "passed": cap_agg.get("passed_count")}
            except Exception as exc:
                behavior["capability"] = {"probe_error": f"{type(exc).__name__}: {exc}"}

        # consistency-variance probe: repeat ONE deterministic anchor N times to
        # catch low-frequency swapping (a fraction routed to a weaker model).
        if with_variance:
            try:
                var_items_path = resolve_path(capability_items
                                              or (ROOT / "judge_golden" / "capability_anchors_v1.json"))
                var_doc = read_json(var_items_path)
                var_items = _as_list(var_doc.get("items") if isinstance(var_doc, dict) else var_doc)
                if not var_items:
                    raise ValueError("no anchor items for variance probe")
                anchor = var_items[0]  # first anchor is a stable single-answer item
                reps = _run_variance_probe(
                    anchor, model, live=True,
                    events_file=out_dir / "variance_events.jsonl",
                    repeats=int(variance_repeats or 12),
                    request_delay=float(request_delay or 0.0),
                    retries=int(retries or 1),
                    retry_backoff=float(retry_backoff or 0.5),
                    max_tokens=256, timeout=120.0, progress=progress,
                )
                behavior["variance"] = score_consistency_variance(reps)
            except Exception as exc:
                behavior["variance"] = {"probe_error": f"{type(exc).__name__}: {exc}"}

        # identity-coherence probe: cross-check the suspect's self-narrated model
        # id against the forge-resistant envelope (returned model field + response
        # id prefix). Cheap (one request), so it runs whenever live identity is
        # requested. expected_model_id = the trusted baseline's served model.
        if with_identity:
            try:
                expected_model_id = (baseline.get("source") or {}).get("model")
                behavior["identity"] = _run_identity_probe(
                    model, live=True,
                    events_file=out_dir / "identity_events.jsonl",
                    expected_model_id=expected_model_id,
                    request_delay=float(request_delay or 0.0),
                    retries=int(retries or 1),
                    retry_backoff=float(retry_backoff or 0.5),
                    progress=progress,
                )
            except Exception as exc:
                behavior["identity"] = {"probe_error": f"{type(exc).__name__}: {exc}"}

    _emit({"stage": "judging", "label": "综合判定中…"})
    return compare_to_baseline(observed, baseline, behavior_signals=behavior or None)


def verify_endpoint(args: argparse.Namespace) -> int:
    """One-shot: compare a suspect provider against a trusted baseline, print a report.

    Loads the named baseline, collects the suspect's fingerprint, renders a
    human-readable verdict. --live actually probes the suspect (cost).
    Thin wrapper over `verify_core` (the shared, argparse-free detection core).
    """
    baselines_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR)
    baseline = load_baseline(baselines_dir, args.baseline_id)
    if baseline is None:
        raise ValueError(f"baseline not found: {args.baseline_id} (build one first with `baseline --live`)")
    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    role = str(args.provider or "tested_model")
    live = bool(getattr(args, "live", False))
    if live:
        load_local_env()
    models = apply_model_overrides(load_two_model_config(models_path), args)
    if role not in models:
        raise ValueError(f"--provider '{role}' not found in providers config (available: {', '.join(sorted(models))})")
    model = models[role]
    verdict = verify_core(
        model, baseline,
        role=role,
        baselines_dir=baselines_dir,
        baseline_id=str(args.baseline_id),
        live=live,
        samples_per_probe=int(getattr(args, "samples", 5) or 5),
        request_delay=float(getattr(args, "request_delay", 0.0) or 0.0),
        retries=int(getattr(args, "retries", 1) or 1),
        retry_backoff=float(getattr(args, "retry_backoff", 0.5) or 0.5),
        with_sse=bool(getattr(args, "with_sse", False)),
        with_error_envelope=bool(getattr(args, "with_error_envelope", False)),
        with_needle=bool(getattr(args, "with_needle", False)),
        needle_tokens=int(getattr(args, "needle_tokens", 120000) or 120000),
        with_capability=bool(getattr(args, "with_capability", False)),
        capability_items=getattr(args, "capability_items", None),
        with_variance=bool(getattr(args, "with_variance", False)),
        variance_repeats=int(getattr(args, "variance_repeats", 12) or 12),
        with_identity=bool(getattr(args, "with_identity", False)),
        providers_path=args.providers,
    )
    print(render_verdict_report(verdict, baseline=baseline))
    if getattr(args, "json", False):
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0



# Fake-1M needle probe. The huge prompt is assembled at run time from a seed
# (NEVER stored in the 495KB task file).
# Empirically (live gateway opus-4-6 ladder test 2026-06-26): ~2.5 chars/token for
# this English filler (400K chars -> 160031 input_tokens). The upstream rejects
# single requests above ~160K-220K tokens (429/timeout), so keep target modest.
NEEDLE_CHARS_PER_TOKEN = 2.5


def _assemble_needle_prompt(target_tokens: int, seed: int, depth: float = 0.01) -> tuple[str, str]:
    """Build a ~target_tokens filler prompt with a planted AUTH_CANARY at `depth`.

    Reproducible from (target_tokens, seed). Returns (prompt, canary_code).
    """
    rng_state = (seed * 2654435761) & 0xFFFFFFFF
    canary_code = f"AUTH_CANARY={rng_state:08x}"
    # small 5% headroom so real tokens just clear target without overshooting
    # into the upstream's large-request rejection band.
    target_chars = int(target_tokens * NEEDLE_CHARS_PER_TOKEN * 1.05)
    sentence = (
        f"Filler line {seed}: the quick brown fox jumps over the lazy dog, "
        "padding context to stress the model's long-context handling. "
    )
    reps = max(1, target_chars // len(sentence) + 1)
    body = sentence * reps
    insert_at = int(len(body) * depth)
    needle_line = f"\n[IMPORTANT MEMO] Remember this exact code and echo it verbatim at the end: {canary_code}\n"
    prompt = (
        body[:insert_at]
        + needle_line
        + body[insert_at:]
        + "\n\nTask: Output ONLY the exact AUTH_CANARY code that appeared in the [IMPORTANT MEMO] above, nothing else."
    )
    return prompt, canary_code


def needle(args: argparse.Namespace) -> int:
    target_tokens = int(getattr(args, "target_tokens", 120000) or 120000)
    # The window the endpoint CLAIMS to support (e.g. 1_000_000 for a "1M" gateway).
    # The probe can only prove context up to target_tokens (upstreams reject single
    # requests above ~160-220K), so when advertised >> target we report that the
    # advertised window is UNPROVEN rather than implying it was verified.
    advertised_tokens = getattr(args, "advertised_tokens", None)
    advertised_tokens = int(advertised_tokens) if advertised_tokens else None
    seed = int(getattr(args, "seed", 1) or 1)
    live = bool(getattr(args, "live", False))
    if live:
        load_local_env()
    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    role = str(args.provider or "tested_model")
    models = apply_model_overrides(load_two_model_config(models_path), args)
    if role not in models:
        raise ValueError("--provider must be tested_model or judge_model")
    model = models[role]

    prompt, canary = _assemble_needle_prompt(target_tokens, seed)
    sent_chars = len(prompt)
    sent_estimate_tokens = sent_chars / NEEDLE_CHARS_PER_TOKEN

    # prefix from the suspect's own baseline canary_code window if available
    prefix_tokens = None
    baseline_id = getattr(args, "baseline_id", None)
    if baseline_id:
        baselines_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR)
        base = load_baseline(baselines_dir, baseline_id)
        if base:
            win = ((base.get("behavior") or {}).get("tokenizer_probe_windows") or {}).get("canary_code")
            if isinstance(win, dict):
                prefix_tokens = win.get("mean")

    out_dir = resolve_path(getattr(args, "baselines_dir", None) or DEFAULT_BASELINES_DIR) / "_needle" / role
    events_file = out_dir / "needle_events.jsonl"

    http_status = 200
    observed_input_tokens = None
    recall = {"score": None, "details": "not measured"}
    if not live:
        result = {
            "probe": "needle_recall",
            "evidence_status": "dry_run_reference_only",
            "target_tokens": target_tokens,
            "advertised_tokens": advertised_tokens,
            "seed": seed,
            "canary_sha256": hashlib.sha256(canary.encode()).hexdigest()[:16],
            "sent_chars": sent_chars,
            "sent_estimate_tokens": round(sent_estimate_tokens, 1),
            "note": "dry-run: prompt assembled + canary planted; no API call. Use --live to actually probe.",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    timeout = httpx.Timeout(float(getattr(args, "timeout", 300.0) or 300.0))
    with httpx.Client(timeout=timeout) as client:
        completion = call_model_with_retries(
            client=client, model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64, temperature=0, live=True,
            events_file=events_file, retries=0, retry_backoff=0.0,
        )
    if not completion.metrics.ok:
        err = str(completion.metrics.error or "")
        m = re.search(r"HTTP\s+(\d+)", err)
        http_status = int(m.group(1)) if m else 0
        recall = {"score": None, "details": f"request failed (HTTP {http_status}); not measured"}
    else:
        obs = _raw_protocol_observation(completion, model)
        observed_input_tokens = obs["input_tokens"]
        recall = score_needle_recall(canary, completion.text)

    def _num(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    recalled = recall.get("score") == 10.0 if recall.get("score") is not None else None
    truncation = evaluate_silent_truncation(
        sent_estimate_tokens=sent_estimate_tokens,
        observed_input_tokens=_num(observed_input_tokens),
        prefix_tokens=_num(prefix_tokens),
        http_status=http_status,
        needle_recalled=recalled,
    )
    verdict = "fake_1m_silent_truncation" if truncation.get("silent_truncation") else (
        "context_ok" if recall.get("score") == 10.0 else "insufficient_or_legit_error"
    )
    # Honesty guard: a successful recall at target_tokens only proves context up
    # to target_tokens. If the endpoint advertises a much larger window, that
    # window is UNPROVEN — don't let context_ok be read as "1M verified".
    advertised_window_proven = None
    if advertised_tokens:
        advertised_window_proven = bool(verdict == "context_ok" and target_tokens >= advertised_tokens)
        if verdict == "context_ok" and target_tokens < advertised_tokens:
            verdict = "context_ok_below_advertised"
    result = {
        "probe": "needle_recall",
        "evidence_status": "live_observed",
        "target_tokens": target_tokens,
        "advertised_tokens": advertised_tokens,
        "advertised_window_proven": advertised_window_proven,
        "seed": seed,
        "http_status": http_status,
        "observed_input_tokens": observed_input_tokens,
        "needle_recall": recall,
        "silent_truncation": truncation,
        "verdict": verdict,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def error_envelope(args: argparse.Namespace) -> int:
    """#8 error-envelope probe: send malformed requests, classify the error body's
    dialect (anthropic / openai / gateway_generic). Independent of call_model.
    """
    live = bool(getattr(args, "live", False))
    if live:
        load_local_env()
    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    role = str(args.provider or "tested_model")
    models = apply_model_overrides(load_two_model_config(models_path), args)
    if role not in models:
        raise ValueError("--provider must be tested_model or judge_model")
    model = models[role]

    variants = {
        "missing_max_tokens": {"model": model.model, "messages": [{"role": "user", "content": "hi"}]},
        "bad_field": {"model": model.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "temperature": 99, "not_a_real_field": True},
        "oversized_max_tokens": {"model": model.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 99999999},
    }
    if not live:
        print(json.dumps({"probe": "error_envelope", "evidence_status": "dry_run_reference_only",
                          "variants": list(variants), "note": "dry-run: no request sent. Use --live to probe."}, ensure_ascii=False, indent=2))
        return 0

    url = f"{model.base_url}/v1/messages" if model.protocol == "anthropic_messages" else f"{model.base_url}/v1/chat/completions"
    secret = auth_value(model)
    headers = {**auth_headers(model, secret), "content-type": "application/json"}
    results = []
    with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
        for name, payload in variants.items():
            try:
                resp = client.post(url, headers=headers, json=payload)
                status = resp.status_code
                body_text = resp.text[:2000]
                cls = classify_error_envelope(body_text, safe_response_headers(resp.headers))
            except Exception as exc:
                status = 0
                cls = {"error_envelope_dialect": "unknown", "error": redact_text(f"{type(exc).__name__}: {exc}", max_chars=200)}
            results.append({"variant": name, "http_status": status, **cls,
                            "body_preview": redact_text(body_text, max_chars=240) if status else None})
    dialects = [r["error_envelope_dialect"] for r in results if r.get("http_status", 0) >= 400]
    overall = "anthropic" if dialects and all(d == "anthropic" for d in dialects) else (
        "suspect" if any(d in ("openai", "gateway_generic") for d in dialects) else "insufficient")
    print(json.dumps({"probe": "error_envelope", "evidence_status": "live_observed",
                      "overall": overall, "results": results}, ensure_ascii=False, indent=2))
    return 0


def sse_fingerprint(args: argparse.Namespace) -> int:
    """#9 SSE event-order fingerprint: open one streaming request, classify the
    event sequence (claude_sse vs openai_sse). Independent of call_model.
    """
    live = bool(getattr(args, "live", False))
    if live:
        load_local_env()
    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    role = str(args.provider or "tested_model")
    models = apply_model_overrides(load_two_model_config(models_path), args)
    if role not in models:
        raise ValueError("--provider must be tested_model or judge_model")
    model = models[role]
    if not live:
        print(json.dumps({"probe": "sse_event_order", "evidence_status": "dry_run_reference_only",
                          "note": "dry-run: no stream opened. Use --live to probe."}, ensure_ascii=False, indent=2))
        return 0

    secret = auth_value(model)
    event_types: list[str] = []
    http_status = 200
    if model.protocol == "anthropic_messages":
        url = f"{model.base_url}/v1/messages"
        payload = {"model": model.model, "messages": [{"role": "user", "content": "Say hi in one short sentence."}], "max_tokens": 32, "stream": True}
    else:
        url = f"{model.base_url}/v1/chat/completions"
        payload = {"model": model.model, "messages": [{"role": "user", "content": "Say hi in one short sentence."}], "max_tokens": 32, "stream": True}
    headers = {**auth_headers(model, secret), "content-type": "application/json", "accept": "text/event-stream"}
    try:
        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                http_status = resp.status_code
                if http_status == 200:
                    # Use the project's canonical SSE parser (buffer-based, handles
                    # multi-line data: fields and CRLF framing) rather than a naive
                    # per-line split — this probe's whole job is SSE-shape fidelity.
                    for event_name, data_str in iter_sse_events(resp.iter_raw()):
                        if data_str == "[DONE]":
                            event_types.append("[DONE]")
                            continue
                        recorded = None
                        if data_str:
                            try:
                                obj = json.loads(data_str)
                                recorded = obj.get("type") or obj.get("object")
                            except (ValueError, TypeError):
                                recorded = None
                        # fall back to the SSE `event:` name when the data has no type
                        if not recorded and event_name and event_name != "message":
                            recorded = event_name
                        if recorded:
                            event_types.append(str(recorded))
    except Exception as exc:
        print(json.dumps({"probe": "sse_event_order", "evidence_status": "live_observed",
                          "http_status": 0, "error": redact_text(f"{type(exc).__name__}: {exc}", max_chars=200)}, ensure_ascii=False, indent=2))
        return 0
    cls = classify_sse_event_order(event_types)
    print(json.dumps({"probe": "sse_event_order", "evidence_status": "live_observed",
                      "http_status": http_status, "event_sequence": event_types, **cls}, ensure_ascii=False, indent=2))
    return 0


def fingerprint(args: argparse.Namespace) -> int:
    models_path = resolve_path(args.providers or DEFAULT_PROVIDERS)
    models = apply_model_overrides(load_two_model_config(models_path), args)
    role = str(args.provider or "tested_model")
    if role not in models:
        raise ValueError("--provider must be tested_model or judge_model")
    identity = campaign_model_identity(models[role], include_key_fingerprint=False)
    fingerprint_doc = build_config_protocol_fingerprint(identity, provider_label=role, live=bool(getattr(args, "live", False)))
    if getattr(args, "campaign_id", None):
        campaigns_dir, runs_dir = campaign_paths(args)
        out_dir = campaign_dir(campaigns_dir, args.campaign_id)
        write_authenticity_evidence(out_dir, runs_dir, persist=True)
        provider_id = str(identity.get("provider_id") or role).replace("/", "_").replace("\\", "_")
        path = out_dir / "protocol_fingerprints" / f"{provider_id}.json"
        fingerprint_doc = read_json(path) if path.exists() else fingerprint_doc
    print(json.dumps(fingerprint_doc, ensure_ascii=False, indent=2))
    return 0


def authenticity_inspect(args: argparse.Namespace) -> int:
    campaigns_dir, runs_dir = campaign_paths(args)
    out_dir = campaign_dir(campaigns_dir, args.campaign_id)
    evidence = load_or_build_authenticity(out_dir, runs_dir, persist=False)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


def authenticity_export(args: argparse.Namespace) -> int:
    campaigns_dir, runs_dir = campaign_paths(args)
    out_dir = campaign_dir(campaigns_dir, args.campaign_id)
    baseline_dir = baseline_campaign_path(args, campaigns_dir)
    write_authenticity_evidence(
        out_dir,
        runs_dir,
        baseline_campaign_dir=baseline_dir,
        baseline_provider=str(args.baseline_provider or "official_baseline"),
        gateway_provider=str(args.gateway_provider or "gateway_candidate"),
        persist=True,
    )
    zip_path = export_campaign(out_dir, runs_dir, include_raw=bool(getattr(args, "include_raw", False)))
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
        return {"auth_type": auth_type, "ok": False, "error": redact_text(f"{type(exc).__name__}: {exc}", max_chars=500)}
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
        "error_preview": redact_text(completion.metrics.error, max_chars=240) if completion.metrics.error else None,
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


def add_model_override_args(parser: argparse.ArgumentParser) -> None:
    for prefix, label in (("tested", "tested model"), ("judge", "judge model")):
        parser.add_argument(f"--{prefix}-provider-id", help=f"override {label} provider id for this run")
        parser.add_argument(f"--{prefix}-base-url", help=f"override {label} base URL for this run")
        parser.add_argument(f"--{prefix}-model", help=f"override {label} model name for this run")
        parser.add_argument(f"--{prefix}-api-key-env", help=f"override {label} API key environment variable for this run")
        parser.add_argument(f"--{prefix}-protocol", choices=sorted(ALLOWED_PROTOCOLS), help=f"override {label} protocol")
        parser.add_argument(f"--{prefix}-auth-type", choices=sorted(ALLOWED_AUTH_TYPES), help=f"override {label} auth type")
        parser.add_argument(f"--{prefix}-display-name", help=f"override {label} display name for this run")
        parser.add_argument(
            f"--{prefix}-reasoning-effort",
            choices=["default", "none", "low", "medium", "high", "xhigh"],
            help=f"override {label} reasoning_effort extra body field",
        )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="Two-model headless eval CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    self_test_parser = sub.add_parser("self-test", help="run eval_cli internal self-tests")
    def _run_self_test(_args: argparse.Namespace) -> int:
        _self_test_judge_payload_sanitization()
        print("eval_cli self-test ok")
        return 0
    self_test_parser.set_defaults(func=_run_self_test)

    run_parser = sub.add_parser("run", help="run a configured two-model job")
    run_parser.add_argument("--job", default=DEFAULT_JOB, help="job name or path")
    run_parser.add_argument("--providers", type=Path, help="providers config path")
    run_parser.add_argument("--runs-dir", type=Path, help="runs directory")
    run_parser.add_argument("--run-id", help="explicit job id")
    run_parser.add_argument("--live", action="store_true", help="call configured live providers")
    run_parser.add_argument("--timeout", type=float, help="per-request timeout seconds")
    run_parser.add_argument("--tested-max-tokens", type=int)
    run_parser.add_argument("--judge-max-tokens", type=int)
    run_parser.add_argument("--max-concurrency", type=int, help="bounded task-level concurrency; defaults to job or benchmark setting")
    run_parser.add_argument("--retries", type=int, help="retry transient live provider failures this many times")
    run_parser.add_argument("--retry-backoff", type=float, help="initial retry backoff seconds for live provider failures")
    run_parser.add_argument("--require-go", action="store_true", help="return exit code 2 unless the final decision is GO")
    run_parser.add_argument("--skip-trace-evaluation", action="store_true", help="skip the default post-run trace evidence pass")
    add_model_override_args(run_parser)
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
    export_parser.add_argument("--include-raw", action="store_true", help="include raw responses, judge responses, and event logs")
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
    campaign_parser.add_argument("--max-concurrency", type=int, help="bounded task-level concurrency for each child run")
    campaign_parser.add_argument("--retries", type=int, help="retry transient live provider failures this many times")
    campaign_parser.add_argument("--retry-backoff", type=float, help="initial retry backoff seconds for live provider failures")
    campaign_parser.add_argument("--require-go", action="store_true", help="return exit code 2 unless the campaign overall decision is GO")
    campaign_parser.add_argument("--skip-trace-evaluation", action="store_true", help="skip the default post-run trace evidence pass for child runs")
    add_model_override_args(campaign_parser)
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
    campaign_export_parser.add_argument("--include-raw", action="store_true", help="include raw responses, judge responses, and event logs")
    campaign_export_parser.set_defaults(func=campaign_export)

    campaign_retest_parser = sub.add_parser("campaign-retest", help="create an explicit retest campaign for a RETEST outcome")
    campaign_retest_parser.add_argument("--campaign-id", required=True, help="source campaign id")
    campaign_retest_parser.add_argument("--new-campaign-id", help="explicit retest campaign id")
    campaign_retest_parser.add_argument("--job", help="override job name or path; defaults to source campaign job")
    campaign_retest_parser.add_argument("--providers", type=Path, help="providers config path")
    campaign_retest_parser.add_argument("--runs-dir", type=Path, help="runs directory")
    campaign_retest_parser.add_argument("--campaigns-dir", type=Path, help="campaigns directory")
    campaign_retest_parser.add_argument("--repeat", type=int, default=1, help="number of retest child runs")
    campaign_retest_parser.add_argument("--live", action="store_true", help="force live provider calls")
    campaign_retest_parser.add_argument("--dry-run", action="store_true", help="force dry-run retest even if the source campaign was live")
    campaign_retest_parser.add_argument("--force", action="store_true", help="allow retesting PASS or FAIL campaigns")
    campaign_retest_parser.add_argument("--timeout", type=float, help="per-request timeout seconds")
    campaign_retest_parser.add_argument("--tested-max-tokens", type=int)
    campaign_retest_parser.add_argument("--judge-max-tokens", type=int)
    campaign_retest_parser.add_argument("--max-concurrency", type=int, help="bounded task-level concurrency for each child run; defaults to 1")
    campaign_retest_parser.add_argument("--retries", type=int, help="retry transient live provider failures this many times")
    campaign_retest_parser.add_argument("--retry-backoff", type=float, help="initial retry backoff seconds for live provider failures")
    campaign_retest_parser.add_argument("--skip-trace-evaluation", action="store_true", help="skip the default post-run trace evidence pass for child runs")
    add_model_override_args(campaign_retest_parser)
    campaign_retest_parser.set_defaults(func=campaign_retest)

    authenticity_parser = sub.add_parser("authenticity", help="run or refresh provider authenticity evidence for a campaign")
    authenticity_parser.add_argument("--job", default=DEFAULT_JOB, help="job name or path")
    authenticity_parser.add_argument("--providers", type=Path, help="providers config path")
    authenticity_parser.add_argument("--runs-dir", type=Path, help="runs directory")
    authenticity_parser.add_argument("--campaigns-dir", type=Path, help="campaigns directory")
    authenticity_parser.add_argument("--campaign-id", help="existing or new campaign id")
    authenticity_parser.add_argument("--repeat", type=int, default=1)
    authenticity_parser.add_argument("--baseline-campaign-id", help="optional official/direct baseline campaign id")
    authenticity_parser.add_argument("--baseline-provider", default="official_baseline", help="baseline provider label for evidence")
    authenticity_parser.add_argument("--gateway-provider", default="gateway_candidate", help="gateway provider label for evidence")
    authenticity_parser.add_argument("--live", action="store_true", help="call configured live providers if a new campaign must be created")
    authenticity_parser.add_argument("--timeout", type=float, help="per-request timeout seconds")
    authenticity_parser.add_argument("--tested-max-tokens", type=int)
    authenticity_parser.add_argument("--judge-max-tokens", type=int)
    authenticity_parser.add_argument("--max-concurrency", type=int)
    authenticity_parser.add_argument("--retries", type=int)
    authenticity_parser.add_argument("--retry-backoff", type=float)
    authenticity_parser.add_argument("--skip-trace-evaluation", action="store_true")
    add_model_override_args(authenticity_parser)
    authenticity_parser.set_defaults(func=authenticity)

    fingerprint_parser = sub.add_parser("fingerprint", help="emit protocol fingerprint evidence for a configured provider")
    fingerprint_parser.add_argument("--provider", choices=["tested_model", "judge_model"], default="tested_model")
    fingerprint_parser.add_argument("--providers", type=Path, help="providers config path")
    fingerprint_parser.add_argument("--campaign-id", help="optional campaign id to refresh and read persisted fingerprint evidence")
    fingerprint_parser.add_argument("--campaigns-dir", type=Path)
    fingerprint_parser.add_argument("--runs-dir", type=Path)
    fingerprint_parser.add_argument("--live", action="store_true", help="mark fingerprint as live-intended; network probing is not performed by this dry-safe command")
    add_model_override_args(fingerprint_parser)
    fingerprint_parser.set_defaults(func=fingerprint)

    authenticity_inspect_parser = sub.add_parser("authenticity-inspect", help="inspect campaign authenticity evidence")
    authenticity_inspect_parser.add_argument("--campaign-id", required=True)
    authenticity_inspect_parser.add_argument("--campaigns-dir", type=Path)
    authenticity_inspect_parser.add_argument("--runs-dir", type=Path)
    authenticity_inspect_parser.set_defaults(func=authenticity_inspect)

    authenticity_export_parser = sub.add_parser("authenticity-export", help="refresh authenticity evidence and export a campaign acceptance pack")
    authenticity_export_parser.add_argument("--campaign-id", required=True)
    authenticity_export_parser.add_argument("--campaigns-dir", type=Path)
    authenticity_export_parser.add_argument("--runs-dir", type=Path)
    authenticity_export_parser.add_argument("--baseline-campaign-id")
    authenticity_export_parser.add_argument("--baseline-provider", default="official_baseline")
    authenticity_export_parser.add_argument("--gateway-provider", default="gateway_candidate")
    authenticity_export_parser.add_argument("--include-raw", action="store_true", help="include raw responses, judge responses, and event logs")
    authenticity_export_parser.set_defaults(func=authenticity_export)

    probe_parser = sub.add_parser("probe", help="probe model/protocol/auth combinations")
    probe_parser.add_argument("--providers", type=Path, help="providers config path")
    probe_parser.add_argument("--model", action="append", default=[], help="candidate model id; repeatable")
    probe_parser.add_argument("--max-models", type=int, default=20)
    probe_parser.add_argument("--timeout", type=float, default=20)
    probe_parser.add_argument("--failure-sample", type=int, default=12)
    probe_parser.add_argument("--stop-after-success", action="store_true")
    probe_parser.set_defaults(func=probe)

    baseline_build_parser = sub.add_parser("baseline", help="build a trusted official Claude fingerprint baseline (dry-run by default; --live needs authorization)")
    baseline_build_parser.add_argument("--providers", type=Path, help="providers config path")
    baseline_build_parser.add_argument("--provider", default="tested_model", help="trusted-source role: tested_model or judge_model")
    baseline_build_parser.add_argument("--baseline-id", help="baseline id (default: timestamped BASE-...)")
    baseline_build_parser.add_argument("--baselines-dir", type=Path, help="baselines output dir")
    baseline_build_parser.add_argument("--samples", type=int, default=2, help="samples per canary probe")
    baseline_build_parser.add_argument("--live", action="store_true", help="REAL API calls to the trusted source (cost). Default off = dry-run placeholder")
    baseline_build_parser.add_argument("--note", help="optional note recorded with this baseline version")
    baseline_build_parser.add_argument("--no-version", action="store_true", help="legacy mode: overwrite baseline.json without keeping a version lineage")
    baseline_build_parser.set_defaults(func=baseline_build)

    baseline_inspect_parser = sub.add_parser("baseline-inspect", help="inspect a stored baseline")
    baseline_inspect_parser.add_argument("--baseline-id", required=True)
    baseline_inspect_parser.add_argument("--baselines-dir", type=Path)
    baseline_inspect_parser.set_defaults(func=baseline_inspect)

    baseline_versions_parser = sub.add_parser("baseline-versions", help="list the version lineage of a baseline (timestamps, drift, dedup count)")
    baseline_versions_parser.add_argument("--baseline-id", required=True)
    baseline_versions_parser.add_argument("--baselines-dir", type=Path)
    baseline_versions_parser.set_defaults(func=baseline_versions)

    baseline_diff_parser = sub.add_parser("baseline-diff", help="diff two baseline versions (protocol + behavior drift). Defaults to prev→latest")
    baseline_diff_parser.add_argument("--baseline-id", required=True)
    baseline_diff_parser.add_argument("--baselines-dir", type=Path)
    baseline_diff_parser.add_argument("--from-version", help="older version label (e.g. v0001); default = the previous version")
    baseline_diff_parser.add_argument("--to-version", help="newer version label or 'latest' (default)")
    baseline_diff_parser.set_defaults(func=baseline_diff)

    judge_calib_parser = sub.add_parser("judge-calibrate", help="calibrate the judge model against an authored golden-set (--live makes real judge calls)")
    judge_calib_parser.add_argument("--golden-set", required=True, type=Path, help="path to a judge golden-set JSON (see judge_calibration --emit-sample)")
    judge_calib_parser.add_argument("--providers", type=Path, help="providers config path")
    judge_calib_parser.add_argument("--provider", default="judge_model", help="role to use as the judge (default judge_model)")
    judge_calib_parser.add_argument("--judge-max-tokens", type=int, default=512)
    judge_calib_parser.add_argument("--min-scored", type=int, default=4, help="minimum scored cases before a non-insufficient verdict")
    judge_calib_parser.add_argument("--retries", type=int, default=0)
    judge_calib_parser.add_argument("--retry-backoff", type=float, default=0.0)
    judge_calib_parser.add_argument("--request-delay", type=float, default=0.0, help="seconds between judge calls (avoid rate limits)")
    judge_calib_parser.add_argument("--timeout", type=float, default=120.0)
    judge_calib_parser.add_argument("--out-dir", type=Path, help="where to write events + last_calibration.json")
    judge_calib_parser.add_argument("--live", action="store_true", help="REAL judge API calls (cost). Default off = dry-run wiring smoke")
    judge_calib_parser.add_argument("--report", action="store_true", help="print a human-readable Chinese report instead of JSON")
    judge_calib_parser.add_argument("--write", action="store_true", help="persist last_calibration.json to out-dir")
    judge_calib_parser.set_defaults(func=judge_calibrate)

    baseline_derive_parser = sub.add_parser("baseline-derive-windows", help="derive token_count_check windows from a trusted live baseline (no offline tokenizer needed)")
    baseline_derive_parser.add_argument("--baseline-id", required=True)
    baseline_derive_parser.add_argument("--baselines-dir", type=Path)
    baseline_derive_parser.add_argument("--long-probe", default="canary_mixed")
    baseline_derive_parser.add_argument("--short-probe", default="canary_zh")
    baseline_derive_parser.add_argument("--write", action="store_true", help="write token_probe_windows.json into the baseline dir")
    baseline_derive_parser.set_defaults(func=baseline_derive_windows)

    baseline_compare_parser = sub.add_parser("baseline-compare", help="compare a suspect provider against a trusted baseline")
    baseline_compare_parser.add_argument("--baseline-id", required=True)
    baseline_compare_parser.add_argument("--providers", type=Path, help="providers config path")
    baseline_compare_parser.add_argument("--provider", default="tested_model", help="suspect role to verify")
    baseline_compare_parser.add_argument("--baselines-dir", type=Path)
    baseline_compare_parser.add_argument("--samples", type=int, default=2)
    baseline_compare_parser.add_argument("--live", action="store_true", help="REAL API calls to the suspect provider (cost)")
    baseline_compare_parser.add_argument("--report", action="store_true", help="print a human-readable Chinese report instead of JSON")
    baseline_compare_parser.set_defaults(func=baseline_compare)

    verify_parser = sub.add_parser("verify-endpoint", help="one-shot: verify a suspect provider against a trusted baseline, print a human-readable verdict")
    verify_parser.add_argument("--baseline-id", required=True, help="trusted baseline to compare against")
    verify_parser.add_argument("--providers", type=Path)
    verify_parser.add_argument("--provider", default="tested_model", help="suspect role to verify")
    verify_parser.add_argument("--baselines-dir", type=Path)
    verify_parser.add_argument("--samples", type=int, default=5)
    verify_parser.add_argument("--live", action="store_true", help="REAL API calls to the suspect (cost)")
    verify_parser.add_argument("--json", action="store_true", help="also print raw JSON verdict")
    verify_parser.add_argument("--with-sse", action="store_true", help="also run the SSE event-order probe (extra live request)")
    verify_parser.add_argument("--with-error-envelope", action="store_true", help="also run the error-envelope probe (malformed requests)")
    verify_parser.add_argument("--with-needle", action="store_true", help="also run the long-context needle probe (~120K request, slow/expensive)")
    verify_parser.add_argument("--needle-tokens", type=int, default=120000, help="needle target prompt size in tokens")
    verify_parser.add_argument("--with-capability", action="store_true", help="also run the capability-anchor probe (silent-downgrade detector; needs a baseline capability_anchor.json)")
    verify_parser.add_argument("--with-variance", action="store_true", help="also run the consistency-variance probe (repeat one anchor N times; low-frequency-swap detector)")
    verify_parser.add_argument("--variance-repeats", type=int, default=12, help="repeats for the variance probe (default 12)")
    verify_parser.add_argument("--with-identity", action="store_true", help="also run the identity-coherence probe (cross-check self-narrated model id vs envelope: returned model field + response id prefix; one cheap live request)")
    verify_parser.add_argument("--capability-items", type=Path, help="capability anchor item set (default judge_golden/capability_anchors_v1.json)")
    verify_parser.add_argument("--request-delay", type=float, default=0.0, help="seconds to wait between probe requests (avoid upstream rate limits)")
    verify_parser.add_argument("--retries", type=int, default=1, help="retries per request on transient failure (429/5xx)")
    verify_parser.add_argument("--retry-backoff", type=float, default=0.5, help="base backoff seconds between retries")
    verify_parser.set_defaults(func=verify_endpoint)

    capability_parser = sub.add_parser("capability-probe", help="probe a provider's capability pass-rate on hard anchors (silent-downgrade detector); writes capability_anchor.json")
    capability_parser.add_argument("--items", type=Path, default=ROOT / "judge_golden" / "capability_anchors_v1.json", help="capability anchor item set")
    capability_parser.add_argument("--providers", type=Path)
    capability_parser.add_argument("--provider", default="tested_model", help="role to probe (trusted source for baseline; suspect otherwise)")
    capability_parser.add_argument("--baseline-id", help="write the result into this baseline dir (its capability_anchor.json becomes the reference rate)")
    capability_parser.add_argument("--baselines-dir", type=Path)
    capability_parser.add_argument("--max-tokens", type=int, default=256)
    capability_parser.add_argument("--request-delay", type=float, default=0.0)
    capability_parser.add_argument("--retries", type=int, default=1)
    capability_parser.add_argument("--retry-backoff", type=float, default=0.5)
    capability_parser.add_argument("--timeout", type=float, default=120.0)
    capability_parser.add_argument("--live", action="store_true", help="REAL API calls (cost). Default off = dry-run wiring smoke")
    capability_parser.set_defaults(func=capability_probe)

    error_env_parser = sub.add_parser("error-envelope", help="#8 probe: send malformed requests, classify error-body dialect (anthropic/openai/generic)")
    error_env_parser.add_argument("--providers", type=Path)
    error_env_parser.add_argument("--provider", default="tested_model")
    error_env_parser.add_argument("--live", action="store_true", help="REAL malformed requests (small cost). Default off = dry-run")
    error_env_parser.set_defaults(func=error_envelope)

    sse_parser = sub.add_parser("sse-fingerprint", help="#9 probe: open one streaming request, classify SSE event-order (claude_sse vs openai_sse)")
    sse_parser.add_argument("--providers", type=Path)
    sse_parser.add_argument("--provider", default="tested_model")
    sse_parser.add_argument("--live", action="store_true", help="REAL streaming request (small cost). Default off = dry-run")
    sse_parser.set_defaults(func=sse_fingerprint)

    needle_parser = sub.add_parser("needle", help="long-context needle probe: plant a needle in a ~120K-token prompt, check recall + silent truncation; --advertised-tokens flags an unproven larger window (--live, expensive)")
    needle_parser.add_argument("--providers", type=Path)
    needle_parser.add_argument("--provider", default="tested_model", help="endpoint to probe")
    needle_parser.add_argument("--target-tokens", type=int, default=120000, help="approx prompt size in tokens actually sent (upstreams reject single requests above ~160-220K)")
    needle_parser.add_argument("--advertised-tokens", type=int, default=None, help="the context window the endpoint CLAIMS (e.g. 1000000); recall at --target-tokens does NOT prove this larger window")
    needle_parser.add_argument("--seed", type=int, default=1, help="reproducible prompt seed")
    needle_parser.add_argument("--baseline-id", help="baseline to read this link's prefix from (for token shortfall)")
    needle_parser.add_argument("--baselines-dir", type=Path)
    needle_parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout seconds (huge prompts are slow)")
    needle_parser.add_argument("--live", action="store_true", help="REAL long-context API call (~120K tokens, expensive). Default off = dry-run assembles prompt only")
    needle_parser.set_defaults(func=needle)

    args = parser.parse_args()
    if args.command in {"inspect", "export"} and not args.latest and not args.job_id:
        parser.error(f"{args.command} requires --latest or --job-id")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
