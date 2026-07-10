from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from acceptance_pack import verify_acceptance_pack
from campaigns import (
    campaign_identity_problem,
    campaign_dir as resolve_campaign_dir,
    campaign_leaderboard,
    campaign_list_payload,
    load_run_index,
    load_summary,
    summary_needs_refresh,
    summarize_campaign,
)
from authenticity import load_or_build_authenticity
from local_env import load_local_env
from redaction import redact_text, redact_value
import eval_cli


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
CAMPAIGNS_DIR = ROOT / "campaigns"
REACT_DIST_DIR = ROOT.parent / "llm_eval_result_site" / "dist"
WEB_DIR = REACT_DIST_DIR if REACT_DIST_DIR.exists() else ROOT / "web"
# The authenticity verify page lives in the plain web/ dir and is served
# independently of WEB_DIR, so it works whether or not the React dist is present.
VERIFY_WEB_DIR = ROOT / "web"
VERIFY_ASSETS = {"/verify", "/verify.html", "/verify.css", "/verify.js", "/vendor/anime.esm.js"}
PROVIDERS_LOCAL = ROOT / "configs" / "providers.local.json"
LOCAL_SECRETS = ROOT / "local_secrets.env"
ALLOWED_PROTOCOLS = {"openai_chat", "anthropic_messages"}
ALLOWED_AUTH_TYPES = {"bearer", "x-api-key"}
ALLOWED_REASONING_EFFORTS = {"", "none", "low", "medium", "high", "xhigh"}
REASONING_PROBE_VALUES = ["none", "minimal", "low", "medium", "high", "xhigh"]
TEXT_MODEL_HINTS = (
    "gpt",
    "claude",
    "gemini",
    "opus",
    "sonnet",
    "haiku",
    "llama",
    "qwen",
    "deepseek",
    "mistral",
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _as_dict(value):
    """Narrow Any -> dict (empty when not a dict) for the type checker."""
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_config_value(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("authorization", "credential", "key", "password", "secret", "token")):
                out[str(key)] = "[REDACTED]"
            else:
                out[str(key)] = sanitize_config_value(item)
        return out
    if isinstance(value, list):
        return [sanitize_config_value(item) for item in value]
    return value


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def safe_run_dir(job_id: str) -> Path:
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        raise ValueError("invalid job id")
    path = RUNS_DIR / job_id
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(job_id)
    return path


def list_jobs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    jobs = []
    for path in RUNS_DIR.iterdir():
        state_path = path / "state.json"
        if not path.is_dir() or not state_path.exists():
            continue
        try:
            state = read_json(state_path)
        except Exception:
            continue
        jobs.append(
            {
                "job_id": state.get("job_id") or path.name,
                "status": state.get("status"),
                "progress": state.get("progress"),
                "final_decision": state.get("final_decision"),
                "started_at": state.get("started_at"),
                "completed_at": state.get("completed_at"),
            }
        )
    return sorted(jobs, key=lambda item: str(item.get("started_at") or item.get("job_id") or ""), reverse=True)


def latest_job() -> dict | None:
    jobs = list_jobs()
    return jobs[0] if jobs else None


def latest_quality_gate(run_dir: Path) -> dict | None:
    gates_dir = run_dir / "quality_gates"
    if not gates_dir.exists():
        return None
    candidates = [path for path in gates_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    gate_dir = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    records = read_jsonl(gate_dir / "quality_gate_records.jsonl")
    manifest_path = gate_dir / "quality_gate_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return {
        "gate_id": gate_dir.name,
        "manifest": manifest,
        "records": records,
        "primary_record": records[0] if records else None,
    }


def score_value(record: dict) -> float | None:
    scoring = record.get("scoring") or {}
    final_score = _as_dict(scoring.get("final_score"))
    value = final_score.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def provider_records(records: list[dict], provider_id: str) -> list[dict]:
    selected = []
    for record in records:
        provider = _as_dict(record.get("provider"))
        if str(provider.get("id") or "") == provider_id:
            selected.append(record)
    return selected or records


def provider_model_name(state: dict, records: list[dict], provider_id: str) -> str:
    tested = ((state.get("models") or {}).get("tested_model") or {})
    if str(tested.get("provider_id") or "") == provider_id and tested.get("model"):
        return str(tested.get("model"))
    for record in records:
        provider = _as_dict(record.get("provider"))
        if str(provider.get("id") or "") != provider_id:
            continue
        for key in ("claimed_model", "model_requested", "model_returned"):
            if provider.get(key):
                return str(provider.get(key))
    return provider_id


def provider_identity(records: list[dict], provider_id: str) -> dict:
    for record in records:
        provider = _as_dict(record.get("provider"))
        if str(provider.get("id") or "") != provider_id:
            continue
        return {
            "provider_display_name": provider.get("provider_display_name") or provider_id,
            "provider_host": provider.get("base_url_host"),
            "source_group": provider.get("leaderboard_group") or "gateway_candidate",
            "baseline_model": provider.get("baseline_model") or provider.get("claimed_model"),
            "provider_channel": provider.get("provider_channel") or "unknown",
        }
    return {
        "provider_display_name": provider_id,
        "provider_host": None,
        "source_group": "gateway_candidate",
        "baseline_model": None,
        "provider_channel": "unknown",
    }


def leaderboard_raw_rows(*, include_dry_run: bool = False) -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    rows: list[dict] = []
    for run_dir in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        state_path = run_dir / "state.json"
        benchmark_path = run_dir / "benchmark_scores.json"
        if not state_path.exists() or not benchmark_path.exists():
            continue
        try:
            state = read_json(state_path)
            benchmark = read_json(benchmark_path)
            records = read_jsonl(run_dir / "run_records.jsonl")
        except Exception:
            continue
        live_provider = state.get("live_provider") is True
        if not include_dry_run and not live_provider:
            continue
        gate = latest_quality_gate(run_dir)
        primary_gate = gate.get("primary_record") if gate else None
        gate_metrics = _as_dict(primary_gate.get("metrics_snapshot") if isinstance(primary_gate, dict) else {})
        providers = _as_dict(benchmark.get("providers"))
        for provider_id, provider_score in providers.items():
            if not isinstance(provider_score, dict):
                continue
            provider_id = str(provider_id)
            selected_records = provider_records(records, provider_id)
            ok_count = 0
            scores: list[float] = []
            latencies: list[float] = []
            for record in selected_records:
                telemetry = _as_dict(record.get("telemetry"))
                if telemetry.get("ok") is True:
                    ok_count += 1
                score = score_value(record)
                if score is not None:
                    scores.append(score)
                latency = to_float(telemetry.get("first_content_token_ms") or telemetry.get("total_ms"))
                if latency is not None:
                    latencies.append(latency)
            total = int(provider_score.get("task_count") or len(selected_records) or 0)
            identity = provider_identity(selected_records, provider_id)
            generated_at = str(
                benchmark.get("generated_at")
                or state.get("completed_at")
                or state.get("started_at")
                or ""
            )
            rows.append(
                {
                    "run_id": state.get("job_id") or run_dir.name,
                    "provider_id": provider_id,
                    "provider_display_name": identity["provider_display_name"],
                    "provider_host": identity["provider_host"],
                    "source_group": identity["source_group"],
                    "provider_channel": identity["provider_channel"],
                    "model": provider_model_name(state, selected_records, provider_id),
                    "baseline_model": identity["baseline_model"],
                    "mode": benchmark.get("benchmark_mode") or provider_score.get("mode") or "custom",
                    "task_count": total,
                    "success_rate": (ok_count / total) if total else None,
                    "average_score_0_10": (sum(scores) / len(scores)) if scores else None,
                    "score": to_float(provider_score.get("benchmark_score")),
                    "benchmark_score": to_float(provider_score.get("benchmark_score")),
                    "quality_score": to_float(provider_score.get("quality_score")),
                    "latency_score": to_float(provider_score.get("latency_score")),
                    "cost_efficiency_score": to_float(provider_score.get("cost_efficiency_score")),
                    "risk_penalty": to_float(provider_score.get("risk_penalty")) or 0.0,
                    "p95_first_content_token_ms": to_float(gate_metrics.get("p95_first_content_token_ms")) or percentile(latencies, 0.95),
                    "gate_decision": (primary_gate.get("decision") if isinstance(primary_gate, dict) else None) or state.get("final_decision"),
                    "status": state.get("status"),
                    "live_provider": live_provider,
                    "generated_at": generated_at,
                    "started_at": state.get("started_at"),
                    "completed_at": state.get("completed_at"),
                }
            )
    return rows


def leaderboard(limit: int = 50, *, include_dry_run: bool = False) -> dict:
    raw_rows = leaderboard_raw_rows(include_dry_run=include_dry_run)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in raw_rows:
        key = (
            str(row.get("provider_id") or ""),
            str(row.get("model") or ""),
            str(row.get("mode") or ""),
        )
        grouped.setdefault(key, []).append(row)

    entries = []
    for history in grouped.values():
        history.sort(key=lambda item: str(item.get("generated_at") or item.get("completed_at") or item.get("started_at") or ""), reverse=True)
        latest = dict(history[0])
        values = [value for value in (to_float(item.get("score")) for item in history) if value is not None]
        average_score = sum(values) / len(values) if values else None
        latest["latest_score"] = latest.get("score")
        latest["score"] = round(average_score, 2) if average_score is not None else None
        latest["average_score"] = latest["score"]
        latest["latest_run_id"] = latest.get("run_id")
        latest["history_count"] = len(history)
        latest["history"] = [
            {
                "run_id": item.get("run_id"),
                "score": item.get("score"),
                "benchmark_score": item.get("benchmark_score"),
                "quality_score": item.get("quality_score"),
                "success_rate": item.get("success_rate"),
                "gate_decision": item.get("gate_decision"),
                "generated_at": item.get("generated_at"),
            }
            for item in history[:10]
        ]
        entries.append(latest)

    entries.sort(
        key=lambda item: (
            item.get("score") is not None,
            item.get("score") or -1.0,
            item.get("success_rate") or -1.0,
            str(item.get("generated_at") or ""),
        ),
        reverse=True,
    )
    for index, entry in enumerate(entries, start=1):
        entry["rank"] = index
    return {
        "entries": entries[:limit],
        "total": len(entries),
        "raw_run_count": len(raw_rows),
        "limit": limit,
        "include_dry_run": include_dry_run,
        "sort": "score desc, success_rate desc, latest run desc",
    }


def summarize_run(run_dir: Path) -> dict:
    state = read_json(run_dir / "state.json")
    run_records = read_jsonl(run_dir / "run_records.jsonl")
    benchmark_path = run_dir / "benchmark_scores.json"
    benchmark = read_json(benchmark_path) if benchmark_path.exists() else {}
    gate = latest_quality_gate(run_dir)
    samples = []
    ok_count = 0
    scores = []
    latencies = []
    for record in run_records:
        task = record.get("task") or {}
        provider = record.get("provider") or {}
        telemetry = record.get("telemetry") or {}
        ok = telemetry.get("ok") is True
        if ok:
            ok_count += 1
        score = score_value(record)
        if score is not None:
            scores.append(score)
        latency = telemetry.get("first_content_token_ms") or telemetry.get("total_ms")
        latency_value = to_float(latency)
        if latency_value is not None:
            latencies.append(latency_value)
        samples.append(
            {
                "task_id": task.get("id"),
                "category": task.get("category"),
                "dimension": task.get("enterprise_dimension"),
                "ok": ok,
                "score": score,
                "error": redact_text(telemetry.get("error"), max_chars=500),
                "latency_ms": latency_value,
                "model_returned": provider.get("model_returned"),
            }
        )
    total = len(run_records)
    success_rate = ok_count / total if total else None
    avg_score = sum(scores) / len(scores) if scores else None
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    primary_gate = gate.get("primary_record") if gate else None
    metrics = _as_dict(primary_gate.get("metrics_snapshot") if isinstance(primary_gate, dict) else {})
    providers = _as_dict(benchmark.get("providers"))
    provider_score = _as_dict(next(iter(providers.values()), {}) if providers else {})
    return {
        "state": state,
        "metrics": {
            "sample_count": total,
            "ok_count": ok_count,
            "failure_count": total - ok_count,
            "success_rate": success_rate,
            "average_score_0_10": avg_score,
            "average_latency_ms": avg_latency,
            "p95_first_content_token_ms": metrics.get("p95_first_content_token_ms"),
            "gate_score": metrics.get("gate_score") or provider_score.get("benchmark_score"),
            "benchmark_score": provider_score.get("benchmark_score"),
            "quality_score": provider_score.get("quality_score"),
            "latency_score": provider_score.get("latency_score"),
            "cost_efficiency_score": provider_score.get("cost_efficiency_score"),
        },
        "quality_gate": {
            "gate_id": gate.get("gate_id") if gate else None,
            "decision": primary_gate.get("decision") if isinstance(primary_gate, dict) else state.get("final_decision"),
            "blockers": primary_gate.get("blockers") if isinstance(primary_gate, dict) else [],
            "review_items": primary_gate.get("review_items") if isinstance(primary_gate, dict) else [],
            "passed_rules": primary_gate.get("passed_rules") if isinstance(primary_gate, dict) else [],
        },
        "samples": samples,
        "benchmark": benchmark,
    }


def sanitized_config() -> dict:
    load_local_env()
    if not PROVIDERS_LOCAL.exists():
        return {"exists": False, "providers": None}
    data = read_json(PROVIDERS_LOCAL)
    out: dict[str, Any] = {"exists": True, "providers": {}}
    for label in ("tested_model", "judge_model"):
        item = data.get(label) or {}
        env_name = str(item.get("api_key_env") or "")
        extra_body = _as_dict(item.get("extra_body"))
        out["providers"][label] = {
            "provider_id": item.get("provider_id"),
            "base_url": item.get("base_url"),
            "model": item.get("model"),
            "protocol": item.get("protocol"),
            "auth_type": item.get("auth_type") or "bearer",
            "api_key_env": env_name,
            "api_key_present": bool(os.environ.get(env_name)),
            "reasoning_effort": extra_body.get("reasoning_effort") or "",
            "extra_body": sanitize_config_value(extra_body),
        }
    return out


def update_env_file(updates: dict[str, str]) -> None:
    existing: dict[str, str] = {}
    if LOCAL_SECRETS.exists():
        for line in LOCAL_SECRETS.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            existing[key.strip()] = value.strip()
    for key, value in updates.items():
        if value:
            existing[key] = value
    lines = [f"{key}={value}" for key, value in sorted(existing.items())]
    LOCAL_SECRETS.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    load_local_env(override=True)


def save_config(payload: dict) -> dict:
    providers = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(providers, dict):
        raise ValueError("providers object is required")
    current = read_json(PROVIDERS_LOCAL) if PROVIDERS_LOCAL.exists() else {}
    env_updates: dict[str, str] = {}
    for label, env_name in (("tested_model", "TESTED_MODEL_API_KEY"), ("judge_model", "JUDGE_MODEL_API_KEY")):
        item = providers.get(label)
        if not isinstance(item, dict):
            raise ValueError(f"{label} is required")
        protocol = str(item["protocol"])
        auth_type = str(item.get("auth_type") or "bearer")
        if protocol not in ALLOWED_PROTOCOLS:
            raise ValueError(f"{label}.protocol must be one of: {', '.join(sorted(ALLOWED_PROTOCOLS))}")
        if auth_type not in ALLOWED_AUTH_TYPES:
            raise ValueError(f"{label}.auth_type must be one of: {', '.join(sorted(ALLOWED_AUTH_TYPES))}")
        existing = current.get(label, {}) if isinstance(current.get(label), dict) else {}
        extra_body = dict(existing.get("extra_body") or {}) if isinstance(existing.get("extra_body"), dict) else {}
        if isinstance(item.get("extra_body"), dict):
            extra_body.update(item["extra_body"])
        reasoning_effort = str(item.get("reasoning_effort") or "").strip()
        if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
            raise ValueError(f"{label}.reasoning_effort must be one of: none, low, medium, high, xhigh")
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort
        else:
            extra_body.pop("reasoning_effort", None)
        current[label] = {
            "provider_id": str(item.get("provider_id") or existing.get("provider_id") or label),
            "base_url": str(item["base_url"]).rstrip("/"),
            "model": str(item["model"]),
            "api_key_env": env_name,
            "protocol": protocol,
            "auth_type": auth_type,
        }
        if extra_body:
            current[label]["extra_body"] = extra_body
        api_key = str(item.get("api_key") or "")
        if api_key:
            env_updates[env_name] = api_key
    write_json(PROVIDERS_LOCAL, current)
    if env_updates:
        update_env_file(env_updates)
    return sanitized_config()


# Server-side floor on the inter-request delay for web-initiated live probes.
# R-001: an account was banned by rapid live probing. The web path NEVER lets a
# caller probe faster than this, regardless of what the body asks for.
WEB_VERIFY_MIN_DELAY = 2.0
# Ephemeral env var name the supplied key is bound to only for the duration of a
# single verify call. Popped in a finally — never persisted to disk.
WEB_VERIFY_KEY_ENV = "WEB_VERIFY_API_KEY"
DEFAULT_BASELINE_ID = "OFFICIAL-CLAUDE-OPUS46"


def run_web_verify(payload: dict, *, live: bool, progress=None) -> dict:
    """Run an authenticity verify from a web request and return a verdict dict.

    Key handling (R-001 iron rule): the suspect key is placed in os.environ only
    for the call and popped in a finally. It is NEVER written to
    providers.local.json or local_secrets.env. The web path also disables the
    dangerous probes (needle fake-1M, malformed error-envelope) and floors the
    request delay — see WEB_VERIFY_MIN_DELAY.
    """
    base_url = str(payload.get("base_url") or "").strip().rstrip("/")
    model_name = str(payload.get("model") or "").strip()
    protocol = str(payload.get("protocol") or "anthropic_messages").strip()
    auth_type = str(payload.get("auth_type") or "x-api-key").strip()
    baseline_id = str(payload.get("baseline_id") or DEFAULT_BASELINE_ID).strip()
    api_key = str(payload.get("api_key") or "")
    with_capability = bool(payload.get("with_capability"))
    with_variance = bool(payload.get("with_variance"))
    with_identity = bool(payload.get("with_identity"))
    if not base_url or not model_name:
        raise ValueError("base_url 和 model 是必填项")
    if protocol not in ALLOWED_PROTOCOLS:
        raise ValueError(f"protocol 必须是 {', '.join(sorted(ALLOWED_PROTOCOLS))} 之一")
    if auth_type not in ALLOWED_AUTH_TYPES:
        raise ValueError(f"auth_type 必须是 {', '.join(sorted(ALLOWED_AUTH_TYPES))} 之一")
    if live and not api_key:
        raise ValueError("live 检测需要提供 api_key")

    baselines_dir = eval_cli.resolve_path(eval_cli.DEFAULT_BASELINES_DIR)
    baseline = eval_cli.load_baseline(baselines_dir, baseline_id)
    if baseline is None:
        raise FileNotFoundError(f"找不到基线 {baseline_id}（请先用可信官方源建立基线）")

    # request_delay floored server-side for LIVE only (R-001). dry-run makes no
    # real requests, so there's nothing to rate-limit — skip the sleeps entirely.
    req_delay = max(WEB_VERIFY_MIN_DELAY, float(payload.get("request_delay") or 0.0)) if live else 0.0

    model = eval_cli.ModelConfig(
        provider_id="web_suspect",
        base_url=base_url,
        model=model_name,
        api_key_env=WEB_VERIFY_KEY_ENV,
        protocol=protocol,
        auth_type=auth_type,
        provider_channel="gateway",
        provider_display_name="web_suspect",
        # Bind the request's key onto this request-scoped ModelConfig instead of a
        # process-global env var. ThreadingHTTPServer serves each request on its
        # own thread; a shared os.environ slot would let concurrent checks clobber
        # each other's key. This instance is local to this request/thread.
        secret_override=(api_key if (live and api_key) else None),
    )
    return _invoke_verify_core(model, baseline, baseline_id, baselines_dir,
                              live=live, api_key=api_key, req_delay=req_delay,
                              with_capability=with_capability, with_variance=with_variance,
                              with_identity=with_identity,
                              progress=progress)


def _invoke_verify_core(model, baseline, baseline_id, baselines_dir, *,
                        live: bool, api_key: str, req_delay: float,
                        with_capability: bool, with_variance: bool = False,
                        with_identity: bool = False, progress=None) -> dict:
    """Run verify_core. The request key rides on model.secret_override (bound by
    the caller), so there is no shared os.environ mutation and no cross-request
    key-isolation race between concurrent threads."""
    verdict = eval_cli.verify_core(
        model, baseline,
        role="web_suspect",
        baselines_dir=baselines_dir,
        baseline_id=baseline_id,
        live=live,
        samples_per_probe=5,
        request_delay=req_delay,
        retries=1,
        retry_backoff=2.0,
        with_sse=False,            # R-001: extra live request, keep web minimal
        with_error_envelope=False, # R-001: malformed requests look like an attack
        with_needle=False,         # R-001: huge request, most dangerous
        with_capability=with_capability,
        providers_path=None,
        with_variance=with_variance,
        with_identity=with_identity,
        variance_repeats=12,
        progress=progress,
    )

    report_text = eval_cli.render_verdict_report(verdict, baseline=baseline)
    safe_verdict = redact_value(verdict)
    return {
        "live": live,
        "baseline_id": baseline_id,
        "verdict": safe_verdict,
        "report_text": redact_text(report_text, max_chars=4000),
        "note": None if live else "dry-run：仅验证配置与管线，无真伪判定意义。勾选风险确认并以 --enable-live-verify 启动后可跑真实检测。",
    }


def authenticity_meta() -> dict:
    """Read-only metadata for the verify page: available baselines + the
    suspect_model config (REDACTED — never the key) so the UI can prefill and
    offer a baseline dropdown instead of free-text. Pure filesystem reads."""
    baselines_dir = eval_cli.resolve_path(eval_cli.DEFAULT_BASELINES_DIR)
    baselines: list[dict] = []
    if baselines_dir.exists():
        for d in sorted(baselines_dir.iterdir()):
            # skip transient/internal dirs (_verify, _CALIB, _judge_calibration…)
            if not d.is_dir() or d.name.startswith("_"):
                continue
            bfile = d / "baseline.json"
            if not bfile.is_file():
                continue
            entry = {"id": d.name, "has_capability": (d / "capability_anchor.json").is_file()}
            try:
                doc = read_json(bfile)
                src = doc.get("source") or {}
                entry["model"] = src.get("model")
                entry["host"] = src.get("base_url_host")
                entry["live"] = doc.get("evidence_status") == "live_observed"
            except Exception:
                pass
            baselines.append(entry)
    # suspect_model prefill (redacted; key is referenced by env name only)
    suspect = None
    if PROVIDERS_LOCAL.exists():
        try:
            load_local_env()
            data = read_json(PROVIDERS_LOCAL)
            item = data.get("suspect_model")
            if isinstance(item, dict):
                suspect = {
                    "base_url": item.get("base_url"),
                    "model": item.get("model"),
                    "protocol": item.get("protocol"),
                    "auth_type": item.get("auth_type") or "x-api-key",
                    "api_key_env": item.get("api_key_env"),
                    "api_key_present": bool(os.environ.get(str(item.get("api_key_env") or ""))),
                }
        except Exception:
            pass
    return {"baselines": baselines, "suspect_model": suspect,
            "default_baseline": DEFAULT_BASELINE_ID}


def model_ids_from_payload(data) -> list[str]:
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


def is_text_model(model_id: str) -> bool:
    lowered = model_id.lower()
    if "image" in lowered or "embedding" in lowered or "moderation" in lowered or "tts" in lowered:
        return False
    return any(hint in lowered for hint in TEXT_MODEL_HINTS)


def provider_auth_headers(item: dict, secret: str) -> dict[str, str]:
    auth_type = str(item.get("auth_type") or "bearer")
    if auth_type == "bearer":
        return {"Authorization": f"Bearer {secret}"}
    if auth_type == "x-api-key":
        return {"x-api-key": secret}
    raise ValueError(f"unsupported auth_type: {auth_type}")


def http_json(method: str, url: str, *, headers: dict[str, str], payload: dict | None = None, timeout: float = 30.0) -> tuple[int, dict, float]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = dict(headers)
    if payload is not None:
        request_headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            data = {"body_preview": raw[:400]}
        return int(response.status), data, elapsed_ms


def probe_reasoning_efforts(item: dict, secret: str, model_id: str) -> dict:
    if str(item.get("protocol") or "") != "openai_chat":
        return {"supported": [], "rejected": [], "skipped": "reasoning_effort probe only supports openai_chat"}
    base_url = str(item.get("base_url") or "").rstrip("/")
    headers = provider_auth_headers(item, secret)
    supported: list[dict] = []
    rejected: list[dict] = []
    for effort in REASONING_PROBE_VALUES:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply exactly OK."}],
            "max_tokens": 32,
            "temperature": 0,
            "reasoning_effort": effort,
        }
        try:
            status, data, elapsed_ms = http_json("POST", f"{base_url}/v1/chat/completions", headers=headers, payload=payload, timeout=90.0)
            choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
            message = _as_dict(choice.get("message"))
            supported.append(
                {
                    "value": effort,
                    "status": status,
                    "elapsed_ms": elapsed_ms,
                    "model_returned": data.get("model") if isinstance(data, dict) else None,
                    "output_tokens": ((data.get("usage") or {}).get("completion_tokens") if isinstance(data, dict) else None),
                    "content_preview": str(message.get("content") or choice.get("text") or "")[:80],
                }
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            rejected.append(
                {
                    "value": effort,
                    "status": exc.code,
                    "error": redact_text(body, max_chars=260),
                }
            )
        except Exception as exc:
            rejected.append({"value": effort, "status": None, "error": redact_text(f"{type(exc).__name__}: {exc}", max_chars=260)})
    return {
        "probe_model": model_id,
        "supported": supported,
        "supported_values": [item["value"] for item in supported],
        "rejected": rejected,
    }


def probe_config_role(role: str, *, include_reasoning: bool = True) -> dict:
    if role not in {"tested_model", "judge_model"}:
        raise ValueError("role must be tested_model or judge_model")
    load_local_env()
    if not PROVIDERS_LOCAL.exists():
        raise FileNotFoundError("providers.local.json")
    data = read_json(PROVIDERS_LOCAL)
    item = data.get(role)
    if not isinstance(item, dict):
        raise ValueError(f"{role} is not configured")
    env_name = str(item.get("api_key_env") or "")
    secret = os.environ.get(env_name)
    if not secret:
        return {
            "role": role,
            "provider_id": item.get("provider_id"),
            "base_url_host": urlparse(str(item.get("base_url") or "")).netloc,
            "api_key_env": env_name,
            "api_key_present": False,
            "error": f"missing environment variable {env_name}",
        }
    base_url = str(item.get("base_url") or "").rstrip("/")
    headers = provider_auth_headers(item, secret)
    try:
        status, payload, elapsed_ms = http_json("GET", f"{base_url}/v1/models", headers=headers, timeout=30.0)
        models = model_ids_from_payload(payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "role": role,
            "provider_id": item.get("provider_id"),
            "base_url_host": urlparse(base_url).netloc,
            "api_key_env": env_name,
            "api_key_present": True,
            "models_ok": False,
            "models_status": exc.code,
            "error": redact_text(body, max_chars=500),
        }
    text_models = [model for model in models if is_text_model(model)]
    result = {
        "role": role,
        "provider_id": item.get("provider_id"),
        "base_url_host": urlparse(base_url).netloc,
        "api_key_env": env_name,
        "api_key_present": True,
        "protocol": item.get("protocol"),
        "auth_type": item.get("auth_type") or "bearer",
        "configured_model": item.get("model"),
        "configured_reasoning_effort": ((item.get("extra_body") or {}).get("reasoning_effort") if isinstance(item.get("extra_body"), dict) else ""),
        "models_ok": status == 200,
        "models_status": status,
        "models_elapsed_ms": elapsed_ms,
        "model_count": len(models),
        "models": models,
        "text_models": text_models,
    }
    configured_model = str(item.get("model") or "")
    probe_model = configured_model if configured_model else (text_models[0] if text_models else "")
    if include_reasoning and probe_model:
        result["reasoning_probe"] = probe_reasoning_efforts(item, secret, probe_model)
    return result


def query_bool(qs: dict[str, list[str]], name: str, default: bool = False) -> bool:
    if name not in qs:
        return default
    return str((qs.get(name) or [str(default)])[0]).lower() in {"1", "true", "yes", "on"}


def artifact_listing(root: Path) -> list[dict]:
    artifacts: list[dict] = []
    if not root.exists():
        return artifacts
    for item in root.iterdir():
        if not item.is_file():
            continue
        row = {"name": item.name, "bytes": item.stat().st_size}
        if item.name == "acceptance_pack.zip":
            row["verification"] = verify_acceptance_pack(item)
        artifacts.append(row)
    return artifacts


class Handler(BaseHTTPRequestHandler):
    server_version = "EvalAutomationAPI/0.2.3"

    def send_json(self, value, status: int = 200) -> None:
        body = json.dumps(redact_value(value), ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/config/probe":
                qs = parse_qs(parsed.query)
                role = (qs.get("role") or ["judge_model"])[0]
                include_reasoning = query_bool(qs, "reasoning", default=True)
                self.send_json(probe_config_role(role, include_reasoning=include_reasoning))
            elif path == "/api/config":
                self.send_json(sanitized_config())
            elif path == "/api/authenticity/meta":
                self.send_json(authenticity_meta())
            elif path == "/api/leaderboard":
                qs = parse_qs(parsed.query)
                raw_limit = (qs.get("limit") or ["50"])[0]
                try:
                    limit = min(max(int(raw_limit), 1), 200)
                except ValueError:
                    limit = 50
                live_filter = None
                if "live_provider" in qs:
                    live_filter = query_bool(qs, "live_provider")
                min_samples_raw = (qs.get("min_samples") or ["1"])[0]
                try:
                    min_samples = max(int(min_samples_raw), 1)
                except ValueError:
                    min_samples = 1
                self.send_json(
                    campaign_leaderboard(
                        CAMPAIGNS_DIR,
                        RUNS_DIR,
                        include_dry_run=query_bool(qs, "include_dry_run"),
                        benchmark_version=(qs.get("benchmark_version") or [""])[0],
                        judge_model=(qs.get("judge_model") or [""])[0],
                        quality_gate_version=(qs.get("quality_gate_version") or [""])[0],
                        live_provider=live_filter,
                        date_from=(qs.get("date_from") or [""])[0],
                        date_to=(qs.get("date_to") or [""])[0],
                        min_samples=min_samples,
                        limit=limit,
                        persist_refresh=False,
                    )
                )
            elif path == "/api/campaigns":
                self.send_json(campaign_list_payload(CAMPAIGNS_DIR, RUNS_DIR, persist_refresh=False))
            elif path == "/api/campaigns/latest":
                campaigns = campaign_list_payload(CAMPAIGNS_DIR, RUNS_DIR, persist_refresh=False).get("campaigns") or []
                self.send_json(campaigns[0] if campaigns else {})
            elif path == "/api/authenticity/latest":
                campaigns = campaign_list_payload(CAMPAIGNS_DIR, RUNS_DIR, persist_refresh=False).get("campaigns") or []
                if not campaigns:
                    self.send_json({})
                else:
                    camp_dir = resolve_campaign_dir(CAMPAIGNS_DIR, str(campaigns[0].get("campaign_id") or ""))
                    self.send_json(load_or_build_authenticity(camp_dir, RUNS_DIR, persist=False))
            elif path.startswith("/api/campaigns/"):
                self.handle_campaign_get(path)
            elif path == "/api/jobs":
                self.send_json({"jobs": list_jobs()})
            elif path == "/api/jobs/latest":
                job = latest_job()
                self.send_json(job or {})
            elif path.startswith("/api/jobs/"):
                self.handle_job_get(path)
            else:
                if path.startswith("/api/"):
                    self.send_error_json(HTTPStatus.NOT_FOUND, "unknown api endpoint")
                    return
                self.serve_static(path)
        except FileNotFoundError:
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        # Always drain the request body first, BEFORE any early return. Leaving an
        # unread body on the socket makes the connection state ambiguous: the client's
        # urllib intermittently sees RemoteDisconnected/ConnectionReset instead of a
        # clean HTTP response, which surfaced as a flaky 403 test. Read once, up front.
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if parsed.path == "/api/authenticity/verify":
            self._handle_verify_post(raw)
            return
        if parsed.path != "/api/config":
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        if not getattr(self.server, "config_write_enabled", False):
            self.send_error_json(HTTPStatus.FORBIDDEN, "config writes are disabled; restart with --enable-config-write to allow this endpoint")
            return
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            self.send_json(save_config(payload))
        except Exception as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_verify_post(self, raw: bytes) -> None:
        """POST /api/authenticity/verify — run a Claude-authenticity check.

        Built-in R-001 ban-avoidance gates (a real account was banned by live
        probing on 2026-06-28):
          - dry-run by default; live requires BOTH risk_ack=true in the body AND
            the server started with --enable-live-verify;
          - request_delay floored to >=2.0s server-side;
          - the dangerous probes (needle fake-1M, malformed error-envelope) are
            never exposed on the web path;
          - the supplied key lives only in os.environ for the call and is popped
            in a finally — never written to providers.local.json / local_secrets.env.
        """
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}")
            return
        if not isinstance(payload, dict):
            self.send_error_json(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
            return
        live = bool(payload.get("live"))
        if live and not bool(payload.get("risk_ack")):
            self.send_error_json(
                HTTPStatus.BAD_REQUEST,
                "live 检测有触发上游风控/封号的风险（参见 R-001）。请先勾选风险确认（risk_ack=true）并使用可弃用的 key。",
            )
            return
        if live and not getattr(self.server, "authenticity_live_enabled", False):
            self.send_error_json(
                HTTPStatus.FORBIDDEN,
                "live 真伪检测未启用。请以 `python api_server.py --enable-live-verify` 重启后再试（dry-run 无需此开关）。",
            )
            return
        if live:
            # live runs many requests over >=2s gaps (tens of seconds). Stream
            # per-probe progress via SSE so the page isn't a blank spinner.
            self._stream_verify(payload)
            return
        try:
            result = run_web_verify(payload, live=False)
            self.send_json(result)
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except FileNotFoundError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except Exception:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error during verify")

    def _stream_verify(self, payload: dict) -> None:
        """Run a live verify in a worker thread, stream SSE progress + final result.

        Each progress dict from verify_core is sent as an SSE `event: progress`;
        the final verdict as `event: result`; any error as `event: error`. The
        client reads the POST response body as a stream (not EventSource, so the
        key still travels in the POST body, never a query string)."""
        import queue, threading
        q: "queue.Queue[tuple[str, dict]]" = queue.Queue()

        def _worker() -> None:
            try:
                result = run_web_verify(payload, live=True,
                                        progress=lambda ev: q.put(("progress", ev)))
                q.put(("result", result))
            except ValueError as exc:
                q.put(("error", {"error": str(exc)}))
            except FileNotFoundError as exc:
                q.put(("error", {"error": str(exc)}))
            except Exception:
                q.put(("error", {"error": "internal server error during verify"}))
            finally:
                q.put(("__done__", {}))

        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("x-accel-buffering", "no")
        self.end_headers()
        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        while True:
            kind, data = q.get()
            if kind == "__done__":
                break
            try:
                payload_json = json.dumps(redact_value(data), ensure_ascii=False)
                self.wfile.write(f"event: {kind}\ndata: {payload_json}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionError):
                break  # client disconnected; worker is daemon, will end
        worker.join(timeout=1)


    def handle_campaign_get(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 3:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "campaign id is required")
            return
        campaign_id = parts[2]
        camp_dir = resolve_campaign_dir(CAMPAIGNS_DIR, campaign_id)
        if not camp_dir.exists() or not camp_dir.is_dir():
            raise FileNotFoundError(campaign_id)
        identity_problem = campaign_identity_problem(camp_dir)
        if identity_problem:
            self.send_error_json(HTTPStatus.CONFLICT, identity_problem)
            return
        tail = parts[3:] if len(parts) > 3 else ["summary"]
        endpoint = tail[0]
        if endpoint == "summary":
            summary = load_summary(camp_dir)
            if summary_needs_refresh(summary):
                summary = summarize_campaign(camp_dir, RUNS_DIR, persist=False)
            self.send_json(summary)
        elif endpoint == "authenticity":
            self.send_json(load_or_build_authenticity(camp_dir, RUNS_DIR, persist=False))
        elif endpoint == "runs":
            self.send_json(load_run_index(camp_dir))
        elif endpoint == "artifacts":
            if len(tail) > 1:
                self.serve_campaign_artifact(camp_dir, tail[1])
            else:
                self.send_json({"artifacts": artifact_listing(camp_dir / "artifacts")})
        else:
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown campaign endpoint")

    def handle_job_get(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 3:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "job id is required")
            return
        job_id = parts[2]
        run_dir = safe_run_dir(job_id)
        tail = parts[3:] if len(parts) > 3 else ["state"]
        endpoint = tail[0]
        if endpoint == "state":
            self.send_json(read_json(run_dir / "state.json"))
        elif endpoint == "events":
            self.send_json({"events": read_jsonl(run_dir / "events.jsonl")})
        elif endpoint == "results":
            results_path = run_dir / "results.json"
            self.send_json(read_json(results_path) if results_path.exists() else [])
        elif endpoint == "summary":
            self.send_json(summarize_run(run_dir))
        elif endpoint == "artifacts":
            if len(tail) > 1:
                self.serve_artifact(run_dir, tail[1])
            else:
                self.send_json({"artifacts": artifact_listing(run_dir / "artifacts")})
        else:
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown job endpoint")

    def serve_campaign_artifact(self, camp_dir: Path, name: str) -> None:
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError("invalid artifact name")
        path = camp_dir / "artifacts" / name
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(name)
        if name == "acceptance_pack.zip":
            verification = verify_acceptance_pack(path)
            if not verification.get("verified"):
                self.send_error_json(HTTPStatus.CONFLICT, f"acceptance pack failed verification: {verification.get('error')}")
                return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("content-disposition", f'attachment; filename="{path.name}"')
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_artifact(self, run_dir: Path, name: str) -> None:
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError("invalid artifact name")
        path = run_dir / "artifacts" / name
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(name)
        if name == "acceptance_pack.zip":
            verification = verify_acceptance_pack(path)
            if not verification.get("verified"):
                self.send_error_json(HTTPStatus.CONFLICT, f"acceptance pack failed verification: {verification.get('error')}")
                return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("content-disposition", f'attachment; filename="{path.name}"')
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, path: str) -> None:
        # Verify-page assets are served from web/ regardless of WEB_DIR, so the
        # authenticity page works even when the React dist is the active root.
        if path in VERIFY_ASSETS:
            rel = "verify.html" if path == "/verify" else path.lstrip("/")
            vpath = (VERIFY_WEB_DIR / rel).resolve()
            if VERIFY_WEB_DIR.resolve() not in vpath.parents:
                raise ValueError("invalid verify asset path")
            if not vpath.exists() or not vpath.is_file():
                raise FileNotFoundError(str(vpath))
            body = vpath.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", mimetypes.guess_type(vpath.name)[0] or "application/octet-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("", "/"):
            file_path = WEB_DIR / "index.html"
        else:
            rel = path.lstrip("/")
            if rel.startswith("web/"):
                rel = rel[len("web/") :]
            file_path = WEB_DIR / rel
        file_path = file_path.resolve()
        if WEB_DIR.resolve() not in file_path.parents and file_path != (WEB_DIR / "index.html").resolve():
            raise ValueError("invalid static path")
        if not file_path.exists() or not file_path.is_file():
            if file_path.suffix == "":
                file_path = (WEB_DIR / "index.html").resolve()
            else:
                raise FileNotFoundError(str(file_path))
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _self_test() -> int:
    """Offline unit checks for the pure helpers. No socket, no files."""
    # sanitize_config_value: SECURITY — any key whose name hints at a secret is
    # redacted, recursively, so /api/config can never leak a key.
    raw = {
        "base_url": "https://x",
        "api_key": "sk-secret",
        "AUTHORIZATION": "Bearer abc",
        "nested": {"password": "p", "token": "t", "model": "claude"},
        "list": [{"secret": "s", "ok": "keep"}],
    }
    clean = sanitize_config_value(raw)
    assert clean["base_url"] == "https://x"
    assert clean["api_key"] == "[REDACTED]"
    assert clean["AUTHORIZATION"] == "[REDACTED]"
    assert clean["nested"]["password"] == "[REDACTED]"
    assert clean["nested"]["token"] == "[REDACTED]"
    assert clean["nested"]["model"] == "claude"  # not a secret
    assert clean["list"][0]["secret"] == "[REDACTED]"
    assert clean["list"][0]["ok"] == "keep"
    assert "sk-secret" not in json.dumps(clean)  # nothing leaked anywhere

    # to_float: numbers parse, junk -> None.
    assert to_float("3.5") == 3.5 and to_float(7) == 7.0
    assert to_float(None) is None and to_float("x") is None

    # percentile: single value + interpolation.
    assert percentile([], 0.5) is None
    assert percentile([10.0], 0.9) == 10.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5

    # is_text_model: text hints pass; image/embedding/etc rejected.
    assert is_text_model("claude-opus-4") is True
    assert is_text_model("gpt-4o-mini") is True
    assert is_text_model("dall-e-3-image") is False
    assert is_text_model("text-embedding-3-large") is False
    assert is_text_model("whisper-tts") is False
    assert is_text_model("some-unknown-model") is False  # no hint

    # model_ids_from_payload: handles {data:[...]}, bare list, str/dict items.
    assert model_ids_from_payload({"data": [{"id": "a"}, "b"]}) == ["a", "b"]
    assert model_ids_from_payload(["x", {"name": "y"}, {"model": "z"}]) == ["x", "y", "z"]
    assert model_ids_from_payload({"models": [{"id": "m"}]}) == ["m"]
    assert model_ids_from_payload("garbage") == []

    # provider_auth_headers: bearer vs x-api-key vs unsupported.
    assert provider_auth_headers({"auth_type": "bearer"}, "K") == {"Authorization": "Bearer K"}
    assert provider_auth_headers({"auth_type": "x-api-key"}, "K") == {"x-api-key": "K"}
    try:
        provider_auth_headers({"auth_type": "weird"}, "K")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    print("api_server self-test ok")
    return 0


def main() -> int:
    import sys
    if "--self-test" in sys.argv:
        return _self_test()
    import argparse

    parser = argparse.ArgumentParser(description="Serve the eval dashboard and read-only run APIs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--enable-config-write", action="store_true", help="enable POST /api/config writes to local provider and secret files")
    parser.add_argument("--enable-live-verify", action="store_true", help="allow POST /api/authenticity/verify to make REAL live calls (cost + R-001 ban risk); dry-run always works without it")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.config_write_enabled = bool(args.enable_config_write)  # type: ignore[attr-defined]
    server.authenticity_live_enabled = bool(args.enable_live_verify)  # type: ignore[attr-defined]
    print(f"serving http://{args.host}:{args.port}")
    if args.enable_live_verify:
        print("⚠ live 真伪检测已启用 — 会对填入的网关发起真实请求（消耗额度，有触发风控风险，请用可弃用 key）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
