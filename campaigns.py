from __future__ import annotations

import json
import math
import re
import statistics
import zipfile
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from redaction import redact_text


DECISION_ORDER = {"GO": 0, "REVIEW": 1, "NO-GO": 2}
SUMMARY_SCHEMA_VERSION = "campaign_summary_v1"
REQUIRED_SUMMARY_METRIC_KEYS = {
    "total_runs",
    "run_history_count",
    "replaced_run_count",
    "completed_runs",
    "total_cases",
    "model_response_success_rate",
    "transport_success_rate",
    "average_quality_score",
    "median_quality_score",
    "protocol_compatibility_score",
    "model_name_consistency_rate",
    "retried_request_count",
    "total_retry_count",
    "p50_latency_ms",
    "p95_latency_ms",
    "latest_tested_at",
    "error_counts",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 6)


def model_name_matches(requested: str, returned: str) -> bool:
    requested = str(requested or "").strip()
    returned = str(returned or "").strip()
    if not requested or not returned:
        return False
    if requested == returned:
        return True
    if returned.startswith(f"{requested}-"):
        suffix = returned[len(requested) + 1 :]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", suffix):
            return True
    return False


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def safe_campaign_id(campaign_id: str) -> str:
    value = str(campaign_id or "").strip()
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError("invalid campaign id")
    return value


def campaign_dir(campaigns_dir: Path, campaign_id: str) -> Path:
    return campaigns_dir / safe_campaign_id(campaign_id)


def load_campaign(campaign_dir_path: Path) -> dict[str, Any]:
    return read_json(campaign_dir_path / "campaign.json")


def load_run_index(campaign_dir_path: Path) -> dict[str, Any]:
    path = campaign_dir_path / "run_ids.json"
    if not path.exists():
        return {"runs": []}
    return read_json(path)


def campaign_identity_problem(campaign_dir_path: Path) -> str | None:
    try:
        campaign = load_campaign(campaign_dir_path)
        run_index = load_run_index(campaign_dir_path)
    except Exception as exc:
        return f"campaign metadata unreadable: {type(exc).__name__}"
    expected = campaign_dir_path.name
    campaign_id = str(campaign.get("campaign_id") or "")
    run_index_id = str(run_index.get("campaign_id") or campaign_id or "")
    if campaign_id and campaign_id != expected:
        return f"campaign_id mismatch: dir={expected}, campaign_json={campaign_id}"
    if run_index_id and run_index_id != expected:
        return f"campaign_id mismatch: dir={expected}, run_ids={run_index_id}"
    return None


def load_summary(campaign_dir_path: Path) -> dict[str, Any] | None:
    path = campaign_dir_path / "summary.json"
    if not path.exists():
        return None
    return read_json(path)


def summary_needs_refresh(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return True
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    if not REQUIRED_SUMMARY_METRIC_KEYS.issubset(metrics):
        return True
    decisions = summary.get("decisions") if isinstance(summary.get("decisions"), dict) else {}
    return not {"model_confidence_decision", "gateway_reliability_decision", "overall_decision"}.issubset(decisions)


def list_campaign_dirs(campaigns_dir: Path) -> list[Path]:
    if not campaigns_dir.exists():
        return []
    valid: list[Path] = []
    for path in campaigns_dir.iterdir():
        if not path.is_dir() or not (path / "campaign.json").exists():
            continue
        if campaign_identity_problem(path):
            continue
        valid.append(path)
    return sorted(
        valid,
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def latest_quality_gate_record(run_dir: Path) -> dict[str, Any] | None:
    gates_dir = run_dir / "quality_gates"
    if not gates_dir.exists():
        return None
    candidates = [path for path in gates_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    gate_dir = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    records = read_jsonl(gate_dir / "quality_gate_records.jsonl")
    return records[0] if records else None


def score_value(record: dict[str, Any]) -> float | None:
    scoring = record.get("scoring") if isinstance(record.get("scoring"), dict) else {}
    final_score = scoring.get("final_score") if isinstance(scoring.get("final_score"), dict) else {}
    return numeric(final_score.get("score"))


def judge_reason(record: dict[str, Any]) -> str:
    scoring = record.get("scoring") if isinstance(record.get("scoring"), dict) else {}
    final_score = scoring.get("final_score") if isinstance(scoring.get("final_score"), dict) else {}
    return redact_text(final_score.get("details") or "", max_chars=500) or ""


def error_type(error: Any) -> str:
    text = str(error or "").strip()
    if not text:
        return "unknown"
    lowered = text.lower()
    if "ssl" in lowered:
        return "ssl"
    if "429" in lowered or "rate" in lowered or "quota" in lowered:
        return "rate_limit"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "readerror" in lowered or "read error" in lowered:
        return "read_error"
    if "connect" in lowered or "connection" in lowered:
        return "connection"
    if "json" in lowered:
        return "json_or_parse"
    return "transport_or_model"


def worst_decision(*decisions: str | None) -> str:
    selected = "GO"
    for decision in decisions:
        if DECISION_ORDER.get(str(decision or "GO"), 0) > DECISION_ORDER[selected]:
            selected = str(decision)
    return selected


def model_confidence_decision(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    decision = "GO"
    reasons: list[str] = []
    consistency = numeric(metrics.get("model_name_consistency_rate"))
    avg_quality = numeric(metrics.get("average_quality_score"))
    protocol_score = numeric(metrics.get("protocol_compatibility_score"))

    if consistency is None:
        decision = worst_decision(decision, "REVIEW")
        reasons.append("model_returned_missing")
    elif consistency < 0.98:
        decision = worst_decision(decision, "NO-GO")
        reasons.append("model_name_consistency_below_0.98")

    if avg_quality is None:
        decision = worst_decision(decision, "REVIEW")
        reasons.append("quality_score_missing")
    elif avg_quality < 6.0:
        decision = worst_decision(decision, "NO-GO")
        reasons.append("average_quality_below_6.0")
    elif avg_quality < 7.5:
        decision = worst_decision(decision, "REVIEW")
        reasons.append("average_quality_below_7.5")

    if protocol_score is None:
        decision = worst_decision(decision, "REVIEW")
        reasons.append("protocol_compatibility_missing")
    elif protocol_score < 1.0:
        decision = worst_decision(decision, "REVIEW")
        reasons.append("protocol_compatibility_not_full")

    return decision, reasons


def gateway_reliability_decision(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    decision = "GO"
    reasons: list[str] = []
    transport_rate = numeric(metrics.get("transport_success_rate"))
    p95_latency = numeric(metrics.get("p95_latency_ms"))

    if transport_rate is None:
        decision = worst_decision(decision, "REVIEW")
        reasons.append("transport_success_missing")
    elif transport_rate < 0.95:
        decision = worst_decision(decision, "NO-GO")
        reasons.append("transport_success_below_0.95")
    elif transport_rate < 0.98:
        decision = worst_decision(decision, "REVIEW")
        reasons.append("transport_success_below_0.98")

    if p95_latency is None:
        decision = worst_decision(decision, "REVIEW")
        reasons.append("latency_missing")
    elif p95_latency > 15000:
        decision = worst_decision(decision, "REVIEW")
        reasons.append("p95_latency_above_15000ms")

    return decision, reasons


def provider_score_for_run(run_dir: Path, provider_id: str | None) -> dict[str, Any]:
    path = run_dir / "benchmark_scores.json"
    if not path.exists():
        return {}
    data = read_json(path)
    providers = data.get("providers") if isinstance(data.get("providers"), dict) else {}
    if provider_id and provider_id in providers and isinstance(providers[provider_id], dict):
        return providers[provider_id]
    first = next(iter(providers.values()), {})
    return first if isinstance(first, dict) else {}


def compatibility_key(campaign: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_version": campaign.get("benchmark_version"),
        "benchmark_mode": campaign.get("benchmark_mode"),
        "judge_model": (campaign.get("judge_model") or {}).get("model"),
        "quality_gate_version": campaign.get("quality_gate_version"),
        "score_formula_version": campaign.get("score_formula_version"),
        "live_provider": campaign.get("live_provider") is True,
    }


def compatibility_key_string(key: dict[str, Any]) -> str:
    return "|".join(f"{name}={key.get(name)}" for name in sorted(key))


def summarize_campaign(campaign_dir_path: Path, runs_dir: Path, *, persist: bool = True) -> dict[str, Any]:
    campaign = load_campaign(campaign_dir_path)
    run_index = load_run_index(campaign_dir_path)
    run_refs = run_index.get("runs") if isinstance(run_index.get("runs"), list) else []
    active_refs = [run_ref for run_ref in run_refs if run_ref.get("status") != "replaced"]
    tested_identity = campaign.get("tested_model") if isinstance(campaign.get("tested_model"), dict) else {}
    tested_provider_id = str(tested_identity.get("provider_id") or "")
    expected_protocol = str(tested_identity.get("protocol") or "")

    all_records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    trend: list[dict[str, Any]] = []
    child_runs: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    scores: list[float] = []
    latencies: list[float] = []
    benchmark_scores: list[float] = []
    provider_quality_scores: list[float] = []
    transport_ok_count = 0
    model_response_count = 0
    protocol_match_count = 0
    returned_seen_count = 0
    returned_match_count = 0
    retried_request_count = 0
    total_retry_count = 0
    latest_tested_at = ""

    for index, run_ref in enumerate(active_refs, start=1):
        run_id = str(run_ref.get("run_id") or "")
        run_dir = runs_dir / run_id
        state = read_json(run_dir / "state.json") if (run_dir / "state.json").exists() else {}
        records = read_jsonl(run_dir / "run_records.jsonl")
        provider_score = provider_score_for_run(run_dir, tested_provider_id)
        gate_record = latest_quality_gate_record(run_dir)
        run_scores: list[float] = []
        run_latencies: list[float] = []
        run_transport_ok = 0

        for record in records:
            all_records.append(record)
            task = record.get("task") if isinstance(record.get("task"), dict) else {}
            provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
            telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
            response = record.get("response") if isinstance(record.get("response"), dict) else {}
            scoring = record.get("scoring") if isinstance(record.get("scoring"), dict) else {}
            judge_score = scoring.get("judge_score") if isinstance(scoring.get("judge_score"), dict) else {}
            ok = telemetry.get("ok") is True
            if ok:
                transport_ok_count += 1
                run_transport_ok += 1
            if ok and numeric(response.get("content_chars")) and numeric(response.get("content_chars")) > 0:
                model_response_count += 1
            if provider.get("api_style") == expected_protocol:
                protocol_match_count += 1

            requested = str(provider.get("model_requested") or "")
            returned = str(provider.get("model_returned") or "")
            if returned:
                returned_seen_count += 1
                if model_name_matches(requested, returned):
                    returned_match_count += 1

            score = score_value(record)
            if ok and score is not None:
                scores.append(score)
                run_scores.append(score)
            latency = numeric(telemetry.get("first_content_token_ms") or telemetry.get("total_ms"))
            if latency is not None:
                latencies.append(latency)
                run_latencies.append(latency)
            if telemetry.get("error"):
                errors[error_type(telemetry.get("error"))] += 1
            if judge_score.get("error"):
                errors["judge_error"] += 1
            retry_count = numeric(telemetry.get("retry_count")) or 0
            if retry_count > 0:
                retried_request_count += 1
                total_retry_count += int(retry_count)

            samples.append(
                {
                    "run_id": run_id,
                    "round": int(run_ref.get("round") or index),
                    "task_id": task.get("id"),
                    "category": task.get("category"),
                    "ok": ok,
                    "score": score,
                    "latency_ms": latency,
                    "error": redact_text(telemetry.get("error"), max_chars=500),
                    "error_type": error_type(telemetry.get("error")) if telemetry.get("error") else None,
                    "model_requested": provider.get("model_requested"),
                    "model_returned": provider.get("model_returned"),
                    "judge_reason": judge_reason(record),
                }
            )

        benchmark_score = numeric(provider_score.get("benchmark_score"))
        quality_score = numeric(provider_score.get("quality_score"))
        if benchmark_score is not None:
            benchmark_scores.append(benchmark_score)
        if quality_score is not None:
            provider_quality_scores.append(quality_score)
        completed_at = str(state.get("completed_at") or run_ref.get("completed_at") or "")
        if completed_at > latest_tested_at:
            latest_tested_at = completed_at
        run_cases = len(records)
        child_runs.append(
            {
                "round": int(run_ref.get("round") or index),
                "run_id": run_id,
                "status": state.get("status") or run_ref.get("status"),
                "final_decision": state.get("final_decision"),
                "started_at": state.get("started_at") or run_ref.get("started_at"),
                "completed_at": completed_at or None,
                "total_cases": run_cases,
                "transport_success_rate": ratio(run_transport_ok, run_cases),
                "average_quality_score": round(sum(run_scores) / len(run_scores), 3) if run_scores else None,
                "benchmark_score": benchmark_score,
                "quality_score": quality_score,
                "p95_latency_ms": percentile(run_latencies, 0.95),
            }
        )
        trend.append(
            {
                "round": int(run_ref.get("round") or index),
                "run_id": run_id,
                "average_quality_score": child_runs[-1]["average_quality_score"],
                "benchmark_score": benchmark_score,
                "transport_success_rate": child_runs[-1]["transport_success_rate"],
                "p95_latency_ms": child_runs[-1]["p95_latency_ms"],
                "decision": (gate_record or {}).get("decision") or state.get("final_decision"),
            }
        )

    total_cases = len(all_records)
    completed_runs = sum(1 for run in child_runs if run.get("status") == "completed")
    metrics = {
        "total_runs": len(active_refs),
        "run_history_count": len(run_refs),
        "replaced_run_count": len(run_refs) - len(active_refs),
        "completed_runs": completed_runs,
        "total_cases": total_cases,
        "successful_model_responses": model_response_count,
        "transport_successful_requests": transport_ok_count,
        "model_response_success_rate": ratio(model_response_count, total_cases),
        "transport_success_rate": ratio(transport_ok_count, total_cases),
        "evaluated_response_count": len(scores),
        "average_quality_score": round(sum(scores) / len(scores), 3) if scores else None,
        "median_quality_score": round(statistics.median(scores), 3) if scores else None,
        "average_benchmark_score": round(sum(benchmark_scores) / len(benchmark_scores), 3) if benchmark_scores else None,
        "average_provider_quality_score": round(sum(provider_quality_scores) / len(provider_quality_scores), 3) if provider_quality_scores else None,
        "protocol_compatibility_score": ratio(protocol_match_count, total_cases),
        "protocol_match_count": protocol_match_count,
        "model_name_consistency_rate": ratio(returned_match_count, returned_seen_count) if returned_seen_count else None,
        "model_returned_seen_count": returned_seen_count,
        "model_returned_match_count": returned_match_count,
        "model_returned_missing_count": total_cases - returned_seen_count,
        "retried_request_count": retried_request_count,
        "total_retry_count": total_retry_count,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "latest_tested_at": latest_tested_at or None,
        "error_counts": dict(sorted(errors.items())),
    }
    model_decision, model_reasons = model_confidence_decision(metrics)
    gateway_decision, gateway_reasons = gateway_reliability_decision(metrics)
    overall = worst_decision(model_decision, gateway_decision)
    campaign_status = str(campaign.get("status") or "")
    if campaign_status == "running":
        status = "running"
    elif campaign_status in {"failed", "partial"}:
        status = campaign_status
    else:
        status = "completed" if completed_runs == len(active_refs) and active_refs else "partial"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "campaign_id": campaign.get("campaign_id") or campaign_dir_path.name,
        "status": status,
        "created_at": campaign.get("created_at"),
        "completed_at": campaign.get("completed_at"),
        "live_provider": campaign.get("live_provider") is True,
        "tested_model": campaign.get("tested_model"),
        "judge_model": campaign.get("judge_model"),
        "benchmark_version": campaign.get("benchmark_version"),
        "benchmark_mode": campaign.get("benchmark_mode"),
        "quality_gate_version": campaign.get("quality_gate_version"),
        "score_formula_version": campaign.get("score_formula_version"),
        "comparison_key": compatibility_key(campaign),
        "comparison_key_id": compatibility_key_string(compatibility_key(campaign)),
        "metrics": metrics,
        "decisions": {
            "model_confidence_decision": model_decision,
            "model_confidence_reasons": model_reasons,
            "gateway_reliability_decision": gateway_decision,
            "gateway_reliability_reasons": gateway_reasons,
            "overall_decision": overall,
        },
        "child_runs": child_runs,
        "trend": trend,
        "samples": samples,
        "latency_values_ms": latencies,
        "failure_counts": metrics["error_counts"],
    }
    if persist:
        write_json(campaign_dir_path / "summary.json", summary)
    return summary


def summary_to_leaderboard_entry(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    decisions = summary.get("decisions") if isinstance(summary.get("decisions"), dict) else {}
    tested = summary.get("tested_model") if isinstance(summary.get("tested_model"), dict) else {}
    judge = summary.get("judge_model") if isinstance(summary.get("judge_model"), dict) else {}
    return {
        "campaign_id": summary.get("campaign_id"),
        "tested_model": tested.get("model"),
        "tested_provider_id": tested.get("provider_id"),
        "judge_model": judge.get("model"),
        "judge_provider_id": judge.get("provider_id"),
        "live_provider": summary.get("live_provider") is True,
        "benchmark_version": summary.get("benchmark_version"),
        "benchmark_mode": summary.get("benchmark_mode"),
        "quality_gate_version": summary.get("quality_gate_version"),
        "score_formula_version": summary.get("score_formula_version"),
        "comparison_key": summary.get("comparison_key"),
        "comparison_key_id": summary.get("comparison_key_id"),
        "total_runs": metrics.get("total_runs"),
        "completed_runs": metrics.get("completed_runs"),
        "total_cases": metrics.get("total_cases"),
        "model_response_success_rate": metrics.get("model_response_success_rate"),
        "transport_success_rate": metrics.get("transport_success_rate"),
        "average_quality_score": metrics.get("average_quality_score"),
        "median_quality_score": metrics.get("median_quality_score"),
        "average_benchmark_score": metrics.get("average_benchmark_score"),
        "protocol_compatibility_score": metrics.get("protocol_compatibility_score"),
        "model_name_consistency_rate": metrics.get("model_name_consistency_rate"),
        "p50_latency_ms": metrics.get("p50_latency_ms"),
        "p95_latency_ms": metrics.get("p95_latency_ms"),
        "model_confidence_decision": decisions.get("model_confidence_decision"),
        "gateway_reliability_decision": decisions.get("gateway_reliability_decision"),
        "overall_decision": decisions.get("overall_decision"),
        "latest_tested_at": metrics.get("latest_tested_at") or summary.get("completed_at"),
        "status": summary.get("status"),
        "score": metrics.get("average_benchmark_score") or ((metrics.get("average_quality_score") or 0) * 100),
    }


def list_campaign_summaries(
    campaigns_dir: Path,
    runs_dir: Path | None = None,
    *,
    refresh_missing: bool = False,
    persist_refresh: bool = True,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in list_campaign_dirs(campaigns_dir):
        summary = load_summary(path)
        if (summary is None or summary_needs_refresh(summary)) and refresh_missing and runs_dir is not None:
            summary = summarize_campaign(path, runs_dir, persist=persist_refresh)
        if summary is not None:
            summaries.append(summary)
    return summaries


def campaign_leaderboard(
    campaigns_dir: Path,
    runs_dir: Path,
    *,
    include_dry_run: bool = False,
    benchmark_version: str = "",
    judge_model: str = "",
    quality_gate_version: str = "",
    live_provider: bool | None = None,
    date_from: str = "",
    date_to: str = "",
    min_samples: int = 1,
    limit: int = 50,
    persist_refresh: bool = True,
) -> dict[str, Any]:
    entries = [
        summary_to_leaderboard_entry(summary)
        for summary in list_campaign_summaries(campaigns_dir, runs_dir, refresh_missing=True, persist_refresh=persist_refresh)
    ]
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("status") != "completed":
            continue
        is_live = entry.get("live_provider") is True
        if live_provider is not None and is_live != live_provider:
            continue
        if live_provider is None and not include_dry_run and not is_live:
            continue
        if benchmark_version and entry.get("benchmark_version") != benchmark_version:
            continue
        if judge_model and entry.get("judge_model") != judge_model:
            continue
        if quality_gate_version and entry.get("quality_gate_version") != quality_gate_version:
            continue
        if int(entry.get("total_cases") or 0) < min_samples:
            continue
        latest = str(entry.get("latest_tested_at") or "")
        if date_from and latest and latest < date_from:
            continue
        if date_to and latest and latest > date_to:
            continue
        filtered.append(entry)

    selected_key = None
    if filtered:
        groups: dict[str, list[dict[str, Any]]] = {}
        for entry in filtered:
            groups.setdefault(str(entry.get("comparison_key_id") or ""), []).append(entry)
        selected_key = max(
            groups,
            key=lambda key: max(str(item.get("latest_tested_at") or "") for item in groups[key]),
        )
        filtered = groups[selected_key]

    filtered.sort(
        key=lambda item: (
            numeric(item.get("score")) is not None,
            numeric(item.get("score")) or -1,
            numeric(item.get("model_response_success_rate")) or -1,
            str(item.get("latest_tested_at") or ""),
        ),
        reverse=True,
    )
    for index, entry in enumerate(filtered, start=1):
        entry["rank"] = index
    return {
        "entries": filtered[:limit],
        "total": len(filtered),
        "limit": limit,
        "include_dry_run": include_dry_run,
        "selected_comparison_key_id": selected_key,
        "sort": "campaign score desc, model response success desc, latest test desc",
    }


def campaign_list_payload(campaigns_dir: Path, runs_dir: Path, *, persist_refresh: bool = True) -> dict[str, Any]:
    campaigns = []
    for summary in list_campaign_summaries(campaigns_dir, runs_dir, refresh_missing=True, persist_refresh=persist_refresh):
        entry = summary_to_leaderboard_entry(summary)
        campaigns.append(entry)
    campaigns.sort(key=lambda item: str(item.get("latest_tested_at") or ""), reverse=True)
    return {"campaigns": campaigns}


def _zip_add_file(zf: zipfile.ZipFile, path: Path, arcname: str, checksums: dict[str, str]) -> None:
    data = path.read_bytes()
    zf.writestr(arcname, data)
    checksums[arcname] = sha256(data).hexdigest()


def _zip_add_json(zf: zipfile.ZipFile, arcname: str, value: Any, checksums: dict[str, str]) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    zf.writestr(arcname, data)
    checksums[arcname] = sha256(data).hexdigest()


def export_campaign(campaign_dir_path: Path, runs_dir: Path, *, include_raw: bool = False) -> Path:
    campaign = load_campaign(campaign_dir_path)
    run_index = load_run_index(campaign_dir_path)
    artifacts_dir = campaign_dir_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    zip_path = artifacts_dir / "acceptance_pack.zip"
    include_campaign_files = ["campaign.json", "summary.json", "run_ids.json", "authenticity_summary.json"]
    include_run_files = [
        "state.json",
        "run_records.jsonl",
        "results.json",
        "summary.csv",
        "benchmark_scores.json",
        "validation.json",
        "job_config.snapshot.json",
        "providers.redacted.json",
    ]
    if include_raw:
        include_run_files.append("events.jsonl")
    checksums: dict[str, str] = {}
    included_runs: list[str] = []
    excluded_replaced_runs: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include_campaign_files:
            path = campaign_dir_path / name
            if path.exists():
                _zip_add_file(zf, path, name, checksums)
        for folder in ["baseline_comparisons", "protocol_fingerprints"]:
            root = campaign_dir_path / folder
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    rel_name = path.relative_to(campaign_dir_path).as_posix()
                    _zip_add_file(zf, path, rel_name, checksums)
        for run_ref in run_index.get("runs") or []:
            if run_ref.get("status") == "replaced":
                if run_ref.get("run_id"):
                    excluded_replaced_runs.append(str(run_ref.get("run_id")))
                continue
            run_id = str(run_ref.get("run_id") or "")
            run_dir = runs_dir / run_id
            if not run_dir.exists():
                continue
            included_runs.append(run_id)
            for name in include_run_files:
                path = run_dir / name
                if path.exists():
                    _zip_add_file(zf, path, f"runs/{run_id}/{name}", checksums)
            folders = ["quality_gates", "trace_evaluations", "baseline_comparisons", "protocol_fingerprints"]
            if include_raw:
                folders.extend(["events", "responses", "judge_responses"])
            for folder in folders:
                root = run_dir / folder
                if not root.exists():
                    continue
                for path in root.rglob("*"):
                    if path.is_file():
                        rel_name = path.relative_to(run_dir).as_posix()
                        _zip_add_file(zf, path, f"runs/{run_id}/{rel_name}", checksums)
        manifest = {
            "schema_version": "acceptance_pack_manifest_v1",
            "pack_type": "campaign",
            "campaign_id": campaign.get("campaign_id") or campaign_dir_path.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "include_raw": include_raw,
            "included_runs": included_runs,
            "excluded_replaced_runs": excluded_replaced_runs,
            "entry_count_without_manifest": len(checksums),
            "raw_entry_policy": "included by explicit request" if include_raw else "excluded by default",
        }
        _zip_add_json(zf, "acceptance_manifest.json", manifest, checksums)
        checksum_lines = [f"{digest}  {name}" for name, digest in sorted(checksums.items())]
        checksum_data = ("\n".join(checksum_lines) + "\n").encode("utf-8")
        zf.writestr("checksums.sha256", checksum_data)
    campaign.setdefault("artifacts", {})["acceptance_pack"] = str(zip_path)
    write_json(campaign_dir_path / "campaign.json", campaign)
    return zip_path
