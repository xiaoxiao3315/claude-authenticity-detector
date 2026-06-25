from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from campaigns import load_campaign, load_run_index, load_summary, summarize_campaign
from redaction import redact_text


AUTHENTICITY_SCHEMA_VERSION = "authenticity_summary_v1"
BASELINE_COMPARISON_SCHEMA_VERSION = "baseline_comparison_v1"
PROTOCOL_FINGERPRINT_SCHEMA_VERSION = "protocol_fingerprint_v1"
DECISION_ORDER = {"GO": 0, "REVIEW": 1, "NO-GO": 2}
REQUEST_ID_KEYS = ("request_id", "upstream_request_id", "openai_request_id", "anthropic_request_id", "cf_ray")
GATEWAY_AUDIT_KEYS = (
    "upstream_provider",
    "upstream_model",
    "upstream_request_id",
    "gateway_route_id",
    "fallback_used",
    "retry_count",
    "cache_hit",
    "gateway_processing_ms",
)
STREAMING_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "message_delta",
    "message_stop",
    "response.output_text.delta",
    "response.completed",
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


def record_event_path(record: dict[str, Any]) -> Path | None:
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    value = artifacts.get("events_file") or response.get("events_file")
    if not value:
        return None
    return Path(str(value))


def record_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    path = record_event_path(record)
    return read_jsonl(path) if path is not None else []


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


def clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def worst_decision(*decisions: str | None) -> str:
    selected = "GO"
    for decision in decisions:
        value = str(decision or "GO")
        if DECISION_ORDER.get(value, 0) > DECISION_ORDER[selected]:
            selected = value
    return selected


def decision_from_score(score: float | None, *, go_threshold: float = 0.85, review_threshold: float = 0.60) -> str:
    if score is None:
        return "REVIEW"
    if score >= go_threshold:
        return "GO"
    if score >= review_threshold:
        return "REVIEW"
    return "NO-GO"


def decision_score(decision: str | None) -> float:
    value = str(decision or "REVIEW")
    if value == "GO":
        return 1.0
    if value == "NO-GO":
        return 0.0
    return 0.5


def active_run_refs(campaign_dir_path: Path) -> list[dict[str, Any]]:
    run_index = load_run_index(campaign_dir_path)
    refs = run_index.get("runs") if isinstance(run_index.get("runs"), list) else []
    return [ref for ref in refs if isinstance(ref, dict) and ref.get("status") != "replaced"]


def campaign_records(campaign_dir_path: Path, runs_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ref in active_run_refs(campaign_dir_path):
        run_id = str(ref.get("run_id") or "")
        if not run_id:
            continue
        records.extend(read_jsonl(runs_dir / run_id / "run_records.jsonl"))
    return records


def record_task_scores(summary: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for sample in summary.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        task_id = str(sample.get("task_id") or "")
        score = numeric(sample.get("score"))
        if task_id and score is not None:
            scores[task_id] = score
    return scores


def record_feature_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    usage_seen = 0
    stop_reason_seen = 0
    request_hash_seen = 0
    event_path_seen = 0
    raw_event_seen = 0
    upstream_request_seen = 0
    response_headers_seen = 0
    request_id_seen = 0
    streaming_order_seen = 0
    cache_usage_seen = 0
    tool_use_seen = 0
    json_probe_seen = 0
    long_context_probe_seen = 0
    gateway_route_seen = 0
    fallback_seen = 0
    retry_count_seen = 0
    cache_hit_seen = 0
    gateway_processing_seen = 0
    stop_reasons: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    gateway_audit_counts: Counter[str] = Counter()
    for record in records:
        task = record.get("task") if isinstance(record.get("task"), dict) else {}
        provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
        request = record.get("request") if isinstance(record.get("request"), dict) else {}
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
        trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
        artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
        response = record.get("response") if isinstance(record.get("response"), dict) else {}
        events = record_events(record)
        record_request_id_present = False
        record_upstream_request_present = False

        if task.get("category"):
            categories[str(task.get("category"))] += 1
        task_label = f"{task.get('id') or ''} {task.get('category') or ''}".lower()
        if "json" in task_label:
            json_probe_seen += 1
        if "long_context" in task_label or "long-context" in task_label:
            long_context_probe_seen += 1
        if usage.get("input_tokens") is not None or usage.get("output_tokens") is not None:
            usage_seen += 1
        if usage.get("cache_creation_input_tokens") is not None or usage.get("cache_read_input_tokens") is not None:
            cache_usage_seen += 1
        if telemetry.get("stop_reason"):
            stop_reason_seen += 1
            stop_reasons[str(telemetry.get("stop_reason"))] += 1
        if request.get("request_hash"):
            request_hash_seen += 1
        if artifacts.get("events_file") or response.get("events_file"):
            event_path_seen += 1
        raw_types = trace.get("raw_event_types") if isinstance(trace.get("raw_event_types"), list) else []
        if raw_types:
            raw_event_seen += 1
            for event_type in raw_types:
                event_types[str(event_type)] += 1
        event_type_values = [str(event.get("type") or "") for event in events if isinstance(event, dict)]
        if any(event_type in STREAMING_EVENT_TYPES for event_type in event_type_values):
            streaming_order_seen += 1
        if any(isinstance(event.get("response_headers"), dict) and event.get("response_headers") for event in events):
            response_headers_seen += 1
        if any(event.get("request_id") for event in events if isinstance(event, dict)):
            record_request_id_present = True
        tool_calls = trace.get("tool_calls") if isinstance(trace.get("tool_calls"), list) else []
        if tool_calls:
            tool_use_seen += 1
        if provider.get("upstream_request_id") or telemetry.get("upstream_request_id"):
            record_upstream_request_present = True
            record_request_id_present = True
        if response.get("request_id") or telemetry.get("request_id") or provider.get("request_id"):
            record_request_id_present = True
        if record_upstream_request_present:
            upstream_request_seen += 1
        if record_request_id_present:
            request_id_seen += 1
        for source in (provider, telemetry, response):
            for key in GATEWAY_AUDIT_KEYS:
                if source.get(key) is not None:
                    gateway_audit_counts[key] += 1
        if provider.get("gateway_route_id") or telemetry.get("gateway_route_id") or response.get("gateway_route_id"):
            gateway_route_seen += 1
        if provider.get("fallback_used") is not None or telemetry.get("fallback_used") is not None or response.get("fallback_used") is not None:
            fallback_seen += 1
        if telemetry.get("retry_count") is not None:
            retry_count_seen += 1
        if provider.get("cache_hit") is not None or telemetry.get("cache_hit") is not None or response.get("cache_hit") is not None:
            cache_hit_seen += 1
        if provider.get("gateway_processing_ms") is not None or telemetry.get("gateway_processing_ms") is not None or response.get("gateway_processing_ms") is not None:
            gateway_processing_seen += 1

    return {
        "record_count": total,
        "usage_presence_rate": ratio(usage_seen, total),
        "stop_reason_presence_rate": ratio(stop_reason_seen, total),
        "request_hash_presence_rate": ratio(request_hash_seen, total),
        "event_path_presence_rate": ratio(event_path_seen, total),
        "raw_event_type_presence_rate": ratio(raw_event_seen, total),
        "upstream_request_id_presence_rate": ratio(upstream_request_seen, total),
        "response_headers_presence_rate": ratio(response_headers_seen, total),
        "request_id_presence_rate": ratio(request_id_seen, total),
        "streaming_event_order_presence_rate": ratio(streaming_order_seen, total),
        "cache_usage_field_presence_rate": ratio(cache_usage_seen, total),
        "tool_use_presence_rate": ratio(tool_use_seen, total),
        "json_probe_task_count": json_probe_seen,
        "long_context_probe_task_count": long_context_probe_seen,
        "gateway_route_id_presence_rate": ratio(gateway_route_seen, total),
        "fallback_used_presence_rate": ratio(fallback_seen, total),
        "retry_count_presence_rate": ratio(retry_count_seen, total),
        "cache_hit_presence_rate": ratio(cache_hit_seen, total),
        "gateway_processing_ms_presence_rate": ratio(gateway_processing_seen, total),
        "gateway_audit_field_counts": dict(sorted(gateway_audit_counts.items())),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "raw_event_type_counts": dict(sorted(event_types.items())),
        "category_counts": dict(sorted(categories.items())),
    }


def percentile(values: list[float], q: float) -> float | None:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[int(index)], 3)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def score_values(summary: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for sample in summary.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        score = numeric(sample.get("score"))
        if score is not None:
            values.append(score)
    return values


def statistical_confidence(summary: dict[str, Any], *, min_sample_threshold: int = 30) -> dict[str, Any]:
    samples = summary.get("samples") if isinstance(summary.get("samples"), list) else []
    child_runs = summary.get("child_runs") if isinstance(summary.get("child_runs"), list) else []
    scores = score_values(summary)
    total_samples = len(samples)
    evaluated_samples = len(scores)
    mean_score = round(sum(scores) / evaluated_samples, 3) if scores else None
    stdev = round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0 if scores else None
    stderr = round((stdev or 0.0) / math.sqrt(evaluated_samples), 6) if evaluated_samples else None
    normal_ci = None
    if mean_score is not None and stderr is not None:
        margin = 1.96 * stderr
        normal_ci = [round(mean_score - margin, 3), round(mean_score + margin, 3)]

    bootstrap_ci = None
    bootstrap_mean = None
    if scores:
        rng = random.Random(202603)
        boot_means: list[float] = []
        iterations = 200 if len(scores) > 1 else 1
        for _ in range(iterations):
            sample = [scores[rng.randrange(len(scores))] for _ in scores]
            boot_means.append(sum(sample) / len(sample))
        bootstrap_mean = round(sum(boot_means) / len(boot_means), 3)
        bootstrap_ci = [percentile(boot_means, 0.025), percentile(boot_means, 0.975)]

    by_task: dict[str, list[float]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        task_id = str(sample.get("task_id") or "")
        score = numeric(sample.get("score"))
        if task_id and score is not None:
            by_task.setdefault(task_id, []).append(score)
    repeated_ranges = [max(values) - min(values) for values in by_task.values() if len(values) > 1]
    repeated_stdevs = [statistics.stdev(values) for values in by_task.values() if len(values) > 1]
    average_repeated_range = round(sum(repeated_ranges) / len(repeated_ranges), 3) if repeated_ranges else None
    average_task_variance = round(sum(value * value for value in repeated_stdevs) / len(repeated_stdevs), 3) if repeated_stdevs else None
    repeated_task_consistency = None
    if average_repeated_range is not None:
        repeated_task_consistency = round(max(0.0, 1.0 - min(average_repeated_range / 10.0, 1.0)), 6)

    run_quality = [numeric(run.get("average_quality_score")) for run in child_runs if isinstance(run, dict)]
    run_quality = [value for value in run_quality if value is not None]
    run_transport = [numeric(run.get("transport_success_rate")) for run in child_runs if isinstance(run, dict)]
    run_transport = [value for value in run_transport if value is not None]
    completed_dates = {
        str(run.get("completed_at") or "")[:10]
        for run in child_runs
        if isinstance(run, dict) and run.get("completed_at")
    }
    anomalies: list[dict[str, Any]] = []
    quality_mean = sum(run_quality) / len(run_quality) if run_quality else None
    quality_stdev = statistics.stdev(run_quality) if len(run_quality) > 1 else 0.0 if run_quality else None
    for run in child_runs:
        if not isinstance(run, dict):
            continue
        run_id = run.get("run_id")
        quality = numeric(run.get("average_quality_score"))
        transport = numeric(run.get("transport_success_rate"))
        p95 = numeric(run.get("p95_latency_ms"))
        if quality is not None and quality_mean is not None and quality_stdev and abs(quality - quality_mean) > quality_stdev * 2:
            anomalies.append({"run_id": run_id, "type": "quality_outlier", "value": quality})
        if transport is not None and transport < 0.95:
            anomalies.append({"run_id": run_id, "type": "transport_below_0.95", "value": transport})
        if p95 is not None and p95 > 15000:
            anomalies.append({"run_id": run_id, "type": "p95_latency_above_15000ms", "value": p95})

    quality_slope = None
    if len(run_quality) >= 2:
        quality_slope = round(run_quality[-1] - run_quality[0], 3)
    transport_slope = None
    if len(run_transport) >= 2:
        transport_slope = round(run_transport[-1] - run_transport[0], 6)

    return {
        "min_sample_threshold": min_sample_threshold,
        "sample_threshold_met": total_samples >= min_sample_threshold,
        "total_samples": total_samples,
        "evaluated_samples": evaluated_samples,
        "mean_quality_score": mean_score,
        "stdev_quality_score": stdev,
        "stderr_quality_score": stderr,
        "normal_95_ci": normal_ci,
        "bootstrap_mean_quality_score": bootstrap_mean,
        "bootstrap_95_ci": bootstrap_ci,
        "repeated_task_count": len(repeated_ranges),
        "repeated_task_consistency": repeated_task_consistency,
        "average_repeated_task_score_range": average_repeated_range,
        "average_per_task_variance": average_task_variance,
        "multi_day_trend": {
            "day_count": len(completed_dates),
            "dates": sorted(date for date in completed_dates if date),
            "quality_delta_first_to_last": quality_slope,
            "transport_delta_first_to_last": transport_slope,
        },
        "anomalies": anomalies,
    }


def build_protocol_fingerprint(
    *,
    campaign_dir_path: Path,
    runs_dir: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    gateway_provider: str,
    persist: bool,
) -> dict[str, Any]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    feature_metrics = record_feature_metrics(records)
    total_cases = int(metrics.get("total_cases") or feature_metrics["record_count"] or 0)
    protocol_rate = clamp01(numeric(metrics.get("protocol_compatibility_score")))
    consistency_rate = clamp01(numeric(metrics.get("model_name_consistency_rate")))
    returned_seen = int(metrics.get("model_returned_seen_count") or 0)
    returned_presence_rate = ratio(returned_seen, total_cases)
    usage_rate = clamp01(numeric(feature_metrics.get("usage_presence_rate")))
    stop_rate = clamp01(numeric(feature_metrics.get("stop_reason_presence_rate")))
    response_headers_rate = clamp01(numeric(feature_metrics.get("response_headers_presence_rate")))
    request_id_rate = clamp01(numeric(feature_metrics.get("request_id_presence_rate")))
    streaming_order_rate = clamp01(numeric(feature_metrics.get("streaming_event_order_presence_rate")))
    cache_usage_rate = clamp01(numeric(feature_metrics.get("cache_usage_field_presence_rate")))
    tool_use_rate = clamp01(numeric(feature_metrics.get("tool_use_presence_rate")))
    live_provider = summary.get("live_provider") is True

    component_values = [
        protocol_rate,
        consistency_rate,
        returned_presence_rate,
        usage_rate,
        stop_rate,
    ]
    available = [value for value in component_values if value is not None]
    score = round(sum(available) / len(available), 6) if available else None
    reasons: list[str] = []
    if not live_provider:
        reasons.append("dry_run_protocol_evidence")
    if protocol_rate is None:
        reasons.append("protocol_compatibility_missing")
    elif protocol_rate < 0.98:
        reasons.append("protocol_compatibility_below_0.98")
    if consistency_rate is None:
        reasons.append("model_name_consistency_missing")
    elif consistency_rate < 0.98:
        reasons.append("model_name_consistency_below_0.98")
    if returned_presence_rate is None or returned_presence_rate < 0.98:
        reasons.append("model_returned_presence_below_0.98")
    if usage_rate is None or usage_rate < 0.95:
        reasons.append("usage_presence_below_0.95")
    if stop_rate is None or stop_rate < 0.95:
        reasons.append("stop_reason_presence_below_0.95")
    if live_provider and (response_headers_rate is None or response_headers_rate <= 0):
        reasons.append("response_headers_missing")
    if live_provider and (request_id_rate is None or request_id_rate <= 0):
        reasons.append("request_id_missing")
    if live_provider and feature_metrics.get("json_probe_task_count") == 0:
        reasons.append("json_probe_missing")
    if live_provider and feature_metrics.get("long_context_probe_task_count") == 0:
        reasons.append("long_context_probe_missing")

    if not live_provider:
        decision = "REVIEW"
    elif protocol_rate is not None and protocol_rate < 0.95:
        decision = "NO-GO"
    elif consistency_rate is not None and consistency_rate < 0.98:
        decision = "NO-GO"
    else:
        decision = decision_from_score(score)

    tested = summary.get("tested_model") if isinstance(summary.get("tested_model"), dict) else {}
    fingerprint = {
        "schema_version": PROTOCOL_FINGERPRINT_SCHEMA_VERSION,
        "campaign_id": summary.get("campaign_id") or campaign_dir_path.name,
        "provider_id": tested.get("provider_id") or gateway_provider,
        "provider_label": gateway_provider,
        "protocol": tested.get("protocol"),
        "model": tested.get("model"),
        "live_provider": live_provider,
        "score": score,
        "decision": decision,
        "reasons": reasons,
        "checks": {
            "protocol_compatibility_score": protocol_rate,
            "model_name_consistency_rate": consistency_rate,
            "model_returned_presence_rate": returned_presence_rate,
            "usage_presence_rate": usage_rate,
            "stop_reason_presence_rate": stop_rate,
            "response_headers_presence_rate": response_headers_rate,
            "request_id_presence_rate": request_id_rate,
            "streaming_event_order_presence_rate": streaming_order_rate,
            "cache_usage_field_presence_rate": cache_usage_rate,
            "tool_use_presence_rate": tool_use_rate,
            "json_probe_task_count": feature_metrics.get("json_probe_task_count"),
            "long_context_probe_task_count": feature_metrics.get("long_context_probe_task_count"),
            "raw_event_type_presence_rate": feature_metrics.get("raw_event_type_presence_rate"),
        },
        "observed": {
            "total_cases": total_cases,
            "model_returned_seen_count": returned_seen,
            "model_returned_missing_count": metrics.get("model_returned_missing_count"),
            "stop_reason_counts": feature_metrics["stop_reason_counts"],
            "raw_event_type_counts": feature_metrics["raw_event_type_counts"],
            "category_counts": feature_metrics["category_counts"],
        },
        "evidence_status": "live_observed" if live_provider else "dry_run_reference_only",
    }
    if persist:
        provider_id = str(fingerprint.get("provider_id") or "provider").replace("/", "_").replace("\\", "_")
        write_json(campaign_dir_path / "protocol_fingerprints" / f"{provider_id}.json", fingerprint)
    return fingerprint


def build_baseline_comparison(
    *,
    campaign_dir_path: Path,
    runs_dir: Path,
    summary: dict[str, Any],
    baseline_campaign_dir: Path | None,
    baseline_provider: str,
    gateway_provider: str,
    persist: bool,
) -> dict[str, Any]:
    baseline_summary = None
    if baseline_campaign_dir is not None and baseline_campaign_dir.exists():
        baseline_summary = load_summary(baseline_campaign_dir)
        if baseline_summary is None:
            baseline_summary = summarize_campaign(baseline_campaign_dir, runs_dir, persist=False)

    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    live_provider = summary.get("live_provider") is True
    reasons: list[str] = []
    comparison_metrics: dict[str, Any] = {
        "quality_delta": None,
        "transport_success_delta": None,
        "model_response_success_delta": None,
        "p95_latency_ratio": None,
        "overlapping_task_count": 0,
        "average_abs_task_score_delta": None,
        "semantic_similarity": None,
        "format_similarity": None,
        "refusal_similarity": None,
        "usage_ratio": None,
    }
    baseline_source = "missing"
    score: float | None = None

    if baseline_summary is None:
        baseline_source = "synthetic_dry_run_reference" if not live_provider else "missing_official_baseline"
        reasons.append("baseline_campaign_missing")
        if not live_provider:
            reasons.append("dry_run_reference_only")
        decision = "REVIEW"
    else:
        baseline_source = "campaign"
        baseline_metrics = baseline_summary.get("metrics") if isinstance(baseline_summary.get("metrics"), dict) else {}
        quality = numeric(metrics.get("average_quality_score"))
        baseline_quality = numeric(baseline_metrics.get("average_quality_score"))
        transport = numeric(metrics.get("transport_success_rate"))
        baseline_transport = numeric(baseline_metrics.get("transport_success_rate"))
        model_success = numeric(metrics.get("model_response_success_rate"))
        baseline_model_success = numeric(baseline_metrics.get("model_response_success_rate"))
        p95 = numeric(metrics.get("p95_latency_ms"))
        baseline_p95 = numeric(baseline_metrics.get("p95_latency_ms"))

        if quality is not None and baseline_quality is not None:
            comparison_metrics["quality_delta"] = round(quality - baseline_quality, 3)
        if transport is not None and baseline_transport is not None:
            comparison_metrics["transport_success_delta"] = round(transport - baseline_transport, 6)
        if model_success is not None and baseline_model_success is not None:
            comparison_metrics["model_response_success_delta"] = round(model_success - baseline_model_success, 6)
        if p95 is not None and baseline_p95 and baseline_p95 > 0:
            comparison_metrics["p95_latency_ratio"] = round(p95 / baseline_p95, 6)

        task_scores = record_task_scores(summary)
        baseline_task_scores = record_task_scores(baseline_summary)
        overlap = sorted(set(task_scores) & set(baseline_task_scores))
        deltas = [abs(task_scores[task_id] - baseline_task_scores[task_id]) for task_id in overlap]
        comparison_metrics["overlapping_task_count"] = len(overlap)
        if deltas:
            comparison_metrics["average_abs_task_score_delta"] = round(sum(deltas) / len(deltas), 3)

        components: list[float] = []
        if comparison_metrics["quality_delta"] is not None:
            components.append(max(0.0, 1.0 - min(abs(float(comparison_metrics["quality_delta"])) / 4.0, 1.0)))
        if comparison_metrics["transport_success_delta"] is not None:
            components.append(max(0.0, 1.0 - min(abs(float(comparison_metrics["transport_success_delta"])) / 0.25, 1.0)))
        if comparison_metrics["model_response_success_delta"] is not None:
            components.append(max(0.0, 1.0 - min(abs(float(comparison_metrics["model_response_success_delta"])) / 0.25, 1.0)))
        if comparison_metrics["p95_latency_ratio"] is not None:
            ratio_value = float(comparison_metrics["p95_latency_ratio"])
            components.append(max(0.0, 1.0 - min(abs(ratio_value - 1.0) / 3.0, 1.0)))
        if comparison_metrics["average_abs_task_score_delta"] is not None:
            components.append(max(0.0, 1.0 - min(float(comparison_metrics["average_abs_task_score_delta"]) / 4.0, 1.0)))
        score = round(sum(components) / len(components), 6) if components else None
        decision = decision_from_score(score, go_threshold=0.82, review_threshold=0.62)
        if comparison_metrics["overlapping_task_count"] == 0:
            decision = worst_decision(decision, "REVIEW")
            reasons.append("no_overlapping_task_scores")
        if comparison_metrics["semantic_similarity"] is None:
            reasons.append("semantic_similarity_not_available_without_raw_or_embeddings")
        if comparison_metrics["usage_ratio"] is None:
            reasons.append("usage_ratio_not_available")

    comparison = {
        "schema_version": BASELINE_COMPARISON_SCHEMA_VERSION,
        "campaign_id": summary.get("campaign_id") or campaign_dir_path.name,
        "baseline_provider": baseline_provider,
        "gateway_provider": gateway_provider,
        "baseline_source": baseline_source,
        "baseline_campaign_id": baseline_summary.get("campaign_id") if isinstance(baseline_summary, dict) else None,
        "score": score,
        "decision": decision,
        "reasons": reasons,
        "metrics": comparison_metrics,
        "note": "black-box API evidence cannot prove upstream identity absolutely",
    }
    if persist:
        write_json(campaign_dir_path / "baseline_comparisons" / "baseline_comparison.json", comparison)
    return comparison


def build_auditability(summary: dict[str, Any], records: list[dict[str, Any]], protocol_fingerprint: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    tested = summary.get("tested_model") if isinstance(summary.get("tested_model"), dict) else {}
    feature_metrics = record_feature_metrics(records)
    total_cases = int(metrics.get("total_cases") or feature_metrics["record_count"] or 0)
    model_returned_seen = int(metrics.get("model_returned_seen_count") or 0)
    model_returned_rate = ratio(model_returned_seen, total_cases)
    request_hash_rate = numeric(feature_metrics.get("request_hash_presence_rate"))
    event_path_rate = numeric(feature_metrics.get("event_path_presence_rate"))
    raw_event_rate = numeric(feature_metrics.get("raw_event_type_presence_rate"))
    upstream_id_rate = numeric(feature_metrics.get("upstream_request_id_presence_rate"))
    gateway_route_rate = numeric(feature_metrics.get("gateway_route_id_presence_rate"))
    retry_count_rate = numeric(feature_metrics.get("retry_count_presence_rate"))
    gateway_processing_rate = numeric(feature_metrics.get("gateway_processing_ms_presence_rate"))
    key_fingerprint_present = bool(tested.get("key_fingerprint")) if summary.get("live_provider") is True else True

    components = {
        "model_returned": (model_returned_rate, 0.20),
        "request_hash": (request_hash_rate, 0.15),
        "event_paths": (event_path_rate, 0.10),
        "raw_event_types": (raw_event_rate, 0.10),
        "upstream_request_id": (upstream_id_rate, 0.20),
        "gateway_route_id": (gateway_route_rate, 0.10),
        "retry_count": (retry_count_rate, 0.05),
        "gateway_processing_ms": (gateway_processing_rate, 0.05),
        "key_fingerprint": (1.0 if key_fingerprint_present else 0.0, 0.05),
    }
    score = 0.0
    for value, weight in components.values():
        score += (clamp01(value) or 0.0) * weight
    score = round(score, 6)
    reasons: list[str] = []
    if model_returned_rate is None or model_returned_rate < 0.98:
        reasons.append("model_returned_audit_gap")
    if raw_event_rate is None or raw_event_rate < 1.0:
        reasons.append("raw_event_type_coverage_incomplete")
    if upstream_id_rate is None or upstream_id_rate <= 0:
        reasons.append("upstream_request_id_missing")
    if gateway_route_rate is None or gateway_route_rate <= 0:
        reasons.append("gateway_route_id_missing")
    if gateway_processing_rate is None or gateway_processing_rate <= 0:
        reasons.append("gateway_processing_ms_missing")
    if summary.get("live_provider") is not True:
        reasons.append("dry_run_auditability_reference")
    if protocol_fingerprint.get("decision") != "GO":
        reasons.append("protocol_fingerprint_not_go")

    decision = decision_from_score(score, go_threshold=0.80, review_threshold=0.50)
    if summary.get("live_provider") is not True:
        decision = worst_decision(decision, "REVIEW")
    return {
        "score": score,
        "decision": decision,
        "reasons": reasons,
        "components": {
            name: {"value": value, "weight": weight}
            for name, (value, weight) in components.items()
        },
        "gateway_audit_fields": {
            "gateway_route_id_presence_rate": feature_metrics.get("gateway_route_id_presence_rate"),
            "fallback_used_presence_rate": feature_metrics.get("fallback_used_presence_rate"),
            "retry_count_presence_rate": feature_metrics.get("retry_count_presence_rate"),
            "cache_hit_presence_rate": feature_metrics.get("cache_hit_presence_rate"),
            "gateway_processing_ms_presence_rate": feature_metrics.get("gateway_processing_ms_presence_rate"),
            "field_counts": feature_metrics.get("gateway_audit_field_counts") or {},
        },
    }


def quality_score_from_summary(summary: dict[str, Any]) -> float | None:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    avg_quality = numeric(metrics.get("average_quality_score"))
    if avg_quality is None:
        return None
    return round(max(0.0, min(avg_quality, 10.0)) / 10.0, 6)


def gateway_score_from_summary(summary: dict[str, Any]) -> float | None:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    transport = clamp01(numeric(metrics.get("transport_success_rate")))
    p95 = numeric(metrics.get("p95_latency_ms"))
    if transport is None and p95 is None:
        return None
    if p95 is None:
        latency_component = 0.5
    elif p95 <= 15000:
        latency_component = 1.0
    elif p95 <= 60000:
        latency_component = 0.5
    else:
        latency_component = 0.0
    return round((transport or 0.0) * 0.7 + latency_component * 0.3, 6)


def write_authenticity_evidence(
    campaign_dir_path: Path,
    runs_dir: Path,
    *,
    baseline_campaign_dir: Path | None = None,
    baseline_provider: str = "official_baseline",
    gateway_provider: str = "gateway_candidate",
    persist: bool = True,
) -> dict[str, Any]:
    summary = load_summary(campaign_dir_path)
    if summary is None:
        summary = summarize_campaign(campaign_dir_path, runs_dir, persist=persist)
    records = campaign_records(campaign_dir_path, runs_dir)
    protocol_fingerprint = build_protocol_fingerprint(
        campaign_dir_path=campaign_dir_path,
        runs_dir=runs_dir,
        summary=summary,
        records=records,
        gateway_provider=gateway_provider,
        persist=persist,
    )
    baseline_comparison = build_baseline_comparison(
        campaign_dir_path=campaign_dir_path,
        runs_dir=runs_dir,
        summary=summary,
        baseline_campaign_dir=baseline_campaign_dir,
        baseline_provider=baseline_provider,
        gateway_provider=gateway_provider,
        persist=persist,
    )
    auditability = build_auditability(summary, records, protocol_fingerprint)
    statistical = statistical_confidence(summary)
    decisions = summary.get("decisions") if isinstance(summary.get("decisions"), dict) else {}
    model_quality_decision = str(decisions.get("model_confidence_decision") or "REVIEW")
    gateway_reliability_decision = str(decisions.get("gateway_reliability_decision") or "REVIEW")
    protocol_decision = str(protocol_fingerprint.get("decision") or "REVIEW")
    baseline_decision = str(baseline_comparison.get("decision") or "REVIEW")
    auditability_decision = str(auditability.get("decision") or "REVIEW")
    overall = worst_decision(
        model_quality_decision,
        gateway_reliability_decision,
        protocol_decision,
        baseline_decision,
        auditability_decision,
    )
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    model_quality_score = quality_score_from_summary(summary)
    gateway_reliability_score = gateway_score_from_summary(summary)
    available_scores = [
        score
        for score in (
            model_quality_score,
            gateway_reliability_score,
            numeric(protocol_fingerprint.get("score")),
            numeric(baseline_comparison.get("score")),
            numeric(auditability.get("score")),
        )
        if score is not None
    ]
    trust_score = round(sum(available_scores) / len(available_scores), 6) if available_scores else None
    if overall == "NO-GO" and trust_score is not None:
        trust_score = min(trust_score, 0.59)
    elif overall == "REVIEW" and trust_score is not None:
        trust_score = min(trust_score, 0.84)

    authenticity = {
        "schema_version": AUTHENTICITY_SCHEMA_VERSION,
        "campaign_id": summary.get("campaign_id") or campaign_dir_path.name,
        "status": summary.get("status"),
        "created_at": summary.get("created_at"),
        "completed_at": summary.get("completed_at"),
        "latest_tested_at": metrics.get("latest_tested_at"),
        "live_provider": summary.get("live_provider") is True,
        "tested_model": summary.get("tested_model"),
        "judge_model": summary.get("judge_model"),
        "baseline_provider": baseline_provider,
        "gateway_provider": gateway_provider,
        "metrics": {
            "model_quality_score": model_quality_score,
            "gateway_reliability_score": gateway_reliability_score,
            "protocol_fingerprint_score": protocol_fingerprint.get("score"),
            "baseline_similarity_score": baseline_comparison.get("score"),
            "auditability_score": auditability.get("score"),
            "overall_trust_score": trust_score,
            "total_cases": metrics.get("total_cases"),
            "completed_runs": metrics.get("completed_runs"),
            "model_response_success_rate": metrics.get("model_response_success_rate"),
            "transport_success_rate": metrics.get("transport_success_rate"),
            "average_quality_score": metrics.get("average_quality_score"),
            "model_name_consistency_rate": metrics.get("model_name_consistency_rate"),
            "protocol_compatibility_score": metrics.get("protocol_compatibility_score"),
            "p50_latency_ms": metrics.get("p50_latency_ms"),
            "p95_latency_ms": metrics.get("p95_latency_ms"),
            "error_counts": metrics.get("error_counts") or {},
            "statistical_confidence": statistical,
        },
        "decisions": {
            "model_quality_decision": model_quality_decision,
            "gateway_reliability_decision": gateway_reliability_decision,
            "protocol_fingerprint_decision": protocol_decision,
            "baseline_similarity_decision": baseline_decision,
            "auditability_decision": auditability_decision,
            "overall_trust_decision": overall,
        },
        "reasons": {
            "model_quality": decisions.get("model_confidence_reasons") or [],
            "gateway_reliability": decisions.get("gateway_reliability_reasons") or [],
            "protocol_fingerprint": protocol_fingerprint.get("reasons") or [],
            "baseline_similarity": baseline_comparison.get("reasons") or [],
            "auditability": auditability.get("reasons") or [],
        },
        "evidence": {
            "protocol_fingerprint": protocol_fingerprint,
            "baseline_comparison": baseline_comparison,
            "auditability": auditability,
            "statistical_confidence": statistical,
            "evidence_status": "live_observed" if summary.get("live_provider") is True else "dry_run_reference_only",
            "limitations": [
                "black-box API evidence cannot prove upstream identity absolutely",
                "missing upstream request ids reduce auditability but do not prove model substitution",
            ],
        },
    }
    if persist:
        write_json(campaign_dir_path / "authenticity_summary.json", authenticity)
    return authenticity


def load_or_build_authenticity(
    campaign_dir_path: Path,
    runs_dir: Path,
    *,
    persist: bool = False,
) -> dict[str, Any]:
    path = campaign_dir_path / "authenticity_summary.json"
    if path.exists():
        return read_json(path)
    return write_authenticity_evidence(campaign_dir_path, runs_dir, persist=persist)


def build_config_protocol_fingerprint(
    provider: dict[str, Any],
    *,
    provider_label: str,
    live: bool = False,
) -> dict[str, Any]:
    protocol = provider.get("protocol")
    reasons = ["config_only_fingerprint"]
    checks = {
        "protocol_configured": protocol in {"openai_chat", "anthropic_messages"},
        "auth_type_configured": bool(provider.get("auth_type")),
        "model_configured": bool(provider.get("model")),
        "base_url_host_configured": bool(provider.get("base_url_host") or provider.get("base_url")),
    }
    score = sum(1 for ok in checks.values() if ok) / len(checks)
    decision = "REVIEW" if not live else decision_from_score(score)
    return {
        "schema_version": PROTOCOL_FINGERPRINT_SCHEMA_VERSION,
        "provider_id": provider.get("provider_id"),
        "provider_label": provider_label,
        "protocol": protocol,
        "model": provider.get("model"),
        "live_provider": live,
        "score": round(score, 6),
        "decision": decision,
        "reasons": reasons,
        "checks": checks,
        "observed": {},
        "evidence_status": "config_only",
    }


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        campaigns_dir = root / "campaigns"
        runs_dir = root / "runs"
        campaign_id = "CMP-AUTH-SELFTEST"
        camp_dir = campaigns_dir / campaign_id
        run_dir = runs_dir / f"{campaign_id}-R01"
        write_json(
            camp_dir / "campaign.json",
            {
                "schema_version": "campaign_v1",
                "campaign_id": campaign_id,
                "status": "completed",
                "created_at": "2026-01-01T00:00:00",
                "completed_at": "2026-01-01T00:01:00",
                "live_provider": False,
                "tested_model": {"provider_id": "tested", "model": "dry", "protocol": "openai_chat"},
                "judge_model": {"provider_id": "judge", "model": "dry-judge"},
                "benchmark_version": "self:test",
                "benchmark_mode": "self",
                "quality_gate_version": "self",
                "score_formula_version": "self",
            },
        )
        write_json(
            camp_dir / "run_ids.json",
            {
                "campaign_id": campaign_id,
                "runs": [
                    {
                        "round": 1,
                        "attempt": 1,
                        "run_id": f"{campaign_id}-R01",
                        "status": "completed",
                        "started_at": "2026-01-01T00:00:00",
                        "completed_at": "2026-01-01T00:01:00",
                    }
                ],
            },
        )
        write_json(
            run_dir / "state.json",
            {
                "job_id": f"{campaign_id}-R01",
                "status": "completed",
                "started_at": "2026-01-01T00:00:00",
                "completed_at": "2026-01-01T00:01:00",
                "final_decision": "GO",
            },
        )
        record = {
            "task": {"id": "task_1", "category": "json"},
            "provider": {"api_style": "openai_chat", "model_requested": "dry", "model_returned": "dry"},
            "request": {"request_hash": "abc"},
            "response": {"content_chars": 12, "events_file": str(run_dir / "events.jsonl")},
            "telemetry": {"ok": True, "first_content_token_ms": 10, "total_ms": 20, "stop_reason": "stop"},
            "usage": {"input_tokens": 4, "output_tokens": 3},
            "scoring": {"final_score": {"score": 8.0, "details": "ok"}},
            "trace": {"raw_event_types": ["dry_completion"]},
        }
        (run_dir / "run_records.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_records.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        evidence = write_authenticity_evidence(camp_dir, runs_dir)
        assert evidence["schema_version"] == AUTHENTICITY_SCHEMA_VERSION
        assert evidence["decisions"]["overall_trust_decision"] in {"GO", "REVIEW", "NO-GO"}
        assert (camp_dir / "authenticity_summary.json").exists()
        assert (camp_dir / "protocol_fingerprints" / "tested.json").exists()
        assert (camp_dir / "baseline_comparisons" / "baseline_comparison.json").exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="authenticity evidence helpers")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("authenticity self-test ok")
        return 0
    parser.error("--self-test is required when running this module directly")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
