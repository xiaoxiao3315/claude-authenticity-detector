from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import archive_registry as archives
import evidence_registry as registry


QUALITY_GATE_RECORD_VERSION = "quality_gate_record_v1"
QUALITY_GATE_POLICY_VERSION = "quality_gate_policy_v1"
DEFAULT_POLICY_ID = "provider_release_v1"

GO = "GO"
REVIEW = "REVIEW"
NO_GO = "NO-GO"


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
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def archive_registry_path(runs_dir: Path) -> Path:
    return runs_dir.parent / "archives" / "archive_registry.json"


def archived_evidence(source_type: str, runs_dir: Path, run_id: str | None, evidence_id: str | None) -> bool:
    return archives.is_archived(archive_registry_path(runs_dir), source_type, run_id=run_id, evidence_id=evidence_id)


def archive_warning(source_type: str, evidence_id: str | None) -> str:
    return f"explicit archived {source_type} evidence was used: {evidence_id}"


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
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def score_value(score: Any) -> float | None:
    if isinstance(score, dict):
        return numeric(score.get("score"))
    return numeric(score)


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(value for value in values if value is not None and value >= 0)
    if not clean:
        return None
    index = max(0, min(len(clean) - 1, math.ceil((pct / 100.0) * len(clean)) - 1))
    return round(clean[index], 3)


def load_policy(policy_path: Path, policy_id: str | None = None) -> dict[str, Any]:
    data = read_json(policy_path)
    wanted = policy_id or DEFAULT_POLICY_ID
    if data.get("policy_id") == wanted:
        policy = dict(data)
    else:
        policies = data.get("policies") or []
        policy = next((dict(item) for item in policies if item.get("policy_id") == wanted), None)
        if policy is None:
            raise ValueError(f"policy not found: {wanted}")
    policy.setdefault("policy_id", wanted)
    policy.setdefault("policy_version", data.get("policy_version") or QUALITY_GATE_POLICY_VERSION)
    policy.setdefault("thresholds", {})
    return policy


def split_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [part.strip() for part in str(value).replace(",", ";").split(";") if part.strip()]


def is_json_sample(sample: dict[str, Any]) -> bool:
    fields = [
        sample.get("category"),
        sample.get("enterprise_dimension"),
        sample.get("scoring_type"),
        " ".join(split_tags(sample.get("risk_tags"))),
    ]
    haystack = " ".join(str(item or "") for item in fields).lower()
    return "json" in haystack or "schema" in haystack or "structured" in haystack


def sample_from_run_record(record: dict[str, Any]) -> dict[str, Any]:
    task = record.get("task") or {}
    provider = record.get("provider") or {}
    telemetry = record.get("telemetry") or {}
    scoring = record.get("scoring") or {}
    final_score = scoring.get("final_score") if isinstance(scoring.get("final_score"), dict) else {}
    judge_score = scoring.get("judge_score") if isinstance(scoring.get("judge_score"), dict) else {}
    return {
        "source_kind": "run_record",
        "source_record_id": record.get("record_id"),
        "provider_id": provider.get("id"),
        "task_id": task.get("id"),
        "category": task.get("category"),
        "enterprise_dimension": task.get("enterprise_dimension"),
        "scoring_type": task.get("scoring_type"),
        "risk_tags": task.get("risk_tags") or [],
        "ok": boolish(telemetry.get("ok")),
        "error": telemetry.get("error"),
        "score_0_10": score_value(final_score),
        "format_ok": boolish(final_score.get("format_ok")) if isinstance(final_score, dict) else None,
        "judge_error": judge_score.get("error") if isinstance(judge_score, dict) else None,
        "stop_reason": telemetry.get("stop_reason"),
        "first_content_token_ms": numeric(telemetry.get("first_content_token_ms")),
        "model_requested": provider.get("model_requested"),
        "model_returned": provider.get("model_returned"),
        "scoring_source": "original",
    }


def sample_from_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_kind": "summary_csv",
        "source_record_id": f"{row.get('run_id')}:{row.get('provider')}:{row.get('task_id')}",
        "provider_id": row.get("provider"),
        "task_id": row.get("task_id"),
        "category": row.get("category"),
        "enterprise_dimension": row.get("enterprise_dimension"),
        "scoring_type": row.get("scoring_type"),
        "risk_tags": split_tags(row.get("risk_tags")),
        "ok": boolish(row.get("ok")),
        "error": row.get("error"),
        "score_0_10": numeric(row.get("quality_0_10"), numeric(row.get("score_0_10"))),
        "format_ok": boolish(row.get("format_ok")),
        "judge_error": row.get("judge_error") or None,
        "stop_reason": row.get("stop_reason"),
        "first_content_token_ms": numeric(row.get("first_content_token_ms")),
        "model_requested": row.get("model_requested"),
        "model_returned": row.get("model_returned"),
        "scoring_source": "original",
    }


def sample_from_result_item(item: dict[str, Any]) -> dict[str, Any]:
    task = item.get("task") or {}
    provider = item.get("provider") or {}
    metrics = item.get("metrics") or {}
    score = item.get("score") if isinstance(item.get("score"), dict) else {}
    return {
        "source_kind": "results_json",
        "source_record_id": f"{item.get('run_id')}:{provider.get('id')}:{task.get('id')}",
        "provider_id": provider.get("id"),
        "task_id": task.get("id"),
        "category": task.get("category"),
        "enterprise_dimension": task.get("enterprise_dimension"),
        "scoring_type": task.get("scoring_type"),
        "risk_tags": task.get("risk_tags") or [],
        "ok": boolish(metrics.get("ok")),
        "error": metrics.get("error"),
        "score_0_10": score_value(score),
        "format_ok": boolish(score.get("format_ok")) if isinstance(score, dict) else None,
        "judge_error": None,
        "stop_reason": metrics.get("stop_reason"),
        "first_content_token_ms": numeric(metrics.get("first_content_token_ms")),
        "model_requested": provider.get("model"),
        "model_returned": metrics.get("server_model"),
        "scoring_source": "original",
    }


def load_run_samples(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    refs: dict[str, str] = {}
    run_records_path = run_dir / "run_records.jsonl"
    summary_path = run_dir / "summary.csv"
    results_path = run_dir / "results.json"
    if run_records_path.exists():
        refs["run_records_file"] = str(run_records_path)
        return [sample_from_run_record(record) for record in read_jsonl(run_records_path)], refs
    if summary_path.exists():
        refs["summary_file"] = str(summary_path)
        return [sample_from_summary_row(row) for row in read_csv_rows(summary_path)], refs
    if results_path.exists():
        refs["results_file"] = str(results_path)
        results = read_json(results_path)
        if not isinstance(results, list):
            raise ValueError(f"{results_path} must contain a JSON array")
        return [sample_from_result_item(item) for item in results if isinstance(item, dict)], refs
    return [], refs


def load_benchmark_provider_scores(run_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = run_dir / "benchmark_scores.json"
    if not path.exists():
        return {}, {}
    data = read_json(path)
    if not isinstance(data, dict):
        return {}, {"benchmark_scores_file": str(path)}
    providers = data.get("providers") if isinstance(data.get("providers"), dict) else {}
    return providers or {}, {"benchmark_scores_file": str(path)}


def compatibility_refs(run_dir: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    manifest_path = run_dir / "compatibility_manifest.json"
    records_path = run_dir / "compatibility_records.jsonl"
    summary_path = run_dir / "compatibility_summary.csv"
    if manifest_path.exists():
        refs["compatibility_manifest_file"] = str(manifest_path)
    if records_path.exists():
        refs["compatibility_records_file"] = str(records_path)
    if summary_path.exists():
        refs["compatibility_summary_file"] = str(summary_path)
    return refs


def load_compatibility_evidence(
    runs_dir: Path,
    provider_id: str,
    compatibility_run_id: str | None = None,
) -> dict[str, Any]:
    if not runs_dir.exists():
        return {"found": False, "run_id": None, "error": "runs directory not found", "refs": {}}

    def matches_compatibility_manifest(manifest: dict[str, Any]) -> bool:
        manifest_run_id = str(manifest.get("run_id") or "")
        if manifest.get("provider_id") != provider_id:
            return False
        if not compatibility_run_id and manifest_run_id and archived_evidence("compatibility_run", runs_dir, manifest_run_id, manifest_run_id):
            return False
        return True

    selection = registry.select_manifest_dir(
        base_dir=runs_dir,
        manifest_name="compatibility_manifest.json",
        explicit_id=compatibility_run_id,
        label="compatibility run",
        id_key="run_id",
        match_manifest=matches_compatibility_manifest if not compatibility_run_id else lambda manifest: manifest.get("provider_id") == provider_id,
        mismatch_error="compatibility provider mismatch",
    )
    if not selection.path:
        return {
            "found": False,
            "run_id": selection.expected_id or compatibility_run_id,
            "error": selection.error or ("compatibility run not found" if compatibility_run_id else "matching compatibility run not found"),
            "manifest": selection.manifest,
            "refs": compatibility_refs(runs_dir / selection.expected_id) if selection.expected_id else {},
        }
    manifest = selection.manifest or {}
    warnings = []
    selected_run_id = str(manifest.get("run_id") or selection.path.name)
    if compatibility_run_id and archived_evidence("compatibility_run", runs_dir, selected_run_id, selected_run_id):
        warnings.append(archive_warning("compatibility_run", selected_run_id))
    return {
        "found": True,
        "run_id": selected_run_id,
        "manifest": manifest,
        "records": read_jsonl(selection.path / "compatibility_records.jsonl"),
        "refs": compatibility_refs(selection.path),
        "warnings": warnings,
        "archive_warning": warnings[0] if warnings else None,
    }


def load_rescore_evidence(run_dir: Path, rescore_id: str | None, provider_id: str | None = None) -> dict[str, Any]:
    def matches_manifest(manifest: dict[str, Any]) -> bool:
        manifest_run_id = manifest.get("source_run_id")
        if manifest_run_id and str(manifest_run_id) != run_dir.name:
            return False
        manifest_rescore_id = str(manifest.get("rescore_id") or "")
        if not rescore_id and manifest_rescore_id and archived_evidence("rescore", run_dir.parent, run_dir.name, manifest_rescore_id):
            return False
        filters = manifest.get("filters") if isinstance(manifest.get("filters"), dict) else {}
        filter_provider_id = filters.get("provider_id")
        return not provider_id or not filter_provider_id or str(filter_provider_id) == str(provider_id)

    selection = registry.select_manifest_dir(
        base_dir=run_dir / "rescores",
        manifest_name="rescore_manifest.json",
        explicit_id=rescore_id,
        label="rescore",
        id_key="rescore_id",
        match_manifest=matches_manifest if not rescore_id else None,
        mismatch_error="rescore provider/source mismatch",
    )
    if not selection.path:
        return {
            "found": False,
            "requested": bool(rescore_id),
            "rescore_id": rescore_id,
            "error": selection.error,
            "records": [],
            "by_record_id": {},
            "refs": {},
        }
    rescore_dir = selection.path
    manifest_path = rescore_dir / "rescore_manifest.json"
    records_path = rescore_dir / "rescore_records.jsonl"
    if not manifest_path.exists() or not records_path.exists():
        return {
            "found": False,
            "requested": bool(rescore_id),
            "rescore_id": rescore_id or rescore_dir.name,
            "error": "rescore not found",
            "records": [],
            "by_record_id": {},
            "refs": {},
        }
    records = read_jsonl(records_path)
    by_record_id = {
        str(record.get("source_record_id")): record
        for record in records
        if record.get("source_record_id")
    }
    return {
        "found": True,
        "requested": bool(rescore_id),
        "rescore_id": (selection.manifest or {}).get("rescore_id") or rescore_dir.name,
        "manifest": selection.manifest or read_json(manifest_path),
        "records": records,
        "by_record_id": by_record_id,
        "warnings": [archive_warning("rescore", rescore_id or rescore_dir.name)] if rescore_id and archived_evidence("rescore", run_dir.parent, run_dir.name, rescore_id) else [],
        "archive_warning": archive_warning("rescore", rescore_id) if rescore_id and archived_evidence("rescore", run_dir.parent, run_dir.name, rescore_id) else None,
        "refs": {
            "rescore_manifest_file": str(manifest_path),
            "rescore_records_file": str(records_path),
            "rescore_summary_file": str(rescore_dir / "rescore_summary.csv"),
        },
    }


def trace_evaluation_refs(eval_dir: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    manifest_path = eval_dir / "trace_eval_manifest.json"
    records_path = eval_dir / "trace_eval_records.jsonl"
    summary_path = eval_dir / "trace_eval_summary.csv"
    if manifest_path.exists():
        refs["trace_eval_manifest_file"] = str(manifest_path)
    if records_path.exists():
        refs["trace_eval_records_file"] = str(records_path)
    if summary_path.exists():
        refs["trace_eval_summary_file"] = str(summary_path)
    return refs


def trace_evaluation_from_dir(eval_dir: Path, provider_id: str, trace_eval_id: str | None) -> dict[str, Any] | None:
    manifest_path = eval_dir / "trace_eval_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    provider_metrics = manifest.get("provider_metrics") if isinstance(manifest.get("provider_metrics"), dict) else {}
    metrics = provider_metrics.get(provider_id) if isinstance(provider_metrics.get(provider_id), dict) else None
    if not metrics:
        if trace_eval_id:
            return {
                "found": False,
                "requested": True,
                "trace_eval_id": trace_eval_id,
                "error": "trace evaluation provider mismatch",
                "manifest": manifest,
                "refs": trace_evaluation_refs(eval_dir),
            }
        return None
    return {
        "found": True,
        "requested": bool(trace_eval_id),
        "trace_eval_id": manifest.get("trace_eval_id") or eval_dir.name,
        "manifest": manifest,
        "provider_metrics": metrics,
        "refs": trace_evaluation_refs(eval_dir),
    }


def load_trace_evaluation_evidence(
    run_dir: Path,
    provider_id: str,
    trace_eval_id: str | None = None,
) -> dict[str, Any]:
    evals_dir = run_dir / "trace_evaluations"
    if not evals_dir.exists():
        return {
            "found": False,
            "requested": bool(trace_eval_id),
            "trace_eval_id": trace_eval_id,
            "error": "trace evaluation not found" if trace_eval_id else None,
            "refs": {},
        }

    def matches_manifest(manifest: dict[str, Any]) -> bool:
        manifest_run_id = manifest.get("source_run_id")
        if manifest_run_id and str(manifest_run_id) != run_dir.name:
            return False
        manifest_trace_id = str(manifest.get("trace_eval_id") or "")
        if not trace_eval_id and manifest_trace_id and archived_evidence("trace_evaluation", run_dir.parent, run_dir.name, manifest_trace_id):
            return False
        provider_metrics = manifest.get("provider_metrics") if isinstance(manifest.get("provider_metrics"), dict) else {}
        return bool(provider_metrics.get(provider_id))

    selection = registry.select_manifest_dir(
        base_dir=evals_dir,
        manifest_name="trace_eval_manifest.json",
        explicit_id=trace_eval_id,
        label="trace evaluation",
        id_key="trace_eval_id",
        match_manifest=matches_manifest,
        mismatch_error="trace evaluation provider mismatch",
    )
    if selection.path:
        result = trace_evaluation_from_dir(selection.path, provider_id, trace_eval_id)
        if result is not None:
            if trace_eval_id and archived_evidence("trace_evaluation", run_dir.parent, run_dir.name, trace_eval_id):
                warning = archive_warning("trace_evaluation", trace_eval_id)
                result["warnings"] = [warning]
                result["archive_warning"] = warning
            return result
    return {
        "found": False,
        "requested": bool(trace_eval_id),
        "trace_eval_id": trace_eval_id,
        "error": selection.error or ("trace evaluation not found" if trace_eval_id else None),
        "manifest": selection.manifest,
        "refs": trace_evaluation_refs(selection.path) if selection.path else {},
    }


def apply_rescore(samples: list[dict[str, Any]], rescore: dict[str, Any], provider_id: str) -> tuple[list[dict[str, Any]], int]:
    if not rescore.get("found"):
        return [dict(sample) for sample in samples], 0
    by_record_id = rescore.get("by_record_id") or {}
    out: list[dict[str, Any]] = []
    applied = 0
    for sample in samples:
        item = dict(sample)
        source_record_id = str(item.get("source_record_id") or "")
        rescore_record = by_record_id.get(source_record_id)
        if rescore_record and str(rescore_record.get("source_provider_id") or provider_id) == provider_id:
            final_score = rescore_record.get("new_final_score") if isinstance(rescore_record.get("new_final_score"), dict) else {}
            new_score = score_value(final_score)
            if new_score is not None:
                item["score_0_10"] = new_score
                item["format_ok"] = boolish(final_score.get("format_ok"))
                item["scoring_source"] = "rescore"
                item["rescore_error"] = rescore_record.get("rescore_error")
                item["judge_error"] = rescore_record.get("judge_error") or item.get("judge_error")
                applied += 1
        out.append(item)
    return out, applied


def provider_ids_from_evidence(samples: list[dict[str, Any]], provider_scores: dict[str, Any], provider_id: str | None) -> list[str]:
    if provider_id:
        return [provider_id]
    ids = {str(sample.get("provider_id")) for sample in samples if sample.get("provider_id")}
    ids.update(str(key) for key in provider_scores.keys())
    return sorted(ids)


def aggregate_metrics(
    *,
    samples: list[dict[str, Any]],
    provider_score: dict[str, Any] | None,
    compatibility: dict[str, Any],
    trace_evaluation: dict[str, Any],
    rescore: dict[str, Any],
    rescore_applied_count: int,
) -> dict[str, Any]:
    sample_count = len(samples)
    ok_count = sum(1 for sample in samples if sample.get("ok") is True)
    failure_count = sample_count - ok_count
    json_samples = [sample for sample in samples if is_json_sample(sample)]
    json_failures = [
        sample
        for sample in json_samples
        if sample.get("ok") is not True or sample.get("format_ok") is False
    ]
    judge_failure_count = sum(1 for sample in samples if sample.get("judge_error"))
    max_tokens_count = sum(1 for sample in samples if sample.get("stop_reason") == "max_tokens")
    model_mismatch_count = sum(
        1
        for sample in samples
        if sample.get("model_requested")
        and sample.get("model_returned")
        and sample.get("model_requested") != sample.get("model_returned")
    )
    model_returned_missing_count = sum(
        1 for sample in samples if sample.get("model_requested") and not sample.get("model_returned")
    )
    latency_values = [
        value
        for value in (numeric(sample.get("first_content_token_ms")) for sample in samples)
        if value is not None
    ]
    scores = [numeric(sample.get("score_0_10")) for sample in samples]
    scores = [value for value in scores if value is not None]
    effective_score_1000 = round((sum(scores) / len(scores)) * 100.0, 2) if scores else None
    benchmark_score = numeric((provider_score or {}).get("benchmark_score")) if provider_score else None
    rescore_coverage = ratio(rescore_applied_count, sample_count)
    if rescore_applied_count and rescore_coverage == 1.0:
        gate_score = effective_score_1000
        gate_score_source = "rescore_full_coverage"
    elif benchmark_score is not None:
        gate_score = benchmark_score
        gate_score_source = "benchmark_scores"
    else:
        gate_score = effective_score_1000
        gate_score_source = "sample_scores"
    manifest = compatibility.get("manifest") or {}
    rescore_manifest = rescore.get("manifest") if isinstance(rescore.get("manifest"), dict) else {}
    trace_metrics = trace_evaluation.get("provider_metrics") if isinstance(trace_evaluation.get("provider_metrics"), dict) else {}
    trace_manifest = trace_evaluation.get("manifest") if isinstance(trace_evaluation.get("manifest"), dict) else {}
    trace_record_count = numeric(trace_metrics.get("record_count"))
    return {
        "sample_count": sample_count,
        "ok_count": ok_count,
        "failure_count": failure_count,
        "success_rate": ratio(ok_count, sample_count),
        "json_task_count": len(json_samples),
        "json_failure_count": len(json_failures),
        "json_failure_rate": ratio(len(json_failures), len(json_samples)),
        "judge_failure_count": judge_failure_count,
        "judge_failure_rate": ratio(judge_failure_count, sample_count),
        "max_tokens_count": max_tokens_count,
        "max_tokens_rate": ratio(max_tokens_count, sample_count),
        "model_mismatch_count": model_mismatch_count,
        "model_returned_missing_count": model_returned_missing_count,
        "p95_first_content_token_ms": percentile(latency_values, 95),
        "average_score_0_10": round(sum(scores) / len(scores), 3) if scores else None,
        "effective_score_1000": effective_score_1000,
        "benchmark_score": benchmark_score,
        "gate_score": gate_score,
        "gate_score_source": gate_score_source,
        "quality_score": numeric((provider_score or {}).get("quality_score")) if provider_score else None,
        "confidence_level": numeric((provider_score or {}).get("confidence_level")) if provider_score else None,
        "benchmark_task_count": numeric((provider_score or {}).get("task_count")) if provider_score else None,
        "compatibility_run_id": compatibility.get("run_id"),
        "compatibility_found": bool(compatibility.get("found")),
        "compatibility_suite_status": manifest.get("suite_status"),
        "compatibility_status": manifest.get("status"),
        "compatibility_stopped": bool(manifest.get("stopped")),
        "compatibility_age_days": manifest_age_days(manifest),
        "compatibility_archive_warning": compatibility.get("archive_warning"),
        "rescore_applied_count": rescore_applied_count,
        "rescore_coverage": rescore_coverage,
        "rescore_age_days": manifest_age_days(rescore_manifest),
        "rescore_archive_warning": rescore.get("archive_warning"),
        "trace_eval_id": trace_evaluation.get("trace_eval_id"),
        "trace_requested": bool(trace_evaluation.get("requested")),
        "trace_found": bool(trace_evaluation.get("found")),
        "trace_status": trace_metrics.get("status"),
        "trace_age_days": manifest_age_days(trace_manifest),
        "trace_archive_warning": trace_evaluation.get("archive_warning"),
        "trace_record_count": trace_record_count,
        "trace_coverage": ratio(trace_record_count, sample_count) if trace_record_count is not None else None,
        "trace_fail_count": trace_metrics.get("fail_count"),
        "trace_warn_count": trace_metrics.get("warn_count"),
        "trace_fail_rate": trace_metrics.get("trace_fail_rate"),
        "trace_warn_rate": trace_metrics.get("trace_warn_rate"),
        "trace_thinking_only_count": trace_metrics.get("thinking_only_count"),
        "trace_missing_events_count": trace_metrics.get("missing_events_count"),
        "trace_max_tokens_count": trace_metrics.get("max_tokens_count"),
    }


def issue(rule_id: str, source: str, metric: str, observed: Any, threshold: Any, details: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "source": source,
        "metric": metric,
        "observed": observed,
        "threshold": threshold,
        "details": details,
    }


def manifest_incomplete(manifest: dict[str, Any]) -> bool:
    status = str(manifest.get("status") or "").strip().lower()
    return bool(manifest.get("stopped")) or bool(manifest.get("partial")) or status in {"stopped", "partial"}


def manifest_source_run_mismatch(manifest: dict[str, Any], source_run_id: str) -> bool:
    manifest_run_id = manifest.get("source_run_id")
    return bool(manifest_run_id and str(manifest_run_id) != str(source_run_id))


def parse_manifest_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def manifest_age_days(manifest: dict[str, Any]) -> float | None:
    for key in ("completed_at", "created_at", "rescored_at", "evaluated_at", "judged_at"):
        stamp = parse_manifest_time(manifest.get(key))
        if stamp is None:
            continue
        now = datetime.now(stamp.tzinfo) if stamp.tzinfo else datetime.now()
        return round(max(0.0, (now - stamp).total_seconds()) / 86400.0, 3)
    return None


def rescore_provider_mismatch(rescore: dict[str, Any], provider_id: str) -> bool:
    manifest = rescore.get("manifest") if isinstance(rescore.get("manifest"), dict) else {}
    filters = manifest.get("filters") if isinstance(manifest.get("filters"), dict) else {}
    filter_provider_id = filters.get("provider_id")
    if filter_provider_id and str(filter_provider_id) != str(provider_id):
        return True
    records = rescore.get("records") if isinstance(rescore.get("records"), list) else []
    record_provider_ids = {
        str(record.get("source_provider_id") or record.get("provider_id") or "")
        for record in records
        if isinstance(record, dict) and (record.get("source_provider_id") or record.get("provider_id"))
    }
    return bool(record_provider_ids and str(provider_id) not in record_provider_ids)


def trace_provider_mismatch(trace_evaluation: dict[str, Any], provider_id: str) -> bool:
    manifest = trace_evaluation.get("manifest") if isinstance(trace_evaluation.get("manifest"), dict) else {}
    provider_metrics = manifest.get("provider_metrics") if isinstance(manifest.get("provider_metrics"), dict) else {}
    return bool(provider_metrics and str(provider_id) not in {str(key) for key in provider_metrics.keys()})


def evaluate_policy(
    *,
    policy: dict[str, Any],
    source_run_id: str,
    provider_id: str,
    metrics: dict[str, Any],
    compatibility: dict[str, Any],
    trace_evaluation: dict[str, Any],
    rescore: dict[str, Any],
    require_rescore: bool,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    thresholds = policy.get("thresholds") or {}
    blockers: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    passed_rules: list[str] = []

    suite_status = metrics.get("compatibility_suite_status")
    compatibility_status = metrics.get("compatibility_status")
    compatibility_stopped = bool(metrics.get("compatibility_stopped"))
    if suite_status == "FAIL":
        blockers.append(issue("compatibility_fail_blocks", "compatibility", "suite_status", suite_status, "not FAIL", "compatibility suite failed"))
    elif not metrics.get("compatibility_found"):
        review_items.append(issue("compatibility_missing_requires_review", "compatibility", "suite_status", None, "PASS", compatibility.get("error") or "matching compatibility evidence missing"))
    elif suite_status == "WARN" or compatibility_stopped or compatibility_status not in {None, "completed"}:
        review_items.append(issue("compatibility_warn_requires_review", "compatibility", "suite_status", suite_status, "PASS", "compatibility evidence is warning, stopped, or partial"))
    else:
        passed_rules.append("compatibility_status_ok")

    if metrics.get("model_mismatch_count", 0) > 0:
        blockers.append(issue("model_mismatch_blocks", "run_records", "model_mismatch_count", metrics.get("model_mismatch_count"), 0, "model_requested and model_returned differ"))
    else:
        passed_rules.append("model_identity_ok")

    # Fake-1M / silent-truncation: needle probe reported HTTP 200 but input_tokens
    # far below sent. Hard blocker. Absent (no needle evidence) -> 0 -> no-op.
    silent_truncation_count = metrics.get("silent_truncation_count", 0) or 0
    silent_truncation_no_go = numeric(thresholds.get("silent_truncation_no_go"), 0)
    if silent_truncation_count > silent_truncation_no_go:
        blockers.append(issue("silent_truncation_blocks", "needle", "silent_truncation_count", silent_truncation_count, silent_truncation_no_go, "endpoint silently truncated a long-context request (suspected fake 1M)"))
    elif metrics.get("needle_found"):
        passed_rules.append("needle_context_ok")

    success_rate = metrics.get("success_rate")
    success_no_go = numeric(thresholds.get("success_rate_no_go"), 0.95)
    success_review = numeric(thresholds.get("success_rate_review"), 0.98)
    if metrics.get("sample_count", 0) <= 0:
        review_items.append(issue("sample_evidence_missing", "run_records", "sample_count", metrics.get("sample_count"), ">0", "no provider-level samples were found"))
    elif success_rate is not None and success_rate < success_no_go:
        blockers.append(issue("success_rate_too_low", "run_records", "success_rate", success_rate, success_no_go, "model/API success rate is below release threshold"))
    elif success_rate is not None and success_rate < success_review:
        review_items.append(issue("success_rate_yellow_band", "run_records", "success_rate", success_rate, success_review, "success rate is below preferred release band"))
    else:
        passed_rules.append("success_rate_ok")

    json_rate = metrics.get("json_failure_rate")
    json_threshold = numeric(thresholds.get("json_failure_rate_no_go"), 0.05)
    if json_rate is not None and json_rate > json_threshold:
        blockers.append(issue("json_failure_rate_too_high", "run_records", "json_failure_rate", json_rate, json_threshold, "JSON task failure rate is too high"))
    else:
        passed_rules.append("json_failure_rate_ok")

    gate_score = metrics.get("gate_score")
    score_no_go = numeric(thresholds.get("gate_score_no_go"), 600)
    score_review = numeric(thresholds.get("gate_score_review"), 750)
    if gate_score is None:
        review_items.append(issue("gate_score_missing", "benchmark_scores", "gate_score", None, f">={score_review}", "benchmark score or rescore-derived gate score is missing"))
    elif gate_score < score_no_go:
        blockers.append(issue("gate_score_too_low", "scoring", "gate_score", gate_score, score_no_go, "gate score is below no-go threshold"))
    elif gate_score < score_review:
        review_items.append(issue("gate_score_yellow_band", "scoring", "gate_score", gate_score, score_review, "gate score is in review band"))
    else:
        passed_rules.append("gate_score_ok")

    judge_rate = metrics.get("judge_failure_rate")
    judge_threshold = numeric(thresholds.get("judge_failure_rate_review"), 0.05)
    if judge_rate is not None and judge_rate > judge_threshold:
        review_items.append(issue("judge_failure_rate_requires_review", "scoring", "judge_failure_rate", judge_rate, judge_threshold, "judge failures require manual review"))
    else:
        passed_rules.append("judge_failure_rate_ok")

    max_tokens_rate = metrics.get("max_tokens_rate")
    max_tokens_threshold = numeric(thresholds.get("max_tokens_rate_review"), 0.1)
    if max_tokens_rate is not None and max_tokens_rate > max_tokens_threshold:
        review_items.append(issue("max_tokens_rate_requires_review", "telemetry", "max_tokens_rate", max_tokens_rate, max_tokens_threshold, "too many responses stopped at max_tokens"))
    else:
        passed_rules.append("max_tokens_rate_ok")

    p95_latency = metrics.get("p95_first_content_token_ms")
    latency_threshold = numeric(thresholds.get("p95_first_content_token_ms_review"), 15000)
    if p95_latency is not None and p95_latency > latency_threshold:
        review_items.append(issue("p95_latency_requires_review", "telemetry", "p95_first_content_token_ms", p95_latency, latency_threshold, "p95 first content token latency is too high"))
    else:
        passed_rules.append("p95_latency_ok")

    evidence_max_age_days = numeric(thresholds.get("evidence_max_age_days"), None)
    if evidence_max_age_days is not None:
        compatibility_age = metrics.get("compatibility_age_days")
        if metrics.get("compatibility_found") and compatibility_age is not None and compatibility_age > evidence_max_age_days:
            review_items.append(issue("compatibility_evidence_stale", "compatibility", "compatibility_age_days", compatibility_age, evidence_max_age_days, "compatibility evidence is stale"))
        elif metrics.get("compatibility_found") and compatibility_age is not None:
            passed_rules.append("compatibility_evidence_fresh")

        rescore_age = metrics.get("rescore_age_days")
        if rescore.get("found") and rescore_age is not None and rescore_age > evidence_max_age_days:
            review_items.append(issue("rescore_evidence_stale", "rescore", "rescore_age_days", rescore_age, evidence_max_age_days, "rescore evidence is stale"))
        elif rescore.get("found") and rescore_age is not None:
            passed_rules.append("rescore_evidence_fresh")

        trace_age = metrics.get("trace_age_days")
        if metrics.get("trace_found") and trace_age is not None and trace_age > evidence_max_age_days:
            review_items.append(issue("trace_evaluation_evidence_stale", "trace_evaluation", "trace_age_days", trace_age, evidence_max_age_days, "trace evaluation evidence is stale"))
        elif metrics.get("trace_found") and trace_age is not None:
            passed_rules.append("trace_evaluation_evidence_fresh")

    rescore_required = require_rescore or bool(rescore.get("requested"))
    if rescore_required:
        required_coverage = numeric(thresholds.get("rescore_required_coverage"), 1.0)
        coverage = metrics.get("rescore_coverage")
        if not rescore.get("found"):
            review_items.append(issue("required_rescore_missing", "rescore", "rescore_id", rescore.get("rescore_id"), "present", rescore.get("error") or "rescore is required but missing"))
        else:
            manifest = rescore.get("manifest") if isinstance(rescore.get("manifest"), dict) else {}
            if manifest_incomplete(manifest):
                review_items.append(issue("rescore_incomplete_requires_review", "rescore", "status", manifest.get("status"), "completed", "rescore manifest is stopped or partial"))
            if manifest_source_run_mismatch(manifest, source_run_id):
                review_items.append(issue("rescore_source_run_mismatch", "rescore", "source_run_id", manifest.get("source_run_id"), source_run_id, "rescore source run does not match quality gate source run"))
            if rescore_provider_mismatch(rescore, provider_id):
                review_items.append(issue("rescore_provider_mismatch", "rescore", "provider_id", provider_id, "matching provider", "rescore evidence is not bound to this provider"))
            if coverage is None or coverage < required_coverage:
                review_items.append(issue("required_rescore_coverage_low", "rescore", "rescore_coverage", coverage, required_coverage, "rescore did not cover all provider samples"))
            else:
                passed_rules.append("required_rescore_coverage_ok")

    trace_fail_threshold = numeric(thresholds.get("trace_fail_rate_review"), 0.0)
    trace_warn_threshold = numeric(thresholds.get("trace_warn_rate_review"), 0.0)
    trace_fail_rate = metrics.get("trace_fail_rate")
    trace_warn_rate = metrics.get("trace_warn_rate")
    if metrics.get("trace_requested") and not metrics.get("trace_found"):
        review_items.append(issue("trace_evaluation_missing_requested", "trace_evaluation", "trace_eval_id", metrics.get("trace_eval_id"), "present", trace_evaluation.get("error") or "requested trace evaluation evidence is missing"))
        if "provider mismatch" in str(trace_evaluation.get("error") or ""):
            review_items.append(issue("trace_evaluation_provider_mismatch", "trace_evaluation", "provider_id", provider_id, "matching provider", "trace evaluation evidence is not bound to this provider"))
    elif metrics.get("trace_requested") and metrics.get("trace_found"):
        manifest = trace_evaluation.get("manifest") if isinstance(trace_evaluation.get("manifest"), dict) else {}
        required_coverage = numeric(thresholds.get("trace_required_coverage"), 1.0)
        coverage = metrics.get("trace_coverage")
        if manifest_incomplete(manifest):
            review_items.append(issue("trace_evaluation_incomplete_requires_review", "trace_evaluation", "status", manifest.get("status"), "completed", "trace evaluation manifest is stopped or partial"))
        if manifest_source_run_mismatch(manifest, source_run_id):
            review_items.append(issue("trace_evaluation_source_run_mismatch", "trace_evaluation", "source_run_id", manifest.get("source_run_id"), source_run_id, "trace evaluation source run does not match quality gate source run"))
        if trace_provider_mismatch(trace_evaluation, provider_id):
            review_items.append(issue("trace_evaluation_provider_mismatch", "trace_evaluation", "provider_id", provider_id, "matching provider", "trace evaluation evidence is not bound to this provider"))
        if coverage is None or coverage < required_coverage:
            review_items.append(issue("trace_evaluation_coverage_low", "trace_evaluation", "trace_coverage", coverage, required_coverage, "trace evaluation did not cover all provider samples"))
        if trace_fail_rate is not None and trace_fail_rate > trace_fail_threshold:
            review_items.append(issue("trace_failures_require_review", "trace_evaluation", "trace_fail_rate", trace_fail_rate, trace_fail_threshold, "offline trace evaluation found failed trace health checks"))
        elif trace_warn_rate is not None and trace_warn_rate > trace_warn_threshold:
            review_items.append(issue("trace_warnings_require_review", "trace_evaluation", "trace_warn_rate", trace_warn_rate, trace_warn_threshold, "offline trace evaluation found warning trace health checks"))
        if not any(item.get("source") == "trace_evaluation" for item in review_items):
            passed_rules.append("trace_evaluation_ok")
    elif metrics.get("trace_found") and trace_fail_rate is not None and trace_fail_rate > trace_fail_threshold:
        review_items.append(issue("trace_failures_require_review", "trace_evaluation", "trace_fail_rate", trace_fail_rate, trace_fail_threshold, "offline trace evaluation found failed trace health checks"))
    elif metrics.get("trace_found") and trace_warn_rate is not None and trace_warn_rate > trace_warn_threshold:
        review_items.append(issue("trace_warnings_require_review", "trace_evaluation", "trace_warn_rate", trace_warn_rate, trace_warn_threshold, "offline trace evaluation found warning trace health checks"))
    elif metrics.get("trace_found"):
        passed_rules.append("trace_evaluation_ok")
    else:
        passed_rules.append("trace_evaluation_not_required")

    decision = NO_GO if blockers else REVIEW if review_items else GO
    return decision, blockers, review_items, sorted(set(passed_rules))


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "gate_id",
        "source_run_id",
        "provider_id",
        "decision",
        "gate_score",
        "gate_score_source",
        "benchmark_score",
        "sample_count",
        "success_rate",
        "json_failure_rate",
        "judge_failure_rate",
        "max_tokens_rate",
        "p95_first_content_token_ms",
        "compatibility_run_id",
        "compatibility_suite_status",
        "trace_eval_id",
        "trace_status",
        "trace_coverage",
        "trace_fail_rate",
        "trace_warn_rate",
        "rescore_applied_count",
        "rescore_coverage",
        "blocker_count",
        "review_item_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics_snapshot") or {}
            row = {
                "gate_id": record.get("gate_id"),
                "source_run_id": record.get("source_run_id"),
                "provider_id": record.get("provider_id"),
                "decision": record.get("decision"),
                "gate_score": metrics.get("gate_score"),
                "gate_score_source": metrics.get("gate_score_source"),
                "benchmark_score": metrics.get("benchmark_score"),
                "sample_count": metrics.get("sample_count"),
                "success_rate": metrics.get("success_rate"),
                "json_failure_rate": metrics.get("json_failure_rate"),
                "judge_failure_rate": metrics.get("judge_failure_rate"),
                "max_tokens_rate": metrics.get("max_tokens_rate"),
                "p95_first_content_token_ms": metrics.get("p95_first_content_token_ms"),
                "compatibility_run_id": metrics.get("compatibility_run_id"),
                "compatibility_suite_status": metrics.get("compatibility_suite_status"),
                "trace_eval_id": metrics.get("trace_eval_id"),
                "trace_status": metrics.get("trace_status"),
                "trace_coverage": metrics.get("trace_coverage"),
                "trace_fail_rate": metrics.get("trace_fail_rate"),
                "trace_warn_rate": metrics.get("trace_warn_rate"),
                "rescore_applied_count": metrics.get("rescore_applied_count"),
                "rescore_coverage": metrics.get("rescore_coverage"),
                "blocker_count": len(record.get("blockers") or []),
                "review_item_count": len(record.get("review_items") or []),
            }
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def list_quality_gates(run_dir: Path) -> list[dict[str, Any]]:
    gates_dir = run_dir / "quality_gates"
    if not gates_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(gates_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        manifest_path = child / "quality_gate_manifest.json"
        if not child.is_dir() or not manifest_path.exists():
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if isinstance(manifest, dict):
            out.append(manifest)
    return out


def read_quality_gate(run_dir: Path, gate_id: str) -> dict[str, Any]:
    gate_dir = run_dir / "quality_gates" / gate_id
    manifest_path = gate_dir / "quality_gate_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"quality gate not found: {gate_id}")
    return {
        "manifest": read_json(manifest_path),
        "summary": read_csv_rows(gate_dir / "quality_gate_summary.csv"),
        "records": read_jsonl(gate_dir / "quality_gate_records.jsonl"),
    }


def run_quality_gate(
    *,
    runs_dir: Path,
    run_id: str,
    policy_path: Path,
    provider_id: str | None = None,
    compatibility_run_id: str | None = None,
    rescore_id: str | None = None,
    trace_eval_id: str | None = None,
    policy_id: str | None = None,
    gate_label: str | None = None,
    require_rescore: bool = False,
) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_id}")

    created_at = datetime.now().isoformat(timespec="seconds")
    gate_id = registry.unique_artifact_id("gate")
    gate_dir = run_dir / "quality_gates" / gate_id
    gate_dir.mkdir(parents=True, exist_ok=False)

    policy = load_policy(policy_path, policy_id)
    samples, sample_refs = load_run_samples(run_dir)
    provider_scores, benchmark_refs = load_benchmark_provider_scores(run_dir)
    provider_ids = provider_ids_from_evidence(samples, provider_scores, provider_id)
    if not provider_ids:
        raise ValueError(f"no providers found for run: {run_id}")

    records: list[dict[str, Any]] = []
    for current_provider_id in provider_ids:
        provider_samples = [
            sample for sample in samples if str(sample.get("provider_id") or "") == current_provider_id
        ]
        rescore = load_rescore_evidence(run_dir, rescore_id, current_provider_id)
        rescored_samples, rescore_applied_count = apply_rescore(provider_samples, rescore, current_provider_id)
        compatibility = load_compatibility_evidence(runs_dir, current_provider_id, compatibility_run_id)
        trace_evaluation = load_trace_evaluation_evidence(run_dir, current_provider_id, trace_eval_id)
        provider_score = provider_scores.get(current_provider_id) if isinstance(provider_scores, dict) else None
        metrics = aggregate_metrics(
            samples=rescored_samples,
            provider_score=provider_score if isinstance(provider_score, dict) else None,
            compatibility=compatibility,
            trace_evaluation=trace_evaluation,
            rescore=rescore,
            rescore_applied_count=rescore_applied_count,
        )
        decision, blockers, review_items, passed_rules = evaluate_policy(
            policy=policy,
            source_run_id=run_id,
            provider_id=current_provider_id,
            metrics=metrics,
            compatibility=compatibility,
            trace_evaluation=trace_evaluation,
            rescore=rescore,
            require_rescore=require_rescore,
        )
        evidence_refs = {
            **sample_refs,
            **benchmark_refs,
            **(compatibility.get("refs") or {}),
            **(trace_evaluation.get("refs") or {}),
            **(rescore.get("refs") or {}),
        }
        evidence_ids = {
            "compatibility_run_id": compatibility.get("run_id"),
            "trace_eval_id": trace_evaluation.get("trace_eval_id") if trace_evaluation.get("found") else None,
            "rescore_id": rescore.get("rescore_id") if rescore.get("found") else None,
        }
        record = {
            "schema_version": QUALITY_GATE_RECORD_VERSION,
            "gate_id": gate_id,
            "source_run_id": run_id,
            "provider_id": current_provider_id,
            "policy_id": policy.get("policy_id"),
            "policy_version": policy.get("policy_version") or QUALITY_GATE_POLICY_VERSION,
            "decision": decision,
            "metrics_snapshot": metrics,
            "blockers": blockers,
            "review_items": review_items,
            "passed_rules": passed_rules,
            "evidence_refs": evidence_refs,
            "evidence_ids": evidence_ids,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        records.append(record)

    records_path = gate_dir / "quality_gate_records.jsonl"
    summary_path = gate_dir / "quality_gate_summary.csv"
    manifest_path = gate_dir / "quality_gate_manifest.json"
    write_jsonl(records_path, records)
    write_summary(summary_path, records)
    decision_counts = {GO: 0, REVIEW: 0, NO_GO: 0}
    for record in records:
        decision_counts[record["decision"]] = decision_counts.get(record["decision"], 0) + 1
    completed_at = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "gate_id": gate_id,
        "source_run_id": run_id,
        "status": "completed",
        "created_at": created_at,
        "completed_at": completed_at,
        "gate_label": gate_label,
        "policy_id": policy.get("policy_id"),
        "policy_version": policy.get("policy_version") or QUALITY_GATE_POLICY_VERSION,
        "provider_ids": provider_ids,
        "filters": {
            "provider_id": provider_id,
            "compatibility_run_id": compatibility_run_id,
            "rescore_id": rescore_id,
            "trace_eval_id": trace_eval_id,
            "require_rescore": require_rescore,
        },
        "record_count": len(records),
        "decision_counts": decision_counts,
        "records_file": str(records_path),
        "summary_file": str(summary_path),
        "manifest_file": str(manifest_path),
    }
    write_json(manifest_path, manifest)
    return {
        "gate_id": gate_id,
        "record_count": len(records),
        "records": records,
        "summary": read_csv_rows(summary_path),
        "manifest": manifest,
    }


def make_fake_run(
    runs_dir: Path,
    *,
    run_id: str,
    provider_id: str = "fake_provider",
    benchmark_score: float = 850,
    ok: bool = True,
    score: float = 8.5,
    category: str = "general",
    scoring_type: str = "keyword_check",
    format_ok: bool | None = True,
    judge_error: str | None = None,
    stop_reason: str = "end_turn",
    first_content_token_ms: float = 1000,
    model_requested: str = "fake-model",
    model_returned: str = "fake-model",
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "run_record_v1",
        "record_id": f"{run_id}:{provider_id}:task_001",
        "run": {"run_id": run_id, "timestamp": "2026-06-20T00:00:00", "benchmark_mode": "test", "formula_version": "score_formula_v1", "runner": "self_test", "status": "completed"},
        "task": {"id": "task_001", "category": category, "enterprise_dimension": category, "difficulty": "easy", "scoring_type": scoring_type, "risk_tags": [], "point_value": 100, "scoring_confidence": 0.8},
        "provider": {"id": provider_id, "api_style": "anthropic_messages", "base_url_host": "example.invalid", "model_requested": model_requested, "model_returned": model_returned},
        "request": {},
        "response": {},
        "telemetry": {"ok": ok, "error": None if ok else "api failed", "first_content_token_ms": first_content_token_ms, "stop_reason": stop_reason},
        "usage": {},
        "scoring": {"final_score": {"score": score, "format_ok": format_ok}, "judge_score": {"error": judge_error} if judge_error else {}, "rule_score": {}},
        "trace": {},
        "artifacts": {},
    }
    write_jsonl(run_dir / "run_records.jsonl", [record])
    write_json(
        run_dir / "benchmark_scores.json",
        {"providers": {provider_id: {"benchmark_score": benchmark_score, "quality_score": benchmark_score, "task_count": 1}}},
    )
    return run_dir


def make_fake_compatibility(
    runs_dir: Path,
    *,
    run_id: str,
    provider_id: str = "fake_provider",
    suite_status: str = "PASS",
    status: str = "completed",
    stopped: bool = False,
) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "compatibility_manifest.json",
        {
            "run_id": run_id,
            "provider_id": provider_id,
            "suite_status": suite_status,
            "status": status,
            "stopped": stopped,
            "record_count": 1,
        },
    )
    write_jsonl(run_dir / "compatibility_records.jsonl", [{"status": suite_status, "provider_id": provider_id}])


def make_fake_trace_evaluation(
    runs_dir: Path,
    *,
    source_run_id: str,
    trace_eval_id: str,
    provider_id: str = "fake_provider",
    status: str = "WARN",
    manifest_status: str = "completed",
    manifest_source_run_id: str | None = None,
    stopped: bool = False,
    record_count: int | None = None,
    fail_count: int = 0,
    warn_count: int = 1,
) -> None:
    eval_dir = runs_dir / source_run_id / "trace_evaluations" / trace_eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    effective_record_count = record_count if record_count is not None else max(1, fail_count + warn_count)
    write_json(
        eval_dir / "trace_eval_manifest.json",
        {
            "schema_version": "trace_eval_manifest_v1",
            "trace_eval_id": trace_eval_id,
            "source_run_id": manifest_source_run_id or source_run_id,
            "status": manifest_status,
            "provider_metrics": {
                provider_id: {
                    "provider_id": provider_id,
                    "record_count": effective_record_count,
                    "pass_count": max(0, effective_record_count - fail_count - warn_count),
                    "warn_count": warn_count,
                    "fail_count": fail_count,
                    "trace_fail_rate": ratio(fail_count, effective_record_count),
                    "trace_warn_rate": ratio(warn_count, effective_record_count),
                    "thinking_only_count": 0,
                    "missing_events_count": warn_count,
                    "max_tokens_count": 0,
                    "status": status,
                }
            },
            "stopped": stopped,
        },
    )
    write_jsonl(eval_dir / "trace_eval_records.jsonl", [{"status": status, "provider_id": provider_id}])


def make_fake_rescore(
    runs_dir: Path,
    *,
    source_run_id: str,
    rescore_id: str,
    provider_id: str = "fake_provider",
    manifest_source_run_id: str | None = None,
    manifest_status: str = "completed",
    stopped: bool = False,
    include_record: bool = True,
    new_score: float = 9.0,
) -> None:
    rescore_dir = runs_dir / source_run_id / "rescores" / rescore_id
    rescore_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if include_record:
        records.append(
            {
                "rescore_id": rescore_id,
                "status": "completed",
                "source_run_id": source_run_id,
                "source_record_id": f"{source_run_id}:{provider_id}:task_001",
                "source_provider_id": provider_id,
                "task_id": "task_001",
                "new_final_score": {"score": new_score, "format_ok": True},
            }
        )
    write_jsonl(rescore_dir / "rescore_records.jsonl", records)
    write_json(
        rescore_dir / "rescore_manifest.json",
        {
            "schema_version": "rescore_manifest_v1",
            "rescore_id": rescore_id,
            "source_run_id": manifest_source_run_id or source_run_id,
            "status": manifest_status,
            "filters": {"provider_id": provider_id, "task_ids": []},
            "record_count": len(records),
            "completed_count": len(records),
            "failure_count": 0,
            "stopped": stopped,
        },
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        policy_path = root / "quality_gate.policy.json"
        write_json(
            policy_path,
            {
                "policy_version": QUALITY_GATE_POLICY_VERSION,
                "policies": [
                    {
                        "policy_id": DEFAULT_POLICY_ID,
                        "thresholds": {
                            "success_rate_no_go": 0.95,
                            "success_rate_review": 0.98,
                            "json_failure_rate_no_go": 0.05,
                            "gate_score_no_go": 600,
                            "gate_score_review": 750,
                            "judge_failure_rate_review": 0.05,
                            "max_tokens_rate_review": 0.1,
                            "p95_first_content_token_ms_review": 15000,
                            "rescore_required_coverage": 1.0,
                            "trace_fail_rate_review": 0,
                            "trace_warn_rate_review": 0,
                        },
                    }
                ],
            },
        )

        make_fake_run(root, run_id="run_go")
        make_fake_compatibility(root, run_id="compat_go")
        go_result = run_quality_gate(runs_dir=root, run_id="run_go", policy_path=policy_path, compatibility_run_id="compat_go")
        assert go_result["records"][0]["decision"] == GO

        make_fake_run(root, run_id="run_missing_compat", provider_id="missing_compat_provider")
        review_result = run_quality_gate(runs_dir=root, run_id="run_missing_compat", policy_path=policy_path)
        assert review_result["records"][0]["decision"] == REVIEW
        assert any(item["rule_id"] == "compatibility_missing_requires_review" for item in review_result["records"][0]["review_items"])

        make_fake_run(root, run_id="run_compat_fail")
        make_fake_compatibility(root, run_id="compat_fail", suite_status="FAIL")
        fail_result = run_quality_gate(runs_dir=root, run_id="run_compat_fail", policy_path=policy_path, compatibility_run_id="compat_fail")
        assert fail_result["records"][0]["decision"] == NO_GO

        make_fake_run(root, run_id="run_model_mismatch", model_returned="other-model")
        make_fake_compatibility(root, run_id="compat_model_mismatch")
        mismatch_result = run_quality_gate(runs_dir=root, run_id="run_model_mismatch", policy_path=policy_path, compatibility_run_id="compat_model_mismatch")
        assert mismatch_result["records"][0]["decision"] == NO_GO

        make_fake_run(root, run_id="run_json_fail", category="json_stability", scoring_type="json_exact", format_ok=False)
        make_fake_compatibility(root, run_id="compat_json_fail")
        json_result = run_quality_gate(runs_dir=root, run_id="run_json_fail", policy_path=policy_path, compatibility_run_id="compat_json_fail")
        assert json_result["records"][0]["decision"] == NO_GO

        make_fake_run(root, run_id="run_judge_fail", judge_error="judge failed")
        make_fake_compatibility(root, run_id="compat_judge_fail")
        judge_result = run_quality_gate(runs_dir=root, run_id="run_judge_fail", policy_path=policy_path, compatibility_run_id="compat_judge_fail")
        assert judge_result["records"][0]["decision"] == REVIEW

        make_fake_run(root, run_id="run_need_rescore")
        make_fake_compatibility(root, run_id="compat_need_rescore")
        rescore_result = run_quality_gate(
            runs_dir=root,
            run_id="run_need_rescore",
            policy_path=policy_path,
            compatibility_run_id="compat_need_rescore",
            require_rescore=True,
        )
        assert rescore_result["records"][0]["decision"] == REVIEW

        make_fake_run(root, run_id="run_partial_rescore", benchmark_score=850, score=5)
        make_fake_compatibility(root, run_id="compat_partial_rescore")
        make_fake_rescore(root, source_run_id="run_partial_rescore", rescore_id="rescore_partial", include_record=False)
        partial_rescore_result = run_quality_gate(
            runs_dir=root,
            run_id="run_partial_rescore",
            policy_path=policy_path,
            compatibility_run_id="compat_partial_rescore",
            rescore_id="rescore_partial",
        )
        partial_record = partial_rescore_result["records"][0]
        assert partial_record["decision"] == REVIEW
        assert partial_record["metrics_snapshot"]["gate_score"] == 850
        assert partial_record["metrics_snapshot"]["gate_score_source"] == "benchmark_scores"
        assert any(item["rule_id"] == "required_rescore_coverage_low" for item in partial_record["review_items"])

        make_fake_run(root, run_id="run_latest_rescore", benchmark_score=850, score=5)
        make_fake_compatibility(root, run_id="compat_latest_rescore")
        make_fake_rescore(root, source_run_id="run_latest_rescore", rescore_id="rescore_latest", new_score=9.0)
        latest_rescore_result = run_quality_gate(
            runs_dir=root,
            run_id="run_latest_rescore",
            policy_path=policy_path,
            compatibility_run_id="compat_latest_rescore",
        )
        latest_rescore_record = latest_rescore_result["records"][0]
        assert latest_rescore_record["evidence_ids"]["rescore_id"] == "rescore_latest"
        assert latest_rescore_record["metrics_snapshot"]["rescore_applied_count"] == 1
        assert latest_rescore_record["metrics_snapshot"]["gate_score_source"] == "rescore_full_coverage"

        make_fake_run(root, run_id="run_rescore_source_mismatch")
        make_fake_compatibility(root, run_id="compat_rescore_source_mismatch")
        make_fake_rescore(
            root,
            source_run_id="run_rescore_source_mismatch",
            rescore_id="rescore_source_mismatch",
            manifest_source_run_id="other_run",
        )
        rescore_mismatch_result = run_quality_gate(
            runs_dir=root,
            run_id="run_rescore_source_mismatch",
            policy_path=policy_path,
            compatibility_run_id="compat_rescore_source_mismatch",
            rescore_id="rescore_source_mismatch",
        )
        assert any(item["rule_id"] == "rescore_source_run_mismatch" for item in rescore_mismatch_result["records"][0]["review_items"])

        make_fake_run(root, run_id="run_trace_warn")
        make_fake_compatibility(root, run_id="compat_trace_warn")
        make_fake_trace_evaluation(root, source_run_id="run_trace_warn", trace_eval_id="trace_warn")
        trace_result = run_quality_gate(
            runs_dir=root,
            run_id="run_trace_warn",
            policy_path=policy_path,
            compatibility_run_id="compat_trace_warn",
            trace_eval_id="trace_warn",
        )
        assert trace_result["records"][0]["decision"] == REVIEW
        assert any(item["rule_id"] == "trace_warnings_require_review" for item in trace_result["records"][0]["review_items"])

        make_fake_run(root, run_id="run_trace_strict")
        make_fake_compatibility(root, run_id="compat_trace_strict")
        make_fake_trace_evaluation(
            root,
            source_run_id="run_trace_strict",
            trace_eval_id="trace_strict",
            status="PASS",
            manifest_status="stopped",
            stopped=True,
            record_count=0,
            warn_count=0,
        )
        trace_strict_result = run_quality_gate(
            runs_dir=root,
            run_id="run_trace_strict",
            policy_path=policy_path,
            compatibility_run_id="compat_trace_strict",
            trace_eval_id="trace_strict",
        )
        trace_strict_items = trace_strict_result["records"][0]["review_items"]
        assert any(item["rule_id"] == "trace_evaluation_incomplete_requires_review" for item in trace_strict_items)
        assert any(item["rule_id"] == "trace_evaluation_coverage_low" for item in trace_strict_items)

        # silent_truncation (fake-1M) is a hard blocker; absent metric is a no-op
        st_policy = {"thresholds": {"silent_truncation_no_go": 0}}
        base_metrics = {"sample_count": 5, "success_rate": 1.0, "compatibility_found": True, "compatibility_suite_status": "PASS"}
        empty = {}
        d_block, blockers, _, _ = evaluate_policy(
            policy=st_policy, source_run_id="r", provider_id="p",
            metrics={**base_metrics, "silent_truncation_count": 1, "needle_found": True},
            compatibility=empty, trace_evaluation=empty, rescore=empty, require_rescore=False,
        )
        assert d_block == NO_GO and any(b["rule_id"] == "silent_truncation_blocks" for b in blockers)
        _, blockers_none, _, passed = evaluate_policy(
            policy=st_policy, source_run_id="r", provider_id="p",
            metrics={**base_metrics, "silent_truncation_count": 0, "needle_found": True},
            compatibility=empty, trace_evaluation=empty, rescore=empty, require_rescore=False,
        )
        assert not any(b["rule_id"] == "silent_truncation_blocks" for b in blockers_none)
        assert "needle_context_ok" in passed

    print("quality gate self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate offline quality gates for an existing run")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--policy", type=Path, default=Path("quality_gate.policy.json"))
    parser.add_argument("--policy-id", default=DEFAULT_POLICY_ID)
    parser.add_argument("--run-id")
    parser.add_argument("--provider-id")
    parser.add_argument("--compatibility-run-id")
    parser.add_argument("--rescore-id")
    parser.add_argument("--trace-eval-id")
    parser.add_argument("--gate-label")
    parser.add_argument("--require-rescore", action="store_true")
    parser.add_argument("--require-go", action="store_true", help="return exit code 2 unless every provider decision is GO")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.run_id:
        raise SystemExit("--run-id is required unless --self-test is used")
    result = run_quality_gate(
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        policy_path=args.policy,
        provider_id=args.provider_id,
        compatibility_run_id=args.compatibility_run_id,
        rescore_id=args.rescore_id,
        trace_eval_id=args.trace_eval_id,
        policy_id=args.policy_id,
        gate_label=args.gate_label,
        require_rescore=args.require_rescore,
    )
    print(json.dumps({"gate_id": result["gate_id"], "manifest": result["manifest"]}, ensure_ascii=False, indent=2))
    if args.require_go and any(record.get("decision") != GO for record in result.get("records") or []):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
