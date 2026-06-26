from __future__ import annotations

import csv
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import evidence_registry as registry


ScoreFn = Callable[[dict[str, Any], str], dict[str, Any]]
JudgeFn = Callable[[dict[str, Any], str, dict[str, Any]], dict[str, Any]]
ProgressFn = Callable[[dict[str, Any]], None]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def csv_value(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def score_value(score: Any) -> Any:
    if not isinstance(score, dict):
        return None
    return score.get("score")


def load_response_text(path_value: Any, run_dir: Path) -> tuple[str, str | None]:
    if not path_value:
        return "", "missing response_file"
    path = Path(str(path_value))
    if not path.exists() and not path.is_absolute():
        path = run_dir / path
    if not path.exists():
        return "", f"response file not found: {path_value}"
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return "", f"response file read failed: {type(exc).__name__}: {exc}"


def source_records_from_run_records(run_dir: Path) -> list[dict[str, Any]]:
    records_path = run_dir / "run_records.jsonl"
    records = read_jsonl(records_path)
    out: list[dict[str, Any]] = []
    for record in records:
        run = record.get("run") or {}
        task = record.get("task") or {}
        provider = record.get("provider") or {}
        scoring = record.get("scoring") or {}
        artifacts = record.get("artifacts") or {}
        telemetry = record.get("telemetry") or {}
        out.append(
            {
                "source_kind": "run_record",
                "source_run_id": run.get("run_id") or run_dir.name,
                "source_record_id": record.get("record_id"),
                "task_id": task.get("id"),
                "provider_id": provider.get("id"),
                "task": task,
                "provider": provider,
                "source_response_file": artifacts.get("response_file") or (record.get("response") or {}).get("response_file"),
                "source_events_file": artifacts.get("events_file") or (record.get("response") or {}).get("events_file"),
                "original_score": scoring.get("final_score"),
                "source_error": telemetry.get("error"),
            }
        )
    return out


def source_records_from_results(run_dir: Path) -> list[dict[str, Any]]:
    results_path = run_dir / "results.json"
    results = read_json(results_path)
    if not isinstance(results, list):
        raise ValueError(f"{results_path} must contain a JSON array")
    out: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        task = item.get("task") or {}
        provider = item.get("provider") or {}
        metrics = item.get("metrics") or {}
        task_id = task.get("id")
        provider_id = provider.get("id")
        out.append(
            {
                "source_kind": "results_json",
                "source_run_id": item.get("run_id") or run_dir.name,
                "source_record_id": f"{item.get('run_id') or run_dir.name}:{provider_id}:{task_id}",
                "task_id": task_id,
                "provider_id": provider_id,
                "task": task,
                "provider": provider,
                "source_response_file": item.get("response_file"),
                "source_events_file": item.get("events_file"),
                "original_score": item.get("score"),
                "source_error": metrics.get("error"),
            }
        )
    return out


def load_source_records(run_dir: Path) -> list[dict[str, Any]]:
    if (run_dir / "run_records.jsonl").exists():
        return source_records_from_run_records(run_dir)
    if (run_dir / "results.json").exists():
        return source_records_from_results(run_dir)
    raise FileNotFoundError(f"no run_records.jsonl or results.json found in {run_dir}")


def filter_source_records(
    records: list[dict[str, Any]],
    provider_id: str | None,
    task_ids: list[str] | None,
) -> list[dict[str, Any]]:
    wanted_tasks = set(task_ids or [])
    out: list[dict[str, Any]] = []
    for record in records:
        if provider_id and record.get("provider_id") != provider_id:
            continue
        if wanted_tasks and record.get("task_id") not in wanted_tasks:
            continue
        out.append(record)
    return out


def source_count(
    run_dir: Path,
    provider_id: str | None = None,
    task_ids: list[str] | None = None,
) -> int:
    return len(filter_source_records(load_source_records(run_dir), provider_id, task_ids))


def task_lookup(task_bank: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(task.get("id")): task for task in task_bank if task.get("id")}


def wait_for_resume(
    job_control: dict[str, Any] | None,
    progress_callback: ProgressFn | None,
    completed: int,
    total: int,
) -> bool:
    if not job_control or not job_control.get("pause_requested"):
        return False
    if progress_callback:
        progress_callback(
            {
                "event": "run_paused",
                "completed_tasks": completed,
                "total_tasks": total,
            }
        )
    resume_event = job_control.get("resume_event")
    while job_control.get("pause_requested") and not job_control.get("stop_requested"):
        if isinstance(resume_event, threading.Event):
            resume_event.wait(0.25)
        else:
            break
    if job_control.get("stop_requested"):
        return True
    if progress_callback:
        progress_callback(
            {
                "event": "run_resumed",
                "completed_tasks": completed,
                "total_tasks": total,
            }
        )
    return False


def write_rescore_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rescore_id",
        "source_run_id",
        "source_record_id",
        "task_id",
        "provider_id",
        "original_score_0_10",
        "new_rule_score_0_10",
        "new_judge_score_0_10",
        "new_final_score_0_10",
        "judge_provider",
        "judge_model_requested",
        "judge_model_returned",
        "rubric_version",
        "status",
        "rescore_error",
        "judge_error",
        "source_response_file",
        "source_events_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def list_rescores(run_dir: Path) -> list[dict[str, Any]]:
    rescores_dir = run_dir / "rescores"
    if not rescores_dir.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for child in sorted(rescores_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        manifest_path = child / "rescore_manifest.json"
        if not child.is_dir() or not manifest_path.exists():
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if isinstance(manifest, dict):
            manifests.append(manifest)
    return manifests


def read_rescore(run_dir: Path, rescore_id: str) -> dict[str, Any]:
    rescore_dir = run_dir / "rescores" / rescore_id
    manifest_path = rescore_dir / "rescore_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"rescore not found: {rescore_id}")
    return {
        "manifest": read_json(manifest_path),
        "summary": read_csv_rows(rescore_dir / "rescore_summary.csv"),
    }


def run_rescore(
    *,
    runs_dir: Path,
    run_id: str,
    task_bank: list[dict[str, Any]],
    score_response: ScoreFn,
    provider_id: str | None = None,
    task_ids: list[str] | None = None,
    judge_response: JudgeFn | None = None,
    judge_provider: dict[str, Any] | None = None,
    judge_rubric_version: str | None = None,
    rescore_label: str | None = None,
    progress_callback: ProgressFn | None = None,
    job_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_id}")

    created_at = datetime.now().isoformat(timespec="seconds")
    rescore_id = registry.unique_artifact_id("rescore")
    rescore_dir = run_dir / "rescores" / rescore_id
    rescore_dir.mkdir(parents=True, exist_ok=False)

    lookup = task_lookup(task_bank)
    all_sources = load_source_records(run_dir)
    sources = filter_source_records(all_sources, provider_id, task_ids)
    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total = len(sources)
    stopped = False

    if progress_callback:
        progress_callback(
            {
                "event": "run_started",
                "total_tasks": total,
                "benchmark_mode": "rescore",
                "current_provider": provider_id,
            }
        )

    for index, source in enumerate(sources):
        if job_control and job_control.get("stop_requested"):
            stopped = True
            break
        if wait_for_resume(job_control, progress_callback, index, total):
            stopped = True
            break

        task_id = str(source.get("task_id") or "")
        source_provider_id = str(source.get("provider_id") or "")
        if progress_callback:
            progress_callback(
                {
                    "event": "task_started",
                    "current_task_id": task_id,
                    "current_provider": source_provider_id,
                    "phase": "rescoring_rule",
                    "completed_tasks": index,
                }
            )

        task = lookup.get(task_id) or source.get("task") or {}
        response_text, response_error = load_response_text(source.get("source_response_file"), run_dir)
        new_rule_score: dict[str, Any] | None = None
        new_judge_score: dict[str, Any] | None = None
        rescore_error = response_error
        judge_error = None

        if not rescore_error:
            # token_count_check needs this run's observed input_tokens (not re-derivable
            # from text); needle_recall depends on the planted prompt. Skip rescoring
            # these from text and preserve the original record score.
            if task.get("scoring_type") in ("token_count_check", "needle_recall"):
                new_rule_score = None
            else:
                try:
                    new_rule_score = score_response(task, response_text)
                except Exception as exc:
                    rescore_error = f"rule score failed: {type(exc).__name__}: {exc}"

        if not rescore_error and judge_response:
            if progress_callback:
                progress_callback(
                    {
                        "event": "task_phase",
                        "current_task_id": task_id,
                        "current_provider": source_provider_id,
                        "phase": "rescoring_judge",
                        "completed_tasks": index,
                    }
                )
            try:
                new_judge_score = judge_response(task, response_text, new_rule_score or {})
                judge_error = new_judge_score.get("error") if isinstance(new_judge_score, dict) else None
            except Exception as exc:
                judge_error = f"{type(exc).__name__}: {exc}"
                new_judge_score = {
                    "score": None,
                    "format_ok": None,
                    "confidence": None,
                    "details": "",
                    "hit_key_points": [],
                    "missed_key_points": [],
                    "risk_flags": [],
                    "raw": None,
                    "provider": (judge_provider or {}).get("id"),
                    "model_requested": (judge_provider or {}).get("model"),
                    "model_returned": None,
                    "error": judge_error,
                }

        if new_judge_score and new_judge_score.get("score") is not None:
            new_final_score = {
                "score": new_judge_score.get("score"),
                "format_ok": new_judge_score.get("format_ok"),
                "details": new_judge_score.get("details") or new_judge_score.get("error"),
            }
        elif new_rule_score:
            new_final_score = {
                "score": new_rule_score.get("score"),
                "format_ok": new_rule_score.get("format_ok"),
                "details": new_rule_score.get("details"),
            }
        else:
            new_final_score = {"score": None, "format_ok": False, "details": rescore_error}

        rescored_at = datetime.now().isoformat(timespec="seconds")
        record_status = "failed" if rescore_error else "judge_failed" if judge_error else "completed"
        record = {
            "rescore_id": rescore_id,
            "status": record_status,
            "source_run_id": source.get("source_run_id"),
            "source_record_id": source.get("source_record_id"),
            "source_kind": source.get("source_kind"),
            "source_provider_id": source_provider_id,
            "task": {
                "id": task.get("id") or task_id,
                "category": task.get("category"),
                "enterprise_dimension": task.get("enterprise_dimension"),
                "scoring_type": task.get("scoring_type"),
            },
            "provider": source.get("provider") or {},
            "source_response_file": source.get("source_response_file"),
            "source_events_file": source.get("source_events_file"),
            "source_error": source.get("source_error"),
            "original_score": source.get("original_score"),
            "new_rule_score": new_rule_score,
            "new_judge_score": new_judge_score,
            "new_final_score": new_final_score,
            "judge_provider": (new_judge_score or {}).get("provider") or (judge_provider or {}).get("id"),
            "judge_model_requested": (new_judge_score or {}).get("model_requested") or (judge_provider or {}).get("model"),
            "judge_model_returned": (new_judge_score or {}).get("model_returned"),
            "rubric_version": judge_rubric_version,
            "rescored_at": rescored_at,
            "rescore_error": rescore_error,
            "judge_error": judge_error,
        }
        records.append(record)
        summary_row = {
            "rescore_id": rescore_id,
            "source_run_id": source.get("source_run_id"),
            "source_record_id": source.get("source_record_id"),
            "task_id": task.get("id") or task_id,
            "provider_id": source_provider_id,
            "original_score_0_10": score_value(source.get("original_score")),
            "new_rule_score_0_10": score_value(new_rule_score),
            "new_judge_score_0_10": score_value(new_judge_score),
            "new_final_score_0_10": score_value(new_final_score),
            "judge_provider": record.get("judge_provider"),
            "judge_model_requested": record.get("judge_model_requested"),
            "judge_model_returned": record.get("judge_model_returned"),
            "rubric_version": judge_rubric_version,
            "status": record_status,
            "rescore_error": rescore_error,
            "judge_error": judge_error,
            "source_response_file": source.get("source_response_file"),
            "source_events_file": source.get("source_events_file"),
        }
        summary_rows.append(summary_row)

        if progress_callback:
            progress_callback(
                {
                    "event": "task_completed",
                    "task_id": task_id,
                    "ok": not rescore_error and not judge_error,
                    "error": rescore_error or judge_error,
                    "phase": "completed",
                    "completed_tasks": index + 1,
                    "total_tasks": total,
                }
            )

    records_path = rescore_dir / "rescore_records.jsonl"
    summary_path = rescore_dir / "rescore_summary.csv"
    manifest_path = rescore_dir / "rescore_manifest.json"
    write_jsonl(records_path, records)
    write_rescore_summary(summary_path, summary_rows)

    completed_count = sum(1 for row in summary_rows if row.get("status") == "completed")
    judge_error_count = sum(1 for row in summary_rows if row.get("status") == "judge_failed")
    failure_count = sum(1 for row in summary_rows if row.get("status") == "failed")
    completed_at = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "rescore_id": rescore_id,
        "source_run_id": run_id,
        "status": "stopped" if stopped else "completed",
        "created_at": created_at,
        "completed_at": completed_at,
        "rescore_label": rescore_label,
        "rubric_version": judge_rubric_version,
        "mode": "rule_and_judge" if judge_response else "rule_only",
        "judge_enabled": bool(judge_response),
        "judge_provider": (judge_provider or {}).get("id"),
        "judge_model_requested": (judge_provider or {}).get("model"),
        "filters": {
            "provider_id": provider_id,
            "task_ids": task_ids or [],
        },
        "total_source_records": len(all_sources),
        "record_count": len(records),
        "completed_count": completed_count,
        "success_count": completed_count,
        "judge_error_count": judge_error_count,
        "failure_count": failure_count,
        "stopped": stopped,
        "records_file": str(records_path),
        "summary_file": str(summary_path),
        "manifest_file": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "rescore_id": rescore_id,
        "record_count": len(records),
        "summary": summary_rows,
        "manifest": manifest,
        "stopped": stopped,
    }
