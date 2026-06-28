from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarking import SCORE_FORMULA_VERSION
from run_records import build_run_record, validate_run_record


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_response_text(response_file: str | None) -> str:
    if not response_file:
        return ""
    path = Path(response_file)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def benchmark_context(results_path: Path) -> tuple[str, str]:
    scores_path = results_path.parent / "benchmark_scores.json"
    if not scores_path.exists():
        return "historical", SCORE_FORMULA_VERSION
    try:
        scores = read_json(scores_path)
    except Exception:
        return "historical", SCORE_FORMULA_VERSION
    return (
        str(scores.get("benchmark_mode") or "historical"),
        str(scores.get("formula_version") or SCORE_FORMULA_VERSION),
    )


def records_from_results(results_path: Path) -> list[dict[str, Any]]:
    items = read_json(results_path)
    if not isinstance(items, list):
        raise ValueError("results JSON must be an array")
    benchmark_mode, formula_version = benchmark_context(results_path)
    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task = item.get("task") or {}
        metrics = item.get("metrics") or {}
        provider = item.get("provider") or {}
        response_file = item.get("response_file") or ""
        events_file = item.get("events_file") or ""
        response_text = load_response_text(response_file)
        max_tokens = task.get("recommended_max_tokens") or 0
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens = 0
        status = "completed" if metrics.get("ok") else "failed"
        records.append(
            build_run_record(
                run_id=str(item.get("run_id") or results_path.parent.name),
                timestamp=str(item.get("timestamp") or ""),
                benchmark_mode=benchmark_mode,
                formula_version=formula_version,
                runner="historical",
                status=status,
                task=task,
                provider=provider,
                metrics=metrics,
                final_score=item.get("score") or {},
                rule_score=item.get("rule_score") or item.get("score") or {},
                judge_score=item.get("judge_score"),
                response_text=response_text,
                response_file=response_file,
                events_file=events_file,
                max_tokens=max_tokens,
                temperature=None,
                system_prompt=None,
            )
        )
    return records


def validate_records(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        for error in validate_run_record(record):
            errors.append(f"record {index}: {error}")
    return errors


def records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def self_test_records() -> list[dict[str, Any]]:
    base_task = {
        "id": "fixture_task",
        "category": "fixture",
        "enterprise_dimension": "fixture_dimension",
        "difficulty": "medium",
        "scoring_type": "keyword_check",
        "risk_tags": ["fixture"],
        "point_value": 100,
        "scoring_confidence": 0.65,
        "recommended_max_tokens": 256,
        "prompt": "fixture prompt",
    }
    base_provider = {
        "id": "fixture_provider",
        "base_url": "https://provider.example.com/anthropic",
        "model": "model-a",
        "auth_env": "FIXTURE_API_KEY",
    }
    base_metrics = {
        "ok": True,
        "error": None,
        "first_event_ms": 100,
        "first_content_token_ms": 200,
        "total_ms": 500,
        "event_count": 5,
        "content_event_count": 2,
        "content_chars": 20,
        "input_tokens": 10,
        "output_tokens": 8,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "server_model": "model-a",
        "stop_reason": "end_turn",
    }

    cases: list[tuple[str, str, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]] = [
        ("success", "completed", base_metrics, {"score": 9.0}, None),
        (
            "api_failed",
            "failed",
            {**base_metrics, "ok": False, "error": "HTTP 500", "server_model": None},
            {"score": None, "format_ok": False, "details": "HTTP 500"},
            None,
        ),
        (
            "judge_failed",
            "completed",
            base_metrics,
            {"score": None, "details": "judge failed"},
            {"error": "judge HTTP 500", "provider": "judge", "model_requested": "judge-a", "model_returned": None},
        ),
        ("partial_stop", "partial", base_metrics, {"score": 8.0}, None),
        (
            "cache_missing_or_zero",
            "completed",
            {**base_metrics, "cache_creation_input_tokens": None, "cache_read_input_tokens": 0},
            {"score": 7.0},
            None,
        ),
        (
            "model_mismatch",
            "completed",
            {**base_metrics, "server_model": "model-b"},
            {"score": 6.0},
            None,
        ),
    ]
    records: list[dict[str, Any]] = []
    for suffix, status, metrics, score, judge_score in cases:
        task = {**base_task, "id": f"fixture_{suffix}"}
        records.append(
            build_run_record(
                run_id="fixture_run",
                timestamp="2026-06-20T00:00:00",
                benchmark_mode="mode_10",
                formula_version=SCORE_FORMULA_VERSION,
                runner="cli",
                status=status,
                task=task,
                provider=base_provider,
                metrics=metrics,
                final_score=score,
                rule_score=score,
                judge_score=judge_score,
                response_text="fixture response",
                response_file=f"responses/fixture/{suffix}.txt",
                events_file=f"events/fixture/{suffix}.jsonl",
                max_tokens=256,
                temperature=0,
                system_prompt=None,
            )
        )
    return records


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Validate run_record_v1 JSONL or historical results.json")
    parser.add_argument("--jsonl", type=Path, help="run_records.jsonl to validate")
    parser.add_argument("--results", type=Path, help="historical results.json to convert and validate")
    parser.add_argument("--write-jsonl", type=Path, help="optional output path for converted records")
    parser.add_argument("--self-test", action="store_true", help="run synthetic coverage fixtures")
    args = parser.parse_args()

    batches: list[tuple[str, list[dict[str, Any]]]] = []
    if args.self_test:
        batches.append(("self-test", self_test_records()))
    if args.jsonl:
        batches.append((str(args.jsonl), records_from_jsonl(args.jsonl)))
    if args.results:
        converted = records_from_results(args.results)
        batches.append((str(args.results), converted))
        if args.write_jsonl:
            write_jsonl(args.write_jsonl, converted)

    if not batches:
        parser.error("provide --self-test, --jsonl, or --results")

    total = 0
    all_errors: list[str] = []
    for label, records in batches:
        total += len(records)
        errors = validate_records(records)
        if errors:
            all_errors.extend(f"{label}: {error}" for error in errors)
        else:
            print(f"ok: {label}: {len(records)} records")

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validated {total} run_record_v1 records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
