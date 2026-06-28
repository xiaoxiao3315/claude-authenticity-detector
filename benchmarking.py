from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCORE_FORMULA_VERSION = "score_formula_v1"

MODE_ORDER = ["mode_10", "mode_100", "mode_1000", "mode_10000", "mode_100000"]

DIMENSION_SCORE_MAP = {
    "API 行为与 Provider 可信度": "provider_trust_score",
    "JSON / 结构化输出稳定性": "json_stability_score",
    "代码审查": "engineering_score",
    "代码生成": "engineering_score",
    "补丁修复 / 最小改动": "engineering_score",
    "Bug 定位": "engineering_score",
    "中文需求拆解": "quality_score",
    "长上下文总结": "quality_score",
    "技术方案设计": "engineering_score",
    "Agent / 工具使用规划": "engineering_score",
    "反幻觉 / 不确定性处理": "provider_trust_score",
    "成本、延迟、质量取舍判断": "cost_efficiency_score",
}

SCORING_CONFIDENCE_DEFAULTS = {
    "json_exact": 0.95,
    "keyword_check": 0.65,
    "manual_rubric": 0.55,
    "artifact_review": 0.55,
    "manual": 0.5,
}

# Some task files express scoring_confidence as a word ("high") rather than a
# 0..1 float. Map known words; anything else falls back to the scoring_type
# default so a single bad value can't crash the whole task-selection pipeline.
SCORING_CONFIDENCE_WORDS = {
    "very_high": 0.95,
    "high": 0.85,
    "medium": 0.6,
    "moderate": 0.6,
    "low": 0.4,
    "very_low": 0.25,
}


def coerce_scoring_confidence(value: Any, scoring_type: str) -> float:
    """Best-effort 0..1 confidence. Accepts a number, a known word, or falls
    back to the scoring-type default. Never raises on bad input."""
    default = SCORING_CONFIDENCE_DEFAULTS.get(scoring_type, 0.5)
    if value is None or value == "":
        return float(default)
    parsed = numeric(value)
    if parsed is not None:
        return parsed
    word = SCORING_CONFIDENCE_WORDS.get(str(value).strip().lower())
    return float(word if word is not None else default)


DIFFICULTY_POINT_VALUES = {
    "easy": 80,
    "medium": 100,
    "hard": 140,
}


def load_benchmark_modes(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    modes = data.get("modes")
    if not isinstance(modes, dict):
        raise ValueError("benchmark modes file must contain a modes object")
    return data


def benchmark_mode_options(config: dict[str, Any]) -> list[dict[str, Any]]:
    modes = config.get("modes") or {}
    return [
        {"id": mode_id, **modes[mode_id]}
        for mode_id in MODE_ORDER
        if mode_id in modes
    ]


def task_benchmark_defaults(task: dict[str, Any]) -> dict[str, Any]:
    difficulty = str(task.get("difficulty") or "medium")
    scoring_type = str(task.get("scoring_type") or "manual")
    roles = task.get("benchmark_roles")
    if not isinstance(roles, list) or not roles:
        roles = ["anchor"] if task.get("id", "").endswith(("_001", "_002", "_003")) else ["long_tail"]
    eligible = task.get("mode_eligible")
    if not isinstance(eligible, list) or not eligible:
        eligible = list(MODE_ORDER)
    return {
        "point_value": int(task.get("point_value") or DIFFICULTY_POINT_VALUES.get(difficulty, 100)),
        "benchmark_roles": roles,
        "mode_eligible": eligible,
        "dimension_weight_group": task.get("dimension_weight_group")
        or task.get("enterprise_dimension")
        or task.get("category")
        or "Unspecified",
        "scoring_confidence": coerce_scoring_confidence(
            task.get("scoring_confidence"), scoring_type
        ),
    }


def enrich_task_metadata(task: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(task)
    enriched.update(task_benchmark_defaults(task))
    return enriched


def select_benchmark_tasks(
    tasks: list[dict[str, Any]],
    mode_id: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = (config.get("modes") or {}).get(mode_id)
    if not isinstance(mode, dict):
        raise ValueError(f"unknown benchmark mode: {mode_id}")

    target_count = int(mode.get("target_count", len(tasks)))
    eligible = [
        enrich_task_metadata(task)
        for task in tasks
        if mode_id in task_benchmark_defaults(task)["mode_eligible"]
    ]
    if target_count <= 0 or target_count >= len(eligible):
        return sorted(eligible, key=task_sort_key)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    calibration = [
        task
        for task in eligible
        if "calibration" in task.get("benchmark_roles", [])
    ]
    for task in sorted(calibration, key=task_sort_key):
        if len(selected) >= target_count:
            break
        selected.append(task)
        selected_ids.add(task["id"])

    quotas = mode.get("dimension_quotas") or {}
    for dimension, quota in quotas.items():
        if len(selected) >= target_count:
            break
        candidates = [
            task
            for task in eligible
            if task.get("dimension_weight_group") == dimension
            and task["id"] not in selected_ids
        ]
        for task in sorted(candidates, key=task_sort_key)[: int(quota)]:
            if len(selected) >= target_count:
                break
            selected.append(task)
            selected_ids.add(task["id"])

    anchors = [
        task
        for task in eligible
        if "anchor" in task.get("benchmark_roles", [])
        and task["id"] not in selected_ids
    ]
    for task in sorted(anchors, key=task_sort_key):
        if len(selected) >= target_count:
            break
        selected.append(task)
        selected_ids.add(task["id"])

    by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in eligible:
        if task["id"] not in selected_ids:
            by_dimension[str(task.get("dimension_weight_group"))].append(task)

    while len(selected) < target_count:
        added = False
        for dimension in sorted(by_dimension):
            candidates = by_dimension[dimension]
            while candidates and candidates[0]["id"] in selected_ids:
                candidates.pop(0)
            if not candidates:
                continue
            task = sorted(candidates, key=task_sort_key)[0]
            candidates.remove(task)
            selected.append(task)
            selected_ids.add(task["id"])
            added = True
            if len(selected) >= target_count:
                break
        if not added:
            break

    return sorted(selected, key=task_sort_key)


def task_sort_key(task: dict[str, Any]) -> tuple[str, int, str]:
    dimension = str(task.get("dimension_weight_group") or task.get("enterprise_dimension") or "")
    role_rank = 0 if "anchor" in task.get("benchmark_roles", []) else 1
    return dimension, role_rank, str(task.get("id"))


def calculate_benchmark_scores(
    summary_rows: list[dict[str, Any]],
    mode_id: str = "custom",
    formula_version: str = SCORE_FORMULA_VERSION,
) -> dict[str, Any]:
    rows_by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        provider = str(row.get("provider") or "unknown")
        rows_by_provider[provider].append(row)

    providers: dict[str, Any] = {}
    for provider, rows in rows_by_provider.items():
        providers[provider] = calculate_provider_score(rows, mode_id, formula_version)

    return {
        "formula_version": formula_version,
        "benchmark_mode": mode_id,
        "providers": providers,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def calculate_provider_score(
    rows: list[dict[str, Any]],
    mode_id: str,
    formula_version: str,
) -> dict[str, Any]:
    component_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    dimension_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    penalties = 0.0
    confidence_weight = 0.0
    confidence_total = 0.0
    point_total = 0.0
    point_earned = 0.0

    for row in rows:
        point_value = numeric(row.get("point_value"), 100.0) or 100.0
        confidence = numeric(row.get("scoring_confidence"), 0.55) or 0.55
        quality = quality_ratio(row)
        ok = str(row.get("ok")).lower() == "true"
        if not ok:
            quality = 0.0
        point_total += point_value
        point_earned += point_value * quality
        confidence_weight += point_value
        confidence_total += confidence * point_value

        dimension = str(row.get("enterprise_dimension") or row.get("dimension_weight_group") or "Unspecified")
        component = DIMENSION_SCORE_MAP.get(dimension, "quality_score")
        component_values[component].append((quality * 1000.0, point_value))
        dimension_values[dimension].append((quality * 1000.0, point_value))

        penalties += risk_penalty(row, point_value)

    quality_score = weighted_average([(point_earned / point_total * 1000.0, point_total)]) if point_total else None
    latency_score = calculate_latency_score(rows)
    cost_efficiency_score = calculate_cost_score(rows)

    components: dict[str, Any] = {
        "quality_score": round_or_none(quality_score),
        "engineering_score": round_or_none(weighted_average(component_values.get("engineering_score", []))),
        "provider_trust_score": round_or_none(weighted_average(component_values.get("provider_trust_score", []))),
        "json_stability_score": round_or_none(weighted_average(component_values.get("json_stability_score", []))),
        "latency_score": round_or_none(latency_score),
        "cost_efficiency_score": round_or_none(cost_efficiency_score),
        "risk_penalty": round(penalties, 2),
        "confidence_level": round(confidence_total / confidence_weight, 3) if confidence_weight else None,
    }
    benchmark_score = (
        (components["quality_score"] or 0) * 0.48
        + (components["engineering_score"] or components["quality_score"] or 0) * 0.12
        + (components["provider_trust_score"] or components["quality_score"] or 0) * 0.12
        + (components["json_stability_score"] or components["quality_score"] or 0) * 0.10
        + (components["latency_score"] or 0) * 0.08
        + (components["cost_efficiency_score"] or 0) * 0.10
        - components["risk_penalty"]
    )
    components["benchmark_score"] = round(max(0.0, benchmark_score), 2)
    components["task_count"] = len(rows)
    components["mode"] = mode_id
    components["formula_version"] = formula_version
    components["dimension_scores"] = {
        dimension: round_or_none(weighted_average(values))
        for dimension, values in sorted(dimension_values.items())
    }
    return components


def quality_ratio(row: dict[str, Any]) -> float:
    score = numeric(row.get("quality_0_10"))
    if score is None:
        score = numeric(row.get("score_0_10"))
    if score is None and row.get("judge_error"):
        return 0.0
    if score is None:
        if str(row.get("format_ok")).lower() == "true":
            score = 8.0
        elif str(row.get("format_ok")).lower() == "false":
            score = 2.0
        else:
            score = 5.0 if str(row.get("ok")).lower() == "true" else 0.0
    return max(0.0, min(1.0, score / 10.0))


def risk_penalty(row: dict[str, Any], point_value: float) -> float:
    penalty = 0.0
    if str(row.get("ok")).lower() != "true":
        penalty += 0.22 * point_value
    if row.get("judge_error"):
        penalty += 0.18 * point_value
    if row.get("judge_provider") and numeric(row.get("judge_score_0_10")) is None:
        penalty += 0.12 * point_value
    if str(row.get("judge_format_ok")).lower() == "false":
        penalty += 0.08 * point_value
    if str(row.get("stop_reason")) == "max_tokens":
        penalty += 0.12 * point_value
    if str(row.get("format_ok")).lower() == "false":
        penalty += 0.10 * point_value
    requested = str(row.get("model_requested") or "")
    returned = str(row.get("model_returned") or "")
    if requested and returned and requested != returned:
        penalty += 0.14 * point_value
    if requested and not returned:
        penalty += 0.04 * point_value
    return penalty


def calculate_latency_score(rows: list[dict[str, Any]]) -> float | None:
    raw = [numeric(row.get("first_content_token_ms")) for row in rows]
    values = [value for value in raw if value is not None and value > 0]
    if not values:
        return None
    avg_ms = sum(values) / len(values)
    return max(0.0, min(1000.0, 1000.0 * (2500.0 / (avg_ms + 2500.0))))


def calculate_cost_score(rows: list[dict[str, Any]]) -> float | None:
    totals: list[float] = []
    cache_reads: list[float] = []
    for row in rows:
        input_tokens = numeric(row.get("input_tokens"), 0.0) or 0.0
        cc = numeric(row.get("cache_creation_input_tokens"), 0.0) or 0.0
        cr = numeric(row.get("cache_read_input_tokens"), 0.0) or 0.0
        output = numeric(row.get("output_tokens"), 0.0) or 0.0
        total = input_tokens + cc + cr + output
        if total > 0:
            totals.append(total)
            cache_reads.append(cr)
    if not totals:
        return None
    avg_tokens = sum(totals) / len(totals)
    cache_ratio = sum(cache_reads) / sum(totals) if sum(totals) else 0.0
    base = 1000.0 * (8000.0 / (avg_tokens + 8000.0))
    return max(0.0, min(1000.0, base + cache_ratio * 160.0))


def weighted_average(values: list[tuple[Any, float]]) -> float | None:
    pairs = [(value, weight) for value, weight in values if value is not None and weight > 0]
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total_weight


def numeric(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)


def ensure_index_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              timestamp TEXT,
              benchmark_mode TEXT,
              formula_version TEXT,
              provider_count INTEGER,
              task_count INTEGER,
              benchmark_scores_json TEXT,
              run_dir TEXT
            );
            CREATE TABLE IF NOT EXISTS results (
              run_id TEXT,
              provider TEXT,
              task_id TEXT,
              enterprise_dimension TEXT,
              benchmark_mode TEXT,
              quality_0_10 REAL,
              benchmark_score REAL,
              ok INTEGER,
              stop_reason TEXT,
              first_content_token_ms REAL,
              total_ms REAL,
              input_tokens REAL,
              cache_creation_input_tokens REAL,
              cache_read_input_tokens REAL,
              output_tokens REAL,
              response_file TEXT,
              PRIMARY KEY (run_id, provider, task_id)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def index_run(
    db_path: Path,
    run_id: str,
    run_dir: Path,
    summary_rows: list[dict[str, Any]],
    benchmark_scores: dict[str, Any],
) -> None:
    ensure_index_schema(db_path)
    providers = {row.get("provider") for row in summary_rows}
    task_ids = {row.get("task_id") for row in summary_rows}
    timestamp = summary_rows[0].get("timestamp") if summary_rows else datetime.now().isoformat(timespec="seconds")
    mode = str(benchmark_scores.get("benchmark_mode") or "custom")
    formula = str(benchmark_scores.get("formula_version") or SCORE_FORMULA_VERSION)
    provider_scores = benchmark_scores.get("providers") or {}

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, timestamp, benchmark_mode, formula_version, provider_count, task_count, benchmark_scores_json, run_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                mode,
                formula,
                len(providers),
                len(task_ids),
                json.dumps(benchmark_scores, ensure_ascii=False),
                str(run_dir),
            ),
        )
        for row in summary_rows:
            provider = str(row.get("provider") or "unknown")
            provider_score = (provider_scores.get(provider) or {}).get("benchmark_score")
            conn.execute(
                """
                INSERT OR REPLACE INTO results
                (run_id, provider, task_id, enterprise_dimension, benchmark_mode, quality_0_10,
                 benchmark_score, ok, stop_reason, first_content_token_ms, total_ms, input_tokens,
                 cache_creation_input_tokens, cache_read_input_tokens, output_tokens, response_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("run_id"),
                    provider,
                    row.get("task_id"),
                    row.get("enterprise_dimension"),
                    mode,
                    numeric(row.get("quality_0_10") or row.get("score_0_10")),
                    provider_score,
                    1 if str(row.get("ok")).lower() == "true" else 0,
                    row.get("stop_reason"),
                    numeric(row.get("first_content_token_ms")),
                    numeric(row.get("total_ms")),
                    numeric(row.get("input_tokens")),
                    numeric(row.get("cache_creation_input_tokens")),
                    numeric(row.get("cache_read_input_tokens")),
                    numeric(row.get("output_tokens")),
                    row.get("response_file"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def paged_runs_from_index(db_path: Path, limit: int = 20, offset: int = 0) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    ensure_index_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        rows = conn.execute(
            """
            SELECT * FROM runs
            ORDER BY timestamp DESC, run_id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    finally:
        conn.close()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": [dict(row) for row in rows],
    }


def _self_test() -> int:
    """Offline unit checks for the pure scoring helpers. No DB, no network."""
    # numeric: empty/None -> default; bad -> default; good -> float.
    assert numeric("3.5") == 3.5 and numeric(2) == 2.0
    assert numeric("") is None and numeric(None) is None
    assert numeric("", default=0.0) == 0.0 and numeric("x", default=-1.0) == -1.0

    # round_or_none.
    assert round_or_none(None) is None
    assert round_or_none(1.23456) == 1.23

    # weighted_average: ignores None values and non-positive weights.
    assert weighted_average([]) is None
    assert weighted_average([(10.0, 1.0), (20.0, 1.0)]) == 15.0
    assert weighted_average([(10.0, 3.0), (20.0, 1.0)]) == 12.5
    assert weighted_average([(10.0, 1.0), (None, 5.0)]) == 10.0  # None dropped
    assert weighted_average([(10.0, 0.0)]) is None  # zero weight dropped

    # quality_ratio: explicit score, judge_error short-circuit, format_ok fallbacks.
    assert quality_ratio({"quality_0_10": 8}) == 0.8
    assert quality_ratio({"score_0_10": 5}) == 0.5
    assert quality_ratio({"judge_error": "boom"}) == 0.0
    assert quality_ratio({"format_ok": "true"}) == 0.8
    assert quality_ratio({"format_ok": "false"}) == 0.2
    assert quality_ratio({"ok": "true"}) == 0.5
    assert quality_ratio({}) == 0.0
    assert quality_ratio({"quality_0_10": 99}) == 1.0  # clamped to 1.0

    # risk_penalty: failed call and model mismatch each add penalty.
    pv = 10.0
    assert risk_penalty({"ok": "true"}, pv) == 0.0
    assert risk_penalty({"ok": "false"}, pv) > 0.0
    mismatch = risk_penalty({"ok": "true", "model_requested": "a", "model_returned": "b"}, pv)
    assert mismatch > 0.0, mismatch

    # task_sort_key returns a comparable tuple.
    k1 = task_sort_key({"category": "a", "point_value": 5, "id": "t1"})
    k2 = task_sort_key({"category": "a", "point_value": 5, "id": "t2"})
    assert isinstance(k1, tuple) and k1 < k2

    # coerce_scoring_confidence: numbers pass through, words map, junk -> type default,
    # and it NEVER raises (this guards the real bug where "high" crashed float()).
    assert coerce_scoring_confidence(0.55, "manual") == 0.55
    assert coerce_scoring_confidence("0.9", "manual") == 0.9
    assert coerce_scoring_confidence("high", "manual") == 0.85
    assert coerce_scoring_confidence("HIGH", "manual") == 0.85
    assert coerce_scoring_confidence(None, "json_exact") == 0.95   # type default
    assert coerce_scoring_confidence("", "keyword_check") == 0.65
    assert coerce_scoring_confidence("nonsense", "manual") == 0.5  # unknown word -> default
    # end-to-end: a task carrying the word form must enrich without raising.
    enriched = task_benchmark_defaults({"id": "x", "scoring_type": "manual", "scoring_confidence": "high"})
    assert enriched["scoring_confidence"] == 0.85

    print("benchmarking self-test ok")
    return 0


if __name__ == "__main__":
    import sys as _sys
    if "--self-test" in _sys.argv:
        raise SystemExit(_self_test())
    _sys.stderr.write("benchmarking.py is a library module; use eval_cli.py for the CLI, or --self-test.\n")
    raise SystemExit(2)
