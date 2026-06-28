"""Unit tests for benchmarking.py — scoring math + task selection.

benchmarking.py computes the multi-component provider score (quality, latency,
cost, risk penalties) and selects which tasks run in each benchmark mode. The
module self-test touched a few cases; these drive the pure scoring helpers and
the selection logic across their branches so a formula or quota regression is
caught.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmarking as B  # noqa: E402


# ---------------------------------------------------------------------------
# numeric / round_or_none / weighted_average
# ---------------------------------------------------------------------------
def test_numeric_coercions():
    assert B.numeric("3.5") == 3.5
    assert B.numeric(None, 1.0) == 1.0
    assert B.numeric("", 2.0) == 2.0
    assert B.numeric("nope", 9.0) == 9.0
    assert B.numeric("nope") is None


def test_round_or_none():
    assert B.round_or_none(None) is None
    assert B.round_or_none(1.23456) == 1.23


def test_weighted_average():
    assert B.weighted_average([]) is None
    assert B.weighted_average([(None, 1.0)]) is None
    assert B.weighted_average([(10.0, 1.0), (20.0, 1.0)]) == 15.0
    # weighted: (10*1 + 20*3) / 4 = 17.5
    assert B.weighted_average([(10.0, 1.0), (20.0, 3.0)]) == 17.5
    # zero-weight pairs are dropped
    assert B.weighted_average([(10.0, 0.0), (20.0, 2.0)]) == 20.0


# ---------------------------------------------------------------------------
# coerce_scoring_confidence
# ---------------------------------------------------------------------------
def test_coerce_confidence_number_passthrough():
    assert B.coerce_scoring_confidence(0.9, "json_exact") == 0.9


def test_coerce_confidence_word_maps():
    assert B.coerce_scoring_confidence("high", "json_exact") == B.SCORING_CONFIDENCE_WORDS["high"]


def test_coerce_confidence_default_on_empty():
    expected = B.SCORING_CONFIDENCE_DEFAULTS.get("keyword_check", 0.5)
    assert B.coerce_scoring_confidence("", "keyword_check") == float(expected)


def test_coerce_confidence_unknown_word_falls_back():
    expected = B.SCORING_CONFIDENCE_DEFAULTS.get("manual", 0.5)
    assert B.coerce_scoring_confidence("ginormous", "manual") == float(expected)


# ---------------------------------------------------------------------------
# quality_ratio — derived from explicit score or format_ok fallback
# ---------------------------------------------------------------------------
def test_quality_ratio_explicit_score():
    assert B.quality_ratio({"quality_0_10": 8.0}) == 0.8


def test_quality_ratio_falls_back_to_score_field():
    assert B.quality_ratio({"score_0_10": 5.0}) == 0.5


def test_quality_ratio_judge_error_is_zero():
    assert B.quality_ratio({"judge_error": "boom"}) == 0.0


def test_quality_ratio_format_ok_true_fallback():
    assert B.quality_ratio({"format_ok": "true"}) == 0.8


def test_quality_ratio_format_ok_false_fallback():
    assert B.quality_ratio({"format_ok": "false"}) == 0.2


def test_quality_ratio_ok_only_fallback():
    assert B.quality_ratio({"ok": "true"}) == 0.5
    assert B.quality_ratio({"ok": "false"}) == 0.0


def test_quality_ratio_clamps():
    assert B.quality_ratio({"quality_0_10": 50.0}) == 1.0


# placeholder-bench


# ---------------------------------------------------------------------------
# risk_penalty — additive penalties per failure signal
# ---------------------------------------------------------------------------
def test_risk_penalty_clean_row_is_zero():
    row = {"ok": "true"}
    assert B.risk_penalty(row, 100.0) == 0.0


def test_risk_penalty_not_ok():
    assert B.risk_penalty({"ok": "false"}, 100.0) == pytest.approx(22.0)


def test_risk_penalty_model_mismatch():
    row = {"ok": "true", "model_requested": "claude-opus-4-6", "model_returned": "gpt-4o"}
    assert B.risk_penalty(row, 100.0) == pytest.approx(14.0)


def test_risk_penalty_max_tokens_truncation():
    row = {"ok": "true", "stop_reason": "max_tokens"}
    assert B.risk_penalty(row, 100.0) == pytest.approx(12.0)


def test_risk_penalty_stacks():
    row = {"ok": "false", "judge_error": "x", "format_ok": "false"}
    # 0.22 + 0.18 + 0.10 = 0.50 of 100
    assert B.risk_penalty(row, 100.0) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# calculate_latency_score / calculate_cost_score
# ---------------------------------------------------------------------------
def test_latency_score_none_without_data():
    assert B.calculate_latency_score([{"first_content_token_ms": None}]) is None


def test_latency_score_lower_is_better():
    fast = B.calculate_latency_score([{"first_content_token_ms": 500}])
    slow = B.calculate_latency_score([{"first_content_token_ms": 5000}])
    assert fast > slow
    assert 0.0 <= slow <= fast <= 1000.0


def test_cost_score_none_without_tokens():
    assert B.calculate_cost_score([{"input_tokens": 0, "output_tokens": 0}]) is None


def test_cost_score_fewer_tokens_better():
    cheap = B.calculate_cost_score([{"input_tokens": 100, "output_tokens": 100}])
    pricey = B.calculate_cost_score([{"input_tokens": 50000, "output_tokens": 50000}])
    assert cheap > pricey


# ---------------------------------------------------------------------------
# task_benchmark_defaults / enrich_task_metadata
# ---------------------------------------------------------------------------
def test_task_defaults_difficulty_point_value():
    d = B.task_benchmark_defaults({"difficulty": "hard", "scoring_type": "json_exact"})
    assert d["point_value"] == B.DIFFICULTY_POINT_VALUES["hard"]


def test_task_defaults_anchor_role_from_id_suffix():
    d = B.task_benchmark_defaults({"id": "task_001", "difficulty": "easy"})
    assert d["benchmark_roles"] == ["anchor"]


def test_task_defaults_long_tail_role():
    d = B.task_benchmark_defaults({"id": "task_random", "difficulty": "easy"})
    assert d["benchmark_roles"] == ["long_tail"]


def test_task_defaults_dimension_group_fallback_chain():
    d = B.task_benchmark_defaults({"id": "x", "category": "Reasoning"})
    assert d["dimension_weight_group"] == "Reasoning"
    d2 = B.task_benchmark_defaults({"id": "x"})
    assert d2["dimension_weight_group"] == "Unspecified"


def test_enrich_task_metadata_merges():
    enriched = B.enrich_task_metadata({"id": "task_001", "difficulty": "hard", "prompt": "p"})
    assert enriched["prompt"] == "p"            # original preserved
    assert enriched["point_value"] == 140       # default added


# ---------------------------------------------------------------------------
# select_benchmark_tasks
# ---------------------------------------------------------------------------
def _tasks():
    return [
        {"id": "task_001", "difficulty": "easy", "scoring_type": "json_exact",
         "enterprise_dimension": "D1", "benchmark_roles": ["anchor"], "mode_eligible": ["m"]},
        {"id": "task_002", "difficulty": "easy", "scoring_type": "json_exact",
         "enterprise_dimension": "D1", "benchmark_roles": ["long_tail"], "mode_eligible": ["m"]},
        {"id": "task_003", "difficulty": "easy", "scoring_type": "json_exact",
         "enterprise_dimension": "D2", "benchmark_roles": ["long_tail"], "mode_eligible": ["m"]},
    ]


def test_select_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown benchmark mode"):
        B.select_benchmark_tasks(_tasks(), "nope", {"modes": {}})


def test_select_returns_all_when_target_exceeds():
    cfg = {"modes": {"m": {"target_count": 99}}}
    out = B.select_benchmark_tasks(_tasks(), "m", cfg)
    assert len(out) == 3


def test_select_respects_target_count_and_includes_anchor():
    cfg = {"modes": {"m": {"target_count": 2}}}
    out = B.select_benchmark_tasks(_tasks(), "m", cfg)
    assert len(out) == 2
    # the anchor task should be selected
    assert any(t["id"] == "task_001" for t in out)


def test_select_filters_by_mode_eligibility():
    tasks = [
        {"id": "a", "difficulty": "easy", "scoring_type": "json_exact",
         "mode_eligible": ["other"]},
        {"id": "b", "difficulty": "easy", "scoring_type": "json_exact",
         "mode_eligible": ["m"]},
    ]
    cfg = {"modes": {"m": {"target_count": 99}}}
    out = B.select_benchmark_tasks(tasks, "m", cfg)
    assert [t["id"] for t in out] == ["b"]


# ---------------------------------------------------------------------------
# calculate_provider_score — integration over the math helpers
# ---------------------------------------------------------------------------
def test_calculate_provider_score_clean_rows():
    rows = [
        {"ok": "true", "quality_0_10": 9.0, "point_value": 100,
         "scoring_confidence": 0.9, "enterprise_dimension": "D1",
         "first_content_token_ms": 800, "input_tokens": 100, "output_tokens": 50},
        {"ok": "true", "quality_0_10": 8.0, "point_value": 100,
         "scoring_confidence": 0.9, "enterprise_dimension": "D1",
         "first_content_token_ms": 900, "input_tokens": 120, "output_tokens": 60},
    ]
    out = B.calculate_provider_score(rows, "m", "v1")
    assert out["task_count"] == 2
    assert out["mode"] == "m"
    assert out["formula_version"] == "v1"
    assert 0.0 <= out["benchmark_score"] <= 1000.0
    assert out["quality_score"] is not None
    assert out["risk_penalty"] == 0.0


def test_calculate_provider_score_failures_lower_score():
    good = B.calculate_provider_score(
        [{"ok": "true", "quality_0_10": 9.0, "point_value": 100,
          "enterprise_dimension": "D1"}], "m", "v1")
    bad = B.calculate_provider_score(
        [{"ok": "false", "quality_0_10": 9.0, "point_value": 100,
          "enterprise_dimension": "D1"}], "m", "v1")
    assert bad["benchmark_score"] < good["benchmark_score"]
    assert bad["risk_penalty"] > 0.0

