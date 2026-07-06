"""Tests for quality_gate.py pure helpers, sample extractors, and guards.

quality_gate.py is the release-policy engine. Its self_test covers evaluate_policy
end-to-end, but the leaf helpers (coercion, sample extraction from each source
shape, manifest/provider mismatch guards, rescore application) were only hit
incidentally. These pin them directly. 77% -> higher.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import quality_gate as Q  # noqa: E402


# ---------------------------------------------------------------------------
# numeric / boolish / score_value / ratio / percentile / split_tags
# ---------------------------------------------------------------------------
def test_numeric():
    assert Q.numeric("3.5") == 3.5
    assert Q.numeric(None, 1.0) == 1.0
    assert Q.numeric("", 2.0) == 2.0
    assert Q.numeric("bad", 9.0) == 9.0
    assert Q.numeric("bad") is None


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False),
    ("true", True), ("1", True), ("yes", True), ("Y", True),
    ("false", False), ("0", False), ("no", False), ("n", False),
    (None, None), ("", None), ("maybe", None),
])
def test_boolish(value, expected):
    assert Q.boolish(value) is expected


def test_score_value():
    assert Q.score_value({"score": 8.5}) == 8.5
    assert Q.score_value(7.0) == 7.0
    assert Q.score_value({"no": 1}) is None
    assert Q.score_value("bad") is None


def test_ratio():
    assert Q.ratio(1, 2) == 0.5
    assert Q.ratio(1, 0) is None
    assert Q.ratio(0, 5) == 0.0


def test_percentile():
    assert Q.percentile([], 95) is None
    assert Q.percentile([10.0], 95) == 10.0
    vals = [float(x) for x in range(1, 101)]
    p95 = Q.percentile(vals, 95)
    assert 94 <= p95 <= 96
    # negative values are filtered out
    assert Q.percentile([-5.0, 10.0], 50) == 10.0


def test_split_tags():
    assert Q.split_tags(["a", "b"]) == ["a", "b"]
    assert Q.split_tags(None) == []
    assert Q.split_tags("") == []
    assert Q.split_tags("a, b; c") == ["a", "b", "c"]


def test_is_json_sample():
    assert Q.is_json_sample({"scoring_type": "json_exact"}) is True
    assert Q.is_json_sample({"category": "Structured output"}) is True
    assert Q.is_json_sample({"risk_tags": ["schema"]}) is True
    assert Q.is_json_sample({"category": "Reasoning"}) is False


# placeholder-qg


# ---------------------------------------------------------------------------
# sample extractors — each source shape -> normalized sample
# ---------------------------------------------------------------------------
def test_sample_from_run_record():
    rec = {
        "record_id": "r1:tested:t1",
        "task": {"id": "t1", "category": "C", "scoring_type": "json_exact", "risk_tags": ["x"]},
        "provider": {"id": "tested", "model_requested": "claude-opus-4-6",
                     "model_returned": "claude-opus-4-6"},
        "telemetry": {"ok": True, "first_content_token_ms": 800, "stop_reason": "end_turn"},
        "scoring": {"final_score": {"score": 9.0, "format_ok": True},
                    "judge_score": {"error": None}},
    }
    s = Q.sample_from_run_record(rec)
    assert s["source_kind"] == "run_record"
    assert s["provider_id"] == "tested"
    assert s["ok"] is True
    assert s["score_0_10"] == 9.0
    assert s["format_ok"] is True
    assert s["first_content_token_ms"] == 800.0
    assert s["scoring_source"] == "original"


def test_sample_from_summary_row():
    row = {"run_id": "r1", "provider": "tested", "task_id": "t1", "category": "C",
           "ok": "true", "quality_0_10": "8.5", "format_ok": "false",
           "risk_tags": "a;b", "first_content_token_ms": "900"}
    s = Q.sample_from_summary_row(row)
    assert s["source_record_id"] == "r1:tested:t1"
    assert s["ok"] is True
    assert s["score_0_10"] == 8.5
    assert s["format_ok"] is False
    assert s["risk_tags"] == ["a", "b"]


def test_sample_from_summary_row_score_fallback():
    # quality_0_10 missing -> falls back to score_0_10
    row = {"run_id": "r", "provider": "p", "task_id": "t", "score_0_10": "6.0"}
    s = Q.sample_from_summary_row(row)
    assert s["score_0_10"] == 6.0


def test_sample_from_result_item():
    item = {"run_id": "r1", "task": {"id": "t1", "category": "C"},
            "provider": {"id": "tested", "model": "m"},
            "metrics": {"ok": True, "server_model": "m", "stop_reason": "end_turn"},
            "score": {"score": 7.0, "format_ok": True}}
    s = Q.sample_from_result_item(item)
    assert s["source_kind"] == "results_json"
    assert s["score_0_10"] == 7.0
    assert s["model_returned"] == "m"


# ---------------------------------------------------------------------------
# manifest guards
# ---------------------------------------------------------------------------
def test_manifest_incomplete():
    assert Q.manifest_incomplete({"stopped": True}) is True
    assert Q.manifest_incomplete({"partial": True}) is True
    assert Q.manifest_incomplete({"status": "stopped"}) is True
    assert Q.manifest_incomplete({"status": "partial"}) is True
    assert Q.manifest_incomplete({"status": "completed"}) is False
    assert Q.manifest_incomplete({}) is False


def test_manifest_source_run_mismatch():
    assert Q.manifest_source_run_mismatch({"source_run_id": "other"}, "run1") is True
    assert Q.manifest_source_run_mismatch({"source_run_id": "run1"}, "run1") is False
    assert Q.manifest_source_run_mismatch({}, "run1") is False  # no id -> no mismatch


def test_parse_manifest_time():
    assert Q.parse_manifest_time(None) is None
    assert Q.parse_manifest_time("not a time") is None
    dt = Q.parse_manifest_time("2026-06-28T00:00:00Z")
    assert dt is not None and dt.year == 2026


def test_manifest_age_days_none_without_timestamp():
    assert Q.manifest_age_days({}) is None


def test_manifest_age_days_computes():
    age = Q.manifest_age_days({"completed_at": "2020-01-01T00:00:00Z"})
    assert age is not None and age > 1000  # years ago


def test_rescore_provider_mismatch_by_filter():
    rescore = {"manifest": {"filters": {"provider_id": "other"}}}
    assert Q.rescore_provider_mismatch(rescore, "tested") is True
    rescore_ok = {"manifest": {"filters": {"provider_id": "tested"}}}
    assert Q.rescore_provider_mismatch(rescore_ok, "tested") is False


def test_rescore_provider_mismatch_by_records():
    rescore = {"manifest": {}, "records": [{"provider_id": "other"}]}
    assert Q.rescore_provider_mismatch(rescore, "tested") is True
    rescore_ok = {"manifest": {}, "records": [{"provider_id": "tested"}]}
    assert Q.rescore_provider_mismatch(rescore_ok, "tested") is False


def test_trace_provider_mismatch():
    trace = {"manifest": {"provider_metrics": {"other": {}}}}
    assert Q.trace_provider_mismatch(trace, "tested") is True
    trace_ok = {"manifest": {"provider_metrics": {"tested": {}}}}
    assert Q.trace_provider_mismatch(trace_ok, "tested") is False


# ---------------------------------------------------------------------------
# apply_rescore
# ---------------------------------------------------------------------------
def test_apply_rescore_not_found_passthrough():
    samples = [{"source_record_id": "x", "score_0_10": 5.0}]
    out, applied = Q.apply_rescore(samples, {"found": False}, "tested")
    assert applied == 0
    assert out[0]["score_0_10"] == 5.0
    assert out is not samples  # copies


def test_apply_rescore_overrides_score():
    samples = [{"source_record_id": "rec1", "score_0_10": 3.0, "scoring_source": "original"}]
    rescore = {
        "found": True,
        "by_record_id": {"rec1": {"source_provider_id": "tested",
                                  "new_final_score": {"score": 9.0, "format_ok": True}}},
    }
    out, applied = Q.apply_rescore(samples, rescore, "tested")
    assert applied == 1
    assert out[0]["score_0_10"] == 9.0
    assert out[0]["scoring_source"] == "rescore"
    assert out[0]["format_ok"] is True


def test_apply_rescore_skips_other_provider():
    samples = [{"source_record_id": "rec1", "score_0_10": 3.0}]
    rescore = {
        "found": True,
        "by_record_id": {"rec1": {"source_provider_id": "other",
                                  "new_final_score": {"score": 9.0}}},
    }
    out, applied = Q.apply_rescore(samples, rescore, "tested")
    assert applied == 0
    assert out[0]["score_0_10"] == 3.0


# ---------------------------------------------------------------------------
# issue
# ---------------------------------------------------------------------------
def test_issue_shape():
    i = Q.issue("R1", "model", "quality", 5.0, 6.0, "below threshold")
    assert i == {"rule_id": "R1", "source": "model", "metric": "quality",
                 "observed": 5.0, "threshold": 6.0, "details": "below threshold"}


# ---------------------------------------------------------------------------
# evaluate_policy — the verdict heart: clean GO, and each hard blocker -> NO-GO
# ---------------------------------------------------------------------------
def _clean_metrics(**over):
    m = {
        "compatibility_found": True,
        "compatibility_suite_status": "PASS",
        "compatibility_status": "completed",
        "model_mismatch_count": 0,
        "silent_truncation_count": 0,
        "sample_count": 20,
        "success_rate": 1.0,
        "json_failure_rate": 0.0,
        "gate_score": 900,
    }
    m.update(over)
    return m


def _policy():
    return {"policy_id": "p", "policy_version": "v1", "thresholds": {}}


def _evaluate(metrics):
    return Q.evaluate_policy(
        policy=_policy(), source_run_id="run1", provider_id="tested",
        metrics=metrics, compatibility={"found": True}, trace_evaluation={},
        rescore={}, require_rescore=False,
    )


def test_evaluate_policy_clean_is_go():
    decision, blockers, review, passed = _evaluate(_clean_metrics())
    assert decision == "GO"
    assert blockers == []
    assert "model_identity_ok" in passed
    assert "success_rate_ok" in passed


def test_evaluate_policy_model_mismatch_is_nogo():
    decision, blockers, _, _ = _evaluate(_clean_metrics(model_mismatch_count=3))
    assert decision == "NO-GO"
    assert any(b["rule_id"] == "model_mismatch_blocks" for b in blockers)


def test_evaluate_policy_silent_truncation_is_nogo():
    # the fake-1M hard blocker — a detector's most important NO-GO
    decision, blockers, _, _ = _evaluate(_clean_metrics(silent_truncation_count=1))
    assert decision == "NO-GO"
    assert any(b["rule_id"] == "silent_truncation_blocks" for b in blockers)


def test_evaluate_policy_low_success_rate_is_nogo():
    decision, blockers, _, _ = _evaluate(_clean_metrics(success_rate=0.5))
    assert decision == "NO-GO"
    assert any(b["rule_id"] == "success_rate_too_low" for b in blockers)


def test_evaluate_policy_compatibility_fail_is_nogo():
    decision, blockers, _, _ = _evaluate(_clean_metrics(compatibility_suite_status="FAIL"))
    assert decision == "NO-GO"
    assert any(b["rule_id"] == "compatibility_fail_blocks" for b in blockers)


def test_evaluate_policy_missing_samples_is_review():
    decision, blockers, review, _ = _evaluate(_clean_metrics(sample_count=0))
    assert decision == "REVIEW"
    assert blockers == []
    assert any(r["rule_id"] == "sample_evidence_missing" for r in review)


def test_evaluate_policy_yellow_band_success_is_review():
    decision, blockers, review, _ = _evaluate(_clean_metrics(success_rate=0.96))
    assert decision == "REVIEW"
    assert any(r["rule_id"] == "success_rate_yellow_band" for r in review)


def test_evaluate_policy_low_gate_score_is_nogo():
    decision, blockers, _, _ = _evaluate(_clean_metrics(gate_score=300))
    assert decision == "NO-GO"
    assert any(b["rule_id"] == "gate_score_too_low" for b in blockers)


def test_evaluate_policy_high_json_failure_is_nogo():
    decision, blockers, _, _ = _evaluate(_clean_metrics(json_failure_rate=0.5))
    assert decision == "NO-GO"
    assert any(b["rule_id"] == "json_failure_rate_too_high" for b in blockers)


# ---------------------------------------------------------------------------
# authenticity verdict from baseline comparison folded into the gate (P3)
# ---------------------------------------------------------------------------
def test_evaluate_policy_wrapper_verdict_is_nogo():
    decision, blockers, _, _ = _evaluate(_clean_metrics(authenticity_verdict="suspected_wrapper"))
    assert decision == "NO-GO"
    assert any(b["rule_id"] == "authenticity_verdict_blocks" for b in blockers)


def test_evaluate_policy_downgrade_verdict_is_nogo():
    decision, blockers, _, _ = _evaluate(_clean_metrics(authenticity_verdict="suspected_downgrade"))
    assert decision == "NO-GO"
    assert any(b["rule_id"] == "authenticity_verdict_blocks" for b in blockers)


def test_evaluate_policy_insufficient_authenticity_is_review():
    decision, blockers, review, _ = _evaluate(_clean_metrics(authenticity_verdict="insufficient_evidence"))
    assert decision == "REVIEW"
    assert blockers == []
    assert any(r["rule_id"] == "authenticity_insufficient_requires_review" for r in review)


def test_evaluate_policy_matches_official_passes():
    decision, blockers, _, passed = _evaluate(_clean_metrics(authenticity_verdict="matches_official"))
    assert decision == "GO"
    assert "authenticity_ok" in passed


def test_evaluate_policy_no_authenticity_verdict_is_noop():
    # absent verdict must not change the clean GO (campaigns without baseline comparison)
    decision, blockers, _, passed = _evaluate(_clean_metrics())
    assert decision == "GO"
    assert not any(b["rule_id"] == "authenticity_verdict_blocks" for b in blockers)
    assert "authenticity_ok" not in passed


