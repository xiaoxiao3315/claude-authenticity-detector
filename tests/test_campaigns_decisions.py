"""Unit tests for campaigns.py decision + outcome logic.

campaigns.py turns per-run metrics into the campaign-level GO/REVIEW/NO-GO
verdict, the PASS/RETEST/FAIL outcome, and the recommended next action. That
logic had only the module's coarse self-test; these drive each pure function
across its branches so a threshold or mapping regression is caught.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import campaigns as C  # noqa: E402


# ---------------------------------------------------------------------------
# model_name_matches — exact, dated-suffix, and reject cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("requested,returned,expected", [
    ("claude-opus-4-6", "claude-opus-4-6", True),
    ("claude-opus-4-6", "claude-opus-4-6-2025-01-01", True),   # dated snapshot suffix
    ("claude-opus-4-6", "claude-opus-4-6-latest", False),       # non-date suffix
    ("claude-opus-4-6", "gpt-4o", False),
    ("", "claude-opus-4-6", False),
    ("claude-opus-4-6", "", False),
])
def test_model_name_matches(requested, returned, expected):
    assert C.model_name_matches(requested, returned) is expected


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------
def test_percentile_empty_is_none():
    assert C.percentile([], 0.95) is None


def test_percentile_single_value():
    assert C.percentile([42.0], 0.95) == 42.0


def test_percentile_interpolates():
    # p50 of 0..10 (11 points) is the midpoint 5.0
    assert C.percentile([float(x) for x in range(11)], 0.5) == 5.0
    # p95 interpolates between the top two-ish points
    assert C.percentile([10.0, 20.0], 0.5) == 15.0


# ---------------------------------------------------------------------------
# safe_campaign_id — path-traversal guard
# ---------------------------------------------------------------------------
def test_safe_campaign_id_accepts_clean():
    assert C.safe_campaign_id("camp_2026") == "camp_2026"


@pytest.mark.parametrize("bad", ["", "  ", "../etc", "a/b", "a\\b", "..", "x/../y"])
def test_safe_campaign_id_rejects_traversal(bad):
    with pytest.raises(ValueError):
        C.safe_campaign_id(bad)


# ---------------------------------------------------------------------------
# error_type classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("", "unknown"),
    ("SSL handshake failed", "ssl"),
    ("HTTP 429 rate limit", "rate_limit"),
    ("quota exceeded", "rate_limit"),
    ("request timed out", "timeout"),
    ("ReadError on socket", "read_error"),
    ("connection refused", "connection"),
    ("response JSON parse failed", "json_or_parse"),
    ("something weird 500", "transport_or_model"),
])
def test_error_type(text, expected):
    assert C.error_type(text) == expected


def test_error_type_priority_ssl_before_rate():
    # ssl is checked first; a string with both still classifies as ssl
    assert C.error_type("SSL error with 429") == "ssl"


# ---------------------------------------------------------------------------
# decision_to_outcome
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decision,outcome", [
    ("GO", "PASS"),
    ("REVIEW", "RETEST"),
    ("NO-GO", "FAIL"),
    ("go", "PASS"),            # case-insensitive
    ("PASS", "PASS"),          # already an outcome
    ("FAIL", "FAIL"),
    (None, "PENDING"),
    ("bogus", "PENDING"),
])
def test_decision_to_outcome(decision, outcome):
    assert C.decision_to_outcome(decision) == outcome


# placeholder-campaigns


# ---------------------------------------------------------------------------
# model_confidence_decision — consistency / quality / protocol thresholds
# ---------------------------------------------------------------------------
def test_model_confidence_all_good_is_go():
    decision, reasons = C.model_confidence_decision({
        "model_name_consistency_rate": 1.0,
        "average_quality_score": 8.5,
        "protocol_compatibility_score": 1.0,
    })
    assert decision == "GO"
    assert reasons == []


def test_model_confidence_low_consistency_is_nogo():
    decision, reasons = C.model_confidence_decision({
        "model_name_consistency_rate": 0.5,
        "average_quality_score": 9.0,
        "protocol_compatibility_score": 1.0,
    })
    assert decision == "NO-GO"
    assert "model_name_consistency_below_0.98" in reasons


def test_model_confidence_low_quality_is_nogo():
    decision, reasons = C.model_confidence_decision({
        "model_name_consistency_rate": 1.0,
        "average_quality_score": 5.0,
        "protocol_compatibility_score": 1.0,
    })
    assert decision == "NO-GO"
    assert "average_quality_below_6.0" in reasons


def test_model_confidence_mid_quality_is_review():
    decision, reasons = C.model_confidence_decision({
        "model_name_consistency_rate": 1.0,
        "average_quality_score": 7.0,
        "protocol_compatibility_score": 1.0,
    })
    assert decision == "REVIEW"
    assert "average_quality_below_7.5" in reasons


def test_model_confidence_missing_metrics_is_review():
    decision, reasons = C.model_confidence_decision({})
    assert decision == "REVIEW"
    assert "model_returned_missing" in reasons
    assert "quality_score_missing" in reasons
    assert "protocol_compatibility_missing" in reasons


def test_model_confidence_partial_protocol_is_review():
    decision, reasons = C.model_confidence_decision({
        "model_name_consistency_rate": 1.0,
        "average_quality_score": 9.0,
        "protocol_compatibility_score": 0.8,
    })
    assert decision == "REVIEW"
    assert "protocol_compatibility_not_full" in reasons


# ---------------------------------------------------------------------------
# gateway_reliability_decision
# ---------------------------------------------------------------------------
def test_gateway_all_good_is_go():
    decision, reasons = C.gateway_reliability_decision({
        "transport_success_rate": 1.0,
        "p95_latency_ms": 1200,
    })
    assert decision == "GO"
    assert reasons == []


def test_gateway_low_transport_is_nogo():
    decision, reasons = C.gateway_reliability_decision({
        "transport_success_rate": 0.90,
        "p95_latency_ms": 1200,
    })
    assert decision == "NO-GO"
    assert "transport_success_below_0.95" in reasons


def test_gateway_mid_transport_is_review():
    decision, reasons = C.gateway_reliability_decision({
        "transport_success_rate": 0.96,
        "p95_latency_ms": 1200,
    })
    assert decision == "REVIEW"
    assert "transport_success_below_0.98" in reasons


def test_gateway_high_latency_is_review():
    decision, reasons = C.gateway_reliability_decision({
        "transport_success_rate": 1.0,
        "p95_latency_ms": 20000,
    })
    assert decision == "REVIEW"
    assert "p95_latency_above_15000ms" in reasons


# placeholder-campaigns2


# ---------------------------------------------------------------------------
# retest_action — the recommended next step per verdict shape
# ---------------------------------------------------------------------------
def test_retest_action_go_accepts():
    action, _ = C.retest_action(
        model_decision="GO", model_reasons=[],
        gateway_decision="GO", gateway_reasons=[], overall_decision="GO")
    assert action == "accept"


def test_retest_action_nogo_rejects():
    action, _ = C.retest_action(
        model_decision="NO-GO", model_reasons=["x"],
        gateway_decision="GO", gateway_reasons=[], overall_decision="NO-GO")
    assert action == "reject"


def test_retest_action_model_quality_review():
    action, _ = C.retest_action(
        model_decision="REVIEW", model_reasons=["average_quality_below_7.5"],
        gateway_decision="GO", gateway_reasons=[], overall_decision="REVIEW")
    assert action == "auto_retest_quality"


def test_retest_action_model_identity_review():
    action, _ = C.retest_action(
        model_decision="REVIEW", model_reasons=["protocol_compatibility_not_full"],
        gateway_decision="GO", gateway_reasons=[], overall_decision="REVIEW")
    assert action == "auto_retest_identity"


def test_retest_action_gateway_latency_review():
    action, _ = C.retest_action(
        model_decision="GO", model_reasons=[],
        gateway_decision="REVIEW", gateway_reasons=["p95_latency_above_15000ms"],
        overall_decision="REVIEW")
    assert action == "auto_retest_latency"


def test_retest_action_gateway_generic_review():
    action, _ = C.retest_action(
        model_decision="GO", model_reasons=[],
        gateway_decision="REVIEW", gateway_reasons=["transport_success_below_0.98"],
        overall_decision="REVIEW")
    assert action == "auto_retest_gateway"


# ---------------------------------------------------------------------------
# campaign_outcomes — folds decisions into outcomes + next_action
# ---------------------------------------------------------------------------
def test_campaign_outcomes_go():
    out = C.campaign_outcomes({}, {
        "model_confidence_decision": "GO",
        "gateway_reliability_decision": "GO",
        "overall_decision": "GO",
    })
    assert out["model_outcome"] == "PASS"
    assert out["gateway_outcome"] == "PASS"
    assert out["overall_outcome"] == "PASS"
    assert out["next_action"] == "accept"
    assert set(C.REQUIRED_SUMMARY_OUTCOME_KEYS).issubset(out)


def test_campaign_outcomes_derives_overall_when_missing():
    # No overall_decision given -> worst_decision(REVIEW, NO-GO) = NO-GO
    out = C.campaign_outcomes({}, {
        "model_confidence_decision": "REVIEW",
        "gateway_reliability_decision": "NO-GO",
    })
    assert out["overall_outcome"] == "FAIL"
    assert out["next_action"] == "reject"


def test_outcomes_from_summary_reuses_existing():
    existing = {k: "PASS" for k in C.REQUIRED_SUMMARY_OUTCOME_KEYS}
    summary = {"outcomes": existing}
    assert C.outcomes_from_summary(summary) is existing


def test_outcomes_from_summary_computes_when_incomplete():
    summary = {
        "outcomes": {"model_outcome": "PASS"},  # incomplete
        "decisions": {
            "model_confidence_decision": "GO",
            "gateway_reliability_decision": "GO",
            "overall_decision": "GO",
        },
    }
    out = C.outcomes_from_summary(summary)
    assert set(C.REQUIRED_SUMMARY_OUTCOME_KEYS).issubset(out)
    assert out["overall_outcome"] == "PASS"


# ---------------------------------------------------------------------------
# summary_needs_refresh
# ---------------------------------------------------------------------------
def test_summary_needs_refresh_on_none():
    assert C.summary_needs_refresh(None) is True


def test_summary_needs_refresh_on_missing_keys():
    assert C.summary_needs_refresh({"metrics": {}, "decisions": {}, "outcomes": {}}) is True


# ---------------------------------------------------------------------------
# summarize_campaign — the on-disk orchestrator, via a minimal fixture
# ---------------------------------------------------------------------------
def _record(task_id, *, ok=True, score=9.0, requested="claude-opus-4-6",
            returned="claude-opus-4-6", api_style="anthropic_messages",
            latency=800.0, error=None):
    return {
        "task": {"id": task_id, "category": "Reasoning"},
        "provider": {"model_requested": requested, "model_returned": returned,
                     "api_style": api_style},
        "telemetry": {"ok": ok, "first_content_token_ms": latency, "error": error,
                      "retry_count": 0},
        "response": {"content_chars": 120 if ok else 0},
        "scoring": {"final_score": {"score": score, "details": "ok"}},
    }


def _build_campaign(tmp_path, records, *, status="completed"):
    """Create a minimal campaign + one run on disk; return (campaign_dir, runs_dir)."""
    campaigns_dir = tmp_path / "campaigns"
    runs_dir = tmp_path / "runs"
    cdir = campaigns_dir / "camp_x"
    rdir = runs_dir / "run_1"
    cdir.mkdir(parents=True)
    rdir.mkdir(parents=True)

    (cdir / "campaign.json").write_text(json.dumps({
        "campaign_id": "camp_x",
        "status": status,
        "live_provider": True,
        "tested_model": {"provider_id": "tested", "protocol": "anthropic_messages"},
        "judge_model": {"provider_id": "judge"},
    }), encoding="utf-8")
    (cdir / "run_ids.json").write_text(json.dumps({
        "campaign_id": "camp_x",
        "runs": [{"run_id": "run_1", "round": 1, "status": "completed"}],
    }), encoding="utf-8")
    (rdir / "state.json").write_text(json.dumps({
        "status": "completed", "final_decision": "GO",
        "completed_at": "2026-06-28T00:00:00Z",
    }), encoding="utf-8")
    with (rdir / "run_records.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return cdir, runs_dir


def test_summarize_campaign_all_good_is_go(tmp_path):
    records = [_record(f"t{i}", ok=True, score=9.0) for i in range(5)]
    cdir, runs_dir = _build_campaign(tmp_path, records)
    summary = C.summarize_campaign(cdir, runs_dir, persist=False)

    assert summary["schema_version"] == C.SUMMARY_SCHEMA_VERSION
    assert summary["campaign_id"] == "camp_x"
    m = summary["metrics"]
    assert m["total_cases"] == 5
    assert m["transport_success_rate"] == 1.0
    assert m["model_name_consistency_rate"] == 1.0
    assert m["protocol_compatibility_score"] == 1.0
    assert m["average_quality_score"] == 9.0
    # all signals clean -> GO across the board
    assert summary["decisions"]["overall_decision"] == "GO"
    assert summary["overall_outcome"] == "PASS"
    assert summary["next_action"] == "accept"


def test_summarize_campaign_model_swap_is_nogo(tmp_path):
    # returned model never matches requested -> consistency 0 -> NO-GO
    records = [_record(f"t{i}", returned="gpt-4o") for i in range(5)]
    cdir, runs_dir = _build_campaign(tmp_path, records)
    summary = C.summarize_campaign(cdir, runs_dir, persist=False)
    assert summary["metrics"]["model_name_consistency_rate"] == 0.0
    assert summary["decisions"]["model_confidence_decision"] == "NO-GO"
    assert summary["overall_outcome"] == "FAIL"


def test_summarize_campaign_transport_failures_lower_verdict(tmp_path):
    # 2 of 5 requests fail transport -> success rate 0.6 -> NO-GO on gateway
    records = ([_record(f"ok{i}") for i in range(3)]
               + [_record(f"bad{i}", ok=False, score=None, error="HTTP 503: down")
                  for i in range(2)])
    cdir, runs_dir = _build_campaign(tmp_path, records)
    summary = C.summarize_campaign(cdir, runs_dir, persist=False)
    assert summary["metrics"]["transport_success_rate"] == 0.6
    assert summary["decisions"]["gateway_reliability_decision"] == "NO-GO"


def test_summarize_campaign_persists_summary_json(tmp_path):
    records = [_record("t1")]
    cdir, runs_dir = _build_campaign(tmp_path, records)
    C.summarize_campaign(cdir, runs_dir, persist=True)
    assert (cdir / "summary.json").exists()
    saved = json.loads((cdir / "summary.json").read_text(encoding="utf-8"))
    assert saved["campaign_id"] == "camp_x"


# ---------------------------------------------------------------------------
# leaderboard layer — entry shaping, listing, filtering
# ---------------------------------------------------------------------------
def test_summary_to_leaderboard_entry_shape(tmp_path):
    records = [_record(f"t{i}", score=9.0) for i in range(3)]
    cdir, runs_dir = _build_campaign(tmp_path, records)
    summary = C.summarize_campaign(cdir, runs_dir, persist=False)
    entry = C.summary_to_leaderboard_entry(summary)
    assert entry["campaign_id"] == "camp_x"
    assert entry["overall_decision"] == "GO"
    assert entry["overall_outcome"] == "PASS"
    assert entry["status"] == "completed"
    assert entry["score"] is not None


def test_list_campaign_summaries_refreshes_missing(tmp_path):
    records = [_record("t1")]
    cdir, runs_dir = _build_campaign(tmp_path, records)
    campaigns_dir = cdir.parent
    # no summary.json yet -> refresh_missing builds it
    summaries = C.list_campaign_summaries(campaigns_dir, runs_dir,
                                          refresh_missing=True, persist_refresh=False)
    assert len(summaries) == 1
    assert summaries[0]["campaign_id"] == "camp_x"


def test_campaign_leaderboard_includes_live_completed(tmp_path):
    records = [_record(f"t{i}", score=9.0) for i in range(3)]
    cdir, runs_dir = _build_campaign(tmp_path, records)
    campaigns_dir = cdir.parent
    board = C.campaign_leaderboard(campaigns_dir, runs_dir,
                                   live_provider=True, persist_refresh=False)
    assert isinstance(board, dict)
    entries = board.get("entries") if isinstance(board.get("entries"), list) else board.get("leaderboard")
    # the live, completed campaign should appear
    flat = json.dumps(board, ensure_ascii=False)
    assert "camp_x" in flat




