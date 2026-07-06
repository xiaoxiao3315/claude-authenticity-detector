"""Fine-grained unit tests for authenticity.py's verdict engine.

This is the product's core: the code that decides GO / REVIEW / NO-GO on
whether a gateway really serves official Claude. Before this file, that logic
had ZERO executed test coverage — the module's own ``_self_test`` only built a
single ``live_provider=False`` dry-run fixture, which short-circuits to REVIEW
at the top of every verdict function (e.g. authenticity.py:468), so the live
NO-GO / GO branches never ran. A regression that mislabels a NO-GO gateway as
GO (the exact "gpt-5.5 scored GO/0.85" self-destruct class) would not be caught.

These tests drive the verdict functions directly with crafted inputs and assert
the *resulting decision and reasons*, not merely "it returned one of three
strings". We feed an input that SHOULD be NO-GO and assert we actually get
NO-GO, and likewise for GO / REVIEW and each live-only reason flag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import authenticity as A  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------

def _clean_record(category: str = "json") -> dict:
    """A record carrying every presence signal the feature metrics look for,
    so feature-derived rates (usage/stop/headers/request_id/...) are all 1.0
    and don't introduce noise reasons we aren't testing."""
    return {
        "task": {"id": f"task_{category}", "category": category},
        "provider": {
            "api_style": "anthropic",
            "model_requested": "claude-opus-4-8",
            "model_returned": "claude-opus-4-8",
            "upstream_request_id": "ups-1",
            "request_id": "req-1",
            "gateway_route_id": "route-1",
            "fallback_used": False,
            "cache_hit": False,
            "gateway_processing_ms": 5,
        },
        "request": {"request_hash": "h1"},
        "response": {
            "content_chars": 20,
            "events_file": "events.jsonl",
            "request_id": "req-1",
        },
        "telemetry": {
            "ok": True,
            "stop_reason": "end_turn",
            "retry_count": 0,
            "request_id": "req-1",
            "upstream_request_id": "ups-1",
            "gateway_route_id": "route-1",
            "gateway_processing_ms": 5,
        },
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "scoring": {"final_score": {"score": 9.0, "details": "ok"}},
        "trace": {
            "raw_event_types": ["message_start", "content_block_delta", "message_stop"],
            "tool_calls": [],
        },
        "events": [
            {"type": "message_start", "response_headers": {"x-request-id": "req-1"}, "request_id": "req-1"},
            {"type": "content_block_delta"},
            {"type": "message_stop"},
        ],
    }


def _live_records(n: int = 3) -> list[dict]:
    # include json + long_context categories so the live-only probe-count
    # reasons (json_probe_missing / long_context_probe_missing) don't fire.
    cats = ["json", "long_context"] + ["json"] * max(0, n - 2)
    return [_clean_record(c) for c in cats[:n]]


def _summary(*, live: bool, protocol_rate, consistency_rate, total_cases=3,
             returned_seen=3) -> dict:
    return {
        "campaign_id": "CMP-TEST",
        "live_provider": live,
        "tested_model": {"provider_id": "tested", "model": "claude-opus-4-8", "protocol": "anthropic"},
        "metrics": {
            "total_cases": total_cases,
            "protocol_compatibility_score": protocol_rate,
            "model_name_consistency_rate": consistency_rate,
            "model_returned_seen_count": returned_seen,
        },
    }


def _fingerprint(summary, records, tmp_path) -> dict:
    return A.build_protocol_fingerprint(
        campaign_dir_path=tmp_path,
        runs_dir=tmp_path,
        summary=summary,
        records=records,
        gateway_provider="tested",
        persist=False,
    )


# ---------------------------------------------------------------------------
# pure decision helpers — the duplicated verdict primitives
# ---------------------------------------------------------------------------

def test_worst_decision_picks_most_severe():
    assert A.worst_decision("GO", "REVIEW", "NO-GO") == "NO-GO"
    assert A.worst_decision("GO", "REVIEW") == "REVIEW"
    assert A.worst_decision("GO", "GO") == "GO"
    assert A.worst_decision() == "GO"
    assert A.worst_decision(None, "GO") == "GO"


def test_decision_from_score_thresholds():
    assert A.decision_from_score(None) == "REVIEW"
    assert A.decision_from_score(0.85) == "GO"      # boundary, inclusive
    assert A.decision_from_score(0.84) == "REVIEW"
    assert A.decision_from_score(0.60) == "REVIEW"  # boundary, inclusive
    assert A.decision_from_score(0.59) == "NO-GO"
    assert A.decision_from_score(0.0) == "NO-GO"


def test_decision_score_mapping():
    assert A.decision_score("GO") == 1.0
    assert A.decision_score("NO-GO") == 0.0
    assert A.decision_score("REVIEW") == 0.5
    assert A.decision_score(None) == 0.5


# ---------------------------------------------------------------------------
# build_protocol_fingerprint — the live-path verdict that had 0% coverage
# ---------------------------------------------------------------------------

def test_fingerprint_dry_run_short_circuits_to_review(tmp_path):
    fp = _fingerprint(_summary(live=False, protocol_rate=1.0, consistency_rate=1.0),
                      _live_records(), tmp_path)
    assert fp["decision"] == "REVIEW"
    assert "dry_run_protocol_evidence" in fp["reasons"]
    assert fp["evidence_status"] == "dry_run_reference_only"


def test_fingerprint_live_clean_is_go(tmp_path):
    fp = _fingerprint(_summary(live=True, protocol_rate=1.0, consistency_rate=1.0),
                      _live_records(), tmp_path)
    assert fp["decision"] == "GO", fp["reasons"]
    assert fp["evidence_status"] == "live_observed"


def test_fingerprint_live_low_protocol_is_nogo(tmp_path):
    # protocol_rate < 0.95 must force NO-GO regardless of score (line 470-471)
    fp = _fingerprint(_summary(live=True, protocol_rate=0.90, consistency_rate=1.0),
                      _live_records(), tmp_path)
    assert fp["decision"] == "NO-GO"
    assert "protocol_compatibility_below_0.98" in fp["reasons"]


def test_fingerprint_live_low_consistency_is_nogo(tmp_path):
    # consistency_rate < 0.98 must force NO-GO (line 472-473)
    fp = _fingerprint(_summary(live=True, protocol_rate=1.0, consistency_rate=0.90),
                      _live_records(), tmp_path)
    assert fp["decision"] == "NO-GO"
    assert "model_name_consistency_below_0.98" in fp["reasons"]


def test_fingerprint_protocol_between_095_and_098_flags_but_not_nogo(tmp_path):
    # 0.95 <= protocol < 0.98: adds the below_0.98 reason but does NOT hard-NO-GO;
    # final decision comes from the averaged score via decision_from_score.
    fp = _fingerprint(_summary(live=True, protocol_rate=0.96, consistency_rate=1.0),
                      _live_records(), tmp_path)
    assert "protocol_compatibility_below_0.98" in fp["reasons"]
    assert fp["decision"] != "NO-GO"


def test_fingerprint_missing_protocol_rate_flags_missing(tmp_path):
    fp = _fingerprint(_summary(live=True, protocol_rate=None, consistency_rate=1.0),
                      _live_records(), tmp_path)
    assert "protocol_compatibility_missing" in fp["reasons"]


def test_fingerprint_live_only_reasons_fire_only_when_live(tmp_path):
    # records with NO header/request_id/probe signals
    bare = [{"task": {"id": "t", "category": "smalltalk"},
             "provider": {}, "telemetry": {"stop_reason": "end_turn"},
             "usage": {"input_tokens": 1, "output_tokens": 1}}]
    live_fp = _fingerprint(_summary(live=True, protocol_rate=1.0, consistency_rate=1.0),
                           bare, tmp_path)
    for reason in ("response_headers_missing", "request_id_missing",
                   "json_probe_missing", "long_context_probe_missing"):
        assert reason in live_fp["reasons"], f"{reason} should fire on live bare records"
    # same bare records under dry-run: the live-only reasons must NOT fire
    dry_fp = _fingerprint(_summary(live=False, protocol_rate=1.0, consistency_rate=1.0),
                          bare, tmp_path)
    for reason in ("response_headers_missing", "request_id_missing",
                   "json_probe_missing", "long_context_probe_missing"):
        assert reason not in dry_fp["reasons"], f"{reason} must be live-only"


def test_fingerprint_persist_writes_file(tmp_path):
    summary = _summary(live=True, protocol_rate=1.0, consistency_rate=1.0)
    A.build_protocol_fingerprint(
        campaign_dir_path=tmp_path, runs_dir=tmp_path, summary=summary,
        records=_live_records(), gateway_provider="tested", persist=True,
    )
    assert (tmp_path / "protocol_fingerprints" / "tested.json").exists()


# ---------------------------------------------------------------------------
# build_baseline_comparison — missing-baseline + populated-baseline paths
# ---------------------------------------------------------------------------

def test_baseline_missing_is_review(tmp_path):
    cmp = A.build_baseline_comparison(
        campaign_dir_path=tmp_path, runs_dir=tmp_path,
        summary=_summary(live=True, protocol_rate=1.0, consistency_rate=1.0),
        baseline_campaign_dir=None, baseline_provider="baseline",
        gateway_provider="tested", persist=False,
    )
    assert cmp["decision"] == "REVIEW"
    assert cmp["baseline_source"] == "missing_official_baseline"
    assert "baseline_campaign_missing" in cmp["reasons"]


def test_baseline_missing_dry_run_marks_synthetic(tmp_path):
    cmp = A.build_baseline_comparison(
        campaign_dir_path=tmp_path, runs_dir=tmp_path,
        summary=_summary(live=False, protocol_rate=1.0, consistency_rate=1.0),
        baseline_campaign_dir=None, baseline_provider="baseline",
        gateway_provider="tested", persist=False,
    )
    assert cmp["baseline_source"] == "synthetic_dry_run_reference"
    assert "dry_run_reference_only" in cmp["reasons"]


def _write_baseline(tmp_path, metrics: dict, samples: list[dict]) -> Path:
    """Write a minimal baseline campaign dir whose summary.json load_summary reads."""
    bdir = tmp_path / "baseline_cmp"
    bdir.mkdir(parents=True, exist_ok=True)
    A.write_json(bdir / "summary.json", {
        "campaign_id": "CMP-BASELINE",
        "metrics": metrics,
        "samples": samples,
    })
    return bdir


def test_baseline_populated_close_metrics_scores_go(tmp_path):
    # Tested metrics nearly identical to baseline -> high similarity -> GO.
    summary = _summary(live=True, protocol_rate=1.0, consistency_rate=1.0)
    summary["metrics"].update({
        "average_quality_score": 8.0, "transport_success_rate": 1.0,
        "model_response_success_rate": 1.0, "p95_latency_ms": 1000,
    })
    summary["samples"] = [{"task_id": "t1", "score": 8.0}, {"task_id": "t2", "score": 9.0}]
    bdir = _write_baseline(tmp_path, {
        "average_quality_score": 8.0, "transport_success_rate": 1.0,
        "model_response_success_rate": 1.0, "p95_latency_ms": 1000,
    }, [{"task_id": "t1", "score": 8.0}, {"task_id": "t2", "score": 9.0}])

    cmp = A.build_baseline_comparison(
        campaign_dir_path=tmp_path, runs_dir=tmp_path, summary=summary,
        baseline_campaign_dir=bdir, baseline_provider="baseline",
        gateway_provider="tested", persist=False,
    )
    assert cmp["baseline_source"] == "campaign"
    assert cmp["metrics"]["overlapping_task_count"] == 2
    assert cmp["metrics"]["quality_delta"] == 0.0
    assert cmp["decision"] == "GO", cmp


def test_baseline_populated_divergent_metrics_degrades(tmp_path):
    # Large quality / transport gaps -> low similarity score -> not GO.
    summary = _summary(live=True, protocol_rate=1.0, consistency_rate=1.0)
    summary["metrics"].update({
        "average_quality_score": 2.0, "transport_success_rate": 0.5,
        "model_response_success_rate": 0.5, "p95_latency_ms": 9000,
    })
    summary["samples"] = [{"task_id": "t1", "score": 2.0}]
    bdir = _write_baseline(tmp_path, {
        "average_quality_score": 9.0, "transport_success_rate": 1.0,
        "model_response_success_rate": 1.0, "p95_latency_ms": 1000,
    }, [{"task_id": "t1", "score": 9.0}])

    cmp = A.build_baseline_comparison(
        campaign_dir_path=tmp_path, runs_dir=tmp_path, summary=summary,
        baseline_campaign_dir=bdir, baseline_provider="baseline",
        gateway_provider="tested", persist=False,
    )
    assert cmp["baseline_source"] == "campaign"
    assert cmp["decision"] in {"REVIEW", "NO-GO"}
    assert cmp["metrics"]["quality_delta"] == -7.0


def test_baseline_no_overlapping_tasks_adds_reason(tmp_path):
    # disjoint task ids -> no overlap -> worst_decision(.., REVIEW) + reason
    summary = _summary(live=True, protocol_rate=1.0, consistency_rate=1.0)
    summary["metrics"].update({
        "average_quality_score": 8.0, "transport_success_rate": 1.0,
        "model_response_success_rate": 1.0, "p95_latency_ms": 1000,
    })
    summary["samples"] = [{"task_id": "tested_only", "score": 8.0}]
    bdir = _write_baseline(tmp_path, {
        "average_quality_score": 8.0, "transport_success_rate": 1.0,
        "model_response_success_rate": 1.0, "p95_latency_ms": 1000,
    }, [{"task_id": "baseline_only", "score": 8.0}])

    cmp = A.build_baseline_comparison(
        campaign_dir_path=tmp_path, runs_dir=tmp_path, summary=summary,
        baseline_campaign_dir=bdir, baseline_provider="baseline",
        gateway_provider="tested", persist=False,
    )
    assert cmp["metrics"]["overlapping_task_count"] == 0
    assert "no_overlapping_task_scores" in cmp["reasons"]
    assert cmp["decision"] in {"REVIEW", "NO-GO"}


# ---------------------------------------------------------------------------
# end-to-end: write_authenticity_evidence trust_score clamp (line 783-786)
# ---------------------------------------------------------------------------

def _seed_campaign(root: Path, *, live: bool, protocol_rate, consistency_rate,
                   stop_reason="end_turn", model_returned="claude-opus-4-8") -> Path:
    """Build a campaign on disk that write_authenticity_evidence can summarize."""
    campaigns_dir = root / "campaigns"
    runs_dir = root / "runs"
    campaign_id = "CMP-E2E"
    camp_dir = campaigns_dir / campaign_id
    run_dir = runs_dir / f"{campaign_id}-R01"
    A.write_json(camp_dir / "campaign.json", {
        "schema_version": "campaign_v1", "campaign_id": campaign_id,
        "status": "completed", "created_at": "2026-01-01T00:00:00",
        "completed_at": "2026-01-01T00:01:00", "live_provider": live,
        "tested_model": {"provider_id": "tested", "model": "claude-opus-4-8", "protocol": "anthropic"},
        "judge_model": {"provider_id": "judge", "model": "dry-judge"},
        "benchmark_version": "self:test", "benchmark_mode": "self",
        "quality_gate_version": "self", "score_formula_version": "self",
        "metrics": {
            "total_cases": 1, "protocol_compatibility_score": protocol_rate,
            "model_name_consistency_rate": consistency_rate,
            "model_returned_seen_count": 1,
        },
    })
    A.write_json(camp_dir / "run_ids.json", {
        "campaign_id": campaign_id,
        "runs": [{"round": 1, "attempt": 1, "run_id": f"{campaign_id}-R01",
                  "status": "completed", "started_at": "2026-01-01T00:00:00",
                  "completed_at": "2026-01-01T00:01:00"}],
    })
    A.write_json(run_dir / "state.json", {
        "job_id": f"{campaign_id}-R01", "status": "completed",
        "started_at": "2026-01-01T00:00:00", "completed_at": "2026-01-01T00:01:00",
        "final_decision": "GO",
    })
    record = {
        "task": {"id": "task_1", "category": "json"},
        "provider": {"api_style": "anthropic", "model_requested": "claude-opus-4-8",
                     "model_returned": model_returned},
        "request": {"request_hash": "abc"},
        "response": {"content_chars": 12, "events_file": str(run_dir / "events.jsonl")},
        "telemetry": {"ok": True, "stop_reason": stop_reason, "total_ms": 20},
        "usage": {"input_tokens": 4, "output_tokens": 3},
        "scoring": {"final_score": {"score": 8.0, "details": "ok"}},
        "trace": {"raw_event_types": ["message_stop"]},
    }
    (run_dir / "run_records.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_records.jsonl").write_text(
        A.json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return camp_dir


def test_evidence_trust_score_clamped_when_not_go(tmp_path):
    # A dry-run campaign can never be GO (everything short-circuits to REVIEW),
    # so the overall trust decision must be REVIEW/NO-GO and the trust score
    # must be clamped accordingly (<=0.84 for REVIEW, <=0.59 for NO-GO).
    camp_dir = _seed_campaign(tmp_path, live=False, protocol_rate=1.0, consistency_rate=1.0)
    evidence = A.write_authenticity_evidence(camp_dir, tmp_path / "runs")
    decision = evidence["decisions"]["overall_trust_decision"]
    score = evidence["metrics"]["overall_trust_score"]
    assert decision in {"REVIEW", "NO-GO"}
    if score is not None:
        if decision == "NO-GO":
            assert score <= 0.59
        else:
            assert score <= 0.84


def test_evidence_live_low_protocol_drives_nogo_and_clamps(tmp_path):
    # A live campaign whose protocol compatibility is far below 0.95 must push
    # the protocol fingerprint to NO-GO, which (via worst_decision) makes the
    # OVERALL trust decision NO-GO and clamps trust_score <= 0.59 (line 783-784).
    # This is the exact "downgraded/swapped gateway mislabeled as GO" guardrail.
    camp_dir = _seed_campaign(tmp_path, live=True, protocol_rate=0.30,
                              consistency_rate=0.30, stop_reason="stop",
                              model_returned="gpt-4o")
    evidence = A.write_authenticity_evidence(camp_dir, tmp_path / "runs")
    proto = evidence["decisions"]["protocol_fingerprint_decision"]
    overall = evidence["decisions"]["overall_trust_decision"]
    score = evidence["metrics"]["overall_trust_score"]
    # protocol fingerprint itself should be NO-GO given protocol_rate 0.30
    assert proto == "NO-GO", evidence["decisions"]
    assert overall == "NO-GO"
    assert score is not None and score <= 0.59

