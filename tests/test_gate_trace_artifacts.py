"""list/read coverage for quality_gate + trace_evaluation artifacts (R21).

run_quality_gate / run_trace_evaluation are exercised by their module self-tests,
but the list_*/read_* accessors over the produced artifacts were not. Reuse the
modules' own fixture builders to produce a gate / trace evaluation, then drive
the accessors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import quality_gate as Q  # noqa: E402
import trace_evaluation as T  # noqa: E402


# ---------------------------------------------------------------------------
# quality_gate: run + list + read
# ---------------------------------------------------------------------------
def _policy_file(tmp_path: Path) -> Path:
    p = tmp_path / "quality_gate.policy.json"
    p.write_text(json.dumps({
        "policy_version": Q.QUALITY_GATE_POLICY_VERSION,
        "policies": [{"policy_id": Q.DEFAULT_POLICY_ID, "thresholds": {}}],
    }), encoding="utf-8")
    return p


def test_quality_gate_run_list_read(tmp_path):
    Q.make_fake_run(tmp_path, run_id="run_go")
    policy = _policy_file(tmp_path)
    result = Q.run_quality_gate(runs_dir=tmp_path, run_id="run_go", policy_path=policy,
                                compatibility_run_id="compat_go")
    assert result["records"]
    gate_id = result["records"][0].get("gate_id") or result.get("gate_id")

    run_dir = tmp_path / "run_go"
    gates = Q.list_quality_gates(run_dir)
    assert isinstance(gates, list) and len(gates) >= 1
    listed_id = gates[0].get("gate_id")
    assert listed_id

    detail = Q.read_quality_gate(run_dir, listed_id)
    assert isinstance(detail, dict)
    assert detail.get("gate_id") == listed_id or "records" in detail or "manifest" in detail


def test_quality_gate_list_empty(tmp_path):
    (tmp_path / "run_x").mkdir()
    assert Q.list_quality_gates(tmp_path / "run_x") == []


# ---------------------------------------------------------------------------
# P3b: a persisted authenticity verdict reaches the gate end-to-end
# ---------------------------------------------------------------------------
def test_quality_gate_consumes_authenticity_wrapper_verdict(tmp_path):
    Q.make_fake_run(tmp_path, run_id="run_wrap", provider_id="fake_provider")
    # persist a compare_to_baseline-shaped verdict where verify-endpoint would
    (tmp_path / "run_wrap" / "authenticity").mkdir()
    (tmp_path / "run_wrap" / "authenticity" / "fake_provider.json").write_text(
        json.dumps({"verdict": "suspected_wrapper", "confidence": 0.8}), encoding="utf-8")
    policy = _policy_file(tmp_path)
    result = Q.run_quality_gate(runs_dir=tmp_path, run_id="run_wrap", policy_path=policy,
                                compatibility_run_id="compat_go")
    rec = result["records"][0]
    assert rec["decision"] == "NO-GO"
    assert any(b["rule_id"] == "authenticity_verdict_blocks" for b in rec["blockers"])
    # the verdict actually rode through aggregate_metrics into the snapshot
    assert rec["metrics_snapshot"]["authenticity_verdict"] == "suspected_wrapper"


def test_quality_gate_authenticity_absent_is_noop(tmp_path):
    Q.make_fake_run(tmp_path, run_id="run_plain", provider_id="fake_provider")
    policy = _policy_file(tmp_path)
    result = Q.run_quality_gate(runs_dir=tmp_path, run_id="run_plain", policy_path=policy,
                                compatibility_run_id="compat_go")
    rec = result["records"][0]
    # no authenticity artifact -> no authenticity blocker, verdict metric is None
    assert rec["metrics_snapshot"]["authenticity_verdict"] is None
    assert not any(b["rule_id"] == "authenticity_verdict_blocks" for b in rec["blockers"])


def test_load_authenticity_evidence_absent(tmp_path):
    (tmp_path / "run_x").mkdir()
    out = Q.load_authenticity_evidence(tmp_path / "run_x", "p")
    assert out["found"] is False
    assert out["verdict"] is None


def test_load_authenticity_evidence_verdict_json_fallback(tmp_path):
    run_dir = tmp_path / "run_x"
    (run_dir / "authenticity").mkdir(parents=True)
    (run_dir / "authenticity" / "verdict.json").write_text(
        json.dumps({"verdict": "matches_official"}), encoding="utf-8")
    out = Q.load_authenticity_evidence(run_dir, provider_id=None)
    assert out["found"] is True
    assert out["verdict"] == "matches_official"


def test_load_authenticity_evidence_corrupt_is_insufficient(tmp_path):
    # a PRESENT-but-unreadable artifact must surface as insufficient_evidence
    # (-> REVIEW), not silently read as absent.
    run_dir = tmp_path / "run_x"
    (run_dir / "authenticity").mkdir(parents=True)
    (run_dir / "authenticity" / "p.json").write_text("{not valid json", encoding="utf-8")
    out = Q.load_authenticity_evidence(run_dir, provider_id="p")
    assert out["found"] is True
    assert out["verdict"] == "insufficient_evidence"
    assert "unreadable" in out.get("error", "")


def test_load_authenticity_evidence_no_verdict_key_is_insufficient(tmp_path):
    run_dir = tmp_path / "run_x"
    (run_dir / "authenticity").mkdir(parents=True)
    (run_dir / "authenticity" / "p.json").write_text(json.dumps({"note": "no verdict"}), encoding="utf-8")
    out = Q.load_authenticity_evidence(run_dir, provider_id="p")
    assert out["verdict"] == "insufficient_evidence"


def test_corrupt_authenticity_artifact_routes_gate_to_review(tmp_path):
    # end-to-end: a corrupt artifact must make the gate REVIEW, never silent GO
    Q.make_fake_run(tmp_path, run_id="run_corrupt", provider_id="fake_provider")
    (tmp_path / "run_corrupt" / "authenticity").mkdir()
    (tmp_path / "run_corrupt" / "authenticity" / "fake_provider.json").write_text(
        "{truncated", encoding="utf-8")
    policy = _policy_file(tmp_path)
    result = Q.run_quality_gate(runs_dir=tmp_path, run_id="run_corrupt", policy_path=policy,
                                compatibility_run_id="compat_go")
    rec = result["records"][0]
    assert rec["decision"] in ("REVIEW", "NO-GO")  # never GO on corrupt evidence
    assert any(r["rule_id"] == "authenticity_insufficient_requires_review" for r in rec["review_items"])


# ---------------------------------------------------------------------------
# trace_evaluation: run + list + read
# ---------------------------------------------------------------------------
def _trace_run(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "trace_fix"
    events_dir = run_dir / "events" / "provider_a"
    good = [
        {"type": "message_start"},
        {"type": "content_block_start", "content_block": {"type": "text"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    T.write_fixture_events(events_dir / "good.jsonl", good)
    records = [T.fixture_record("trace_fix", "good", "provider_a", "events/provider_a/good.jsonl")]
    T.write_jsonl(run_dir / "run_records.jsonl", records)
    policy = tmp_path / "trace_evaluation.policy.json"
    T.write_json(policy, {"policy_version": T.TRACE_EVAL_POLICY_VERSION,
                          "policies": [{"policy_id": T.DEFAULT_POLICY_ID, "thresholds": {}}]})
    return runs_dir, policy


def test_trace_run_list_read(tmp_path):
    runs_dir, policy = _trace_run(tmp_path)
    result = T.run_trace_evaluation(runs_dir=runs_dir, run_id="trace_fix", policy_path=policy)
    assert result["records"]
    trace_eval_id = result.get("trace_eval_id") or result["manifest"].get("trace_eval_id")

    run_dir = runs_dir / "trace_fix"
    listed = T.list_trace_evaluations(run_dir)
    assert isinstance(listed, list) and len(listed) >= 1
    listed_id = listed[0].get("trace_eval_id") or trace_eval_id
    detail = T.read_trace_evaluation(run_dir, listed_id)
    assert isinstance(detail, dict)


def test_trace_run_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        T.run_trace_evaluation(runs_dir=tmp_path, run_id="ghost",
                               policy_path=tmp_path / "p.json")


def test_trace_list_empty(tmp_path):
    (tmp_path / "run_x").mkdir()
    assert T.list_trace_evaluations(tmp_path / "run_x") == []
