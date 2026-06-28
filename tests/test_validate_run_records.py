"""Tests for validate_run_records.py pure functions.

This module validates historical results.json and run_record JSONL against the
v1 schema. It was only exercised via its --self-test subprocess (0% under
pytest). These import it directly and drive the pure conversion + validation
functions, including its own self-test fixtures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import validate_run_records as V  # noqa: E402


# ---------------------------------------------------------------------------
# self_test_records — the module's own fixtures must all validate clean
# ---------------------------------------------------------------------------
def test_self_test_records_are_schema_valid():
    records = V.self_test_records()
    assert len(records) >= 1
    errors = V.validate_records(records)
    assert errors == [], errors


# ---------------------------------------------------------------------------
# load_response_text
# ---------------------------------------------------------------------------
def test_load_response_text_empty_and_missing(tmp_path):
    assert V.load_response_text(None) == ""
    assert V.load_response_text("") == ""
    assert V.load_response_text(str(tmp_path / "nope.txt")) == ""


def test_load_response_text_reads_file(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text("body text", encoding="utf-8")
    assert V.load_response_text(str(p)) == "body text"


# ---------------------------------------------------------------------------
# benchmark_context
# ---------------------------------------------------------------------------
def test_benchmark_context_no_scores_file(tmp_path):
    results = tmp_path / "results.json"
    mode, formula = V.benchmark_context(results)
    assert mode == "historical"
    assert formula == V.SCORE_FORMULA_VERSION


def test_benchmark_context_reads_scores(tmp_path):
    (tmp_path / "benchmark_scores.json").write_text(
        json.dumps({"benchmark_mode": "smoke_10", "formula_version": "vX"}), encoding="utf-8")
    mode, formula = V.benchmark_context(tmp_path / "results.json")
    assert mode == "smoke_10"
    assert formula == "vX"


def test_benchmark_context_bad_scores_falls_back(tmp_path):
    (tmp_path / "benchmark_scores.json").write_text("not json", encoding="utf-8")
    mode, formula = V.benchmark_context(tmp_path / "results.json")
    assert mode == "historical"


# ---------------------------------------------------------------------------
# records_from_results — convert legacy results.json into v1 records
# ---------------------------------------------------------------------------
def test_records_from_results_builds_valid_records(tmp_path):
    results = [{
        "run_id": "run1",
        "timestamp": "2026-06-28T00:00:00Z",
        "task": {"id": "t1", "category": "C", "prompt": "p", "scoring_type": "json_exact",
                 "recommended_max_tokens": 256},
        "provider": {"id": "tested", "model": "claude-opus-4-6",
                     "base_url": "https://gw.x", "auth_env": "K", "provider_channel": "gateway"},
        "metrics": {"ok": True, "server_model": "claude-opus-4-6", "content_chars": 10},
        "score": {"score": 9.0},
    }]
    rp = tmp_path / "results.json"
    rp.write_text(json.dumps(results), encoding="utf-8")
    records = V.records_from_results(rp)
    assert len(records) == 1
    assert V.validate_records(records) == []
    assert records[0]["run"]["runner"] == "historical"
    assert records[0]["run"]["status"] == "completed"


def test_records_from_results_failed_status(tmp_path):
    results = [{
        "run_id": "run1",
        "task": {"id": "t1", "category": "C", "prompt": "p", "scoring_type": "json_exact"},
        "provider": {"id": "tested", "model": "m", "base_url": "u", "auth_env": "K"},
        "metrics": {"ok": False, "error": "boom"},
    }]
    rp = tmp_path / "results.json"
    rp.write_text(json.dumps(results), encoding="utf-8")
    records = V.records_from_results(rp)
    assert records[0]["run"]["status"] == "failed"


def test_records_from_results_rejects_non_array(tmp_path):
    rp = tmp_path / "results.json"
    rp.write_text(json.dumps({"not": "an array"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an array"):
        V.records_from_results(rp)


def test_records_from_results_bad_max_tokens_coerced(tmp_path):
    results = [{
        "run_id": "r",
        "task": {"id": "t1", "category": "C", "prompt": "p", "scoring_type": "json_exact",
                 "recommended_max_tokens": "not-a-number"},
        "provider": {"id": "tested", "model": "m", "base_url": "u", "auth_env": "K"},
        "metrics": {"ok": True, "server_model": "m", "content_chars": 1},
    }]
    rp = tmp_path / "results.json"
    rp.write_text(json.dumps(results), encoding="utf-8")
    records = V.records_from_results(rp)  # must not raise
    assert records[0]["request"]["max_tokens"] == 0


# ---------------------------------------------------------------------------
# records_from_jsonl
# ---------------------------------------------------------------------------
def test_records_from_jsonl_reads(tmp_path):
    p = tmp_path / "rec.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert V.records_from_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_records_from_jsonl_rejects_bad_json(tmp_path):
    p = tmp_path / "rec.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL"):
        V.records_from_jsonl(p)


def test_records_from_jsonl_rejects_non_object(tmp_path):
    p = tmp_path / "rec.jsonl"
    p.write_text("[1,2,3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        V.records_from_jsonl(p)


# ---------------------------------------------------------------------------
# validate_records — prefixes errors with record index
# ---------------------------------------------------------------------------
def test_validate_records_flags_bad_record():
    errors = V.validate_records([{"schema_version": "wrong"}])
    assert errors
    assert all(e.startswith("record 1:") for e in errors)


def test_validate_records_empty_input():
    assert V.validate_records([]) == []
