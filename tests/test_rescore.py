"""Tests for rescore.py pure helpers — source extraction, filtering, formatting.

rescore re-grades a prior run's stored responses (no new model calls). It reads
source records from either run_records.jsonl or results.json, filters by
provider/task, and reloads response text. These pure pieces were at 27%; the
big run_rescore orchestrator stays out of scope (it drives the judge model).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rescore as RS  # noqa: E402


# ---------------------------------------------------------------------------
# csv_value / score_value
# ---------------------------------------------------------------------------
def test_csv_value():
    assert RS.csv_value(["a", "b"]) == "a;b"
    assert RS.csv_value(None) == ""
    assert RS.csv_value(5) == "5"
    assert RS.csv_value("x") == "x"


def test_score_value():
    assert RS.score_value({"score": 9.0}) == 9.0
    assert RS.score_value({"no_score": 1}) is None
    assert RS.score_value("not a dict") is None
    assert RS.score_value(None) is None


# ---------------------------------------------------------------------------
# load_response_text
# ---------------------------------------------------------------------------
def test_load_response_text_missing_path():
    text, err = RS.load_response_text(None, Path("."))
    assert text == ""
    assert err == "missing response_file"


def test_load_response_text_not_found(tmp_path):
    text, err = RS.load_response_text("nope.txt", tmp_path)
    assert text == ""
    assert "not found" in err


def test_load_response_text_relative_to_run_dir(tmp_path):
    (tmp_path / "resp.txt").write_text("hello body", encoding="utf-8")
    text, err = RS.load_response_text("resp.txt", tmp_path)
    assert text == "hello body"
    assert err is None


def test_load_response_text_absolute(tmp_path):
    p = tmp_path / "abs.txt"
    p.write_text("abs body", encoding="utf-8")
    text, err = RS.load_response_text(str(p), tmp_path)
    assert text == "abs body"
    assert err is None


# ---------------------------------------------------------------------------
# read_jsonl validation
# ---------------------------------------------------------------------------
def test_read_jsonl_rejects_non_object(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"ok": 1}\n[1,2,3]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        RS.read_jsonl(p)


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "ok.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    rows = RS.read_jsonl(p)
    assert rows == [{"a": 1}, {"b": 2}]


# placeholder-rescore


# ---------------------------------------------------------------------------
# source_records_from_run_records / from_results / load_source_records
# ---------------------------------------------------------------------------
def _write_run_records(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "record_id": "run1:tested:t1",
        "run": {"run_id": "run1"},
        "task": {"id": "t1", "category": "C"},
        "provider": {"id": "tested"},
        "scoring": {"final_score": {"score": 8.0}},
        "artifacts": {"response_file": "r.txt", "events_file": "e.jsonl"},
        "telemetry": {"error": None},
    }
    (run_dir / "run_records.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")


def test_source_records_from_run_records(tmp_path):
    _write_run_records(tmp_path)
    out = RS.source_records_from_run_records(tmp_path)
    assert len(out) == 1
    r = out[0]
    assert r["source_kind"] == "run_record"
    assert r["task_id"] == "t1"
    assert r["provider_id"] == "tested"
    assert r["original_score"] == {"score": 8.0}
    assert r["source_response_file"] == "r.txt"


def test_source_records_from_results(tmp_path):
    results = [{
        "run_id": "run1",
        "task": {"id": "t2"},
        "provider": {"id": "p2"},
        "metrics": {"error": "boom"},
        "response_file": "x.txt",
        "score": {"score": 3.0},
    }]
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    out = RS.source_records_from_results(tmp_path)
    assert len(out) == 1
    r = out[0]
    assert r["source_kind"] == "results_json"
    assert r["task_id"] == "t2"
    assert r["source_record_id"] == "run1:p2:t2"
    assert r["source_error"] == "boom"


def test_load_source_records_prefers_run_records(tmp_path):
    _write_run_records(tmp_path)
    (tmp_path / "results.json").write_text("[]", encoding="utf-8")
    out = RS.load_source_records(tmp_path)
    assert out[0]["source_kind"] == "run_record"  # run_records.jsonl wins


def test_load_source_records_falls_back_to_results(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps([{"run_id": "r", "task": {"id": "t"}, "provider": {"id": "p"}}]),
        encoding="utf-8")
    out = RS.load_source_records(tmp_path)
    assert out[0]["source_kind"] == "results_json"


def test_load_source_records_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RS.load_source_records(tmp_path)


# ---------------------------------------------------------------------------
# filter_source_records / source_count / task_lookup
# ---------------------------------------------------------------------------
def _records():
    return [
        {"provider_id": "a", "task_id": "t1"},
        {"provider_id": "a", "task_id": "t2"},
        {"provider_id": "b", "task_id": "t1"},
    ]


def test_filter_by_provider():
    out = RS.filter_source_records(_records(), "a", None)
    assert len(out) == 2
    assert all(r["provider_id"] == "a" for r in out)


def test_filter_by_task_ids():
    out = RS.filter_source_records(_records(), None, ["t1"])
    assert len(out) == 2
    assert all(r["task_id"] == "t1" for r in out)


def test_filter_by_both():
    out = RS.filter_source_records(_records(), "a", ["t1"])
    assert out == [{"provider_id": "a", "task_id": "t1"}]


def test_filter_no_constraints_returns_all():
    assert len(RS.filter_source_records(_records(), None, None)) == 3


def test_source_count(tmp_path):
    _write_run_records(tmp_path)
    assert RS.source_count(tmp_path) == 1
    assert RS.source_count(tmp_path, provider_id="other") == 0


def test_task_lookup():
    bank = [{"id": "t1", "x": 1}, {"id": "t2", "x": 2}, {"no_id": True}]
    lut = RS.task_lookup(bank)
    assert lut["t1"] == {"id": "t1", "x": 1}
    assert "t2" in lut
    assert len(lut) == 2  # entry without id dropped


# ---------------------------------------------------------------------------
# run_rescore — the orchestrator, driven fully offline with injected fakes
# ---------------------------------------------------------------------------
def _run_with_response(tmp_path: Path, response_text="the answer"):
    run_dir = tmp_path / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "resp.txt").write_text(response_text, encoding="utf-8")
    rec = {
        "record_id": "run1:tested:t1",
        "run": {"run_id": "run1"},
        "task": {"id": "t1", "category": "C", "scoring_type": "keyword_check"},
        "provider": {"id": "tested"},
        "scoring": {"final_score": {"score": 5.0}},
        "artifacts": {"response_file": "resp.txt", "events_file": "e.jsonl"},
        "telemetry": {"error": None},
    }
    (run_dir / "run_records.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    return run_dir.parent, "run1"


def test_run_rescore_rule_only(tmp_path):
    runs_dir, run_id = _run_with_response(tmp_path, "contains the keyword apple")

    def fake_score(task, response_text):
        # deterministic rule scorer: full marks if 'apple' present
        return {"score": 10.0 if "apple" in response_text else 0.0, "format_ok": True,
                "details": "rule"}

    result = RS.run_rescore(
        runs_dir=runs_dir, run_id=run_id,
        task_bank=[{"id": "t1", "scoring_type": "keyword_check"}],
        score_response=fake_score,
    )
    assert result["rescore_id"]
    assert result["record_count"] == 1
    # the written records file carries the per-record new scores
    rescore_dir = runs_dir / run_id / "rescores" / result["rescore_id"]
    assert rescore_dir.exists()
    records = RS.read_jsonl(rescore_dir / "rescore_records.jsonl")
    assert records[0]["new_final_score"]["score"] == 10.0


def test_run_rescore_with_judge(tmp_path):
    runs_dir, run_id = _run_with_response(tmp_path)

    def fake_score(task, response_text):
        return {"score": 6.0, "format_ok": True, "details": "rule"}

    def fake_judge(task, response_text, rule_score):
        return {"score": 9.0, "format_ok": True, "decision": "GO", "reason": "good"}

    result = RS.run_rescore(
        runs_dir=runs_dir, run_id=run_id,
        task_bank=[{"id": "t1", "scoring_type": "keyword_check"}],
        score_response=fake_score, judge_response=fake_judge,
    )
    rescore_dir = runs_dir / run_id / "rescores" / result["rescore_id"]
    records = RS.read_jsonl(rescore_dir / "rescore_records.jsonl")
    assert records[0]["new_judge_score"]["score"] == 9.0


def test_run_rescore_missing_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="run not found"):
        RS.run_rescore(runs_dir=tmp_path, run_id="ghost",
                       task_bank=[], score_response=lambda t, r: {})


def test_run_rescore_handles_scorer_exception(tmp_path):
    runs_dir, run_id = _run_with_response(tmp_path)

    def boom_score(task, response_text):
        raise RuntimeError("scorer crashed")

    result = RS.run_rescore(
        runs_dir=runs_dir, run_id=run_id,
        task_bank=[{"id": "t1", "scoring_type": "keyword_check"}],
        score_response=boom_score,
    )
    # the crash is captured per-record as a failed status, not propagated
    assert result["record_count"] == 1
    assert result["manifest"]["failure_count"] == 1
    rescore_dir = runs_dir / run_id / "rescores" / result["rescore_id"]
    records = RS.read_jsonl(rescore_dir / "rescore_records.jsonl")
    assert records[0]["rescore_error"] is not None
    assert "scorer crashed" in records[0]["rescore_error"]


