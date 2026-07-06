"""Branch coverage for trace_evaluation helpers, evaluate_source, main (T1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trace_evaluation as T  # noqa: E402


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def test_csv_value():
    assert T.csv_value({"a": 1}) == '{"a":1}'
    assert T.csv_value([1, 2]) == "[1,2]"
    assert T.csv_value(None) == ""
    assert T.csv_value(5) == "5"


def test_numeric():
    assert T.numeric("3.5") == 3.5
    assert T.numeric(None, 1.0) == 1.0
    assert T.numeric("", 2.0) == 2.0
    assert T.numeric("bad", 9.0) == 9.0
    assert T.numeric("bad") is None


def test_boolish():
    assert T.boolish(True) is True
    assert T.boolish("ok") is True       # trace's extra truthy token
    assert T.boolish("yes") is True
    assert T.boolish("false") is False
    assert T.boolish(None) is None
    assert T.boolish("maybe") is None


def test_ratio():
    assert T.ratio(1, 2) == 0.5
    assert T.ratio(1, 0) is None


def test_read_csv_rows_missing(tmp_path):
    assert T.read_csv_rows(tmp_path / "nope.csv") == []


def test_read_csv_rows(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    rows = T.read_csv_rows(p)
    assert rows == [{"a": "1", "b": "2"}]


# ---------------------------------------------------------------------------
# load_policy
# ---------------------------------------------------------------------------
def test_load_policy_missing_file_defaults(tmp_path):
    pol = T.load_policy(tmp_path / "nope.json")
    assert pol["policy_id"] == T.DEFAULT_POLICY_ID
    assert pol["thresholds"]["first_content_token_ms_warn"] == 15000


def test_load_policy_top_level(tmp_path):
    p = tmp_path / "pol.json"
    p.write_text(json.dumps({"policy_id": T.DEFAULT_POLICY_ID, "thresholds": {}}), encoding="utf-8")
    pol = T.load_policy(p)
    assert pol["policy_id"] == T.DEFAULT_POLICY_ID
    # defaults backfilled
    assert pol["thresholds"]["thinking_delta_count_warn"] == 100


def test_load_policy_from_policies_list(tmp_path):
    p = tmp_path / "pol.json"
    p.write_text(json.dumps({"policies": [{"policy_id": "custom", "thresholds": {}}]}), encoding="utf-8")
    pol = T.load_policy(p, policy_id="custom")
    assert pol["policy_id"] == "custom"


def test_load_policy_not_found_raises(tmp_path):
    p = tmp_path / "pol.json"
    p.write_text(json.dumps({"policies": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="trace policy not found"):
        T.load_policy(p, policy_id="ghost")


# placeholder-t1


# ---------------------------------------------------------------------------
# evaluate_source branches
# ---------------------------------------------------------------------------
def _events(tmp_path, events, name="ev.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return p


def _clean_stream():
    return [
        {"type": "message_start"},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]


def _policy():
    return {"thresholds": {"first_content_token_ms_warn": 15000, "thinking_delta_count_warn": 100}}


def test_evaluate_source_missing_events_warns():
    src = {"provider_id": "p", "telemetry": {"ok": True}, "trace": {"tool_calls": []},
           "events_path": "/no/such/events.jsonl"}
    result = T.evaluate_source(src, _policy(), "te1")
    by = {c["name"]: c["status"] for c in result["checks"]}
    assert by["events_file_present"] == "WARN"
    assert result["evidence"].get("missing_events") is True


def test_evaluate_source_max_tokens_warn(tmp_path):
    ev = _events(tmp_path, _clean_stream())
    src = {"provider_id": "p", "telemetry": {"ok": True, "stop_reason": "max_tokens",
                                             "first_content_token_ms": 500},
           "trace": {"tool_calls": []}, "events_path": str(ev),
           "raw_event_types": ["message_start"]}
    result = T.evaluate_source(src, _policy(), "te1")
    by = {c["name"]: c["status"] for c in result["checks"]}
    assert by["max_tokens_stop"] == "WARN"


def test_evaluate_source_latency_warn(tmp_path):
    ev = _events(tmp_path, _clean_stream())
    src = {"provider_id": "p",
           "telemetry": {"ok": True, "stop_reason": "end_turn", "first_content_token_ms": 99999},
           "trace": {"tool_calls": []}, "events_path": str(ev), "raw_event_types": []}
    result = T.evaluate_source(src, _policy(), "te1")
    by = {c["name"]: c["status"] for c in result["checks"]}
    assert by["latency_trace"] == "WARN"


def test_evaluate_source_missing_latency_warn(tmp_path):
    ev = _events(tmp_path, _clean_stream())
    src = {"provider_id": "p", "telemetry": {"ok": True, "stop_reason": "end_turn"},
           "trace": {"tool_calls": []}, "events_path": str(ev), "raw_event_types": []}
    result = T.evaluate_source(src, _policy(), "te1")
    by = {c["name"]: c["status"] for c in result["checks"]}
    assert by["latency_trace"] == "WARN"


def test_evaluate_source_thinking_only_fails(tmp_path):
    thinking = [
        {"type": "message_start"},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    ev = _events(tmp_path, thinking)
    src = {"provider_id": "p", "telemetry": {"ok": True, "stop_reason": "end_turn",
                                             "first_content_token_ms": 100},
           "trace": {"tool_calls": []}, "events_path": str(ev), "raw_event_types": []}
    result = T.evaluate_source(src, _policy(), "te1")
    by = {c["name"]: c["status"] for c in result["checks"]}
    assert by["visible_text_path"] == "FAIL"
    assert result["evidence"].get("thinking_only") is True


def test_evaluate_source_invalid_json_fails(tmp_path):
    p = tmp_path / "ev.jsonl"
    p.write_text('{"type":"message_start"}\nnot json\n{"type":"message_stop"}\n'
                 '{"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n', encoding="utf-8")
    src = {"provider_id": "p", "telemetry": {"ok": True, "first_content_token_ms": 100},
           "trace": {"tool_calls": []}, "events_path": str(p), "raw_event_types": []}
    result = T.evaluate_source(src, _policy(), "te1")
    by = {c["name"]: c["status"] for c in result["checks"]}
    assert by.get("event_json_validity") == "FAIL"


# ---------------------------------------------------------------------------
# main() via argv
# ---------------------------------------------------------------------------
def test_main_self_test(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["trace_evaluation.py", "--self-test"])
    rc = T.main()
    assert rc == 0
    assert "self-test ok" in capsys.readouterr().out


def test_main_requires_run_id(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["trace_evaluation.py", "--runs-dir", "/tmp/x"])
    with pytest.raises(SystemExit):
        T.main()

