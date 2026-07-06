"""Tests for trace_evaluation.py — SSE event parsing + per-source grading.

trace_evaluation grades the recorded SSE event stream of each run (event order,
tool-call structure, thinking/text deltas, stop reason) into PASS/WARN/FAIL.
These drive the pure analytic functions: parse_events, validate_tool_calls,
provider_metrics, worst_status, plus evaluate_source via on-disk event fixtures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trace_evaluation as T  # noqa: E402


# ---------------------------------------------------------------------------
# worst_status
# ---------------------------------------------------------------------------
def test_worst_status():
    assert T.worst_status(["PASS", "WARN", "FAIL"]) == "FAIL"
    assert T.worst_status(["PASS", "WARN"]) == "WARN"
    assert T.worst_status(["PASS", "PASS"]) == "PASS"


# ---------------------------------------------------------------------------
# validate_tool_calls
# ---------------------------------------------------------------------------
def test_validate_tool_calls_none():
    status, details, ev = T.validate_tool_calls(None)
    assert status == T.NOT_APPLICABLE
    assert ev["tool_call_count"] == 0


def test_validate_tool_calls_valid():
    status, _, ev = T.validate_tool_calls([{"name": "search", "arguments": {"q": "x"}}])
    assert status == T.PASS
    assert ev["tool_call_count"] == 1


def test_validate_tool_calls_not_a_list():
    status, _, ev = T.validate_tool_calls("oops")
    assert status == T.FAIL
    assert ev["actual_type"] == "str"


def test_validate_tool_calls_missing_name():
    status, _, ev = T.validate_tool_calls([{"arguments": {}}])
    assert status == T.FAIL
    assert ev["invalid"][0]["error"] == "missing name/tool_name"


def test_validate_tool_calls_bad_arguments_type():
    status, _, ev = T.validate_tool_calls([{"name": "t", "arguments": 123}])
    assert status == T.FAIL


def test_validate_tool_calls_tool_name_alias():
    status, _, _ = T.validate_tool_calls([{"tool_name": "t", "tool_input": {"a": 1}}])
    assert status == T.PASS


# ---------------------------------------------------------------------------
# parse_events — reads an SSE events file
# ---------------------------------------------------------------------------
def _events_file(tmp_path, events):
    p = tmp_path / "events.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return p


def test_parse_events_counts_deltas(tmp_path):
    events = [
        {"type": "message_start"},
        {"type": "content_block_start", "content_block": {"type": "text"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    out = T.parse_events(_events_file(tmp_path, events))
    assert out["event_count"] == 6
    assert out["text_delta_count"] == 1
    assert out["thinking_delta_count"] == 1
    assert out["message_delta_stop_reason"] == "end_turn"
    assert out["message_delta_usage"] == {"output_tokens": 5}
    assert "message_start" in out["unique_event_types"]


def test_parse_events_counts_tool_use(tmp_path):
    events = [
        {"type": "content_block_start", "content_block": {"type": "tool_use"}},
        {"type": "content_block_delta", "delta": {"type": "input_json_delta"}},
    ]
    out = T.parse_events(_events_file(tmp_path, events))
    assert out["tool_event_count"] == 2


def test_parse_events_handles_invalid_json(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"type": "message_start"}\nnot json\n', encoding="utf-8")
    out = T.parse_events(p)
    assert out["invalid_json_count"] == 1
    assert "non_json_event" in out["event_types"]


# ---------------------------------------------------------------------------
# provider_metrics — aggregate per-provider grading
# ---------------------------------------------------------------------------
def test_provider_metrics_groups_and_rates():
    records = [
        {"provider_id": "a", "status": "PASS", "checks": [], "evidence": {}},
        {"provider_id": "a", "status": "FAIL", "checks": [], "evidence": {"missing_events": True}},
        {"provider_id": "b", "status": "WARN", "checks": [], "evidence": {}},
    ]
    out = T.provider_metrics(records)
    assert out["a"]["record_count"] == 2
    assert out["a"]["fail_count"] == 1
    assert out["a"]["trace_fail_rate"] == 0.5
    assert out["a"]["missing_events_count"] == 1
    assert out["a"]["status"] == "FAIL"
    assert out["b"]["status"] == "WARN"


def test_provider_metrics_all_pass():
    records = [{"provider_id": "a", "status": "PASS", "checks": [], "evidence": {}}]
    out = T.provider_metrics(records)
    assert out["a"]["status"] == "PASS"
    assert out["a"]["trace_fail_rate"] == 0.0


# ---------------------------------------------------------------------------
# evaluate_source — full grade over a fixture source (uses fixture_record path)
# ---------------------------------------------------------------------------
def test_evaluate_source_clean_passes(tmp_path):
    events = [
        {"type": "message_start"},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    ev_path = _events_file(tmp_path, events)
    source = {
        "provider_id": "tested",
        "telemetry": {"ok": True, "first_content_token_ms": 500, "stop_reason": "end_turn"},
        "trace": {"tool_calls": []},
        "events_path": str(ev_path),
        "raw_event_types": ["message_start", "content_block_delta", "message_delta", "message_stop"],
    }
    policy = {"thresholds": {}}
    result = T.evaluate_source(source, policy, "trace_eval_1")
    assert result["status"] in {"PASS", "WARN"}  # clean stream should not FAIL
    assert "checks" in result
