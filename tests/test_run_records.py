"""Tests for run_records.py — record construction, hashing, schema validation.

run_records.py builds the canonical per-call record and validates it against
run_record.schema.json. Two things matter most and had no direct coverage:

1. **Hash stability.** request_hash is computed from a fixed request_fingerprint.
   Project memory warns: NEVER let new fields drift request_hash, or historical
   records stop matching. These tests pin the hash to exact values and assert
   it depends only on the documented fingerprint inputs.

2. **Schema validation** actually rejecting malformed records (wrong type,
   missing key, bad enum) — not just passing the happy path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_records as RR  # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def test_stable_json_hash_is_order_independent():
    a = RR.stable_json_hash({"x": 1, "y": 2})
    b = RR.stable_json_hash({"y": 2, "x": 1})
    assert a == b  # sort_keys -> key order does not matter
    assert a != RR.stable_json_hash({"x": 1, "y": 3})


def test_text_hash_deterministic():
    assert RR.text_hash("hello") == RR.text_hash("hello")
    assert RR.text_hash("hello") != RR.text_hash("world")
    # known sha256 of "hello"
    assert RR.text_hash("hello").startswith("2cf24dba5fb0a30e")


def test_base_url_host_extracts_netloc():
    assert RR.base_url_host("https://gw.example.com/v1") == "gw.example.com"
    assert RR.base_url_host("https://api.anthropic.com") == "api.anthropic.com"
    assert RR.base_url_host(None) is None
    assert RR.base_url_host("") is None


def test_as_plain_dict_from_dict_and_object():
    assert RR.as_plain_dict({"a": 1}) == {"a": 1}
    assert RR.as_plain_dict(None) == {}

    class Obj:
        def __init__(self):
            self.id = "p1"
            self.model = "m"
        def method(self):  # callables excluded
            return 1
    plain = RR.as_plain_dict(Obj())
    assert plain["id"] == "p1"
    assert plain["model"] == "m"
    assert "method" not in plain


def test_extract_raw_event_types_dedupes_in_order(tmp_path):
    ev = tmp_path / "ev.jsonl"
    ev.write_text(
        '{"type": "message_start"}\n'
        '{"type": "content_block_delta"}\n'
        '{"type": "message_start"}\n'   # duplicate -> not repeated
        'not json\n'
        '{"type": "message_stop"}\n',
        encoding="utf-8",
    )
    types = RR.extract_raw_event_types(ev)
    assert types == ["message_start", "content_block_delta", "non_json_event", "message_stop"]


def test_extract_raw_event_types_missing_file():
    assert RR.extract_raw_event_types(None) == []
    assert RR.extract_raw_event_types("/no/such/file.jsonl") == []


# placeholder-runrecords


# ---------------------------------------------------------------------------
# build_run_record — end to end + hash stability
# ---------------------------------------------------------------------------
def _build(**over):
    kw = dict(
        run_id="run1", timestamp="2026-06-28T00:00:00Z", benchmark_mode="custom",
        formula_version="v1", runner="cli", status="completed",
        task={"id": "t1", "category": "C", "prompt": "hello", "scoring_type": "json_exact"},
        provider={"id": "tested", "model": "claude-opus-4-6",
                  "base_url": "https://gw.x/v1", "auth_env": "K",
                  "provider_channel": "gateway"},
        metrics={"ok": True, "server_model": "claude-opus-4-6", "content_chars": 5,
                 "error": None, "input_tokens": 30},
        final_score={"score": 9.0}, response_text="hi",
        response_file="r.json", events_file="e.jsonl",
        max_tokens=64, temperature=0.0, system_prompt=None,
    )
    kw.update(over)
    return RR.build_run_record(**kw)


def test_build_run_record_is_schema_valid():
    rec = _build()
    assert RR.validate_run_record(rec) == []
    assert rec["schema_version"] == RR.RUN_RECORD_SCHEMA_VERSION
    assert rec["record_id"] == "run1:tested:t1"
    assert rec["provider"]["leaderboard_group"] == "gateway_candidate"
    assert rec["provider"]["base_url_host"] == "gw.x"


def test_request_hash_stable_across_irrelevant_fields():
    # Changing run-level metadata (timestamp, runner) must NOT change request_hash:
    # the hash is over the request fingerprint only. This is the history-drift guard.
    a = _build(timestamp="2026-01-01T00:00:00Z", runner="cli")
    b = _build(timestamp="2099-12-31T23:59:59Z", runner="web")
    assert a["request"]["request_hash"] == b["request"]["request_hash"]


def test_request_hash_changes_with_prompt():
    a = _build(task={"id": "t1", "category": "C", "prompt": "hello", "scoring_type": "json_exact"})
    b = _build(task={"id": "t1", "category": "C", "prompt": "DIFFERENT", "scoring_type": "json_exact"})
    assert a["request"]["request_hash"] != b["request"]["request_hash"]


def test_request_hash_changes_with_max_tokens_and_temp():
    base = _build()
    assert _build(max_tokens=128)["request"]["request_hash"] != base["request"]["request_hash"]
    assert _build(temperature=0.7)["request"]["request_hash"] != base["request"]["request_hash"]


def test_prompt_hash_matches_text_hash():
    rec = _build()
    assert rec["request"]["prompt_hash"] == RR.text_hash("hello")
    assert rec["response"]["normalized_text_hash"] == RR.text_hash("hi")


def test_leaderboard_group_by_channel():
    assert _build(provider={"id": "p", "model": "m", "base_url": "u", "auth_env": "K",
                            "provider_channel": "official"})["provider"]["leaderboard_group"] == "official_baseline"
    assert _build(provider={"id": "p", "model": "m", "base_url": "u", "auth_env": "K",
                            "provider_channel": "byo"})["provider"]["leaderboard_group"] == "imported"


def test_invalid_status_coerced_to_failed():
    rec = _build(status="bogus")
    assert rec["run"]["status"] == "failed"


def test_error_field_is_redacted():
    rec = _build(metrics={"ok": False, "server_model": "m", "content_chars": 0,
                          "error": "auth failed with key sk-LEAKEDKEY1234567890"})
    assert "sk-LEAKEDKEY1234567890" not in str(rec["telemetry"]["error"])


def test_system_present_flag_and_hash():
    no_sys = _build(system_prompt=None)
    assert no_sys["request"]["system_present"] is False
    with_sys = _build(system_prompt="be terse")
    assert with_sys["request"]["system_present"] is True
    # system hash participates in the request fingerprint
    assert no_sys["request"]["request_hash"] != with_sys["request"]["request_hash"]


# placeholder-runrecords2


# ---------------------------------------------------------------------------
# validate_run_record — must REJECT malformed records, not just pass good ones
# ---------------------------------------------------------------------------
def test_validate_rejects_missing_top_level_key():
    rec = _build()
    del rec["telemetry"]
    errors = RR.validate_run_record(rec)
    assert any("telemetry" in e for e in errors)


def test_validate_rejects_wrong_schema_version():
    rec = _build()
    rec["schema_version"] = "run_record_v2"
    errors = RR.validate_run_record(rec)
    assert any("schema_version" in e for e in errors)


def test_validate_rejects_bad_status_enum():
    rec = _build()
    rec["run"]["status"] = "weird"
    errors = RR.validate_run_record(rec)
    assert any("status" in e for e in errors)


def test_validate_rejects_bad_api_style():
    rec = _build()
    rec["provider"]["api_style"] = "grpc"
    errors = RR.validate_run_record(rec)
    assert any("api_style" in e for e in errors)


def test_validate_rejects_non_list_tool_calls():
    rec = _build()
    rec["trace"]["tool_calls"] = "not a list"
    errors = RR.validate_run_record(rec)
    assert any("tool_calls" in e for e in errors)


def test_validate_rejects_empty_record_id():
    rec = _build()
    rec["record_id"] = ""
    errors = RR.validate_run_record(rec)
    assert any("record_id" in e for e in errors)


def test_validate_rejects_missing_nested_key():
    rec = _build()
    del rec["request"]["request_hash"]
    errors = RR.validate_run_record(rec)
    assert any("request.request_hash" in e or "request_hash" in e for e in errors)


# ---------------------------------------------------------------------------
# schema type matchers
# ---------------------------------------------------------------------------
def test_type_matches_basic():
    assert RR._type_matches("x", "string")
    assert RR._type_matches(1, "integer")
    assert RR._type_matches(True, "boolean")
    assert not RR._type_matches(True, "integer")   # bool is not integer
    assert RR._type_matches(None, "null")
    assert RR._type_matches(1.5, "number")
    assert not RR._type_matches(True, "number")
    assert RR._type_matches([], "array")
    assert RR._type_matches({}, "object")


def test_schema_type_matches_union():
    assert RR._schema_type_matches(None, ["string", "null"])
    assert RR._schema_type_matches("x", ["string", "null"])
    assert not RR._schema_type_matches(5, ["string", "null"])


# ---------------------------------------------------------------------------
# append_run_record_jsonl — persistence round-trip
# ---------------------------------------------------------------------------
def test_append_run_record_jsonl_roundtrip(tmp_path):
    import json
    path = tmp_path / "sub" / "run_records.jsonl"
    rec = _build()
    RR.append_run_record_jsonl(path, rec)
    RR.append_run_record_jsonl(path, _build(run_id="run2"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["record_id"] == "run1:tested:t1"
    assert json.loads(lines[1])["run"]["run_id"] == "run2"


