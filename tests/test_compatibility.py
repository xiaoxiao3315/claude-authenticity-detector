"""Tests for compatibility.py — the per-case protocol-conformance evaluators.

compatibility.py runs Anthropic-protocol probes (SSE event order, usage fields,
JSON-no-markdown, cache_control, tool_call, model identity) and grades each as
PASS/WARN/FAIL. The evaluators are pure functions over a ProbeMetrics + case;
they were exercised only end-to-end. These pin each grader's branches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import compatibility as K  # noqa: E402
from run_eval import Provider  # noqa: E402


def _provider(model="claude-opus-4-6"):
    return Provider(id="tested", base_url="https://gw.x", model=model,
                    auth_type="x-api-key", auth_env="K")


def test_read_jsonl_tolerates_corrupt_line(tmp_path):
    # a truncated/corrupt events line must be skipped, not crash the probe (P7)
    p = tmp_path / "events.jsonl"
    p.write_text('{"a": 1}\n{truncated...\n{"b": 2}\n', encoding="utf-8")
    rows = K.read_jsonl(p)
    assert rows == [{"a": 1}, {"b": 2}]


def test_read_jsonl_missing_file(tmp_path):
    assert K.read_jsonl(tmp_path / "nope.jsonl") == []


def test_read_jsonl_skips_non_dict(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"a": 1}\n[1,2,3]\n"str"\n', encoding="utf-8")
    assert K.read_jsonl(p) == [{"a": 1}]


def _metrics(**over):
    m = K.ProbeMetrics()
    for k, v in over.items():
        setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------
# worst_status / index_of / type_name / check
# ---------------------------------------------------------------------------
def test_worst_status_picks_most_severe():
    assert K.worst_status(["PASS", "WARN", "FAIL"]) == "FAIL"
    assert K.worst_status(["PASS", "WARN"]) == "WARN"
    assert K.worst_status(["PASS", "PASS"]) == "PASS"
    assert K.worst_status([]) == "PASS"


def test_index_of():
    assert K.index_of(["a", "b", "c"], "b") == 1
    assert K.index_of(["a", "b"], "z") is None


def test_type_name():
    assert K.type_name(True) == "bool"
    assert K.type_name(1) == "int"
    assert K.type_name(1.5) == "float"
    assert K.type_name("x") == "str"
    assert K.type_name([]) == "list"
    assert K.type_name({}) == "dict"
    assert K.type_name(None) == "null"


def test_check_shape():
    c = K.check("n", K.PASS, "ok", {"e": 1})
    assert c == {"name": "n", "status": "PASS", "details": "ok", "evidence": {"e": 1}}
    assert K.check("n", K.FAIL, "bad")["evidence"] == {}


# ---------------------------------------------------------------------------
# evaluate_common
# ---------------------------------------------------------------------------
def test_evaluate_common_all_pass():
    m = _metrics(http_status=200, content_chars=50, server_model="claude-opus-4-6")
    checks = K.evaluate_common(_provider(), m)
    by = {c["name"]: c["status"] for c in checks}
    assert by["http_status"] == "PASS"
    assert by["displayable_text"] == "PASS"
    assert by["model_identity"] == "PASS"


def test_evaluate_common_model_mismatch_fails():
    m = _metrics(http_status=200, content_chars=50, server_model="gpt-4o")
    checks = K.evaluate_common(_provider("claude-opus-4-6"), m)
    by = {c["name"]: c["status"] for c in checks}
    assert by["model_identity"] == "FAIL"


def test_evaluate_common_missing_model_warns():
    m = _metrics(http_status=200, content_chars=50, server_model=None)
    by = {c["name"]: c["status"] for c in K.evaluate_common(_provider(), m)}
    assert by["model_identity"] == "WARN"


def test_evaluate_common_http_error_and_no_text_fail():
    m = _metrics(http_status=500, content_chars=0, server_model="claude-opus-4-6", error="boom")
    by = {c["name"]: c["status"] for c in K.evaluate_common(_provider(), m)}
    assert by["http_status"] == "FAIL"
    assert by["displayable_text"] == "FAIL"


def test_evaluate_common_require_text_false():
    m = _metrics(http_status=200, content_chars=0, server_model="claude-opus-4-6")
    by = {c["name"]: c["status"] for c in K.evaluate_common(_provider(), m, require_text=False)}
    assert by["displayable_text"] == "PASS"  # text optional for this case


# placeholder-compat


# ---------------------------------------------------------------------------
# evaluate_sse — required events + order + latency
# ---------------------------------------------------------------------------
def test_evaluate_sse_valid_stream():
    m = _metrics(
        event_types=["message_start", "content_block_start", "content_block_delta",
                     "message_delta", "message_stop"],
        first_event_ms=10.0, first_content_token_ms=20.0,
    )
    by = {c["name"]: c["status"] for c in K.evaluate_sse(m)}
    assert by["sse_required_events"] == "PASS"
    assert by["sse_event_order"] == "PASS"
    assert by["sse_latency"] == "PASS"


def test_evaluate_sse_missing_events_fail():
    m = _metrics(event_types=["message_start", "message_stop"])
    by = {c["name"]: c["status"] for c in K.evaluate_sse(m)}
    assert by["sse_required_events"] == "FAIL"


def test_evaluate_sse_bad_order_fail():
    m = _metrics(
        event_types=["message_stop", "content_block_delta", "message_delta", "message_start"],
        first_event_ms=10.0, first_content_token_ms=20.0,
    )
    by = {c["name"]: c["status"] for c in K.evaluate_sse(m)}
    assert by["sse_event_order"] == "FAIL"


def test_evaluate_sse_missing_latency_fail():
    m = _metrics(
        event_types=["message_start", "content_block_delta", "message_delta", "message_stop"],
        first_event_ms=None, first_content_token_ms=None,
    )
    by = {c["name"]: c["status"] for c in K.evaluate_sse(m)}
    assert by["sse_latency"] == "FAIL"


# ---------------------------------------------------------------------------
# evaluate_usage
# ---------------------------------------------------------------------------
def test_evaluate_usage_required_present():
    m = _metrics(input_tokens=10, output_tokens=5,
                 cache_creation_input_tokens=0, cache_read_input_tokens=0)
    by = {c["name"]: c["status"] for c in K.evaluate_usage(m, None)}
    assert by["usage_required_fields"] == "PASS"
    assert by["usage_cache_fields"] == "PASS"


def test_evaluate_usage_missing_required_fails():
    m = _metrics(input_tokens=None, output_tokens=5)
    by = {c["name"]: c["status"] for c in K.evaluate_usage(m, None)}
    assert by["usage_required_fields"] == "FAIL"


def test_evaluate_usage_missing_cache_warns():
    m = _metrics(input_tokens=10, output_tokens=5,
                 cache_creation_input_tokens=None, cache_read_input_tokens=None)
    by = {c["name"]: c["status"] for c in K.evaluate_usage(m, None)}
    assert by["usage_cache_fields"] == "WARN"


def test_evaluate_usage_prompt_scale():
    short = _metrics(input_tokens=100, output_tokens=5)
    long = _metrics(input_tokens=5000, output_tokens=5)
    by = {c["name"]: c["status"] for c in K.evaluate_usage(short, long)}
    assert by["usage_prompt_scale"] == "PASS"
    # longer prompt does NOT report more -> FAIL
    by2 = {c["name"]: c["status"] for c in K.evaluate_usage(long, short)}
    assert by2["usage_prompt_scale"] == "FAIL"


# ---------------------------------------------------------------------------
# evaluate_json
# ---------------------------------------------------------------------------
def test_evaluate_json_perfect():
    case = {"expected_json": {"a": 1, "b": "x"}}
    by = {c["name"]: c["status"] for c in K.evaluate_json(case, '{"a": 1, "b": "x"}')}
    assert by["json_no_markdown"] == "PASS"
    assert by["json_parse"] == "PASS"
    assert by["json_schema"] == "PASS"


def test_evaluate_json_markdown_wrapped_fails():
    case = {"expected_json": {"a": 1}}
    checks = K.evaluate_json(case, '```json\n{"a": 1}\n```')
    by = {c["name"]: c["status"] for c in checks}
    assert by["json_no_markdown"] == "FAIL"


def test_evaluate_json_parse_failure():
    case = {"expected_json": {"a": 1}}
    by = {c["name"]: c["status"] for c in K.evaluate_json(case, "not json")}
    assert by["json_parse"] == "FAIL"


def test_evaluate_json_value_mismatch_fails_schema():
    case = {"expected_json": {"a": 1}}
    by = {c["name"]: c["status"] for c in K.evaluate_json(case, '{"a": 2}')}
    assert by["json_schema"] == "FAIL"


def test_evaluate_json_no_expected_skips():
    assert K.evaluate_json({}, "anything") == []


# ---------------------------------------------------------------------------
# request_has_cache_control + evaluate_cache
# ---------------------------------------------------------------------------
def test_request_has_cache_control_nested():
    assert K.request_has_cache_control({"messages": [{"cache_control": {"type": "ephemeral"}}]})
    assert K.request_has_cache_control([{"a": {"cache_control": 1}}])
    assert not K.request_has_cache_control({"messages": [{"role": "user"}]})


def test_evaluate_cache_sent_and_observed():
    case = {"request": {"messages": [{"cache_control": {"type": "ephemeral"}}]}}
    primary = _metrics(cache_creation_input_tokens=100, cache_read_input_tokens=0)
    by = {c["name"]: c["status"] for c in K.evaluate_cache(case, primary, None)}
    assert by["cache_control_sent"] == "PASS"
    assert by["cache_usage_observed"] == "PASS"


def test_evaluate_cache_not_sent_fails():
    case = {"request": {"messages": [{"role": "user"}]}}
    primary = _metrics()
    by = {c["name"]: c["status"] for c in K.evaluate_cache(case, primary, None)}
    assert by["cache_control_sent"] == "FAIL"
    assert by["cache_usage_observed"] == "WARN"


# ---------------------------------------------------------------------------
# evaluate_expected_substring + build_payload
# ---------------------------------------------------------------------------
def test_evaluate_expected_substring():
    assert K.evaluate_expected_substring({"expected_substring": "FOO"}, "the foo is here")[0]["status"] == "PASS"
    assert K.evaluate_expected_substring({"expected_substring": "FOO"}, "no marker")[0]["status"] == "WARN"
    assert K.evaluate_expected_substring({}, "x") == []


def test_build_payload_defaults_and_messages():
    p = _provider()
    payload = K.build_payload(p, {"default_max_tokens": 256}, {}, {"prompt": "hi"})
    assert payload["model"] == "claude-opus-4-6"
    assert payload["stream"] is True
    assert payload["max_tokens"] == 256
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_build_payload_passes_tools_and_system():
    p = _provider()
    req = {"messages": [{"role": "user", "content": "x"}], "system": "sys",
           "tools": [{"name": "t"}], "tool_choice": {"type": "any"}}
    payload = K.build_payload(p, {}, {}, req)
    assert payload["system"] == "sys"
    assert payload["tools"] == [{"name": "t"}]
    assert payload["tool_choice"] == {"type": "any"}


# ---------------------------------------------------------------------------
# evaluate_case — the per-category dispatcher tying graders together
# ---------------------------------------------------------------------------
def _good_metrics(**over):
    return _metrics(http_status=200, content_chars=20, server_model="claude-opus-4-6", **over)


def test_evaluate_case_messages_substring():
    checks = K.evaluate_case(
        suite={}, case={"category": "messages", "expected_substring": "FOO"},
        provider=_provider(), primary_metrics=_good_metrics(), primary_response_text="has FOO here")
    names = {c["name"] for c in checks}
    assert "expected_substring" in names
    assert "http_status" in names


def test_evaluate_case_json_category():
    checks = K.evaluate_case(
        suite={}, case={"category": "json", "expected_json": {"a": 1}},
        provider=_provider(), primary_metrics=_good_metrics(), primary_response_text='{"a": 1}')
    by = {c["name"]: c["status"] for c in checks}
    assert by["json_schema"] == "PASS"


def test_evaluate_case_sse_category():
    m = _good_metrics(event_types=["message_start", "content_block_delta", "message_delta", "message_stop"],
                      first_event_ms=10.0, first_content_token_ms=20.0)
    checks = K.evaluate_case(suite={}, case={"category": "sse"}, provider=_provider(),
                             primary_metrics=m, primary_response_text="hi")
    by = {c["name"]: c["status"] for c in checks}
    assert by["sse_required_events"] == "PASS"


def test_evaluate_case_usage_category():
    primary = _good_metrics(input_tokens=100, output_tokens=10)
    secondary = _good_metrics(input_tokens=5000, output_tokens=10)
    checks = K.evaluate_case(suite={}, case={"category": "usage"}, provider=_provider(),
                             primary_metrics=primary, primary_response_text="hi",
                             secondary_metrics=secondary)
    by = {c["name"]: c["status"] for c in checks}
    assert by["usage_required_fields"] == "PASS"
    assert by["usage_prompt_scale"] == "PASS"


def test_evaluate_case_cache_category():
    case = {"category": "cache", "request": {"messages": [{"cache_control": {"type": "ephemeral"}}]}}
    primary = _good_metrics(cache_creation_input_tokens=50, cache_read_input_tokens=0)
    checks = K.evaluate_case(suite={}, case=case, provider=_provider(),
                             primary_metrics=primary, primary_response_text="hi")
    by = {c["name"]: c["status"] for c in checks}
    assert by["cache_control_sent"] == "PASS"


def test_evaluate_case_tool_call_skips_text_requirement():
    # tool_call category does not require displayable text
    m = _metrics(http_status=200, content_chars=0, server_model="claude-opus-4-6",
                 tool_use_event_count=1)
    checks = K.evaluate_case(suite={}, case={"category": "tool_call"}, provider=_provider(),
                             primary_metrics=m, primary_response_text="")
    by = {c["name"]: c["status"] for c in checks}
    assert by["displayable_text"] == "PASS"  # optional for tool_call


def test_evaluate_case_forced_tool_choice_unsupported():
    # an HTTP 400 with a tool_choice/thinking error on the tool_call_probe is treated as WARN
    m = _metrics(http_status=400, error="tool_choice not supported with thinking",
                 server_model="claude-opus-4-6")
    checks = K.evaluate_case(
        suite={}, case={"id": "tool_call_probe", "category": "tool_call"},
        provider=_provider(), primary_metrics=m, primary_response_text="")
    by = {c["name"]: c["status"] for c in checks}
    assert by["tool_choice_support"] == "WARN"

