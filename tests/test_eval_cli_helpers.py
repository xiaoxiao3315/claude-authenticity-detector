"""Tests for eval_cli.py pure helpers.

eval_cli.py is the 1500-line CLI surface; its command handlers need a network
or argparse harness, but a layer of pure helpers underneath does the real work:
config sanitization (secret-safety), model-config loading/validation, judge
JSON parsing, rule scoring, score normalization, needle-prompt assembly, and
/v1/models payload parsing. Those had 0% coverage. These drive them directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_cli as E  # noqa: E402
from model_client import CallMetrics, Completion  # noqa: E402


# ---------------------------------------------------------------------------
# sanitize_config_value — redact secret-named keys recursively
# ---------------------------------------------------------------------------
def test_sanitize_redacts_sensitive_keys():
    cfg = {"api_key": "secret", "model": "claude", "nested": {"authorization": "Bearer x"}}
    out = E.sanitize_config_value(cfg)
    assert out["api_key"] == "[REDACTED]"
    assert out["model"] == "claude"
    assert out["nested"]["authorization"] == "[REDACTED]"


def test_sanitize_each_sensitive_token():
    for token in E.SENSITIVE_CONFIG_KEY_TOKENS:
        out = E.sanitize_config_value({f"my_{token}_field": "v"})
        assert out[f"my_{token}_field"] == "[REDACTED]", token


def test_sanitize_walks_lists():
    out = E.sanitize_config_value([{"secret": "s"}, {"ok": 1}])
    assert out[0]["secret"] == "[REDACTED]"
    assert out[1]["ok"] == 1


def test_sanitize_passes_scalars():
    assert E.sanitize_config_value(42) == 42
    assert E.sanitize_config_value("plain") == "plain"


# ---------------------------------------------------------------------------
# load_extra_body / load_model_config
# ---------------------------------------------------------------------------
def test_load_extra_body_ok():
    assert E.load_extra_body({"extra_body": {"top_p": 0.9}}, "tested") == {"top_p": 0.9}
    assert E.load_extra_body({}, "tested") == {}


def test_load_extra_body_rejects_non_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        E.load_extra_body({"extra_body": [1, 2]}, "tested")


def test_load_model_config_ok():
    raw = {"provider_id": "tested", "base_url": "https://gw.x/", "model": "claude-opus-4-6",
           "api_key_env": "K", "protocol": "anthropic_messages", "auth_type": "x-api-key"}
    cfg = E.load_model_config(raw, "tested")
    assert cfg.provider_id == "tested"
    assert cfg.base_url == "https://gw.x"   # trailing slash stripped
    assert cfg.protocol == "anthropic_messages"


def test_load_model_config_rejects_bad_protocol():
    raw = {"provider_id": "p", "base_url": "u", "model": "m", "api_key_env": "K",
           "protocol": "grpc", "auth_type": "bearer"}
    with pytest.raises(ValueError, match="protocol must be one of"):
        E.load_model_config(raw, "p")


def test_load_model_config_rejects_bad_auth_type():
    raw = {"provider_id": "p", "base_url": "u", "model": "m", "api_key_env": "K",
           "protocol": "openai_chat", "auth_type": "weird"}
    with pytest.raises(ValueError, match="auth_type must be one of"):
        E.load_model_config(raw, "p")


# placeholder-evalcli


# ---------------------------------------------------------------------------
# parse_judge_json
# ---------------------------------------------------------------------------
def test_parse_judge_json_plain():
    value, err = E.parse_judge_json('{"score_0_10": 8, "decision": "GO"}')
    assert err is None
    assert value["decision"] == "GO"


def test_parse_judge_json_embedded():
    # judge wraps JSON in prose -> regex rescue
    value, err = E.parse_judge_json('Here is my verdict: {"score_0_10": 5} thanks')
    assert err is None
    assert value["score_0_10"] == 5


def test_parse_judge_json_no_json():
    value, err = E.parse_judge_json("no json here at all")
    assert value is None
    assert "did not return a JSON object" in err


def test_parse_judge_json_non_object():
    value, err = E.parse_judge_json("[1, 2, 3]")
    assert value is None
    assert err


# ---------------------------------------------------------------------------
# normalize_score — clamp to 0..10
# ---------------------------------------------------------------------------
def test_normalize_score():
    assert E.normalize_score(8.5) == 8.5
    assert E.normalize_score(15) == 10.0
    assert E.normalize_score(-3) == 0.0
    assert E.normalize_score("7") == 7.0
    assert E.normalize_score("bad") is None
    assert E.normalize_score(None) is None


# ---------------------------------------------------------------------------
# json_exact_rule_score
# ---------------------------------------------------------------------------
def test_json_exact_rule_score_match():
    task = {"scoring_type": "json_exact", "expected_json": {"a": 1}}
    res = E.json_exact_rule_score(task, '{"a": 1}')
    assert res["score"] == 10.0
    assert res["format_ok"] is True


def test_json_exact_rule_score_mismatch():
    task = {"scoring_type": "json_exact", "expected_json": {"a": 1}}
    res = E.json_exact_rule_score(task, '{"a": 2}')
    assert res["score"] == 2.0
    assert res["format_ok"] is False


def test_json_exact_rule_score_invalid_json():
    task = {"scoring_type": "json_exact", "expected_json": {"a": 1}}
    res = E.json_exact_rule_score(task, "not json")
    assert res["score"] == 0.0
    assert "invalid JSON" in res["details"]


def test_json_exact_rule_score_not_applicable():
    assert E.json_exact_rule_score({"scoring_type": "keyword_check"}, "x") is None


# ---------------------------------------------------------------------------
# final_score_from_judge
# ---------------------------------------------------------------------------
def _ok(text="answer"):
    return Completion(text=text, metrics=CallMetrics(ok=True))


def test_final_score_from_judge_uses_judge_payload():
    final, judge = E.final_score_from_judge(
        tested=_ok(), judge=None,
        judge_payload={"score_0_10": 9, "decision": "GO", "format_ok": True, "reason": "good"},
        judge_error=None, rule_score=None)
    assert final["score"] == 9.0
    assert final["decision"] == "GO"
    assert judge["score_0_10"] == 9.0


def test_final_score_from_judge_tested_failed():
    final, judge = E.final_score_from_judge(
        tested=Completion(text="", metrics=CallMetrics(ok=False, error="boom")),
        judge=None, judge_payload=None, judge_error=None, rule_score=None)
    assert final["score"] == 0.0
    assert final["format_ok"] is False
    assert judge is None


def test_final_score_from_judge_falls_back_to_rule():
    rule = {"score": 10.0, "format_ok": True, "details": "rule match"}
    final, judge = E.final_score_from_judge(
        tested=_ok(), judge=None, judge_payload=None, judge_error="judge down", rule_score=rule)
    assert final == rule
    assert "judge" in judge["error"].lower()


def test_final_score_from_judge_invalid_decision_defaults_review():
    final, _ = E.final_score_from_judge(
        tested=_ok(), judge=None,
        judge_payload={"score_0_10": 7, "decision": "MAYBE"},
        judge_error=None, rule_score=None)
    assert final["decision"] == "REVIEW"


def test_final_score_from_judge_redacts_reason():
    final, _ = E.final_score_from_judge(
        tested=_ok(), judge=None,
        judge_payload={"score_0_10": 7, "reason": "leaked sk-ABCDEF1234567890"},
        judge_error=None, rule_score=None,
        raw_redaction_values=["sk-ABCDEF1234567890"])
    assert "sk-ABCDEF1234567890" not in final["details"]


# ---------------------------------------------------------------------------
# expected_context / judge_messages
# ---------------------------------------------------------------------------
def test_expected_context_filters_keys():
    task = {"expected_json": {"a": 1}, "rubric": "r", "unrelated": "x"}
    ctx = E.expected_context(task)
    assert ctx == {"expected_json": {"a": 1}, "rubric": "r"}


def test_judge_messages_shape():
    msgs = E.judge_messages({"id": "t1", "prompt": "p", "category": "C"}, "the answer")
    assert msgs[0]["role"] == "system"
    assert "JSON object" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "the answer" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# model_ids_from_payload — /v1/models parsing across shapes
# ---------------------------------------------------------------------------
def test_model_ids_openai_shape():
    data = {"data": [{"id": "claude-opus-4-6"}, {"id": "claude-sonnet-4-6"}]}
    assert E.model_ids_from_payload(data) == ["claude-opus-4-6", "claude-sonnet-4-6"]


def test_model_ids_list_of_strings():
    assert E.model_ids_from_payload(["a", "b"]) == ["a", "b"]


def test_model_ids_name_or_model_keys():
    data = {"models": [{"name": "m1"}, {"model": "m2"}]}
    assert E.model_ids_from_payload(data) == ["m1", "m2"]


def test_model_ids_unexpected_shape():
    assert E.model_ids_from_payload(42) == []


# ---------------------------------------------------------------------------
# _raw_protocol_observation — header dialect detection (P1 regression)
# ---------------------------------------------------------------------------
def _model(protocol="anthropic_messages"):
    from model_client import ModelConfig
    return ModelConfig(provider_id="tested", base_url="https://gw.x", model="claude-opus-4-6",
                       api_key_env="K", protocol=protocol, auth_type="x-api-key")


def test_raw_observation_detects_anthropic_headers():
    c = Completion(text="hi", metrics=CallMetrics(ok=True),
                   raw={"stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 2}},
                   response_headers={"anthropic-request-id": "req_xyz", "request-id": "req_xyz"})
    obs = E._raw_protocol_observation(c, _model())
    assert obs["has_anthropic_request_id"] is True
    assert obs["has_anthropic_headers"] is True
    assert obs["raw_stop_reason"] == "end_turn"
    assert obs["input_tokens"] == 10


def test_raw_observation_wrapper_without_anthropic_headers():
    # an OpenAI-style wrapper: no anthropic-* headers, non-req_ id
    c = Completion(text="hi", metrics=CallMetrics(ok=True),
                   raw={"choices": [{"finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
                   response_headers={"openai-request-id": "chatcmpl-1", "cf-ray": "abc"})
    obs = E._raw_protocol_observation(c, _model(protocol="openai_chat"))
    assert obs["has_anthropic_request_id"] is False
    assert obs["has_anthropic_headers"] is False
    assert obs["raw_stop_reason"] == "stop"


def test_raw_observation_non_req_id_is_not_anthropic():
    # has an anthropic-request-id header but the value isn't req_-prefixed
    c = Completion(text="hi", metrics=CallMetrics(ok=True),
                   raw={"stop_reason": "end_turn", "usage": {}},
                   response_headers={"anthropic-request-id": "weird-format-123"})
    obs = E._raw_protocol_observation(c, _model())
    assert obs["has_anthropic_request_id"] is False  # not req_-prefixed
    assert obs["has_anthropic_headers"] is True       # still an anthropic-* header present


# ---------------------------------------------------------------------------
# _assemble_needle_prompt — reproducible, canary planted
# ---------------------------------------------------------------------------
def test_assemble_needle_prompt_reproducible():
    p1, c1 = E._assemble_needle_prompt(1000, seed=7)
    p2, c2 = E._assemble_needle_prompt(1000, seed=7)
    assert p1 == p2 and c1 == c2          # deterministic from (target, seed)
    assert c1.startswith("AUTH_CANARY=")
    assert c1 in p1                        # canary is actually planted in the body


def test_assemble_needle_prompt_seed_varies_canary():
    _, c1 = E._assemble_needle_prompt(1000, seed=1)
    _, c2 = E._assemble_needle_prompt(1000, seed=2)
    assert c1 != c2


def test_assemble_needle_prompt_scales_with_target():
    small, _ = E._assemble_needle_prompt(500, seed=1)
    big, _ = E._assemble_needle_prompt(5000, seed=1)
    assert len(big) > len(small)

