"""Tests for api_server.py pure helpers.

api_server.py serves the observability/admin HTTP layer; its request handlers
need a running server, but a layer of pure helpers does the data shaping:
config sanitization (secret-safety), job-id path-traversal guard, score/float
coercion, percentile, provider record/identity selection, /v1/models parsing,
text-model classification, auth-header dialect, query-bool parsing. 23% -> up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import api_server as S  # noqa: E402


# ---------------------------------------------------------------------------
# sanitize_config_value (secret-safety, mirrors eval_cli)
# ---------------------------------------------------------------------------
def test_sanitize_redacts_secret_keys():
    out = S.sanitize_config_value({"api_key": "s", "model": "m", "n": {"secret": "x"}})
    assert out["api_key"] == "[REDACTED]"
    assert out["model"] == "m"
    assert out["n"]["secret"] == "[REDACTED]"


def test_sanitize_walks_list_and_scalars():
    out = S.sanitize_config_value([{"token": "t"}, 5])
    assert out[0]["token"] == "[REDACTED]"
    assert out[1] == 5


# ---------------------------------------------------------------------------
# safe_run_dir — path traversal guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "../etc", "a/b", "a\\b", ".."])
def test_safe_run_dir_rejects_traversal(bad):
    with pytest.raises(ValueError):
        S.safe_run_dir(bad)


def test_safe_run_dir_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RUNS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        S.safe_run_dir("no_such_job")


def test_safe_run_dir_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RUNS_DIR", tmp_path)
    (tmp_path / "job1").mkdir()
    assert S.safe_run_dir("job1") == tmp_path / "job1"


# ---------------------------------------------------------------------------
# to_float / score_value / percentile
# ---------------------------------------------------------------------------
def test_to_float():
    assert S.to_float("3.5") == 3.5
    assert S.to_float(None) is None
    assert S.to_float("bad") is None


def test_score_value():
    assert S.score_value({"scoring": {"final_score": {"score": 8.0}}}) == 8.0
    assert S.score_value({"scoring": {}}) is None
    assert S.score_value({}) is None


def test_percentile():
    assert S.percentile([], 0.95) is None
    assert S.percentile([5.0], 0.95) == 5.0
    assert S.percentile([10.0, 20.0], 0.5) == 15.0


# ---------------------------------------------------------------------------
# provider_records / provider_identity
# ---------------------------------------------------------------------------
def test_provider_records_filters():
    records = [{"provider": {"id": "a"}}, {"provider": {"id": "b"}}]
    out = S.provider_records(records, "a")
    assert len(out) == 1
    assert out[0]["provider"]["id"] == "a"


def test_provider_records_falls_back_to_all_when_no_match():
    records = [{"provider": {"id": "a"}}]
    # no provider 'z' -> return all rather than empty
    assert S.provider_records(records, "z") == records


def test_provider_identity_found():
    records = [{"provider": {"id": "tested", "provider_display_name": "Tested GW",
                             "base_url_host": "gw.x", "leaderboard_group": "gateway_candidate",
                             "provider_channel": "gateway"}}]
    ident = S.provider_identity(records, "tested")
    assert ident["provider_display_name"] == "Tested GW"
    assert ident["provider_host"] == "gw.x"
    assert ident["provider_channel"] == "gateway"


def test_provider_identity_default_when_absent():
    ident = S.provider_identity([], "ghost")
    assert ident["provider_display_name"] == "ghost"
    assert ident["source_group"] == "gateway_candidate"
    assert ident["provider_host"] is None


# ---------------------------------------------------------------------------
# model_ids_from_payload / is_text_model / provider_auth_headers / query_bool
# ---------------------------------------------------------------------------
def test_model_ids_from_payload():
    assert S.model_ids_from_payload({"data": [{"id": "a"}]}) == ["a"]
    assert S.model_ids_from_payload(["x", "y"]) == ["x", "y"]
    assert S.model_ids_from_payload({"models": [{"name": "n"}]}) == ["n"]
    assert S.model_ids_from_payload(None) == []


def test_is_text_model():
    assert S.is_text_model("claude-opus-4-6") is True
    assert S.is_text_model("gpt-4o") is True
    assert S.is_text_model("deepseek-chat") is True
    assert S.is_text_model("text-embedding-3") is False  # embedding excluded
    assert S.is_text_model("dall-e-image") is False       # image excluded
    assert S.is_text_model("some-random-model") is False  # no known hint


def test_provider_auth_headers():
    assert S.provider_auth_headers({"auth_type": "bearer"}, "tok") == {"Authorization": "Bearer tok"}
    assert S.provider_auth_headers({"auth_type": "x-api-key"}, "k") == {"x-api-key": "k"}
    assert S.provider_auth_headers({}, "tok") == {"Authorization": "Bearer tok"}  # default bearer


def test_provider_auth_headers_rejects_unknown():
    with pytest.raises(ValueError, match="unsupported auth_type"):
        S.provider_auth_headers({"auth_type": "weird"}, "k")


def test_query_bool():
    assert S.query_bool({"live": ["1"]}, "live") is True
    assert S.query_bool({"live": ["true"]}, "live") is True
    assert S.query_bool({"live": ["no"]}, "live") is False
    assert S.query_bool({}, "live", default=True) is True
    assert S.query_bool({}, "live") is False
