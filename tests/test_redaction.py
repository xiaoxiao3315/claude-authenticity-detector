"""Secret-safety tests for redaction.py.

This is the module that keeps API keys out of every persisted record, log
line, and report. The detector's iron rule is "never read or store keys", so
a redaction miss is the single worst class of bug in the repo — worse than a
wrong verdict. These tests pin every pattern, the raw-fragment scrubber, and
the recursive structure walker, and assert NO known secret shape survives.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redaction import redact_raw_fragments, redact_text, redact_value  # noqa: E402


# ---------------------------------------------------------------------------
# redact_text — each SECRET_PATTERN
# ---------------------------------------------------------------------------
def test_redact_sk_key():
    out = redact_text("my key is sk-ABCDEF1234567890 trailing")
    assert "sk-ABCDEF1234567890" not in out
    assert "[REDACTED]" in out
    assert "trailing" in out  # surrounding text preserved


def test_redact_anthropic_sk_ant_key():
    out = redact_text("sk-ant-api03-abcDEF123456789xyz")
    assert "abcDEF123456789xyz" not in out
    assert "[REDACTED]" in out


def test_redact_bearer_header():
    out = redact_text("Authorization: Bearer abcdef123456789")
    assert "abcdef123456789" not in out
    assert "[REDACTED]" in out


def test_redact_x_api_key_header():
    out = redact_text("x-api-key: supersecretvalue12345")
    assert "supersecretvalue12345" not in out
    assert "[REDACTED]" in out


def test_redact_api_key_json():
    out = redact_text('{"api_key": "verysecretvalue123"}')
    assert "verysecretvalue123" not in out
    assert "[REDACTED]" in out


def test_redact_api_key_variants():
    for text in ['apikey="secretvalue123456"',
                 'api-key = secretvalue123456',
                 'API_KEY: secretvalue123456']:
        out = redact_text(text)
        assert "secretvalue123456" not in out, text


def test_redact_short_sk_not_a_key_passes():
    # fewer than 12 chars after sk- is not a key shape — must not over-redact
    assert redact_text("sk-short") == "sk-short"


def test_redact_none_returns_none():
    assert redact_text(None) is None


def test_redact_non_string_coerced():
    assert redact_text(12345) == "12345"


def test_redact_max_chars_truncates():
    out = redact_text("x" * 100, max_chars=10)
    assert out == "xxxxxxxxxx...[truncated]"


def test_redact_multiple_secrets_in_one_string():
    text = "Bearer tok123456789 and sk-LIVEKEY1234567890 both"
    out = redact_text(text)
    assert "tok123456789" not in out
    assert "sk-LIVEKEY1234567890" not in out


# placeholder-redaction


# ---------------------------------------------------------------------------
# redact_raw_fragments — scrub known raw secret values verbatim
# ---------------------------------------------------------------------------
def test_raw_fragments_scrubs_full_value():
    out = redact_raw_fragments("response has MYSECRETKEY1234567890 in it",
                               ["MYSECRETKEY1234567890"])
    assert "MYSECRETKEY1234567890" not in out
    assert "[REDACTED_RAW]" in out


def test_raw_fragments_scrubs_multiline_value():
    raw = "line-one-secret-aaaaaaaa\nline-two-secret-bbbbbbbb"
    leaked = f"echo: {raw}"
    out = redact_raw_fragments(leaked, [raw])
    assert "line-one-secret-aaaaaaaa" not in out
    assert "line-two-secret-bbbbbbbb" not in out


def test_raw_fragments_scrubs_long_value_substrings():
    # a long key echoed back with surrounding noise still gets scrubbed
    raw = "A" * 80
    out = redact_raw_fragments(f"prefix {raw} suffix", [raw])
    assert raw not in out


def test_raw_fragments_ignores_short_values():
    # values under 16 chars are not treated as fragments (too noisy)
    out = redact_raw_fragments("the word hello appears", ["hello"])
    assert "hello" in out  # not scrubbed


def test_raw_fragments_none_returns_none():
    assert redact_raw_fragments(None, ["x"]) is None


def test_raw_fragments_also_applies_text_patterns():
    # a raw list value plus an sk- key: both gone
    out = redact_raw_fragments("sk-ABCDEF1234567890 and RAWVALUE1234567890abc",
                               ["RAWVALUE1234567890abc"])
    assert "sk-ABCDEF1234567890" not in out
    assert "RAWVALUE1234567890abc" not in out


# ---------------------------------------------------------------------------
# redact_value — recursive walk over dict/list/tuple
# ---------------------------------------------------------------------------
def test_redact_value_in_dict():
    out = redact_value({"headers": {"x-api-key": "secretvalue123456"}, "n": 5})
    assert "secretvalue123456" not in str(out)
    assert out["n"] == 5


def test_redact_value_in_list_and_tuple():
    out = redact_value(["sk-LIVEKEY1234567890", ("Bearer tok123456789",)])
    flat = str(out)
    assert "sk-LIVEKEY1234567890" not in flat
    assert "tok123456789" not in flat
    # tuple becomes list (json-friendly)
    assert isinstance(out[1], list)


def test_redact_value_passes_non_str_scalars():
    assert redact_value(42) == 42
    assert redact_value(None) is None
    assert redact_value(True) is True


def test_redact_value_nested_structure():
    payload = {"runs": [{"auth": "Authorization: Bearer abc123456789",
                         "meta": {"key": 'api_key="deepsecret123456"'}}]}
    out = redact_value(payload)
    flat = str(out)
    assert "abc123456789" not in flat
    assert "deepsecret123456" not in flat

