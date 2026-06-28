from __future__ import annotations

import re
from typing import Any


SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{12,}"), "sk-[REDACTED]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,'\"}]+"), r"\1[REDACTED]"),
    # Bare "Bearer <token>" with no Authorization label (e.g. a header value
    # logged on its own, or a JSON value "Authorization": "Bearer xyz" where the
    # walker sees only the value). Without this, such tokens leaked.
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._-]{8,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,'\"}]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)[^\"'\s,}]+"), r"\1[REDACTED]"),
]

# Dict-key name fragments whose VALUE must be redacted regardless of the value's
# own shape — a bare key string under "x-api-key" has no intrinsic secret marker
# (no sk- / bearer), so the structural walker would otherwise pass it through.
SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "auth_secret",
    "authorization",
    "bearer",
    "client_secret",
    "password",
    "secret",
    "token",
    "x-api-key",
)

# Suffixes that mark a key as METADATA about a credential, not the credential
# itself — an env-var NAME, a presence flag, a salted fingerprint, an id. These
# must NOT be redacted (e.g. api_key_present=False, api_key_env="TESTED_KEY"),
# or the config/observability surface loses the fields the UI needs.
SECRET_KEY_METADATA_SUFFIXES = (
    "_present",
    "_env",
    "_name",
    "_id",
    "_fingerprint",
    "_hash",
    "_set",
    "_configured",
    "_count",
)


def is_secret_key(key: Any) -> bool:
    """True if a dict key name signals its value is a raw credential.

    Matches credential fragments (api_key, secret, token, ...) but explicitly
    NOT metadata-about-a-credential keys (api_key_env, api_key_present, ...),
    whose values are names/flags, not secrets.
    """
    lowered = str(key).lower()
    if lowered.endswith(SECRET_KEY_METADATA_SUFFIXES):
        return False
    return any(fragment in lowered for fragment in SECRET_KEY_FRAGMENTS)



def redact_text(value: Any, *, max_chars: int | None = None) -> str | None:
    if value is None:
        return None
    text = str(value)
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def redact_raw_fragments(value: Any, raw_values: list[Any] | None = None, *, max_chars: int | None = None) -> str | None:
    text = redact_text(value, max_chars=None)
    if text is None:
        return None
    for raw in raw_values or []:
        raw_text = str(raw or "")
        fragments: set[str] = set()
        if len(raw_text) >= 16:
            fragments.add(raw_text)
        for line in raw_text.splitlines():
            line = line.strip()
            if len(line) >= 16:
                fragments.add(line)
        if len(raw_text) > 64:
            for start in range(0, len(raw_text), 32):
                fragment = raw_text[start : start + 64].strip()
                if len(fragment) >= 32:
                    fragments.add(fragment)
        for fragment in sorted(fragments, key=len, reverse=True):
            text = text.replace(fragment, "[REDACTED_RAW]")
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def redact_value(value: Any, *, max_chars: int | None = None, parent_key: Any = None) -> Any:
    # A value sitting under a credential-named key is redacted whole, even if the
    # value itself carries no intrinsic secret marker (e.g. {"x-api-key": "abc123"}).
    # Booleans/numbers are never secrets (a presence flag like api_key_present=False
    # must survive), so only string credential values are masked here.
    if (parent_key is not None and is_secret_key(parent_key)
            and isinstance(value, str)):
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value, max_chars=max_chars)
    if isinstance(value, list):
        return [redact_value(item, max_chars=max_chars, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, max_chars=max_chars, parent_key=parent_key) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, max_chars=max_chars, parent_key=key) for key, item in value.items()}
    return value


def _self_test() -> int:
    # Pattern coverage.
    assert "sk-ABCDEF1234567890" not in (redact_text("x sk-ABCDEF1234567890 y") or "")
    assert redact_text("sk-short") == "sk-short"            # too short to be a key
    assert "tok123456789" not in (redact_text("Authorization: Bearer tok123456789") or "")
    assert "tok123456789" not in (redact_text("Bearer tok123456789") or "")  # bare bearer
    assert "secretval12345" not in (redact_text("x-api-key: secretval12345") or "")
    assert redact_text(None) is None
    assert (redact_text("y" * 50, max_chars=5) or "").endswith("...[truncated]")

    # Key-aware structural redaction: a value under a credential key is scrubbed
    # even with no intrinsic marker — this was a real leak before.
    out = redact_value({"headers": {"x-api-key": "barevalue123456"}, "n": 5})
    assert "barevalue123456" not in str(out), out
    assert out["n"] == 5
    assert is_secret_key("X-Api-Key") and is_secret_key("authorization") and not is_secret_key("model")

    # Metadata-about-a-credential keys must SURVIVE (not the secret itself):
    # api_key_present is a flag, api_key_env is an env-var name, _fingerprint a hash.
    assert not is_secret_key("api_key_present")
    assert not is_secret_key("api_key_env")
    assert not is_secret_key("key_fingerprint")
    meta = redact_value({"api_key_present": False, "api_key_env": "TESTED_KEY"})
    assert meta["api_key_present"] is False
    assert meta["api_key_env"] == "TESTED_KEY"
    # but the raw credential value under api_key IS masked
    assert redact_value({"api_key": "rawsecretvalue"})["api_key"] == "[REDACTED]"

    # Raw-fragment scrub for a known echoed key.
    assert "RAWKEY1234567890" not in (redact_raw_fragments("got RAWKEY1234567890", ["RAWKEY1234567890"]) or "")

    print("redaction self-test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
