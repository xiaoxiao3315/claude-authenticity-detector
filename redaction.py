from __future__ import annotations

import re
from typing import Any


SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{12,}"), "sk-[REDACTED]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,'\"}]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,'\"}]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)[^\"'\s,}]+"), r"\1[REDACTED]"),
]


def redact_text(value: Any, *, max_chars: int | None = None) -> str | None:
    if value is None:
        return None
    text = str(value)
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def redact_value(value: Any, *, max_chars: int | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, max_chars=max_chars)
    if isinstance(value, list):
        return [redact_value(item, max_chars=max_chars) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, max_chars=max_chars) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, max_chars=max_chars) for key, item in value.items()}
    return value
