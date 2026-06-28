"""Tests for audit_export pure helpers, redaction, checksum round-trip (T7)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import audit_export as A  # noqa: E402


# ---------------------------------------------------------------------------
# hashing / path helpers
# ---------------------------------------------------------------------------
def test_sha256_helpers():
    assert A.sha256_text("hello").startswith("2cf24dba5fb0a30e")
    assert A.sha256_bytes(b"hello") == A.sha256_text("hello")


def test_file_sha256(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    assert A.file_sha256(p) == A.sha256_text("hello")
    assert A.file_sha256(tmp_path / "nope") is None


def test_rel_path(tmp_path):
    inside = tmp_path / "a" / "b.txt"
    assert A.rel_path(inside, tmp_path) == "a/b.txt"
    # outside base -> returns str(path)
    assert A.rel_path(Path("/elsewhere/x"), tmp_path) == str(Path("/elsewhere/x"))


def test_resolve_path(tmp_path):
    (tmp_path / "r.txt").write_text("x", encoding="utf-8")
    found = A.resolve_path("r.txt", root_dir=tmp_path, run_dir=tmp_path)
    assert found is not None and found.name == "r.txt"
    assert A.resolve_path(None, root_dir=tmp_path, run_dir=tmp_path) is None


def test_record_id_provider_task():
    rec = {"record_id": "r1", "provider": {"id": "p"}, "task": {"id": "t"}}
    assert A.record_id(rec) == "r1"
    assert A.record_id({"source_record_id": "s1"}) == "s1"
    assert A.provider_from_record(rec) == "p"
    assert A.task_from_record(rec) == "t"


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------
def test_redacted_scalar():
    out = A.redacted_scalar("secret")
    assert out["redacted"] is True
    assert out["chars"] == 6
    assert "sha256" in out


def test_should_redact_key():
    assert A.should_redact_key("api_key") is True
    assert A.should_redact_key("authorization") is True
    assert A.should_redact_key("model") is False


def test_looks_like_secret():
    assert A.looks_like_secret("Bearer abc") is True
    assert A.looks_like_secret("sk-XYZ") is True
    assert A.looks_like_secret("just text") is False


def test_redact_value_redacts_text_keys_and_secrets():
    out = A.redact_value({"prompt": "long prompt text", "api_key": "x", "n": 5})
    # TEXT_KEYS (prompt) and secret keys are redacted; plain scalars kept
    assert isinstance(out["prompt"], dict) and out["prompt"]["redacted"] is True
    assert out["api_key"]["redacted"] is True
    assert out["n"] == 5


def test_redact_record_adds_marker():
    out = A.redact_record({"a": 1})
    assert out["_audit_redaction"] == A.REDACTION_MODE


def test_redact_event_non_object():
    out = A.redact_event([1, 2], line_number=3)
    assert out["type"] == "non_object_event"
    assert out["line_number"] == 3
    assert out["redacted"] is True


# ---------------------------------------------------------------------------
# checksum generate / verify round-trip + traversal guard
# ---------------------------------------------------------------------------
def test_generate_and_verify_checksums(tmp_path):
    export = tmp_path / "export"
    (export / "sub").mkdir(parents=True)
    (export / "a.json").write_text("{}", encoding="utf-8")
    (export / "sub" / "b.txt").write_text("data", encoding="utf-8")
    A.generate_checksums(export)
    assert (export / "checksums.sha256").exists()
    assert A.verify_checksums(export) is True


def test_verify_checksums_detects_tampering(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "a.json").write_text("{}", encoding="utf-8")
    A.generate_checksums(export)
    # tamper after checksums generated
    (export / "a.json").write_text("{tampered}", encoding="utf-8")
    assert A.verify_checksums(export) is False


def test_verify_checksums_missing_file(tmp_path):
    assert A.verify_checksums(tmp_path / "noexport") is False


def test_normalize_checksum_entry_rejects_traversal(tmp_path):
    assert A.normalize_checksum_entry(tmp_path, "../escape") is None
    assert A.normalize_checksum_entry(tmp_path, "checksums.sha256") is None
    ok = A.normalize_checksum_entry(tmp_path, "sub/file.txt")
    assert ok is not None and ok[0] == "sub/file.txt"


# ---------------------------------------------------------------------------
# csv_value
# ---------------------------------------------------------------------------
def test_csv_value():
    assert A.csv_value({"a": 1}) == '{"a":1}'
    assert A.csv_value(["x", "y"]) == '["x","y"]'
    assert A.csv_value(None) == ""
    assert A.csv_value(5) == "5"
