"""Tests for acceptance_pack.verify_acceptance_pack.

Verifies a delivered acceptance .zip: it must contain acceptance_manifest.json
and checksums.sha256, and every checksummed entry must hash-match. This is the
integrity gate on a handoff bundle, and was at 17%. We build real zips in tmp
and drive every branch (success, missing pack, bad zip, missing manifest,
checksum mismatch, extra entry).
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import acceptance_pack as AP  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_pack(tmp_path, entries: dict[str, bytes], checksums: dict[str, str] | None = None,
               include_checksums=True, include_manifest=True) -> Path:
    """Build a .zip. checksums maps name->digest line; default = correct hashes."""
    path = tmp_path / "pack.zip"
    with zipfile.ZipFile(path, "w") as zf:
        if include_manifest and "acceptance_manifest.json" not in entries:
            entries = {"acceptance_manifest.json": b'{"v":1}', **entries}
        for name, data in entries.items():
            zf.writestr(name, data)
        if include_checksums:
            if checksums is None:
                checksums = {name: _sha(data) for name, data in entries.items()}
            lines = "\n".join(f"{digest}  {name}" for name, digest in checksums.items())
            zf.writestr("checksums.sha256", lines.encode("utf-8"))
    return path


def test_verify_success(tmp_path):
    pack = _make_pack(tmp_path, {"report.txt": b"hello"})
    result = AP.verify_acceptance_pack(pack)
    assert result["verified"] is True
    assert result["error"] is None


def test_verify_missing_pack(tmp_path):
    result = AP.verify_acceptance_pack(tmp_path / "nope.zip")
    assert result["verified"] is False
    assert result["error"] == "pack_missing"


def test_verify_invalid_zip(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_text("not a zip file", encoding="utf-8")
    result = AP.verify_acceptance_pack(bad)
    assert result["error"] == "invalid_zip"


def test_verify_missing_manifest(tmp_path):
    # build a zip with checksums but no manifest
    path = tmp_path / "p.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("report.txt", b"x")
        zf.writestr("checksums.sha256", f"{_sha(b'x')}  report.txt".encode("utf-8"))
    result = AP.verify_acceptance_pack(path)
    assert result["error"] == "missing_manifest_or_checksums"
    assert "acceptance_manifest.json" in result["missing"]


def test_verify_checksum_mismatch(tmp_path):
    # declare a wrong digest for report.txt; manifest digest stays correct
    path = tmp_path / "p.zip"
    manifest = b'{"v":1}'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("acceptance_manifest.json", manifest)
        zf.writestr("report.txt", b"real content")
        lines = f"{_sha(manifest)}  acceptance_manifest.json\n{'0'*64}  report.txt"
        zf.writestr("checksums.sha256", lines.encode("utf-8"))
    result = AP.verify_acceptance_pack(path)
    assert result["error"] == "checksum_mismatch"
    assert "report.txt" in result["mismatches"]


def test_verify_checksum_entry_file_absent(tmp_path):
    # checksums list a file that isn't in the zip -> mismatch
    path = tmp_path / "p.zip"
    manifest = b'{"v":1}'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("acceptance_manifest.json", manifest)
        lines = f"{_sha(manifest)}  acceptance_manifest.json\n{'a'*64}  ghost.txt"
        zf.writestr("checksums.sha256", lines.encode("utf-8"))
    result = AP.verify_acceptance_pack(path)
    assert result["error"] == "checksum_mismatch"
    assert "ghost.txt" in result["mismatches"]


def test_verify_extra_entry(tmp_path):
    # a file present in the zip but not in checksums -> extra_entries
    path = tmp_path / "p.zip"
    manifest = b'{"v":1}'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("acceptance_manifest.json", manifest)
        zf.writestr("surprise.txt", b"unlisted")
        lines = f"{_sha(manifest)}  acceptance_manifest.json"
        zf.writestr("checksums.sha256", lines.encode("utf-8"))
    result = AP.verify_acceptance_pack(path)
    assert result["error"] == "checksum_mismatch"
    assert "surprise.txt" in result["extra_entries"]


def test_checksum_line_regex():
    m = AP.CHECKSUM_LINE.match(f"{'a'*64}  some/file.txt")
    assert m and m.group(2) == "some/file.txt"
    assert AP.CHECKSUM_LINE.match("tooshort  file") is None
