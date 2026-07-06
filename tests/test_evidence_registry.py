"""Tests for evidence_registry pure helpers (T4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evidence_registry as ER  # noqa: E402


# ---------------------------------------------------------------------------
# safe_evidence_id
# ---------------------------------------------------------------------------
def test_safe_evidence_id_ok():
    assert ER.safe_evidence_id("gate_20260628_abc") == "gate_20260628_abc"


@pytest.mark.parametrize("bad", ["", "  ", "../x", "a/b", "a b", "x;y"])
def test_safe_evidence_id_rejects(bad):
    with pytest.raises(ValueError):
        ER.safe_evidence_id(bad)


def test_unique_artifact_id_shape():
    aid = ER.unique_artifact_id("gate")
    assert aid.startswith("gate_")
    # prefix + timestamp + token; deterministic prefix, unique tail
    assert ER.unique_artifact_id("gate") != aid


def test_unique_artifact_id_bad_prefix():
    with pytest.raises(ValueError):
        ER.unique_artifact_id("bad/prefix")


# ---------------------------------------------------------------------------
# read_manifest / manifest_id
# ---------------------------------------------------------------------------
def test_read_manifest(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"id": "x"}), encoding="utf-8")
    assert ER.read_manifest(p) == {"id": "x"}


def test_read_manifest_non_object(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        ER.read_manifest(p)


def test_manifest_id_from_key():
    assert ER.manifest_id({"gate_id": "g1"}, "gate_id", Path("dir/g_fallback")) == "g1"


def test_manifest_id_fallback_to_path_name():
    assert ER.manifest_id({}, "gate_id", Path("dir/g_fallback")) == "g_fallback"
    assert ER.manifest_id(None, None, Path("dir/g_fallback")) == "g_fallback"


# ---------------------------------------------------------------------------
# dedupe_paths / dedupe_strings
# ---------------------------------------------------------------------------
def test_dedupe_paths(tmp_path):
    a = tmp_path / "x"
    out = ER.dedupe_paths([a, a, tmp_path / "y"])
    assert len(out) == 2


def test_dedupe_strings():
    assert ER.dedupe_strings(["b", "a", "a", None, ""]) == ["a", "b"]


# ---------------------------------------------------------------------------
# path_is_within / path_is_within_any
# ---------------------------------------------------------------------------
def test_path_is_within(tmp_path):
    inside = tmp_path / "sub" / "f.txt"
    assert ER.path_is_within(inside, tmp_path) is True
    assert ER.path_is_within(tmp_path.parent / "other", tmp_path) is False


def test_path_is_within_any(tmp_path):
    inside = tmp_path / "f"
    assert ER.path_is_within_any(inside, [tmp_path]) is True
    assert ER.path_is_within_any(inside, []) is True   # no roots -> unrestricted
    assert ER.path_is_within_any(inside, [tmp_path / "elsewhere"]) is False


# ---------------------------------------------------------------------------
# resolve_evidence_ref_path / evidence_id_from_ref
# ---------------------------------------------------------------------------
def test_resolve_evidence_ref_path_relative(tmp_path):
    (tmp_path / "art.json").write_text("{}", encoding="utf-8")
    found = ER.resolve_evidence_ref_path("art.json", search_roots=[tmp_path], allowed_roots=[tmp_path])
    assert found is not None and found.name == "art.json"


def test_resolve_evidence_ref_path_none():
    assert ER.resolve_evidence_ref_path(None, search_roots=[Path(".")]) is None
    assert ER.resolve_evidence_ref_path("", search_roots=[Path(".")]) is None


def test_resolve_evidence_ref_path_missing(tmp_path):
    assert ER.resolve_evidence_ref_path("ghost.json", search_roots=[tmp_path]) is None


def test_evidence_id_from_ref_dir(tmp_path):
    d = tmp_path / "gate_1"
    d.mkdir()
    assert ER.evidence_id_from_ref(str(d), resolved_path=d) == "gate_1"


def test_evidence_id_from_ref_file_uses_parent(tmp_path):
    d = tmp_path / "gate_2"
    d.mkdir()
    f = d / "manifest.json"
    f.write_text("{}", encoding="utf-8")
    assert ER.evidence_id_from_ref(str(f), resolved_path=f) == "gate_2"


def test_evidence_id_from_ref_none():
    assert ER.evidence_id_from_ref(None) is None
