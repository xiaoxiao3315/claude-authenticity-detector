from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ARCHIVE_REGISTRY_VERSION = "archive_registry_v1"
ARCHIVE_SOURCE_TYPES = {
    "run",
    "byo_import",
    "compatibility_run",
    "rescore",
    "trace_evaluation",
    "quality_gate",
    "audit_export",
    "campaign",
}
SAFE_ARCHIVE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def utc_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_id(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text or not SAFE_ARCHIVE_ID.fullmatch(text):
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def safe_source_type(value: Any) -> str:
    source_type = str(value or "").strip()
    if source_type not in ARCHIVE_SOURCE_TYPES:
        raise ValueError(f"invalid source_type: {value!r}")
    return source_type


def archive_key(source_type: str, run_id: str | None = None, evidence_id: str | None = None) -> str:
    run_part = run_id or ""
    evidence_part = evidence_id or run_part
    return f"{source_type}:{run_part}:{evidence_part}"


def new_registry() -> dict[str, Any]:
    return {"schema_version": ARCHIVE_REGISTRY_VERSION, "entries": []}


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return new_registry()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("archive registry must be a JSON object")
    entries = data.get("entries")
    if not isinstance(entries, list):
        data["entries"] = []
    data.setdefault("schema_version", ARCHIVE_REGISTRY_VERSION)
    return data


def write_registry(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tmp:
        tmp.write(payload)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    source_type = safe_source_type(entry.get("source_type"))
    run_id = str(entry.get("run_id") or "").strip() or None
    evidence_id = str(entry.get("evidence_id") or "").strip() or None
    if run_id:
        safe_id(run_id, "run_id")
    if evidence_id:
        safe_id(evidence_id, "evidence_id")
    if source_type in {"run", "byo_import"} and not run_id:
        run_id = safe_id(evidence_id, "run_id")
    if source_type in {"run", "byo_import"} and not evidence_id:
        evidence_id = run_id
    if source_type not in {"run", "byo_import", "compatibility_run", "campaign"} and not run_id:
        raise ValueError(f"run_id is required for {source_type}")
    if source_type in {"compatibility_run", "campaign"} and not evidence_id:
        evidence_id = safe_id(run_id, "evidence_id")
    if source_type not in {"run", "byo_import"} and not evidence_id:
        raise ValueError(f"evidence_id is required for {source_type}")
    key = archive_key(source_type, run_id, evidence_id)
    normalized = dict(entry)
    normalized.update({"source_type": source_type, "run_id": run_id, "evidence_id": evidence_id, "archive_key": key})
    return normalized


def list_archives(path: Path, *, include_restored: bool = False, source_type: str | None = None) -> list[dict[str, Any]]:
    data = load_registry(path)
    entries = []
    for raw in data.get("entries") or []:
        if not isinstance(raw, dict):
            continue
        try:
            entry = normalize_entry(raw)
        except ValueError:
            continue
        if source_type and entry.get("source_type") != source_type:
            continue
        if not include_restored and entry.get("restored_at"):
            continue
        entries.append(entry)
    entries.sort(key=lambda item: item.get("archived_at") or "", reverse=True)
    return entries


def active_archive(
    path: Path,
    source_type: str,
    *,
    run_id: str | None = None,
    evidence_id: str | None = None,
) -> dict[str, Any] | None:
    source_type = safe_source_type(source_type)
    key = archive_key(source_type, run_id, evidence_id)
    for entry in list_archives(path, include_restored=False):
        if entry.get("archive_key") == key:
            return entry
    return None


def is_archived(path: Path, source_type: str, *, run_id: str | None = None, evidence_id: str | None = None) -> bool:
    return active_archive(path, source_type, run_id=run_id, evidence_id=evidence_id) is not None


def any_archived(path: Path, checks: Iterable[tuple[str, str | None, str | None]]) -> bool:
    return any(is_archived(path, source_type, run_id=run_id, evidence_id=evidence_id) for source_type, run_id, evidence_id in checks)


def archive_evidence(
    path: Path,
    *,
    source_type: str,
    run_id: str | None = None,
    evidence_id: str | None = None,
    provider_id: str | None = None,
    label: str | None = None,
    reason: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    entry = normalize_entry(
        {
            "source_type": source_type,
            "run_id": run_id,
            "evidence_id": evidence_id,
            "provider_id": provider_id,
            "label": label,
            "reason": reason,
            "source_path": source_path,
        }
    )
    data = load_registry(path)
    entries = [item for item in data.get("entries") or [] if isinstance(item, dict)]
    now = utc_now()
    for item in entries:
        try:
            item_key = normalize_entry(item).get("archive_key")
        except ValueError:
            continue
        if item_key == entry["archive_key"]:
            item.update(entry)
            item.setdefault("archived_at", now)
            item["restored_at"] = None
            write_registry(path, {"schema_version": ARCHIVE_REGISTRY_VERSION, "entries": entries})
            return normalize_entry(item)
    entry.update({"archived_at": now, "restored_at": None})
    entries.append(entry)
    write_registry(path, {"schema_version": ARCHIVE_REGISTRY_VERSION, "entries": entries})
    return entry


def restore_evidence(
    path: Path,
    *,
    source_type: str,
    run_id: str | None = None,
    evidence_id: str | None = None,
) -> dict[str, Any]:
    target = normalize_entry({"source_type": source_type, "run_id": run_id, "evidence_id": evidence_id})
    data = load_registry(path)
    entries = [item for item in data.get("entries") or [] if isinstance(item, dict)]
    for item in entries:
        try:
            item_key = normalize_entry(item).get("archive_key")
        except ValueError:
            continue
        if item_key == target["archive_key"]:
            item["restored_at"] = utc_now()
            write_registry(path, {"schema_version": ARCHIVE_REGISTRY_VERSION, "entries": entries})
            return normalize_entry(item)
    raise FileNotFoundError(f"archive entry not found: {target['archive_key']}")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "archives" / "archive_registry.json"
        run = archive_evidence(path, source_type="run", run_id="run_a", label="smoke")
        assert run["archive_key"] == "run:run_a:run_a"
        assert is_archived(path, "run", run_id="run_a", evidence_id="run_a")
        rescore = archive_evidence(path, source_type="rescore", run_id="run_a", evidence_id="rescore_a")
        assert rescore["archive_key"] == "rescore:run_a:rescore_a"
        assert len(list_archives(path)) == 2
        restore_evidence(path, source_type="run", run_id="run_a", evidence_id="run_a")
        assert not is_archived(path, "run", run_id="run_a", evidence_id="run_a")
        assert len(list_archives(path)) == 1
        assert len(list_archives(path, include_restored=True)) == 2
    print("archive registry self-test ok")


if __name__ == "__main__":
    self_test()
