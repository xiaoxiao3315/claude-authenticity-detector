from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import archive_registry as archives
import evidence_registry as registry
from job_runtime import check_job_pause_stop


AUDIT_EXPORT_SCHEMA_VERSION = "audit_export_v1"
REDACTION_MODE = "redacted"

ProgressFn = Callable[[dict[str, Any]], None]

TEXT_KEYS = {
    "assistant_response",
    "content",
    "input",
    "input_json",
    "messages",
    "partial_json",
    "prompt",
    "response",
    "response_text",
    "text",
    "thinking",
}
SECRET_KEY_FRAGMENTS = (
    "api_key",
    "auth_secret",
    "authorization",
    "bearer",
    "client_secret",
    "password",
    "secret",
)


def archive_registry_path(runs_dir: Path) -> Path:
    return runs_dir.parent / "archives" / "archive_registry.json"


def archived_evidence(source_type: str, runs_dir: Path, run_id: str | None, evidence_id: str | None) -> bool:
    return archives.is_archived(archive_registry_path(runs_dir), source_type, run_id=run_id, evidence_id=evidence_id)


def archive_warning(source_type: str, evidence_id: str | None) -> str:
    return f"explicit archived {source_type} evidence was used: {evidence_id}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_dict(value: Any) -> dict[str, Any]:
    """Narrow Any -> dict (empty when not a dict) for the type checker."""
    return value if isinstance(value, dict) else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def resolve_path(path_value: Any, *, root_dir: Path, run_dir: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    raw = Path(str(path_value))
    candidates = [raw] if raw.is_absolute() else [run_dir / raw, root_dir / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def provider_from_record(record: dict[str, Any]) -> str | None:
    provider = _as_dict(record.get("provider"))
    return provider.get("id") or record.get("provider_id") or record.get("provider")


def task_from_record(record: dict[str, Any]) -> str | None:
    task = _as_dict(record.get("task"))
    return task.get("id") or record.get("task_id")


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("record_id") or record.get("source_record_id") or "")


def redacted_scalar(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        raw = value
        chars = None
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        chars = len(value)
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        raw = text.encode("utf-8")
        chars = len(text)
    out = {
        "redacted": True,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
    }
    if chars is not None:
        out["chars"] = chars
    return out


def should_redact_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in TEXT_KEYS:
        return True
    return any(fragment in lowered for fragment in SECRET_KEY_FRAGMENTS)


def redact_value(value: Any, *, parent_key: str = "") -> Any:
    if parent_key and should_redact_key(parent_key):
        return redacted_scalar(value)
    if isinstance(value, dict):
        return {str(key): redact_value(item, parent_key=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item, parent_key=parent_key) for item in value]
    if isinstance(value, str) and looks_like_secret(value):
        return redacted_scalar(value)
    return value


def looks_like_secret(value: str) -> bool:
    lowered = value.lower()
    return (
        "bearer " in lowered
        or "authorization:" in lowered
        or "api_key" in lowered
        or "sk-" in value
        or "sk_proj" in lowered
        or "sk-proj" in lowered
    )


def redact_record(record: dict[str, Any]) -> dict[str, Any]:
    redacted = redact_value(record)
    if isinstance(redacted, dict):
        redacted["_audit_redaction"] = REDACTION_MODE
    return redacted if isinstance(redacted, dict) else {"value": redacted}


def redact_event(event: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {
            "line_number": line_number,
            "type": "non_object_event",
            "redacted": True,
            "value": redacted_scalar(event),
        }
    out = redact_value(event)
    assert isinstance(out, dict)
    out["_audit_line_number"] = line_number
    out["_audit_redaction"] = REDACTION_MODE
    return out


def redact_response_file(source_path: Path, export_path: Path) -> dict[str, Any]:
    data = source_path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    payload = {
        "redaction_mode": REDACTION_MODE,
        "redacted": True,
        "source_file": str(source_path),
        "source_sha256": sha256_bytes(data),
        "bytes": len(data),
        "chars": len(text),
        "line_count": len(text.splitlines()),
        "content": redacted_scalar(text),
    }
    write_json(export_path, payload)
    return payload


def redact_events_file(source_path: Path, export_path: Path) -> dict[str, Any]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    event_count = 0
    invalid_json_count = 0
    event_types: list[str] = []
    with source_path.open("r", encoding="utf-8", errors="replace") as source, export_path.open("w", encoding="utf-8", newline="\n") as out:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            event_count += 1
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_json_count += 1
                row = {
                    "line_number": line_number,
                    "type": "invalid_json",
                    "redacted": True,
                    "raw": redacted_scalar(stripped),
                }
            else:
                event_type = str(event.get("type") or event.get("event") or "unknown") if isinstance(event, dict) else "non_object_event"
                event_types.append(event_type)
                row = redact_event(event, line_number)
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            out.write("\n")
    return {
        "redaction_mode": REDACTION_MODE,
        "redacted": True,
        "event_count": event_count,
        "invalid_json_count": invalid_json_count,
        "event_types": list(dict.fromkeys(event_types)),
    }


def evidence_index_row(
    *,
    source_type: str,
    source_path: Path | None,
    export_path: Path | None,
    export_dir: Path,
    provider_id: str | None = None,
    task_id: str | None = None,
    included: bool = True,
    missing: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "provider_id": provider_id,
        "task_id": task_id,
        "source_path": str(source_path) if source_path else "",
        "export_path": rel_path(export_path, export_dir) if export_path else "",
        "source_sha256": file_sha256(source_path) if source_path and source_path.exists() else None,
        "export_sha256": file_sha256(export_path) if export_path and export_path.exists() else None,
        "redaction": REDACTION_MODE,
        "included": included,
        "missing": missing,
        "notes": notes,
    }


def write_evidence_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source_type",
        "provider_id",
        "task_id",
        "included",
        "missing",
        "redaction",
        "source_sha256",
        "export_sha256",
        "source_path",
        "export_path",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def load_run_records(run_dir: Path, provider_id: str | None) -> list[dict[str, Any]]:
    records = read_jsonl(run_dir / "run_records.jsonl")
    if provider_id:
        records = [record for record in records if provider_from_record(record) == provider_id]
    return records


def provider_ids_from_records(records: list[dict[str, Any]], provider_id: str | None = None) -> list[str]:
    if provider_id:
        return [provider_id]
    return sorted({str(provider_from_record(record)) for record in records if provider_from_record(record)})


def row_matches_provider(row: dict[str, Any], provider_ids: set[str]) -> bool:
    if not provider_ids:
        return True
    provider = row.get("provider_id") or row.get("source_provider_id") or row.get("provider")
    if provider and str(provider) in provider_ids:
        return True
    metrics = _as_dict(row.get("metrics_snapshot"))
    provider = metrics.get("provider_id")
    return bool(provider and str(provider) in provider_ids)


def collect_gate_dir(run_dir: Path, provider_ids: set[str], gate_id: str | None) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    gates_dir = run_dir / "quality_gates"
    def matches_provider(manifest: dict[str, Any]) -> bool:
        manifest_gate_id = str(manifest.get("gate_id") or "")
        if not gate_id and manifest_gate_id and archived_evidence("quality_gate", run_dir.parent, run_dir.name, manifest_gate_id):
            return False
        if not provider_ids:
            return True
        ids = set(str(item) for item in (manifest.get("provider_ids") or []))
        return bool(ids & provider_ids)

    selection = registry.select_manifest_dir(
        base_dir=gates_dir,
        manifest_name="quality_gate_manifest.json",
        explicit_id=gate_id,
        label="quality gate",
        id_key="gate_id",
        match_manifest=matches_provider,
        mismatch_error="quality gate provider mismatch",
    )
    if selection.path:
        selected_gate_id = selection.bound_id or gate_id or selection.path.name
        if gate_id and archived_evidence("quality_gate", run_dir.parent, run_dir.name, selected_gate_id):
            warnings.append(archive_warning("quality_gate", selected_gate_id))
        return selection.path, warnings
    if gate_id and selection.error == "quality gate not found":
        warnings.append(f"quality gate not found: {gate_id}")
    else:
        warnings.append(selection.error or "matching quality gate evidence not found")
    return None, warnings


def gate_ref_dirs(
    *,
    gate_dir: Path | None,
    provider_ids: set[str],
    ref_name: str,
    id_key: str,
    base_dir: Path,
    manifest_name: str,
    manifest_id_key: str | None,
    evidence_kind: str,
    match_manifest: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[list[Path], list[str], list[str], list[str]]:
    if not gate_dir:
        return [], [], [], []
    records = collect_gate_records(gate_dir, provider_ids)
    run_dir = gate_dir.parents[1] if len(gate_dir.parents) >= 2 else gate_dir.parent
    runs_dir = gate_dir.parents[2] if len(gate_dir.parents) >= 3 else run_dir.parent
    root_dir = runs_dir.parent
    bound = registry.collect_bound_evidence_dirs(
        records=records,
        ref_name=ref_name,
        id_key=id_key,
        base_dir=base_dir,
        manifest_name=manifest_name,
        manifest_id_key=manifest_id_key,
        evidence_kind=evidence_kind,
        search_roots=[Path.cwd(), root_dir, runs_dir, run_dir, gate_dir],
        allowed_roots=[runs_dir],
        match_manifest=match_manifest,
    )
    return bound.dirs, bound.warnings, bound.expected_ids, bound.bound_ids


def collect_compatibility_dirs(
    *,
    runs_dir: Path,
    provider_ids: set[str],
    compatibility_run_id: str | None,
    gate_dir: Path | None,
) -> tuple[list[Path], list[str], list[str], list[str]]:
    warnings: list[str] = []
    dirs: list[Path] = []
    def matches_provider(manifest: dict[str, Any]) -> bool:
        manifest_run_id = str(manifest.get("run_id") or "")
        if not compatibility_run_id and manifest_run_id and archived_evidence("compatibility_run", runs_dir, manifest_run_id, manifest_run_id):
            return False
        return not provider_ids or str(manifest.get("provider_id") or "") in provider_ids

    if compatibility_run_id:
        selection = registry.select_manifest_dir(
            base_dir=runs_dir,
            manifest_name="compatibility_manifest.json",
            explicit_id=compatibility_run_id,
            label="compatibility run",
            id_key="run_id",
            match_manifest=matches_provider,
            mismatch_error="compatibility provider mismatch",
        )
        if selection.path:
            selected_id = selection.bound_id or compatibility_run_id
            if archived_evidence("compatibility_run", runs_dir, selected_id, selected_id):
                warnings.append(archive_warning("compatibility_run", selected_id))
            return [selection.path], warnings, [compatibility_run_id], [selection.bound_id or compatibility_run_id]
        warnings.append(f"compatibility run not found: {compatibility_run_id}" if selection.error == "compatibility run not found" else selection.error or f"compatibility run not found: {compatibility_run_id}")
        return dirs, warnings, [compatibility_run_id], []
    if gate_dir:
        return gate_ref_dirs(
            gate_dir=gate_dir,
            provider_ids=provider_ids,
            ref_name="compatibility_manifest_file",
            id_key="compatibility_run_id",
            base_dir=runs_dir,
            manifest_name="compatibility_manifest.json",
            manifest_id_key="run_id",
            evidence_kind="compatibility",
            match_manifest=matches_provider,
        )

    for provider_id in provider_ids or {""}:
        def matches_provider_for_id(manifest: dict[str, Any], provider_id: str = provider_id) -> bool:
            manifest_run_id = str(manifest.get("run_id") or "")
            if manifest_run_id and archived_evidence("compatibility_run", runs_dir, manifest_run_id, manifest_run_id):
                return False
            return not provider_id or manifest.get("provider_id") == provider_id

        selection = registry.select_manifest_dir(
            base_dir=runs_dir,
            manifest_name="compatibility_manifest.json",
            label="compatibility run",
            id_key="run_id",
            match_manifest=matches_provider_for_id,
        )
        if selection.path:
            dirs.append(selection.path)
    deduped = registry.dedupe_paths(dirs)
    if not deduped:
        warnings.append("matching compatibility evidence not found")
    ids = [path.name for path in deduped]
    return deduped, warnings, ids, ids


def collect_trace_dirs(run_dir: Path, provider_ids: set[str], trace_eval_id: str | None, gate_dir: Path | None) -> tuple[list[Path], list[str], list[str], list[str]]:
    warnings: list[str] = []
    def matches_provider(manifest: dict[str, Any]) -> bool:
        manifest_run_id = manifest.get("source_run_id")
        if manifest_run_id and str(manifest_run_id) != run_dir.name:
            return False
        manifest_trace_id = str(manifest.get("trace_eval_id") or "")
        if not trace_eval_id and manifest_trace_id and archived_evidence("trace_evaluation", run_dir.parent, run_dir.name, manifest_trace_id):
            return False
        metrics = _as_dict(manifest.get("provider_metrics"))
        return not provider_ids or bool(set(str(key) for key in metrics.keys()) & provider_ids)

    if trace_eval_id:
        selection = registry.select_manifest_dir(
            base_dir=run_dir / "trace_evaluations",
            manifest_name="trace_eval_manifest.json",
            explicit_id=trace_eval_id,
            label="trace evaluation",
            id_key="trace_eval_id",
            match_manifest=matches_provider,
            mismatch_error="trace evaluation provider/source mismatch",
        )
        if selection.path:
            selected_id = selection.bound_id or trace_eval_id
            if archived_evidence("trace_evaluation", run_dir.parent, run_dir.name, selected_id):
                warnings.append(archive_warning("trace_evaluation", selected_id))
            return [selection.path], warnings, [trace_eval_id], [selection.bound_id or trace_eval_id]
        warnings.append(f"trace evaluation not found: {trace_eval_id}" if selection.error == "trace evaluation not found" else selection.error or f"trace evaluation not found: {trace_eval_id}")
        return [], warnings, [trace_eval_id], []
    if gate_dir:
        return gate_ref_dirs(
            gate_dir=gate_dir,
            provider_ids=provider_ids,
            ref_name="trace_eval_manifest_file",
            id_key="trace_eval_id",
            base_dir=run_dir / "trace_evaluations",
            manifest_name="trace_eval_manifest.json",
            manifest_id_key="trace_eval_id",
            evidence_kind="trace evaluation",
            match_manifest=matches_provider,
        )

    selection = registry.select_manifest_dir(
        base_dir=run_dir / "trace_evaluations",
        manifest_name="trace_eval_manifest.json",
        label="trace evaluation",
        id_key="trace_eval_id",
        match_manifest=matches_provider,
    )
    if selection.path:
        return [selection.path], warnings, [selection.bound_id or selection.path.name], [selection.bound_id or selection.path.name]
    warnings.append("matching trace evaluation evidence not found")
    return [], warnings, [], []


def collect_rescore_dirs(run_dir: Path, provider_ids: set[str], rescore_id: str | None, gate_dir: Path | None) -> tuple[list[Path], list[str], list[str], list[str]]:
    warnings: list[str] = []
    def matches_provider(manifest: dict[str, Any]) -> bool:
        manifest_run_id = manifest.get("source_run_id")
        if manifest_run_id and str(manifest_run_id) != run_dir.name:
            return False
        manifest_rescore_id = str(manifest.get("rescore_id") or "")
        if not rescore_id and manifest_rescore_id and archived_evidence("rescore", run_dir.parent, run_dir.name, manifest_rescore_id):
            return False
        filters = _as_dict(manifest.get("filters"))
        filter_provider_id = filters.get("provider_id")
        return not provider_ids or not filter_provider_id or str(filter_provider_id) in provider_ids

    if rescore_id:
        selection = registry.select_manifest_dir(
            base_dir=run_dir / "rescores",
            manifest_name="rescore_manifest.json",
            explicit_id=rescore_id,
            label="rescore",
            id_key="rescore_id",
            match_manifest=matches_provider,
            mismatch_error="rescore provider/source mismatch",
        )
        if selection.path:
            selected_id = selection.bound_id or rescore_id
            if archived_evidence("rescore", run_dir.parent, run_dir.name, selected_id):
                warnings.append(archive_warning("rescore", selected_id))
            return [selection.path], warnings, [rescore_id], [selection.bound_id or rescore_id]
        warnings.append(f"rescore not found: {rescore_id}" if selection.error == "rescore not found" else selection.error or f"rescore not found: {rescore_id}")
        return [], warnings, [rescore_id], []
    if gate_dir:
        return gate_ref_dirs(
            gate_dir=gate_dir,
            provider_ids=provider_ids,
            ref_name="rescore_manifest_file",
            id_key="rescore_id",
            base_dir=run_dir / "rescores",
            manifest_name="rescore_manifest.json",
            manifest_id_key="rescore_id",
            evidence_kind="rescore",
            match_manifest=matches_provider,
        )
    selection = registry.select_manifest_dir(
        base_dir=run_dir / "rescores",
        manifest_name="rescore_manifest.json",
        label="rescore",
        id_key="rescore_id",
        match_manifest=matches_provider,
    )
    if selection.path:
        return [selection.path], warnings, [selection.bound_id or selection.path.name], [selection.bound_id or selection.path.name]
    warnings.append("matching rescore evidence not found")
    return [], warnings, [], []


def write_redacted_records(
    *,
    source_path: Path,
    export_path: Path,
    source_type: str,
    export_dir: Path,
    provider_ids: set[str],
    index_rows: list[dict[str, Any]],
    record_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> int:
    rows = read_jsonl(source_path)
    written = 0
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_path.exists():
        export_path.write_text("", encoding="utf-8")
    for row in rows:
        if record_filter and not record_filter(row):
            continue
        if provider_ids and not row_matches_provider(row, provider_ids):
            # run_records use nested provider objects, which row_matches_provider does not cover.
            nested_provider = provider_from_record(row)
            if nested_provider not in provider_ids:
                continue
        append_jsonl(export_path, redact_record(row))
        written += 1
    index_rows.append(
        evidence_index_row(
            source_type=source_type,
            source_path=source_path,
            export_path=export_path,
            export_dir=export_dir,
            included=True,
            missing=False,
            notes=f"{written} records",
        )
    )
    return written


def write_redacted_csv_records(
    *,
    source_path: Path,
    export_path: Path,
    source_type: str,
    export_dir: Path,
    index_rows: list[dict[str, Any]],
) -> int:
    rows = read_csv_rows(source_path)
    if not rows:
        return 0
    fieldnames = sorted({key for row in rows for key in row.keys()})
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            redacted = redact_record(row)
            writer.writerow({key: csv_value(redacted.get(key)) for key in fieldnames})
    index_rows.append(
        evidence_index_row(
            source_type=source_type,
            source_path=source_path,
            export_path=export_path,
            export_dir=export_dir,
            included=True,
            missing=False,
            notes=f"{len(rows)} rows",
        )
    )
    return len(rows)


def artifact_paths_from_run_record(record: dict[str, Any], root_dir: Path, run_dir: Path) -> tuple[Path | None, Path | None]:
    response = _as_dict(record.get("response"))
    artifacts = _as_dict(record.get("artifacts"))
    response_file = artifacts.get("response_file") or response.get("response_file")
    events_file = artifacts.get("events_file") or response.get("events_file")
    return (
        resolve_path(response_file, root_dir=root_dir, run_dir=run_dir),
        resolve_path(events_file, root_dir=root_dir, run_dir=run_dir),
    )


def export_artifacts(
    *,
    records: list[dict[str, Any]],
    root_dir: Path,
    run_dir: Path,
    export_dir: Path,
    index_rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {"responses": 0, "events": 0, "missing_responses": 0, "missing_events": 0}
    for record in records:
        provider_id = str(provider_from_record(record) or "unknown")
        task_id = str(task_from_record(record) or record_id(record) or "unknown")
        response_path, events_path = artifact_paths_from_run_record(record, root_dir, run_dir)
        if response_path and response_path.exists():
            export_path = export_dir / "artifacts" / "responses" / provider_id / f"{task_id}.redacted.json"
            redact_response_file(response_path, export_path)
            index_rows.append(
                evidence_index_row(
                    source_type="response_file",
                    provider_id=provider_id,
                    task_id=task_id,
                    source_path=response_path,
                    export_path=export_path,
                    export_dir=export_dir,
                )
            )
            counts["responses"] += 1
        else:
            counts["missing_responses"] += 1
            index_rows.append(
                evidence_index_row(
                    source_type="response_file",
                    provider_id=provider_id,
                    task_id=task_id,
                    source_path=response_path,
                    export_path=None,
                    export_dir=export_dir,
                    included=False,
                    missing=True,
                    notes="response file missing",
                )
            )
        if events_path and events_path.exists():
            export_path = export_dir / "artifacts" / "events" / provider_id / f"{task_id}.redacted.jsonl"
            redact_events_file(events_path, export_path)
            index_rows.append(
                evidence_index_row(
                    source_type="events_file",
                    provider_id=provider_id,
                    task_id=task_id,
                    source_path=events_path,
                    export_path=export_path,
                    export_dir=export_dir,
                )
            )
            counts["events"] += 1
        else:
            counts["missing_events"] += 1
            index_rows.append(
                evidence_index_row(
                    source_type="events_file",
                    provider_id=provider_id,
                    task_id=task_id,
                    source_path=events_path,
                    export_path=None,
                    export_dir=export_dir,
                    included=False,
                    missing=True,
                    notes="events file missing",
                )
            )
    return counts


def collect_gate_records(gate_dir: Path | None, provider_ids: set[str]) -> list[dict[str, Any]]:
    if not gate_dir:
        return []
    records_path = gate_dir / "quality_gate_records.jsonl"
    if not records_path.exists():
        return []
    records = read_jsonl(records_path)
    if provider_ids:
        records = [record for record in records if str(record.get("provider_id") or "") in provider_ids]
    return records


def write_summary_markdown(
    *,
    path: Path,
    manifest: dict[str, Any],
    gate_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    lines = [
        "# Audit Export Summary",
        "",
        f"- Audit export id: `{manifest.get('audit_export_id')}`",
        f"- Source run id: `{manifest.get('source_run_id')}`",
        f"- Providers: `{', '.join(manifest.get('provider_ids') or [])}`",
        f"- Redaction mode: `{manifest.get('redaction_mode')}`",
        f"- Created at: `{manifest.get('created_at')}`",
        "",
        "## Quality Gate",
    ]
    if gate_records:
        for record in gate_records:
            provider_id = record.get("provider_id")
            decision = record.get("decision")
            metrics = _as_dict(record.get("metrics_snapshot"))
            lines.extend(
                [
                    "",
                    f"### {provider_id}",
                    f"- Decision: `{decision}`",
                    f"- Gate score: `{metrics.get('gate_score')}`",
                    f"- Success rate: `{metrics.get('success_rate')}`",
                    f"- Compatibility: `{metrics.get('compatibility_suite_status')}`",
                    f"- Trace status: `{metrics.get('trace_status')}`",
                    f"- Blockers: `{len(record.get('blockers') or [])}`",
                    f"- Review items: `{len(record.get('review_items') or [])}`",
                ]
            )
            for item in record.get("blockers") or []:
                lines.append(f"  - BLOCKER `{item.get('rule_id')}`: {item.get('details')}")
            for item in record.get("review_items") or []:
                lines.append(f"  - REVIEW `{item.get('rule_id')}`: {item.get('details')}")
    else:
        lines.append("")
        lines.append("No Quality Gate record was exported.")
    lines.extend(["", "## Bound Evidence"])
    bound = _as_dict(manifest.get("bound_evidence"))
    for key, value in bound.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Warnings"])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Redaction",
            "- This v1 export does not include raw response text.",
            "- Response files are represented by hash, byte count, char count, and line count.",
            "- Event text/thinking/input JSON fields are replaced with redacted metadata and hashes.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_checksums(export_dir: Path) -> None:
    checksum_path = export_dir / "checksums.sha256"
    rows: list[str] = []
    for path in sorted(p for p in export_dir.rglob("*") if p.is_file() and p != checksum_path):
        digest = file_sha256(path)
        if digest:
            rows.append(f"{digest}  {rel_path(path, export_dir)}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def normalize_checksum_entry(export_dir: Path, relative: str) -> tuple[str, Path] | None:
    normalized = relative.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None

    candidate = Path(normalized)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        return None

    normalized_relative = "/".join(parts)
    if normalized_relative == "checksums.sha256":
        return None

    export_root = export_dir.resolve(strict=False)
    path = export_dir.joinpath(*parts)
    try:
        path.resolve(strict=False).relative_to(export_root)
    except ValueError:
        return None
    return normalized_relative, path


def verify_checksums(export_dir: Path) -> bool:
    checksum_path = export_dir / "checksums.sha256"
    if not checksum_path.exists():
        return False
    listed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "  " not in line:
            return False
        expected, relative = line.split("  ", 1)
        if len(expected) != 64 or any(char not in "0123456789abcdefABCDEF" for char in expected):
            return False
        entry = normalize_checksum_entry(export_dir, relative)
        if not entry:
            return False
        normalized_relative, path = entry
        if normalized_relative in listed:
            return False
        if file_sha256(path) != expected.lower():
            return False
        listed[normalized_relative] = expected.lower()

    actual = {
        rel_path(path, export_dir)
        for path in export_dir.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(listed) != actual:
        return False
    return True


def audit_export_source_count(runs_dir: Path, run_id: str, provider_id: str | None = None) -> int:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        return 0
    return len(load_run_records(run_dir, provider_id))


def write_stopped_audit_export_result(
    *,
    export_dir: Path,
    audit_export_id: str,
    run_id: str,
    created_at: str,
    audit_label: str | None,
    redaction_mode: str,
    provider_id: str | None,
    gate_id: str | None,
    compatibility_run_id: str | None,
    rescore_id: str | None,
    trace_eval_id: str | None,
    selected_records: list[dict[str, Any]],
    warnings: list[str],
    progress_callback: ProgressFn | None,
    completed_tasks: int,
    total_tasks: int,
    stop_reason: str = "user_stop_requested",
) -> dict[str, Any]:
    provider_ids = set(provider_ids_from_records(selected_records, provider_id))
    evidence_index_path = export_dir / "evidence_index.jsonl"
    evidence_summary_path = export_dir / "evidence_summary.csv"
    write_jsonl(evidence_index_path, [])
    write_evidence_summary(evidence_summary_path, [])
    manifest_path = export_dir / "audit_export_manifest.json"
    summary_path = export_dir / "audit_export_summary.md"
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_EXPORT_SCHEMA_VERSION,
        "audit_export_id": audit_export_id,
        "source_run_id": run_id,
        "status": "stopped",
        "created_at": created_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "audit_label": audit_label,
        "redaction_mode": redaction_mode,
        "provider_ids": sorted(provider_ids),
        "filters": {
            "provider_id": provider_id,
            "gate_id": gate_id,
            "compatibility_run_id": compatibility_run_id,
            "rescore_id": rescore_id,
            "trace_eval_id": trace_eval_id,
        },
        "expected_evidence": {
            "gate_id": gate_id,
            "compatibility_run_ids": [compatibility_run_id] if compatibility_run_id else [],
            "trace_eval_ids": [trace_eval_id] if trace_eval_id else [],
            "rescore_ids": [rescore_id] if rescore_id else [],
        },
        "bound_evidence": {
            "gate_id": None,
            "compatibility_run_ids": [],
            "trace_eval_ids": [],
            "rescore_ids": [],
        },
        "record_counts": {},
        "artifact_counts": {},
        "evidence_count": 0,
        "warnings": warnings,
        "manifest_file": str(manifest_path),
        "summary_file": str(summary_path),
        "evidence_index_file": str(evidence_index_path),
        "evidence_summary_file": str(evidence_summary_path),
        "checksums_file": None,
        "stopped": True,
        "stop_reason": stop_reason,
    }
    write_summary_markdown(path=summary_path, manifest=manifest, gate_records=[], warnings=warnings)
    write_json(manifest_path, manifest)
    if progress_callback:
        progress_callback(
            {
                "event": "run_stopped",
                "run_id": run_id,
                "audit_export_id": audit_export_id,
                "completed_tasks": completed_tasks,
                "total_tasks": total_tasks,
                "stop_reason": stop_reason,
            }
        )
    return {
        "audit_export_id": audit_export_id,
        "source_run_id": run_id,
        "record_count": len(selected_records),
        "manifest": manifest,
        "summary": summary_path.read_text(encoding="utf-8"),
        "evidence_index": [],
        "evidence_summary": read_csv_rows(evidence_summary_path),
        "stopped": True,
        "stop_reason": stop_reason,
    }


def run_audit_export(
    *,
    runs_dir: Path,
    run_id: str,
    provider_id: str | None = None,
    gate_id: str | None = None,
    compatibility_run_id: str | None = None,
    rescore_id: str | None = None,
    trace_eval_id: str | None = None,
    audit_label: str | None = None,
    redaction_mode: str = REDACTION_MODE,
    progress_callback: ProgressFn | None = None,
    job_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if redaction_mode != REDACTION_MODE:
        raise ValueError("audit export v1 only supports redaction_mode=redacted")
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_id}")
    run_records_path = run_dir / "run_records.jsonl"
    if not run_records_path.exists():
        raise FileNotFoundError(f"run_records.jsonl not found for run: {run_id}")

    created_at = datetime.now().isoformat(timespec="seconds")
    audit_export_id = registry.unique_artifact_id("audit_export")
    export_dir = run_dir / "audit_exports" / audit_export_id
    export_dir.mkdir(parents=True, exist_ok=False)

    total_source_count = audit_export_source_count(runs_dir, run_id, provider_id)
    if progress_callback:
        progress_callback({"event": "run_started", "total_tasks": total_source_count, "benchmark_mode": "audit_export", "current_provider": provider_id})

    if check_job_pause_stop(
        job_control,
        progress_callback,
        completed_tasks=0,
        total_tasks=total_source_count,
    ):
        return write_stopped_audit_export_result(
            export_dir=export_dir,
            audit_export_id=audit_export_id,
            run_id=run_id,
            created_at=created_at,
            audit_label=audit_label,
            redaction_mode=redaction_mode,
            provider_id=provider_id,
            gate_id=gate_id,
            compatibility_run_id=compatibility_run_id,
            rescore_id=rescore_id,
            trace_eval_id=trace_eval_id,
            selected_records=load_run_records(run_dir, provider_id),
            warnings=["audit export stopped before evidence collection"],
            progress_callback=progress_callback,
            completed_tasks=0,
            total_tasks=total_source_count,
        )

    root_dir = runs_dir.parent
    selected_records = load_run_records(run_dir, provider_id)
    provider_ids = set(provider_ids_from_records(selected_records, provider_id))
    warnings: list[str] = []
    if not selected_records:
        warnings.append("no run records matched export filters")

    if progress_callback:
        progress_callback({"event": "task_phase", "phase": "audit_collecting", "completed_tasks": 0, "total_tasks": len(selected_records)})
    if check_job_pause_stop(
        job_control,
        progress_callback,
        completed_tasks=0,
        total_tasks=len(selected_records),
    ):
        return write_stopped_audit_export_result(
            export_dir=export_dir,
            audit_export_id=audit_export_id,
            run_id=run_id,
            created_at=created_at,
            audit_label=audit_label,
            redaction_mode=redaction_mode,
            provider_id=provider_id,
            gate_id=gate_id,
            compatibility_run_id=compatibility_run_id,
            rescore_id=rescore_id,
            trace_eval_id=trace_eval_id,
            selected_records=selected_records,
            warnings=warnings + ["audit export stopped before evidence binding"],
            progress_callback=progress_callback,
            completed_tasks=0,
            total_tasks=len(selected_records),
        )

    gate_dir, gate_warnings = collect_gate_dir(run_dir, provider_ids, gate_id)
    warnings.extend(gate_warnings)
    compatibility_dirs, compatibility_warnings, expected_compatibility_ids, bound_compatibility_ids = collect_compatibility_dirs(
        runs_dir=runs_dir,
        provider_ids=provider_ids,
        compatibility_run_id=compatibility_run_id,
        gate_dir=gate_dir,
    )
    warnings.extend(compatibility_warnings)
    trace_dirs, trace_warnings, expected_trace_ids, bound_trace_ids = collect_trace_dirs(run_dir, provider_ids, trace_eval_id, gate_dir)
    warnings.extend(trace_warnings)
    rescore_dirs, rescore_warnings, expected_rescore_ids, bound_rescore_ids = collect_rescore_dirs(run_dir, provider_ids, rescore_id, gate_dir)
    warnings.extend(rescore_warnings)

    index_rows: list[dict[str, Any]] = []
    record_counts: dict[str, int] = defaultdict(int)

    if progress_callback:
        progress_callback({"event": "task_phase", "phase": "audit_redacting", "completed_tasks": 0, "total_tasks": len(selected_records)})

    records_dir = export_dir / "records"
    record_counts["run_records"] = write_redacted_records(
        source_path=run_records_path,
        export_path=records_dir / "run_records.redacted.jsonl",
        source_type="run_records",
        export_dir=export_dir,
        provider_ids=provider_ids,
        index_rows=index_rows,
    )
    if (run_dir / "summary.csv").exists():
        record_counts["summary_rows"] = write_redacted_csv_records(
            source_path=run_dir / "summary.csv",
            export_path=records_dir / "summary.redacted.csv",
            source_type="summary_csv",
            export_dir=export_dir,
            index_rows=index_rows,
        )
    if (run_dir / "benchmark_scores.json").exists():
        export_path = records_dir / "benchmark_scores.redacted.json"
        write_json(export_path, redact_record(read_json(run_dir / "benchmark_scores.json")))
        index_rows.append(evidence_index_row(source_type="benchmark_scores", source_path=run_dir / "benchmark_scores.json", export_path=export_path, export_dir=export_dir))

    gate_records = collect_gate_records(gate_dir, provider_ids)
    if gate_dir and (gate_dir / "quality_gate_records.jsonl").exists():
        record_counts["quality_gate_records"] = write_redacted_records(
            source_path=gate_dir / "quality_gate_records.jsonl",
            export_path=records_dir / "quality_gate_records.jsonl",
            source_type="quality_gate_records",
            export_dir=export_dir,
            provider_ids=provider_ids,
            index_rows=index_rows,
        )
        for name in ("quality_gate_manifest.json", "quality_gate_summary.csv"):
            source = gate_dir / name
            if source.exists():
                if source.suffix == ".json":
                    export_path = records_dir / f"{source.stem}.redacted.json"
                    write_json(export_path, redact_record(read_json(source)))
                    source_type = source.stem
                else:
                    export_path = records_dir / f"{source.stem}.redacted.csv"
                    write_redacted_csv_records(source_path=source, export_path=export_path, source_type=source.stem, export_dir=export_dir, index_rows=index_rows)
                    continue
                index_rows.append(evidence_index_row(source_type=source_type, source_path=source, export_path=export_path, export_dir=export_dir))

    for compat_dir in compatibility_dirs:
        if (compat_dir / "compatibility_records.jsonl").exists():
            record_counts["compatibility_records"] += write_redacted_records(
                source_path=compat_dir / "compatibility_records.jsonl",
                export_path=records_dir / "compatibility_records.jsonl",
                source_type="compatibility_records",
                export_dir=export_dir,
                provider_ids=provider_ids,
                index_rows=index_rows,
            )
        for name in ("compatibility_manifest.json", "compatibility_summary.csv"):
            source = compat_dir / name
            if source.exists():
                export_name = f"{compat_dir.name}_{source.stem}.redacted{source.suffix}"
                export_path = records_dir / export_name
                if source.suffix == ".json":
                    write_json(export_path, redact_record(read_json(source)))
                    index_rows.append(evidence_index_row(source_type=source.stem, source_path=source, export_path=export_path, export_dir=export_dir))
                else:
                    write_redacted_csv_records(source_path=source, export_path=export_path, source_type=source.stem, export_dir=export_dir, index_rows=index_rows)

    for trace_dir in trace_dirs:
        if (trace_dir / "trace_eval_records.jsonl").exists():
            record_counts["trace_eval_records"] += write_redacted_records(
                source_path=trace_dir / "trace_eval_records.jsonl",
                export_path=records_dir / "trace_eval_records.jsonl",
                source_type="trace_eval_records",
                export_dir=export_dir,
                provider_ids=provider_ids,
                index_rows=index_rows,
            )
        for name in ("trace_eval_manifest.json", "trace_eval_summary.csv"):
            source = trace_dir / name
            if source.exists():
                export_name = f"{trace_dir.name}_{source.stem}.redacted{source.suffix}"
                export_path = records_dir / export_name
                if source.suffix == ".json":
                    write_json(export_path, redact_record(read_json(source)))
                    index_rows.append(evidence_index_row(source_type=source.stem, source_path=source, export_path=export_path, export_dir=export_dir))
                else:
                    write_redacted_csv_records(source_path=source, export_path=export_path, source_type=source.stem, export_dir=export_dir, index_rows=index_rows)

    for rescore_dir in rescore_dirs:
        if (rescore_dir / "rescore_records.jsonl").exists():
            record_counts["rescore_records"] += write_redacted_records(
                source_path=rescore_dir / "rescore_records.jsonl",
                export_path=records_dir / "rescore_records.jsonl",
                source_type="rescore_records",
                export_dir=export_dir,
                provider_ids=provider_ids,
                index_rows=index_rows,
            )
        for name in ("rescore_manifest.json", "rescore_summary.csv"):
            source = rescore_dir / name
            if source.exists():
                export_name = f"{rescore_dir.name}_{source.stem}.redacted{source.suffix}"
                export_path = records_dir / export_name
                if source.suffix == ".json":
                    write_json(export_path, redact_record(read_json(source)))
                    index_rows.append(evidence_index_row(source_type=source.stem, source_path=source, export_path=export_path, export_dir=export_dir))
                else:
                    write_redacted_csv_records(source_path=source, export_path=export_path, source_type=source.stem, export_dir=export_dir, index_rows=index_rows)

    artifact_counts = export_artifacts(
        records=selected_records,
        root_dir=root_dir,
        run_dir=run_dir,
        export_dir=export_dir,
        index_rows=index_rows,
    )

    if progress_callback:
        progress_callback({"event": "task_phase", "phase": "audit_writing", "completed_tasks": len(selected_records), "total_tasks": len(selected_records)})

    evidence_index_path = export_dir / "evidence_index.jsonl"
    evidence_summary_path = export_dir / "evidence_summary.csv"
    write_jsonl(evidence_index_path, index_rows)
    write_evidence_summary(evidence_summary_path, index_rows)

    bound_evidence = {
        "gate_id": gate_dir.name if gate_dir else None,
        "compatibility_run_ids": bound_compatibility_ids or [path.name for path in compatibility_dirs],
        "trace_eval_ids": bound_trace_ids or [path.name for path in trace_dirs],
        "rescore_ids": bound_rescore_ids or [path.name for path in rescore_dirs],
    }
    expected_evidence = {
        "gate_id": gate_id or (gate_dir.name if gate_dir else None),
        "compatibility_run_ids": expected_compatibility_ids,
        "trace_eval_ids": expected_trace_ids,
        "rescore_ids": expected_rescore_ids,
    }
    stopped = check_job_pause_stop(
        job_control,
        progress_callback,
        completed_tasks=len(selected_records),
        total_tasks=len(selected_records),
    )
    stop_reason = "user_stop_requested" if stopped else None
    manifest_path = export_dir / "audit_export_manifest.json"
    summary_path = export_dir / "audit_export_summary.md"
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_EXPORT_SCHEMA_VERSION,
        "audit_export_id": audit_export_id,
        "source_run_id": run_id,
        "status": "stopped" if stopped else "completed",
        "created_at": created_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "audit_label": audit_label,
        "redaction_mode": redaction_mode,
        "provider_ids": sorted(provider_ids),
        "filters": {
            "provider_id": provider_id,
            "gate_id": gate_id,
            "compatibility_run_id": compatibility_run_id,
            "rescore_id": rescore_id,
            "trace_eval_id": trace_eval_id,
        },
        "expected_evidence": expected_evidence,
        "bound_evidence": bound_evidence,
        "record_counts": dict(record_counts),
        "artifact_counts": artifact_counts,
        "evidence_count": len(index_rows),
        "warnings": warnings,
        "manifest_file": str(manifest_path),
        "summary_file": str(summary_path),
        "evidence_index_file": str(evidence_index_path),
        "evidence_summary_file": str(evidence_summary_path),
        "checksums_file": None if stopped else str(export_dir / "checksums.sha256"),
        "stopped": stopped,
        "stop_reason": stop_reason,
    }
    write_summary_markdown(path=summary_path, manifest=manifest, gate_records=gate_records, warnings=warnings)
    write_json(manifest_path, manifest)

    if not stopped:
        if progress_callback:
            progress_callback({"event": "task_phase", "phase": "audit_checksumming", "completed_tasks": len(selected_records), "total_tasks": len(selected_records)})
        stopped = check_job_pause_stop(
            job_control,
            progress_callback,
            completed_tasks=len(selected_records),
            total_tasks=len(selected_records),
        )
        if stopped:
            stop_reason = "user_stop_requested"
            manifest["status"] = "stopped"
            manifest["checksums_file"] = None
            manifest["stopped"] = True
            manifest["stop_reason"] = stop_reason
            write_summary_markdown(path=summary_path, manifest=manifest, gate_records=gate_records, warnings=warnings)
            write_json(manifest_path, manifest)
        else:
            generate_checksums(export_dir)

    if stopped and progress_callback:
        progress_callback({"event": "run_stopped", "run_id": run_id, "audit_export_id": audit_export_id, "completed_tasks": len(selected_records), "total_tasks": len(selected_records), "stop_reason": stop_reason})

    return {
        "audit_export_id": audit_export_id,
        "source_run_id": run_id,
        "record_count": len(selected_records),
        "manifest": manifest,
        "summary": summary_path.read_text(encoding="utf-8"),
        "evidence_index": read_jsonl(evidence_index_path),
        "evidence_summary": read_csv_rows(evidence_summary_path),
        "stopped": stopped,
        "stop_reason": stop_reason,
    }


def list_audit_exports(run_dir: Path) -> list[dict[str, Any]]:
    exports_dir = run_dir / "audit_exports"
    if not exports_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted((p for p in exports_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        manifest_path = child / "audit_export_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if isinstance(manifest, dict):
            out.append(manifest)
    return out


def read_audit_export(run_dir: Path, audit_export_id: str) -> dict[str, Any]:
    export_dir = run_dir / "audit_exports" / audit_export_id
    manifest_path = export_dir / "audit_export_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"audit export not found: {audit_export_id}")
    summary_path = export_dir / "audit_export_summary.md"
    return {
        "manifest": read_json(manifest_path),
        "summary_markdown": summary_path.read_text(encoding="utf-8") if summary_path.exists() else "",
        "evidence_index": read_jsonl(export_dir / "evidence_index.jsonl"),
        "evidence_summary": read_csv_rows(export_dir / "evidence_summary.csv"),
        "checksums_valid": verify_checksums(export_dir),
    }


def make_fake_run(root: Path) -> tuple[Path, str]:
    runs_dir = root / "runs"
    run_id = "audit_fixture"
    run_dir = runs_dir / run_id
    provider_id = "provider_a"
    task_id = "task_001"
    response_path = run_dir / "responses" / provider_id / f"{task_id}.txt"
    events_path = run_dir / "events" / provider_id / f"{task_id}.jsonl"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text("SECRET_RESPONSE_TEXT FAKE_SECRET_TOKEN", encoding="utf-8")
    events: list[dict[str, Any]] = [
        {"type": "message_start"},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "SECRET_EVENT_TEXT"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 1, "output_tokens": 2}},
        {"type": "message_stop"},
    ]
    write_jsonl(events_path, events)
    record = {
        "schema_version": "run_record_v1",
        "record_id": f"{run_id}:{provider_id}:{task_id}",
        "run": {"run_id": run_id, "timestamp": "2026-06-20T00:00:00", "benchmark_mode": "fixture", "formula_version": "score_formula_v1", "runner": "self_test", "status": "completed"},
        "task": {"id": task_id, "category": "fixture", "enterprise_dimension": "fixture", "difficulty": "easy", "scoring_type": "manual_rubric", "risk_tags": [], "point_value": 100, "scoring_confidence": 0.5},
        "provider": {"id": provider_id, "api_style": "anthropic_messages", "base_url_host": "example.invalid", "auth_secret": "FAKE_SECRET_TOKEN", "model_requested": "model-a", "model_returned": "model-a"},
        "request": {"messages": [{"role": "user", "content": "SECRET_PROMPT"}]},
        "response": {"response_file": str(response_path), "events_file": str(events_path)},
        "telemetry": {"ok": True, "stop_reason": "end_turn"},
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "scoring": {"final_score": {"score": 9}},
        "trace": {"tool_calls": []},
        "artifacts": {"response_file": str(response_path), "events_file": str(events_path)},
    }
    write_jsonl(run_dir / "run_records.jsonl", [record])
    write_json(run_dir / "benchmark_scores.json", {"providers": {provider_id: {"benchmark_score": 900}}})
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "provider", "task_id", "ok", "response_text"])
        writer.writeheader()
        writer.writerow({"run_id": run_id, "provider": provider_id, "task_id": task_id, "ok": "true", "response_text": "SECRET_RESPONSE_TEXT"})

    gate_dir = run_dir / "quality_gates" / "gate_fixture"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate_record = {
        "schema_version": "quality_gate_record_v1",
        "gate_id": "gate_fixture",
        "source_run_id": run_id,
        "provider_id": provider_id,
        "decision": "GO",
        "metrics_snapshot": {"gate_score": 900, "success_rate": 1.0, "compatibility_suite_status": "PASS", "trace_status": "PASS"},
        "blockers": [],
        "review_items": [],
        "passed_rules": ["fixture"],
        "evidence_refs": {},
    }
    write_jsonl(gate_dir / "quality_gate_records.jsonl", [gate_record])
    write_json(gate_dir / "quality_gate_manifest.json", {"gate_id": "gate_fixture", "source_run_id": run_id, "provider_ids": [provider_id], "decision_counts": {"GO": 1}})
    with (gate_dir / "quality_gate_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gate_id", "provider_id", "decision"])
        writer.writeheader()
        writer.writerow({"gate_id": "gate_fixture", "provider_id": provider_id, "decision": "GO"})

    compat_dir = runs_dir / "compat_fixture"
    compat_dir.mkdir(parents=True, exist_ok=True)
    write_json(compat_dir / "compatibility_manifest.json", {"run_id": "compat_fixture", "provider_id": provider_id, "suite_status": "PASS"})
    write_jsonl(compat_dir / "compatibility_records.jsonl", [{"provider_id": provider_id, "status": "PASS", "evidence": {"text": "SECRET_EVENT_TEXT"}}])

    trace_dir = run_dir / "trace_evaluations" / "trace_fixture"
    trace_dir.mkdir(parents=True, exist_ok=True)
    write_json(trace_dir / "trace_eval_manifest.json", {"trace_eval_id": "trace_fixture", "provider_metrics": {provider_id: {"status": "PASS", "record_count": 1}}})
    write_jsonl(trace_dir / "trace_eval_records.jsonl", [{"provider_id": provider_id, "task_id": task_id, "status": "PASS"}])

    rescore_dir = run_dir / "rescores" / "rescore_fixture"
    rescore_dir.mkdir(parents=True, exist_ok=True)
    write_json(rescore_dir / "rescore_manifest.json", {"rescore_id": "rescore_fixture", "record_count": 1})
    write_jsonl(rescore_dir / "rescore_records.jsonl", [{"source_provider_id": provider_id, "source_record_id": record["record_id"], "new_final_score": {"score": 9}}])

    gate_record["evidence_refs"] = {
        "compatibility_manifest_file": str(compat_dir / "compatibility_manifest.json"),
        "trace_eval_manifest_file": str(trace_dir / "trace_eval_manifest.json"),
        "rescore_manifest_file": str(rescore_dir / "rescore_manifest.json"),
    }
    gate_record["evidence_ids"] = {
        "compatibility_run_id": "compat_fixture",
        "trace_eval_id": "trace_fixture",
        "rescore_id": "rescore_fixture",
    }
    write_jsonl(gate_dir / "quality_gate_records.jsonl", [gate_record])
    return runs_dir, run_id


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs_dir, run_id = make_fake_run(root)
        result = run_audit_export(
            runs_dir=runs_dir,
            run_id=run_id,
            provider_id="provider_a",
            gate_id="gate_fixture",
            compatibility_run_id="compat_fixture",
            rescore_id="rescore_fixture",
            trace_eval_id="trace_fixture",
            audit_label="fixture",
        )
        export_dir = runs_dir / run_id / "audit_exports" / result["audit_export_id"]
        required = [
            "audit_export_manifest.json",
            "audit_export_summary.md",
            "evidence_index.jsonl",
            "evidence_summary.csv",
            "checksums.sha256",
            "records/run_records.redacted.jsonl",
            "records/quality_gate_records.jsonl",
            "records/compatibility_records.jsonl",
            "records/trace_eval_records.jsonl",
            "records/rescore_records.jsonl",
            "artifacts/responses/provider_a/task_001.redacted.json",
            "artifacts/events/provider_a/task_001.redacted.jsonl",
        ]
        for relative in required:
            assert (export_dir / relative).exists(), relative
        exported_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in export_dir.rglob("*") if path.is_file())
        assert "SECRET_RESPONSE_TEXT" not in exported_text
        assert "SECRET_EVENT_TEXT" not in exported_text
        assert "SECRET_PROMPT" not in exported_text
        assert "FAKE_SECRET_TOKEN" not in exported_text
        assert "source_sha256" in exported_text
        assert verify_checksums(export_dir)
        checksum_path = export_dir / "checksums.sha256"
        checksum_text = checksum_path.read_text(encoding="utf-8")

        extra_path = export_dir / "unexpected_extra.txt"
        extra_path.write_text("tamper", encoding="utf-8")
        assert not verify_checksums(export_dir)
        extra_path.unlink()

        summary_path = export_dir / "audit_export_summary.md"
        summary_text = summary_path.read_text(encoding="utf-8")
        summary_path.unlink()
        assert not verify_checksums(export_dir)
        summary_path.write_text(summary_text, encoding="utf-8")

        checksum_path.write_text(checksum_text + "malformed checksum row\n", encoding="utf-8")
        assert not verify_checksums(export_dir)
        checksum_path.write_text(checksum_text + f"{'0' * 64}  ../escape.txt\n", encoding="utf-8")
        assert not verify_checksums(export_dir)
        checksum_path.write_text(checksum_text, encoding="utf-8")
        assert verify_checksums(export_dir)

        missing_result = run_audit_export(runs_dir=runs_dir, run_id=run_id, provider_id="provider_a", gate_id="missing_gate")
        assert missing_result["manifest"]["warnings"]

        latest_result = run_audit_export(runs_dir=runs_dir, run_id=run_id, provider_id="provider_a")
        assert latest_result["manifest"]["bound_evidence"]["gate_id"] == "gate_fixture"
        assert latest_result["manifest"]["expected_evidence"]["compatibility_run_ids"] == ["compat_fixture"]
        assert latest_result["manifest"]["bound_evidence"]["trace_eval_ids"] == ["trace_fixture"]
        assert latest_result["manifest"]["bound_evidence"]["rescore_ids"] == ["rescore_fixture"]

        ids_only_gate_dir = runs_dir / run_id / "quality_gates" / "gate_ids_only"
        ids_only_gate_dir.mkdir(parents=True, exist_ok=True)
        ids_only_record = dict(read_jsonl(runs_dir / run_id / "quality_gates" / "gate_fixture" / "quality_gate_records.jsonl")[0])
        ids_only_record["gate_id"] = "gate_ids_only"
        ids_only_record["evidence_refs"] = {}
        ids_only_record["evidence_ids"] = {
            "compatibility_run_id": "compat_fixture",
            "trace_eval_id": "trace_fixture",
            "rescore_id": "rescore_fixture",
        }
        write_jsonl(ids_only_gate_dir / "quality_gate_records.jsonl", [ids_only_record])
        write_json(ids_only_gate_dir / "quality_gate_manifest.json", {"gate_id": "gate_ids_only", "source_run_id": run_id, "provider_ids": ["provider_a"], "decision_counts": {"GO": 1}})
        ids_only_result = run_audit_export(runs_dir=runs_dir, run_id=run_id, provider_id="provider_a", gate_id="gate_ids_only")
        assert ids_only_result["manifest"]["expected_evidence"]["compatibility_run_ids"] == ["compat_fixture"]
        assert ids_only_result["manifest"]["bound_evidence"]["trace_eval_ids"] == ["trace_fixture"]
        assert ids_only_result["manifest"]["bound_evidence"]["rescore_ids"] == ["rescore_fixture"]

        broken_gate_dir = runs_dir / run_id / "quality_gates" / "gate_broken_refs"
        broken_gate_dir.mkdir(parents=True, exist_ok=True)
        broken_record = dict(read_jsonl(runs_dir / run_id / "quality_gates" / "gate_fixture" / "quality_gate_records.jsonl")[0])
        broken_record["gate_id"] = "gate_broken_refs"
        broken_record["evidence_refs"] = {"rescore_manifest_file": str(runs_dir / run_id / "rescores" / "missing_rescore" / "rescore_manifest.json")}
        write_jsonl(broken_gate_dir / "quality_gate_records.jsonl", [broken_record])
        write_json(broken_gate_dir / "quality_gate_manifest.json", {"gate_id": "gate_broken_refs", "source_run_id": run_id, "provider_ids": ["provider_a"], "decision_counts": {"GO": 1}})
        with (broken_gate_dir / "quality_gate_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["gate_id", "provider_id", "decision"])
            writer.writeheader()
            writer.writerow({"gate_id": "gate_broken_refs", "provider_id": "provider_a", "decision": "GO"})
        broken_result = run_audit_export(runs_dir=runs_dir, run_id=run_id, provider_id="provider_a", gate_id="gate_broken_refs")
        assert "missing_rescore" in broken_result["manifest"]["expected_evidence"]["rescore_ids"]
        assert broken_result["manifest"]["bound_evidence"]["rescore_ids"] == []
        assert any("could not be resolved" in warning for warning in broken_result["manifest"]["warnings"])

        stopped_events: list[dict[str, Any]] = []
        stopped_result = run_audit_export(
            runs_dir=runs_dir,
            run_id=run_id,
            provider_id="provider_a",
            job_control={"stop_requested": True},
            progress_callback=stopped_events.append,
        )
        stopped_export_dir = runs_dir / run_id / "audit_exports" / stopped_result["audit_export_id"]
        assert stopped_result["stopped"] is True
        assert stopped_result["manifest"]["status"] == "stopped"
        assert stopped_result["manifest"]["checksums_file"] is None
        assert not (stopped_export_dir / "checksums.sha256").exists()
        assert any(event.get("event") == "run_stopped" for event in stopped_events)

    print("audit export self-test ok")


def main() -> int:
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="Export a redacted offline audit evidence package")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--provider-id")
    parser.add_argument("--gate-id")
    parser.add_argument("--compatibility-run-id")
    parser.add_argument("--rescore-id")
    parser.add_argument("--trace-eval-id")
    parser.add_argument("--audit-label")
    parser.add_argument("--redaction-mode", default=REDACTION_MODE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.run_id:
        parser.error("--run-id is required unless --self-test is used")
    result = run_audit_export(
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        provider_id=args.provider_id,
        gate_id=args.gate_id,
        compatibility_run_id=args.compatibility_run_id,
        rescore_id=args.rescore_id,
        trace_eval_id=args.trace_eval_id,
        audit_label=args.audit_label,
        redaction_mode=args.redaction_mode,
    )
    print(json.dumps({"audit_export_id": result["audit_export_id"], "manifest": result["manifest"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
