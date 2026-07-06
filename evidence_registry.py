from __future__ import annotations

import argparse
import json
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


SAFE_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ManifestSelection:
    path: Path | None
    manifest: dict[str, Any] | None
    expected_id: str | None
    bound_id: str | None
    error: str | None = None


@dataclass(frozen=True)
class BoundEvidenceDirs:
    dirs: list[Path]
    warnings: list[str]
    expected_ids: list[str]
    bound_ids: list[str]


def safe_evidence_id(value: Any, label: str = "evidence id") -> str:
    text = str(value or "").strip()
    if not text or not SAFE_EVIDENCE_ID.fullmatch(text):
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def unique_artifact_id(prefix: str) -> str:
    safe_prefix = safe_evidence_id(prefix, "artifact prefix")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{safe_prefix}_{stamp}_{secrets.token_hex(3)}"


def read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def manifest_id(manifest: dict[str, Any] | None, id_key: str | None, path: Path) -> str:
    if manifest and id_key and manifest.get(id_key):
        return str(manifest.get(id_key))
    return path.name


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(resolve_non_strict(path))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def dedupe_strings(values: Iterable[str | None]) -> list[str]:
    return sorted({str(value) for value in values if value})


def resolve_non_strict(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def path_is_within(path: Path, root: Path) -> bool:
    resolved_path = resolve_non_strict(path)
    resolved_root = resolve_non_strict(root)
    try:
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def path_is_within_any(path: Path, roots: Sequence[Path]) -> bool:
    return not roots or any(path_is_within(path, root) for root in roots)


def resolve_evidence_ref_path(
    value: Any,
    *,
    search_roots: Sequence[Path],
    allowed_roots: Sequence[Path] = (),
) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [root / raw for root in search_roots]
    for candidate in dedupe_paths(candidates):
        if candidate.exists() and path_is_within_any(candidate, allowed_roots):
            return candidate
    return None


def evidence_id_from_ref(value: Any, resolved_path: Path | None = None) -> str | None:
    if value in (None, "") and resolved_path is None:
        return None
    path = resolved_path or Path(str(value))
    if path.exists() and path.is_dir():
        return path.name or None
    if path.parent.name:
        return path.parent.name
    return path.stem or None


def select_manifest_dir(
    *,
    base_dir: Path,
    manifest_name: str,
    explicit_id: str | None = None,
    label: str = "evidence",
    id_key: str | None = None,
    match_manifest: Callable[[dict[str, Any]], bool] | None = None,
    mismatch_error: str | None = None,
) -> ManifestSelection:
    if explicit_id:
        try:
            wanted = safe_evidence_id(explicit_id, f"{label} id")
        except ValueError as exc:
            return ManifestSelection(None, None, None, None, str(exc))
        child = base_dir / wanted
        manifest_path = child / manifest_name
        if not manifest_path.exists():
            return ManifestSelection(None, None, wanted, None, f"{label} not found")
        try:
            manifest = read_manifest(manifest_path)
        except Exception as exc:
            return ManifestSelection(None, None, wanted, None, f"{label} manifest unreadable: {type(exc).__name__}: {exc}")
        if match_manifest and not match_manifest(manifest):
            return ManifestSelection(None, manifest, wanted, manifest_id(manifest, id_key, child), mismatch_error or f"{label} manifest mismatch")
        return ManifestSelection(child, manifest, wanted, manifest_id(manifest, id_key, child))

    if not base_dir.exists():
        return ManifestSelection(None, None, None, None, f"{label} directory not found")

    for child in sorted((p for p in base_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        manifest_path = child / manifest_name
        if not manifest_path.exists():
            continue
        try:
            manifest = read_manifest(manifest_path)
        except Exception:
            continue
        if match_manifest and not match_manifest(manifest):
            continue
        return ManifestSelection(child, manifest, None, manifest_id(manifest, id_key, child))
    return ManifestSelection(None, None, None, None, f"matching {label} evidence not found")


def collect_bound_evidence_dirs(
    *,
    records: list[dict[str, Any]],
    ref_name: str,
    id_key: str,
    base_dir: Path,
    manifest_name: str,
    manifest_id_key: str | None,
    evidence_kind: str,
    search_roots: Sequence[Path],
    allowed_roots: Sequence[Path],
    match_manifest: Callable[[dict[str, Any]], bool] | None = None,
) -> BoundEvidenceDirs:
    warnings: list[str] = []
    dirs: list[Path] = []
    expected_ids: list[str] = []
    bound_ids: list[str] = []
    if not records:
        warnings.append(f"quality gate records unavailable for bound {evidence_kind} evidence")
        return BoundEvidenceDirs(dirs, warnings, expected_ids, bound_ids)

    for record in records:
        raw_refs = record.get("evidence_refs")
        refs: dict[str, Any] = raw_refs if isinstance(raw_refs, dict) else {}
        raw_ev_ids = record.get("evidence_ids")
        evidence_ids: dict[str, Any] = raw_ev_ids if isinstance(raw_ev_ids, dict) else {}

        expected_id: str | None = None
        raw_expected_id = evidence_ids.get(id_key)
        if raw_expected_id:
            try:
                expected_id = safe_evidence_id(raw_expected_id, f"{evidence_kind} id")
                expected_ids.append(expected_id)
            except ValueError as exc:
                warnings.append(str(exc))

        ref_value = refs.get(ref_name)
        if ref_value:
            ref_id = evidence_id_from_ref(ref_value)
            if ref_id and not expected_id:
                expected_id = ref_id
                expected_ids.append(ref_id)
            elif ref_id and expected_id and ref_id != expected_id:
                warnings.append(f"quality gate bound {evidence_kind} id/ref mismatch: {expected_id} != {ref_id}")
                expected_ids.append(ref_id)

            resolved = resolve_evidence_ref_path(ref_value, search_roots=search_roots, allowed_roots=allowed_roots)
            if not resolved:
                warnings.append(f"quality gate bound {evidence_kind} ref could not be resolved: {ref_value}")
                continue
            evidence_dir = resolved if resolved.is_dir() else resolved.parent
            manifest_path = evidence_dir / manifest_name
            if not manifest_path.exists():
                warnings.append(f"quality gate bound {evidence_kind} ref missing manifest: {ref_value}")
                continue
            try:
                manifest = read_manifest(manifest_path)
            except Exception as exc:
                warnings.append(f"quality gate bound {evidence_kind} manifest unreadable: {type(exc).__name__}: {exc}")
                continue
            if match_manifest and not match_manifest(manifest):
                warnings.append(f"quality gate bound {evidence_kind} ref did not match requested provider/run: {ref_value}")
                continue
            dirs.append(evidence_dir)
            bound_ids.append(manifest_id(manifest, manifest_id_key, evidence_dir))
            continue

        if expected_id:
            selection = select_manifest_dir(
                base_dir=base_dir,
                manifest_name=manifest_name,
                explicit_id=expected_id,
                label=evidence_kind,
                id_key=manifest_id_key,
                match_manifest=match_manifest,
            )
            if selection.path:
                dirs.append(selection.path)
                bound_ids.append(selection.bound_id or expected_id)
            else:
                warnings.append(f"quality gate bound {evidence_kind} id could not be resolved: {expected_id}")

    if not expected_ids:
        warnings.append(f"quality gate did not bind {evidence_kind} evidence; not falling back to latest")
    return BoundEvidenceDirs(
        dedupe_paths(dirs),
        warnings,
        dedupe_strings(expected_ids),
        dedupe_strings(bound_ids),
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs_dir = root / "runs"
        run_dir = runs_dir / "run_a"
        compat_dir = runs_dir / "compat_a"
        compat_dir.mkdir(parents=True)
        manifest_path = compat_dir / "compatibility_manifest.json"
        manifest_path.write_text(json.dumps({"run_id": "compat_a", "provider_id": "provider_a"}), encoding="utf-8")

        selection = select_manifest_dir(
            base_dir=runs_dir,
            manifest_name="compatibility_manifest.json",
            explicit_id="compat_a",
            label="compatibility run",
            id_key="run_id",
            match_manifest=lambda manifest: manifest.get("provider_id") == "provider_a",
        )
        assert selection.path == compat_dir
        assert selection.bound_id == "compat_a"
        assert select_manifest_dir(base_dir=runs_dir, manifest_name="compatibility_manifest.json", explicit_id="../bad").error

        gate_dir = run_dir / "quality_gates" / "gate_a"
        gate_dir.mkdir(parents=True)
        records = [
            {
                "evidence_ids": {"compatibility_run_id": "compat_a"},
                "evidence_refs": {"compatibility_manifest_file": str(manifest_path)},
            }
        ]
        bound = collect_bound_evidence_dirs(
            records=records,
            ref_name="compatibility_manifest_file",
            id_key="compatibility_run_id",
            base_dir=runs_dir,
            manifest_name="compatibility_manifest.json",
            manifest_id_key="run_id",
            evidence_kind="compatibility",
            search_roots=[root, runs_dir, run_dir, gate_dir],
            allowed_roots=[runs_dir],
            match_manifest=lambda manifest: manifest.get("provider_id") == "provider_a",
        )
        assert bound.dirs == [compat_dir]
        assert bound.expected_ids == ["compat_a"]
        assert bound.bound_ids == ["compat_a"]
        assert unique_artifact_id("gate").startswith("gate_")

    print("evidence registry self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence registry helper checks")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
