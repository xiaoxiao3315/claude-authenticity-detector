"""Shared low-level IO + path helpers for the eval toolchain.

Extracted from eval_cli.py so the CLI module stays focused on command logic.
These are pure leaf helpers (JSON/JSONL read-write, path resolution, small id
helpers) with no dependency on the rest of the CLI — safe to import anywhere.

ROOT resolves to the project root (this file lives at the repo top level, same
as eval_cli.py), so resolve_path's default base is unchanged from before.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def utcish_job_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # tolerate a corrupt/partial line rather than crashing the caller
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def resolve_path(path_value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def resolve_job(job: str | Path) -> Path:
    raw = Path(job)
    candidates = []
    if raw.suffix:
        candidates.append(raw)
    else:
        candidates.append(Path("configs/jobs") / f"{raw}.json")
        candidates.append(Path("configs/jobs") / str(raw))
    for candidate in candidates:
        path = resolve_path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"job config not found: {job}")


def base_url_host(base_url: str) -> str | None:
    parsed = urlparse(base_url)
    return parsed.netloc or parsed.path or None


def _self_test() -> int:
    """Offline checks for the IO/path helpers."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "sub" / "x.json"
        write_json(p, {"a": 1, "中文": "保留"})
        assert read_json(p) == {"a": 1, "中文": "保留"}
        jl = Path(tmp) / "y.jsonl"
        write_jsonl(jl, [{"i": 1}, {"i": 2}])
        append_jsonl(jl, {"i": 3})
        assert read_jsonl(jl) == [{"i": 1}, {"i": 2}, {"i": 3}]
        assert read_jsonl(Path(tmp) / "missing.jsonl") == []
    assert resolve_path("/abs/x").is_absolute()
    assert resolve_path("rel/x") == ROOT / "rel" / "x"
    assert base_url_host("https://api.example.com/v1") == "api.example.com"
    assert utcish_job_id("CMP").startswith("CMP-")
    assert "T" in now_iso()
    print("cli_io self-test ok")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_self_test())
