from __future__ import annotations

import os
import re
from pathlib import Path


LOCAL_ENV_FILE = Path(__file__).resolve().parent / "local_secrets.env"
ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    key, sep, value = line.partition("=")
    if not sep:
        return None
    key = key.strip()
    if not ENV_KEY.match(key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_local_env(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load local KEY=VALUE pairs into this process without overwriting shell env by default."""
    env_path = path or LOCAL_ENV_FILE
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if value == "":
            continue
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def env_override(raw: dict, field: str) -> str:
    env_name = str(raw.get(f"{field}_env") or "")
    if env_name:
        value = os.environ.get(env_name)
        if value:
            return value
    return str(raw[field])
