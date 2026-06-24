from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any


CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{64})  (.+)$")


def verify_acceptance_pack(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "verified": False,
        "error": None,
        "missing": [],
        "mismatches": [],
        "extra_entries": [],
        "entry_count": 0,
    }
    if not path.exists() or not path.is_file():
        result["error"] = "pack_missing"
        return result
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            result["entry_count"] = len(names)
            required = {"acceptance_manifest.json", "checksums.sha256"}
            missing = sorted(required - names)
            if missing:
                result["missing"] = missing
                result["error"] = "missing_manifest_or_checksums"
                return result
            expected: dict[str, str] = {}
            for line in zf.read("checksums.sha256").decode("utf-8").splitlines():
                match = CHECKSUM_LINE.match(line.strip())
                if match:
                    expected[match.group(2)] = match.group(1).lower()
            mismatches = []
            for name, digest in expected.items():
                if name not in names:
                    mismatches.append(name)
                    continue
                if hashlib.sha256(zf.read(name)).hexdigest() != digest:
                    mismatches.append(name)
            extra = sorted(names - {"checksums.sha256"} - set(expected))
            if mismatches or extra:
                result["mismatches"] = mismatches
                result["extra_entries"] = extra
                result["error"] = "checksum_mismatch"
                return result
    except zipfile.BadZipFile:
        result["error"] = "invalid_zip"
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["verified"] = True
    return result
