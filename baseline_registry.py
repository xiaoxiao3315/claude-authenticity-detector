"""Claude official-authenticity baseline registry.

Builds and stores a "golden" fingerprint baseline from a TRUSTED official Claude
source (situation A: 小小 vouches for drhknode.airouting.com using official
Claude). A baseline is the standard answer that suspect providers are later
compared against, to detect model swapping / downgrade / wrapper / fake-1M.

This module is framework-only for now: it can build/read/compare baselines and
self-test fully with dry-run sample data. Live collection (real API calls) is
driven from eval_cli.py and gated behind an explicit --live flag.

Design notes:
- Samples carry the RAW observed protocol values (raw usage key names, raw
  stop_reason) taken from the upstream response BEFORE eval_cli.call_model's
  L639/L640 fallback would rewrite them. Never normalize here.
- Reuses authenticity.py json/stat helpers; verdict mirrors the 4-class scheme.
- Secrets are never stored; only base_url_host + a salted key fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from authenticity import numeric, percentile, read_json, write_json

CLAUDE_BASELINE_SCHEMA_VERSION = "claude_baseline_v1"
BASELINE_COMPARISON_RESULT_VERSION = "baseline_compare_v1"

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINES_DIR = ROOT / "baselines"

# Anthropic / Claude official protocol expectations (the "standard answer").
CLAUDE_STOP_REASONS = {"end_turn", "max_tokens", "stop_sequence", "tool_use"}
OPENAI_FINISH_REASONS = {"stop", "length", "content_filter", "tool_calls", "function_call"}
ANTHROPIC_USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
}
OPENAI_USAGE_KEYS = {"prompt_tokens", "completion_tokens", "total_tokens"}

# Verdict labels (mirrors the authenticity 4-class scheme).
VERDICT_MATCHES = "matches_official"
VERDICT_DOWNGRADE = "suspected_downgrade"
VERDICT_WRAPPER = "suspected_wrapper"
VERDICT_INSUFFICIENT = "insufficient_evidence"


def key_fingerprint(secret: str | None, *, salt: str = "claude-baseline-v1") -> str | None:
    """Salted, truncated hash of a key. Never store the raw key."""
    if not secret:
        return None
    digest = hashlib.sha256(f"{salt}:{secret}".encode("utf-8")).hexdigest()
    return digest[:12]


def usage_naming_dialect(usage_keys: set[str] | list[str]) -> str:
    keys = set(usage_keys)
    has_anthropic = bool(keys & ANTHROPIC_USAGE_KEYS)
    has_openai = bool(keys & OPENAI_USAGE_KEYS)
    if has_anthropic and has_openai:
        return "mixed"
    if has_anthropic:
        return "anthropic"
    if has_openai:
        return "openai"
    return "unknown"


def make_sample(
    *,
    protocol: str | None,
    raw_stop_reason: Any,
    raw_usage_keys: list[str] | None,
    input_tokens: Any,
    output_tokens: Any,
    total_ms: Any,
    has_anthropic_request_id: bool = False,
    has_anthropic_headers: bool = False,
    probe_id: str | None = None,
    expected_input_tokens: Any = None,
    live: bool = False,
    ok: bool = True,
) -> dict[str, Any]:
    """Build one collection sample from RAW observed values.

    raw_stop_reason / raw_usage_keys must come from the upstream response as-is,
    not from CallMetrics (which applies a "stop" / model fallback).
    ok=False marks a failed request (HTTP error / timeout) — distinguishes
    "request failed" from "succeeded but field missing".
    """
    return {
        "protocol": protocol,
        "ok": bool(ok),
        "raw_stop_reason": None if raw_stop_reason is None else str(raw_stop_reason),
        "raw_usage_keys": sorted(raw_usage_keys or []),
        "usage_naming_dialect": usage_naming_dialect(raw_usage_keys or []),
        "input_tokens": numeric(input_tokens),
        "output_tokens": numeric(output_tokens),
        "total_ms": numeric(total_ms),
        "has_anthropic_request_id": bool(has_anthropic_request_id),
        "has_anthropic_headers": bool(has_anthropic_headers),
        "probe_id": probe_id,
        "expected_input_tokens": numeric(expected_input_tokens),
        "live": bool(live),
    }


def _rate(count: int, total: int) -> float | None:
    if not total:
        return None
    return round(count / total, 6)


def _distribution(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"count": 0}
    dist: dict[str, Any] = {
        "count": len(clean),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "mean": round(statistics.fmean(clean), 3),
        "p50": percentile(clean, 0.5),
        "p95": percentile(clean, 0.95),
    }
    if len(clean) >= 2:
        dist["stdev"] = round(statistics.stdev(clean), 3)
    return dist


def build_baseline_from_samples(
    samples: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    baseline_id: str,
    live: bool = False,
    collected_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate N collection samples into a baseline fingerprint document."""
    total = len(samples)
    stop_counter: Counter[str] = Counter()
    usage_dialects: Counter[str] = Counter()
    anthropic_req_id = 0
    anthropic_headers = 0
    failed_requests = 0
    input_token_values: list[float] = []
    latency_values: list[float] = []
    # per-probe input_tokens, for tokenizer expected windows
    probe_tokens: dict[str, list[float]] = {}

    for sample in samples:
        if sample.get("ok") is False:
            failed_requests += 1
        sr = sample.get("raw_stop_reason")
        if sr is not None:
            stop_counter[str(sr)] += 1
        usage_dialects[str(sample.get("usage_naming_dialect") or "unknown")] += 1
        if sample.get("has_anthropic_request_id"):
            anthropic_req_id += 1
        if sample.get("has_anthropic_headers"):
            anthropic_headers += 1
        it = numeric(sample.get("input_tokens"))
        if it is not None:
            input_token_values.append(it)
        lat = numeric(sample.get("total_ms"))
        if lat is not None:
            latency_values.append(lat)
        probe_id = sample.get("probe_id")
        if probe_id and it is not None:
            probe_tokens.setdefault(str(probe_id), []).append(it)

    stop_reason_in_claude_enum = all(
        value in CLAUDE_STOP_REASONS for value in stop_counter
    ) if stop_counter else None

    fingerprint = {
        "schema_version": CLAUDE_BASELINE_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "evidence_status": "live_observed" if live else "dry_run_reference_only",
        "sample_count": total,
        "failed_request_count": failed_requests,
        "request_failure_rate": _rate(failed_requests, total),
        "collected_window": collected_window or {},
        "source": {
            "provider_id": source.get("provider_id"),
            "provider_label": source.get("provider_label"),
            "base_url_host": source.get("base_url_host"),
            "model": source.get("model"),
            "protocol": source.get("protocol"),
            "channel": "trusted_official",
            "key_fingerprint": source.get("key_fingerprint"),
        },
        "protocol_fingerprint": {
            "stop_reason_counts": dict(sorted(stop_counter.items())),
            "stop_reason_in_claude_enum": stop_reason_in_claude_enum,
            "usage_naming_dialect_counts": dict(sorted(usage_dialects.items())),
            "anthropic_request_id_rate": _rate(anthropic_req_id, total),
            "anthropic_headers_rate": _rate(anthropic_headers, total),
        },
        "behavior": {
            "input_tokens_distribution": _distribution(input_token_values),
            "latency_ms_distribution": _distribution(latency_values),
            "tokenizer_probe_windows": {
                probe_id: _distribution(values)
                for probe_id, values in sorted(probe_tokens.items())
            },
            # NOTE: capability-anchor pass-rate is NOT stored here. It lives in a
            # separate sidecar file `baselines/<id>/capability_anchor.json` and is
            # compared independently (see eval_cli.py: base_cap_path / score_capability_vs_baseline).
            # This key is kept as an always-None placeholder ONLY so the content
            # signature (content_fingerprint hashes the whole behavior dict) stays
            # stable for existing baselines — do not populate it; read the sidecar instead.
            "capability_anchor_pass_rate": None,
        },
    }
    return fingerprint


def baseline_dir(baselines_dir: Path, baseline_id: str) -> Path:
    safe = str(baseline_id).replace("/", "_").replace("\\", "_")
    return baselines_dir / safe


def write_baseline(baselines_dir: Path, baseline_id: str, doc: dict[str, Any]) -> Path:
    out_dir = baseline_dir(baselines_dir, baseline_id)
    path = out_dir / "baseline.json"
    write_json(path, doc)
    return path


def load_baseline(baselines_dir: Path, baseline_id: str) -> dict[str, Any] | None:
    path = baseline_dir(baselines_dir, baseline_id) / "baseline.json"
    if not path.exists():
        return None
    return read_json(path)


def list_baselines(baselines_dir: Path) -> list[str]:
    if not baselines_dir.exists():
        return []
    out: list[str] = []
    for child in sorted(baselines_dir.iterdir()):
        if child.is_dir() and (child / "baseline.json").exists():
            out.append(child.name)
    return out


# --------------------------------------------------------------------------
# Versioning: keep a lineage of immutable historical snapshots per baseline_id.
#
# Layout (additive — never breaks existing single-file consumers):
#   baselines/<id>/baseline.json            <- LATEST pointer (unchanged shape)
#   baselines/<id>/versions.json            <- lineage manifest (sidecar)
#   baselines/<id>/versions/<vNNNN>/baseline.json   <- immutable snapshot
#
# load_baseline / list_baselines keep reading baseline.json exactly as before,
# so all current callers and self-tests are untouched.
# --------------------------------------------------------------------------

BASELINE_VERSIONS_SCHEMA_VERSION = "claude_baseline_versions_v1"

# Fields that legitimately vary between rebuilds of the "same" baseline and so
# must be excluded from the content fingerprint used for dedup. Everything else
# (protocol_fingerprint, behavior) is genuine signal whose change = a new version.
_CONTENT_HASH_EXCLUDE = {
    "baseline_id",
    "evidence_status",
    "sample_count",
    "failed_request_count",
    "request_failure_rate",
    "collected_window",
}


def content_fingerprint(doc: dict[str, Any]) -> str:
    """Stable hash of a baseline's SIGNAL content (protocol + behavior + source
    identity), excluding volatile metadata like sample_count. Two rebuilds that
    observed the same fingerprint hash equal → no new version is created."""
    core = {k: v for k, v in doc.items() if k not in _CONTENT_HASH_EXCLUDE}
    blob = json.dumps(core, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def versions_manifest_path(baselines_dir: Path, baseline_id: str) -> Path:
    return baseline_dir(baselines_dir, baseline_id) / "versions.json"


def load_versions_manifest(baselines_dir: Path, baseline_id: str) -> dict[str, Any] | None:
    path = versions_manifest_path(baselines_dir, baseline_id)
    if not path.exists():
        return None
    return read_json(path)


def list_baseline_versions(baselines_dir: Path, baseline_id: str) -> list[dict[str, Any]]:
    """Return the lineage entries (oldest→newest) for a baseline_id, or []."""
    manifest = load_versions_manifest(baselines_dir, baseline_id)
    if not manifest:
        return []
    return list(manifest.get("versions") or [])


def load_baseline_version(
    baselines_dir: Path, baseline_id: str, version: str
) -> dict[str, Any] | None:
    """Load one immutable historical snapshot by its version label (e.g. v0003)."""
    safe = str(version).replace("/", "_").replace("\\", "_")
    path = baseline_dir(baselines_dir, baseline_id) / "versions" / safe / "baseline.json"
    if not path.exists():
        return None
    return read_json(path)


def write_baseline_version(
    baselines_dir: Path,
    baseline_id: str,
    doc: dict[str, Any],
    *,
    now: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Commit a baseline as a new immutable version in its lineage.

    - Updates baseline.json (the LATEST pointer) — same shape as write_baseline.
    - If the content fingerprint equals the current latest version, NO new
      snapshot is created (returns dedup=True); the pointer is still refreshed.
    - Otherwise archives an immutable snapshot under versions/<vNNNN>/ and
      appends an entry (with parent + drift summary) to versions.json.

    `now` is an ISO-ish timestamp string injected by the caller (this module
    never reads the clock, to stay deterministic / resume-safe).
    """
    manifest = load_versions_manifest(baselines_dir, baseline_id) or {
        "schema_version": BASELINE_VERSIONS_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "versions": [],
    }
    versions: list[dict[str, Any]] = list(manifest.get("versions") or [])
    new_hash = content_fingerprint(doc)
    prev = versions[-1] if versions else None

    # Always refresh the latest pointer so existing consumers see fresh metadata.
    pointer_path = write_baseline(baselines_dir, baseline_id, doc)

    # Dedup against the ENTIRE lineage, not just the latest — a fingerprint that
    # regresses to an already-seen value re-observes that version rather than
    # forging a spurious new one.
    match = next((v for v in versions if v.get("content_hash") == new_hash), None)
    if match is not None:
        match["last_seen"] = now
        match["observed_count"] = int(match.get("observed_count") or 1) + 1
        regressed = match is not prev  # came back to an older version, not the tip
        manifest["versions"] = versions
        manifest["updated_at"] = now
        write_json(versions_manifest_path(baselines_dir, baseline_id), manifest)
        return {
            "version": match.get("version"),
            "dedup": True,
            "regressed": regressed,
            "content_hash": new_hash,
            "pointer_path": str(pointer_path),
            "drift": None,
        }

    seq = len(versions) + 1
    version_label = f"v{seq:04d}"
    snap_rel = f"versions/{version_label}/baseline.json"
    snap_path = baseline_dir(baselines_dir, baseline_id) / "versions" / version_label / "baseline.json"
    write_json(snap_path, doc)

    drift = diff_baselines(load_baseline_version(baselines_dir, baseline_id, prev["version"]), doc) if prev else None

    entry = {
        "version": version_label,
        "created_at": now,
        "last_seen": now,
        "observed_count": 1,
        "content_hash": new_hash,
        "parent": prev.get("version") if prev else None,
        "evidence_status": doc.get("evidence_status"),
        "sample_count": doc.get("sample_count"),
        "model": (doc.get("source") or {}).get("model"),
        "base_url_host": (doc.get("source") or {}).get("base_url_host"),
        "note": note,
        "drift_from_parent": drift,
        "snapshot_path": snap_rel,  # relative to the baseline dir — portable
    }
    versions.append(entry)
    manifest["versions"] = versions
    manifest["updated_at"] = now
    write_json(versions_manifest_path(baselines_dir, baseline_id), manifest)
    return {
        "version": version_label,
        "dedup": False,
        "content_hash": new_hash,
        "pointer_path": str(pointer_path),
        "drift": drift,
    }


def _band_shift(old_dist: dict[str, Any] | None, new_dist: dict[str, Any] | None) -> dict[str, Any] | None:
    """Relative mean shift between two _distribution() dicts, if both present."""
    if not old_dist or not new_dist:
        return None
    om, nm = numeric(old_dist.get("mean")), numeric(new_dist.get("mean"))
    if om is None or nm is None:
        return None
    abs_delta = round(nm - om, 3)
    rel = round(abs_delta / om, 4) if om else None
    return {"old_mean": om, "new_mean": nm, "abs_delta": abs_delta, "rel_delta": rel}


def diff_baselines(old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drift summary between two baseline docs (old→new). Surfaces protocol-layer
    changes (stop_reason enum / usage dialect / model id) and behavior-layer
    distribution shifts (input_tokens, latency, per-probe tokenizer windows).

    `changed` is True when any protocol field flipped or a tracked mean moved
    by more than `TOKEN_DRIFT_REL` (tokens/probes) / `LATENCY_DRIFT_REL` (latency).
    Pure function — used both for lineage records and the diff CLI."""
    if old is None or new is None:
        return None
    TOKEN_DRIFT_REL = 0.02   # >2% mean token shift is worth flagging
    LATENCY_DRIFT_REL = 0.50  # latency is noisy; only flag big (>50%) moves

    old_proto = old.get("protocol_fingerprint") or {}
    new_proto = new.get("protocol_fingerprint") or {}
    old_beh = old.get("behavior") or {}
    new_beh = new.get("behavior") or {}
    old_src = old.get("source") or {}
    new_src = new.get("source") or {}

    protocol_changes: dict[str, Any] = {}
    for field in ("stop_reason_in_claude_enum",):
        if old_proto.get(field) != new_proto.get(field):
            protocol_changes[field] = {"old": old_proto.get(field), "new": new_proto.get(field)}
    for field in ("stop_reason_counts", "usage_naming_dialect_counts"):
        if old_proto.get(field) != new_proto.get(field):
            protocol_changes[field] = {"old": old_proto.get(field), "new": new_proto.get(field)}
    for field in ("model", "base_url_host", "protocol"):
        if old_src.get(field) != new_src.get(field):
            protocol_changes[f"source.{field}"] = {"old": old_src.get(field), "new": new_src.get(field)}

    behavior_changes: dict[str, Any] = {}
    it_shift = _band_shift(old_beh.get("input_tokens_distribution"), new_beh.get("input_tokens_distribution"))
    if it_shift and it_shift.get("rel_delta") is not None and abs(it_shift["rel_delta"]) > TOKEN_DRIFT_REL:
        behavior_changes["input_tokens"] = it_shift
    lat_shift = _band_shift(old_beh.get("latency_ms_distribution"), new_beh.get("latency_ms_distribution"))
    if lat_shift and lat_shift.get("rel_delta") is not None and abs(lat_shift["rel_delta"]) > LATENCY_DRIFT_REL:
        behavior_changes["latency_ms"] = lat_shift

    old_probes = old_beh.get("tokenizer_probe_windows") or {}
    new_probes = new_beh.get("tokenizer_probe_windows") or {}
    probe_changes: dict[str, Any] = {}
    for probe_id in sorted(set(old_probes) | set(new_probes)):
        if probe_id not in old_probes:
            probe_changes[probe_id] = {"status": "added"}
            continue
        if probe_id not in new_probes:
            probe_changes[probe_id] = {"status": "removed"}
            continue
        shift = _band_shift(old_probes[probe_id], new_probes[probe_id])
        if shift and shift.get("rel_delta") is not None and abs(shift["rel_delta"]) > TOKEN_DRIFT_REL:
            probe_changes[probe_id] = shift
    if probe_changes:
        behavior_changes["tokenizer_probe_windows"] = probe_changes

    changed = bool(protocol_changes or behavior_changes)
    return {
        "changed": changed,
        "protocol_changes": protocol_changes,
        "behavior_changes": behavior_changes,
    }


def compare_to_baseline(
    observed: dict[str, Any],
    baseline: dict[str, Any],
    *,
    behavior_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a suspect endpoint's observed fingerprint against the baseline.

    Protocol layer always runs. Optional behavior_signals (tokenizer / sse /
    error_envelope, gathered live by verify_endpoint) are folded in: matching
    behavior raises confidence; a behavior mismatch hard-fails to wrapper.
    Returns a verdict with confidence + reasons + evidence_chain.
    """
    reasons: list[str] = []
    evidence: list[dict[str, Any]] = []

    base_status = baseline.get("evidence_status")
    obs_status = observed.get("evidence_status")

    # If either side is dry-run reference only, we cannot conclude authenticity.
    if base_status != "live_observed" or obs_status != "live_observed":
        return {
            "schema_version": BASELINE_COMPARISON_RESULT_VERSION,
            "verdict": VERDICT_INSUFFICIENT,
            "confidence": 0.0,
            "reasons": ["baseline_or_observed_not_live"],
            "evidence_chain": [
                {"check": "evidence_status", "baseline": base_status, "observed": obs_status}
            ],
        }

    base_fp = baseline.get("protocol_fingerprint") or {}
    obs_fp = observed.get("protocol_fingerprint") or {}

    # If most observed requests FAILED (HTTP error / 429 / timeout), we cannot
    # fingerprint the endpoint at all — that is "insufficient: requests failed",
    # NOT a suspicious empty fingerprint. Distinguishes a rate-limited/broken key
    # from a real wrapper that returns empty fields on successful calls.
    obs_fail_rate = numeric(observed.get("request_failure_rate"))
    obs_success = obs_fp.get("stop_reason_counts") or {}
    if (obs_fail_rate is not None and obs_fail_rate >= 0.5) or (
        not obs_success and (observed.get("failed_request_count") or 0) > 0
    ):
        return {
            "schema_version": BASELINE_COMPARISON_RESULT_VERSION,
            "baseline_id": baseline.get("baseline_id"),
            "verdict": VERDICT_INSUFFICIENT,
            "confidence": 0.0,
            "reasons": ["observed_requests_failed", f"failure_rate:{obs_fail_rate}"],
            "evidence_chain": [
                {"check": "request_failure_rate", "baseline": baseline.get("request_failure_rate"), "observed": obs_fail_rate},
                {"check": "failed_request_count", "observed": observed.get("failed_request_count")},
            ],
            "note": "endpoint mostly returned errors (e.g. 429 rate-limit/quota); cannot fingerprint — retry with a healthy key",
        }

    hard_fail = False

    # 1. stop_reason must stay in Claude enum
    obs_stop_counts = obs_fp.get("stop_reason_counts") or {}
    openai_dialect_stop = [s for s in obs_stop_counts if s in OPENAI_FINISH_REASONS and s not in CLAUDE_STOP_REASONS]
    if openai_dialect_stop:
        hard_fail = True
        reasons.append(f"stop_reason_openai_dialect:{','.join(sorted(openai_dialect_stop))}")
    evidence.append({
        "check": "stop_reason_enum",
        "baseline": base_fp.get("stop_reason_counts"),
        "observed": obs_stop_counts,
    })

    # 2. usage naming dialect must be anthropic
    obs_usage = obs_fp.get("usage_naming_dialect_counts") or {}
    if obs_usage.get("openai") or obs_usage.get("mixed"):
        hard_fail = True
        reasons.append("usage_naming_openai_or_mixed")
    evidence.append({
        "check": "usage_naming_dialect",
        "baseline": base_fp.get("usage_naming_dialect_counts"),
        "observed": obs_usage,
    })

    # 3. model identity drift (wrapper often returns a different / no model id)
    base_model = (baseline.get("source") or {}).get("model")
    obs_model = (observed.get("source") or {}).get("model")
    model_match = bool(base_model) and base_model == obs_model
    evidence.append({"check": "model_id", "baseline": base_model, "observed": obs_model})

    # 4. anthropic header / request-id presence (soft)
    soft_misses = 0
    for field in ("anthropic_request_id_rate", "anthropic_headers_rate"):
        base_rate = numeric(base_fp.get(field))
        obs_rate = numeric(obs_fp.get(field))
        if base_rate and base_rate > 0.5 and (obs_rate is None or obs_rate < 0.5):
            soft_misses += 1
            reasons.append(f"{field}_below_baseline")
        evidence.append({"check": field, "baseline": base_rate, "observed": obs_rate})

    # 5. behavior signals (tokenizer / SSE / error envelope) — hard-to-fake layer.
    # Each can hard-fail (mismatch) or add a positive vote (match).
    sig = behavior_signals or {}
    behavior_votes = 0          # positive matches
    behavior_checked = 0
    behavior_soft_misses = 0    # behavior probes that disagree but must NOT solely convict
    probe_errors: list[str] = []  # requested probes that were supplied but crashed

    def _probe_failed(name: str, val: Any) -> bool:
        """A probe the caller ran but that errored out. We must surface this as an
        INCOMPLETE check (not silently treat the signal as absent), otherwise a
        crashed downgrade/wrapper probe would leave a falsely-clean verdict."""
        if isinstance(val, dict) and val.get("probe_error"):
            probe_errors.append(f"{name}:{val.get('probe_error')}")
            evidence.append({"check": name, "probe_error": val.get("probe_error"), "incomplete": True})
            return True
        return False

    tok = sig.get("tokenizer")
    if not _probe_failed("tokenizer", tok):
      if isinstance(tok, dict) and tok.get("score") is not None:
        behavior_checked += 1
        if tok.get("score") == 0.0:
            # tokenizer is corroborating ONLY — never sole grounds for wrapper.
            # (self-reported token counts + short-canary differencing are noisy.)
            behavior_soft_misses += 1
            reasons.append(f"tokenizer_delta_off_baseline:{tok.get('suspected_tokenizer','unknown')}")
        elif tok.get("score") == 10.0:
            behavior_votes += 1
        evidence.append({"check": "tokenizer_delta", "observed": tok.get("observed"), "result": tok.get("details")})
      elif isinstance(tok, dict):
        # advisory only (e.g. too few samples) — show but do not vote/penalize
        evidence.append({"check": "tokenizer_delta", "observed": tok.get("observed"), "result": tok.get("details"), "advisory": True})

    sse = sig.get("sse")
    if not _probe_failed("sse", sse):
      if isinstance(sse, dict) and sse.get("sse_family"):
        behavior_checked += 1
        if sse.get("sse_family") == "openai_sse":
            hard_fail = True  # OpenAI SSE frames on an anthropic endpoint IS a strong wrapper signal
            reasons.append("sse_openai_family")
        elif sse.get("is_claude_shaped"):
            behavior_votes += 1
        evidence.append({"check": "sse_event_order", "observed": sse.get("sse_family"), "order_ok": sse.get("claude_event_order_ok")})

    env = sig.get("error_envelope")
    if not _probe_failed("error_envelope", env):
      if isinstance(env, dict) and env.get("error_envelope_dialect"):
        d = env.get("error_envelope_dialect")
        if d in ("openai", "gateway_generic"):
            # generic on a tolerant gateway is weak (it 200s bad requests); soft only
            reasons.append(f"error_envelope_{d}")
        elif d == "anthropic":
            behavior_votes += 1
            behavior_checked += 1
        evidence.append({"check": "error_envelope", "observed": d})

    nd = sig.get("needle")
    if not _probe_failed("needle", nd):
      if isinstance(nd, dict) and nd.get("evidence_status") == "live_observed":
        behavior_checked += 1
        trunc = nd.get("silent_truncation") or {}
        if trunc.get("silent_truncation") is True:
            hard_fail = True  # fake-1M: claimed long context but silently truncated
            reasons.append("needle_silent_truncation")
        elif (nd.get("needle_recall") or {}).get("score") == 10.0:
            behavior_votes += 1
        evidence.append({"check": "needle_fake_1m", "observed": nd.get("verdict"),
                         "silent_truncation": trunc.get("silent_truncation")})

    # capability anchors: the dedicated DOWNGRADE detector. A model that keeps
    # Claude's protocol/tokenizer but solves far fewer hard anchors than baseline
    # is a silent downgrade (opus -> haiku). This is the one signal that convicts
    # downgrade on its own (unlike model_id, which is a trivially-forged string).
    cap = sig.get("capability")
    capability_downgrade = False
    capability_borderline = False
    if not _probe_failed("capability", cap):
      if isinstance(cap, dict) and cap.get("score") is not None:
        behavior_checked += 1
        if cap.get("score") == 0.0:
            capability_downgrade = True
            reasons.append(f"capability_pass_rate_below_baseline:gap={cap.get('gap')}")
        elif cap.get("score") == 10.0:
            behavior_votes += 1
        elif cap.get("score") == 5.0:
            # borderline gap or large-gap-on-too-few-samples: corroborating
            # REVIEW signal, NOT a stand-alone downgrade conviction.
            capability_borderline = True
            reasons.append(f"capability_pass_rate_borderline:gap={cap.get('gap')}")
        evidence.append({"check": "capability_anchor", "baseline": cap.get("baseline"),
                         "observed": cap.get("observed"), "result": cap.get("detail")})
      elif isinstance(cap, dict):
        evidence.append({"check": "capability_anchor", "observed": cap.get("observed"),
                         "result": cap.get("detail"), "advisory": True})

    # consistency-variance: the LOW-FREQUENCY swap detector. One deterministic
    # anchor repeated N times; statistically-significant failures or answer
    # non-determinism on a temp-0 anchor => a fraction of requests is being
    # routed to a weaker model. score 0 convicts downgrade (like capability);
    # score 5 is a soft REVIEW corroborator.
    var = sig.get("variance")
    if not _probe_failed("variance", var):
      if isinstance(var, dict) and var.get("score") is not None and not var.get("advisory"):
        behavior_checked += 1
        if var.get("score") == 0.0:
            capability_downgrade = True  # treated as a downgrade conviction
            reasons.append(f"consistency_variance_swap:p={var.get('p_value')},fails={var.get('failures')}/{var.get('answered')}")
        elif var.get("score") == 5.0:
            capability_borderline = True
            reasons.append("consistency_variance_borderline")
        elif var.get("score") == 10.0:
            behavior_votes += 1
        evidence.append({"check": "consistency_variance", "baseline": "deterministic@temp0",
                         "observed": var.get("observed_pass_rate"), "result": var.get("detail")})
      elif isinstance(var, dict):
        evidence.append({"check": "consistency_variance", "observed": var.get("observed_pass_rate"),
                         "result": var.get("detail"), "advisory": True})

    # A borderline capability gap is a soft signal: it lowers confidence and, with
    # one more soft miss, routes to REVIEW — but never convicts downgrade alone.
    if capability_borderline:
        behavior_soft_misses += 1

    # Verdict logic. hard_fail = a STRONG protocol/SSE signal (openai stop_reason,
    # openai usage naming, openai SSE frames). Tokenizer/header are corroborating only.
    if hard_fail:
        verdict = VERDICT_WRAPPER
        confidence = 0.9 if behavior_checked else 0.85
    elif capability_downgrade:
        # genuine-protocol but under-performing -> the textbook silent downgrade
        verdict = VERDICT_DOWNGRADE
        confidence = 0.8
        reasons.append("capability_downgrade_detected")
    elif not model_match:
        verdict = VERDICT_DOWNGRADE
        confidence = 0.6
        reasons.append("model_id_mismatch")
    elif soft_misses >= 2 or behavior_soft_misses >= 2:
        # multiple soft signals disagree -> needs human review, not an outright verdict
        verdict = VERDICT_DOWNGRADE
        confidence = 0.5
    else:
        verdict = VERDICT_MATCHES
        # protocol match baseline 0.7; each matching behavior probe raises confidence.
        # a single soft miss (e.g. noisy tokenizer delta) nudges confidence down but
        # does not flip the verdict.
        confidence = min(0.97, 0.7 + 0.08 * behavior_votes - 0.1 * behavior_soft_misses)
        confidence = max(0.4, confidence)
        if behavior_soft_misses:
            reasons.append("protocol_match_with_noisy_behavior_signal")
        elif behavior_votes:
            reasons.append(f"protocol_plus_{behavior_votes}_behavior_probes_matched")
        else:
            reasons.append("protocol_layer_match_only_behavior_probes_pending")

    # A requested probe that crashed leaves a hole in the evidence. Never let that
    # masquerade as a clean pass: cap confidence and flag the incomplete check so a
    # MATCHES/insufficient verdict is visibly provisional, not authoritative.
    if probe_errors:
        for pe in probe_errors:
            reasons.append(f"probe_failed:{pe.split(':')[0]}")
        confidence = round(min(confidence, 0.5), 3)
        reasons.append("verdict_incomplete_due_to_probe_error")

    note = "protocol + behavior comparison" if behavior_checked else "protocol-layer only (no behavior signals supplied)"
    if probe_errors:
        note = f"INCOMPLETE — {len(probe_errors)} requested probe(s) errored: {note}"

    return {
        "schema_version": BASELINE_COMPARISON_RESULT_VERSION,
        "baseline_id": baseline.get("baseline_id"),
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "evidence_chain": evidence,
        "behavior_probes_checked": behavior_checked,
        "probe_errors": probe_errors,
        "note": note,
    }


def score_token_count(
    *,
    delta: float | None,
    text_tokens: float | None,
    claude_delta_window: list[float] | None,
    claude_window: list[float] | None,
    competitor_windows: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Score a tokenizer probe.

    delta = input_tokens(self) - input_tokens(diff_partner): prefix-cancelled,
    the PRIMARY signal. text_tokens = input_tokens(self) - prefix_baseline:
    secondary cross-check. Returns 10.0 if both land in the Claude window(s),
    0.0 + suspected_tokenizer if either lands in a competitor window, None if
    there is not enough evidence (missing tokens / no windows / all out-of-range).
    """
    competitor_windows = competitor_windows or {}

    def _in(value: float | None, window: list[float] | None) -> bool | None:
        if value is None or not window or len(window) != 2:
            return None
        return window[0] <= value <= window[1]

    delta_claude = _in(delta, claude_delta_window)
    text_claude = _in(text_tokens, claude_window)

    # competitor match on either metric = hard fail
    for label, win in competitor_windows.items():
        if _in(delta, win) is True or _in(text_tokens, win) is True:
            return {
                "score": 0.0,
                "format_ok": False,
                "details": f"token count matches competitor tokenizer window: {label}",
                "suspected_tokenizer": label,
                "observed": {"delta": delta, "text_tokens": text_tokens},
            }

    # primary: delta must be in claude delta window; text is cross-check if present
    if delta_claude is True and text_claude is not False:
        return {
            "score": 10.0,
            "format_ok": True,
            "details": "token count matches Claude tokenizer (prefix-free delta)",
            "observed": {"delta": delta, "text_tokens": text_tokens},
        }
    if delta_claude is False or text_claude is False:
        return {
            "score": 0.0,
            "format_ok": False,
            "details": "token count outside Claude window (no competitor match)",
            "suspected_tokenizer": "unknown",
            "observed": {"delta": delta, "text_tokens": text_tokens},
        }
    return {
        "score": None,
        "format_ok": None,
        "details": "insufficient token evidence (missing tokens or no calibrated windows)",
        "observed": {"delta": delta, "text_tokens": text_tokens},
    }


def score_capability_item(item: dict[str, Any], response_text: str) -> dict[str, Any]:
    """Objectively score one capability-anchor item against a model's answer.

    Capability anchors are graded WITHOUT a judge model (a judge could itself be
    swapped/unreliable — that's calibrated separately). Grading is deterministic:
      - check="exact":    normalized answer must equal one of expected_any
      - check="contains": answer must contain ALL of expected_all (case-insensitive)
      - check="regex":    answer must match the pattern
    Returns {passed: bool, detail: str}.
    """
    text = (response_text or "").strip()
    check = str(item.get("check") or "contains").lower()

    if check == "exact":
        expected = [str(e).strip() for e in (item.get("expected_any") or [])]
        norm = text.strip().strip(".。").strip()
        passed = any(norm == e or text == e for e in expected)
        return {"passed": passed, "detail": ("exact match" if passed else "no exact match")}

    if check == "regex":
        import re as _re
        pat = item.get("pattern")
        try:
            passed = bool(pat) and _re.search(str(pat), text) is not None
        except _re.error as exc:
            return {"passed": False, "detail": f"bad regex: {exc}"}
        return {"passed": passed, "detail": ("regex matched" if passed else "regex no match")}

    # default: contains ALL expected_all (case-insensitive substring)
    needles = [str(e).strip() for e in (item.get("expected_all") or [])]
    low = text.lower()
    missing = [n for n in needles if n.lower() not in low]
    passed = bool(needles) and not missing
    detail = "all key points present" if passed else f"missing: {missing}"
    return {"passed": passed, "detail": detail}


def aggregate_capability(item_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item capability results into a pass-rate fingerprint.

    item_results: [{id, passed, ok}]. ok=False (call failed) is EXCLUDED from the
    rate denominator — a 429 is not a wrong answer. Returns pass_rate over the
    items that actually returned, plus counts."""
    answered = [r for r in item_results if r.get("ok", True)]
    failed = [r for r in item_results if not r.get("ok", True)]
    passed = [r for r in answered if r.get("passed")]
    n = len(answered)
    return {
        "capability_anchor_pass_rate": round(len(passed) / n, 4) if n else None,
        "answered_count": n,
        "passed_count": len(passed),
        "failed_request_count": len(failed),
        "total_items": len(item_results),
    }


def score_capability_vs_baseline(
    observed_pass_rate: float | None,
    baseline_pass_rate: float | None,
    *,
    answered_count: int = 0,
    min_items: int = 5,
    downgrade_margin: float = 0.25,
    review_margin: float = 0.12,
    confident_items: int = 10,
) -> dict[str, Any]:
    """Compare an observed capability pass-rate against the baseline's.

    A genuine same-tier model tracks the baseline; a silently-downgraded model
    (opus -> haiku) solves far fewer hard anchors while protocol/tokenizer look
    identical — capability is the signal that catches it.

    Threshold rationale (anchors are objectively-graded, single-interpretation,
    temperature 0, so a same-tier model is near-deterministic at ~1.0):
      - downgrade_margin 0.25: a real tier drop (opus->haiku on hard anchors)
        moves pass-rate by far more than sampling noise on ~12 items; 0.25 is
        a conservative floor that a same-tier jitter won't cross.
      - review_margin 0.12: gaps in [0.12, 0.25) are suspicious but within the
        range a small-N estimate could produce by noise -> REVIEW, not convict.
      - confident_items 10: a hard downgrade conviction (score 0) needs enough
        answered anchors to trust the point estimate. With min_items..confident_items
        answered, even a >=downgrade_margin gap is only a borderline REVIEW (the
        estimate is too noisy to convict on its own).

    Returns a graded result feeding compare_to_baseline's behavior layer:
      score 10  = at/near baseline (match vote)
      score 0   = confident downgrade (gap >= downgrade_margin AND enough samples)
      score 5   = borderline (review_margin <= gap, but not a confident convict)
      score None = insufficient (too few answered, or no baseline rate)
    """
    if baseline_pass_rate is None or observed_pass_rate is None:
        return {"score": None, "detail": "no baseline or observed pass-rate", "observed": observed_pass_rate}
    if answered_count < min_items:
        return {"score": None, "detail": f"only {answered_count} answered (< {min_items}); advisory",
                "observed": observed_pass_rate, "advisory": True}
    gap = round(baseline_pass_rate - observed_pass_rate, 4)
    if gap >= downgrade_margin and answered_count >= confident_items:
        return {"score": 0.0, "suspected_downgrade": True, "gap": gap,
                "observed": observed_pass_rate, "baseline": baseline_pass_rate,
                "detail": f"pass-rate {observed_pass_rate} is {gap} below baseline {baseline_pass_rate} "
                          f"(confident: {answered_count} anchors)"}
    if gap >= review_margin:
        # either a mid-band gap, or a large gap on too few samples to convict
        reason = ("borderline gap" if gap < downgrade_margin
                  else f"large gap but only {answered_count} anchors (< {confident_items} to convict)")
        return {"score": 5.0, "borderline": True, "gap": gap,
                "observed": observed_pass_rate, "baseline": baseline_pass_rate,
                "detail": f"pass-rate {observed_pass_rate} is {gap} below baseline {baseline_pass_rate} ({reason}); review"}
    return {"score": 10.0, "gap": gap, "observed": observed_pass_rate, "baseline": baseline_pass_rate,
            "detail": f"pass-rate {observed_pass_rate} tracks baseline {baseline_pass_rate}"}


def _binom_tail_at_least(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Used to test whether observed failures
    on a repeated deterministic anchor are too many to be sampling noise."""
    if n <= 0 or k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0 if k > 0 else 1.0
    if p >= 1.0:
        return 1.0
    # exact sum of binomial pmf from k..n
    from math import comb
    total = 0.0
    for i in range(k, n + 1):
        total += comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return min(1.0, total)


def score_consistency_variance(
    repeats: list[dict[str, Any]],
    *,
    baseline_pass_rate: float = 1.0,
    min_repeats: int = 8,
    p_value_convict: float = 0.01,
    p_value_review: float = 0.10,
    max_distinct_answers: int = 1,
) -> dict[str, Any]:
    """Detect LOW-FREQUENCY swapping by repeating ONE deterministic anchor N times.

    A genuine same-tier model at temperature 0 is near-deterministic: it should
    pass every repeat and return the same normalized answer each time. If an
    upstream silently swaps a FRACTION of requests to a weaker model (e.g. 10%
    routed to haiku), single-shot capability probing misses it — but repeating
    one hard, single-answer anchor surfaces it as occasional wrong/varying
    answers. We test the failure count against a binomial null (the model SHOULD
    pass at `baseline_pass_rate`, ~1.0) and also flag answer non-determinism.

    repeats: [{passed: bool, answer_norm: str|None, ok: bool}]. ok=False
    (429/transport) is EXCLUDED — a rate-limit is not a wrong answer.

    Returns a graded result for compare_to_baseline's behavior layer:
      score 0   = statistically-significant failures (p < p_value_convict) -> swap
      score 5   = borderline (p < p_value_review, or answers vary) -> REVIEW
      score 10  = consistent, deterministic (tracks baseline)
      score None = too few answered repeats -> advisory
    """
    answered = [r for r in repeats if r.get("ok", True)]
    n = len(answered)
    if n < min_repeats:
        return {"score": None, "advisory": True, "answered": n,
                "detail": f"only {n} answered repeats (< {min_repeats}); advisory"}
    failures = sum(1 for r in answered if not r.get("passed"))
    distinct = {str(r.get("answer_norm")) for r in answered if r.get("answer_norm") is not None}
    n_distinct = len(distinct)
    # binomial null: a same-tier model passes at baseline_pass_rate, so failures
    # follow Binomial(n, 1 - baseline_pass_rate). Too many failures => swap.
    p_fail_null = max(0.0, min(1.0, 1.0 - baseline_pass_rate))
    pval = _binom_tail_at_least(failures, n, p_fail_null) if p_fail_null > 0 else (
        0.0 if failures > 0 else 1.0)
    observed_pass_rate = round((n - failures) / n, 4)
    base = {"answered": n, "failures": failures, "p_value": round(pval, 5),
            "distinct_answers": n_distinct, "observed_pass_rate": observed_pass_rate}
    if pval < p_value_convict:
        return {**base, "score": 0.0, "suspected_swap": True,
                "detail": f"{failures}/{n} repeats failed a deterministic anchor "
                          f"(p={pval:.4f} < {p_value_convict}); low-frequency swap suspected"}
    if pval < p_value_review or n_distinct > max_distinct_answers:
        why = (f"borderline failure rate (p={pval:.4f})" if pval < p_value_review
               else f"{n_distinct} distinct answers on a temp-0 anchor (expected {max_distinct_answers})")
        return {**base, "score": 5.0, "borderline": True,
                "detail": f"{why}; review (corroborating, not a stand-alone conviction)"}
    return {**base, "score": 10.0,
            "detail": f"{n - failures}/{n} consistent, deterministic; tracks baseline"}


def score_needle_recall(canary_code: str | None, response_text: str) -> dict[str, Any]:
    """Score a needle probe: did the response echo the planted AUTH_CANARY code?"""
    if not canary_code:
        return {"score": None, "format_ok": None, "details": "missing canary code"}
    hit = canary_code in (response_text or "")
    return {
        "score": 10.0 if hit else 0.0,
        "format_ok": hit,
        "details": "needle recalled" if hit else "needle missed (corroborating only)",
    }


def evaluate_silent_truncation(
    *,
    sent_estimate_tokens: float,
    observed_input_tokens: float | None,
    prefix_tokens: float | None,
    shortfall_ratio: float = 0.5,
    http_status: int = 200,
    needle_recalled: bool | None = None,
) -> dict[str, Any]:
    """Detect fake-1M silent truncation.

    PRIMARY signal is needle recall: the canary is planted near the START of the
    prompt, so if it was recalled, the context was NOT truncated — regardless of
    token counts. Token shortfall is only a CORROBORATING signal, and unreliable
    because our char->token send estimate varies a lot by filler text. So:
      - recall succeeded  -> NOT truncated (hard).
      - recall failed AND observed tokens are FAR below sent (gap > shortfall) ->
        silent_truncation.
      - non-200 -> legit error, not fakery.
    """
    if http_status != 200:
        return {"silent_truncation": False, "reason": f"non_200_status:{http_status}_legit_not_fakery"}
    # needle recalled = context reached the planted code = NOT truncated.
    if needle_recalled is True:
        return {
            "silent_truncation": False,
            "reason": "needle_recalled_context_intact",
            "observed_input_tokens": observed_input_tokens,
        }
    if observed_input_tokens is None:
        return {"silent_truncation": False, "reason": "no_observed_input_tokens", "prefix_assumed": prefix_tokens is None}
    effective = observed_input_tokens - (prefix_tokens or 0.0)
    threshold = sent_estimate_tokens * shortfall_ratio
    # only suspect truncation when recall did NOT succeed AND tokens are far short.
    # (send estimate is unreliable, so use a generous 0.5 ratio + require failed recall.)
    truncated = (needle_recalled is False) and (effective < threshold)
    return {
        "silent_truncation": bool(truncated),
        "effective_text_tokens": round(effective, 1),
        "sent_estimate_tokens": round(sent_estimate_tokens, 1),
        "threshold": round(threshold, 1),
        "prefix_assumed": prefix_tokens is None,
        "needle_recalled": needle_recalled,
        "reason": (
            "needle_missed_and_tokens_far_below_sent" if truncated
            else "needle_recall_unknown_token_estimate_unreliable" if needle_recalled is None
            else "tokens_short_but_needle_not_failed_or_estimate_noise"
        ),
    }


def derive_token_windows(
    baseline: dict[str, Any],
    *,
    long_probe: str = "canary_mixed",
    short_probe: str = "canary_zh",
    tolerance: float = 0.05,
    abs_floor: float = 8.0,
) -> dict[str, Any]:
    """Derive token_count_check windows from a TRUSTED live baseline's observed
    per-probe input_tokens (no offline tokenizer needed).

    For each probe: claude_window = mean ± max(tolerance*mean, abs_floor).
    Also derives the prefix-free claude_delta_window between long and short probes
    (their observed-token difference cancels the shared injected prefix).
    """
    if baseline.get("evidence_status") != "live_observed":
        return {"ok": False, "error": "baseline is not live_observed; cannot derive real windows"}
    windows = (baseline.get("behavior") or {}).get("tokenizer_probe_windows") or {}

    def _win(mean: float | None) -> list[float] | None:
        if mean is None:
            return None
        tol = max(tolerance * mean, abs_floor)
        return [round(mean - tol, 1), round(mean + tol, 1)]

    per_probe: dict[str, Any] = {}
    means: dict[str, float] = {}
    for probe_id, dist in windows.items():
        if isinstance(dist, dict) and dist.get("mean") is not None:
            m = float(dist["mean"])
            means[probe_id] = m
            per_probe[probe_id] = {"observed_mean": m, "claude_window": _win(m)}

    delta_window = None
    if long_probe in means and short_probe in means:
        delta = means[long_probe] - means[short_probe]
        tol = max(tolerance * abs(delta), abs_floor)
        delta_window = [round(delta - tol, 1), round(delta + tol, 1)]

    return {
        "ok": True,
        "schema_version": "token_probe_windows_v1",
        "source_baseline_id": baseline.get("baseline_id"),
        "source_model": (baseline.get("source") or {}).get("model"),
        "derived_from": "live_baseline_observation",
        "diff_pair": {"long": long_probe, "short": short_probe},
        "claude_delta_window": delta_window,
        "per_probe": per_probe,
        "note": "windows derived from trusted live baseline (no offline tokenizer); recalibrate if the gateway prefix changes",
    }


def classify_error_envelope(body_text: str | None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Classify an error response body's dialect: anthropic vs openai vs generic.

    Anthropic: {"type":"error","error":{"type":"invalid_request_error",...}}
    OpenAI:    {"error":{"message":...,"type":"invalid_request_error","code":...}}
    A wrapper that proxies 200s but generates its own errors usually looks generic.
    Pure parser — no network. headers optional (anthropic-* / request-id hints).
    """
    headers = headers or {}
    parsed: Any = None
    try:
        parsed = json.loads(body_text) if body_text else None
    except (ValueError, TypeError):
        parsed = None

    dialect = "unknown"
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if parsed.get("type") == "error" and isinstance(err, dict) and "type" in err:
            dialect = "anthropic"
        elif isinstance(err, dict) and ("message" in err or "code" in err) and parsed.get("type") != "error":
            dialect = "openai"
        else:
            # any other JSON object (FastAPI {"detail":...}, bare {"message":...},
            # {"code":...,"msg":...}, etc.) is a non-Anthropic generic gateway error
            dialect = "gateway_generic"

    lowered = {str(k).lower(): v for k, v in headers.items()}
    has_anthropic_header = any(k.startswith("anthropic-") for k in lowered)
    req_id = lowered.get("anthropic-request-id") or lowered.get("request-id") or ""
    anthropic_req_format = bool(str(req_id).startswith("req_"))

    return {
        "error_envelope_dialect": dialect,
        "has_anthropic_header": has_anthropic_header,
        "anthropic_request_id_format": anthropic_req_format,
        "is_claude_shaped": dialect == "anthropic",
    }


def classify_sse_event_order(event_types: list[str]) -> dict[str, Any]:
    """Check whether an observed SSE event sequence matches Claude's messages stream.

    Claude order: message_start → content_block_start → content_block_delta →
    content_block_stop → message_delta → message_stop (ping interspersed).
    OpenAI: repeated chat.completion.chunk deltas then [DONE]. Pure — no network.
    """
    claude_markers = ["message_start", "content_block_start", "content_block_delta", "message_stop"]
    openai_markers = {"chat.completion.chunk", "[DONE]", "completion"}
    seen = [e for e in event_types if e]
    seen_set = set(seen)

    if seen_set & openai_markers:
        family = "openai_sse"
    elif any(m in seen_set for m in ("message_start", "content_block_delta", "message_stop")):
        family = "claude_sse"
    else:
        family = "unknown"

    # ordered subsequence check for the 4 key claude markers
    idx = 0
    for ev in seen:
        if idx < len(claude_markers) and ev == claude_markers[idx]:
            idx += 1
    order_ok = idx == len(claude_markers)

    return {
        "sse_family": family,
        "claude_event_order_ok": bool(order_ok and family == "claude_sse"),
        "event_types_seen": sorted(seen_set),
        "is_claude_shaped": family == "claude_sse" and order_ok,
    }


VERDICT_LABELS_ZH = {
    VERDICT_MATCHES: "✅ 真·官方 Claude",
    VERDICT_DOWNGRADE: "⚠️ 疑似降级（可能换了更小/更弱的模型）",
    VERDICT_WRAPPER: "❌ 疑似套壳（可能是别家模型伪装）",
    VERDICT_INSUFFICIENT: "❔ 证据不足（无法判定，需更多 live 采集）",
}


def render_verdict_report(verdict: dict[str, Any], *, baseline: dict[str, Any] | None = None) -> str:
    """Render a compare_to_baseline verdict into a human-readable Chinese report."""
    v = verdict.get("verdict", VERDICT_INSUFFICIENT)
    lines: list[str] = []
    lines.append("=" * 48)
    lines.append("  Claude 真伪检测报告")
    lines.append("=" * 48)
    if baseline:
        src = baseline.get("source") or {}
        lines.append(f"对照基线: {baseline.get('baseline_id')} (model={src.get('model')}, host={src.get('base_url_host')})")
    lines.append(f"结论: {VERDICT_LABELS_ZH.get(v, v)}")
    conf = verdict.get("confidence")
    if conf is not None:
        lines.append(f"置信度: {conf}")
    reasons = verdict.get("reasons") or []
    if reasons:
        lines.append("理由:")
        for r in reasons:
            lines.append(f"  - {r}")
    chain = verdict.get("evidence_chain") or []
    # Surface crashed probes FIRST and loudly — a requested probe that errored is a
    # hole in the evidence, and the verdict below is only provisional because of it.
    perr = verdict.get("probe_errors") or []
    if perr:
        lines.append("⚠ 探针未完成（结论不完整，请重跑）:")
        for pe in perr:
            lines.append(f"  ✗ {pe}")
    if chain:
        # #5: group evidence by strength so users see what's definitive vs advisory.
        STRONG = {"stop_reason_enum", "usage_naming_dialect", "model_id",
                  "sse_event_order", "error_envelope", "needle_fake_1m", "request_failure_rate",
                  "capability_anchor", "consistency_variance"}
        # crashed-probe rows are shown in the dedicated block above, not here.
        chain = [e for e in chain if not e.get("probe_error")]
        strong = [e for e in chain if e.get("check") in STRONG and not e.get("advisory")]
        corro = [e for e in chain if e.get("check") not in STRONG and not e.get("advisory")]
        advisory = [e for e in chain if e.get("advisory")]
        def _fmt(e):
            extra = "".join(f" {k}={e[k]}" for k in ("order_ok", "silent_truncation", "result") if k in e and e[k] is not None)
            return f"  · {e.get('check')}: baseline={e.get('baseline')} | observed={e.get('observed')}{extra}"
        if strong:
            lines.append("强证据（定罪级）:")
            for e in strong:
                lines.append(_fmt(e))
        if corro:
            lines.append("佐证（参考，单独不定罪）:")
            for e in corro:
                lines.append(_fmt(e))
        if advisory:
            lines.append("仅参考（样本不足/不计票）:")
            for e in advisory:
                lines.append(_fmt(e))
    note = verdict.get("note")
    if note:
        lines.append(f"备注: {note}")
    lines.append("=" * 48)
    return "\n".join(lines)


def _fake_official_samples(n: int = 6) -> list[dict[str, Any]]:
    """Synthetic samples that look like genuine official Claude (for self-test)."""
    samples = []
    for i in range(n):
        samples.append(make_sample(
            protocol="anthropic_messages",
            raw_stop_reason="end_turn" if i % 3 else "max_tokens",
            raw_usage_keys=["input_tokens", "output_tokens", "cache_read_input_tokens"],
            input_tokens=100 + i,
            output_tokens=40 + i,
            total_ms=900 + i * 10,
            has_anthropic_request_id=True,
            has_anthropic_headers=True,
            probe_id="canary_mixed",
            expected_input_tokens=102,
            live=True,
        ))
    return samples


def _fake_wrapper_observed() -> dict[str, Any]:
    """Synthetic observed fingerprint from an OpenAI-wrapper pretending to be Claude."""
    samples = [make_sample(
        protocol="anthropic_messages",
        raw_stop_reason="stop",  # OpenAI dialect leak
        raw_usage_keys=["prompt_tokens", "completion_tokens"],  # OpenAI naming
        input_tokens=100,
        output_tokens=40,
        total_ms=700,
        has_anthropic_request_id=False,
        has_anthropic_headers=False,
        probe_id="canary_mixed",
        live=True,
    ) for _ in range(5)]
    return build_baseline_from_samples(
        samples,
        {"provider_id": "suspect", "model": "claude-opus-4-8", "protocol": "anthropic_messages"},
        baseline_id="suspect_observed",
        live=True,
    )


def _self_test() -> None:
    # 1. aggregate official samples -> baseline
    source = {
        "provider_id": "tested_model",
        "provider_label": "tested_model",
        "base_url_host": "drhknode.airouting.com",
        "model": "claude-opus-4-8",
        "protocol": "anthropic_messages",
        "key_fingerprint": key_fingerprint("dummy-secret"),
    }
    baseline = build_baseline_from_samples(
        _fake_official_samples(), source, baseline_id="selftest_official", live=True
    )
    assert baseline["schema_version"] == CLAUDE_BASELINE_SCHEMA_VERSION
    assert baseline["protocol_fingerprint"]["stop_reason_in_claude_enum"] is True
    assert baseline["protocol_fingerprint"]["usage_naming_dialect_counts"].get("anthropic")
    assert baseline["behavior"]["input_tokens_distribution"]["count"] == 6
    assert baseline["source"]["key_fingerprint"] and "dummy-secret" not in json.dumps(baseline)

    # 2. write -> read round-trip in a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp) / "baselines"
        write_baseline(base_dir, "selftest_official", baseline)
        loaded = load_baseline(base_dir, "selftest_official")
        assert loaded == baseline
        assert list_baselines(base_dir) == ["selftest_official"]

    # 3. compare: genuine-looking observed -> matches; wrapper -> suspected_wrapper
    genuine_observed = build_baseline_from_samples(
        _fake_official_samples(), source, baseline_id="genuine_observed", live=True
    )
    good = compare_to_baseline(genuine_observed, baseline)
    assert good["verdict"] == VERDICT_MATCHES, good

    # behavior signals raise confidence on a match, and hard-fail on tokenizer mismatch
    good_beh = compare_to_baseline(genuine_observed, baseline, behavior_signals={
        "tokenizer": {"score": 10.0, "observed": {"delta": 80}},
        "sse": {"sse_family": "claude_sse", "is_claude_shaped": True, "claude_event_order_ok": True},
        "error_envelope": {"error_envelope_dialect": "anthropic"},
    })
    assert good_beh["verdict"] == VERDICT_MATCHES and good_beh["confidence"] > good["confidence"], good_beh
    # a single noisy tokenizer miss must NOT flip the verdict to wrapper (corroborating only)
    bad_beh = compare_to_baseline(genuine_observed, baseline, behavior_signals={
        "tokenizer": {"score": 0.0, "suspected_tokenizer": "cl100k", "observed": {"delta": 55}},
    })
    assert bad_beh["verdict"] == VERDICT_MATCHES and bad_beh["confidence"] < good["confidence"], bad_beh
    # but OpenAI SSE frames (a strong signal) DO hard-fail to wrapper
    sse_fail = compare_to_baseline(genuine_observed, baseline, behavior_signals={
        "sse": {"sse_family": "openai_sse", "is_claude_shaped": False},
    })
    assert sse_fail["verdict"] == VERDICT_WRAPPER, sse_fail

    # a REQUESTED probe that crashed must surface as an incomplete check — never a
    # silently-clean pass. confidence is capped and the failure is named.
    probe_err = compare_to_baseline(genuine_observed, baseline, behavior_signals={
        "capability": {"probe_error": "RuntimeError: upstream 500"},
    })
    assert probe_err["probe_errors"] and "capability" in probe_err["probe_errors"][0], probe_err
    assert probe_err["confidence"] <= 0.5, probe_err
    assert any(r.startswith("probe_failed:capability") for r in probe_err["reasons"]), probe_err
    assert "INCOMPLETE" in probe_err["note"], probe_err
    # the crashed probe must NOT count as a completed behavior check
    assert probe_err["behavior_probes_checked"] == 0, probe_err
    # a healthy capability vote alongside a different crashed probe still flags incomplete
    mixed = compare_to_baseline(genuine_observed, baseline, behavior_signals={
        "sse": {"sse_family": "claude_sse", "is_claude_shaped": True},
        "needle": {"probe_error": "TimeoutError"},
    })
    assert mixed["probe_errors"] and mixed["confidence"] <= 0.5, mixed
    assert mixed["behavior_probes_checked"] == 1, mixed  # sse counted, needle did not

    # mostly-failed requests (e.g. 429) -> insufficient, NOT a suspicious empty fingerprint
    failed_samples = [make_sample(
        protocol="anthropic_messages", raw_stop_reason=None, raw_usage_keys=[],
        input_tokens=None, output_tokens=None, total_ms=None, probe_id="canary_mixed",
        live=True, ok=False,
    ) for _ in range(6)]
    failed_obs = build_baseline_from_samples(failed_samples, source, baseline_id="failed", live=True)
    assert failed_obs["request_failure_rate"] == 1.0, failed_obs
    failed_cmp = compare_to_baseline(failed_obs, baseline)
    assert failed_cmp["verdict"] == VERDICT_INSUFFICIENT and "observed_requests_failed" in failed_cmp["reasons"], failed_cmp

    wrapper = compare_to_baseline(_fake_wrapper_observed(), baseline)
    assert wrapper["verdict"] == VERDICT_WRAPPER, wrapper

    # 4. dry-run baseline -> comparison must be insufficient
    dry = build_baseline_from_samples(
        _fake_official_samples(), source, baseline_id="dry", live=False
    )
    assert dry["evidence_status"] == "dry_run_reference_only"
    assert compare_to_baseline(genuine_observed, dry)["verdict"] == VERDICT_INSUFFICIENT

    # 5. tokenizer probe scoring
    tok_match = score_token_count(
        delta=42.0, text_tokens=18.0,
        claude_delta_window=[38.0, 46.0], claude_window=[14.0, 22.0],
        competitor_windows={"cl100k": [50.0, 60.0]},
    )
    assert tok_match["score"] == 10.0, tok_match
    tok_comp = score_token_count(
        delta=55.0, text_tokens=18.0,
        claude_delta_window=[38.0, 46.0], claude_window=[14.0, 22.0],
        competitor_windows={"cl100k": [50.0, 60.0]},
    )
    assert tok_comp["score"] == 0.0 and tok_comp["suspected_tokenizer"] == "cl100k", tok_comp
    tok_insufficient = score_token_count(
        delta=None, text_tokens=None, claude_delta_window=None, claude_window=None,
    )
    assert tok_insufficient["score"] is None, tok_insufficient

    # 6. needle recall scoring
    assert score_needle_recall("AUTH_CANARY=ab12", "... AUTH_CANARY=ab12 ...")["score"] == 10.0
    assert score_needle_recall("AUTH_CANARY=ab12", "no code here")["score"] == 0.0

    # 7. silent truncation — needle recall is the PRIMARY signal
    # recall FAILED + tokens far short -> truncated
    trunc = evaluate_silent_truncation(
        sent_estimate_tokens=210000, observed_input_tokens=12000, prefix_tokens=4166,
        needle_recalled=False,
    )
    assert trunc["silent_truncation"] is True, trunc
    # recall SUCCEEDED -> NOT truncated even if token estimate looks short
    recalled_short = evaluate_silent_truncation(
        sent_estimate_tokens=126000, observed_input_tokens=78000, prefix_tokens=4166,
        needle_recalled=True,
    )
    assert recalled_short["silent_truncation"] is False, recalled_short
    # full tokens, recall unknown -> not truncated
    ok_full = evaluate_silent_truncation(
        sent_estimate_tokens=210000, observed_input_tokens=214166, prefix_tokens=4166,
    )
    assert ok_full["silent_truncation"] is False, ok_full
    # non-200 -> legit error, not fakery
    http_err = evaluate_silent_truncation(
        sent_estimate_tokens=210000, observed_input_tokens=None, prefix_tokens=4166, http_status=413,
    )
    assert http_err["silent_truncation"] is False, http_err

    # 8. derive token windows from a live baseline
    derived = derive_token_windows(baseline, long_probe="canary_mixed", short_probe="canary_zh")
    assert derived["ok"] is True, derived
    assert derived["claude_delta_window"] is None or len(derived["claude_delta_window"]) == 2
    # dry baseline must refuse
    assert derive_token_windows(dry)["ok"] is False

    # 9. human-readable report renders for every verdict
    rep = render_verdict_report(wrapper, baseline=baseline)
    assert "套壳" in rep and "Claude 真伪检测报告" in rep, rep
    assert "真·官方" in render_verdict_report(good, baseline=baseline)

    # 9c. consistency-variance: the low-frequency swap detector.
    # all-pass deterministic -> match vote (10)
    allpass = score_consistency_variance([{"passed": True, "answer_norm": "42", "ok": True}] * 12)
    assert allpass["score"] == 10.0, allpass
    # a 10%-swap signature: several failures on a should-be-deterministic anchor -> convict
    swap = score_consistency_variance(
        [{"passed": True, "answer_norm": "42", "ok": True}] * 12 +
        [{"passed": False, "answer_norm": "41", "ok": True}] * 4)
    assert swap["score"] == 0.0 and swap.get("suspected_swap") is True, swap
    # answer non-determinism on temp-0 (all "passed" but varying answers) -> borderline review
    vary = score_consistency_variance(
        [{"passed": True, "answer_norm": "42", "ok": True}] * 10 +
        [{"passed": True, "answer_norm": "forty-two", "ok": True}] * 2)
    assert vary["score"] == 5.0 and vary.get("borderline") is True, vary
    # too few answered (429s excluded) -> advisory
    thin_v = score_consistency_variance(
        [{"passed": True, "answer_norm": "42", "ok": True}] * 3 +
        [{"passed": None, "answer_norm": None, "ok": False}] * 10)
    assert thin_v["score"] is None and thin_v.get("advisory") is True, thin_v
    # binomial tail sanity: many failures under a ~0 null prob is highly significant
    assert _binom_tail_at_least(4, 16, 0.0) == 0.0
    assert _binom_tail_at_least(0, 16, 0.0) == 1.0
    # variance score 0 folds into compare_to_baseline as a downgrade conviction
    var_dg = compare_to_baseline(
        build_baseline_from_samples(_fake_official_samples(), source, baseline_id="o", live=True),
        baseline,
        behavior_signals={"variance": swap})
    assert var_dg["verdict"] == VERDICT_DOWNGRADE, var_dg

    # 10. error envelope dialect (top-level type:error is the discriminator,
    #     NOT error.type — both Anthropic & OpenAI use invalid_request_error)
    assert classify_error_envelope(
        '{"type":"error","error":{"type":"invalid_request_error","message":"x"}}',
        {"anthropic-request-id": "req_011AbC"},
    )["error_envelope_dialect"] == "anthropic"
    assert classify_error_envelope(
        '{"error":{"message":"x","type":"invalid_request_error","code":null}}', {},
    )["error_envelope_dialect"] == "openai"
    assert classify_error_envelope('{"detail":"Unprocessable"}', {})["error_envelope_dialect"] == "gateway_generic"
    assert classify_error_envelope(None, {})["error_envelope_dialect"] == "unknown"

    # 11. SSE event-order fingerprint
    claude_seq = ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert classify_sse_event_order(claude_seq)["is_claude_shaped"] is True
    assert classify_sse_event_order(["chat.completion.chunk", "chat.completion.chunk", "[DONE]"])["sse_family"] == "openai_sse"
    assert classify_sse_event_order(["message_stop", "message_start"])["claude_event_order_ok"] is False

    # 12. baseline versioning — lineage, dedup, drift detection
    with tempfile.TemporaryDirectory() as tmp:
        vdir = Path(tmp) / "baselines"
        bid = "selftest_versioned"
        v1 = build_baseline_from_samples(_fake_official_samples(), source, baseline_id=bid, live=True)
        r1 = write_baseline_version(vdir, bid, v1, now="2026-06-27T00:00:00Z", note="first")
        assert r1["dedup"] is False and r1["version"] == "v0001", r1
        # latest pointer is readable by the OLD single-file API
        assert load_baseline(vdir, bid) == v1
        # identical rebuild -> dedup, no new version, observed_count bumps
        r2 = write_baseline_version(vdir, bid, v1, now="2026-06-27T01:00:00Z")
        assert r2["dedup"] is True and r2["version"] == "v0001", r2
        assert len(list_baseline_versions(vdir, bid)) == 1
        assert list_baseline_versions(vdir, bid)[0]["observed_count"] == 2
        # a genuinely different fingerprint -> new version + drift summary
        shifted = [make_sample(
            protocol="anthropic_messages", raw_stop_reason="end_turn",
            raw_usage_keys=["input_tokens", "output_tokens"],
            input_tokens=500 + i, output_tokens=40 + i, total_ms=900 + i * 10,
            probe_id="canary_mixed", live=True,
        ) for i in range(6)]
        v2 = build_baseline_from_samples(shifted, source, baseline_id=bid, live=True)
        r3 = write_baseline_version(vdir, bid, v2, now="2026-06-27T02:00:00Z", note="changed")
        assert r3["dedup"] is False and r3["version"] == "v0002", r3
        assert r3["drift"] and r3["drift"]["changed"] is True, r3
        assert len(list_baseline_versions(vdir, bid)) == 2
        # historical snapshot is immutable & retrievable
        assert load_baseline_version(vdir, bid, "v0001") == v1
        assert load_baseline_version(vdir, bid, "v0002") == v2
        # snapshot_path in the manifest is RELATIVE (portable across machines)
        man = load_versions_manifest(vdir, bid)
        assert man is not None
        assert man["versions"][0]["snapshot_path"] == "versions/v0001/baseline.json", man["versions"][0]
        # latest pointer reflects v2 (the most recent distinct write)
        assert load_baseline(vdir, bid) == v2
        # regressing to an earlier fingerprint re-observes v0001, does NOT forge v0003
        r4 = write_baseline_version(vdir, bid, v1, now="2026-06-27T03:00:00Z")
        assert r4["dedup"] is True and r4["version"] == "v0001" and r4.get("regressed") is True, r4
        assert len(list_baseline_versions(vdir, bid)) == 2  # still just v1, v2
        # diff of identical docs -> no change
        _diff_same = diff_baselines(v1, v1)
        assert _diff_same is not None and _diff_same["changed"] is False
        # content fingerprint is stable & excludes volatile metadata
        v1b = dict(v1)
        v1b["sample_count"] = 999  # volatile field must not affect the hash
        assert content_fingerprint(v1b) == content_fingerprint(v1)

    # 13. capability anchors — item scoring, aggregation, downgrade detection
    # item scoring: exact / contains / regex
    assert score_capability_item({"check": "exact", "expected_any": ["391"]}, "391")["passed"] is True
    assert score_capability_item({"check": "exact", "expected_any": ["391"]}, "392")["passed"] is False
    assert score_capability_item({"check": "exact", "expected_any": ["391"]}, "391.")["passed"] is True  # trailing punct normalized
    assert score_capability_item({"check": "contains", "expected_all": ["SYN", "ACK"]}, "SYN then SYN-ACK and ACK")["passed"] is True
    assert score_capability_item({"check": "contains", "expected_all": ["SYN", "ACK"]}, "only SYN here")["passed"] is False
    assert score_capability_item({"check": "regex", "pattern": r"\b42\b"}, "the answer is 42 ok")["passed"] is True
    assert score_capability_item({"check": "regex", "pattern": "["}, "x")["passed"] is False  # bad regex -> fail, no crash

    # aggregation: failed calls excluded from denominator
    agg = aggregate_capability([
        {"id": "a", "passed": True, "ok": True}, {"id": "b", "passed": True, "ok": True},
        {"id": "c", "passed": False, "ok": True}, {"id": "d", "passed": None, "ok": False},
    ])
    assert agg["answered_count"] == 3 and agg["passed_count"] == 2, agg
    assert agg["capability_anchor_pass_rate"] == round(2 / 3, 4), agg
    assert agg["failed_request_count"] == 1, agg

    # vs-baseline: tracks -> match vote; far below -> downgrade; small-N -> advisory
    near = score_capability_vs_baseline(0.9, 0.95, answered_count=10)
    assert near["score"] == 10.0, near
    down = score_capability_vs_baseline(0.5, 0.95, answered_count=10)
    assert down["score"] == 0.0 and down["suspected_downgrade"] is True, down
    thin = score_capability_vs_baseline(0.4, 0.95, answered_count=2)
    assert thin["score"] is None and thin.get("advisory") is True, thin
    # borderline band: a mid-range gap (>=review_margin, <downgrade_margin) -> REVIEW (5.0)
    border = score_capability_vs_baseline(0.80, 0.95, answered_count=12)  # gap 0.15
    assert border["score"] == 5.0 and border.get("borderline") is True, border
    # a large gap but too few answered to convict -> borderline REVIEW, not a hard 0.0
    big_thin = score_capability_vs_baseline(0.5, 0.95, answered_count=6)  # gap 0.45, <confident_items
    assert big_thin["score"] == 5.0 and big_thin.get("borderline") is True, big_thin

    # compare_to_baseline: a capability downgrade signal flips a protocol-match to DOWNGRADE
    cap_src = {"provider_id": "t", "provider_label": "t", "base_url_host": "h",
               "model": "claude-opus-4-6", "protocol": "anthropic_messages", "key_fingerprint": None}
    cap_base = build_baseline_from_samples(_fake_official_samples(), cap_src, baseline_id="cap", live=True)
    cap_obs = build_baseline_from_samples(_fake_official_samples(), cap_src, baseline_id="cap_obs", live=True)
    cap_down = compare_to_baseline(cap_obs, cap_base, behavior_signals={
        "capability": {"score": 0.0, "suspected_downgrade": True, "gap": 0.45,
                       "observed": 0.5, "baseline": 0.95},
    })
    assert cap_down["verdict"] == VERDICT_DOWNGRADE and "capability_downgrade_detected" in cap_down["reasons"], cap_down
    # a matching capability signal is a positive vote, verdict stays match
    cap_ok = compare_to_baseline(cap_obs, cap_base, behavior_signals={
        "capability": {"score": 10.0, "gap": 0.02, "observed": 0.93, "baseline": 0.95},
    })
    assert cap_ok["verdict"] == VERDICT_MATCHES, cap_ok

    print("baseline_registry self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Claude authenticity baseline registry")
    parser.add_argument("--self-test", action="store_true", help="run internal self-tests")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
