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
) -> dict[str, Any]:
    """Build one collection sample from RAW observed values.

    raw_stop_reason / raw_usage_keys must come from the upstream response as-is,
    not from CallMetrics (which applies a "stop" / model fallback).
    """
    return {
        "protocol": protocol,
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
    input_token_values: list[float] = []
    latency_values: list[float] = []
    # per-probe input_tokens, for tokenizer expected windows
    probe_tokens: dict[str, list[float]] = {}

    for sample in samples:
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
            # capability-anchor pass rate is a later (P1) addition; placeholder.
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


def compare_to_baseline(observed: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Compare a suspect endpoint's observed fingerprint against the baseline.

    Framework stage: protocol-layer comparison only (behavior layer is P1).
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

    # Verdict logic (protocol layer only at this stage).
    if hard_fail:
        verdict = VERDICT_WRAPPER
        confidence = 0.85
    elif not model_match:
        verdict = VERDICT_DOWNGRADE
        confidence = 0.6
        reasons.append("model_id_mismatch")
    elif soft_misses >= 2:
        verdict = VERDICT_DOWNGRADE
        confidence = 0.5
    else:
        verdict = VERDICT_MATCHES
        confidence = 0.7  # protocol-only match; behavior probes (P1) would raise this
        reasons.append("protocol_layer_match_only_behavior_probes_pending")

    return {
        "schema_version": BASELINE_COMPARISON_RESULT_VERSION,
        "baseline_id": baseline.get("baseline_id"),
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "evidence_chain": evidence,
        "note": "framework stage: protocol-layer comparison only; tokenizer/needle/capability behavior probes are P1",
    }


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

    wrapper = compare_to_baseline(_fake_wrapper_observed(), baseline)
    assert wrapper["verdict"] == VERDICT_WRAPPER, wrapper

    # 4. dry-run baseline -> comparison must be insufficient
    dry = build_baseline_from_samples(
        _fake_official_samples(), source, baseline_id="dry", live=False
    )
    assert dry["evidence_status"] == "dry_run_reference_only"
    assert compare_to_baseline(genuine_observed, dry)["verdict"] == VERDICT_INSUFFICIENT

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
