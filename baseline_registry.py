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
    shortfall_ratio: float = 0.9,
    http_status: int = 200,
) -> dict[str, Any]:
    """Hard signal for fake-1M: HTTP 200 but reported input_tokens far below sent.

    Only HTTP 200 + token shortfall counts as silent_truncation. HTTP 400/413 etc.
    are legitimate errors, not fakery. Prefix is subtracted before comparing.
    """
    if http_status != 200:
        return {"silent_truncation": False, "reason": f"non_200_status:{http_status}_legit_not_fakery"}
    if observed_input_tokens is None:
        return {"silent_truncation": False, "reason": "no_observed_input_tokens", "prefix_assumed": prefix_tokens is None}
    effective = observed_input_tokens - (prefix_tokens or 0.0)
    threshold = sent_estimate_tokens * shortfall_ratio
    truncated = effective < threshold
    return {
        "silent_truncation": bool(truncated),
        "effective_text_tokens": round(effective, 1),
        "sent_estimate_tokens": round(sent_estimate_tokens, 1),
        "threshold": round(threshold, 1),
        "prefix_assumed": prefix_tokens is None,
        "reason": "input_tokens_far_below_sent" if truncated else "input_tokens_consistent_with_sent",
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

    # 7. silent truncation
    trunc = evaluate_silent_truncation(
        sent_estimate_tokens=210000, observed_input_tokens=12000, prefix_tokens=4166,
    )
    assert trunc["silent_truncation"] is True, trunc
    ok_full = evaluate_silent_truncation(
        sent_estimate_tokens=210000, observed_input_tokens=214166, prefix_tokens=4166,
    )
    assert ok_full["silent_truncation"] is False, ok_full
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
