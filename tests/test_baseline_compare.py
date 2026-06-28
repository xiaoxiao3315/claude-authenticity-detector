"""Granular tests for compare_to_baseline's capability verdict bands (P5 + P8).

The audit flagged (P5) the capability check as a hard 0.25 cliff with no
borderline treatment, and (P8) that compare_to_baseline lacked granular pytest
cases. These drive the verdict directly with crafted capability signals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import baseline_registry as B  # noqa: E402


@pytest.fixture
def matched_pair():
    src = {"provider_id": "t", "provider_label": "t", "base_url_host": "h",
           "model": "claude-opus-4-6", "protocol": "anthropic_messages", "key_fingerprint": None}
    base = B.build_baseline_from_samples(B._fake_official_samples(), src, baseline_id="b", live=True)
    obs = B.build_baseline_from_samples(B._fake_official_samples(), src, baseline_id="o", live=True)
    return obs, base


# ---------------------------------------------------------------------------
# score_capability_vs_baseline bands
# ---------------------------------------------------------------------------
def test_capability_tracks_baseline_is_match():
    r = B.score_capability_vs_baseline(0.93, 0.95, answered_count=12)
    assert r["score"] == 10.0


def test_capability_confident_downgrade():
    r = B.score_capability_vs_baseline(0.5, 0.95, answered_count=12)
    assert r["score"] == 0.0 and r["suspected_downgrade"] is True


def test_capability_midband_is_borderline_review():
    r = B.score_capability_vs_baseline(0.80, 0.95, answered_count=12)  # gap 0.15
    assert r["score"] == 5.0 and r["borderline"] is True


def test_capability_large_gap_but_few_samples_is_borderline_not_convict():
    # gap 0.45 but only 6 anchors (< confident_items 10) -> REVIEW, not a hard 0.0
    r = B.score_capability_vs_baseline(0.5, 0.95, answered_count=6)
    assert r["score"] == 5.0 and r["borderline"] is True


def test_capability_too_few_is_insufficient():
    r = B.score_capability_vs_baseline(0.4, 0.95, answered_count=2)
    assert r["score"] is None and r.get("advisory") is True


# ---------------------------------------------------------------------------
# compare_to_baseline: borderline capability must NOT convict downgrade
# ---------------------------------------------------------------------------
def test_borderline_capability_does_not_convict_downgrade(matched_pair):
    obs, base = matched_pair
    verdict = B.compare_to_baseline(obs, base, behavior_signals={
        "capability": {"score": 5.0, "borderline": True, "gap": 0.15,
                       "observed": 0.80, "baseline": 0.95},
    })
    # a single borderline soft signal must not flip an otherwise-matching
    # protocol comparison to a downgrade conviction
    assert verdict["verdict"] != B.VERDICT_DOWNGRADE
    assert any("borderline" in str(r) for r in verdict["reasons"])


def test_confident_downgrade_still_convicts(matched_pair):
    obs, base = matched_pair
    verdict = B.compare_to_baseline(obs, base, behavior_signals={
        "capability": {"score": 0.0, "suspected_downgrade": True, "gap": 0.45,
                       "observed": 0.5, "baseline": 0.95},
    })
    assert verdict["verdict"] == B.VERDICT_DOWNGRADE
    assert "capability_downgrade_detected" in verdict["reasons"]


def test_matching_capability_is_positive_vote(matched_pair):
    obs, base = matched_pair
    verdict = B.compare_to_baseline(obs, base, behavior_signals={
        "capability": {"score": 10.0, "gap": 0.02, "observed": 0.93, "baseline": 0.95},
    })
    assert verdict["verdict"] == B.VERDICT_MATCHES
