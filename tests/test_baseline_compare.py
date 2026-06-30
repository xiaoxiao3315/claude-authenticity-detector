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
    # thresholds calibrated 2026-06-29: downgrade 0.15, review 0.06. gap 0.10 = midband
    r = B.score_capability_vs_baseline(0.85, 0.95, answered_count=12)  # gap 0.10
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


# ---------------------------------------------------------------------------
# classify_response_id — supply-chain fingerprint from the id prefix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rid,family", [
    ("msg_01ABCdef", "anthropic_native"),
    ("msg_bdrk_9z", "anthropic_bedrock"),
    ("chatcmpl-9XyZ", "openai_compat"),
    ("gen-abc123", "openrouter"),
    ("deadbeefdeadbeef", "bare_hex"),
    ("weird::id", "other"),
    ("", "absent"),
    (None, "absent"),
])
def test_classify_response_id(rid, family):
    assert B.classify_response_id(rid) == family


# ---------------------------------------------------------------------------
# score_identity_coherence — narration vs envelope
# ---------------------------------------------------------------------------
def test_identity_coherent_match():
    r = B.score_identity_coherence(
        narrated_model_id="claude-opus-4-8", returned_model_field="claude-opus-4-8",
        response_id="msg_01ZZ", expected_model_id="claude-opus-4-8")
    assert r["score"] == 10.0


def test_identity_bedrock_is_genuine():
    r = B.score_identity_coherence(
        narrated_model_id="claude-opus-4-8", returned_model_field="claude-opus-4-8",
        response_id="msg_bdrk_x", expected_model_id="claude-opus-4-8")
    assert r["score"] == 10.0


def test_identity_returned_model_swap_hard_fails():
    # thinkai case: narrates opus but the envelope returns a different model.
    r = B.score_identity_coherence(
        narrated_model_id="claude-opus-4-8", returned_model_field="xiaomi/mimo-v2.5",
        response_id="gen-abc", expected_model_id="claude-opus-4-8")
    assert r["score"] == 0.0 and r["suspected_wrapper"] is True


def test_identity_openai_compat_id_hard_fails():
    # even if the model field looks right, a chatcmpl- id proves an OpenAI relay.
    r = B.score_identity_coherence(
        narrated_model_id="claude-opus-4-8", returned_model_field="claude-opus-4-8",
        response_id="chatcmpl-1", expected_model_id="claude-opus-4-8")
    assert r["score"] == 0.0


def test_identity_weak_envelope_is_soft():
    r = B.score_identity_coherence(
        narrated_model_id="claude-opus-4-8", returned_model_field="claude-opus-4-8",
        response_id="deadbeefdeadbeef", expected_model_id="claude-opus-4-8")
    assert r["score"] == 5.0 and r.get("borderline")


def test_identity_no_envelope_is_insufficient():
    r = B.score_identity_coherence(
        narrated_model_id="claude-opus-4-8", returned_model_field=None,
        response_id=None, expected_model_id="claude-opus-4-8")
    assert r["score"] is None and r.get("advisory")


def test_identity_date_suffix_still_matches():
    r = B.score_identity_coherence(
        narrated_model_id="claude-opus-4-8", returned_model_field="claude-opus-4-8-20251101",
        response_id="msg_01QQ", expected_model_id="claude-opus-4-8")
    assert r["score"] == 10.0


# ---------------------------------------------------------------------------
# identity signal folded into compare_to_baseline
# ---------------------------------------------------------------------------
def test_identity_incoherent_flips_to_wrapper(matched_pair):
    obs, base = matched_pair
    verdict = B.compare_to_baseline(obs, base, behavior_signals={
        "identity": {"score": 0.0, "suspected_wrapper": True,
                     "detail": "returned_model_mismatch", "observed": {}},
    })
    assert verdict["verdict"] == B.VERDICT_WRAPPER
    assert any("identity_incoherent" in str(r) for r in verdict["reasons"])


def test_identity_match_is_positive_vote(matched_pair):
    obs, base = matched_pair
    verdict = B.compare_to_baseline(obs, base, behavior_signals={
        "identity": {"score": 10.0, "detail": "coherent", "observed": {}},
    })
    assert verdict["verdict"] == B.VERDICT_MATCHES


def test_identity_probe_error_marks_incomplete(matched_pair):
    obs, base = matched_pair
    verdict = B.compare_to_baseline(obs, base, behavior_signals={
        "identity": {"probe_error": "Timeout"},
    })
    assert verdict["probe_errors"]
    assert any("identity" in str(r) for r in verdict["reasons"])
