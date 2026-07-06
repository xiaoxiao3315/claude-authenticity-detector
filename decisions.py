"""Shared GO / REVIEW / NO-GO decision helpers.

Single source of truth for the release-decision vocabulary. Previously
`worst_decision`, `decision_from_score`, `decision_score`, and `DECISION_ORDER`
were duplicated in both campaigns.py and authenticity.py — two copies that had
already drifted (one assigned the normalized form, the other the raw string).
A detector that ranks severity must rank it the same way everywhere, so these
live in one module that both import.

This module has NO project imports, so importing it from campaigns/authenticity
cannot create a cycle.
"""
from __future__ import annotations

# Severity ranking: higher number = more severe. worst_decision picks the max.
DECISION_ORDER = {"GO": 0, "REVIEW": 1, "NO-GO": 2}


def worst_decision(*decisions: str | None) -> str:
    """Return the most severe decision among the arguments.

    None / unknown values are treated as "GO" (least severe) so a missing
    signal never silently escalates a verdict. With no arguments, returns "GO".
    """
    selected = "GO"
    for decision in decisions:
        value = str(decision or "GO")
        if DECISION_ORDER.get(value, 0) > DECISION_ORDER[selected]:
            selected = value
    return selected


def decision_from_score(
    score: float | None,
    *,
    go_threshold: float = 0.85,
    review_threshold: float = 0.60,
) -> str:
    """Map a 0..1 score to GO / REVIEW / NO-GO. Missing score => REVIEW.

    Thresholds are inclusive lower bounds: score >= go_threshold => GO,
    score >= review_threshold => REVIEW, else NO-GO.
    """
    if score is None:
        return "REVIEW"
    if score >= go_threshold:
        return "GO"
    if score >= review_threshold:
        return "REVIEW"
    return "NO-GO"


def decision_score(decision: str | None) -> float:
    """Map a decision back to a representative 0..1 score. Unknown => 0.5."""
    value = str(decision or "REVIEW")
    if value == "GO":
        return 1.0
    if value == "NO-GO":
        return 0.0
    return 0.5


def _self_test() -> int:
    # worst_decision: picks the most severe; None/unknown treated as GO; default GO.
    assert worst_decision("GO", "REVIEW", "NO-GO") == "NO-GO"
    assert worst_decision("GO", "REVIEW") == "REVIEW"
    assert worst_decision("GO", "GO") == "GO"
    assert worst_decision() == "GO"
    assert worst_decision("GO", None) == "GO"
    assert worst_decision(None, "NO-GO") == "NO-GO"
    assert worst_decision("bogus", "REVIEW") == "REVIEW"  # unknown ranks as GO(0)

    # decision_from_score: inclusive lower-bound thresholds; None => REVIEW.
    assert decision_from_score(None) == "REVIEW"
    assert decision_from_score(0.85) == "GO"      # boundary, inclusive
    assert decision_from_score(0.84) == "REVIEW"
    assert decision_from_score(0.60) == "REVIEW"  # boundary, inclusive
    assert decision_from_score(0.59) == "NO-GO"
    assert decision_from_score(0.0) == "NO-GO"
    assert decision_from_score(0.5, go_threshold=0.4, review_threshold=0.2) == "GO"

    # decision_score: inverse mapping; unknown/None => 0.5.
    assert decision_score("GO") == 1.0
    assert decision_score("NO-GO") == 0.0
    assert decision_score("REVIEW") == 0.5
    assert decision_score(None) == 0.5
    assert decision_score("bogus") == 0.5

    print("decisions self-test ok")
    return 0


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    sys.stderr.write("decisions.py is a library module; run with --self-test.\n")
    raise SystemExit(2)
