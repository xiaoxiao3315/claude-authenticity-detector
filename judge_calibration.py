"""Judge golden-set calibration.

The judge model is the thing that turns a tested model's answer into a
GO / REVIEW / NO-GO decision and a 0-10 score (see eval_cli.final_score_from_judge).
But who checks the judge? If the gateway silently swaps / downgrades the judge
model, scores drift and every campaign verdict is quietly corrupted.

A golden-set is a small, fixed bank of cases where a human has already decided
the correct outcome:

    case = {
        "id": ...,
        "task": { ... a normal eval task ... },
        "candidate_answer": "...",      # the answer the judge will grade
        "expected_decision": "GO" | "REVIEW" | "NO-GO",
        "expected_score_band": [lo, hi] (optional),  # acceptable 0-10 range
        "rationale": "why a human ruled this way",
    }

This module is split in two:

  1. OFFLINE (this file's core): pure functions that take the judge's observed
     decisions/scores and the golden expectations, and compute agreement metrics
     — accuracy, false-GO / false-NO-GO rates, confusion matrix, Cohen's kappa,
     score-band hit rate. Fully self-testable with synthetic data, no API calls.

  2. LIVE (run from eval_cli, gated behind --live): actually send each golden
     case to the judge model, parse its JSON, and feed the observed decisions
     into the offline metrics. Costs gateway quota — user-triggered only.

Design rules followed:
- No clock / network here; deterministic and self-testable.
- Reuses authenticity.read_json/write_json for IO.
- Never store secrets. Golden cases are authored content, not live responses.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from authenticity import numeric, read_json, write_json

JUDGE_GOLDEN_SCHEMA_VERSION = "judge_golden_set_v1"
JUDGE_CALIBRATION_RESULT_VERSION = "judge_calibration_v1"

ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDEN_DIR = ROOT / "judge_golden"

DECISIONS = ("GO", "REVIEW", "NO-GO")

# Which confusions are dangerous. A judge that says GO when truth is NO-GO is a
# FALSE GO (a bad model passes) — the most expensive error for a release gate.
# The reverse (NO-GO when truth is GO) is a FALSE NO-GO (a good model blocked).


def normalize_decision(value: Any) -> str | None:
    """Map a raw decision string to one of GO / REVIEW / NO-GO, else None."""
    if value is None:
        return None
    token = str(value).strip().upper().replace("_", "-")
    if token in {"NO-GO", "NOGO", "NO GO", "BLOCK", "REJECT"}:
        return "NO-GO"
    if token in {"GO", "PASS", "ACCEPT"}:
        return "GO"
    if token in {"REVIEW", "MAYBE", "HOLD"}:
        return "REVIEW"
    return None


def validate_golden_set(doc: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems (empty = valid)."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["golden-set must be a JSON object"]
    cases = doc.get("cases")
    if not isinstance(cases, list) or not cases:
        return ["golden-set must have a non-empty 'cases' array"]
    seen_ids: set[str] = set()
    for i, case in enumerate(cases):
        where = f"case[{i}]"
        if not isinstance(case, dict):
            problems.append(f"{where}: not an object")
            continue
        cid = case.get("id")
        if not cid:
            problems.append(f"{where}: missing 'id'")
        elif cid in seen_ids:
            problems.append(f"{where}: duplicate id '{cid}'")
        else:
            seen_ids.add(cid)
        if not isinstance(case.get("task"), dict):
            problems.append(f"{where} ({cid}): missing 'task' object")
        if not str(case.get("candidate_answer") or "").strip():
            problems.append(f"{where} ({cid}): missing 'candidate_answer'")
        exp = normalize_decision(case.get("expected_decision"))
        if exp is None:
            problems.append(f"{where} ({cid}): expected_decision must be GO / REVIEW / NO-GO")
        band = case.get("expected_score_band")
        if band is not None:
            lo = numeric(band[0]) if isinstance(band, (list, tuple)) and len(band) == 2 else None
            hi = numeric(band[1]) if isinstance(band, (list, tuple)) and len(band) == 2 else None
            if (not isinstance(band, (list, tuple)) or len(band) != 2
                    or lo is None or hi is None or lo > hi):
                problems.append(f"{where} ({cid}): expected_score_band must be [lo, hi] with lo<=hi")
    return problems


def load_golden_set(path: Path) -> dict[str, Any]:
    doc = read_json(path)
    problems = validate_golden_set(doc)
    if problems:
        raise ValueError("invalid golden-set:\n  - " + "\n  - ".join(problems))
    return doc


def _confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """pairs = list of (expected, observed). Returns matrix[expected][observed]."""
    matrix = {e: {o: 0 for o in DECISIONS} for e in DECISIONS}
    for expected, observed in pairs:
        if expected in matrix and observed in matrix[expected]:
            matrix[expected][observed] += 1
    return matrix


def _cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa over the 3 decision classes. None if undefined.

    kappa = (p_observed - p_expected) / (1 - p_expected). 1.0 = perfect,
    0 = chance-level, <0 = worse than chance."""
    n = len(pairs)
    if n == 0:
        return None
    agree = sum(1 for e, o in pairs if e == o)
    p_o = agree / n
    exp_counts = {d: 0 for d in DECISIONS}
    obs_counts = {d: 0 for d in DECISIONS}
    for e, o in pairs:
        if e in exp_counts:
            exp_counts[e] += 1
        if o in obs_counts:
            obs_counts[o] += 1
    p_e = sum((exp_counts[d] / n) * (obs_counts[d] / n) for d in DECISIONS)
    if abs(1.0 - p_e) < 1e-9:
        # All in one class with perfect agreement -> kappa is degenerate; report 1.0.
        return 1.0 if p_o >= 1.0 else 0.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def compute_calibration(
    cases: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Core offline metric. Pure function.

    cases: golden cases (each with id, expected_decision, optional expected_score_band).
    observations: list of {id, observed_decision, observed_score, ok, error}.
      ok=False or a missing/invalid decision counts as 'unscored' (judge failed
      to produce a usable verdict) — NOT silently treated as agreement.

    Returns accuracy over scored cases, false-GO / false-NO-GO rates, the
    confusion matrix, Cohen's kappa, score-band hit rate, and a per-case trace.
    """
    obs_by_id = {str(o.get("id")): o for o in observations}
    expected_by_id = {str(c.get("id")): c for c in cases}
    # Duplicate ids would silently collapse in the dict above and undercount
    # total_cases without any error. A pure metric must fail loudly on bad input
    # rather than quietly drop cases. (load_golden_set also guards this, but
    # compute_calibration is callable directly.)
    if len(expected_by_id) != len(cases):
        counts = Counter(str(c.get("id")) for c in cases)
        dups = sorted(cid for cid, n in counts.items() if n > 1)
        raise ValueError(f"duplicate case id(s) in golden-set: {dups}")

    pairs: list[tuple[str, str]] = []
    per_case: list[dict[str, Any]] = []
    unscored = 0
    band_total = 0
    band_hits = 0
    false_go = 0          # expected NO-GO, judge said GO
    false_nogo = 0        # expected GO, judge said NO-GO
    nogo_truth = 0
    go_truth = 0

    for cid, case in expected_by_id.items():
        expected = normalize_decision(case.get("expected_decision"))
        obs = obs_by_id.get(cid)
        observed = normalize_decision(obs.get("observed_decision")) if obs else None
        ok = bool(obs.get("ok", True)) if obs else False
        score = numeric(obs.get("observed_score")) if obs else None

        if expected == "NO-GO":
            nogo_truth += 1
        elif expected == "GO":
            go_truth += 1

        if not ok or observed is None or expected is None:
            unscored += 1
            per_case.append({
                "id": cid, "expected": expected, "observed": observed,
                "agree": None, "unscored": True,
                "error": (obs.get("error") if obs else "no observation"),
            })
            continue

        agree = expected == observed
        pairs.append((expected, observed))
        if expected == "NO-GO" and observed == "GO":
            false_go += 1
        if expected == "GO" and observed == "NO-GO":
            false_nogo += 1

        band = case.get("expected_score_band")
        band_ok = None
        if band is not None and score is not None:
            band_total += 1
            lo, hi = numeric(band[0]), numeric(band[1])
            band_ok = lo is not None and hi is not None and lo <= score <= hi
            if band_ok:
                band_hits += 1

        per_case.append({
            "id": cid, "expected": expected, "observed": observed,
            "agree": agree, "unscored": False,
            "observed_score": score, "score_band_ok": band_ok,
        })

    scored = len(pairs)
    accuracy = round(sum(1 for e, o in pairs if e == o) / scored, 4) if scored else None

    return {
        "result_version": JUDGE_CALIBRATION_RESULT_VERSION,
        "total_cases": len(expected_by_id),
        "scored_cases": scored,
        "unscored_cases": unscored,
        "accuracy": accuracy,
        "cohens_kappa": _cohens_kappa(pairs),
        "false_go_count": false_go,
        "false_go_rate": round(false_go / nogo_truth, 4) if nogo_truth else None,
        "false_nogo_count": false_nogo,
        "false_nogo_rate": round(false_nogo / go_truth, 4) if go_truth else None,
        "score_band_total": band_total,
        "score_band_hit_rate": round(band_hits / band_total, 4) if band_total else None,
        "confusion_matrix": _confusion_matrix(pairs),
        "per_case": per_case,
    }


# Calibration verdict thresholds. Tunable; these are deliberately lenient for a
# first cut so a noisy run is REVIEW rather than a false alarm.
JUDGE_RELIABLE = "judge_reliable"
JUDGE_REVIEW = "judge_needs_review"
JUDGE_UNRELIABLE = "judge_unreliable"
JUDGE_INSUFFICIENT = "insufficient_evidence"


def classify_judge(result: dict[str, Any], *, min_scored: int = 4) -> dict[str, Any]:
    """Turn calibration metrics into a verdict + reasons.

    - Any false-GO (a bad model passed) is the worst error -> unreliable.
    - Too few scored cases -> insufficient_evidence (don't over-claim).
    - High accuracy + no false-GO -> reliable; middling -> review.
    """
    scored = int(result.get("scored_cases") or 0)
    if scored < min_scored:
        return {"verdict": JUDGE_INSUFFICIENT, "reasons": [f"only {scored} scored cases (< {min_scored})"]}

    reasons: list[str] = []
    acc = result.get("accuracy")
    false_go = int(result.get("false_go_count") or 0)
    false_nogo_rate = result.get("false_nogo_rate")
    kappa = result.get("cohens_kappa")

    if false_go > 0:
        reasons.append(f"{false_go} false-GO (judge passed a should-block answer)")
        return {"verdict": JUDGE_UNRELIABLE, "reasons": reasons, "accuracy": acc, "cohens_kappa": kappa}

    # A high false-NO-GO rate is less dangerous than a false-GO (it blocks good
    # models rather than passing bad ones) but still makes the judge untrustworthy
    # as a gate — cap it at needs_review even if accuracy looks ok.
    FALSE_NOGO_REVIEW_CEIL = 0.34  # >1/3 of good answers wrongly blocked
    if false_nogo_rate is not None and false_nogo_rate > FALSE_NOGO_REVIEW_CEIL:
        reasons.append(f"false-NO-GO rate {false_nogo_rate} (judge over-blocks good answers)")
        return {"verdict": JUDGE_REVIEW, "reasons": reasons, "accuracy": acc, "cohens_kappa": kappa}

    if acc is not None and acc >= 0.9 and (kappa is None or kappa >= 0.6):
        reasons.append(f"accuracy {acc}, kappa {kappa}, no false-GO")
        return {"verdict": JUDGE_RELIABLE, "reasons": reasons, "accuracy": acc, "cohens_kappa": kappa}

    reasons.append(f"accuracy {acc}, kappa {kappa} below the reliable bar")
    return {"verdict": JUDGE_REVIEW, "reasons": reasons, "accuracy": acc, "cohens_kappa": kappa}


def render_calibration_report(result: dict[str, Any], verdict: dict[str, Any]) -> str:
    """Human-readable Chinese calibration report."""
    label = {
        JUDGE_RELIABLE: "✅ 评审模型可信",
        JUDGE_REVIEW: "⚠️ 评审模型需复核",
        JUDGE_UNRELIABLE: "❌ 评审模型不可信",
        JUDGE_INSUFFICIENT: "❔ 证据不足",
    }.get(str(verdict.get("verdict")), verdict.get("verdict"))

    lines = ["=" * 48, "Judge 金标校准报告", "=" * 48]
    lines.append(f"判定: {label}")
    lines.append(f"用例: 共 {result.get('total_cases')}，已评 {result.get('scored_cases')}，未评 {result.get('unscored_cases')}")
    lines.append(f"准确率: {result.get('accuracy')}    Cohen's kappa: {result.get('cohens_kappa')}")
    fg, fgr = result.get("false_go_count"), result.get("false_go_rate")
    fn, fnr = result.get("false_nogo_count"), result.get("false_nogo_rate")
    lines.append(f"误放行 (False-GO): {fg} (率 {fgr})   误拦截 (False-NO-GO): {fn} (率 {fnr})")
    if result.get("score_band_total"):
        lines.append(f"分数落入期望区间命中率: {result.get('score_band_hit_rate')} (共 {result.get('score_band_total')})")
    for r in verdict.get("reasons", []):
        lines.append(f"理由: {r}")
    # surface disagreements
    bad = [c for c in result.get("per_case", []) if c.get("agree") is False]
    if bad:
        lines.append("分歧用例:")
        for c in bad:
            lines.append(f"  - {c['id']}: 期望 {c['expected']} / judge {c['observed']}")
    unscored = [c for c in result.get("per_case", []) if c.get("unscored")]
    if unscored:
        lines.append(f"未评用例: {len(unscored)} 个（judge 调用失败或返回无法解析）")
    lines.append("=" * 48)
    return "\n".join(lines)


def _sample_golden_set() -> dict[str, Any]:
    """A tiny built-in golden-set scaffold so users have a starting template.
    These are illustrative cases; expand with real authored cases before trusting
    a live verdict."""
    return {
        "schema_version": JUDGE_GOLDEN_SCHEMA_VERSION,
        "description": "Starter judge calibration golden-set. Replace/extend with real authored cases.",
        "cases": [
            {
                "id": "golden_json_exact_pass",
                "task": {
                    "id": "g_json_1", "category": "json_stability", "difficulty": "easy",
                    "scoring_type": "json_exact",
                    "prompt": "只输出一个 JSON object: {\"status\": \"ok\", \"count\": 3}，不要任何解释。",
                    "expected_json": {"status": "ok", "count": 3},
                },
                "candidate_answer": "{\"status\": \"ok\", \"count\": 3}",
                "expected_decision": "GO",
                "expected_score_band": [8, 10],
                "rationale": "Exact JSON match, no extra text — a clear pass.",
            },
            {
                "id": "golden_json_wrong_value",
                "task": {
                    "id": "g_json_2", "category": "json_stability", "difficulty": "easy",
                    "scoring_type": "json_exact",
                    "prompt": "只输出一个 JSON object: {\"status\": \"ok\", \"count\": 3}，不要任何解释。",
                    "expected_json": {"status": "ok", "count": 3},
                },
                "candidate_answer": "{\"status\": \"ok\", \"count\": 99}",
                "expected_decision": "NO-GO",
                "expected_score_band": [0, 3],
                "rationale": "Wrong value (count 99 vs 3) — must be blocked.",
            },
            {
                "id": "golden_json_with_markdown",
                "task": {
                    "id": "g_json_3", "category": "json_stability", "difficulty": "easy",
                    "scoring_type": "json_exact",
                    "prompt": "只输出一个 JSON object，不要 Markdown: {\"ok\": true}",
                    "expected_json": {"ok": True},
                },
                "candidate_answer": "```json\n{\"ok\": true}\n```",
                "expected_decision": "REVIEW",
                "expected_score_band": [4, 7],
                "rationale": "Correct content but wrapped in markdown despite explicit instruction — borderline.",
            },
            {
                "id": "golden_partial_answer",
                "task": {
                    "id": "g_qa_1", "category": "reasoning", "difficulty": "medium",
                    "scoring_type": "judge",
                    "prompt": "列出 HTTP 429 状态码的含义，以及客户端应如何正确处理（至少两点）。",
                },
                "candidate_answer": "429 表示请求过多。",
                "expected_decision": "NO-GO",
                "expected_score_band": [0, 4],
                "rationale": "Missing the required handling steps — incomplete.",
            },
        ],
    }


def _self_test() -> None:
    # 1. decision normalization
    assert normalize_decision("no_go") == "NO-GO"
    assert normalize_decision("Go") == "GO"
    assert normalize_decision("garbage") is None

    # 2. golden-set validation
    gs = _sample_golden_set()
    assert validate_golden_set(gs) == [], validate_golden_set(gs)
    bad = {"cases": [{"id": "x", "task": {}, "candidate_answer": "", "expected_decision": "MAYBE-NOT"}]}
    assert validate_golden_set(bad), "should catch missing answer + bad decision"
    assert validate_golden_set({"cases": []}), "empty cases invalid"

    # 2b. compute_calibration fails loudly on duplicate ids (no silent undercount)
    try:
        compute_calibration(
            [{"id": "d", "expected_decision": "GO"}, {"id": "d", "expected_decision": "NO-GO"}],
            [{"id": "d", "observed_decision": "GO", "ok": True}],
        )
        raise AssertionError("expected duplicate-id ValueError")
    except ValueError as exc:
        assert "duplicate case id" in str(exc), exc

    cases = gs["cases"]

    # 3. perfect agreement -> reliable
    perfect = [
        {"id": c["id"], "observed_decision": c["expected_decision"],
         "observed_score": (c["expected_score_band"][0] + c["expected_score_band"][1]) / 2, "ok": True}
        for c in cases
    ]
    res = compute_calibration(cases, perfect)
    assert res["accuracy"] == 1.0, res
    assert res["false_go_count"] == 0 and res["false_nogo_count"] == 0
    assert res["score_band_hit_rate"] == 1.0, res
    assert res["cohens_kappa"] == 1.0, res
    v = classify_judge(res)
    assert v["verdict"] == JUDGE_RELIABLE, v

    # 4. a false-GO (judge passes a should-block answer) -> unreliable
    fg_obs = []
    for c in cases:
        d = c["expected_decision"]
        if c["id"] == "golden_json_wrong_value":
            d = "GO"  # judge wrongly passes a NO-GO case
        fg_obs.append({"id": c["id"], "observed_decision": d, "observed_score": 9, "ok": True})
    res2 = compute_calibration(cases, fg_obs)
    assert res2["false_go_count"] == 1, res2
    assert res2["false_go_rate"] is not None
    v2 = classify_judge(res2)
    assert v2["verdict"] == JUDGE_UNRELIABLE, v2

    # 5. failed judge calls -> unscored, not silently agreed
    failed = [{"id": c["id"], "observed_decision": None, "ok": False, "error": "429"} for c in cases]
    res3 = compute_calibration(cases, failed)
    assert res3["scored_cases"] == 0 and res3["unscored_cases"] == len(cases), res3
    assert classify_judge(res3)["verdict"] == JUDGE_INSUFFICIENT, res3

    # 5b. high false-NO-GO (over-blocking good answers) -> needs_review, not reliable,
    #     even with no false-GO. Synthetic: 6 GO-truth cases, judge blocks 3 of them.
    nogo_cases = [{"id": f"go_{i}", "task": {"id": f"t{i}"}, "candidate_answer": "x",
                   "expected_decision": "GO"} for i in range(6)]
    over_block = [
        {"id": f"go_{i}", "observed_decision": ("NO-GO" if i < 3 else "GO"), "ok": True}
        for i in range(6)
    ]
    res_ob = compute_calibration(nogo_cases, over_block)
    assert res_ob["false_go_count"] == 0 and res_ob["false_nogo_count"] == 3, res_ob
    assert res_ob["false_nogo_rate"] == 0.5, res_ob
    v_ob = classify_judge(res_ob)
    assert v_ob["verdict"] == JUDGE_REVIEW, v_ob  # over-blocking caps at review

    # 6. report renders for a verdict
    rep = render_calibration_report(res2, v2)
    assert "不可信" in rep and "False-GO" in rep, rep
    assert "可信" in render_calibration_report(res, v)

    # 7. confusion matrix sums to scored count
    total_in_matrix = sum(res["confusion_matrix"][e][o] for e in DECISIONS for o in DECISIONS)
    assert total_in_matrix == res["scored_cases"], res["confusion_matrix"]

    # 8. write/read round-trip + validation on load
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "gs.json"
        write_json(p, gs)
        loaded = load_golden_set(p)
        assert loaded["cases"][0]["id"] == cases[0]["id"]

    print("judge_calibration self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge golden-set calibration (offline metrics)")
    parser.add_argument("--self-test", action="store_true", help="run internal self-tests")
    parser.add_argument("--emit-sample", type=Path, help="write a starter golden-set template to this path")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.emit_sample:
        write_json(args.emit_sample, _sample_golden_set())
        print(f"wrote starter golden-set to {args.emit_sample}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
