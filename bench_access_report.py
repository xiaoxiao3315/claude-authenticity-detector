"""Access assessment report: tiered TTFT / OTPS / RPM benchmark with grading.

Reproduces the structure of third-party gateway "access assessment" reports:
per-input-size tiers (small/medium/large) run at fixed concurrency, TTFT P50/P90
and output tokens/s are scored against baseline (pass) and perfect (full-marks)
thresholds, plus an optional RPM/TPM saturation probe. Everything folds into a
V_TOTAL 0-100 score, a letter grade, and an accept/reject verdict. Response
envelopes (id prefix / server model) are captured on the side as authenticity
evidence; a server-model mismatch flags a suspected non-official upstream
(run `eval_cli.py quickcheck` for the full authenticity verdict).

Usage:
    python bench_access_report.py --provider gpt_az --tiers small --samples 4 --concurrency 2 --skip-rpm
    python bench_access_report.py --provider tested_model            # full run, image-equivalent tiers
Results land in benchmarks/access_<provider>_<timestamp>.json plus a markdown report on stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from eval_cli import load_model_config
from local_env import load_local_env
from model_client import ModelConfig, auth_headers, auth_value

ROOT = Path(__file__).resolve().parent
PROVIDERS_PATH = ROOT / "configs" / "providers.local.json"
OUT_DIR = ROOT / "benchmarks"

PAD_PARAGRAPH = (
    "Gateway acceptance testing separates first-token latency from sustained "
    "throughput because they stress different parts of the serving stack: "
    "queueing and prefill dominate the former, decode scheduling the latter. "
)  # ~40 tokens; repeated to hit tier input sizes

ANSWER_INSTRUCTION = (
    "\n\nIgnore the repeated filler above. Reply with a short paragraph (about "
    "120 words) on why TTFT and tokens/s must be measured separately."
)

CHARS_PER_TOKEN = 4.2  # rough English estimate for padding; actual usage is recorded


@dataclass
class Threshold:
    baseline: float          # pass line (score 60)
    perfect: float           # full marks (score 100)
    lower_is_better: bool


@dataclass
class Tier:
    name: str
    input_tokens: int
    concurrency: int
    samples: int
    ttft_p50: Threshold
    ttft_p90: Threshold
    otps: Threshold          # output tokens/s per stream


# Thresholds mirror the published access-assessment baselines:
#   <8k @50cc:      TTFT baseline 2.5/6s, perfect 1/2.5s;  OTPS baseline 31.25 tok/s (32ms/tok), perfect 66.7 (15ms/tok)
#   32k-64k @10cc:  TTFT baseline 6.5/14s, perfect 3/6s;   OTPS baseline 14, perfect 45
#   128k-200k @1cc: TTFT baseline 30/60s, perfect 10/20s;  OTPS baseline 8,  perfect 22
DEFAULT_TIERS: dict[str, Tier] = {
    "small": Tier("small(<8k)", 6000, 50, 50,
                  Threshold(2.5, 1.0, True), Threshold(6.0, 2.5, True), Threshold(31.25, 66.7, False)),
    "medium": Tier("medium(32k-64k)", 48000, 10, 10,
                   Threshold(6.5, 3.0, True), Threshold(14.0, 6.0, True), Threshold(14.0, 45.0, False)),
    "large": Tier("large(128k-200k)", 160000, 1, 2,
                  Threshold(30.0, 10.0, True), Threshold(60.0, 20.0, True), Threshold(8.0, 22.0, False)),
}

RPM_CAP = 600.0
TPM_CAP = 6_000_000.0

GRADE_BANDS = [(85.0, "A"), (70.0, "B"), (60.0, "C")]  # else D


def load_provider(name: str) -> ModelConfig:
    raw = json.loads(PROVIDERS_PATH.read_text(encoding="utf-8"))
    if name not in raw:
        raise SystemExit(f"provider '{name}' not in {PROVIDERS_PATH.name}; have: {', '.join(sorted(raw))}")
    return load_model_config(raw[name], name)


def build_prompt(input_tokens: int) -> str:
    pad_tokens = max(0, input_tokens - 60)
    reps = max(1, round(pad_tokens * CHARS_PER_TOKEN / len(PAD_PARAGRAPH)))
    return PAD_PARAGRAPH * reps + ANSWER_INSTRUCTION


def request_spec(cfg: ModelConfig, prompt: str, max_tokens: int) -> tuple[str, dict[str, str], dict[str, Any]]:
    secret = auth_value(cfg)
    if cfg.protocol == "anthropic_messages":
        url = f"{cfg.base_url}/v1/messages"
        headers = {**auth_headers(cfg, secret), "anthropic-version": "2023-06-01", "content-type": "application/json"}
        payload = {"model": cfg.model, "max_tokens": max_tokens, "stream": True,
                   "messages": [{"role": "user", "content": prompt}]}
    else:
        url = f"{cfg.base_url}/v1/chat/completions"
        headers = {**auth_headers(cfg, secret), "content-type": "application/json"}
        payload = {"model": cfg.model, "max_tokens": max_tokens, "stream": True,
                   "stream_options": {"include_usage": True},
                   "messages": [{"role": "user", "content": prompt}]}
    return url, headers, payload


async def stream_once(client: httpx.AsyncClient, cfg: ModelConfig, prompt: str, max_tokens: int) -> dict[str, Any]:
    url, headers, payload = request_spec(cfg, prompt, max_tokens)
    anthropic = cfg.protocol == "anthropic_messages"
    t0 = time.perf_counter()
    ttft: float | None = None
    chars = 0
    in_tok: int | None = None
    out_tok: int | None = None
    envelope: dict[str, Any] = {}
    try:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                body = raw.decode("utf-8", "replace")[:200]
                etype = None
                try:
                    etype = (json.loads(body).get("error") or {}).get("type")
                except Exception:
                    pass
                return {"ok": False, "error": f"http {resp.status_code}",
                        "status": resp.status_code, "error_type": etype,
                        "error_body": body, "total_s": time.perf_counter() - t0}
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if anthropic:
                    etype = event.get("type")
                    if etype == "message_start":
                        msg = event.get("message") or {}
                        envelope = {"id_prefix": str(msg.get("id", ""))[:12], "model": msg.get("model")}
                        in_tok = (msg.get("usage") or {}).get("input_tokens")
                    elif etype == "content_block_delta":
                        text = (event.get("delta") or {}).get("text")
                        if text:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            chars += len(text)
                    elif etype == "message_delta":
                        out_tok = (event.get("usage") or {}).get("output_tokens") or out_tok
                else:
                    if not envelope and event.get("id"):
                        envelope = {"id_prefix": str(event["id"])[:12], "model": event.get("model")}
                    usage = event.get("usage")
                    if usage:
                        in_tok = usage.get("prompt_tokens") or in_tok
                        out_tok = usage.get("completion_tokens") or out_tok
                    for choice in event.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            chars += len(delta)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": type(exc).__name__, "total_s": time.perf_counter() - t0}
    total = time.perf_counter() - t0
    tokens = out_tok if out_tok else round(chars / CHARS_PER_TOKEN)
    gen_window = total - (ttft or total)
    return {
        "ok": ttft is not None,
        "error": None if ttft is not None else "no content received",
        "ttft_s": ttft, "total_s": total,
        "input_tokens": in_tok, "output_tokens": tokens, "tokens_estimated": out_tok is None,
        "tokens_per_s": (tokens / gen_window) if (tokens and gen_window > 0.05) else None,
        "envelope": envelope,
    }


async def run_tier(cfg: ModelConfig, tier: Tier, max_tokens: int, timeout: float,
                   budget: dict[str, float] | None = None) -> list[dict[str, Any]]:
    prompt = build_prompt(tier.input_tokens)
    sem = asyncio.Semaphore(tier.concurrency)
    limits = httpx.Limits(max_connections=tier.concurrency + 5)
    done = 0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def one(i: int) -> dict[str, Any]:
            nonlocal done
            if budget and budget.get("cap", 0) and budget["spent"] >= budget["cap"]:
                done += 1
                print(f"  [{tier.name}] {done:>3}/{tier.samples} SKIPPED (cost cap ${budget['cap']:.2f} hit)", flush=True)
                return {"ok": False, "error": "cost_cap_skipped", "status": None, "total_s": 0.0}
            async with sem:
                r = await stream_once(client, cfg, prompt, max_tokens)
            if budget is not None:
                cin = (r.get("input_tokens") or 0) / 1e6 * budget.get("price_in", 0)
                cout = (r.get("output_tokens") or 0) / 1e6 * budget.get("price_out", 0)
                budget["spent"] += cin + cout
            r["sample"] = i
            done += 1
            state = f"ttft={r['ttft_s']:.2f}s tok/s={round(r['tokens_per_s'], 1) if r.get('tokens_per_s') else '?'}" \
                if r["ok"] else f"ERROR {r['error']}"
            spent_note = f" spent=${budget['spent']:.3f}" if budget else ""
            print(f"  [{tier.name}] {done:>3}/{tier.samples} {state}{spent_note}", flush=True)
            return r

        return list(await asyncio.gather(*(one(i) for i in range(1, tier.samples + 1))))


async def run_rpm_probe(cfg: ModelConfig, workers: int, window_s: float, timeout: float) -> dict[str, Any]:
    """Saturation probe: short requests from N workers for a fixed window."""
    prompt = "Reply with the single word: ok"
    deadline = time.perf_counter() + window_s
    ok_count = 0
    err_count = 0
    tokens_total = 0
    limits = httpx.Limits(max_connections=workers + 2)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def worker() -> None:
            nonlocal ok_count, err_count, tokens_total
            while time.perf_counter() < deadline:
                r = await stream_once(client, cfg, prompt, 8)
                if r["ok"]:
                    ok_count += 1
                    tokens_total += (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)
                else:
                    err_count += 1

        await asyncio.gather(*(worker() for _ in range(workers)))

    rpm = ok_count * 60.0 / window_s
    tpm = tokens_total * 60.0 / window_s
    return {"workers": workers, "window_s": window_s, "ok": ok_count, "errors": err_count,
            "rpm_measured": round(rpm, 1), "tpm_measured": round(tpm, 1)}


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[idx]


def score_metric(measured: float | None, th: Threshold) -> float | None:
    """Linear score: perfect=100, baseline=60, 0 at 2x-baseline-worse. Clamped."""
    if measured is None:
        return None
    if th.lower_is_better:
        if measured <= th.perfect:
            return 100.0
        if measured <= th.baseline:
            return 60.0 + 40.0 * (th.baseline - measured) / (th.baseline - th.perfect)
        return max(0.0, 60.0 * (2 * th.baseline - measured) / th.baseline)
    if measured >= th.perfect:
        return 100.0
    if measured >= th.baseline:
        return 60.0 + 40.0 * (measured - th.baseline) / (th.perfect - th.baseline)
    return max(0.0, 60.0 * measured / th.baseline)


def grade(v_total: float) -> str:
    for floor, letter in GRADE_BANDS:
        if v_total >= floor:
            return letter
    return "D"


def evaluate(cfg: ModelConfig, tier_runs: dict[str, list[dict[str, Any]]],
             rpm: dict[str, Any] | None, tiers: dict[str, Tier]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    envelopes: set[str] = set()
    server_models: set[str] = set()

    for key, runs in tier_runs.items():
        tier = tiers[key]
        ok = [r for r in runs if r["ok"]]
        for r in ok:
            if r.get("envelope"):
                envelopes.add(json.dumps(r["envelope"], ensure_ascii=False, sort_keys=True))
                if r["envelope"].get("model"):
                    server_models.add(str(r["envelope"]["model"]))
        ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
        tps = [r["tokens_per_s"] for r in ok if r.get("tokens_per_s")]
        in_toks = [r["input_tokens"] for r in ok if r.get("input_tokens")]
        for label, measured, th in (
            (f"TTFT P50 {tier.name}", pct(ttfts, 0.50), tier.ttft_p50),
            (f"TTFT P90 {tier.name}", pct(ttfts, 0.90), tier.ttft_p90),
            (f"OTPS {tier.name}", statistics.median(tps) if tps else None, tier.otps),
        ):
            s = score_metric(measured, th)
            items.append({
                "item": label, "measured": round(measured, 2) if measured is not None else None,
                "baseline": th.baseline, "perfect": th.perfect,
                "unit": "s" if th.lower_is_better else "tok/s",
                "score": round(s, 2) if s is not None else None,
                "pass": (s is not None and s >= 60.0),
            })
        errs = [r for r in runs if not r["ok"]]
        status_breakdown: dict[str, int] = {}
        for r in errs:
            key = str(r.get("error_type") or r.get("status") or r.get("error") or "unknown")
            status_breakdown[key] = status_breakdown.get(key, 0) + 1
        note_parts = []
        if in_toks:
            note_parts.append(f"input tokens median {statistics.median(in_toks):.0f}")
        if status_breakdown:
            note_parts.append("errors: " + ", ".join(f"{k}×{v}" for k, v in sorted(status_breakdown.items())))
        items.append({
            "item": f"errors {tier.name}", "measured": len(runs) - len(ok),
            "baseline": 0, "perfect": 0, "unit": "count",
            "score": 100.0 if len(runs) == len(ok) else round(max(0.0, 100.0 * len(ok) / len(runs)), 2),
            "pass": len(runs) == len(ok),
            "error_breakdown": status_breakdown or None,
            "note": "; ".join(note_parts) if note_parts else None,
        })

    if rpm is not None:
        for label, measured, cap in (("RPM vs cap", rpm["rpm_measured"], RPM_CAP),
                                     ("TPM vs cap", rpm["tpm_measured"], TPM_CAP)):
            th = Threshold(baseline=cap, perfect=cap, lower_is_better=False)
            s = min(100.0, 100.0 * measured / cap)
            items.append({"item": label, "measured": measured, "baseline": cap, "perfect": cap,
                          "unit": "per-min", "score": round(s, 2), "pass": measured >= cap})

    scored = [i for i in items if i["score"] is not None]
    v_total = round(sum(i["score"] for i in scored) / len(scored), 2) if scored else 0.0
    coverage = f"{len(scored)}/{len(items)}"

    veto_notes: list[str] = []
    mismatched = {m for m in server_models if m and cfg.model not in m and m not in cfg.model}
    if mismatched:
        veto_notes.append(f"server-reported model {sorted(mismatched)} does not match requested '{cfg.model}' "
                          f"— suspected non-official upstream; run eval_cli.py quickcheck for the full verdict")

    # Response-id family: `resp_` on an OpenAI-chat path means the ChatGPT/Codex
    # Responses API backend (chatgpt.com/backend-api), not the standard OpenAI API
    # (which returns `chatcmpl-`). This convicts as a resold consumer/Pro account pool.
    id_prefixes = {json.loads(e).get("id_prefix", "") for e in envelopes}
    if any(p.startswith("resp_") for p in id_prefixes) and cfg.protocol == "openai_chat":
        veto_notes.append("response-id family 'resp_' on an OpenAI-chat endpoint — this is the ChatGPT/Codex "
                          "Responses API backend, not the standard OpenAI API (chatcmpl-); "
                          "suspected reverse-engineered ChatGPT account pool")

    # Size-conditional degradation (the toy-passes / real-fails signal): if a larger
    # tier's error rate is >=50% while a smaller tier succeeds (<=10% errors), the
    # backend serves toy requests honestly but collapses on real-size ones.
    tier_err: dict[str, float] = {}
    tier_id_families: dict[str, set[str]] = {}
    for key, runs in tier_runs.items():
        if runs:
            tier_err[key] = sum(1 for r in runs if not r["ok"]) / len(runs)
        fams = {str((r.get("envelope") or {}).get("id_prefix", ""))[:4]
                for r in runs if r.get("ok") and (r.get("envelope") or {}).get("id_prefix")}
        if fams:
            tier_id_families[key] = fams
    order = [k for k in ("small", "medium", "large") if k in tier_err]
    for i, small_k in enumerate(order):
        for big_k in order[i + 1:]:
            if tier_err.get(small_k, 1.0) <= 0.10 and tier_err.get(big_k, 0.0) >= 0.50:
                veto_notes.append(
                    f"size-conditional degradation: '{small_k}' tier err {tier_err[small_k]*100:.0f}% but "
                    f"'{big_k}' tier err {tier_err[big_k]*100:.0f}% — toy requests pass, real-size requests fail "
                    f"(supply-constrained/reverse-engineered upstream)")
    # Size-conditional routing: id-prefix family differs across tiers.
    all_fams = {f for fams in tier_id_families.values() for f in fams}
    if len(all_fams) > 1:
        veto_notes.append(f"size-conditional routing: response-id families differ across tiers "
                          f"{ {k: sorted(v) for k, v in tier_id_families.items()} } — requests routed to "
                          f"different backends by size")

    all_pass = all(i["pass"] for i in items) and not veto_notes
    return {
        "v_total": v_total, "grade": grade(v_total), "coverage": coverage,
        "verdict": "ACCEPTED" if all_pass else "REJECTED",
        "veto": veto_notes, "items": items,
        "envelopes_seen": sorted(envelopes), "server_models": sorted(server_models),
    }


def render_markdown(cfg: ModelConfig, result: dict[str, Any], rpm: dict[str, Any] | None) -> str:
    lines = [
        f"\n## Access Assessment Report — {cfg.provider_id} ({cfg.model})",
        f"\n**Grade {result['grade']}** · V_TOTAL **{result['v_total']}**/100 · coverage {result['coverage']} · "
        f"verdict **{result['verdict']}**\n",
        "| item | measured | baseline(pass) | perfect | score | pass |",
        "|---|---|---|---|---|---|",
    ]
    for i in result["items"]:
        lines.append(f"| {i['item']} | {i['measured']} {i['unit']} | {i['baseline']} | {i['perfect']} "
                     f"| {i['score']} | {'PASS' if i['pass'] else 'FAIL'} |")
        if i.get("note"):
            lines.append(f"|   ↳ {i['note']} | | | | | |")
    if rpm is not None:
        lines.append(f"\nRPM probe: {rpm['ok']} ok / {rpm['errors']} errors in {rpm['window_s']}s "
                     f"with {rpm['workers']} workers")
    if result["veto"]:
        lines.append("\n**VETO / authenticity flags:**")
        lines.extend(f"- {n}" for n in result["veto"])
    lines.append(f"\nenvelopes: {json.dumps(result['envelopes_seen'], ensure_ascii=False)}")
    return "\n".join(lines)


def self_test() -> int:
    th_lat = Threshold(2.5, 1.0, True)
    assert score_metric(1.0, th_lat) == 100.0
    assert score_metric(2.5, th_lat) == 60.0
    assert score_metric(5.0, th_lat) == 0.0
    assert score_metric(10.0, th_lat) == 0.0
    mid = score_metric(1.75, th_lat)
    assert mid is not None and 60.0 < mid < 100.0
    th_tps = Threshold(14.0, 45.0, False)
    assert score_metric(45.0, th_tps) == 100.0
    assert score_metric(14.0, th_tps) == 60.0
    assert score_metric(7.0, th_tps) == 30.0
    assert score_metric(None, th_tps) is None
    assert grade(90) == "A" and grade(72) == "B" and grade(61) == "C" and grade(22.84) == "D"
    assert pct([1.0, 2.0, 3.0, 4.0], 0.5) in (2.0, 3.0)
    prompt = build_prompt(6000)
    assert 5000 * CHARS_PER_TOKEN * 0.8 < len(prompt) < 6500 * CHARS_PER_TOKEN * 1.2
    cfg = ModelConfig(provider_id="p", base_url="https://x", model="claude-opus-4-6",
                      api_key_env="K", protocol="anthropic_messages", auth_type="x-api-key")
    fake_runs = {"small": [
        {"ok": True, "ttft_s": 1.2, "total_s": 5.0, "tokens_per_s": 40.0, "input_tokens": 6100,
         "output_tokens": 150, "envelope": {"id_prefix": "msg_01abc", "model": "claude-opus-4-6"}},
        {"ok": True, "ttft_s": 2.0, "total_s": 6.0, "tokens_per_s": 35.0, "input_tokens": 6100,
         "output_tokens": 150, "envelope": {"id_prefix": "msg_01abd", "model": "claude-opus-4-6"}},
        {"ok": False, "error": "http 500", "total_s": 1.0},
    ]}
    res = evaluate(cfg, fake_runs, {"rpm_measured": 56.1, "tpm_measured": 42.9, "workers": 5,
                                    "window_s": 20, "ok": 19, "errors": 0}, DEFAULT_TIERS)
    assert res["verdict"] == "REJECTED"  # error + RPM below cap
    assert res["grade"] in "ABCD" and 0 <= res["v_total"] <= 100
    assert not res["veto"]
    fake_runs["small"][0]["envelope"]["model"] = "gpt-5.5"
    res2 = evaluate(cfg, fake_runs, None, DEFAULT_TIERS)
    assert res2["veto"] and res2["verdict"] == "REJECTED"
    # size-conditional degradation: small clean, large mostly-503 -> veto
    deg_runs = {
        "small": [
            {"ok": True, "ttft_s": 1.0, "total_s": 4.0, "tokens_per_s": 40.0, "input_tokens": 6000,
             "output_tokens": 120, "envelope": {"id_prefix": "msg_01aaa", "model": "claude-opus-4-6"}},
            {"ok": True, "ttft_s": 1.1, "total_s": 4.2, "tokens_per_s": 39.0, "input_tokens": 6000,
             "output_tokens": 120, "envelope": {"id_prefix": "msg_01aab", "model": "claude-opus-4-6"}},
        ],
        "large": [
            {"ok": False, "error": "http 503", "status": 503, "error_type": "overloaded_error", "total_s": 2.0},
            {"ok": False, "error": "http 503", "status": 503, "error_type": "overloaded_error", "total_s": 2.0},
        ],
    }
    res3 = evaluate(cfg, deg_runs, None, DEFAULT_TIERS)
    assert any("size-conditional degradation" in n for n in res3["veto"]), res3["veto"]
    # resp_ family on openai-chat path -> account-pool veto
    cfg_oai = ModelConfig(provider_id="g", base_url="https://x", model="gpt-5.5",
                          api_key_env="K", protocol="openai_chat", auth_type="bearer")
    resp_runs = {"small": [
        {"ok": True, "ttft_s": 4.4, "total_s": 8.0, "tokens_per_s": 50.0, "input_tokens": 4000,
         "output_tokens": 150, "envelope": {"id_prefix": "resp_09c73ed", "model": "gpt-5.5"}},
    ]}
    res4 = evaluate(cfg_oai, resp_runs, None, DEFAULT_TIERS)
    assert any("account pool" in n for n in res4["veto"]), res4["veto"]
    # error breakdown surfaces the 503 type
    big_err = [i for i in res3["items"] if i["item"].startswith("errors large")][0]
    assert big_err.get("error_breakdown", {}).get("overloaded_error") == 2, big_err
    print(render_markdown(cfg, res, {"rpm_measured": 56.1, "tpm_measured": 42.9, "workers": 5,
                                     "window_s": 20, "ok": 19, "errors": 0}))
    print("\nself-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="provider id from providers.local.json")
    parser.add_argument("--tiers", default="small,medium,large", help="comma-separated subset of: small,medium,large")
    parser.add_argument("--samples", type=int, help="override samples per tier")
    parser.add_argument("--concurrency", type=int, help="override concurrency per tier")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip-rpm", action="store_true", help="skip the RPM/TPM saturation probe")
    parser.add_argument("--rpm-workers", type=int, default=10)
    parser.add_argument("--rpm-window", type=float, default=30.0)
    parser.add_argument("--price-in", type=float, default=0.0, help="$ per 1M input tokens (for cost cap)")
    parser.add_argument("--price-out", type=float, default=0.0, help="$ per 1M output tokens (for cost cap)")
    parser.add_argument("--max-cost-usd", type=float, default=0.0, help="0 = no cap; else abort new dispatch past this")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.provider:
        parser.error("--provider is required (or use --self-test)")

    load_local_env()
    cfg = load_provider(args.provider)

    tiers: dict[str, Tier] = {}
    for key in (k.strip() for k in args.tiers.split(",") if k.strip()):
        if key not in DEFAULT_TIERS:
            raise SystemExit(f"unknown tier '{key}'; choose from {', '.join(DEFAULT_TIERS)}")
        t = DEFAULT_TIERS[key]
        tiers[key] = Tier(t.name, t.input_tokens,
                          args.concurrency or t.concurrency, args.samples or t.samples,
                          t.ttft_p50, t.ttft_p90, t.otps)

    est_in = sum(t.input_tokens * t.samples for t in tiers.values())
    print(f"provider={cfg.provider_id} model={cfg.model} protocol={cfg.protocol}")
    print(f"tiers: " + ", ".join(f"{k}(n={t.samples},cc={t.concurrency},~{t.input_tokens}tok)" for k, t in tiers.items()))
    print(f"estimated input tokens: ~{est_in:,} (plus outputs and RPM probe); this spends real quota\n")

    budget = {"cap": args.max_cost_usd, "spent": 0.0,
              "price_in": args.price_in, "price_out": args.price_out} if args.max_cost_usd else None
    if budget:
        print(f"cost cap: ${args.max_cost_usd:.2f} (price_in=${args.price_in}/1M price_out=${args.price_out}/1M)\n")

    tier_runs: dict[str, list[dict[str, Any]]] = {}
    for key, tier in tiers.items():
        print(f"tier {tier.name}: {tier.samples} samples @ concurrency {tier.concurrency}")
        tier_runs[key] = asyncio.run(run_tier(cfg, tier, args.max_tokens, args.timeout, budget))

    rpm = None
    if not args.skip_rpm:
        print(f"RPM probe: {args.rpm_workers} workers for {args.rpm_window}s")
        rpm = asyncio.run(run_rpm_probe(cfg, args.rpm_workers, args.rpm_window, args.timeout))

    result = evaluate(cfg, tier_runs, rpm, tiers)

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"access_{cfg.provider_id}_{stamp}.json"
    out.write_text(json.dumps({
        "at": stamp, "provider_id": cfg.provider_id, "model": cfg.model, "protocol": cfg.protocol,
        "tiers": {k: {"input_tokens": t.input_tokens, "concurrency": t.concurrency, "samples": t.samples}
                  for k, t in tiers.items()},
        "result": result, "rpm_probe": rpm, "runs": tier_runs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(render_markdown(cfg, result, rpm))
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
