"""bench_gateway.py - open-loop load probe for a relay/gateway endpoint.

Reuses model_client.call_model (sync httpx, non-streaming) so auth/config/redaction
stay identical to the eval toolchain. This measures the GATEWAY's sustainable
throughput, NOT any official provider quota - the point is to see whether the
relay's claimed RPM/ITPM/OTPM hold up and whether its x-ratelimit-* headers are real.

Key properties:
  * OPEN-LOOP arrival: requests are dispatched at a fixed rate regardless of how
    long earlier ones take. Falling behind schedule = backpressure (reported).
  * Three independent buckets: RPM, input-TPM, output-TPM (Anthropic-style).
  * Header authenticity cross-check: flags 429s that occur while the gateway's
    remaining-tokens/requests headers still claim > 0.
  * Cost cap: aborts new dispatch once estimated $ or total tokens exceed a limit.

Usage (60-request calibration first!):
  python bench_gateway.py --config configs/providers.local.json --key tested_model \
      --rate 10 --max-requests 60 --input-len 8000 --output-len 1500 \
      --price-in 5 --price-out 25 --max-cost-usd 5

Dry run (no network, no cost - validates wiring against model_client):
  python bench_gateway.py --config configs/providers.example.json --dry-run --max-requests 5
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import string
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from model_client import ModelConfig, call_model

@dataclass
class Rec:
    idx: int
    scheduled: float          # intended send time (open-loop clock)
    sent: float               # actual POST start
    done: float               # completion time
    ok: bool
    status: int | None        # parsed HTTP status (429 etc.) if error
    in_tok: int
    out_tok: int
    total_ms: float | None
    remaining_req: str | None
    remaining_tok: str | None


def load_model(config_path: Path, key: str) -> ModelConfig:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if key not in data:
        raise SystemExit(f"key {key!r} not found in {config_path}; available: {sorted(data)}")
    node = data[key]
    return ModelConfig(
        provider_id=node["provider_id"],
        base_url=node["base_url"],
        model=node["model"],
        api_key_env=node["api_key_env"],
        protocol=node["protocol"],
        auth_type=node.get("auth_type", "bearer"),
        extra_body=node.get("extra_body", {}),
    )


def make_prompt(approx_tokens: int) -> list[dict[str, str]]:
    # ~4 chars/token. Unique nonce per request defeats gateway prompt-caching,
    # which would otherwise falsify input-token accounting.
    nonce = uuid.uuid4().hex
    n_chars = max(40, approx_tokens * 4 - len(nonce) - 64)
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9)))
             for _ in range(max(1, n_chars // 7))]
    body = " ".join(words)[:n_chars]
    return [{"role": "user",
             "content": f"[{nonce}] Summarize the following text in one sentence:\n{body}"}]


def parse_status(err: str | None) -> int | None:
    if not err:
        return None
    m = re.search(r"HTTP\s+(\d+)", err)
    return int(m.group(1)) if m else None


def self_test() -> int:
    """Offline checks: prompt sizing, status parsing, open-loop scheduling math."""
    # parse_status mirrors model_client's "HTTP {code}: ..." format.
    assert parse_status("HTTP 429: too many") == 429
    assert parse_status("HTTP 503: overloaded") == 503
    assert parse_status("ConnectError: boom") is None
    assert parse_status(None) is None
    # make_prompt: returns a messages list; content carries a unique nonce
    # (defeats prompt-cache) and scales with the requested token size.
    p1 = make_prompt(2000)
    p2 = make_prompt(2000)
    c1 = p1[0]["content"]
    assert c1 != p2[0]["content"], "prompts must carry a unique nonce"
    assert len(c1) > 1000, f"expected sizeable prompt, got {len(c1)} chars"
    assert len(make_prompt(2000)[0]["content"]) > len(make_prompt(200)[0]["content"])
    # open-loop schedule: request i is due at start + i*interval, independent of latency.
    rate = 10.0
    interval = 1.0 / rate
    due = [i * interval for i in range(5)]
    assert all(abs(d - i * 0.1) < 1e-9 for i, d in enumerate(due))
    print("bench_gateway self-test ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/providers.local.json"))
    ap.add_argument("--key", default="tested_model", help="top-level key in the config json")
    ap.add_argument("--rate", type=float, default=10.0, help="requests/sec (10 = 600 RPM)")
    ap.add_argument("--max-requests", type=int, default=60, help="start small; 60 = calibration")
    ap.add_argument("--input-len", type=int, default=8000, help="approx prompt tokens")
    ap.add_argument("--output-len", type=int, default=1500, help="max_tokens per request")
    ap.add_argument("--arrival", choices=["constant", "poisson"], default="constant")
    ap.add_argument("--max-workers", type=int, default=256)
    ap.add_argument("--price-in", type=float, default=0.0, help="$ per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="$ per 1M output tokens")
    ap.add_argument("--max-cost-usd", type=float, default=0.0, help="0 = no cap")
    ap.add_argument("--max-total-tokens", type=int, default=0, help="0 = no cap")
    ap.add_argument("--dry-run", action="store_true", help="no network: use model_client dry path")
    ap.add_argument("--dump-jsonl", type=Path, default=None, help="write per-request records here")
    ap.add_argument("--events", type=Path, default=Path("bench_gateway_events.jsonl"))
    ap.add_argument("--self-test", action="store_true", help="offline checks, no network")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    model = load_model(args.config, args.key)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    live = not args.dry_run

    client = httpx.Client(timeout=httpx.Timeout(120.0),
                          limits=httpx.Limits(max_connections=args.max_workers,
                                              max_keepalive_connections=args.max_workers)) if live else None
    recs: list[Rec] = []
    recs_lock = threading.Lock()
    spent = {"tokens": 0, "cost": 0.0}
    spent_lock = threading.Lock()
    abort = threading.Event()

    def worker(idx: int, scheduled: float) -> None:
        sent = time.time()
        # retries=0 by calling call_model directly: never mask a 429.
        comp = call_model(client=client, model=model, messages=make_prompt(args.input_len),
                          max_tokens=args.output_len, temperature=0.0,
                          live=live, events_file=args.events)
        done = time.time()
        m = comp.metrics
        it, ot = m.input_tokens or 0, m.output_tokens or 0
        with spent_lock:
            spent["tokens"] += it + ot
            spent["cost"] += it / 1e6 * args.price_in + ot / 1e6 * args.price_out
            if ((args.max_cost_usd and spent["cost"] >= args.max_cost_usd) or
                    (args.max_total_tokens and spent["tokens"] >= args.max_total_tokens)):
                abort.set()
        h = comp.response_headers or {}
        with recs_lock:
            recs.append(Rec(idx, scheduled, sent, done, m.ok, parse_status(m.error),
                            it, ot, m.total_ms,
                            h.get("x-ratelimit-remaining-requests"),
                            h.get("x-ratelimit-remaining-tokens")))

    mode = "DRY-RUN (no network)" if args.dry_run else "LIVE"
    print(f"[bench] {mode} gateway={model.base_url} model={model.model} "
          f"rate={args.rate}/s target_RPM={args.rate*60:.0f} n={args.max_requests}")
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for i in range(args.max_requests):
            if abort.is_set():
                print(f"[bench] cost/token cap hit - stopping dispatch at request {i}")
                break
            interval = 1.0 / args.rate
            if args.arrival == "poisson":
                target = time.time() + random.expovariate(args.rate)
            else:
                target = start + i * interval
            now = time.time()
            if target > now:
                time.sleep(target - now)
            ex.submit(worker, i, target)
    if client is not None:
        client.close()

    report(recs, start, args)
    if args.dump_jsonl:
        with args.dump_jsonl.open("w", encoding="utf-8") as fh:
            for r in sorted(recs, key=lambda x: x.idx):
                fh.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
        print(f"[bench] wrote {len(recs)} records -> {args.dump_jsonl}")
    return 0

def report(recs: list[Rec], start: float, args: argparse.Namespace) -> None:
    if not recs:
        print("[bench] no records")
        return
    recs.sort(key=lambda r: r.idx)
    elapsed = max(1e-6, max(r.done for r in recs) - start)
    ok = [r for r in recs if r.ok]
    h429 = [r for r in recs if r.status == 429]
    other_err = [r for r in recs if not r.ok and r.status != 429]

    def per_minute(items, ts_fn, val_fn):
        buckets: dict[int, float] = {}
        for r in items:
            w = int((ts_fn(r) - start) // 60)
            buckets[w] = buckets.get(w, 0) + val_fn(r)
        return buckets

    rpm = per_minute(ok, lambda r: r.sent, lambda r: 1)
    itpm = per_minute(ok, lambda r: r.done, lambda r: r.in_tok)
    otpm = per_minute(ok, lambda r: r.done, lambda r: r.out_tok)
    lateness = [r.sent - r.scheduled for r in recs]
    lat_ms = [r.total_ms for r in ok if r.total_ms is not None]

    def pct(xs, p):
        if not xs:
            return None
        if len(xs) < 2:
            return round(xs[0], 2)
        return round(statistics.quantiles(xs, n=100)[p - 1], 2)

    print("\n===== gateway load report =====")
    print(f"elapsed            : {elapsed:.1f}s")
    print(f"dispatched / ok    : {len(recs)} / {len(ok)}  success={len(ok)/len(recs)*100:.1f}%")
    print(f"429 / other errors : {len(h429)} / {len(other_err)}  (429 rate {len(h429)/len(recs)*100:.1f}%)")
    print(f"observed RPM/min   : {dict(sorted(rpm.items()))}   (raw per-60s bucket; partial windows under-count)")
    print(f"observed ITPM/min  : {dict(sorted(itpm.items()))}")
    print(f"observed OTPM/min  : {dict(sorted(otpm.items()))}")
    # Normalized rates over actual elapsed time — the meaningful figure for
    # sub-minute calibration runs where no 60s bucket is full.
    norm_rpm = len(ok) * 60.0 / elapsed
    norm_itpm = sum(r.in_tok for r in ok) * 60.0 / elapsed
    norm_otpm = sum(r.out_tok for r in ok) * 60.0 / elapsed
    print(f"normalized/min     : RPM={norm_rpm:.1f} ITPM={norm_itpm:.0f} OTPM={norm_otpm:.0f}  "
          f"(ok*60/elapsed; use this for runs shorter than a minute)")
    tot_in = sum(r.in_tok for r in ok)
    tot_out = sum(r.out_tok for r in ok)
    est_cost = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
    print(f"tokens in/out      : {tot_in} / {tot_out}   est cost=${est_cost:.4f} "
          f"(price_in={args.price_in} price_out={args.price_out} /1M)")
    print(f"latency ms p50/p95/p99 : {pct(lat_ms,50)} / {pct(lat_ms,95)} / {pct(lat_ms,99)}")
    print(f"sched lateness s p50/p99: {pct(lateness,50)} / {pct(lateness,99)}   "
          f"(high => gateway can't sustain the rate)")

    # ---- header authenticity cross-check (feeds the detector) ----
    seen_hdr = [r for r in recs if r.remaining_tok is not None or r.remaining_req is not None]
    if not seen_hdr:
        print("HEADER CHECK      : gateway sent NO x-ratelimit-* headers "
              "(can't be verified; likely stripped or not emitted)")
    else:
        def as_int(x):
            try:
                return int(str(x).replace(",", ""))
            except Exception:
                return None
        min_tok = min((v for v in (as_int(r.remaining_tok) for r in seen_hdr) if v is not None), default=None)
        min_req = min((v for v in (as_int(r.remaining_req) for r in seen_hdr) if v is not None), default=None)
        contradiction = h429 and ((min_tok is None or min_tok > 0) and (min_req is None or min_req > 0))
        print(f"HEADER CHECK      : min remaining tokens={min_tok} requests={min_req}")
        if contradiction:
            print("  !! SUSPICIOUS: 429s occurred while headers still claimed remaining>0 "
                  "-> ratelimit headers are probably synthetic/passthrough, not the real limiter")
        elif h429:
            print("  ok: headers approached 0 as 429s appeared (consistent with a real limiter)")


if __name__ == "__main__":
    raise SystemExit(main())
