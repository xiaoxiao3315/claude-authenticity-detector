from __future__ import annotations

import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from acceptance_pack import verify_acceptance_pack
from campaigns import (
    campaign_identity_problem,
    campaign_dir as resolve_campaign_dir,
    campaign_leaderboard,
    campaign_list_payload,
    load_run_index,
    load_summary,
    summary_needs_refresh,
    summarize_campaign,
)
from authenticity import load_or_build_authenticity
from local_env import load_local_env
from redaction import redact_text, redact_value


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
CAMPAIGNS_DIR = ROOT / "campaigns"
WEB_DIR = ROOT / "web"
PROVIDERS_LOCAL = ROOT / "configs" / "providers.local.json"
LOCAL_SECRETS = ROOT / "local_secrets.env"


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def safe_run_dir(job_id: str) -> Path:
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        raise ValueError("invalid job id")
    path = RUNS_DIR / job_id
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(job_id)
    return path


def list_jobs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    jobs = []
    for path in RUNS_DIR.iterdir():
        state_path = path / "state.json"
        if not path.is_dir() or not state_path.exists():
            continue
        try:
            state = read_json(state_path)
        except Exception:
            continue
        jobs.append(
            {
                "job_id": state.get("job_id") or path.name,
                "status": state.get("status"),
                "progress": state.get("progress"),
                "final_decision": state.get("final_decision"),
                "started_at": state.get("started_at"),
                "completed_at": state.get("completed_at"),
            }
        )
    return sorted(jobs, key=lambda item: str(item.get("started_at") or item.get("job_id") or ""), reverse=True)


def latest_job() -> dict | None:
    jobs = list_jobs()
    return jobs[0] if jobs else None


def latest_quality_gate(run_dir: Path) -> dict | None:
    gates_dir = run_dir / "quality_gates"
    if not gates_dir.exists():
        return None
    candidates = [path for path in gates_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    gate_dir = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    records = read_jsonl(gate_dir / "quality_gate_records.jsonl")
    manifest_path = gate_dir / "quality_gate_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return {
        "gate_id": gate_dir.name,
        "manifest": manifest,
        "records": records,
        "primary_record": records[0] if records else None,
    }


def score_value(record: dict) -> float | None:
    scoring = record.get("scoring") or {}
    final_score = scoring.get("final_score") if isinstance(scoring.get("final_score"), dict) else {}
    value = final_score.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def provider_records(records: list[dict], provider_id: str) -> list[dict]:
    selected = []
    for record in records:
        provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
        if str(provider.get("id") or "") == provider_id:
            selected.append(record)
    return selected or records


def provider_model_name(state: dict, records: list[dict], provider_id: str) -> str:
    tested = ((state.get("models") or {}).get("tested_model") or {})
    if str(tested.get("provider_id") or "") == provider_id and tested.get("model"):
        return str(tested.get("model"))
    for record in records:
        provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
        if str(provider.get("id") or "") != provider_id:
            continue
        for key in ("claimed_model", "model_requested", "model_returned"):
            if provider.get(key):
                return str(provider.get(key))
    return provider_id


def provider_identity(records: list[dict], provider_id: str) -> dict:
    for record in records:
        provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
        if str(provider.get("id") or "") != provider_id:
            continue
        return {
            "provider_display_name": provider.get("provider_display_name") or provider_id,
            "provider_host": provider.get("base_url_host"),
            "source_group": provider.get("leaderboard_group") or "gateway_candidate",
            "baseline_model": provider.get("baseline_model") or provider.get("claimed_model"),
            "provider_channel": provider.get("provider_channel") or "unknown",
        }
    return {
        "provider_display_name": provider_id,
        "provider_host": None,
        "source_group": "gateway_candidate",
        "baseline_model": None,
        "provider_channel": "unknown",
    }


def leaderboard_raw_rows(*, include_dry_run: bool = False) -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    rows: list[dict] = []
    for run_dir in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        state_path = run_dir / "state.json"
        benchmark_path = run_dir / "benchmark_scores.json"
        if not state_path.exists() or not benchmark_path.exists():
            continue
        try:
            state = read_json(state_path)
            benchmark = read_json(benchmark_path)
            records = read_jsonl(run_dir / "run_records.jsonl")
        except Exception:
            continue
        live_provider = state.get("live_provider") is True
        if not include_dry_run and not live_provider:
            continue
        gate = latest_quality_gate(run_dir)
        primary_gate = gate.get("primary_record") if gate else None
        gate_metrics = primary_gate.get("metrics_snapshot") if isinstance(primary_gate, dict) else {}
        providers = benchmark.get("providers") if isinstance(benchmark.get("providers"), dict) else {}
        for provider_id, provider_score in providers.items():
            if not isinstance(provider_score, dict):
                continue
            provider_id = str(provider_id)
            selected_records = provider_records(records, provider_id)
            ok_count = 0
            scores: list[float] = []
            latencies: list[float] = []
            for record in selected_records:
                telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
                if telemetry.get("ok") is True:
                    ok_count += 1
                score = score_value(record)
                if score is not None:
                    scores.append(score)
                latency = to_float(telemetry.get("first_content_token_ms") or telemetry.get("total_ms"))
                if latency is not None:
                    latencies.append(latency)
            total = int(provider_score.get("task_count") or len(selected_records) or 0)
            identity = provider_identity(selected_records, provider_id)
            generated_at = str(
                benchmark.get("generated_at")
                or state.get("completed_at")
                or state.get("started_at")
                or ""
            )
            rows.append(
                {
                    "run_id": state.get("job_id") or run_dir.name,
                    "provider_id": provider_id,
                    "provider_display_name": identity["provider_display_name"],
                    "provider_host": identity["provider_host"],
                    "source_group": identity["source_group"],
                    "provider_channel": identity["provider_channel"],
                    "model": provider_model_name(state, selected_records, provider_id),
                    "baseline_model": identity["baseline_model"],
                    "mode": benchmark.get("benchmark_mode") or provider_score.get("mode") or "custom",
                    "task_count": total,
                    "success_rate": (ok_count / total) if total else None,
                    "average_score_0_10": (sum(scores) / len(scores)) if scores else None,
                    "score": to_float(provider_score.get("benchmark_score")),
                    "benchmark_score": to_float(provider_score.get("benchmark_score")),
                    "quality_score": to_float(provider_score.get("quality_score")),
                    "latency_score": to_float(provider_score.get("latency_score")),
                    "cost_efficiency_score": to_float(provider_score.get("cost_efficiency_score")),
                    "risk_penalty": to_float(provider_score.get("risk_penalty")) or 0.0,
                    "p95_first_content_token_ms": to_float(gate_metrics.get("p95_first_content_token_ms")) or percentile(latencies, 0.95),
                    "gate_decision": (primary_gate.get("decision") if isinstance(primary_gate, dict) else None) or state.get("final_decision"),
                    "status": state.get("status"),
                    "live_provider": live_provider,
                    "generated_at": generated_at,
                    "started_at": state.get("started_at"),
                    "completed_at": state.get("completed_at"),
                }
            )
    return rows


def leaderboard(limit: int = 50, *, include_dry_run: bool = False) -> dict:
    raw_rows = leaderboard_raw_rows(include_dry_run=include_dry_run)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in raw_rows:
        key = (
            str(row.get("provider_id") or ""),
            str(row.get("model") or ""),
            str(row.get("mode") or ""),
        )
        grouped.setdefault(key, []).append(row)

    entries = []
    for history in grouped.values():
        history.sort(key=lambda item: str(item.get("generated_at") or item.get("completed_at") or item.get("started_at") or ""), reverse=True)
        latest = dict(history[0])
        values = [value for value in (to_float(item.get("score")) for item in history) if value is not None]
        average_score = sum(values) / len(values) if values else None
        latest["latest_score"] = latest.get("score")
        latest["score"] = round(average_score, 2) if average_score is not None else None
        latest["average_score"] = latest["score"]
        latest["latest_run_id"] = latest.get("run_id")
        latest["history_count"] = len(history)
        latest["history"] = [
            {
                "run_id": item.get("run_id"),
                "score": item.get("score"),
                "benchmark_score": item.get("benchmark_score"),
                "quality_score": item.get("quality_score"),
                "success_rate": item.get("success_rate"),
                "gate_decision": item.get("gate_decision"),
                "generated_at": item.get("generated_at"),
            }
            for item in history[:10]
        ]
        entries.append(latest)

    entries.sort(
        key=lambda item: (
            item.get("score") is not None,
            item.get("score") or -1.0,
            item.get("success_rate") or -1.0,
            str(item.get("generated_at") or ""),
        ),
        reverse=True,
    )
    for index, entry in enumerate(entries, start=1):
        entry["rank"] = index
    return {
        "entries": entries[:limit],
        "total": len(entries),
        "raw_run_count": len(raw_rows),
        "limit": limit,
        "include_dry_run": include_dry_run,
        "sort": "score desc, success_rate desc, latest run desc",
    }


def summarize_run(run_dir: Path) -> dict:
    state = read_json(run_dir / "state.json")
    run_records = read_jsonl(run_dir / "run_records.jsonl")
    benchmark_path = run_dir / "benchmark_scores.json"
    benchmark = read_json(benchmark_path) if benchmark_path.exists() else {}
    gate = latest_quality_gate(run_dir)
    samples = []
    ok_count = 0
    scores = []
    latencies = []
    for record in run_records:
        task = record.get("task") or {}
        provider = record.get("provider") or {}
        telemetry = record.get("telemetry") or {}
        ok = telemetry.get("ok") is True
        if ok:
            ok_count += 1
        score = score_value(record)
        if score is not None:
            scores.append(score)
        latency = telemetry.get("first_content_token_ms") or telemetry.get("total_ms")
        try:
            latency_value = float(latency)
            latencies.append(latency_value)
        except (TypeError, ValueError):
            latency_value = None
        samples.append(
            {
                "task_id": task.get("id"),
                "category": task.get("category"),
                "dimension": task.get("enterprise_dimension"),
                "ok": ok,
                "score": score,
                "error": redact_text(telemetry.get("error"), max_chars=500),
                "latency_ms": latency_value,
                "model_returned": provider.get("model_returned"),
            }
        )
    total = len(run_records)
    success_rate = ok_count / total if total else None
    avg_score = sum(scores) / len(scores) if scores else None
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    primary_gate = gate.get("primary_record") if gate else None
    metrics = primary_gate.get("metrics_snapshot") if isinstance(primary_gate, dict) else {}
    providers = benchmark.get("providers") if isinstance(benchmark.get("providers"), dict) else {}
    provider_score = next(iter(providers.values()), {}) if providers else {}
    return {
        "state": state,
        "metrics": {
            "sample_count": total,
            "ok_count": ok_count,
            "failure_count": total - ok_count,
            "success_rate": success_rate,
            "average_score_0_10": avg_score,
            "average_latency_ms": avg_latency,
            "p95_first_content_token_ms": metrics.get("p95_first_content_token_ms"),
            "gate_score": metrics.get("gate_score") or provider_score.get("benchmark_score"),
            "benchmark_score": provider_score.get("benchmark_score"),
            "quality_score": provider_score.get("quality_score"),
            "latency_score": provider_score.get("latency_score"),
            "cost_efficiency_score": provider_score.get("cost_efficiency_score"),
        },
        "quality_gate": {
            "gate_id": gate.get("gate_id") if gate else None,
            "decision": primary_gate.get("decision") if isinstance(primary_gate, dict) else state.get("final_decision"),
            "blockers": primary_gate.get("blockers") if isinstance(primary_gate, dict) else [],
            "review_items": primary_gate.get("review_items") if isinstance(primary_gate, dict) else [],
            "passed_rules": primary_gate.get("passed_rules") if isinstance(primary_gate, dict) else [],
        },
        "samples": samples,
        "benchmark": benchmark,
    }


def sanitized_config() -> dict:
    load_local_env()
    if not PROVIDERS_LOCAL.exists():
        return {"exists": False, "providers": None}
    data = read_json(PROVIDERS_LOCAL)
    out = {"exists": True, "providers": {}}
    for label in ("tested_model", "judge_model"):
        item = data.get(label) or {}
        env_name = str(item.get("api_key_env") or "")
        out["providers"][label] = {
            "provider_id": item.get("provider_id"),
            "base_url": item.get("base_url"),
            "model": item.get("model"),
            "protocol": item.get("protocol"),
            "auth_type": item.get("auth_type") or "bearer",
            "api_key_env": env_name,
            "api_key_present": bool(os.environ.get(env_name)),
        }
    return out


def update_env_file(updates: dict[str, str]) -> None:
    existing: dict[str, str] = {}
    if LOCAL_SECRETS.exists():
        for line in LOCAL_SECRETS.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            existing[key.strip()] = value.strip()
    for key, value in updates.items():
        if value:
            existing[key] = value
    lines = [f"{key}={value}" for key, value in sorted(existing.items())]
    LOCAL_SECRETS.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    load_local_env(override=True)


def save_config(payload: dict) -> dict:
    providers = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(providers, dict):
        raise ValueError("providers object is required")
    current = read_json(PROVIDERS_LOCAL) if PROVIDERS_LOCAL.exists() else {}
    env_updates: dict[str, str] = {}
    for label, env_name in (("tested_model", "TESTED_MODEL_API_KEY"), ("judge_model", "JUDGE_MODEL_API_KEY")):
        item = providers.get(label)
        if not isinstance(item, dict):
            raise ValueError(f"{label} is required")
        current[label] = {
            "provider_id": str(item.get("provider_id") or current.get(label, {}).get("provider_id") or label),
            "base_url": str(item["base_url"]).rstrip("/"),
            "model": str(item["model"]),
            "api_key_env": env_name,
            "protocol": str(item["protocol"]),
            "auth_type": str(item.get("auth_type") or "bearer"),
        }
        api_key = str(item.get("api_key") or "")
        if api_key:
            env_updates[env_name] = api_key
    write_json(PROVIDERS_LOCAL, current)
    if env_updates:
        update_env_file(env_updates)
    return sanitized_config()


def query_bool(qs: dict[str, list[str]], name: str, default: bool = False) -> bool:
    if name not in qs:
        return default
    return str((qs.get(name) or [str(default)])[0]).lower() in {"1", "true", "yes", "on"}


def artifact_listing(root: Path) -> list[dict]:
    artifacts = []
    if not root.exists():
        return artifacts
    for item in root.iterdir():
        if not item.is_file():
            continue
        row = {"name": item.name, "bytes": item.stat().st_size}
        if item.name == "acceptance_pack.zip":
            row["verification"] = verify_acceptance_pack(item)
        artifacts.append(row)
    return artifacts


class Handler(BaseHTTPRequestHandler):
    server_version = "EvalAutomationAPI/0.2.2"

    def send_json(self, value, status: int = 200) -> None:
        body = json.dumps(redact_value(value), ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/config":
                self.send_json(sanitized_config())
            elif path == "/api/leaderboard":
                qs = parse_qs(parsed.query)
                raw_limit = (qs.get("limit") or ["50"])[0]
                try:
                    limit = min(max(int(raw_limit), 1), 200)
                except ValueError:
                    limit = 50
                live_filter = None
                if "live_provider" in qs:
                    live_filter = query_bool(qs, "live_provider")
                min_samples_raw = (qs.get("min_samples") or ["1"])[0]
                try:
                    min_samples = max(int(min_samples_raw), 1)
                except ValueError:
                    min_samples = 1
                self.send_json(
                    campaign_leaderboard(
                        CAMPAIGNS_DIR,
                        RUNS_DIR,
                        include_dry_run=query_bool(qs, "include_dry_run"),
                        benchmark_version=(qs.get("benchmark_version") or [""])[0],
                        judge_model=(qs.get("judge_model") or [""])[0],
                        quality_gate_version=(qs.get("quality_gate_version") or [""])[0],
                        live_provider=live_filter,
                        date_from=(qs.get("date_from") or [""])[0],
                        date_to=(qs.get("date_to") or [""])[0],
                        min_samples=min_samples,
                        limit=limit,
                        persist_refresh=False,
                    )
                )
            elif path == "/api/campaigns":
                self.send_json(campaign_list_payload(CAMPAIGNS_DIR, RUNS_DIR, persist_refresh=False))
            elif path == "/api/campaigns/latest":
                campaigns = campaign_list_payload(CAMPAIGNS_DIR, RUNS_DIR, persist_refresh=False).get("campaigns") or []
                self.send_json(campaigns[0] if campaigns else {})
            elif path == "/api/authenticity/latest":
                campaigns = campaign_list_payload(CAMPAIGNS_DIR, RUNS_DIR, persist_refresh=False).get("campaigns") or []
                if not campaigns:
                    self.send_json({})
                else:
                    camp_dir = resolve_campaign_dir(CAMPAIGNS_DIR, str(campaigns[0].get("campaign_id") or ""))
                    self.send_json(load_or_build_authenticity(camp_dir, RUNS_DIR, persist=False))
            elif path.startswith("/api/campaigns/"):
                self.handle_campaign_get(path)
            elif path == "/api/jobs":
                self.send_json({"jobs": list_jobs()})
            elif path == "/api/jobs/latest":
                job = latest_job()
                self.send_json(job or {})
            elif path.startswith("/api/jobs/"):
                self.handle_job_get(path)
            else:
                self.serve_static(path)
        except FileNotFoundError:
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/config":
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        if not getattr(self.server, "config_write_enabled", False):
            self.send_error_json(HTTPStatus.FORBIDDEN, "config writes are disabled; restart with --enable-config-write to allow this endpoint")
            return
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            self.send_json(save_config(payload))
        except Exception as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def handle_campaign_get(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 3:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "campaign id is required")
            return
        campaign_id = parts[2]
        camp_dir = resolve_campaign_dir(CAMPAIGNS_DIR, campaign_id)
        if not camp_dir.exists() or not camp_dir.is_dir():
            raise FileNotFoundError(campaign_id)
        identity_problem = campaign_identity_problem(camp_dir)
        if identity_problem:
            self.send_error_json(HTTPStatus.CONFLICT, identity_problem)
            return
        tail = parts[3:] if len(parts) > 3 else ["summary"]
        endpoint = tail[0]
        if endpoint == "summary":
            summary = load_summary(camp_dir)
            if summary_needs_refresh(summary):
                summary = summarize_campaign(camp_dir, RUNS_DIR, persist=False)
            self.send_json(summary)
        elif endpoint == "authenticity":
            self.send_json(load_or_build_authenticity(camp_dir, RUNS_DIR, persist=False))
        elif endpoint == "runs":
            self.send_json(load_run_index(camp_dir))
        elif endpoint == "artifacts":
            if len(tail) > 1:
                self.serve_campaign_artifact(camp_dir, tail[1])
            else:
                self.send_json({"artifacts": artifact_listing(camp_dir / "artifacts")})
        else:
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown campaign endpoint")

    def handle_job_get(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 3:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "job id is required")
            return
        job_id = parts[2]
        run_dir = safe_run_dir(job_id)
        tail = parts[3:] if len(parts) > 3 else ["state"]
        endpoint = tail[0]
        if endpoint == "state":
            self.send_json(read_json(run_dir / "state.json"))
        elif endpoint == "events":
            self.send_json({"events": read_jsonl(run_dir / "events.jsonl")})
        elif endpoint == "results":
            results_path = run_dir / "results.json"
            self.send_json(read_json(results_path) if results_path.exists() else [])
        elif endpoint == "summary":
            self.send_json(summarize_run(run_dir))
        elif endpoint == "artifacts":
            if len(tail) > 1:
                self.serve_artifact(run_dir, tail[1])
            else:
                self.send_json({"artifacts": artifact_listing(run_dir / "artifacts")})
        else:
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown job endpoint")

    def serve_campaign_artifact(self, camp_dir: Path, name: str) -> None:
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError("invalid artifact name")
        path = camp_dir / "artifacts" / name
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(name)
        if name == "acceptance_pack.zip":
            verification = verify_acceptance_pack(path)
            if not verification.get("verified"):
                self.send_error_json(HTTPStatus.CONFLICT, f"acceptance pack failed verification: {verification.get('error')}")
                return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("content-disposition", f'attachment; filename="{path.name}"')
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_artifact(self, run_dir: Path, name: str) -> None:
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError("invalid artifact name")
        path = run_dir / "artifacts" / name
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(name)
        if name == "acceptance_pack.zip":
            verification = verify_acceptance_pack(path)
            if not verification.get("verified"):
                self.send_error_json(HTTPStatus.CONFLICT, f"acceptance pack failed verification: {verification.get('error')}")
                return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("content-disposition", f'attachment; filename="{path.name}"')
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, path: str) -> None:
        if path in ("", "/"):
            file_path = WEB_DIR / "index.html"
        else:
            rel = path.lstrip("/")
            if rel.startswith("web/"):
                rel = rel[len("web/") :]
            file_path = WEB_DIR / rel
        file_path = file_path.resolve()
        if WEB_DIR.resolve() not in file_path.parents and file_path != (WEB_DIR / "index.html").resolve():
            raise ValueError("invalid static path")
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(str(file_path))
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the eval dashboard and read-only run APIs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--enable-config-write", action="store_true", help="enable POST /api/config writes to local provider and secret files")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.config_write_enabled = bool(args.enable_config_write)
    print(f"serving http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
