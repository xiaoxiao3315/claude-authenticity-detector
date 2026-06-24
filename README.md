# eval_automation_v0_2

Backend-only core for the local LLM evaluation workflow.

This folder intentionally excludes the old web console, historical runs, local secrets, campaigns, archives, screenshots, and generated logs. It keeps the evaluation kernel: benchmark execution, `run_record_v1`, compatibility checks, offline re-score, trace evaluation, quality gate, and redacted audit export.

## v0.2.1 Two-Model Headless Flow

The preferred entrypoint is now CLI-first:

```powershell
python .\eval_cli.py run --job smoke_10
python .\eval_cli.py inspect --latest
python .\eval_cli.py export --latest
```

`run` reads `configs/providers.local.json` and `configs/jobs/smoke_10.json`, writes a new `runs/JOB-.../` folder, asks the tested model to answer, asks the judge model to score, writes `run_record_v1`, runs the quality gate, and leaves the dashboard to observe the files.

Live provider calls are off by default. Use `--live` only when local keys are configured:

```powershell
python .\eval_cli.py run --job smoke_10 --live
```

To open the local dashboard/API:

```powershell
python .\api_server.py --host 127.0.0.1 --port 18081
```

To identify the correct gateway model/protocol/auth combination, use `probe`.
It tests OpenAI chat and Anthropic messages separately, and tries both Bearer
and `x-api-key` auth headers:

```powershell
python .\eval_cli.py probe --model opus4.8 --model claude-opus-4-8 --model gpt-5.5 --stop-after-success
```

API routes:

```text
GET /api/jobs
GET /api/jobs/latest
GET /api/jobs/<job_id>/state
GET /api/jobs/<job_id>/events
GET /api/jobs/<job_id>/results
GET /api/jobs/<job_id>/artifacts
```

## v0.2.2 Campaign Aggregation

Campaigns group repeated runs under the same tested model, judge model,
benchmark version, quality-gate version, score formula, and live/dry-run mode.
The leaderboard ranks campaign summaries by default instead of treating one
run as a model-level conclusion.

```powershell
python .\eval_cli.py campaign --job smoke_10 --repeat 3
python .\eval_cli.py campaign-list
python .\eval_cli.py campaign-status --campaign-id CMP-...
python .\eval_cli.py campaign-inspect --campaign-id CMP-...
python .\eval_cli.py campaign-export --campaign-id CMP-...
```

Use `--live` only after the dry-run path is verified:

```powershell
python .\eval_cli.py campaign --job smoke_10 --repeat 3 --live
```

Campaign files are runtime artifacts under `campaigns/CMP-.../`:

```text
campaign.json   fixed test condition, redacted model identity, versions, git commit, config hash
run_ids.json    child run IDs and per-round status
summary.json    aggregate metrics, separated decisions, trend and evidence data
artifacts/      campaign acceptance pack after campaign-export
```

The model confidence decision and gateway reliability decision are separate.
Transport failures lower gateway reliability, but are not treated as proof of
poor model identity. API keys are never written; campaign metadata stores only
redacted config and a SHA-256 key fingerprint prefix.

Campaign API routes:

```text
GET /api/leaderboard
GET /api/leaderboard?include_dry_run=true
GET /api/campaigns
GET /api/campaigns/latest
GET /api/campaigns/<campaign_id>/summary
GET /api/campaigns/<campaign_id>/runs
GET /api/campaigns/<campaign_id>/artifacts
GET /api/campaigns/<campaign_id>/artifacts/acceptance_pack.zip
```

`/api/leaderboard` defaults to completed live campaigns only. It excludes
dry-run campaigns unless `include_dry_run=true` is passed, and only ranks
campaigns within one compatible comparison group.

## Install

```powershell
cd E:\ai\ai测试\eval_automation_v0_2
python -m pip install -r requirements.txt
```

## Config

Create a local provider file from the template:

```powershell
Copy-Item .\providers.example.json .\providers.local.json
Copy-Item .\configs\providers.example.json .\configs\providers.local.json
```

Edit `providers.local.json` with your provider id, Base URL, model, auth type, and auth environment variable. Do not commit `providers.local.json`.

Set secrets in the shell or create `local_secrets.env`:

```env
ANTHROPIC_API_KEY=your_key_here
```

Do not commit `local_secrets.env`.

## Inspect Tasks And Modes

```powershell
python .\run_eval.py --tasks .\tasks\enterprise_v0_2.json --benchmarks .\benchmarks\enterprise_modes.json --list-modes
python .\run_eval.py --tasks .\tasks\pilot_v0_1.json --benchmarks .\benchmarks\enterprise_modes.json --list-tasks
```

## Run A Smoke Benchmark

This calls the configured provider and writes a new run under `runs/`.

```powershell
python .\run_eval.py --providers .\providers.local.json --tasks .\tasks\pilot_v0_1.json --benchmarks .\benchmarks\enterprise_modes.json --benchmark-mode mode_10 --out .\runs
```

## Validate Run Records

After a benchmark run, validate the generated JSONL:

```powershell
python .\validate_run_records.py --jsonl .\runs\<run_id>\run_records.jsonl
```

## Local Self-Checks

```powershell
python -m py_compile run_eval.py run_records.py benchmarking.py local_env.py compatibility.py trace_evaluation.py rescore.py quality_gate.py audit_export.py evidence_registry.py archive_registry.py job_runtime.py validate_run_records.py
python .\validate_run_records.py --self-test
python .\compatibility.py --self-test
python .\quality_gate.py --self-test
python .\trace_evaluation.py --self-test
python .\audit_export.py --self-test
python .\eval_cli.py campaign --job smoke_10 --repeat 3
python .\eval_cli.py campaign-export --campaign-id CMP-...
```
