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

Dry-runs do not load `local_secrets.env`; local secrets are loaded only for
live provider calls, config-status display, or explicit probe/config-write
flows.

To open the local dashboard/API:

```powershell
python .\api_server.py --host 127.0.0.1 --port 18081
```

The dashboard/API is read-only by default. To allow the web form to write
`configs/providers.local.json` and `local_secrets.env`, start it explicitly:

```powershell
python .\api_server.py --host 127.0.0.1 --port 18081 --enable-config-write
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
python .\eval_cli.py campaign --job smoke_10 --repeat 3 --retries 2 --retry-backoff 2
python .\eval_cli.py campaign --job smoke_10 --campaign-id CMP-... --resume
python .\eval_cli.py campaign --job smoke_100 --repeat 1
python .\eval_cli.py campaign-list
python .\eval_cli.py campaign-status --campaign-id CMP-...
python .\eval_cli.py campaign-inspect --campaign-id CMP-...
python .\eval_cli.py campaign-export --campaign-id CMP-...
```

Use `smoke_10` for quick checks and `smoke_100` for a small comparison run.
`smoke_100` selects up to 100 eligible tasks from the current 130-task private
bank.

Use `--live` only after the dry-run path is verified:

```powershell
python .\eval_cli.py campaign --job smoke_10 --repeat 3 --live
python .\eval_cli.py campaign --job smoke_100 --repeat 1 --live
```

For CI-style gates, add `--require-go`. The command still writes the run or
campaign evidence, but exits `2` when the final decision is not `GO`:

```powershell
python .\eval_cli.py run --job smoke_10 --require-go
python .\eval_cli.py campaign --job smoke_10 --repeat 3 --require-go
```

For control campaigns, keep the same local provider keys and override only the
tested model identity at the CLI layer:

```powershell
python .\eval_cli.py campaign --job smoke_10 --repeat 1 --live --tested-model claude-sonnet-4-6 --tested-provider-id tested_airouting_sonnet46 --campaign-id CMP-LIVE-SONNET46-SMOKE10
python .\eval_cli.py campaign --job smoke_10 --repeat 1 --live --tested-model invalid-negative-control-model --tested-provider-id negative_invalid_model --campaign-id CMP-LIVE-NEGATIVE-INVALID --retries 0
```

These overrides are runtime-only; they do not write back to
`configs/providers.local.json`.

Campaign files are runtime artifacts under `campaigns/CMP-.../`:

```text
campaign.json   fixed test condition, redacted model identity, versions, git commit, config hash
run_ids.json    child run IDs and per-round status
summary.json    aggregate metrics, separated decisions, trend and evidence data
artifacts/      campaign acceptance pack after campaign-export
```

The model identity/quality evidence decision and gateway reliability decision
are separate. Transport failures lower gateway reliability, but are not treated
as proof of poor model identity or proof that a provider is not using the
claimed upstream model. API keys are never written; campaign metadata stores
only redacted config and a SHA-256 key fingerprint prefix.

Live provider calls retry transient transport failures by default (`2` retries
unless overridden by job config or CLI). Resume does not overwrite old child
runs: incomplete rounds are rerun with a new attempt id and the previous run is
marked `replaced`.

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

Acceptance packs are safe-by-default. `export` and `campaign-export` include
summary evidence, quality gates, `acceptance_manifest.json`, and
`checksums.sha256`; raw model responses, judge responses, and per-request event
logs are excluded unless `--include-raw` is passed:

```powershell
python .\eval_cli.py export --latest --include-raw
python .\eval_cli.py campaign-export --campaign-id CMP-... --include-raw
```

## Install

```powershell
cd E:\ai\ai测试\eval_automation_v0_2
python -m pip install -r requirements.txt
```

## Config

For the v0.2.x CLI/dashboard flow, create the two-model provider file from the
template:

```powershell
Copy-Item .\configs\providers.example.json .\configs\providers.local.json
```

Edit `configs/providers.local.json` with your tested model, judge model, Base
URL, model names, protocol, auth type, and auth environment variable names. Do
not commit `configs/providers.local.json`.

The root-level `providers.example.json` is retained for the older `run_eval.py`
flow only.

Set secrets in the shell or create `local_secrets.env`:

```env
TESTED_MODEL_API_KEY=your_tested_model_key_here
JUDGE_MODEL_API_KEY=your_judge_model_key_here
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

Preferred full local gate:

```powershell
.\scripts\check_all.ps1
```

Fresh package smoke only:

```powershell
.\scripts\package_smoke.ps1
```

Lower-level checks:

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
