# eval_automation_v0_2

Backend-only core for the local LLM evaluation workflow.

This folder intentionally excludes the old web console, historical runs, local secrets, campaigns, archives, screenshots, and generated logs. It keeps the evaluation kernel: benchmark execution, `run_record_v1`, compatibility checks, offline re-score, trace evaluation, quality gate, and redacted audit export.

## Install

```powershell
cd E:\ai\ai测试\eval_automation_v0_2
python -m pip install -r requirements.txt
```

## Config

Create a local provider file from the template:

```powershell
Copy-Item .\providers.example.json .\providers.local.json
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
```
