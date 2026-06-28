# eval_automation_v0_2

Backend-only core for local LLM evaluation **and Claude authenticity detection**.

This folder keeps the evaluation kernel (benchmark execution, `run_record_v1`,
compatibility checks, offline re-score, trace evaluation, quality gate, redacted
audit export) **and a Claude official-authenticity detector** that answers one
question: *is the model behind a gateway really official Claude, or has it been
swapped / wrapped / downgraded / faking its context window?*

It intentionally excludes the old web console, historical runs, local secrets,
campaigns, archives, screenshots, and generated logs.

---

## ⭐ Claude 真伪检测（项目头号能力）

Black-box detection of four ways a gateway can fake "official Claude":

| 攻击 | 含义 | 探针 |
|---|---|---|
| 换模型 / 套壳 | 后面其实是 GPT 等，套个 Claude 壳 | 协议指纹 / SSE / 错误体 |
| 换 tokenizer | 用了非 Claude 分词器 | token 差分 |
| 假 1M 上下文 | 声称长上下文，实际静默截断 | needle 召回 |
| 偷降级 | 还是 Claude，但偷换更小的（opus→haiku） | 能力锚点通过率 |

最终给四类结论之一：**✅真·官方 / ⚠️疑似降级 / ❌疑似套壳 / ❔证据不足**，附分级证据链（强证据 / 佐证 / 仅参考）。

### 自助三步走（脱离记 CLI）

完整使用手册见 **[`docs/真伪检测使用手册.md`](docs/真伪检测使用手册.md)**。最短路径：

```powershell
# 1. 填要测的网关：configs/providers.local.json 的 suspect_model
#    填 key：local_secrets.env 的 SUSPECT_MODEL_API_KEY（key 绝不进 json）
# 2. dry-run 验证配置（不花额度）
.\scripts\run_authenticity_check.ps1
# 3. 真实检测（调网关、出中文判定）
.\scripts\run_authenticity_check.ps1 -Live
.\scripts\run_authenticity_check.ps1 -Live -Full   # 额外跑 needle 假1M探针（慢/贵）
```

### 底层 CLI（脚本就是它的封装）

```powershell
# 建可信基线（用一个你信任的真官方源；默认 OFFICIAL-CLAUDE-OPUS46 已就绪）
python .\eval_cli.py baseline --provider tested_model --baseline-id OFFICIAL-CLAUDE-OPUS46 --live --request-delay 1.0
python .\eval_cli.py capability-probe --provider tested_model --baseline-id OFFICIAL-CLAUDE-OPUS46 --live   # 能力基线（抓降级）
# 检测一个可疑网关（四合一）
python .\eval_cli.py verify-endpoint --baseline-id OFFICIAL-CLAUDE-OPUS46 --provider suspect_model --live --with-sse --with-error-envelope --with-capability
# 基线版本化 / 漂移
python .\eval_cli.py baseline-versions --baseline-id OFFICIAL-CLAUDE-OPUS46
python .\eval_cli.py baseline-diff --baseline-id OFFICIAL-CLAUDE-OPUS46
# 校准评审模型本身是否可信
python .\eval_cli.py judge-calibrate --golden-set judge_golden/golden_set_v1.json --live --report
```

### 重要边界

- **基线会过期**：网关上游会变（已实测到 drhknode 的 error-envelope 行为漂移）。基线超过一段时间应重建；版本化（`baseline-versions`/`baseline-diff`）会留痕历史指纹。
- **黑盒限制**：能证明"接口表现得像/不像官方 Claude"、能抓套壳/降级/假长上下文，但不能单独证明背后一定是官方上游——真正身份保证仍需官方账号/合同/日志。
- 永不读写 key；待测网关必须配在 `providers.local.json`（CLI 无 suspect 覆盖参数）。
- 可视化报告页：React 站 `../llm_eval_result_site` 的 `/authenticity`。

---

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

Each run performs a post-run trace evidence pass before the quality gate by
default. Use `--skip-trace-evaluation` only when you intentionally want to omit
trace evidence from the gate:

```powershell
python .\eval_cli.py run --job smoke_10 --skip-trace-evaluation
```

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

Task-level concurrency is bounded and conservative. Job configs default to `1`
to avoid gateway rate-limit noise; raise it explicitly only when you want a
controlled throughput test:

```powershell
python .\eval_cli.py run --job smoke_100 --max-concurrency 2
python .\eval_cli.py campaign --job smoke_100 --repeat 1 --max-concurrency 2
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

The API marks `acceptance_pack.zip` artifacts with checksum verification status
and refuses to download packs that are missing the manifest/checksum pair. If a
run or campaign was exported before this safe-pack format, re-run `export` or
`campaign-export` for that item.

## v0.3 Provider Authenticity Evidence Layer

The v0.3 layer treats provider trust as evidence, not as a single benchmark
score. It keeps the existing campaign runner and adds campaign-level evidence
for model quality, gateway reliability, protocol fingerprint, official or
direct baseline similarity, auditability, and overall trust.

Black-box API testing cannot prove upstream identity absolutely. Missing
upstream request IDs or official baselines lower auditability or similarity
confidence; they do not by themselves prove model substitution.

Dry-safe commands:

```powershell
python .\eval_cli.py authenticity --job smoke_10 --campaign-id CMP-AUTH-SMOKE --repeat 1 --baseline-provider official_dry_run --gateway-provider gateway_dry_run
python .\eval_cli.py fingerprint --provider tested_model
python .\eval_cli.py authenticity-inspect --campaign-id CMP-AUTH-SMOKE
python .\eval_cli.py authenticity-export --campaign-id CMP-AUTH-SMOKE
```

To compare a gateway campaign with an official/direct baseline campaign, run
both campaigns under comparable job, judge, policy, and live/dry-run settings,
then pass the official campaign as the baseline:

```powershell
python .\eval_cli.py authenticity --job smoke_10 --campaign-id CMP-GATEWAY --baseline-campaign-id CMP-OFFICIAL --baseline-provider official_direct --gateway-provider gateway_candidate
```

`authenticity_summary.json` contains separated decisions:

```text
model_quality_decision
gateway_reliability_decision
protocol_fingerprint_decision
baseline_similarity_decision
auditability_decision
overall_trust_decision
```

Campaign evidence files:

```text
authenticity_summary.json
baseline_comparisons/baseline_comparison.json
protocol_fingerprints/<provider_id>.json
artifacts/acceptance_pack.zip
```

The authenticity export path remains safe-by-default. It includes sanitized
campaign summaries, protocol and baseline evidence, quality gates, manifest,
and checksums. Raw responses, judge responses, and event logs remain excluded
unless `--include-raw` is passed explicitly.

Authenticity API routes:

```text
GET /api/authenticity/latest
GET /api/campaigns/<campaign_id>/authenticity
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
