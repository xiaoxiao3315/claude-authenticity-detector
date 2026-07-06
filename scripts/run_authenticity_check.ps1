# Claude 真伪检测 — 一键自助脚本
#
# 用途：对配置好的 suspect_model 网关跑完整真伪检测（四类攻击），输出中文判定。
# 不需要记 CLI 命令，填好配置后一条命令搞定。
#
# 用法（先按 docs\真伪检测使用手册.md 填好 key 和网关）：
#   .\scripts\run_authenticity_check.ps1                 # dry-run（不花额度，验证配置/管线）
#   .\scripts\run_authenticity_check.ps1 -Live           # 真实检测（调网关，花额度）
#   .\scripts\run_authenticity_check.ps1 -Live -Full     # 真实检测 + needle 假1M探针（慢/贵）
#
# 参数：
#   -Provider    要检测的角色（默认 suspect_model；配置文件里定义）
#   -BaselineId  对比基线（默认 OFFICIAL-CLAUDE-OPUS46，已内置真官方 opus-4-6 指纹）
#   -Live        真实调用网关（默认 off = dry-run）
#   -Full        额外跑 needle 假1M探针（>200K请求，慢且贵）
#   -Delay       请求间隔秒数（默认 1.0，避免限流）

[CmdletBinding()]
param(
    [string]$Provider = "suspect_model",
    [string]$BaselineId = "OFFICIAL-CLAUDE-OPUS46",
    [switch]$Live,
    [switch]$Full,
    [double]$Delay = 1.0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    Write-Host ""
    Write-Host "==== Claude 真伪检测 ====" -ForegroundColor Cyan
    Write-Host ""

    # --- 1. 前置检查：基线存在吗 ---
    $baselineFile = Join-Path $RepoRoot "baselines\$BaselineId\baseline.json"
    if (-not (Test-Path $baselineFile)) {
        Write-Host "[X] 找不到基线 $BaselineId" -ForegroundColor Red
        Write-Host "    需要先用一个【可信的官方 Claude 源】建立基线："
        Write-Host "    python .\eval_cli.py baseline --provider tested_model --baseline-id $BaselineId --live --request-delay 1.0"
        Write-Host "    （tested_model 必须指向你信任的真官方源，见使用手册）"
        exit 1
    }
    Write-Host "[OK] 基线就绪：$BaselineId" -ForegroundColor Green

    $capFile = Join-Path $RepoRoot "baselines\$BaselineId\capability_anchor.json"
    $withCapability = Test-Path $capFile
    if ($withCapability) {
        Write-Host "[OK] 能力基线就绪（可检测偷降级）" -ForegroundColor Green
    } else {
        Write-Host "[!] 无能力基线，跳过降级检测。补建：" -ForegroundColor Yellow
        Write-Host "    python .\eval_cli.py capability-probe --provider tested_model --baseline-id $BaselineId --live"
    }

    # --- 2. 检查配置里有没有这个角色 ---
    $providersFile = Join-Path $RepoRoot "configs\providers.local.json"
    if (-not (Test-Path $providersFile)) {
        Write-Host "[X] 找不到 configs\providers.local.json，请先按使用手册配置。" -ForegroundColor Red
        exit 1
    }

    # --- 3. 组装命令 ---
    $cliArgs = @(
        ".\eval_cli.py", "verify-endpoint",
        "--baseline-id", $BaselineId,
        "--provider", $Provider,
        "--request-delay", $Delay,
        "--retries", "1",
        "--retry-backoff", "2.0",
        "--with-sse",
        "--with-error-envelope"
    )
    if ($withCapability) { $cliArgs += "--with-capability" }
    if ($Full) { $cliArgs += @("--with-needle", "--needle-tokens", "120000") }
    if ($Live) {
        $cliArgs += "--live"
        Write-Host ""
        Write-Host "[模式] 真实检测（会调用网关、消耗额度）" -ForegroundColor Magenta
    } else {
        Write-Host ""
        Write-Host "[模式] dry-run 演练（不调网关、不花额度，仅验证配置与管线）" -ForegroundColor Yellow
        Write-Host "       真实检测请加 -Live" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "正在检测角色 '$Provider'，对比基线 '$BaselineId' ..." -ForegroundColor Cyan
    Write-Host ("命令: python " + ($cliArgs -join " ")) -ForegroundColor DarkGray
    Write-Host ""

    # --- 4. 执行 ---
    $env:PYTHONUTF8 = "1"
    python @cliArgs

    Write-Host ""
    Write-Host "==== 检测完成 ====" -ForegroundColor Cyan
    if (-not $Live) {
        Write-Host "以上为 dry-run 结果（无真实判定意义）。确认配置无误后，加 -Live 跑真实检测。" -ForegroundColor Yellow
    } else {
        Write-Host "判定说明：✅真·官方 / ⚠️疑似降级 / ❌疑似套壳 / ❔证据不足" -ForegroundColor Green
        Write-Host "结果含分级证据链（强证据/佐证/仅参考）。可视化报告页见 React 站 /authenticity。"
    }
}
finally {
    Pop-Location
}
