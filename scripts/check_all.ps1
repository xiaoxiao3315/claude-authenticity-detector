[CmdletBinding()]
param(
    [switch]$SkipDryRun,
    [switch]$SkipPackageSmoke
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    Write-Host "[check] python compile"
    python -m py_compile eval_cli.py campaigns.py api_server.py run_records.py benchmarking.py quality_gate.py trace_evaluation.py acceptance_pack.py redaction.py authenticity.py

    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) {
        Write-Host "[check] web app syntax"
        node --check web\app.js
    } else {
        Write-Warning "node not found; skipping web/app.js syntax check"
    }

    Write-Host "[check] self-tests"
    python .\scripts\run_all_selftests.py
    if ($LASTEXITCODE -ne 0) { throw "self-tests failed (exit $LASTEXITCODE)" }

    if (-not $SkipDryRun) {
        $campaignId = "CMP-CHECK-" + (Get-Date -Format "yyyyMMddHHmmss")
        Write-Host "[check] dry-run campaign $campaignId"
        python .\eval_cli.py campaign --job smoke_10 --providers configs\providers.example.json --repeat 1 --campaign-id $campaignId
        python .\eval_cli.py campaign-status --campaign-id $campaignId
        Write-Host "[check] dry-run capability-probe"
        python .\eval_cli.py capability-probe --providers configs\providers.example.json --baselines-dir $env:TEMP\capcheck
        python .\eval_cli.py authenticity --job smoke_10 --providers configs\providers.example.json --campaign-id $campaignId --repeat 1 --baseline-provider official_dry_run --gateway-provider gateway_dry_run
        python .\eval_cli.py authenticity-inspect --campaign-id $campaignId
        python .\eval_cli.py authenticity-export --campaign-id $campaignId --baseline-provider official_dry_run --gateway-provider gateway_dry_run
        $packPath = Join-Path $RepoRoot "campaigns\$campaignId\artifacts\acceptance_pack.zip"
        $env:CHECK_ACCEPTANCE_PACK = $packPath
        try {
            @'
import os
from pathlib import Path
from acceptance_pack import verify_acceptance_pack
pack = Path(os.environ["CHECK_ACCEPTANCE_PACK"])
result = verify_acceptance_pack(pack)
print(result)
raise SystemExit(0 if result.get("verified") else 1)
'@ | python -
        } finally {
            Remove-Item Env:\CHECK_ACCEPTANCE_PACK -ErrorAction SilentlyContinue
        }
    }

    if (-not $SkipPackageSmoke) {
        Write-Host "[check] package smoke"
        & (Join-Path $PSScriptRoot "package_smoke.ps1")
    }

    Write-Host "CHECK-ALL-OK"
} finally {
    Pop-Location
}
