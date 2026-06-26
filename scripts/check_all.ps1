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
    python .\eval_cli.py self-test
    python .\validate_run_records.py --self-test
    python .\quality_gate.py --self-test
    python .\compatibility.py --self-test
    python .\trace_evaluation.py --self-test
    python .\audit_export.py --self-test
    python .\authenticity.py --self-test
    python .\baseline_registry.py --self-test

    if (-not $SkipDryRun) {
        $campaignId = "CMP-CHECK-" + (Get-Date -Format "yyyyMMddHHmmss")
        Write-Host "[check] dry-run campaign $campaignId"
        python .\eval_cli.py campaign --job smoke_10 --providers configs\providers.example.json --repeat 1 --campaign-id $campaignId
        python .\eval_cli.py campaign-status --campaign-id $campaignId
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
