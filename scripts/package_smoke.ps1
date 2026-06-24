[CmdletBinding()]
param(
    [string]$PackageRoot = "",
    [int]$Port = 18082,
    [string]$CampaignId = "",
    [switch]$KeepTemp,
    [switch]$SkipPipDryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $CampaignId) {
    $CampaignId = "CMP-FRESH-PACKAGE-SMOKE-" + (Get-Date -Format "yyyyMMddHHmmss")
}

$createdTemp = $false
if (-not $PackageRoot) {
    $PackageRoot = Join-Path $env:TEMP ("eval_pkg_test_" + (Get-Date -Format "yyyyMMddHHmmss"))
    $createdTemp = $true
}

New-Item -ItemType Directory -Path $PackageRoot -Force | Out-Null
$PackageRoot = (Resolve-Path $PackageRoot).Path

Write-Host "[package] source $RepoRoot"
Write-Host "[package] temp   $PackageRoot"

$tracked = & git -C $RepoRoot ls-files
if ($LASTEXITCODE -ne 0) {
    throw "git ls-files failed"
}
$untracked = & git -C $RepoRoot ls-files --others --exclude-standard
if ($LASTEXITCODE -ne 0) {
    throw "git ls-files --others failed"
}

foreach ($file in @($tracked + $untracked)) {
    $src = Join-Path $RepoRoot $file
    $dest = Join-Path $PackageRoot $file
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $src -Destination $dest -Force
}

$forbidden = @(
    "local_secrets.env",
    ".env",
    "configs\providers.local.json",
    "runs",
    "campaigns"
)
foreach ($relative in $forbidden) {
    $path = Join-Path $PackageRoot $relative
    if (Test-Path $path) {
        throw "forbidden package path copied before setup: $relative"
    }
}

Copy-Item -LiteralPath (Join-Path $PackageRoot "configs\providers.example.json") -Destination (Join-Path $PackageRoot "configs\providers.local.json") -Force

Push-Location $PackageRoot
$server = $null
try {
    if (-not $SkipPipDryRun) {
        Write-Host "[package] pip dry-run"
        python -m pip install -r requirements.txt --dry-run
    }

    Write-Host "[package] syntax checks"
    python -m py_compile eval_cli.py campaigns.py api_server.py run_records.py benchmarking.py quality_gate.py trace_evaluation.py acceptance_pack.py redaction.py
    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) {
        node --check web\app.js
    } else {
        Write-Warning "node not found; skipping web/app.js syntax check"
    }

    Write-Host "[package] dry-run campaign $CampaignId"
    python .\eval_cli.py campaign --job smoke_10 --repeat 1 --campaign-id $CampaignId
    python .\eval_cli.py campaign-status --campaign-id $CampaignId
    python .\eval_cli.py campaign-export --campaign-id $CampaignId

    Write-Host "[package] acceptance pack contents"
    $packPath = Join-Path $PackageRoot "campaigns\$CampaignId\artifacts\acceptance_pack.zip"
    if (-not (Test-Path $packPath)) {
        throw "acceptance pack missing: $packPath"
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($packPath)
    try {
        $entryNames = @($zip.Entries | ForEach-Object { $_.FullName })
    } finally {
        $zip.Dispose()
    }
    foreach ($required in @(
        "campaign.json",
        "summary.json",
        "run_ids.json",
        "acceptance_manifest.json",
        "checksums.sha256",
        "runs/$CampaignId-R01/run_records.jsonl",
        "runs/$CampaignId-R01/quality_gates/",
        "runs/$CampaignId-R01/trace_evaluations/"
    )) {
        $found = $false
        foreach ($name in $entryNames) {
            if ($name -eq $required -or $name.StartsWith($required)) {
                $found = $true
                break
            }
        }
        if (-not $found) {
            throw "acceptance pack missing required entry prefix: $required"
        }
    }
    foreach ($forbiddenPrefix in @(
        "runs/$CampaignId-R01/responses/",
        "runs/$CampaignId-R01/judge_responses/",
        "runs/$CampaignId-R01/events/",
        "local_secrets.env",
        ".env",
        "configs/providers.local.json"
    )) {
        foreach ($name in $entryNames) {
            if ($name -eq $forbiddenPrefix -or $name.StartsWith($forbiddenPrefix)) {
                throw "acceptance pack contains forbidden entry prefix: $forbiddenPrefix"
            }
        }
    }
    $env:PACKAGE_SMOKE_PACK = $packPath
    try {
        @'
import os
from pathlib import Path
from acceptance_pack import verify_acceptance_pack
pack = Path(os.environ["PACKAGE_SMOKE_PACK"])
result = verify_acceptance_pack(pack)
print(result)
raise SystemExit(0 if result.get("verified") else 1)
'@ | python -
    } finally {
        Remove-Item Env:\PACKAGE_SMOKE_PACK -ErrorAction SilentlyContinue
    }

    Write-Host "[package] api smoke on port $Port"
    $server = Start-Process -FilePath python -ArgumentList ".\api_server.py", "--host", "127.0.0.1", "--port", "$Port" -WorkingDirectory $PackageRoot -WindowStyle Hidden -PassThru
    $url = "http://127.0.0.1:$Port/api/leaderboard?include_dry_run=true"
    $content = $null
    for ($i = 0; $i -lt 20; $i++) {
        try {
            $content = (Invoke-WebRequest -UseBasicParsing $url).Content
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $content) {
        throw "API smoke failed: $url"
    }
    if ($content -notmatch [regex]::Escape($CampaignId)) {
        throw "API smoke did not include campaign id $CampaignId"
    }

    Write-Host "PACKAGE-SMOKE-OK $PackageRoot"
} finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
    Pop-Location
    if ($createdTemp -and -not $KeepTemp) {
        $resolvedTemp = Resolve-Path $PackageRoot
        $resolvedBase = Resolve-Path $env:TEMP
        if ($resolvedTemp.Path.StartsWith($resolvedBase.Path)) {
            Remove-Item -LiteralPath $resolvedTemp.Path -Recurse -Force
        } else {
            Write-Warning "refusing to remove non-temp path: $resolvedTemp"
        }
    }
}
