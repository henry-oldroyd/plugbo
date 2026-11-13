<#
.SYNOPSIS
  Re-run the LLM-agent (sara-lenz) experiments only, forcing the agent to use the whole budget.
  Runs in the foreground: Ctrl+C aborts. Finished runs are skipped, so rerunning resumes.

.EXAMPLE
  .\scripts\ollama_agent_rerun.ps1
  .\scripts\ollama_agent_rerun.ps1 -Model gpt-oss:20b -Seeds 42,43 -Budget 30
  .\scripts\ollama_agent_rerun.ps1 -Force        # redo runs that already completed
  .\scripts\ollama_agent_rerun.ps1 -NoEnforce    # old behaviour: agent may end the run early
#>
param(
  [string]$Model = "qwen3.6:27b",
  [int[]]$Seeds = @(42, 43, 44, 45, 46),
  [string[]]$Benches = @("hartmann6"),
  [string[]]$Disclosures = @("blind"),
  [string[]]$Backends = @("sara-lenz"),
  [int]$Budget = 50,
  [switch]$NoEnforce,
  [switch]$Force
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$bash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git Bash not found at $bash" }

$env:OLLAMA_MODEL      = $Model
$env:SEEDS             = $Seeds -join " "
$env:BENCHES           = $Benches -join " "
$env:DISCLOSURES       = $Disclosures -join " "
$env:BACKENDS          = $Backends -join ","
$env:BUDGET            = "$Budget"
$env:SARA_ENFORCE_BUDGET = if ($NoEnforce) { "0" } else { "1" }

Write-Host "model=$Model seeds=$($Seeds -join ',') benches=$($Benches -join ',') budget=$Budget backends=$($Backends -join ',') enforce_budget=$(-not $NoEnforce)"
$extra = @(); if ($Force) { $extra += "--force" }
& $bash "scripts/run_ollama_batch.sh" @extra
exit $LASTEXITCODE
