<#
.SYNOPSIS
  Keep running new seeds (vanilla BO on CPU, in parallel with the forced-continue LLM agent on GPU) and refresh the
  mean +/- standard-error regret plot after EVERY finished seed. Runs until you press
  Ctrl+C, or until -MaxSeeds / -Hours is reached (a seed already started is always finished).

.EXAMPLE
  .\scripts\ollama_seed_loop.ps1                    # seeds 47, 48, 49, ... until Ctrl+C
  .\scripts\ollama_seed_loop.ps1 -Hours 8           # stop starting new seeds after 8 hours
  .\scripts\ollama_seed_loop.ps1 -StartSeed 100 -MaxSeeds 10

  Output plot (rewritten after each seed): results\figures\regret_seed_distribution.png
  Resume: just run it again. Unfinished runs from an interrupted session are deleted and redone.
  Run only ONE copy at a time (it assumes nothing else is writing to the results folders).
#>
param(
  [string]$Model = "qwen3.6:27b",
  [string]$Bench = "hartmann6",
  [int]$StartSeed = 47,
  [int]$MaxSeeds = 0,        # 0 = no limit
  [double]$Hours = 0,        # 0 = no limit
  [int]$Budget = 50
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$bash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git Bash not found at $bash" }

$env:OLLAMA_MODEL = $Model
$env:BENCHES = $Bench
$env:BUDGET = "$Budget"
$env:SARA_ENFORCE_BUDGET = "1"
$root = "results\logs\ollama-batch\$($Model -replace '[:/]','_')\$Bench-blind"
$deadline = if ($Hours -gt 0) { (Get-Date).AddHours($Hours) } else { $null }

function Remove-StaleRuns {
  # Unfinished sandboxes (status 'running') from an interrupted session would be skipped forever as in-flight.
  if (-not (Test-Path $root)) { return }
  Get-ChildItem $root -Recurse -Filter run_meta.json -ErrorAction SilentlyContinue | ForEach-Object {
    try { $s = (Get-Content $_.FullName -Raw | ConvertFrom-Json).status } catch { $s = "running" }
    if ($s -ne "completed") { Write-Host "removing unfinished run $($_.DirectoryName)"; Remove-Item -Recurse -Force $_.DirectoryName }
  }
}

function Stop-Children {
  Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'run_ollama_batch|run_synthetic|run_blind|benchmarks\.|/tmp/tmp\.' -and $_.Name -match '^(bash|python)\.exe$'
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  & ollama stop $Model 2>$null
}

$done = 0
$seed = $StartSeed
try {
  Remove-StaleRuns
  while ($true) {
    if ($MaxSeeds -gt 0 -and $done -ge $MaxSeeds) { Write-Host "Reached -MaxSeeds $MaxSeeds."; break }
    if ($deadline -and (Get-Date) -gt $deadline) { Write-Host "Reached -Hours $Hours."; break }

    Write-Host "`n######## seed $seed  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
    $env:SEEDS = "$seed"
    New-Item -ItemType Directory -Force "results\logs" | Out-Null

    # CPU-only vanilla BO for this seed runs in parallel, in its own process, while the agent
    # uses the GPU. They stay separate experiments (own folders, own logs). The vanilla job
    # skips the model preflight and must not unload the model out from under the agent.
    $env:BACKENDS = "vanilla"; $env:SKIP_PREFLIGHT = "1"; $env:NO_UNLOAD = "1"
    $vlog = Join-Path "results\logs" "vanilla-seed$seed.log"
    $vp = Start-Process $bash -ArgumentList "scripts/run_ollama_batch.sh" -NoNewWindow -PassThru `
            -RedirectStandardOutput $vlog -RedirectStandardError "$vlog.err"
    Write-Host "vanilla (CPU) started in parallel: pid $($vp.Id), log $vlog"
    Remove-Item Env:SKIP_PREFLIGHT, Env:NO_UNLOAD -ErrorAction SilentlyContinue

    $env:BACKENDS = "sara-lenz"   # agent leg (GPU) in the foreground
    & $bash "scripts/run_ollama_batch.sh"
    $agentExit = $LASTEXITCODE
    if ($agentExit -ne 0) { Write-Warning "seed $seed agent leg reported failures (exit $agentExit); continuing" }

    if (-not $vp.HasExited) { Write-Host "waiting for vanilla seed $seed to finish..."; $vp.WaitForExit() }
    if ($vp.ExitCode -ne 0) { Write-Warning "seed $seed vanilla leg failed (exit $($vp.ExitCode)); see $vlog.err" }

    # Refresh the comparison group + plot after every seed
    & python scripts/make_comparison_group.py | Out-Null
    & python scripts/plot_seed_distribution.py
    $done++; $seed++
  }
}
finally {
  Write-Host "`nCleaning up (stopping any child runs, unloading $Model from VRAM)..."
  Stop-Children
  Write-Host "Seeds finished this session: $done. Plot: results\figures\regret_seed_distribution.png"
}
