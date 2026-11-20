<#
.SYNOPSIS
  Experiment 2: vanilla BO vs an LLM agent that may RECONFIGURE the optimizer (acquisition function,
  surrogate incl. CAKE, bounds, objectives, constraints; -Freedom all adds region/sampler/belief).
  Runs seed after seed (vanilla on CPU in parallel with the agent on GPU) and, after EVERY finished seed,
  rebuilds the comparison and saves the plot. Runs until Ctrl+C, -MaxSeeds or -Hours.

  Writes to NEW folders, so experiment 1 is never touched:
    runs:   results\logs\ollama-free\...
    group:  results\logs\analysis-free-hartmann6\...
    plot:   results\figures\regret_free_vs_vanilla.png   (+ a snapshot per seed count in results\figures\free_snapshots\)
  Experiment 1's pinned-agent runs (results\logs\ollama-batch) are only READ, as a third line on the plot.

.EXAMPLE
  .\scripts\ollama_free_loop.ps1                       # seeds 42, 43, ... until Ctrl+C
  .\scripts\ollama_free_loop.ps1 -Hours 8
  .\scripts\ollama_free_loop.ps1 -Freedom all -MaxSeeds 10
  .\scripts\ollama_free_loop.ps1 -SmokeTest            # one short run (budget 12) to a scratch folder

  Resume: run it again; unfinished runs from an interrupted session are deleted and redone.
  Run only ONE loop at a time (do not run it together with ollama_seed_loop.ps1: same GPU, same model).
#>
param(
  [string]$Model = "qwen3.6:27b",
  [string]$Bench = "hartmann6",
  [int]$StartSeed = 42,      # same seeds as experiment 1 so the comparison is paired
  [int]$MaxSeeds = 0,        # 0 = no limit
  [double]$Hours = 0,        # 0 = no limit
  [int]$Budget = 50,
  [ValidateSet("paper", "all")][string]$Freedom = "paper",
  [int]$BashTimeout = 600,   # seconds per agent shell command; CAKE/LLAMBO call an LLM inside lenz and are slow
  [switch]$Hint,             # ALSO run a third condition per seed: free agent + a vague, true hint (doubles agent time)
  [switch]$HintOnly,         # run ONLY the hinted free agent (plus vanilla) -- no unhinted free agent
  [switch]$SmokeTest
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$bash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git Bash not found at $bash" }

$out = "results/logs/ollama-free"
$group = "results/logs/analysis-free-hartmann6"
$fig = "results/figures/regret_free_vs_vanilla.png"
if ($SmokeTest) {
  # everything goes to scratch locations so real results and the real plot are untouched
  $Budget = 12; $MaxSeeds = 1; $StartSeed = 9001
  $out = "results/logs/_free_smoketest"; $group = "results/logs/_free_smoketest_group"; $fig = "results/figures/_free_smoketest.png"
}
$agentBackend = if ($Freedom -eq "all") { "sara-lenz-free-all" } else { "sara-lenz-free" }
if ($Hint) { $agentBackend = "$agentBackend,sara-lenz-free-hint" }
if ($HintOnly) { $agentBackend = "sara-lenz-free-hint" }
$tag = $Model -replace '[:/]', '_'
$root = Join-Path $out.Replace('/', '\') "$tag\$Bench-blind"

$env:OLLAMA_MODEL = $Model
$env:BENCHES = $Bench
$env:BUDGET = "$Budget"
$env:OUT = $out
$env:SARA_ENFORCE_BUDGET = "1"
$env:SARA_BASH_TIMEOUT = "$BashTimeout"
$deadline = if ($Hours -gt 0) { (Get-Date).AddHours($Hours) } else { $null }

function Log-Failures {
  & python scripts/log_failures.py --free-root $out --model-tag $tag --bench $Bench
}

function Remove-StaleRuns {
  # Unfinished sandboxes from an interrupted session would be skipped forever as "in flight".
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
  Log-Failures        # keep a record of failures before any cleanup deletes them
  Remove-StaleRuns
  while ($true) {
    if ($MaxSeeds -gt 0 -and $done -ge $MaxSeeds) { Write-Host "Reached -MaxSeeds $MaxSeeds."; break }
    if ($deadline -and (Get-Date) -gt $deadline) { Write-Host "Reached -Hours $Hours."; break }

    Write-Host "`n######## [free: $Freedom] seed $seed  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
    $env:SEEDS = "$seed"
    New-Item -ItemType Directory -Force "results\logs" | Out-Null

    # Vanilla BO (CPU) for this seed in its own process, in parallel with the agent (GPU).
    $env:BACKENDS = "vanilla"; $env:SKIP_PREFLIGHT = "1"; $env:NO_UNLOAD = "1"
    $vlog = Join-Path "results\logs" "free-vanilla-seed$seed.log"
    $vp = Start-Process $bash -ArgumentList "scripts/run_ollama_batch.sh" -NoNewWindow -PassThru `
            -RedirectStandardOutput $vlog -RedirectStandardError "$vlog.err"
    Write-Host "vanilla (CPU) started in parallel: pid $($vp.Id), log $vlog"
    Remove-Item Env:SKIP_PREFLIGHT, Env:NO_UNLOAD -ErrorAction SilentlyContinue

    $env:BACKENDS = $agentBackend
    & $bash "scripts/run_ollama_batch.sh"
    if ($LASTEXITCODE -ne 0) { Write-Warning "seed $seed agent leg reported failures (exit $LASTEXITCODE); continuing" }

    if (-not $vp.HasExited) { Write-Host "waiting for vanilla seed $seed to finish..."; $vp.WaitForExit() }
    if ($vp.ExitCode -ne 0) { Write-Warning "seed $seed vanilla leg failed (exit $($vp.ExitCode)); see $vlog.err" }

    Log-Failures
    # Save the plot after every seed
    & python scripts/make_free_group.py --free-root $out --model-tag $tag --bench $Bench --group $group
    & python scripts/plot_free_comparison.py --free-root $out --model-tag $tag --bench $Bench --group $group --fig $fig
    $done++; $seed++
  }
}
finally {
  Write-Host "`nCleaning up (stopping any child runs, unloading $Model from VRAM)..."
  Stop-Children
  Write-Host "Seeds finished this session: $done. Plot: $fig"
}
