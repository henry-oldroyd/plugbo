<#
.SYNOPSIS
  Full local-Ollama pipeline: configure Ollama -> pull model -> smoke test -> batch runs -> viewer.

.EXAMPLE
  .\scripts\ollama_pipeline.ps1                       # foreground, defaults
  .\scripts\ollama_pipeline.ps1 -Background           # hidden window, log in results\logs\pipeline-*.log
  .\scripts\ollama_pipeline.ps1 -Seeds 43,44,45 -Benches hartmann6,branin -Budget 50 -Background
  .\scripts\ollama_pipeline.ps1 -SkipSmoke -Model gpt-oss:20b

  If PowerShell blocks scripts:  powershell -ExecutionPolicy Bypass -File scripts\ollama_pipeline.ps1 -Background

  Windows only (Git Bash + Start-Process). On Linux/macOS use  ./scripts/ollama_pipeline.sh  (same options).
  Default model is chosen from the detected GPUs (same rule as scripts/_gpu_profile.sh):
    >= 40 GiB total VRAM (e.g. 2x RTX 3090): deepseek-r1:32b-qwen-distill-q8_0, i.e. Qwen2.5-32B (DeepSeek-R1
      distill) at int8. At -Context 32768 with the q8 KV cache it loads as 39 GB, 100% GPU, ~19 GB per card.
    otherwise (e.g. one 24 GB RTX 4090): qwen3.6:27b.
  Pass -Model to override, or -GpuProfile large|small to force a profile.

  NOTE: this restarts Ollama (so the context / KV-cache settings take effect).
  When it finishes the model is unloaded from VRAM; `ollama serve` is left running.
#>
param(
  [string]$Model = "",                 # empty = pick from the detected GPUs
  [ValidateSet("", "large", "small")][string]$GpuProfile = "",
  [int]$Context = 32768,
  [int[]]$Seeds = @(42, 43, 44, 45, 46),
  [string[]]$Benches = @("hartmann6"),
  [string[]]$Disclosures = @("blind"),
  [string[]]$Backends = @("vanilla", "sara-lenz"),
  [int]$Budget = 50,
  [switch]$SkipSmoke,
  [switch]$NoViz,
  [switch]$Background
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
New-Item -ItemType Directory -Force "results\logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

# --- Pick the model from the GPUs present (mirror of scripts/_gpu_profile.sh) -------------
$totalMiB = 0; $gpuCount = 0
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  try {
    foreach ($line in (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null)) {
      if ("$line".Trim() -match '^\d+$') { $totalMiB += [int]"$line".Trim(); $gpuCount++ }
    }
  } catch { }
}
$gpuTier = if ($totalMiB -ge 40960) { "large" } else { "small" }
if ($GpuProfile) { $gpuTier = $GpuProfile }
$tierModel = if ($gpuTier -eq "large") { "deepseek-r1:32b-qwen-distill-q8_0" } else { "qwen3.6:27b" }
if (-not $Model) { $Model = $tierModel }
Write-Host "Detected: $gpuCount GPU(s), $([math]::Floor($totalMiB / 1024)) GiB total VRAM -> profile '$gpuTier'"
Write-Host "Using:    model=$Model context=$Context"

# --- Relaunch hidden in the background, logging to a file ---------------------------------
if ($Background) {
  $log = Join-Path $repo "results\logs\pipeline-$stamp.log"
  $ps = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }
  $argsList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"",
    "-Model", $Model, "-Context", $Context, "-Budget", $Budget,
    "-Seeds", ($Seeds -join ","), "-Benches", ($Benches -join ","),
    "-Disclosures", ($Disclosures -join ","), "-Backends", ($Backends -join ","))
  if ($SkipSmoke) { $argsList += "-SkipSmoke" }
  if ($NoViz) { $argsList += "-NoViz" }
  $p = Start-Process $ps -ArgumentList $argsList -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  Write-Host "Started in background (pid $($p.Id)).`n  log:  Get-Content '$log' -Wait`n  errs: $log.err`n  stop: Stop-Process -Id $($p.Id) -Force   (then: ollama stop $Model)"
  exit 0
}

$bash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git Bash not found at $bash" }

function Step($msg) { Write-Host "`n=== $msg  ($(Get-Date -Format HH:mm:ss)) ===" -ForegroundColor Cyan }

# --- 1. Configure + (re)start Ollama ------------------------------------------------------
Step "Restarting Ollama (context=$Context, flash attention, q8 KV cache)"
$env:OLLAMA_CONTEXT_LENGTH  = "$Context"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE   = "q8_0"
Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
$serve = Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden -PassThru
$ready = $false
foreach ($i in 1..120) {
  if ($serve.HasExited) { throw "ollama serve exited immediately (code $($serve.ExitCode)); is another instance holding port 11434?" }
  try { Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null; $ready = $true; break }
  catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { throw "Ollama did not respond on 127.0.0.1:11434 within 2 minutes" }

# --- 2. Pull model ------------------------------------------------------------------------
Step "Pulling $Model"
& ollama pull $Model
if ($LASTEXITCODE -ne 0) { throw "ollama pull $Model failed (check the tag with 'ollama list' / the Ollama library)" }

$env:OLLAMA_MODEL = $Model

# --- 3. Viewer (live, so you can watch runs appear) --------------------------------------
if (-not $NoViz) {
  Step "Starting viewer at http://127.0.0.1:8765"
  Start-Process "plugbo-viz" -ArgumentList "--root", "results/logs" -WindowStyle Hidden
}

# --- 4. Smoke test ------------------------------------------------------------------------
if (-not $SkipSmoke) {
  Step "Smoke test"
  & $bash "scripts/run_ollama_smoke.sh"
  if ($LASTEXITCODE -ne 0) { throw "Smoke test failed (exit $LASTEXITCODE); not starting the long run" }
}

# --- 5. Batch -----------------------------------------------------------------------------
Step "Batch: seeds=$($Seeds -join ',') benches=$($Benches -join ',') budget=$Budget backends=$($Backends -join ',')"
$env:SEEDS       = $Seeds -join " "
$env:BENCHES     = $Benches -join " "
$env:DISCLOSURES = $Disclosures -join " "
$env:BACKENDS    = $Backends -join ","
$env:BUDGET      = "$Budget"
& $bash "scripts/run_ollama_batch.sh"
$code = $LASTEXITCODE

# --- 6. Done ------------------------------------------------------------------------------
Step "Finished (batch exit code $code)"
Write-Host "Results: results\logs\ollama-batch   Viewer: http://127.0.0.1:8765  (or: plugbo-viz --root results/logs)"
exit $code
