# PowerShell launcher: runs a run_ollama_*.sh script under Git Bash (not WSL).
#   .\scripts\ollama.ps1 synthetic hartmann6
#   .\scripts\ollama.ps1 sweep
#   .\scripts\ollama.ps1 bolt
param([Parameter(Mandatory=$true)][ValidateSet('smoke','batch','synthetic','sweep','bolt')][string]$Kind,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
$bash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git Bash not found at $bash" }
Set-Location (Split-Path $PSScriptRoot -Parent)
& $bash "scripts/run_ollama_$Kind.sh" @Rest
exit $LASTEXITCODE
