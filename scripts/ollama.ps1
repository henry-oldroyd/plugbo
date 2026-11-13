# PowerShell launcher: runs a run_ollama_*.sh script under bash. Works on Windows (Git Bash, NOT WSL)
# and on Linux/macOS with PowerShell 7 (plain `bash`). On Linux you can also just run the .sh directly.
#   .\scripts\ollama.ps1 synthetic hartmann6
#   .\scripts\ollama.ps1 sweep
#   .\scripts\ollama.ps1 bolt
param([Parameter(Mandatory=$true)][ValidateSet('smoke','batch','synthetic','sweep','bolt')][string]$Kind,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
# $IsWindows only exists in PowerShell 6+; Windows PowerShell 5.1 is always Windows.
$onWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)
if ($onWindows) {
  $bash = "C:\Program Files\Git\bin\bash.exe"
  if (-not (Test-Path $bash)) { throw "Git Bash not found at $bash" }
} else {
  $bash = "bash"
}
Set-Location (Split-Path $PSScriptRoot -Parent)
& $bash "scripts/run_ollama_$Kind.sh" @Rest
exit $LASTEXITCODE
