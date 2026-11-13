# Dot-sourced by ollama_pipeline.ps1 and ollama_agent_rerun.ps1: picks a default local model from the GPUs.
# PowerShell mirror of _gpu_profile.sh -- keep the two in sync.
#   large : >= 40 GiB total VRAM (e.g. 2x RTX 3090 = 48 GB) -> deepseek-r1:32b-qwen-distill-q8_0
#           (Qwen2.5-32B DeepSeek-R1 distill at int8; 39 GB at context 32768, 100% on GPU, ~19 GB per card)
#   small : everything else (e.g. one 24 GB RTX 4090, no NVIDIA GPU) -> qwen3.6:27b
# Usage:  . "$PSScriptRoot\_gpu_profile.ps1"; $p = Get-GpuProfile [-Force large|small]
function Get-GpuProfile {
  param([string]$Force = "")
  $totalMiB = 0; $count = 0
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
      foreach ($line in (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null)) {
        $t = "$line".Trim()
        if ($t -match '^\d+$') { $totalMiB += [int]$t; $count++ }
      }
    } catch { }
  }
  $tier = if ($totalMiB -ge 40960) { "large" } else { "small" }
  if ($Force) { $tier = $Force }
  $model = if ($tier -eq "large") { "deepseek-r1:32b-qwen-distill-q8_0" } else { "qwen3.6:27b" }
  [pscustomobject]@{
    Tier     = $tier
    Model    = $model
    Context  = 32768
    Gpus     = $count
    TotalGiB = [math]::Floor($totalMiB / 1024)
  }
}
