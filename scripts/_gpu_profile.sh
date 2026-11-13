# Sourced by _ollama_env.sh and ollama_pipeline.sh: picks a default local model from the GPUs present.
# Exports GPU_PROFILE (large|small), GPU_PROFILE_MODEL, GPU_PROFILE_CONTEXT, GPU_PROFILE_DESC.
#
#   large : >= 40 GiB total VRAM (e.g. 2x RTX 3090 = 48 GB)  -> Qwen2.5-32B (DeepSeek-R1 distill), int8.
#           Weights ~34 GB; at context 32768 with a q8 KV cache it loads as 39 GB, 100% on GPU,
#           split ~19 GB per card by Ollama. Anything that spills to CPU RAM is several times slower.
#   small : everything else (e.g. one 24 GB RTX 4090, no NVIDIA GPU, macOS) -> the earlier default.
#
# Overrides (any of): OLLAMA_MODEL / --model (pick the model directly), GPU_PROFILE_FORCE=large|small,
# GPU_LARGE_MIN_MIB (threshold in MiB, default 40960).
# The Windows equivalent of this logic is at the top of ollama_pipeline.ps1.

_gpu_total_mib=0
_gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
  while read -r _m; do
    case "$_m" in ''|*[!0-9]*) continue ;; esac
    _gpu_total_mib=$((_gpu_total_mib + _m))
    _gpu_count=$((_gpu_count + 1))
  done < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)
fi

GPU_PROFILE="small"
[ "$_gpu_total_mib" -ge "${GPU_LARGE_MIN_MIB:-40960}" ] && GPU_PROFILE="large"
[ -n "${GPU_PROFILE_FORCE:-}" ] && GPU_PROFILE="$GPU_PROFILE_FORCE"

case "$GPU_PROFILE" in
  large) GPU_PROFILE_MODEL="deepseek-r1:32b-qwen-distill-q8_0"; GPU_PROFILE_CONTEXT=32768 ;;
  *)     GPU_PROFILE="small"; GPU_PROFILE_MODEL="qwen3.6:27b";  GPU_PROFILE_CONTEXT=32768 ;;
esac
GPU_PROFILE_DESC="${_gpu_count} GPU(s), $((_gpu_total_mib / 1024)) GiB total VRAM -> profile '$GPU_PROFILE' ($GPU_PROFILE_MODEL)"
export GPU_PROFILE GPU_PROFILE_MODEL GPU_PROFILE_CONTEXT GPU_PROFILE_DESC
unset _gpu_total_mib _gpu_count _m
