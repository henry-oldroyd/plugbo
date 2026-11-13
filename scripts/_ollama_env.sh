# Sourced by run_ollama_*.sh: points the standard run scripts at a local Ollama model
# and runs preflight checks. Override with env vars:
#   OLLAMA_MODEL=qwen2.5-coder:32b  OLLAMA_BASE_URL=http://host:11434/v1
#   NEED_BOLT=1 (also require the BoLT extras)   SKIP_PREFLIGHT=1
# Default model: on a >=40 GiB-VRAM box (e.g. 2x RTX 3090) use the int8 32B model that fits in VRAM
# (see _gpu_profile.sh); otherwise keep the original single-GPU default.
source "$(dirname "${BASH_SOURCE[0]}")/_gpu_profile.sh"
_default_model="qwen3-coder:30b"
[ "$GPU_PROFILE" = "large" ] && _default_model="$GPU_PROFILE_MODEL"
export PROVIDER=ollama
export MODEL="${OLLAMA_MODEL:-$_default_model}"
export BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
export API_KEY="ollama"
export EXTRA_BODY="${EXTRA_BODY-}"
export KERNEL_LLM_API_KEY_ENV="${KERNEL_LLM_API_KEY_ENV-}"

_die() { echo "preflight error: $*" >&2; exit 1; }

# WSL bash cannot see Windows Python packages; require Git Bash on Windows.
if grep -qi microsoft /proc/version 2>/dev/null; then
  _die "running under WSL. Use Git Bash instead: & \"C:\Program Files\Git\bin\bash.exe\" $0 ..."
fi

# Windows: `python3` is often a Microsoft Store stub. Shim it to `python` if so.
if ! python3 -c "" >/dev/null 2>&1; then
  command -v python >/dev/null 2>&1 || _die "no working python/python3 on PATH"
  _shim="$(mktemp -d)"
  printf '#!/usr/bin/env bash\nexec python "$@"\n' > "$_shim/python3"
  chmod +x "$_shim/python3"
  export PATH="$_shim:$PATH"
fi

if [ -z "${SKIP_PREFLIGHT:-}" ]; then
  # 1. Python modules
  mods="torch gpytorch botorch numpy scipy anthropic openai dotenv"
  [ -n "${NEED_BOLT:-}" ] && mods="$mods huggingface_hub safetensors"
  missing="$(python3 - $mods <<'PY'
import importlib.util, sys
print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))
PY
)"
  [ -z "$missing" ] || _die "missing Python modules: $missing  -> run: python -m pip install -e '${NEED_BOLT:+.[bolt]}${NEED_BOLT:-.}'"
  python3 -c "import lenz, sara, benchmarks, llm" 2>/dev/null \
    || _die "plugbo packages not importable -> run: python -m pip install -e ."

  # 2. Ollama server reachable and model present
  _root="${BASE_URL%/v1}"
  tags="$(curl -sf "$_root/api/tags")" || _die "Ollama not reachable at $_root (start it with 'ollama serve')"
  echo "$tags" | grep -q "\"name\":\"$MODEL\"" \
    || _die "model '$MODEL' not pulled (try: ollama pull $MODEL). Available: $(echo "$tags" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')"

  # 3. Model actually answers (loads it, so the first real call is not a cold start)
  echo "preflight: loading $MODEL ..."
  reply="$(curl -sf --max-time 300 "$BASE_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}")" \
    || _die "model '$MODEL' did not answer a test request at $BASE_URL"
  echo "$reply" | grep -q '"choices"' || _die "unexpected reply from Ollama: $reply"
  echo "preflight: ok"
fi

# Free VRAM when the script ends (success, error or Ctrl-C).
_unload_model() { python3 -m llm.ollama_unload "$MODEL" "$BASE_URL" || true; }
trap _unload_model EXIT
