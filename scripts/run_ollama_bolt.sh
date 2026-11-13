#!/usr/bin/env bash
# BoLT LoRA HPO with a local Ollama Qwen model (needs: pip install -e '.[bolt]').
#   ./scripts/run_ollama_bolt.sh [run_bolt.sh flags...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NEED_BOLT=1 source scripts/_ollama_env.sh
args=("$@")
has() { for a in "${args[@]+"${args[@]}"}"; do [ "$a" = "$1" ] && return 0; done; return 1; }
has --budget  || args+=(--budget 20)
has --backend || args+=(--backend vanilla,sara-lenz)
./scripts/run_bolt.sh "${args[@]}"
