#!/usr/bin/env bash
# Synthetic-function experiments with a local Ollama Qwen model.
#   ./scripts/run_ollama_synthetic.sh [benchmark] [run_synthetic.sh flags...]
#   OLLAMA_MODEL=qwen2.5-coder:32b ./scripts/run_ollama_synthetic.sh branin
# Defaults: hartmann6, budget 20, sara-lenz backend only (cheap first run).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_ollama_env.sh
BENCH="${1:-hartmann6}"; [ $# -gt 0 ] && shift
args=("$@")
has() { for a in "${args[@]+"${args[@]}"}"; do [ "$a" = "$1" ] && return 0; done; return 1; }
has --budget  || args+=(--budget 20)
has --backend || args+=(--backend vanilla,sara-lenz)
has --root    || args+=(--root "./results/logs/ollama/${BENCH}-${MODEL//[:\/]/_}")
./scripts/run_synthetic.sh "$BENCH" "${args[@]}"
