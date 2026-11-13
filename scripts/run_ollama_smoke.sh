#!/usr/bin/env bash
# Quick end-to-end check (a few minutes): tiny budget, 1 seed, 2 cheap benchmarks.
# Writes to results/logs/ollama-smoke (separate from real runs).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SEEDS="${SEEDS:-42}" BENCHES="${BENCHES:-branin hartmann6}" BUDGET="${BUDGET:-12}" \
  OUT=results/logs/ollama-smoke ./scripts/run_ollama_batch.sh --force "$@"
