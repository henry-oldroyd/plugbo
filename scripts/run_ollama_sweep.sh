#!/usr/bin/env bash
# Small sweep: several benchmarks x disclosure levels with a local Ollama model.
#   ./scripts/run_ollama_sweep.sh                # branin hartmann6, blind + revealed-shift
#   BENCHES="ackley" DISCLOSURES="blind" ./scripts/run_ollama_sweep.sh --budget 15
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for b in ${BENCHES:-branin hartmann6}; do
  for d in ${DISCLOSURES:-blind revealed-shift}; do
    ./scripts/run_ollama_synthetic.sh "$b" --disclosure "$d" \
      --root "./results/logs/ollama/${b}-${d}" "$@"
  done
done
