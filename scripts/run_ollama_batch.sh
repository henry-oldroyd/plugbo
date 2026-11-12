#!/usr/bin/env bash
# Many seeds x benchmarks with a local Ollama model. Finished legs are skipped, so it
# is safe to stop (Ctrl-C) and re-run to resume.
#   SEEDS="43 44 45" BENCHES="branin hartmann6" BUDGET=50 ./scripts/run_ollama_batch.sh
# Env: SEEDS (default 42 43 44 45 46)  BENCHES (default hartmann6)  BUDGET (default 50)
#      BACKENDS (default vanilla,sara-lenz)  DISCLOSURES (default blind)  OUT (results/logs/ollama-batch)
# Extra args are passed to run_synthetic.sh.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_ollama_env.sh
OUT="${OUT:-results/logs/ollama-batch}"
tag="${MODEL//[:\/]/_}"
fail=0
for seed in ${SEEDS:-42 43 44 45 46}; do
  for b in ${BENCHES:-hartmann6}; do
    for d in ${DISCLOSURES:-blind}; do
      echo "######## $b disclosure=$d seed=$seed ($(date +%H:%M:%S))"
      ./scripts/run_synthetic.sh "$b" --disclosure "$d" --seed "$seed" \
        --budget "${BUDGET:-50}" --backend "${BACKENDS:-vanilla,sara-lenz}" \
        --root "$OUT/$tag/${b}-${d}/seed${seed}" "$@" || { echo "!! failed: $b $d seed=$seed" >&2; fail=1; }
    done
  done
done
echo "batch done (failures: $fail). View: plugbo-viz --root $OUT"
exit $fail
