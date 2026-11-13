#!/usr/bin/env bash
# Linux counterpart of ollama_seed_loop.ps1, aimed at ONE experiment on a 2-GPU box: keep running new seeds of
# vanilla BO vs the sara-lenz agent on one benchmark, with ONE model split across ALL GPUs (default
# deepseek-r1:32b int8), and redraw the cross-seed "mean +/- standard error" regret plot after EVERY finished seed.
# (For many benchmarks / one model per GPU see ollama_experiments.sh; this script is the single-experiment loop.)
#
#   ./scripts/ollama_seed_loop.sh                      # seeds 42, 43, 44, ... until Ctrl-C
#   ./scripts/ollama_seed_loop.sh --hours 12           # stop starting new seeds after 12 hours
#   ./scripts/ollama_seed_loop.sh --max-seeds 5 --start-seed 42
#   nohup ./scripts/ollama_seed_loop.sh > results/logs/seed-loop.log 2>&1 &      # run in the background
#
# Options:  --bench B (hartmann6)  --model M (deepseek-r1:32b-qwen-distill-q8_0)  --budget N (50)
#           --start-seed N (42, the same seeds as the qwen runs, so warm-up points match)  --max-seeds N (0 = no limit)
#           --hours H (0 = no limit)   --text-tools 1|all|0 (all)   -- anything after `--` goes to ollama_experiments.sh
# Output:   plot  results/figures/regret_seed_distribution_<model>.png  (+ the interactive .html next to it)
#           runs  results/logs/ollama-experiments/<model>/<bench>-blind/seed<N>/{vanilla,sara-lenz}
# Resume:   just run it again; finished seeds are skipped, and an interrupted seed is redone.
#
# Per seed it runs `ollama_experiments.sh shared`: one Ollama server on all GPUs (the model is reloaded for each
# seed, ~1-2 min, small next to the run), the agent leg on the GPUs and the vanilla leg on a CPU worker at the
# same time. deepseek-r1 distills do not emit structured tool calls, so SARA_TEXT_TOOLS (default `all`) makes
# the agent run the fenced shell commands in its replies and keeps its prose out of the history; see
# sara/agent.py and LOCAL_OLLAMA_NOTES.md. The agent may evaluate points it made up rather than the optimiser's
# suggestion, and it is slow: judge the plot with that in mind.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BENCH=hartmann6; MODEL=deepseek-r1:32b-qwen-distill-q8_0; BUDGET=50; START=42; MAXSEEDS=0; HOURS=0; TEXT=all
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --bench) BENCH="$2"; shift 2 ;;      --model) MODEL="$2"; shift 2 ;;     --budget) BUDGET="$2"; shift 2 ;;
    --start-seed) START="$2"; shift 2 ;; --max-seeds) MAXSEEDS="$2"; shift 2 ;;
    --hours) HOURS="$2"; shift 2 ;;      --text-tools) TEXT="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

TAG="${MODEL//[:\/]/_}"
ROOT="results/logs/ollama-experiments/$TAG"
FIG="results/figures/regret_seed_distribution_${TAG}"
mkdir -p results/figures
export SARA_TEXT_TOOLS="$TEXT"
deadline=0; [ "$HOURS" != 0 ] && deadline=$(python3 -c "import time; print(int(time.time() + $HOURS * 3600))")

plot() {
  python3 -m benchmarks.plot_seeds --root "$ROOT" --stat mean --out "$FIG.html" \
    --name "$TAG=Agent, forced to continue ($MODEL)" --name "vanilla=Vanilla BO (no LLM)" > /dev/null 2>&1 \
    && cp "$FIG/${BENCH}-blind.png" "$FIG.png" 2>/dev/null && echo "plot: $FIG.png" || echo "(plot not updated yet)"
}

done_seeds=0; seed="$START"
trap 'echo; echo "stopped (Ctrl-C). Seeds finished this session: $done_seeds. Plot: $FIG.png"; exit 130' INT TERM
while true; do
  [ "$MAXSEEDS" -gt 0 ] && [ "$done_seeds" -ge "$MAXSEEDS" ] && { echo "Reached --max-seeds $MAXSEEDS."; break; }
  [ "$deadline" -gt 0 ] && [ "$(date +%s)" -gt "$deadline" ] && { echo "Reached --hours $HOURS."; break; }
  echo; echo "######## seed $seed  ($(date '+%Y-%m-%d %H:%M:%S'))"
  ./scripts/ollama_experiments.sh shared --model "$MODEL" --benches "$BENCH" --no-extended --seeds "$seed" \
      --budget "$BUDGET" --cpu-workers 1 "${EXTRA[@]+"${EXTRA[@]}"}" \
    || echo "WARNING: seed $seed reported failures (see the run files above); continuing"
  plot
  done_seeds=$((done_seeds + 1)); seed=$((seed + 1))
done
echo "Seeds finished this session: $done_seeds. Plot: $FIG.png"
