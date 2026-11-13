#!/usr/bin/env bash
# Linux/macOS equivalent of ollama_agent_rerun.ps1: re-run only the LLM-agent (sara-lenz) experiments,
# forcing the agent to use the whole evaluation budget. Finished runs are skipped, so rerunning resumes.
#
#   ./scripts/ollama_agent_rerun.sh
#   ./scripts/ollama_agent_rerun.sh --model gpt-oss:20b --seeds "42 43" --budget 30
#   ./scripts/ollama_agent_rerun.sh --force            # redo runs that already completed
#   ./scripts/ollama_agent_rerun.sh --no-enforce       # old behaviour: agent may end the run early
#
# Options: --model M  --seeds "a b"  --benches "x y"  --disclosures "x"  --backends a,b  --budget N
#          --no-enforce  --force  -h|--help
# Defaults: model from the detected GPUs (scripts/_gpu_profile.sh: 2x RTX 3090 ->
# deepseek-r1:32b-qwen-distill-q8_0, one 4090 -> qwen3.6:27b), seeds 42-46, hartmann6, blind,
# backend sara-lenz, budget 50.
#
# Assumes `ollama serve` is already running with the settings you want (context, KV cache, models dir).
# ollama_pipeline.sh does that for you; here, for the 2x3090 box, start it as e.g.
#   OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=32768 ollama serve
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_gpu_profile.sh

MODEL_TAG="${OLLAMA_MODEL:-$GPU_PROFILE_MODEL}"
SEEDS="${SEEDS:-42 43 44 45 46}"
BENCHES="${BENCHES:-hartmann6}"
DISCLOSURES="${DISCLOSURES:-blind}"
BACKENDS="${BACKENDS:-sara-lenz}"
BUDGET="${BUDGET:-50}"
ENFORCE=1
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --model)       MODEL_TAG="$2"; shift 2 ;;
    --seeds)       SEEDS="$2"; shift 2 ;;
    --benches)     BENCHES="$2"; shift 2 ;;
    --disclosures) DISCLOSURES="$2"; shift 2 ;;
    --backends)    BACKENDS="$2"; shift 2 ;;
    --budget)      BUDGET="$2"; shift 2 ;;
    --no-enforce)  ENFORCE=0; shift ;;
    --force)       EXTRA+=(--force); shift ;;
    -h|--help)     sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

echo "Detected: $GPU_PROFILE_DESC"
echo "model=$MODEL_TAG seeds=$SEEDS benches=$BENCHES budget=$BUDGET backends=$BACKENDS enforce_budget=$ENFORCE"
export OLLAMA_MODEL="$MODEL_TAG" SEEDS BENCHES DISCLOSURES BACKENDS BUDGET
export SARA_ENFORCE_BUDGET="$ENFORCE"
exec ./scripts/run_ollama_batch.sh "${EXTRA[@]+"${EXTRA[@]}"}"
