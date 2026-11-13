#!/usr/bin/env bash
# Linux/macOS equivalent of ollama_pipeline.ps1 (use that one on Windows).
#   configure + (re)start Ollama -> pull model -> viewer -> smoke test -> batch runs
#
#   ./scripts/ollama_pipeline.sh                              # model picked from the detected GPUs
#   ./scripts/ollama_pipeline.sh --seeds "43 44 45" --benches "hartmann6 branin" --budget 50
#   ./scripts/ollama_pipeline.sh --skip-smoke --model gpt-oss:20b
#   nohup ./scripts/ollama_pipeline.sh > results/logs/pipeline.log 2>&1 &     # run in background
#
# Options (each also settable via the env var in brackets):
#   --model M          [PIPE_MODEL]        default: auto from GPUs (see below)
#   --context N        [PIPE_CONTEXT]      default 32768
#   --seeds "a b c"    [SEEDS]             default "42 43 44 45 46"
#   --benches "x y"    [BENCHES]           default hartmann6
#   --disclosures "x"  [DISCLOSURES]       default blind
#   --backends a,b     [BACKENDS]          default vanilla,sara-lenz
#   --budget N         [BUDGET]            default 50
#   --skip-smoke, --no-viz
#   --smoke-only       restart Ollama, pull, run only the smoke test, then stop (no batch)
#
# Default model is chosen from the detected GPUs (scripts/_gpu_profile.sh):
#   >= 40 GiB total VRAM (e.g. 2x RTX 3090): deepseek-r1:32b-qwen-distill-q8_0, i.e. Qwen2.5-32B
#       (DeepSeek-R1 distill) at int8. At context 32768 with a q8 KV cache it loads as 39 GB, 100% GPU,
#       ~19 GB per card (Ollama splits layers over both GPUs itself). Spilling to CPU RAM is slow.
#   otherwise (e.g. one 24 GB RTX 4090): qwen3.6:27b.
# Force one with GPU_PROFILE_FORCE=large|small, or pick any model with --model.
#
# Model storage: if OLLAMA_MODELS is unset and /scratch/$USER exists, models go in
# /scratch/$USER/ollama-models (home on this lab's machines is a small, nearly full NFS share).
#
# NOTE: this restarts `ollama serve` (so the context / KV-cache settings take effect). It cannot
# do that for a systemd-managed Ollama; see the check below. On exit the model is unloaded from
# VRAM (done by _ollama_env.sh); `ollama serve` is left running.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source scripts/_gpu_profile.sh
MODEL_TAG="${PIPE_MODEL:-$GPU_PROFILE_MODEL}"
CONTEXT="${PIPE_CONTEXT:-$GPU_PROFILE_CONTEXT}"
SEEDS="${SEEDS:-42 43 44 45 46}"
BENCHES="${BENCHES:-hartmann6}"
DISCLOSURES="${DISCLOSURES:-blind}"
BACKENDS="${BACKENDS:-vanilla,sara-lenz}"
BUDGET="${BUDGET:-50}"
SKIP_SMOKE=""
NO_VIZ=""
SMOKE_ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model)       MODEL_TAG="$2"; shift 2 ;;
    --context)     CONTEXT="$2"; shift 2 ;;
    --seeds)       SEEDS="$2"; shift 2 ;;
    --benches)     BENCHES="$2"; shift 2 ;;
    --disclosures) DISCLOSURES="$2"; shift 2 ;;
    --backends)    BACKENDS="$2"; shift 2 ;;
    --budget)      BUDGET="$2"; shift 2 ;;
    --skip-smoke)  SKIP_SMOKE=1; shift ;;
    --no-viz)      NO_VIZ=1; shift ;;
    --smoke-only)  SMOKE_ONLY=1; shift ;;
    -h|--help)     sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

mkdir -p results/logs
echo "Detected: $GPU_PROFILE_DESC"
echo "Using:    model=$MODEL_TAG context=$CONTEXT"
step() { printf '\n=== %s  (%s) ===\n' "$1" "$(date +%H:%M:%S)"; }
die()  { echo "pipeline error: $*" >&2; exit 1; }

command -v ollama >/dev/null 2>&1 || die "ollama not on PATH (user-local installs live in ~/.local/bin)"

# --- 1. Configure + (re)start Ollama ------------------------------------------------------
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama 2>/dev/null; then
  die "Ollama is running as a systemd service, so this script cannot change its settings. Either
  'sudo systemctl edit ollama' and add Environment= lines for OLLAMA_CONTEXT_LENGTH=$CONTEXT,
  OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0 (then restart it and rerun with the same
  settings), or 'sudo systemctl stop ollama' and rerun this script."
fi

if [ -z "${OLLAMA_MODELS:-}" ] && [ -d "/scratch/${USER:-$(id -un)}" ]; then
  export OLLAMA_MODELS="/scratch/${USER:-$(id -un)}/ollama-models"
  mkdir -p "$OLLAMA_MODELS"
fi

step "Restarting Ollama (context=$CONTEXT, flash attention, q8 KV cache, models=${OLLAMA_MODELS:-default})"
export OLLAMA_CONTEXT_LENGTH="$CONTEXT" OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
pkill -x ollama 2>/dev/null || true       # -x: exact process name, only our own processes
sleep 2
setsid nohup ollama serve > results/logs/ollama-serve.log 2>&1 < /dev/null &
SERVE_PID=$!
ready=""
for _ in $(seq 1 120); do
  kill -0 "$SERVE_PID" 2>/dev/null || die "ollama serve exited immediately; see results/logs/ollama-serve.log (port 11434 already in use?)"
  if curl -sf --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null; then ready=1; break; fi
  sleep 1
done
[ -n "$ready" ] || die "Ollama did not respond on 127.0.0.1:11434 within 2 minutes"

# --- 2. Pull model ------------------------------------------------------------------------
step "Pulling $MODEL_TAG"
ollama pull "$MODEL_TAG" || die "ollama pull $MODEL_TAG failed (check the tag against the Ollama library)"
export OLLAMA_MODEL="$MODEL_TAG"

# --- 3. Viewer ----------------------------------------------------------------------------
if [ -z "$NO_VIZ" ]; then
  if command -v plugbo-viz >/dev/null 2>&1; then
    step "Starting viewer at http://127.0.0.1:8765"
    setsid nohup plugbo-viz --root results/logs > results/logs/viz.log 2>&1 < /dev/null &
  else
    echo "plugbo-viz not found (pip install -e .); skipping viewer"
  fi
fi

# --- 4. Smoke test ------------------------------------------------------------------------
if [ -z "$SKIP_SMOKE" ]; then
  step "Smoke test"
  ./scripts/run_ollama_smoke.sh || die "smoke test failed (exit $?); not starting the long run"
fi
if [ -n "$SMOKE_ONLY" ]; then
  step "Smoke test finished (--smoke-only); skipping batch"
  echo "Results: results/logs/ollama-smoke"
  exit 0
fi

# --- 5. Batch -----------------------------------------------------------------------------
step "Batch: seeds=$SEEDS benches=$BENCHES budget=$BUDGET backends=$BACKENDS"
set +e
SEEDS="$SEEDS" BENCHES="$BENCHES" DISCLOSURES="$DISCLOSURES" BACKENDS="$BACKENDS" BUDGET="$BUDGET" \
  ./scripts/run_ollama_batch.sh
code=$?
set -e

# --- 6. Done ------------------------------------------------------------------------------
step "Finished (batch exit code $code)"
echo "Results: results/logs/ollama-batch   Viewer: http://127.0.0.1:8765  (or: plugbo-viz --root results/logs)"
exit "$code"
