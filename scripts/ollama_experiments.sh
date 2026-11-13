#!/usr/bin/env bash
# Vanilla BO vs an LLM agent (sara-lenz) on a diverse set of synthetic benchmarks, over several seeds,
# with an ordered job queue worked through by the GPUs. Run the two commands in order:
#
#   ./scripts/ollama_experiments.sh parallel   # one Ollama server PER GPU, each running qwen3.6:27b; every
#                                              #   GPU is a separate worker taking the next job in the queue
#   ./scripts/ollama_experiments.sh shared     # ONE server using ALL GPUs for deepseek-r1:32b (int8);
#                                              #   the same queue, one job at a time
#   ./scripts/ollama_experiments.sh both       # parallel, then shared
#   ./scripts/ollama_experiments.sh plots      # (re)draw the multi-seed plots from whatever has finished
#
# THE QUEUE (identical in both modes). Two tiers, each ordered seed-major:
#   tier 1 "core":     seed 42 on every core benchmark, then seed 43 on every core benchmark, ... (5 seeds)
#   tier 2 "extended": the same pattern over the extended benchmarks
# so after one pass of tier 1 you have a result for every core benchmark, and each further pass adds a seed
# (more seeds = tighter uncertainty bands). Stop (Ctrl-C) whenever; re-running resumes where it left off.
# Each job is one (seed, benchmark, backend) leg. GPU workers take the LLM legs (sara-lenz) in queue order;
# vanilla BO makes no LLM calls so a CPU worker takes those legs, in the same order, and never occupies a GPU.
#
# Core benchmarks (diverse dimension / landscape):  branin (2-D, 3 optima), hartmann6 (6-D smooth multimodal),
#   ackley10 (10-D, flat plateau + needle), rosenbrock6 (narrow curved valley), rastrigin6 (6-D highly
#   multimodal grid), michalewicz5 (steep ridges and valleys).
# Extended: shekel (4-D narrow peaks), six_hump_camel (2-D, two global optima), levy10, styblinski_tang10,
#   griewank10, constrained_hartmann6.
#
# Results: results/logs/ollama-experiments/<model>/<bench>-blind/seed<N>/{vanilla,sara-lenz}
# Plots:   results/logs/ollama-experiments/<model>/seeds.html  (median + 25-75% band across seeds, one chart
#          per benchmark, refreshed every few minutes while running), and with `plots` an overlay of the
#          models at results/logs/ollama-experiments/compare-models.html.  View: plugbo-viz or open the html.
#
# Options (env var in brackets):
#   --seeds "42 43 44 45 46"   [SEEDS]     seeds per benchmark            --budget N  [BUDGET]  evaluations (100)
#   --benches "a b ..."        core set                                    --no-extended   skip tier 2
#   --extended-benches "..."   tier-2 set                                  --all-extended  tier 2 = every other
#   --model M     override the mode's model      --context N  Ollama context (32768)    synthetic benchmark
#   --gpus "0 1"  GPUs to use (default all)      --threads N  CPU threads per job (8)
#   --cpu-workers N   parallel workers for the vanilla legs (default 2; they only use CPU)
#   --label NAME      write to <model>_NAME instead of <model>, e.g. to keep a second arm (such as
#                     --no-enforce-budget) apart from the main one      --agent-only  skip the vanilla legs
#   --base-port N first Ollama port (11434)      --no-enforce-budget  let the agent end runs early
#   --text-tools / --no-text-tools   run a fenced ```bash block in a reply as a tool call (SARA_TEXT_TOOLS).
#                 Default: ON for `shared` (deepseek-r1 distills write commands as text instead of making
#                 structured tool calls, so without this the agent never acts), OFF for `parallel` (qwen).
#   --plan        print the queue and exit (starts nothing)
#
# This RESTARTS Ollama: it stops all of your own `ollama` processes, starts its own server(s), and stops them
# when it exits. It refuses to run if Ollama is a systemd service. Models go in /scratch/$USER/ollama-models
# when /scratch/$USER exists (home is small and networked).
#
# Time: an agent run of 100 evaluations takes roughly half an hour to over an hour on one GPU (the sara-lenz run
# on a 4090 took ~30 min for 32 evaluations), so tier 1 at 5 seeds is 30 agent runs: about a day per GPU-pair.
# Use --plan, then cut with --seeds / --benches / --budget as needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SELF="$(pwd)/scripts/ollama_experiments.sh"

die() { echo "ollama_experiments: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d'; }

MODE="${1:-}"
case "$MODE" in
  parallel|shared|both|plots) shift ;;
  _worker) shift; WK_KIND="$1"; WK_IDX="$2"; shift 2 ;;     # internal: spawned by parallel/shared
  -h|--help|"") usage; exit 0 ;;
  *) die "first argument must be parallel, shared, both or plots (see --help), got '$MODE'" ;;
esac

# ---------------------------------------------------------------- options
CORE_DEFAULT="branin hartmann6 ackley10 rosenbrock6 rastrigin6 michalewicz5"
EXT_DEFAULT="shekel six_hump_camel levy10 styblinski_tang10 griewank10 constrained_hartmann6"
SEEDS="${SEEDS:-42 43 44 45 46}"
BENCHES="${BENCHES:-}"
EXT_BENCHES=""
EXTENDED=1
ALL_EXTENDED=""
BUDGET="${BUDGET:-100}"
MODEL_OVERRIDE=""
CONTEXT=32768
GPUS=""
THREADS=8
CPU_WORKERS=2
LABEL="${LABEL:-}"
AGENT_ONLY=""
BASE_PORT=11434
ENFORCE=1
TEXT_TOOLS=""
PLAN=""
PASSTHRU=()
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds)    SEEDS="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --benches)  BENCHES="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --extended-benches) EXT_BENCHES="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --no-extended) EXTENDED=0; PASSTHRU+=("$1"); shift ;;
    --all-extended) ALL_EXTENDED=1; PASSTHRU+=("$1"); shift ;;
    --budget)   BUDGET="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --model)    MODEL_OVERRIDE="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --context)  CONTEXT="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --gpus)     GPUS="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --threads)  THREADS="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --cpu-workers) CPU_WORKERS="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --label)    LABEL="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --agent-only) AGENT_ONLY=1; PASSTHRU+=("$1"); shift ;;
    --base-port) BASE_PORT="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --no-enforce-budget) ENFORCE=0; PASSTHRU+=("$1"); shift ;;
    --text-tools)    TEXT_TOOLS=1; PASSTHRU+=("$1"); shift ;;
    --no-text-tools) TEXT_TOOLS=0; PASSTHRU+=("$1"); shift ;;
    --plan)     PLAN=1; PASSTHRU+=("$1"); shift ;;
    -h|--help)  usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

OUT="results/logs/ollama-experiments"

if [ "$MODE" = "both" ]; then
  rc=0
  "$SELF" parallel "${PASSTHRU[@]+"${PASSTHRU[@]}"}" || rc=$?
  echo; echo "######## parallel finished (exit $rc); starting shared ########"; echo
  "$SELF" shared "${PASSTHRU[@]+"${PASSTHRU[@]}"}" || rc=$?
  [ -n "$PLAN" ] || "$SELF" plots || true
  exit "$rc"
fi

if [ "$MODE" = "plots" ]; then
  python3 -c "import matplotlib" 2>/dev/null || die "matplotlib missing: python -m pip install -e '.[plots]'"
  roots=(); for d in "$OUT"/*/; do [ -d "$d" ] && roots+=(--root "${d%/}"); done
  [ "${#roots[@]}" -gt 0 ] || die "no results under $OUT yet"
  for ((i = 1; i < ${#roots[@]}; i += 2)); do python3 -m benchmarks.plot_seeds --root "${roots[$i]}"; done
  [ "${#roots[@]}" -le 2 ] || python3 -m benchmarks.plot_seeds "${roots[@]}" --out "$OUT/compare-models.html" \
    --title "Models compared: best regret vs evaluations (median and 25-75% over seeds)"
  exit 0
fi

# ---------------------------------------------------------------- environment shared by all modes
# (A _worker inherits all of this, already exported, from the parent; do not reset it from defaults.)
if [ "$MODE" != "_worker" ]; then
  [ -n "$TEXT_TOOLS" ] || TEXT_TOOLS="${SARA_TEXT_TOOLS:-}"   # an already-exported SARA_TEXT_TOOLS (e.g. "all") wins
  [ -n "$TEXT_TOOLS" ] || { [ "$MODE" = shared ] && TEXT_TOOLS=1 || TEXT_TOOLS=0; }
  export PROVIDER=ollama API_KEY=ollama EXTRA_BODY="" KERNEL_LLM_API_KEY_ENV="" SARA_ENFORCE_BUDGET="$ENFORCE" SARA_TEXT_TOOLS="$TEXT_TOOLS"
  export BASE_URL="http://127.0.0.1:${BASE_PORT}/v1" MODEL="${MODEL_OVERRIDE:-placeholder}"
fi
python3 -c "import lenz, sara, benchmarks, llm" 2>/dev/null || die "plugbo is not importable; activate the virtualenv and run: python -m pip install -e ."

if [ "$MODE" != "_worker" ]; then
  REGISTRY="$(python3 -c 'from benchmarks.functions import REGISTRY; print(" ".join(k for k in REGISTRY if k != "bolt_lora"))')" \
    || die "could not list benchmarks"
  [ -n "$BENCHES" ] || BENCHES="$CORE_DEFAULT"
  if [ -n "$ALL_EXTENDED" ]; then
    EXT_BENCHES=""; for b in $REGISTRY; do case " $BENCHES " in *" $b "*) ;; *) EXT_BENCHES+="$b " ;; esac; done
  fi
  [ -n "$EXT_BENCHES" ] || EXT_BENCHES="$EXT_DEFAULT"
  [ "$EXTENDED" = 1 ] || EXT_BENCHES=""
  for b in $BENCHES $EXT_BENCHES; do
    case " $REGISTRY " in *" $b "*) ;; *) die "unknown benchmark '$b'. Known: $REGISTRY" ;; esac
  done
fi

if [ -z "$GPUS" ]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ' || true)"
  [ -n "$GPUS" ] || die "no NVIDIA GPUs found (nvidia-smi)"
fi
read -r -a GPU_ARR <<< "$GPUS"

source scripts/_compare_env.sh   # provides condition_status

case "$MODE" in
  parallel) MODEL_TAG="${MODEL_OVERRIDE:-qwen3.6:27b}"; N_SERVERS=${#GPU_ARR[@]} ;;
  shared)   MODEL_TAG="${MODEL_OVERRIDE:-deepseek-r1:32b-qwen-distill-q8_0}"; N_SERVERS=1 ;;
  _worker)  MODEL_TAG="$MODEL"; N_SERVERS=0 ;;
esac
export MODEL="$MODEL_TAG"
TAG="${MODEL_TAG//[:\/]/_}${LABEL:+_$LABEL}"
BASE="$OUT/$TAG"

# ---------------------------------------------------------------- worker (internal)
# Claims jobs from the ordered queue by atomic mkdir, so several workers never take the same one.
if [ "$MODE" = "_worker" ]; then
  n=0; ndone=0; nfail=0
  while read -r tier seed bench backend; do
    n=$((n + 1))
    mkdir "$RUN_DIR/claims/$WK_KIND.$n" 2>/dev/null || continue     # another worker has this job
    root="$BASE/${bench}-blind/seed${seed}"
    st="$(condition_status "$root/$backend" "$seed")"
    if [ "$st" = completed ]; then echo "SKIP  [$tier] $bench $backend seed=$seed (completed)"; continue; fi
    force=(); [ "$st" = missing ] || force=(--force)               # stale "running"/"failed" from an old run
    echo "######## [$WK_KIND$WK_IDX] $tier: $bench $backend seed=$seed budget=$BUDGET status=$st ($(date +%H:%M:%S))"
    if ./scripts/run_synthetic.sh "$bench" --disclosure blind --backend "$backend" --seed "$seed" \
         --budget "$BUDGET" --root "$root" "${force[@]+"${force[@]}"}" < /dev/null; then
      ndone=$((ndone + 1))
    else
      nfail=$((nfail + 1)); echo "!! failed: $bench $backend seed=$seed" >&2
    fi
  done < "$RUN_DIR/jobs.$WK_KIND"
  echo "worker $WK_KIND$WK_IDX finished: ran=$ndone failed=$nfail"
  exit 0
fi

# ---------------------------------------------------------------- job queue + plan
RUN_DIR="results/logs/experiment-runs/${MODE}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR/claims" "$BASE"
: > "$RUN_DIR/jobs.llm"; : > "$RUN_DIR/jobs.cpu"
total=0; todo=0; PLAN_TXT=""
for tier in core extended; do
  if [ "$tier" = core ]; then tier_benches="$BENCHES"; else tier_benches="$EXT_BENCHES"; fi
  [ -n "$tier_benches" ] || continue
  t_total=0; t_todo=0
  for seed in $SEEDS; do                 # seed-major: every benchmark once for a seed, then the next seed
    for bench in $tier_benches; do
      for backend in vanilla sara-lenz; do
        [ "$backend" = vanilla ] && [ -n "$AGENT_ONLY" ] && continue
        st="$(condition_status "$BASE/${bench}-blind/seed${seed}/$backend" "$seed")"
        t_total=$((t_total + 1)); [ "$st" = completed ] || t_todo=$((t_todo + 1))
        if [ "$backend" = vanilla ]; then echo "$tier $seed $bench $backend" >> "$RUN_DIR/jobs.cpu"
        else echo "$tier $seed $bench $backend" >> "$RUN_DIR/jobs.llm"; fi
      done
    done
  done
  total=$((total + t_total)); todo=$((todo + t_todo))
  PLAN_TXT+="  tier $tier: $tier_benches"$'\n'"      x seeds {$SEEDS}: $t_total legs, $t_todo not yet completed"$'\n'
done
cat <<EOF
== mode: $MODE ==
model:      $MODEL_TAG   (context $CONTEXT, q8 KV cache, flash attention)
servers:    $N_SERVERS $([ "$MODE" = parallel ] && echo "(one per GPU: ${GPU_ARR[*]}; ports $BASE_PORT..$((BASE_PORT + N_SERVERS - 1)); each GPU is a worker)" || echo "(all GPUs: ${GPU_ARR[*]}; port $BASE_PORT; one worker)")
queue:      $(wc -l < "$RUN_DIR/jobs.llm") LLM legs (GPU workers) + $(wc -l < "$RUN_DIR/jobs.cpu") vanilla legs ($CPU_WORKERS CPU workers)
$PLAN_TXT  budget $BUDGET evaluations per leg, disclosure=blind, budget enforcement=$ENFORCE, text-tool fallback=${SARA_TEXT_TOOLS:-0}
results:    $BASE
run files:  $RUN_DIR
EOF
echo "first LLM jobs in the queue:"; head -8 "$RUN_DIR/jobs.llm" | sed 's/^/  /'
if [ -n "$PLAN" ]; then rm -rf "$RUN_DIR"; echo "(--plan: nothing started)"; exit 0; fi
[ "$todo" -gt 0 ] || { echo "Everything is already completed."; "$SELF" plots || true; exit 0; }

# ---------------------------------------------------------------- servers
command -v ollama >/dev/null 2>&1 || die "ollama not on PATH (user-local installs live in ~/.local/bin)"
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama 2>/dev/null; then
  die "Ollama runs as a systemd service; this script needs to start its own servers. 'sudo systemctl stop ollama' first."
fi
if [ -z "${OLLAMA_MODELS:-}" ] && [ -d "/scratch/${USER:-$(id -un)}" ]; then
  export OLLAMA_MODELS="/scratch/${USER:-$(id -un)}/ollama-models"; mkdir -p "$OLLAMA_MODELS"
fi
exec 9> "$OUT/.lock"; flock -n 9 || die "another ollama_experiments.sh is already running (lock: $OUT/.lock)"

PIDS=()
cleanup() {
  trap '' INT TERM
  echo; echo "stopping workers and Ollama servers ..."
  for p in "${PIDS[@]+"${PIDS[@]}"}"; do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true; done
  sleep 2
  for p in "${PIDS[@]+"${PIDS[@]}"}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo; echo "=== starting $N_SERVERS Ollama server(s) ($(date +%H:%M:%S)) ==="
pkill -x ollama 2>/dev/null || true; sleep 3
SERVER_PORTS=(); SERVER_DEVS=()
for ((i = 0; i < N_SERVERS; i++)); do
  port=$((BASE_PORT + i))
  if curl -sf --max-time 2 "http://127.0.0.1:$port/api/tags" > /dev/null; then
    die "port $port is already served by something else (another user's Ollama?); use --base-port"
  fi
  # parallel: this server sees only its own GPU; shared: it sees them all and splits the model across them
  if [ "$MODE" = parallel ]; then dev="${GPU_ARR[$i]}"; else dev="$(IFS=,; echo "${GPU_ARR[*]}")"; fi
  CUDA_VISIBLE_DEVICES="$dev" OLLAMA_HOST="127.0.0.1:$port" OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
    OLLAMA_CONTEXT_LENGTH="$CONTEXT" OLLAMA_KEEP_ALIVE=24h OLLAMA_NUM_PARALLEL=1 \
    setsid nohup ollama serve > "$RUN_DIR/ollama-$i.log" 2>&1 < /dev/null &
  PIDS+=("$!"); SERVER_PORTS+=("$port"); SERVER_DEVS+=("$dev")
  ready=""
  for _ in $(seq 1 120); do
    kill -0 "${PIDS[-1]}" 2>/dev/null || die "ollama serve (GPU $dev) exited; see $RUN_DIR/ollama-$i.log"
    curl -sf --max-time 2 "http://127.0.0.1:$port/api/tags" > /dev/null && { ready=1; break; }; sleep 1
  done
  [ -n "$ready" ] || die "server on port $port did not start; see $RUN_DIR/ollama-$i.log"
  echo "server $i: port $port, CUDA_VISIBLE_DEVICES=$dev"
done

echo "=== pulling $MODEL_TAG ==="
OLLAMA_HOST="127.0.0.1:${SERVER_PORTS[0]}" ollama pull "$MODEL_TAG" || die "ollama pull $MODEL_TAG failed"

echo "=== loading the model on each server (first load reads it from disk; can take minutes) ==="
for ((i = 0; i < N_SERVERS; i++)); do
  curl -sf --max-time 1800 "http://127.0.0.1:${SERVER_PORTS[$i]}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_TAG\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}" > /dev/null \
    || die "server $i (port ${SERVER_PORTS[$i]}) could not run $MODEL_TAG; see $RUN_DIR/ollama-$i.log"
  echo "--- server $i (GPU ${SERVER_DEVS[$i]}):"
  OLLAMA_HOST="127.0.0.1:${SERVER_PORTS[$i]}" ollama ps
  OLLAMA_HOST="127.0.0.1:${SERVER_PORTS[$i]}" ollama ps | grep -q "100% GPU" \
    || echo "WARNING: server $i is not 100% on GPU; it will be much slower. Lower --context or use a smaller --model."
done
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/GPU /'

# ---------------------------------------------------------------- workers
export RUN_DIR BASE BUDGET OUT LABEL
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
echo; echo "=== running $todo legs ($(date +%H:%M:%S)) ===   logs: tail -f $RUN_DIR/worker-*.log"
WORKER_PIDS=()
for ((i = 0; i < N_SERVERS; i++)); do
  # each LLM worker talks to its own server; in parallel mode its python also sees only its own GPU
  if [ "$MODE" = parallel ]; then gpu_env="${SERVER_DEVS[$i]}"; else gpu_env="${CUDA_VISIBLE_DEVICES:-}"; fi
  BASE_URL="http://127.0.0.1:${SERVER_PORTS[$i]}/v1" CUDA_VISIBLE_DEVICES="$gpu_env" \
    setsid "$SELF" _worker llm "$i" > "$RUN_DIR/worker-llm$i.log" 2>&1 < /dev/null &
  WORKER_PIDS+=("$!"); PIDS+=("$!")
done
for ((i = 0; i < CPU_WORKERS; i++)); do   # vanilla legs: CPU only, so several can run alongside the GPU workers
  CUDA_VISIBLE_DEVICES="" setsid "$SELF" _worker cpu "$i" > "$RUN_DIR/worker-cpu$i.log" 2>&1 < /dev/null &
  WORKER_PIDS+=("$!"); PIDS+=("$!")
done

refresh_plots() { python3 -m benchmarks.plot_seeds --root "$BASE" > /dev/null 2>&1 || true; }
alive() { for p in "${WORKER_PIDS[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
tick=0
while alive; do                          # redraw the multi-seed plots every ~5 minutes while jobs run
  sleep 10; tick=$((tick + 1))
  [ $((tick % 30)) -eq 0 ] && refresh_plots
done
refresh_plots

# ---------------------------------------------------------------- summary
echo; echo "=== finished ($(date +%H:%M:%S)) ==="
other=0; completed=0
for seed in $SEEDS; do for bench in $BENCHES $EXT_BENCHES; do for backend in vanilla sara-lenz; do
  [ "$backend" = vanilla ] && [ -n "$AGENT_ONLY" ] && continue
  st="$(condition_status "$BASE/${bench}-blind/seed${seed}/$backend" "$seed")"
  if [ "$st" = completed ]; then completed=$((completed + 1)); else other=$((other + 1)); echo "  not completed: $bench $backend seed=$seed ($st)"; fi
done; done; done
echo "completed legs: $completed / $total   (re-run the same command to retry the rest)"
echo "plots: $BASE/seeds.html   (all models: ./scripts/ollama_experiments.sh plots)   viewer: plugbo-viz --root $OUT"
[ "$other" -eq 0 ]
