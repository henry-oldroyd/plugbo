#!/usr/bin/env python3
"""How many agent slots (and BO workers) fit on a layer-split multi-GPU box?

Model of the memory (all in GiB):

    usable  = gpus * (vram_per_gpu - overhead_per_gpu - desktop_per_gpu)
    free    = usable - weights                       # what is left for KV caches
    kv_slot = kv_bytes_per_token * ctx               # one agent's KV cache
    slots   = floor(free / kv_slot)                  # "max"
    comfy   = floor((1 - headroom) * free / kv_slot) # "comfortable"

With a LAYER split (llama.cpp default, what Ollama does when a model spans GPUs) GPU 0 holds
the first half of the layers and GPU 1 the second half, *including the matching half of every
slot's KV cache*. Weights and KV therefore divide evenly, so the pooled formula above is also
the per-GPU formula (each GPU carries weights/g + slots*kv_slot/g + overhead). Only the hidden
state of each token crosses the GPU boundary (hidden_size * 2 bytes ~ 10 KiB for a 5120-wide
model), which is why PCIe without NVLink is enough.

KV bytes per token (standard attention, GQA):
    2 (K and V) * layers * kv_heads * head_dim * bytes_per_element
Hybrid / recurrent models (e.g. Qwen3.5/3.6 "qwen35") keep a KV cache for only some layers plus a
fixed per-slot state, so this formula over-estimates their KV (it is then an upper bound).

Usage:
    python scripts/dual_gpu/dg_capacity.py --model qwen3:32b          # reads Ollama + nvidia-smi
    python scripts/dual_gpu/dg_capacity.py --preset qwen3-32b-q4 --ctx 16384 --kv-type q8_0
    python scripts/dual_gpu/dg_capacity.py --preset qwen2.5-32b-q8 --json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.request

GIB = 1024**3
# bytes per KV element
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}

# (weights GiB, layers, kv_heads, head_dim, hidden). Approximate file sizes; prefer --model (reads Ollama).
PRESETS = {
    "qwen3-32b-q4": dict(weights_gib=18.8, layers=64, kv_heads=8, head_dim=128, hidden=5120),
    "qwen2.5-32b-q4": dict(weights_gib=18.5, layers=64, kv_heads=8, head_dim=128, hidden=5120),
    "qwen2.5-32b-q8": dict(weights_gib=32.5, layers=64, kv_heads=8, head_dim=128, hidden=5120),
    "qwen3-14b-q4": dict(weights_gib=9.0, layers=40, kv_heads=8, head_dim=128, hidden=5120),
}


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, kv_type: str) -> float:
    return 2 * layers * kv_heads * head_dim * KV_BYTES[kv_type]


def compute(
    weights_gib: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    ctx: int,
    kv_type: str = "q8_0",
    gpus: int = 2,
    vram_gib: float = 24.0,
    overhead_gib: float = 1.2,
    desktop_gib: float = 0.5,
    headroom: float = 0.15,
    safety: float = 0.03,
    hidden: int = 5120,
) -> dict:
    per_tok = kv_bytes_per_token(layers, kv_heads, head_dim, kv_type)
    kv_slot = per_tok * ctx / GIB
    usable = gpus * (vram_gib * (1 - safety) - overhead_gib - desktop_gib)
    free = usable - weights_gib
    slots_max = max(0, math.floor(free / kv_slot)) if free > 0 else 0
    slots_comfy = max(0, math.floor((1 - headroom) * free / kv_slot)) if free > 0 else 0
    n = slots_comfy
    per_gpu = (weights_gib + n * kv_slot) / gpus + overhead_gib + desktop_gib
    return dict(
        ctx=ctx,
        kv_type=kv_type,
        kv_kib_per_token=per_tok / 1024,
        kv_gib_per_slot=kv_slot,
        weights_gib=weights_gib,
        usable_gib=usable,
        free_for_kv_gib=free,
        slots_max=slots_max,
        slots_comfortable=slots_comfy,
        per_gpu_gib_at_comfortable=per_gpu,
        vram_gib_per_gpu=vram_gib,
        boundary_kib_per_token=hidden * 2 / 1024,
    )


def cpu_plan(
    cpus: int,
    agents: int,
    bo_threads: int = 2,
    agent_threads: int = 2,
    os_reserve: int = 2,
    t_bo_min: float = 10.0,
    t_agent_min: float = 30.0,
) -> dict:
    """BO workers that fit on the CPU, and how many are needed to stay ahead of the agents.

    Needed: BO throughput >= agent throughput  ->  bo_workers >= agents * t_bo / t_agent.
    Available: (cpus - os_reserve - agents*agent_threads) // bo_threads.
    t_bo / t_agent are *assumptions* (minutes per run) -- measure them and pass real values.
    """
    available = max(0, (cpus - os_reserve - agents * agent_threads) // max(1, bo_threads))
    needed = max(1, math.ceil(agents * t_bo_min / t_agent_min))
    return dict(cpus=cpus, bo_workers_available=available, bo_workers_needed_to_stay_ahead=needed,
                bo_workers_recommended=max(1, min(available, max(needed, 2))), bo_threads=bo_threads,
                agent_threads=agent_threads, t_bo_min=t_bo_min, t_agent_min=t_agent_min)


# ---------------------------------------------------------------- detection helpers
def ollama_model_info(model: str, base: str = "http://127.0.0.1:11434") -> dict | None:
    """Layers / kv heads / head dim from /api/show and weight size from /api/tags."""
    try:
        req = urllib.request.Request(f"{base}/api/show", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
        info = json.load(urllib.request.urlopen(req, timeout=10))["model_info"]
        arch = info["general.architecture"]
        layers = info[f"{arch}.block_count"]
        kv_heads = info[f"{arch}.attention.head_count_kv"]
        hidden = info[f"{arch}.embedding_length"]
        head_dim = info.get(f"{arch}.attention.key_length") or hidden // info[f"{arch}.attention.head_count"]
        tags = json.load(urllib.request.urlopen(f"{base}/api/tags", timeout=10))["models"]
        size = next(m["size"] for m in tags if m["name"] == model or m["model"] == model)
        return dict(weights_gib=size / GIB, layers=layers, kv_heads=kv_heads, head_dim=head_dim, hidden=hidden,
                    arch=arch)
    except Exception as exc:  # noqa: BLE001
        print(f"[capacity] could not read {model} from Ollama at {base}: {exc}", file=sys.stderr)
        return None


def detect_gpus() -> tuple[int, float] | None:
    """(count, VRAM GiB of the smallest GPU) from nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.split()
        mib = [float(x) for x in out]
        return (len(mib), min(mib) / 1024) if mib else None
    except Exception:  # noqa: BLE001
        return None


def resolve_model(args) -> dict:
    if args.preset:
        return dict(PRESETS[args.preset])
    if args.weights_gib:
        return dict(weights_gib=args.weights_gib, layers=args.layers, kv_heads=args.kv_heads,
                    head_dim=args.head_dim, hidden=5120)
    if args.model:
        info = ollama_model_info(args.model, args.ollama)
        if info:
            return info
    print("[capacity] falling back to preset qwen3-32b-q4", file=sys.stderr)
    return dict(PRESETS["qwen3-32b-q4"])


# ---------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="Ollama model tag; architecture + weight size are read from Ollama")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama base URL used for --model")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--weights-gib", type=float)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--gpus", type=int, help="default: detect with nvidia-smi, else 2")
    ap.add_argument("--vram-gib", type=float, help="per GPU; default: detect, else 24")
    ap.add_argument("--overhead-gib", type=float, default=1.2, help="CUDA context + compute buffers per GPU")
    ap.add_argument("--desktop-gib", type=float, default=0.5, help="VRAM used by other apps per GPU")
    ap.add_argument("--headroom", type=float, default=0.15, help="fraction kept free for 'comfortable'")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--kv-type", choices=sorted(KV_BYTES), default="q8_0")
    ap.add_argument("--budget", type=int, default=50, help="evaluations per run (for the context-need estimate)")
    ap.add_argument("--tokens-per-eval", type=int, default=1300, help="observed prompt growth per evaluation")
    ap.add_argument("--cpus", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--t-bo-min", type=float, default=10.0)
    ap.add_argument("--t-agent-min", type=float, default=30.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    det = detect_gpus()
    gpus = a.gpus or (det[0] if det else 2)
    vram = a.vram_gib or (det[1] if det else 24.0)
    m = resolve_model(a)
    base = dict(layers=m["layers"], kv_heads=m["kv_heads"], head_dim=m["head_dim"], weights_gib=m["weights_gib"],
                gpus=gpus, vram_gib=vram, overhead_gib=a.overhead_gib, desktop_gib=a.desktop_gib,
                headroom=a.headroom, hidden=m.get("hidden", 5120))
    chosen = compute(ctx=a.ctx, kv_type=a.kv_type, **base)
    cpu = cpu_plan(a.cpus, max(1, chosen["slots_comfortable"]), t_bo_min=a.t_bo_min, t_agent_min=a.t_agent_min)
    need_ctx = 4000 + a.tokens_per_eval * a.budget

    if a.json:
        print(json.dumps(dict(model=m, gpus=gpus, vram_gib=vram, chosen=chosen, cpu=cpu,
                              untrimmed_ctx_needed=need_ctx), indent=2))
        return

    print(f"GPUs: {gpus} x {vram:.1f} GiB   weights: {m['weights_gib']:.1f} GiB   "
          f"layers {m['layers']}, KV heads {m['kv_heads']}, head dim {m['head_dim']}")
    print(f"KV cache per token: f16 {kv_bytes_per_token(m['layers'], m['kv_heads'], m['head_dim'], 'f16')/1024:.0f} KiB, "
          f"q8_0 {kv_bytes_per_token(m['layers'], m['kv_heads'], m['head_dim'], 'q8_0')/1024:.0f} KiB")
    print(f"Layer split: each GPU holds ~{m['weights_gib']/gpus:.1f} GiB of weights + its half of every KV cache; "
          f"only {chosen['boundary_kib_per_token']:.0f} KiB per token crosses between GPUs.")
    print(f"Memory for KV caches after weights/overhead/desktop: {chosen['free_for_kv_gib']:.1f} GiB "
          f"(of {chosen['usable_gib']:.1f} GiB usable)\n")
    print("Concurrent agent slots (max / comfortable = keeps "
          f"{a.headroom:.0%} of the KV pool free):")
    ctxs = [8192, 16384, 24576, 32768, 49152, 65536]
    print(f"{'context':>9} | " + " | ".join(f"{t:>14}" for t in KV_BYTES))
    for c in ctxs:
        cells = []
        for t in KV_BYTES:
            r = compute(ctx=c, kv_type=t, **base)
            cells.append(f"{r['slots_max']:>3} / {r['slots_comfortable']:<3} ({r['kv_gib_per_slot']:.1f}G)")
        print(f"{c:>9} | " + " | ".join(f"{x:>14}" for x in cells))
    print(f"\nSelected: ctx {a.ctx}, KV {a.kv_type}: {chosen['kv_gib_per_slot']:.2f} GiB per agent -> "
          f"max {chosen['slots_max']}, comfortable {chosen['slots_comfortable']} concurrent agents; "
          f"~{chosen['per_gpu_gib_at_comfortable']:.1f} / {vram:.0f} GiB per GPU at the comfortable count.")
    print(f"An UNTRIMMED agent history needs about {need_ctx:,} tokens for {a.budget} evaluations "
          f"(~{a.tokens_per_eval} tokens/eval): "
          + ("the selected context is large enough." if a.ctx >= need_ctx else
             f"the selected context ({a.ctx:,}) is too small, so agents will hit the window before the budget "
             "unless the history is trimmed."))
    print(f"\nCPU ({cpu['cpus']} threads): agents use ~{cpu['agent_threads']} threads each, BO workers {cpu['bo_threads']}. "
          f"BO workers available: {cpu['bo_workers_available']}; needed to stay ahead of "
          f"{max(1, chosen['slots_comfortable'])} agents (assuming BO {cpu['t_bo_min']:.0f} min vs agent "
          f"{cpu['t_agent_min']:.0f} min per run): {cpu['bo_workers_needed_to_stay_ahead']}; "
          f"recommended: {cpu['bo_workers_recommended']}.")
    print("(Assumed run times are placeholders: measure them with a short run and pass --t-bo-min/--t-agent-min.)")


if __name__ == "__main__":
    main()
