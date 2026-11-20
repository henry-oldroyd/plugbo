#!/usr/bin/env python3
"""Start a SEPARATE Ollama server (default port 11435) configured for layer-split, multi-agent use.

It never touches an existing Ollama on :11434, so your normal setup keeps working.

What the settings do
    OLLAMA_SCHED_SPREAD=1       spread the model's layers across ALL visible GPUs even if it would fit
                                on one (layer / pipeline split: GPU0 = first half of the layers and
                                their KV cache, GPU1 = second half; only per-token hidden states cross)
    OLLAMA_NUM_PARALLEL=N       N concurrent sequences ("slots") sharing ONE copy of the weights; each
                                slot gets its own KV cache of OLLAMA_CONTEXT_LENGTH tokens
    OLLAMA_CONTEXT_LENGTH=C     per-slot context window
    OLLAMA_FLASH_ATTENTION=1    required for a quantised KV cache
    OLLAMA_KV_CACHE_TYPE=q8_0   ~half the KV memory of f16
    OLLAMA_MAX_LOADED_MODELS=1  never load a second copy of the weights
    OLLAMA_KEEP_ALIVE=24h       keep the model resident between runs

  python scripts/dual_gpu/start_ollama.py --agents 4 --ctx 16384            # foreground server
  python scripts/dual_gpu/start_ollama.py --agents 4 --ctx 16384 --model qwen3:32b --check
      (--check: also start it, load the model, run one request and print per-GPU memory, then exit)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def server_env(a) -> dict:
    env = dict(os.environ)
    env.update(
        OLLAMA_HOST=f"127.0.0.1:{a.port}",
        OLLAMA_SCHED_SPREAD="1",
        OLLAMA_NUM_PARALLEL=str(a.agents),
        OLLAMA_CONTEXT_LENGTH=str(a.ctx),
        OLLAMA_FLASH_ATTENTION="1",
        OLLAMA_KV_CACHE_TYPE=a.kv_type,
        OLLAMA_MAX_LOADED_MODELS="1",
        OLLAMA_KEEP_ALIVE="24h",
    )
    if a.gpus:
        env["CUDA_VISIBLE_DEVICES"] = a.gpus
    return env


def gpu_mem() -> list[str]:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--agents", type=int, default=4, help="OLLAMA_NUM_PARALLEL (concurrent agent slots)")
    ap.add_argument("--ctx", type=int, default=16384, help="context per slot")
    ap.add_argument("--kv-type", default="q8_0", choices=["f16", "q8_0", "q4_0"])
    ap.add_argument("--gpus", default="0,1", help="CUDA_VISIBLE_DEVICES ('' = all)")
    ap.add_argument("--model", default=None, help="with --check: model to load and test")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    env = server_env(a)
    print("Starting `ollama serve` with:")
    for k in ("OLLAMA_HOST", "OLLAMA_SCHED_SPREAD", "OLLAMA_NUM_PARALLEL", "OLLAMA_CONTEXT_LENGTH",
              "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KEEP_ALIVE",
              "CUDA_VISIBLE_DEVICES"):
        print(f"  {k}={env.get(k, '')}")
    proc = subprocess.Popen(["ollama", "serve"], env=env)
    if not a.check:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return 130
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{base}/api/tags", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        else:
            print("server did not come up")
            return 1
        model = a.model or json.load(urllib.request.urlopen(f"{base}/api/tags"))["models"][0]["name"]
        print(f"\nloading {model} and sending one request ...")
        req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(
            {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "Say OK"}]}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=600).read()
        print("per-GPU memory (index, used, total) -- BOTH GPUs should hold a share of the model:")
        for line in gpu_mem():
            print("  ", line)
        ps = json.load(urllib.request.urlopen(f"{base}/api/ps"))["models"]
        for m in ps:
            print(f"  ollama ps: {m['name']} total {m['size']/2**30:.1f} GiB, in VRAM {m['size_vram']/2**30:.1f} GiB, "
                  f"context {m.get('context_length')}")
        return 0
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
