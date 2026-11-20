#!/usr/bin/env python3
"""TEST STUB standing in for a real worker (no GPU, no LLM, no BO fitting).

Copies an existing finished run (e.g. from experiment 1) into --root after a delay, so the
orchestrator's scheduling, single-writer logging, failure handling and plotting can be exercised
with real data in seconds. It never touches anything outside --root and never knows where the
orchestrator keeps its logs or figures.

  python dg_worker_fake.py --kind agent --seed 42 --root R --source SRC --secs 3 [--fail]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["bo", "agent"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--source", required=True, help="dir containing seed<N>/{vanilla,sara-lenz}")
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--fail", action="store_true")
    a = ap.parse_args()

    cond = "vanilla" if a.kind == "bo" else "sara-lenz"
    src = Path(a.source) / f"seed{a.seed}" / cond
    boxes = sorted(src.glob("sandbox_*"), key=lambda p: p.stat().st_mtime)
    boxes = [b for b in boxes if json.loads((b / "run_meta.json").read_text()).get("status") == "completed"]
    if not boxes:
        print(f"[fake {a.kind}] no completed source run for seed {a.seed} in {src}", flush=True)
        return 2
    print(f"[fake {a.kind}] seed {a.seed}: working for {a.secs}s", flush=True)
    time.sleep(a.secs)

    root = Path(a.root)
    (root / "_answers").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "_answers", root / "_answers", dirs_exist_ok=True)
    dest = root / boxes[-1].name
    shutil.copytree(boxes[-1], dest, dirs_exist_ok=True)
    if a.fail:  # simulate a crash mid-run: mark the copy as failed, truncate nothing else
        meta_path = dest / "run_meta.json"
        meta = json.loads(meta_path.read_text())
        meta.update(status="failed", error="simulated crash (selftest)")
        meta_path.write_text(json.dumps(meta))
        print(f"[fake {a.kind}] seed {a.seed}: simulated failure", flush=True)
        return 1
    print(f"[fake {a.kind}] seed {a.seed}: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
