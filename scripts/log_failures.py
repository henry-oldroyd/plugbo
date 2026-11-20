"""Record failed runs (run_meta status 'failed') of experiment 2 in <free-root>/failures.jsonl.

Called by the loop after every seed and before stale-run cleanup, so failure counts survive reruns.
Each failed sandbox is recorded once. Prints the totals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-root", default="results/logs/ollama-free")
    ap.add_argument("--model-tag", default="qwen3.6_27b")
    ap.add_argument("--bench", default="hartmann6")
    a = ap.parse_args()
    root = Path(a.free_root)
    log = root / "failures.jsonl"
    seen = set()
    if log.exists():
        seen = {json.loads(l)["sandbox"] for l in log.read_text(encoding="utf-8").splitlines() if l.strip()}
    base = root / a.model_tag / f"{a.bench}-blind"
    new = []
    for meta_path in base.glob("seed*/*/sandbox_*/run_meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("status") != "failed":
            continue
        sb = meta_path.parent
        if sb.name in seen:
            continue
        new.append({
            "sandbox": sb.name,
            "seed": meta.get("seed"),
            "condition": sb.parent.name,
            "error": (meta.get("error") or "")[:300],
            "ended_at": meta.get("ended_at"),
        })
    if new:
        root.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
    total = sum(1 for l in log.read_text(encoding="utf-8").splitlines() if l.strip()) if log.exists() else 0
    print(f"failures logged: {total} total ({len(new)} new)")


if __name__ == "__main__":
    main()
