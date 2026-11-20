"""Collect completed runs of experiment 2 into one viewer 'group' (read-only on all sources).

  <group>/vanilla-no-LLM/sandbox_*     vanilla BO, from the new free-experiment folder
  <group>/agent-free/sandbox_*         agent allowed to reconfigure lenz, from the new folder
  <group>/agent-pinned/sandbox_*       experiment 1's pinned agent (surrogate/acqf fixed), copied
                                       from results/logs/ollama-batch (never modified)

Only runs whose run_meta.json says "completed" are copied, so a half-finished seed never enters a mean.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def collect(label: str, sources: list[Path], group: Path) -> int:
    n = 0
    for src in sources:
        if not src.is_dir():
            continue
        cond = group / label
        (cond / "_answers").mkdir(parents=True, exist_ok=True)
        for sb in src.glob("sandbox_*"):
            try:
                if json.loads((sb / "run_meta.json").read_text()).get("status") != "completed":
                    continue
            except (OSError, json.JSONDecodeError):
                continue
            shutil.copytree(sb, cond / sb.name, dirs_exist_ok=True)
            n += 1
        if (src / "_answers").is_dir():
            shutil.copytree(src / "_answers", cond / "_answers", dirs_exist_ok=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-root", default="results/logs/ollama-free")
    ap.add_argument("--pinned-root", default="results/logs/ollama-batch")
    ap.add_argument("--model-tag", default="qwen3.6_27b")
    ap.add_argument("--bench", default="hartmann6")
    ap.add_argument("--group", default="results/logs/analysis-free-hartmann6")
    a = ap.parse_args()
    free = Path(a.free_root) / a.model_tag / f"{a.bench}-blind"
    pinned = Path(a.pinned_root) / a.model_tag / f"{a.bench}-blind"
    group = Path(a.group)
    if group.exists():
        shutil.rmtree(group)
    seeds = sorted(free.glob("seed*"))
    agent_dirs = [d / name for d in seeds for name in ("sara-lenz-free", "sara-lenz-free-all")]
    plan = {
        "vanilla-no-LLM": [d / "vanilla" for d in seeds],
        "agent-free": agent_dirs,
        "agent-free-hint": [d / "sara-lenz-free-hint" for d in seeds],
        "agent-pinned": [d / "sara-lenz" for d in sorted(pinned.glob("seed*"))],
    }
    for label, srcs in plan.items():
        print(f"{label}: {collect(label, srcs, group)} runs copied")


if __name__ == "__main__":
    main()
