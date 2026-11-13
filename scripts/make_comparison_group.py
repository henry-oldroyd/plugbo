"""Collect per-seed runs into one viewer 'group' so the Experiments tab shows cross-seed bands.

The batch runner writes one folder per seed (seed42/{vanilla,sara-lenz}). plugbo-viz aggregates
across seeds only when the seeds are sandboxes under the SAME condition folder, so this copies
runs into  <out>/<condition>/sandbox_*  (plus the _answers key the scorer needs).

  python scripts/make_comparison_group.py            # defaults below
  plugbo-viz --root results/logs                     # then: Experiments tab -> analysis-hartmann6
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

LIVE = Path("results/logs/ollama-batch/qwen3.6_27b/hartmann6-blind")
BACKUP = Path("results/backup/ollama-batch-qwen3.6_27b-earlystop-20261113/qwen3.6_27b/hartmann6-blind")


def collect(label: str, sources: list[Path], out: Path) -> int:
    n = 0
    for src in sources:
        cond_out = out / label
        (cond_out / "_answers").mkdir(parents=True, exist_ok=True)
        for sb in src.glob("sandbox_*"):
            try:
                if json.loads((sb / "run_meta.json").read_text()).get("status") != "completed":
                    continue  # skip in-flight / crashed runs so partial traces never enter the mean
            except (OSError, json.JSONDecodeError):
                continue
            shutil.copytree(sb, cond_out / sb.name, dirs_exist_ok=True)
            n += 1
        if (src / "_answers").is_dir():
            shutil.copytree(src / "_answers", cond_out / "_answers", dirs_exist_ok=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/logs/analysis-hartmann6")
    ap.add_argument("--live", default=str(LIVE))
    ap.add_argument("--backup", default=str(BACKUP))
    a = ap.parse_args()
    out, live, backup = Path(a.out), Path(a.live), Path(a.backup)
    if out.exists():
        shutil.rmtree(out)
    plan = {
        "vanilla-no-LLM": [d / "vanilla" for d in sorted(live.glob("seed*"))],
        "agent-first-stop": [d / "sara-lenz" for d in sorted(backup.glob("seed*"))],
        "agent-forced-continue": [d / "sara-lenz" for d in sorted(live.glob("seed*"))],
    }
    for label, srcs in plan.items():
        print(f"{label}: {collect(label, [s for s in srcs if s.is_dir()], out)} runs copied")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
