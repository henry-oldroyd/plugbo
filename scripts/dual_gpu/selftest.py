#!/usr/bin/env python3
"""No-GPU self-test of the orchestrator (scheduling, single writer, failure handling, plotting).

Runs the real orchestrator with FAKE workers that copy finished runs from experiment 1
(results/logs/ollama-batch/...), so the figure is drawn from real data. Needs those results to exist
(or pass --fake-source). Takes about 30-60 seconds and writes to a temp directory.

  python scripts/dual_gpu/selftest.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> int:
    out = Path(tempfile.mkdtemp(prefix="dg_selftest_"))
    cmd = [sys.executable, str(HERE / "orchestrator.py"), "--fake", "--out", str(out), "--start-seed", "42",
           "--max-seeds", "6", "--agents", "3", "--bo-workers", "2", "--bo-lead", "4", "--heartbeat", "2",
           "--fake-agent-secs", "4", "--fake-bo-secs", "1", "--fake-fail-seeds", "44", "--model", "selftest-model"]
    print("running:", " ".join(cmd[:4]), "...\n")
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=300)
    print(r.stdout[-2500:])
    check(r.returncode == 0, f"orchestrator exited cleanly (rc={r.returncode}) {r.stderr[-300:]}")

    logs, figs = out / "logs", out / "figures"
    rows = list(csv.DictReader(open(logs / "results.csv", encoding="utf-8")))
    check(len(rows) == 12, f"12 results rows (6 seeds x BO+agent), got {len(rows)}")
    failed = [r for r in rows if r["status"] != "completed"]
    check(len(failed) == 1 and failed[0]["seed"] == "44" and failed[0]["condition"] == "agent",
          "exactly one failure: agent seed 44")
    fl = [json.loads(l) for l in open(logs / "failures.jsonl", encoding="utf-8")]
    check(len(fl) == 1, "failures.jsonl has one record")
    status = json.load(open(logs / "status.json", encoding="utf-8"))
    check(sorted(status["paired_seeds"]) == [42, 43, 45, 46, 47], f"paired seeds exclude 44: {status['paired_seeds']}")
    check(status["excluded_seeds"].get("44", status["excluded_seeds"].get(44)) is not None, "seed 44 recorded as excluded")
    check(status["failures"] == 1, "status.json failure count == 1")
    check((figs / "regret_dual_gpu.png").stat().st_size > 20_000, "figure exists")
    snaps = sorted((figs / "snapshots").glob("regret_n*.png"))
    check([s.name for s in snaps] == [f"regret_n{i:03d}.png" for i in range(1, 6)],
          f"one snapshot per paired seed (figure updated only on pairing): {[s.name for s in snaps]}")
    ev = (logs / "events.log").read_text(encoding="utf-8")
    check(ev.count("FIGURE updated") == 5, "figure updated exactly 5 times (once per paired seed)")
    check("EXCLUDED from the figure" in ev, "exclusion logged")
    check("HEARTBEAT" in ev, "heartbeat lines written throughout")
    check(list((figs).glob("*.tmp*")) == [] and list(logs.glob("*.tmp")) == [], "no stray temp files (atomic writes)")
    check(any((logs / "workers").glob("agent-seed42.log")), "per-worker logs written by the orchestrator")
    # single writer: worker run dirs contain only run artifacts, never logs/figures
    stray = [p for p in (out / "runs").rglob("*") if p.suffix in (".png", ".csv", ".jsonl") and "trace" not in p.name]
    check(not stray, "no log/figure files inside worker run directories")
    print(f"\nALL CHECKS PASSED  (outputs left in {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
