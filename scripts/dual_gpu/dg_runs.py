"""Read-only helpers for finished (or running) runs: status, regret trace, first forced continuation.

Used only by the single writer process (the orchestrator) AFTER a worker has exited, or for
read-only progress peeks. Nothing here writes to a run directory.

Run directory layout (same as the rest of PlugBO):
    <root>/_answers/<token>.json      hidden answer key (needed to compute true regret)
    <root>/sandbox_<token>/           state.json, trace.jsonl (agents), run_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.plot_compare import regret_trace_for_state  # reuse the repo's scoring, unmodified


def find_sandbox(root: Path) -> Path | None:
    """The newest sandbox_* under a run root (there is normally exactly one)."""
    boxes = [p for p in root.glob("sandbox_*") if p.is_dir()]
    return max(boxes, key=lambda p: p.stat().st_mtime) if boxes else None


def read_meta(sb: Path) -> dict:
    try:
        return json.loads((sb / "run_meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def observed_times(sb: Path) -> list[float]:
    try:
        st = json.loads((sb / "state.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return sorted(t["observed_at"] for t in st.get("trials", []) if t.get("status") == "observed" and t.get("observed_at"))


def n_observed(sb: Path) -> int:
    return len(observed_times(sb))


def first_forced(sb: Path) -> tuple[int | None, int]:
    """(evaluations observed when the FIRST forced continuation happened, total forced continuations).

    The agent harness writes {"forced_continuation": k, "ts": ...} into trace.jsonl whenever it
    tells the model to keep going. Its timestamps and the trials' `observed_at` share one clock, so
    the evaluation index is the number of trials observed no later than that timestamp.
    Returns (None, 0) when the run never needed a forced continuation (or is not an agent run).
    """
    trace = sb / "trace.jsonl"
    if not trace.exists():
        return None, 0
    first_ts, count = None, 0
    for line in trace.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("forced_continuation"):
            count += 1
            if first_ts is None:
                first_ts = e["ts"]
    if first_ts is None:
        return None, 0
    return sum(1 for t in observed_times(sb) if t <= first_ts), count


def summarize(root: Path) -> dict:
    """Everything the orchestrator needs to know about one finished run."""
    sb = find_sandbox(root)
    if sb is None:
        return {"status": "missing", "root": str(root)}
    meta = read_meta(sb)
    status = meta.get("status") or "unknown"
    out = {"status": status, "root": str(root), "sandbox": sb.name, "seed": meta.get("seed"),
           "started_at": meta.get("started_at"), "ended_at": meta.get("ended_at"),
           "error": (meta.get("error") or "")[:300], "n_evals": n_observed(sb)}
    if status == "completed":
        trace = regret_trace_for_state(sb / "state.json", root)
        if trace:
            out["trace"] = trace
            out["best_regret"] = trace[-1]
        idx, count = first_forced(sb)
        out["first_forced_idx"], out["forced_count"] = idx, count
    return out
