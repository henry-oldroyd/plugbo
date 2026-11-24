"""Tool-use statistics from agent traces (read-only). Used by make_tooluse_report.py."""
from __future__ import annotations
import glob, json, re, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPS = {
    "Exp 1: optimizer pinned": str(ROOT / "results/logs/ollama-batch/**/sara-lenz/*/trace.jsonl"),
    "Exp 2: free + vague hint": str(ROOT / "results/logs/ollama-free/**/sara-lenz-free-hint/*/trace.jsonl"),
}
CATS = ["suggest", "submit", "oracle", "reconfigure (set-*)", "inspect", "other"]
SET_VERBS = ["set-acqf", "set-surrogate", "set-bounds", "set-objectives", "set-constraints", "set-region", "set-sampler", "set-belief"]
WARMUP = 7


def classify(cmd: str) -> collections.Counter:
    c = collections.Counter()
    c["suggest"] += len(re.findall(r"lenz suggest", cmd))
    c["submit"] += len(re.findall(r"lenz submit", cmd))
    c["oracle"] += len(re.findall(r"\./oracle", cmd))
    c["reconfigure (set-*)"] += len(re.findall(r"lenz set-", cmd))
    c["inspect"] += len(re.findall(r"lenz (?:status|trials|show|best|diagnos\w*|posterior|fit|cv|history|summary)", cmd))
    if not c:
        c["other"] += 1
    return c


def load_run(path: str) -> dict:
    ev = WARMUP
    per_eval = collections.defaultdict(collections.Counter)  # eval index at call time -> category counts
    tot = collections.Counter(); verbs = collections.Counter(); qs = collections.Counter(); forced = 0
    others = collections.Counter()
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d.get("forced_continuation"):
            forced += 1; per_eval[ev]["forced reprompt"] += 1; continue
        if d["role"] != "tool":
            continue
        a = d["arguments"]; cmd = a.get("cmd", "") if isinstance(a, dict) else ""
        c = classify(cmd)
        for k, v in c.items():
            tot[k] += v; per_eval[ev][k] += v
        for v in re.findall(r"lenz (set-[a-z]+)", cmd): verbs[v] += 1
        for s in re.findall(r"lenz suggest[^\n;|&]*", cmd):
            m = re.search(r"--q\s+(\d+)", s); qs[int(m.group(1)) if m else 1] += 1
        if c.get("other"): others[cmd.split()[0] if cmd.split() else "?"] += 1
        ev += c.get("oracle", 0)
    tot["forced reprompt"] = forced
    n_state = None
    try:
        n_state = len(json.load(open(path.replace("trace.jsonl", "state.json"), encoding="utf-8")).get("trials", []))
    except Exception:  # noqa: BLE001
        pass
    return dict(tot=tot, per_eval=per_eval, verbs=verbs, qs=qs, others=others, evals=n_state if n_state else min(ev, 50))


def load_all() -> dict:
    out = {}
    for name, pat in EXPS.items():
        out[name] = [load_run(p) for p in sorted(glob.glob(pat, recursive=True))]
    return out

if __name__ == "__main__":
    for k, runs in load_all().items():
        t = collections.Counter(); o = collections.Counter()
        for r in runs: t.update(r["tot"]); o.update(r["others"])
        print(k, len(runs), dict(t)); print("  other first-words:", o.most_common(8))
