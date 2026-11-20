#!/usr/bin/env python3
"""Experiment 3 orchestrator: many seeds of vanilla BO and of an LLM agent, concurrently.

Architecture (no locks, one writer)
-----------------------------------
                       +-------------------------------+
   BO workers  (CPU) --|  stdout pipes                 |
   (subprocesses)      |   ->  supervisor coroutines   |--> asyncio.Queue --> WRITER coroutine
   Agent workers (GPU)-|   (read lines, wait for exit, |                     (the ONLY code that
   (subprocesses)      |    summarise the finished run)|                      writes logs / csv /
                       +-------------------------------+                      status.json / figure)

* A worker (a `python -m benchmarks.run_blind_*` subprocess) writes ONLY inside its own run
  directory `<out>/runs/seed<N>/<condition>/`, which no other process touches.
* Everything shared -- events.log, per-worker logs, results.csv, failures.jsonl, status.json and the
  figure -- is written by the writer coroutine of this one process. Workers "send their results" by
  exiting; a supervisor reads the finished run directory (read-only) and puts a message on the queue.
  Because the loop is single-threaded and there is a single writer, no file lock is ever needed.
* The figure is re-rendered only when a seed is PAIRED: its BO run and its agent run have both
  completed. If either failed, the seed is excluded from the figure (so every line uses the same
  seeds) and the failure is logged and counted.
* BO workers run ahead of the agents (they are cheaper): a BO worker may start seed s only while
  s < next_agent_seed + --bo-lead.

Typical use on the 2-GPU box (see experiments/03-dual-gpu-parallel.md):
    python scripts/dual_gpu/start_ollama.py --agents 4 --ctx 16384 &      # layer-split server on :11435
    python scripts/dual_gpu/orchestrator.py --model qwen3:32b --agents 4 --bo-workers 3 --hours 12

No-GPU smoke test of all the scheduling / logging / plotting logic:
    python scripts/dual_gpu/selftest.py
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import dg_capacity  # noqa: E402
import dg_plot  # noqa: E402
import dg_runs  # noqa: E402


# ------------------------------------------------------------------ helpers
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def kill_tree(pid: int) -> None:
    """Kill a worker and everything it spawned (agents start lenz and ./oracle subprocesses)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=20)
        else:
            os.killpg(pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


def gpu_snapshot() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        rows = []
        for line in out:
            i, util, used, total = [x.strip() for x in line.split(",")]
            rows.append(dict(gpu=int(i), util_pct=int(util), mem_used_mib=int(used), mem_total_mib=int(total)))
        return rows
    except Exception:  # noqa: BLE001
        return []


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Job:
    kind: str  # "bo" | "agent"
    seed: int
    root: Path
    cmd: list[str]
    env: dict
    key: str = ""
    started: float = 0.0
    pid: int | None = None
    task: asyncio.Task | None = field(default=None, repr=False)

    def __post_init__(self):
        self.key = f"{self.kind}-seed{self.seed}"


# ------------------------------------------------------------------ orchestrator
class Orchestrator:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(a.out).resolve()
        self.runs_dir = self.out / "runs"
        self.logs_dir = self.out / "logs"
        self.fig_path = Path(a.fig).resolve() if a.fig else self.out / "figures" / "regret_dual_gpu.png"
        self.snap_dir = self.fig_path.parent / "snapshots"
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stop = asyncio.Event()
        self.running: dict[str, Job] = {}
        # in-memory results (mutated only by the writer coroutine)
        self.done: dict[str, dict[int, dict]] = {"bo": {}, "agent": {}}  # seed -> summary
        self.paired: dict[int, dict] = {}
        self.excluded: dict[int, str] = {}
        self.failures = 0
        self.agent_ptr = a.start_seed  # next seed an agent will start
        self.bo_ptr = a.start_seed
        self.t0 = time.time()
        self.deadline = self.t0 + a.hours * 3600 if a.hours else None
        self.last_stats: dict = {}
        self._handles: dict[str, object] = {}
        self.events_fh = None
        self.shim_dir: str | None = None

    # ---------------------------------------------------------- command construction
    def worker_env(self, kind: str) -> dict:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        threads = str(self.a.bo_threads if kind == "bo" else self.a.agent_threads)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = threads
        if kind == "agent":
            env.update(SARA_ENFORCE_BUDGET="1", SARA_BASH_TIMEOUT=str(self.a.bash_timeout),
                       OLLAMA_BASE_URL=self.a.base_url)
        if os.name == "nt" and self.shim_dir:  # `python3` is a Store stub on Windows: shim it for ./oracle
            env["PATH"] = self.shim_dir + os.pathsep + env.get("PATH", "")
        return env

    def make_job(self, kind: str, seed: int) -> Job:
        cond = "vanilla" if kind == "bo" else "agent"
        root = self.runs_dir / f"seed{seed}" / cond
        py = self.a.python
        if self.a.fake:
            cmd = [py, str(HERE / "dg_worker_fake.py"), "--kind", kind, "--seed", str(seed), "--root", str(root),
                   "--source", self.a.fake_source,
                   "--secs", str(self.a.fake_bo_secs if kind == "bo" else self.a.fake_agent_secs)]
            if kind == "agent" and seed in self.a.fake_fail:
                cmd.append("--fail")
        elif kind == "bo":
            cmd = [py, "-m", "benchmarks.run_blind_baseline", "--benchmark", self.a.bench, "--budget",
                   str(self.a.budget), "--seed", str(seed), "--root", str(root), "--policy", "vanilla"]
        else:
            cmd = [py, "-m", "benchmarks.run_blind_test", "--benchmark", self.a.bench, "--provider", "ollama",
                   "--model", self.a.model, "--base-url", self.a.base_url, "--api-key", "ollama",
                   "--budget", str(self.a.budget), "--seed", str(seed), "--root", str(root),
                   "--acqf", "noisy_logei", "--surrogate", "fixed", "--freedom", self.a.freedom]
            if self.a.hint != "none":
                cmd += ["--hint", self.a.hint]
            if self.a.extra_body:
                cmd += ["--extra-body", self.a.extra_body]
        return Job(kind=kind, seed=seed, root=root, cmd=cmd, env=self.worker_env(kind))

    # ---------------------------------------------------------- supervisor (one per worker)
    async def supervise(self, job: Job) -> None:
        job.root.mkdir(parents=True, exist_ok=True)
        job.started = time.time()
        kwargs = dict(stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=job.env,
                      cwd=str(REPO), limit=1 << 24)
        if os.name != "nt":
            kwargs["start_new_session"] = True  # own process group so kill_tree works
        try:
            proc = await asyncio.create_subprocess_exec(*job.cmd, **kwargs)
        except Exception as exc:  # noqa: BLE001
            await self.queue.put(("finished", job, {"status": "failed", "error": f"spawn failed: {exc}"}))
            return
        job.pid = proc.pid
        self.running[job.key] = job
        await self.queue.put(("started", job, None))
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                await self.queue.put(("line", job, raw.decode("utf-8", "replace").rstrip()))
        except Exception as exc:  # noqa: BLE001 - never let a log-reading problem orphan a worker
            await self.queue.put(("line", job, f"[orchestrator] stdout reader error: {exc}"))
        rc = await proc.wait()
        # read the finished run directory (read-only) and hand the summary to the writer
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(None, dg_runs.summarize, job.root)
        summary["exit_code"] = rc
        if summary.get("status") == "completed" and rc != 0:
            summary["status"] = "failed"
            summary["error"] = f"worker exited with code {rc}"
        if summary.get("status") == "completed" and not summary.get("trace"):
            summary["status"] = "failed"
            summary["error"] = "run completed but no regret trace could be computed (missing answer key?)"
        if summary.get("status") not in ("completed",):
            summary.setdefault("error", f"worker exited with code {rc}; status={summary.get('status')}")
            summary["status"] = "failed"
        summary["duration_s"] = round(time.time() - job.started, 1)
        self.running.pop(job.key, None)
        await self.queue.put(("finished", job, summary))

    # ---------------------------------------------------------- the single writer
    def log_event(self, msg: str) -> None:
        line = f"{now()}  {msg}"
        self.events_fh.write(line + "\n")
        self.events_fh.flush()
        print(line, flush=True)

    def worker_log(self, job: Job):
        fh = self._handles.get(job.key)
        if fh is None:
            d = self.logs_dir / "workers"
            d.mkdir(parents=True, exist_ok=True)
            fh = open(d / f"{job.key}.log", "a", encoding="utf-8")
            self._handles[job.key] = fh
        return fh

    def progress(self) -> dict:
        out = {}
        for j in list(self.running.values()):
            sb = dg_runs.find_sandbox(j.root)
            out[j.key] = dict(evals=dg_runs.n_observed(sb) if sb else 0, minutes=round((time.time() - j.started) / 60, 1))
        return out

    def write_status(self, extra: dict | None = None) -> None:
        agents = [j for j in self.running.values() if j.kind == "agent"]
        bos = [j for j in self.running.values() if j.kind == "bo"]
        progress = self.progress()
        status = dict(
            updated=now(), elapsed_min=round((time.time() - self.t0) / 60, 1),
            running_agents=len(agents), running_bo=len(bos), progress=progress,
            finished_agent=len(self.done["agent"]), finished_bo=len(self.done["bo"]),
            paired_seeds=sorted(self.paired), excluded_seeds=self.excluded, failures=self.failures,
            next_agent_seed=self.agent_ptr, next_bo_seed=self.bo_ptr, last_plot_stats=self.last_stats,
            gpu=(extra or {}).get("gpu", []),
        )
        atomic_write_json(self.logs_dir / "status.json", status)

    def append_result_row(self, job: Job, s: dict) -> None:
        path = self.logs_dir / "results.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "condition", "seed", "status", "n_evals", "best_regret", "first_forced_idx",
                            "forced_count", "duration_s", "error"])
            w.writerow([now(), job.kind, job.seed, s.get("status"), s.get("n_evals"), s.get("best_regret"),
                        s.get("first_forced_idx"), s.get("forced_count"), s.get("duration_s"), s.get("error", "")])

    async def maybe_pair(self, seed: int) -> None:
        bo, ag = self.done["bo"].get(seed), self.done["agent"].get(seed)
        if bo is None or ag is None or seed in self.paired or seed in self.excluded:
            return
        if bo["status"] != "completed" or ag["status"] != "completed" or not bo.get("trace") or not ag.get("trace"):
            bad = [k for k, v in (("bo", bo), ("agent", ag)) if v["status"] != "completed" or not v.get("trace")]
            self.excluded[seed] = "failed: " + ",".join(bad)
            self.log_event(f"seed {seed} EXCLUDED from the figure (failed: {', '.join(bad)}); "
                           f"total failures so far {self.failures}")
            return
        self.paired[seed] = {"bo": bo["trace"], "agent": ag["trace"], "ff": ag.get("first_forced_idx")}
        await self.render()

    async def render(self) -> None:
        loop = asyncio.get_running_loop()
        n = len(self.paired)
        title = f"Hartmann-6, {self.a.model}: mean regret over {n} paired seeds (± 1 SE)" if not self.a.fake else \
            f"[SELFTEST] mean regret over {n} paired seeds (± 1 SE)"
        footer = (f"Seeds in figure = seeds where BO and agent both completed. Failed runs excluded: {self.failures}. "
                  f"Agent runs started {len(self.done['agent']) + sum(1 for j in self.running.values() if j.kind == 'agent')}; "
                  f"updated {now()}.")
        stats = await loop.run_in_executor(
            None, lambda: dg_plot.render(dict(self.paired), self.fig_path, title, footer, self.a.budget))
        self.last_stats = stats
        snap = self.snap_dir / f"regret_n{n:03d}.png"
        snap.parent.mkdir(parents=True, exist_ok=True)
        await loop.run_in_executor(None, shutil.copyfile, self.fig_path, snap)
        self.log_event(f"FIGURE updated: {n} paired seeds | BO final {stats['bo_final_mean']:.2f}+/-{stats['bo_final_se']:.2f} "
                       f"| agent final {stats['agent_final_mean']:.2f}+/-{stats['agent_final_se']:.2f} | "
                       f"median first forced continuation: {stats['median_first_forced']} "
                       f"({stats['n_with_forced']}/{n} runs) -> {self.fig_path}")

    async def writer(self) -> None:
        while True:
            kind, job, payload = await self.queue.get()
            if kind == "stop":
                return
            if kind == "line":
                fh = self.worker_log(job)
                fh.write(payload + "\n")
                fh.flush()
            elif kind == "started":
                self.log_event(f"START  {job.key}  (pid {job.pid})")
                self.write_status()
            elif kind == "heartbeat":
                self.write_status(payload)
                prog = ", ".join(f"{k}:{v['evals']}ev" for k, v in sorted(self.progress().items()))
                self.log_event(f"HEARTBEAT paired={len(self.paired)} excluded={len(self.excluded)} "
                               f"agents={sum(1 for j in self.running.values() if j.kind == 'agent')} "
                               f"bo={sum(1 for j in self.running.values() if j.kind == 'bo')} | {prog}")
            elif kind == "finished":
                s = payload
                self.done[job.kind][job.seed] = s
                self.append_result_row(job, s)
                if s["status"] == "completed":
                    extra = (f" first_forced@{s.get('first_forced_idx')} forced={s.get('forced_count')}"
                             if job.kind == "agent" else "")
                    self.log_event(f"FINISH {job.key}  evals={s.get('n_evals')} best_regret={s.get('best_regret'):.3f}"
                                   f"{extra} ({s.get('duration_s')}s)")
                elif self.stop.is_set():
                    self.log_event(f"INTERRUPTED {job.key} (orchestrator stopping; not counted as a failure)")
                else:
                    self.failures += 1
                    with open(self.logs_dir / "failures.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(time=now(), condition=job.kind, seed=job.seed, error=s.get("error"),
                                                n_evals=s.get("n_evals"))) + "\n")
                    self.log_event(f"FAILED {job.key}: {s.get('error')}  (failures so far: {self.failures})")
                await self.maybe_pair(job.seed)
                self.write_status()
                fh = self._handles.pop(job.key, None)
                if fh:
                    fh.close()

    # ---------------------------------------------------------- dispatchers
    def seeds_exhausted(self, ptr: int) -> bool:
        return bool(self.a.max_seeds) and ptr >= self.a.start_seed + self.a.max_seeds

    def should_stop(self) -> bool:
        return self.stop.is_set() or (self.deadline is not None and time.time() > self.deadline)

    async def dispatch(self, kind: str, sem: asyncio.Semaphore) -> None:
        tasks = []

        async def run_one(job: Job):
            try:
                await self.supervise(job)
            finally:
                sem.release()

        while not self.should_stop():
            ptr = self.agent_ptr if kind == "agent" else self.bo_ptr
            if self.seeds_exhausted(ptr):
                break
            seed = ptr
            if kind == "agent":
                self.agent_ptr += 1
            else:
                # BO stays ahead of the agents, but never runs unboundedly far ahead
                while not self.should_stop() and self.bo_ptr >= self.agent_ptr + self.a.bo_lead:
                    await asyncio.sleep(1.0)
                if self.should_stop():
                    break
                seed = self.bo_ptr
                self.bo_ptr += 1
            if seed in self.done[kind] and self.done[kind][seed]["status"] == "completed":
                continue  # resumed run
            await sem.acquire()
            if self.should_stop():
                sem.release()
                break
            job = self.make_job(kind, seed)
            job.task = asyncio.create_task(run_one(job))
            tasks.append(job.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_watcher(self) -> None:
        await self.stop.wait()
        self.log_event("STOP requested: terminating workers")
        for j in list(self.running.values()):
            kill_tree(j.pid)

    async def heartbeat(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(self.a.heartbeat)
            gpu = await loop.run_in_executor(None, gpu_snapshot)
            await self.queue.put(("heartbeat", None, {"gpu": gpu}))

    # ---------------------------------------------------------- resume
    def resume_scan(self) -> None:
        """Pick up completed runs from a previous session; move unfinished ones aside (never delete)."""
        stale = self.out / "_stale"
        for seed_dir in sorted(self.runs_dir.glob("seed*")) if self.runs_dir.exists() else []:
            seed = int(seed_dir.name.removeprefix("seed"))
            for kind, cond in (("bo", "vanilla"), ("agent", "agent")):
                root = seed_dir / cond
                if not root.is_dir():
                    continue
                s = dg_runs.summarize(root)
                if s["status"] == "completed":
                    self.done[kind][seed] = s
                else:
                    dest = stale / f"seed{seed}-{cond}-{int(time.time())}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(root), str(dest))
                    self.log_event(f"RESUME: moved unfinished {kind} run for seed {seed} to {dest}")
        for seed in sorted(set(self.done["bo"]) & set(self.done["agent"])):
            bo, ag = self.done["bo"][seed], self.done["agent"][seed]
            if bo.get("trace") and ag.get("trace"):
                self.paired[seed] = {"bo": bo["trace"], "agent": ag["trace"], "ff": ag.get("first_forced_idx")}

    # ---------------------------------------------------------- main
    async def main(self) -> int:
        a = self.a
        for d in (self.logs_dir, self.fig_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        self.events_fh = open(self.logs_dir / "events.log", "a", encoding="utf-8")
        self.log_event(f"=== orchestrator start: model={a.model} agents={a.agents} bo_workers={a.bo_workers} "
                       f"bench={a.bench} budget={a.budget} seeds from {a.start_seed}"
                       f"{' (max ' + str(a.max_seeds) + ')' if a.max_seeds else ''} "
                       f"{'[FAKE WORKERS]' if a.fake else ''} out={self.out}")
        if os.name == "nt" and not a.fake:
            self.shim_dir = self.make_python3_shim()
        self.resume_scan()
        if self.paired:
            self.log_event(f"resumed {len(self.paired)} paired seeds from disk")
            await self.render()
        writer = asyncio.create_task(self.writer())
        hb = asyncio.create_task(self.heartbeat())
        watcher = asyncio.create_task(self.stop_watcher())
        agent_sem, bo_sem = asyncio.Semaphore(a.agents), asyncio.Semaphore(a.bo_workers)
        loop = asyncio.get_running_loop()
        if os.name != "nt":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self.stop.set)
        try:
            await asyncio.gather(self.dispatch("agent", agent_sem), self.dispatch("bo", bo_sem))
            # let in-flight pairs finish and be written
            await asyncio.sleep(0.2)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.stop.set()
        finally:
            if self.stop.is_set():
                for j in list(self.running.values()):
                    kill_tree(j.pid)
            hb.cancel()
            watcher.cancel()
            await self.queue.put(("stop", None, None))
            await writer
            self.write_status({"gpu": await loop.run_in_executor(None, gpu_snapshot)})
            self.log_event(f"=== orchestrator end: paired={len(self.paired)} excluded={len(self.excluded)} "
                           f"failures={self.failures}")
            self.events_fh.close()
        return 0

    def make_python3_shim(self) -> str | None:
        probe = subprocess.run(["python3", "-c", ""], capture_output=True)
        if probe.returncode == 0:
            return None
        d = tempfile.mkdtemp(prefix="dg_py3_")
        Path(d, "python3").write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
        return d


# ------------------------------------------------------------------ CLI
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/dual_gpu", help="root for runs/, logs/, figures/")
    ap.add_argument("--fig", default=None, help="figure path (default <out>/figures/regret_dual_gpu.png)")
    ap.add_argument("--bench", default="hartmann6")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=42)
    ap.add_argument("--max-seeds", type=int, default=0, help="0 = no limit (until --hours or Ctrl+C)")
    ap.add_argument("--hours", type=float, default=0.0, help="stop starting new seeds after this long (0 = no limit)")
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--base-url", default="http://127.0.0.1:11435/v1", help="OpenAI-compatible Ollama endpoint")
    ap.add_argument("--agents", default="auto", help="concurrent agent runs (int) or 'auto' = capacity-calculator")
    ap.add_argument("--bo-workers", default="auto", help="concurrent BO runs (int) or 'auto'")
    ap.add_argument("--bo-lead", type=int, default=0, help="BO may start seed s while s < next_agent_seed + lead "
                    "(default 3 x agents)")
    ap.add_argument("--ctx", type=int, default=16384, help="per-agent context (used by 'auto' sizing)")
    ap.add_argument("--kv-type", default="q8_0", choices=["f16", "q8_0", "q4_0"])
    ap.add_argument("--freedom", default="paper", choices=["pinned", "paper", "all"])
    ap.add_argument("--hint", default="vague", choices=["none", "vague"])
    ap.add_argument("--extra-body", default=None, help="JSON merged into the agent's LLM requests (e.g. thinking off)")
    ap.add_argument("--max-auto-agents", type=int, default=6, help="cap for --agents auto (throughput does not scale linearly)")
    ap.add_argument("--bash-timeout", type=int, default=600)
    ap.add_argument("--bo-threads", type=int, default=2)
    ap.add_argument("--agent-threads", type=int, default=2)
    ap.add_argument("--heartbeat", type=float, default=30.0, help="seconds between status/log refreshes")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--fake", action="store_true", help="use dg_worker_fake.py (selftest; no GPU/LLM)")
    ap.add_argument("--fake-source", default=str(REPO / "results/logs/ollama-batch/qwen3.6_27b/hartmann6-blind"))
    ap.add_argument("--fake-agent-secs", type=float, default=4.0)
    ap.add_argument("--fake-bo-secs", type=float, default=1.5)
    ap.add_argument("--fake-fail-seeds", default="", help="comma list of seeds whose fake agent run fails")
    a = ap.parse_args(argv)
    a.fake_fail = {int(x) for x in a.fake_fail_seeds.split(",") if x.strip()}

    n_agents, cap = None, None
    if a.agents == "auto" or a.bo_workers == "auto":
        info = None if a.fake else dg_capacity.ollama_model_info(a.model, a.base_url.removesuffix("/v1"))
        m = info or dict(dg_capacity.PRESETS["qwen3-32b-q4"])
        det = dg_capacity.detect_gpus()
        gpus, vram = (det if det else (2, 24.0))
        c = dg_capacity.compute(ctx=a.ctx, kv_type=a.kv_type, gpus=gpus, vram_gib=vram, weights_gib=m["weights_gib"],
                                layers=m["layers"], kv_heads=m["kv_heads"], head_dim=m["head_dim"])
        n_agents = max(1, min(a.max_auto_agents, c["slots_comfortable"]))
        cap = dg_capacity.cpu_plan(os.cpu_count() or 8, n_agents, a.bo_threads, a.agent_threads)
    a.agents = int(a.agents) if a.agents != "auto" else n_agents
    a.bo_workers = int(a.bo_workers) if a.bo_workers != "auto" else cap["bo_workers_recommended"]
    if a.bo_lead <= 0:
        a.bo_lead = max(3, 3 * a.agents)
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    orch = Orchestrator(a)
    try:
        return asyncio.run(orch.main())
    except KeyboardInterrupt:  # Windows (no loop signal handlers): clean up
        for j in list(orch.running.values()):
            kill_tree(j.pid)
        return 130
    finally:
        if not a.fake:
            subprocess.run([a.python, "-m", "llm.ollama_unload", a.model, a.base_url], cwd=str(REPO),
                           capture_output=True, timeout=60)


if __name__ == "__main__":
    sys.exit(main())
