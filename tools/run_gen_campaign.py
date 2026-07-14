"""One-off launcher for the generalization redesign campaign (Exp 1/2/3).

Runs ALL jobs as a SINGLE rolling pool at a fixed concurrency, each job
wall-clock time-boxed and resumable (--only-missing skips a run dir that
already has >= iterations recorded). Unlike run_campaign.py, each job carries
its OWN effort, so the strong/weak/high efforts and the variant runs all share
one concurrency cap (best packing -> shortest wall clock).

Job matrix (5 seeds each, 1h box, 40-iter cap, gpt-5.5, codex-timeout 1200):
  Exp 1 (prefix E1-):
    - 4 perfect-info tasks  x effort HIGH            (low/none already on disk)
    - 3 generalization tasks x efforts HIGH/LOW/NONE (new train+test default)
  Exp 2 (prefix E2-):  3 <task>_e2  variants x LOW
  Exp 3 (prefix E3-):  3 <task>_r8 + 3 <task>_r16 variants x LOW

Usage:
  python3.12 tools/run_gen_campaign.py [--concurrency 20] [--runs 5]
      [--timebox 3600] [--iterations 40] [--codex-timeout 1200]
      [--only-missing] [--dry-run]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "runs" / "_campaign"
LOG = CAMP / "gen_campaign.jsonl"
MODEL = "gpt-5.5"

PERFECT = ["mem_index", "mem_str", "mem_infer", "ops_connect"]
GEN = ["easy_word_problems", "tag_seq", "compress_heldout"]


def _interleave(groups):
    """Round-robin flatten a list of lists, so consecutive items come from
    different groups (spreads heavy high-effort jobs across the campaign and
    keeps every wave diverse for monitoring)."""
    out, i = [], 0
    while any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g):
                out.append(g[i])
        i += 1
    return out


def build_jobs(runs):
    """Full job list of (task, effort, prefix, k), interleaved + seed-major so
    the first wave samples all experiment types and high-effort load is spread."""
    g_perf_high = [(t, "high", "E1-") for t in PERFECT]
    g_gen_high = [(t, "high", "E1-") for t in GEN]
    g_gen_low = [(t, "low", "E1-") for t in GEN]
    g_gen_none = [(t, "none", "E1-") for t in GEN]
    g_e2 = [(f"{t}_e2", "low", "E2-") for t in GEN]
    g_e3 = [x for t in GEN for x in ((f"{t}_r8", "low", "E3-"), (f"{t}_r16", "low", "E3-"))]
    specs = _interleave([g_perf_high, g_gen_high, g_gen_low, g_gen_none, g_e2, g_e3])
    # seed-major: all r1 across specs, then r2, ... -> each wave is diverse
    jobs = [(t, eff, pfx, k) for k in range(1, runs + 1) for (t, eff, pfx) in specs]
    return jobs


def run_dir_for(task, eff, pfx, k):
    return ROOT / "runs" / task / f"{pfx}r{k}-{MODEL}-{eff}"


def iters_done(rd):
    log = rd / "log.jsonl"
    if not log.exists():
        return 0
    n = 0
    for line in log.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e.get("iter"), int) and e.get("event") != "baseline":
            n += 1
    return n


def log_event(ev):
    CAMP.mkdir(parents=True, exist_ok=True)
    ev = dict(ev)
    ev.setdefault("t", round(time.time(), 1))
    with open(LOG, "a") as f:
        f.write(json.dumps(ev) + "\n")
    print(f"[gencamp] {ev.get('event')}: {ev.get('job','')} {ev.get('detail','')}",
          flush=True)


def launch(task, eff, pfx, k, args):
    rd = run_dir_for(task, eff, pfx, k)
    rd.mkdir(parents=True, exist_ok=True)
    stdout = open(rd / "launcher_stdout.txt", "a")
    cmd = [sys.executable, "-m", "loop.optimize",
           "--task", task, "--model", MODEL, "--effort", eff,
           "--iterations", str(args.iterations),
           "--codex-timeout", str(args.codex_timeout),
           "--run-dir", str(rd)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return {"proc": proc, "stdout": stdout, "start": time.time(),
            "rd": rd, "job": f"{task}:{eff}:r{k}"}


def kill_job(job, sig=signal.SIGTERM):
    try:
        os.killpg(os.getpgid(job["proc"].pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timebox", type=int, default=3600)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--codex-timeout", type=int, default=1200)
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--poll", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.runs)
    if args.only_missing:
        jobs = [j for j in jobs
                if iters_done(run_dir_for(*j)) < args.iterations]

    print(f"[gencamp] {len(jobs)} jobs, concurrency={args.concurrency}, "
          f"timebox={args.timebox}s, iters<={args.iterations}")
    by_pfx = {}
    for (t, eff, pfx, k) in jobs:
        by_pfx.setdefault(pfx, 0)
        by_pfx[pfx] += 1
    print(f"[gencamp] by experiment: {by_pfx}")
    if args.dry_run:
        for (t, eff, pfx, k) in jobs:
            print(f"  {pfx} {t:<24} {eff:<5} r{k}")
        return

    log_event({"event": "start", "detail": f"{len(jobs)} jobs conc={args.concurrency}"})
    pending = list(jobs)
    running = []
    done = 0
    total = len(jobs)
    while pending or running:
        while pending and len(running) < args.concurrency:
            t, eff, pfx, k = pending.pop(0)
            job = launch(t, eff, pfx, k, args)
            running.append(job)
            log_event({"event": "launch", "job": job["job"],
                       "detail": f"pid={job['proc'].pid} ({total - len(pending)}/{total})"})
        time.sleep(args.poll)
        still = []
        for job in running:
            rc = job["proc"].poll()
            elapsed = time.time() - job["start"]
            if rc is not None:
                job["stdout"].close()
                done += 1
                log_event({"event": "finish", "job": job["job"],
                           "detail": f"rc={rc} elapsed={elapsed:.0f}s "
                           f"iters={iters_done(job['rd'])} ({done}/{total})"})
            elif elapsed >= args.timebox:
                kill_job(job, signal.SIGTERM)
                time.sleep(5)
                if job["proc"].poll() is None:
                    kill_job(job, signal.SIGKILL)
                job["proc"].wait()
                job["stdout"].close()
                done += 1
                log_event({"event": "timeout", "job": job["job"],
                           "detail": f"boxed at {elapsed:.0f}s "
                           f"iters={iters_done(job['rd'])} ({done}/{total})"})
            else:
                still.append(job)
        running = still
    log_event({"event": "done", "detail": f"{done} jobs finished"})


if __name__ == "__main__":
    main()
