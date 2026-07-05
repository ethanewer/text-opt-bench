"""Campaign launcher: run N independent loop.optimize runs per task under a
fixed concurrency cap, each wall-clock time-boxed, and resumable.

Each job runs:
    python3.12 -m loop.optimize --task T --model M --effort E
        --iterations ITERS --run-dir runs/T/<prefix>r<k>-M-E
in its own process group (codex spawns children; we kill the whole group on
timeout). Because loop.optimize resumes an existing --run-dir, re-running the
same job later simply extends it — that's how "let good runs go longer" works.

Progress is appended to runs/_campaign/launcher.jsonl (one JSON event per
line: launch / finish / timeout / kill), so the campaign timeline survives
even if we lose the terminal.

Usage:
    python3.12 tools/run_campaign.py --tasks a,b,c --runs 5 --concurrency 10 \
        --timebox 3600 --iterations 30 [--only-missing] [--jobs t:k,t:k]
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
LOG = CAMP / "launcher.jsonl"


def now():
    return time.time()


def log_event(ev):
    CAMP.mkdir(parents=True, exist_ok=True)
    ev = dict(ev)
    ev.setdefault("t", round(now(), 1))
    with open(LOG, "a") as f:
        f.write(json.dumps(ev) + "\n")
    print(f"[campaign] {ev.get('event')}: {ev.get('job','')} {ev.get('detail','')}",
          flush=True)


def run_dir_for(task, k, model, effort, prefix):
    return ROOT / "runs" / task / f"{prefix}r{k}-{model}-{effort}"


def iters_done(rd):
    """Count accepted+rejected+invalid loop iterations recorded so far."""
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


def launch(task, k, args):
    rd = run_dir_for(task, k, args.model, args.effort, args.prefix)
    rd.mkdir(parents=True, exist_ok=True)
    stdout = open(rd / "launcher_stdout.txt", "a")
    cmd = [
        sys.executable, "-m", "loop.optimize",
        "--task", task,
        "--model", args.model,
        "--effort", args.effort,
        "--iterations", str(args.iterations),
        "--codex-timeout", str(args.codex_timeout),
        "--run-dir", str(rd),
    ]
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group -> killpg reaches codex
    )
    return {"task": task, "k": k, "rd": rd, "proc": proc, "stdout": stdout,
            "start": now(), "job": f"{task}:r{k}"}


def kill_job(job, sig=signal.SIGTERM):
    try:
        os.killpg(os.getpgid(job["proc"].pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="", help="comma-separated task names")
    ap.add_argument("--jobs", default="", help="explicit task:k pairs, comma-sep "
                    "(overrides --tasks/--runs; for targeted extends)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--timebox", type=int, default=3600, help="per-run wall seconds")
    ap.add_argument("--iterations", type=int, default=30, help="iteration cap per run")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--prefix", default="5x-")
    ap.add_argument("--codex-timeout", type=int, default=900)
    ap.add_argument("--only-missing", action="store_true",
                    help="skip jobs whose run dir already has >= iterations done")
    ap.add_argument("--poll", type=int, default=15)
    args = ap.parse_args()

    if args.jobs:
        jobs = []
        for pair in args.jobs.split(","):
            t, k = pair.split(":")
            jobs.append((t.strip(), int(k.strip().lstrip("r"))))
    else:
        tasks = [t for t in args.tasks.split(",") if t.strip()]
        jobs = [(t.strip(), k) for t in tasks for k in range(1, args.runs + 1)]

    if args.only_missing:
        jobs = [(t, k) for (t, k) in jobs
                if iters_done(run_dir_for(t, k, args.model, args.effort, args.prefix))
                < args.iterations]

    # Rotate any prior campaign log so a new campaign starts fresh (a stale
    # campaign_done / old finish events otherwise confuse the monitor+dashboard).
    if LOG.exists():
        n = 0
        while (LOG.parent / f"{LOG.name}.{n}").exists():
            n += 1
        LOG.rename(LOG.parent / f"{LOG.name}.{n}")

    log_event({"event": "campaign_start", "detail": f"{len(jobs)} jobs, "
               f"conc={args.concurrency}, timebox={args.timebox}s, "
               f"iters<={args.iterations}, model={args.model}-{args.effort}"})

    pending = list(jobs)
    running = []
    done = 0
    while pending or running:
        # Fill free slots.
        while pending and len(running) < args.concurrency:
            t, k = pending.pop(0)
            job = launch(t, k, args)
            running.append(job)
            log_event({"event": "launch", "job": job["job"],
                       "detail": f"pid={job['proc'].pid} dir={job['rd'].name}"})
        # Poll running jobs.
        time.sleep(args.poll)
        still = []
        for job in running:
            rc = job["proc"].poll()
            elapsed = now() - job["start"]
            if rc is not None:
                job["stdout"].close()
                done += 1
                log_event({"event": "finish", "job": job["job"],
                           "detail": f"rc={rc} elapsed={elapsed:.0f}s "
                           f"iters={iters_done(job['rd'])} ({done}/{len(jobs)})"})
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
                           f"iters={iters_done(job['rd'])} ({done}/{len(jobs)})"})
            else:
                still.append(job)
        running = still

    log_event({"event": "campaign_done", "detail": f"{done} jobs finished"})


if __name__ == "__main__":
    main()
