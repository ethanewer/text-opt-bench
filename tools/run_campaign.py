"""Campaign launcher: run N independent loop.optimize runs per task under a
fixed concurrency cap, each wall-clock time-boxed, and resumable.

Each job runs:
    python3.12 -m loop.optimize --task T --agent A --model M --effort E
        --iterations ITERS --run-dir runs/T/<prefix>r<k>-A-M-E
in its own process group (the agent may spawn children; we kill the whole group on
timeout). Because loop.optimize resumes an existing --run-dir, re-running the
same job later simply extends it — that's how "let good runs go longer" works.

Progress is appended to runs/_campaign/launcher.jsonl (one JSON event per
line: launch / finish / timeout / kill), so the campaign timeline survives
even if we lose the terminal.

Usage:
    python3.12 tools/run_campaign.py --tasks a,b,c --runs 5 --concurrency 20 \
        --timebox 3600 --iterations 30 [--only-missing] [--jobs t:k,t:k]
"""

import argparse
from contextlib import nullcontext
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.calibrate import recommend_concurrency
from bench import deferred, runner
from bench.resource_lock import LIMITS_ENV, LOCK_DIR_ENV, WAIT_LOG_ENV
from bench.slm_mps_lock import exclusive_campaign_mps_phase
from bench.slm_private import require_private_slm_operator_state_absent

CAMP = ROOT / "runs" / "_campaign"
LOG = CAMP / "launcher.jsonl"
DEFERRED_CACHE_ROOT = CAMP / "deferred"
DEFERRED_MANAGED_ENV = "TEXTOPT_DEFERRED_MANAGED"


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


def run_dir_for(task, k, agent, model, effort, prefix):
    return ROOT / "runs" / task / f"{prefix}r{k}-{agent}-{model}-{effort}"


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


def eval_queue_seconds(wait_log, since, until=None):
    """Union of this launch's completed and currently-active queue intervals."""
    until = now() if until is None else until
    path = Path(wait_log)
    if not path.exists():
        return 0.0
    starts = {}
    intervals = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return 0.0
    for line in lines:
        try:
            event = json.loads(line)
            event_id = event["id"]
            ts = float(event["ts"])
            pid = int(event["pid"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if event.get("event") == "start":
            # Ignore stale intervals belonging to an earlier resumed launch.
            if ts >= since:
                starts[event_id] = (ts, pid)
        elif event.get("event") == "end" and event_id in starts:
            start, _ = starts.pop(event_id)
            intervals.append((start, min(ts, until)))
    for start, pid in starts.values():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass  # alive, but owned by another user
        intervals.append((start, until))
    intervals = sorted((max(start, since), min(end, until))
                       for start, end in intervals if end > start)
    if not intervals:
        return 0.0
    total = 0.0
    covered_until = since
    for start, end in intervals:
        if end <= covered_until:
            continue
        total += end - max(start, covered_until)
        covered_until = end
    return max(0.0, total)


def launch(task, k, args):
    rd = run_dir_for(task, k, args.agent, args.model, args.effort, args.prefix)
    rd.mkdir(parents=True, exist_ok=True)
    stdout = open(rd / "launcher_stdout.txt", "a")
    cmd = [
        sys.executable, "-m", "loop.optimize",
        "--task", task,
        "--agent", args.agent,
        "--model", args.model,
        "--effort", args.effort,
        "--iterations", str(args.iterations),
        "--codex-timeout", str(args.codex_timeout),
        "--run-dir", str(rd),
    ]
    if task == "slm_weight_compression_lfm25":
        cmd.extend(("--device", args.slm_device))
    env = os.environ.copy()
    # Codex self-evaluations run in a workspace-write sandbox. Its writable
    # roots include the iteration workspace and the system temp directory,
    # but not the parent run/campaign directories. Put only coordination
    # files in a unique per-launch temp namespace so self-evaluation and
    # parent-driven submissions share the same gates.
    wait_log = args.eval_coord_dir / "waits" / f"{task}-r{k}.jsonl"
    wait_log.parent.mkdir(parents=True, exist_ok=True)
    env[LOCK_DIR_ENV] = str(args.eval_coord_dir / "locks")
    env[WAIT_LOG_ENV] = str(wait_log)
    env[LIMITS_ENV] = json.dumps({
        "cpu": args.eval_cpu_concurrency,
        "accelerator": args.eval_accelerator_concurrency,
    }, sort_keys=True)
    env[DEFERRED_MANAGED_ENV] = "1"
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # own process group -> killpg reaches agent
        )
    except BaseException:
        stdout.close()
        raise
    return {"task": task, "k": k, "rd": rd, "proc": proc, "stdout": stdout,
            # start_new_session=True makes the child PID the process-group ID.
            # Retain it explicitly: once the group leader exits, getpgid(pid)
            # can no longer find the still-running descendants we must reap.
            "pgid": proc.pid,
            "start": now(), "job": f"{task}:r{k}", "wait_log": wait_log}


def launch_deferred(request, args):
    """Launch one resumable low-priority held-out test shard."""
    stdout_path = CAMP / "holdout_worker.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = open(stdout_path, "a")
    cmd = [
        sys.executable, "-m", "bench.deferred", "score-shard",
        request["run_dir"], str(request["n"]),
        str(args.deferred_cache_dir),
        request["shard"],
    ]
    env = os.environ.copy()
    env[LOCK_DIR_ENV] = str(args.eval_coord_dir / "locks")
    # Deliberately omit WAIT_LOG_ENV: background holdout queueing must not
    # extend an optimizer's active-time budget.
    env.pop(WAIT_LOG_ENV, None)
    env[LIMITS_ENV] = json.dumps({
        "cpu": args.eval_cpu_concurrency,
        "accelerator": args.eval_accelerator_concurrency,
    }, sort_keys=True)
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, env=env,
            # A worker owns an evaluator child.  Give both a process group so
            # launcher cleanup cannot orphan model compute after an exception.
            start_new_session=True,
        )
    except BaseException:
        stdout.close()
        raise
    return {"proc": proc, "pgid": proc.pid, "stdout": stdout,
            "request": request,
            "start": now()}


def _latest_accepted_record(run_dir):
    """Return the newest immutable accepted-incumbent record for one run."""
    records = []
    for line in deferred._record_lines(run_dir):
        record = json.loads(line)
        if record.get("ok") and record.get("best"):
            records.append(record)
    if not records:
        return None
    latest = records[-1]
    if latest.get("n") != max(int(record["n"]) for record in records):
        raise RuntimeError(f"accepted submission order is corrupt in {run_dir}")
    return latest


def deferred_request(run_dirs, args, skip_tasks=()):
    """Return work only for each run's newest accepted incumbent.

    This is deliberately a launcher policy rather than a change to the sealed
    evaluator protocol. Older immutable submissions and any holdouts already
    attached to them remain valid, but a campaign no longer drains every
    superseded incumbent merely because it was briefly best online.
    """
    skipped = frozenset(skip_tasks)
    for raw_run_dir in sorted(map(Path, run_dirs), key=str):
        run_dir = Path(raw_run_dir)
        session_path = run_dir / "session.json"
        if not session_path.exists():
            continue
        meta = json.loads(session_path.read_text())
        task = meta["task"]
        config = runner.load_config(task)
        if task in skipped or not config.get("deferred_test"):
            continue
        record = _latest_accepted_record(run_dir)
        if record is None:
            continue
        number = int(record["n"])
        program_sha256 = record["program_sha256"]
        completed = deferred.read_results(run_dir)
        if number in completed:
            continue
        if deferred.assemble_cached(run_dir, number, args.deferred_cache_dir):
            continue
        development_profile = config.get("development_profile", "mixed")
        for shard in config.get("test_shards", ()):
            if deferred.read_shard(
                    args.deferred_cache_dir, task, development_profile,
                    program_sha256, shard) is None:
                return {
                    "run_dir": str(run_dir), "n": number,
                    "task": task, "program_sha256": program_sha256,
                    "development_profile": development_profile,
                    "shard": shard,
                }
    return None


def kill_job(job, sig=signal.SIGTERM):
    try:
        pgid = job.get("pgid")
        if pgid is None:
            # Compatibility for callers constructing legacy job dictionaries.
            # Real launcher jobs always retain the group ID at creation time.
            pgid = os.getpgid(job["proc"].pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def cleanup_processes(running, background, grace_seconds=5.0):
    """Terminate, reap, and close every process still owned by the launcher.

    All optimizer jobs and deferred workers start their own sessions, so one
    process-group signal also reaches any Codex/evaluator descendants.  Signal
    every live group before waiting for any one of them; this keeps cleanup
    bounded by one grace interval rather than one interval per job.
    """
    targets = list(running)
    if background is not None:
        targets.append(background)
    # Signal the retained group even when its leader has already exited. A
    # failed optimizer/worker can otherwise leave an evaluator or Codex child
    # consuming CPU/MPS indefinitely.
    for target in targets:
        kill_job(target, signal.SIGTERM)

    deadline = time.monotonic() + max(0.0, grace_seconds)
    for target in targets:
        proc = target["proc"]
        if proc.poll() is None:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass

    # Always make a second group-wide pass. The leader may have exited in
    # response to SIGTERM while a descendant ignored it; checking only
    # proc.poll() would miss exactly that orphan case.
    for target in targets:
        kill_job(target, signal.SIGKILL)

    for target in targets:
        proc = target["proc"]
        try:
            # SIGKILL cannot be handled by the child. Reap without leaving an
            # unowned descendant behind; already-polled children return here
            # immediately as well.
            proc.wait()
        except ChildProcessError:
            pass
        try:
            target["stdout"].close()
        except (AttributeError, OSError):
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="", help="comma-separated task names")
    ap.add_argument("--jobs", default="", help="explicit task:k pairs, comma-sep "
                    "(overrides --tasks/--runs; for targeted extends)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument(
        "--eval-cpu-concurrency", type=int, default=None,
        help="simultaneous CPU evaluations (default: host-calibrated)",
    )
    ap.add_argument(
        "--eval-accelerator-concurrency", type=int, default=1,
        help="simultaneous CUDA/MPS/model evaluations (default: 1)",
    )
    ap.add_argument("--timebox", type=int, default=3600, help="per-run wall seconds")
    ap.add_argument("--iterations", type=int, default=30, help="iteration cap per run")
    ap.add_argument("--agent", default="codex", choices=["codex", "cursor"])
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--prefix", default="5x-")
    ap.add_argument("--codex-timeout", type=int, default=900)
    ap.add_argument(
        "--slm-device", choices=("mps", "cuda"), default="mps",
        help="backend for slm_weight_compression_lfm25 jobs (default: mps)",
    )
    ap.add_argument("--only-missing", action="store_true",
                    help="skip jobs whose run dir already has >= iterations done")
    ap.add_argument("--poll", type=int, default=15)
    args = ap.parse_args()

    # Iteration workspaces cannot write their parent repository, but they can
    # read it. Never launch while plaintext generation references, selections,
    # judge records, operator-final score curves, or the source prompt/reference
    # catalogs remain in the checkout.
    require_private_slm_operator_state_absent()

    args.eval_coord_dir = (Path(tempfile.gettempdir()) /
                           f"text-opt-bm-campaign-{os.getpid()}-{time.time_ns()}")
    # Keep sealed scientific results out of the temp coordination namespace
    # exposed to optimizer subprocesses. This stable operator-owned directory
    # also permits resumable content-addressed reuse across launcher restarts.
    args.deferred_cache_dir = DEFERRED_CACHE_ROOT
    args.deferred_cache_dir.mkdir(parents=True, exist_ok=True)
    args.deferred_cache_dir.chmod(0o700)

    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    if args.eval_cpu_concurrency is None:
        args.eval_cpu_concurrency = recommend_concurrency(os.cpu_count() or 1)
    if args.eval_cpu_concurrency < 1:
        ap.error("--eval-cpu-concurrency must be >= 1")
    if args.eval_accelerator_concurrency < 1:
        ap.error("--eval-accelerator-concurrency must be >= 1")

    if args.jobs:
        jobs = []
        for pair in args.jobs.split(","):
            t, k = pair.split(":")
            jobs.append((t.strip(), int(k.strip().lstrip("r"))))
    else:
        tasks = [t for t in args.tasks.split(",") if t.strip()]
        jobs = [(t.strip(), k) for t in tasks for k in range(1, args.runs + 1)]

    all_requested_jobs = list(jobs)

    if args.only_missing:
        jobs = [(t, k) for (t, k) in jobs
                if iters_done(run_dir_for(t, k, args.agent, args.model,
                                          args.effort, args.prefix))
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
               f"eval_cpu={args.eval_cpu_concurrency}, "
               f"eval_accelerator={args.eval_accelerator_concurrency}, "
               f"iters<={args.iterations}, agent={args.agent}, "
               f"model={args.model}-{args.effort}"})

    pending = list(jobs)
    running = []
    # Include skipped/resumed runs in the deferred drain. `--only-missing`
    # must not strand holdouts from optimizers that were already complete.
    known_run_dirs = [
        run_dir_for(task, k, args.agent, args.model, args.effort, args.prefix)
        for task, k in all_requested_jobs
    ]
    background = None
    deferred_failures = {}
    done = 0
    active_tasks = set(runner.list_tasks())
    has_slm_mps = args.slm_device == "mps" and any(
        task in active_tasks and
        runner.load_config(task).get("required_device") == "mps"
        for task, _run in all_requested_jobs)
    campaign_phase = (exclusive_campaign_mps_phase()
                      if has_slm_mps else nullcontext())
    campaign_phase.__enter__()
    try:
        while pending or running:
            # Fill free slots.
            while pending and len(running) < args.concurrency:
                t, k = pending.pop(0)
                job = launch(t, k, args)
                running.append(job)
                log_event({"event": "launch", "job": job["job"],
                           "detail": f"pid={job['proc'].pid} dir={job['rd'].name}"})
            # Poll the low-priority worker independently of optimization slots.
            if background is not None and background["proc"].poll() is not None:
                rc = background["proc"].returncode
                request = background["request"]
                cleanup_processes([], background, grace_seconds=0.0)
                key = (request["task"], request["program_sha256"], request["shard"])
                if rc:
                    deferred_failures[key] = deferred_failures.get(key, 0) + 1
                    log_event({"event": "holdout_error",
                               "job": f"{request['task']}:{request['n']}",
                               "detail": f"shard={request['shard']} rc={rc} "
                                         f"attempt={deferred_failures[key]}"})
                    if deferred_failures[key] >= 3:
                        raise RuntimeError(
                            f"deferred holdout shard failed three times: {key}")
                else:
                    log_event({"event": "holdout_shard",
                               "job": f"{request['task']}:{request['n']}",
                               "detail": f"shard={request['shard']} "
                                         f"elapsed={now()-background['start']:.1f}s"})
                background = None
            if background is None:
                # Model-bearing holdouts are non-preemptive once launched.
                # Postpone them while an optimizer for the same accelerator
                # task is alive; otherwise a background test can make
                # foreground validation wait for minutes even though it has
                # higher queue priority. CPU tasks still use idle capacity and
                # coalesce naturally to their newest incumbent.
                active_accelerator_tasks = {
                    job["task"] for job in running
                    if (job["task"] in active_tasks and
                        runner.load_config(job["task"]).get(
                            "evaluation_resource") == "accelerator")
                }
                request = deferred_request(
                    known_run_dirs, args, skip_tasks=active_accelerator_tasks)
                if request is not None:
                    background = launch_deferred(request, args)

            # Poll running jobs.
            time.sleep(args.poll)
            still = []
            for job in running:
                rc = job["proc"].poll()
                wall_elapsed = now() - job["start"]
                queue_elapsed = eval_queue_seconds(job["wait_log"], job["start"])
                elapsed = max(0.0, wall_elapsed - queue_elapsed)
                if rc is not None:
                    cleanup_processes([job], None, grace_seconds=0.0)
                    done += 1
                    event = "finish" if rc == 0 else "optimizer_error"
                    log_event({"event": event, "job": job["job"],
                               "detail": f"rc={rc} active={elapsed:.0f}s "
                               f"wall={wall_elapsed:.0f}s "
                               f"queue_refund={queue_elapsed:.0f}s "
                               f"iters={iters_done(job['rd'])} "
                               f"({done}/{len(jobs)})"})
                    if rc != 0:
                        raise RuntimeError(
                            f"optimizer job {job['job']} exited with status {rc}")
                elif elapsed >= args.timebox:
                    cleanup_processes([job], None, grace_seconds=5.0)
                    done += 1
                    log_event({"event": "timeout", "job": job["job"],
                               "detail": f"boxed at active={elapsed:.0f}s "
                               f"wall={wall_elapsed:.0f}s "
                               f"queue_refund={queue_elapsed:.0f}s "
                               f"iters={iters_done(job['rd'])} "
                               f"({done}/{len(jobs)})"})
                else:
                    still.append(job)
            running = still

        log_event({"event": "optimizers_done", "detail": f"{done} jobs finished"})
        log_event({"event": "holdout_drain_start",
                   "detail": "draining final incumbents"})
        while True:
            if background is not None:
                rc = background["proc"].wait()
                request = background["request"]
                cleanup_processes([], background, grace_seconds=0.0)
                key = (request["task"], request["program_sha256"], request["shard"])
                if rc:
                    deferred_failures[key] = deferred_failures.get(key, 0) + 1
                    log_event({"event": "holdout_error",
                               "job": f"{request['task']}:{request['n']}",
                               "detail": f"shard={request['shard']} rc={rc} "
                                         f"attempt={deferred_failures[key]}"})
                    if deferred_failures[key] >= 3:
                        raise RuntimeError(
                            f"deferred holdout shard failed three times: {key}")
                else:
                    log_event({"event": "holdout_shard",
                               "job": f"{request['task']}:{request['n']}",
                               "detail": f"shard={request['shard']} "
                                         f"elapsed={now()-background['start']:.1f}s"})
                background = None
            request = deferred_request(known_run_dirs, args)
            if request is None:
                break
            background = launch_deferred(request, args)
        log_event({"event": "holdout_drain_done",
                   "detail": "all accepted incumbents scored"})
        log_event({"event": "campaign_done", "detail": f"{done} jobs finished"})
    except BaseException as exc:
        log_event({"event": "campaign_error",
                   "detail": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        try:
            cleanup_processes(running, background)
        finally:
            campaign_phase.__exit__(None, None, None)


if __name__ == "__main__":
    main()
