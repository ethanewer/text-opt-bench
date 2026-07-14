"""Resource-aware, pausable benchmark campaign runner.

This is the durable runner for large mixed-task campaigns. Agent rollout
concurrency is independent of evaluator capacity: by default 24 agents may
think concurrently, while evaluations share weighted CPU/accelerator queues.
The default profile gives CPU capacity 16, cheap tasks one unit, and heavier
tasks more units. Accelerator tasks may request CPU capacity at the same time.

Examples::

    python3.12 tools/run_benchmark.py start july --tasks word_problems,optimizer_generalization --runs 5
    python3.12 tools/run_benchmark.py status july
    python3.12 tools/run_benchmark.py pause july
    python3.12 tools/run_benchmark.py resume july

Ctrl-C requests the same clean pause as the ``pause`` command. Sessions,
iteration history, active seconds, and the latest accepted submission remain
in their run directories. Resume does not re-grade the baseline. Evaluator
queue wait and time between pause/resume are excluded from the active budget.
"""

import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import runner  # noqa: E402
from bench.resource_lock import (LIMITS_ENV, LOCK_DIR_ENV, REQUESTS_ENV,
                                 WAIT_LOG_ENV)  # noqa: E402
from bench.slm_mps_lock import exclusive_campaign_mps_phase  # noqa: E402
from bench.slm_private import require_private_slm_operator_state_absent  # noqa: E402
from loop.history import HistoryRepo  # noqa: E402
from tools import run_campaign as legacy  # noqa: E402

CAMP_ROOT = ROOT / "runs" / "_campaign"
STATE_ROOT = CAMP_ROOT / "benchmarks"
PROFILE = ROOT / "tools" / "benchmark_resources.json"
FORMAT = 1
DEFERRED_MANAGED_ENV = "TEXTOPT_DEFERRED_MANAGED"


def now():
    return time.time()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                               dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w") as handle:
            fd = -1
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if fd != -1:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _campaign_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name or ""):
        raise ValueError(
            "campaign name must be 1-80 letters, digits, '.', '_' or '-'")
    return name


class StateStore:
    """Small lock-serialized durable state file shared with pause/status."""

    def __init__(self, name):
        self.name = _campaign_name(name)
        self.directory = STATE_ROOT / self.name
        self.path = self.directory / "state.json"
        self.state_lock = self.directory / "state.lock"
        self.controller_lock = self.directory / "controller.lock"
        self.log_path = CAMP_ROOT / f"launcher.jsonl.{self.name}"

    def exists(self):
        return self.path.exists()

    def read(self):
        with open(self.state_lock, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            return json.loads(self.path.read_text())

    def replace(self, value):
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.state_lock, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = dict(value)
            value["updated_ts"] = round(now(), 3)
            _atomic_json(self.path, value)
            return value

    def update(self, mutate):
        with open(self.state_lock, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text())
            replacement = mutate(state)
            if replacement is not None:
                state = replacement
            state["updated_ts"] = round(now(), 3)
            _atomic_json(self.path, state)
            return state

    def controller(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = open(self.controller_lock, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise RuntimeError(
                f"campaign {self.name!r} already has a live controller")
        return handle

    def event(self, event, job="", detail="", **fields):
        record = {"event": event, "campaign": self.name,
                  "t": round(now(), 1)}
        if job:
            record["job"] = job
        if detail:
            record["detail"] = detail
        record.update(fields)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[benchmark:{self.name}] {event}: {job} {detail}", flush=True)


def load_profile(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("resource profile must be a format-1 JSON object")
    capacities = payload.get("resource_capacities")
    if not isinstance(capacities, dict) or not capacities:
        raise ValueError("resource profile needs resource_capacities")
    clean_capacities = {}
    for resource, units in capacities.items():
        if (not isinstance(resource, str) or not resource or
                not resource.replace("_", "").replace("-", "").isalnum() or
                not isinstance(units, int) or isinstance(units, bool) or
                units < 1):
            raise ValueError("resource capacities must be positive integers")
        clean_capacities[resource] = units
    default_units = payload.get("default_units", 1)
    if (not isinstance(default_units, int) or isinstance(default_units, bool)
            or default_units < 1):
        raise ValueError("default_units must be a positive integer")
    task_units = payload.get("task_units", {})
    if not isinstance(task_units, dict):
        raise ValueError("task_units must be an object")
    clean_tasks = {}
    for task, units in task_units.items():
        if (not isinstance(task, str) or not task or
                not isinstance(units, int) or isinstance(units, bool)
                or units < 1):
            raise ValueError("task unit costs must be positive integers")
        clean_tasks[task] = units
    task_requests = payload.get("task_requests", {})
    if not isinstance(task_requests, dict):
        raise ValueError("task_requests must be an object")
    clean_requests = {}
    for task, requests in task_requests.items():
        if not isinstance(task, str) or not task or not isinstance(requests, dict):
            raise ValueError("task resource requests must be nested objects")
        clean = {}
        for resource, units in requests.items():
            if (not isinstance(resource, str) or not resource or
                    not resource.replace("_", "").replace("-", "").isalnum() or
                    not isinstance(units, int) or isinstance(units, bool) or
                    units < 1):
                raise ValueError(
                    "task resource requests must be positive integers")
            clean[resource] = units
        if not clean:
            raise ValueError("task resource requests cannot be empty")
        clean_requests[task] = clean
    return {
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "resource_capacities": clean_capacities,
        "default_units": default_units,
        "task_units": clean_tasks,
        "task_requests": clean_requests,
    }


def task_requests(task, profile):
    """Return all capacity requests made by one evaluation of ``task``."""
    primary = runner.load_config(task).get("evaluation_resource", "cpu")
    explicit = profile.get("task_requests", {}).get(task)
    if explicit is None:
        requests = {
            primary: profile["task_units"].get(task, profile["default_units"])
        }
    else:
        requests = dict(explicit)
        if primary not in requests:
            raise ValueError(
                f"task {task!r} resource request omits primary resource "
                f"{primary!r}")
    for resource, units in requests.items():
        capacity = profile["resource_capacities"].get(resource)
        if capacity is None:
            raise ValueError(
                f"task {task!r} uses unconfigured resource {resource!r}")
        if units > capacity:
            raise ValueError(
                f"task {task!r} requests {units} {resource} units, above "
                f"capacity {capacity}")
    return requests


def task_request(task, profile):
    """Backward-compatible primary-resource view of :func:`task_requests`."""
    primary = runner.load_config(task).get("evaluation_resource", "cpu")
    return primary, task_requests(task, profile)[primary]


def _jobs(args):
    if args.jobs:
        result = []
        for raw in args.jobs.split(","):
            try:
                task, run = raw.split(":", 1)
                number = int(run.strip().lstrip("r"))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid --jobs entry {raw!r}") from exc
            result.append((task.strip(), number))
    else:
        tasks = [task.strip() for task in args.tasks.split(",")
                 if task.strip()]
        # Round-robin tasks so a campaign with many runs does not occupy all
        # agent slots with one task before the mixed resource queue gets work.
        result = [(task, run) for run in range(1, args.runs + 1)
                  for task in tasks]
    if not result:
        raise ValueError("provide --tasks or --jobs")
    known = set(runner.list_tasks())
    unknown = sorted({task for task, _run in result if task not in known})
    if unknown:
        raise ValueError(f"unknown/retired tasks: {', '.join(unknown)}")
    if any(run < 1 for _task, run in result):
        raise ValueError("run numbers must be >= 1")
    if len(set(result)) != len(result):
        raise ValueError("duplicate task:run jobs are not allowed")
    return result


def _submission_count(run_dir):
    path = Path(run_dir) / "submissions.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def iterations_done(run_dir):
    """Count durable loop attempts across every interruption window.

    A pause can land after any one of the session append, history ref update,
    or loop-log append. Their union prevents either repeating a completed
    attempt or granting an extra iteration after resume.
    """
    run_dir = Path(run_dir)
    numbers = set()
    log = run_dir / "log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (isinstance(record.get("iter"), int)
                    and record.get("event") != "baseline"):
                numbers.add(record["iter"])
    submissions = run_dir / "submissions.jsonl"
    if submissions.exists():
        for line in submissions.read_text().splitlines():
            try:
                note = str(json.loads(line).get("note", ""))
                if note.startswith("loop iter "):
                    numbers.add(int(note.rsplit(" ", 1)[1]))
            except (json.JSONDecodeError, ValueError):
                continue
    for ref in HistoryRepo(run_dir).attempt_refs():
        try:
            numbers.add(int(ref.rsplit("-", 1)[1]))
        except ValueError:
            continue
    return len(numbers)


def _new_state(args):
    profile = load_profile(args.resource_profile)
    if args.cpu_capacity is not None:
        profile["resource_capacities"]["cpu"] = args.cpu_capacity
    if args.accelerator_capacity is not None:
        profile["resource_capacities"]["accelerator"] = (
            args.accelerator_capacity)
    requested = _jobs(args)
    for task, _run in requested:
        task_requests(task, profile)
    prefix = args.prefix if args.prefix is not None else f"{args.name}-"
    jobs = []
    for task, run in requested:
        run_dir = legacy.run_dir_for(
            task, run, args.agent, args.model, args.effort, prefix)
        jobs.append({
            "id": f"{task}:r{run}", "task": task, "run": run,
            "run_dir": str(run_dir), "status": "pending",
            "active_seconds": 0.0, "launches": 0,
            "last_submission": -1, "pid": None, "pgid": None,
            "launch_started": None, "launch_active_base": None,
            "wait_log": None,
        })
    created = now()
    return {
        "format": FORMAT, "name": args.name, "status": "created",
        "created_ts": round(created, 3), "updated_ts": round(created, 3),
        "controller_pid": None, "pause_requested": False,
        "last_error": None,
        "config": {
            "campaign_name": args.name,
            "agent_concurrency": args.agent_concurrency,
            "time_budget_seconds": args.time_budget,
            "iterations": args.iterations, "agent": args.agent,
            "model": args.model, "effort": args.effort,
            "feedback": args.feedback, "prefix": prefix,
            "codex_timeout": args.codex_timeout, "poll": args.poll,
            "resource_profile": profile,
        },
        "jobs": jobs,
    }


def _coord_dir(name):
    identity = hashlib.sha256(
        str(ROOT.resolve()).encode()).hexdigest()[:20]
    path = Path(tempfile.gettempdir()) / f"text-opt-bm-benchmark-{identity}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_active(runtime, until=None):
    until = now() if until is None else until
    wall = max(0.0, until - runtime["start"])
    queued = legacy.eval_queue_seconds(
        runtime["wait_log"], runtime["start"], until)
    return runtime["active_base"] + max(0.0, wall - queued), wall, queued


def _process_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _kill_pgid(pgid, sig):
    if not isinstance(pgid, int) or pgid < 1:
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _owned_optimizer_process(job):
    """Guard crash recovery against PID/process-group reuse."""
    pid, pgid = job.get("pid"), job.get("pgid")
    if not isinstance(pid, int) or not isinstance(pgid, int):
        return False
    try:
        if os.getpgid(pid) != pgid:
            return False
        result = subprocess.run(
            ["ps", "-ww", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5, check=False)
    except (OSError, ProcessLookupError, subprocess.SubprocessError):
        return False
    command = result.stdout.strip()
    return (result.returncode == 0 and "loop.optimize" in command
            and str(job.get("run_dir", "")) in command)


def _recover_orphans(store, state):
    """Reap process groups left by a controller crash before resuming."""
    orphans = [job for job in state["jobs"] if job["status"] == "running"]
    if not orphans:
        return state
    timestamp = now()
    owned = {job["id"]: _owned_optimizer_process(job) for job in orphans}
    for job in orphans:
        started = job.get("launch_started")
        wait_log = job.get("wait_log")
        if owned[job["id"]] and started and wait_log:
            wall = max(0.0, timestamp - float(started))
            queued = legacy.eval_queue_seconds(wait_log, float(started), timestamp)
            job["active_seconds"] = round(
                float(job.get("launch_active_base") or 0.0) +
                max(0.0, wall - queued), 4)
        if owned[job["id"]]:
            _kill_pgid(job.get("pgid"), signal.SIGTERM)
    time.sleep(0.25)
    for job in orphans:
        if owned[job["id"]]:
            _kill_pgid(job.get("pgid"), signal.SIGKILL)
        store.event("pause", job["id"], "controller crash recovery")
        job.update(status="paused", pid=None, pgid=None,
                   launch_started=None, launch_active_base=None, wait_log=None,
                   last_submission=_submission_count(job["run_dir"]) - 1)
    state["status"] = "paused"
    state["controller_pid"] = None
    state["pause_requested"] = False
    state["last_error"] = "recovered optimizer processes after controller exit"
    store.event("orphan_recovery", detail=f"{len(orphans)} jobs stopped")
    return store.replace(state)


def _launch(job_state, config, coord_dir):
    task = job_state["task"]
    run = job_state["run"]
    run_dir = Path(job_state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = iterations_done(run_dir)
    remaining = max(0, config["iterations"] - completed)
    if remaining == 0:
        return None
    stdout = open(run_dir / "launcher_stdout.txt", "a")
    cmd = [
        sys.executable, "-m", "loop.optimize", "--task", task,
        "--agent", config["agent"], "--model", config["model"],
        "--effort", config["effort"], "--feedback", config["feedback"],
        "--iterations", str(remaining),
        "--codex-timeout", str(config["codex_timeout"]),
        "--run-dir", str(run_dir),
    ]
    launch_number = int(job_state.get("launches", 0)) + 1
    wait_log = (coord_dir / "waits" / config["campaign_name"] /
                f"{task}-r{run}-launch{launch_number}.jsonl")
    wait_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        wait_log.unlink()
    except FileNotFoundError:
        pass
    profile = config["resource_profile"]
    requests = task_requests(task, profile)
    resource, units = task_request(task, profile)
    env = os.environ.copy()
    env[LOCK_DIR_ENV] = str(coord_dir / "locks")
    env[WAIT_LOG_ENV] = str(wait_log)
    env[LIMITS_ENV] = json.dumps(
        profile["resource_capacities"], sort_keys=True)
    env[REQUESTS_ENV] = json.dumps(requests, sort_keys=True)
    env[DEFERRED_MANAGED_ENV] = "1"
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT,
            env=env, start_new_session=True)
    except BaseException:
        stdout.close()
        raise
    started = now()
    return {
        "id": job_state["id"], "task": task, "run": run,
        "rd": run_dir, "proc": proc, "stdout": stdout, "pgid": proc.pid,
        "start": started, "wait_log": wait_log,
        "active_base": float(job_state.get("active_seconds", 0.0)),
        "launch_number": launch_number, "resource": resource, "units": units,
        "requests": requests,
    }


def _launch_deferred(request, config, coord_dir):
    stdout_path = CAMP_ROOT / "holdout_worker.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = open(stdout_path, "a")
    cmd = [sys.executable, "-m", "bench.deferred", "score-shard",
           request["run_dir"], str(request["n"]),
           str(legacy.DEFERRED_CACHE_ROOT), request["shard"]]
    profile = config["resource_profile"]
    requests = task_requests(request["task"], profile)
    env = os.environ.copy()
    env[LOCK_DIR_ENV] = str(coord_dir / "locks")
    env.pop(WAIT_LOG_ENV, None)
    env[LIMITS_ENV] = json.dumps(
        profile["resource_capacities"], sort_keys=True)
    env[REQUESTS_ENV] = json.dumps(requests, sort_keys=True)
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT,
            env=env, start_new_session=True)
    except BaseException:
        stdout.close()
        raise
    return {"proc": proc, "pgid": proc.pid, "stdout": stdout,
            "request": request, "start": now()}


def _checkpoint(store, runtimes, status=None, error=None):
    current = {}
    for runtime in runtimes:
        active, _wall, _queued = _runtime_active(runtime)
        current[runtime["id"]] = (runtime, active)

    def mutate(state):
        for job in state["jobs"]:
            if job["id"] not in current:
                continue
            runtime, active = current[job["id"]]
            job["active_seconds"] = round(active, 4)
            job["last_submission"] = _submission_count(job["run_dir"]) - 1
            job["pid"] = runtime["proc"].pid
            job["pgid"] = runtime["pgid"]
            job["launch_started"] = runtime["start"]
            job["launch_active_base"] = runtime["active_base"]
            job["wait_log"] = str(runtime["wait_log"])
        if status is not None:
            state["status"] = status
        if error is not None:
            state["last_error"] = error
        return state

    return store.update(mutate)


def _pause(store, runtimes, background, reason):
    # Snapshot while queued PIDs are still alive; unmatched live wait markers
    # can then be recognized and fully refunded before process-group cleanup.
    state = _checkpoint(store, runtimes, status="pause_requested")
    runtime_ids = {runtime["id"] for runtime in runtimes}
    active_by_id = {
        job["id"]: job["active_seconds"] for job in state["jobs"]
        if job["id"] in runtime_ids
    }
    for runtime in runtimes:
        active, wall, queued = _runtime_active(runtime)
        store.event("pause", runtime["id"],
                    f"active={active:.1f}s wall={wall:.1f}s "
                    f"queue_refund={queued:.1f}s")
    legacy.cleanup_processes(runtimes, background)

    def mutate(current):
        for job in current["jobs"]:
            if job["id"] in active_by_id:
                job["active_seconds"] = active_by_id[job["id"]]
                job["status"] = "paused"
                job["last_submission"] = _submission_count(job["run_dir"]) - 1
                job.update(pid=None, pgid=None, launch_started=None,
                           launch_active_base=None, wait_log=None)
            elif job["status"] == "pending":
                job["status"] = "paused"
        current["status"] = "paused"
        current["pause_requested"] = False
        current["controller_pid"] = None
        return current

    store.update(mutate)
    store.event("campaign_paused", detail=reason)


def _drain_holdouts(store, state, coord_dir, stop_requested):
    known = [Path(job["run_dir"]) for job in state["jobs"]]
    args = argparse.Namespace(deferred_cache_dir=legacy.DEFERRED_CACHE_ROOT)
    failures = {}
    while True:
        if stop_requested() or store.read().get("pause_requested"):
            return False
        request = legacy.deferred_request(known, args)
        if request is None:
            return True
        worker = _launch_deferred(request, state["config"], coord_dir)
        while worker["proc"].poll() is None:
            if stop_requested() or store.read().get("pause_requested"):
                legacy.cleanup_processes([], worker)
                return False
            time.sleep(min(1.0, state["config"]["poll"]))
        rc = worker["proc"].returncode
        legacy.cleanup_processes([], worker, grace_seconds=0.0)
        key = (request["task"], request["program_sha256"], request["shard"])
        if rc:
            failures[key] = failures.get(key, 0) + 1
            store.event("holdout_error", f"{request['task']}:{request['n']}",
                        f"shard={request['shard']} rc={rc} "
                        f"attempt={failures[key]}")
            if failures[key] >= 3:
                raise RuntimeError(
                    f"deferred holdout shard failed three times: {key}")
        else:
            store.event("holdout_shard", f"{request['task']}:{request['n']}",
                        f"shard={request['shard']}")


def run_controller(store):
    require_private_slm_operator_state_absent()
    legacy.DEFERRED_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    legacy.DEFERRED_CACHE_ROOT.chmod(0o700)
    controller_lock = store.controller()
    interrupted = False

    def request_pause(_signum=None, _frame=None):
        nonlocal interrupted
        interrupted = True

    previous_int = signal.signal(signal.SIGINT, request_pause)
    previous_term = signal.signal(signal.SIGTERM, request_pause)
    runtimes = []
    background = None
    try:
        state = store.read()
        if state.get("format") != FORMAT:
            raise RuntimeError("unsupported benchmark state format")
        # Holding controller.lock is the authoritative liveness check. The PID
        # in durable state is informational and may have been reused after a
        # crash.
        state = _recover_orphans(store, state)
        if state["status"] == "complete":
            print(f"campaign {store.name!r} is already complete")
            return 0

        budget = float(state["config"]["time_budget_seconds"])
        target_iterations = int(state["config"]["iterations"])
        for job in state["jobs"]:
            if (job["status"] != "complete" and
                    float(job["active_seconds"]) < budget and
                    iterations_done(Path(job["run_dir"])) < target_iterations):
                job["status"] = "pending"
            elif job["status"] != "complete":
                job["status"] = "complete"
            job.update(pid=None, pgid=None, launch_started=None,
                       launch_active_base=None, wait_log=None)
        state["status"] = "running"
        state["pause_requested"] = False
        state["controller_pid"] = os.getpid()
        state["last_error"] = None
        state = store.replace(state)
        store.event("campaign_resume" if any(job["launches"] for job in state["jobs"])
                    else "campaign_start",
                    detail=(f"jobs={len(state['jobs'])} "
                            f"agents={state['config']['agent_concurrency']} "
                            f"budget={budget:g}s capacities="
                            f"{state['config']['resource_profile']['resource_capacities']}"))

        active_tasks = set(runner.list_tasks())
        has_mps = any(
            job["task"] in active_tasks and
            runner.load_config(job["task"]).get("required_device") == "mps"
            for job in state["jobs"])
        campaign_phase = (exclusive_campaign_mps_phase()
                          if has_mps else nullcontext())
        coord_dir = _coord_dir(store.name)

        with campaign_phase:
            while True:
                state = store.read()
                if interrupted or state.get("pause_requested"):
                    _pause(store, runtimes, background,
                           "signal" if interrupted else "pause command")
                    return 0

                live_ids = {runtime["id"] for runtime in runtimes}
                pending = [job for job in state["jobs"]
                           if job["status"] == "pending"
                           and job["id"] not in live_ids]
                while (pending and len(runtimes) <
                       state["config"]["agent_concurrency"]):
                    job_state = pending.pop(0)
                    runtime = _launch(job_state, state["config"], coord_dir)
                    if runtime is None:
                        def mark_done(current, job_id=job_state["id"]):
                            next(job for job in current["jobs"]
                                 if job["id"] == job_id)["status"] = "complete"
                        store.update(mark_done)
                        continue
                    runtimes.append(runtime)

                    def mark_running(current, rt=runtime):
                        job = next(job for job in current["jobs"]
                                   if job["id"] == rt["id"])
                        job.update(status="running", pid=rt["proc"].pid,
                                   pgid=rt["pgid"], launch_started=rt["start"],
                                   launch_active_base=rt["active_base"],
                                   wait_log=str(rt["wait_log"]),
                                   launches=rt["launch_number"])
                    store.update(mark_running)
                    request_text = ",".join(
                        f"{name}={units}/"
                        f"{state['config']['resource_profile']['resource_capacities'][name]}"
                        for name, units in sorted(runtime["requests"].items()))
                    store.event("launch", runtime["id"],
                                f"pid={runtime['proc'].pid} "
                                f"dir={runtime['rd'].name} "
                                f"eval={request_text}")

                if not runtimes:
                    state = store.read()
                    if all(job["status"] == "complete" for job in state["jobs"]):
                        store.event("holdout_drain_start",
                                    detail="draining final incumbents")
                        if not _drain_holdouts(
                                store, state, coord_dir, lambda: interrupted):
                            _pause(store, [], None, "paused during holdout drain")
                            return 0
                        def complete(current):
                            current["status"] = "complete"
                            current["controller_pid"] = None
                        store.update(complete)
                        store.event("campaign_done",
                                    detail=f"{len(state['jobs'])} jobs finished")
                        return 0

                time.sleep(state["config"]["poll"])
                _checkpoint(store, runtimes)
                survivors = []
                fatal = None
                for runtime in runtimes:
                    rc = runtime["proc"].poll()
                    active, wall, queued = _runtime_active(runtime)
                    if rc is None and active < budget:
                        survivors.append(runtime)
                        continue
                    if rc is None:
                        # Preserve the live queue refund before cleanup.
                        legacy.cleanup_processes([runtime], None)
                        event = "timeout"
                        detail = (f"active={active:.1f}s wall={wall:.1f}s "
                                  f"queue_refund={queued:.1f}s")
                        final_status = "complete"
                    else:
                        legacy.cleanup_processes(
                            [runtime], None, grace_seconds=0.0)
                        event = "finish" if rc == 0 else "optimizer_error"
                        detail = (f"rc={rc} active={active:.1f}s wall={wall:.1f}s "
                                  f"queue_refund={queued:.1f}s")
                        final_status = "complete" if rc == 0 else "paused"
                        if rc != 0:
                            fatal = (f"optimizer {runtime['id']} exited "
                                     f"with status {rc}")

                    def mark_final(current, rt=runtime, measured=active,
                                   job_status=final_status):
                        job = next(job for job in current["jobs"]
                                   if job["id"] == rt["id"])
                        job.update(status=job_status,
                                   active_seconds=round(measured, 4), pid=None,
                                   pgid=None, launch_started=None,
                                   launch_active_base=None, wait_log=None,
                                   last_submission=(
                                       _submission_count(job["run_dir"]) - 1))
                    store.update(mark_final)
                    store.event(event, runtime["id"], detail)
                runtimes = survivors
                if fatal:
                    _pause(store, runtimes, background, fatal)
                    store.update(lambda current: current.update(
                        last_error=fatal) or current)
                    return 1
    except BaseException as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        if runtimes or background is not None:
            try:
                _pause(store, runtimes, background,
                       f"controller error: {error_message}")
            except BaseException:
                legacy.cleanup_processes(runtimes, background)
        store.update(lambda current: current.update(
            status="error", controller_pid=None,
            last_error=error_message) or current)
        store.event("campaign_error", detail=error_message)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        controller_lock.close()


def command_start(args):
    store = StateStore(args.name)
    if store.exists():
        raise RuntimeError(
            f"campaign {args.name!r} already exists; use resume/status")
    if (args.agent_concurrency < 1 or not math.isfinite(args.time_budget)
            or args.time_budget <= 0 or args.iterations < 1):
        raise ValueError("concurrency, budget, and iterations must be positive")
    if not math.isfinite(args.poll) or args.poll <= 0 or args.codex_timeout <= 0:
        raise ValueError("poll and codex timeout must be positive")
    if ((args.cpu_capacity is not None and args.cpu_capacity < 1) or
            (args.accelerator_capacity is not None and
             args.accelerator_capacity < 1)):
        raise ValueError("resource capacities must be positive")
    state = _new_state(args)
    store.replace(state)
    return run_controller(store)


def command_pause(args):
    store = StateStore(args.name)
    if not store.exists():
        raise RuntimeError(f"unknown campaign {args.name!r}")
    state = store.read()
    if state["status"] in ("paused", "complete"):
        print(f"campaign {args.name!r} is already {state['status']}")
        return 0
    store.update(lambda current: current.update(pause_requested=True) or current)
    store.event("pause_requested", detail=f"by pid {os.getpid()}")
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        state = store.read()
        if state["status"] in ("paused", "error", "complete"):
            print(f"campaign {args.name!r}: {state['status']}")
            return 0
        if not _process_alive(state.get("controller_pid")):
            print("controller is not live; resume will recover its process groups")
            return 0
        time.sleep(0.2)
    print("pause requested; controller has not acknowledged it yet")
    return 0


def command_status(args):
    store = StateStore(args.name)
    if not store.exists():
        raise RuntimeError(f"unknown campaign {args.name!r}")
    state = store.read()
    config = state["config"]
    print(f"campaign={args.name} status={state['status']} "
          f"controller={state.get('controller_pid')}")
    print(f"agents={config['agent_concurrency']} budget="
          f"{config['time_budget_seconds']}s capacities="
          f"{config['resource_profile']['resource_capacities']}")
    for job in state["jobs"]:
        requests = ",".join(
            f"{name}:{units}" for name, units in sorted(
                task_requests(job["task"], config["resource_profile"]).items()))
        print(f"{job['id']:<36} {job['status']:<9} "
              f"active={job['active_seconds']:8.1f}s "
              f"eval_units={requests:<20} last_submission="
              f"{job['last_submission']}")
    if state.get("last_error"):
        print(f"last_error={state['last_error']}")
    print(f"state={store.path}")
    print(f"events={store.log_path}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="create and run a campaign")
    start.add_argument("name")
    start.add_argument("--tasks", default="", help="comma-separated tasks")
    start.add_argument("--jobs", default="", help="task:rN pairs")
    start.add_argument("--runs", type=int, default=5)
    start.add_argument("--agent-concurrency", type=int, default=24)
    start.add_argument("--time-budget", type=float, default=3600,
                       help="active seconds per rollout")
    start.add_argument("--iterations", type=int, default=1000)
    start.add_argument("--agent", choices=("codex", "cursor"), default="codex")
    start.add_argument("--model", default="gpt-5.5")
    start.add_argument("--effort", default="low",
                       choices=("none", "minimal", "low", "medium", "high",
                                "xhigh"))
    start.add_argument("--feedback", choices=("full", "train-only"),
                       default="full")
    start.add_argument("--prefix", default=None)
    start.add_argument("--codex-timeout", type=int, default=900)
    start.add_argument("--poll", type=float, default=2.0)
    start.add_argument("--resource-profile", type=Path, default=PROFILE)
    start.add_argument("--cpu-capacity", type=int, default=None)
    start.add_argument("--accelerator-capacity", type=int, default=None)
    start.set_defaults(func=command_start)

    pause = commands.add_parser("pause", help="request a durable clean pause")
    pause.add_argument("name")
    pause.add_argument("--wait", type=float, default=30.0,
                       help="seconds to wait for acknowledgement")
    pause.set_defaults(func=command_pause)

    resume = commands.add_parser("resume", help="resume a saved campaign")
    resume.add_argument("name")
    resume.set_defaults(func=lambda args: run_controller(StateStore(args.name)))

    status = commands.add_parser("status", help="show durable campaign state")
    status.add_argument("name")
    status.set_defaults(func=command_status)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        sys.exit(f"run_benchmark: {exc}")


if __name__ == "__main__":
    main()
