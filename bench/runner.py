"""Evaluate a program against a task in an isolated child process.

Determinism measures:
  - minimal fixed child environment (allowlist; shell vars not inherited),
    PYTHONHASHSEED=0, UTF-8 forced, user site-packages disabled
  - no bytecode-cache writes (PYTHONDONTWRITEBYTECODE=1) and a fresh
    throwaway PYTHONPYCACHEPREFIX against stale-cache reads — pyc writing
    uses address-dependent temp names and cold/warm cache state shifts
    tracemalloc scores by tens of bytes
  - all task data generated from fixed seeds inside the evaluator
  - memory evaluators pre-warm the program's imports outside the
    measurement window (bench.eval_lib.preimport)
  - scores are allocation/instruction/byte counts, never wall-clock time
  - a generous CPU-time rlimit (not wall time) guards against runaway
    programs without introducing load-dependent flakiness
"""

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from bench.resource_lock import (configured_limits, evaluation_slots,
                                 record_wait_interval)
from bench.slm_mps_lock import (operator_mps_phase,
                                require_canonical_mps_lock_identity)

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def list_tasks():
    result = []
    for path in TASKS_DIR.iterdir():
        if not (path / "evaluate.py").exists():
            continue
        try:
            config = json.loads((path / "config.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not config.get("retired", False):
            result.append(path.name)
    return sorted(result)


def default_tasks():
    """Tasks runnable in the base environment (exclude optional ML models)."""
    return [task for task in list_tasks()
            if not load_config(task).get("optional", False)]


def task_dir(task):
    d = TASKS_DIR / task
    if not (d / "evaluate.py").exists():
        raise ValueError(f"unknown task: {task!r} (available: {list_tasks()})")
    return d


def load_config(task):
    return json.loads((task_dir(task) / "config.json").read_text())


def read_spec(task):
    return (task_dir(task) / "spec.md").read_text()


def initial_program(task):
    return task_dir(task) / "initial_program.py"


def evaluate(task, program_path, python=None, final=False, train_only=False,
             test_only=False, test_shard=None,
             evaluation_priority="foreground", development_profile=None,
             calibration_size=None, device=None):
    """Run the task's evaluator on program_path. Returns the result dict.

    For generalization tasks (config ``kind == "generalization"``),
    ``final=True`` asks the evaluator for held-out test scores (which must
    never be exposed to the optimizing agent).  ``train_only=True`` selects
    a visible training objective only on tasks that define one.  Some tasks,
    notably SLM compression, expose calibration data but deliberately define
    no train score: their ordinary online score is validation-only.
    """
    cfg = load_config(task)
    # The interpreter is sys.executable (the process running the harness) or
    # an explicit caller argument — never taken from the environment. An
    # env-configurable interpreter would let an agent point scoring at a
    # fake `python` that prints any score; to run under a different
    # interpreter, launch the harness with it (`python3.13 -m bench ...`).
    python = python or sys.executable
    cpu_s = cfg.get("cpu_s", 120)
    wall_s = cfg.get("timeout_s", 600)
    evaluation_resource = cfg.get("evaluation_resource", "cpu")
    required_device = cfg.get("required_device")

    if final and test_only:
        raise ValueError("final and test_only are mutually exclusive")
    if test_shard is not None and not test_only:
        raise ValueError("test_shard requires test_only=True")
    if test_only and not cfg.get("deferred_test", False):
        raise ValueError(f"task {task!r} does not support deferred test scoring")
    if development_profile is None:
        development_profile = cfg.get("development_profile")
    if development_profile not in (None, "mixed", "full"):
        raise ValueError("development_profile must be mixed or full")
    if calibration_size not in (None, 32, 64, 128):
        raise ValueError("calibration_size must be 32, 64, or 128")
    if device not in (None, "auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if required_device is not None:
        if required_device not in ("cpu", "cuda", "mps"):
            raise ValueError(
                f"task {task!r} has invalid required_device {required_device!r}")
        if device in (None, "auto"):
            device = required_device
        elif device != required_device:
            raise ValueError(
                f"task {task!r} requires device={required_device}; "
                f"refusing device={device}")

    cmd = [python, str(task_dir(task) / "evaluate.py"), str(Path(program_path).resolve())]
    if final:
        cmd.append("--final")
    if train_only:
        cmd.append("--train-only")
    if test_only:
        cmd.append("--test-only")
    if test_shard is not None:
        cmd.extend(("--test-shard", str(test_shard)))
    if development_profile is not None:
        cmd.extend(("--development-profile", development_profile))
    if calibration_size is not None:
        cmd.extend(("--calibration-size", str(calibration_size)))
    if device is not None:
        cmd.extend(("--device", device))

    def set_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 10))

    with tempfile.TemporaryDirectory() as tmp, ExitStack() as local_phases:
        # Minimal, fixed child environment. Inherited shell variables and
        # bytecode-cache writes both perturb interpreter state ahead of
        # the measurement window enough to move tracemalloc scores by tens
        # of bytes (pyc writing even uses address-dependent temp names, so
        # it can differ between two runs of the same command). The child
        # therefore gets an allowlist env, never writes .pyc files, and
        # points any stale-cache *reads* at a fresh empty prefix.
        # Allowlist only: no TEXTOPT_* passthrough (no evaluator reads such
        # vars, and forwarding agent-set env into the scorer is needless
        # attack surface). PATH/HOME/TMPDIR are what the interpreter needs.
        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "HOME", "TMPDIR")}
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPYCACHEPREFIX"] = str(Path(tmp) / "pyc")
        # Canonical SLM evaluation is MPS-only.  Setting this even for CPU
        # tasks is harmless and prevents PyTorch from silently moving an
        # unsupported MPS operator onto CPU in evaluator children.
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        # Result protocol: the evaluator prefixes its one result line with
        # this nonce and we accept only a nonce-prefixed line, and it
        # os._exit()s right after emitting. This defeats CASUAL forgery
        # (stray prints, atexit-appended lines). It is NOT unforgeable: a
        # candidate that escapes to os (via string-hidden attribute access
        # the AST scan can't catch) can read this nonce from the env and
        # forge — the determined-adversary class that is out of scope under
        # the cooperative threat model (see bench/eval_lib.py).
        nonce = "%016x" % int.from_bytes(os.urandom(8), "big")
        env["TEXTOPT_RESULT_NONCE"] = nonce
        # A direct/preflight/operator evaluation launched outside the campaign
        # also participates in phase separation. Campaign optimizer/deferred
        # parents carry the trusted resource-gate environment and are already
        # covered by the launcher's exclusive campaign phase lease.
        if required_device == "mps" and configured_limits() is None:
            local_phases.enter_context(
                operator_mps_phase(f"operator-eval:{task}"))
        # Measure the evaluation's LOCAL cost: wall time, and the child's
        # CPU time via a RUSAGE_CHILDREN delta. CPU time is the rescale
        # basis (bench/trace.py) because, unlike wall, it is not inflated
        # when several graders run at once — it counts cycles actually used.
        # Wait for a resource slot only at the scoring boundary. The outer
        # optimization loop remains live while other loops are evaluating.
        with evaluation_slots(evaluation_resource,
                              priority=evaluation_priority) as eval_queue:
            ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=tmp,
                    timeout=wall_s,
                    preexec_fn=set_limits,
                )
            except subprocess.TimeoutExpired:
                # Attach the elapsed wall so a timed-out grading is recorded
                # with its true local cost rather than as instantaneous.
                return _err(
                    f"wall-clock timeout after {wall_s}s (safety guard)",
                    {"eval_wall_seconds": round(time.monotonic() - t0, 4),
                     "eval_cpu_seconds": None,
                     "eval_queue_seconds": round(eval_queue, 4)},
                    evaluator_completed=True,
                    failure_kind="timeout",
                )
            eval_wall = time.monotonic() - t0
            ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            eval_cpu = ((ru1.ru_utime - ru0.ru_utime)
                        + (ru1.ru_stime - ru0.ru_stime))

    children_cpu = round(max(0.0, eval_cpu), 4)

    prefix = nonce + " "
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        # Only a line the evaluator emitted (carrying the secret nonce) is
        # a valid result; anything a candidate wrote to real stdout lacks it.
        if line.startswith(prefix):
            try:
                result = json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                continue
            # Prefer the child's own RUSAGE_SELF (per-process accurate) over
            # the parent's process-global RUSAGE_CHILDREN delta, which would
            # misattribute CPU if a caller ran evaluations concurrently.
            self_cpu = result.pop("eval_self_cpu_seconds", None)
            inner_mps_wait = 0.0
            if required_device == "mps" and result.get("ok"):
                lock_record = (result.get("metrics") or {}).get(
                    "exclusive_mps_lock")
                try:
                    require_canonical_mps_lock_identity(
                        lock_record, "ranked evaluator MPS lock")
                    wait_started = float(lock_record["wait_started_unix"])
                    acquired = float(lock_record["acquired_unix"])
                    declared_wait = float(lock_record["wait_seconds"])
                    inner_mps_wait = acquired - wait_started
                    if (inner_mps_wait < 0 or declared_wait < 0 or
                            abs(inner_mps_wait - declared_wait) > 0.25 or
                            inner_mps_wait > eval_wall + 0.25):
                        raise RuntimeError("MPS lock wait telemetry is inconsistent")
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    return _err(
                        f"canonical MPS evaluator provenance is invalid: {exc}",
                        {"eval_wall_seconds": round(eval_wall, 4),
                         "eval_cpu_seconds": children_cpu,
                         "eval_queue_seconds": round(eval_queue, 4)},
                        evaluator_completed=False,
                        failure_kind="infrastructure")
                record_wait_interval(
                    wait_started, acquired, category="slm-mps-lock")
            result["eval_wall_seconds"] = round(eval_wall, 4)
            result["eval_cpu_seconds"] = (round(self_cpu, 4)
                                          if self_cpu is not None
                                          else children_cpu)
            result["eval_queue_seconds"] = round(
                eval_queue + inner_mps_wait, 4)
            # A nonce-authenticated evaluator payload is a completed benchmark
            # outcome even when candidate validation failed. Deferred scoring
            # may cache that deterministic failure. This differs from a child
            # crash/no-result, which remains retryable infrastructure failure.
            result["evaluator_completed"] = True
            result["failure_kind"] = (
                None if result.get("ok") else "candidate")
            return result

    timing = {"eval_wall_seconds": round(eval_wall, 4),
              "eval_cpu_seconds": children_cpu,
              "eval_queue_seconds": round(eval_queue, 4)}

    if proc.returncode == -signal.SIGXCPU:
        return _err(
            f"CPU time limit exceeded ({cpu_s}s of CPU time)", timing,
            evaluator_completed=True, failure_kind="cpu_limit")
    stderr_tail = (proc.stderr or "")[-2000:]
    return _err(
        f"evaluator produced no result (exit code {proc.returncode}); "
        f"stderr tail:\n{stderr_tail}",
        timing, evaluator_completed=False, failure_kind="infrastructure",
    )


def _err(msg, timing=None, evaluator_completed=False,
         failure_kind="infrastructure"):
    r = {"ok": False, "score": None, "metrics": {}, "error": msg}
    if timing:
        r.update(timing)
    r["evaluator_completed"] = bool(evaluator_completed)
    r["failure_kind"] = failure_kind
    return r
