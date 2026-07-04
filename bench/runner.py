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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def list_tasks():
    return sorted(
        p.name for p in TASKS_DIR.iterdir() if (p / "evaluate.py").exists()
    )


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


def evaluate(task, program_path, python=None, final=False, train_only=False):
    """Run the task's evaluator on program_path. Returns the result dict.

    For generalization tasks (config "kind" == "generalization"), the
    default result contains train/val scores only; final=True passes
    --final so the evaluator also reports the held-out test score (never
    expose final results to the optimizing agent), train_only=True passes
    --train-only so it reports the train score only (blind mode).
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

    cmd = [python, str(task_dir(task) / "evaluate.py"), str(Path(program_path).resolve())]
    if final:
        cmd.append("--final")
    if train_only:
        cmd.append("--train-only")

    def set_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 10))

    with tempfile.TemporaryDirectory() as tmp:
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
        # Measure the evaluation's LOCAL cost: wall time, and the child's
        # CPU time via a RUSAGE_CHILDREN delta. CPU time is the rescale
        # basis (bench/trace.py) because, unlike wall, it is not inflated
        # when several graders run at once — it counts cycles actually used.
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
            # Attach the elapsed wall so a timed-out grading is recorded with
            # its true (large) local cost rather than as instantaneous.
            return _err(f"wall-clock timeout after {wall_s}s (safety guard)",
                        {"eval_wall_seconds": round(time.monotonic() - t0, 4),
                         "eval_cpu_seconds": None})
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
            result["eval_wall_seconds"] = round(eval_wall, 4)
            result["eval_cpu_seconds"] = (round(self_cpu, 4)
                                          if self_cpu is not None
                                          else children_cpu)
            return result

    timing = {"eval_wall_seconds": round(eval_wall, 4),
              "eval_cpu_seconds": children_cpu}

    if proc.returncode == -signal.SIGXCPU:
        return _err(f"CPU time limit exceeded ({cpu_s}s of CPU time)", timing)
    stderr_tail = (proc.stderr or "")[-2000:]
    return _err(
        f"evaluator produced no result (exit code {proc.returncode}); "
        f"stderr tail:\n{stderr_tail}",
        timing,
    )


def _err(msg, timing=None):
    r = {"ok": False, "score": None, "metrics": {}, "error": msg}
    if timing:
        r.update(timing)
    return r
