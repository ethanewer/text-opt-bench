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
    python = python or os.environ.get("TEXTOPT_PYTHON", sys.executable)
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
        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "HOME", "TMPDIR") or k.startswith("TEXTOPT_")}
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPYCACHEPREFIX"] = str(Path(tmp) / "pyc")
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
            return _err(f"wall-clock timeout after {wall_s}s (safety guard)")

    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    if proc.returncode == -signal.SIGXCPU:
        return _err(f"CPU time limit exceeded ({cpu_s}s of CPU time)")
    stderr_tail = (proc.stderr or "")[-2000:]
    return _err(
        f"evaluator produced no result (exit code {proc.returncode}); "
        f"stderr tail:\n{stderr_tail}"
    )


def _err(msg):
    return {"ok": False, "score": None, "metrics": {}, "error": msg}
