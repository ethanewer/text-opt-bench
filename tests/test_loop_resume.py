"""A paused optimizer resumes its session without a baseline re-grade."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import runner  # noqa: E402
from loop import optimize  # noqa: E402


def main():
    old_tasks = runner.TASKS_DIR
    old_agent = optimize.run_agent
    old_profile = optimize.calibrate.machine_profile
    old_argv = sys.argv
    with tempfile.TemporaryDirectory(prefix="loop-resume-test-") as raw:
        root = Path(raw)
        tasks = root / "tasks"
        task = tasks / "fixture"
        task.mkdir(parents=True)
        (task / "config.json").write_text(json.dumps({
            "name": "fixture", "kind": "perfect", "metric": "fixture",
            "direction": "min", "cpu_s": 5, "timeout_s": 5,
        }))
        (task / "spec.md").write_text("Keep program.py valid.\n")
        (task / "initial_program.py").write_text("VALUE = 1\n")
        (task / "evaluate.py").write_text(
            "from bench import eval_lib\n"
            "eval_lib.succeed(1.0, {'fixture': 1})\n")
        run_dir = root / "run"
        try:
            runner.TASKS_DIR = tasks
            optimize.run_agent = lambda *_args, **_kwargs: None
            optimize.calibrate.machine_profile = lambda: {"rate": 1}
            base_argv = [
                "optimize.py", "--task", "fixture", "--iterations", "1",
                "--run-dir", str(run_dir), "--codex-timeout", "1",
            ]
            sys.argv = base_argv
            optimize.main()

            submissions = (run_dir / "submissions.jsonl").read_text().splitlines()
            assert len(submissions) == 1, "first run should submit one baseline"
            # Simulate both interruption windows repaired by resume: a stale
            # derived best cache and the next workspace cloned only partially.
            (run_dir / "best_program.py").write_text("CORRUPT = True\n")
            orphan = run_dir / "iter_002"
            orphan.mkdir()
            (orphan / "program.py").write_text("PARTIAL = True\n")

            sys.argv = base_argv
            optimize.main()

            submissions = (run_dir / "submissions.jsonl").read_text().splitlines()
            assert len(submissions) == 1, (
                "resume re-graded and re-submitted the baseline")
            events = [json.loads(line) for line in
                      (run_dir / "log.jsonl").read_text().splitlines()]
            assert sum(event.get("event") == "baseline" for event in events) == 1
            assert sum(event.get("event") == "resume" for event in events) == 1
            assert (run_dir / "best_program.py").read_text() == "VALUE = 1\n"
            assert (orphan / "program.py").read_text() == "VALUE = 1\n"
        finally:
            runner.TASKS_DIR = old_tasks
            optimize.run_agent = old_agent
            optimize.calibrate.machine_profile = old_profile
            sys.argv = old_argv
    print("loop resume checks passed")


if __name__ == "__main__":
    main()
