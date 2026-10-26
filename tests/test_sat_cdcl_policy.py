"""Focused checks for the lightweight SAT-policy task."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import runner

EVALUATE = ROOT / "bench/tasks/sat_cdcl_policy/evaluate.py"


def _module():
    spec = importlib.util.spec_from_file_location("sat_policy_eval_test", EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formula_constructions_and_solver_correctness():
    module = _module()
    candidate = ROOT / "tests/solutions/sat_cdcl_policy.py"
    cases = [
        ["random_3sat", 101, 20, 4.1, 60_000],
        ["coloring", 103, 10, 0.5, 60_000],
        ["pigeonhole", 105, 6, 0, 300_000],
        ["xor", 107, 16, 0.7, 100_000],
        ["xor", 108, 16, 0.7, 100_000],
    ]
    result = module.score_split(candidate, cases)
    assert result["solved"] == len(cases)
    assert 0 <= result["score"] < 1.25


def test_runner_determinism_and_reference_headroom():
    baseline = runner.evaluate("sat_cdcl_policy", runner.initial_program("sat_cdcl_policy"))
    repeated = runner.evaluate("sat_cdcl_policy", runner.initial_program("sat_cdcl_policy"))
    reference = runner.evaluate("sat_cdcl_policy", ROOT / "tests/solutions/sat_cdcl_policy.py")
    assert baseline["ok"], baseline["error"]
    assert repeated["ok"], repeated["error"]
    assert reference["ok"], reference["error"]
    assert baseline["score"] == repeated["score"]
    assert reference["score"] < baseline["score"] * 0.85
    assert not any(key.startswith("test_") for key in baseline["metrics"])


def test_invalid_policy_outputs_are_rejected():
    for name in ("sat_cdcl_policy_bad_branch.py", "sat_cdcl_policy_bad_restart.py"):
        result = runner.evaluate("sat_cdcl_policy", ROOT / "tests/broken" / name)
        assert not result["ok"], (name, result)


if __name__ == "__main__":
    test_formula_constructions_and_solver_correctness()
    test_runner_determinism_and_reference_headroom()
    test_invalid_policy_outputs_are_rejected()
    print("sat_cdcl_policy checks passed")
