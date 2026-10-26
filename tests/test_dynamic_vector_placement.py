"""Focused protocol and quality checks for dynamic_vector_placement."""

import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import heldout, runner
from bench.tasks.dynamic_vector_placement import evaluate


def test_generator_is_deterministic_and_split_disjoint():
    first = evaluate.generate_workload("mixed", 12345, n=40)
    assert first == evaluate.generate_workload("mixed", 12345, n=40)
    assert first != evaluate.generate_workload("mixed", 12346, n=40)
    data = ROOT / "bench/tasks/dynamic_vector_placement/data"
    splits = (
        json.loads((data / "train.json").read_text()),
        heldout.read(data / "heldout_val.bin"),
        heldout.read(data / "heldout_test.bin"),
    )
    seeds = [seed for members in splits for _, seed in members]
    assert len(seeds) == len(set(seeds))
    assert all(len(evaluate.generate_workload(regime, seed)) == 720
               for members in splits for regime, seed in members)


def test_starter_is_valid_deterministic_and_reference_has_headroom():
    starter = runner.evaluate(
        "dynamic_vector_placement",
        ROOT / "bench/tasks/dynamic_vector_placement/initial_program.py")
    repeated = runner.evaluate(
        "dynamic_vector_placement",
        ROOT / "bench/tasks/dynamic_vector_placement/initial_program.py")
    reference = runner.evaluate(
        "dynamic_vector_placement",
        ROOT / "tests/solutions/dynamic_vector_placement.py")
    assert starter["ok"], starter["error"]
    assert repeated["ok"], repeated["error"]
    assert reference["ok"], reference["error"]
    assert starter["score"] == repeated["score"]
    assert reference["score"] < starter["score"] * 0.93
    assert not any(key.startswith("test_") for key in starter["metrics"])
    assert starter["metrics"]["future_lifetimes_visible"] is False


def test_invalid_returns_and_infeasible_placement_fail():
    for name in (
        "dynamic_vector_placement_invalid_id.py",
        "dynamic_vector_placement_overfill.py",
        "dynamic_vector_placement_bad_return.py",
        "dynamic_vector_placement_cheat_file.py",
    ):
        result = runner.evaluate(
            "dynamic_vector_placement", ROOT / "tests/broken" / name)
        assert not result["ok"], (name, result)


def test_final_mode_seals_test_metrics_from_online_evaluation():
    evaluator = ROOT / "bench/tasks/dynamic_vector_placement/evaluate.py"
    program = ROOT / "tests/solutions/dynamic_vector_placement.py"
    completed = subprocess.run(
        [sys.executable, str(evaluator), str(program), "--final"],
        text=True, capture_output=True, check=True)
    # Direct evaluator output may have a harness nonce prefix only under runner.
    assert '"test_score"' in completed.stdout

    deferred = runner.evaluate(
        "dynamic_vector_placement", program, test_only=True, test_shard="full")
    assert deferred["ok"], deferred["error"]
    assert deferred["metrics"]["deferred_test_shard"] == "full"


def main():
    test_generator_is_deterministic_and_split_disjoint()
    test_starter_is_valid_deterministic_and_reference_has_headroom()
    test_invalid_returns_and_infeasible_placement_fail()
    test_final_mode_seals_test_metrics_from_online_evaluation()
    print("dynamic vector placement checks passed")


if __name__ == "__main__":
    main()
