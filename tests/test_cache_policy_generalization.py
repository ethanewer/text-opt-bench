"""Focused checks for the lightweight cache-policy task."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import runner
from bench.tasks.cache_policy_generalization import evaluate


def test_trace_families_are_equal_length_and_deterministic():
    for regime in range(5):
        first = evaluate.make_trace(7100 + regime, regime)
        second = evaluate.make_trace(7100 + regime, regime)
        assert first == second
        assert len(first) == evaluate.TRACE_LENGTH
        assert all(len(request) == 3 for request in first)


def test_initial_and_reference_are_deterministic_with_headroom():
    initial = runner.evaluate(
        "cache_policy_generalization",
        ROOT / "bench/tasks/cache_policy_generalization/initial_program.py")
    repeated = runner.evaluate(
        "cache_policy_generalization",
        ROOT / "bench/tasks/cache_policy_generalization/initial_program.py")
    reference = runner.evaluate(
        "cache_policy_generalization",
        ROOT / "tests/solutions/cache_policy_generalization.py")
    assert initial["ok"], initial
    assert repeated["ok"], repeated
    assert reference["ok"], reference
    assert initial["score"] == repeated["score"]
    assert reference["score"] < initial["score"] * 0.95


def test_reference_headroom_transfers_to_every_sealed_split():
    initial = runner.evaluate(
        "cache_policy_generalization",
        ROOT / "bench/tasks/cache_policy_generalization/initial_program.py",
        final=True)
    reference = runner.evaluate(
        "cache_policy_generalization",
        ROOT / "tests/solutions/cache_policy_generalization.py",
        final=True)
    assert initial["ok"] and reference["ok"]
    for split in ("train", "val", "test"):
        assert (reference["metrics"][split + "_score"]
                < initial["metrics"][split + "_score"] * 0.95)


def test_invalid_policies_fail_closed():
    for name in (
        "cache_policy_generalization_unknown_victim.py",
        "cache_policy_generalization_no_space.py",
        "cache_policy_generalization_hidden_state.py",
        "cache_policy_generalization_annotation_state.py",
    ):
        result = runner.evaluate(
            "cache_policy_generalization", ROOT / "tests/broken" / name)
        assert not result["ok"], (name, result)


def test_opaque_ids_and_no_length_fingerprint():
    traces = [evaluate.make_trace(8100 + regime, regime) for regime in range(5)]
    assert {len(trace) for trace in traces} == {evaluate.TRACE_LENGTH}
    first_ids = [trace[0][0] for trace in traces]
    assert len(set(first_ids)) == len(first_ids)
    assert all(object_id >= 10_000 for object_id in first_ids)


if __name__ == "__main__":
    test_trace_families_are_equal_length_and_deterministic()
    test_initial_and_reference_are_deterministic_with_headroom()
    test_reference_headroom_transfers_to_every_sealed_split()
    test_invalid_policies_fail_closed()
    test_opaque_ids_and_no_length_fingerprint()
    print("cache policy generalization checks passed")
