"""Integration checks for the revised ML benchmark tasks."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import runner
from bench.session import visible_metrics
from bench.tasks.optimizer_generalization import evaluate as optimizer_eval
from bench.slm_sft import candidate_activation_stats
from bench.lfm25_model_identity import expected_files

CPU_TASKS = ("llm_routing", "optimizer_generalization")
MODEL_TASKS = ("slm_compression_3_5bpw", "slm_compression_4_5bpw")
MODEL_TARGETS = {
    "slm_compression_3_5bpw": 3.5,
    "slm_compression_4_5bpw": 4.5,
}
SOLUTIONS = {
    "llm_routing": "llm_routing.py",
    "optimizer_generalization": "optimizer_generalization.py",
    "slm_compression_3_5bpw": None,
    "slm_compression_4_5bpw": None,
}
RETIRED = (
    "gradient_compression", "hpo_taskset", "kv_cache_policy",
    "kv_prefill_compression", "optimizer_synthesis", "slm_compression",
    "slm_compression_qwen35",
    "slm_weight_compression_qwen35",
)


class OptimizerProtocolProbe:
    def init(self, parameter_shapes):
        return [0]

    def update(self, parameter_blocks, gradient_blocks, state, step):
        assert state == [0], "view mutated live optimizer state"
        return [parameter_blocks, state]

    def view(self, parameter_blocks, state, step):
        assert step > 0, "evaluator exposed the initial denominator to view"
        state[0] = 1
        return parameter_blocks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rms_source, max_source = [1.0, 2.0], [3.0, 4.0]
    rms_view, max_view = candidate_activation_stats(
        {"rms": rms_source, "max": max_source})
    assert isinstance(rms_view, tuple) and isinstance(max_view, tuple)
    try:
        rms_view[0] = 99.0
        raise AssertionError("candidate activation stats remained mutable")
    except TypeError:
        pass
    assert rms_source == [1.0, 2.0] and max_source == [3.0, 4.0]
    first_optimizer_task = optimizer_eval._read(
        ROOT / "bench/tasks/optimizer_generalization/data/train.json")[0]
    probe = optimizer_eval.run_task(OptimizerProtocolProbe(), first_optimizer_task)
    assert abs(probe["auc"] - 1.0) < 1e-12
    loop_source = (ROOT / "loop/optimize.py").read_text()
    assert "full_result(session.best)" not in loop_source
    probe = {"train_score": 1, "val_nll_delta": 2,
             "test_nll_delta": 3, "test_perplexity": 4}
    assert "test_nll_delta" not in visible_metrics(probe, "full")
    assert "test_perplexity" not in visible_metrics(probe, "full")
    assert "val_nll_delta" in visible_metrics(probe, "full")
    assert "val_nll_delta" not in visible_metrics(probe, "train-only")
    manifest = json.loads((ROOT / "bench/tasks/ml_assets.json").read_text())
    assert manifest.get("format") == 3
    assert tuple(manifest.get("suite", ())) == CPU_TASKS + MODEL_TASKS
    active = runner.list_tasks()
    for task in CPU_TASKS:
        config = runner.load_config(task)
        assert config["deferred_test"] is True
        assert config["deferred_aggregation"] == "single_shard"
        assert config["test_shards"] == ["full"]
    for task in MODEL_TASKS:
        config = runner.load_config(task)
        assert config["online_objective"] == "validation"
        assert config["required_device"] == "mps"
        assert config["supported_devices"] == ["mps", "cuda"]
        assert config["canonical_device"] == "mps"
        assert config["canonical_devices"] == ["mps", "cuda"]
        assert config["mps_fallback_allowed"] is False
        assert config["calibration_conversations"] == 128
        assert config["validation_examples"] == 100
        assert config["calibration_conversations_scored"] == 0
        assert config["feedback_modes"] == ["full"]
        assert config["scoring_inference_dtype"] == "bfloat16"
        assert config["deferred_aggregation"] == "lfm_behavior_single_shard"
        assert config["test_shards"] == ["lfm25@regression"]
        attestation = json.loads((
            ROOT / "bench/tasks" / task / "data/model_attestation.json"
        ).read_text())
        assert attestation["files"] == expected_files()
        assert (f"bench/tasks/{task}/model_identity.py"
                in config["fingerprint_code"])
        assert "bench/lfm25_model_identity.py" in config["fingerprint_code"]
        assert "bench/deferred.py" in config["fingerprint_code"]
    for task in RETIRED:
        assert task not in active
        config = runner.load_config(task)
        assert config.get("retired") is True
        assert config.get("retired_reason")
    for relative, expected in manifest["artifacts"].items():
        path = ROOT / relative
        assert path.exists() and sha(path) == expected, relative
    tasks = CPU_TASKS + (MODEL_TASKS if os.environ.get("TEXTOPT_TEST_MODELS") else ())
    for task in tasks:
        baseline = runner.evaluate(task, runner.initial_program(task))
        solution = (baseline if SOLUTIONS[task] is None else runner.evaluate(
            task, ROOT / "tests/solutions" / SOLUTIONS[task]))
        assert baseline["ok"], (task, baseline["error"])
        assert solution["ok"], (task, solution["error"])
        assert not any(key.startswith("test_")
                       for key in baseline.get("metrics", {})), task
        assert not any(key.startswith("test_")
                       for key in solution.get("metrics", {})), task
        if task in MODEL_TASKS:
            metrics = baseline["metrics"]
            assert not any(key.startswith("train_") for key in metrics)
            assert "val_tracks" not in metrics
            assert metrics["examples_per_dataset"] == 20
            assert set(metrics["dataset_regression_rates"]) == {
                "gpqa", "ifbench", "bfcl", "gsm8k", "mmlupro"}
            assert metrics["target_bpw"] == MODEL_TARGETS[task]
            assert metrics["compression_device"] == "mps"
            assert metrics["canonical_device"] == "mps"
            assert metrics["calibration_backend"] == "mps"
            assert metrics["mps_fallback_enabled"] is False
            assert abs(baseline["score"] - metrics["val_score"]) <= 1e-8
        if SOLUTIONS[task] is not None:
            assert solution["score"] < baseline["score"], (
                task, baseline["score"], solution["score"])
        repeated = runner.evaluate(task, runner.initial_program(task))
        tolerance = runner.load_config(task).get("score_tolerance", 0)
        assert abs(repeated["score"] - baseline["score"]) <= tolerance, task
        print(f"[PASS] {task}: {baseline['score']:g} -> {solution['score']:g}")
    print("all ML task checks passed")


if __name__ == "__main__":
    main()
