"""Fail-fast readiness check for the revised ML tasks."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import runner
from bench.ml_models import mps_fallback_enabled

TASKS = ("llm_routing", "optimizer_generalization",
         "slm_weight_compression_lfm25")
MODEL_TASKS = ("slm_weight_compression_lfm25",)
SLM_PROTOCOL_VERSIONS = {
    "slm_weight_compression_lfm25": 3,
}
RETIRED = (
    "gradient_compression", "hpo_taskset", "kv_cache_policy",
    "kv_prefill_compression", "optimizer_synthesis", "slm_compression",
    "slm_compression_qwen35",
    "slm_weight_compression_qwen35",
)
EXPECTED_VERSIONS = {
    "numpy": "2.5.1", "torch": "2.13.0", "transformers": "5.2.0",
    "jax": "0.10.2", "jaxlib": "0.10.2",
    "safetensors": "0.8.0", "pandas": "2.3.3", "scipy": "1.18.0",
    "sklearn": "1.9.0", "huggingface_hub": "1.23.0",
}
DISTRIBUTIONS = {
    "numpy": "numpy", "torch": "torch", "transformers": "transformers",
    "jax": "jax", "jaxlib": "jaxlib",
    "safetensors": "safetensors", "pandas": "pandas", "scipy": "scipy",
    "sklearn": "scikit-learn", "huggingface_hub": "huggingface-hub",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin_mps_fallback_before_import():
    """Return whether fallback was requested, then pin the safe import value."""
    inherited = mps_fallback_enabled()
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    return inherited


def optimizer_jax_backend():
    """Inspect JAX in a fresh process so preflight never forks after JAX.

    JAX owns background threads after import. Importing it in this parent and
    then invoking the ordinary evaluator subprocess path causes Python's
    multithreaded-fork warning on macOS and is not a robust readiness check.
    """
    source = (
        "import json; "
        "from bench.tasks.optimizer_generalization import real_workloads_jax; "
        "print(json.dumps(real_workloads_jax.backend()))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], cwd=ROOT, capture_output=True,
        text=True, env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=60,
    )
    if completed.returncode:
        raise RuntimeError(
            "CPU JAX backend probe failed: " + completed.stderr.strip())
    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true",
                        help=("also score the online objective for every baseline "
                              "(loads the active LFM checkpoint)"))
    args = parser.parse_args()
    errors = []
    # Dependency checks below include PyTorch. Reject an unsafe inherited
    # request, then pin the disabled value before even metadata/preflight code
    # has an opportunity to import torch and latch operator fallback.
    inherited_fallback = pin_mps_fallback_before_import()
    if inherited_fallback:
        errors.append(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; canonical SLM "
            "evaluation requires it to be disabled")
    manifest_path = ROOT / "bench/tasks/ml_assets.json"
    if not manifest_path.exists():
        errors.append("missing ml_assets.json; run tools/prepare_ml_benchmark.py")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != 3:
            errors.append("ml_assets.json is not format 3; rerun preparation")
        if tuple(manifest.get("suite", ())) != TASKS:
            errors.append("ml_assets.json suite does not match the active ML tasks")
    for relative, expected in manifest.get("artifacts", {}).items():
        path = ROOT / relative
        if not path.exists():
            errors.append(f"missing artifact: {relative}")
        elif digest(path) != expected:
            errors.append(f"artifact checksum mismatch: {relative}")
    for task in TASKS:
        try:
            cfg = runner.load_config(task)
            for name in ("spec.md", "initial_program.py", "evaluate.py"):
                if not (runner.task_dir(task) / name).exists():
                    errors.append(f"{task}: missing {name}")
            expected = "accelerator" if task in MODEL_TASKS else "cpu"
            if cfg.get("evaluation_resource", "cpu") != expected:
                errors.append(f"{task}: wrong evaluation_resource")
            if task == "llm_routing":
                if (cfg.get("protocol_version") != 7 or
                        cfg.get("benchmark_status") != "custom_tweaked" or
                        cfg.get("direct_paper_reproduction") is not False or
                        cfg.get(
                            "minimum_prompts_per_development_source_per_scored_split") != 16 or
                        cfg.get("deferred_test") is not True or
                        cfg.get("deferred_aggregation") != "single_shard" or
                        cfg.get("test_shards") != ["full"]):
                    errors.append(
                        f"{task}: custom-v7 protocol metadata is wrong")
                split_manifest = json.loads((
                    runner.task_dir(task) / "data/split_manifest.json"
                ).read_text())
                leakage = split_manifest.get("leakage_audit", {})
                independent = leakage.get(
                    "independent_cross_role_similarity_audit", {})
                routing_hashes = split_manifest.get("sha256", {})
                routing_data = runner.task_dir(task) / "data"
                if (split_manifest.get("task_protocol") !=
                        "llm_routing_v7_custom" or
                        leakage.get(
                            "exact_normalized_templates_crossing_roles") != 0 or
                        leakage.get(
                            "fuzzy_components_crossing_roles") != 0 or
                        independent.get(
                            "cross_role_high_similarity_pairs") != 0 or
                        leakage.get(
                            "all_scored_dataset_minimums_satisfied") is not True):
                    errors.append(
                        f"{task}: split leakage/minimum audit is invalid")
                if set(split_manifest.get("test_only_datasets", ())) != {
                        "livecodebench", "swe-bench", "tau2"}:
                    errors.append(f"{task}: test-only domain split is wrong")
                for name in ("train.json", "heldout_val.bin",
                             "heldout_test.bin", "routing_reference_choices.bin"):
                    path = routing_data / name
                    expected_hash = routing_hashes.get(name)
                    if (not path.is_file() or not isinstance(expected_hash, str)
                            or digest(path) != expected_hash):
                        errors.append(
                            f"{task}: split manifest hash is stale for {name}")
            if task == "optimizer_generalization":
                backend = optimizer_jax_backend()
                if backend.get("platforms") != ["cpu"]:
                    errors.append(f"{task}: JAX backend is not CPU-only")
                if (cfg.get("protocol_version") != 9 or
                        cfg.get("benchmark_status") !=
                        "research_candidate_real_workload_primary" or
                        cfg.get("direct_paper_reproduction") is not False or
                        cfg.get("development_real_architectures") != 5 or
                        cfg.get("sealed_test_known_real_architectures") != 5 or
                        cfg.get("sealed_test_only_real_architectures") != 3 or
                        cfg.get("development_analytic_families") != 5 or
                        cfg.get("sealed_test_only_analytic_families") != 5 or
                        cfg.get("deferred_test") is not True or
                        cfg.get("deferred_aggregation") != "single_shard" or
                        cfg.get("test_shards") != ["full"]):
                    errors.append(
                        f"{task}: expanded real-primary protocol-v9 metadata is wrong")
                optimizer_manifest = json.loads((
                    runner.task_dir(task) / "data/data_manifest.json"
                ).read_text())
                if (optimizer_manifest.get("schema") != 8 or
                        optimizer_manifest.get("protocol") != 9 or
                        optimizer_manifest.get("counts", {}).get("test") != 688 or
                        optimizer_manifest.get("real_counts", {}).get(
                            "test", {}).get("ood") != 64 or
                        optimizer_manifest.get(
                            "sealed_test_unseen_architecture_score_weight") != 0.5):
                    errors.append(
                        f"{task}: generated v9 real/analytic split metadata is wrong")
            if task in MODEL_TASKS and (
                    cfg.get("protocol_version") != SLM_PROTOCOL_VERSIONS[task] or
                    cfg.get("online_objective") != "validation" or
                    cfg.get("required_device") != "mps" or
                    cfg.get("canonical_device") != "mps" or
                    cfg.get("mps_fallback_allowed") is not False or
                    cfg.get("calibration_conversations") != 128 or
                    cfg.get("validation_conversations") != 128 or
                    cfg.get("calibration_conversations_scored") != 0 or
                    not cfg.get("fingerprint_manifest") or
                    cfg.get("require_data_fingerprint") is not True or
                    cfg.get("feedback_modes") != ["full"] or
                    cfg.get("scoring_inference_dtype") != "float32" or
                    cfg.get("target_whole_model_bits_per_parameter") != [3.5]):
                errors.append(
                    f"{task}: calibration/validation scoring contract is wrong")
            if task in MODEL_TASKS:
                from bench.tasks.slm_weight_compression_lfm25.model_identity import (
                    expected_files)
                data_dir = runner.task_dir(task) / "data"
                data_manifest = json.loads((data_dir / "data_manifest.json").read_text())
                if data_manifest.get("counts") != {
                        "calibration": 128, "validation": 128,
                        "test_id": 128, "test_ood": 128}:
                    errors.append(f"{task}: dataset split counts are wrong")
                hashes = data_manifest.get("sha256", {})
                for name in ("train.json", "heldout_val.bin",
                             "heldout_test.bin", "model_attestation.json"):
                    if hashes.get(name) != digest(data_dir / name):
                        errors.append(f"{task}: stale data hash for {name}")
                attestation = json.loads(
                    (data_dir / "model_attestation.json").read_text())
                if (attestation.get("model_id") != "LiquidAI/LFM2.5-230M" or
                        attestation.get("revision") !=
                        "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7" or
                        attestation.get("files") != expected_files()):
                    errors.append(f"{task}: model attestation identity is wrong")
        except Exception as exc:
            errors.append(f"{task}: {exc}")
    active = runner.list_tasks()
    for task in RETIRED:
        try:
            config = runner.load_config(task)
            if not config.get("retired") or not config.get("retired_reason"):
                errors.append(f"{task}: retirement metadata is incomplete")
            if task in active:
                errors.append(f"{task}: retired task remains runnable")
        except Exception as exc:
            errors.append(f"{task}: {exc}")
    versions = {}
    for module, expected in EXPECTED_VERSIONS.items():
        try:
            versions[module] = importlib.metadata.version(DISTRIBUTIONS[module])
            if versions[module] != expected:
                errors.append(f"{module}=={versions[module]} installed; need {expected}")
        except importlib.metadata.PackageNotFoundError as exc:
            errors.append(f"missing dependency {module}: {exc}")
    try:
        import torch
        if not (getattr(torch.backends, "mps", None) and
                torch.backends.mps.is_available()):
            errors.append(
                "MPS is unavailable; active SLM compilation and evaluation "
                "fail closed instead of falling back to CPU")
    except ImportError:
        # The dependency error above is already more specific.
        pass
    for model, record in manifest.get("models", {}).items():
        if not isinstance(record, dict):
            errors.append(f"{model}: manifest lacks pinned revision metadata")
            continue
        path = Path(record["path"])
        if not record.get("revision"):
            errors.append(f"{model}: missing pinned revision")
        for name, expected in record.get("files", {}).items():
            file_path = path / name
            if not file_path.exists():
                errors.append(f"{model}: missing {file_path}")
            elif digest(file_path) != expected:
                errors.append(f"{model}: checksum mismatch for {name}")
    results = {}
    if args.evaluate and not errors:
        for task in TASKS:
            result = runner.evaluate(task, runner.initial_program(task))
            results[task] = {"ok": result["ok"], "score": result["score"],
                             "seconds": result.get("eval_wall_seconds")}
            if not result["ok"]:
                errors.append(f"{task} baseline failed: {result['error']}")
    payload = {"ok": not errors, "python": sys.executable,
               "versions": versions, "tasks": list(TASKS),
               "artifacts": len(manifest.get("artifacts", {})),
               "evaluations": results, "errors": errors,
               "recommended_campaign": (
                   f"{sys.executable} tools/run_benchmark.py start ml-v9 "
                   f"--tasks {','.join(TASKS)} --runs 5 "
                   "--agent-concurrency 24 --time-budget 3600 "
                   "--iterations 1000 "
                   "--model gpt-5.6-sol --effort high "
                   "--prefix 5x-gpt56-sol-high-")}
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
