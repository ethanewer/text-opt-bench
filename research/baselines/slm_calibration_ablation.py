#!/usr/bin/env python3
"""Run 32/64/128-conversation calibration stability audits.

RTN is the calibration-independent control. AWQ-style and Wanda-style are
evaluator adapters over channel statistics; these results must not be labeled
as full reproductions of the corresponding papers. GPTQ is intentionally not
claimed because the online plan API does not implement its Hessian update.

The ordinary audit scores only the 64-row Qwen2.5/Qwen3.5 online validation
objective.  ``--sealed-test-after-optimization`` additionally measures the
4.125-bpw qwen25/qwen3/qwen35 test shards and must be run only after the
optimization campaign, so target-model/test calibration sensitivity cannot
affect the trajectory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench import deferred, runner
from bench.slm_mps_lock import (canonical_mps_lock_identity,
                                operator_mps_phase,
                                require_canonical_mps_lock_identity)

PLANS = {
    "rtn": ROOT / "research/baselines/slm_plans/rtn.py",
    "awq_style": ROOT / "research/baselines/slm_plans/awq_style.py",
    "magnitude_sparse": ROOT / "research/baselines/slm_plans/magnitude_sparse.py",
    "wanda_style": ROOT / "research/baselines/slm_plans/wanda_style.py",
}
TASKS = ("slm_compression_v2", "slm_compression_qwen35")


def evaluate_operator(task, program, **kwargs):
    """One background-priority operator evaluation, excluded from campaigns."""
    with operator_mps_phase("slm-calibration-ablation"):
        result = runner.evaluate(
            task, program, device="mps", evaluation_priority="background",
            **kwargs)
    metrics = result.get("metrics") or {}
    if result.get("ok"):
        if (metrics.get("canonical_device") != "mps" or
                metrics.get("device") != "mps" or
                metrics.get("compression_device") != "mps" or
                metrics.get("mps_fallback_enabled") is not False or
                metrics.get("calibration_backend") != "mps"):
            raise RuntimeError("calibration ablation returned invalid MPS provenance")
        require_canonical_mps_lock_identity(
            metrics.get("exclusive_mps_lock"),
            "calibration-ablation MPS lock")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/baselines/slm_calibration_ablation.json")
    parser.add_argument(
        "--sealed-test-after-optimization", action="store_true",
        help=("operator-only: audit the 4.125-bpw sealed test shard for every "
              "model after all optimization loops finish"))
    args = parser.parse_args()
    results = {
        "format": 3,
        "scope": (
            "evaluator-owned RTN/AWQ-style/Wanda-style mechanisms; not full "
            "AWQ, GPTQ, Wanda, or SparseGPT reproductions"),
        "online_objective": "64 validation conversations only",
        "canonical_device": "mps",
        "mps_fallback_enabled": False,
        "mps_lock": canonical_mps_lock_identity(),
        "benchmark_fingerprints": {},
        "protocol_versions": {},
        "scorer_versions": {},
        "planner_sha256": {
            method: hashlib.sha256(path.read_bytes()).hexdigest()
            for method, path in PLANS.items()
        },
        "tasks": {},
    }
    for task in [value for value in args.tasks.split(",") if value]:
        fingerprint = deferred.benchmark_fingerprint(task)
        results["benchmark_fingerprints"][task] = fingerprint
        results["protocol_versions"][task] = runner.load_config(task)[
            "protocol_version"]
        results["tasks"][task] = {}
        for method, program in PLANS.items():
            results["tasks"][task][method] = {}
            for size in (32, 64, 128):
                result = evaluate_operator(
                    task, program, calibration_size=size)
                if not result["ok"]:
                    raise RuntimeError(
                        f"{task}/{method}/{size}: {result['error']}")
                if deferred.benchmark_fingerprint(task) != fingerprint:
                    raise RuntimeError(
                        f"{task}: benchmark identity changed during calibration audit")
                if (hashlib.sha256(program.read_bytes()).hexdigest() !=
                        results["planner_sha256"][method]):
                    raise RuntimeError(
                        f"{method}: planner changed during calibration audit")
                scorer_version = result["metrics"].get("scorer_version")
                prior_scorer = results["scorer_versions"].get(task)
                if (not isinstance(scorer_version, str) or
                        prior_scorer not in (None, scorer_version)):
                    raise RuntimeError(
                        f"{task}: missing or mixed scorer version")
                results["scorer_versions"][task] = scorer_version
                results["tasks"][task][method][str(size)] = {
                    "score": result["score"],
                    "metrics": result["metrics"],
                    "eval_wall_seconds": result.get("eval_wall_seconds"),
                }
                print(json.dumps({
                    "task": task, "method": method,
                    "calibration_conversations": size,
                    "score": result["score"],
                }), flush=True)
    if args.sealed_test_after_optimization:
        results["sealed_test_after_optimization"] = {}
        for task in [value for value in args.tasks.split(",") if value]:
            fingerprint = results["benchmark_fingerprints"][task]
            shards = [shard for shard in runner.load_config(task)["test_shards"]
                      if shard.endswith("@4.125")]
            results["sealed_test_after_optimization"][task] = {}
            for method in ("rtn", "awq_style", "wanda_style"):
                program = PLANS[method]
                results["sealed_test_after_optimization"][task][method] = {}
                for size in (32, 64, 128):
                    local = {}
                    for shard in shards:
                        result = evaluate_operator(
                            task, program, test_only=True, test_shard=shard,
                            calibration_size=size)
                        if not result["ok"]:
                            raise RuntimeError(
                                f"{task}/{method}/{size}/{shard}: "
                                f"{result['error']}")
                        if deferred.benchmark_fingerprint(task) != fingerprint:
                            raise RuntimeError(
                                f"{task}: benchmark identity changed during "
                                "sealed calibration audit")
                        if (hashlib.sha256(program.read_bytes()).hexdigest() !=
                                results["planner_sha256"][method]):
                            raise RuntimeError(
                                f"{method}: planner changed during sealed audit")
                        local[shard] = {
                            "score": result["score"],
                            "metrics": result["metrics"],
                            "eval_wall_seconds": result.get(
                                "eval_wall_seconds"),
                        }
                        print(json.dumps({
                            "task": task, "method": method,
                            "calibration_conversations": size,
                            "sealed_test_shard": shard,
                            "score": result["score"],
                        }), flush=True)
                    results["sealed_test_after_optimization"][task][method][
                        str(size)] = local
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
