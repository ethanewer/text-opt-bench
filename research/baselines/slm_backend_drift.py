#!/usr/bin/env python3
"""Measure repeatability of the canonical MPS SLM scorer.

The filename is retained for compatibility with earlier experiment scripts,
but cross-backend scoring is no longer an admissible benchmark protocol.
Every repeat performs calibration-owned compression and scoring on MPS with
PyTorch CPU fallback disabled.
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


DEFAULT_PROGRAMS = {
    "slm_compression_v2":
        ROOT / "research/baselines/slm_plans/rtn.py",
    "slm_compression_qwen35":
        ROOT / "research/baselines/slm_plans/rtn.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(DEFAULT_PROGRAMS), required=True)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calibration-size", type=int, default=128,
                        choices=(32, 64, 128))
    parser.add_argument("--sealed-shard-after-optimization",
                        help="operator-only MODEL@BUDGET test shard")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2")

    program = args.program or DEFAULT_PROGRAMS[args.task]
    program = program.resolve()
    program_sha256 = hashlib.sha256(program.read_bytes()).hexdigest()
    benchmark_fingerprint = deferred.benchmark_fingerprint(args.task)
    results = []
    for repeat in range(args.repeats):
        kwargs = {
            "calibration_size": args.calibration_size,
            "device": "mps",
            "evaluation_priority": "background",
        }
        if args.sealed_shard_after_optimization:
            kwargs.update(
                test_only=True,
                test_shard=args.sealed_shard_after_optimization)
        with operator_mps_phase("slm-repeatability-audit"):
            result = runner.evaluate(args.task, program, **kwargs)
        if not result["ok"]:
            raise RuntimeError(f"MPS repeat {repeat}: {result['error']}")
        if deferred.benchmark_fingerprint(args.task) != benchmark_fingerprint:
            raise RuntimeError("benchmark identity changed during MPS repeatability audit")
        if hashlib.sha256(program.read_bytes()).hexdigest() != program_sha256:
            raise RuntimeError("planner changed during MPS repeatability audit")
        if (result["metrics"].get("device") != "mps" or
                result["metrics"].get("canonical_device") != "mps" or
                result["metrics"].get("compression_device") != "mps" or
                result["metrics"].get("mps_fallback_enabled") is not False or
                result["metrics"].get("calibration_backend") != "mps"):
            raise RuntimeError("scorer returned invalid MPS provenance")
        require_canonical_mps_lock_identity(
            result["metrics"].get("exclusive_mps_lock"),
            "repeatability-audit MPS lock")
        results.append({
            "repeat": repeat,
            "score": result["score"],
            "metrics": result["metrics"],
            "eval_wall_seconds": result.get("eval_wall_seconds"),
            "eval_cpu_seconds": result.get("eval_cpu_seconds"),
        })
        print(json.dumps({
            "repeat": repeat, "device": "mps", "score": result["score"],
            "eval_wall_seconds": result.get("eval_wall_seconds"),
        }), flush=True)

    scores = [row["score"] for row in results]
    payload = {
        "format": 3,
        "task": args.task,
        "program": str(program),
        "program_sha256": program_sha256,
        "benchmark_fingerprint": benchmark_fingerprint,
        "protocol_version": runner.load_config(args.task)["protocol_version"],
        "scorer_version": results[0]["metrics"].get("scorer_version"),
        "calibration_conversations": args.calibration_size,
        "canonical_device": "mps",
        "mps_fallback_enabled": False,
        "mps_lock": canonical_mps_lock_identity(),
        "scoring_inference_dtype": "float32",
        "objective": (
            "sealed test shard (post-optimization only)"
            if args.sealed_shard_after_optimization
            else "64-conversation validation"),
        "sealed_shard": args.sealed_shard_after_optimization,
        "results": results,
        "max_absolute_repeat_difference": max(scores) - min(scores),
        "interpretation": (
            "Only canonical-MPS repeats may be pooled. CPU, CUDA, and MPS "
            "runs with operator fallback enabled are outside the protocol."),
    }
    output = args.output or (
        ROOT / "research/baselines" /
        f"slm_mps_repeatability_{args.task}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "max_absolute_repeat_difference":
            payload["max_absolute_repeat_difference"],
    }, indent=2))


if __name__ == "__main__":
    main()
