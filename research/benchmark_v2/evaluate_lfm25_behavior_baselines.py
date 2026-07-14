#!/usr/bin/env python3
"""Evaluate fixed LFM QWeight baselines on the current behavior splits.

The large QWeight payloads remain operator artifacts outside Git.  This tool
wraps each payload in a tiny producer program, sends it through the canonical
task evaluator, and publishes only aggregate validation/test metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import deferred, runner  # noqa: E402

TASK = "slm_weight_compression_lfm25"
PARAMETERS = 229_693_184


def named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path).expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = str(path.relative_to(directory)).encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def copy_program(bundle: Path, destination: Path) -> Path:
    program = destination / "copy_bundle.py"
    program.write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "import shutil\n\n"
        f"SOURCE = Path({str(bundle)!r})\n\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--model')\n"
        "parser.add_argument('--calibration')\n"
        "parser.add_argument('--output', required=True)\n"
        "parser.add_argument('--targets', required=True)\n"
        "parser.add_argument('--device')\n"
        "args = parser.parse_args()\n"
        "shutil.copytree(SOURCE, Path(args.output) / args.targets)\n"
    )
    return program


def aggregate(result: dict) -> dict:
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "baseline evaluation failed")
    metrics = result.get("metrics") or {}
    return {
        "regression_rate": float(result["score"]),
        "dataset_regression_rates": dict(metrics["dataset_regression_rates"]),
        "wall_seconds": float(result["eval_wall_seconds"]),
        "cpu_seconds": float(result["eval_cpu_seconds"]),
    }


def evaluate(name: str, program: Path, device: str, kind: str,
             bundle: Path | None = None) -> dict:
    validation = runner.evaluate(TASK, program, device=device)
    test = runner.evaluate(TASK, program, final=True, device=device)
    if not validation.get("ok"):
        raise RuntimeError(f"{name} validation failed: {validation.get('error')}")
    metrics = validation.get("metrics") or {}
    result = {
        "name": name,
        "kind": kind,
        "program_sha256": sha256(program),
        "bundle_storage_bytes": int(metrics["bundle_storage_bytes"]),
        "whole_model_bits_per_parameter": float(
            metrics["whole_model_bits_per_parameter"]),
        "validation": aggregate(validation),
        "sealed_test": aggregate(test),
    }
    if bundle is not None:
        manifest = json.loads((bundle / "manifest.json").read_text())
        result.update(
            bundle_sha256=bundle_sha256(bundle),
            producer=manifest.get("producer"),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", action="append", type=named_path, default=[])
    parser.add_argument("--bundle", action="append", type=named_path, default=[])
    parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.program and not args.bundle:
        parser.error("at least one --program or --bundle is required")

    methods = []
    for name, program in args.program:
        methods.append(evaluate(name, program, args.device, "program"))
    for name, bundle in args.bundle:
        if not (bundle / "manifest.json").is_file():
            raise RuntimeError(f"missing QWeight manifest: {bundle}")
        with tempfile.TemporaryDirectory(prefix="lfm-baseline-producer-") as tmp:
            program = copy_program(bundle, Path(tmp))
            methods.append(evaluate(
                name, program, args.device, "qweight", bundle=bundle))

    task_config = runner.load_config(TASK)
    packages = {}
    for package in task_config.get("fingerprint_packages", ()):
        packages[package] = importlib.metadata.version(package)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    payload = {
        "format": 1,
        "task": TASK,
        "protocol_version": task_config.get("protocol_version"),
        "benchmark_fingerprint": deferred.benchmark_fingerprint(TASK),
        "evaluated_commit": commit,
        "device": args.device,
        "package_versions": packages,
        "parameters": PARAMETERS,
        "bf16_reference_regression_rate": 0.0,
        "methods": methods,
        "privacy": ("Only aggregate split metrics are published; per-example "
                    "sealed-test rows remain outside Git."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
