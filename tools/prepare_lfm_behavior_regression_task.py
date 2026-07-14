"""Freeze behavioral data for slm_weight_compression_lfm25."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unicodedata

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import heldout  # noqa: E402
from bench.ifbench_subset import configure_nltk_data, loose_pass  # noqa: E402

DEFAULT_CALIBRATION = ROOT / "bench/tasks/slm_weight_compression_lfm25/data"
DEFAULT_OUTPUT = ROOT / "bench/tasks/slm_weight_compression_lfm25/data"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior-dir",
        type=Path,
        required=True,
        help="operator-only frozen behavior source outside the repository",
    )
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    args = parser.parse_args()
    behavior = args.behavior_dir.resolve()
    calibration_dir = args.calibration_dir.resolve()
    output = args.output.resolve()
    if behavior == ROOT or behavior.is_relative_to(ROOT):
        raise RuntimeError(
            "behavior source must remain outside the optimizer-readable repository"
        )
    source_manifest_path = behavior / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    for name in ("train.json", "heldout_test.bin"):
        if sha256(behavior / name) != source_manifest.get("sha256", {}).get(name):
            raise RuntimeError(f"behavior source hash mismatch: {name}")

    output.mkdir(parents=True, exist_ok=True)
    for name in ("train.json", "model_attestation.json"):
        source, destination = calibration_dir / name, output / name
        if source != destination:
            shutil.copy2(source, destination)

    train = json.loads((behavior / "train.json").read_text())
    test = heldout.read(behavior / "heldout_test.bin")
    heldout.write(output / "heldout_val.bin", train)
    heldout.write(output / "heldout_test.bin", {"regression": test})

    nltk_output = output / "ifbench_nltk_data"
    if nltk_output.exists():
        shutil.rmtree(nltk_output)
    shutil.copytree(args.nltk_data.resolve(), nltk_output)
    nltk_files = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(nltk_output.rglob("*"))
        if path.is_file()
    }
    (output / "ifbench_nltk_manifest.json").write_text(
        json.dumps({"format": 1, "files": nltk_files}, indent=2, sort_keys=True) + "\n"
    )

    import nltk

    previous_nltk_path = configure_nltk_data(nltk_output)
    try:
        for resource in (
            "corpora/stopwords",
            "taggers/averaged_perceptron_tagger_eng",
            "tokenizers/punkt_tab/english",
        ):
            nltk.data.find(resource)
        ifbench_rows = [
            row
            for payload in (train, test)
            for row in payload["datasets"]["ifbench"]
        ]
        failed = [
            row["id"]
            for row in ifbench_rows
            if not loose_pass(row, row["bf16_response"])
        ]
        if failed:
            raise RuntimeError(f"frozen BF16 IFBench self-test failed: {failed}")
    finally:
        nltk.data.path[:] = list(previous_nltk_path)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json.resolve()))
    calibration = json.loads((output / "train.json").read_text())["records"]
    calibration_text = [
        normalize(tokenizer.decode(row["input_ids"])) for row in calibration
    ]
    benchmark_text = []
    for payload in (train, test):
        benchmark_text.extend(row["question"] for row in payload["datasets"]["gpqa"])
        benchmark_text.extend(row["prompt"] for row in payload["datasets"]["ifbench"])
        benchmark_text.extend(
            json.dumps(row["messages"], sort_keys=True)
            for row in payload["datasets"]["bfcl"]
        )
    overlap = []
    for index, value in enumerate(map(normalize, benchmark_text)):
        if any(value == local or value in local for local in calibration_text):
            overlap.append(index)
    if overlap:
        raise RuntimeError(f"benchmark/calibration overlap: {overlap}")

    artifacts = [
        "train.json",
        "heldout_val.bin",
        "heldout_test.bin",
        "model_attestation.json",
        "ifbench_nltk_manifest.json",
    ]
    manifest = {
        "format": 1,
        "task": "slm_weight_compression_lfm25",
        "model": {
            "id": "LiquidAI/LFM2.5-230M",
            "revision": "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7",
        },
        "counts": {
            "calibration": 128,
            "validation": 60,
            "test": 60,
            "per_benchmark_per_split": 20,
        },
        "calibration_overlap_audit": {
            "method": "NFKC-casefold exact/substring comparison after decoding",
            "benchmark_observables": len(benchmark_text),
            "overlaps": 0,
        },
        "ifbench_pinned_self_test": {
            "examples": len(ifbench_rows),
            "failed": 0,
            "nltk_search_path": "task_local_only",
        },
        "behavior_source_manifest_sha256": sha256(source_manifest_path),
        "sha256": {name: sha256(output / name) for name in artifacts},
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
