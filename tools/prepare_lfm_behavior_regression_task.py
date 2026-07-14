"""Freeze data for slm_weight_compression_lfm25_regression."""

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

SOURCE = ROOT / "bench/tasks/slm_weight_compression_lfm25/data"
BEHAVIOR = ROOT / "research/benchmark_v2/lfm25_behavior_data_fast"
OUTPUT = ROOT / "bench/tasks/slm_weight_compression_lfm25_regression/data"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "train.json", OUTPUT / "train.json")
    shutil.copy2(SOURCE / "model_attestation.json", OUTPUT / "model_attestation.json")

    train = json.loads((BEHAVIOR / "train.json").read_text())
    test = heldout.read(BEHAVIOR / "heldout_test.bin")
    heldout.write(OUTPUT / "heldout_val.bin", train)
    heldout.write(OUTPUT / "heldout_test.bin", {"regression": test})

    nltk_output = OUTPUT / "ifbench_nltk_data"
    if nltk_output.exists():
        shutil.rmtree(nltk_output)
    shutil.copytree(args.nltk_data.resolve(), nltk_output)
    nltk_files = {
        str(path.relative_to(OUTPUT)): sha256(path)
        for path in sorted(nltk_output.rglob("*"))
        if path.is_file()
    }
    (OUTPUT / "ifbench_nltk_manifest.json").write_text(
        json.dumps({"format": 1, "files": nltk_files}, indent=2, sort_keys=True) + "\n"
    )

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json.resolve()))
    calibration = json.loads((OUTPUT / "train.json").read_text())["records"]
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
        "task": "slm_weight_compression_lfm25_regression",
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
        "behavior_source_manifest_sha256": sha256(BEHAVIOR / "manifest.json"),
        "sha256": {name: sha256(OUTPUT / name) for name in artifacts},
    }
    (OUTPUT / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
