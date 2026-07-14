"""Seal a CUDA-selected GSM8K/MMLU-Pro expansion into the LFM task."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import heldout  # noqa: E402
TASK = ROOT / "bench/tasks/slm_weight_compression_lfm25"
DATA = TASK / "data"
ML_ASSETS = ROOT / "bench/tasks/ml_assets.json"
PUBLIC_AUDIT = ROOT / "research/benchmark_v2/lfm25_behavior_expansion_results.json"
DATASETS = ("gpqa", "ifbench", "bfcl", "gsm8k", "mmlupro")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def observable(dataset, row):
    if dataset in ("gpqa", "mmlupro"):
        return row["question"]
    if dataset in ("ifbench", "gsm8k"):
        return row["prompt"]
    if dataset == "bfcl":
        return json.dumps(row["messages"], sort_keys=True)
    raise ValueError(dataset)


def validate_expansion(payload):
    if payload.get("format") != 1:
        raise RuntimeError("selection has the wrong format")
    selected = payload.get("selected")
    if set(selected or {}) != {"validation", "test"}:
        raise RuntimeError("selection must contain validation and test")
    seen = set()
    for split in ("validation", "test"):
        if set(selected[split]) != {"gsm8k", "mmlupro"}:
            raise RuntimeError(f"selection {split} has the wrong datasets")
        for dataset in ("gsm8k", "mmlupro"):
            rows = selected[split][dataset]
            if len(rows) != 20:
                raise RuntimeError(f"selection {split}/{dataset} needs 20 rows")
            for row in rows:
                if row["id"] in seen:
                    raise RuntimeError(f"duplicate selected id: {row['id']}")
                seen.add(row["id"])
                if dataset == "gsm8k":
                    if (row["output_tokens"] != 1 or row["input_tokens"] > 192 or
                            len(row["options"]) != 4 or
                            row["bf16_prediction"] != row["answer"] or
                            row["answer"] not in "ABCD"):
                        raise RuntimeError(f"invalid GSM8K pass: {row['id']}")
                elif (row["output_tokens"] != 1 or row["input_tokens"] > 256 or
                      row["bf16_prediction"] != row["answer"]):
                    raise RuntimeError(f"invalid MMLU-Pro pass: {row['id']}")


def update_ml_assets(manifest):
    assets = json.loads(ML_ASSETS.read_text())
    prefix = "bench/tasks/slm_weight_compression_lfm25/data/"
    for name, digest in manifest["sha256"].items():
        assets["artifacts"][prefix + name] = digest
    assets["artifacts"][prefix + "data_manifest.json"] = sha256(
        DATA / "data_manifest.json")
    assets.setdefault("slm_behavioral_compression", {})[
        "slm_weight_compression_lfm25"] = manifest
    ML_ASSETS.write_text(
        json.dumps(assets, separators=(",", ":"), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, required=True,
        help="operator-only output of prepare_lfm25_behavior_expansion.py",
    )
    parser.add_argument(
        "--tokenizer-json", type=Path, required=True,
        help="hash-attested LFM tokenizer.json used to audit calibration overlap",
    )
    args = parser.parse_args()
    selection_path = args.selection.resolve()
    if selection_path == ROOT or selection_path.is_relative_to(ROOT):
        raise RuntimeError("plaintext selection must remain outside the repository")
    selection = json.loads(selection_path.read_text())
    validate_expansion(selection)

    validation = heldout.read(DATA / "heldout_val.bin")
    test_container = heldout.read(DATA / "heldout_test.bin")
    test = test_container["regression"]
    if (set(validation["datasets"]) != {"gpqa", "ifbench", "bfcl"} or
            set(test["datasets"]) != {"gpqa", "ifbench", "bfcl"}):
        raise RuntimeError("task is not the expected unexpanded protocol")
    validation["datasets"].update(selection["selected"]["validation"])
    test["datasets"].update(selection["selected"]["test"])

    ids = set()
    normalized_observables = set()
    benchmark_text = []
    for split_name, split in (("validation", validation), ("test", test)):
        if set(split["datasets"]) != set(DATASETS):
            raise RuntimeError(f"{split_name} has the wrong datasets")
        for dataset in DATASETS:
            rows = split["datasets"][dataset]
            if len(rows) != 20:
                raise RuntimeError(f"{split_name}/{dataset} does not have 20 rows")
            for row in rows:
                if row["id"] in ids:
                    raise RuntimeError(f"cross-split duplicate id: {row['id']}")
                ids.add(row["id"])
                value = observable(dataset, row)
                normalized = normalize(value)
                if normalized in normalized_observables:
                    raise RuntimeError(
                        f"cross-split duplicate observable in {split_name}/{dataset}")
                normalized_observables.add(normalized)
                benchmark_text.append(value)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json.resolve()))
    calibration = json.loads((DATA / "train.json").read_text())["records"]
    calibration_text = [
        normalize(tokenizer.decode(row["input_ids"])) for row in calibration
    ]
    overlap = []
    for index, value in enumerate(map(normalize, benchmark_text)):
        if any(value == local or value in local for local in calibration_text):
            overlap.append(index)
    if overlap:
        raise RuntimeError(f"benchmark/calibration overlap: {overlap}")

    heldout.write(DATA / "heldout_val.bin", validation)
    heldout.write(DATA / "heldout_test.bin", {"regression": test})
    old_manifest = json.loads((DATA / "data_manifest.json").read_text())
    artifacts = [
        "train.json",
        "heldout_val.bin",
        "heldout_test.bin",
        "model_attestation.json",
        "ifbench_nltk_manifest.json",
    ]
    manifest = {
        **old_manifest,
        "format": 2,
        "counts": {
            "calibration": 128,
            "validation": 100,
            "test": 100,
            "per_benchmark_per_split": 20,
            "benchmarks": list(DATASETS),
        },
        "calibration_overlap_audit": {
            "method": "NFKC-casefold exact/substring comparison after decoding",
            "benchmark_observables": len(benchmark_text),
            "overlaps": 0,
        },
        "expansion": {
            "selection_sha256": sha256(selection_path),
            "sources": selection["sources"],
            "policies": selection["policies"],
            "pool": selection["pool"],
            "selected_summary": selection["selected_summary"],
            "environment": selection["environment"],
        },
        "sha256": {name: sha256(DATA / name) for name in artifacts},
    }
    (DATA / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    update_ml_assets(manifest)

    public = {
        "format": 1,
        "task": "slm_weight_compression_lfm25",
        "selection_sha256": sha256(selection_path),
        "model": selection["model"],
        "sources": selection["sources"],
        "policies": selection["policies"],
        "pool": selection["pool"],
        "selected_summary": selection["selected_summary"],
        "environment": selection["environment"],
        "task_artifacts": manifest["sha256"],
        "calibration_overlap_audit": manifest["calibration_overlap_audit"],
        "privacy": (
            "selected test ids, prompts, answers, and BF16 responses are present "
            "only in the sealed task artifact"
        ),
    }
    PUBLIC_AUDIT.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n")
    print(json.dumps(public, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
