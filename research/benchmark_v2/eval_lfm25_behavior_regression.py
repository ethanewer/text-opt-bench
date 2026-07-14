"""Evaluate QWeight checkpoints on frozen BF16-pass behavioral subsets."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from bench import heldout  # noqa: E402
from research.benchmark_v2.lfm25_behavior_regression import (  # noqa: E402
    bfcl_pass,
    dump_json,
    gpqa_predictions,
    greedy_generate,
    load_model,
    load_tokenizer,
    sha256,
)

DEFAULT_DATA = ROOT / "research/benchmark_v2/lfm25_behavior_data"


def reference_scaled_caps(tokenizer, rows, hard_limit: int) -> dict[str, int]:
    """Round BF16 response length up to 16, scale by 1.25, and cap."""
    result = {}
    for row in rows:
        length = len(
            tokenizer(row["bf16_response"], add_special_tokens=False).input_ids
        )
        result[row["id"]] = min(hard_limit, max(20, ((length + 15) // 16) * 20))
    return result


def load_ifbench_verifier(repo: Path):
    if not (repo / "evaluation_lib.py").is_file():
        raise RuntimeError(f"IFBench checkout is missing at {repo}")
    sys.path.insert(0, str(repo))
    return importlib.import_module("evaluation_lib")


def ifbench_loose_pass(verifier, row: dict, response: str) -> bool:
    example = verifier.InputExample(
        key=row["key"],
        instruction_id_list=list(row["instruction_id_list"]),
        prompt=row["prompt"],
        kwargs=[
            {key: value for key, value in values.items() if value is not None}
            for values in row["kwargs"]
        ],
    )
    result = verifier.test_instruction_following_loose(
        example, {row["prompt"]: response}
    )
    return bool(result.follow_all_instructions)


def load_data(data_dir: Path, splits: list[str]):
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = {
        "train": data_dir / "train.json",
        "test": data_dir / "heldout_test.bin",
    }
    for name, path in paths.items():
        if sha256(path) != manifest["sha256"][path.name]:
            raise RuntimeError(f"behavior data hash mismatch: {path.name}")
    payloads = {}
    if "train" in splits:
        payloads["train"] = json.loads(paths["train"].read_text())
    if "test" in splits:
        payloads["test"] = heldout.read(paths["test"])
    return manifest, payloads


def score_dataset(rows, passed_by_id, detail_by_id):
    details = []
    for row in rows:
        passed = bool(passed_by_id[row["id"]])
        details.append(
            {
                "id": row["id"],
                "regression": int(not passed),
                **detail_by_id[row["id"]],
            }
        )
    regressions = sum(row["regression"] for row in details)
    return {
        "regression_rate": regressions / len(details),
        "regressions": regressions,
        "items": len(details),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--name", default="bf16")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ifbench-repo", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument(
        "--reference-scaled-caps",
        action="store_true",
        help="use min(hard cap, round_up_to_16(BF16 response tokens) * 1.25)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("behavior regression evaluation requires local MPS")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("MPS fallback must be disabled")
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    splits = ["train", "test"] if args.split == "both" else [args.split]
    manifest, payloads = load_data(args.data.resolve(), splits)
    generation_batch_size = manifest["generation"]["scoring_batch_size"]
    verifier = load_ifbench_verifier(args.ifbench_repo.resolve())
    tokenizer = load_tokenizer()
    model, model_metadata = load_model(args.bundle)

    all_rows = {
        dataset: [
            row for split in splits for row in payloads[split]["datasets"][dataset]
        ]
        for dataset in ("gpqa", "ifbench", "bfcl")
    }

    gpqa_output, gpqa_seconds = gpqa_predictions(model, tokenizer, all_rows["gpqa"])
    gpqa_passed = {
        row["id"]: gpqa_output[row["id"]] == row["bf16_prediction"]
        for row in all_rows["gpqa"]
    }
    gpqa_detail = {
        row["id"]: {
            "bf16_prediction": row["bf16_prediction"],
            "prediction": gpqa_output[row["id"]],
        }
        for row in all_rows["gpqa"]
    }

    ifbench_cap = manifest["generation"]["ifbench_max_new_tokens"]
    if args.reference_scaled_caps:
        ifbench_cap = reference_scaled_caps(tokenizer, all_rows["ifbench"], ifbench_cap)
    ifbench_output, ifbench_seconds = greedy_generate(
        model,
        tokenizer,
        all_rows["ifbench"],
        max_new_tokens=ifbench_cap,
        batch_size=generation_batch_size,
    )
    ifbench_passed = {
        row["id"]: ifbench_loose_pass(verifier, row, ifbench_output[row["id"]])
        for row in all_rows["ifbench"]
    }
    ifbench_detail = {
        row["id"]: {
            "response": ifbench_output[row["id"]],
            "matches_bf16_text": (ifbench_output[row["id"]] == row["bf16_response"]),
        }
        for row in all_rows["ifbench"]
    }

    bfcl_cap = manifest["generation"]["bfcl_max_new_tokens"]
    if args.reference_scaled_caps:
        bfcl_cap = reference_scaled_caps(tokenizer, all_rows["bfcl"], bfcl_cap)
    bfcl_output, bfcl_seconds = greedy_generate(
        model,
        tokenizer,
        all_rows["bfcl"],
        max_new_tokens=bfcl_cap,
        batch_size=generation_batch_size,
    )
    bfcl_passed = {
        row["id"]: bfcl_pass(bfcl_output[row["id"]], row["ground_truth"])
        for row in all_rows["bfcl"]
    }
    bfcl_detail = {
        row["id"]: {
            "response": bfcl_output[row["id"]],
            "matches_bf16_text": bfcl_output[row["id"]] == row["bf16_response"],
        }
        for row in all_rows["bfcl"]
    }

    passed = {
        "gpqa": gpqa_passed,
        "ifbench": ifbench_passed,
        "bfcl": bfcl_passed,
    }
    details = {
        "gpqa": gpqa_detail,
        "ifbench": ifbench_detail,
        "bfcl": bfcl_detail,
    }
    results = {}
    for split in splits:
        datasets = {
            dataset: score_dataset(
                payloads[split]["datasets"][dataset],
                passed[dataset],
                details[dataset],
            )
            for dataset in ("gpqa", "ifbench", "bfcl")
        }
        score = statistics.fmean(
            result["regression_rate"] for result in datasets.values()
        )
        results[split] = {"score": score, "datasets": datasets}

    output = {
        "protocol": manifest["protocol"],
        "generation_policy": (
            "reference_scaled_caps" if args.reference_scaled_caps else "fixed_caps"
        ),
        "name": args.name,
        "model": model_metadata,
        "splits": results,
        "timing_seconds": {
            "gpqa": gpqa_seconds,
            "ifbench": ifbench_seconds,
            "bfcl": bfcl_seconds,
            "total_inference": gpqa_seconds + ifbench_seconds + bfcl_seconds,
        },
        "device": "mps",
        "mps_fallback": False,
        "greedy": True,
    }
    dump_json(args.output, output)
    for split, result in results.items():
        summary = ", ".join(
            f"{name}={value['regressions']}/{value['items']}"
            for name, value in result["datasets"].items()
        )
        print(f"{split}: score={result['score']:.6f}; {summary}", flush=True)
    print(
        f"inference_seconds={output['timing_seconds']['total_inference']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
