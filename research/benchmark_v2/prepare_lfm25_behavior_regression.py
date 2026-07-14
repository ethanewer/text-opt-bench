"""Freeze BF16-pass behavioral-regression subsets for LFM2.5-230M."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench import heldout  # noqa: E402
from bench.tasks.slm_weight_compression_lfm25.model_identity import REVISION  # noqa: E402
from research.benchmark_v2.lfm25_behavior_regression import (  # noqa: E402
    bfcl_pass,
    dump_json,
    gpqa_predictions,
    greedy_generate,
    load_model,
    load_tokenizer,
)

GPQA_ID = "fingertap/GPQA-Diamond"
GPQA_REVISION = "68be7564497676e07a77a042fdb587deb88c51c3"
IFBENCH_ID = "allenai/IFBench_test"
IFBENCH_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
IFBENCH_COMMIT = "1091c4c3de6c1f6ed12c012ed68f11ea450b0117"
BFCL_VERSION = "2025.12.17"
PER_DATASET_PER_SPLIT = 16
IFBENCH_CANDIDATES = 128
BFCL_CANDIDATES_PER_CATEGORY = 128
IFBENCH_MAX_NEW_TOKENS = 256


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_key(dataset: str, row_id: str) -> str:
    return hashlib.sha256(f"behavior-v1|{dataset}|{row_id}".encode()).hexdigest()


def split_rows(dataset: str, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    rows = sorted(rows, key=lambda row: order_key(dataset, row["id"]))
    needed = 2 * PER_DATASET_PER_SPLIT
    if len(rows) < needed:
        raise RuntimeError(f"{dataset} has only {len(rows)} BF16 passes; need {needed}")
    return rows[:PER_DATASET_PER_SPLIT], rows[PER_DATASET_PER_SPLIT:needed]


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


def load_bfcl_rows(data_dir: Path) -> list[dict]:
    rows = []
    for category in ("simple_python", "live_simple"):
        questions_path = data_dir / f"BFCL_v4_{category}.json"
        answers_path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
        questions = {
            row["id"]: row
            for row in map(json.loads, questions_path.read_text().splitlines())
        }
        answers = {
            row["id"]: row["ground_truth"]
            for row in map(json.loads, answers_path.read_text().splitlines())
        }
        for row_id, question in questions.items():
            rows.append(
                {
                    "id": row_id,
                    "category": category,
                    "messages": question["question"][0],
                    "tools": question["function"],
                    "ground_truth": answers[row_id],
                }
            )
    selected = []
    for category in ("simple_python", "live_simple"):
        category_rows = [row for row in rows if row["category"] == category]
        category_rows.sort(key=lambda row: order_key("bfcl-pool", row["id"]))
        selected.extend(category_rows[:BFCL_CANDIDATES_PER_CATEGORY])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifbench-repo", type=Path, required=True)
    parser.add_argument("--bfcl-data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="operator-only output directory outside the benchmark repository",
    )
    parser.add_argument("--generation-batch-size", type=int, default=4)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output == ROOT or args.output.is_relative_to(ROOT):
        raise RuntimeError("behavior data output must remain outside the repository")
    args.output.mkdir(parents=True, exist_ok=True)

    verifier = load_ifbench_verifier(args.ifbench_repo.resolve())
    tokenizer = load_tokenizer()
    model, model_metadata = load_model(None)

    gpqa_source = load_dataset(
        GPQA_ID, revision=GPQA_REVISION, split="test"
    )
    gpqa_rows = [
        {
            "id": f"gpqa_{index:03d}",
            "source_index": index,
            "question": row["question"],
            "answer": row["answer"].strip().upper(),
        }
        for index, row in enumerate(gpqa_source)
    ]
    gpqa_output, gpqa_seconds = gpqa_predictions(model, tokenizer, gpqa_rows)
    gpqa_passes = [
        {**row, "bf16_prediction": gpqa_output[row["id"]]}
        for row in gpqa_rows
        if gpqa_output[row["id"]] == row["answer"]
    ]
    print(f"GPQA BF16 passes: {len(gpqa_passes)}/{len(gpqa_rows)}", flush=True)

    ifbench_source = load_dataset(
        IFBENCH_ID, revision=IFBENCH_REVISION, split="train"
    )
    ifbench_rows = [
        {
            "id": f"ifbench_{row['key']}",
            "key": row["key"],
            "prompt": row["prompt"],
            "instruction_id_list": list(row["instruction_id_list"]),
            "kwargs": [dict(value) for value in row["kwargs"]],
        }
        for row in ifbench_source
    ]
    ifbench_rows.sort(key=lambda row: order_key("ifbench-pool", row["id"]))
    ifbench_rows = ifbench_rows[:IFBENCH_CANDIDATES]
    ifbench_output, ifbench_seconds = greedy_generate(
        model,
        tokenizer,
        ifbench_rows,
        max_new_tokens=IFBENCH_MAX_NEW_TOKENS,
        batch_size=args.generation_batch_size,
        cache_path=args.output / "ifbench_bf16_cache_256.json",
    )
    ifbench_passes = []
    for row in ifbench_rows:
        response = ifbench_output[row["id"]]
        if ifbench_loose_pass(verifier, row, response):
            ifbench_passes.append({**row, "bf16_response": response})
    stable_ifbench_output, _ = greedy_generate(
        model,
        tokenizer,
        ifbench_passes,
        max_new_tokens=IFBENCH_MAX_NEW_TOKENS,
        batch_size=1,
        cache_path=args.output / "ifbench_bf16_single_cache_256.json",
    )
    ifbench_passes = [
        {**row, "bf16_response": stable_ifbench_output[row["id"]]}
        for row in ifbench_passes
        if ifbench_loose_pass(verifier, row, stable_ifbench_output[row["id"]])
    ]
    print(
        f"IFBench BF16 passes: {len(ifbench_passes)}/{len(ifbench_rows)}",
        flush=True,
    )

    bfcl_rows = load_bfcl_rows(args.bfcl_data.resolve())
    bfcl_output, bfcl_seconds = greedy_generate(
        model,
        tokenizer,
        bfcl_rows,
        max_new_tokens=96,
        batch_size=args.generation_batch_size,
        cache_path=args.output / "bfcl_bf16_cache.json",
    )
    bfcl_passes = []
    for row in bfcl_rows:
        response = bfcl_output[row["id"]]
        if bfcl_pass(response, row["ground_truth"]):
            bfcl_passes.append({**row, "bf16_response": response})
    stable_bfcl_output, _ = greedy_generate(
        model,
        tokenizer,
        bfcl_passes,
        max_new_tokens=96,
        batch_size=1,
        cache_path=args.output / "bfcl_bf16_single_cache.json",
    )
    bfcl_passes = [
        {**row, "bf16_response": stable_bfcl_output[row["id"]]}
        for row in bfcl_passes
        if bfcl_pass(stable_bfcl_output[row["id"]], row["ground_truth"])
    ]
    print(f"BFCL BF16 passes: {len(bfcl_passes)}/{len(bfcl_rows)}", flush=True)

    del model
    import torch

    torch.mps.empty_cache()

    train = {"format": 1, "datasets": {}}
    test = {"format": 1, "datasets": {}}
    pools = {
        "gpqa": gpqa_passes,
        "ifbench": ifbench_passes,
        "bfcl": bfcl_passes,
    }
    for dataset, rows in pools.items():
        train_rows, test_rows = split_rows(dataset, rows)
        train["datasets"][dataset] = train_rows
        test["datasets"][dataset] = test_rows

    train_path = args.output / "train.json"
    test_path = args.output / "heldout_test.bin"
    dump_json(train_path, train)
    heldout.write(test_path, test)
    manifest = {
        "format": 1,
        "protocol": "lfm25-bf16-pass-regression-v1",
        "model": {"id": "LiquidAI/LFM2.5-230M", "revision": REVISION},
        "sources": {
            "gpqa": {"id": GPQA_ID, "revision": GPQA_REVISION},
            "ifbench": {
                "id": IFBENCH_ID,
                "revision": IFBENCH_REVISION,
                "verifier_commit": IFBENCH_COMMIT,
            },
            "bfcl": {
                "package": "bfcl-eval",
                "version": BFCL_VERSION,
                "categories": ["simple_python", "live_simple"],
                "question_sha256": {
                    name: file_sha256(args.bfcl_data / f"BFCL_v4_{name}.json")
                    for name in ("simple_python", "live_simple")
                },
            },
        },
        "selection": {
            "rule": "sha256 behavior-v1 ordering among BF16 loose/AST passes",
            "per_dataset_per_split": PER_DATASET_PER_SPLIT,
            "candidate_pool_counts": {
                "gpqa": len(gpqa_rows),
                "ifbench": len(ifbench_rows),
                "bfcl": len(bfcl_rows),
            },
            "bf16_pass_pool_counts": {
                dataset: len(rows) for dataset, rows in pools.items()
            },
        },
        "counts": {
            split: {
                dataset: len(payload["datasets"][dataset])
                for dataset in payload["datasets"]
            }
            for split, payload in (("train", train), ("test", test))
        },
        "generation": {
            "do_sample": False,
            "ifbench_max_new_tokens": IFBENCH_MAX_NEW_TOKENS,
            "bfcl_max_new_tokens": 96,
            "batch_size": args.generation_batch_size,
            "scoring_batch_size": 1,
        },
        "timing_seconds": {
            "gpqa": gpqa_seconds,
            "ifbench": ifbench_seconds,
            "bfcl": bfcl_seconds,
        },
        "model_metadata": model_metadata,
        "sha256": {
            "train.json": file_sha256(train_path),
            "heldout_test.bin": file_sha256(test_path),
        },
    }
    dump_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
