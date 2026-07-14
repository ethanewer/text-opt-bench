"""Build diverse 20-item BF16-pass behavioral-regression splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench import heldout  # noqa: E402
from research.benchmark_v2.lfm25_behavior_regression import (  # noqa: E402
    bfcl_pass,
    dump_json,
    gpqa_predictions,
    greedy_generate,
    load_model,
    load_tokenizer,
)
from research.benchmark_v2.prepare_lfm25_behavior_regression import (  # noqa: E402
    BFCL_VERSION,
    GPQA_ID,
    GPQA_REVISION,
    IFBENCH_COMMIT,
    IFBENCH_ID,
    IFBENCH_REVISION,
    ifbench_loose_pass,
    load_bfcl_rows,
    load_ifbench_verifier,
    order_key,
)

DEFAULT_SOURCE = ROOT / "research/benchmark_v2/lfm25_behavior_data"
DEFAULT_OUTPUT = ROOT / "research/benchmark_v2/lfm25_behavior_data_fast"
PER_SPLIT = 20
IFBENCH_SELECTION_MAX_NEW_TOKENS = 112
IFBENCH_EVALUATION_MAX_NEW_TOKENS = 128
BFCL_MAX_NEW_TOKENS = 96
WORD_RE = re.compile(r"[a-z0-9_]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def words(value: object) -> set[str]:
    return set(WORD_RE.findall(json.dumps(value, sort_keys=True).lower()))


def features(dataset: str, row: dict) -> set[str]:
    if dataset == "gpqa":
        return words(row["question"])
    if dataset == "ifbench":
        result = words(row["prompt"])
        for instruction_id in row["instruction_id_list"]:
            result.add(f"instruction={instruction_id}")
            result.add(f"family={instruction_id.split(':', 1)[0]}")
        return result
    if dataset == "bfcl":
        return words(
            {
                "category": row["category"],
                "messages": row["messages"],
                "tools": row["tools"],
            }
        )
    raise ValueError(f"unknown dataset: {dataset}")


def coverage_features(dataset: str, row: dict) -> set[str]:
    if dataset == "ifbench":
        return {
            marker
            for instruction_id in row["instruction_id_list"]
            for marker in (
                f"instruction={instruction_id}",
                f"family={instruction_id.split(':', 1)[0]}",
            )
        }
    if dataset == "bfcl":
        result = {f"category={row['category']}"}
        for tool in row["tools"]:
            function = tool.get("function", tool)
            if function.get("name"):
                result.add(f"function={function['name']}")
            properties = function.get("parameters", {}).get("properties", {})
            for name, schema in properties.items():
                result.add(f"parameter={name}")
                if schema.get("type"):
                    result.add(f"type={schema['type']}")
        return result
    return set()


def jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else 1.0 - len(left & right) / len(union)


def diverse_select(
    dataset: str, rows: list[dict], count: int, selection_name: str
) -> list[dict]:
    """Greedily maximize new structural coverage, then lexical distance."""
    if len(rows) < count:
        raise RuntimeError(f"{dataset} has only {len(rows)} rows; need {count}")
    remaining = list(rows)
    selected: list[dict] = []
    while len(selected) < count:
        best_row = choose_diverse(dataset, remaining, selected, selection_name)
        selected.append(best_row)
        remaining.remove(best_row)
    return selected


def choose_diverse(
    dataset: str,
    remaining: list[dict],
    selected: list[dict],
    selection_name: str,
) -> dict:
    selected_features = [features(dataset, row) for row in selected]
    covered = set().union(*(coverage_features(dataset, row) for row in selected))
    best_row = None
    best_score = None
    best_tie = None
    for row in remaining:
        row_features = features(dataset, row)
        novelty = len(coverage_features(dataset, row) - covered)
        distance = (
            1.0
            if not selected_features
            else min(
                jaccard_distance(row_features, chosen) for chosen in selected_features
            )
        )
        score = (novelty, distance)
        tie = order_key(selection_name, row["id"])
        if (
            best_score is None
            or score > best_score
            or (score == best_score and tie < best_tie)
        ):
            best_row, best_score, best_tie = row, score, tie
    assert best_row is not None
    return best_row


def diverse_train_test(
    dataset: str, rows: list[dict], count: int = PER_SPLIT, name: str | None = None
) -> tuple[list[dict], list[dict]]:
    """Alternate allocation so neither split can hoard rare features."""
    if len(rows) < 2 * count:
        raise RuntimeError(f"{dataset} has only {len(rows)} rows; need {2 * count}")
    name = name or dataset
    remaining = list(rows)
    splits = {"train": [], "test": []}
    for round_index in range(count):
        split_order = ("train", "test") if round_index % 2 == 0 else ("test", "train")
        for split in split_order:
            chosen = choose_diverse(
                dataset,
                remaining,
                splits[split],
                f"{name}-{split}-v3",
            )
            splits[split].append(chosen)
            remaining.remove(chosen)
    return splits["train"], splits["test"]


def diverse_bfcl_train_test(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train, test = [], []
    for category in ("simple_python", "live_simple"):
        pool = [row for row in rows if row["category"] == category]
        category_train, category_test = diverse_train_test(
            "bfcl",
            pool,
            count=PER_SPLIT // 2,
            name=f"bfcl-{category}",
        )
        train.extend(category_train)
        test.extend(category_test)
    return train, test


def diversity_summary(dataset: str, rows: list[dict]) -> dict:
    pairwise = [
        jaccard_distance(features(dataset, left), features(dataset, right))
        for index, left in enumerate(rows)
        for right in rows[index + 1 :]
    ]
    coverage = set().union(*(coverage_features(dataset, row) for row in rows))
    return {
        "mean_pairwise_jaccard_distance": sum(pairwise) / len(pairwise),
        "structural_features": len(coverage),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ifbench-repo", type=Path, required=True)
    parser.add_argument("--bfcl-data", type=Path, required=True)
    parser.add_argument(
        "--gpqa-cache",
        type=Path,
        default=Path("/private/tmp/lfm25_gpqa_bf16_predictions.json"),
    )
    parser.add_argument(
        "--ifbench-cache",
        type=Path,
        default=Path("/private/tmp/lfm25_ifbench_bf16_112_batch2.json"),
    )
    parser.add_argument(
        "--bfcl-cache",
        type=Path,
        default=Path("/private/tmp/lfm25_bfcl_bf16_96_batch2.json"),
    )
    parser.add_argument("--candidate-batch-size", type=int, default=2)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    verifier = load_ifbench_verifier(args.ifbench_repo.resolve())
    tokenizer = load_tokenizer()
    model, model_metadata = load_model(None)

    gpqa_source = load_dataset(GPQA_ID, revision=GPQA_REVISION, split="test")
    gpqa_rows = [
        {
            "id": f"gpqa_{index:03d}",
            "source_index": index,
            "question": row["question"],
            "answer": row["answer"].strip().upper(),
        }
        for index, row in enumerate(gpqa_source)
    ]
    if args.gpqa_cache.is_file():
        gpqa_output = json.loads(args.gpqa_cache.read_text())
        gpqa_seconds = 0.0
    else:
        gpqa_output, gpqa_seconds = gpqa_predictions(model, tokenizer, gpqa_rows)
        dump_json(args.gpqa_cache, gpqa_output)
    gpqa_passing = [
        {**row, "bf16_prediction": gpqa_output[row["id"]]}
        for row in gpqa_rows
        if gpqa_output[row["id"]] == row["answer"]
    ]

    ifbench_source = load_dataset(IFBENCH_ID, revision=IFBENCH_REVISION, split="train")
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
    ifbench_output, ifbench_seconds = greedy_generate(
        model,
        tokenizer,
        ifbench_rows,
        max_new_tokens=IFBENCH_SELECTION_MAX_NEW_TOKENS,
        batch_size=args.candidate_batch_size,
        cache_path=args.ifbench_cache.resolve(),
    )
    ifbench_candidate_passing = [
        {**row, "bf16_response": ifbench_output[row["id"]]}
        for row in ifbench_rows
        if ifbench_loose_pass(verifier, row, ifbench_output[row["id"]])
    ]
    ifbench_stable_cache = args.ifbench_cache.with_name(
        f"{args.ifbench_cache.stem}_single.json"
    )
    ifbench_stable_output, ifbench_stable_seconds = greedy_generate(
        model,
        tokenizer,
        ifbench_candidate_passing,
        max_new_tokens=IFBENCH_SELECTION_MAX_NEW_TOKENS,
        batch_size=1,
        cache_path=ifbench_stable_cache.resolve(),
    )
    ifbench_passing = [
        {**row, "bf16_response": ifbench_stable_output[row["id"]]}
        for row in ifbench_candidate_passing
        if ifbench_loose_pass(verifier, row, ifbench_stable_output[row["id"]])
        and len(
            tokenizer(
                ifbench_stable_output[row["id"]], add_special_tokens=False
            ).input_ids
        )
        < IFBENCH_SELECTION_MAX_NEW_TOKENS
    ]

    bfcl_rows = load_bfcl_rows(args.bfcl_data.resolve())
    bfcl_output, bfcl_seconds = greedy_generate(
        model,
        tokenizer,
        bfcl_rows,
        max_new_tokens=BFCL_MAX_NEW_TOKENS,
        batch_size=args.candidate_batch_size,
        cache_path=args.bfcl_cache.resolve(),
    )
    bfcl_candidate_passing = [
        {**row, "bf16_response": bfcl_output[row["id"]]}
        for row in bfcl_rows
        if bfcl_pass(bfcl_output[row["id"]], row["ground_truth"])
    ]
    bfcl_stable_cache = args.bfcl_cache.with_name(f"{args.bfcl_cache.stem}_single.json")
    bfcl_stable_output, bfcl_stable_seconds = greedy_generate(
        model,
        tokenizer,
        bfcl_candidate_passing,
        max_new_tokens=BFCL_MAX_NEW_TOKENS,
        batch_size=1,
        cache_path=bfcl_stable_cache.resolve(),
    )
    bfcl_passing = [
        {**row, "bf16_response": bfcl_stable_output[row["id"]]}
        for row in bfcl_candidate_passing
        if bfcl_pass(bfcl_stable_output[row["id"]], row["ground_truth"])
    ]

    selected = {}
    selected["gpqa"] = diverse_train_test("gpqa", gpqa_passing)
    selected["ifbench"] = diverse_train_test("ifbench", ifbench_passing)
    selected["bfcl"] = diverse_bfcl_train_test(bfcl_passing)
    train = {
        "format": 2,
        "datasets": {dataset: splits[0] for dataset, splits in selected.items()},
    }
    test = {
        "format": 2,
        "datasets": {dataset: splits[1] for dataset, splits in selected.items()},
    }

    train_path = output / "train.json"
    test_path = output / "heldout_test.bin"
    dump_json(train_path, train)
    heldout.write(test_path, test)

    source_manifest = json.loads((source / "manifest.json").read_text())
    pools = {
        "gpqa": gpqa_passing,
        "ifbench": ifbench_passing,
        "bfcl": bfcl_passing,
    }
    manifest = {
        **source_manifest,
        "format": 3,
        "protocol": "lfm25-bf16-pass-regression-diverse-v3",
        "derived_from": {
            "protocol": source_manifest["protocol"],
            "manifest_sha256": sha256(source / "manifest.json"),
        },
        "selection": {
            "rule": (
                "deterministic greedy structural-coverage then max-min lexical "
                "Jaccard diversity among stable BF16 passes; BFCL is balanced "
                "10/10 across simple_python and live_simple per split"
            ),
            "candidate_pool_counts": {
                "gpqa": len(gpqa_rows),
                "ifbench": len(ifbench_rows),
                "bfcl": len(bfcl_rows),
            },
            "bf16_pass_pool_counts": {
                dataset: len(rows) for dataset, rows in pools.items()
            },
            "per_dataset_per_split": PER_SPLIT,
            "ids": {
                split_name: {
                    dataset: [row["id"] for row in payload["datasets"][dataset]]
                    for dataset in ("gpqa", "ifbench", "bfcl")
                }
                for split_name, payload in (("train", train), ("test", test))
            },
            "diversity": {
                split_name: {
                    dataset: diversity_summary(dataset, payload["datasets"][dataset])
                    for dataset in ("gpqa", "ifbench", "bfcl")
                }
                for split_name, payload in (("train", train), ("test", test))
            },
        },
        "counts": {
            split_name: {
                dataset: len(payload["datasets"][dataset])
                for dataset in ("gpqa", "ifbench", "bfcl")
            }
            for split_name, payload in (("train", train), ("test", test))
        },
        "generation": {
            **source_manifest["generation"],
            "do_sample": False,
            "ifbench_selection_max_new_tokens": IFBENCH_SELECTION_MAX_NEW_TOKENS,
            "ifbench_max_new_tokens": IFBENCH_EVALUATION_MAX_NEW_TOKENS,
            "bfcl_max_new_tokens": BFCL_MAX_NEW_TOKENS,
            "candidate_batch_size": args.candidate_batch_size,
            "scoring_batch_size": 1,
        },
        "timing_seconds": {
            "gpqa": gpqa_seconds,
            "ifbench_candidates": ifbench_seconds,
            "ifbench_stabilization": ifbench_stable_seconds,
            "bfcl_candidates": bfcl_seconds,
            "bfcl_stabilization": bfcl_stable_seconds,
        },
        "model_metadata": model_metadata,
        "sources": {
            **source_manifest["sources"],
            "gpqa": {"id": GPQA_ID, "revision": GPQA_REVISION},
            "ifbench": {
                "id": IFBENCH_ID,
                "revision": IFBENCH_REVISION,
                "verifier_commit": IFBENCH_COMMIT,
            },
            "bfcl": {
                **source_manifest["sources"]["bfcl"],
                "package": "bfcl-eval",
                "version": BFCL_VERSION,
                "question_sha256": {
                    category: sha256(
                        args.bfcl_data.resolve() / f"BFCL_v4_{category}.json"
                    )
                    for category in ("simple_python", "live_simple")
                },
            },
        },
        "sha256": {
            "train.json": sha256(train_path),
            "heldout_test.bin": sha256(test_path),
        },
    }
    dump_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "pass_pools": manifest["selection"]["bf16_pass_pool_counts"],
                "counts": manifest["counts"],
                "timing_seconds": manifest["timing_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
