"""Evaluate LFM2.5-230M QWeight checkpoints on GPQA-Diamond using MPS.

The prompt and choices follow lm-evaluation-harness' zero-shot GPQA v2.2
multiple-choice task.  ``fingertap/GPQA-Diamond`` is already shuffled and
formatted, so its four answer letters are scored directly as ``(A)`` through
``(D)``.  All four continuations for a question are evaluated together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from bench.qweight import bundle_bytes, decode_bundle  # noqa: E402
from bench.lfm25_model_identity import (  # noqa: E402
    MODEL_ID,
    MODEL_PATH,
    REVISION,
)


DATASET_ID = "fingertap/GPQA-Diamond"
DATASET_REVISION = "68be7564497676e07a77a042fdb587deb88c51c3"
CHOICES = ("(A)", "(B)", "(C)", "(D)")
PROMPT = "What is the correct answer to this question:{question}\nAnswer:"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize() -> None:
    torch.mps.synchronize()


def load_model(bundle: Path | None):
    """Load native BF16 weights or decode a QWeight bundle into BF16."""
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, dtype=torch.bfloat16
    ).eval()
    metadata = {
        "kind": "bf16_native",
        "weights_sha256": sha256(MODEL_PATH / "model.safetensors"),
        "storage_bytes": (MODEL_PATH / "model.safetensors").stat().st_size,
    }
    if bundle is not None:
        bundle = bundle.resolve()
        state = model.state_dict()
        manifest, decoded = decode_bundle(
            bundle,
            {name: tuple(value.shape) for name, value in state.items()},
            MODEL_ID,
            REVISION,
            torch.device("mps"),
        )
        with torch.no_grad():
            for name, destination in state.items():
                destination.copy_(decoded[name].to(dtype=torch.bfloat16))
        del decoded, state
        torch.mps.empty_cache()
        metadata = {
            "kind": "qweight_decoded_to_bf16",
            "bundle": str(bundle),
            "producer": manifest.get("producer"),
            "target_bpw": manifest.get("target_bpw"),
            "storage_bytes": bundle_bytes(bundle),
            "manifest_sha256": sha256(bundle / "manifest.json"),
            "weights_sha256": sha256(bundle / "weights.safetensors"),
        }
    model.to("mps").eval()
    return model, metadata


def prepare_examples(dataset, tokenizer):
    examples = []
    max_length = 0
    for question_index, row in enumerate(dataset):
        context = PROMPT.format(question=row["question"])
        context_ids = tokenizer(context, add_special_tokens=True).input_ids
        candidates = []
        for choice_index, choice in enumerate(CHOICES):
            full_ids = tokenizer(context + choice, add_special_tokens=True).input_ids
            # Match lm-evaluation-harness' causal _encode_pair: tokenize the
            # whole pair, then split it at the standalone context token count.
            # Some tokenizers legitimately merge at this boundary.
            continuation_start = len(context_ids)
            continuation_tokens = len(full_ids) - continuation_start
            if continuation_tokens < 1:
                raise RuntimeError("empty GPQA answer continuation")
            max_length = max(max_length, len(full_ids))
            candidates.append(
                {
                    "question_index": question_index,
                    "choice_index": choice_index,
                    "ids": full_ids,
                    "continuation_start": continuation_start,
                    "continuation_tokens": continuation_tokens,
                }
            )
        answer = row["answer"].strip().upper()
        examples.append({"answer": ord(answer) - ord("A"), "candidates": candidates})
    return examples, max_length


@torch.inference_mode()
def score(
    model,
    examples,
    pad_token_id: int,
    batch_size: int,
    max_batch_tokens: int,
):
    candidates = [candidate for example in examples for candidate in example["candidates"]]
    # Length sorting minimizes padding while retaining large MPS batches.
    candidates.sort(key=lambda candidate: len(candidate["ids"]))
    scores = {}
    scored_tokens = 0
    padded_tokens = 0
    synchronize()
    started = time.perf_counter()
    offset = 0
    while offset < len(candidates):
        size = min(batch_size, len(candidates) - offset)
        while (
            size > 1
            and size * len(candidates[offset + size - 1]["ids"])
            > max_batch_tokens
        ):
            size -= 1
        batch = candidates[offset : offset + size]
        offset += size
        width = max(len(candidate["ids"]) for candidate in batch)
        input_ids = torch.full(
            (len(batch), width), pad_token_id, dtype=torch.long, device="mps"
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, candidate in enumerate(batch):
            length = len(candidate["ids"])
            input_ids[row_index, :length] = torch.tensor(
                candidate["ids"], dtype=torch.long, device="mps"
            )
            attention_mask[row_index, :length] = 1
        # Keep the full vocabulary projection in BF16 and cast only the few
        # continuation positions. Casting the full [batch, seq, vocab] tensor
        # to FP32 can need tens of GB for long GPQA questions.
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        for row_index, candidate in enumerate(batch):
            start = candidate["continuation_start"]
            stop = len(candidate["ids"])
            target = input_ids[row_index, start:stop]
            selected_log_probs = torch.log_softmax(
                logits[row_index, start - 1 : stop - 1].float(), dim=-1
            )
            token_scores = selected_log_probs.gather(
                -1, target.unsqueeze(-1)
            ).squeeze(-1)
            scores[(candidate["question_index"], candidate["choice_index"])] = (
                float(token_scores.sum().item()),
                float(token_scores.mean().item()),
            )
            scored_tokens += stop
        padded_tokens += input_ids.numel()
        del input_ids, attention_mask, logits
    synchronize()
    seconds = time.perf_counter() - started

    raw_correct = 0
    normalized_correct = 0
    predictions = []
    for question_index, example in enumerate(examples):
        raw = [scores[(question_index, choice)][0] for choice in range(4)]
        normalized = [scores[(question_index, choice)][1] for choice in range(4)]
        raw_prediction = max(range(4), key=raw.__getitem__)
        normalized_prediction = max(range(4), key=normalized.__getitem__)
        raw_correct += raw_prediction == example["answer"]
        normalized_correct += normalized_prediction == example["answer"]
        predictions.append(
            {
                "index": question_index,
                "answer": chr(ord("A") + example["answer"]),
                "prediction": chr(ord("A") + raw_prediction),
                "normalized_prediction": chr(ord("A") + normalized_prediction),
                "choice_loglikelihood": raw,
                "choice_mean_loglikelihood": normalized,
            }
        )
    count = len(examples)
    return {
        "accuracy": raw_correct / count,
        "accuracy_normalized": normalized_correct / count,
        "correct": raw_correct,
        "correct_normalized": normalized_correct,
        "questions": count,
        "inference_seconds": seconds,
        "sequences_per_second": len(candidates) / seconds,
        "unpadded_tokens_per_second": scored_tokens / seconds,
        "padding_fraction": 1.0 - scored_tokens / padded_tokens,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aqlm", type=Path)
    parser.add_argument("--hqq", type=Path)
    parser.add_argument("--optimized", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="additional quantized bundle to evaluate (repeatable)",
    )
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=12_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("this evaluation requires local MPS")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("MPS fallback must be disabled")
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    dataset = load_dataset(
        DATASET_ID, revision=DATASET_REVISION, split="test"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    examples, max_length = prepare_examples(dataset, tokenizer)

    output = {
        "protocol": {
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_fingerprint": dataset._fingerprint,
            "split": "test",
            "questions": len(dataset),
            "prompt": PROMPT,
            "choices": list(CHOICES),
            "fewshot": 0,
            "scoring": "continuation loglikelihood and per-token normalized loglikelihood",
            "execution_dtype": "bfloat16",
            "logit_scoring_dtype": "float32",
            "device": "mps",
            "mps_fallback": False,
            "batch_size": args.batch_size,
            "max_batch_tokens": args.max_batch_tokens,
            "max_sequence_tokens": max_length,
        },
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "datasets": __import__("datasets").__version__,
        },
        "results": {},
    }
    models: list[tuple[str, Path | None]] = []
    if not args.no_bf16:
        models.append(("bf16", None))
    models.extend(
        (name, path)
        for name, path in (
            ("aqlm", args.aqlm),
            ("hqq", args.hqq),
            ("optimized", args.optimized),
        )
        if path is not None
    )
    for checkpoint in args.checkpoint:
        name, separator, path = checkpoint.partition("=")
        if not separator or not name or not path:
            parser.error(f"invalid --checkpoint {checkpoint!r}; expected NAME=PATH")
        models.append((name, Path(path)))
    if not models:
        parser.error("no models selected")
    names = [name for name, _ in models]
    if len(names) != len(set(names)):
        parser.error("model names must be unique")
    for name, bundle in models:
        load_started = time.perf_counter()
        model, metadata = load_model(bundle)
        synchronize()
        metadata["load_seconds"] = time.perf_counter() - load_started
        result = score(
            model,
            examples,
            pad_token_id,
            args.batch_size,
            args.max_batch_tokens,
        )
        result["model"] = metadata
        output["results"][name] = result
        # Preserve each completed model in case a later model run is stopped.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(
            f"{name}: {result['correct']}/{result['questions']} "
            f"acc={result['accuracy']:.6f} "
            f"acc_norm={result['accuracy_normalized']:.6f} "
            f"seconds={result['inference_seconds']:.2f}",
            flush=True,
        )
        del model
        torch.mps.empty_cache()

    accuracies = [row["accuracy"] for row in output["results"].values()]
    output["summary"] = {
        "mean_accuracy": statistics.fmean(accuracies),
        "chance_accuracy": 0.25,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
