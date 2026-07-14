"""Select short BF16-passing GSM8K and diverse MMLU-Pro task rows.

This operator-only preparation command is intended for CUDA hosts.  It emits
the selected plaintext rows to an explicit output path outside the repository;
the task build step seals those rows before they enter the benchmark checkout.
"""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time


MODEL_ID = "LiquidAI/LFM2.5-230M"
MODEL_REVISION = "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7"
MODEL_PATH = Path("/tmp/lfm25-230m-source")
MODEL_FILES = {
    "chat_template.jinja":
        "6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963",
    "config.json":
        "f7d0bcc454b7a30fa471b1e7b9e359e11fb25b56f5b4ffd59bb18248e3c2ea3d",
    "generation_config.json":
        "4f88574c47c3215f7f952e1f520d1df7387422dde0345655228fb7b3fde6858c",
    "model.safetensors":
        "f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd",
    "tokenizer.json":
        "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd",
    "tokenizer_config.json":
        "75c287923e252b08b0a0f1c367bbe557ab23a681d0b71c5a34e0932ddbe2f5ee",
}
GSM8K_ID = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MMLUPRO_ID = "TIGER-Lab/MMLU-Pro"
MMLUPRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
LABELS = tuple("ABCDEFGHIJ")
NUMBER = re.compile(r"^[\s$]*([+-]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?)[\s%]*$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_model(snapshot_download):
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_PATH,
        allow_patterns=sorted(MODEL_FILES),
    )
    for name, expected in MODEL_FILES.items():
        actual = sha256(MODEL_PATH / name)
        if actual != expected:
            raise RuntimeError(f"pinned model hash mismatch for {name}: {actual}")


def canonical_number(value):
    match = NUMBER.fullmatch(value)
    if not match or not match.group(1):
        return None
    try:
        number = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in ("", "-0") else normalized


def gsm_answer(row):
    return canonical_number(row["answer"].rsplit("####", 1)[-1])


def gsm_prompt(question):
    return (
        "Solve this grade-school math problem. Reply with only the final "
        "numeric answer and no explanation.\n\n" + question.strip()
    )


def trigrams(value):
    words = re.findall(r"[a-z0-9]+", value.casefold())
    padded = ("<s>", *words, "</s>")
    return set(zip(padded, padded[1:], padded[2:]))


def distance(left, right):
    union = left | right
    return 1.0 if not union else 1.0 - len(left & right) / len(union)


def stable_noise(seed, row_id):
    raw = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") / 2**64


def diverse_short(rows, count, seed, category_key=None):
    """Deterministic max-min selection with a mild short-input preference."""
    if len(rows) < count:
        raise RuntimeError(f"need {count} candidates, found {len(rows)}")
    features = {row["id"]: trigrams(row["question"]) for row in rows}
    selected = []
    remaining = list(rows)
    if category_key:
        categories = sorted({row[category_key] for row in rows})
        rng = random.Random(seed)
        rng.shuffle(categories)
        for category in categories:
            local = [row for row in remaining if row[category_key] == category]
            if not local or len(selected) >= count:
                continue
            choice = min(
                local,
                key=lambda row: (
                    row["input_tokens"],
                    stable_noise(seed, row["id"]),
                    row["id"],
                ),
            )
            selected.append(choice)
            remaining.remove(choice)
    if not selected:
        selected.append(min(
            remaining,
            key=lambda row: (
                row["input_tokens"], stable_noise(seed, row["id"]), row["id"]
            ),
        ))
        remaining.remove(selected[0])
    while len(selected) < count:
        category_counts = {}
        for row in selected:
            if category_key:
                category_counts[row[category_key]] = (
                    category_counts.get(row[category_key], 0) + 1)
        choice = max(
            remaining,
            key=lambda row: (
                min(distance(features[row["id"]], features[item["id"]])
                    for item in selected),
                -(category_counts.get(row.get(category_key), 0)
                  if category_key else 0),
                -row["input_tokens"] / 256,
                stable_noise(seed, row["id"]),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def generate_gsm(torch, model, tokenizer, source_rows, source_split,
                 batch_size):
    prepared = []
    for index, raw in enumerate(source_rows):
        prompt = gsm_prompt(raw["question"])
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_tokens = len(tokenizer(
            rendered, add_special_tokens=False).input_ids)
        if input_tokens <= 192:
            prepared.append((input_tokens, index, raw, prompt, rendered))
    prepared.sort(key=lambda item: (item[0], item[1]))
    passed = []
    started = time.perf_counter()
    for offset in range(0, len(prepared), batch_size):
        batch = prepared[offset:offset + batch_size]
        encoded = tokenizer(
            [item[4] for item in batch],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")
        width = encoded.input_ids.shape[1]
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=16,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        suffixes = generated[:, width:]
        for item, suffix in zip(batch, suffixes):
            input_tokens, index, raw, prompt, _rendered = item
            terminated = bool((suffix == tokenizer.eos_token_id).any().item())
            response = tokenizer.decode(
                suffix, skip_special_tokens=True).strip()
            expected = gsm_answer(raw)
            if (terminated and expected is not None and
                    canonical_number(response) == expected):
                passed.append({
                    "id": f"gsm8k_{source_split}_{index:05d}",
                    "source_split": source_split,
                    "source_index": index,
                    "question": raw["question"].strip(),
                    "prompt": prompt,
                    "answer": expected,
                    "bf16_response": response,
                    "input_tokens": input_tokens,
                    "output_tokens": len(tokenizer(
                        response, add_special_tokens=False).input_ids),
                })
        del encoded, generated, suffixes
    torch.cuda.synchronize()
    return passed, time.perf_counter() - started, len(prepared)


def mmlupro_prompt(row):
    options = "\n".join(
        f"{label}. {option.strip()}"
        for label, option in zip(LABELS, row["options"])
    )
    return (
        "What is the correct answer to this multiple-choice question?\n"
        f"{row['question'].strip()}\n\n{options}\nAnswer:"
    )


def score_mmlupro(torch, model, tokenizer, source_rows, batch_size):
    label_ids = []
    for label in LABELS:
        ids = tokenizer(" " + label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise RuntimeError(f"MMLU-Pro label {label!r} is not one token: {ids}")
        label_ids.append(ids[0])
    prepared = []
    for index, raw in enumerate(source_rows):
        prompt = mmlupro_prompt(raw)
        input_tokens = len(tokenizer(prompt, add_special_tokens=True).input_ids)
        if input_tokens <= 256:
            prepared.append((input_tokens, index, raw, prompt))
    prepared.sort(key=lambda item: (item[0], item[1]))
    passed = []
    started = time.perf_counter()
    label_tensor = torch.tensor(label_ids, device="cuda")
    for offset in range(0, len(prepared), batch_size):
        batch = prepared[offset:offset + batch_size]
        encoded = tokenizer(
            [item[3] for item in batch],
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).to("cuda")
        logits = model(
            **encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        choice_logits = logits.index_select(-1, label_tensor)
        predictions = choice_logits.argmax(-1).tolist()
        for item, prediction in zip(batch, predictions):
            input_tokens, index, raw, prompt = item
            answer = raw["answer"]
            if LABELS[prediction] == answer:
                row_id = raw.get("question_id", index)
                passed.append({
                    "id": f"mmlupro_{int(row_id):05d}",
                    "source_split": "test",
                    "source_index": index,
                    "question_id": int(row_id),
                    "question": raw["question"].strip(),
                    "options": [option.strip() for option in raw["options"]],
                    "answer": answer,
                    "bf16_prediction": answer,
                    "category": raw["category"],
                    "src": raw["src"],
                    "prompt": prompt,
                    "input_tokens": input_tokens,
                    "output_tokens": 1,
                })
        del encoded, logits, choice_logits
    torch.cuda.synchronize()
    return passed, time.perf_counter() - started, len(prepared)


def split_mmlupro(rows):
    validation, test = [], []
    for row in rows:
        digest = hashlib.sha256(row["id"].encode()).digest()
        (validation if digest[0] & 1 else test).append(row)
    return validation, test


def summary(rows):
    categories = {}
    for row in rows:
        if "category" in row:
            categories[row["category"]] = categories.get(row["category"], 0) + 1
    result = {
        "rows": len(rows),
        "input_tokens": {
            "min": min(row["input_tokens"] for row in rows),
            "max": max(row["input_tokens"] for row in rows),
            "mean": sum(row["input_tokens"] for row in rows) / len(rows),
        },
        "output_tokens": {
            "min": min(row["output_tokens"] for row in rows),
            "max": max(row["output_tokens"] for row in rows),
            "mean": sum(row["output_tokens"] for row in rows) / len(rows),
        },
    }
    if categories:
        result["categories"] = dict(sorted(categories.items()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepare_model(snapshot_download)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()

    gsm_train = load_dataset(
        GSM8K_ID, "main", revision=GSM8K_REVISION, split="train")
    gsm_test = load_dataset(
        GSM8K_ID, "main", revision=GSM8K_REVISION, split="test")
    mmlupro = load_dataset(
        MMLUPRO_ID, revision=MMLUPRO_REVISION, split="test")

    with torch.inference_mode():
        gsm_validation_pool, gsm_validation_time, gsm_validation_candidates = (
            generate_gsm(
                torch, model, tokenizer, gsm_train, "train", args.batch_size))
        gsm_test_pool, gsm_test_time, gsm_test_candidates = generate_gsm(
            torch, model, tokenizer, gsm_test, "test", args.batch_size)
        mmlupro_pool, mmlupro_time, mmlupro_candidates = score_mmlupro(
            torch, model, tokenizer, mmlupro, args.batch_size)
    mmlu_validation_pool, mmlu_test_pool = split_mmlupro(mmlupro_pool)

    selected = {
        "validation": {
            "gsm8k": diverse_short(gsm_validation_pool, 20, 2026071401),
            "mmlupro": diverse_short(
                mmlu_validation_pool, 20, 2026071402, "category"),
        },
        "test": {
            "gsm8k": diverse_short(gsm_test_pool, 20, 2026071403),
            "mmlupro": diverse_short(
                mmlu_test_pool, 20, 2026071404, "category"),
        },
    }
    payload = {
        "format": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "sources": {
            "gsm8k": {"id": GSM8K_ID, "revision": GSM8K_REVISION},
            "mmlupro": {"id": MMLUPRO_ID, "revision": MMLUPRO_REVISION},
        },
        "policies": {
            "gsm8k": {
                "maximum_input_tokens": 192,
                "maximum_generated_tokens": 16,
                "greedy": True,
                "eos_required": True,
                "pass": "normalized exact final number",
            },
            "mmlupro": {
                "maximum_input_tokens": 256,
                "continuation": "one space-prefixed answer-label token",
                "greedy": True,
                "pass": "argmax answer label equals ground truth",
                "split": "SHA-256 row-id parity before selection",
            },
        },
        "pool": {
            "gsm8k_validation": {
                "candidates": gsm_validation_candidates,
                "passes": len(gsm_validation_pool),
                "seconds": gsm_validation_time,
            },
            "gsm8k_test": {
                "candidates": gsm_test_candidates,
                "passes": len(gsm_test_pool),
                "seconds": gsm_test_time,
            },
            "mmlupro": {
                "candidates": mmlupro_candidates,
                "passes": len(mmlupro_pool),
                "validation_pool": len(mmlu_validation_pool),
                "test_pool": len(mmlu_test_pool),
                "seconds": mmlupro_time,
            },
        },
        "selected": selected,
        "selected_summary": {
            split: {name: summary(rows) for name, rows in datasets.items()}
            for split, datasets in selected.items()
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "pool": payload["pool"],
        "selected_summary": payload["selected_summary"],
        "environment": payload["environment"],
    }, indent=2))


if __name__ == "__main__":
    main()
