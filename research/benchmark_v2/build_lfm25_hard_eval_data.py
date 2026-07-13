"""Build longer, public-reference LFM validation and test splits.

The existing 128-row calibration split is retained byte-for-byte at the row
level.  Scored rows are rebuilt from pinned public corpora and deliberately
constrain assistant targets to a narrow, long range.  This makes the metric
less dominated by short labels and greedy self-generated text while keeping
the 512-token per-conversation ceiling and fast local grading.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from datasets import load_dataset
from transformers import AutoTokenizer

from bench.slm_data import conversation_record


PRIVATE = Path(
    "/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/"
    "slm_sft_data/generated"
)
OLD_DATA = PRIVATE / "lfm25_generated_selected.json"
OUTPUT = PRIVATE / "lfm25_hard_eval_selected.json"
MODEL = Path("/private/tmp/lfm25-230m-source")
SITE_PACKAGES = Path("/tmp/text-opt-bm-ml/lib/python3.12/site-packages")
TARGET_PER_FAMILY = 16
MIN_ASSISTANT_TOKENS = 220
MAX_ASSISTANT_TOKENS = 340
MAX_USER_TOKENS = 112

SOURCES = {
    "ultrachat": {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "config": "default",
        "split": "train_sft",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "license": "MIT",
    },
    "dolly": {
        "dataset": "databricks/databricks-dolly-15k",
        "config": None,
        "split": "train",
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "license": "CC-BY-SA-3.0",
    },
    "hh_rlhf": {
        "dataset": "Anthropic/hh-rlhf",
        "config": None,
        "split": "train",
        "revision": "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
        "license": "MIT",
    },
    "wikitext": {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-v1",
        "split": "train",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "license": "CC-BY-SA-3.0/GFDL",
    },
    "math": {
        "dataset": "nvidia/Nemotron-SFT-Math-v3",
        "config": None,
        "split": "train",
        "revision": "ff4439c1073c87e006ab7ee5f1e5e28c4790dab3",
        "license": "CC-BY-4.0/CC-BY-SA-4.0",
    },
    "science_mcq": {
        "dataset": "nvidia/Nemotron-Science-v1",
        "config": None,
        "split": "MCQ",
        "revision": "82e1af468197076b4f0f392c239274eac032adc7",
        "license": "CC-BY-4.0",
    },
    "science_rqa": {
        "dataset": "nvidia/Nemotron-Science-v1",
        "config": None,
        "split": "RQA",
        "revision": "82e1af468197076b4f0f392c239274eac032adc7",
        "license": "CC-BY-4.0",
    },
    "transformers_code": {
        "dataset": "huggingface/transformers",
        "config": None,
        "split": "source release",
        "revision": "v5.2.0",
        "license": "Apache-2.0",
    },
    "aiohttp_code": {
        "dataset": "aio-libs/aiohttp",
        "config": None,
        "split": "source release",
        "revision": "v3.14.1",
        "license": "Apache-2.0",
    },
    "pybind11_code": {
        "dataset": "pybind/pybind11",
        "config": None,
        "split": "source release bundled by torch",
        "revision": "v3.0.4",
        "license": "BSD-3-Clause",
    },
}


def dataset(key):
    spec = SOURCES[key]
    args = [spec["dataset"]]
    if spec["config"]:
        args.append(spec["config"])
    return load_dataset(
        *args, split=spec["split"], streaming=True,
        revision=spec["revision"])


def messages(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def adjacent_pair(value):
    rows = messages(value)
    for index in range(len(rows) - 2, -1, -1):
        if rows[index].get("role") == "user" and rows[index + 1].get("role") == "assistant":
            user = rows[index].get("content", "")
            assistant = rows[index + 1].get("content", "")
            if isinstance(user, str) and isinstance(assistant, str):
                return user, assistant
    return "", ""


def hh_pair(text):
    turns = re.findall(
        r"(?:^|\n\n)(Human|Assistant):\s*(.*?)(?=(?:\n\n(?:Human|Assistant):)|$)",
        text, flags=re.S)
    for index in range(len(turns) - 2, -1, -1):
        if turns[index][0] == "Human" and turns[index + 1][0] == "Assistant":
            return turns[index][1].strip(), turns[index + 1][1].strip()
    return "", ""


def clean(text):
    return re.sub(r"\n{4,}", "\n\n\n", str(text)).strip()


class Builder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL), local_files_only=True)
        self.used = set()
        self.used_assistants = set()
        self.provenance = {}

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def crop(self, text, limit):
        ids = self.tokenizer.encode(clean(text), add_special_tokens=False)[:limit]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def fit(self, split, family, source_key, source_id, user, assistant,
            cluster=None):
        user, assistant = clean(user), clean(assistant)
        if not user or not assistant:
            return None
        assistant_ids = self.tokenizer.encode(assistant, add_special_tokens=False)
        if len(assistant_ids) < MIN_ASSISTANT_TOKENS:
            return None
        user = self.crop(user, MAX_USER_TOKENS)
        assistant = self.tokenizer.decode(
            assistant_ids[:MAX_ASSISTANT_TOKENS], skip_special_tokens=True).strip()
        relation = "heldout" if split == "ood_test" else "development"
        source_hash = hashlib.sha256(
            (source_key + "\0" + str(source_id) + "\0" + user + "\0" + assistant)
            .encode()).hexdigest()
        if source_hash in self.used:
            return None
        candidate_id = f"lfm25_hard_{split}_{family}_{source_hash[:12]}"
        candidate = {
            "candidate_id": candidate_id,
            "family": family,
            "domain_relation": relation,
            "template_cluster": cluster or f"{family}:{source_key}",
            "messages": [
                {"role": "system", "content": "Complete the task accurately and preserve the requested format."},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        }
        # Tokenizer decode boundaries can add a token; trim until the rendered
        # conversation satisfies the benchmark's hard 512-token limit.
        while True:
            try:
                record = conversation_record(candidate, "lfm25", self.tokenizer)
                break
            except ValueError as exc:
                if "rendered conversation has" not in str(exc):
                    raise
                ids = self.tokenizer.encode(
                    candidate["messages"][-1]["content"],
                    add_special_tokens=False)
                if len(ids) <= MIN_ASSISTANT_TOKENS:
                    return None
                candidate["messages"][-1]["content"] = self.tokenizer.decode(
                    ids[:-8], skip_special_tokens=True).strip()
        if record["assistant_tokens"] < MIN_ASSISTANT_TOKENS:
            return None
        normalized_assistant = " ".join(
            candidate["messages"][-1]["content"].lower().split())
        assistant_hash = hashlib.sha256(normalized_assistant.encode()).hexdigest()
        if assistant_hash in self.used_assistants:
            return None
        self.used.add(source_hash)
        self.used_assistants.add(assistant_hash)
        spec = SOURCES[source_key]
        record.update({
            "split": split,
            "target_origin": "public_reference",
            "source_key": source_key,
            "source_id": str(source_id),
            "source_revision": spec["revision"],
            "source_license": spec["license"],
        })
        self.provenance[candidate_id] = {
            "dataset": spec["dataset"], "config": spec["config"],
            "split": spec["split"], "revision": spec["revision"],
            "record_id": str(source_id), "license": spec["license"],
            "source_content_sha256": source_hash,
        }
        return record

    def collect(self, split, family, source_key, extractor, accept=lambda row: True,
                limit=TARGET_PER_FAMILY, scan_limit=20000):
        result = []
        for index, row in enumerate(dataset(source_key)):
            if index >= scan_limit:
                break
            if not accept(row):
                continue
            pair = extractor(row)
            if not pair:
                continue
            user, assistant = pair
            source_id = (row.get("uuid") or row.get("prompt_id") or
                         row.get("nemotron_pretraining_id") or index)
            record = self.fit(
                split, family, source_key, source_id, user, assistant)
            if record:
                result.append(record)
            if len(result) == limit:
                return result
        raise RuntimeError(
            f"{source_key}/{family} supplied only {len(result)} of {limit} rows")

    def collect_code_files(self, split, family, source_key, root, pattern,
                           limit=TARGET_PER_FAMILY):
        result = []
        paths = sorted(path for path in root.rglob(pattern)
                       if path.is_file() and "__pycache__" not in path.parts)
        for path in paths:
            try:
                text = path.read_text(errors="strict")
            except (OSError, UnicodeDecodeError):
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            # Prefer one independent file per row. A long window supplies
            # contextual prefix tokens followed by a high-entropy target.
            if len(ids) < 420:
                continue
            prefix = self.tokenizer.decode(ids[:80], skip_special_tokens=True)
            target = self.tokenizer.decode(ids[80:420], skip_special_tokens=True)
            relative = path.relative_to(root)
            user = (f"Continue this public {source_key.replace('_', ' ')} source "
                    f"excerpt exactly ({relative}):\n\n{prefix}")
            record = self.fit(
                split, family, source_key, relative, user, target)
            if record:
                result.append(record)
            if len(result) == limit:
                return result
        raise RuntimeError(
            f"{source_key}/{family} supplied only {len(result)} of {limit} files")


def continuation_pair(row):
    text = clean(row["text"])
    pivot = min(len(text) // 3, 420)
    return ("Continue this public reference excerpt exactly and preserve its style:\n\n" +
            text[:pivot], text[pivot:])


def dolly_pair(row):
    user = row["instruction"]
    if row.get("context"):
        user += "\n\nContext:\n" + row["context"]
    return user, row["response"]


def keyword_pair(words):
    pattern = re.compile("|".join(re.escape(word) for word in words), re.I)

    def accept(row):
        user, _ = adjacent_pair(row["messages"])
        return bool(pattern.search(user))

    return accept


def main():
    builder = Builder()
    old = json.loads(OLD_DATA.read_text())
    calibration = [row for row in old["records"] if row["split"] == "calibration"]
    if len(calibration) != 128 or sum(row["total_tokens"] for row in calibration) != 32621:
        raise RuntimeError("canonical calibration split changed unexpectedly")

    scored = []
    for split in ("validation", "id_test"):
        scored += builder.collect(split, "general_chat", "ultrachat",
                                  lambda row: adjacent_pair(row["messages"]))
        scored += builder.collect(split, "instruction_response", "dolly", dolly_pair)
        scored += builder.collect(split, "safety_dialogue", "hh_rlhf",
                                  lambda row: hh_pair(row["chosen"]))
        scored += builder.collect(split, "math_reasoning", "math",
                                  lambda row: adjacent_pair(row["messages"]))
        scored += builder.collect(split, "science_reasoning", "science_mcq",
                                  lambda row: adjacent_pair(row["messages"]))
        scored += builder.collect_code_files(
            split, "ml_library_code", "transformers_code",
            SITE_PACKAGES / "transformers", "*.py")
        scored += builder.collect_code_files(
            split, "network_library_code", "aiohttp_code",
            SITE_PACKAGES / "aiohttp", "*.py")
        scored += builder.collect(split, "encyclopedic_prose", "wikitext",
                                  continuation_pair)

    scored += builder.collect(
        "ood_test", "legal_policy", "ultrachat",
        lambda row: adjacent_pair(row["messages"]),
        keyword_pair(("legal", "law", "court", "contract", "regulation")))
    scored += builder.collect(
        "ood_test", "finance_business", "ultrachat",
        lambda row: adjacent_pair(row["messages"]),
        keyword_pair(("finance", "accounting", "investment", "business", "econom")))
    scored += builder.collect(
        "ood_test", "humanities_history", "ultrachat",
        lambda row: adjacent_pair(row["messages"]),
        keyword_pair(("history", "philosophy", "literature", "culture", "religion")))
    scored += builder.collect(
        "ood_test", "medicine_health", "science_rqa",
        lambda row: adjacent_pair(row["messages"]))
    scored += builder.collect(
        "ood_test", "advanced_mathematics", "math",
        lambda row: adjacent_pair(row["messages"]),
        lambda row: row.get("data_source") == "StackExchange-Math")
    scored += builder.collect_code_files(
        "ood_test", "nonpython_code", "pybind11_code",
        SITE_PACKAGES / "torch/include/pybind11", "*.h")
    scored += builder.collect(
        "ood_test", "creative_writing", "dolly", dolly_pair,
        lambda row: row.get("category") == "creative_writing")
    scored += builder.collect(
        "ood_test", "technical_web", "ultrachat",
        lambda row: adjacent_pair(row["messages"]),
        keyword_pair(("network", "database", "security", "server", "software")))

    # Preserve calibration order as well as row contents so quantization
    # runtimes and order-sensitive calibration methods remain comparable.
    scored.sort(key=lambda row: (row["split"], row["domain"], row["prompt_id"]))
    records = calibration + scored
    expected = {"calibration": 128, "validation": 128,
                "id_test": 128, "ood_test": 128}
    counts = Counter(row["split"] for row in records)
    if dict(counts) != expected:
        raise RuntimeError(f"wrong split counts: {counts}")
    family_counts = {
        split: dict(sorted(Counter(
            row["domain"] for row in records if row["split"] == split).items()))
        for split in expected
    }
    for split in ("validation", "id_test", "ood_test"):
        if set(family_counts[split].values()) != {TARGET_PER_FAMILY}:
            raise RuntimeError(f"unbalanced {split}: {family_counts[split]}")
    payload = {
        "format": 2,
        "model": "LiquidAI/LFM2.5-230M",
        "model_revision": old["model_revision"],
        "calibration_policy": {
            "source": str(OLD_DATA), "rows_unchanged": True,
            "conversations": 128, "total_tokens": 32621,
        },
        "scored_target_policy": {
            "origin": "pinned public reference completions",
            "assistant_token_range": [MIN_ASSISTANT_TOKENS, MAX_ASSISTANT_TOKENS],
            "max_conversation_tokens": 512,
            "aggregation": "mean per-conversation assistant-token NLL",
        },
        "counts": expected,
        "family_counts": family_counts,
        "sources": SOURCES,
        "records": records,
        "provenance": builder.provenance,
    }
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    audit = {
        "dataset_status": "canonical",
        "output": str(OUTPUT), "sha256": digest, "counts": expected,
        "calibration_rows_equal_to_previous_canonical": True,
        "scored_mix": {
            "validation_and_id": "62.5% SFT/reference responses, 37.5% permissively licensed code/encyclopedic continuation",
            "ood": "87.5% held-out-domain SFT/reference responses, 12.5% BSD-licensed C++ headers",
        },
        "family_counts": family_counts,
        "splits": {},
    }
    for split in expected:
        rows = [row for row in records if row["split"] == split]
        audit["splits"][split] = {
            "conversations": len(rows),
            "total_tokens": sum(row["total_tokens"] for row in rows),
            "assistant_tokens": sum(row.get("assistant_tokens", 0) for row in rows),
            "min_assistant_tokens": min((row.get("assistant_tokens", 0) for row in rows), default=0),
            "max_assistant_tokens": max((row.get("assistant_tokens", 0) for row in rows), default=0),
            "target_origins": dict(Counter(row.get("target_origin", "model_generated") for row in rows)),
        }
    old_by_split = {
        split: [row for row in old["records"] if row["split"] == split]
        for split in expected
    }
    audit["comparison_to_previous"] = {
        split: {
            "old_assistant_tokens": sum(row.get("assistant_tokens", 0)
                                        for row in old_by_split[split]),
            "new_assistant_tokens": audit["splits"][split]["assistant_tokens"],
            "assistant_token_multiplier": round(
                audit["splits"][split]["assistant_tokens"] /
                sum(row.get("assistant_tokens", 0)
                    for row in old_by_split[split]), 4),
        }
        for split in expected
    }
    audit_path = Path(__file__).with_name("lfm25_hard_eval_dataset_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
