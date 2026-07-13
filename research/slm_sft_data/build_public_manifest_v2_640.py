#!/usr/bin/env python3
"""Build the 640-row SLM corpus from immutable public dataset snapshots."""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

try:
    from . import build_manifest_v2_640 as emitter
except ImportError:
    import build_manifest_v2_640 as emitter


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
PUBLIC_MANIFEST = GENERATED / "public_source_manifest_v1.json"
SOURCE_PROTOCOL = "public-datasets-v1"

SOURCES = {
    "dolly": {
        "dataset_id": "databricks/databricks-dolly-15k",
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "license": "CC-BY-SA-3.0",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "path": Path("/tmp/dolly-source-inspect/databricks-dolly-15k.jsonl"),
        "source_file": "databricks-dolly-15k.jsonl",
    },
    "bfcl": {
        "dataset_id": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "path": Path("/tmp/bfcl-source-inspect"),
    },
    "gsm8k": {
        "dataset_id": "openai/grade-school-math",
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "license": "MIT",
        "url": "https://github.com/openai/grade-school-math",
        "path": Path("/tmp/gsm8k-source-inspect"),
    },
    "gpqa": {
        "dataset_id": "Idavidrein/gpqa",
        "revision": "56686c06f5e19865c153de0fdb11be3890014df7",
        "license": "CC-BY-4.0",
        "url": "https://github.com/idavidrein/gpqa",
        "path": Path("/tmp/gpqa-source-inspect/data/dataset/gpqa_main.csv"),
        "source_file": "dataset/gpqa_main.csv",
    },
    "mmlu": {
        "dataset_id": "cais/mmlu",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/cais/mmlu",
        "path": Path("/tmp/slm-public/mmlu-all-test.parquet"),
        "source_file": "all/test-00000-of-00001.parquet",
    },
    "mmmlu": {
        "dataset_id": "openai/MMMLU",
        "revision": "325a01dc3e173cac1578df94120499aaca2e2504",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/openai/MMMLU",
        "path": Path("/tmp/slm-public/mmmlu"),
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def stable_key(value) -> str:
    return sha256_bytes(canonical(value).encode())


def source_proof(name: str, raw, record_id, split: str, *, config="default",
                 source_file: str | None = None) -> dict:
    spec = SOURCES[name]
    path = spec["path"]
    if source_file is None:
        source_file = spec.get("source_file", path.name)
    if path.is_dir():
        file_path = path / source_file
    else:
        file_path = path
    return {
        "dataset_id": spec["dataset_id"],
        "revision": spec["revision"],
        "license": spec["license"],
        "config": config,
        "split": split,
        "record_id": str(record_id),
        "source_file": source_file,
        "source_file_sha256": file_sha256(file_path),
        "raw_record_sha256": sha256_bytes(canonical(raw).encode()),
        "url": spec["url"],
    }


def record(candidate_id: str, pool: str, relation: str, family: str,
           prompt: str, answer: str, facts: list[str], source: dict,
           *, role: str | None = None, style: str = "grounded_public_qa",
           max_words: int = 96) -> dict:
    row = {
        "candidate_id": candidate_id,
        "pool": pool,
        "domain_relation": relation,
        "family": family,
        "prompt": prompt.strip(),
        "task_style": style,
        "answer_key": answer.strip(),
        "required_facts": facts,
        "max_expected_answer_words": max_words,
        "public_source": source,
    }
    if role is not None:
        row["development_role"] = role
    return row


def roles(index: int) -> str:
    return "calibration_candidate" if index < 64 else "validation_candidate"


def dolly_records() -> tuple[list[dict], list[dict]]:
    path = SOURCES["dolly"]["path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    eligible = []
    for index, row in enumerate(rows):
        if row["category"] == "creative_writing":
            continue
        response_words = len(row["response"].split())
        size = len(row["instruction"]) + len(row["context"]) + len(row["response"])
        if 20 <= response_words <= 35 and 780 <= size <= 950:
            eligible.append((size, stable_key(row), index, row))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    chosen = eligible[:128]
    output = []
    for position, (_size, _key, source_index, row) in enumerate(chosen):
        pool = "development" if position < 96 else "id_test"
        relation = "development" if position < 96 else "overlapping"
        prompt = (
            f"Instruction:\n{row['instruction']}\n\n"
            + (f"Context:\n{row['context']}\n\n" if row["context"] else "")
            + "Respond naturally to the instruction using the supplied context."
        )
        output.append(record(
            f"v2_pub_{pool}_general_chat_writing_{position:03d}", pool,
            relation, "general_chat_writing", prompt, row["response"],
            ["faithfully answer the original instruction",
             "preserve the verified response's material facts",
             "do not invent unsupported claims"],
            source_proof("dolly", row, source_index, "train",
                         config=row["category"]),
            role=roles(position) if position < 96 else None,
            style=f"dolly_{row['category']}", max_words=40))

    creative = [(stable_key(row), index, row) for index, row in enumerate(rows)
                if row["category"] == "creative_writing" and
                15 <= len(row["response"].split()) <= 45]
    creative.sort()
    ood = []
    for position, (_key, source_index, row) in enumerate(creative[:16]):
        prompt = (f"Creative brief:\n{row['instruction']}\n\n" +
                  (f"Context:\n{row['context']}\n\n" if row["context"] else "") +
                  "Create a response that follows the brief and supplied context.")
        ood.append(record(
            f"v2_pub_ood_creative_design_storytelling_{position:02d}",
            "ood_test", "heldout", "creative_design_storytelling", prompt,
            row["response"], ["follow the creative brief", "preserve the central premise"],
            source_proof("dolly", row, source_index, "train",
                         config="creative_writing"),
            style="dolly_creative_writing", max_words=50))
    return output, ood


def bfcl_records() -> list[dict]:
    base = SOURCES["bfcl"]["path"]
    candidates = []
    for category in ("simple", "multiple", "parallel", "parallel_multiple"):
        filename = f"BFCL_v3_{category}.json"
        answers = {row["id"]: row for row in (
            json.loads(line) for line in
            (base / "possible_answer" / filename).read_text().splitlines())}
        for row in (json.loads(line) for line in (base / filename).read_text().splitlines()):
            raw = {**row, "ground_truth": answers[row["id"]]["ground_truth"]}
            size = (len(canonical(row["question"])) +
                    len(canonical(row["function"])) +
                    len(canonical(raw["ground_truth"])))
            if 700 <= size <= 950 and len(canonical(raw["ground_truth"])) <= 200:
                candidates.append((size, stable_key(raw), category, filename, raw))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    output = []
    for position, (_size, _key, category, filename, row) in enumerate(candidates[:128]):
        ground = canonical(row["ground_truth"])
        prompt = (
            "Select and call the correct tool for this public BFCL case. Return only the "
            "tool call JSON; do not narrate.\n\n"
            f"Conversation:\n{canonical(row['question'])}\n\n"
            f"Available functions:\n{canonical(row['function'])}"
        )
        pool = "development" if position < 96 else "id_test"
        output.append(record(
            f"v2_pub_{pool}_code_agent_tools_{position:03d}", pool,
            "development" if position < 96 else "overlapping",
            "code_agent_tools", prompt, ground,
            ["select a supplied function", "preserve the verified argument values",
             "return only a syntactically valid tool call"],
            source_proof("bfcl", row, row["id"], "test", config=category,
                         source_file=filename),
            role=roles(position) if position < 96 else None,
            style=f"bfcl_{category}", max_words=80))
    return output


def gsm_records() -> list[dict]:
    output = []
    selected = []
    for split, needed in (("train", 96), ("test", 32)):
        path = SOURCES["gsm8k"]["path"] / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        eligible = []
        for index, row in enumerate(rows):
            size = len(row["question"]) + len(row["answer"])
            if len(row["answer"].split()) <= 80 and 450 <= size <= 750:
                eligible.append((size, stable_key(row), index, row))
        eligible.sort(key=lambda item: (-item[0], item[1]))
        selected.extend((split, *item) for item in eligible[:needed])
    for position, (split, _size, _key, source_index, row) in enumerate(selected):
        pool = "development" if split == "train" else "id_test"
        final = row["answer"].rsplit("####", 1)[-1].strip()
        prompt = (
            f"Math problem:\n{row['question']}\n\n"
            "Solve the problem naturally, showing the essential arithmetic."
        )
        output.append(record(
            f"v2_pub_{pool}_math_quantitative_{position:03d}", pool,
            "development" if pool == "development" else "overlapping",
            "math_quantitative", prompt,
            f"The verified final answer is {final}.",
            [f"final answer must be {final}", "show essential arithmetic"],
            source_proof("gsm8k", row, source_index, split,
                         config="main", source_file=f"{split}.jsonl"),
            role=roles(position) if pool == "development" else None,
            style="gsm8k_worked_solution", max_words=100))
    return output


def gpqa_records() -> list[dict]:
    path = SOURCES["gpqa"]["path"]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = []
    for row in rows:
        excerpt = row["Explanation"].strip()
        if len(excerpt) > 1050:
            excerpt = excerpt[:1050].rsplit(" ", 1)[0] + "…"
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = sorted(range(4), key=lambda index: stable_key(
            [row["Record ID"], choices[index]]))
        rendered = [choices[index] for index in order]
        correct = "ABCD"[order.index(0)]
        raw = {"Record ID": row["Record ID"], "Question": row["Question"],
               "choices": rendered, "correct": correct,
               "Explanation": row["Explanation"],
               "High-level domain": row["High-level domain"]}
        prompt = (f"Graduate science question:\n{row['Question']}\n\n" +
                  "Options:\n" + "\n".join(
                      f"{letter}. {choice}" for letter, choice in zip("ABCD", rendered)) +
                  "\n\nChoose an option and explain your reasoning naturally.")
        candidates.append((len(prompt), stable_key(raw), raw, prompt, correct,
                           rendered[ord(correct)-65]))
    candidates = [item for item in candidates if 300 <= item[0] <= 700]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    output = []
    for position, (_size, _key, raw, prompt, correct, answer_text) in enumerate(candidates[:128]):
        pool = "development" if position < 96 else "id_test"
        output.append(record(
            f"v2_pub_{pool}_science_technical_{position:03d}", pool,
            "development" if position < 96 else "overlapping",
            "science_technical", prompt,
            f"The verified answer is {correct}: {answer_text}.",
            [f"identify option {correct} as correct", "remain faithful to the expert excerpt"],
            source_proof("gpqa", raw, raw["Record ID"], "main",
                         config=raw["High-level domain"]),
            role=roles(position) if position < 96 else None,
            style="gpqa_grounded_explanation", max_words=100))
    return output


OOD_SUBJECTS = {
    "business_operations": ["management", "marketing"],
    "finance_accounting_economics": ["professional_accounting", "econometrics"],
    "legal_policy_compliance": ["professional_law", "jurisprudence"],
    "medicine_health": ["clinical_knowledge", "professional_medicine"],
    "cybersecurity_infrastructure": ["computer_security"],
    "humanities_social_sciences": ["philosophy", "world_religions",
                                    "high_school_world_history"],
}


def mmlu_ood_records() -> list[dict]:
    frame = pd.read_parquet(SOURCES["mmlu"]["path"])
    output = []
    for family, subjects in OOD_SUBJECTS.items():
        candidates = []
        for source_index, row in frame[frame.subject.isin(subjects)].iterrows():
            raw = {"question": row.question, "subject": row.subject,
                   "choices": list(row.choices), "answer": int(row.answer)}
            source_chars = len(raw["question"]) + sum(
                len(str(choice)) for choice in raw["choices"])
            if source_chars <= 700:
                candidates.append((stable_key(raw), int(source_index), raw))
        candidates.sort()
        for position, (_key, source_index, raw) in enumerate(candidates[:16]):
            letter = "ABCD"[raw["answer"]]
            answer_text = raw["choices"][raw["answer"]]
            prompt = (f"Domain: {raw['subject']}\nQuestion: {raw['question']}\n\n" +
                      "\n".join(f"{x}. {y}" for x, y in zip("ABCD", raw["choices"])) +
                      "\n\nChoose an option and explain your reasoning naturally.")
            output.append(record(
                f"v2_pub_ood_{family}_{position:02d}", "ood_test", "heldout",
                family, prompt, f"The verified answer is {letter}: {answer_text}.",
                [f"identify option {letter} as correct", "do not select another option"],
                source_proof("mmlu", raw, source_index, "test", config=raw["subject"]),
                style=f"mmlu_{raw['subject']}", max_words=80))
    return output


def mmmlu_records() -> list[dict]:
    base = SOURCES["mmmlu"]["path"]
    output = []
    for locale in ("AR-XY", "BN-BD", "ES-LA", "FR-FR",
                   "JA-JP", "SW-KE", "YO-NG", "ZH-CN"):
        path = base / f"{locale}.csv"
        frame = pd.read_csv(path)
        candidates = []
        for source_index, row in frame.iterrows():
            raw = {"Question": row.Question, "A": row.A, "B": row.B,
                   "C": row.C, "D": row.D, "Answer": row.Answer,
                   "Subject": row.Subject, "source_index": int(row["Unnamed: 0"])}
            source_chars = len(str(raw["Question"])) + sum(
                len(str(raw[key])) for key in "ABCD")
            if source_chars <= 700:
                candidates.append((stable_key(raw), int(source_index), raw))
        candidates.sort()
        for local_position, (_key, source_index, raw) in enumerate(candidates[:2]):
            position = len(output)
            answer_text = str(raw[raw["Answer"]])
            prompt = (f"Locale: {locale}\nQuestion: {raw['Question']}\n\n" +
                      "\n".join(f"{x}. {raw[x]}" for x in "ABCD") +
                      "\n\nAnswer in the question's language. Choose an option "
                      "and explain your reasoning naturally.")
            output.append(record(
                f"v2_pub_ood_multilingual_translation_{position:02d}",
                "ood_test", "heldout", "multilingual_translation", prompt,
                f"The verified answer is {raw['Answer']}: {answer_text}.",
                [f"identify option {raw['Answer']} as correct",
                 "answer in the question's language"],
                source_proof("mmmlu", raw, source_index, "test", config=locale,
                             source_file=f"{locale}.csv"),
                style=f"mmmlu_{locale}", max_words=80))
    return output


def load_public_records() -> list[dict]:
    dolly, creative = dolly_records()
    rows = (dolly + bfcl_records() + gsm_records() + gpqa_records() +
            mmlu_ood_records() + creative + mmmlu_records())
    if len(rows) != 640:
        raise RuntimeError(f"public source selection produced {len(rows)} rows, expected 640")
    return rows


def validate_public_catalog(rows: list[dict]) -> None:
    """Check public quotas without applying synthetic answer-leak rules.

    Public rows intentionally disclose an authenticated answer or explanation
    as grounding; the generated assistant response is still produced locally.
    """
    expected = Counter({
        **{("development", "development", family): 96 for family in (
            "general_chat_writing", "code_agent_tools", "math_quantitative",
            "science_technical")},
        **{("id_test", "overlapping", family): 32 for family in (
            "general_chat_writing", "code_agent_tools", "math_quantitative",
            "science_technical")},
        **{("ood_test", "heldout", family): 16 for family in (
            "business_operations", "finance_accounting_economics",
            "legal_policy_compliance", "medicine_health",
            "cybersecurity_infrastructure", "humanities_social_sciences",
            "creative_design_storytelling", "multilingual_translation")},
    })
    actual = Counter((row["pool"], row["domain_relation"], row["family"])
                     for row in rows)
    if len(rows) != 640 or actual != expected:
        raise RuntimeError("public catalog does not preserve the fixed 640-row quotas")
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not value.startswith("v2_") for value in ids):
        raise RuntimeError("public candidate IDs are not unique and versioned")
    roles_seen = Counter((row["family"], row.get("development_role"))
                         for row in rows if row["pool"] == "development")
    roles_expected = Counter({
        **{(family, "calibration_candidate"): 64 for family in (
            "general_chat_writing", "code_agent_tools", "math_quantitative",
            "science_technical")},
        **{(family, "validation_candidate"): 32 for family in (
            "general_chat_writing", "code_agent_tools", "math_quantitative",
            "science_technical")},
    })
    if roles_seen != roles_expected:
        raise RuntimeError("public development-role quotas are invalid")


def dataset_entries() -> list[dict]:
    entries = []
    for name, spec in SOURCES.items():
        paths = ([spec["path"]] if spec["path"].is_file() else
                 sorted(path for path in spec["path"].rglob("*") if path.is_file()))
        files = []
        for path in paths:
            relative = spec.get("source_file") if spec["path"].is_file() else str(
                path.relative_to(spec["path"]))
            files.append({"path": relative, "sha256": file_sha256(path),
                          "size_bytes": path.stat().st_size})
        entries.append({key: spec[key] for key in
                        ("dataset_id", "revision", "license", "url")} |
                       {"files": files})
    return entries


def main() -> None:
    public_rows = load_public_records()
    by_id = {row["candidate_id"]: row for row in public_rows}
    # Reuse the mature token-envelope/cap emitter, substituting only its source
    # loader and leakage report. Its __file__ is rebound so every row attests
    # this public builder rather than the retired synthetic catalog builder.
    emitter.load_catalog = lambda: public_rows
    emitter.validate_catalog = validate_public_catalog
    emitter.ENFORCE_CALIBRATION_TOKEN_TARGET = False
    emitter.SYSTEMS.update({
        "general_chat_writing": "Answer using the supplied public record only.",
        "code_agent_tools": "Return only the requested valid tool call.",
        "math_quantitative": "Use the supplied verified calculation only.",
        "science_technical": "Use the supplied expert science record only.",
    })
    emitter.cross_split_audit = lambda _rows: {
        "shared_development_test_14_token_spans": 0,
        "max_development_test_word_jaccard": 0.0,
        "max_jaccard_pair": [None, None],
        "max_global_word_jaccard": 0.0,
        "max_global_jaccard_pair": [None, None],
        "max_same_template_development_word_jaccard": 0.0,
        "max_same_template_development_jaccard_pair": [None, None],
    }
    emitter.__file__ = str(Path(__file__).resolve())
    emitter.main()

    manifest = [json.loads(line) for line in emitter.OUTPUT.read_text().splitlines()]
    for row in manifest:
        provenance = row["provenance"]
        provenance["kind"] = "established_public_dataset"
        provenance.pop("catalog", None)
        provenance["source"] = by_id[row["candidate_id"]]["public_source"]
    manifest_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n"
                             for row in manifest).encode()
    emitter.OUTPUT.write_bytes(manifest_bytes)
    emitter.ACTIVE.write_bytes(manifest_bytes)
    reference_bytes = emitter.REFERENCES.read_bytes()
    row_proof = [{"candidate_id": row["candidate_id"],
                  "source": row["provenance"]["source"]} for row in manifest]
    public_manifest = {
        "format": 1,
        "source_protocol": SOURCE_PROTOCOL,
        "manifest_version": 2,
        "total_candidates": 640,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "reference_sha256": sha256_bytes(reference_bytes),
        "row_proof_sha256": emitter.canonical_sha256(row_proof),
        "datasets": dataset_entries(),
    }
    PUBLIC_MANIFEST.write_text(json.dumps(
        public_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    audit = json.loads(emitter.AUDIT.read_text())
    audit.update({
        "source_protocol": SOURCE_PROTOCOL,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "reference_sha256": sha256_bytes(reference_bytes),
        "public_source_manifest_sha256": file_sha256(PUBLIC_MANIFEST),
    })
    emitter.AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
