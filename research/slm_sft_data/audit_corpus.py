#!/usr/bin/env python3
"""Audit generated SFT conversations and report final-split viability."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from generate_responses import (GENERATED, MAX_CONVERSATION_TOKENS,
                                SEMANTIC_REVIEW_FAMILIES, read_jsonl)


MODELS = ("qwen25", "qwen3", "qwen35")
REVIEW = GENERATED / "quality_review.csv"

TARGETS = {
    "train_development": {
        "general_chat_writing": 8,
        "code_agent_tools": 8,
        "math_quantitative": 8,
        "science_technical": 8,
    },
    "test_overlapping": {
        "general_chat_writing": 16,
        "code_agent_tools": 16,
        "math_quantitative": 16,
        "science_technical": 16,
    },
    "test_heldout": {
        "business_operations": 8,
        "finance_accounting_economics": 8,
        "legal_policy_compliance": 8,
        "medicine_health": 8,
        "cybersecurity_infrastructure": 8,
        "humanities_social_sciences": 8,
        "creative_design_storytelling": 8,
        "multilingual_translation": 8,
    },
}


def latest_rows(model: str) -> dict[str, dict]:
    latest = {}
    for row in read_jsonl(GENERATED / "raw" / f"{model}.jsonl"):
        latest[row["candidate_id"]] = row
    return latest


def assistant_text(row: dict) -> str:
    return "\n\n--- NEXT ASSISTANT TURN ---\n\n".join(
        message["content"] for message in row["messages"]
        if message["role"] == "assistant")


def lint(row: dict) -> list[str]:
    """Conservative semantic/format flags for later human or LLM review."""
    prompt = row["messages"][1]["content"]
    output = assistant_text(row)
    flags = []
    final = [m["content"] for m in row["messages"]
             if m["role"] == "assistant"][-1]

    limit_match = re.search(
        r"(?:at most|no more than|under)\s+(\d+)\s+words?", prompt, re.I)
    if limit_match:
        limit = int(limit_match.group(1))
        words = len(re.findall(r"\b\w+(?:[-']\w+)*\b", final))
        if words > limit:
            flags.append(f"word_limit:{words}>{limit}")

    if re.search(r"(?:only|provide only|return only)\s+valid JSON", prompt, re.I):
        candidate = final.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate,
                               flags=re.I | re.S)
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            flags.append("invalid_requested_json")

    if re.search(r"\[(?:insert|your|supplier|name|date|position)[^]]*\]", output,
                 re.I):
        flags.append("unresolved_placeholder")
    if len(re.findall(r"\b(?:sorry|apologize|apologies)\b", output, re.I)) > 4:
        flags.append("possible_apology_loop")
    if max(row["quality"]["token_counts"].values()) > MAX_CONVERSATION_TOKENS:
        flags.append("token_limit_metadata_violation")
    if any(output.get("hit_generation_cap")
           for output in row.get("generation_outputs", ())):
        flags.append("generation_cap_without_eos")
    # Hard spot validators for arithmetic failures discovered during raw
    # review. They supplement, rather than replace, independent semantic
    # review over every math/code/science and high-stakes conversation.
    candidate_id = row["candidate_id"]
    compact = re.sub(r"[,$*`_\\]", "", output.lower())
    if candidate_id == "overlap_math_quantitative_06":
        if not re.search(r"\b12(?:\.0+)?\s*(?:ml|milliliters?)\b", compact):
            flags.append("known_answer_missing:12_mL")
        if re.search(r"\b300(?:\.0+)?\s*(?:ml|milliliters?)\b", compact):
            flags.append("known_answer_wrong:300_mL")
    if candidate_id == "overlap_math_quantitative_07":
        if not re.search(r"\b2464(?:\.0+)?\b", compact):
            flags.append("known_answer_missing:2464")
        if re.search(r"\b44\s+(?:dollars?\s+)?per\s+year\b", compact):
            flags.append("known_answer_wrong:44_per_year")
    if not row["quality"]["accepted"]:
        flags.extend("surface:" + reason
                     for reason in row["quality"]["rejection_reasons"])
    return sorted(set(flags))


def read_annotations() -> dict[tuple[str, str], tuple[str, str]]:
    if not REVIEW.exists():
        return {}
    result = {}
    with REVIEW.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("review") or row.get("review_reason"):
                result[(row["model"], row["candidate_id"])] = (
                    row.get("review", ""), row.get("review_reason", ""))
    return result


def read_judgments(model: str) -> dict[str, dict]:
    path = GENERATED / "judge" / f"{model}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {row["candidate_id"]: row for row in payload.get("reviews", [])}


def target_group(row: dict) -> str:
    if row["pool"] == "train_development":
        return "train_development"
    return ("test_overlapping" if row["domain_relation"] == "overlapping"
            else "test_heldout")


def quota_report(rows_by_model: dict[str, dict], judgments: dict[str, dict],
                 required_models: tuple[str, ...], groups: tuple[str, ...]) -> dict:
    common_ids = None
    for model in required_models:
        accepted_ids = {
            key for key, row in rows_by_model[model].items()
            if row["quality"]["accepted"]
        }
        common_ids = accepted_ids if common_ids is None else common_ids & accepted_ids
    common_ids = common_ids or set()
    representative = rows_by_model[required_models[0]]
    available = Counter()
    lint_clean = Counter()
    judge_accepted = Counter()
    for candidate_id in common_ids:
        row = representative[candidate_id]
        group = target_group(row)
        if group not in groups:
            continue
        key = (group, row["family"])
        available[key] += 1
        if all(not lint(rows_by_model[model][candidate_id])
               for model in required_models):
            lint_clean[key] += 1
        if all(judgments[model].get(candidate_id, {}).get("verdict") == "accept"
               for model in required_models):
            judge_accepted[key] += 1
    details = {}
    viable = True
    for group in groups:
        for family, needed in TARGETS[group].items():
            key = (group, family)
            surface = available[key]
            clean = lint_clean[key]
            judged = judge_accepted[key]
            if surface < needed:
                viable = False
            details[f"{group}|{family}"] = {
                "target": needed,
                "surface_accepted": surface,
                "lint_clean": clean,
                "independent_judge_accepted": judged,
                "surface_viable": surface >= needed,
                "lint_clean_viable": clean >= needed,
                "independent_judge_viable": judged >= needed,
            }
    return {
        "required_models": list(required_models),
        "surface_viable": viable,
        "groups": details,
    }


def main() -> None:
    rows_by_model = {model: latest_rows(model) for model in MODELS}
    judgments = {model: read_judgments(model) for model in MODELS}
    annotations = read_annotations()
    review_rows = []
    summaries = {}
    for model in MODELS:
        rows = rows_by_model[model]
        flag_counts = Counter()
        for candidate_id in sorted(rows):
            row = rows[candidate_id]
            flags = lint(row)
            flag_counts.update(flags)
            annotation = annotations.get((model, candidate_id), ("", ""))
            judgment = judgments[model].get(candidate_id, {})
            review_rows.append({
                "model": model,
                "candidate_id": candidate_id,
                "pool": row["pool"],
                "domain_relation": row["domain_relation"],
                "family": row["family"],
                "surface_accepted": str(row["quality"]["accepted"]).lower(),
                "semantic_review_required": str(
                    row["family"] in SEMANTIC_REVIEW_FAMILIES).lower(),
                "lint_flags": "|".join(flags),
                "judge_verdict": judgment.get("verdict", ""),
                "judge_score": judgment.get("score", ""),
                "judge_reasons": "|".join(judgment.get("reasons", [])),
                "review": annotation[0],
                "review_reason": annotation[1],
                "prompt": row["messages"][1]["content"].replace("\n", "\\n"),
                "assistant": assistant_text(row).replace("\n", "\\n"),
            })
        summaries[model] = {
            "generated_candidates": len(rows),
            "surface_accepted": sum(
                row["quality"]["accepted"] for row in rows.values()),
            "semantic_review_required": sum(
                row["family"] in SEMANTIC_REVIEW_FAMILIES
                for row in rows.values()),
            "lint_clean": sum(
                row["quality"]["accepted"] and not lint(row)
                for row in rows.values()),
            "lint_flags": dict(flag_counts),
            "independently_judged": len(judgments[model]),
            "independent_judge_accepted": sum(
                row["verdict"] == "accept" for row in judgments[model].values()),
        }

    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model", "candidate_id", "pool", "domain_relation", "family",
            "surface_accepted", "semantic_review_required", "lint_flags",
            "judge_verdict", "judge_score", "judge_reasons", "review",
            "review_reason", "prompt", "assistant",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    report = {
        "models": summaries,
        "task_qwen25_to_qwen3": quota_report(
            rows_by_model, judgments, ("qwen25",), ("train_development",))
        if rows_by_model["qwen25"] else None,
        "task_qwen25_to_qwen3_test": quota_report(
            rows_by_model, judgments, ("qwen25", "qwen3"),
            ("test_overlapping", "test_heldout"))
        if rows_by_model["qwen25"] and rows_by_model["qwen3"] else None,
        "task_qwen35": quota_report(
            rows_by_model, judgments, ("qwen35",),
            ("train_development", "test_overlapping", "test_heldout"))
        if rows_by_model["qwen35"] else None,
        "review_csv": str(REVIEW),
        "note": (
            "Surface checks and lint are not semantic correctness review. "
            "Set review=accept/reject in quality_review.csv after manual or "
            "independent-model review before selecting sealed splits."),
    }
    (GENERATED / "corpus_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
