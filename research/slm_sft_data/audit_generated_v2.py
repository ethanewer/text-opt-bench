#!/usr/bin/env python3
"""Report v2 generation/semantic-quota and calibration-token feasibility."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from generate_responses import (GENERATED, RAW, generation_backend_is_canonical,
                                read_jsonl, versioned_directory)
from run_quality_judges import (QUALITY_GATES, conversation_sha256,
                                latest as validated_generation_rows,
                                load_validated_judge_aggregate)
try:
    from .pipeline_contract import require_current_reference_audit
except ImportError:  # Direct script execution.
    from pipeline_contract import require_current_reference_audit


MANIFEST_VERSION = 2
MANIFEST = GENERATED / "prompt_candidates_v2.jsonl"
REPORT = GENERATED / "corpus_audit_v2.json"

REQUIRED_GENERATIONS = {"qwen25": 640, "qwen3": 256, "qwen35": 640}
DEVELOPMENT_FAMILIES = (
    "general_chat_writing", "code_agent_tools", "math_quantitative",
    "science_technical",
)
OOD_FAMILIES = (
    "business_operations", "finance_accounting_economics",
    "legal_policy_compliance", "medicine_health",
    "cybersecurity_infrastructure", "humanities_social_sciences",
    "creative_design_storytelling", "multilingual_translation",
)


def latest(model: str) -> dict[str, dict]:
    path = versioned_directory(RAW, MANIFEST_VERSION) / f"{model}.jsonl"
    rows = {}
    for row in read_jsonl(path):
        rows[row["candidate_id"]] = row
    return rows


def judgments(model: str) -> dict[str, dict]:
    path = versioned_directory(GENERATED / "judge", MANIFEST_VERSION) / f"{model}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {row["candidate_id"]: row for row in payload["reviews"]}


def semantic_ids(rows: dict[str, dict], reviews: dict[str, dict]) -> set[str]:
    result = set()
    for candidate_id, row in rows.items():
        review = reviews.get(candidate_id)
        backend = row.get("generation_backend", {})
        if (generation_backend_is_canonical(backend) and
                row["quality"]["accepted"] and review and
                review["verdict"] == "accept" and review["score"] >= 4 and
                set(review.get("gates", {})) == QUALITY_GATES and
                all(review["gates"].values()) and
                review.get("conversation_sha256") == conversation_sha256(row) and
                review.get("generation_input_sha256") ==
                row.get("provenance", {}).get("generation_input_sha256") and
                review.get("manifest_row_sha256") ==
                row.get("provenance", {}).get("manifest_row_sha256") and
                review.get("quality_reference_sha256") ==
                row.get("provenance", {}).get("reference_sha256")):
            result.add(candidate_id)
    return result


def quota_counts(manifest: dict[str, dict], ids: set[str]) -> dict[str, int]:
    return {"|".join(key): value for key, value in sorted(Counter(
        (manifest[candidate_id]["pool"],
         manifest[candidate_id]["family"])
        for candidate_id in ids).items())}


def quota_viability(manifest: dict[str, dict], ids: set[str],
                    include_development: bool = True) -> dict:
    requirements = {
        **{("id_test", family): 16 for family in DEVELOPMENT_FAMILIES},
        **{("ood_test", family): 8 for family in OOD_FAMILIES},
    }
    if include_development:
        requirements.update({
            ("development", family): 48 for family in DEVELOPMENT_FAMILIES})
    counts = Counter((manifest[candidate_id]["pool"],
                      manifest[candidate_id]["family"])
                     for candidate_id in ids)
    detail = {"|".join(key): {
        "semantic_accepted": counts[key],
        "required": required,
        "viable": counts[key] >= required,
    } for key, required in sorted(requirements.items())}
    return {
        "viable": all(item["viable"] for item in detail.values()),
        "groups": detail,
    }


def paired_test_viability(manifest: dict[str, dict], left: set[str],
                          right: set[str]) -> dict:
    paired = {candidate_id for candidate_id in left & right
              if manifest[candidate_id]["pool"] != "development"}
    requirements = {
        **{("id_test", family): 16 for family in DEVELOPMENT_FAMILIES},
        **{("ood_test", family): 8 for family in OOD_FAMILIES},
    }
    counts = Counter((manifest[candidate_id]["pool"],
                      manifest[candidate_id]["family"])
                     for candidate_id in paired)
    detail = {"|".join(key): {
        "paired_semantic_accepted": counts[key],
        "required": required,
        "viable": counts[key] >= required,
    } for key, required in sorted(requirements.items())}
    return {
        "viable": all(item["viable"] for item in detail.values()),
        "paired_candidates": len(paired),
        "groups": detail,
    }


def development_viability(manifest: dict[str, dict], ids: set[str]) -> dict:
    counts = Counter((manifest[candidate_id]["family"],
                      manifest[candidate_id].get("development_role"))
                     for candidate_id in ids
                     if manifest[candidate_id]["pool"] == "development")
    detail = {}
    for family in DEVELOPMENT_FAMILIES:
        for role, required in (("calibration_candidate", 32),
                               ("validation_candidate", 16)):
            count = counts[(family, role)]
            detail[f"{family}|{role}"] = {
                "jointly_semantic_accepted": count,
                "required": required,
                "viable": count >= required,
            }
    return {
        "viable": all(item["viable"] for item in detail.values()),
        "groups": detail,
    }


def calibration_token_feasibility(manifest: dict[str, dict], ids: set[str],
                                  token_value) -> dict:
    """Range for 32 calibration-only rows/family; this does not select rows."""
    family_ranges = {}
    minimum_total = 0
    maximum_total = 0
    for family in DEVELOPMENT_FAMILIES:
        values = sorted(
            token_value(candidate_id)
            for candidate_id in ids
            if manifest[candidate_id]["pool"] == "development" and
            manifest[candidate_id].get("development_role") ==
            "calibration_candidate" and
            manifest[candidate_id]["family"] == family)
        if len(values) < 32:
            family_ranges[family] = {
                "available": len(values), "required": 32, "viable": False}
            continue
        low, high = sum(values[:32]), sum(values[-32:])
        minimum_total += low
        maximum_total += high
        family_ranges[family] = {
            "available": len(values),
            "required": 32,
            "minimum_32_tokens": low,
            "maximum_32_tokens": high,
            "viable": True,
        }
    enough = all(item["viable"] for item in family_ranges.values())
    intersects = enough and maximum_total >= 50_000 and minimum_total <= 65_536
    return {
        "target_tokens": [50_000, 65_536],
        "minimum_possible_tokens": minimum_total if enough else None,
        "maximum_possible_tokens": maximum_total if enough else None,
        "target_range_intersection": intersects,
        "families": family_ranges,
        "note": (
            "Feasibility for 128 calibration-only rows; these rows receive no "
            "optimization score. Deterministic selection is external."),
    }


def main() -> None:
    contract = require_current_reference_audit(MANIFEST)
    manifest_rows = contract["manifest"]
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    models = {}
    rows_by_model = {}
    ids_by_model = {}
    for model, expected in REQUIRED_GENERATIONS.items():
        # This validates every latest raw row against the recomputed canonical
        # plan and fails unless the complete 640/256 matrix is present.
        judge_sources = validated_generation_rows(
            model, MANIFEST_VERSION, contract)
        rows = latest(model)
        expected_ids = {
            row["candidate_id"] for row in manifest_rows
            if (model != "qwen3" or row["pool"] != "development")
        }
        unexpected_ids = sorted(set(rows) - expected_ids)
        rows = {candidate_id: row for candidate_id, row in rows.items()
                if candidate_id in expected_ids}
        reviews, _aggregate = load_validated_judge_aggregate(
            model, judge_sources)
        accepted = semantic_ids(rows, reviews)
        rows_by_model[model] = rows
        ids_by_model[model] = accepted
        models[model] = {
            "generated": len(rows),
            "required_generated": expected,
            "unexpected_generation_ids": len(unexpected_ids),
            "row_count_complete": set(rows) == expected_ids,
            "generation_complete": (
                set(rows) == expected_ids and not unexpected_ids and all(
                    generation_backend_is_canonical(
                        row.get("generation_backend"))
                    for row in rows.values())),
            "mps_generated": sum(
                generation_backend_is_canonical(row.get("generation_backend"))
                for row in rows.values()),
            "mps_generation_complete": (
                set(rows) == expected_ids and not unexpected_ids and all(
                    generation_backend_is_canonical(
                        row.get("generation_backend"))
                    for row in rows.values())),
            "surface_accepted": sum(
                row["quality"]["accepted"] for row in rows.values()),
            "independently_judged": len(reviews),
            "semantic_accepted": len(accepted),
            "semantic_counts": quota_counts(manifest, accepted),
            "quota": quota_viability(
                manifest, accepted, include_development=(model != "qwen3")),
        }
    # The shared contract above has already required and source-authenticated a
    # current 640/0 audit. Missing audit state can never become implicit passes.
    reference_passes = set(manifest)
    jointly_eligible_development = (
        ids_by_model["qwen25"] & ids_by_model["qwen35"] & reference_passes)
    jointly_eligible_tests = (
        ids_by_model["qwen25"] & ids_by_model["qwen3"] &
        ids_by_model["qwen35"] & reference_passes)
    result = {
        "manifest_version": MANIFEST_VERSION,
        "manifest_candidates": len(manifest),
        "models": models,
        "qwen25_qwen3_paired_tests": paired_test_viability(
            manifest, ids_by_model["qwen25"], ids_by_model["qwen3"]),
        "all_model_paired_tests": quota_viability(
            manifest, jointly_eligible_tests, include_development=False),
        "joint_development_quota": development_viability(
            manifest, jointly_eligible_development),
        "calibration_token_feasibility": {
            "qwen25": calibration_token_feasibility(
                manifest, jointly_eligible_development,
                lambda candidate_id: rows_by_model["qwen25"][candidate_id]
                ["quality"]["token_counts"]["qwen25"]),
            "qwen3": calibration_token_feasibility(
                manifest, jointly_eligible_development,
                lambda candidate_id: manifest[candidate_id]
                ["calibration_prompt_token_counts"]["qwen3"]),
            "qwen35": calibration_token_feasibility(
                manifest, jointly_eligible_development,
                lambda candidate_id: rows_by_model["qwen35"][candidate_id]
                ["quality"]["token_counts"]["qwen35"]),
        },
        "judge_aggregate_schema": str(
            Path(__file__).resolve().parent / "judge_aggregate_schema.json"),
        "judge_aggregate_paths": {
            model: str(versioned_directory(
                GENERATED / "judge", MANIFEST_VERSION) / f"{model}.json")
            for model in REQUIRED_GENERATIONS
        },
    }
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
