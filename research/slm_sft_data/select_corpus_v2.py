#!/usr/bin/env python3
"""Select the final calibration/validation/test corpus from the 2x pool.

Selection uses only prompt metadata, independent semantic-quality judgments,
and calibration sequence lengths.  It never observes compression losses or
candidate-policy performance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

from bench.slm_data import calibration_record
from generate_responses import (generation_backend_is_canonical,
                                load_tokenizers, read_jsonl,
                                versioned_directory)
from bench.slm_mps_lock import canonical_mps_lock_identity
from run_quality_judges import (QUALITY_GATES, RUBRIC, RUBRIC_VERSION, SCHEMA,
                                conversation_sha256,
                                load_validated_judge_aggregate,
                                latest as judge_source_rows)
try:
    from .pipeline_contract import require_current_reference_audit
except ImportError:  # Direct script execution.
    from pipeline_contract import require_current_reference_audit


GENERATED = ROOT / "generated"
MANIFEST_VERSION = 2
MANIFEST = GENERATED / "prompt_candidates_v2.jsonl"
RAW = GENERATED / "raw"
JUDGE = GENERATED / "judge"
REFERENCE_AUDIT = GENERATED / "reference_audit_v2.json"
OUTPUT = GENERATED / "selected_corpus.json"

MODEL_PATHS = {
    "qwen25": "/tmp/qwen2.5-0.5b-instruct",
    "qwen3": "/tmp/qwen3-06b",
    "qwen35": "/tmp/qwen35-08b",
}
TRAIN_FAMILIES = (
    "general_chat_writing", "code_agent_tools",
    "math_quantitative", "science_technical",
)
OOD_FAMILIES = (
    "business_operations", "finance_accounting_economics",
    "legal_policy_compliance", "medicine_health",
    "cybersecurity_infrastructure", "humanities_social_sciences",
    "creative_design_storytelling", "multilingual_translation",
)


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_rows(model: str) -> dict[str, dict]:
    path = versioned_directory(RAW, MANIFEST_VERSION) / f"{model}.jsonl"
    result = {}
    for row in read_jsonl(path):
        result[row["candidate_id"]] = row
    return result


def load_reviews(model: str, source_rows: list[dict]) -> tuple[dict[str, dict], dict]:
    return load_validated_judge_aggregate(model, source_rows)


def load_reference_passes() -> tuple[set[str], dict]:
    contract = require_current_reference_audit(MANIFEST)
    if contract.get("source_protocol") == "public-datasets-v1":
        public = contract["public_source"]
        return ({row["candidate_id"] for row in contract["manifest"]}, {
            "model": "pinned-public-dataset",
            "reasoning": "deterministic-source-authentication",
            "source_protocol": contract["source_protocol"],
            "sha256": public["sha256"],
        })
    payload = json.loads(REFERENCE_AUDIT.read_text())
    return ({row["candidate_id"] for row in contract["manifest"]}, payload)


def valid_pair(candidate_id: str, model: str, manifest: dict[str, dict],
               rows: dict[str, dict], reviews: dict[str, dict],
               reference_passes: set[str]) -> bool:
    source = rows.get(candidate_id)
    review = reviews.get(candidate_id)
    if source is None or review is None or candidate_id not in reference_passes:
        return False
    expected_input = manifest[candidate_id]["provenance"]["input_sha256"]
    gates = review.get("gates")
    return bool(
        source.get("manifest_version") == MANIFEST_VERSION and
        source.get("record_format") == 2 and
        generation_backend_is_canonical(source.get("generation_backend")) and
        source.get("provenance", {}).get("input_sha256") == expected_input and
        source.get("provenance", {}).get("manifest_row_sha256") ==
        canonical_sha256(manifest[candidate_id]) and
        review.get("generation_input_sha256") ==
        source.get("provenance", {}).get("generation_input_sha256") and
        review.get("manifest_row_sha256") ==
        source.get("provenance", {}).get("manifest_row_sha256") and
        review.get("quality_reference_sha256") ==
        source.get("provenance", {}).get("reference_sha256") and
        source.get("quality", {}).get("accepted") and
        source["quality"].get("max_conversation_tokens", 513) <= 512 and
        review.get("conversation_sha256") == conversation_sha256(source) and
        review.get("verdict") == "accept" and
        review.get("score", 0) >= 4 and
        isinstance(gates, dict) and set(gates) == QUALITY_GATES and
        all(gates.values()))


def valid_public_generation(candidate_id: str, manifest: dict[str, dict],
                            rows: dict[str, dict],
                            reference_passes: set[str]) -> bool:
    """Authenticate generation without imposing a semantic-answer judge.

    Public questions are calibration seeds, not labeled SFT targets. Naturally
    generated answers (including answers stopped at the token cap) are retained.
    """
    source = rows.get(candidate_id)
    if source is None or candidate_id not in reference_passes:
        return False
    return bool(
        source.get("manifest_version") == MANIFEST_VERSION and
        source.get("record_format") == 2 and
        generation_backend_is_canonical(source.get("generation_backend")) and
        source.get("provenance", {}).get("input_sha256") ==
        manifest[candidate_id]["provenance"]["input_sha256"] and
        source.get("provenance", {}).get("manifest_row_sha256") ==
        canonical_sha256(manifest[candidate_id]) and
        source.get("quality", {}).get("accepted") and
        source["quality"].get("max_conversation_tokens", 513) <= 512)


def quality_rank(candidate_id: str, models: tuple[str, ...], reviews) -> tuple:
    if not all(reviews.get(model) for model in models):
        return (candidate_id,)
    scores = [reviews[model][candidate_id]["score"] for model in models]
    return (-min(scores), -sum(scores), candidate_id)


def diverse_select(candidate_ids: list[str], count: int,
                   manifest: dict[str, dict], rank) -> list[str]:
    """Round-robin task styles, then use the supplied within-style rank."""
    groups = defaultdict(list)
    for candidate_id in candidate_ids:
        groups[manifest[candidate_id]["template_cluster"]].append(candidate_id)
    for values in groups.values():
        values.sort(key=rank)
    selected = []
    styles = sorted(groups)
    while len(selected) < count:
        progressed = False
        for style in styles:
            if groups[style] and len(selected) < count:
                selected.append(groups[style].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} quality-passing rows available; need {count}")
    return selected


def nested_calibration_coverage(calibration: list[str], manifest: dict[str, dict]):
    """Require every nested per-family prefix to maximize cluster coverage."""
    result = {}
    by_family = {
        family: [candidate_id for candidate_id in calibration
                 if manifest[candidate_id]["family"] == family]
        for family in TRAIN_FAMILIES
    }
    for size, per_family in (("32", 8), ("64", 16), ("128", 32)):
        family_counts = {}
        minimum_required = {}
        for family, rows in by_family.items():
            available = len({manifest[candidate_id]["template_cluster"]
                             for candidate_id in rows})
            prefix = rows[:per_family]
            observed = len({manifest[candidate_id]["template_cluster"]
                            for candidate_id in prefix})
            required = min(per_family, available)
            if observed < required:
                raise RuntimeError(
                    f"nested calibration {size}/{family} covers {observed} "
                    f"template clusters; expected {required}")
            family_counts[family] = observed
            minimum_required[family] = required
        result[size] = {
            "rows": per_family * len(TRAIN_FAMILIES),
            "template_clusters": sum(family_counts.values()),
            "template_clusters_by_family": family_counts,
            "minimum_required_by_family": minimum_required,
        }
    return result


def selected_quality_proof(candidate_id: str, model: str, rows, reviews,
                           aggregate: dict) -> dict:
    source = rows[model][candidate_id]
    review = reviews[model][candidate_id]
    message_sha = canonical_sha256(source["messages"])
    return {
        "conversation_sha256": message_sha,
        "judge_conversation_sha256": review["conversation_sha256"],
        "semantic_verdict": review["verdict"],
        "semantic_score": review["score"],
        "gates": dict(review["gates"]),
        "surface_quality_sha256": canonical_sha256(source["quality"]),
        "generation_backend": dict(source["generation_backend"]),
        "judge_model": aggregate["judge_model"],
        "judge_reasoning": aggregate["reasoning"],
        "judge_rubric_version": aggregate["rubric_version"],
    }


def selected_public_proof(candidate_id: str, model: str, rows) -> dict:
    source = rows[model][candidate_id]
    return {
        "conversation_sha256": canonical_sha256(source["messages"]),
        "surface_quality_sha256": canonical_sha256(source["quality"]),
        "generation_backend": dict(source["generation_backend"]),
        "acceptance_protocol": "authenticated-natural-generation-v1",
        "semantic_judge_used": False,
        "length_truncation_allowed": True,
    }


def main() -> None:
    import transformers

    contract = require_current_reference_audit(MANIFEST)
    public_mode = contract.get("source_protocol") == "public-datasets-v1"
    manifest_rows = contract["manifest"]
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    if len(manifest_rows) != 640 or len(manifest) != 640:
        raise RuntimeError("v2 candidate manifest must contain 640 unique IDs")
    expected_pool = Counter({"development": 384, "id_test": 128,
                             "ood_test": 128})
    if Counter(row["pool"] for row in manifest_rows) != expected_pool:
        raise RuntimeError("candidate manifest does not preserve the 2x pools")
    expected_roles = Counter({
        (family, "calibration_candidate"): 64
        for family in TRAIN_FAMILIES})
    expected_roles.update({
        (family, "validation_candidate"): 32
        for family in TRAIN_FAMILIES})
    actual_roles = Counter(
        (row["family"], row.get("development_role"))
        for row in manifest_rows if row["pool"] == "development")
    if actual_roles != expected_roles:
        raise RuntimeError(
            "candidate manifest does not preserve fixed development roles")

    reference_passes, reference_audit = load_reference_passes()
    source_rows = ({model: list(latest_rows(model).values())
                    for model in MODEL_PATHS} if public_mode else {
        model: judge_source_rows(model, MANIFEST_VERSION, contract)
        for model in MODEL_PATHS})
    rows = {
        model: {row["candidate_id"]: row for row in source_rows[model]}
        for model in MODEL_PATHS}
    review_pairs = ({} if public_mode else {
        model: load_reviews(model, source_rows[model]) for model in MODEL_PATHS})
    reviews = ({model: {} for model in MODEL_PATHS} if public_mode else
               {model: pair[0] for model, pair in review_pairs.items()})
    aggregates = ({} if public_mode else
                  {model: pair[1] for model, pair in review_pairs.items()})

    eligible = {}
    for candidate_id, item in manifest.items():
        required_models = (("qwen25", "qwen35")
                           if item["pool"] == "development"
                           else ("qwen25", "qwen3", "qwen35"))
        if all((valid_public_generation(candidate_id, manifest, rows[model],
                                        reference_passes) if public_mode else
                valid_pair(candidate_id, model, manifest, rows[model],
                           reviews[model], reference_passes))
               for model in required_models):
            eligible[candidate_id] = required_models

    tokenizers = load_tokenizers(transformers)
    calibration_tokens = {}
    qwen3_prompt_only_records = {}
    for candidate_id in eligible:
        if (manifest[candidate_id]["pool"] != "development" or
                manifest[candidate_id].get("development_role") !=
                "calibration_candidate"):
            continue
        qwen3_record = calibration_record(
            rows["qwen25"][candidate_id], "qwen3",
            tokenizers["qwen3"], prompt_only=True)
        if not (qwen3_record["prompt_only"] and
                qwen3_record["add_generation_prompt"] and
                qwen3_record["generation_scaffold_tokens"] > 0 and
                qwen3_record["fabricated_assistant_targets"] is False):
            raise RuntimeError(
                f"{candidate_id} lacks strict Qwen3 prompt-only provenance")
        qwen3_prompt_only_records[candidate_id] = qwen3_record
        calibration_tokens[candidate_id] = {
            "qwen25": len(calibration_record(
                rows["qwen25"][candidate_id], "qwen25",
                tokenizers["qwen25"])["input_ids"]),
            "qwen3": len(qwen3_record["input_ids"]),
            "qwen35": len(calibration_record(
                rows["qwen35"][candidate_id], "qwen35",
                tokenizers["qwen35"])["input_ids"]),
        }

    calibration, validation = [], []
    for family in TRAIN_FAMILIES:
        calibration_local = [
            candidate_id for candidate_id in eligible
            if manifest[candidate_id]["pool"] == "development" and
            manifest[candidate_id].get("development_role") ==
            "calibration_candidate" and
            manifest[candidate_id]["family"] == family]
        validation_local = [
            candidate_id for candidate_id in eligible
            if manifest[candidate_id]["pool"] == "development" and
            manifest[candidate_id].get("development_role") ==
            "validation_candidate" and
            manifest[candidate_id]["family"] == family]
        if len(calibration_local) < 32 or len(validation_local) < 16:
            raise RuntimeError(
                f"{family} has jointly passing fixed-role rows "
                f"calibration={len(calibration_local)}, "
                f"validation={len(validation_local)}; need 32 and 16")

        def calibration_rank(candidate_id):
            lengths = calibration_tokens[candidate_id]
            quality = quality_rank(
                candidate_id, ("qwen25", "qwen35"), reviews)
            return (-min(lengths.values()), -sum(lengths.values()), *quality)

        chosen = diverse_select(
            calibration_local, 32, manifest, calibration_rank)
        # Preserve diverse_select's cluster round-robin ordering. The compiler
        # takes the first 8/16/32 rows per family for nested ablations.
        calibration.extend(chosen)
        validation.extend(diverse_select(
            validation_local, 16, manifest,
            lambda candidate_id: quality_rank(
                candidate_id, ("qwen25", "qwen35"), reviews)))

    totals = {model: sum(calibration_tokens[candidate_id][model]
                         for candidate_id in calibration)
              for model in MODEL_PATHS}
    if (not public_mode and
            any(not 50_000 <= value <= 65_536 for value in totals.values())):
        raise RuntimeError(
            "selected 128-row calibration set misses the 50k--65,536 token "
            f"target: {totals}; create a separate test-disjoint calibration "
            "corpus instead of padding")
    calibration_prefix_coverage = nested_calibration_coverage(
        calibration, manifest)

    tests = {}
    for group, pool, families, per_family in (
            ("overlap", "id_test", TRAIN_FAMILIES, 16),
            ("heldout", "ood_test", OOD_FAMILIES, 8)):
        selected = []
        for family in families:
            local = [candidate_id for candidate_id in eligible
                     if manifest[candidate_id]["pool"] == pool and
                     manifest[candidate_id]["family"] == family]
            selected.extend(diverse_select(
                local, per_family, manifest,
                lambda candidate_id: quality_rank(
                    candidate_id, ("qwen25", "qwen3", "qwen35"), reviews)))
        tests[group] = selected

    final_ids = calibration + validation + tests["overlap"] + tests["heldout"]
    if len(final_ids) != 320 or len(set(final_ids)) != 320:
        raise RuntimeError("final selection must contain 320 unique prompt IDs")
    if (any(manifest[candidate_id].get("development_role") !=
            "calibration_candidate" for candidate_id in calibration) or
            any(manifest[candidate_id].get("development_role") !=
                "validation_candidate" for candidate_id in validation)):
        raise RuntimeError("selection crossed a fixed development-role boundary")
    quality_proof = {model: {} for model in MODEL_PATHS}
    for candidate_id in final_ids:
        required_models = eligible[candidate_id]
        for model in required_models:
            quality_proof[model][candidate_id] = (
                selected_public_proof(candidate_id, model, rows)
                if public_mode else selected_quality_proof(
                    candidate_id, model, rows, reviews, aggregates[model]))

    output = {
        "format": 1,
        "manifest_version": MANIFEST_VERSION,
        "development": {
            "calibration": calibration,
            "validation": validation,
        },
        "test": tests,
        "quality_proof": quality_proof,
        "selection_protocol": {
            "compression_performance_used": False,
            "required_generation_backend": "mps",
            "required_generation_dtype": "bfloat16",
            "required_mps_lock": canonical_mps_lock_identity(),
            "calibration_rows_scored": 0,
            "online_validation_rows_scored": 64,
            "candidate_counts": dict(expected_pool),
            "fixed_development_subpool_counts": {
                "calibration_candidate": 256,
                "validation_candidate": 128,
            },
            "final_development_role_counts": {
                "calibration_only": 128,
                "validation_score": 64,
            },
            "final_counts": {
                "calibration_only": 128, "validation_score": 64,
                "id_test": 64, "ood_test": 64,
            },
            "calibration_tokens": totals,
            "qwen3_prompt_only_calibration": {
                "add_generation_prompt": True,
                "fabricated_assistant_targets": False,
                "selected_rows": len(calibration),
                "generation_scaffold_tokens": sum(
                    qwen3_prompt_only_records[candidate_id]
                    ["generation_scaffold_tokens"]
                    for candidate_id in calibration),
            },
            "nested_calibration_coverage": calibration_prefix_coverage,
            "template_cluster_counts": {
                "calibration_only": len({
                    manifest[candidate_id]["template_cluster"]
                    for candidate_id in calibration}),
                "validation_score": len({
                    manifest[candidate_id]["template_cluster"]
                    for candidate_id in validation}),
                "id_test": len({
                    manifest[candidate_id]["template_cluster"]
                    for candidate_id in tests["overlap"]}),
                "ood_test": len({
                    manifest[candidate_id]["template_cluster"]
                    for candidate_id in tests["heldout"]}),
            },
            "quality_gates": sorted(QUALITY_GATES),
            "semantic_judge_used": not public_mode,
            "length_truncation_allowed": public_mode,
            "reference_audit_model": reference_audit["model"],
            "reference_audit_reasoning": reference_audit["reasoning"],
            "source_protocol": contract.get(
                "source_protocol", "synthetic-reference-audit-v2"),
            "manifest_sha256": file_sha256(MANIFEST),
            "reference_sha256": contract["reference_sha256"],
            "manifest_audit_sha256": contract["manifest_audit_sha256"],
            "reference_audit_sha256": contract["reference_audit_sha256"],
            "judge_aggregate_sha256": {} if public_mode else {
                model: file_sha256(
                    versioned_directory(JUDGE, MANIFEST_VERSION) /
                    f"{model}.json")
                for model in MODEL_PATHS
            },
        },
    }
    OUTPUT.write_text(json.dumps(
        output, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "selected": len(final_ids),
        "calibration_tokens": totals,
        "eligible": Counter(manifest[candidate_id]["pool"]
                            for candidate_id in eligible),
        "output": str(OUTPUT),
    }, indent=2, default=dict))


if __name__ == "__main__":
    main()
