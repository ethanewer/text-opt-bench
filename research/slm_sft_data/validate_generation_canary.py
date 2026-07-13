#!/usr/bin/env python3
"""Authenticate one exact-8 SLM MPS generation canary before full resume."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

try:
    from .generate_responses import (
        CANONICAL_BATCH_SIZE,
        GENERATED,
        MODEL_SPECS,
        RAW,
        _AUTHENTICATED_FINGERPRINTS,
        build_generation_validation_context,
        canonical_generation_plan,
        conversation_token_counts,
        generation_backend_is_canonical,
        generation_input_provenance,
        generation_plan_bindings,
        load_tokenizers,
        read_jsonl,
        render,
        verify_checkpoint,
        validate_generated_record,
        versioned_directory,
    )
    from .pipeline_contract import require_current_reference_audit
except ImportError:  # Direct script execution.
    from generate_responses import (
        CANONICAL_BATCH_SIZE,
        GENERATED,
        MODEL_SPECS,
        RAW,
        _AUTHENTICATED_FINGERPRINTS,
        build_generation_validation_context,
        canonical_generation_plan,
        conversation_token_counts,
        generation_backend_is_canonical,
        generation_input_provenance,
        generation_plan_bindings,
        load_tokenizers,
        read_jsonl,
        render,
        verify_checkpoint,
        validate_generated_record,
        versioned_directory,
    )
    from pipeline_contract import require_current_reference_audit


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_canary(model_key: str, manifest_path: Path,
                    batch_size: int = CANONICAL_BATCH_SIZE,
                    base_seed: int = 7357) -> dict:
    if batch_size != CANONICAL_BATCH_SIZE:
        raise RuntimeError(f"canary batch size must be {CANONICAL_BATCH_SIZE}")
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("transformers is required for canary validation") from exc

    audit = require_current_reference_audit(manifest_path)
    context = build_generation_validation_context(
        transformers, audit, model_key, base_seed, batch_size)
    spec = context["spec"]
    manifest = context["manifest"]
    tokenizers = context["tokenizers"]
    tokenizer = context["tokenizer"]
    plan = context["plan"]
    bindings = context["bindings"]
    status_path = GENERATED / f"status_{model_key}_v2.json"
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing canary status {status_path}") from exc
    canary = status.get("canary")
    if not isinstance(canary, dict):
        raise RuntimeError("latest generation status is not a canary run")
    require_equal(status.get("model"), model_key, "canary status model")
    require_equal(status.get("model_id"), spec["hub_id"],
                  "canary status model ID")
    require_equal(status.get("device"), "mps", "canary status device")
    require_equal(status.get("manifest_version"), 2,
                  "canary status manifest version")
    require_equal(status.get("canonical_batch_plan_sha256"),
                  plan["plan_sha256"], "canary plan SHA")
    candidate_ids = canary.get("candidate_ids")
    if (not isinstance(candidate_ids, list) or
            len(candidate_ids) != batch_size or
            len(set(candidate_ids)) != batch_size):
        raise RuntimeError("canary does not identify eight unique candidates")
    planned_batch = next((batch for batch in plan["batches"]
                          if batch["batch_sha256"] == canary.get("batch_sha256")),
                         None)
    if planned_batch is None:
        raise RuntimeError("canary batch SHA is absent from the current plan")
    expected_ids = [member["candidate_id"]
                    for member in planned_batch["identity"]["members"]]
    require_equal(candidate_ids, expected_ids, "canary candidate order")
    require_equal(canary.get("first_turn_size"), batch_size,
                  "canary first-turn size")
    require_equal(canary.get("follow_up_turn_size"), len(
        planned_batch["identity"]["follow_up_members"]),
        "canary follow-up size")
    require_equal(canary.get("actual_generation_cap"),
                  planned_batch["identity"]["actual_generation_cap"],
                  "canary actual generation cap")
    require_equal(status.get("scheduled_batches"), 1,
                  "canary scheduled-batch count")
    require_equal(status.get("computed_this_run"), batch_size,
                  "canary computed-row count")
    write_candidate_ids = canary.get("write_candidate_ids")
    if (not isinstance(write_candidate_ids, list) or
            not write_candidate_ids or
            len(write_candidate_ids) != len(set(write_candidate_ids)) or
            write_candidate_ids != [candidate_id for candidate_id in candidate_ids
                                    if candidate_id in set(write_candidate_ids)]):
        raise RuntimeError("canary write-ID proof is invalid")
    require_equal(status.get("completed_this_run"), len(write_candidate_ids),
                  "canary appended-row count")

    raw_path = versioned_directory(RAW, 2) / f"{model_key}.jsonl"
    latest = {}
    for row in read_jsonl(raw_path):
        latest[row["candidate_id"]] = row
    missing = set(candidate_ids) - set(latest)
    if missing:
        raise RuntimeError(f"canary raw artifact is missing rows: {sorted(missing)}")
    for position, candidate_id in enumerate(candidate_ids):
        row = latest[candidate_id]
        require_equal(bindings[candidate_id]["position"], position,
                      f"{candidate_id} canary position")
        validate_generated_record(
            row, context, require_surface_acceptance=True)

    return {
        "status": "canary_valid",
        "model": model_key,
        "batch_size": batch_size,
        "candidate_ids": candidate_ids,
        "batch_sha256": planned_batch["batch_sha256"],
        "plan_sha256": plan["plan_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--batch-size", type=int, default=CANONICAL_BATCH_SIZE)
    parser.add_argument("--base-seed", type=int, default=7357)
    parser.add_argument(
        "--manifest", type=Path,
        default=GENERATED / "prompt_candidates_v2.jsonl")
    args = parser.parse_args()
    print(json.dumps(validate_canary(
        args.model, args.manifest, args.batch_size, args.base_seed),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
