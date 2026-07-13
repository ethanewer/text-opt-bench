"""Model-free adversarial checks for full-row/canary generation proofs."""

from __future__ import annotations

import copy

import pytest

from bench.slm_mps_lock import canonical_mps_lock_identity
from research.slm_sft_data.generate_responses import (
    generate_batch, load_model, validate_generated_record)
from research.slm_sft_data.tokenizer_pins import PINNED_TOKENIZER_FILES


SPEC = {
    "hub_id": "Qwen/Qwen3.5-0.8B",
    "text_only": True,
    "generation_eos": {1, 2},
}
FINGERPRINT = {"weights_sha256": "a" * 64}
EXPECTED_INPUT = {
    "generation_input_format": 2,
    "generation_input_sha256": "b" * 64,
    "manifest_row_sha256": "c" * 64,
    "rendered_prompt_sha256": "d" * 64,
    "chat_template_mode": "enable_thinking_false",
    "generator_script_sha256": "e" * 64,
    "generation_plan_sha256": "f" * 64,
    "generation_batch_sha256": "1" * 64,
    "generation_batch_position": 0,
    "actual_generation_cap": 80,
    "cross_tokenizer_snapshot_sha256": "2" * 64,
}
SOURCE = {
    "candidate_id": "probe",
    "messages": [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Prompt"},
    ],
    "follow_up": "Confirm briefly.",
    "provenance": {
        "input_sha256": "input",
        "reference_sha256": "reference",
    },
}
REFERENCE = {"candidate_id": "probe", "reference_sha256": "reference"}
BATCH_IDENTITY = {
    "actual_generation_cap": 80,
    "follow_up_members": [
        {"candidate_id": "probe"},
        {"candidate_id": "peer-1"},
        {"candidate_id": "peer-2"},
    ],
}


def context() -> dict:
    return {
        "model_key": "qwen35",
        "spec": copy.deepcopy(SPEC),
        "manifest_by_id": {"probe": copy.deepcopy(SOURCE)},
        "references_by_id": {"probe": copy.deepcopy(REFERENCE)},
        "expected_inputs": {"probe": copy.deepcopy(EXPECTED_INPUT)},
        "fingerprint": copy.deepcopy(FINGERPRINT),
        "runtime_versions": {
            "torch": "test-torch", "transformers": "test-transformers"},
        "bindings": {"probe": {
            "position": 0, "batch_identity": copy.deepcopy(BATCH_IDENTITY)}},
        "base_seed": 7357,
        "batch_size": 8,
        "tokenizers": {},
    }


def surface_values(rejected: bool = False, over_limit: bool = False):
    counts = {"qwen25": 513 if over_limit else 100,
              "qwen3": 101, "qwen35": 102}
    assistants = {"qwen25": 20, "qwen3": 21, "qwen35": 22}
    reasons = ["generation_cap_without_eos_turn_0"] if rejected else []
    flags = ["hit_generation_cap_turn_0"] if rejected else []
    return reasons, flags, counts, assistants


def quality(values=None) -> dict:
    reasons, flags, counts, assistants = values or surface_values()
    return {
        "accepted": not reasons,
        "acceptance_scope": "surface_only",
        "semantic_review_required": True,
        "rejection_reasons": reasons,
        "flags": flags,
        "token_counts": counts,
        "assistant_target_token_counts": assistants,
        "nonassistant_token_counts": {
            key: counts[key] - assistants[key] for key in counts},
        "max_conversation_tokens": max(counts.values()),
    }


def valid_record() -> dict:
    provenance = {
        "input_sha256": "input", "reference_sha256": "reference",
        **EXPECTED_INPUT,
    }
    return {
        "record_format": 2,
        "manifest_version": 2,
        "candidate_id": "probe",
        "model_key": "qwen35",
        "model_id": SPEC["hub_id"],
        "checkpoint": copy.deepcopy(FINGERPRINT),
        "nonthinking": True,
        "text_only": True,
        "follow_up": SOURCE["follow_up"],
        "messages": [
            *copy.deepcopy(SOURCE["messages"]),
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": SOURCE["follow_up"]},
            {"role": "assistant", "content": "Confirmed."},
        ],
        "provenance": provenance,
        "generation_backend": {
            "device_backend": "mps",
            "mps_fallback_enabled": False,
            "model_weight_dtype": "bfloat16",
            "torch_version": "test-torch",
            "transformers_version": "test-transformers",
            "cross_tokenizer_snapshots": copy.deepcopy(PINNED_TOKENIZER_FILES),
            "exclusive_mps_lock": canonical_mps_lock_identity(),
            "model_device_dtype_attestation": {
                "attested": True,
                "parameter_count": 100,
                "parameter_elements": 1000,
                "floating_parameter_count": 100,
                "floating_parameter_elements": 1000,
                "buffer_count": 10,
                "buffer_elements": 100,
                "parameter_devices": ["mps"],
                "buffer_devices": ["mps"],
                "floating_parameter_dtypes": ["torch.bfloat16"],
            },
        },
        "generation_outputs": [
            {
                "turn_index": 0, "text": "Answer", "generated_tokens": 10,
                "generation_steps": 11, "max_new_tokens": 80,
                "terminal_eos_token_id": 1, "stop_reason": "eos",
                "hit_generation_cap": False, "batch_elapsed_seconds": 0.5,
                "batch_size": 8, "deterministic_seed": 7357,
                "seed_scope": "model_wide",
            },
            {
                "turn_index": 1, "text": "Confirmed.", "generated_tokens": 4,
                "generation_steps": 5, "max_new_tokens": 80,
                "terminal_eos_token_id": 2, "stop_reason": "eos",
                "hit_generation_cap": False, "batch_elapsed_seconds": 0.4,
                "batch_size": 3, "deterministic_seed": 107358,
                "seed_scope": "model_wide",
            },
        ],
        "quality": quality(),
    }


def run(row: dict, values=None, require_acceptance=True):
    values = values or surface_values()
    return validate_generated_record(
        row, context(), require_surface_acceptance=require_acceptance,
        surface_quality_fn=lambda _messages, _outputs, _tokenizers: values)


def test_valid_model_free_generation_record() -> None:
    assert run(valid_record())["candidate_id"] == "probe"


def test_generation_helpers_require_active_canonical_mps_lease() -> None:
    with pytest.raises(RuntimeError, match="active canonical"):
        load_model(None, None, "qwen25", {}, None, None)
    with pytest.raises(RuntimeError, match="active canonical"):
        generate_batch(
            None, None, None, None, [], 0, 7357, "qwen25", 80, [])


@pytest.mark.parametrize("mutation", [
    "cpu", "fallback", "weight_dtype", "attested_dtype", "alternate_lock",
    "zero_parameters", "buffer_count",
])
def test_generation_rejects_noncanonical_backend(mutation: str) -> None:
    row = valid_record()
    backend = row["generation_backend"]
    if mutation == "cpu":
        backend["device_backend"] = "cpu"
    elif mutation == "fallback":
        backend["mps_fallback_enabled"] = True
    elif mutation == "weight_dtype":
        backend["model_weight_dtype"] = "float16"
    elif mutation == "attested_dtype":
        backend["model_device_dtype_attestation"][
            "floating_parameter_dtypes"] = ["torch.float16"]
    elif mutation == "alternate_lock":
        backend["exclusive_mps_lock"]["path"] = "/tmp/alternate.lock"
    elif mutation == "zero_parameters":
        backend["model_device_dtype_attestation"]["parameter_count"] = 0
    else:
        backend["model_device_dtype_attestation"]["buffer_count"] = 0
    with pytest.raises(RuntimeError, match="source/model/backend"):
        run(row)


@pytest.mark.parametrize("field", ["nonthinking", "text_only"])
def test_generation_rejects_thinking_or_vision_mode(field: str) -> None:
    row = valid_record()
    row[field] = False
    with pytest.raises(RuntimeError, match="source/model/backend"):
        run(row)


@pytest.mark.parametrize("field", sorted(EXPECTED_INPUT))
def test_every_exposed_generation_provenance_field_is_bound(field: str) -> None:
    row = valid_record()
    row["provenance"][field] = "tampered"
    with pytest.raises(RuntimeError, match=field):
        run(row)


@pytest.mark.parametrize("turn,size", [(0, 7), (1, 2)])
def test_generation_rejects_wrong_actual_turn_batch_size(
        turn: int, size: int) -> None:
    row = valid_record()
    row["generation_outputs"][turn]["batch_size"] = size
    with pytest.raises(RuntimeError, match="stop proof"):
        run(row)


def test_generation_rejects_output_message_mismatch() -> None:
    row = valid_record()
    row["generation_outputs"][0]["text"] = "Different"
    with pytest.raises(RuntimeError, match="alignment"):
        run(row)


def test_eos_on_final_allowed_step_is_not_a_cap_hit() -> None:
    row = valid_record()
    row["generation_outputs"][0].update({
        "generated_tokens": 79,
        "generation_steps": 80,
        "terminal_eos_token_id": 1,
        "stop_reason": "eos",
        "hit_generation_cap": False,
    })
    run(row)


def test_length_stop_is_integral_but_not_surface_accepted() -> None:
    row = valid_record()
    row["generation_outputs"][0].update({
        "generated_tokens": 80,
        "generation_steps": 80,
        "terminal_eos_token_id": None,
        "stop_reason": "length",
        "hit_generation_cap": True,
    })
    values = surface_values(rejected=True)
    row["quality"] = quality(values)
    run(row, values, require_acceptance=False)
    with pytest.raises(RuntimeError, match="surface quality"):
        run(row, values, require_acceptance=True)


def test_generation_rejects_conversation_over_512() -> None:
    row = valid_record()
    values = surface_values(over_limit=True)
    row["quality"] = quality(values)
    # The recomputed quality is authentic but deterministic surface acceptance
    # still fails because response_quality includes the over-limit reason in
    # production. Model the explicit stale-acceptance guard here.
    row["quality"]["accepted"] = False
    with pytest.raises(RuntimeError):
        run(row, values)


@pytest.mark.parametrize("field,value", [
    ("acceptance_scope", "semantic"),
    ("semantic_review_required", False),
])
def test_generation_rejects_quality_protocol_drift(field: str, value) -> None:
    row = valid_record()
    row["quality"][field] = value
    with pytest.raises(RuntimeError, match="surface-quality"):
        run(row)


def test_generation_rejects_nonfinite_elapsed_time() -> None:
    row = valid_record()
    row["generation_outputs"][0]["batch_elapsed_seconds"] = float("nan")
    with pytest.raises(RuntimeError, match="stop proof"):
        run(row)
