"""Adversarial model-free checks for semantic-judge cache authentication."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.slm_sft_data.run_quality_judges import (
    JUDGE_BATCH_SIZE,
    aggregate,
    expected_provenance,
    file_sha256,
    require_complete_generation_matrix,
    validate_judge_batch_payload,
    validate_raw_reviews,
)


def source_rows() -> list[dict]:
    return [{
        "candidate_id": "probe",
        "model_id": "Qwen/probe",
        "messages": [
            {"role": "user", "content": "Prompt"},
            {"role": "assistant", "content": "Answer"},
        ],
        "generation_backend": {"device_backend": "mps"},
        "_quality_reference": {"reference_sha256": "a" * 64},
        "provenance": {
            "generation_input_sha256": "b" * 64,
            "manifest_row_sha256": "c" * 64,
        },
    }]


def raw_review() -> dict:
    return {
        "candidate_id": "probe",
        "verdict": "accept",
        "score": 5,
        "gates": {
            "semantic_correct": True,
            "instruction_compliant": True,
            "safe": True,
            "format_compliant": True,
            "complete": True,
            "no_truncation": True,
            "no_repetition": True,
        },
        "reasons": ["correct and complete"],
    }


def write_valid_batch(root: Path):
    rows = source_rows()
    provenance = expected_provenance(
        rows, "gpt-5.6-sol", "high", "codex-test", 2)
    raw_path = root / "batch.raw.json"
    output_path = root / "batch.json"
    log_path = root / "batch.log"
    raw = {"reviews": [raw_review()]}
    raw_path.write_text(json.dumps(raw))
    log_path.write_text("immutable invocation transcript")
    source = provenance["sources"][0]
    normalized_review = {
        **raw["reviews"][0],
        "conversation_sha256": source["conversation_sha256"],
        "quality_reference_sha256": source["quality_reference_sha256"],
        "generation_input_sha256": source["generation_input_sha256"],
        "manifest_row_sha256": source["manifest_row_sha256"],
    }
    payload = {
        "reviews": [normalized_review],
        "provenance": provenance,
        "model_output_sha256": file_sha256(raw_path),
        "invocation_log_sha256": file_sha256(log_path),
    }
    output_path.write_text(json.dumps(payload))
    return rows, output_path, raw_path, log_path


def validate(paths):
    rows, output, raw, log = paths
    return validate_judge_batch_payload(
        output, raw, log, rows, "gpt-5.6-sol", "high", "codex-test", 2)


def test_valid_judge_batch_proof(tmp_path: Path) -> None:
    paths = write_valid_batch(tmp_path)
    payload = validate(paths)
    assert payload["reviews"][0]["verdict"] == "accept"
    assert payload["provenance"]["configured_batch_size"] == JUDGE_BATCH_SIZE


def test_semantic_judge_batch_size_is_fixed() -> None:
    with pytest.raises(RuntimeError, match="batch size is fixed"):
        aggregate("qwen3", [], JUDGE_BATCH_SIZE // 2,
                  "gpt-5.6-sol", "high", "codex-test", 2)


def test_incomplete_generation_matrix_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        require_complete_generation_matrix(
            "qwen3", {"row-0": {}}, {"row-0", "row-1"})
    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        require_complete_generation_matrix(
            "qwen3", {"row-0": {}, "unexpected": {}}, {"row-0"})


@pytest.mark.parametrize("target", ["normalized", "raw", "log"])
def test_edited_judge_cache_is_rejected(tmp_path: Path, target: str) -> None:
    paths = write_valid_batch(tmp_path)
    _rows, output, raw, log = paths
    if target == "normalized":
        payload = json.loads(output.read_text())
        payload["reviews"][0]["verdict"] = "reject"
        output.write_text(json.dumps(payload))
    elif target == "raw":
        payload = json.loads(raw.read_text())
        payload["reviews"][0]["score"] = 4
        raw.write_text(json.dumps(payload))
    else:
        log.write_text("edited transcript")
    with pytest.raises(RuntimeError):
        validate(paths)


@pytest.mark.parametrize("mutation", [
    "duplicate_id", "bad_score", "bad_gate", "bad_verdict", "empty_reason",
    "extra_field",
])
def test_raw_judge_schema_is_revalidated(mutation: str) -> None:
    review = raw_review()
    reviews = [review]
    expected_ids = ["probe"]
    if mutation == "duplicate_id":
        reviews.append(copy.deepcopy(review))
    elif mutation == "bad_score":
        review["score"] = 6
    elif mutation == "bad_gate":
        review["gates"].pop("safe")
    elif mutation == "bad_verdict":
        review["verdict"] = "reject"
    elif mutation == "empty_reason":
        review["reasons"] = [""]
    else:
        review["extra"] = True
    with pytest.raises(RuntimeError):
        validate_raw_reviews(reviews, expected_ids)
