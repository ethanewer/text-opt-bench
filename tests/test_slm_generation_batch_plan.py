"""Model-free adversarial checks for immutable batched SLM generation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.slm_sft_data.generate_responses import (
    CANONICAL_BATCH_SIZE,
    canonical_generation_plan,
    require_generation_plan_integrity,
    schedule_generation_batches,
    select_canary_batch,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "research" / "slm_sft_data" / "generated"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_sources(intervals: list[tuple[int, int, int]]):
    rows, references, rendered = [], [], {}
    for index, (required, preferred, ceiling) in enumerate(intervals):
        candidate_id = f"candidate_{index:03d}"
        messages = [{"role": "user", "content": candidate_id}]
        input_sha = hashlib.sha256(candidate_id.encode()).hexdigest()
        reference_sha = hashlib.sha256(f"ref:{candidate_id}".encode()).hexdigest()
        row = {
            "candidate_id": candidate_id,
            "messages": messages,
            "follow_up": "confirm" if index % 5 == 0 else None,
            "generation": {
                "max_new_tokens_per_turn": ceiling,
                "declared_max_new_tokens_per_turn": ceiling,
            },
            "prompt_token_counts": {
                "qwen25": 200 + index,
                "qwen3": 204 + index,
                "qwen35": 208 + index,
            },
            "provenance": {"input_sha256": input_sha},
        }
        reference = {
            "candidate_id": candidate_id,
            "reference_sha256": reference_sha,
            "reference_required_generation_cap": max(1, required - 12),
            "word_required_generation_cap": 0,
            "required_generation_cap": required,
            "preferred_generation_cap": preferred,
            "declared_max_generation_cap": ceiling,
        }
        rows.append(row)
        references.append(reference)
        rendered[candidate_id] = hashlib.sha256(
            f"rendered:{candidate_id}".encode()).hexdigest()
    return rows, references, rendered


def _plan(rows, references, rendered, model="qwen25"):
    return canonical_generation_plan(
        rows, references, model, CANONICAL_BATCH_SIZE, rendered,
        "a" * 64, "b" * 64)


def test_interval_remainders_pack_into_exact_batches() -> None:
    # Exact-cap grouping would strand 4 rows at cap 64 and 4 at cap 65. Their
    # authenticated intervals permit one exact-8 cap-64 batch instead.
    intervals = (
        [(64, 64, 64)] * 4 +
        [(64, 65, 65)] * 4 +
        [(65, 80, 100)] * 8
    )
    rows, references, rendered = _fake_sources(intervals)
    first = _plan(rows, references, rendered)
    second = _plan(rows, references, rendered)
    assert first["plan_sha256"] == second["plan_sha256"]
    assert len(first["batches"]) == 2
    assert all(len(batch["rows"]) == 8 for batch in first["batches"])
    assert first["batches"][0]["identity"]["actual_generation_cap"] == 64
    require_generation_plan_integrity(first)


def test_qwen35_authenticated_exact16_plan() -> None:
    rows, references, rendered = _fake_sources([(64, 80, 120)] * 32)
    plan = canonical_generation_plan(
        rows, references, "qwen35", 16, rendered, "a" * 64, "b" * 64)
    assert plan["identity"]["algorithm"] == "exact16_interval_edf_v1"
    assert len(plan["batches"]) == 2
    assert all(len(batch["rows"]) == 16 for batch in plan["batches"])
    require_generation_plan_integrity(plan)


def test_impossible_exact8_intervals_fail_closed() -> None:
    intervals = [(64, 64, 64)] * 4 + [(65, 65, 65)] * 12
    rows, references, rendered = _fake_sources(intervals)
    with pytest.raises(RuntimeError, match="packing is impossible"):
        _plan(rows, references, rendered)


def test_canary_resume_and_partial_crash_keep_batch_identity() -> None:
    rows, references, rendered = _fake_sources([(64, 80, 120)] * 16)
    plan = _plan(rows, references, rendered)
    all_ids = {row["candidate_id"] for row in rows}
    scheduled = schedule_generation_batches(plan, all_ids)
    canary = select_canary_batch(scheduled)
    canary_ids = [member["candidate_id"]
                  for member in canary["identity"]["members"]]
    assert len(canary_ids) == len(set(canary_ids)) == 8

    resumed = schedule_generation_batches(plan, all_ids - set(canary_ids))
    assert len(resumed) == 1
    assert resumed[0]["batch_sha256"] != canary["batch_sha256"]

    # An interrupted append with only three durable rows recomputes all eight
    # companions, while marking only the five missing records for append.
    missing = set(canary_ids[3:])
    partial = schedule_generation_batches(plan, missing)
    assert len(partial) == 1
    assert partial[0]["batch_sha256"] == canary["batch_sha256"]
    assert len(partial[0]["rows"]) == 8
    assert partial[0]["write_candidate_ids"] == canary_ids[3:]

    # If other untouched full batches remain after a 3/8 canary crash, the
    # next canary invocation resumes the partial canonical batch first.
    after_crash_pending = (all_ids - set(canary_ids[:3]))
    after_crash = schedule_generation_batches(plan, after_crash_pending)
    resumed_canary = select_canary_batch(after_crash)
    assert resumed_canary["batch_sha256"] == canary["batch_sha256"]
    assert resumed_canary["write_candidate_ids"] == canary_ids[3:]


@pytest.mark.parametrize("mutation", ["row_order", "member_order", "cap"])
def test_plan_tampering_is_rejected(mutation: str) -> None:
    rows, references, rendered = _fake_sources([(64, 80, 120)] * 8)
    plan = copy.deepcopy(_plan(rows, references, rendered))
    if mutation == "row_order":
        plan["batches"][0]["rows"][0:2] = reversed(
            plan["batches"][0]["rows"][0:2])
    elif mutation == "member_order":
        plan["batches"][0]["identity"]["members"][0:2] = reversed(
            plan["batches"][0]["identity"]["members"][0:2])
    else:
        plan["batches"][0]["identity"]["actual_generation_cap"] += 1
    with pytest.raises(RuntimeError):
        require_generation_plan_integrity(plan)


@pytest.mark.parametrize("model,expected_rows,expected_batches", [
    ("qwen25", 640, 80),
    ("qwen3", 256, 32),
    ("qwen35", 640, 80),
])
@pytest.mark.skipif(not GENERATED.is_dir(),
                    reason="operator-private generated corpus quarantined")
def test_current_artifacts_have_exact_model_specific_plans(
        model: str, expected_rows: int, expected_batches: int) -> None:
    manifest_path = GENERATED / "prompt_candidates_v2.jsonl"
    reference_path = GENERATED / "quality_reference_v2.jsonl"
    all_rows = _jsonl(manifest_path)
    references = _jsonl(reference_path)
    rows = (all_rows if model != "qwen3" else [
        row for row in all_rows if row["pool"] in {"id_test", "ood_test"}
    ])
    rendered = {
        row["candidate_id"]: hashlib.sha256(
            f"{model}:{row['candidate_id']}".encode()).hexdigest()
        for row in rows
    }
    plan = canonical_generation_plan(
        rows, references, model, 8, rendered,
        _file_sha256(manifest_path), _file_sha256(reference_path))
    assert len(rows) == expected_rows
    assert len(plan["batches"]) == expected_batches
    assert all(len(batch["rows"]) == 8 for batch in plan["batches"])
    assert len({
        member["candidate_id"] for batch in plan["batches"]
        for member in batch["identity"]["members"]
    }) == expected_rows
    require_generation_plan_integrity(plan)
