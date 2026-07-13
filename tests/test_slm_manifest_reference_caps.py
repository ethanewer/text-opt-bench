"""Regression checks for reference-aware SLM generation caps."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from research.slm_sft_data import build_manifest_v2_640 as builder
from research.slm_sft_data import tokenizer_pins


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "research" / "slm_sft_data" / "generated"
CATALOG = ROOT / "research" / "slm_sft_data" / "catalog_v2"
if CATALOG.is_dir():
    from research.slm_sft_data.catalog_v2 import development as development_catalog
    from research.slm_sft_data.catalog_v2 import tests as sealed_catalog
else:  # Operator-private sources are intentionally absent during campaigns.
    development_catalog = sealed_catalog = None


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_required_generation_cap_formula() -> None:
    maximum, slack, required = builder.required_generation_cap({
        "qwen25": 100,
        "qwen3": 149,
        "qwen35": 150,
    })
    assert maximum == 150
    assert slack == max(12, math.ceil(0.08 * 150)) == 12
    assert required == 162


def test_creative_reference_word_range_gate() -> None:
    record = {
        "candidate_id": "v2_ood_creative_design_storytelling_11",
        "answer_key": (
            "Build practical bicycle maintenance confidence across six "
            "welcoming beginner sessions; all tools are provided, so bring "
            "your questions and get ready to practice together."),
    }
    assert builder.reference_word_count(record["answer_key"]) == 24
    assert builder.require_reference_word_range(record) == 24

    record["answer_key"] = " ".join(f"word{index}" for index in range(19))
    with pytest.raises(RuntimeError, match="reference has 19 words"):
        builder.require_reference_word_range(record)


@pytest.mark.skipif(not CATALOG.is_dir(), reason="operator-private catalog quarantined")
def test_audited_code_boundary_facts_are_explicit() -> None:
    development = {row["candidate_id"]: row
                   for row in development_catalog.RECORDS}
    for index in range(8):
        row = development[f"v2_dev_code_agent_tools_08_{index + 1:02d}"]
        assert development_catalog.SMALL_CODE_REVIEW_BOUNDARIES[index] in (
            row["required_facts"])
    owner = development["v2_dev_code_agent_tools_08_08"]["answer_key"]
    assert "blank text leaked an empty string" in owner

    sealed = {row["candidate_id"]: row for row in sealed_catalog.RECORDS}
    score_sort = sealed["v2_id_code_agent_tools_28"]
    assert "string keys to finite numeric values" in score_sort["prompt"]
    assert "finite numeric score values support unary negation" in (
        score_sort["required_facts"])


def test_all_tokenizer_snapshot_files_are_authenticated(monkeypatch) -> None:
    paths = {key: f"/snapshot/{key}" for key in
             tokenizer_pins.PINNED_TOKENIZER_FILES}
    filename_to_field = {
        "tokenizer_config.json": "tokenizer_config_sha256",
        "tokenizer.json": "tokenizer_json_sha256",
        "vocab.json": "vocab_json_sha256",
        "merges.txt": "merges_txt_sha256",
    }

    def pinned_hash(path: Path) -> str:
        return tokenizer_pins.PINNED_TOKENIZER_FILES[path.parent.name][
            filename_to_field[path.name]]

    monkeypatch.setattr(tokenizer_pins, "file_sha256", pinned_hash)
    assert tokenizer_pins.require_pinned_tokenizer_snapshots(paths) == (
        tokenizer_pins.PINNED_TOKENIZER_FILES)

    def tampered_hash(path: Path) -> str:
        if path.parent.name == "qwen3" and path.name == "merges.txt":
            return "0" * 64
        return pinned_hash(path)

    monkeypatch.setattr(tokenizer_pins, "file_sha256", tampered_hash)
    with pytest.raises(RuntimeError, match="qwen3"):
        tokenizer_pins.require_pinned_tokenizer_snapshots(paths)


@pytest.mark.skipif(not GENERATED.is_dir(), reason="operator-private generated corpus quarantined")
def test_built_manifest_reference_caps() -> None:
    manifest = _jsonl(GENERATED / "prompt_candidates_v2.jsonl")
    references = {
        row["candidate_id"]: row
        for row in _jsonl(GENERATED / "quality_reference_v2.jsonl")
    }
    assert len(manifest) == len(references) == 640
    script_sha = hashlib.sha256(Path(builder.__file__).read_bytes()).hexdigest()

    for row in manifest:
        reference = references[row["candidate_id"]]
        maximum, slack, reference_required = builder.required_generation_cap(
            reference["answer_token_counts"])
        required = reference["required_generation_cap"]
        declared_maximum = row["generation"]["max_new_tokens_per_turn"]
        prompt_maximum = max(row["prompt_token_counts"].values())
        assert reference["max_answer_tokens"] == maximum
        assert reference["generation_cap_safety_tokens"] == slack
        assert reference["reference_required_generation_cap"] == reference_required
        assert required == max(
            64,
            reference_required,
            reference["word_required_generation_cap"],
        )
        assert required <= reference["preferred_generation_cap"]
        assert reference["preferred_generation_cap"] <= declared_maximum
        assert declared_maximum == row["generation"][
            "declared_max_new_tokens_per_turn"]
        assert reference["declared_max_generation_cap"] == declared_maximum
        assert declared_maximum >= required
        assert (prompt_maximum + declared_maximum <=
                builder.PROMPT_AND_CAP_LIMIT)
        assert row["provenance"]["build_script_sha256"] == script_sha
        if row["interaction_format"] == "sql_zero_preserving_aggregate":
            assert prompt_maximum <= 330
