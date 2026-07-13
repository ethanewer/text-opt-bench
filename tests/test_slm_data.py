"""Tokenizer-level checks for manually masked Qwen SFT conversations."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.slm_data import (calibration_record, conversation_record,
                            tokenize_with_assistant_mask)


def main():
    from transformers import AutoTokenizer

    models = (
        "/tmp/qwen2.5-0.5b-instruct",
        "/tmp/qwen3-06b",
        "/tmp/qwen35-08b",
    )
    messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Name one prime greater than ten."},
        {"role": "assistant", "content": "Eleven."},
        {"role": "user", "content": "Now name another."},
        {"role": "assistant", "content": "Thirteen."},
    ]
    for model in models:
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        ids, mask = tokenize_with_assistant_mask(tokenizer, messages)
        assert len(ids) == len(mask) and 0 < sum(mask) < len(ids)
        targets = [token for token, selected in zip(ids, mask) if selected]
        text = tokenizer.decode(targets)
        assert "Eleven" in text and "Thirteen" in text
        assert "Name one prime" not in text and "Now name" not in text
        # The Qwen3 nonthinking scaffold is part of the prompt and must not be
        # optimized as assistant SFT output.
        assert "<think>" not in text and "</think>" not in text
    tokenizer = AutoTokenizer.from_pretrained(models[0], local_files_only=True)
    candidate = {
        "candidate_id": "relation-probe",
        "family": "general_chat_writing",
        "template_cluster": "general_chat_writing:probe",
        "messages": messages[:3],
    }
    for relation in ("training", "development", "overlapping"):
        row = conversation_record(
            {**candidate, "domain_relation": relation}, "qwen25", tokenizer)
        assert row["domain_group"] == "overlap", relation
    row = conversation_record(
        {**candidate, "domain_relation": "heldout"}, "qwen25", tokenizer)
    assert row["domain_group"] == "heldout"
    try:
        conversation_record(
            {**candidate, "domain_relation": "unknown"}, "qwen25", tokenizer)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown domain relation was silently treated as OOD")

    qwen3_tokenizer = AutoTokenizer.from_pretrained(
        models[1], local_files_only=True)
    prompt_only = calibration_record(
        {**candidate, "domain_relation": "development"},
        "qwen3", qwen3_tokenizer, prompt_only=True)
    prompt_messages = messages[:2]
    expected = qwen3_tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False)
    if hasattr(expected, "get") and expected.get("input_ids") is not None:
        expected = expected["input_ids"]
    if hasattr(expected, "tolist"):
        expected = expected.tolist()
    if expected and isinstance(expected[0], list):
        assert len(expected) == 1
        expected = expected[0]
    assert prompt_only["input_ids"] == expected
    assert prompt_only["prompt_only"]
    assert prompt_only["add_generation_prompt"]
    assert prompt_only["generation_scaffold_tokens"] > 0
    assert prompt_only["fabricated_assistant_targets"] is False
    assert all(message["role"] != "assistant"
               for message in prompt_only["messages"])
    print("SLM conversation masking checks passed")


if __name__ == "__main__":
    main()
