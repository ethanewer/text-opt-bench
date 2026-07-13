"""Regenerate the frozen SLM prompts with LFM2.5-230M on strict Apple MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated")
MODEL = Path("/private/tmp/lfm25-230m-source")


def load_inputs():
    selected = json.loads((PRIVATE / "selected_corpus.json").read_text())
    candidates = {}
    with (PRIVATE / "prompt_candidates_v2.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            candidates[row["candidate_id"]] = row
    caps = {}
    with (PRIVATE / "raw_v2/qwen35.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            caps[row["candidate_id"]] = int(
                row.get("generation", {}).get("max_new_tokens_per_turn", 128))
    split_ids = {
        "calibration": selected["development"]["calibration"],
        "validation": selected["development"]["validation"],
        "id_test": selected["test"]["overlap"],
        "ood_test": selected["test"]["heldout"],
    }
    return candidates, caps, split_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expansion-prompts", type=Path)
    parser.add_argument("--base-data", type=Path)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bench.slm_data import conversation_record
    from bench.slm_mps_lock import exclusive_mps_lock, operator_mps_phase

    if not torch.backends.mps.is_available():
        raise RuntimeError("strict MPS is required")
    if args.expansion_prompts:
        expansion = json.loads(args.expansion_prompts.read_text())
        selected_rows = [(row["split"], row, 192)
                         for row in expansion["candidates"]]
        split_ids = expansion["counts"]
    else:
        candidates, caps, split_ids = load_inputs()
        selected_rows = []
        for split, ids in split_ids.items():
            for candidate_id in ids:
                row = candidates[candidate_id]
                selected_rows.append((split, row, caps.get(candidate_id, 128)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with operator_mps_phase("lfm25-selected-datagen"):
        with exclusive_mps_lock(purpose="paper-native:lfm25-selected-datagen") as lock:
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = 0
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL), local_files_only=True, torch_dtype=torch.bfloat16)
            model.to("mps").eval()
            loaded = time.monotonic()
            prepared = []
            for split, candidate, cap in selected_rows:
                prompt_messages = [dict(m) for m in candidate["messages"]
                                   if m["role"] != "assistant"]
                ids = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=True, add_generation_prompt=True)
                if hasattr(ids, "get"):
                    ids = ids["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                # Reserve chat-template/end-marker and decode/re-encode drift.
                cap = max(1, min(cap, 512 - len(ids) - 16))
                prepared.append((split, candidate, prompt_messages, ids, cap))
            # Group equal caps to avoid changing generation length through padding.
            prepared.sort(key=lambda x: (x[4], len(x[3])))
            records = []
            generated_tokens = 0
            for cap in sorted({x[4] for x in prepared}):
                group = [x for x in prepared if x[4] == cap]
                for offset in range(0, len(group), args.batch_size):
                    batch = group[offset:offset + args.batch_size]
                    encoded = tokenizer.pad(
                        {"input_ids": [x[3] for x in batch]},
                        padding=True, return_tensors="pt")
                    input_ids = encoded["input_ids"].to("mps")
                    attention_mask = encoded["attention_mask"].to("mps")
                    with torch.inference_mode():
                        output = model.generate(
                            input_ids=input_ids, attention_mask=attention_mask,
                            max_new_tokens=cap, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
                    continuation = output[:, input_ids.shape[1]:].cpu().tolist()
                    for item, token_ids in zip(batch, continuation):
                        split, candidate, prompt_messages, _ids, row_cap = item
                        while token_ids and token_ids[-1] == tokenizer.pad_token_id:
                            token_ids.pop()
                        text = tokenizer.decode(token_ids, skip_special_tokens=True)
                        if not text.strip():
                            text = tokenizer.decode(token_ids, skip_special_tokens=False).strip() or " "
                        generated = dict(candidate)
                        generated["messages"] = prompt_messages + [
                            {"role": "assistant", "content": text}]
                        record = conversation_record(generated, "lfm25", tokenizer)
                        record.update({
                            "split": split, "generation_cap": row_cap,
                            "generated_tokens": len(token_ids),
                            "generation_truncated": len(token_ids) >= row_cap,
                        })
                        records.append(record)
                        generated_tokens += len(token_ids)
            finished = time.monotonic()
    if args.base_data:
        base = json.loads(args.base_data.read_text())
        base_records = base["records"]
        overlap = {r["prompt_id"] for r in base_records} & {
            r["prompt_id"] for r in records}
        if overlap:
            raise RuntimeError(f"base/expansion prompt overlap: {sorted(overlap)[:3]}")
        records = base_records + records
        generated_tokens = sum(r.get("generated_tokens", r["assistant_tokens"])
                               for r in records)
    records.sort(key=lambda x: (x["split"], x["prompt_id"]))
    payload = {
        "format": 1, "model": "LiquidAI/LFM2.5-230M",
        "model_revision": "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7",
        "device": "mps", "mps_fallback": False,
        "records": records,
        "timing_seconds": {"load": loaded - started,
                           "generate": finished - loaded,
                           "total": finished - started},
        "counts": {split: sum(r["split"] == split for r in records)
                   for split in ("calibration", "validation", "id_test", "ood_test")},
        "generated_tokens": generated_tokens,
        "lock": lock,
    }
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {k: payload[k] for k in (
        "model", "model_revision", "device", "mps_fallback",
        "timing_seconds", "counts", "generated_tokens")}
    summary["data_sha256"] = digest
    summary["output"] = str(args.output)
    summary_name = ("lfm25_datagen_results_x2.json" if args.base_data
                    else "lfm25_datagen_results.json")
    (ROOT / "research/benchmark_v2" / summary_name).write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
