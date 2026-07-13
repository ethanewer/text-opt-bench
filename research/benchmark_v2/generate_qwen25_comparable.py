"""Regenerate the canonical LFM prompt splits with Qwen2.5-0.5B-Instruct."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/private/tmp/qwen2.5-0.5b-instruct")
SOURCE = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated/lfm25_generated_selected.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    from bench.ml_models import (attest_fresh_mps_torch_import,
                                 require_fresh_torch_import)
    label = "Qwen2.5 comparable corpus generation"
    require_fresh_torch_import(label)
    import torch
    attest_fresh_mps_torch_import(torch, label)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bench.slm_data import conversation_record
    from bench.slm_mps_lock import exclusive_mps_lock, operator_mps_phase

    source = json.loads(SOURCE.read_text())
    jobs = []
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for row in source["records"]:
        prompt_messages = [dict(m) for m in row["messages"] if m["role"] != "assistant"]
        ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True,
                                            add_generation_prompt=True)
        if hasattr(ids, "get"):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        cap = max(1, min(192, 512 - len(ids) - 16))
        candidate = {
            "candidate_id": row["prompt_id"], "family": row["domain"],
            "domain_relation": ("heldout" if row["split"] == "ood_test"
                                else "development"),
            "template_cluster": row["template_cluster"],
        }
        jobs.append((row["split"], candidate, prompt_messages, ids, cap))
    jobs.sort(key=lambda x: (x[4], len(x[3])))
    records, generated_tokens = [], 0
    started = time.monotonic()
    with operator_mps_phase("qwen25-comparable-datagen"):
        with exclusive_mps_lock(purpose="paper-native:qwen25-comparable-datagen") as lock:
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL), local_files_only=True, dtype=torch.bfloat16)
            model.to("mps").eval()
            loaded = time.monotonic()
            for cap in sorted({x[4] for x in jobs}):
                group = [x for x in jobs if x[4] == cap]
                for offset in range(0, len(group), args.batch_size):
                    batch = group[offset:offset + args.batch_size]
                    encoded = tokenizer.pad(
                        {"input_ids": [x[3] for x in batch]}, padding=True,
                        return_tensors="pt")
                    input_ids = encoded["input_ids"].to("mps")
                    attention_mask = encoded["attention_mask"].to("mps")
                    with torch.inference_mode():
                        output = model.generate(
                            input_ids=input_ids, attention_mask=attention_mask,
                            max_new_tokens=cap, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
                    continuations = output[:, input_ids.shape[1]:].cpu().tolist()
                    for item, token_ids in zip(batch, continuations):
                        split, candidate, prompt_messages, _ids, row_cap = item
                        while token_ids and token_ids[-1] == tokenizer.pad_token_id:
                            token_ids.pop()
                        text = tokenizer.decode(token_ids, skip_special_tokens=True)
                        if not text.strip():
                            text = tokenizer.decode(token_ids, skip_special_tokens=False).strip() or " "
                        candidate = dict(candidate)
                        candidate["messages"] = prompt_messages + [
                            {"role": "assistant", "content": text}]
                        record = conversation_record(candidate, "qwen25", tokenizer)
                        record.update({"split": split, "generation_cap": row_cap,
                                       "generated_tokens": len(token_ids),
                                       "generation_truncated": len(token_ids) >= row_cap})
                        records.append(record); generated_tokens += len(token_ids)
            finished = time.monotonic()
    records.sort(key=lambda row: (row["split"], row["prompt_id"]))
    counts = {split: sum(r["split"] == split for r in records)
              for split in ("calibration", "validation", "id_test", "ood_test")}
    if counts != {"calibration": 128, "validation": 128,
                  "id_test": 128, "ood_test": 128}:
        raise RuntimeError(counts)
    payload = {"format": 1, "model": "Qwen/Qwen2.5-0.5B-Instruct",
               "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
               "device": "mps", "mps_fallback": False, "records": records,
               "counts": counts, "generated_tokens": generated_tokens,
               "timing_seconds": {"load": loaded-started,
                                  "generate": finished-loaded,
                                  "total": finished-started}, "lock": lock}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    summary = {k: payload[k] for k in ("model", "model_revision", "device",
               "mps_fallback", "counts", "generated_tokens", "timing_seconds")}
    summary["data_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    (ROOT / "research/benchmark_v2/qwen25_datagen_results.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
