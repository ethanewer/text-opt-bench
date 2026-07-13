"""Score LFM checkpoints on validation, ID test, and OOD test in one load."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
MODEL = Path("/private/tmp/lfm25-230m-source")
DATA = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated/lfm25_hard_eval_selected.json")
GGUF = Path("/private/tmp/lfm25-230m-gguf")
SPLITS = ("validation", "id_test", "ood_test")


def stratified_ci(deltas, families, seed, repeats=2000):
    """Paired, family-stratified bootstrap interval for a mean delta."""
    grouped = {}
    for value, family in zip(deltas, families):
        grouped.setdefault(family, []).append(value)
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        sampled = []
        for family in sorted(grouped):
            values = grouped[family]
            sampled.extend(rng.choice(values) for _ in values)
        draws.append(statistics.fmean(sampled))
    draws.sort()
    return [draws[int(0.025 * repeats)], draws[int(0.975 * repeats)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gguf", action="store_true")
    parser.add_argument("--gptqmodel", type=Path, action="append", default=[])
    parser.add_argument("--qweight", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    from bench.ml_models import (attest_fresh_mps_torch_import,
                                 require_fresh_torch_import)
    label = "LFM2.5 multisplit checkpoint grading"
    require_fresh_torch_import(label)
    import torch
    attest_fresh_mps_torch_import(torch, label)
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from bench.slm_mps_lock import exclusive_mps_lock, operator_mps_phase
    from bench.slm_sft import per_conversation_nll

    payload = json.loads(args.data.read_text())
    rows = {split: [r for r in payload["records"] if r["split"] == split]
            for split in SPLITS}
    if any(len(rows[split]) != 128 for split in SPLITS):
        raise RuntimeError({split: len(value) for split, value in rows.items()})
    jobs = [("BF16-native", None, "native")]
    if args.gguf:
        for path in sorted(GGUF.glob("*.gguf")):
            jobs.append((path.stem.removeprefix("LFM2.5-230M-"), path, "gguf"))
    for path in args.gptqmodel:
        jobs.append((path.name, path, "gptqmodel"))
    for path in args.qweight:
        jobs.append((path.name, path, "qweight"))

    device = torch.device("mps")
    results, all_score_values = [], []
    with operator_mps_phase("lfm25-multisplit-grading"):
        with exclusive_mps_lock(purpose="paper-native:lfm25-multisplit") as lock:
            for name, path, kind in jobs:
                started = time.monotonic()
                if kind == "gguf":
                    model = AutoModelForCausalLM.from_pretrained(
                        str(MODEL), gguf_file=str(path), local_files_only=True,
                        dtype=torch.float32)
                elif kind == "gptqmodel":
                    from gptqmodel import BACKEND, GPTQModel
                    model = GPTQModel.load(str(path), backend=BACKEND.TORCH,
                                           device="mps:0").model
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
                elif kind == "qweight":
                    from bench.qweight import decode_bundle
                    model = AutoModelForCausalLM.from_pretrained(
                        str(MODEL), local_files_only=True, dtype=torch.float32)
                    shapes = {key: tuple(value.shape)
                              for key, value in model.state_dict().items()}
                    manifest, decoded = decode_bundle(
                        path, shapes, "LiquidAI/LFM2.5-230M",
                        "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7", device)
                    with torch.no_grad():
                        for key, destination in model.state_dict().items():
                            destination.copy_(decoded[key])
                    name = manifest["producer"]
                    del decoded
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(MODEL), local_files_only=True, dtype=torch.float32)
                loaded = time.monotonic()
                model.to(device).eval()
                score_values, score_seconds = {}, {}
                for split in SPLITS:
                    split_started = time.monotonic()
                    values = per_conversation_nll(
                        torch, F, model, rows[split], device, args.batch_size)
                    score_seconds[split] = time.monotonic() - split_started
                    score_values[split] = values
                finished = time.monotonic()
                if path is None:
                    size = (MODEL / "model.safetensors").stat().st_size
                elif path.is_file():
                    size = path.stat().st_size
                elif kind == "qweight":
                    from bench.qweight import bundle_bytes
                    size = bundle_bytes(path)
                else:
                    size = (path / "model.safetensors").stat().st_size
                means = {split: statistics.fmean(score_values[split])
                         for split in SPLITS}
                test_all = statistics.fmean(
                    score_values["id_test"] + score_values["ood_test"])
                results.append({
                    "name": name, "kind": kind, "bytes": size,
                    "bpw": size * 8 / 229_693_184,
                    "mean_nll": {**means, "test_all": test_all},
                    "seconds": {"load": loaded-started,
                                **score_seconds, "total": finished-started},
                })
                all_score_values.append(score_values)
                del model
                gc.collect()
                torch.mps.empty_cache()
    base = results[0]["mean_nll"]
    base_values = all_score_values[0]
    for result_index, (result, values_by_split) in enumerate(
            zip(results, all_score_values)):
        result["delta_nll"] = {
            split: result["mean_nll"][split] - base[split]
            for split in (*SPLITS, "test_all")}
        family_deltas = {}
        ci95 = {}
        mean_abs = {}
        mean_positive = {}
        negative_fraction = {}
        combined_deltas, combined_families = [], []
        for split_index, split in enumerate(SPLITS):
            deltas = [value - reference for value, reference in zip(
                values_by_split[split], base_values[split])]
            families = [row["domain"] for row in rows[split]]
            mean_abs[split] = statistics.fmean(abs(value) for value in deltas)
            mean_positive[split] = statistics.fmean(max(value, 0.0)
                                                    for value in deltas)
            negative_fraction[split] = sum(
                value < 0 for value in deltas) / len(deltas)
            grouped = {}
            for value, family in zip(deltas, families):
                grouped.setdefault(family, []).append(value)
            family_deltas[split] = {
                family: statistics.fmean(grouped[family])
                for family in sorted(grouped)}
            ci95[split] = stratified_ci(
                deltas, families, seed=20260711 + 100 * result_index + split_index)
            if split in ("id_test", "ood_test"):
                combined_deltas.extend(deltas)
                combined_families.extend(f"{split}:{family}" for family in families)
        ci95["test_all"] = stratified_ci(
            combined_deltas, combined_families,
            seed=20261711 + 100 * result_index)
        mean_abs["test_all"] = statistics.fmean(
            abs(value) for value in combined_deltas)
        mean_positive["test_all"] = statistics.fmean(
            max(value, 0.0) for value in combined_deltas)
        negative_fraction["test_all"] = sum(
            value < 0 for value in combined_deltas) / len(combined_deltas)
        result["delta_nll_ci95"] = ci95
        result["family_delta_nll"] = family_deltas
        result["mean_absolute_delta_nll"] = mean_abs
        result["mean_positive_delta_nll"] = mean_positive
        result["negative_delta_fraction"] = negative_fraction
    output = {"model": "LiquidAI/LFM2.5-230M", "device": "mps",
              "mps_fallback": False, "metric": "mean per-conversation assistant-token NLL",
              "split_counts": {split: len(value) for split, value in rows.items()},
              "results": results, "lock": lock}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
