"""Grade several LFM QWeight bundles with one shared native reference pass."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

MODEL = Path("/private/tmp/lfm25-230m-source")
DATA = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated/lfm25_hard_eval_selected.json")
BASE_MODEL = "LiquidAI/LFM2.5-230M"
BASE_REVISION = "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7"
PARAMETERS = 229_693_184
SPLITS = ("validation", "id_test", "ood_test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from bench.ml_models import attest_fresh_mps_torch_import, require_fresh_torch_import
    require_fresh_torch_import("LFM QWeight sweep grading")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    attest_fresh_mps_torch_import(torch, "LFM QWeight sweep grading")
    from bench.qweight import bundle_bytes, decode_bundle
    from bench.slm_mps_lock import exclusive_mps_lock
    from bench.slm_sft import per_conversation_nll

    payload = json.loads(DATA.read_text())
    rows = {split: [row for row in payload["records"] if row["split"] == split]
            for split in SPLITS}
    device = torch.device("mps")
    results = []
    with exclusive_mps_lock(purpose="paper-native:lfm25-rtn-bit-sweep") as lock:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), local_files_only=True, dtype=torch.float32).to(device).eval()
        reference = {split: per_conversation_nll(
            torch, F, model, rows[split], device, 4) for split in SPLITS}
        shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for bundle in args.bundle:
            started = time.monotonic()
            manifest, decoded = decode_bundle(
                bundle, shapes, BASE_MODEL, BASE_REVISION, device)
            with torch.no_grad():
                for name, destination in model.state_dict().items():
                    destination.copy_(decoded[name])
            del decoded
            gc.collect()
            torch.mps.empty_cache()
            compressed = {split: per_conversation_nll(
                torch, F, model, rows[split], device, 4) for split in SPLITS}
            deltas = {split: [value - base for value, base in zip(
                compressed[split], reference[split])] for split in SPLITS}
            size = bundle_bytes(bundle)
            compressed_mean = {
                split: statistics.fmean(compressed[split]) for split in SPLITS}
            compressed_mean["test_all"] = statistics.fmean(
                compressed["id_test"] + compressed["ood_test"])
            results.append({
                "name": manifest["producer"], "bundle": str(bundle),
                "bytes": size, "bpw": size * 8 / PARAMETERS,
                "mean_nll": compressed_mean,
                "delta_nll": {split: statistics.fmean(deltas[split])
                              for split in SPLITS},
                "mean_absolute_delta_nll": {
                    split: statistics.fmean(abs(value) for value in deltas[split])
                    for split in SPLITS},
                "mean_positive_delta_nll": {
                    split: statistics.fmean(max(value, 0.0)
                                            for value in deltas[split])
                    for split in SPLITS},
                "negative_delta_fraction": {
                    split: sum(value < 0 for value in deltas[split]) / len(deltas[split])
                    for split in SPLITS},
                "test_all_delta_nll": statistics.fmean(
                    deltas["id_test"] + deltas["ood_test"]),
                "test_all_mean_absolute_delta_nll": statistics.fmean(
                    abs(value) for value in
                    (deltas["id_test"] + deltas["ood_test"])),
                "test_all_mean_positive_delta_nll": statistics.fmean(
                    max(value, 0.0) for value in
                    (deltas["id_test"] + deltas["ood_test"])),
                "test_all_negative_delta_fraction": sum(
                    value < 0 for value in
                    (deltas["id_test"] + deltas["ood_test"])) / (
                        len(deltas["id_test"]) + len(deltas["ood_test"])),
                "score_seconds": time.monotonic() - started,
            })
    native_mean = {split: statistics.fmean(reference[split]) for split in SPLITS}
    native_mean["test_all"] = statistics.fmean(
        reference["id_test"] + reference["ood_test"])
    output = {"model": BASE_MODEL, "method": "symmetric groupwise RTN",
              "group_size": 40, "device": "mps", "mps_fallback": False,
              "native_mean_nll": native_mean,
              "results": results, "lock": lock}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
