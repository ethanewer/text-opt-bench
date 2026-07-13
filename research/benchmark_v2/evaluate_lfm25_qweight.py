"""Grade one LFM QWeight bundle on the canonical hard-eval splits."""

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
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from bench.ml_models import attest_fresh_mps_torch_import, require_fresh_torch_import
    require_fresh_torch_import("LFM QWeight grading")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    attest_fresh_mps_torch_import(torch, "LFM QWeight grading")
    from bench.qweight import bundle_bytes, decode_bundle
    from bench.slm_mps_lock import exclusive_mps_lock
    from bench.slm_sft import per_conversation_nll

    payload = json.loads(DATA.read_text())
    rows = {split: [row for row in payload["records"] if row["split"] == split]
            for split in SPLITS}
    device = torch.device("mps")
    started = time.monotonic()
    with exclusive_mps_lock(purpose="paper-native:lfm25-qweight-starter") as lock:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), local_files_only=True, dtype=torch.float32).to(device).eval()
        reference = {split: per_conversation_nll(
            torch, F, model, rows[split], device, 4) for split in SPLITS}
        shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        manifest, decoded = decode_bundle(
            args.bundle, shapes, BASE_MODEL, BASE_REVISION, device)
        with torch.no_grad():
            for name, destination in model.state_dict().items():
                destination.copy_(decoded[name])
        del decoded
        gc.collect()
        torch.mps.empty_cache()
        compressed = {split: per_conversation_nll(
            torch, F, model, rows[split], device, 4) for split in SPLITS}
        del model
        torch.mps.empty_cache()
    deltas = {split: [value - base for value, base in zip(
        compressed[split], reference[split])] for split in SPLITS}
    result = {
        "name": manifest["producer"],
        "bundle_bytes": bundle_bytes(args.bundle),
        "bpw": bundle_bytes(args.bundle) * 8 / PARAMETERS,
        "delta_nll": {split: statistics.fmean(deltas[split]) for split in SPLITS},
        "test_all_delta_nll": statistics.fmean(
            deltas["id_test"] + deltas["ood_test"]),
        "seconds": time.monotonic() - started,
        "device": "mps", "mps_fallback": False, "lock": lock,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
