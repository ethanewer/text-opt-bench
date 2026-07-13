"""Time strict-MPS validation grading for native/GGUF/GPTQModel LFM checkpoints."""

from __future__ import annotations

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
GGUF = Path("/private/tmp/lfm25-230m-gguf")
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--gguf", action="store_true")
    parser.add_argument("--gptqmodel", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    from bench.ml_models import (attest_fresh_mps_torch_import,
                                 require_fresh_torch_import)
    strict_label = "LFM2.5 checkpoint grading"
    require_fresh_torch_import(strict_label)
    import torch
    attest_fresh_mps_torch_import(torch, strict_label)
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from bench.slm_mps_lock import exclusive_mps_lock, operator_mps_phase
    from bench.slm_sft import per_conversation_nll

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    rows = [r for r in json.loads(args.data.read_text())["records"]
            if r["split"] == "validation"]
    device = torch.device("mps")
    jobs = []
    if args.include_native:
        jobs.append(("BF16-native", None, None))
    if args.gguf:
        for path in sorted(GGUF.glob("*.gguf")):
            jobs.append((path.stem.removeprefix("LFM2.5-230M-"), path, "gguf"))
    for path in args.gptqmodel:
        jobs.append((path.name, path, "gptqmodel"))
    results = []
    with operator_mps_phase("lfm25-checkpoint-grading"):
        with exclusive_mps_lock(purpose="paper-native:lfm25-checkpoint-grading") as lock:
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
                    # GPTQModel mutates this environment variable after torch
                    # has already been imported under the strict disabled
                    # policy. Restore the attested policy before scoring.
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(MODEL), local_files_only=True, dtype=torch.float32)
                loaded = time.monotonic()
                model.to(device).eval()
                placed = time.monotonic()
                values = per_conversation_nll(
                    torch, F, model, rows, device, args.batch_size)
                scored = time.monotonic()
                size = path.stat().st_size if path and path.is_file() else (
                    (path / "model.safetensors").stat().st_size
                    if path else (MODEL / "model.safetensors").stat().st_size)
                results.append({
                    "name": name, "kind": kind or "native", "path": str(path or MODEL),
                    "bytes": size, "whole_file_bpw": size * 8 / 229_693_184,
                    "mean_assistant_nll": statistics.fmean(values),
                    "median_assistant_nll": statistics.median(values),
                    "conversations": len(values),
                    "seconds": {"load": loaded-started, "place": placed-loaded,
                                "score": scored-placed, "total": scored-started},
                })
                del model
                gc.collect()
                torch.mps.empty_cache()
    base = next((x["mean_assistant_nll"] for x in results
                 if x["name"] == "BF16-native"), None)
    if base is not None:
        for item in results:
            item["delta_nll"] = item["mean_assistant_nll"] - base
    payload = {"model": "LiquidAI/LFM2.5-230M", "device": "mps",
               "mps_fallback": False, "batch_size": args.batch_size,
               "validation_conversations": len(rows), "results": results,
               "lock": lock}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
