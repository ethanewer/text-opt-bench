"""Timed GPTQ/AWQ W4 quantization of Qwen2.5-0.5B-Instruct on MPS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
MODEL = Path("/private/tmp/qwen2.5-0.5b-instruct")
DATA = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated/qwen25_generated_selected.json")


def calibration_rows():
    rows = [r for r in json.loads(DATA.read_text())["records"]
            if r["split"] == "calibration"]
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 calibration rows, got {len(rows)}")
    return [{"input_ids": r["input_ids"],
             "attention_mask": [1] * len(r["input_ids"])} for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gptq", "awq"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    import gptqmodel
    gptqmodel.run_torch_linalg_warmup = lambda device, warmup_ctx: None
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from gptqmodel.quantization import METHOD
    from bench.slm_mps_lock import exclusive_mps_lock, operator_mps_phase

    method = METHOD.GPTQ if args.method == "gptq" else METHOD.AWQ
    backend = BACKEND.GPTQ_TORCH if args.method == "gptq" else BACKEND.AWQ_TORCH
    config = QuantizeConfig(
        bits=4, group_size=128, method=method,
        sym=args.method == "gptq", desc_act=False, lm_head=False,
        device="mps", calibration_data_device="mps", offload_to_disk=False)
    config.auto_forward_data_parallel = False
    if args.method == "gptq":
        config.true_sequential = False
    else:
        config.scale_search_chunked_activations = False
    calibration = calibration_rows()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with operator_mps_phase(f"qwen25-{args.method}-w4"):
        with exclusive_mps_lock(purpose=f"paper-native:qwen25-{args.method}") as lock:
            model = GPTQModel.load(str(MODEL), quantize_config=config,
                                   trust_remote_code=False)
            loaded = time.monotonic()
            log = model.quantize(calibration, batch_size=args.batch_size,
                                 backend=backend, calibration_sort=None)
            quantized = time.monotonic()
            model.save(str(args.output))
            saved = time.monotonic()
    metadata = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "method": args.method,
        "bits": 4, "group_size": 128, "symmetric": args.method == "gptq",
        "calibration_conversations": len(calibration),
        "calibration_tokens": sum(len(r["input_ids"]) for r in calibration),
        "backend": backend.value, "device": "mps", "mps_fallback": False,
        "seconds": {"load": loaded-started, "quantize": quantized-loaded,
                    "save": saved-quantized, "total": saved-started},
        "lock": lock, "quant_log": log,
    }
    (args.output / "benchmark_calibration.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n")
    print(json.dumps(metadata["seconds"], indent=2))


if __name__ == "__main__":
    main()
