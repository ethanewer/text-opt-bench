"""Timed paper-style GPTQ/AWQ quantization of LFM2.5-230M."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
MODEL = Path("/private/tmp/lfm25-230m-source")
DATA = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated/lfm25_hard_eval_selected.json")


def patch_lfm2(method):
    import torch
    from transformers.masking_utils import create_causal_mask
    from gptqmodel.models.definitions.lfm2 import LFM2QModel
    # Upstream plain LFM2 omitted the hybrid conv/attention replay-mask fix.
    def prepare_layer_replay_kwargs(self, layer, layer_input,
                                    additional_inputs, target_device):
        attention_mask = additional_inputs.get("attention_mask")
        if attention_mask is None:
            return additional_inputs
        if not getattr(layer, "is_attention_layer", False):
            if torch.is_tensor(attention_mask) and attention_mask.dtype != torch.bool:
                additional_inputs["attention_mask"] = attention_mask.bool()
            return additional_inputs
        if not layer_input or not torch.is_tensor(layer_input[0]):
            return additional_inputs
        layer_config = getattr(layer, "config", None)
        if layer_config is None:
            layer_config = getattr(getattr(layer, "self_attn", None), "config", None)
        if layer_config is not None:
            additional_inputs["attention_mask"] = create_causal_mask(
                config=layer_config, inputs_embeds=layer_input[0],
                attention_mask=attention_mask,
                past_key_values=additional_inputs.get("past_key_values"),
                position_ids=additional_inputs.get("position_ids"))
        return additional_inputs
    LFM2QModel.prepare_layer_replay_kwargs = prepare_layer_replay_kwargs
    if method == "awq":
        tree = dict(LFM2QModel.module_tree[-1])
        # conv.out_proj consumes a nonlinear convolution result, not in_proj's
        # linear output, so it must be a separate AWQ equalization stage.
        tree["conv"] = ("in_proj:0", "out_proj:1")
        LFM2QModel.module_tree = ["model", "layers", "#", tree]


def calibration_rows(path=DATA):
    rows = json.loads(path.read_text())["records"]
    if rows and "split" in rows[0]:
        rows = [r for r in rows if r["split"] == "calibration"]
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 calibration rows, got {len(rows)}")
    return [{"input_ids": r["input_ids"],
             "attention_mask": [1] * len(r["input_ids"])} for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gptq", "awq"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--device", choices=("mps", "cuda"), default="mps")
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 8), default=4)
    args = parser.parse_args()
    if args.method == "awq" and args.bits != 4:
        parser.error("GPTQModel's deployed AWQ quantized-linear kernel supports W4 only")
    import torch
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    patch_lfm2(args.method)
    import gptqmodel
    # GPTQModel 7.2 unnecessarily warms up SVD on MPS, where PyTorch can only
    # execute it through forbidden CPU fallback. Quantization itself does not
    # use that warmup result.
    gptqmodel.run_torch_linalg_warmup = lambda device, warmup_ctx: None
    # Upstream can report an accelerator without its equivalent index and then
    # assert exact device equality in threaded quantization. Canonicalize both
    # paths without enabling fallback.
    from gptqmodel.looper import stage_subset
    original_get_device = stage_subset.get_device
    def canonical_get_device(value):
        device = original_get_device(value)
        if getattr(device, "type", None) in ("mps", "cuda"):
            return torch.device(f"{device.type}:0")
        return device
    stage_subset.get_device = canonical_get_device
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from gptqmodel.quantization import METHOD
    from bench.slm_mps_lock import exclusive_mps_lock, operator_mps_phase

    method = METHOD.GPTQ if args.method == "gptq" else METHOD.AWQ
    backend = BACKEND.GPTQ_TORCH if args.method == "gptq" else BACKEND.AWQ_TORCH
    accelerator = f"{args.device}:0"
    if args.device == "cuda":
        # GPTQModel defaults to four quantization workers per CUDA device.
        # Concurrent first-use torch.linalg calls can race in newer PyTorch
        # nightlies ("lazy wrapper should be called at most once").  One
        # worker is deterministic and is already enough to saturate this
        # small model's single-GPU quantization workload.
        from gptqmodel.utils.threadx import DeviceThreadPool

        gptqmodel._DEVICE_THREAD_POOL = DeviceThreadPool(
            devices=[torch.device(accelerator), torch.device("cpu")],
            inference_mode=True,
            workers={accelerator: 1, "cpu": 1},
            empty_cache_every_n=512,
        )
    config = QuantizeConfig(
        bits=args.bits, group_size=128, method=method,
        sym=args.method == "gptq", desc_act=False, lm_head=False,
        # GPTQModel's threaded worker compares canonical torch devices; using
        # ``mps:0`` avoids treating the equivalent bare ``mps`` as a mismatch.
        device=accelerator, calibration_data_device=accelerator,
        offload_to_disk=False)
    config.auto_forward_data_parallel = False
    if args.method == "gptq":
        config.true_sequential = False
    else:
        config.scale_search_chunked_activations = False
    calibration = calibration_rows(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.device == "mps":
        phase = operator_mps_phase(f"lfm25-{args.method}-w{args.bits}")
        lock_context = exclusive_mps_lock(
            purpose=f"paper-native:lfm25-{args.method}")
    else:
        from bench.slm_cuda_lock import exclusive_cuda_lock
        phase = nullcontext()
        lock_context = exclusive_cuda_lock(
            purpose=f"paper-native:lfm25-{args.method}")
    with phase:
        with lock_context as lock:
            model = GPTQModel.load(str(args.model), quantize_config=config,
                                   trust_remote_code=False)
            loaded = time.monotonic()
            log = model.quantize(calibration, batch_size=args.batch_size,
                                 backend=backend, calibration_sort=None)
            quantized = time.monotonic()
            model.save(str(args.output))
            saved = time.monotonic()
    metadata = {
        "model": "LiquidAI/LFM2.5-230M", "method": args.method,
        "bits": args.bits, "group_size": 128,
        "symmetric": args.method == "gptq",
        "calibration_conversations": len(calibration),
        "calibration_tokens": sum(len(r["input_ids"]) for r in calibration),
        "backend": backend.value, "device": args.device, "mps_fallback": False,
        "quant_workers": 1 if args.device == "cuda" else None,
        "seconds": {"load": loaded-started, "quantize": quantized-loaded,
                    "save": saved-quantized, "total": saved-started},
        "lock": lock, "quant_log": log,
    }
    (args.output / "benchmark_calibration.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n")
    print(json.dumps(metadata["seconds"], indent=2))


if __name__ == "__main__":
    main()
