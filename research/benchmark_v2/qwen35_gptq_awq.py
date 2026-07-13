"""Calibrate Qwen3.5 GPTQ/AWQ baselines with GPTQModel on Apple MPS.

GPTQModel 7.0's Qwen3.5 definition currently assumes the multimodal wrapper.
The benchmark pins the text-only view of the same checkpoint, so this runner
narrows that definition to AutoModelForCausalLM without changing its layer
grouping or quantization algorithms.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

ROOT = Path(__file__).resolve().parents[2]
TEXT_MODEL = Path("/tmp/qwen35-08b-text-gptq-v3")
DATA = ROOT / "bench/tasks/slm_compression_qwen35/data/train.json"


def patch_text_only_definition():
    from transformers import AutoModelForCausalLM
    from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel

    Qwen3_5QModel.loader = AutoModelForCausalLM
    Qwen3_5QModel.require_load_processor = False
    Qwen3_5QModel.pre_lm_head_norm_module = "model.norm"
    Qwen3_5QModel.rotary_embedding = "model.rotary_emb"
    Qwen3_5QModel.module_tree = [
        "model", "layers", "#", Qwen3_5QModel.module_tree[-1]]


def patch_awq_shared_linear_attention_scaling():
    """Compensate every consumer of Qwen3.5's shared input RMSNorm.

    GPTQModel's generic AWQ walker learns a scale for ``in_proj_qkv`` and
    applies the inverse transform to the shared ``input_layernorm``.  Qwen3.5
    has three additional projections fed by that same norm.  Two are
    intentionally excluded from weight quantization, so the generic walker
    omits them and silently changes the model function.  Apply the same
    function-preserving weight/input transform to all sibling consumers.
    """
    import torch
    from gptqmodel.looper import awq_processor
    from gptqmodel.quantization.awq.utils.module import get_op_by_name

    if getattr(awq_processor.apply_scale,
               "_qwen35_shared_consumers_patched", False):
        return
    original = awq_processor.apply_scale

    @torch.inference_mode()
    def apply_scale(module, scales_list, input_feat_dict=None):
        original(module, scales_list, input_feat_dict=input_feat_dict)
        for prev_name, layer_names, scales, _loss in scales_list:
            if (prev_name != "input_layernorm" or
                    "linear_attn.in_proj_qkv" not in layer_names):
                continue
            for name in ("linear_attn.in_proj_a", "linear_attn.in_proj_b"):
                if name in layer_names:
                    continue
                sibling = get_op_by_name(module, name)
                if sibling is None:
                    continue
                sibling.to(scales.device)
                sibling.weight.mul_(scales.view(1, -1).to(
                    device=sibling.weight.device, dtype=sibling.weight.dtype))
                sibling.cpu()
                if input_feat_dict is not None and name in input_feat_dict:
                    features = input_feat_dict[name]
                    features.div_(scales.view(1, -1).to(
                        device=features.device, dtype=features.dtype))

    apply_scale._qwen35_shared_consumers_patched = True
    awq_processor.apply_scale = apply_scale


def patch_awq_parallel_linear_attention_groups():
    """Describe Qwen3.5's qkv/z/a/b projections as parallel consumers.

    GPTQModel's upstream definition places qkv and z in sequential AWQ groups,
    which creates a function-changing qkv->z equalization edge.  They are
    actually parallel projections of the same normalized hidden state.
    """
    from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel

    tree = dict(Qwen3_5QModel.module_tree[-1])
    tree["linear_attn"] = (
        "norm:!", "conv1d:!", "in_proj_qkv:0", "in_proj_z:0",
        "in_proj_b:!:0", "in_proj_a:!:0", "out_proj:1")
    Qwen3_5QModel.module_tree = ["model", "layers", "#", tree]


def calibration_rows():
    payload = json.loads(DATA.read_text())
    rows = payload["calibration"]["qwen35"]
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 calibration rows, found {len(rows)}")
    return [{"input_ids": row["input_ids"],
             "attention_mask": [1] * len(row["input_ids"])} for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gptq", "awq"), required=True)
    parser.add_argument("--bits", type=int, choices=(3, 4), required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.method == "awq" and args.bits != 4:
        parser.error("the paper-compatible AWQ torch backend supports 4 bits")

    import torch
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    patch_text_only_definition()
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from gptqmodel.quantization import METHOD
    from bench.slm_mps_lock import exclusive_mps_lock, operator_mps_phase

    method = METHOD.GPTQ if args.method == "gptq" else METHOD.AWQ
    if args.method == "awq":
        patch_awq_parallel_linear_attention_groups()
        patch_awq_shared_linear_attention_scaling()
    backend = (BACKEND.GPTQ_TORCH if args.method == "gptq"
               else BACKEND.AWQ_TORCH)
    device = "mps:0" if args.device == "mps" else "cpu"
    # GPTQ's standard W4 configuration is symmetric.  AWQ's published/default
    # W4A16 configuration instead uses affine (asymmetric) zero points.  Using
    # ``sym=True`` for AWQ is accepted by GPTQModel but is a materially
    # different method and catastrophically degraded this hybrid checkpoint.
    symmetric = args.method == "gptq"
    config = QuantizeConfig(
        bits=args.bits, group_size=args.group_size, method=method,
        sym=symmetric, desc_act=False, lm_head=False, device=device,
        calibration_data_device=device, offload_to_disk=False)
    if args.method == "gptq":
        # One joint Hessian/capture pass per decoder layer. This is the
        # original GPTQ layerwise formulation and avoids five redundant
        # activation replays per hybrid Qwen decoder block.
        config.true_sequential = False
    else:
        # The 0.8B model fits the eager AWQ reconstruction search comfortably;
        # retaining full activations avoids 20 repeated micro-batch traversals
        # for every scale candidate.
        config.scale_search_chunked_activations = False
    # GPTQModel 7.0 otherwise aliases the sole Apple device as both ``mps``
    # and ``mps:0`` and attempts a spurious two-device forward pool.
    config.auto_forward_data_parallel = False
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with operator_mps_phase(f"{args.method}-{args.bits}bit-calibration"):
        with exclusive_mps_lock(
                purpose=f"paper-native:{args.method}-{args.bits}bit") as lock:
            model = GPTQModel.load(
                str(TEXT_MODEL), quantize_config=config,
                trust_remote_code=False)
            loaded = time.monotonic()
            log = model.quantize(
                calibration_rows(), batch_size=args.batch_size, backend=backend,
                calibration_sort=None)
            quantized = time.monotonic()
            model.save(str(args.output))
            saved = time.monotonic()
    metadata = {
        "method": args.method, "bits": args.bits,
        "symmetric": symmetric,
        "group_size": args.group_size, "calibration_conversations": 128,
        "calibration_tokens": sum(len(row["input_ids"])
                                  for row in calibration_rows()),
        "backend": backend.value, "device": args.device,
        "mps_fallback": False, "lock": lock,
        "seconds": {"load": loaded - started,
                    "quantize": quantized - loaded,
                    "save": saved - quantized,
                    "total": saved - started},
        "quant_log": log,
    }
    (args.output / "benchmark_calibration.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n")
    print(json.dumps(metadata["seconds"], indent=2))


if __name__ == "__main__":
    main()
