"""Export calibrated GPTQModel weights to safe, whole-model QWeight bundles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qwen35_gptq_awq import patch_text_only_definition


BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def pack(codes, bits):
    values = codes.detach().to("cpu", torch.uint8).numpy().reshape(-1)
    output = np.zeros((math.ceil(values.size * bits / 8),), dtype=np.uint8)
    for start in range(0, values.size, 1_000_000):
        local = values[start:start + 1_000_000].astype(np.uint16)
        positions = np.arange(start, start + local.size, dtype=np.int64) * bits
        byte, shift = positions // 8, positions % 8
        np.bitwise_or.at(output, byte, ((local << shift) & 255).astype(np.uint8))
        crossing = shift + bits > 8
        if crossing.any():
            np.bitwise_or.at(output, byte[crossing] + 1,
                             (local[crossing] >> (8 - shift[crossing])).astype(np.uint8))
    return torch.from_numpy(output)


def rtn(weight, bits, group_size):
    shape = list(weight.shape)
    columns = shape[-1] if shape else 1
    outer = weight.numel() // columns
    groups = math.ceil(columns / group_size)
    matrix = weight.float().reshape(outer, columns)
    matrix = torch.nn.functional.pad(
        matrix, (0, groups * group_size - columns)).reshape(
            outer, groups, group_size)
    levels = (1 << (bits - 1)) - 1
    scales = (matrix.abs().amax(-1).clamp_min(1e-8) / levels).half()
    codes = (matrix / scales.float().unsqueeze(-1)).round().clamp(
        -levels, levels).to(torch.int16) + (1 << (bits - 1))
    codes = codes.reshape(outer, groups * group_size)[:, :columns]
    return codes, scales


def unpack_gptq(module):
    from gptqmodel.utils.model_dequant import unpack_cols, unpack_rows
    codes = unpack_rows(module.qweight.cpu(), module.bits)[
        :module.in_features, :module.out_features]
    zeros = unpack_cols(module.qzeros.cpu(), module.bits)[
        :, :module.out_features]
    return codes.t().contiguous(), zeros.t().contiguous(), module.scales.cpu().t().contiguous()


def unpack_awq_module(module):
    from gptqmodel.quantization.awq.utils.packing_utils import (
        reverse_awq_order, unpack_awq)
    codes, zeros = unpack_awq(module.qweight.cpu(), module.qzeros.cpu(), module.bits)
    codes, zeros = reverse_awq_order(codes, zeros, module.bits)
    mask = (1 << module.bits) - 1
    return ((codes & mask)[:module.in_features, :module.out_features].t().contiguous(),
            (zeros & mask)[:, :module.out_features].t().contiguous(),
            module.scales.cpu().t().contiguous())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("gptq", "awq"), required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-qwen35-awq-shared-inputs",
                        action="store_true")
    args = parser.parse_args()

    patch_text_only_definition()
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    backend = (BACKEND.GPTQ_TORCH if args.method == "gptq"
               else BACKEND.AWQ_TORCH)
    wrapped = GPTQModel.load(str(args.checkpoint), backend=backend, device="cpu")
    model = wrapped.model.eval()
    quantized = {name: module for name, module in model.named_modules()
                 if isinstance(module, BaseQuantLinear)}
    state = model.state_dict()
    repair_factors = {}
    if args.repair_qwen35_awq_shared_inputs:
        if args.method != "awq":
            parser.error("the shared-input repair is specific to AWQ")
        base_path = Path("/tmp/qwen35-08b-text-gptq-v3/model.safetensors")
        if not base_path.is_file():
            raise RuntimeError(f"missing pinned dense checkpoint: {base_path}")
        with safe_open(base_path, framework="pt", device="cpu") as base:
            with torch.no_grad():
                for layer in range(24):
                    prefix = f"model.layers.{layer}"
                    norm_name = prefix + ".input_layernorm.weight"
                    a_name = prefix + ".linear_attn.in_proj_a.weight"
                    b_name = prefix + ".linear_attn.in_proj_b.weight"
                    z_name = prefix + ".linear_attn.in_proj_z"
                    if a_name not in state:
                        continue
                    dense_norm = base.get_tensor(norm_name).float()
                    factor = (state[norm_name].float() / dense_norm).clamp_min(1e-6)
                    repair_factors[z_name] = factor
                    state[a_name].div_(factor.to(state[a_name].dtype).view(1, -1))
                    state[b_name].div_(factor.to(state[b_name].dtype).view(1, -1))
    tensors, records, pointers = {}, {}, {}
    index = 0
    for name, weight in state.items():
        # Packed implementation details are represented by the corresponding
        # logical ``module.weight`` record below.
        if any(name == prefix + suffix for prefix in quantized for suffix in
               (".qweight", ".qzeros", ".scales", ".g_idx", ".bias")):
            continue
        logical = name
        if logical.endswith(".weight") and logical[:-7] in quantized:
            continue
        pointer = (weight.untyped_storage().data_ptr(), weight.storage_offset(),
                   tuple(weight.shape), tuple(weight.stride()))
        if pointer in pointers:
            records[logical] = {"codec": "alias", "source": pointers[pointer]}
            continue
        pointers[pointer] = logical
        if weight.is_floating_point() and weight.ndim:
            codes, scales = rtn(weight, args.bits, 128)
            ck, sk = f"codes_{index}", f"scales_{index}"
            tensors[ck], tensors[sk] = pack(codes, args.bits), scales
            records[logical] = {
                "codec": "affine", "codes": f"weights.safetensors:{ck}",
                "bits": args.bits, "shape": list(weight.shape),
                "group_size": 128, "scales": f"weights.safetensors:{sk}"}
        else:
            key = f"dense_{index}"
            tensors[key] = weight.cpu()
            records[logical] = {"codec": "dense",
                                "tensor": f"weights.safetensors:{key}"}
        index += 1

    for name, module in quantized.items():
        codes, zeros, scales = (unpack_gptq(module) if args.method == "gptq"
                                else unpack_awq_module(module))
        if name in repair_factors:
            expanded_zeros = zeros.repeat_interleave(
                module.group_size, dim=1)[:, :module.in_features]
            expanded_scales = scales.repeat_interleave(
                module.group_size, dim=1)[:, :module.in_features]
            logical = ((codes.float() - expanded_zeros.float()) *
                       expanded_scales.float())
            logical.div_(repair_factors[name].view(1, -1))
            codes, scales = rtn(logical, args.bits, module.group_size)
            zeros = torch.full_like(scales, 1 << (args.bits - 1))
        ck, sk, zk = f"codes_{index}", f"scales_{index}", f"zeros_{index}"
        tensors[ck], tensors[sk] = pack(codes, args.bits), scales.half()
        record = {
            "codec": "affine", "codes": f"weights.safetensors:{ck}",
            "bits": args.bits,
            "shape": [module.out_features, module.in_features],
            "group_size": module.group_size,
            "scales": f"weights.safetensors:{sk}",
        }
        midpoint = 1 << (args.bits - 1)
        if not torch.equal(zeros, torch.full_like(zeros, midpoint)):
            tensors[zk] = zeros.to(torch.int16)
            record["zeros"] = f"weights.safetensors:{zk}"
        g_idx = getattr(module, "g_idx", None)
        expected = torch.arange(module.in_features) // module.group_size
        if g_idx is not None and not torch.equal(g_idx.cpu().long(), expected):
            gk = f"gidx_{index}"
            tensors[gk] = g_idx.cpu().to(torch.int32)
            record["g_idx"] = f"weights.safetensors:{gk}"
        records[name + ".weight"] = record
        index += 1

    # Tied output projection is not materialized in GPTQModel's state dict.
    if "lm_head.weight" not in records and "model.embed_tokens.weight" in records:
        records["lm_head.weight"] = {
            "codec": "alias", "source": "model.embed_tokens.weight"}
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output / "weights.safetensors")
    manifest = {
        "format": "qweight-1", "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION, "target_bpw": 99.0,
        "producer": (f"GPTQModel-7.0-{args.method}-w{args.bits}-g128+RTN-nonlinears" +
                     ("+qwen35-shared-input-repair" if repair_factors else "")),
        "tensors": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n")
    parameters = 752_393_024
    size = sum(path.stat().st_size for path in args.output.iterdir())
    print(json.dumps({"bytes": size, "bpw": 8 * size / parameters,
                      "quantized_modules": len(quantized)}, indent=2))


if __name__ == "__main__":
    main()
