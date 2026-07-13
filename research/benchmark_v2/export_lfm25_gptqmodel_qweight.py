"""Export an LFM2.5 GPTQModel checkpoint as a compact whole-model QWeight bundle.

GPTQModel quantizes supported Linear modules.  To make physical BPW comparable
under the benchmark, every other floating state tensor is encoded with
groupwise RTN at the same bit width instead of retaining a hidden BF16 tail.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from lfm25_gptq_awq import patch_lfm2


BASE_MODEL = "LiquidAI/LFM2.5-230M"
BASE_REVISION = "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7"
PARAMETERS = 229_693_184


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
    return codes.reshape(outer, groups * group_size)[:, :columns], scales


def unpack_gptq(module):
    from gptqmodel.utils.model_dequant import unpack_cols, unpack_rows
    codes = unpack_rows(module.qweight.cpu(), module.bits)[
        :module.in_features, :module.out_features]
    zeros = unpack_cols(module.qzeros.cpu(), module.bits)[
        :, :module.out_features]
    return (codes.t().contiguous(), zeros.t().contiguous(),
            module.scales.cpu().t().contiguous())


def unpack_awq(module):
    from gptqmodel.quantization.awq.utils.packing_utils import (
        reverse_awq_order, unpack_awq as unpack)
    codes, zeros = unpack(module.qweight.cpu(), module.qzeros.cpu(), module.bits)
    codes, zeros = reverse_awq_order(codes, zeros, module.bits)
    mask = (1 << module.bits) - 1
    return ((codes & mask)[:module.in_features, :module.out_features].t().contiguous(),
            (zeros & mask)[:, :module.out_features].t().contiguous(),
            module.scales.cpu().t().contiguous())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("gptq", "awq"), required=True)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    patch_lfm2(args.method)
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    backend = (BACKEND.GPTQ_TORCH if args.method == "gptq"
               else BACKEND.AWQ_TORCH)
    wrapped = GPTQModel.load(str(args.checkpoint), backend=backend, device="cpu")
    model = wrapped.model.eval()
    quantized = {name: module for name, module in model.named_modules()
                 if isinstance(module, BaseQuantLinear)}
    state = model.state_dict()
    tensors, records, pointers = {}, {}, {}
    index = 0
    for name, weight in state.items():
        if any(name == prefix + suffix for prefix in quantized for suffix in
               (".qweight", ".qzeros", ".scales", ".g_idx", ".bias")):
            continue
        if name.endswith(".weight") and name[:-7] in quantized:
            continue
        pointer = (weight.untyped_storage().data_ptr(), weight.storage_offset(),
                   tuple(weight.shape), tuple(weight.stride()))
        if pointer in pointers:
            records[name] = {"codec": "alias", "source": pointers[pointer]}
            continue
        pointers[pointer] = name
        if weight.is_floating_point() and weight.ndim:
            codes, scales = rtn(weight, args.bits, 128)
            ck, sk = f"codes_{index}", f"scales_{index}"
            tensors[ck], tensors[sk] = pack(codes, args.bits), scales
            records[name] = {"codec": "affine",
                             "codes": f"weights.safetensors:{ck}",
                             "bits": args.bits, "shape": list(weight.shape),
                             "group_size": 128,
                             "scales": f"weights.safetensors:{sk}"}
        else:
            key = f"dense_{index}"
            tensors[key] = weight.cpu()
            records[name] = {"codec": "dense",
                             "tensor": f"weights.safetensors:{key}"}
        index += 1

    for name, module in quantized.items():
        codes, zeros, scales = (unpack_gptq(module) if args.method == "gptq"
                                else unpack_awq(module))
        ck, sk, zk = f"codes_{index}", f"scales_{index}", f"zeros_{index}"
        tensors[ck], tensors[sk] = pack(codes, args.bits), scales.half()
        record = {"codec": "affine", "codes": f"weights.safetensors:{ck}",
                  "bits": args.bits,
                  "shape": [module.out_features, module.in_features],
                  "group_size": module.group_size,
                  "scales": f"weights.safetensors:{sk}"}
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

    if "lm_head.weight" not in records and "model.embed_tokens.weight" in records:
        records["lm_head.weight"] = {
            "codec": "alias", "source": "model.embed_tokens.weight"}
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output / "weights.safetensors")
    manifest = {"format": "qweight-1", "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION, "target_bpw": 3.5,
                "producer": (f"GPTQModel-{args.method}-w{args.bits}-g128"
                             f"+RTN-w{args.bits}-g128-tail"),
                "tensors": records}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n")
    size = sum(path.stat().st_size for path in args.output.iterdir())
    print(json.dumps({"bytes": size, "bpw": 8 * size / PARAMETERS,
                      "quantized_modules": len(quantized)}, indent=2))


if __name__ == "__main__":
    main()
