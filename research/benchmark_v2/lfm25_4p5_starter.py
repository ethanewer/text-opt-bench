"""Deterministic 4-bit groupwise-RTN QWeight starter for LFM2.5-230M."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


BASE_MODEL = "LiquidAI/LFM2.5-230M"
BASE_REVISION = "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7"
GROUP_SIZE = 40


def pack(codes, bits):
    values = codes.detach().to("cpu", torch.uint8).numpy().reshape(-1)
    output = np.zeros((math.ceil(values.size * bits / 8),), dtype=np.uint8)
    chunk = 1_000_000
    for start in range(0, values.size, chunk):
        local = values[start:start + chunk].astype(np.uint16)
        positions = np.arange(start, start + local.size, dtype=np.int64) * bits
        byte, shift = positions // 8, positions % 8
        np.bitwise_or.at(output, byte, ((local << shift) & 255).astype(np.uint8))
        crossing = shift + bits > 8
        if crossing.any():
            np.bitwise_or.at(
                output, byte[crossing] + 1,
                (local[crossing] >> (8 - shift[crossing])).astype(np.uint8))
    return torch.from_numpy(output)


def quantize(weight, bits=4, group_size=GROUP_SIZE):
    shape = list(weight.shape)
    columns = shape[-1] if shape else 1
    outer = weight.numel() // columns
    groups = math.ceil(columns / group_size)
    padded = groups * group_size
    matrix = weight.float().reshape(outer, columns)
    if padded != columns:
        matrix = torch.nn.functional.pad(matrix, (0, padded - columns))
    matrix = matrix.reshape(outer, groups, group_size)
    levels = (1 << (bits - 1)) - 1
    scales = (matrix.abs().amax(-1).clamp_min(1e-8) / levels).to(torch.float16)
    codes = (matrix / scales.float().unsqueeze(-1)).round().clamp(
        -levels, levels).to(torch.int16) + (1 << (bits - 1))
    codes = codes.reshape(outer, padded)[:, :columns].reshape(-1)
    return pack(codes, bits), scales.cpu()


def build(model_path, output, targets, device):
    if tuple(targets) != (4.5,):
        raise ValueError("this starter supports the 4.500-BPW operating point")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    tensors, records, pointers = {}, {}, {}
    for index, (name, weight) in enumerate(model.state_dict().items()):
        pointer = (weight.untyped_storage().data_ptr(), weight.storage_offset(),
                   tuple(weight.shape), tuple(weight.stride()))
        if pointer in pointers:
            records[name] = {"codec": "alias", "source": pointers[pointer]}
            continue
        pointers[pointer] = name
        if not weight.is_floating_point() or weight.ndim == 0:
            key = f"dense_{index}"
            tensors[key] = weight.detach().cpu()
            records[name] = {
                "codec": "dense", "tensor": f"weights.safetensors:{key}"}
            continue
        code_key, scale_key = f"codes_{index}", f"scales_{index}"
        codes, scales = quantize(weight)
        tensors[code_key], tensors[scale_key] = codes, scales
        records[name] = {
            "codec": "affine", "codes": f"weights.safetensors:{code_key}",
            "bits": 4, "shape": list(weight.shape),
            "group_size": GROUP_SIZE,
            "scales": f"weights.safetensors:{scale_key}",
        }
    directory = Path(output) / "4.500"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, directory / "weights.safetensors")
    manifest = {
        "format": "qweight-1", "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION, "target_bpw": 4.5,
        "producer": "symmetric-groupwise-rtn-w4-g40",
        "tensors": records,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--device", choices=("mps",), required=True)
    args = parser.parse_args()
    build(args.model, args.output,
          tuple(float(value) for value in args.targets.split(",")), args.device)
