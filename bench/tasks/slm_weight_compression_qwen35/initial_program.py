"""Deterministic symmetric groupwise-RTN QWeight baseline."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM


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
            np.bitwise_or.at(output, byte[crossing] + 1,
                             (local[crossing] >> (8 - shift[crossing])).astype(np.uint8))
    return torch.from_numpy(output)


def quantize(weight, bits, group_size):
    shape = list(weight.shape)
    columns = shape[-1]
    outer = weight.numel() // columns
    groups = math.ceil(columns / group_size)
    padded = groups * group_size
    matrix = weight.float().reshape(outer, columns)
    if padded != columns:
        matrix = torch.nn.functional.pad(matrix, (0, padded - columns))
    matrix = matrix.reshape(outer, groups, group_size)
    levels = (1 << (bits - 1)) - 1
    scale = matrix.abs().amax(-1).clamp_min(1e-8) / levels
    # Metadata is physically FP16, so fake-dequantization is exactly what the
    # trusted decoder reconstructs rather than a higher-precision surrogate.
    scale = scale.to(torch.float16)
    codes = (matrix / scale.float().unsqueeze(-1)).round().clamp(
        -levels, levels).to(torch.int16) + (1 << (bits - 1))
    codes = codes.reshape(outer, padded)[:, :columns].reshape(-1)
    return pack(codes, bits), scale.cpu()


def build(model_path, output, targets, device):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config.text_config, local_files_only=True,
        key_mapping={r"^model\.language_model\.": "model."}).to(
            device=device, dtype=torch.bfloat16).eval()
    states = model.state_dict()
    pointers = {}
    for target in targets:
        bits = 3 if target < 3.5 else 4
        # 256-wide groups leave room under the cap for all physical headers.
        group_size = 256
        tensors, records = {}, {}
        pointers.clear()
        for index, (name, weight) in enumerate(states.items()):
            pointer = (weight.untyped_storage().data_ptr(), weight.storage_offset(),
                       tuple(weight.shape), tuple(weight.stride()))
            if pointer in pointers:
                records[name] = {"codec": "alias", "source": pointers[pointer]}
                continue
            pointers[pointer] = name
            if not weight.is_floating_point() or weight.ndim == 0:
                key = f"dense_{index}"
                tensors[key] = weight.detach().cpu()
                records[name] = {"codec": "dense", "tensor": f"weights.safetensors:{key}"}
                continue
            code_key, scale_key = f"codes_{index}", f"scales_{index}"
            codes, scales = quantize(weight, bits, group_size)
            tensors[code_key], tensors[scale_key] = codes, scales
            records[name] = {
                "codec": "affine", "codes": f"weights.safetensors:{code_key}",
                "bits": bits, "shape": list(weight.shape),
                "group_size": group_size,
                "scales": f"weights.safetensors:{scale_key}",
            }
        directory = Path(output) / f"{target:.3f}"
        directory.mkdir(parents=True, exist_ok=True)
        save_file(tensors, directory / "weights.safetensors")
        manifest = {
            "format": "qweight-1", "base_model": "Qwen/Qwen3.5-0.8B",
            "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "target_bpw": target, "producer": "symmetric-groupwise-rtn-256",
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
