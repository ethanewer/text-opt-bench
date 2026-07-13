"""Deterministic 3-bit groupwise-RTN starter for the 3.5-BPW task."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from research.benchmark_v2.lfm25_4p5_starter import (
    BASE_MODEL, BASE_REVISION, GROUP_SIZE, quantize)


def build(model_path, output, targets, device):
    if tuple(targets) != (3.5,):
        raise ValueError("this starter supports the 3.500-BPW operating point")
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
        codes, scales = quantize(weight, bits=3, group_size=GROUP_SIZE)
        tensors[code_key], tensors[scale_key] = codes, scales
        records[name] = {
            "codec": "affine", "codes": f"weights.safetensors:{code_key}",
            "bits": 3, "shape": list(weight.shape),
            "group_size": GROUP_SIZE,
            "scales": f"weights.safetensors:{scale_key}",
        }
    directory = Path(output) / "3.500"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, directory / "weights.safetensors")
    manifest = {
        "format": "qweight-1", "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION, "target_bpw": 3.5,
        "producer": "symmetric-groupwise-rtn-w3-g40",
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
