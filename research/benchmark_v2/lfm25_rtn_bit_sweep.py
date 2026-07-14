"""Build comparable 2/3/4/5-bit groupwise-RTN LFM QWeight bundles."""

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from lfm25_4p5_starter import (
    BASE_MODEL, BASE_REVISION, GROUP_SIZE, quantize)


PARAMETERS = 229_693_184


def bundle_bytes(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob("*")
               if path.is_file())


def build_one(states, bits, output):
    tensors, records, pointers = {}, {}, {}
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
            records[name] = {
                "codec": "dense", "tensor": f"weights.safetensors:{key}"}
            continue
        code_key, scale_key = f"codes_{index}", f"scales_{index}"
        codes, scales = quantize(weight, bits=bits, group_size=GROUP_SIZE)
        tensors[code_key], tensors[scale_key] = codes, scales
        records[name] = {
            "codec": "affine", "codes": f"weights.safetensors:{code_key}",
            "bits": bits, "shape": list(weight.shape),
            "group_size": GROUP_SIZE,
            "scales": f"weights.safetensors:{scale_key}",
        }
    directory = Path(output) / f"w{bits}_g{GROUP_SIZE}"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, directory / "weights.safetensors")
    del tensors
    manifest = {
        "format": "qweight-1", "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION, "target_bpw": 0.0,
        "producer": f"symmetric-groupwise-rtn-w{bits}-g{GROUP_SIZE}",
        "tensors": records,
    }
    manifest_path = directory / "manifest.json"
    for _ in range(8):
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
        bpw = bundle_bytes(directory) * 8 / PARAMETERS
        if manifest["target_bpw"] == bpw:
            break
        manifest["target_bpw"] = bpw
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    size = bundle_bytes(directory)
    return {"bits": bits, "group_size": GROUP_SIZE,
            "bundle": str(directory), "bytes": size,
            "bpw": size * 8 / PARAMETERS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16).to(
            args.device).eval()
    states = model.state_dict()
    results = []
    for bits in (5, 4, 3, 2):
        results.append(build_one(states, bits, args.output))
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
    print(json.dumps({"method": "symmetric groupwise RTN",
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
