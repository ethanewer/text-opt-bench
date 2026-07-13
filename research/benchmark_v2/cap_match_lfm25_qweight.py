"""Use the remaining 3.5-BPW budget of an existing W3 QWeight baseline.

The underlying GPTQ/HQQ/AQLM records are preserved. Selected affine W3
records are upgraded to W4 symmetric RTN, in a deterministic architecture-aware
order, until no further tensor fits. This is a fixed-cap mixed-precision
adaptation, not a new paper-native algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from safetensors.torch import load_file, save_file

from export_lfm25_hqq_qweight import pack, rtn


MODEL = Path("/private/tmp/lfm25-230m-source/model.safetensors")
PARAMETERS = 229_693_184
CAP_BYTES = math.floor(3.5 * PARAMETERS / 8)


def payload_key(reference: str) -> str:
    prefix = "weights.safetensors:"
    if not reference.startswith(prefix):
        raise ValueError(f"unsupported payload reference: {reference}")
    return reference[len(prefix):]


def priority(name: str):
    if name == "model.embed_tokens.weight":
        rank = 0
    elif ".conv.conv.weight" in name:
        rank = 1
    elif any(token in name for token in ("out_proj", "in_proj")):
        rank = 2
    elif ".feed_forward.w2." in name:
        rank = 3
    elif ".feed_forward." in name:
        rank = 4
    elif "norm" in name or name.endswith("bias"):
        rank = 5
    else:
        rank = 6
    return rank, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.source / "manifest.json").read_text())
    if not torch.backends.mps.is_available():
        raise RuntimeError("cap matching requires MPS")
    device = torch.device("mps")
    tensors = load_file(args.source / "weights.safetensors", device="cpu")
    native = load_file(MODEL, device="cpu")
    source_bytes = sum(path.stat().st_size for path in args.source.iterdir())
    remaining = CAP_BYTES - source_bytes
    if remaining < 0:
        raise RuntimeError("source bundle already exceeds 3.5 BPW")

    zero_references = {}
    for record in manifest["tensors"].values():
        reference = record.get("zeros")
        if reference is not None:
            zero_references[reference] = zero_references.get(reference, 0) + 1

    candidates = []
    for name, record in manifest["tensors"].items():
        if (record.get("codec") == "affine" and record.get("bits") == 3
                and name in native):
            old_key = payload_key(record["codes"])
            old_bytes = tensors[old_key].numel() * tensors[old_key].element_size()
            new_bytes = math.ceil(native[name].numel() * 4 / 8)
            zero_reference = record.get("zeros")
            removable_zero_key = None
            removable_zero_bytes = 0
            if (zero_reference is not None
                    and zero_references.get(zero_reference) == 1):
                removable_zero_key = payload_key(zero_reference)
                removable_zero_bytes = (
                    tensors[removable_zero_key].numel()
                    * tensors[removable_zero_key].element_size())
            candidates.append((
                priority(name), name,
                new_bytes - old_bytes - removable_zero_bytes,
                removable_zero_key,
            ))

    upgraded = []
    for _rank, name, extra, removable_zero_key in sorted(candidates):
        if extra > remaining:
            continue
        record = manifest["tensors"][name]
        group_size = int(record["group_size"])
        codes, scales = rtn(native[name].to(device), 4, group_size)
        code_key = payload_key(record["codes"])
        scale_key = payload_key(record["scales"])
        tensors[code_key] = pack(codes, 4)
        tensors[scale_key] = scales.cpu()
        record["bits"] = 4
        record.pop("zeros", None)
        if removable_zero_key is not None:
            tensors.pop(removable_zero_key)
        remaining -= extra
        upgraded.append(name)

    manifest["producer"] = (
        manifest.get("producer", "qweight-w3") +
        "+cap-matched-mixed-W4-RTN-tail")
    manifest["target_bpw"] = 3.5
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output / "weights.safetensors")
    (args.output / "manifest.json").write_text(json.dumps(
        manifest, separators=(",", ":")) + "\n")
    size = sum(path.stat().st_size for path in args.output.iterdir())
    if size > CAP_BYTES:
        raise RuntimeError(f"cap-matched bundle exceeds budget: {size}>{CAP_BYTES}")
    print(json.dumps({"device": "mps", "mps_fallback": False,
                      "source_bytes": source_bytes, "bytes": size,
                      "bpw": 8 * size / PARAMETERS,
                      "upgraded_tensors": upgraded}, indent=2))


if __name__ == "__main__":
    main()
