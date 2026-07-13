"""Build a physically compact HQQ W3 baseline for the LFM benchmark.

HQQ is applied to every matrix whose input dimension admits group-128
quantization.  Small vectors and exceptional tensors use the benchmark's
group-128 symmetric RTN tail codec so that no uncounted BF16 weights remain.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


MODEL = Path("/private/tmp/lfm25-230m-source")
BASE_MODEL = "LiquidAI/LFM2.5-230M"
BASE_REVISION = "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7"
PARAMETERS = 229_693_184


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
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


def rtn(weight: torch.Tensor, bits: int, group_size: int):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=3, choices=(2, 3, 4, 8))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--axis", type=int, default=1, choices=(0, 1))
    args = parser.parse_args()

    from hqq.core.quantize import Quantizer

    device = torch.device("mps")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), local_files_only=True, dtype=torch.float32).eval()
    tensors, records, pointers = {}, {}, {}
    hqq_count = tail_count = 0
    started = __import__("time").monotonic()
    for index, (name, weight) in enumerate(model.state_dict().items()):
        pointer = (weight.untyped_storage().data_ptr(), weight.storage_offset(),
                   tuple(weight.shape), tuple(weight.stride()))
        if pointer in pointers:
            records[name] = {"codec": "alias", "source": pointers[pointer]}
            continue
        pointers[pointer] = name
        if not weight.is_floating_point() or not weight.ndim:
            key = f"dense_{index}"
            tensors[key] = weight.cpu()
            records[name] = {"codec": "dense",
                             "tensor": f"weights.safetensors:{key}"}
            continue

        hqq_eligible = (weight.ndim >= 2 and
                        (weight.numel() if args.axis == 0 else weight.shape[-1])
                        % args.group_size == 0)
        if hqq_eligible:
            codes, meta = Quantizer.quantize(
                weight.to(device), nbits=args.bits, group_size=args.group_size,
                optimize=True, round_zero=False, axis=args.axis, bitpack=False,
                device="mps")
            if args.axis == 0:
                codes = codes.reshape(args.group_size, -1).to(torch.int16).cpu()
                scales = meta["scale"].reshape(1, -1).half().cpu()
                zeros = meta["zero"].reshape(1, -1).half().cpu()
            else:
                codes = codes.reshape(weight.numel() // weight.shape[-1],
                                      weight.shape[-1]).to(torch.int16).cpu()
                scales = meta["scale"].reshape(
                    weight.numel() // weight.shape[-1],
                    weight.shape[-1] // args.group_size).half().cpu()
                zeros = meta["zero"].reshape_as(scales).half().cpu()
            hqq_count += 1
        else:
            codes, scales = rtn(weight, args.bits, args.group_size)
            zeros = None
            tail_count += 1

        ck, sk = f"codes_{index}", f"scales_{index}"
        tensors[ck], tensors[sk] = pack(codes, args.bits), scales
        if hqq_eligible and args.axis == 0:
            zk = f"zeros_{index}"
            tensors[zk] = zeros
            record = {"codec": "graph", "shape": list(weight.shape),
                      "nodes": [
                          {"id": "packed", "op": "payload",
                           "tensor": f"weights.safetensors:{ck}"},
                          {"id": "codes", "op": "unpack", "input": "packed",
                           "bits": args.bits, "count": weight.numel()},
                          {"id": "groups", "op": "reshape", "input": "codes",
                           "shape": [args.group_size, weight.numel() // args.group_size]},
                          {"id": "zeros", "op": "payload",
                           "tensor": f"weights.safetensors:{zk}"},
                          {"id": "centered", "op": "sub", "left": "groups",
                           "right": "zeros"},
                          {"id": "scales", "op": "payload",
                           "tensor": f"weights.safetensors:{sk}"},
                          {"id": "scaled", "op": "mul", "left": "centered",
                           "right": "scales"},
                          {"id": "out", "op": "reshape", "input": "scaled",
                           "shape": list(weight.shape)}], "output": "out"}
        else:
            record = {"codec": "affine", "codes": f"weights.safetensors:{ck}",
                      "bits": args.bits, "shape": list(weight.shape),
                      "group_size": args.group_size,
                      "scales": f"weights.safetensors:{sk}"}
            if zeros is not None:
                zk = f"zeros_{index}"
                tensors[zk] = zeros
                record["zeros"] = f"weights.safetensors:{zk}"
        records[name] = record

    if "lm_head.weight" not in records and "model.embed_tokens.weight" in records:
        records["lm_head.weight"] = {
            "codec": "alias", "source": "model.embed_tokens.weight"}
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output / "weights.safetensors")
    manifest = {"format": "qweight-1", "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION, "target_bpw": 3.5,
                "producer": (f"HQQ-w{args.bits}-g{args.group_size}-axis{args.axis}"
                             f"+RTN-w{args.bits}-g{args.group_size}-tail"),
                "tensors": records}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n")
    size = sum(path.stat().st_size for path in args.output.iterdir())
    print(json.dumps({"bytes": size, "bpw": 8 * size / PARAMETERS,
                      "hqq_tensors": hqq_count, "tail_tensors": tail_count,
                      "seconds": __import__("time").monotonic() - started}, indent=2))


if __name__ == "__main__":
    main()
