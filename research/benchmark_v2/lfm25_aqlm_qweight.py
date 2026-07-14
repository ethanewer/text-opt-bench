"""Calibrate an exact AQLM 3x8 checkpoint for LFM2.5 and export QWeight."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from export_lfm25_hqq_qweight import (BASE_MODEL, BASE_REVISION, MODEL,
                                      PARAMETERS, pack, rtn)

DATA = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated/lfm25_hard_eval_selected.json")
AQLM = Path("/private/tmp/aqlm")


def calibration_chunks(path: Path, length: int) -> list[torch.Tensor]:
    rows = [row for row in json.loads(path.read_text())["records"]
            if row.get("split", "calibration") == "calibration"]
    tokens = [token for row in rows for token in row["input_ids"]]
    return [torch.tensor(tokens[i:i + length], dtype=torch.long)
            for i in range(0, len(tokens) - length + 1, length)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--kmeans-iterations", type=int, default=10)
    parser.add_argument("--points-per-centroid", type=int, default=32)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--aqlm", type=Path, default=AQLM)
    parser.add_argument("--device", choices=("mps", "cuda"), default="mps")
    args = parser.parse_args()
    sys.path.insert(0, str(args.aqlm))
    from aq_engine import AQEngine

    device = torch.device(args.device)
    torch.manual_seed(20260712)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.float32).to(device).eval()
    chunks = calibration_chunks(args.data, args.sequence_length)
    aq_weights = {}
    started = time.monotonic()
    aq_args = argparse.Namespace(
        devices=[device], out_group_size=1, in_group_size=8,
        num_codebooks=3, nbits_per_codebook=8,
        codebook_value_nbits=16, codebook_value_num_groups=1,
        scale_nbits=0, init_max_iter=args.kmeans_iterations,
        init_max_points_per_centroid=args.points_per_centroid,
        lr=1e-4, max_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch, relative_mse_tolerance=.01,
        beam_size=1, print_frequency=1000)

    for layer_index, layer in enumerate(model.model.layers):
        linears = {name: module for name, module in layer.named_modules()
                   if isinstance(module, torch.nn.Linear) and
                   module.weight.shape[1] % 8 == 0}
        engines = {name: AQEngine(module, accumulator_dtype=torch.float32)
                   for name, module in linears.items()}
        handles = []
        for name, module in linears.items():
            handles.append(module.register_forward_pre_hook(
                lambda _module, inputs, name=name: engines[name].add_batch(inputs[0].detach())))
        with torch.no_grad():
            for start in range(0, len(chunks), args.batch_size):
                ids = torch.stack(chunks[start:start + args.batch_size]).to(device)
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        for handle in handles:
            handle.remove()
        for name, engine in engines.items():
            print(f"layer {layer_index}: AQLM {name}", flush=True)
            quantized = engine.quantize(args=aq_args, verbose=False)
            aq_weights[f"model.layers.{layer_index}.{name}.weight"] = {
                "codes": quantized.get_codes().detach().cpu().to(torch.uint8),
                "codebooks": quantized.get_codebooks().detach().cpu().half(),
                "scales": quantized.get_scales().detach().cpu().half(),
            }
            with torch.no_grad():
                linears[name].weight.copy_(quantized().to(device))
            del engine, quantized
            if args.device == "mps":
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

    tensors, records, pointers = {}, {}, {}
    for index, (name, weight) in enumerate(model.state_dict().items()):
        pointer = (weight.untyped_storage().data_ptr(), weight.storage_offset(),
                   tuple(weight.shape), tuple(weight.stride()))
        if pointer in pointers:
            records[name] = {"codec": "alias", "source": pointers[pointer]}
            continue
        pointers[pointer] = name
        if name in aq_weights:
            item = aq_weights[name]
            code_key, scale_key = f"aq_codes_{index}", f"aq_scales_{index}"
            tensors[code_key], tensors[scale_key] = item["codes"], item["scales"]
            nodes = [{"id": "codes", "op": "payload",
                      "tensor": f"weights.safetensors:{code_key}"}]
            vector_ids = []
            for codebook_index in range(3):
                table_key = f"aq_table_{index}_{codebook_index}"
                tensors[table_key] = item["codebooks"][codebook_index].reshape(256, 8)
                nodes.extend([
                    {"id": f"idx{codebook_index}", "op": "slice", "input": "codes",
                     "dim": 2, "start": codebook_index, "stop": codebook_index + 1},
                    {"id": f"idxflat{codebook_index}", "op": "reshape",
                     "input": f"idx{codebook_index}",
                     "shape": [weight.shape[0], weight.shape[1] // 8]},
                    {"id": f"table{codebook_index}", "op": "payload",
                     "tensor": f"weights.safetensors:{table_key}"},
                    {"id": f"vec{codebook_index}", "op": "vector_lookup",
                     "table": f"table{codebook_index}",
                     "indices": f"idxflat{codebook_index}"},
                ])
                vector_ids.append(f"vec{codebook_index}")
            nodes.extend([
                {"id": "sum01", "op": "add", "left": vector_ids[0], "right": vector_ids[1]},
                {"id": "sum", "op": "add", "left": "sum01", "right": vector_ids[2]},
                {"id": "scales_raw", "op": "payload",
                 "tensor": f"weights.safetensors:{scale_key}"},
                {"id": "scales", "op": "reshape", "input": "scales_raw",
                 "shape": [weight.shape[0], 1, 1]},
                {"id": "scaled", "op": "mul", "left": "sum", "right": "scales"},
                {"id": "out", "op": "reshape", "input": "scaled",
                 "shape": list(weight.shape)},
            ])
            records[name] = {"codec": "graph", "shape": list(weight.shape),
                             "nodes": nodes, "output": "out"}
        elif weight.is_floating_point() and weight.ndim:
            codes, scales = rtn(weight.cpu(), 3, 128)
            code_key, scale_key = f"tail_codes_{index}", f"tail_scales_{index}"
            tensors[code_key], tensors[scale_key] = pack(codes, 3), scales
            records[name] = {"codec": "affine",
                             "codes": f"weights.safetensors:{code_key}",
                             "bits": 3, "shape": list(weight.shape),
                             "group_size": 128,
                             "scales": f"weights.safetensors:{scale_key}"}
        else:
            key = f"dense_{index}"
            tensors[key] = weight.cpu()
            records[name] = {"codec": "dense", "tensor": f"weights.safetensors:{key}"}
    if "lm_head.weight" not in records and "model.embed_tokens.weight" in records:
        records["lm_head.weight"] = {"codec": "alias", "source": "model.embed_tokens.weight"}
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output / "weights.safetensors")
    manifest = {"format": "qweight-1", "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION, "target_bpw": 3.5,
                "producer": "AQLM-3x8-g8-calibrated+RTN-w3-g128-tail",
                "tensors": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    size = sum(path.stat().st_size for path in args.output.iterdir())
    print(json.dumps({"bpw": size * 8 / PARAMETERS, "bytes": size,
                      "seconds": time.monotonic() - started,
                      "calibration_sequences": len(chunks),
                      "device": args.device}, indent=2))


if __name__ == "__main__":
    main()
