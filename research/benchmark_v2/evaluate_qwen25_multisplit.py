"""Score Qwen2.5-0.5B checkpoints on validation, ID, and OOD splits."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
MODEL = Path("/private/tmp/qwen2.5-0.5b-instruct")
DATA = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated/qwen25_generated_selected.json")
GGUF = Path("/private/tmp/qwen25-05b-gguf")
PARAMETERS = 494_032_768
SPLITS = ("validation", "id_test", "ood_test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gguf", action="store_true")
    parser.add_argument("--gptqmodel", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from bench.ml_models import (attest_fresh_mps_torch_import,
                                 require_fresh_torch_import)
    label = "Qwen2.5 multisplit checkpoint grading"
    require_fresh_torch_import(label)
    import torch
    attest_fresh_mps_torch_import(torch, label)
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from bench.slm_mps_lock import exclusive_mps_lock, operator_mps_phase
    from bench.slm_sft import per_conversation_nll

    payload = json.loads(DATA.read_text())
    rows = {split: [r for r in payload["records"] if r["split"] == split]
            for split in SPLITS}
    if any(len(value) != 128 for value in rows.values()):
        raise RuntimeError({key: len(value) for key, value in rows.items()})
    jobs = [("BF16-native", None, "native")]
    if args.gguf:
        for path in sorted(GGUF.glob("*.gguf")):
            name = path.stem.removeprefix("qwen2.5-0.5b-instruct-").upper()
            jobs.append((name, path, "gguf"))
    for path in args.gptqmodel:
        jobs.append((path.name, path, "gptqmodel"))
    device = torch.device("mps")
    results = []
    with operator_mps_phase("qwen25-multisplit-grading"):
        with exclusive_mps_lock(purpose="paper-native:qwen25-multisplit") as lock:
            for name, path, kind in jobs:
                started = time.monotonic()
                if kind == "gguf":
                    model = AutoModelForCausalLM.from_pretrained(
                        str(MODEL), gguf_file=str(path), local_files_only=True,
                        dtype=torch.float32)
                elif kind == "gptqmodel":
                    from gptqmodel import BACKEND, GPTQModel
                    model = GPTQModel.load(str(path), backend=BACKEND.TORCH,
                                           device="mps:0").model
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(MODEL), local_files_only=True, dtype=torch.float32)
                loaded = time.monotonic(); model.to(device).eval()
                values, seconds = {}, {}
                for split in SPLITS:
                    split_started = time.monotonic()
                    values[split] = per_conversation_nll(
                        torch, F, model, rows[split], device, args.batch_size)
                    seconds[split] = time.monotonic() - split_started
                finished = time.monotonic()
                size = ((MODEL / "model.safetensors").stat().st_size if path is None
                        else path.stat().st_size if path.is_file()
                        else (path / "model.safetensors").stat().st_size)
                means = {split: statistics.fmean(values[split]) for split in SPLITS}
                means["test_all"] = statistics.fmean(
                    values["id_test"] + values["ood_test"])
                results.append({"name": name, "kind": kind, "bytes": size,
                                "bpw": size*8/PARAMETERS, "mean_nll": means,
                                "seconds": {"load": loaded-started, **seconds,
                                            "total": finished-started}})
                del model; gc.collect(); torch.mps.empty_cache()
    base = results[0]["mean_nll"]
    for result in results:
        result["delta_nll"] = {key: result["mean_nll"][key]-base[key]
                               for key in (*SPLITS, "test_all")}
    output = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "parameters": PARAMETERS,
              "device": "mps", "mps_fallback": False,
              "metric": "mean per-conversation assistant-token NLL",
              "split_counts": {key: len(value) for key, value in rows.items()},
              "results": results, "lock": lock}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
