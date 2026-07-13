"""Score calibrated GPTQ/AWQ QWeight bundles with the canonical MPS metric."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from bench.qweight import bundle_bytes, decode_bundle
from bench.ml_models import (attest_fresh_mps_torch_import,
                             require_fresh_torch_import)
from bench.slm_mps_lock import exclusive_mps_lock
from bench.slm_sft import (clear_accelerator_cache, load_model,
                           per_conversation_nll, read_data,
                           select_online_validation, summarize,
                           validate_data_manifest)
from bench.tasks.slm_weight_compression_qwen35.evaluate import SPEC


def validation_rows():
    data = ROOT / "bench/tasks/slm_compression_qwen35/data"
    manifest = validate_data_manifest(data, "slm_compression_qwen35", (SPEC,))
    calibration, visible, sealed, _ = read_data(
        data, SPEC.key, (SPEC.key,), manifest,
        include_validation=True, include_test=False)
    return select_online_validation("mixed", visible, sealed,
                                    calibration[SPEC.key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--results", type=Path, default=ROOT /
                        "research/benchmark_v2/qwen35_gptq_awq_results.json")
    args = parser.parse_args()
    require_fresh_torch_import("Qwen3.5 GPTQ/AWQ baseline scoring")
    import torch
    import torch.nn.functional as F
    attest_fresh_mps_torch_import(torch, "Qwen3.5 GPTQ/AWQ baseline scoring")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    device = torch.device("mps")
    rows = validation_rows()
    records = []
    with exclusive_mps_lock(purpose="paper-native:qwen35-gptq-awq") as lock:
        reference_model = load_model(None, SPEC).to(
            device=device, dtype=torch.float32).eval()
        reference = per_conversation_nll(
            torch, F, reference_model, rows, device, 2)
        shapes = {name: tuple(value.shape)
                  for name, value in reference_model.state_dict().items()}
        parameters = sum(parameter.numel()
                         for parameter in reference_model.parameters())
        del reference_model
        clear_accelerator_cache(torch, device)
        for bundle in args.bundle:
            started = time.monotonic()
            manifest, decoded = decode_bundle(
                bundle, shapes, SPEC.hub_name, SPEC.revision, device)
            model = load_model(None, SPEC).to(
                device=device, dtype=torch.float32).eval()
            with torch.no_grad():
                for name, destination in model.state_dict().items():
                    destination.copy_(decoded[name].to(destination.dtype))
            del decoded
            values = per_conversation_nll(torch, F, model, rows, device, 2)
            prepared = [{
                "id": row["id"], "prompt_id": row["prompt_id"],
                "domain": row["domain"], "domain_group": row["domain_group"],
                "template_cluster": row["template_cluster"],
                "base": base, "compressed": value, "delta": value - base,
            } for row, base, value in zip(rows, reference, values)]
            bpw = 8 * bundle_bytes(bundle) / parameters
            summary = summarize({SPEC.key: {f"{bpw:.9f}": prepared}})
            records.append({
                "name": bundle.name, "producer": manifest["producer"],
                "bytes": bundle_bytes(bundle), "bpw": bpw,
                "score": summary["score"],
                "signed_nll_delta": summary["signed_nll_delta"],
                "perplexity_ratio": summary["perplexity_ratio"],
                "ci95": summary["paired_bootstrap_ci95"],
                "seconds": time.monotonic() - started,
            })
            del model
            clear_accelerator_cache(torch, device)
    payload = {
        "format": 1,
        "metric": "64-conversation ID validation assistant-token delta-NLL",
        "direction": "lower is better", "device": "mps",
        "scoring_dtype": "float32", "mps_fallback": False,
        "parameters": parameters, "mps_lock": lock, "results": records,
    }
    args.results.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
