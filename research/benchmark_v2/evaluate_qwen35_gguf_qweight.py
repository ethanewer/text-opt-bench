"""Losslessly wrap every Qwen3.5 GGUF as QWeight and compare its score."""

import argparse
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from bench.gguf_qwen35 import EXPECTED_PARAMETERS
from bench.ml_models import (attest_fresh_mps_torch_import,
                             require_fresh_torch_import)
from bench.qweight import bundle_bytes, decode_bundle
from bench.slm_mps_lock import exclusive_mps_lock
from bench.slm_sft import (clear_accelerator_cache, load_model,
                           per_conversation_nll)
from bench.tasks.slm_weight_compression_qwen35.evaluate import SPEC
from research.benchmark_v2.evaluate_qwen35_gguf import (
    atomic_json, prepared_validation, score_rows, sha256, write_svg)


def write_manifest(directory, source, digest):
    manifest = {
        "format": "qweight-1", "base_model": SPEC.hub_name,
        "base_revision": SPEC.revision, "target_bpw": 0.0,
        "producer": "lossless-native-gguf-wrapper-v1", "tensors": {},
        "native_gguf": {
            "file": "model.gguf", "sha256": digest,
            "architecture": "qwen35",
            "importer": "transformers-5.2-gguf-0.19-qwen35-v3",
        },
    }
    path = directory / "manifest.json"
    # Include the manifest itself in its declared physical BPW. Iteration is
    # needed only because changing the decimal can change JSON length.
    for _ in range(8):
        path.write_text(json.dumps(manifest, separators=(",", ":")))
        value = bundle_bytes(directory) * 8 / EXPECTED_PARAMETERS
        if manifest["target_bpw"] == value:
            break
        manifest["target_bpw"] = value
    path.write_text(json.dumps(manifest, separators=(",", ":")))
    return manifest


def convert(source, root, digest):
    directory = root / source.name.removesuffix(".gguf")
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / "model.gguf"
    if payload.exists():
        if sha256(payload) != digest:
            raise RuntimeError(f"stale QWeight GGUF payload: {payload}")
    else:
        # A hard link makes the local 13-GB round-trip practical. Copying the
        # resulting directory produces an ordinary self-contained bundle.
        os.link(source, payload)
    write_manifest(directory, source, digest)
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-directory", type=Path,
                        default=Path("/tmp/qwen35-gguf"))
    parser.add_argument("--qweight-directory", type=Path,
                        default=Path("/tmp/qwen35-qweight-gguf"))
    parser.add_argument("--direct-results", type=Path, default=(
        ROOT / "research/benchmark_v2/qwen35_gguf_results.json"))
    parser.add_argument("--results", type=Path, default=(
        ROOT / "research/benchmark_v2/qwen35_gguf_qweight_results.json"))
    parser.add_argument("--plot", type=Path, default=(
        ROOT / "docs/qwen35_gguf_qweight_bpw_vs_score.svg"))
    args = parser.parse_args()

    direct_payload = json.loads(args.direct_results.read_text())
    direct = {row["name"]: row for row in direct_payload["results"]}
    if len(direct) != 22 or any(row.get("status") != "ok"
                                for row in direct.values()):
        raise RuntimeError("direct GGUF sweep is incomplete")
    bundles = []
    for name, row in sorted(direct.items()):
        source = args.gguf_directory / name
        if (not source.is_file() or source.stat().st_size != row["bytes"] or
                sha256(source) != row["sha256"]):
            raise RuntimeError(f"GGUF source authentication failed: {source}")
        bundles.append((name, convert(
            source, args.qweight_directory, row["sha256"])))

    require_fresh_torch_import("Qwen3.5 GGUF QWeight round trip")
    import torch
    import torch.nn.functional as F
    attest_fresh_mps_torch_import(torch, "Qwen3.5 GGUF QWeight round trip")
    device = torch.device("mps")
    rows = prepared_validation()
    existing = {}
    if args.results.exists():
        existing = {row["name"]: row for row in
                    json.loads(args.results.read_text()).get("results", [])}

    with exclusive_mps_lock(
            purpose="paper-native:qwen35-gguf-qweight-roundtrip") as lock:
        reference_model = load_model(None, SPEC).to(
            device=device, dtype=torch.float32).eval()
        expected_shapes = {name: tuple(value.shape)
                           for name, value in reference_model.state_dict().items()}
        reference = per_conversation_nll(
            torch, F, reference_model, rows, device, 2)
        del reference_model
        clear_accelerator_cache(torch, device)
        for name, directory in bundles:
            direct_row = direct[name]
            prior = existing.get(name)
            bundle_sha = sha256(directory / "manifest.json")
            if (prior and prior.get("manifest_sha256") == bundle_sha and
                    prior.get("status") == "ok"):
                continue
            started = time.monotonic()
            try:
                model = load_model(None, SPEC).to(
                    device=device, dtype=torch.float32).eval()
                manifest, decoded = decode_bundle(
                    directory, expected_shapes, SPEC.hub_name, SPEC.revision,
                    device)
                state = model.state_dict()
                with torch.no_grad():
                    for tensor_name, destination in state.items():
                        destination.copy_(decoded[tensor_name])
                del state, decoded
                bpw = bundle_bytes(directory) * 8 / EXPECTED_PARAMETERS
                summary = score_rows(
                    torch, F, model, rows, reference, device, bpw)
                score = float(summary["score"])
                record = {
                    "status": "ok", "name": name,
                    "quant": direct_row["quant"],
                    "source_gguf_sha256": direct_row["sha256"],
                    "manifest_sha256": bundle_sha,
                    "source_gguf_bpw": direct_row["bpw"],
                    "bpw": bpw, "bundle_bytes": bundle_bytes(directory),
                    "manifest_overhead_bytes": (
                        bundle_bytes(directory) - direct_row["bytes"]),
                    "direct_score": direct_row["score"],
                    "score": score,
                    "absolute_score_difference": abs(
                        score - float(direct_row["score"])),
                    "ci95": summary["paired_bootstrap_ci95"],
                    "seconds": time.monotonic() - started,
                }
                del model
                clear_accelerator_cache(torch, device)
            except Exception as exc:
                record = {"status": "error", "name": name,
                          "manifest_sha256": bundle_sha,
                          "error": repr(exc),
                          "seconds": time.monotonic() - started}
            existing[name] = record
            payload = {
                "format": 1, "comparison": "native GGUF vs QWeight-wrapped GGUF",
                "metric": direct_payload["metric"], "device": "mps",
                "scoring_dtype": "float32", "mps_fallback": False,
                "parameters": EXPECTED_PARAMETERS, "mps_lock": lock,
                "results": sorted(existing.values(), key=lambda item: item["name"]),
            }
            atomic_json(args.results, payload)
            good = [item for item in payload["results"]
                    if item.get("status") == "ok"]
            if good:
                write_svg(
                    args.plot, good,
                    title="QWeight-wrapped Qwen3.5 GGUF: physical BPW vs validation ΔNLL",
                    x_label="physical QWeight bundle bits per base-model parameter (lower is smaller)")
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
