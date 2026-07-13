"""Score every Unsloth Qwen3.5-0.8B GGUF on the benchmark validation set.

GGUF parsing/dequantization is an offline weight-import operation performed by
the pinned gguf package. All reference and imported-model inference is FP32 on
MPS with fallback disabled, using the benchmark's exact assistant-token metric.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from bench.slm_mps_lock import exclusive_mps_lock
from bench.gguf_qwen35 import (EXPECTED_PARAMETERS, IMPORTER_VERSION,
                               load_model as load_gguf)
from bench.ml_models import (attest_fresh_mps_torch_import,
                             require_fresh_torch_import)
from bench.slm_sft import (clear_accelerator_cache, load_model,
                           per_conversation_nll, read_data,
                           select_online_validation, summarize,
                           validate_data_manifest)
from bench.tasks.slm_weight_compression_qwen35.evaluate import SPEC

REPO_ID = "unsloth/Qwen3.5-0.8B-GGUF"
REPO_REVISION = "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepared_validation():
    data = ROOT / "bench/tasks/slm_compression_qwen35/data"
    manifest = validate_data_manifest(
        data, "slm_compression_qwen35", (SPEC,))
    calibration, visible, sealed, _test = read_data(
        data, SPEC.key, (SPEC.key,), manifest,
        include_validation=True, include_test=False)
    validation = select_online_validation(
        "mixed", visible, sealed, calibration[SPEC.key])
    return validation


def score_rows(torch, F, model, rows, reference, device, bpw):
    values = per_conversation_nll(torch, F, model, rows, device, 2)
    prepared = []
    for row, base, value in zip(rows, reference, values):
        prepared.append({
            "id": row["id"], "prompt_id": row["prompt_id"],
            "domain": row["domain"], "domain_group": row["domain_group"],
            "template_cluster": row["template_cluster"],
            "base": base, "compressed": value, "delta": value - base,
        })
    return summarize({SPEC.key: {f"{bpw:.9f}": prepared}})


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def family(name):
    if "BF16" in name: return "BF16"
    if "UD-" in name: return "Unsloth Dynamic"
    if "IQ" in name: return "IQ"
    return "K/Q"


def write_svg(path, records, title=None, x_label=None):
    good = sorted((row for row in records if row.get("status") == "ok"),
                  key=lambda row: row["bpw"])
    width, height = 1120, 720
    left, right, top, bottom = 90, 300, 55, 80
    xmin = min(row["bpw"] for row in good) - .25
    xmax = max(row["bpw"] for row in good) + .25
    epsilon = 1e-4
    ymin = math.log10(epsilon)
    ymax = math.log10(max(row["score"] for row in good) + epsilon) + .04
    x = lambda value: left + (width-left-right)*(value-xmin)/(xmax-xmin)
    y = lambda value: top + (height-top-bottom)*(ymax-math.log10(max(0, value)+epsilon))/(ymax-ymin)
    colors = {"BF16": "#183b56", "IQ": "#d97706", "K/Q": "#7555a4",
              "Unsloth Dynamic": "#087f75"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
        '<style>text{font-family:ui-sans-serif,system-ui;fill:#263638}.small{font-size:12px}.label{font-size:11px}</style>',
        f'<text x="90" y="28" font-size="20" font-weight="700">{title or "Unsloth Qwen3.5-0.8B GGUF: physical BPW vs validation ΔNLL"}</text>',
    ]
    for value in (0, .0001, .001, .01, .1, 1.0):
        if value > max(row["score"] for row in good) * 1.05:
            continue
        py = y(value)
        lines += [f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#dfe3df"/>',
                  f'<text class="small" x="{left-10}" y="{py+4:.2f}" text-anchor="end">{value:g}</text>']
    for value in range(math.floor(xmin), math.ceil(xmax)+1):
        px = x(value)
        lines += [f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{height-bottom}" stroke="#eef0ed"/>',
                  f'<text class="small" x="{px:.2f}" y="{height-bottom+23}" text-anchor="middle">{value}</text>']
    lines += [f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#596866"/>',
              f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#596866"/>',
              f'<text x="{(left+width-right)/2}" y="{height-25}" text-anchor="middle">{x_label or "physical GGUF bits per base-model parameter (lower is smaller)"}</text>',
              f'<text transform="translate(24 {(top+height-bottom)/2}) rotate(-90)" text-anchor="middle">assistant-token ΔNLL · log scale (lower is better)</text>']
    frontier, best = [], float("inf")
    for row in good:
        if row["score"] < best:
            frontier.append(row)
            best = row["score"]
    lines.append('<polyline fill="none" stroke="#be3455" stroke-width="2.5" opacity=".7" points="' +
                 " ".join(f'{x(row["bpw"]):.2f},{y(row["score"]):.2f}' for row in frontier) + '"/>')
    for index, row in enumerate(good):
        px, py = x(row["bpw"]), y(row["score"])
        color = colors[family(row["name"])]
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6" fill="{color}" stroke="white" stroke-width="1.5"/>')
        # A numbered key avoids unreadable collisions among near-identical BPW.
        lines.append(f'<text class="label" x="{px+7:.2f}" y="{py-7:.2f}">{index+1}</text>')
    key_x = width-right+25
    for index, row in enumerate(good):
        py = top + index * 25
        color = colors[family(row["name"])]
        lines += [f'<circle cx="{key_x}" cy="{py}" r="5" fill="{color}"/>',
                  f'<text class="label" x="{key_x+11}" y="{py+4}">{index+1}. {row["quant"]} · {row["bpw"]:.3f} · {row["score"]:.4f}</text>']
    lines.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("/tmp/qwen35-gguf"))
    parser.add_argument("--results", type=Path, default=(
        ROOT / "research/benchmark_v2/qwen35_gguf_results.json"))
    parser.add_argument("--plot", type=Path, default=(
        ROOT / "docs/qwen35_gguf_bpw_vs_score.svg"))
    parser.add_argument("--only", action="append", default=[],
                        help="score only this GGUF filename (repeatable)")
    args = parser.parse_args()

    require_fresh_torch_import("Qwen3.5 GGUF baseline sweep")
    import torch
    import torch.nn.functional as F
    attest_fresh_mps_torch_import(torch, "Qwen3.5 GGUF baseline sweep")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    files = sorted(args.directory.glob("Qwen3.5-0.8B-*.gguf"))
    if args.only:
        files = [path for path in files if path.name in set(args.only)]
    existing = {}
    if args.results.exists():
        existing = {row["name"]: row for row in
                    json.loads(args.results.read_text()).get("results", [])}
    rows = prepared_validation()
    device = torch.device("mps")
    with exclusive_mps_lock(purpose="paper-native:qwen35-gguf-sweep") as lock:
        reference_model = load_model(None, SPEC).to(
            device=device, dtype=torch.float32).eval()
        reference = per_conversation_nll(
            torch, F, reference_model, rows, device, 2)
        del reference_model
        clear_accelerator_cache(torch, device)
        for path in files:
            digest = sha256(path)
            prior = existing.get(path.name)
            if (prior and prior.get("sha256") == digest and
                    prior.get("importer_version") == IMPORTER_VERSION and
                    prior.get("status") == "ok"):
                continue
            started = time.monotonic()
            try:
                model = load_gguf(path).to(
                    device=device, dtype=torch.float32).eval()
                bpw = path.stat().st_size * 8 / EXPECTED_PARAMETERS
                summary = score_rows(torch, F, model, rows, reference, device, bpw)
                record = {
                    "status": "ok", "name": path.name,
                    "importer_version": IMPORTER_VERSION,
                    "quant": path.name.removeprefix("Qwen3.5-0.8B-").removesuffix(".gguf"),
                    "bytes": path.stat().st_size, "bpw": bpw,
                    "sha256": digest, "score": summary["score"],
                    "signed_nll_delta": summary["signed_nll_delta"],
                    "perplexity_ratio": summary["perplexity_ratio"],
                    "ci95": summary["paired_bootstrap_ci95"],
                    "seconds": time.monotonic() - started,
                }
                del model
                clear_accelerator_cache(torch, device)
            except Exception as exc:
                record = {"status": "error", "name": path.name,
                          "importer_version": IMPORTER_VERSION,
                          "sha256": digest, "error": repr(exc),
                          "seconds": time.monotonic() - started}
            existing[path.name] = record
            payload = {
                "format": 1, "source_repo": REPO_ID,
                "source_revision": REPO_REVISION,
                "metric": "64-conversation ID validation assistant-token signed delta-NLL",
                "direction": "lower is better", "device": "mps",
                "scoring_dtype": "float32", "mps_fallback": False,
                "gguf_import": "gguf 0.19 CPU dequantization; exact imported weights; MPS FP32 scoring",
                "parameters": EXPECTED_PARAMETERS,
                "importer_version": IMPORTER_VERSION,
                "mps_lock": lock,
                "results": sorted(existing.values(), key=lambda item: item["name"]),
            }
            atomic_json(args.results, payload)
            good = [item for item in payload["results"] if item.get("status") == "ok"]
            if good:
                write_svg(args.plot, good)
            print(json.dumps(record, sort_keys=True), flush=True)
    final = [item for item in existing.values() if item.get("status") == "ok"]
    if final:
        write_svg(args.plot, final)


if __name__ == "__main__":
    main()
