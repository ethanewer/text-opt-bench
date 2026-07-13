"""Evaluator-owned Qwen3.5 per-layer quantization/pruning policy task."""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, finite, integer, load_candidate, split_metrics
from bench.ml_models import (attest_fresh_mps_torch_import,
                             attest_model_device_dtype, choose_slm_device,
                             linear_modules, load_qwen35_text, model_path,
                             mps_fallback_enabled, nll,
                             require_fresh_torch_import, round_metric,
                             token_window)
from bench.slm_mps_lock import require_active_mps_lock, serialized_mps_job

DATA = Path(__file__).resolve().parent / "data"
WINDOW = 96
TARGET_BITS = 4.25


def apply_policy(torch, model, mod):
    total_weights = 0
    total_bits = 0.0
    policies = {}
    with torch.no_grad():
        for name, layer in linear_modules(torch, model):
            weight = layer.weight.data
            mean_abs = float(weight.abs().mean())
            max_abs = float(weight.abs().max())
            answer = call(mod.policy, name, weight.shape[0], weight.shape[1],
                          mean_abs, max_abs, TARGET_BITS)
            if type(answer) not in (list, tuple) or len(answer) != 4:
                eval_lib.fail("policy must return [bits, group_size, clip, prune_fraction]")
            bits = integer(answer[0], "quantization bits", 2, 8)
            group = integer(answer[1], "group size", 16, 128)
            if group not in (16, 32, 64, 128):
                eval_lib.fail("group size must be 16, 32, 64, or 128")
            clip = finite(answer[2], "clip")
            prune = finite(answer[3], "prune fraction")
            if not (0.5 <= clip <= 1.2 and 0.0 <= prune <= 0.75):
                eval_lib.fail("clip/prune are outside allowed ranges")
            k = 0
            if prune:
                k = min(weight.shape[1] - 1, int(weight.shape[1] * prune))
                if k:
                    indices = torch.topk(weight.abs(), k, dim=1, largest=False).indices
                    weight.scatter_(1, indices, 0)
            levels = 2 ** (bits - 1) - 1
            result = torch.empty_like(weight)
            for start in range(0, weight.shape[1], group):
                block = weight[:, start:start + group]
                scale = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                scale = scale * clip / levels
                result[:, start:start + group] = (
                    block.clamp(-levels * scale, levels * scale) /
                    scale).round().clamp(-levels, levels) * scale
            layer.weight.data = result
            count = weight.numel()
            # Packed nonzeros + optional bitmap + FP16 scale per row/group.
            pruned_count = weight.shape[0] * k
            layer_bits = (count - pruned_count) * bits
            if k:
                layer_bits += count
            layer_bits += weight.shape[0] * math.ceil(weight.shape[1] / group) * 16
            total_weights += count
            total_bits += layer_bits
            policies[name] = [bits, group, round(clip, 4), round(prune, 4)]
    return total_bits / total_weights, policies


def score_nll(torch, model, tokenizer, device, rows, base_value):
    ids = token_window(tokenizer, rows, WINDOW)
    value = nll(torch, model, ids, device)
    delta = max(0.0, value - base_value)
    return delta, math.exp(value)


@serialized_mps_job("slm-legacy-evaluator")
def main():
    try:
        require_active_mps_lock("retired SLM compression evaluation")
    except RuntimeError as exc:
        eval_lib.fail(str(exc))
    try:
        require_fresh_torch_import("retired SLM compression evaluation")
    except RuntimeError as exc:
        eval_lib.fail(str(exc))
    if mps_fallback_enabled():
        eval_lib.fail(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing SLM evaluation")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        eval_lib.fail("model task dependencies are missing; run tools/prepare_ml_benchmark.py: " + str(exc))
    try:
        attest_fresh_mps_torch_import(
            torch, "retired SLM compression evaluation")
    except RuntimeError as exc:
        eval_lib.fail(str(exc))
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    mod = load_candidate(sys.argv[1], ("policy",))
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = choose_slm_device(torch, "mps")
    path = model_path("qwen35-08b", "Qwen/Qwen3.5-0.8B")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = load_qwen35_text(path)
    model.eval().to(device=device, dtype=torch.bfloat16)
    try:
        attest_model_device_dtype(
            torch, model, device, "retired SLM compression model",
            torch.bfloat16)
    except RuntimeError as exc:
        eval_lib.fail(str(exc))
    train_rows = json.loads((DATA / "train.json").read_text())
    val_rows = heldout.read(DATA / "heldout_val.bin")
    test_rows = heldout.read(DATA / "heldout_test.bin") if final else None
    train_ids = token_window(tokenizer, train_rows, WINDOW)
    val_ids = token_window(tokenizer, val_rows, WINDOW)
    test_ids = token_window(tokenizer, test_rows, WINDOW) if final else None
    base_train = nll(torch, model, train_ids, device)
    base_val = nll(torch, model, val_ids, device)
    base_test = nll(torch, model, test_ids, device) if final else None
    bits_per_weight, policies = apply_policy(torch, model, mod)
    model.to(device=device, dtype=torch.bfloat16).eval()
    try:
        attest_model_device_dtype(
            torch, model, device, "retired compressed SLM model",
            torch.bfloat16)
    except RuntimeError as exc:
        eval_lib.fail(str(exc))
    train_nll = nll(torch, model, train_ids, device)
    train_delta = max(0.0, train_nll - base_train)
    budget_penalty = max(0.0, bits_per_weight - TARGET_BITS) * 4.0
    train = {"score": round_metric(train_delta + budget_penalty),
             "nll_delta": round_metric(train_delta),
             "perplexity": round_metric(math.exp(train_nll)),
             "bits_per_weight": round_metric(bits_per_weight)}
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    val_nll = nll(torch, model, val_ids, device)
    val_delta = max(0.0, val_nll - base_val)
    val = {"score": round_metric(val_delta + budget_penalty),
           "nll_delta": round_metric(val_delta),
           "perplexity": round_metric(math.exp(val_nll)),
           "bits_per_weight": round_metric(bits_per_weight)}
    test = None
    if final:
        test_nll = nll(torch, model, test_ids, device)
        test_delta = max(0.0, test_nll - base_test)
        test = {"score": round_metric(test_delta + budget_penalty),
                "nll_delta": round_metric(test_delta),
                "perplexity": round_metric(math.exp(test_nll)),
                "bits_per_weight": round_metric(bits_per_weight)}
    metrics = split_metrics(train, val, test)
    metrics.update(model="Qwen/Qwen3.5-0.8B language model only",
                   device=str(device), target_bits_per_weight=TARGET_BITS,
                   canonical_device="mps", compression_device=str(device),
                   mps_fallback_enabled=mps_fallback_enabled(),
                   configured_layers=len(policies),
                   paper_metric="held-out perplexity/NLL under exact packed-bit budget")
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
