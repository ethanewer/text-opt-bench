"""Backend-agnostic pruning/quantization evaluation on a tiny causal LM.

The official-quality candidate metric tested here is held-out perplexity.
Calibration logits KL and weight MSE are retained only to test whether those
surrogates preserve the downstream method ranking.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import choose_device, dump, load_rows, spearman, timed


def linear_modules(model):
    return [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name]


def token_windows(tokenizer, texts, window, count):
    ids = tokenizer("\n".join(texts), return_tensors="pt").input_ids[0]
    usable = min(count, max(1, len(ids) // window))
    return [ids[i * window:(i + 1) * window].unsqueeze(0) for i in range(usable)]


@torch.inference_mode()
def logits_and_nll(model, windows, device):
    logits, losses, tokens = [], 0.0, 0
    for cpu_ids in windows:
        ids = cpu_ids.to(device)
        out = model(ids)
        pred = out.logits[:, :-1].float()
        target = ids[:, 1:]
        losses += float(F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), target.reshape(-1),
            reduction="sum").cpu())
        tokens += target.numel()
        logits.append(pred.cpu())
    nll = losses / tokens
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in logits]), nll


def collect_input_norms(model, windows, device):
    sums = {}
    handles = []
    for name, mod in linear_modules(model):
        sums[name] = torch.zeros(mod.in_features, dtype=torch.float64)

        def hook(_mod, args, _out, key=name):
            # MPS has no float64 kernels; move first, then accumulate in
            # float64 on CPU for backend-independent activation statistics.
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu().double()
            sums[key].add_((x * x).sum(dim=0))

        handles.append(mod.register_forward_hook(hook))
    with torch.inference_mode():
        for ids in windows:
            model(ids.to(device))
    for handle in handles:
        handle.remove()
    return {key: value.sqrt().float() for key, value in sums.items()}


def prune_per_row(model, fraction, mode, input_norms=None, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, mod in linear_modules(model):
            weight = mod.weight.data
            if mode == "magnitude":
                importance = weight.abs()
            elif mode == "wanda":
                importance = weight.abs() * input_norms[name].to(weight.device)
            elif mode == "random":
                importance = torch.rand(weight.shape, generator=generator,
                                        dtype=weight.dtype).to(weight.device)
            else:
                raise ValueError(mode)
            k = max(1, int(weight.shape[1] * fraction))
            indices = torch.topk(importance, k, dim=1, largest=False).indices
            weight.scatter_(1, indices, 0)


def quantize(model, bits, granularity, clip=1.0, group_size=32):
    levels = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for _name, mod in linear_modules(model):
            w = mod.weight.data
            if granularity == "tensor":
                scale = w.abs().amax().clamp_min(1e-8) * clip / levels
                mod.weight.data = (w.clamp(-levels * scale, levels * scale) /
                                   scale).round().clamp(-levels, levels) * scale
            elif granularity == "row":
                scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                scale = scale * clip / levels
                mod.weight.data = (w.clamp(-levels * scale, levels * scale) /
                                   scale).round().clamp(-levels, levels) * scale
            elif granularity == "group":
                result = torch.empty_like(w)
                for start in range(0, w.shape[1], group_size):
                    block = w[:, start:start + group_size]
                    scale = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                    scale = scale * clip / levels
                    result[:, start:start + group_size] = (
                        block.clamp(-levels * scale, levels * scale) /
                        scale).round().clamp(-levels, levels) * scale
                mod.weight.data = result
            else:
                raise ValueError(granularity)


def weight_mse(reference, candidate):
    ref = dict(linear_modules(reference))
    total_error = total_energy = 0.0
    for name, mod in linear_modules(candidate):
        a = ref[name].weight.detach().float().cpu()
        b = mod.weight.detach().float().cpu()
        total_error += float(((a - b) ** 2).sum())
        total_energy += float((a ** 2).sum())
    return total_error / total_energy


def logits_kl(reference_logits, candidate_logits):
    ref_log = F.log_softmax(reference_logits.float(), dim=-1)
    cand_log = F.log_softmax(candidate_logits.float(), dim=-1)
    return float(F.kl_div(cand_log, ref_log.exp(), reduction="batchmean"))


def load_causal_model(path, language_model_only):
    """Load a causal LM, optionally excluding Qwen3.5's vision tower.

    Qwen3.5 checkpoints use ``model.language_model.*`` names inside a
    multimodal wrapper. Transformers' explicit key mapping lets us instantiate
    only Qwen3_5ForCausalLM and stream those text weights into it, so the vision
    encoder is never resident.
    """
    if not language_model_only:
        return AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    root_config = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(root_config, "model_type", None) != "qwen3_5":
        raise ValueError("--language-model-only currently requires a Qwen3.5 checkpoint")
    return AutoModelForCausalLM.from_pretrained(
        path,
        config=root_config.text_config,
        local_files_only=True,
        key_mapping={r"^model\.language_model\.": "model."},
    )


def main():
    raise SystemExit(
        "retired SLM prototype: use slm_compression_v2, "
        "slm_compression_qwen35, or research/baselines/slm_paper_native; "
        "all admissible SLM work is MPS-only and globally serialized")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--windows-per-split", type=int, default=2)
    parser.add_argument("--quick", action="store_true",
                        help="run a six-method subset suitable for 0.5B models")
    parser.add_argument("--language-model-only", action="store_true",
                        help="load Qwen3.5 text weights without its vision tower")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = choose_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = load_causal_model(args.model, args.language_model_only)
    base.eval().to(device)
    rows = load_rows(args.rows)
    thirds = max(1, len(rows) // 3)
    cal = token_windows(tokenizer, rows[:thirds], args.window,
                        args.windows_per_split)
    val = token_windows(tokenizer, rows[thirds:2 * thirds], args.window,
                        args.windows_per_split)
    test = token_windows(tokenizer, rows[2 * thirds:], args.window,
                         args.windows_per_split)

    (base_cal_logits, base_cal_nll), base_cal_s = timed(
        torch, device, lambda: logits_and_nll(base, cal, device))
    (_base_val_logits, base_val_nll), base_val_s = timed(
        torch, device, lambda: logits_and_nll(base, val, device))
    (_base_test_logits, base_test_nll), base_test_s = timed(
        torch, device, lambda: logits_and_nll(base, test, device))
    norms = collect_input_norms(base, cal, device)

    methods = {
        "prune_magnitude_30": lambda m: prune_per_row(m, .3, "magnitude"),
        "prune_wanda_30": lambda m: prune_per_row(m, .3, "wanda", norms),
        "prune_random_50": lambda m: prune_per_row(m, .5, "random", seed=7),
        "prune_magnitude_50": lambda m: prune_per_row(m, .5, "magnitude"),
        "prune_wanda_50": lambda m: prune_per_row(m, .5, "wanda", norms),
        "quant_tensor_4b": lambda m: quantize(m, 4, "tensor"),
        "quant_row_4b": lambda m: quantize(m, 4, "row"),
        "quant_group32_4b": lambda m: quantize(m, 4, "group"),
        "quant_group32_clip90_4b": lambda m: quantize(m, 4, "group", .90),
        "quant_tensor_3b": lambda m: quantize(m, 3, "tensor"),
        "quant_row_3b": lambda m: quantize(m, 3, "row"),
        "quant_group32_3b": lambda m: quantize(m, 3, "group"),
        "quant_group32_clip85_3b": lambda m: quantize(m, 3, "group", .85),
    }
    if args.quick:
        selected = {
            "prune_magnitude_50", "prune_wanda_50",
            "quant_tensor_4b", "quant_row_4b", "quant_group32_4b",
            "quant_group32_clip90_4b",
        }
        methods = {name: method for name, method in methods.items()
                   if name in selected}
    results = []
    for name, transform in methods.items():
        candidate = copy.deepcopy(base).to(device)
        transform(candidate)
        candidate.eval()
        (cal_logits, cal_nll), cal_s = timed(
            torch, device, lambda: logits_and_nll(candidate, cal, device))
        (_val_logits, val_nll), val_s = timed(
            torch, device, lambda: logits_and_nll(candidate, val, device))
        (_test_logits, test_nll), test_s = timed(
            torch, device, lambda: logits_and_nll(candidate, test, device))
        results.append({
            "method": name,
            "weight_nmse": weight_mse(base, candidate),
            "calibration_logits_kl": logits_kl(base_cal_logits, cal_logits),
            "calibration_nll_delta": cal_nll - base_cal_nll,
            "validation_ppl": math.exp(val_nll),
            "validation_nll_delta": val_nll - base_val_nll,
            "test_ppl": math.exp(test_nll),
            "test_nll_delta": test_nll - base_test_nll,
            "eval_seconds": cal_s + val_s + test_s,
        })
        del candidate

    test_deltas = [r["test_nll_delta"] for r in results]
    payload = {
        "device": str(device),
        "model_parameters": sum(p.numel() for p in base.parameters()),
        "tokens_per_split": sum(x.numel() - 1 for x in cal),
        "base": {
            "calibration_ppl": math.exp(base_cal_nll),
            "validation_ppl": math.exp(base_val_nll),
            "test_ppl": math.exp(base_test_nll),
            "eval_seconds": base_cal_s + base_val_s + base_test_s,
        },
        "methods": sorted(results, key=lambda x: x["test_ppl"]),
        "rank_correlations_to_test_nll": {
            "weight_nmse": spearman([r["weight_nmse"] for r in results],
                                    test_deltas),
            "calibration_logits_kl": spearman(
                [r["calibration_logits_kl"] for r in results], test_deltas),
            "calibration_nll_delta": spearman(
                [r["calibration_nll_delta"] for r in results], test_deltas),
            "validation_nll_delta": spearman(
                [r["validation_nll_delta"] for r in results], test_deltas),
        },
    }
    dump(payload)


if __name__ == "__main__":
    main()
