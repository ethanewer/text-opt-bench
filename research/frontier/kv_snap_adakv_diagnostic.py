"""Compare equal-head SnapKV and head-adaptive Ada-SnapKV locally.

This diagnostic uses Qwen3 eager attention and teacher-forced NLL on the exact
benchmark documents.  It keeps dense cache tensors for backend portability but
masks evicted entries independently per KV head and accounts only retained KV
entries.  Thus it tests the defining head-allocation mechanism on CPU, CUDA, or
MPS without the official CUDA-only variable-length storage kernel.

It is a diagnostic for revising the task API, not a reproduction of the paper's
RULER/LongBench protocol: the local task evicts during token-by-token scoring,
whereas published SnapKV compresses a completed prompt using an observation
window.
"""

import argparse
import json
import math
from pathlib import Path

from bench import heldout
from bench.ml_models import choose_device, model_path, token_window


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench/tasks/kv_cache_policy/data"


def _frontier_eager(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, **kwargs):
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    keys = repeat_kv(key, module.num_key_value_groups)
    values = repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, keys.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    keep = getattr(module, "_frontier_keep", None)
    if keep is not None:
        expanded = keep.repeat_interleave(module.num_key_value_groups, dim=0)
        blocked = ~expanded[:, :key.shape[-2]]
        weights = weights.masked_fill(blocked[None, :, None, :],
                                      torch.finfo(weights.dtype).min)
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(weights, values).transpose(1, 2).contiguous()
    return output, weights


def _choose(scores, alive, budget, recent, adaptive):
    """Return per-head masks with an equal total retained-token budget."""
    import torch

    heads, length = scores.shape
    recent_start = max(0, length - recent)
    recent_ids = list(range(recent_start, length))
    old_budget = max(0, budget - len(recent_ids))
    result = torch.zeros_like(alive)
    result[:, recent_start:] = True
    if not old_budget or not recent_start:
        return result
    old_scores = scores[:, :recent_start].masked_fill(
        ~alive[:, :recent_start], torch.finfo(scores.dtype).min)
    if not adaptive:
        chosen = old_scores.topk(min(old_budget, recent_start), dim=1).indices
        result[:, :recent_start].scatter_(1, chosen, True)
        return result
    floor = int(old_budget * 0.5)
    if floor:
        fixed = old_scores.topk(min(floor, recent_start), dim=1).indices
        result[:, :recent_start].scatter_(1, fixed, True)
    remaining = old_scores.masked_fill(result[:, :recent_start],
                                       torch.finfo(scores.dtype).min)
    left = heads * (old_budget - min(floor, recent_start))
    if left:
        flat = remaining.flatten().topk(min(left, remaining.numel())).indices
        result[flat // recent_start, flat % recent_start] = True
    return result


def evaluate(ids, model, device, mode, budget=None, recent=4):
    import torch
    import torch.nn.functional as F
    from transformers.cache_utils import DynamicCache

    attentions = [layer.self_attn for layer in model.model.layers]
    cache = DynamicCache()
    alive = None
    scores = None
    loss = 0.0
    retained_peak = 0
    with torch.inference_mode():
        for position in range(ids.shape[1] - 1):
            if mode != "full" and alive is not None:
                for layer, attn in enumerate(attentions):
                    newest = torch.ones((alive.shape[1], 1), dtype=torch.bool,
                                        device=device)
                    attn._frontier_keep = torch.cat([alive[layer], newest], dim=1)
            output = model(
                ids[:, position:position + 1].to(device),
                past_key_values=cache,
                use_cache=True,
                output_attentions=mode != "full",
                position_ids=torch.tensor([[position]], device=device),
                cache_position=torch.tensor([position], device=device),
            )
            cache = output.past_key_values
            target = ids[:, position + 1].to(device)
            loss += float(F.cross_entropy(output.logits[:, -1].float(), target).cpu())
            if mode == "full":
                continue
            layer_count = len(attentions)
            kv_heads = attentions[0].k_proj.out_features // attentions[0].head_dim
            length = position + 1
            if alive is None:
                alive = torch.ones((layer_count, kv_heads, length),
                                   dtype=torch.bool, device=device)
                scores = torch.zeros((layer_count, kv_heads, length),
                                     dtype=torch.float32, device=device)
            else:
                alive = torch.cat([alive, torch.ones(
                    (layer_count, kv_heads, 1), dtype=torch.bool, device=device)], dim=2)
                scores = torch.cat([scores, torch.zeros(
                    (layer_count, kv_heads, 1), dtype=torch.float32, device=device)], dim=2)
            for layer, weights in enumerate(output.attentions):
                groups = attentions[layer].num_key_value_groups
                mass = weights[0, :, -1, :].reshape(kv_heads, groups, -1).mean(1)
                scores[layer, :, :mass.shape[-1]] += mass
            if length > budget:
                for layer in range(layer_count):
                    alive[layer] = _choose(scores[layer], alive[layer], budget,
                                           recent, mode == "ada_snapkv")
            retained_peak = max(retained_peak,
                                layer_count * kv_heads * min(length, budget))
    for attn in attentions:
        if hasattr(attn, "_frontier_keep"):
            del attn._frontier_keep
    return loss / (ids.shape[1] - 1), retained_peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3 import modeling_qwen3

    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = choose_device(torch)
    path = model_path("qwen3-06b", "Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, attn_implementation="eager").eval().to(device)
    rows = (json.loads((DATA / "train.json").read_text()) if args.split == "train"
            else heldout.read(DATA / "heldout_val.bin"))
    original = modeling_qwen3.eager_attention_forward
    modeling_qwen3.eager_attention_forward = _frontier_eager
    try:
        deltas = {"snapkv": [], "ada_snapkv": []}
        for row in rows:
            ids = token_window(tokenizer, [row], 64)
            full, _ = evaluate(ids, model, device, "full")
            for budget in (16, 24):
                for method in deltas:
                    value, _ = evaluate(ids, model, device, method, budget)
                    deltas[method].append(value - full)
        result = {"split": args.split, "device": str(device), "budgets": [16, 24],
                  "protocol": "local online teacher-forced NLL diagnostic"}
        for method, values in deltas.items():
            ordered = sorted(values)
            worst = ordered[-max(1, len(ordered) // 4):]
            result[method] = {
                "mean_nll_delta": sum(values) / len(values),
                "worst_quartile_nll_delta": sum(worst) / len(worst),
                "score": sum(values) / len(values) + 0.25 * sum(worst) / len(worst),
            }
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text + "\n")
        print(text)
    finally:
        modeling_qwen3.eager_attention_forward = original


if __name__ == "__main__":
    main()
