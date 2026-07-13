"""Portable SnapKV/Ada-SnapKV comparison for the v2 prefill protocol."""

import argparse
import json
import math
from pathlib import Path

from bench import heldout
from bench.ml_models import choose_device, model_path


ROOT = Path(__file__).resolve().parents[2]
TEXT = Path("/tmp/text-opt-bm-tinystories-rows.json")
MODELS = (("qwen3", "qwen3-06b", "Qwen/Qwen3-0.6B"),
          ("qwen25", "qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B"))
PROMPT = 64
CONTINUATION = 32
OBSERVATION = 8
BUDGETS = (16, 24)


def rows_for(split, limit=None):
    payload = json.loads(TEXT.read_text())
    rows = [item["row"]["text"] for item in payload["rows"]]
    selected = {"train": rows[:16], "val": rows[20:52],
                "test": rows[52:100]}[split]
    return selected[:limit] if limit else selected


def frontier_eager(module, query, key, value, attention_mask, scaling,
                   dropout=0.0, **kwargs):
    import torch
    import torch.nn.functional as F

    groups = module.num_key_value_groups
    keys = key[:, :, None, :, :].expand(-1, -1, groups, -1, -1)
    values = value[:, :, None, :, :].expand(-1, -1, groups, -1, -1)
    keys = keys.reshape(key.shape[0], -1, key.shape[-2], key.shape[-1])
    values = values.reshape(value.shape[0], -1, value.shape[-2], value.shape[-1])
    weights = torch.matmul(query, keys.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    keep = getattr(module, "_frontier_keep", None)
    if keep is not None:
        expanded = keep.repeat_interleave(groups, dim=1)
        weights = weights.masked_fill(~expanded[:, :, None, :key.shape[-2]],
                                      torch.finfo(weights.dtype).min)
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, values).transpose(1, 2).contiguous(), weights


def select_mask(scores, budget, adaptive):
    """Select prompt entries independently for every batch/layer/KV head."""
    import torch

    batch, layers, heads, length = scores.shape
    recent_start = length - OBSERVATION
    old_budget = budget - OBSERVATION
    result = torch.zeros_like(scores, dtype=torch.bool)
    result[..., recent_start:] = True
    old = scores[..., :recent_start]
    if not adaptive:
        chosen = old.topk(old_budget, dim=-1).indices
        result[..., :recent_start].scatter_(-1, chosen, True)
        return result
    floor = max(1, old_budget // 2)
    fixed = old.topk(floor, dim=-1).indices
    result[..., :recent_start].scatter_(-1, fixed, True)
    remaining = old.masked_fill(result[..., :recent_start],
                                torch.finfo(old.dtype).min)
    left = heads * (old_budget - floor)
    flat = remaining.reshape(batch, layers, heads * recent_start)
    chosen = flat.topk(left, dim=-1).indices
    head = chosen // recent_start
    token = chosen % recent_start
    for b in range(batch):
        for layer in range(layers):
            result[b, layer, head[b, layer], token[b, layer]] = True
    return result


def token_batch(tokenizer, rows):
    encoded = tokenizer(rows, padding=True, truncation=True,
                        max_length=PROMPT + CONTINUATION,
                        return_tensors="pt")
    ids = encoded.input_ids[:, :PROMPT + CONTINUATION]
    if ids.shape[1] < PROMPT + CONTINUATION:
        raise RuntimeError("a v2 document is shorter than the token window")
    return ids


def full_losses(torch, F, model, ids, device):
    with torch.inference_mode():
        local = ids.to(device)
        logits = model(local).logits[:, PROMPT:-1].float()
        targets = local[:, PROMPT + 1:]
        values = F.cross_entropy(logits.transpose(1, 2), targets,
                                 reduction="none").mean(dim=1)
    return values.cpu().tolist()


def compressed_losses(torch, F, DynamicCache, model, ids, device, budget, method):
    attentions = [layer.self_attn for layer in model.model.layers]
    cache = DynamicCache()
    batch = ids.shape[0]
    with torch.inference_mode():
        output = model(ids[:, :PROMPT].to(device), past_key_values=cache,
                       use_cache=True, output_attentions=True)
        cache = output.past_key_values
        scores = []
        for layer, weights in enumerate(output.attentions):
            kv_heads = attentions[layer].k_proj.out_features // attentions[layer].head_dim
            groups = attentions[layer].num_key_value_groups
            mass = weights[:, :, -OBSERVATION:, :].mean(dim=2)
            scores.append(mass.reshape(batch, kv_heads, groups, PROMPT).mean(2))
        keep = select_mask(torch.stack(scores, dim=1), budget,
                           method == "ada_snapkv")
        losses = torch.zeros(batch, device=device)
        count = 0
        for position in range(PROMPT, PROMPT + CONTINUATION - 1):
            call_keep = torch.cat([keep, torch.ones((*keep.shape[:-1], 1),
                                                     dtype=torch.bool, device=device)], dim=-1)
            for layer, attention in enumerate(attentions):
                attention._frontier_keep = call_keep[:, layer]
            output = model(ids[:, position:position + 1].to(device),
                           past_key_values=cache, use_cache=True,
                           position_ids=torch.full((batch, 1), position, device=device),
                           cache_position=torch.tensor([position], device=device))
            cache = output.past_key_values
            target = ids[:, position + 1].to(device)
            losses += F.cross_entropy(output.logits[:, -1].float(), target,
                                      reduction="none")
            count += 1
            keep = call_keep
    for attention in attentions:
        if hasattr(attention, "_frontier_keep"):
            del attention._frontier_keep
    return (losses / count).cpu().tolist()


def evaluate_model(model_key, local_name, hub_name, rows, batch_size):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen3 import modeling_qwen3

    device = choose_device(torch)
    path = model_path(local_name, hub_name)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, attn_implementation="eager").eval().to(device)
    originals = (modeling_qwen2.eager_attention_forward,
                 modeling_qwen3.eager_attention_forward)
    modeling_qwen2.eager_attention_forward = frontier_eager
    modeling_qwen3.eager_attention_forward = frontier_eager
    result = {method: [] for method in ("snapkv", "ada_snapkv")}
    try:
        for start in range(0, len(rows), batch_size):
            ids = token_batch(tokenizer, rows[start:start + batch_size])
            full = full_losses(torch, F, model, ids, device)
            for budget in BUDGETS:
                for method in result:
                    compressed = compressed_losses(torch, F, DynamicCache, model,
                                                   ids, device, budget, method)
                    result[method].extend(c - f for c, f in zip(compressed, full))
    finally:
        modeling_qwen2.eager_attention_forward = originals[0]
        modeling_qwen3.eager_attention_forward = originals[1]
    summary = {method: {"mean_nll_delta": sum(values) / len(values),
                        "standard_error": (sum((x - sum(values) / len(values)) ** 2
                                               for x in values) /
                                           max(1, len(values) - 1) / len(values)) ** 0.5,
                        "n": len(values)} for method, values in result.items()}
    paired = [a - s for a, s in zip(result["ada_snapkv"], result["snapkv"])]
    paired_mean = sum(paired) / len(paired)
    paired_se = (sum((x - paired_mean) ** 2 for x in paired) /
                 max(1, len(paired) - 1) / len(paired)) ** 0.5
    summary["ada_minus_snapkv"] = {
        "mean": paired_mean, "standard_error": paired_se,
        "ci95": [paired_mean - 1.96 * paired_se,
                 paired_mean + 1.96 * paired_se]}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    rows = rows_for(args.split, args.limit)
    output = {key: evaluate_model(key, local, hub, rows, args.batch_size)
              for key, local, hub in MODELS}
    print(json.dumps({"split": args.split, "documents": len(rows),
                      "models": output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
