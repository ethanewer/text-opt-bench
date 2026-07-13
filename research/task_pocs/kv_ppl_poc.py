"""Score causal KV eviction with the paper-standard perplexity metric."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).parent))
from common import choose_device, dump, load_rows, timed


def slice_cache(cache, indices):
    for layer in cache.layers:
        if not layer.is_initialized:
            continue
        idx = indices.to(layer.keys.device)
        layer.keys = layer.keys.index_select(-2, idx)
        layer.values = layer.values.index_select(-2, idx)


def cache_bytes(cache):
    total = 0
    for layer in cache.layers:
        if layer.is_initialized:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    return total


@torch.inference_mode()
def evaluate_policy(model, ids, device, policy, budget, sink=4):
    cache = DynamicCache()
    scores = torch.empty(0, dtype=torch.float64)
    loss_sum = 0.0
    peak_bytes = 0
    for position in range(ids.shape[1] - 1):
        token = ids[:, position:position + 1].to(device)
        out = model(
            token,
            past_key_values=cache,
            use_cache=True,
            output_attentions=(policy == "h2o"),
            position_ids=torch.tensor([[position]], device=device),
            cache_position=torch.tensor([position], device=device),
        )
        cache = out.past_key_values
        target = ids[:, position + 1].to(device)
        loss_sum += float(F.cross_entropy(out.logits[:, -1].float(), target,
                                          reduction="sum").cpu())

        length = cache.get_seq_length()
        if len(scores) < length:
            scores = torch.cat([scores, torch.zeros(length - len(scores))])
        if policy == "h2o":
            # Accumulated attention mass is H2O's heavy-hitter signal. Average
            # layers and heads so every layer keeps a common set of positions.
            attention = torch.stack([
                value[0, :, -1, :].float().mean(dim=0).cpu()
                for value in out.attentions
            ]).mean(dim=0).double()
            scores[:len(attention)] += attention

        if budget and length > budget:
            if policy == "recent":
                keep = torch.arange(length - budget, length)
            elif policy == "sink_recent":
                recent = budget - min(sink, budget)
                keep = torch.cat([
                    torch.arange(min(sink, length)),
                    torch.arange(max(sink, length - recent), length),
                ]).unique(sorted=True)
            elif policy == "h2o":
                recent = max(1, budget // 2)
                recent_idx = torch.arange(length - recent, length)
                candidate_end = length - recent
                heavy_count = budget - recent
                if candidate_end:
                    heavy = torch.topk(scores[:candidate_end],
                                       min(heavy_count, candidate_end)).indices
                else:
                    heavy = torch.empty(0, dtype=torch.long)
                keep = torch.cat([heavy, recent_idx]).unique(sorted=True)
            else:
                raise ValueError(policy)
            slice_cache(cache, keep)
            scores = scores.index_select(0, keep)
        peak_bytes = max(peak_bytes, cache_bytes(cache))
    nll = loss_sum / (ids.shape[1] - 1)
    return {
        "policy": policy,
        "budget_tokens": budget,
        "nll": nll,
        "perplexity": math.exp(nll),
        "peak_cache_bytes": peak_bytes,
    }


def main():
    raise SystemExit(
        "retired KV prototype: long-context evaluation could not meet the "
        "benchmark runtime/quality constraint and is not part of the active suite")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--budget", type=int, default=48)
    parser.add_argument("--offset", type=int, default=70)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = choose_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # H2O needs attention weights. Qwen3 otherwise defaults to SDPA, which
    # deliberately does not materialize them, so use the eager implementation
    # consistently for every policy in this comparison.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, attn_implementation="eager")
    model.eval().to(device)
    rows = load_rows(args.rows)
    text = "\n".join(rows[args.offset:])
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :args.tokens]

    results = []
    for policy, budget in [
        ("recent", args.budget),
        ("sink_recent", args.budget),
        ("h2o", args.budget),
    ]:
        result, seconds = timed(
            torch, device,
            lambda p=policy, b=budget: evaluate_policy(
                model, ids, device, p, b))
        result["eval_seconds"] = seconds
        results.append(result)

    # Full-cache last because it has the highest memory footprint.
    result, seconds = timed(
        torch, device,
        lambda: evaluate_policy(model, ids, device, "recent", None))
    result["policy"] = "full"
    result["eval_seconds"] = seconds
    results.append(result)
    dump({
        "device": str(device),
        "tokens": ids.numel(),
        "results": sorted(results, key=lambda value: value["perplexity"]),
    })


if __name__ == "__main__":
    main()
