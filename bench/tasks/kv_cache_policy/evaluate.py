"""End-to-end Qwen3 KV eviction scored by teacher-forced NLL."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, int_list, load_candidate, split_metrics
from bench.ml_models import choose_device, model_path, round_metric, token_window

DATA = Path(__file__).resolve().parent / "data"
BUDGETS = (16, 24)
TOKENS = 64


def slice_cache(cache, indices):
    for layer in cache.layers:
        if layer.is_initialized:
            local = indices.to(layer.keys.device)
            layer.keys = layer.keys.index_select(-2, local)
            layer.values = layer.values.index_select(-2, local)


def evaluate_one(torch, F, DynamicCache, model, ids, device, mod, budget=None):
    cache = DynamicCache()
    scores = torch.empty(0, dtype=torch.float64)
    retained_tokens = []
    loss_sum = 0.0
    peak_bytes = 0
    with torch.inference_mode():
        for position in range(ids.shape[1] - 1):
            token_id = int(ids[0, position])
            retained_tokens.append(token_id)
            out = model(
                ids[:, position:position + 1].to(device),
                past_key_values=cache, use_cache=True,
                output_attentions=budget is not None,
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
            if budget is not None:
                attention = torch.stack([
                    value[0, :, -1, :].float().mean(dim=0).cpu()
                    for value in out.attentions
                ]).mean(dim=0).double()
                scores[:len(attention)] += attention
            if budget is not None and length > budget:
                keep = int_list(
                    call(mod.select, list(retained_tokens), scores.tolist(),
                         budget, position),
                    "select result", unique=True, low=0, high=length - 1,
                    max_len=budget,
                )
                if len(keep) != budget or length - 1 not in keep:
                    eval_lib.fail(
                        "select must retain exactly budget entries including the newest token")
                keep.sort()
                index = torch.tensor(keep, dtype=torch.long)
                slice_cache(cache, index)
                scores = scores.index_select(0, index)
                retained_tokens = [retained_tokens[i] for i in keep]
            cache_size = 0
            for layer in cache.layers:
                if layer.is_initialized:
                    cache_size += ((layer.keys.numel() + layer.values.numel()) *
                                   layer.keys.element_size())
            peak_bytes = max(peak_bytes, cache_size)
    return loss_sum / (ids.shape[1] - 1), peak_bytes


def score_split(torch, F, DynamicCache, model, tokenizer, device, mod, rows):
    deltas, ratios = [], []
    for row in rows:
        ids = token_window(tokenizer, [row], TOKENS)
        full_nll, full_bytes = evaluate_one(
            torch, F, DynamicCache, model, ids, device, mod, None)
        for budget in BUDGETS:
            candidate_nll, candidate_bytes = evaluate_one(
                torch, F, DynamicCache, model, ids, device, mod, budget)
            deltas.append(max(0.0, candidate_nll - full_nll))
            ratios.append(candidate_bytes / full_bytes)
    ordered = sorted(deltas)
    worst = ordered[-max(1, len(ordered) // 4):]
    mean = sum(deltas) / len(deltas)
    return {
        "score": round_metric(mean + .25 * sum(worst) / len(worst)),
        "mean_nll_delta": round_metric(mean),
        "worst_quartile_nll_delta": round_metric(sum(worst) / len(worst)),
        "perplexity_ratio": round_metric(math.exp(mean)),
        "peak_cache_ratio": round_metric(sum(ratios) / len(ratios)),
        "n_sequences": len(rows),
    }


def main():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:
        eval_lib.fail("model task dependencies are missing; run tools/prepare_ml_benchmark.py: " + str(exc))
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    program_path = sys.argv[1]
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = choose_device(torch)
    path = model_path("qwen3-06b", "Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, attn_implementation="eager")
    model.eval().to(device)
    train_rows = json.loads((DATA / "train.json").read_text())
    val_rows = heldout.read(DATA / "heldout_val.bin")
    test_rows = heldout.read(DATA / "heldout_test.bin") if final else None
    def fresh_score(rows):
        candidate = load_candidate(program_path, ("select",))
        return score_split(torch, F, DynamicCache, model, tokenizer, device,
                           candidate, rows)

    train = fresh_score(train_rows)
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    val = fresh_score(val_rows)
    test = fresh_score(test_rows) if final else None
    metrics = split_metrics(train, val, test)
    metrics.update(model="Qwen/Qwen3-0.6B", device=str(device), budgets=list(BUDGETS),
                   paper_metric="teacher-forced perplexity/NLL at fixed KV budget")
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
