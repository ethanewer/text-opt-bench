"""Two-model, per-head KV prefill-compression evaluator."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.kv_prefill import (OBSERVATION, PROMPT, compressed_loss,
                              eager_with_head_mask, full_loss, token_ids)
from bench.ml_eval import call, int_list, load_candidate, split_metrics
from bench.ml_models import choose_device, model_path, round_metric

DATA = Path(__file__).resolve().parent / "data"
BUDGETS = (16, 24)
MODELS = (("qwen3", "qwen3-06b", "Qwen/Qwen3-0.6B"),
          ("qwen25", "qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B"))


def score_split(torch, F, DynamicCache, model, tokenizer, device, family, mod, rows):
    values = []
    for start in range(0, len(rows), 8):
        ids = token_ids(tokenizer, rows[start:start + 8])
        base = full_loss(torch, F, model, ids, device)
        for budget in BUDGETS:
            def selector(layer, scores, local_budget):
                masks = []
                for sample_scores in scores:
                    answer = call(mod.select, family, layer, sample_scores.tolist(),
                                  local_budget, OBSERVATION)
                    if (type(answer) not in (list, tuple) or
                            len(answer) != sample_scores.shape[0]):
                        eval_lib.fail("select must return one index list per KV head")
                    mask = torch.zeros_like(sample_scores, dtype=torch.bool)
                    total = 0
                    required = set(range(PROMPT - OBSERVATION, PROMPT))
                    for head, row in enumerate(answer):
                        indices = int_list(row, "retained indices", unique=True,
                                           low=0, high=PROMPT - 1, max_len=PROMPT)
                        if not required.issubset(indices):
                            eval_lib.fail("every head must retain the observation window")
                        total += len(indices)
                        mask[head, indices] = True
                    if total != sample_scores.shape[0] * local_budget:
                        eval_lib.fail("select must satisfy the global per-layer cache budget")
                    masks.append(mask)
                return torch.stack(masks)
            candidate = compressed_loss(torch, F, DynamicCache, model, ids,
                                        device, budget, selector)
            values.extend(a - b for a, b in zip(candidate, base))
    return values


def summarize(by_model):
    means = {key: sum(values) / len(values) for key, values in by_model.items()}
    macro, worst = sum(means.values()) / len(means), max(means.values())
    rows = [value for values in by_model.values() for value in values]
    row_mean = sum(rows) / len(rows)
    variance = sum((x - row_mean) ** 2 for x in rows) / max(1, len(rows) - 1)
    return {"score": macro + 0.25 * worst,
            "signed_nll_delta": round_metric(macro),
            "worst_model_nll_delta": round_metric(worst),
            "delta_standard_error": round_metric(math.sqrt(variance / len(rows))),
            "model_nll_delta": {key: round_metric(value) for key, value in means.items()},
            "n_model_budget_documents": len(rows)}


def main():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen2 import modeling_qwen2
        from transformers.models.qwen3 import modeling_qwen3
    except ImportError as exc:
        eval_lib.fail("model dependencies are missing: " + str(exc))
    final, train_only = "--final" in sys.argv[2:], "--train-only" in sys.argv[2:]
    path = sys.argv[1]
    rows = {"train": json.loads((DATA / "train.json").read_text()),
            "val": heldout.read(DATA / "heldout_val.bin")}
    if final:
        rows["test"] = heldout.read(DATA / "heldout_test.bin")
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = choose_device(torch)
    values = {key: {} for key in rows}
    originals = (modeling_qwen2.eager_attention_forward,
                 modeling_qwen3.eager_attention_forward)
    modeling_qwen2.eager_attention_forward = eager_with_head_mask
    modeling_qwen3.eager_attention_forward = eager_with_head_mask
    try:
        for family, local_name, hub_name in MODELS:
            model = AutoModelForCausalLM.from_pretrained(
                model_path(local_name, hub_name), local_files_only=True,
                attn_implementation="eager").eval().to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path(local_name, hub_name), local_files_only=True)
            for key, local_rows in rows.items():
                candidate = load_candidate(path, ("select",))
                values[key][family] = score_split(
                    torch, F, DynamicCache, model, tokenizer, device,
                    family, candidate, local_rows)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
    finally:
        modeling_qwen2.eager_attention_forward = originals[0]
        modeling_qwen3.eager_attention_forward = originals[1]
    train, val = summarize(values["train"]), summarize(values["val"])
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    test = summarize(values["test"]) if final else None
    metrics = split_metrics(train, val, test)
    metrics.update(models=[row[2] for row in MODELS], device=str(device),
                   budgets=list(BUDGETS), prompt_tokens=PROMPT,
                   paper_metric="continuation NLL after per-head prefill compression")
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
