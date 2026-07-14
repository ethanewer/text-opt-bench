# Task: per-head KV prefill compression

Compress a completed prompt cache before scoring a disjoint continuation on
Qwen3-0.6B and Qwen2.5-0.5B.

```python
def select(model_family, layer_index, attention_scores, budget,
           observation_window):
    return retained_indices_per_kv_head
```

`attention_scores[head][token]` is observation-window attention mass for one
layer. Return one unique index list per KV head. Every head must retain the final
eight observation tokens; the total retained entries across heads must equal
`heads * budget`, allowing Ada-KV-style nonuniform allocation. The evaluator
scores signed continuation-NLL change at 16- and 24-token average head budgets.

Visible, validation, and sealed test contain 4, 32, and 48 independent
documents. Scores macro-average models and add a worst-model term. One eager
attention implementation with per-head masks runs on CPU, CUDA, or MPS.
