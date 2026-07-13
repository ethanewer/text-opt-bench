# Task: end-to-end KV-cache eviction policy

Design a causal token-retention policy for `Qwen/Qwen3-0.6B`. The evaluator
owns the model, cache tensors, attention, text, and NLL computation.

```python
def select(token_ids, attention_scores, budget, absolute_position):
    return retained_indices
```

Indices address the currently retained cache, must be unique, must include the
newest token, and must contain exactly `budget` entries. The evaluator applies
the selection to every layer's K/V tensors. `attention_scores` contains
accumulated mean attention mass, enabling H2O-like policies; token IDs and
positions enable sink, recency, and content-aware hybrids.

The score is mean teacher-forced NLL degradation from a full cache at 16- and
24-token budgets, plus 0.25 times worst-quartile degradation. Metrics report
perplexity and exact peak-cache ratios. Visible, validation, and sealed test
documents differ; the candidate never emits logits or attention outputs.

Scoring uses one PyTorch implementation on CPU, CUDA, or MPS and a local-only
checkpoint. Candidate policy code is import-free safe Python limited to 32 KB.
