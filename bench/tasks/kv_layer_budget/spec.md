# Task: kv_layer_budget - layer-wise KV-cache budget allocation

Allocate token-retention and quantization budgets across selected
real-model KV-cache layers. The evaluator owns the compression algorithm:
it keeps sink tokens, recent tokens, and observation heavy hitters, then
quantizes each retained layer according to your budget.

## Required API

```python
def allocate(cache_info, config):
    """Return one [keep_tokens, key_levels, value_levels] list per layer."""
```

Allowed quantization levels are in `config["allowed_levels"]`. The
encoded cache must fit `config["max_encoded_bytes"]`.

## Scoring

Lower is better:

`error_weight * layer_weighted_attention_MSE + instruction_weight * allocate_instructions`

This isolates PyramidKV/AdaKV-style layer budget allocation from the
lower-level implementation details of a compressor.

## Rules

No imports. Deterministic behavior required. Programs run with curated
builtins only.
