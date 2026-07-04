# Task: kv_fixed_budget - KV-cache compression under a hard byte cap

Compress real-model KV-cache slices while staying under a fixed encoded
byte budget. The data comes from selected heads of an open-weight
TinyStories GPT-2-style model on public-domain text.

## Required API

```python
def encode(cache, config):
    """Return a deterministic compressed KV representation."""

def attend(encoded, queries, config):
    """Return one attention output per selected layer/query."""
```

Each layer has `keys`, `values`, and observation-window `importance`
scores. `config["max_encoded_bytes"]` is a hard cap. Exceeding it is
invalid.

## Scoring

Lower is better:

`error_weight * layer_weighted_attention_MSE + instruction_weight * bytecode_instructions`

This setting models deployments where the KV-cache memory budget is
fixed and the goal is to maximize fidelity under that budget. Token
selection, recent/sink retention, layer-wise budgets, and quantization
are all valid.

## Rules

Same cooperative sandbox rules as `kv_quant`: no imports, curated
builtins only, deterministic behavior required. The candidate module is
reloaded between `encode` and `attend`, so the encoded object must be
self-contained.
