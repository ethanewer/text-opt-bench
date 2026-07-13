# Task: evaluator-owned SLM compression policy

Choose per-layer quantization and pruning for the language-only component of
`Qwen/Qwen3.5-0.8B`. The vision tower is never instantiated.

```python
def policy(layer_name, rows, columns, mean_abs, max_abs, target_bits):
    return [bits, group_size, clip, prune_fraction]
```

`bits` is 2–8, group size is 16/32/64/128, clip is 0.5–1.2, and pruning is
0–0.75. The evaluator performs magnitude pruning and symmetric groupwise
quantization, accounts packed nonzero values, pruning bitmaps, and FP16 scales,
then executes the complete model. Policies exceeding 4.25 bits/weight receive
a steep deterministic penalty.

The ranked metric is held-out NLL/perplexity degradation from the uncompressed
model under the exact storage budget. Weight MSE and logit KL are deliberately
not optimization targets. Calibration, validation, and sealed test documents
are disjoint. Metrics report exact bits/weight and perplexity.

One implementation runs locally on CPU, CUDA, or MPS using a local-only model
snapshot. Candidate policy code is deterministic, import-free safe Python and
limited to 32 KB.
