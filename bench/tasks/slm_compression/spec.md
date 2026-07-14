# Task: Qwen GQA SFT compression policy

Design a globally budgeted post-training weight-compression plan on
Qwen2.5-0.5B-Instruct. The same algorithm is applied without score feedback to
nonthinking Qwen3-0.6B. The final report separates in-distribution, data,
model, and joint model-and-data generalization.

Each 64-conversation scoring cell must contain at least 512 assistant target
tokens; exact counts are pinned in the data manifest.

```python
def plan(layers, target_bits):
    return [layer_policy for layer in layers]
```

`layers` is an immutable tuple in execution order. Each descriptor is:

```text
(layer_role, normalized_depth, rows, columns,
 weight_mean_abs, weight_max_abs, activation_rms, activation_max_abs)
```

The activation arrays are immutable tuples with one value per input channel.
They are computed separately on each target model. Qwen2.5 uses the fixed SFT
calibration conversations. Qwen3 target calibration is strictly prompt-only on
the same 128 Qwen2.5 training prompt IDs: it consumes neither Qwen2.5 assistant
answers nor Qwen3-generated assistant answers. Calibration statistics are
allowed in ordinary post-training quantization; no Qwen3 response, loss, or
performance outcome is exposed to optimization.

Return exactly one six-value policy per layer:

```text
[bits, group_size, clip, prune_fraction,
 prune_activation_power, quant_activation_power]
```

- `bits`: integer 2--8.
- `group_size`: 16, 32, 64, or 128.
- `clip`: 0.5--1.2.
- `prune_fraction`: 0--0.75.
- both activation powers: 0--1.

The evaluator owns the transforms. A zero quantization-activation power is
symmetric groupwise round-to-nearest; a positive value supplies AWQ-style
channel scaling. Every active channel scale is rounded to FP16, stored as one
multiplier per input channel, and charged to the budget; no uncounted scale
fusion is assumed. Row/group scales are likewise rounded to FP16 before the
scored fake-dequantization. A zero pruning-activation power is magnitude
pruning; one is Wanda-style activation weighting. This task synthesizes allocation and
hyperparameters over these evaluator-owned transforms; it is not an arbitrary
compressor and does not claim to reproduce GPTQ or SparseGPT.

Plans are independently evaluated at hard 3.125- and 4.125-bit eligible-weight
caps. Packed nonzeros, a one-bit sparsity bitmap when used, FP16 row/group
scales, any FP16 per-input-channel multipliers, and one self-decoding header
byte per eligible Linear count toward the cap. The header records the local
bitwidth, group size, and sparse/channel-section flags. Any
per-model violation is invalid. Consequently a dense activation-scaled plan
must allocate fewer codes on some layers; it cannot reuse a nominal dense RTN
bit allocation and ignore its channel metadata. Metrics also
report honest whole-model storage including embeddings, norms, and other
uncompressed parameters.

Data are model-generated, text-only SFT conversations of at most 512 tokens.
Only assistant response tokens contribute to cross-entropy. The signed score
is compressed NLL minus an uncompressed reference measured on the same runtime
backend, equivalently the log perplexity ratio; improvements remain negative.
Build-time reference losses are retained only as an audit diagnostic and do
not enter the score. Generation, activation calibration, the candidate's
quantization/pruning transform, and both scoring passes use Apple MPS with
PyTorch CPU fallback disabled. The BF16 checkpoint and compressed weights are
converted to FP32 for scoring. Every result and compiled artifact records this
canonical backend; CPU, CUDA, MLX, and fallback-enabled results are
inadmissible.
Every model-bearing SLM process—including generation, compilation and
activation calibration, paper-native diagnostics, online validation, and
sealed testing—must hold the suite's single non-preemptive cross-process MPS
lease for its entire model-work interval.
Aggregation proceeds through
conversation, operation-template, domain, overlapping/held-out domain group,
model, and storage point, giving equal weight at each macro level. To avoid
treating templated variants as independent, confidence
intervals resample operation-level template clusters within each domain while
keeping all cross-model prompt IDs paired.
The final report includes deterministic cluster-bootstrap intervals for every
model/domain-group curve and every storage point, with the same resampled
clusters reused across both storage points.

The 128 training conversations are **calibration only** and contribute no loss
to the optimization objective. They contain 32 conversations from each
of four training-domain families and target 50,000--65,536 useful calibration
tokens. Every optimization submission is ranked only on 64 strictly-ID,
Qwen2.5 validation conversations (16/family). In the default mixed regime that
validation set is sealed; the regenerable full-visible regime exposes those
same 64 validation conversations while keeping the scored set unchanged.
Neither regime ever scores a calibration conversation. Calibration statistics are separately
prepared at 32, 64, and 128 balanced conversations for stability audits.
The preparation tool materializes only one regime at a time in the canonical
task directory, so a full-visible copy can never disclose the mixed regime's
sealed validation prompts during a run.
Online feedback reports the aggregate, storage-point summaries, and
cluster-bootstrap uncertainty, but withholds per-domain/template validation
tracks to limit adaptive overfitting.

The final 128 prompt IDs comprise 64 ID prompts (16/training family) and 64 OOD
prompts (eight from each of eight held-out families). The identical prompts are
answered separately by Qwen2.5 and Qwen3. Sealed test curves are:

- Qwen2.5 + overlapping domains: in-distribution;
- Qwen2.5 + held-out domains: data generalization;
- Qwen3 + overlapping domains: model generalization;
- Qwen3 + held-out domains: joint generalization.

Qwen3 supplies no training response, validation loss, or performance feedback;
its target-model activations are calibrated only from the prompt sides of the
128 Qwen2.5 training prompt IDs, with every assistant output excluded. Test
scoring is deferred and cannot affect selection. Foreground
64-conversation validation evaluations have accelerator priority; accepted-incumbent
test shards run only in idle capacity and are fully drained after optimization.
