# Task: Qwen3.5 hybrid SFT compression policy

Design a globally budgeted post-training weight-compression plan specialized
for the text-only, nonthinking Qwen3.5-0.8B language model. Unlike the Qwen GQA
task, this task exposes distinct semantic roles for Qwen3.5 full-attention,
linear-attention, and MLP projections, allowing the plan to allocate bits and
sparsity differently across its hybrid layer types.

```python
def plan(layers, target_bits):
    return [layer_policy for layer in layers]
```

Each immutable layer descriptor is:

```text
(layer_role, normalized_depth, rows, columns,
 weight_mean_abs, weight_max_abs, activation_rms, activation_max_abs)
```

Return one policy per descriptor:

```text
[bits, group_size, clip, prune_fraction,
 prune_activation_power, quant_activation_power]
```

Bits are integer 2--8; group size is 16, 32, 64, or 128; clipping is
0.5--1.2; pruning is 0--0.75; and the two activation powers are 0--1. The
evaluator performs FP32 RTN/AWQ-style quantization and magnitude/Wanda-style
pruning and then casts weights back to their model dtype. Every active
per-input-channel multiplier and every row/group scale is first rounded to
FP16 and is charged to packed storage; the benchmark assumes no free scale
fusion. The API optimizes
these owned transform families rather than claiming arbitrary GPTQ or
SparseGPT support.

The hard online operating points are 3.125 and 4.125 eligible bits/weight.
Packed values, sparse bitmaps, FP16 row/group scales, and FP16 channel
multipliers count toward each cap, as does one self-decoding byte per eligible
Linear for the bitwidth, group size, and sparse/channel-section flags. Whole
language-model storage is also reported.

The model itself generated every SFT assistant response in nonthinking,
text-only mode. Conversations span diverse domains, contain at most 512 model
tokens, and are scored only on assistant targets. The signed objective is
compressed minus uncompressed assistant-token NLL. Generation, activation
calibration, the candidate's quantization/pruning transform, and both scoring
passes use Apple MPS with PyTorch CPU fallback disabled. Build-time reference
losses are audit diagnostics only; runtime references and compressed FP32
inference share canonical MPS. CPU, CUDA, MLX, and fallback-enabled results are
inadmissible. Every model-bearing SLM process—including generation, compilation
and activation calibration, paper-native diagnostics, online validation, and
sealed testing—must hold the suite's single non-preemptive cross-process MPS
lease for its entire model-work interval. Final curves separately report
overlapping-domain in-distribution retention and held-out-domain data
generalization. Any equal-weight aggregate is an optional diagnostic and never
replaces either generalization curve.
Scoring macro-averages conversations through operation templates and domains,
so repeated variants of one operation cannot dominate a family.
Each 64-conversation scoring cell contains at least 512 assistant target tokens,
with exact counts pinned in the data manifest.
Confidence intervals use a domain-stratified operation-template cluster
bootstrap, so closely related prompt variants are not treated as independent.
The report gives a deterministic clustered interval at each storage point and
for each model/domain-group curve, using paired cluster draws across budgets.

The 128 training conversations (32 per training-domain family) are
calibration-only and target 50,000--65,536 useful tokens. They never contribute
to the optimization loss. Submissions are ranked solely on 64 ID validation
conversations (16/family), sealed in the mixed regime and exposed only in the
regenerable full-visible regime. Those are the same 64 validation rows in both
regimes; calibration rows are never scored. A balanced 32/64/128
calibration-size audit is prepared separately. Final testing uses 64 new ID and
64 OOD conversations; the OOD half contains eight examples from each of eight
held-out families.
Preparation materializes only one development regime at a time in the
canonical task directory, preventing a full-visible copy from disclosing the
mixed regime's sealed validation prompts during optimization.
Online feedback reports aggregate and storage-point results with clustered
uncertainty, not per-domain/template validation tracks, to limit adaptive
overfitting.

Sealed test shards run at background accelerator priority and cannot influence
the optimization trajectory. All pending accepted-incumbent tests are drained
after the main loop completes.
