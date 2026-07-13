# Paper-native SLM compression diagnostic

This directory defines an **offline diagnostic**, not a ranked optimization
task. It answers one narrow question: when real GPTQ, AWQ, SparseGPT, and Wanda
are applied to the pinned Qwen2.5-0.5B-Instruct model, does the benchmark's
64-conversation validation score recover the broad compression directions in
Zhou, Kurz, and Zhao, *Revisiting Pruning vs Quantization for Small Language
Models* (Findings of EMNLP 2025)? A paired nonthinking Qwen3-0.6B extension
then checks transfer across the closely related GQA architecture; it is a local
transfer curve, not a paper-reported Qwen3 result.

Run `/tmp/text-opt-bm-ml/bin/python research/baselines/slm_paper_native/protocol.py describe` for
the complete machine-readable protocol and storage table. The intended native
cells are:

- GPTQ and AWQ at dense INT8 and dense INT4, group size 128.
- SparseGPT and Wanda at pure unstructured S50 and S75. The pruning cells must
  not add quantization.
- Qwen2.5 cells calibrate only on its 128 training conversations and score only
  its 64 ID validation conversations during development. The 128 calibration
  conversations are quantization/pruning data, never a scored split.
- Qwen3 uses target-model activations on the exact same selected 128 training
  prompt IDs. Its records are prompt-only nonthinking prefills. It receives no
  training loss, validation loss, or performance feedback.
- Operator-final test scoring has four paired 64-row curves: Qwen2.5 ID/OOD
  and Qwen3 ID/OOD. The same prompt IDs across models isolate model transfer
  from prompt difficulty.

## What counts as a reproduction

The existing `awq_style.py` and `wanda_style.py` plans are useful ranked-task
adapters, but they are not entries in this diagnostic. They combine simplified
activation rules with the evaluator-owned quantize/prune transform and, at the
ranked 3.125/4.125-bpw points, pruning is followed by quantization.

A completed native row must record the implementation repository and commit,
any local patch hash, the ordered calibration-prompt hash, exact calibration
tokens, compression and scoring wall times, the dense fake-quant scoring
checkpoint bytes, and (when materialized) native packed artifact bytes/hash. Use
the method's real reconstruction/search procedure:

- GPTQ: approximate-Hessian sequential quantization and error compensation.
- AWQ: architecture-correct activation-aware scale search (and clipping if the
  selected pinned implementation enables it), followed by actual INT4/INT8
  fake-quantized weights for MPS scoring.
- SparseGPT: approximate-Hessian mask selection plus weight update/error
  compensation.
- Wanda: `abs(weight) * sqrt(channel activation energy)` row-wise selection,
  with the official sequential calibration flow.

`protocol.py validate-results` rejects the ranked `awq_style`/`wanda_style`
paths and requires method-specific native-algorithm checks. A result is still
best described as a **method-native local reproduction under benchmark
calibration**, not an exact reproduction of Zhou et al., because that paper
does not pin all tool versions/hyperparameters and uses different corpora.

The scoring checkpoint may hold fake-quantized FP32 weights or dense BF16
zeros so PyTorch can evaluate it on MPS. Those materialized bytes
are **not** its compression rate. Report the native packed representation and
the serialized scoring artifact separately.

## Calibration mismatch with the paper

Zhou et al. use 128 C4 sequences of 2,048 tokens (262,144 tokens) for GPTQ,
SparseGPT, and Wanda, but 128 Pile sequences of 512 tokens (65,536 tokens) for
AWQ. The benchmark intentionally uses its 128 SFT calibration conversations,
targeting 50,000--65,536 useful tokens. Therefore local rows can validate a
direction under the benchmark protocol, not numerically reproduce the paper.
An optional paper-corpus sensitivity run may be reported separately, but it
must never replace the SFT-calibrated marker used for the benchmark.

## Paper metric alignment

Appendix Table 8 is the primary paper comparison. It reports multilingual PPL
for Qwen2.5-0.5B under every native method/setting. `protocol.py` preserves all
seven PPL values and derives
`mean_language(log(PPL_compressed / PPL_full))`. This is exactly a signed delta
NLL/log-perplexity ratio, so its unit aligns with the local assistant-token
score.

The paper and local points still remain in separate panels: Table 8 averages
seven language-modeling corpora equally, while the benchmark macro-averages
assistant-token deltas by task template and domain over 64 SFT conversations.
Their direction and approximate magnitude are comparable; they are not scores
on the same corpus. Appendix Table 7 activation SNR/error remains a useful
secondary fidelity panel and must stay on a separate, unlike-metric axis.

The Appendix Table 8 PDF extraction has an inconsistent header/body ordering.
Table 2 contains the same full-size Qwen row and establishes the canonical
`en, ar, hi, zh, th, de, es` order used by the protocol.

## Storage

The paper explicitly assumes zero sparsity-mask overhead, which is why it
compares S50 with INT8 and S75 with INT4. Both views must be shown:

1. **Paper-logical rate** reproduces that zero-overhead convention.
2. **Canonical packed rate** adds a one-bit dense mask to FP16 pruning and
   adds FP16 group scales (plus packed zero-points for asymmetric AWQ) to
   quantization.
3. **Native packed artifact bytes** measure the actual deployment artifact. They
   can exceed the canonical minimum because of alignment, `g_idx`, tensor
   padding, and file headers. The dense fake-quant checkpoint used for MPS
   scoring is reported separately and never treated as compressed storage.

The paper does not pin GPTQModel/Optimum versions, GPTQ symmetry, activation
ordering, or whether the chosen format materializes a zero-point tensor. The
protocol therefore reports both implicit-symmetric-zero and packed-zero-point
GPTQ minima; a completed result must declare which representation it actually
uses. AWQ uses packed asymmetric zero-points in the referenced implementation.

For this pinned model the transformer-linear eligible set has 357,826,560
weights; 136,206,208 parameters remain FP16. As a result, the embedding-heavy
whole model compresses much less than the nominal transformer weights. The
protocol command gives exact byte counts.

## Runtime boundary

Native compression is cached offline. It must not run inside a one-hour
optimization loop: GPTQ and SparseGPT form and factor a 4,864-square Hessian
for every MLP down-projection, while AWQ performs repeated reconstruction
searches. Once a compressed fake-quant checkpoint exists, scoring the shared
64 validation conversations uses the ordinary PyTorch MPS path. Every local
model command must check `torch.backends.mps.is_available()` before loading a
checkpoint, reject an enabled `PYTORCH_ENABLE_MPS_FALLBACK`, and fail closed
otherwise. The result schema requires both
compression and scoring backends to be exactly `mps`; model inference on CPU,
CUDA, or MLX does not produce an admissible local marker. Model-free schema and
storage unit tests may still run on CPU.

Measured July 11, 2026 on this 32-GB, 10-core M5 MacBook with PyTorch 2.13 and
Transformers 5.2 (representative synthetic lengths, not benchmark scores):

- MPS scoring of 64 rows / 25,536 total tokens / 4,096 assistant targets:
  3.35 seconds for one model state.
- MPS dense activation statistics over 128 rows / 51,072 tokens: 20.13 seconds.
- MPS full-model pass with one 4,864-square input Gram accumulator: 15.41
  seconds; factorization of that Gram took another 0.33 seconds.

Those primitives imply the following conservative planning envelope after a
runner is debugged: 30--90 seconds per Wanda setting, 1--3 minutes per
GPTQ/SparseGPT setting, and 3--10 minutes per AWQ setting because its repeated
reconstruction grid search dominates. Budget 15--40 minutes for all eight MPS
compressions, then under one minute to score all cached states if the dense
reference is reused. First-run compilation, checkpoint serialization, or a
less vectorized port can push the total toward an hour; record actual wall
times. Run only one compressor/scorer at a time on MPS because model weights,
calibration states, and Hessians share unified memory.

### Implementation feasibility

The published repositories are algorithm references, not drop-in portable
runners in this environment:

- Original GPTQ calls CUDA synchronization/cache APIs, has no Qwen driver,
  and fails unmodified on MPS. Its Hessian/column-update math is ordinary
  PyTorch and runs on MPS after a small device abstraction and Qwen sequential
  layer driver.
- Original SparseGPT and the official Wanda runner hard-code CUDA/device maps
  and older Llama-style layer calls. Their core Gram, Cholesky, pruning, and
  error-update operations work on MPS, but the orchestration must be ported.
- MIT `llm-awq` hard-codes `.cuda()` and uses CUDA packed kernels. Archived
  AutoAWQ has an MPS device chooser for calibration search, but native packed
  inference remains CUDA/Triton/IPEX-oriented and its pinned dependency range
  predates Transformers 5.2. Use an isolated, pinned pure-PyTorch search port
  and a dense fake-quant state for MPS scoring. If no MPS-native packer exists,
  leave native artifact bytes null and use the exact canonical accounting.
- A current GPTQModel release advertises Apple CPU/MLX inference paths, but
  those are inadmissible here. It is also not the unversioned
  GPTQModel/Optimum stack used by the paper. Substituting it silently would
  turn version drift into an algorithm comparison.

`qwen_native_runner.py` implements the isolated port. It uses only the small,
audited algorithm cores at pinned upstream commits, replaces CUDA-only
cache/sync calls with MPS helpers, asserts MPS availability, and keeps explicit
Qwen2.5 and Qwen3 sequential-layer adapters. Qwen3 AWQ runs reconstruction
through the real attention module, including `q_norm` and `k_norm`; Q/K
clipping remains disabled as in AutoAWQ. Every cell starts from the
authenticated base checkpoint. Do not reuse a
model already compressed at another setting. Tokenized calibration can be
shared, but method/setting-specific sequential activations and Hessians cannot
be shared when earlier compressed layers change later inputs.

### Staged minimum credible run

1. Smoke-test each port on 32 calibration conversations and exactly one decoder
   layer. Check exact INT grids or exact S50/S75 zero counts, MPS tensor/device
   provenance, and AWQ function preservation. Do not score calibration or
   validation rows and never plot a smoke artifact.
2. Run full-128 GPTQ INT8/INT4, SparseGPT S50/S75, and Wanda S50/S75, then score
   the common 64-row validation set. This six-cell stage tests the strongest
   paper direction (dense quantization versus pure pruning) and the
   second-order-versus-heuristic pruning contrast.
3. Add full-128 AWQ INT8/INT4 only after the Qwen2 scaling map passes
   function-preservation tests before fake quantization. These complete the
   eight-row paper-native matrix.
4. Run the previously required 32/64/128 calibration-size ablation as a
   sensitivity appendix. Keep the full-128 result as the plotted marker.
5. Materialize native packed artifacts only where an MPS-compatible packer
   supports them and record their bytes/hash. Otherwise plot canonical packed bytes and
   explicitly leave native artifact bytes null; never use dense fake-quant file
   size as compression storage.

Inspect runner contracts and create a provenance-complete result skeleton with:

```sh
/tmp/text-opt-bm-ml/bin/python \
  research/baselines/slm_paper_native/qwen_native_runner.py describe
/tmp/text-opt-bm-ml/bin/python \
  research/baselines/slm_paper_native/protocol.py \
  init-results /tmp/slm-paper-native-results.json
```

Once the selected paired corpus exists, one-layer smoke syntax is:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 /tmp/text-opt-bm-ml/bin/python \
  research/baselines/slm_paper_native/qwen_native_runner.py compress \
  --model qwen25 --method gptq_int4 --calibration-size 32 --smoke
```

The runner holds `/tmp/text-opt-bm-slm-mps.lock` for the entire model/method
job. It takes the operator side of the campaign phase lease and fails closed
while an SLM optimization campaign or deferred drain is live. The exact path
and common lock-helper SHA are content-bound in cache/result provenance. It
writes an overlay after every layer and resumes only when the checkpoint,
calibration prompt hash, source commits, and local code hash all match.

### Operator-only scoring export

Plaintext validation/test rows never enter an agent-visible task artifact. The
post-selection compiler writes exactly one ignored file:

`research/slm_sft_data/generated/operator_final_native_score_curves_v1.json`

It uses schema `slm-paper-native-score-export-v1`, role
`operator_final_native_score_curves`, and contains Qwen2.5 validation/ID/OOD
plus paired Qwen3 ID/OOD curves (64 rows each). It contains no calibration rows
or conversation messages. Its provenance binds the compiler, selected-corpus
manifest, active task data manifest, sealed validation/test binaries, full-128
calibration prompt hashes, and ordered prompt/record hashes for every curve.
The runner recomputes those hashes, requires the export's external file SHA-256
on the command line, and rejects the same schema at any path outside the
ignored operator location.

The mixed-profile corpus compiler emits it automatically only after both task
manifests are complete. To reauthenticate and rebuild it manually afterward:

```sh
/tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/export_native_score_curves.py --operator-final
```

The command has no input/output path overrides, reauthenticates all compiled
source bytes, writes atomically with mode `0600`, and prints hashes/counts only.
It performs no model computation.

After real method runners fill all eight rows, validate and emit plot-ready
separate published/local panels with:

```sh
/tmp/text-opt-bm-ml/bin/python research/baselines/slm_paper_native/protocol.py \
  validate-results /tmp/slm-paper-native-results.json
/tmp/text-opt-bm-ml/bin/python research/baselines/slm_paper_native/protocol.py \
  compare /tmp/slm-paper-native-results.json \
  --output /tmp/slm-paper-native-plot.json
```

The primary published panel uses the derived Table 8 delta log-PPL marker. The
local panel uses assistant-token signed NLL delta with prompt-clustered paired
uncertainty. These share a mathematical unit but must identify their different
corpora. Table 7 activation SNR/error remains a separate secondary panel.

The canonical PyTorch 2.13 strict-MPS kernel proof is committed at
`results/mps_kernel_smoke_torch213.json`. With
`PYTORCH_ENABLE_MPS_FALLBACK=0`, GPTQ, AWQ, SparseGPT, and Wanda tensors all
remained on `mps:0`; it took 0.16 seconds of kernel wall time. That proof also
exposed that PyTorch's `cholesky_inverse` is not implemented on MPS. The port
uses the algebraically identical `L^-T L^-1` triangular-solve construction,
which is regression-tested against CPU `cholesky_inverse`. This tiny artifact
is backend evidence only, not a model result.

Qwen3.5 is an exploratory extension only. Zhou et al. do not provide a hybrid
full-/linear-attention comparator, and native AWQ requires a separately tested
architecture-specific scaling map. Do not draw a published-SOTA line on the
Qwen3.5 task from this paper.
