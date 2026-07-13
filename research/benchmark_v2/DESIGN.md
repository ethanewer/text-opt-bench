# Active research ML suite design

The active suite contains exactly four tasks: LLM routing, optimizer
generalization, Qwen2.5-to-Qwen3 compression, and Qwen3.5 compression. The
first two are custom CPU benchmarks. The latter two are architecture-specific,
MPS-only post-training-compression benchmarks. They share a data protocol but
remain separate tasks so an agent can specialize for conventional GQA versus
Qwen3.5's hybrid full-attention/linear-attention stack.

## Evidence labels

Results are labeled by what was actually run:

- **paper-reported**: a number copied or transformed from a cited paper, shown
  only on that paper's corpus and aggregation;
- **method-native local reproduction**: the published algorithm itself was
  ported and run locally, with implementation provenance and benchmark data;
- **mechanism adapter**: a local method expresses a paper's core allocation or
  balancing idea but is not the paper implementation or protocol;
- **local baseline**: a recognizable reference implementation run on this
  evaluator;
- **agent result**: an optimization-loop submission selected by the benchmark;
  and
- **evaluator upper bound**: an evaluator-only diagnostic that may use hidden
  realized outcomes and is not an attainable method or research frontier.

Paper-reported and local values are never put on one numerical axis merely
because both are lower-is-better. A local score is not called state of the art
until the relevant method-native comparisons have actually completed.

## Acceptance gates

1. Independent fit/calibration, validation, and sealed test roles, with exact
   and fuzzy leakage audits where prompts are reused or templated.
2. Signed objectives and uncertainty, with family/domain macro aggregation so
   a large subgroup cannot dominate.
3. APIs that can express meaningful algorithmic choices without exposing
   evaluator-owned family, domain, model-performance, or test labels.
4. Recognizable literature methods executed locally, and discrepancies with
   their expected direction reported rather than hidden.
5. Online scoring within roughly two minutes on the target MacBook. Expensive
   native compression is an offline diagnostic; cached compressed states may
   be scored through the ordinary evaluator.
6. CPU-only execution for routing and optimizer synthesis. Every model action
   in either SLM task uses PyTorch MPS with
   `PYTORCH_ENABLE_MPS_FALLBACK=0`; CPU, CUDA, MLX, and fallback-enabled SLM
   measurements are outside the protocol. Generation, compilation and
   activation calibration, paper-native diagnostics, online validation, and
   sealed testing all acquire one shared, non-preemptive cross-process MPS
   lease. Model-free CPU checks may overlap it; a second model job may not.
7. Online selection never executes sealed test data. For all four tasks, each
   accepted incumbent queues low-priority sealed-test work on otherwise idle
   capacity, and every pending job is drained after optimization. Test-only
   crashes become sealed operator failures and cannot reject an incumbent,
   alter later prompts, or affect selection.

## `llm_routing_v2`: cost-aware routing, custom v5

- Source: a pinned LLMRouterBench performance-cost release with twelve models
  and ten datasets.
- Roles: 6,086 fit rows, 1,218 visible scoring rows, 2,455 validation rows,
  and 2,576 sealed test rows (12,335 total).
- Leakage control: dataset-scoped exact/fuzzy template components cannot cross
  roles; an independent similarity audit found no high-similarity cross-role
  pair. Every dataset has at least eight rows in every scored role.
- Candidate: learn from recorded quality/cost outcomes, then choose one model
  for each prompt and each of 21 cost preferences.
- Ranked scalar: dataset-macro normalized regret relative to the realized
  per-prompt utility oracle. Lower is better; the realized oracle is an
  unattainable evaluator upper bound, not a learned frontier.
- Reporting: deterministic dataset-then-prompt bootstrap intervals, paper-style
  accuracy/cost diagnostics, and a fixed-oracle frontier-distance diagnostic.
- Campaign semantics: v5 preserves the v4 data generation and ranked metric,
  while binding prepared artifacts to sessions and deferring the full sealed
  test to accepted-incumbent background CPU work.
- Status: custom/tweaked local benchmark, not a numerical LLMRouterBench or
  Avengers-Pro reproduction. The Avengers-Pro row is a mechanism adapter.

## `optimizer_generalization_v2`: optimizer synthesis, custom v6

- Roles: 80 visible training workloads, 240 validation workloads, and 640
  sealed test workloads.
- Development families: quadratic, multilabel logistic, robust multi-output
  regression, matrix factorization, and multiclass softmax.
- Test-only families: tanh neural regression, Poisson regression, quantile
  regression, pairwise ranking, and learned-Fourier regression. Sealed test
  gives exactly half its macro weight to known families and half to unseen
  families; every family has both ID and OOD tracks.
- Candidate: one stateful optimizer over a natural matrix block and vector
  block. A separate `view` permits Schedule-Free evaluation iterates.
- Ranked scalar: TaskSet-inspired reference-normalized validation-loss curve
  AUC over 17 checkpoints, upper-clipped at one and macro-averaged over
  family/track cells. Better-than-reference progress remains negative.
- Red team: shape, initialization RMS, first-gradient RMS, and horizon supports
  overlap; the former exact structural signature dispatcher has zero advantage
  and the scalar-signature dispatcher stays below the committed exploit gate.
- Campaign semantics: v6 preserves the v5 workload generation and AUC, while
  binding prepared artifacts to sessions and deferring the full sealed test to
  accepted-incumbent background CPU work.
- Status: custom small-workload direction check. Local Adam, Momentum,
  Schedule-Free Algorithm 1, and compact Shampoo are baselines, not paper-table
  reproductions; the compact Shampoo implementation is not Distributed
  Shampoo.

## Shared SFT compression data protocol

The two SLM tasks use model-generated conversations of at most 512 tokens.
Prompts cover four balanced development-domain families and eight held-out
domain families. Before model-specific generation, 640 prompt candidates are
audited; final selection retains:

- 128 calibration conversations, 32 per development family;
- 64 ID validation conversations, 16 per development family;
- 64 new ID test conversations, 16 per development family; and
- 64 OOD test conversations, eight per held-out family.

The 128 training conversations supply activation/quantization calibration only.
They contribute **zero** loss to candidate scoring. During optimization, the
quantized model is scored on the 64 validation conversations only. Mixed mode
seals those 64 rows; full-visible mode exposes the same rows but does not change
which rows are scored. A nested 32/64/128 calibration ablation measures
sensitivity, and the selected 128-row calibration corpus targets 50,000--65,536
useful tokens. If it falls short, a test-disjoint calibration-only supplement
must be used rather than padding or leaking validation/test data.

Only assistant tokens contribute to NLL. The score is compressed NLL minus an
uncompressed reference measured in FP32 on the same MPS runtime, equivalently
`log(PPL_compressed / PPL_full)`. Operation-template clusters, domains,
domain-relation groups, models, and storage points receive equal macro weight.
Clustered paired intervals keep shared prompt IDs paired where applicable.

Candidates are evaluated at two hard eligible-weight caps, 3.125 and 4.125
bits per weight. Packed codes/nonzeros, sparsity bitmaps, group scales, channel
scales, and a self-decoding one-byte policy header per eligible Linear count
toward eligibility. Whole-model storage is reported separately. The two fixed
caps produce retention-versus-storage points; ID/OOD/model-transfer curves are
reported separately rather than collapsed into a falsely precise universal
Pareto frontier.

Online validation has foreground accelerator priority. Test/holdout shards run
only when the MPS queue is otherwise idle, cannot affect the optimization
trajectory, and are drained after the main loop. A deterministic candidate
failure becomes a sealed failed shard rather than aborting unrelated campaigns.

## `slm_compression_v2`: Qwen GQA transfer

- Optimize a globally budgeted quantization/pruning plan on
  Qwen2.5-0.5B-Instruct.
- The evaluator owns RTN/AWQ-style quantization and magnitude/Wanda-style
  pruning; candidate choices allocate bits, group sizes, clipping, sparsity,
  and activation powers by layer role.
- Training and validation feedback comes only from Qwen2.5. Nonthinking
  Qwen3-0.6B receives no training loss, validation loss, or performance
  feedback. Target-model activation calibration is permitted as standard PTQ
  calibration, but it is prompt-only on the 128 Qwen2.5 training prompt IDs:
  Qwen3 consumes no Qwen2.5 or Qwen3 assistant output.
- The same 128 sealed test prompt IDs are scored on both models, producing four
  64-conversation curves: Qwen2.5 ID, Qwen2.5 OOD, Qwen3 ID (model transfer),
  and Qwen3 OOD (joint model-and-data transfer).

## `slm_compression_qwen35`: hybrid-architecture specialization

- Optimize on the text-only, nonthinking Qwen3.5-0.8B model.
- Layer descriptors preserve Qwen3.5 full-attention, linear-attention, and MLP
  roles so the policy can specialize storage and transforms by architecture.
- Online feedback is the same 64-row strictly-ID validation protocol. Final
  reporting has separate 64-row ID and 64-row OOD curves at both hard storage
  points.
- Results from the GQA task do not serve as Qwen3.5 baselines unless their
  algorithm is separately run through the Qwen3.5 evaluator.

## Paper-native SLM diagnostic

The offline diagnostic runs actual GPTQ and AWQ at INT8/INT4 and actual
SparseGPT and Wanda at S50/S75. It is deliberately separate from the ranked
evaluator-owned transform API. A completed row must include source commit,
patch hash, calibration prompt/token hashes, MPS provenance, packed storage,
and compression/scoring runtime.

Zhou, Kurz, and Zhao (Findings of EMNLP 2025) provide the paper reference. Their
Table 8 multilingual perplexities are transformed to an equal-language mean
`log(PPL_compressed / PPL_full)`, which is the same unit as signed delta NLL.
The paper still uses different language-modeling corpora and calibration data,
so its markers and the local SFT scores remain in separate panels. Local
method-native and agent-result panels stay marked pending until their artifacts
exist.

## Reproducibility

`tools/prepare_ml_benchmark.py` pins dataset sources, compact artifacts, model
revisions, and checkpoint/tokenizer hashes. `tools/prepare_slm_sft_benchmark.py`
compiles exactly one mixed or full-visible SLM development regime after the
conversation-quality and selection audits pass. `tools/preflight_ml_benchmark.py`
verifies all four active tasks, strict MPS availability/fallback settings,
split cardinalities, hashes, and optional baseline execution before a campaign.
