# Final harder ML task set

## Review of the retired compression branch

The `feature/ml-systems-benchmark-tasks` branch contains four tasks that overlap
with model or KV-cache compression. All four were later merged, attacked, and
removed from `main` in commit `46759fd`.

| Retired task | What it optimized | Important limitation | Confirmed failure |
| --- | --- | --- | --- |
| `kv_quant` | Encoded bytes + selected-head attention-output MSE + Python instruction count | Four fixed contexts from four heads of a 10M TinyStories model; the candidate emitted attention outputs; MSE was not end-to-end model quality | A content fingerprint selected a one-byte marker and replayed packed public outputs, scoring 1,020x better than the honest reference while behaving honestly on the held-out gate. |
| `kv_fixed_budget` | Attention-output MSE under a 55 KB cache cap | Same fixed slices and answer-emission surface | The same public-score/hidden-gate strategy produced an approximately 225x bypass. |
| `kv_layer_budget` | Evaluator-owned compression followed by attention MSE | Structurally safer, but four public token counts acted as instance identifiers | All five campaign winners branched on token count; later an attacker memorized all scoring and validation plans for a 27.7x bypass. |
| `weight_quant` | Encoded bytes + normalized linear-layer output MSE | Four matrix slices and 20 activation rows per layer, not an end-to-end model score | It resisted replay only because its 9,920-value answer table did not fit the 16 KB source limit. The protection was accidental rather than structural. |

This history rules out restoring the old tasks unchanged. A replacement must:

1. Have the evaluator own model execution and quality scoring. A candidate may
   emit a policy or an artifact in a restricted evaluator-defined format, but
   never logits, attention outputs, or reconstructed layer outputs.
2. Compute the ranked score on sealed data. A public score plus a hidden
   pass/fail gate is insufficient.
3. Use an end-to-end paper metric where one exists. Reconstruction MSE and
   logit KL may be diagnostics only.
4. Hold out data families, model seeds or architectures, and operating budgets,
   rather than merely changing a recognizable sequence length.

## Common evaluation protocol

Paper-reproduction tasks report two scores:

- **Replication score:** the exact released split, budget, and metric, used to
  compare against published methods.
- **Generalization score:** sealed task instances, coordinate permutations,
  later datasets, or held-out families. This score controls benchmark ranking;
  alternatively use the worse normalized score of replication and
  generalization.

This preserves scientific comparability without making a public benchmark
lookup table the optimization target. Candidates receive visible meta-training
data, aggregate feedback on hidden validation, and no test examples or labels.

Cross-platform tasks use deterministic counts--queries, trials, examples,
bytes, steps, or function evaluations--rather than wall time. PyTorch tasks use
one `cpu|mps|cuda` implementation, with CPU as the canonical scorer where it
fits and tolerance-based comparison on accelerators.

## A. Paper benchmarks with direct baseline comparison

### A1. HPO-B transfer black-box hyperparameter optimization

**Candidate API.** Given past evaluations and optional meta-training tasks,
choose the next hyperparameter configuration. The evaluator owns the HPO-B
lookup table and returns the corresponding response.

**Metric.** Average normalized regret over trials and average rank, using the
official HPO-B-v3 transfer and non-transfer protocols.

**Published comparison.** Random search, GP, BOHAMIANN, DNGO, HEBO, RGPE,
ABLR, TST-R, TAF-R, and FSBO results from HPO-B can be reproduced directly.
HPO-B contains 6.4 million evaluations across 176 search spaces and 196
datasets.

**Generalization defense.** Preserve the exact public protocol as the
replication score. Rank on sealed meta-test tasks with opaque identifiers and
random bijections of normalized hyperparameter coordinates. Hold out complete
algorithm/search-space families, not random rows.

**Cost.** Table lookup; comfortably below one second for hundreds of trials.

**Why it is research-grade.** Transfer HPO remains a difficult economic
problem, and improvement is measured against published optimizers rather than
an invented proxy.

Source: [HPO-B](https://arxiv.org/abs/2106.06257).

### A2. LLMRouterBench offline cost-aware routing

**Candidate API.** For a prompt and a pool of model metadata, return a score or
selection for each model and cost preference. The evaluator owns precomputed
correctness, cost, and latency outcomes; it never calls an LLM.

**Metric.** `AvgAcc`, `Gain@best`, `Gap@oracle`, `PerfGain`, `CostSave`, and
`ParetoDist`, exactly as defined by LLMRouterBench. Use `ParetoDist` plus
best-single-relative constraints for the scalar ranking.

**Published comparison.** The released benchmark has more than 400,000
instances from 21 datasets and 33 models and includes ten routing baselines.
The paper reports that several sophisticated routers remain close to simple
baselines and far from the oracle.

**Generalization defense.** Remove dataset names and source IDs; cluster
near-duplicates before splitting; hold out whole prompt sources and candidate
models. The replication score uses the official split. The ranked score uses a
sealed split and a model-transfer split. A compact task artifact contains only
prompt, correctness, token count/cost, and opaque IDs.

**Cost.** The RouterBench proof of concept in this repository evaluates 36,497
prompts, including TF-IDF training, in 37.9 seconds on CPU. Compact data should
be faster.

Source: [LLMRouterBench](https://arxiv.org/abs/2601.07206) and its
[metric definitions](https://aclanthology.org/2026.findings-acl.1881.pdf).

### A3. TaskSet hyperparameter portfolio and multi-fidelity search

**Candidate API.** Sequentially select optimizer/hyperparameter configurations
and optionally allocate additional training steps after observing prefixes of
released training curves. The evaluator performs lookup into the TaskSet curve
archive rather than training networks.

**Metric.** Best normalized validation loss after `k` trials, normalized regret
versus consumed training steps, and area under the regret-versus-budget curve.
Use the exact task groups, curve normalization, and trial budgets from the
paper for the replication score.

**Published comparison.** TaskSet contains over a thousand diverse optimization
tasks and roughly 29 million training curves. The paper compares learned
ordered hyperparameter lists with random search and studies transfer to unseen
tasks.

**Generalization defense.** Split by complete task family and architecture,
not individual curves. Apply opaque task IDs and hyperparameter-coordinate
permutations to the ranked sealed split. Include a sealed family that was not
used to construct the candidate's feedback score.

**Cost.** Array lookup and aggregation; normally below one second.

Source: [TaskSet](https://arxiv.org/abs/2002.11887) and the
[released optimizer archives](https://learned-optimization.readthedocs.io/en/latest/optimizer_baselines.html).

### A4. BBOB/IOHprofiler black-box optimizer design

**Candidate API.** Given a bounded continuous search domain and its own query
history, choose the next batch of points. The evaluator owns shifted and
rotated objective instances.

**Metric.** Expected running time in function evaluations, fixed-budget best
value, and aggregate ECDF over standard target precisions. These are the
standard BBOB/IOH metrics, with no wall-clock component.

**Published comparison.** Run the exact 24 noiseless BBOB functions, standard
dimensions, instance IDs, target values, and budgets to compare with the large
public archive of CMA-ES, differential evolution, Bayesian optimization, and
other solvers.

**Generalization defense.** The ranked component uses held-out instance IDs,
dimensions, and MA-BBOB affine mixtures. The exact BBOB component remains
visible as a replication result. Aggregate across function groups and include
worst-group performance so a solver cannot sacrifice multimodal or
ill-conditioned families.

**Cost.** Pure analytic functions. Thousands of evaluations across a compact
suite should fit comfortably within the scoring budget.

Sources: [BBOB workshop and data archive](https://numbbo.github.io/workshops/),
[IOHprofiler metrics](https://iohprofiler.github.io/Background), and
[MA-BBOB](https://arxiv.org/abs/2306.10627).

## B. Custom or deliberately scaled-down benchmarks

### B1. End-to-end KV-cache policy generalization

This is the valid successor to the retired KV branch tasks.

**Candidate API.** Return a causal cache policy: token priorities, per-layer and
per-head budgets, K/V bit widths, group sizes, or low-rank ranks. The evaluator
applies the policy and owns cache storage, attention, and model execution.

**Metric.** Teacher-forced held-out NLL/perplexity degradation from full cache
at several exact peak-byte budgets. Aggregate mean and worst-corpus degradation
and report the Pareto area over cache ratios.

**Split.** Visible calibration stories; hidden validation from different
documents; sealed test corpus families, context lengths, and model seeds. Do
not expose unique length-to-instance mappings. Score the sealed data directly.

**Baselines.** Recent-only, StreamingLLM/sinks, H2O, SnapKV-style observation,
KIVI-style K/V quantization, and simple hybrids. Published absolute numbers are
not claimed because the model is scaled down, but the methods and PPL metric
are paper-derived.

**Feasibility.** The local proof of concept scores a 192-token, 48-token-cache
run in 2.9 seconds on CPU. Its winning compressed policy changes when context
and budget change, demonstrating useful hidden-instance variation.

Sources: [H2O](https://arxiv.org/abs/2306.14048),
[KIVI](https://arxiv.org/abs/2402.02750), and
[SnapKV](https://arxiv.org/abs/2404.14469).

### B2. Evaluator-owned SLM quantization and pruning

This replaces the retired `weight_quant`; it is not a reconstruction task.

**Candidate API.** Return quantization/pruning parameters and packed weights in
an evaluator-defined format. The candidate has no `infer` function. The
evaluator decodes the artifact, executes the complete 0.5B model, and computes
quality on sealed text.

**Metric.** Held-out NLL/perplexity degradation under exact total bits,
metadata, sparsity, and calibration-operation budgets. Report mean and
worst-language/corpus degradation at INT4/INT8 and 50%/75% sparsity operating
points. Weight MSE, activation MSE, and logit KL are diagnostics only.

**Split.** Public calibration samples; hidden validation documents; sealed
test corpora and languages. Quantization is performed before test inputs are
revealed, eliminating fingerprint-and-replay inference. Use more than one
model seed or architecture in the sealed score.

**Baselines.** GPTQ, AWQ, SparseGPT, Wanda, magnitude pruning, and naive
groupwise quantization. The published 0.5B study supplies method expectations,
but local results should be described as a scaled/subsampled protocol unless
the full paper evaluation is run.

**Feasibility.** Qwen2.5-0.5B with six transformations and 510 tokens per split
scored in 9.8 seconds on MPS. A CPU smoke run took 26.8 seconds and preserved
the method ordering.

Source: [Revisiting Pruning vs Quantization for Small Language Models](https://aclanthology.org/2025.findings-emnlp.645/).

### B3. Generalizing optimizer/update-rule synthesis

**Candidate API.** Implement a bounded-state first-order update rule. The
evaluator owns model parameters, gradients, data, training loops, and all
validation evaluation.

**Metric.** Fixed-step normalized validation-loss curve area, examples or steps
to hidden targets, aggregate performance-profile area, and worst-quartile
workload score. Never use cross-platform wall time.

**Split.** Hold out complete combinations of architecture, loss, conditioning,
gradient noise, sparsity, batch size, and scale. Include tiny real neural
networks alongside analytic quadratics and matrix factorization. Hyperparameters
must be self-tuning or shared across workloads.

**Baselines.** SGD, momentum/Nesterov, RMSprop, Adam/AdamW, Lion, Adafactor, and
one non-diagonal preconditioner where size permits.

**Feasibility.** The proof of concept runs five optimizers over eight workload
instances and 150 steps in 1.3 seconds on CPU. There is ample room to expand to
dozens of sealed workloads under one minute.

Sources: [TaskSet](https://arxiv.org/abs/2002.11887) and
[AlgoPerf](https://mlcommons.org/benchmarks/algorithms/).

### B4. Communication-budgeted gradient compression

**Candidate API.** Compress each worker gradient into an evaluator-accounted
bitstream and update bounded error-feedback state. The evaluator owns workers,
data partitions, aggregation, and training.

**Metric.** Validation loss/accuracy curves versus exact cumulative transmitted
bits, plus steps-to-quality under fixed bit budgets. Include encoder metadata
and residual state in the accounting. Aggregate mean and worst workload.

**Split.** Hidden worker counts, IID and non-IID partitions, gradient sparsity,
noise, model families, and bandwidth schedules. Score actual training quality;
do not score gradient MSE.

**Baselines.** Dense SGD, sign/ternary quantization, random-k, top-k, top-k with
error feedback, DGC-style warmup/momentum correction, and EF21.

**Cost.** Tiny logistic, MLP, CNN, and language-model workloads should fit in
one to two minutes on CPU or PyTorch MPS/CUDA. The score is backend-independent
because it uses bits, steps, and validation quality.

Sources: [Deep Gradient Compression](https://arxiv.org/abs/1712.01887) and
[EF21](https://papers.nips.cc/paper_files/paper/2021/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html).

### B5. OOD, filtered, and streaming ANN index design

**Candidate API.** Build an index and answer nearest-neighbor queries through
evaluator-owned distance and storage primitives.

**Metric.** Recall@10 under exact index-byte, distance-evaluation, and update
budgets. Report the recall frontier and worst track across ordinary, filtered,
OOD-query, and streaming-update workloads. Do not use local QPS in the scalar
score.

**Split.** Hidden query distributions, filters, insert/delete traces, dimensions,
and clustered/adversarial vector families. The evaluator owns ground truth and
charges every serving transient.

**Baselines.** Random candidates, LSH, IVF, product quantization, graph search,
and hybrid coarse-quantizer methods.

**Feasibility.** The NumPy proof of concept scores 12,000 vectors and 160
queries in 0.17 seconds, leaving large headroom for harder instances.

Source: [Big ANN competition results](https://proceedings.neurips.cc/paper_files/paper/2025/file/63092d79154adebd7305dfd498cbff70-Paper-Datasets_and_Benchmarks_Track.pdf).

## Recommended implementation order

1. `hpob_transfer`
2. `llm_router_offline`
3. `taskset_portfolio`
4. `kv_policy_ppl`
5. `optimizer_generalization`
6. `slm_compress_ppl`
7. `grad_compress_train`
8. `bbob_optimizer`
9. `ann_ood_stream`

The first three have the best combination of research comparability, economic
value, very cheap scoring, and structural resistance to answer replay. The KV
and SLM tasks are worth implementing only in their evaluator-owned,
sealed-score form; the branch versions should remain retired regression cases.
