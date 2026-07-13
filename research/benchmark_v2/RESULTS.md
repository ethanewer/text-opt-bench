# Active four-task suite: baseline and literature record

Lower is better throughout. Every number is tagged by evidence type. A
**local baseline** or **mechanism adapter** was executed on this benchmark. A
**paper-reported marker** was taken from a paper and remains on that paper's
corpus. A **method-native local reproduction** must execute the actual research
algorithm locally. No SLM method-native or agent score is reported here until
that run has produced a validated artifact.

Machine-readable values and pending fields live in `literature_results.json`.

## 1. LLM routing — custom v5

The ranked metric is dataset-macro normalized utility regret over 21 cost
preferences. This is a custom/tweaked split over pinned LLMRouterBench realized
outcomes, not a numerical LLMRouterBench or Avengers-Pro reproduction. Exact
and fuzzy template components cannot cross roles; all ten datasets contribute
to every scored role. The 12,335 rows comprise 6,086 fit, 1,218 visible
scoring, 2,455 validation, and 2,576 sealed-test rows.

The v5 campaign wrapper preserves the v4 data generation and ranked metric.
It adds fingerprint binding and defers sealed test to low-priority,
accepted-incumbent background CPU work. The test values below are completed
operator baseline diagnostics, not online optimization feedback; pending work
drains after optimization, and a test-only crash cannot influence selection.

| Evidence | Local method | Validation (95% hierarchical bootstrap CI) | Test (95% hierarchical bootstrap CI) |
|---|---|---:|---:|
| local baseline | Global best-single router | 0.191930 [0.142908, 0.229212] | 0.193270 [0.152065, 0.234706] |
| local baseline | Embedding kNN | 0.176300 [0.131961, 0.208462] | 0.173126 [0.134294, 0.212823] |
| local baseline | Semantic-cluster centroid | **0.171798** [0.129608, 0.202940] | **0.171958** [0.132302, 0.207952] |
| mechanism adapter | Avengers-Pro balance mechanics | 0.182114 [0.135841, 0.219694] | 0.177968 [0.134974, 0.219926] |
| evaluator upper bound | Realized-sample utility oracle | 0.000000 | 0.000000 |

All prompt-aware point estimates beat the global baseline on both splits. Only
kNN's paired test improvement over global has a 95% interval strictly above
zero. The centroid interval misses zero by less than 0.0001 on the paired test
comparison, so it is still reported as a statistical tie. The balance adapter
uses released Avengers-Pro mechanics but differs in model pool, split,
duplicate handling, and embedding. The zero-regret oracle observes each
realized outcome and is unattainable by a learned router; it is not a frontier
claim.

The paper-style diagnostics are deliberately separate from ranked regret:

| Evidence | Method/corpus | Performance gain | Cost save |
|---|---|---:|---:|
| paper-reported | Avengers-Pro in LLMRouterBench Figure 7 | +4.00% | +31.70% |
| mechanism adapter | Local custom-v5 validation | +2.45% | -4.88% |
| mechanism adapter | Local custom-v5 sealed test | +5.33% | -7.05% |

The local adapter recovers an accuracy gain but not the paper's cost saving:
at its compared operating point it costs more than the best single model. The
different model pool, split, duplicate treatment, and embedding prevent a
numerical reproduction claim.

Source context: [LLMRouterBench](https://aclanthology.org/2026.findings-acl.1881/).

## 2. Optimizer generalization — custom v6

The scalar is TaskSet-inspired reference-normalized validation-loss curve AUC,
macro-averaged by family and ID/OOD track. Hyperparameters are selected on the
80 training workloads only. Validation has 240 workloads from five development
families; sealed test has 640 workloads and gives equal weight to those five
families and five unseen families.

The v6 campaign wrapper preserves the v5 workload generation and AUC. It adds
fingerprint binding and defers sealed test to low-priority,
accepted-incumbent background CPU work. The test values below are completed
operator baseline diagnostics, not online optimization feedback; pending work
drains after optimization, and a test-only crash cannot influence selection.

| Evidence | Local method | Validation (95% CI) | Sealed test (95% CI) | Test known | Test unseen |
|---|---|---:|---:|---:|---:|
| local baseline | Momentum | 0.571660 [0.546807, 0.596514] | 0.544043 [0.528456, 0.559629] | 0.547906 | 0.540179 |
| local baseline | Adam | 0.492964 [0.467688, 0.518240] | **0.502814** [0.489310, 0.516318] | **0.482368** | **0.523259** |
| method-native local reproduction | Schedule-Free Adam, Algorithm 1 | **0.474195** [0.450329, 0.498061] | 0.507620 [0.494097, 0.521143] | 0.483167 | 0.532073 |
| mechanism adapter | Compact Shampoo with Adam grafting | 0.480319 [0.456477, 0.504162] | 0.507529 [0.494183, 0.520874] | 0.490408 | 0.524649 |

Schedule-Free improves over Adam on validation by -0.018768, with paired 95%
CI [-0.029738, -0.007798]. That advantage does not transfer to sealed test:
Schedule-Free minus Adam is +0.004806, CI [-0.000511, 0.010123]. Shampoo minus
Adam is also unresolved on both validation and test. Adam therefore has the
best sealed-test point estimate, while the established methods are a
statistical group on test. This is useful generalization evidence, not a claim
that Adam is globally better than Schedule-Free or Shampoo.

The exact Algorithm 1 implementation supports a Schedule-Free literature
direction check. The Shampoo row is a compact legal structured preconditioner,
not Meta's Distributed Shampoo and not an AlgoPerf reproduction. Primary
method references: [Adam](https://arxiv.org/abs/1412.6980),
[Schedule-Free](https://arxiv.org/abs/2405.15682),
[Shampoo](https://proceedings.mlr.press/v80/gupta18a.html), and
[TaskSet](https://arxiv.org/abs/2002.11887).

## 3. Qwen2.5-to-Qwen3 SLM compression

The online objective scores only 64 ID Qwen2.5 validation conversations. The
128 training conversations are calibration data, target 50,000--65,536 useful
tokens, and are never scored. Mixed and full-visible materializations use the
same 192-row development pool: mixed exposes 128 calibration inputs and seals
the 64 validation inputs, while full-visible exposes all 192 inputs. Both rank
on exactly the same 64 validation conversations. Final testing will produce
four 64-conversation curves at each hard 3.125/4.125 eligible-bit cap: Qwen2.5
ID, Qwen2.5 OOD, nonthinking Qwen3 ID, and nonthinking Qwen3 OOD. Qwen3
receives no validation-loss or performance feedback, and its target activation
calibration is prompt-only on the 128 Qwen2.5 training prompts, with no
assistant output consumed.

All model-bearing SLM work shares one non-preemptive cross-process MPS lease;
CPU/CUDA/MLX and fallback-enabled model results are inadmissible. The lease
covers response generation, compilation and activation calibration,
paper-native diagnostics, online validation, and sealed testing.

| Local evidence panel | Status |
|---|---|
| Ranked-task RTN/AWQ-style and magnitude/Wanda-style mechanism adapters | **Pending validated MPS scores** |
| Method-native GPTQ, AWQ, SparseGPT, and Wanda on Qwen2.5 validation | **Pending** |
| Paired Qwen2.5/Qwen3 ID/OOD method-native transfer curves | **Pending** |
| Agent optimization results | **Pending benchmark campaign** |

No local number is inferred from a smoke test, an earlier corpus, or the paper
markers below.

### Paper-reported, axis-aligned reference

Zhou, Kurz, and Zhao report Qwen2.5-0.5B multilingual perplexities in Appendix
Table 8 of *Revisiting Pruning vs Quantization for Small Language Models*. The
table below transforms the seven language values to
`mean_language(log(PPL_compressed / PPL_full))`. This is the same mathematical
unit as signed delta NLL, but it is **not the same evaluation**: their corpora,
targets, calibration data, and aggregation differ from the 64-conversation SFT
protocol.

| Evidence | Paper method/setting | Paper Table 8 mean log-PPL ratio |
|---|---|---:|
| paper-reported | AWQ INT8 | **-0.032556** |
| paper-reported | GPTQ INT8 | -0.032230 |
| paper-reported | AWQ INT4 | 0.186753 |
| paper-reported | GPTQ INT4 | 0.204029 |
| paper-reported | SparseGPT S50 | 0.813780 |
| paper-reported | Wanda S50 | 0.849252 |
| paper-reported | SparseGPT S75 | 4.453510 |
| paper-reported | Wanda S75 | 5.511682 |

These are reference markers, not red frontier thresholds. They establish
recognizable method orderings to test locally. The paper's zero-mask-overhead
storage convention is also reported separately from deployable packed storage
in the native protocol. Source: [Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.645/).

## 4. Qwen3.5 hybrid SLM compression

This task independently specializes the compression policy for text-only,
nonthinking Qwen3.5-0.8B while retaining full-attention, linear-attention, and
MLP layer roles. As above, 128 conversations are calibration-only, target
50,000--65,536 useful tokens, and the online score comes from 64 ID validation
conversations. Mixed exposes only the 128 calibration inputs while sealing the
64 validation inputs; full-visible exposes all 192 development inputs. The
scored validation rows are identical in both profiles, and no calibration row
is scored. Final MPS reporting has separate 64-row ID and 64-row OOD curves at
both hard storage points; an aggregate may be diagnostic but is not a
replacement for either curve.

| Local evidence panel | Status |
|---|---|
| Ranked-task mechanism adapters | **Pending validated MPS scores** |
| Qwen3.5 ID/OOD curves | **Pending** |
| Agent optimization results | **Pending benchmark campaign** |

The Zhou Qwen2.5 markers are not Qwen3.5 baselines. Any method shown for this
task must be separately executed on the hybrid architecture. There is
therefore no paper frontier line or local best-method claim yet.

## Interpretation

The CPU tasks are ready for quantitative comparison, and their diagnostics
show why validation alone is insufficient: routing improvements are mostly
uncertain, while Schedule-Free's optimizer validation gain disappears on
unseen-family test. The SLM protocols now have paper-aligned units, robust
calibration/validation/test roles, and explicit architecture transfer, but
their local method-native and agent panels intentionally remain pending. A
research claim should begin only after those MPS artifacts reproduce—or
credibly explain a departure from—the expected paper method directions.
