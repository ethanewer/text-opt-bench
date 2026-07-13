# Task: generalizable cost-aware LLM routing v6 (custom/tweaked)

Learn a router from precomputed LLMRouterBench outcomes. This is a
custom/tweaked local generalization benchmark, not a direct numerical
reproduction of LLMRouterBench or Avengers-Pro: the pinned model intersection,
template-disjoint split, duplicate aggregation, embedding, and ranked utility
regret differ. No LLM is called during scoring. The evaluator owns every
validation/test quality, cost, and dataset label and executes the selected
model choice.

Version 6 retains the ranked metric and version-5 deferred-test semantics, but
repairs the economic data. Every retained model outcome now has a strictly
positive observed or token-reconstructed inference cost. Unrecoverable
zero-cost rows are removed before forming the all-model prompt intersection,
and Qwen3-235B-Thinking is excluded because its release data has 1,251 such
rows. The split manifest records the price table, reconstruction counts,
exclusions, and complete-cost model pool.

Implement:

```python
def fit(training_rows): ...
def route(prompt, embedding, model_stats, cost_preference, state): ...
```

Each fit row is `[prompt, embedding, quality, cost]`. `quality` and `cost` are
lists for a stably permuted model pool. `embedding` is a pinned local embedding
of the prompt. `model_stats[i]` is
`(mean_fit_quality, relative_mean_fit_cost)`. `route` must return a plain
integer model index. The runtime API passes indices rather than model names, so
portable solutions should use the supplied arrays rather than fixed external
assumptions. The preparation manifest retains the name permutation for audits;
this is interface design, not a secrecy boundary.

The evaluator calls `route` at 21 cost preferences: zero plus a quarter-decade
grid from `0.0001` through `5.6234`. For preference `p`, utility is

```text
quality - p * cost / median_positive_fit_cost
```

## Ranked metric

Lower is better. For every dataset and cost preference, the evaluator sums the
candidate's utility gap from the per-prompt oracle and divides by the summed
gap between the oracle and worst available model. This cell regret is bounded
between zero (oracle) and one (always selecting the worst model). Preferences
are averaged within each dataset, then datasets are macro-averaged so a large
source cannot dominate the score.

Every dataset contributes at least eight prompts to each scored split.
Validation and sealed test report a deterministic two-level percentile
bootstrap interval: datasets are resampled first and prompts are then resampled
within each selected dataset (256 replicates, pinned seed). A secondary
dataset-level Student-t interval is retained for continuity and is explicitly
labeled as not resampling prompts. Dataset identifiers are used only inside the
evaluator for macro aggregation and are never passed to `fit` or `route`.

The evaluator also reports paper-native diagnostics, plus one clearly labeled
custom frontier diagnostic, without mixing them into the ranked scalar:

- dataset-macro `AvgAcc`, `Gain@B`, and `Gap@O`;
- `PerfGain` and `CostSave` relative to the best single model;
- the complete accuracy/cost curve and its non-dominated points;
- a custom normalized L1 distance to a fixed, candidate-independent
  realized-sample upper-bound frontier (`oracle_frontier_distance`).

The first five diagnostics use LLMRouterBench's paper definitions. The fixed
oracle frontier distance is deliberately custom: it makes cross-run distances
comparable, but it is an upper-bound diagnostic rather than the paper's
empirical ParetoDist over a changing collection of submitted methods. Every
"oracle" value in this task means the realized-sample upper bound obtained by
selecting among the pinned models after observing their recorded outcomes. It
is evaluator-only and unattainable by a learned router; it is not a population
frontier or a claim about the best possible routing algorithm.

Training rows, visible scoring rows, validation rows, and sealed test rows are
disjoint. Before splitting, exact and fuzzy character-five-shingle template
matches are joined into deterministic, dataset-scoped connected components.
Candidate edges use pinned MinHash banding and are verified by exact
Jaccard/containment thresholds. SWE-Bench compares only the `<issue>` section,
so its shared partial-code-base wrapper cannot collapse unrelated issues into a
single component. The generated split manifest records thresholds, component
counts, role counts, deterministic rebalancing, and zero-cross-role leakage
assertions. Every scored split is macro-aggregated by evaluator-owned dataset
IDs.

During optimization, candidate validity and the visible scalar are computed
from training/validation only. The sealed test is never executed by an online
submission. Accepted incumbents receive one separately queued, low-priority
full-test evaluation whose result is attached only as a sealed operator
artifact and cannot affect acceptance or later prompts. Background test work
runs only in otherwise idle CPU evaluation capacity and is fully drained after
optimization. A test-only crash produces a sealed failed artifact; it cannot
reject the incumbent or influence the optimization trajectory.

The program is limited to 32 KB and must be deterministic, import-free Python
using the safe builtins listed in the repository README. It may learn from the
supplied fit outcomes but must not read files or emit scored outcomes.
