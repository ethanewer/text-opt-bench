# text-opt-bm — official coverage results

## Current seven-task Experiment 1 (2026-07-13)

The generated blogpost uses only complete N=5 model/task series. Its current
run-set mapping is:

- gpt-5.6-sol high: `n5-main-56sol-20260713-*` for five retained tasks, plus
  `v7v9-20260713-*` in the immutable `llm_routing_v2` and
  `optimizer_generalization_v2` recording directories. All seven series are
  complete.
- gpt-5.5 high: `E1-*` for `mem_index`, `tag_seq`, and repaired/rescored
  `compress_heldout`; `v9-35-gpt55-20260713-*` for routing and optimizer
  generalization; `n5-main-55-20260713-*` for revised `mem_infer` and revised
  behavioral LFM2.5 compression. The blogpost omits any latter line until all
  five trials in that task series finish, and omits the gpt-5.5 seven-task mean
  until all seven series are complete.

The `_v2` suffixes above are recording-path compatibility only; the public
task names are `llm_routing` and `optimizer_generalization`.

## Historical coverage experiments

**Canonical setup (identical for every run, comparable + reproducible):** gpt-5.5, 1h box (3600s), 40-iter cap, codex-timeout 1200, 5 runs.

Reproduce with `tools/run_official_coverage.sh` (config -> prefix: low <- 5xE-+GENF2-, none <- CMPN-+cov-none-, lowvv <- cov-lowvv-; `tools/make_exposed_variants.py` builds the val-exposed variants).

## Coverage (N runs) and best score per task

| task | N none | N low | N low(val-vis) | best none | best low | best low(val-vis) |
|---|---|---|---|---|---|---|
| checkpoint_plan | 5 | 5 | — | 1.542e+05 | 1.423e+05 | — |
| compress | 5 | 5 | — | 9.609e+04 | 6.624e+04 | — |
| compress_heldout | 5 | 5 | 5 | 4.245e+04 | 9708 | 9594 |
| mem_index | 5 | 5 | — | 2.923e+06 | 1.828e+06 | — |
| mem_infer | 5 | 5 | — | 1.257e+04 | 1.284e+04 | — |
| mem_intset | 5 | 5 | — | 1.296e+05 | 9.404e+04 | — |
| mem_kv | 5 | 5 | — | 1.566e+07 | 1.362e+06 | — |
| mem_str | 5 | 5 | — | 2.107e+05 | 1.891e+05 | — |
| normalize | 5 | 5 | 5 | 0.1 | 0.04 | 0 |
| ops_connect | 5 | 5 | — | 5.479e+04 | 5.054e+04 | — |
| rule_list | 5 | 5 | 5 | 0.3233 | 0.2783 | 0 |
| tag_seq | 5 | 5 | 5 | 0.2346 | 0.0861 | 0 |
| easy_word_problems | 5 | 5 | 5 | 0.224 | 0.12 | 0 |

## Overfitting arm — hidden-val (low) vs exposed-val (low, val visible)

Same val instances, both evaluated on the always-hidden test split. Exposing the eval data lets the optimizer drive val→~0 while test does not follow.

| task | hidden val | hidden test | exposed val | exposed test |
|---|---|---|---|---|
| easy_word_problems | 0.1624 | 0.151 | 0 | 0.05333 |
| compress_heldout | 1.528e+04 | 1.527e+04 | 1.526e+04 | 2.08e+04 |
| normalize | 0.05 | 0.1009 | 0.01 | 0.1014 |
| rule_list | 0.312 | 0.3643 | 0 | 0.3968 |
| tag_seq | 0.1672 | 0.1751 | 0 | 0.174 |
