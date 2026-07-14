# Model discrimination experiment: gpt-5.5 low vs gpt-5.5 none

**Question:** does text-opt-bm discriminate a stronger optimizer from a weaker one?

**Setup:** same model `gpt-5.5`, only reasoning effort differs — **strong = `low`**,
**weak = `none`** (non-reasoning). 8 tasks spanning all families/tiers, **5 independent
runs per model per task**, identical 1-hour box / 40-iteration cap / deterministic
tasks. Only the model changes. Strong arm prefix `CMPL-`, weak arm `CMPN-`.

**Metric:** probability of superiority **POS = P(strong run beats weak run)** over all
5×5 run pairs (direction=min, lower score better). POS 1.0 = strong wins every pairing
(perfect discrimination); 0.5 = indistinguishable. Reported at iteration 1 (first-attempt
quality), at equal iteration budgets (K=5, K=10 — removes the wall-clock confound), and
at final best (full 1-hour wall-clock).

## Result: the benchmark discriminates clearly

| task | strong best (mean) | weak best (mean) | POS iter-1 | POS @5 iters | POS @10 iters | POS final (1h) |
|---|---|---|---|---|---|---|
| compress | 7.7e4 | 9.93e4 | 0.96 | 1.00 | 1.00 | **1.00** |
| compress_heldout | 1.43e4 | 5.95e4 | 0.84 | 0.92 | 1.00 | **1.00** |
| easy_word_problems | 0.177 | 0.249 | 0.98 | 1.00 | 1.00 | **1.00** |
| normalize | 0.049 | 0.114 | 1.00 | 1.00 | 1.00 | **1.00** |
| tag_seq | 0.212 | 0.245 | 1.00 | 0.92 | 1.00 | **1.00** |
| mem_intset | 1.09e5 | 1.87e5 | 1.00 | 1.00 | 1.00 | **0.96** |
| ops_connect | 5.34e4 | 6.09e4 | 0.76 | 1.00 | 1.00 | **0.96** |
| rule_list | 0.339 | 0.351 | 0.84 | 0.72 | 0.68 | **0.52** |
| **AGGREGATE (mean POS)** | | | **0.92** | **0.945** | **0.960** | **0.93** |

- **7 of 8 tasks discriminate strongly on final best** (POS 0.96–1.00). All 8 discriminate
  at iteration 1 (POS 0.76–1.00) and at equal iteration budgets (POS ≥0.68).
- **Aggregate POS ≈ 0.93** — the stronger model beats the weaker one ~93% of the time.
- The separation is large in magnitude too: e.g. compress_heldout strong ~1.4e4 vs weak
  ~6e4 (weak can't find a generalizing codec), mem_intset ~1.1e5 vs ~1.9e5, normalize
  0.049 vs 0.114, easy_word_problems 0.177 vs 0.249.

## The one exception, and what it teaches

**rule_list** ties on final best (POS 0.52) but the strong model is ahead per-iteration
(POS 0.68–0.84 at fixed budgets). Cause: `none` iterations have no reasoning overhead, so
the weak model ran **~4× more iterations** in the hour (18–40 vs 5–10). rule_list is a
slow-converging task (a long idiosyncratic exception tail; both models plateau far above
the 0.08 oracle floor), so the weak model's iteration-count advantage closes the final gap
at a fixed *wall-clock* budget — but not at a fixed *iteration* budget.

**Takeaways for benchmark use:**
- The benchmark robustly separates model strength; **first-attempt quality (iter-1) and
  fixed-iteration comparisons are the cleanest discriminators** (they isolate per-step
  reasoning quality from raw speed).
- At a fixed wall-clock budget, a faster non-reasoning model can partially compensate with
  more iterations on slow-converging tasks — so for pure capability ranking, compare at a
  fixed iteration/token budget, not fixed wall-clock.
- Every family discriminates; the convergent tasks (ops_connect) and the hardest task
  (rule_list) discriminate least on final, as expected.

Data: `runs/{task}/CMPL-*` (strong) and `runs/{task}/CMPN-*` (weak); analysis script
`compare_models.py`.
