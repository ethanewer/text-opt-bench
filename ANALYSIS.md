# Run analysis (2026-07-02 / 2026-07-03)

All runs: greedy hill-climb loop (`loop/optimize.py`), codex CLI agents,
one focused edit per iteration, harness re-scores and accepts strict
improvements. Raw per-iteration data: for runs since the session layer
landed (2026-07-03), the canonical record is
`runs/<task>/<run>/submissions.jsonl` (hash-chained; hidden splits
sealed); `log.jsonl` holds loop diagnostics. Older runs have `log.jsonl`
only. Regenerate the tables with `python3.12 tools/analyze_runs.py`.

> **Score-scale note (2026-07-03):** the determinism hardening (minimal
> child env, no pyc writes, `eval_lib.preimport` moving module loading
> out of the tracemalloc window) shifted **memory-task** absolute scores
> down by a few KB. Re-measured under the current harness: mem_kv
> baseline 33,779,543 / loop best 1,356,592 (24.9x); mem_index baseline
> 13,958,993 / reference 4,469,742; mem_infer baseline 582,334 / loop
> best 135,971 / reference 58,190. Numbers quoted below are as recorded
> at run time; relative comparisons are unaffected except for older
> mem_index runs noted below. Instruction-count, byte-count, and
> error-rate tasks did not move.
>
> **mem_index metric drift note (2026-07-08):** commit `7ee5737`
> changed mem_index from retained/resident bytes to serving peak bytes,
> closing the compress-then-decompress-per-query loophole. Historical
> `5x-*` and `5xB-*` mem_index sessions were recorded under the old
> resident-only metric and must not be compared directly with current
> serving-peak runs. I rescored all 261 recorded mem_index submissions
> (232 unique program hashes) under the current evaluator. The best
> current score is 1,590,560 from
> `runs/mem_index/E1-r5-gpt-5.5-high/submissions/005.py`; the old
> resident-only 1,183,802 result now scores 8,444,826. Full rescore
> artifact: `runs/mem_index/current_metric_rescore.json`.

## 1. Iterative improvement (not one-shotting)

The code-optimization tasks show sustained multi-iteration progress with
essentially no saturation within 7 iterations:

**compress** (score = compressed bytes, baseline 600,364)

| iter | gpt-5.5/low | gpt-5.5/none | gpt-5.4/low |
|---|---|---|---|
| 1 | 108,998 | 169,278 | 146,755 |
| 2 | 94,471 | 126,384 | 134,883 |
| 3 | 90,053 | 125,447 | 92,592 |
| 4 | 81,981 | 105,779 | 81,082 |
| 5 | 81,973 | 80,510 | 80,841 |
| 6 | 81,771 | 80,080 | 78,701 |
| 7 | 81,195 | 79,640 | 69,031 |

21/21 iterations accepted (every iteration strictly improved).

**tsp_budget** (score = total tour length, baseline 61.565)

| iter | gpt-5.5/low | gpt-5.5/none | gpt-5.4/low |
|---|---|---|---|
| 1 | 54.108 | 55.017 | 54.862 |
| 2 | 53.383 | 54.556 | 53.048 |
| 3 | 53.363 | 54.554 | 52.843 |
| 4 | 53.355 | 53.914 | 52.843 (rej) |
| 5 | 53.340 | 53.893 | 52.816 |
| 6 | 53.316 | 53.658 | 52.688 |
| 7 | 52.894 | 53.225 | 52.656 |

20/21 accepted. Other tasks behaved the same way in shorter runs:
mem_kv 33.78 MB → 1.37 MB (3/3 accepted), ops_connect 7.01 M → 52.5 K
instructions (3/3), mem_infer 583 KB → 137 KB (4/4),
compress_heldout 240 K → 49 K val bytes (6/6).

## 2. Model-strength trends

- **Iteration 1 quality tracks model strength** on both comparison
  tasks: compress 109.0 K (5.5/low) < 146.8 K (5.4/low) < 169.3 K
  (5.5/none); tsp 54.11 < 54.86 < 55.02. First-shot quality is the
  cleanest separation signal in this data.
- **Final scores after 7 iterations converge and cross** (gpt-5.4/low
  actually finished best on compress at 69,031). With a greedy
  single-trajectory loop, one lucky algorithmic idea dominates several
  iterations of tuning, so late-run scores are noisy across single runs.
- Practical guidance: to compare models, use score-at-iteration-k curves
  averaged over repeated runs (different run dirs give independent
  trajectories), not a single final score. Weaker configs also produced
  faster iterations (gpt-5.5/none averaged ~70 s vs ~170 s for 5.5/low
  on compress), so score-vs-wall-clock curves are a second useful lens.

## 3. One-shot resistance: what it took (word_problems)

The GSM8K-style task is where one-shotting had to be engineered away.
gpt-5.5/low + a visible train set is a brutally strong few-shot grammar
inducer — each codex call can also self-evaluate repeatedly against the
train file before returning, so one "iteration" is internally many
attempts:

| generator version | design | gpt-5.5/low val error after iter 1 |
|---|---|---|
| v1 | 10 rigid templates, train 200 | **0.000** (perfect one-shot, all splits) |
| v2 | 16 templates, distractor sentences, number words | 0.030 |
| v3 | compositional event chains, transfers, pronouns, varied question targets | 0.016 |
| v4 | v3 + idiomatic tail (coins, "all but N", "doubles their pile", ratios, week+day durations, "half as many again") + **train shrunk to 100** vs val 250 / test 600 | **0.264**, then 0.256 / 0.252 / 0.224 / 0.220 over 8 iterations |

Both levers mattered: the idiomatic long tail resists a one-pass parser,
and the small train split means many family x phrasing x idiom
combinations never appear in the visible data, so progress beyond ~0.25
must come from generalizing under aggregate validation feedback — which
is exactly the iterative signal the benchmark wants to measure.

Final v4 best-program splits: **train 0.000 / val 0.220 / test 0.192** —
the program fully fits the visible train set while still missing ~20% of
hidden data: a real, measurable generalization gap, which makes this the
right task for the full-vs-blind feedback experiment. (Operational note:
two iterations hit the 900 s codex timeout on this task — the agent runs
long self-test loops against train; use `--codex-timeout 1500` for
word_problems runs.)

## 4. Cheating observed and countered

- **ops_connect, first ever run**: codex read the evaluator, re-derived
  the fixed-seed workload at import time (uncounted), precomputed all
  answers, scored 24 instructions. Preserved as
  `tests/broken/ops_connect_hardcode.py`. Countered by unseen-data
  validation instances in every task (and, in `mem_infer`, by including
  a held-out instance's peak inside the scored maximum). The re-run
  produced an honest union-by-size algorithm.
- **mem_kv**: the winner reverse-engineered the record schema and packs
  each record into 10 bytes, resynthesizing the JSON at lookup. Ruled
  legitimate: it stores the actual input data (validated on an unseen
  dataset) — schema-aware compaction is the point of the task.
- **compress (gpt-5.4)**: embedded the corpus CSV header as a literal.
  Legal in the perfect-information variant; the unseen-corpus ratio
  guard confirms the compressor still generalizes.
- Memory/size metrics are largely self-policing (stored answers count
  against the score); instruction-count metrics are the ones that need
  validation instances.

## 5. The overfitting experiment (full vs train-only feedback)

`compress_heldout`, gpt-5.5/low, 6 iterations per arm, hidden test
scored quietly every iteration:

| arm | selection signal | final train ratio | val ratio | test ratio |
|---|---|---|---|---|
| full feedback | val score | 0.2053 | 0.2043 | 0.2035 |
| train-only (blind) | train score | 0.0448 | 0.0454 | 0.0452 |

The blind arm won on held-out data by 4.5x — not by overfitting (its
three ratios are identical) but because that trajectory discovered
format-aware field-level codecs for the four genres, with dictionaries
mined from the visible train docs. n=1 per arm: this says the harness
measures the right things, not that blind feedback is better in general.

**Caveat found by this experiment**: with synthetic corpora, the train
split reveals the generator's full template structure, so
`compress_heldout`'s train/val gap axis is weak — modeling train ≈
modeling the distribution. `word_problems` v4 fixed the analogous
problem by shrinking train below the distribution's surface diversity;
if a sharper compression overfitting axis is wanted, the corpora need
value/template diversity that 300 KB of train text cannot enumerate.

## 6. Benchmark/optimizer separation + goal-mode validation (2026-07-03)

The benchmark was restructured so that nothing about it requires the
bundled loop: `bench/session.py` records every submission (exact program
bytes, score, wall-clock timeline, hash-chained, hidden splits sealed),
and the loop became one consumer of that API. Two live gpt-5.5/low
validations of the new surface:

- **Loop path** (`runs/ops_connect/20260703-133153-gpt-5.5-low`):
  3/3 iterations accepted, 7.01 M → 52,563 instructions; submission
  record and git attempt history both intact; `bench verify --rescore`
  reproduced all 4 records bit-exactly.
- **Goal-mode path** (no loop, no git — one `codex exec` session pointed
  at a `bench workspace` dir with the instruction "follow GOAL.md"):
  the agent made **9 submissions in a single session** via
  `bench submit`, improving monotonically 33.78 MB → 3.15 MB (10.7x) on
  mem_kv with inter-submission lapses of 20-116 s recorded; all 9
  records re-scored bit-exactly.

Same-session determinism hardening (see README): minimal child env, no
pyc writes, and `eval_lib.preimport` moved module loading out of the
tracemalloc window. This eliminated a previously unexplained ±40-140
byte flicker in memory scores (C-extension init inside the window was
the main culprit — `import zlib` made the mem_kv reference solution
bimodal) and is what makes `verify --rescore`'s bit-exact claim hold.

## 7. ML systems tasks (2026-07-03, from the taskset-expansion review)

Four ML systems tasks were built and adversarially hardened in a
separate working copy; after a 55-agent adversarial review plus live
gpt-5.5/low loop runs, **three were integrated** and one rejected.

Live loop evidence (5 iterations each):

| task | baseline | iter-by-iter | accepted |
|---|---|---|---|
| rl_async_sched | 235,202 | 196,612 → 196,518 → 195,600 → 195,428 → 195,227 | 5/5 |
| inference_batching | 395,023 | 345,038 → 342,099 → 341,373 → 341,366 → 340,262 | 5/5 |
| checkpoint_plan | 372,389 | 151,111 → 149,563 → 148,143 → (no-change) → 147,992 | 4/5 |
| pipeline_partition | 3,504,047 | **3,309,453 on iter 1 = provable exact optimum; frozen after** | 1/5 |

- **pipeline_partition rejected**: its objective reduces to the classic
  contiguous-partition bottleneck DP (n≤48, k≤8). Review verifiers
  solved it exactly offline, and codex-low then hit that exact optimum
  on iteration 1 and could never improve again — the word_problems-v1
  failure mode. A redesign (non-decomposable objective) could revive it.
- The accepted three have the compress/tsp profile: a large first win
  (the canonical heuristic/DP) followed by a genuine multi-iteration
  tail. checkpoint_plan's tail is backed by a known offline optimum of
  141,946 (loop best 147,992, 4% above), with the 5M-instruction budget
  providing real algorithmic friction.
- Review fixes applied at integration: clean protocol failures for
  non-int id lists (was: uncaught TypeError in `sorted()`),
  rl_async_sched spec now describes the actual dispatch semantics
  (dependencies satisfied at *dispatch*, node waits until they finish),
  specs list the exact curated-builtins subset, and the rl/inference
  reference solutions were replaced with honest ones (the shipped ones
  scored *worse* than the obvious first move: 229,311 vs LPT's 196,623
  on rl_async_sched).
- Known texture (accepted): scheduling scores carry irreducible floors
  (total work / nodes), so relative ratios understate the optimization
  signal; inference_batching's traces are overload-heavy, making p95 ≈
  constant. Candidates for future sharpening, not blockers.

## 8. Open items

- Repeat comparison runs (3-5 seeds per config) before claiming model
  rankings; the harness and analyzer already support it.
- `word_problems` full-vs-blind experiment (the sharper axis) is ready
  to run: `--task word_problems --feedback train-only`.
- tsp_budget saturates slowly (2-opt-quality plateau ≈ 52.5-53); if more
  headroom is wanted, raise the instruction budget so Or-opt/candidate
  lists differentiate further.
