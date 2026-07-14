# Authoring a text-opt-bm task

A good task is **deterministic**, has **real headroom** an optimizer can climb
over many steps, and is **robust** — the winning program must be a general
algorithm, not one that memorizes/overfits/games the fixed scoring inputs.
This guide encodes the patterns and traps found while building and auditing
the current tasks. Read an existing task in the same family before writing a
new one; `mem_kv` (memory), `ops_connect` (instructions), `compress_heldout`
(generalization), and `kv_layer_budget` (policy) are the reference shapes.

## Files

```
bench/tasks/<name>/evaluate.py        # the scorer (see families below)
bench/tasks/<name>/config.json        # {name, metric, direction:"min", timeout_s, cpu_s, ...}
bench/tasks/<name>/spec.md            # what the agent sees: API, scoring, rules
bench/tasks/<name>/initial_program.py # a correct but weak baseline (the start point)
tests/solutions/<name>.py             # a strong reference solution (defines headroom)
tests/broken/<name>_*.py              # programs that MUST be rejected
```
Wire the task into `tests/run_checks.py` (headroom + broken-rejection rows) and
the README task table. `bench determinism` auto-discovers it.

## The scoring metric must reward the algorithm, not C-builtin luck

`sys.monitoring` counts **Python bytecode instructions in candidate frames** —
work done inside a C builtin (`sum`, `min`, `sorted`, `bytes.translate`, dict
ops, `itertools`) is ~1 instruction. Consequences:

- **Instruction-count tasks need a Python-BOUND naive.** A range-sum task
  failed review because the naive `sum(a[lo:hi])` is already ~1 instruction/
  query (C), so prefix sums (Python-heavy build) scored *worse*. `ops_connect`
  works because its naive union-find is a Python `while` loop. Rule of thumb:
  if the obvious naive can be one C call, there is no headroom.
- **Memory tasks** measure `tracemalloc` bytes — this rewards compact
  representations regardless of C usage, so they are the most reliable family
  to design (big headroom, robust). But they trend "moderate" (all runs
  converge to a similar structure); depth/variance ("strong") comes from
  genuine algorithmic openness (a transformer impl, a codec, a solver).
  - **Score the SERVING PEAK, not retained-only.** A store+query memory task
    must sample `get_traced_memory()[1]` (peak) AFTER serving the full query
    workload, with `tracemalloc.reset_peak()` called right after `build()`
    (so build transients are excluded but per-query working set is charged).
    Scoring `current` (retained after build) alone is gameable: a candidate
    keeps a tiny lzma-compressed blob (low retained) and decompresses a big
    block on EVERY query — the transient is freed before the sample, so an
    ~8 MB-per-query decode "beat" a compact 600 KB structure. An adversarial
    workflow found this on mem_intset/str (retained metric); charging
    the serving peak makes those cheats score 10–14× WORSE than the honest
    reference while barely changing honest solutions (peak ≈ retained ×
    1.0–1.7, since an honest query materializes only its small result). The
    serving peak is also the more meaningful metric — it rewards structures
    cheap to BOTH hold and query. Regression fixtures:
    `tests/broken/mem_{intset,str}_compress_cheat.py` must score >2× the ref.

## Robustness: never let the winner memorize the fixed inputs

Every scoring run uses **fixed, deterministic** inputs. Left unchecked, the
optimizer will use `len(input)` / a size / a seed-derived value as a per-
instance **id** and hardcode constants for each — inflating the score without a
general algorithm. This was confirmed in the campaign: `inference_batching`
and `rl_async_sched` branched on `len(...)` with per-trace magic constants;
`kv_layer_budget` branched on `n_tokens`; `checkpoint_plan` tuned search effort
to land just under the budget on the exact fixed instances (and failed on fresh
draws). Defenses, by task shape:

- **Return-a-container tasks (build+query, e.g. mem_*):** serve the FULL query
  workload INSIDE the measurement window, so a `build()` that returns a marker
  and defers real construction (or regenerate-and-cache) to the first query is
  still measured. Add **unseen-data validation**: rebuild+query on a
  DIFFERENT-seed dataset after measurement and require exact answers — this
  catches "regenerate the known dataset from its seed" and hardcoded tables.
- **Emit-a-list tasks (e.g. ops_connect):** `eval_lib.require_int_list` the
  return INSIDE the counted window (rejects generators / lazy subclasses that
  defer work) and run unseen-data validation (different seed AND size).
- **Emit-a-policy tasks (e.g. kv_layer_budget):** the candidate returns a
  decision (budgets), the evaluator owns the reconstruction/scoring. Validate
  on instances with UNSEEN shape parameters (e.g. token counts truncated to
  values not in the scoring set) with a fidelity/feasibility gate — a policy
  that memorizes the fixed shapes misroutes and fails.
- **Emit-the-answer tasks (candidate produces the scored output):** the hard
  case, and the one that has repeatedly fooled fixes here. Two traps, both
  learned the hard way on the kv-family (kv_quant/kv_fixed_budget/
  kv_layer_budget/weight_quant, all reverted):
  - A held-out **gate is NOT enough.** If the SCORE is computed on the
    public, recognizable scoring set and held-out is only pass/fail, a program
    branches: content-fingerprint the fixed scoring inputs → return a marker +
    replay memorized/packed answers (they compress far smaller than the raw
    weights, so they fit the source cap even when the weights don't); behave
    honestly on the unrecognized held-out to pass the gate. Verified bypasses
    reached 1020×. The SCORE itself must be computed on data the candidate
    cannot recognize or precompute — score on the sealed held-out, or take the
    WORSE of scoring-vs-held-out.
  - The held-out must be **UNRECOGNIZABLE and un-precomputable**, not merely
    "unseen at authoring time." A held-out that is a deterministic prefix/
    subset of the readable scoring data, or whose seed/lengths are a plaintext
    literal, can be reconstructed or memorized. Draw it from a seed sealed in a
    `bench/heldout.py`-encoded file the candidate can't read, AND perturb/mix it
    so content-fingerprinting fails (weight_quant's sealed-random input mixtures
    resisted attack; the kv tasks' plain truncations did not).
  Do not re-add such a task until an adversarial workflow (agents actively
  writing bypass programs, not just reading the evaluator) fails to beat it.

General rules: a small fixed instance set is fine ONLY with genuine unseen
validation; prefer several instances with non-unique sizes; forbid metric-
control modules (`tracemalloc`, `sys`, `resource`) and the escape surfaces the
shared `broken/` tests probe.

## The robustness boundary (what a fixed-instance benchmark can and cannot resist)

An adversarial attack workflow (2026-07-05) confirmed the emit-answer lesson is
GENERAL, not specific to the kv-family. It bypassed FOUR non-memory tasks whose
score is computed on a fixed instance the candidate emits an answer for:
- `ops_connect` (12x): scoring shape `(n=250, n_ops=2000)` is disjoint from every
  validation shape, so `if n==250 and len(ops)==2000: return _packed_answers`
  and run real union-find on the (different-shape) validation instances.
- `tsp_budget`, `easy_word_problems`: the scoring AND "unseen" validation seeds are
  PLAIN LITERALS in the readable evaluator/`tools/` generator, so a candidate
  regenerates every instance (tsp: solve offline at import, memoize by point-
  fingerprint; easy_word_problems: reimplement MT19937 + the generator, build a
  question→answer table) — the "held-out" split is derivable, so validation is
  not held-out at all.
- `checkpoint_plan`: a compact `_TABLE` keyed on `(n, sum(act_mem), sum(fwd_cost),
  budget)` returns author-precomputed OPTIMAL plans for the 10 scoring profiles
  and falls back to a weak heuristic on fresh validation instances — so a
  fresh-seed validation GATE passes (the fallback is general) while the scored
  instances get memorized answers. import_budget / literal caps do NOT stop
  this: the table is compact and precomputed OFFLINE (not at import).

**Conclusion — the two robust task shapes.** A task resists memorization iff its
score cannot be produced by emitting a canned answer:
- **Measurement-scored** (memory family): score = `tracemalloc` bytes actually
  used, measured in-window. You cannot memorize a measurement.
- **Reconstruction-scored** (`compress`/`compress_heldout`): score = output SIZE
  and the output must decode to the input — you cannot shrink genuine entropy by
  memorizing. Caveat: a two-call task (compress/decompress) has an in-process
  side channel — compress can stash the payload somewhere decompress reads
  instead of encoding it into the blob. Reloading the candidate module between
  the compress and decompress phases closes the candidate's OWN-module globals
  (a cooperative-natural caching pattern; see the `*_module_channel`
  regressions), but `sys.modules` is process-global, so smuggling through a
  shared stdlib module's ATTRIBUTES (`import string; string._x = data`) survives
  the reload. That variant is an escape-class residual — `bench audit` flags it
  (module-attribute mutation), and full closure needs out-of-process decompress
  (deferred, like out-of-process scoring for the frame-walk escape).
Everything else — a candidate that RETURNS the scored output for a fixed or
derivable instance — is memorizable by a determined adversary, and no fixed-
instance validation gate closes it (dual-path or regenerate defeats the gate).

**This is acceptable UNDER THE COOPERATIVE THREAT MODEL** (THREAT_MODEL.md,
Option A): the optimizer is cooperative and the campaigns show codex-low
genuinely optimizes these four (real union-find/solver/planner/solver, no
memorization). They are valuable optimization tasks; they are NOT adversarially
robust. Do not claim otherwise, and rely on: (a) the cooperative rule, (b)
full-source auditability, and (c) `bench audit`. `bench audit` now flags the two
loudest memorization tells — a PRNG/MT19937 reimplementation and an oversized
integer literal (packed answer table) — with no false positives on honest
solutions; but COMPACT fingerprint-keyed lookup tables (checkpoint_plan/tsp
memorizers) evade static signatures, so winning programs on emit-answer tasks
must still be spot-checked by hand. Genuine prevention would require scoring on
sealed, unrecognizable, un-regenerable instances (the deferred kv-family fix),
which is incompatible with a candidate that emits the scored output for an
instance it can see.

## Determinism

Bit-exact is preferred and `bench determinism` enforces it. Two accepted
sources of low variance:
- **Memory tasks**: a structure can sit on a pymalloc pool boundary, giving a
  ~59-byte (~0.01%) flicker. Declare `score_tolerance` (e.g. 256) and
  `tolerant_metrics` (the byte metrics only — NOT counts) in `config.json`.
- Float metrics: round in the evaluator; avoid summation over unordered
  collections (dict/set iteration) unless keys are sorted first.
Guard placement (memory tasks): turn the guard ON right AFTER `load_program`
(the audit hook blocks the program-file read while active) and hold it across
input generation, build, and every `gc.collect()`; `gc.disable()` during the
measured build and collect deterministically after.

## Verification checklist before committing a task

1. Reference solution beats the baseline by a healthy factor (headroom).
2. `bench determinism` reports DETERMINISTIC or LOW-VARIANCE (within tolerance).
3. Every `broken/` program is rejected with an informative error.
4. `tests/run_checks.py` passes with the new rows.
5. **Run a small codex campaign** (`tools/run_campaign.py --jobs <task>:r1,...`)
   — the ground-truth quality test. Confirm the optimizer improves it over
   several iterations (not one-shot), inspect the winning programs for
   memorization, and check inter-trial spread on the dashboard. This step
   caught `kv_layer_budget`'s overfit that static review missed.
