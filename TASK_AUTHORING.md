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
- **Memory tasks** measure `tracemalloc` retained/peak bytes — this rewards
  compact representations regardless of C usage, so they are the most reliable
  family to design (big headroom, robust). But they trend "moderate" (all
  runs converge to a similar structure); depth/variance ("strong") comes from
  genuine algorithmic openness (a transformer impl, a codec, a solver).

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
  case. In-window + unseen validation is NOT enough if the candidate can
  regenerate outputs it was given. These need to be **scored on sealed
  held-out data** the candidate never saw and cannot regenerate
  (`bench/heldout.py`). The kv_quant / weight_quant / kv_fixed_budget tasks
  were removed pending this treatment.

General rules: a small fixed instance set is fine ONLY with genuine unseen
validation; prefer several instances with non-unique sizes; forbid metric-
control modules (`tracemalloc`, `sys`, `resource`) and the escape surfaces the
shared `broken/` tests probe.

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
