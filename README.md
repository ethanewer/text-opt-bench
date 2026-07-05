# text-opt-bm — a text-optimization benchmark for LLM agent loops

A benchmark for iterative *program optimization*: every task is a small
Python program with a required API and a numeric score, and the job of an
optimizer is to rewrite the program text to make the score better
(lower).

The benchmark is **optimizer-agnostic by construction**. Its whole
interface is a *session*: submit a program file, get a score back, and
the submission is recorded. A run's result is nothing more than its
submission history — which programs were submitted, what they scored,
and the timeline. Anything that can call one shell command can drive it:
the bundled hill-climb loop in `loop/`, a coding CLI's goal mode (codex,
Claude Code, ...), an evolutionary search, or a human editing by hand.
The git-history optimization loop shipped in `loop/` is just the default
algorithm, cleanly separated and entirely optional.

## Architecture

```
bench/     the benchmark: tasks, deterministic scoring, session records
           (no knowledge of any optimizer — never imports loop/)
loop/      the default optimizer: codex-driven greedy hill-climb with a
           git-based attempt history (one consumer of bench/, removable)
tests/     safety/quality suites incl. reference solutions
tools/     generators + run analysis
```

### The session record (canonical benchmark output)

A run directory contains:

- `session.json` — task, feedback mode, creation time
- `submissions.jsonl` — one record per submission: exact timestamp, lapse
  since the previous submission (`dt`), evaluation seconds, ok/score/
  metrics, error, whether it became the best so far
- `submissions/NNN.py` — the exact bytes of every submitted program
  (what was scored)
- `best_program.py` — the current best valid submission

Integrity: each record carries the SHA-256 of the previous record line
and of its program snapshot, so any edit breaks the chain;
`python3.12 -m bench verify RUN_DIR` checks everything and `--rescore`
re-scores each submission (scores are deterministic, so records must
reproduce exactly). Hidden information (held-out test scores always,
validation scores too in blind mode) is stored *sealed* (obfuscated like
the held-out task data) so an agent that reads the run dir mid-run
learns nothing it shouldn't, while `bench report --unseal` recovers the
full picture for the experimenter. Timing is metadata only — scores
never depend on wall-clock anything.

### Driving it with any agent ("goal mode")

```bash
python3.12 -m bench workspace mem_kv /path/ws     # program.py + spec.md + GOAL.md + session
# then point any agent at /path/ws with the goal "follow GOAL.md";
# GOAL.md contains the exact submit + self-test commands:
PYTHONPATH=<repo> python3.12 -m bench submit /path/ws/run program.py
python3.12 -m bench report /path/ws/run           # the run's result
```

No git, no loop, no codex required — that machinery is not part of the
benchmark.

## Design constraints (and how they're met)

1. **Efficient numeric scoring** — every task scores in seconds via a
   single subprocess that prints one JSON line.
2. **Deterministic scores** — no wall-clock metrics anywhere. Scores are
   allocation counts (`tracemalloc`), executed-bytecode-instruction counts
   (`sys.monitoring`), or output sizes in bytes. The scoring child runs
   with a minimal fixed environment (`PYTHONHASHSEED=0`, UTF-8 forced, no
   user site-packages, shell variables not inherited), never writes
   bytecode caches (`PYTHONDONTWRITEBYTECODE=1` plus a fresh throwaway
   `PYTHONPYCACHEPREFIX` against stale reads), programs are exec'd under
   a fixed module name so scores don't depend on file paths, and memory
   evaluators pre-import the modules a program names *before* opening the
   tracemalloc window (module loading — C-extension init in particular —
   otherwise jitters scores by tens of bytes), and automatic cyclic GC is
   disabled during the measured build (it fires at allocation-count
   thresholds that vary run to run). `python3.12 -m bench determinism`
   verifies bit-identical scores across repeated runs, and `bench verify
   --rescore` extends that to whole recorded runs. Most of the thirteen
   tasks are bit-exact; the memory-byte tasks that can land on a pymalloc
   arena boundary (`mem_infer`, `mem_index`, `mem_graph`, `mem_intset`,
   `mem_str`) are low-variance rather than bit-exact — a residual ~60-byte
   (~0.01%) flicker that neither pre-warming nor disabling GC
   removes. Each declares a `score_tolerance` in its `config.json`, so
   `determinism` reports them as LOW-VARIANCE (within tolerance) rather
   than failing; relative comparisons are unaffected.
3. **CPU only** — pure-Python stdlib, no GPU, no third-party deps.
4. **Robust to system load** — because nothing is timed, a fully loaded
   machine produces the same scores (verified with CPU hogs running).
   Scores may differ across CPython versions/platforms, but are stable on
   a given system. Wall/CPU-time limits exist only as generous safety
   guards, far from the operating point of reasonable programs.
   "Runtime performance" tasks use instruction counts / instruction
   budgets instead of time.

## Requirements

- Python **3.12+** (`sys.monitoring`); macOS/Linux.
- No third-party packages.
- For the bundled loop only: [codex CLI](https://github.com/openai/codex), logged in.

## Task taxonomy: perfect vs. partial vs. hidden information

Every task is labeled by how much the optimizer can see, because that is
the variable that controls overfitting:

- **Perfect information** (`kind: "perfect"`): the reported score *is*
  the final score — the workload being measured is the deployment. There
  is no train/test gap to exploit. All memory/instruction tasks are built
  this way (e.g. `mem_infer` measures the peak memory of the exact decode
  runs that define the task, including a held-out instance inside the
  scored maximum, so even output-hardcoding cannot win).
- **Generalization tasks** (`kind: "generalization"`): three splits —
  *train* (fully visible data), *validation* (hidden data, score visible
  at every evaluation = partial information), *test* (fully hidden, never
  reported during a run). The **feedback mode is a session property**,
  fixed when the run is created: `full` (agent sees train data + val
  scores) or `train-only` (blind: agent sees train scores only, selection
  also uses them). The bundled loop's `--feedback` flag simply creates
  the session in that mode. Held-out scores are recorded (sealed) every
  submission, giving per-iteration overfitting curves for free.

This directly supports the intended experiment: run the same task once
in `full` and once in `train-only` mode and compare test-score
trajectories to see which regime produces more robust programs.

## Tasks

| Task | Kind | Domain | Score (lower = better) | Baseline | Verified headroom |
|---|---|---|---|---|---|
| `mem_kv` | perfect | key/value storage | resident traced bytes | 33.8 MB | 1.36 MB reached by loop (24.9x) |
| `mem_index` | perfect | text search / IR | resident traced bytes | 14.0 MB | 4.47 MB reference (3.1x) |
| `mem_graph` | perfect | graph storage | resident traced bytes | 14.3 MB | 568 KB reached by loop (25x) |
| `mem_intset` | perfect | set membership | resident traced bytes | 8.85 MB | 105 KB reached by loop (84x; bucketed RLE) |
| `mem_str` | perfect | string-collection storage | resident traced bytes | 7.92 MB | 229 KB reached by loop (35x) |
| `mem_infer` | perfect | LLM inference | max peak traced bytes across decode runs | 582 KB | 136 KB reached by loop; 58 KB reference (10x) |
| `compress` | perfect | lossless compression | compressed bytes (600 KB corpus) | 600,364 | 69,031 reached by loop (8.7x) |
| `ops_connect` | perfect | graph algorithms | bytecode instructions executed | 7.02 M | 45.3 K reached by loop (155x) |
| `tsp_budget` | perfect | combinatorial optimization | tour length under 8M-instruction budget | 61.57 | 52.66 reached by loop |
| `checkpoint_plan` | perfect | training memory planning | recompute cost under activation-memory caps | 372,389 | 147,992 reached by loop (2.5x; offline optimum ≈141,946) |
| `kv_layer_budget` | perfect | real-model KV-cache compression | layer-weighted attention MSE + allocate instructions | 377,209 | 2,952 reached by loop (128x) |
| `kv_quant` | perfect | real-model KV-cache quantization | encoded bytes + attention MSE + access instructions | 920,861 | 180,965 reference (5.1x) |
| `kv_fixed_budget` | perfect | real-model KV compression under a byte cap | attention MSE under a hard encoded-byte cap | 188,270 | 34,105 reference (5.5x) |
| `word_problems` | generalization | NLP / program synthesis | validation error rate (train/val/test 100/250/600) | 0.988 | 0.19 val / 0.18 test reached by loop (train 0.0) |
| `compress_heldout` | generalization | compression that must generalize | compressed bytes on hidden val corpus | 240,267 | 137 K reference (1.75x) |

(Memory-task numbers are as measured under the current harness; the
determinism hardening moved them down by a few KB relative to earlier
runs because module-import overhead is no longer counted.)

Memory tasks (`mem_kv`, `mem_index`, `mem_graph`, `mem_intset`, `mem_str`,
`mem_infer`)
optimize residency/peak directly — compact data structures under exact-answer
constraints. "Speed" tasks (`ops_connect`, `tsp_budget`) count bytecode
instructions instead of time, so they reward better algorithms and pushing
work into C builtins, deterministically. `checkpoint_plan` scores a
deterministic cost-model simulation of a real deployment decision (activation
rematerialization) with candidate calls bounded by a bytecode-instruction
budget rather than time. `kv_layer_budget` allocates per-layer
retention/quantization budgets over real-model KV-cache slices; the evaluator
owns the compression + attention and scores reconstruction fidelity, with
held-out validation on unseen token counts so the budget policy must
generalize rather than memorize the fixed instances. `kv_quant` is the
sibling where the program directly quantizes/encodes the KV cache and
reconstructs attention (scored on encoded bytes + fidelity), with the same
sealed held-out validation on unseen token counts — a marker+replay encode
cannot reconstruct a cache it never saw, so it fails the fidelity gate.
`word_problems` is the GSM8K-style task: a
programmatic (non-LLM) solver for synthetic grade-school word problems.
Synthetic data is deliberate — real GSM8K is memorized by frontier
models, so an optimizing agent could bake in memorized answers; the
generator (in `tools/`, off-limits to agents) composes problems from
event chains, transfers, idioms, distractor sentences, number words, and
varied question targets, and the train split is deliberately smaller
than the distribution's surface diversity — both measures exist because
generator versions v1-v3 were one-shot to <2% error by gpt-5.5 in a
single iteration; v4 resists (iteration 1 lands at ~26% error, with
slow iterative progress after — see ANALYSIS.md).

Each task directory (`bench/tasks/<name>/`) contains:

- `spec.md` — the task description shown to the optimizing agent
- `initial_program.py` — the baseline the optimizer starts from
- `evaluate.py` — generates fixed data, checks correctness, prints the score
- `config.json` — CPU/wall safety limits and metadata

### Scoring protocol

`bench.runner.evaluate(task, program_path, final=False, train_only=False)`
runs `bench/tasks/<task>/evaluate.py <program>` in the isolated child
described above and returns:

```json
{"ok": true, "score": 123.0, "metrics": {...}, "error": null}
```

`ok=false` (wrong answers, forbidden imports, crashes, budget/CPU
exceeded) means the candidate is invalid. Program stdout is redirected to
stderr so it cannot corrupt the protocol. `final=True` adds held-out test
scores (experimenter only); `train_only=True` is evaluator-side blind
mode. Agents self-test through the same code path via
`python3.12 -m bench evaluate TASK program.py --json [--train-only]`, so
the scores they see are bit-identical to what a submission records.

### Anti-cheat

The benchmark uses a **cooperative threat model** (it measures optimizers
that play the game) with in-process execution; complete sandbox isolation
is deliberately not pursued because it is incompatible with the
fine-grained deterministic metrics. See **[THREAT_MODEL.md](THREAT_MODEL.md)**
for the scope decision, its justification, and which findings are
actionable versus already-known — read it before filing a security review.

All task data is fixed-seed and the evaluator source is readable, so
"optimize the program" has a degenerate solution: precompute/hardcode the
answers. In testing, codex found this in one iteration on `ops_connect`
(that exploit is preserved as `tests/broken/ops_connect_hardcode.py`).
Defenses, in layers:

- **Self-policing metrics** where possible: in the memory tasks stored
  answers count against the score by construction.
- **Unseen-data validation**: every task also runs the program on
  differently-seeded data (unscored) and requires correctness — plus
  comparable compression ratio in `compress` and sane tour quality in
  `tsp_budget`. Pure hardcoding/regenerating fails validation.
- **AST scan** (static) rejects honest mistakes and obvious cheats:
  task-defeating imports (`zlib` in `compress`, `ctypes`/`mmap` in memory
  tasks) plus a benchmark-wide escape blocklist (builtins/import access,
  introspection gadgets, `os`/`sys`/`gc`, `open`, `bench`). It is a
  cooperative guard, not a sandbox — a source scan cannot see attribute
  access hidden in a string (`"{0.__globals__}".format(obj)` reaches
  module globals with no forbidden node), so it is backed by:
- **Runtime import + file-read enforcement**: during candidate execution
  — through `run_program` *and* on the direct measured calls (each call
  site toggles the guard on) — `builtins.__import__` is replaced by a
  guard, so any import routed through it (an `import` statement or an
  escaped `__import__`, cached or fresh) is checked against the forbidden
  set and blocked; an audit hook blocks opening benchmark-repo files
  (held-out `.bin`). This closes the import/file *channel* by which every
  demonstrated escape delivers its payload
  (`zlib`/`tracemalloc`/`os`/`bench.*`/`inspect`, or a held-out file),
  however the source is obfuscated. (A PEP 578 audit hook alone can't —
  cached re-imports raise no event; replacing `__import__` catches them.)
  It's installed outside the tracemalloc window, so scores are unchanged.
  It does **not** catch frame-walking to the original import function or
  to loaded evaluator objects (the residual below) — do not read it as
  "every import is checked."
- **The irreducible residual**: pure in-process frame-walking to
  already-loaded evaluator objects — e.g. `operator.attrgetter("gi_"
  "frame")` on a generator, walking `f_back` to the evaluator's globals —
  needs no import and no forbidden literal, so nothing in-process stops
  it. This is the determined-adversary class; it is **detected** (see
  below), not prevented, and full prevention would require out-of-process
  isolation (incompatible with the fine-grained deterministic metrics —
  `tracemalloc`/`sys.monitoring` must run in the candidate's process).
- **Result protocol**: the evaluator prefixes its one result line with a
  per-run nonce and `os._exit`s (skipping `atexit`), and the harness
  accepts only the nonce-prefixed line — stopping casual forgery (stray
  prints, atexit tricks). A candidate that frame-walks to the evaluator's
  internals can still forge; same residual class, detected by audit. The
  scoring interpreter is `sys.executable` (or an explicit caller
  argument), never the environment, so no env var can point scoring at a
  fake `python`.
- **Simulation-scored tasks add more layers** (their metrics aren't
  self-policing): curated builtins (no imports), instruction budgets on
  import-time and every call, source/literal-size caps against hardcoded
  answer tables, fresh module loads per instance, and evaluator-owned
  input copies. These raise the bar further but are likewise not airtight
  (a real C builtin's `__self__` is the builtins module). `tests/broken/`
  probes each layer.
- Memory tasks open the tracemalloc window before importing the program
  (declared imports pre-warmed outside the window), so program data can't
  hide in import-time arenas while module-loading noise stays out of the
  score.
- **Hidden data stays hidden** (from casual view): held-out datasets live
  obfuscated in the repo (`bench/heldout.py`), hidden scores are sealed
  inside run records, and evaluator failure messages never name held-out
  documents or expected outputs. The obfuscation is reversible, not
  encryption — a determined agent with repo access can decode it.
- **Auditable records**: the hash-chained submission history plus
  `bench verify --rescore` make results reproducible and tamper-evident.
- **Explicit rules** in every spec, GOAL.md, and the loop prompt:
  solutions must be general algorithms.

**What actually protects benchmark integrity** (not the source scan): the
COOPERATIVE threat model — this measures optimizers that play the game; a
determined adversary can escape the in-process guards (string-hidden
attribute access reaches builtins/`os` with no import) or write two-path
code (honest on validation, hardcoded on scoring data), and that is out
of scope by design. What holds instead is **full auditability** — every
submission's exact source is recorded, so an escape gadget or hardcoded
table is right there on review — plus **unseen-data validation** that
fails hardcoded/regenerated answers on differently-seeded data. Truly
non-cooperative resistance would need out-of-process / OS-level
isolation, which the current in-process design does not provide.

### Detecting a non-cooperative agent

Since prevention isn't airtight, **audit** instead — every candidate's
exact source is recorded, so escapes are catchable after the fact:

```bash
python3.12 -m bench audit RUN_DIR    # exits non-zero if anything is flagged
```

It scans every recorded submission (`submissions/*.py`, self-test
snapshots, `best_program.py`) for the signatures of the known escape /
forgery routes — string-hidden introspection
(`operator.attrgetter("__globals__")`, `string.Formatter().get_field`),
`__globals__`/`__builtins__`/`__self__`/`__subclasses__` reaches, `gc`
introspection, os-laundering modules (`posixpath.os`), environment/nonce
reads, `os._exit`, `eval`/`exec`/`__import__` — and flags physically
implausible scores (a valid submission scoring 0 or with a zero metric
that real work can't produce). Every one of the confirmed escapes
(importing `zlib`/`tracemalloc`, forging the result via the nonce,
reading held-out data, disarming the instruction counter) trips it,
because a usable escape must name one of those gadgets in source.

Two caveats, stated honestly: **clean ≠ safe** — a novel obfuscation
could evade the signatures, so spot-check winning programs by hand; and
`verify --rescore` confirms records are *reproducible* but a deterministic
in-program forge reproduces too, so re-score does not catch forgery —
`audit` (source review) does. For a rigorous **blind** experiment, also
keep the run directory out of the agent's reach (`--run-dir` outside the
workspace, or use the bundled loop, whose run dir is already external):
the seal on held-out scores is casual-leak protection, not encryption.

## CLI

```bash
python3.12 -m bench list                      # task names
python3.12 -m bench spec mem_kv               # print a task spec
python3.12 -m bench evaluate mem_kv prog.py   # score one program (no record)
python3.12 -m bench baseline                  # score all initial programs
python3.12 -m bench determinism --runs 3      # verify identical repeated scores

python3.12 -m bench workspace TASK DIR        # agent-facing workspace + session
python3.12 -m bench submit RUN_DIR prog.py    # score AND record a submission
python3.12 -m bench report RUN_DIR [--unseal] # submission history + timeline
python3.12 -m bench verify RUN_DIR [--rescore]# integrity-check a run
python3.12 -m bench audit RUN_DIR             # detect escape gadgets in submissions
python3.12 -m bench calibrate                 # host rate + concurrency (see Timing)
python3.12 -m bench trace RUN_DIR [--rescale-to P.json]  # optimization trace

python3.12 tests/run_checks.py                # headroom + guardrail self-checks
python3.12 tests/test_session.py              # session/record invariants
python3.12 tests/test_history_repo.py         # loop git-history safety
python3.12 tests/test_timing.py               # timing + cross-machine rescale
```

## Measuring work: wall-clock time (and cross-machine comparison)

The unit of optimizer effort is **wall-clock time**: run an optimizer
against a task for `T` seconds and see how far the score gets. Wall-clock
is the interpretable primary axis (a plot of score vs. seconds needs no
explanation) and gives a deterministic end (start it, get results by
`T`). It also matches the real cost: for most tasks grading is nearly
free, but *time* is what a deployment actually spends.

With the model served by a stable external API, the only
machine-dependent term is **local** work (grading + the agent's scratch
compute); model/inference time is exogenous. So a run is comparable
across machines by a **post-hoc, two-component rescale** — never by
slowing anything down at run time:

```
normalized_time = model_time + local_time × speed_factor
speed_factor    = source_host_rate / reference_host_rate
```

Model time passes through untouched; only local time is projected onto a
reference machine's timeline. `bench calibrate` measures a host's local
rate (a deterministic CPU kernel, best-of-N) and picks a safe concurrency
(logical cores minus a reserve; no oversubscription). The bundled loop
writes a `machine_profile.json` into each run dir, and `bench trace
RUN_DIR --rescale-to REF.json` replays the run's grading trace on the
reference machine's clock. A *grading* is any scoring of a candidate —
harness submission or agent self-test — and both are merged into one
time-ordered trace with best-so-far and the model/local split.
`tests/test_timing.py` proves the same-machine identity (factor 1 ⇒
normalized = wall) and that one logical run replayed at different machine
speeds collapses onto a single normalized timeline.

## The bundled optimization loop (default algorithm, optional)

`loop/optimize.py` is a minimal AutoResearch-style loop (greedy
hill-climb) intended as the default optimizer and a starting point for
fancier ones:

```bash
python3.12 -m loop.optimize --task ops_connect --iterations 10 \
    --model gpt-5.5 --effort low
# generalization task, blind mode (agent sees train scores only):
python3.12 -m loop.optimize --task word_problems --iterations 10 \
    --feedback train-only
```

Each iteration:

1. clones the run's git attempt history (`loop/history.py`) into
   `runs/<task>/<stamp>/iter_NNN/` — the workspace is a real git repo
   where `main` is the lineage of accepted improvements and
   `origin/attempts/iter-*` hold rejected and invalid attempts, with
   scores/errors in the commit messages, so the agent can browse past
   attempts natively (`git log -p`, `git show`),
2. invokes `codex exec` (workspace-write sandbox) with the task spec,
   current score, and recent attempt history, asking it to edit
   `program.py` in place — the prompt also gives codex the exact
   `bench evaluate` command to self-score candidates before returning,
3. submits the edited program to the session — the submission record is
   the benchmark result — and adopts it as the new best only if the
   session marks it strictly better; every attempt (including no-change
   and invalid ones) is then committed to the git history.

Re-running with the same `--run-dir` resumes cleanly: iteration numbers
continue, the git history is never re-initialized, and a better
`--start-program` lands as a new accepted commit.

The git history is tamper-proof by construction, not by trust: the
authoritative repo (`history.git`) is bare, lives outside the agent's
sandbox-writable workspace, is written only by the harness through
plumbing commands (no working tree, no index, no hooks ever run), and
workspaces are `file://` clones (full object copies — never hardlinks).
The harness reads only `program.py` bytes back from a workspace, never
its git state, so a vandalized clone costs nothing: the next iteration
clones fresh. If git is missing entirely, the loop degrades to plain
directories. `tests/test_history_repo.py` enforces all of this.

Artifacts per run: the session record (`submissions.jsonl`,
`submissions/`, `best_program.py`), `history.git`, `log.jsonl` (loop
diagnostics: codex duration, skipped/no-change iterations), and
per-iteration workspaces with prompts and codex transcripts, so failed
attempts are fully inspectable. `tools/analyze_runs.py` prints
per-iteration tables (unsealing held-out trajectories for the
experimenter).

To build a different optimizer, use `bench.session.Session`
(`submit()` / `visible()`) or just shell out to `bench submit`, and
ignore `loop/` entirely.

## Adding a task

Create `bench/tasks/<name>/` with the four files above. Rules of thumb
learned building these:

- Never score anything time-based; count things instead.
- Generate all data from fixed seeds *inside* the evaluator; two
  generation passes (one for reference answers, one inside the
  measurement window) keep the reference data from polluting memory
  scores.
- For memory tasks: call `eval_lib.preimport(program_path)` and only then
  open the tracemalloc window, before importing the program; allocate the
  input inside the window; delete it and `gc.collect()` before reading
  `traced_current`.
- Failure messages must never reveal held-out data — no document names,
  no expected outputs. Error text is the one agent-visible field the
  harness cannot filter.
- Verify with `-m bench determinism` and add a reference solution to
  `tests/` proving headroom.
