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
python3.12 -m bench workspace mem_index /path/ws  # program.py + spec.md + GOAL.md + session
# then point any agent at /path/ws with the goal "follow GOAL.md";
# GOAL.md contains the exact submit + self-test commands:
PYTHONPATH=<repo> python3.12 -m bench submit /path/ws/run program.py
python3.12 -m bench report /path/ws/run           # the run's result
```

No git, no loop, no codex required — that machinery is not part of the
benchmark.

## Base-suite design constraints (and how they're met)

The constraints in this section describe the seven lightweight tasks in the
first seven rows of the table below. Three additional generalization tasks have
heavier data, dependency, or device contracts documented later in this file.

1. **Efficient numeric scoring** — every base task scores in seconds via a
   single subprocess that prints one JSON line.
2. **Deterministic scores** — no base-task score uses a wall-clock metric.
   Scores are
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
   --rescore` extends that to whole recorded runs. Most of the eight
   tasks are bit-exact; the tracemalloc tasks that can land on a pymalloc
   arena boundary (`mem_index`, `mem_str`) are low-variance
   rather than bit-exact — a residual ~60-byte
   (~0.01%) flicker that neither pre-warming nor disabling GC
   removes. Each declares a `score_tolerance` in its `config.json`, so
   `determinism` reports them as LOW-VARIANCE (within tolerance) rather
   than failing; relative comparisons are unaffected.
3. **CPU only** — the base suite is pure-Python stdlib, with no GPU or
   third-party dependencies. The research SLM tasks are deliberately strict
   MPS workloads.
4. **Robust to system load** — because nothing is timed, a fully loaded
   machine produces the same scores (verified with CPU hogs running).
   Scores may differ across CPython versions/platforms, but are stable on
   a given system. Wall/CPU-time limits exist only as generous safety
   guards, far from the operating point of reasonable programs.
   "Runtime performance" tasks use instruction counts / instruction
   budgets instead of time.

## Base-suite requirements

- Python **3.12+** (`sys.monitoring`); macOS/Linux.
- No third-party packages for the base suite. The three dependency-heavy
  generalization tasks use the separately prepared environment described below.
- For the bundled loop only: [codex CLI](https://github.com/openai/codex), logged in.

## Task taxonomy: perfect information vs. generalization

Every task is labeled by how much the optimizer can see, because that is
the variable that controls overfitting:

- **Perfect information** (`kind: "perfect"`): the reported score *is*
  the final score — the workload being measured is the deployment. There
  is no train/test gap to exploit. All memory/instruction tasks are built
  this way (e.g. `mem_infer` measures the peak memory of the exact decode
  runs that define the task, including a held-out instance inside the
  scored maximum, so even output-hardcoding cannot win).
- **Generalization tasks** (`kind: "generalization"`): the score used to guide
  optimization is distinct from the final sealed-test score. The three
  dependency-free tasks expose a visible training set and seal a larger test
  from the same distribution on every submission. The three ML-systems tasks
  use explicit fit/calibration, online-validation, and sealed-test roles, with
  sealed tests deferred to accepted-incumbent background work. All six belong
  to the same generalization family: improvements must transfer beyond the
  examples or workloads that supplied optimization feedback. A
  restricted-information variant (`<task>_e2`) instead grades on a hidden
  *validation* score with only a handful of visible train examples — the agent
  sees a number, not the data, so it cannot memorize.

This supports the experiments in the writeup: compare reasoning efforts
(high/low/none), and compare grading on the visible train (which the agent can
memorize) against a hidden validation score (which it cannot) — the hidden test
reveals which regime actually generalizes.

## Tasks

| Task | Kind | Domain | Score (lower = better) | Baseline | Verified headroom |
|---|---|---|---|---|---|
| `mem_index` | perfect | text search / IR | serving peak bytes | 14.0 MB | 1.59 MB reached by loop (8.8x) |
| `mem_str` | perfect | string-collection storage | serving peak bytes | 7.92 MB | 189 KB reached by loop (42x) |
| `mem_infer` | perfect | hybrid LLM inference | peak live tensor bytes under 18M deterministic work units | 1.05 MB | 17.8 KB reference (59.2x); campaign pending |
| `ops_connect` | perfect | graph algorithms | bytecode instructions executed | 7.02 M | 50.5 K reached by loop (139x) |
| `word_problems` | generalization | language parsing + compositional arithmetic | 50/50 easy/hard train error; hidden test (train/test 1100/4400) | 0.991 train | hard-regime reference reaches 0.499 train; combined campaign pending |
| `compress_heldout` | generalization | compression that must generalize | train compressed bytes; hidden test corpus (4/4 docs, 50/200 KB) | ~200 KB train | train 13 KB; hidden test 70 KB (low) |
| `tag_seq` | generalization | sequence labeling | train per-token error; hidden test (500/2000) | 0.747 train | train→0 (overfits); hidden test 0.34 |
| `llm_routing` | generalization | cost-aware LLM routing | online validation regret; sealed ID/OOD test | see ML setup below | campaign results below |
| `optimizer_generalization` | generalization | learned optimizer transfer | normalized validation-loss curve AUC; sealed ID/OOD architectures | see ML setup below | campaign results below |
| `slm_compression_3_5bpw` | generalization | behavior-preserving SLM weight compression | online behavioral regression at 3.5 BPW; sealed behavior test | RTN W3 | completed campaign and method study below |
| `slm_compression_4_5bpw` | generalization | behavior-preserving SLM weight compression | online behavioral regression at 4.5 BPW; sealed behavior test | RTN W4 | see ML setup below |

(Store+query memory tasks score the **serving peak** — the tracemalloc peak
reached while answering the full query workload, with the peak reset right
after `build` so build-time transients are excluded. This charges both what a
structure retains AND what each query transiently materializes, so a store
that keeps a tiny compressed blob but decompresses a big block per query is
correctly penalized.)

Except for the new combined `word_problems` reference result, the "Verified
headroom" column reports the best score found in the campaign
(5 independent runs per task per effort, 1-hour box each) under the current
harness; all winning programs were audited clean (no escape gadgets, no
memorized/regenerated answers on perfect-info tasks). On the perfect-information
tasks more reasoning effort reliably lowers the score (high < low < none), and
the serving-peak metric makes the store+query memory tasks discriminating
(1.6–2.9x inter-run spread) versus retained-only scoring. In the three original
generalization campaigns, the agent drives the visible-train error to ~0 at
every effort; the hidden test then separates them: the historical easy
word-problem task generalizes strongly, while `tag_seq` exposes a substantial
train/test gap and benefits from hidden validation feedback. The three
ML-systems generalization tasks use online validation rather than visible-train
error and defer sealed testing, so their curves are presented separately below.

Memory tasks (`mem_index`, `mem_str`, `mem_infer`)
optimize serving footprint directly — compact data structures or inference
state that are cheap to hold and use, under strict correctness constraints.
`mem_infer` runs a Torch-backed DeltaNet/GQA hybrid through a metered tensor
API, so its memory and work totals are logical, deterministic quantities rather
than allocator or wall-clock measurements. It checks the full vocabulary-logit
trace against float32 inference while leaving room for safe state/cache
quantization, buffer reuse, and blocked kernels. The "speed" task
(`ops_connect`) counts bytecode instructions instead of time, so it rewards
better algorithms and pushing work into C builtins, deterministically.
`word_problems` combines the former easy and hard protocols behind one
`solve(question)` interface and one macro-averaged score. Its 1,100-example visible
train set contains 500 GSM8K-style language-diversity problems and 600 deeper
compositional problems; its sealed test contains a disjoint 2,000 + 2,400.
Component errors remain available as diagnostics, and the benchmark weights
the two regimes equally so an optimizer must improve one solver across both.

Synthetic data is deliberate — real GSM8K is memorized by frontier models, so
an optimizing agent could bake in memorized answers. The easy regime composes
event chains, transfers, idioms, distractors, number words, and varied question
targets. The hard regime usually combines four or more decisions, including
inverse operations, changing rates, ratio transfers, averages, elapsed-time
arithmetic, successive percentages, tiered prices, and composite geometry.
Train and test wording and operation combinations are independently generated,
so useful solutions must build reusable parsing and arithmetic semantics.

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
  differently-seeded data (unscored) and requires correctness — plus a
  comparable compression ratio in `compress_heldout`. Pure hardcoding/regenerating
  fails validation. (Caveat: for emit-answer tasks a validation *gate* is
  bypassable by a dual-path program — see TASK_AUTHORING.md's robustness
  boundary; the robust core is measurement- and reconstruction-scored tasks.)
- **AST scan** (static) rejects honest mistakes and obvious cheats:
  task-defeating imports (`zlib` in `compress_heldout`, `ctypes`/`mmap` in memory
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

## Results blogpost (generated — do not hand-edit)

`docs/blogpost.html` is produced by a generator; never edit the HTML directly
(hand-built charts are how the figures historically drifted apart). To change
it, edit `tools/make_blogpost.py` (charts/data/layout),
`tools/blogpost_content.py` (prose), or `tools/blogpost_exp4_data.py`
(Experiment 4 traces), then rebuild:

```bash
python3 tools/make_blogpost.py
```

The generator plots optimizer-active time reconstructed from the campaign
launcher logs (interrupted runs stitched, cut at 60 active minutes) and keeps
paired panels on one shared y-scale — preserve both properties when editing.
Agent-facing copies of these instructions live in `CLAUDE.md` and `AGENTS.md`.

## CLI

```bash
python3.12 -m bench list                      # task names
python3.12 -m bench spec mem_index               # print a task spec
python3.12 -m bench evaluate mem_index prog.py   # score one program (no record)
python3.12 -m bench baseline                  # base-environment tasks
python3.12 -m bench determinism --runs 3      # base tasks; verify repeatability

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

For mixed-task benchmark campaigns, use the durable resource-aware runner. It
defaults to 5 independent runs per task, 24 live agent rollouts, and a
one-hour *active* budget per rollout:

```bash
python3.12 tools/run_benchmark.py start july \
    --tasks word_problems,optimizer_generalization \
    --runs 5 --agent-concurrency 24 --time-budget 3600
```

Agent and evaluation concurrency are independent. Agents can think, edit, or
wait on an API in all 24 rollout slots, but acquire weighted capacity only
while grading. The default [resource profile](tools/benchmark_resources.json)
has 16 CPU units and one accelerator unit. Brief graders such as word problems
cost one CPU unit, so 16 can grade together. Sustained memory-task graders cost
two, and `optimizer_generalization` costs four, so at most eight and four can
grade together respectively. The MPS SLM grader requests the one accelerator
unit and two CPU units together. Mixed workloads share the same capacities (for
example, three optimizer evaluations leave four CPU units for other graders),
so per-task limits cannot oversubscribe the host in combination. Foreground
evaluation admission is FIFO, preventing expensive requests from starving
behind a stream of cheap ones. Edit or supply a different profile with
`--resource-profile`; `--cpu-capacity` and `--accelerator-capacity` provide
one-run capacity overrides.

Campaign state is durable. Pause from another terminal, use Ctrl-C on the
controller, inspect progress, and resume with:

```bash
python3.12 tools/run_benchmark.py pause july
python3.12 tools/run_benchmark.py status july
python3.12 tools/run_benchmark.py resume july
```

Task configs use `"evaluation_resource": "accelerator"` for local-model
evaluators; omitted means `"cpu"`. The profile can add a CPU request to an
accelerator task; otherwise the pools are independent. Queue time is recorded as
`eval_queue_seconds`, and locks are released automatically if a loop exits or
is killed. Each job's active seconds and last submission are checkpointed in
`runs/_campaign/benchmarks/<name>/state.json`. Resume opens that session's
incumbent directly and does not re-grade its baseline; a half-created iteration
workspace is discarded. Evaluation-queue intervals are subtracted from the
active budget, including a wait still in progress when paused. Time between
pause and resume is never charged. Thus a one-hour run delayed by 90 seconds of
queue contention and paused for an hour receives approximately 61 minutes 30
seconds of running wall time, across both launches, while retaining a one-hour
optimization budget. Overlapping queue intervals within one run are counted
once, not summed. `tools/run_campaign.py` remains available as the simpler
legacy, non-durable launcher.

This accounting and sealing inherit the benchmark's cooperative threat model.
Deferred cache entries live in an operator-owned campaign directory, benchmark
fingerprints rehash all declared data and scoring-code bytes, and a held-out
worker evaluates a private copy of the authenticated submission. Fingerprints
also bind Python/package versions plus the OS release, macOS version, and
architecture, so the stable cache is reusable only across short same-host
restarts—not across runtime or Metal-driver updates. However, the
optimizer and launcher still run as the same OS user: a deliberately hostile
agent that escapes its workspace could read casually sealed files or forge its
queue-wait telemetry to claim extra time. Treat the time refund and held-out
confidentiality as auditable protocol controls, not a security boundary. A
non-cooperative contest must put cache/seals and wait accounting behind a
broker owned by a different OS principal (or an equivalent external service).

Three generalization tasks have additional ML-system dependencies: two are
CPU-only algorithm tasks and one is an MPS-only small-language-model
weight-compression task. They require the optional environment and prepared
compact artifacts/models:

```bash
uv venv /tmp/text-opt-bm-ml --python python3.12
uv pip install --python /tmp/text-opt-bm-ml/bin/python -r requirements-ml.txt
/tmp/text-opt-bm-ml/bin/python tools/prepare_ml_benchmark.py
/tmp/text-opt-bm-ml/bin/python tools/preflight_ml_benchmark.py --evaluate
```

Preparation compacts the pinned LLMRouterBench performance-cost release,
generates the stochastic optimizer workloads, verifies the selected SFT
conversation artifacts, downloads the public pinned model snapshot when it is
not already present, and authenticates the pinned
LiquidAI/LFM2.5-230M snapshot outside the
repository. `ml_assets.json` records source hashes, model revisions, and
compact-artifact hashes. The active tasks are:

- `llm_routing`: custom-v7 cost-aware routing over 4,960 fit, 1,089 visible
  scoring, 2,193 validation, and 3,648 sealed-test prompt rows (11,890 total).
  Three domains occur only in sealed test, whose scalar weights known and
  unseen-domain cells 50/50; evaluation is CPU-only;
- `optimizer_generalization`: research-protocol v9 synthesis of one
  optimizer, ranked on real MNIST/Fashion-MNIST neural classifiers,
  convolutional models, autoencoders, and character RNNs, with ten analytic
  objective families retained as non-compensating diagnostics. Sealed test
  adds residual, gated, and bottleneck architectures and weights known versus
  unseen architectures 50/50; evaluator-owned losses use CPU-only JAX JIT;
- `slm_compression_3_5bpw` and `slm_compression_4_5bpw`: arbitrary emitted
  LFM2.5-230M weights in a safe generic packed format under explicit 3.5- and
  4.5-whole-model-BPW caps.
  GPTQ/AWQ/HQQ-style affine groups, block floats, dense records, and
  GGUF-style codebooks/graphs are representable; every submitted byte counts.
  Compression quality is behavioral regression from native BF16 on GPQA,
  IFBench, single-turn BFCL, short GSM8K, and MMLU-Pro rather than conversation
  NLL.

Routing v7 adds sealed-only domains and a denser sealed cost-preference grid.
Optimizer protocol v9 is a score-incompatible expanded research protocol:
its primary scalar is TaskSet-style empirical-reference curve AUC on real
neural workloads, while its former synthetic AUC is reported separately.

The prepared N=5 research campaign admits ten live optimization loops and one
non-preemptive MPS scoring slot. All model-bearing SLM work—including response
generation, compilation and activation calibration, paper-native diagnostics,
online validation, and sealed testing—also acquires the same exclusive
cross-process MPS lease. Thus no two model jobs share the device, while
model-free CPU checks and either CPU task may overlap the MPS lease holder.
An SLM campaign additionally holds an exclusive phase lease: operator-side
datagen, compilation, direct evaluation, calibration audits, repeatability
checks, and paper-native jobs fail closed until the campaign and deferred drain
finish. Successful evaluator waits on the inner shared MPS lease carry the
canonical lock-helper hash and trusted timestamps; the parent runner validates
and refunds that interval just like accelerator-semaphore queue time.

```bash
/tmp/text-opt-bm-ml/bin/python tools/run_benchmark.py start ml-v9 \
    --tasks llm_routing,optimizer_generalization,slm_compression_3_5bpw,slm_compression_4_5bpw \
    --runs 5 --agent-concurrency 24 --time-budget 3600 --iterations 1000 \
    --model gpt-5.6-sol --effort high --prefix 5x-gpt56-sol-high-
```

For both SLM tasks, 128 training conversations are calibration data only and
are never scored. Online ranking uses 100 hidden BF16-passing examples: 20 each
from GPQA Diamond, IFBench, single-turn BFCL, short GSM8K, and MMLU-Pro. A
disjoint 100-example split is sealed for final evaluation.
The producer receives only the pinned LFM checkpoint and the 128 calibration
conversations; it receives no validation or test inputs.
Calibration, compression, trusted decoding, and compressed-model inference use
PyTorch MPS or CUDA with operator fallback disabled. Behavioral generation is greedy,
requires EOS, and uses a native-response-relative token cap. CPU, CUDA, MLX,
and fallback-enabled SLM results are inadmissible.
The hard 3.5-BPW cap charges every byte in the emitted weight bundle,
including codes, scales, zero points, codebooks, permutations, padding,
safetensors headers, and the manifest. This is not a Pareto-frontier task;
future operating points are separate benchmark runs. A trusted tensor-only decoder supports
affine, codebook, block-float, dense, alias, and bounded graph records; custom
submitted decoder code is never executed during grading. GGUF-style scalar,
mixed-bit, and codebook schemes can be represented through those trusted
records after conversion to QWeight.
All three active tasks defer sealed testing outside online submissions. The SLM
test split uses the otherwise idle exclusive MPS lease; routing-v7 and optimizer-v9
use otherwise idle CPU evaluation capacity. Online submissions evaluate only
training/validation data, and each accepted incumbent queues low-priority
sealed-test work. All pending CPU and
SLM test work drains after optimization. A test-only crash is recorded in its
sealed operator artifact; it cannot reject an incumbent, alter later prompts,
or otherwise influence selection.

Always launch campaigns with the optional environment's Python; the loop
propagates its exact interpreter into agent self-evaluation commands. The
paper-native GPTQ/AWQ/SparseGPT/Wanda diagnostic runs offline, writes
content-addressed caches, and remains separate from the ranked optimization
loop. Keep all operator-only source corpora and scoring material outside the
optimizer-readable repository.

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
