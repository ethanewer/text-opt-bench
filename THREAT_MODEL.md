# Threat model and security scope

**Decision (project owner, 2026-07-03): this benchmark uses a cooperative
threat model with in-process execution (Option A). Complete sandbox
isolation (Option B) is deliberately NOT pursued.** This note explains
why, defines what is in and out of scope, and tells reviewers which
findings are actionable versus already-known-and-accepted.

## What this benchmark measures

An *optimizer* (an LLM agent loop, a coding-CLI goal mode, a search
algorithm, or a person) that is **trying to play the game**: rewrite a
program to legitimately improve a numeric score. The benchmark's value is
in comparing how well such optimizers improve real programs, recorded as
a reproducible, auditable submission history. It is **not** a system for
safely scoring code from an unknown, hostile third party.

## Why in-process, and why that caps tamper-resistance

The benchmark's defining property is **fine-grained deterministic
scoring**: exact `tracemalloc` byte counts, exact `sys.monitoring`
bytecode-instruction counts, exact output sizes — bit-identical across
runs and immune to machine load. Those measurements **must run in the
same process as the candidate** (`tracemalloc` and `sys.monitoring` are
per-process, in-process facilities). There is no external, deterministic
way to measure them.

Consequence: **anything in the candidate's process is reachable by the
candidate.** Untrusted Python executed in-process cannot be securely
sandboxed. This is not a bug to be patched — it is a proven property,
established here across every layer we tried (AST source scan, regex
audit, PEP 578 audit hook, `__import__` replacement). The irreducible
route, verified live, needs no import and no forbidden token:

```python
import operator                                  # itemgetter is legitimate
frame = operator.attrgetter("gi_" "frame")((x for x in []))   # a frame
# walk frame.f_back to the evaluator's globals, reach any loaded object
```

Full prevention would require Option B — running the candidate in an
isolated process and measuring it from outside (peak RSS instead of
`tracemalloc`; no instruction counting) — which destroys the fine-grained
determinism that is the benchmark's entire point. That trade is not worth
it for a cooperative benchmark.

## What actually protects integrity (defense in depth)

1. **Static scan** (`bench.eval_lib.scan_forbidden`): rejects honest
   mistakes and obvious cheats — task-defeating imports, builtins/import
   access, introspection gadgets, `bench`, file IO — at load time.
2. **Runtime import + file enforcement** (`bench.eval_lib`): during
   candidate execution — whether the candidate is called through
   `run_program` OR directly by the evaluator on the measured/scored path
   (the flag is toggled around every call site) — `builtins.__import__`
   is replaced by a guard, so any import that goes through it (an `import`
   statement, or an escaped `__import__` reached through obfuscation,
   cached or fresh) is checked against the forbidden set and blocked; an
   audit hook blocks reading benchmark-repo files. It runs outside the
   measurement window, so scores are unchanged. **Scope, stated precisely:**
   this blocks the import/file *channel* — which is how every escape
   demonstrated to date delivers its payload (`zlib`/`tracemalloc`/`os`/
   `bench.*`/`inspect`, or a held-out `.bin`). It does **not** catch an
   adversary who frame-walks to the *original* import function
   (`eval_lib`'s saved reference) or to already-loaded evaluator objects
   and uses them without ever calling the replaced `__import__`. That is
   the same in-process frame-walk residual below — detected, not
   prevented. Do not read this layer as "every import is checked."
   The candidate-execution span also covers **more than the call itself**:
   measured evaluators materialize/type-check the return value INSIDE the
   measurement window (rejecting generators and list/tuple subclasses, so
   work can't be deferred to a lazy `__iter__`/`__getitem__` consumed after
   the window), and — on the memory tasks — keep the guard active across the
   **entire measured region**: from immediately after the program file is
   read (the guard can't be on during that read — the audit hook would
   block it) through input generation, the build/decode, and **every**
   `gc.collect()` (explicit or automatic). A `__del__` finalizer collected
   at any GC point — whether created at import time and made unreachable, or
   during the build — therefore can't import a metric-control module while
   the guard is off. On mem_infer GC also stays disabled through the
   correctness check, so a cyclic finalizer can't mutate a placeholder
   return into the right answer after the peak is sampled.
   For the store/index tasks (mem_kv, mem_index) the returned object is
   opaque — the candidate picks the data structure — so it can't be
   type-materialized like a list. Instead the evaluator **serves the full
   query workload INSIDE the measurement window** and samples memory after:
   the score is the memory needed to *answer* the workload, so a `build()`
   that returns a marker and defers the real construction (including a
   regenerate-and-cache) to the first `lookup()`/`query()` has that
   construction happen — and be measured — in-window. It gets no benefit.
3. **Full auditability**: every submission's exact source is recorded
   (hash-chained `submissions.jsonl` + `submissions/NNN.py`), so any
   escape gadget or hardcoded table is visible on review.
4. **Detection** (`bench audit`): scans recorded sources (with string
   constant-folding, so split-string obfuscation is caught) for the
   signatures of the escape/forgery routes, and flags physically
   implausible scores. This is the accepted mitigation for the one class
   that runtime enforcement cannot prevent (in-process frame-walking).
5. **Unseen-data validation** + reproducible records (`verify --rescore`):
   catch hardcoded/regenerated answers and record tampering.

## Guidance for reviewers

**Already known and accepted — NOT new findings.** Do not re-file these
as bugs; they are inherent to Option A and documented above:

- "A candidate can escape the in-process guard via
  `inspect`/`operator.attrgetter`/`string.Formatter`/frame-walking/split
  strings to reach builtins, `os`, evaluator internals, the result nonce,
  held-out data, or the instruction counter." — Known. In-process Python
  is not sandboxable; this is why detection + auditability + cooperative
  use are the guarantees, not prevention.
- "Held-out data / sealed scores are only obfuscated, decodable after an
  escape." — Known and intended (`bench/heldout.py` says so).
- "The nonce result protocol is forgeable after an escape." — Known;
  it only stops casual forgery, as its comments state.

**Actionable — please DO report these:**

- A forbidden import or file read that the **runtime** guard fails to
  block (not just the static scan) — the runtime layer is meant to be
  airtight for the import/file channels, on EVERY candidate call
  (measured/direct calls included) and across return-value consumption and
  the post-build `gc.collect()`.
- A candidate that moves real work or a metric-control access OUTSIDE the
  measurement window — a lazy return object consumed after measurement, a
  `__del__` finalizer, or (on the store/index tasks) a `build()` that
  returns a marker and defers construction to a post-measurement
  `lookup()`/`query()` — on a measured task. (Measured calls must
  materialize their return inside the window OR serve the full workload
  in-window, and keep the guard on across the post-build collect; a task
  that doesn't is a bug.)
- A realistic escape whose recorded source **`bench audit` does not
  flag** — detection is the accepted mitigation, so detection gaps are
  in scope. (Bonus: propose the signature.)
- Any **determinism break**, scoring bug, or a legitimate program
  wrongly rejected by a guard.
- A leak of hidden information through a channel that is supposed to be
  closed to a *cooperative* agent: error-message content, telemetry, the
  loop prompt, `bench evaluate` default output, filtered metrics.
- A false positive in `bench audit` on a plausible legitimate program.

## When to revisit this decision

If the benchmark is ever used to score submissions from untrusted or
unknown parties (not cooperative optimizers), Option A is insufficient
and Option B (out-of-process isolation, with coarser externally-measured
metrics) becomes necessary. That is a deliberate, owner-level change — not
a patch to the in-process guards.
