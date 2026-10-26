# Task: `sat_cdcl_policy` — guide a deterministic SAT solver

Write a search policy for an evaluator-owned DPLL-style SAT solver with a
small, sound nogood-learning extension. The evaluator parses and solves every
formula, performs unit propagation, learns decision-path nogoods, validates
SAT assignments, and checks outcomes against each formula's construction.
Your program never emits a SAT/UNSAT answer. This compact core is deliberately
not a full production CDCL solver: it has no implication-graph analysis,
non-chronological backjumping, or learned-clause deletion.

This is a lightweight research prototype and a **legacy task**, not an
official alpha benchmark task.

## Required API

```python
def prepare(problem):
    """Return private policy state for one formula."""

def branch(search, state):
    """Return [variable, phase], where phase is 0 or 1."""

def restart(conflict, state):
    """Return a bool requesting a restart after this conflict."""
```

`problem` contains `family`, `n_variables`, `n_clauses`, and a clause-width
histogram. `search` contains immutable snapshots of the unassigned variables,
their positive and negative unresolved-clause occurrence counts,
Jeroslow-Wang weights, conflict activities, saved phases, and global search
counters. Variable indices start at 1.

`conflict` contains the conflicting clause, current decision literals, and
the same counters. Restarts retain learned clauses and saved phases. At most
16 policy-requested restarts are honored per formula.

Programs use a restricted set of Python builtins and cannot import modules.
Source is limited to 32 KB and large literal tables are rejected. The task
uses the repository's cooperative threat model: held-out files are lightly
obfuscated, not cryptographically secret, and reading or reconstructing them
is cheating. General policies only.

## Workloads

The deterministic suite contains compact CNF formulas from several standard,
structurally different families:

- planted random 3-SAT;
- graph 3-coloring;
- pigeonhole-principle formulas;
- XOR/parity systems encoded into CNF.

Development metadata and examples are visible in `data/train.json`. Online
feedback is computed on a disjoint sealed validation split. Final evaluation
also reports a separately sealed test split with independently generated and
relabelled instances. Matching shape grids across splits prevent shape alone
from revealing which split is running.

## Score

Lower is better. Solver work counts literal inspections during propagation,
branch decisions, and conflicts. It never uses elapsed time. For each formula,
work is capped at its fixed budget; an unsolved formula consumes the full
budget. `work / budget` is the area under that formula's unsolved indicator
over a uniformly selected work checkpoint. The split score is the mean of
this normalized effort plus 0.25 times its worst-quartile mean. Metrics also
report solved count, decisions, conflicts, restarts, and raw work.

The same formula always produces the same score. Invalid policy outputs,
exceptions, and incorrect solver outcomes invalidate the submission.

## Research provenance

This task is inspired by programmable policy choices in conflict-driven
clause-learning SAT solvers, especially VSIDS/EVSIDS branching, phase saving,
restart schedules, conflict-history branching (CHB), and learning-rate
branching (LRB). It is not an exact reproduction of those algorithms, a paper,
or a SAT Competition track: its chronological DPLL core, decision-nogood
learning, policy observations, and normalized-work objective are custom.

Primary background:

- Moskewicz et al., *Chaff: Engineering an Efficient SAT Solver* (DAC 2001),
  DOI 10.1145/378239.379017 — VSIDS and practical CDCL engineering.
- Audemard and Simon, *Predicting Learnt Clauses Quality in Modern SAT
  Solvers* (IJCAI 2009) — LBD-based learned-clause reasoning.
- Liang et al., *Learning Rate Based Branching Heuristic for SAT Solvers*
  (SAT 2016), DOI 10.1007/978-3-319-40970-2_8 — LRB.
- Liang et al., *Exponential Recency Weighted Average Branching Heuristic for SAT Solvers* (AAAI
  2016), DOI 10.1609/aaai.v30i1.10439 — CHB.

Results from this task must be labeled as this benchmark's local policy
simulation, not as paper or SAT Competition results.
