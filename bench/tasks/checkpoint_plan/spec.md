# Task: checkpoint_plan - activation rematerialization planner

Choose activation checkpoint boundaries for transformer-like training
profiles. The cost model is architecture-agnostic and CPU-stable, but it
captures the deployment tradeoff between activation memory and recompute
work.

## Required API

```python
def plan(layers, memory_budget):
    """layers: list[dict] with act_mem and fwd_cost.
    memory_budget: max allowed activation memory units.
    Return list[int] of checkpoint boundary indices in 0..len(layers)."""
```

Boundary `0` is the input and boundary `len(layers)` is the output. A
stored boundary `i` with `0 < i < len(layers)` is charged as the
activation after layer `i - 1`. You may include the endpoints or omit
them; the evaluator normalizes them in.

## Scoring

A checkpoint plan splits the network into segments. During backward, each
segment is recomputed from its stored input checkpoint. Peak activation
memory is:

`sum(stored checkpoint activation memories) + max(segment activation sum)`

Plans over `memory_budget` are invalid. Score is total recompute forward
cost across deterministic model profiles: each segment recomputes its
interior activations during backward, while single-layer segments require
no recompute. Lower is better.

This is a perfect-information task. Validation profiles use the same
layer-count shapes as scored profiles. Candidate import-time code and
each planner call each have a 5,000,000 bytecode-instruction budget. The
evaluator scores an untouched copy of each profile, so mutating layer
dictionaries cannot change the memory or recompute model.
Programs are also limited to 12,000 bytes of source, no literal
container with more than 80 items, at most 300 total literal container
items, and no string/bytes literal over 2,000 bytes.

## Rules

- No imports. Programs run under a curated builtins subset. Available:
  `abs, all, any, bool, dict, enumerate, filter, float, int, isinstance,
  len, list, map, max, min, print, range, reversed, round, set, slice,
  sorted, str, sum, tuple, zip` plus common exception types
  (`Exception`, `ValueError`, `KeyError`, `IndexError`, `TypeError`,
  `RuntimeError`, `StopIteration`, `ZeroDivisionError`, `LookupError`,
  `BaseException`). Anything else — including class definitions and
  builtins like `getattr`, `iter`, `next`, `divmod`, `pow`, `hasattr`,
  `frozenset`, `bytes` — is unavailable and fails at run time. Use
  operators (`**`, `//`, `%`) instead of math functions.
- Forbidden (checked): filesystem/process/threading modules, `sys`,
  `bench`, `builtins`, `__builtins__`, `importlib`, `__import__`, and
  introspection/eval helpers such as `globals`, `locals`, `vars`, `dir`,
  `getattr`, `type`, `object`, `eval`, `exec`, `compile`, and traceback
  frame attributes.
- Must be deterministic.
