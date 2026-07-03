# Task: rl_async_sched - asynchronous distributed RL scheduling

Schedule rollout and learner-update work across a simulated 8-node RL
cluster. The workload models a real RLHF/RLAIF-style engine where rollout
sequence lengths vary heavily, so stragglers and learner lag matter.

## Required API

```python
def schedule(tasks, n_nodes):
    """tasks: list[dict], each with id, kind, ready, duration, deps.
    n_nodes: number of identical worker nodes.
    Return a list[int] giving task ids in dispatch-priority order."""
```

The evaluator treats the returned list as a dispatch-priority order. When
a node is free, it dispatches the highest-priority task whose `ready` time
has arrived and whose dependencies have all been *dispatched*. If those
dependencies have not finished running yet, the dispatched task occupies
the node and waits: it starts at max(now, ready, latest dependency finish
time). Dispatching an update too early therefore parks a node that could
have been running rollouts.

## Workload

There are 8 nodes. Rollout tasks have short, medium, and very long
sequence lengths mixed together. Learner update tasks depend on rollout
groups and have all-reduce-like overheads.

## Scoring

Score is a deterministic simulated cost:

`makespan + 0.10 * mean_rollout_completion + 0.35 * mean_update_lag`

Lower is better. Invalid or missing task ids fail. Candidate import-time
code and each scheduler call each have a 5,000,000 bytecode-instruction
budget. The evaluator scores an untouched copy of each trace, so mutating
the input task dictionaries cannot change the workload being measured.
Programs are also limited to 12,000 bytes of source, no literal
container with more than 80 items, at most 300 total literal container
items, and no string/bytes literal over 2,000 bytes.

This is a perfect-information task: the fixed traces are the deployment
workload. Validation traces check that the scheduler is not only valid but
also at least roughly heuristic-quality on different seeds.

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
