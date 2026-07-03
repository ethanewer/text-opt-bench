# Task: ops_connect — dynamic connectivity with minimal executed instructions

Answer a stream of union/connected operations on an undirected graph using
as few **executed Python bytecode instructions** as possible.

## Required API (module-level function in program.py)

```python
def process(n, ops):
    """n: number of nodes, labeled 0..n-1.
    ops: list of tuples. Each is either
        ("u", a, b)  — add an undirected edge between a and b
        ("q", a, b)  — query: are a and b currently in the same
                        connected component?
    Return a list[bool]: one answer per "q" op, in order."""
```

## Workload

- n ≈ 250 nodes, ~2,000 operations (about half unions, half queries),
  in a fixed deterministic order. Unions may repeat or be redundant.

## Scoring (lower is better)

Score = the exact number of Python bytecode instructions executed during
your `process` call, counted with `sys.monitoring`. The count is fully
deterministic for a deterministic program and independent of machine load.

Notes on the metric:
- Work done inside CPython builtins written in C (e.g. `list.sort`,
  `dict` operations, `str.split`) counts only as the instructions of the
  calling code, so both *better algorithms* and *pushing work into
  builtins* reduce the score.
- Only the `process(...)` call on the scoring instance is counted.

All answers must be exactly correct or the score is invalid.

**General algorithms only.** Before scoring, `process` is also run on
validation instances with different sizes and different data, and must be
exactly correct on all of them (those runs are not counted). Hardcoding,
memoizing, or precomputing answers or input data for the specific
benchmark instance — including anything derived from the evaluator's
seeds or source — is cheating and will fail validation. Data-independent
precomputation (lookup tables, etc.) is fine.

## Rules

- Python 3.12 stdlib only. **Forbidden** (checked): `sys`, `os`, `ctypes`,
  `socket`, `subprocess`, `multiprocessing`, `threading`, `signal`,
  `importlib`, `__import__` (no tampering with the instruction counter).
- Must be deterministic.
- Avoid deep recursion (recursion limit is the default 1000).
- CPU-time guard: 150 s (instruction counting slows execution ~10-30x;
  an O(n·m) approach still fits comfortably).
