# Task: tsp_budget — best TSP tours within a fixed instruction budget

Produce the shortest travelling-salesman tours you can, with computation
limited by a **deterministic budget of executed bytecode instructions**
(not wall-clock time).

## Required API (module-level function in program.py)

```python
def solve(points):
    """points: list[tuple[float, float]] — city coordinates in [0, 1]^2.
    Return a tour: a list of the indices 0..len(points)-1, each exactly
    once, in visiting order (the tour implicitly returns to the start)."""
```

## Workload

- 6 independent instances, 120 cities each, uniformly random in the unit
  square (fixed seeds). `solve` is called once per instance.

## Budget

- Each `solve` call may execute at most **8,000,000 Python bytecode
  instructions**, counted with `sys.monitoring`. Exceeding the budget
  raises `bench.opcount.BudgetExceeded` inside your code and invalidates
  the whole run — so leave a safety margin.
- You may check your consumption from inside the program:

  ```python
  from bench.opcount import remaining  # instructions left in the budget
  ```

  Call it periodically (e.g. once per improvement pass) and return your
  best-so-far tour before the budget runs out.
- Calls into C builtins (`math.dist`, `list.sort`, `min`, ...) count only
  as the instructions of the calling code, so vectorizing inner loops into
  builtins buys you more search within the budget.

**General algorithms only.** `solve` is also called on validation
instances with different seeds and sizes (under the same budget, not
scored); their tours must be valid and of reasonable quality. Hardcoding
or precomputing tours for the specific benchmark instances — including
anything derived from the evaluator's seeds or source — is cheating and
will fail validation. Data-independent precomputation at import time is
fine.

## Scoring (lower is better)

Score = sum over the 6 instances of the Euclidean length of your tour
(including the closing edge back to the start), rounded to 6 decimals.
Invalid tours (wrong length, repeated or missing indices) invalidate the
run.

## Rules

- Python 3.12 stdlib only, plus `bench.opcount.remaining`. **Forbidden**
  (checked): `sys`, `os`, `ctypes`, `socket`, `subprocess`,
  `multiprocessing`, `threading`, `signal`, `importlib`, `__import__`.
- Must be deterministic: any randomness must use a fixed seed
  (e.g. `random.Random(0)`).
- Avoid deep recursion (default recursion limit).
