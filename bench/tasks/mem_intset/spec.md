# Task: mem_intset — compact integer-set membership

Store a set of integers (a clustered distribution: dense runs plus sparse
noise, over a universe of ~5,000,000) so that membership queries are answered
exactly, using as little **serving memory** (peak while answering queries) as
possible.

## Required API

```python
def build(ints):      # ints: list of distinct ints (shuffled); return your index
def contains(index, x):   # return True iff x is in the set
```

## Scoring (lower is better)

Score = **peak traced bytes while serving** (tracemalloc peak, sampled after
the full query workload runs, with the peak reset right after `build`) —
charging both retained bytes AND per-query transients, so decompressing a
large block on every `contains()` does not help. ~150,000 members. The input
list is allocated outside the traced window (reading it is free).
`contains()` is called for the whole workload INSIDE the measured window, so
deferring construction to the first query does not help. Build-time transients
are excluded; only the serving footprint is scored.

## Rules

- Python 3.12 stdlib only. **Forbidden**: `os`, `io`, `open`, `mmap`,
  `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `pathlib`, `shutil`, `sqlite3`, `dbm`, `shelve`, `importlib`,
  `__import__`, `tracemalloc`, `sys`, `resource`. `array`, `struct`, bytes,
  bitmaps, delta/varint, run-length — all fair game.
- Must be **exact**: no false positives or false negatives (membership is
  checked on both members and non-members, incl. unseen validation data).
- Off-limits: regenerating the scoring set from its seed / hardcoding.
