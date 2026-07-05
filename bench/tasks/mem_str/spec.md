# Task: mem_str — compact string-collection storage

Store a list of strings (heavy duplication and shared prefixes) so each can be
retrieved EXACTLY by its index, using as little **serving memory** (peak while
answering retrievals) as possible.

## Required API

```python
def build(strings):   # strings: list of str; return your index
def get(index, i):    # return the i-th string, exactly
```

## Scoring (lower is better)

Score = **peak traced bytes while serving** (tracemalloc peak, sampled after
the full retrieval workload runs, with the peak reset right after `build`) —
charging both retained bytes AND per-query transients, so decompressing a
large block on every `get()` does not help. ~100,000 strings drawn with heavy
duplication. The input list is allocated outside the traced window (reading it
is free; retaining per-string objects is not). `get()` runs for the whole
workload INSIDE the measured window, so deferring construction to the first
retrieval does not help. Build-time transients are excluded; only serving is
scored.

## Rules

- Python 3.12 stdlib only. **Forbidden**: `os`, `io`, `open`, `mmap`,
  `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `pathlib`, `shutil`, `sqlite3`, `dbm`, `shelve`, `importlib`,
  `__import__`, `tracemalloc`, `sys`, `resource`. `array`, `bytes`,
  deduplication, interning, prefix factoring — all fair game.
- Must return each string **exactly** (checked on members and on unseen
  validation data with a different seed).
- Off-limits: regenerating the scoring strings from their seed / hardcoding.
