# Task: mem_graph — compact directed-graph neighbor index

Store a directed graph (given as an edge list, with duplicate edges) so that
out-neighbor queries are answered exactly, using as little retained memory as
possible.

## Required API

```python
def build(edges):
    """edges: list of (int u, int v) tuples (may contain duplicates).
    Return your index object."""

def neighbors(index, u):
    """Return the sorted list of DISTINCT out-neighbors of node u
    ([] if u has no out-edges)."""
```

## Scoring (lower is better)

Score = **resident traced bytes** of your index after `build` (tracemalloc
current allocation, sampled after the full query workload has run). The graph
has ~30,000 nodes and ~300,000 edges, with duplicate edges and hub skew
(so it compresses well). The input edge list is allocated outside the traced
window — reading it is free; copying it is not.

`neighbors()` is called for the whole query workload **inside** the measured
window, so returning a marker from `build()` and constructing the real index
on the first query does not help — that construction is measured too.

## Rules

- Python 3.12 stdlib only. **Forbidden** (checked): `os`, `io`, `open`,
  `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `pathlib`, `shutil`, `sqlite3`, `dbm`, `shelve`, `importlib`,
  `__import__`, `tracemalloc`, `sys`, `resource`. `array`, `struct`, bytes
  packing, delta/varint encoding, in-place reuse — all fair game.
- Must be deterministic. `neighbors(index, u)` must return the sorted list of
  distinct out-neighbor ids.
- Off-limits: regenerating the scoring graph from its seed, or hardcoding
  answers. Validation on a different-seed graph checks that your index holds
  the edges it is given.
