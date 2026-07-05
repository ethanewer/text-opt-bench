# Task: mem_kv — memory-efficient key/value store

Implement an in-memory key/value store for string data that minimizes
**serving memory** — the peak memory used while answering exact-match
lookups — not just what is retained at rest.

## Required API (module-level functions in program.py)

```python
def build(pairs):
    """pairs: list[tuple[str, str]] of (key, value). Keys are unique.
    Return an opaque store object (any Python object)."""

def lookup(store, key):
    """Return the exact value string for `key`, or None if absent."""
```

## Data characteristics

- About 150,000 pairs.
- Keys are short ASCII strings of the form `category:decimal-id:hex-suffix`,
  e.g. `order:00012345:a3f9c2`. Keys are unique; categories repeat.
- Values are short one-line ASCII strings (roughly 60–160 chars) with a
  JSON-like record structure; there is substantial redundancy across values.

## Scoring (lower is better)

Score = the **peak** `tracemalloc`-traced bytes reached while serving the
lookup workload. The evaluator builds your store, resets the peak, then calls
`lookup` ~40,000 times (present and absent keys) INSIDE the measurement window
and samples the high-water mark. So the score charges both what you retain at
rest AND whatever each lookup transiently materializes: a store that holds a
tiny compressed blob but decompresses a large block on every lookup pays for
that block. The window opens **before your module is imported** and the input
`pairs` list is allocated **inside** it, so retained references to the input
strings count too. Every answer must be exactly correct, else the score is
invalid. Build-time transients are excluded (the peak is reset after `build`),
so a one-time expensive build is fine — only the serving footprint is scored.

Practical implication: aim for a structure that is small to hold AND cheap to
query. Heavy per-lookup decompression/reconstruction no longer helps — it
raises the serving peak.

## Rules

- Python 3.12 stdlib only. **Forbidden** (checked): `os`, `io`, `open`,
  `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `pathlib`, `shutil`, `sqlite3`, `dbm`, `shelve`,
  `importlib`, `__import__`. Everything must stay in process memory that
  tracemalloc can see. Compression codecs like `zlib`/`lzma`/`bz2` ARE
  allowed and are a legitimate tool here.
- Must be deterministic (no unseeded randomness, no time-dependent logic).
- Store the data you are given. After the measurement, `build`/`lookup`
  are also validated on an unseen dataset (different seed) — memorizing or
  re-generating the known scoring dataset instead of storing the input is
  cheating and fails validation.
- Lookups may be slower than a dict, but the whole evaluation must fit in
  150 s of CPU time.
