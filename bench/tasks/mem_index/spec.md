# Task: mem_index — memory-efficient inverted text index

Build an inverted index over a document collection that minimizes
**serving memory** — the peak memory used while answering term-lookup
queries — not just what is retained at rest.

## Required API (module-level functions in program.py)

```python
def build(docs):
    """docs: list[str]. Each doc is a line of space-separated lowercase
    ASCII words. Return an opaque index object."""

def query(index, term):
    """Return a sorted list[int] of the ids (positions in `docs`) of all
    documents that contain `term` as a whole word. Return [] if none."""
```

## Data characteristics

- About 20,000 documents, each 20–100 words.
- Vocabulary of roughly 25,000 distinct words with a skewed (Zipf-like)
  frequency distribution: a few words appear in thousands of documents,
  most appear in only a handful.

## Scoring (lower is better)

Score = the **peak** `tracemalloc`-traced bytes reached while serving the
query workload. The evaluator builds your index, resets the peak, then calls
`query` ~4,000 times (existing and missing terms) INSIDE the measurement
window and samples the high-water mark — charging both what you retain at rest
AND whatever each query transiently materializes (so decompressing a large
posting block per query pays for that block). The window opens **before your
module is imported**, and `docs` is allocated **inside** it, so retained
references to the input strings count. Every returned list must exactly equal
the sorted list of matching doc ids, else the score is invalid. Build-time
transients are excluded (peak reset after `build`); only serving is scored.

## Rules

- Python 3.12 stdlib only. **Forbidden** (checked): `os`, `io`, `open`,
  `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `pathlib`, `shutil`, `sqlite3`, `dbm`, `shelve`,
  `importlib`, `__import__`. Compression (`zlib` etc.), `array`, and
  compact encodings are allowed and encouraged.
- Must be deterministic.
- Index the documents you are given. After the measurement, `build`/`query`
  are also validated on an unseen dataset (different seed) — memorizing or
  re-generating the known scoring dataset instead of indexing the input is
  cheating and fails validation.
- Queries may be slower than a dict lookup, but the whole evaluation must
  fit in 150 s of CPU time.
