# Task: mem_index — memory-efficient inverted text index

Build an inverted index over a document collection that minimizes
**resident memory** while answering term-lookup queries correctly.

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

Score = Python-allocated bytes still resident after `build` returns
(`tracemalloc` traced-current, after the input list is deleted and
`gc.collect()` runs). The measurement window opens **before your module is
imported**, and `docs` is allocated **inside** the window — retained
references to the input strings count against you.

After measurement, `query` is called ~4,000 times (existing and missing
terms). Every returned list must exactly equal the sorted list of matching
doc ids, else the score is invalid.

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
