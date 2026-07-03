# Task: compress_heldout — lossless compression that must generalize

Same shape as a classic compressor task — pure-Python `compress` /
`decompress`, no codec libraries — but you are scored on documents you
cannot see. Only a training corpus is visible.

## Required API (module-level functions in program.py)

```python
def compress(data):
    """data: bytes. Return bytes."""

def decompress(blob):
    """blob: bytes. Must return exactly the original data."""
```

## Data splits (this is a train/validation/test task)

- **train (visible)**: four documents in
  `bench/tasks/compress_heldout/data/train_*.txt` — web-server logs,
  JSON-lines records, English-like prose, CSV telemetry (~75 KB each).
  Inspect them freely; tune dictionaries and models on them.
- **validation (hidden)**: four unseen documents of the same four genres
  (~60 KB each). Score = total compressed bytes over these; reported at
  every evaluation.
- **test (hidden)**: four more unseen documents, never reported during
  optimization; used afterwards to measure generalization.

## Scoring (lower is better)

Score = `sum(len(compress(doc)))` over the hidden validation corpus.
Every document in every evaluated split must round-trip exactly
(`decompress(compress(doc)) == doc`) or the run is invalid. Metrics also
report your compressed size on the visible train corpus — the ratio gap
is your overfitting signal. Baking train-derived dictionaries into your
source is legal; whether it generalizes is the game.

## Rules

- Pure Python 3.12 stdlib; **all compression codecs forbidden**
  (checked): `zlib`, `gzip`, `bz2`, `lzma`, `zstd`, `compression`,
  `zipfile`, `tarfile`, `codecs`, `encodings`; also `open`, `os`, `io`,
  `sys`, `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`,
  `threading`, `importlib`, `__import__`. The program gets bytes in,
  bytes out — any dictionaries must live in the source text.
- Must be deterministic.
- Off-limits: `data/heldout_*.bin`, the generator in `tools/`, and any
  attempt to reconstruct hidden documents. Only `data/train_*.txt` may
  be read while developing.
- CPU budget 300 s for the whole evaluation — pure-Python inner loops
  are slow; mind your complexity.
