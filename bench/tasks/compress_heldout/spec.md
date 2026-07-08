# Task: compress_heldout — lossless compression that must generalize

Same shape as a classic compressor task — pure-Python `compress` /
`decompress`, no codec libraries — but you are scored on a small visible
training corpus, while a hidden test set of the same genres you cannot
see measures whether it generalizes. Only a training corpus is visible.

## Required API (module-level functions in program.py)

```python
def compress(data):
    """data: bytes. Return bytes."""

def decompress(blob):
    """blob: bytes. Must return exactly the original data."""
```

## Data (train + test)

- **train (visible, graded)**: four documents in
  `bench/tasks/compress_heldout/data/train_*.txt`, one per genre —
  web-server logs, JSON-lines records, English-like prose, CSV telemetry
  (~50 KB each). You see the full train data and your train score;
  inspect, fit, and smoke-test on it freely; tune dictionaries and models
  on them.
- **test (hidden)**: four unseen documents of the same four genres
  (~200 KB each), drawn from the same generator but never shown and never
  reported during optimization; used afterwards to measure
  generalization.
- No validation split.

The train corpus is small relative to the distribution's diversity, so
unseen combinations dominate the hidden test — fitting only what you saw
in train scores well on train but says little about the test.

## Scoring (lower is better)

Score = `sum(len(compress(doc)))` over the four visible **train**
documents. Every document must round-trip exactly
(`decompress(compress(doc)) == doc`) or the run is invalid. Baking
train-derived dictionaries into your source is legal; whether it
generalizes is the game. The hidden test is sealed and reported only to
the operator afterwards — the train-vs-test gap is your generalization
signal, so overfitting the train documents (e.g. memorizing them) buys
nothing on the hidden test.

## Rules

- Pure Python 3.12 stdlib; **all compression codecs forbidden**
  (checked): `zlib`, `gzip`, `bz2`, `lzma`, `zstd`, `compression`,
  `zipfile`, `tarfile`, `codecs`, `encodings`; also `open`, `os`, `io`,
  `sys`, `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`,
  `threading`, `importlib`, `__import__`. The program gets bytes in,
  bytes out — any dictionaries must live in the source text.
- Must be deterministic.
- Off-limits: `data/heldout_test.bin`, the generator in `tools/`, and any
  attempt to reconstruct hidden test items. Only `data/train_*.txt` may
  be read while developing.
- CPU budget 300 s for the whole evaluation — pure-Python inner loops
  are slow; mind your complexity.
