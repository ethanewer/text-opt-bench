# Task: compress — lossless compressor, pure Python

Write a lossless compressor and decompressor in pure Python. The score is
the total size in bytes of your compressed output over a fixed corpus.

## Required API (module-level functions in program.py)

```python
def compress(data):
    """data: bytes. Return bytes."""

def decompress(blob):
    """blob: bytes. Must return exactly the original data."""
```

## Corpus

Four documents of roughly 150 KB each (~600 KB total), compressed
independently:

1. Web-server access logs (timestamped lines, IPs, request paths, statuses).
2. A JSON-lines dump of small records with a repeated schema.
3. English-like prose built from a limited vocabulary.
4. CSV sensor telemetry (ids, monotonically increasing timestamps,
   slowly-drifting numeric readings).

All four are highly redundant, each in a different way.

## Scoring (lower is better)

Score = `sum(len(compress(doc)) for doc in corpus)`. For every document,
`decompress(compress(doc))` must equal the original bytes exactly, or the
score is invalid.

**General compression only.** Your program is also run on a validation
corpus: unseen documents of the same four genres. It must round-trip them
exactly, and its compression ratio there must be in the same ballpark as
on the scoring corpus. Embedding, memorizing, or re-generating the known
scoring documents (or anything derived from the evaluator's seeds or
source) is cheating and will fail validation. Tuning your algorithm to
the *genres* (log-like, JSON-like, prose-like, CSV-like data) is fine and
encouraged.

## Rules

- Pure Python 3.12 stdlib, but **all compression codecs are forbidden**
  (checked): `zlib`, `gzip`, `bz2`, `lzma`, `zstd`, `compression`,
  `zipfile`, `tarfile`, `codecs`, `encodings`, plus `os`, `io`, `open`,
  `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`,
  `threading`, `importlib`, `__import__`. You must implement the algorithm
  yourself (e.g. RLE, LZ77/LZSS, Huffman, arithmetic/range coding, BWT,
  PPM — your choice).
- `bytes`/`bytearray`/`memoryview`, `struct`, `array`, `collections`,
  `itertools`, `math`, `heapq` etc. are all fine. `str.encode`/
  `bytes.decode` with standard encodings are fine.
- Must be deterministic.
- CPU budget for the whole evaluation (all compress + decompress calls):
  240 s. Stay well under it — pure-Python bit twiddling is slow, so mind
  your inner loops.
