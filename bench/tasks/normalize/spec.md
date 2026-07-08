# Task: normalize — canonicalize messy duration strings

Write a **pure-Python, non-LLM** program that maps a messy, free-form
**duration** string to a single canonical normal form: the **total number
of whole seconds**, rendered as a plain decimal integer string.
Minimize the error rate.

## Required API (module-level function in program.py)

```python
def predict(raw):
    """raw: str, one duration in some surface format.
    Return the canonical form: the total whole seconds as a decimal
    integer string, e.g. "5400" for 'one hour thirty minutes'."""
```

Scoring is **exact string match** against the reference canonical form.
An exception inside `predict`, or any non-matching string, counts that
item as wrong.

## Data (train + test)

- **train (500 strings, fully visible and GRADED)**: `raw` AND
  `canonical` at `bench/tasks/normalize/data/train.jsonl` (one
  `{"raw": ..., "canonical": ...}` per line). You see the full train
  data and your train score, so you may study them, mine the format
  patterns, fit, and smoke-test on them freely.
- **hidden test (2000 strings)**: drawn from the same generator, but
  NEVER shown and NEVER reported during optimization; used afterwards to
  measure generalization. There is no validation split.

Train is SMALL relative to the diversity of the distribution, so unseen
combinations of value and surface format dominate the hidden test. Train
also **overweights the common formats**: rare surface formats occur far
more often in the hidden test than in train, so handling only what you
saw in bulk will not be enough.

Train and the hidden test share one distribution. The same duration value
is rendered in one of MANY surface formats, including:

- compact unit strings: `1h30m`, `2h`, `45m`, `1h30m15s`;
- single unit with a full word: `90 minutes`, `3 days`, `45 seconds`;
- decimal quantities: `1.5 hours`, `2.5 h`, `0.75 min`;
- spaced abbreviations: `1 h 30 m`, `4hr 28min`, `2 hrs 15 mins`;
- conjunction phrasings: `1 hour and 30 minutes`, `2 hours, 16 minutes`;
- spelled-out numbers: `ninety minutes`, `one hour thirty minutes`,
  `forty-eight seconds`;
- word fractions: `half an hour`, `a quarter of an hour`,
  `three quarters of an hour`, `an hour and a half`, `half a day`,
  `two and a half hours`;
- unicode fractions: `1½ hours`, `¼ hour`, `¾ h`;
- ISO-8601 durations: `PT1H30M`, `PT90M`, `PT2H`, `PT3H30M40S`;
- clock notation with three fields: `01:30:00`, `1:30:00` (H:MM:SS);
- prime / double-prime marks: `90'` (minutes), `30''` (seconds),
  `5' 30''`;
- thousands separators: `5,400 seconds`, `3,600 s`;
- locale decimal commas: `1,5 h`, `2,5 hrs` (i.e. 1.5 h, 2.5 h);
- idiomatic shorthands: `a couple of hours`;
- multi-unit forms: `2 hours, 15 minutes and 30 seconds`,
  `1d 6h`, `2 days and 3 hours`;
- weeks: `1 week`, `2 weeks`, `1wk`;
- noisy wrappers: `approx. 90 min`, `about 1 hour`, `~2h`,
  `duration: 1h30m`, `lasted 2 hours`, `90 minutes long`.

Unit conventions: `week`=604800 s, `day`=86400 s, `hour`=3600 s,
`minute`=60 s, `second`=1 s, and their usual abbreviations.

### One inherently ambiguous form (the error floor)

A slice of items uses a **bare `X:YY`** with no third field (e.g.
`5:30`). This is genuinely ambiguous — it can mean `H:MM` (5 h 30 m) or
`M:SS` (5 min 30 s) — and the intended reading was fixed by a hidden coin
when the data was generated, so it **cannot be recovered from the surface
string**. The majority reading is `H:MM`. Defaulting these to the
majority reading is the best any deterministic solver can do; the
resulting residual (~10% error) is the irreducible floor. Three-field
`H:MM:SS` is unambiguous and always parses normally.

## Scoring (lower is better)

Score = exact-match error rate = wrong / 500 on the visible **train**
set, in [0, 1]. You see this score and the full train data, so you can
fit and smoke-test against it freely.

The hidden test (2000 strings) is sealed: it is never shown and never
reported during optimization, and is scored only by the operator
afterwards. The gap between your train error and your hidden-test error
is the generalization signal. A low train error obtained by memorizing or
otherwise overfitting train `raw`→`canonical` pairs is pointless, because
every hidden-test `raw` string is unseen.

## Rules

- Python 3.12 stdlib only (`re`, `math`, `fractions`, ...). **Forbidden**
  (checked): `open`, `os`, `io`, `sys`, `pathlib`, `mmap`, `ctypes`,
  `socket`, `subprocess`, `multiprocessing`, `threading`, `tempfile`,
  `shutil`, `importlib`, `__import__`, `tracemalloc` — the program gets
  the `raw` string and nothing else; all knowledge must be in the source.
- Must be deterministic.
- Off-limits: the held-out test file (`data/heldout_test.bin`), the data
  generator (`tools/`), and any attempt to reconstruct hidden test
  instances. Only `data/train*` may be read while developing.
