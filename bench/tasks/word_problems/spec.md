# Task: word_problems — combined arithmetic reasoning

Write a deterministic, pure-Python, non-LLM program that solves synthetic
arithmetic word problems. The task combines grade-school language diversity
with deeper compositional arithmetic. Minimize exact-answer error.

## Required API

```python
def solve(question):
    """Return the numeric answer to question (an int or equal float)."""
```

An exception, `None`, or a nonnumeric return counts as a wrong answer.

## Data

- **Train:** 1,100 fully visible, graded questions in `data/train.jsonl`:
  500 easy-regime and 600 hard-regime examples, deterministically shuffled.
- **Sealed test:** 4,400 questions: 2,000 easy-regime and 2,400 hard-regime
  examples, scored only after optimization.

The easy regime emphasizes varied natural-language rendering of arithmetic:
number words, pronouns, transfers, prices and change, rates, sharing,
percentages, comparisons, geometry, durations, idioms, and distractors. The
hard regime usually requires four or more decisions and adds inverse
inventory chains, proportional transfers, changing work rates, values implied
by averages, elapsed schedules, successive percentage changes, tiered prices,
ratio changes, composite geometry, and production/packing chains.

The sealed split uses independently generated wording and operation
combinations from the same two regimes. A useful solver must learn reusable
quantity and operation semantics rather than memorize the visible questions.

## Scoring

The optimization score gives the two former tasks equal weight:

`0.5 * (easy wrong / 500 + hard wrong / 600)`

in `[0, 1]`, rounded to six decimals. Easy and hard component errors are also
reported as diagnostics. Numeric answers use tolerance `1e-6`. The sealed
test applies the same 50/50 macro-average to its 2,000 easy and 2,400 hard
examples and is never exposed during a run.

## Rules

- Python 3.12 standard library only; `re`, `math`, `fractions`, and
  `collections` are allowed.
- Forbidden: `open`, `os`, `io`, `sys`, `pathlib`, `mmap`, `ctypes`, `socket`,
  `subprocess`, `multiprocessing`, `threading`, `tempfile`, `shutil`, `sqlite3`,
  `dbm`, `shelve`, `importlib`, and `__import__`.
- The program receives only the question string and must be deterministic.
- The sealed data and generators are off-limits. Do not reconstruct or
  memorize hidden questions. Visible-train memorization may lower the graded
  score, but the sealed test measures generalization.
