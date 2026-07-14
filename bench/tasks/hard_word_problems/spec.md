# Task: hard_word_problems — compositional arithmetic reasoning

Write a deterministic, pure-Python, non-LLM program that solves challenging
multi-step arithmetic word problems. Minimize exact-answer error.

## Required API

```python
def solve(question):
    """Return the numeric answer to question (an int or equal float)."""
```

An exception or nonnumeric return counts as a wrong answer.

## Data

- **Train:** 600 fully visible, graded questions in `data/train.jsonl`.
- **Sealed test:** 2,400 questions, scored only after optimization.

Problems typically require four or more arithmetic decisions. Families include
reverse inventory chains, proportional transfers, changing work rates,
missing values implied by averages, elapsed-time schedules, successive
percentage changes, tiered pricing, ratio changes, and composite geometry.
Quantities may be written as digits, words, dozens, or mixed units. Irrelevant
sentences and reordered clauses occur frequently. The sealed split uses unseen
combinations of operation sequences and surface forms from the same underlying
families.

## Scoring

The optimization score is visible-train error rate, in `[0, 1]`. Numeric
answers use tolerance `1e-6`. The sealed test error is never exposed during a
run; it measures whether the solver learned reusable arithmetic semantics
rather than memorizing the 600 visible questions.

## Rules

- Python 3.12 standard library only; `re`, `math`, `fractions`, and
  `collections` are allowed.
- Forbidden: `open`, `os`, `io`, `sys`, `pathlib`, `mmap`, `ctypes`, `socket`,
  `subprocess`, `multiprocessing`, `threading`, `tempfile`, `shutil`, `sqlite3`,
  `dbm`, `shelve`, `importlib`, and `__import__`.
- The program receives only the question string and must be deterministic.
- The sealed data and generator are off-limits. Do not reconstruct or
  memorize hidden questions. Training-only memorization may lower the graded
  score, but it provides no value on the sealed test.
