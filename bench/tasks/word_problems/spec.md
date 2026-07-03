# Task: word_problems — programmatic math word-problem solver

Write a **pure-Python, non-LLM** program that solves grade-school math
word problems (GSM8K-style): multi-step arithmetic dressed in natural
language. Minimize the error rate.

## Required API (module-level function in program.py)

```python
def solve(question):
    """question: str, one word problem ending in a question.
    Return the numeric answer (int, or float equal to it)."""
```

An exception inside `solve` just counts that question as wrong.

## Data splits (this is a train/validation/test task)

- **train (100 problems, fully visible)**: questions AND answers at
  `bench/tasks/word_problems/data/train.jsonl` (one
  `{"question": ..., "answer": ...}` per line). Study them, mine
  patterns, tune on them freely — but note the train set is SMALL
  relative to the variety of the distribution: phrasings, idioms, and
  structure combinations WILL appear in the hidden splits that never
  occur in train. Handling only what you saw will not be enough.
- **validation (250 problems, hidden)**: you never see these questions;
  every evaluation reports your error rate on them. This is the score.
- **test (600 problems, fully hidden)**: never seen, never reported
  during optimization; used afterwards to measure generalization.

All three splits are drawn from the same distribution: a broad family of
arithmetic structures (multi-entity quantity chains with transfers,
prices and change, rates and distances, sharing with remainders, percent
changes and discounts, comparisons, areas and fencing, savings with
daily/weekly rates, packing and reading with round-up division,
multi-purchase totals, two-stage problems, fractions of groups, round
trips, coin values, ratios, durations in weeks and days) with varied
names, objects, numbers, and phrasings. Beware: numbers up to
ninety-nine may be written as words; idiomatic quantity changes occur
("doubles their pile", "loses all but four", "gives half of them away",
"half as many again", "three short of double"); "they" can refer to the
previous subject; and many problems contain a distractor sentence with
an irrelevant quantity.

## Scoring (lower is better)

Score = validation error rate = wrong_answers / 250, in [0, 1].
Answers count as correct when numerically equal to the reference
(tolerance 1e-6). The evaluation also reports your train error in the
metrics — the gap between them is your overfitting signal.

## Rules

- Python 3.12 stdlib only (`re`, `math`, `fractions`, `collections`, ...).
  **Forbidden** (checked): `open`, `os`, `io`, `sys`, `pathlib`, `mmap`,
  `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `importlib`, `__import__` — the program gets the question string and
  nothing else; all knowledge must be baked into the source.
- Must be deterministic.
- Off-limits: the held-out data files (`data/heldout_*.bin`), the data
  generator (`tools/`), and any attempt to reconstruct hidden questions
  or answers. Only `data/train.jsonl` may be read while developing.
  Hardcoding train answers is pointless — train error is not the score.
