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

## Data (train + test)

- **train (500 problems, fully visible & graded)**: questions AND answers
  at `bench/tasks/word_problems/data/train.jsonl` (one
  `{"question": ..., "answer": ...}` per line). This is the graded set —
  study them, mine patterns, tune and smoke-test on them freely. But note
  the train set is SMALL relative to the variety of the distribution:
  phrasings, idioms, and structure combinations WILL appear in the hidden
  test that never occur in train, so unseen combinations dominate it.
  Handling only what you saw will not be enough.
- **test (2000 problems, fully hidden)**: never shown and never reported
  during optimization; used afterwards to measure generalization. There
  is no validation split.

Both train and test are drawn from the same distribution: a broad family of
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

Score = train error rate = wrong_answers / 500 on the fully visible train
set, in [0, 1]. Answers count as correct when numerically equal to the
reference (tolerance 1e-6).

The hidden test set is sealed: it is never shown or reported during
optimization and is scored only by the operator afterwards. The gap
between your train error and that hidden test error is your
generalization signal — fitting only what you saw in train will drive
train error down but says nothing about how you do on the hidden test.

## Rules

- Python 3.12 stdlib only (`re`, `math`, `fractions`, `collections`, ...).
  **Forbidden** (checked): `open`, `os`, `io`, `sys`, `pathlib`, `mmap`,
  `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `importlib`, `__import__` — the program gets the question string and
  nothing else; all knowledge must be baked into the source.
- Must be deterministic.
- Off-limits: the held-out test file (`data/heldout_test.bin`), the data
  generator (`tools/`), and any attempt to reconstruct the hidden test
  questions or answers. Only `data/train*` may be read while developing.
  Memorizing train answers will drive train error to zero but is
  pointless — the hidden test measures generalization.
