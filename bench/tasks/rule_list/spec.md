# Task: rule_list — decision-list classification with a long exception tail

Write a **pure-Python, non-LLM** program that learns to classify rows of
numeric features into one of **K = 4** classes. Minimize the error rate on
the visible train set.

## Required API (module-level functions in program.py)

```python
def fit(train_examples):
    """train_examples: a list of {"features": [float]*8, "label": int}.
    Called once. Store whatever state you need (module globals, etc.)."""

def predict(features):
    """features: a list of 8 floats (one row). Return the class label
    (an int in 0..3)."""
```

An exception inside `predict`, or a non-matching return, just counts that
row as wrong.

## The label-generating process

Each row has **D = 8 real features**. The true label is produced by an
**ordered decision list** over the features. Crucially, the label depends
**only on the WITHIN-ROW ORDER STRUCTURE** of the features, never on their
absolute magnitudes: **every row is generated with its own random center
and scale** (`features ≈ center + scale · noise`), so two rows of the same
class can live in completely different numeric ranges. A model that
thresholds raw feature values (`feature_3 > 4.2`) will *not* generalize;
only order/relational predicates carry signal. The predicate family used by
the decision list is:

- **which feature is the argmax / argmin** of the row;
- **within-row comparisons** `features[i] > features[j]`;
- the **count of features above the row's own mean**;
- the **within-row rank** of a given feature (how many features it exceeds).

The decision list is evaluated **last-match-wins**: the **first, broad
rules** (the *head*) capture the common general structure and set a base
label; a **long tail of narrow, idiosyncratic rules** then **overrides**
that base label for small, specific feature regions. There are dozens of
these exception rules; each fires on only a small fraction of rows, and
different exceptions apply to different regions.

A small amount of **label noise** is injected, so even a perfect model of
the rules cannot reach 0 error — there is a Bayes floor strictly above zero.

## Data (train + test)

- **train (1200 rows, fully visible and graded)** at
  `bench/tasks/rule_list/data/train.jsonl` (one
  `{"features": [...], "label": ...}` per line). You see every train row
  and its label, and your score is measured on exactly these rows — study,
  fit, and smoke-test on them freely.
- **hidden test (4800 rows)**: drawn from the same generator and the same
  decision list, but never shown and never reported during optimization.
  It is used afterward to measure generalization.
- There is **no validation split**.

The train set is **small relative to the diversity of the distribution**:
the decision list's long exception tail means many feature-order
combinations never appear in your 1200 rows, so unseen combinations
dominate the hidden test. Fitting only what you saw in train will score
well on train but leave the exception tail — where most of the hidden-test
error lives — uncovered.

## Scoring (lower is better)

Score = train error rate = `wrong / 1200` on the visible train set, in
`[0, 1]`. A prediction is correct when it exactly equals the reference
label. The hidden test set is **sealed**: its error is reported only to the
operator afterward, never during optimization. The gap between your train
score and the hidden-test error is the **generalization signal** — because
the score is measured on rows you can see, memorizing train labels drives
the train score to zero but does nothing on the hidden test; only rules
that generalize transfer.

## Rules

- Python 3.12 stdlib only (`math`, `statistics`, `collections`, ...).
  **Forbidden** (checked): `open`, `os`, `io`, `sys`, `pathlib`, `mmap`,
  `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `importlib`, `__import__`, `tracemalloc` — `fit` gets the visible train
  rows and `predict` gets one feature row; nothing else.
- Must be deterministic.
- Off-limits: the held-out test file (`data/heldout_test.bin`), the data
  generator (`tools/`), and any attempt to reconstruct the hidden test
  rows, labels, or the decision list. Only `data/train*` may be read while
  developing.
