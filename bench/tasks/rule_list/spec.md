# Task: rule_list — decision-list classification with a long exception tail

Write a **pure-Python, non-LLM** program that learns to classify rows of
numeric features into one of **K = 4** classes. Minimize the error rate on
a hidden validation split.

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

## Data splits (train / validation / test)

- **train (300 rows, fully visible)** at
  `bench/tasks/rule_list/data/train.jsonl` (one
  `{"features": [...], "label": ...}` per line). **The train split
  UNDERSAMPLES the exception rows**: the rare tail rules are far less
  frequent here than in the hidden splits. Capturing only what is common
  in train will leave you well short — the exception tail is where most of
  the remaining error lives.
- **validation (600 rows, hidden)**: never shown; every full evaluation
  reports your error rate on them. This is the score.
- **test (1500 rows, fully hidden)**: never seen, never reported during
  optimization; used afterward to measure generalization.

All splits share the same distribution and the same decision list.

## Scoring (lower is better)

Score = validation error rate = `wrong / 600`, in `[0, 1]`. A prediction is
correct when it exactly equals the reference label. The evaluation also
reports your train error; the gap between train and validation is your
overfitting signal.

## Rules

- Python 3.12 stdlib only (`math`, `statistics`, `collections`, ...).
  **Forbidden** (checked): `open`, `os`, `io`, `sys`, `pathlib`, `mmap`,
  `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `importlib`, `__import__`, `tracemalloc` — `fit` gets the visible train
  rows and `predict` gets one feature row; nothing else.
- Must be deterministic.
- Off-limits: the held-out data files (`data/heldout_*.bin`), the data
  generator (`tools/`), and any attempt to reconstruct the hidden rows,
  labels, or the decision list. Only `data/train.jsonl` may be read while
  developing. Memorizing train rows is pointless — validation rows never
  coincide with train rows, and train error is not the score.
