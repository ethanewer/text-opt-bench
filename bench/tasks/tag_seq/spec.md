# Task: tag_seq — per-token sequence labeling with an exception tail

Write a **pure-Python, non-LLM** program that labels each token of a
sequence with a **tag** (one of the letters `A B C D E F`). Minimize the
per-token error rate on the visible train set.

## Required API (module-level functions in program.py)

```python
def fit(train_examples):
    """train_examples: a list of {"tokens": [str, ...], "tags": [str, ...]}
    (tags[i] is the gold tag for tokens[i]). Called once. Store whatever
    state you need (module globals, etc.)."""

def predict(tokens):
    """tokens: a tuple of token strings (one sequence). Return a sequence
    of the SAME length: one tag (a letter "A".."F") per token."""
```

An exception inside `predict`, a wrong-length return, or a non-matching
tag counts the affected token(s) as wrong.

## The label-generating process

Each token is a made-up word of the form **`stem + suffix`**. The **last
two characters** (the *suffix*) determine the token's **class** (one of
`C = 12`). The stem is random and carries no signal — and stems almost
never repeat across sequences or splits, so a token STRING you saw in
train will essentially never reappear in the hidden data. **Memorizing
`token -> tag` will not generalize**; the signal lives in the suffix
(class) and the local context.

- **The general rule (head).** A token's tag is *usually* a fixed function
  of its own class: `tag = BASE[class]`. This broad mapping is easy to read
  off the data and gets most tokens right.
- **The exception tail (the hard part).** A long, ordered list of narrow
  **exception rules** OVERRIDES the general tag when a token appears in a
  specific **local class context**. Each exception is keyed on things like:
  - the class of the **previous** token and the current class (a bigram);
  - the current class and the class of the **next** token;
  - a **three-token** class context (prev, current, next);
  - the class **two** tokens back and the current class (a skip-bigram);
  - the current class together with being the **first** / **last** token.

  The list is evaluated **last-match-wins**. There are **dozens** of these
  rules; each fires on only a small fraction of tokens, the override tags
  are **idiosyncratic** (no formula generates them), and different
  exceptions apply to different contexts. Because they are keyed on
  **classes** (not specific stems), the exceptions **generalize**: they fire
  on hidden sequences built from unseen tokens.

A small amount of per-token **label noise** is injected, so even a perfect
model of the rules cannot reach 0 error — there is a Bayes floor strictly
above zero.

## Data (train + test)

- **train (visible, graded)** at `bench/tasks/tag_seq/data/train.jsonl`
  (one `{"tokens": [...], "tags": [...]}` per line): **500 sequences**. The
  full train data is visible — you may study, fit, and smoke-test on it
  freely — and your score is measured on it.
- **test (hidden)**: **2000 sequences** drawn from the same generator, never
  shown and never reported during optimization; used afterward to measure
  generalization.
- There is **no validation split**.

Train and test share the same class/suffix scheme, the same head mapping,
and the same exception list. Train is **small relative to the
distribution's diversity**: 500 sequences cover only a fraction of the
possible class contexts and exception combinations, so unseen combinations
dominate the hidden test.

## Scoring (lower is better)

Score = per-token error rate on the **visible train set** =
`wrong_tokens / total_tokens`, in `[0, 1]`. A token is correct when its
predicted tag exactly equals the gold tag. The hidden test is **sealed**:
its error is reported only to the operator afterward, never during
optimization. The gap between your train error and the hidden-test error is
the generalization signal. Fitting only what you saw in train will score
well on train, but the hidden test measures whether your rules actually
generalize — memorizing or overfitting train is pointless for the test.

## Rules

- Python 3.12 stdlib only (`re`, `math`, `collections`, `statistics`, ...).
  **Forbidden** (checked): `open`, `os`, `io`, `sys`, `pathlib`, `mmap`,
  `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `shutil`, `importlib`, `__import__`, `tracemalloc` — `fit`
  gets the visible train rows and `predict` gets one token tuple; nothing
  else.
- Must be deterministic.
- Off-limits: the held-out test file (`data/heldout_test.bin`), the data
  generator (`tools/`), and any attempt to reconstruct the hidden test
  sequences, tags, or the exception list. Only `data/train*` may be read
  while developing.
