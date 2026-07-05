# Task: tag_seq — per-token sequence labeling with an exception tail

Write a **pure-Python, non-LLM** program that labels each token of a
sequence with a **tag** (one of the letters `A B C D E F`). Minimize the
per-token error rate on a hidden validation split.

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

## Data splits (train / validation / test)

- **train (visible)** at `bench/tasks/tag_seq/data/train.jsonl` (one
  `{"tokens": [...], "tags": [...]}` per line). **The train split heavily
  UNDERSAMPLES the exception contexts**: the tokens that would trigger a
  tail rule are largely resampled away, so tail rules are far rarer in
  train than in the hidden splits. Capturing only what is common in train
  (i.e. the head mapping) leaves you well short — most of the remaining
  error lives in the exception tail, which you must recover from repeated
  validation feedback.
- **validation (hidden)**: never shown; every `--full` evaluation reports
  your per-token error rate on it. This is the score.
- **test (hidden)**: never seen, never reported during optimization; used
  afterward to measure generalization.

All splits share the same class/suffix scheme, the same head mapping, and
the same exception list. Only the exception *frequency* differs (train
undersamples it).

## Scoring (lower is better)

Score = validation per-token error rate = `wrong_tokens / total_tokens`, in
`[0, 1]`. A token is correct when its predicted tag exactly equals the gold
tag. The evaluation also reports your train error; the gap between train
and validation is your overfitting signal. Memorizing train sequences is
pointless — validation tokens are unseen strings, and train error is not
the score.

## Rules

- Python 3.12 stdlib only (`re`, `math`, `collections`, `statistics`, ...).
  **Forbidden** (checked): `open`, `os`, `io`, `sys`, `pathlib`, `mmap`,
  `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `shutil`, `importlib`, `__import__`, `tracemalloc` — `fit`
  gets the visible train rows and `predict` gets one token tuple; nothing
  else.
- Must be deterministic.
- Off-limits: the held-out data files (`data/heldout_*.bin`), the data
  generator (`tools/`), and any attempt to reconstruct the hidden
  sequences, tags, or the exception list. Only `data/train.jsonl` may be
  read while developing.
