"""Overfit fixture for tag_seq: memorize the visible train tokens.

Builds an exact lookup table from the full token STRING to its most common
train tag, falling back to the global majority tag. Because stems are random
and almost never repeat across sequences or splits, validation tokens are
unseen strings -> the table almost never hits on hidden data, so this scores
near the global-majority baseline: FAR above the reference floor (and worse
than even the head-only mapping), proving the generalization gap
(memorizing train tokens does not transfer).
"""

from collections import Counter, defaultdict

_TABLE = {}
_GLOBAL = "A"


def fit(train_examples):
    global _GLOBAL
    by_tok = defaultdict(Counter)
    allc = Counter()
    for row in train_examples:
        for tk, tg in zip(row["tokens"], row["tags"]):
            by_tok[tk][tg] += 1
            allc[tg] += 1
    for tok, ctr in by_tok.items():
        _TABLE[tok] = ctr.most_common(1)[0][0]
    if allc:
        _GLOBAL = allc.most_common(1)[0][0]


def predict(tokens):
    return tuple(_TABLE.get(tk, _GLOBAL) for tk in tokens)
