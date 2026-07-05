"""Plausible ITERATION-1 attempt for tag_seq: learn only the general head
mapping (suffix -> majority tag).

A first pass over the train data reveals the dominant signal: a token's tag
is almost always a fixed function of its suffix (class). Grouping train
tokens by their last two characters and taking the majority tag recovers the
broad head mapping BASE. This handles the common head of the distribution
but ignores the entire long tail of narrow, context-keyed exception rules
(which train barely contains), so it plateaus far above the floor. This is
what a one-shot solver writes; beating it requires discovering the exception
tail over many iterations of validation feedback.
"""

from collections import Counter, defaultdict

_MAJ_BY_SUF = {}
_GLOBAL = "A"


def fit(train_examples):
    global _GLOBAL
    by_suf = defaultdict(Counter)
    allc = Counter()
    for row in train_examples:
        for tk, tg in zip(row["tokens"], row["tags"]):
            by_suf[tk[-2:]][tg] += 1
            allc[tg] += 1
    for suf, ctr in by_suf.items():
        _MAJ_BY_SUF[suf] = ctr.most_common(1)[0][0]
    if allc:
        _GLOBAL = allc.most_common(1)[0][0]


def predict(tokens):
    return tuple(_MAJ_BY_SUF.get(tk[-2:], _GLOBAL) for tk in tokens)
