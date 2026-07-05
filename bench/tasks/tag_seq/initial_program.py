"""Baseline tagger: ignore context, predict the single most common tag
seen anywhere in training for every token."""

from collections import Counter

_MAJORITY = "A"


def fit(train_examples):
    global _MAJORITY
    counts = Counter()
    for row in train_examples:
        counts.update(row["tags"])
    if counts:
        _MAJORITY = counts.most_common(1)[0][0]


def predict(tokens):
    return tuple(_MAJORITY for _ in tokens)
