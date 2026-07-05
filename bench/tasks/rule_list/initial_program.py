"""Baseline classifier: ignore the features, always predict the most
common label seen in training."""

from collections import Counter

_MAJORITY = 0


def fit(train_examples):
    global _MAJORITY
    counts = Counter(row["label"] for row in train_examples)
    _MAJORITY = counts.most_common(1)[0][0] if counts else 0


def predict(features):
    return _MAJORITY
