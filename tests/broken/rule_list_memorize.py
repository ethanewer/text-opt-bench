"""Overfit fixture for rule_list: memorize the visible train rows.

Builds an exact lookup table from train feature-vectors to labels and
falls back to the train majority class. Because every row is generated
with its own random center and scale, validation rows never coincide with
train rows numerically -> the table never hits on hidden data, so this
scores at the majority-class baseline: FAR above the reference floor,
proving the generalization gap (memorizing train does not transfer).
"""

from collections import Counter

_TABLE = {}
_MAJORITY = 0


def _key(features):
    return tuple(round(x, 6) for x in features)


def fit(train_examples):
    global _MAJORITY
    counts = Counter()
    for row in train_examples:
        _TABLE[_key(row["features"])] = row["label"]
        counts[row["label"]] += 1
    _MAJORITY = counts.most_common(1)[0][0] if counts else 0


def predict(features):
    return _TABLE.get(_key(features), _MAJORITY)
