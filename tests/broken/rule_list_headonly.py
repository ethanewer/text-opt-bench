"""Plausible ITERATION-1 attempt for rule_list: capture only the obvious
head rule.

A first pass over the train data reveals the dominant signal — the label
tracks WHICH feature is the argmax (grouped into pairs) — plus the one
broad "mostly above the mean -> class 3" rule. This handles the common
head of the distribution but ignores the long tail of narrow exception
rules, so it plateaus far above the floor. This is what a one-shot solver
writes; beating it requires discovering the exception tail over many
iterations.
"""

D = 8


def fit(train_examples):
    # Nothing to tune: the head mapping is read straight off the data.
    pass


def predict(features):
    v = features
    amax = 0
    for i in range(1, D):
        if v[i] > v[amax]:
            amax = i
    mean = sum(v) / D
    above = sum(1 for x in v if x > mean)
    # 5th head rule: strongly right-skewed rows -> class 3 (overrides).
    if above >= 6:
        return 3
    # 4 broad argmax-group rules.
    if amax in (0, 1):
        return 0
    if amax in (2, 3):
        return 1
    if amax in (4, 5):
        return 2
    return 3
