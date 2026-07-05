"""Strong reference for rule_list: embody the full decision list.

This is the author's ground-truth model -- the same ordered rule list the
generator uses. It handles the entire long exception tail, so its only
error is the injected label noise, defining the achievable floor. A
candidate cannot read this file; it must recover the rules from data +
validation feedback. (The rules are hardcoded rather than rebuilt with
`random`, because `random` transitively imports the forbidden `os`.)
"""

D = 8

# Ordered decision list, evaluated LAST-MATCH-WINS. First 5 are the broad
# head; the remaining 34 are narrow exception rules.
_RULES = [
    ('amax_in', (0, 1), 0), ('amax_in', (2, 3), 1), ('amax_in', (4, 5), 2),
    ('amax_in', (6, 7), 3), ('above_ge', 6, 3),
    ('and2_amin', ((0, 6), (1, 5), 0), 3),
    ('rank_gt_above', (0, 7, (5, 6), 3), 1),
    ('rank_gt_above', (3, 6, (4, 7), 2), 0),
    ('rank_gt_above', (0, 1, (6, 1), 3), 1),
    ('and2_amin', ((6, 0), (1, 5), 3), 0),
    ('and2_amin', ((3, 5), (0, 6), 0), 3),
    ('amax_above_gt', (5, 5, (6, 2)), 2),
    ('rank_gt_above', (3, 6, (2, 1), 2), 2),
    ('and2_above_eq', ((0, 6), (5, 3), 2), 0),
    ('and2_above_eq', ((1, 0), (2, 6), 5), 1),
    ('amax_above_gt', (6, 2, (5, 2)), 2),
    ('rank_gt_above', (5, 7, (0, 1), 2), 2),
    ('and2_above_eq', ((3, 7), (1, 6), 2), 1),
    ('and2_above_eq', ((5, 1), (2, 6), 5), 1),
    ('rank_gt_above', (0, 1, (1, 2), 4), 1),
    ('rank_gt_above', (1, 0, (3, 2), 2), 1),
    ('and2_above_eq', ((0, 3), (2, 7), 2), 2),
    ('and2_amin', ((2, 1), (5, 0), 7), 1),
    ('amax_above_gt', (1, 3, (0, 2)), 0),
    ('amax_above_gt', (2, 5, (5, 0)), 1),
    ('and2_amin', ((3, 2), (6, 5), 4), 3),
    ('rank_gt_above', (7, 6, (2, 5), 5), 3),
    ('rank_gt_above', (5, 0, (1, 2), 4), 1),
    ('and2_amin', ((5, 1), (3, 4), 6), 1),
    ('and2_amin', ((6, 4), (5, 2), 5), 0),
    ('and2_above_eq', ((5, 0), (6, 1), 4), 1),
    ('rank_gt_above', (0, 7, (1, 2), 3), 3),
    ('and2_above_eq', ((4, 2), (5, 3), 3), 3),
    ('and2_amin', ((4, 7), (5, 2), 5), 1),
    ('and2_amin', ((0, 5), (2, 4), 7), 3),
    ('and2_above_eq', ((2, 6), (3, 4), 5), 3),
    ('and2_above_eq', ((2, 0), (5, 4), 3), 1),
    ('amax_above_gt', (5, 5, (0, 4)), 3),
    ('and2_amin', ((5, 0), (3, 7), 5), 2),
]


def _stats(v):
    amax = amin = 0
    for i in range(1, D):
        if v[i] > v[amax]:
            amax = i
        if v[i] < v[amin]:
            amin = i
    mean = sum(v) / D
    above = sum(1 for x in v if x > mean)
    rank = [sum(1 for j in range(D) if v[j] < v[i]) for i in range(D)]
    return amax, amin, above, rank


def _match(kind, args, v, amax, amin, above, rank):
    if kind == "amax_in":
        return amax in args
    if kind == "above_ge":
        return above >= args
    if kind == "and2_above_eq":
        (a, b), (c, d), k = args
        return v[a] > v[b] and v[c] > v[d] and above == k
    if kind == "and2_amin":
        (a, b), (c, d), m = args
        return v[a] > v[b] and v[c] > v[d] and amin == m
    if kind == "rank_gt_above":
        i, rk, (j, k), cnt = args
        return rank[i] == rk and v[j] > v[k] and above == cnt
    if kind == "amax_above_gt":
        mx, cnt, (j, k) = args
        return amax == mx and above == cnt and v[j] > v[k]
    return False


def _label(v, rules):
    amax, amin, above, rank = _stats(v)
    label = 0
    for kind, args, lab in rules:
        if _match(kind, args, v, amax, amin, above, rank):
            label = lab
    return label


def fit(train_examples):
    # The model is the rule list; nothing to learn from train.
    pass


def predict(features):
    return _label(features, _RULES)
