"""Strong reference for tag_seq: embody the full head mapping + the entire
ordered exception list.

This is the author's ground-truth model -- the same suffix table, head
mapping, and last-match-wins exception list the generator uses. It handles
the whole idiosyncratic exception tail, so its only error is the injected
per-token label noise, defining the achievable floor. A candidate cannot
read this file; it must recover the head and the ~72 exceptions from data +
validation feedback. (The tables are hardcoded literals rather than rebuilt
with `random`, because `random` transitively imports the forbidden `os`.)
"""

TAGS = ["A", "B", "C", "D", "E", "F"]

SUFFIXES = ['mu', 'gi', 'ko', 'va', 'zu', 'do', 'so', 'te', 'to', 'le',
            'vo', 'zi']
BASE = [2, 0, 4, 5, 2, 0, 3, 2, 3, 3, 0, 5]
RULES = [('pc', (0, 8), 3), ('pcn', (0, 0, 8), 0), ('cn', (6, 2), 2),
         ('pcn', (7, 8, 3), 5), ('pc', (0, 11), 5), ('cn', (3, 5), 0),
         ('pc', (9, 0), 2), ('pc', (1, 7), 0), ('pc', (10, 7), 1),
         ('pc', (4, 8), 4), ('skip', (4, 6), 2), ('pc', (9, 0), 2),
         ('cn', (2, 1), 1), ('cn', (7, 2), 4), ('first', (1,), 4),
         ('pc', (6, 8), 1), ('pc', (11, 5), 1), ('pc', (6, 11), 0),
         ('cn', (7, 3), 5), ('pc', (8, 4), 0), ('cn', (3, 0), 2),
         ('cn', (8, 7), 0), ('skip', (10, 1), 5), ('pc', (10, 10), 2),
         ('skip', (3, 5), 1), ('skip', (2, 8), 3), ('pc', (9, 5), 2),
         ('last', (10,), 3), ('skip', (7, 4), 1), ('cn', (0, 0), 0),
         ('pc', (9, 4), 3), ('cn', (0, 7), 1), ('cn', (4, 6), 0),
         ('cn', (2, 8), 2), ('skip', (0, 4), 5), ('cn', (4, 4), 4),
         ('pc', (3, 0), 4), ('last', (7,), 3), ('cn', (5, 7), 2),
         ('pc', (0, 2), 5), ('pc', (0, 1), 4), ('first', (11,), 4),
         ('pc', (10, 0), 3), ('pc', (10, 10), 4), ('pc', (1, 11), 5),
         ('cn', (1, 1), 1), ('pc', (8, 3), 3), ('pc', (4, 8), 2),
         ('pc', (11, 0), 4), ('pc', (7, 11), 5), ('cn', (0, 3), 4),
         ('cn', (10, 10), 1), ('skip', (1, 11), 5), ('cn', (8, 0), 5),
         ('skip', (6, 4), 1), ('pcn', (10, 11, 10), 0), ('skip', (11, 8), 3),
         ('last', (9,), 5), ('pc', (3, 10), 1), ('pc', (8, 11), 3),
         ('cn', (6, 4), 2), ('cn', (0, 10), 1), ('pc', (6, 9), 1),
         ('cn', (8, 3), 4), ('skip', (11, 11), 4), ('pc', (5, 9), 0),
         ('pc', (7, 9), 4), ('last', (2,), 2), ('pc', (3, 8), 2),
         ('cn', (5, 1), 3), ('cn', (3, 9), 5), ('cn', (10, 10), 3)]

_SUF = {s: i for i, s in enumerate(SUFFIXES)}


def _classes(tokens):
    return [_SUF.get(tk[-2:], -1) for tk in tokens]


def _tag_at(classes, i):
    L = len(classes)
    c = classes[i]
    if c < 0:
        return 0
    p = classes[i - 1] if i - 1 >= 0 else -1
    n = classes[i + 1] if i + 1 < L else -1
    pp = classes[i - 2] if i - 2 >= 0 else -1
    tag = BASE[c]
    for form, params, t in RULES:
        if form == "pc":
            if p == params[0] and c == params[1]:
                tag = t
        elif form == "cn":
            if c == params[0] and n == params[1]:
                tag = t
        elif form == "skip":
            if pp == params[0] and c == params[1]:
                tag = t
        elif form == "pcn":
            if p == params[0] and c == params[1] and n == params[2]:
                tag = t
        elif form == "first":
            if i == 0 and c == params[0]:
                tag = t
        elif form == "last":
            if i == L - 1 and c == params[0]:
                tag = t
    return tag


def fit(train_examples):
    # The model is the head mapping + exception list; nothing to learn.
    pass


def predict(tokens):
    classes = _classes(tokens)
    return tuple(TAGS[_tag_at(classes, i)] for i in range(len(classes)))
