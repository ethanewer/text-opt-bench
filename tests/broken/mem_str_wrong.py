"""Broken: returns a truncated / altered string (drops the last char) — must
be rejected on the exact-retrieval check."""


def build(strings):
    return [s[:-1] for s in strings]


def get(index, i):
    return index[i]
