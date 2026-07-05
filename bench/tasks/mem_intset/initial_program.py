"""Baseline: a Python set of the integers. Correct but memory-heavy
(a hash-table slot plus a boxed int object per element)."""


def build(ints):
    return set(ints)


def contains(index, x):
    return x in index
