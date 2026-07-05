"""Baseline: keep the strings in a Python list, retrieved by index.
Correct but memory-heavy: a str object (with per-object overhead) per entry,
even when many entries are duplicates or share long prefixes."""


def build(strings):
    return list(strings)


def get(index, i):
    return index[i]
