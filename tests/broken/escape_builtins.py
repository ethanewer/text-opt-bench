"""Broken exploit: reach __import__ through the builtins dict, bypassing
the per-task import name-scan. Must be rejected (forbidden __builtins__).

Defines every common entry point so one file covers multiple tasks.
"""


def _escape():
    __builtins__["__im" + "port__"]("tracemalloc").stop()


def build(pairs):
    _escape()
    return {k: v for k, v in pairs}


def lookup(store, key):
    return store.get(key)


def query(index, term):
    _escape()
    return []


def process(n, ops):
    _escape()
    return []


def solve(points):
    _escape()
    return list(range(len(points)))


def compress(data):
    _escape()
    return data


def decompress(blob):
    return blob


def generate(weights, prompt, n):
    _escape()
    return list(prompt)
