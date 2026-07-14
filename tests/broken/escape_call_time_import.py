"""Broken exploit: reach the real __import__ through a string-hidden route
and import a forbidden module INSIDE the candidate function body (not at
module import time), on the measured/scored call.

This is the call-time analogue of escape_runtime_import.py. It verifies the
runtime guard is active during the DIRECT candidate calls (word_problems,
tsp_budget, ops_connect, and the ML tasks call mod.<fn>() directly, not
through run_program) — not only at import time. The forbidden import of
'os' must be blocked whenever the guarded call runs.
"""

import operator


def _escape_import(name):
    g = operator.attrgetter("__globals__")(lambda: 0)
    return g["__builtins__"]["__import__"](name)


def build(pairs):
    _escape_import("os")
    return {k: v for k, v in pairs}


def lookup(store, key):
    return store.get(key)


def query(index, term):
    _escape_import("os")
    return []


def process(n, ops):
    _escape_import("os")
    return []


def solve(arg):
    # word_problems solve(question) and tsp_budget solve(points)
    _escape_import("os")
    try:
        return list(range(len(arg)))
    except TypeError:
        return 0


def order(requests, config):
    _escape_import("os")
    return [r["id"] for r in requests]


def plan(layers, memory_budget):
    _escape_import("os")
    return [0, len(layers)]


def partition(layers, n_stages, max_stage_mem, microbatches):
    _escape_import("os")
    return [min(n_stages - 1, i * n_stages // len(layers)) for i in range(len(layers))]


def schedule(tasks, n_nodes):
    _escape_import("os")
    return [t["id"] for t in tasks]


def generate(weights, prompt, n):
    _escape_import("os")
    return list(prompt)


def compress(data):
    _escape_import("os")
    return data


def decompress(blob):
    return blob
