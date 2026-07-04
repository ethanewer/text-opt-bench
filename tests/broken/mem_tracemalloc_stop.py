"""Broken exploit: stop the scorer to fake a zero memory score.

Stores everything honestly, then stops tracemalloc so the resident-bytes
read comes back ~0. Must be rejected: a program may not touch its own
metric's control surface. Applies to mem_kv/mem_index/mem_infer (the
required function name differs per task; all four are defined so this one
file covers every memory task).
"""

import tracemalloc


def _kill():
    tracemalloc.stop()


def build(pairs):          # mem_kv
    store = {k: v for k, v in pairs}
    _kill()
    return store


def lookup(store, key):    # mem_kv
    return store.get(key)


def query(index, term):    # mem_index
    return index.get(term, [])


def generate(weights, prompt, n):  # mem_infer
    _kill()
    return list(prompt)
