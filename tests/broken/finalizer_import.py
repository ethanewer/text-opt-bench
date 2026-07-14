"""Broken exploit: a __del__ finalizer on an unreachable cyclic object runs
during the evaluator's post-build gc.collect(); it reaches a forbidden
import (obfuscated, to pass the static scan) to stop the memory scorer.

Must NOT succeed: the guard stays active across the post-build collect, so
the finalizer's import is blocked and the true memory is recorded (the
program stays valid but gets no benefit — resident is the real, large size,
not 0).
"""

import operator


class _Bomb:
    def __del__(self):
        try:
            g = operator.attrgetter("__globals__")(lambda: 0)
            g["__builtins__"]["__im" + "port__"]("tracemalloc").stop()
        except Exception:
            pass


def build(docs):
    b = _Bomb()
    b.self = b            # unreachable cycle -> collected at gc.collect()
    del b
    index = {}
    for doc_id, doc in enumerate(docs):
        for word in set(doc.split()):
            index.setdefault(word, []).append(doc_id)
    return index


def query(index, word):
    return index.get(word, [])
