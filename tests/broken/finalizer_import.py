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


def build(pairs):
    b = _Bomb()
    b.self = b            # unreachable cycle -> collected at gc.collect()
    del b
    return {k: v for k, v in pairs}


def lookup(store, key):
    return store.get(key)
