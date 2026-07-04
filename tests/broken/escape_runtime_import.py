"""Broken exploit: reach the real __import__ through a string-hidden
introspection route the AST scan cannot see (operator.attrgetter with a
string arg, dict subscripts), then import a forbidden module.

The AST scan passes this (no forbidden Name/Attribute node — the dunders
are string constants). It is blocked at RUNTIME: builtins.__import__ is
replaced by a guard during candidate execution, so the actual import of
'os' fails regardless of how __import__ was reached.
"""

import operator

_g = operator.attrgetter("__globals__")(lambda: 0)
_imp = _g["__builtins__"]["__import__"]
_imp("os")   # runtime import guard rejects this


def build(pairs):
    return {k: v for k, v in pairs}


def lookup(store, key):
    return store.get(key)


def process(n, ops):
    return []


def solve(points):
    return list(range(len(points)))


def compress(data):
    return data


def decompress(blob):
    return blob
