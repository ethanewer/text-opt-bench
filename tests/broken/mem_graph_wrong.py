"""Broken: returns out-neighbors WITHOUT deduping (duplicate edges leak
through) and unsorted — must be rejected as wrong."""


def build(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)   # no dedup, no sort
    return adj


def neighbors(index, u):
    return list(index.get(u, []))
