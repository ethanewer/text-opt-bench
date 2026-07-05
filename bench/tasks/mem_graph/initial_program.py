"""Baseline: a dict mapping each node to a Python list of its distinct,
sorted out-neighbors. Correct but memory-heavy (a dict entry + a Python list
of boxed ints per node)."""


def build(edges):
    adj = {}
    for u, v in edges:
        s = adj.get(u)
        if s is None:
            adj[u] = s = set()
        s.add(v)
    return {u: sorted(s) for u, s in adj.items()}


def neighbors(index, u):
    return list(index.get(u, []))
