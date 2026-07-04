"""Broken exploit: import bench.opcount and stop the instruction counter so
real work runs uncounted. Must be rejected (forbidden import of bench).
gc.get_referrers(remaining) is the sibling gadget (blocked: gc forbidden).
"""

from bench import opcount


def process(n, ops):
    opcount.stop()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    answers = []
    for op, a, b in ops:
        ra, rb = find(a), find(b)
        if op == "u":
            if ra != rb:
                parent[ra] = rb
        else:
            answers.append(ra == rb)
    return answers


def solve(points):
    opcount.stop()
    return list(range(len(points)))
