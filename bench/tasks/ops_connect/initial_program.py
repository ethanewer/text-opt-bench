"""Baseline: rebuild adjacency incrementally, BFS for every query."""

from collections import deque


def process(n, ops):
    adj = {i: [] for i in range(n)}
    answers = []
    for op, a, b in ops:
        if op == "u":
            adj[a].append(b)
            adj[b].append(a)
        else:
            seen = {a}
            frontier = deque([a])
            found = False
            while frontier:
                node = frontier.popleft()
                if node == b:
                    found = True
                    break
                for nxt in adj[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
            answers.append(found)
    return answers
