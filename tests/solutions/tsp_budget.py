"""Reference improved solution: nearest neighbour + 2-opt until budget low.

`remaining()` is injected into the program namespace by the evaluator
(expose_budget) — no import needed; importing bench is forbidden.
"""

import math


def solve(points):
    n = len(points)
    dist = math.dist
    unvisited = set(range(1, n))
    tour = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist(points[cur], points[j]))
        unvisited.remove(nxt)
        tour.append(nxt)
        cur = nxt

    improved = True
    while improved and remaining() > 1_500_000:
        improved = False
        for i in range(n - 1):
            if remaining() < 800_000:
                return tour
            a, b = points[tour[i]], points[tour[i + 1]]
            for j in range(i + 2, n):
                c = points[tour[j]]
                d = points[tour[(j + 1) % n]]
                if dist(a, c) + dist(b, d) < dist(a, b) + dist(c, d) - 1e-12:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    b = points[tour[i + 1]]
                    improved = True
    return tour
