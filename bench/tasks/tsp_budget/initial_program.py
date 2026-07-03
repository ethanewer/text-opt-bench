"""Baseline TSP: plain nearest-neighbour from city 0."""

import math


def solve(points):
    n = len(points)
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: math.dist(points[current], points[j]))
        unvisited.remove(nxt)
        tour.append(nxt)
        current = nxt
    return tour
