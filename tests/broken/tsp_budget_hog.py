"""Ignores the instruction budget: must be rejected with BudgetExceeded."""


def solve(points):
    n = len(points)
    x = 0
    for _ in range(10 ** 9):
        x = (x + 1) % 7
    return list(range(n))
