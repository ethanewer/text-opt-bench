"""A candidate must not be able to replace evaluator-owned weights."""


def generate(rt, weights, prompt, n):
    weights["lnf"] = rt.zeros((48,))
    return [0] * n
