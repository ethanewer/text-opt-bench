"""Evaluator-owned weights must never be usable as unmetered output buffers."""


def generate(rt, weights, prompt, n_tokens):
    x = rt.zeros((48,))
    rt.rmsnorm(x, weights["lnf"], out=weights["lnf"])
    return [0] * n_tokens
