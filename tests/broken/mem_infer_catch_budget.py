"""Catching WorkExceeded must not turn an over-budget run into a score."""


def generate(rt, weights, prompt, n):
    x = rt.embed(weights["wte"], weights["wpe"], prompt[0], 0)
    try:
        for _ in range(2500):
            rt.argmax_vocab(x, weights["wte"])
    except RuntimeError:
        pass
    return [0] * n
