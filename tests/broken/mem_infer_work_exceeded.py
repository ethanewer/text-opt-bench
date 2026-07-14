"""Must fail the deterministic work ceiling before returning."""


def generate(rt, weights, prompt, n_tokens):
    x = rt.embed(weights["wte"], weights["wpe"], prompt[0], 0)
    for _ in range(2200):
        y = rt.linear(x, weights["wte"])
        rt.free(y)
    return [0] * n_tokens
