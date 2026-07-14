"""Overflowing logits to NaN must not bypass the error-tolerance check."""


def generate(rt, weights, prompt, n_tokens):
    x = rt.embed(weights["wte"], weights["wpe"], prompt[0], 0)
    for _ in range(200):
        rt.add(x, x, out=x)
    output = []
    for _ in range(n_tokens):
        output.append(rt.argmax_vocab(x, weights["wte"]))
    return output
