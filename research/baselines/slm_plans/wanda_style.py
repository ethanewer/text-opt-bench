"""Activation-weighted pruning adapter; not a full Wanda reproduction."""


def plan(layers, target_bits):
    bits = 4 if target_bits < 3.5 else 6
    result = []
    for layer in layers:
        columns = layer[3]
        prune = (columns // 2 + 1) / columns
        result.append([bits, 128, 1.0, prune, 1.0, 0.0])
    return result
