"""Higher-bit magnitude-pruned packed baseline."""


def plan(layers, target_bits):
    bits = 4 if target_bits < 3.5 else 6
    result = []
    for layer in layers:
        columns = layer[3]
        # One additional pruned input per row pays for the layer's one-byte
        # decode header while preserving the recognizable 50%-sparse regime.
        prune = (columns // 2 + 1) / columns
        result.append([bits, 128, 1.0, prune, 0.0, 0.0])
    return result
