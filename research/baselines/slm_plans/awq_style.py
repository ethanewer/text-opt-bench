"""Activation-scaled RTN adapter; not a full reproduction of AWQ."""


def plan(layers, target_bits):
    # Activation scaling needs one stored FP16 multiplier per input channel.
    # Begin below dense RTN, account for every group/channel scale, then spend
    # the remaining global budget on whole-layer code-width upgrades.
    base_bits = 2 if target_bits < 3.5 else 3
    result = []
    priorities = []
    total_weights = 0
    used_bits = 0
    for index, layer in enumerate(layers):
        role, depth, rows, columns = layer[:4]
        count = rows * columns
        total_weights += count
        used_bits += count * base_bits
        used_bits += rows * ((columns + 127) // 128) * 16
        used_bits += columns * 16
        used_bits += 8
        power = 0.55 if "attention" in role else 0.45
        importance = 3 if role in (
            "full_attention_out", "linear_attention_out", "mlp_down"
        ) else 2 if "attention" in role else 1
        priorities.append((-importance, depth, index, count))
        result.append([base_bits, 128, 1.0, 0.0, 0.0, power])
    cap = int(target_bits * total_weights)
    for _importance, _depth, index, count in sorted(priorities):
        if used_bits + count <= cap:
            result[index][0] += 1
            used_bits += count
    return result
