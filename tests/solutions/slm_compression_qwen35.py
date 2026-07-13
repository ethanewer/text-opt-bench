def plan(layers, target_bits):
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
        if role in ("linear_attention_out", "full_attention_out", "mlp_down"):
            importance = 3
        elif role.startswith("linear_attention"):
            importance = 2
        else:
            importance = 1
        priorities.append((-importance, depth, index, count))
        activation_power = 0.5 if role.startswith("linear_attention") else 0.35
        result.append(
            [base_bits, 128, 1.0, 0.0, 0.0, activation_power])
    cap = int(target_bits * total_weights)
    for _importance, _depth, index, count in sorted(priorities):
        if used_bits + count <= cap:
            result[index][0] += 1
            used_bits += count
    return result
