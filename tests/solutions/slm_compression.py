def plan(layers, target_bits):
    # Activation scaling needs one stored FP16 multiplier per input channel.
    # Start one code bit below dense RTN, then spend the remaining exact global
    # budget on whole-layer upgrades instead of pretending that metadata is free.
    base_bits = 2 if target_bits < 3.5 else 3
    policies = []
    total_weights = 0
    used_bits = 0
    priorities = []
    for index, layer in enumerate(layers):
        role, depth, rows, columns = layer[:4]
        count = rows * columns
        total_weights += count
        used_bits += count * base_bits
        used_bits += rows * ((columns + 127) // 128) * 16
        used_bits += columns * 16
        used_bits += 8
        importance = 2 if role in (
            "full_attention_out", "mlp_down") else 1
        priorities.append((-importance, depth, index, count))
        policies.append([base_bits, 128, 1.0, 0.0, 0.0, 0.5])
    cap = int(target_bits * total_weights)
    for _importance, _depth, index, count in sorted(priorities):
        if used_bits + count <= cap:
            policies[index][0] += 1
            used_bits += count
    return policies
