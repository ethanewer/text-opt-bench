"""Symmetric groupwise round-to-nearest baseline with honest headers."""


def plan(layers, target_bits):
    # Dense groupwise RTN would sit exactly on the nominal cap before its
    # per-layer decode headers. Keep the representation honestly under the
    # hard cap by lowering the least-sensitive whole layers as needed.
    desired_bits = 3 if target_bits < 3.5 else 4
    policies = []
    downgrade_order = []
    total_weights = 0
    used_bits = 0
    for index, layer in enumerate(layers):
        role, depth, rows, columns = layer[:4]
        count = rows * columns
        importance = 3 if role in (
            "full_attention_out", "linear_attention_out", "mlp_down"
        ) else 2 if "attention" in role else 1
        policies.append([desired_bits, 128, 1.0, 0.0, 0.0, 0.0])
        downgrade_order.append((importance, -depth, count, index))
        total_weights += count
        used_bits += count * desired_bits
        used_bits += rows * ((columns + 127) // 128) * 16
        used_bits += 8
    cap = int(target_bits * total_weights)
    ordered = sorted(downgrade_order)
    while used_bits > cap:
        changed = False
        for _importance, _depth, count, index in ordered:
            if policies[index][0] > 2:
                policies[index][0] -= 1
                used_bits -= count
                changed = True
                if used_bits <= cap:
                    break
        if not changed:
            break
    return policies

