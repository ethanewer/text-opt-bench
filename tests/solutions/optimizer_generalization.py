"""Schedule-Free AdamW Algorithm 1 headroom solution for protocol v9."""

LR = 0.06


def zeros(shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [[0.0] * shape[1] for _ in range(shape[0])]


def copy_blocks(blocks):
    return [[list(row) for row in block]
            if block and isinstance(block[0], list) else list(block)
            for block in blocks]


def init(parameter_shapes):
    return [None, None, [zeros(shape) for shape in parameter_shapes], 0.0]


def update(parameter_blocks, gradient_blocks, state, step):
    z, x, second, weight_sum = state
    if z is None:
        z, x = copy_blocks(parameter_blocks), copy_blocks(parameter_blocks)
    beta1, beta2 = 0.8, 0.999
    scheduled_lr = LR * min(1.0, step / 10.0)
    bias_correction2 = 1.0 - beta2 ** step
    weight = scheduled_lr * scheduled_lr
    weight_sum += weight
    coefficient = weight / weight_sum
    output, next_z, next_x, next_second = [], [], [], []
    for gradients, zb, xb, vb in zip(gradient_blocks, z, x, second):
        matrix = gradients and isinstance(gradients[0], list)
        rows = zip(gradients if matrix else [gradients],
                   zb if matrix else [zb], xb if matrix else [xb],
                   vb if matrix else [vb])
        y_rows, z_rows, x_rows, v_rows = [], [], [], []
        for grow, zrow, xrow, vrow in rows:
            yr, zr, xr, vr = [], [], [], []
            for g, zi, xi, vi in zip(grow, zrow, xrow, vrow):
                vi = beta2 * vi + (1.0 - beta2) * g * g
                v_hat = vi / bias_correction2
                zi = zi - scheduled_lr * g / (v_hat ** 0.5 + 1e-8)
                xi = (1.0 - coefficient) * xi + coefficient * zi
                yr.append(beta1 * xi + (1.0 - beta1) * zi)
                zr.append(zi)
                xr.append(xi)
                vr.append(vi)
            y_rows.append(yr)
            z_rows.append(zr)
            x_rows.append(xr)
            v_rows.append(vr)
        output.append(y_rows if matrix else y_rows[0])
        next_z.append(z_rows if matrix else z_rows[0])
        next_x.append(x_rows if matrix else x_rows[0])
        next_second.append(v_rows if matrix else v_rows[0])
    return [output, [next_z, next_x, next_second, weight_sum]]


def view(parameter_blocks, state, step):
    return parameter_blocks if state[1] is None else state[1]
