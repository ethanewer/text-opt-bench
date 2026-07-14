"""Schedule-Free AdamW paper Algorithm 1, with zero weight decay.

The evaluator supplies the gradient at ``y_t`` and ``view`` exposes ``x_t``.
Algorithm 1 applies second-moment bias correction in the preconditioner.  Its
iterate-average weight is the square of the scheduled learning-rate maximum,
which equals the scheduled learning rate for this monotone warmup schedule.
"""

LR = 0.03
BETA1 = 0.9
BETA2 = 0.999
WARMUP_STEPS = 10


def zeros(shape):
    return [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]


def copy_blocks(blocks):
    return [[list(row) for row in block] if block and isinstance(block[0], list)
            else list(block) for block in blocks]


def init(parameter_shapes):
    # z, x, uncorrected second moment, sum_i scheduled_lr_i^2
    return [None, None, [zeros(shape) for shape in parameter_shapes], 0.0]


def update(parameter_blocks, gradient_blocks, state, step):
    z, x, second, weight_sum = state
    if z is None:
        z, x = copy_blocks(parameter_blocks), copy_blocks(parameter_blocks)
    beta1, beta2 = BETA1, BETA2
    warmup = min(1.0, step / WARMUP_STEPS)
    scheduled_lr = LR * warmup
    bias_correction2 = 1.0 - beta2 ** step
    # Algorithm 1 uses lr_max**2 for averaging.  The schedule here is
    # nondecreasing, so lr_max_t == scheduled_lr_t at every step.
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
                xi = (1 - coefficient) * xi + coefficient * zi
                yr.append(beta1 * xi + (1.0 - beta1) * zi)
                zr.append(zi); xr.append(xi); vr.append(vi)
            y_rows.append(yr); z_rows.append(zr); x_rows.append(xr); v_rows.append(vr)
        output.append(y_rows if matrix else y_rows[0])
        next_z.append(z_rows if matrix else z_rows[0])
        next_x.append(x_rows if matrix else x_rows[0])
        next_second.append(v_rows if matrix else v_rows[0])
    return [output, [next_z, next_x, next_second, weight_sum]]


def view(parameter_blocks, state, step):
    return parameter_blocks if state[1] is None else state[1]
