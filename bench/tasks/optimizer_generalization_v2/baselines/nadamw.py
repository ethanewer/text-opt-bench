"""NAdamW baseline following Dozat's Nesterov-Adam update.

Weight decay defaults to zero because the benchmark ranks validation-loss
optimization rather than model regularization; the driver records and tunes
this choice explicitly.
"""

LR = 0.01
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.0


def zeros(shape):
    return [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]


def init(parameter_shapes):
    return [[zeros(shape) for shape in parameter_shapes],
            [zeros(shape) for shape in parameter_shapes]]


def update(parameter_blocks, gradient_blocks, state, step):
    first, second = state
    output, next_first, next_second = [], [], []
    bc1, bc2 = 1.0 - BETA1 ** step, 1.0 - BETA2 ** step
    for parameters, gradients, moments, values in zip(
            parameter_blocks, gradient_blocks, first, second):
        matrix = parameters and isinstance(parameters[0], list)
        rows = zip(parameters if matrix else [parameters],
                   gradients if matrix else [gradients],
                   moments if matrix else [moments], values if matrix else [values])
        out_rows, moment_rows, value_rows = [], [], []
        for prow, grow, mrow, vrow in rows:
            out, nm, nv = [], [], []
            for p, g, m, v in zip(prow, grow, mrow, vrow):
                m = BETA1 * m + (1.0 - BETA1) * g
                v = BETA2 * v + (1.0 - BETA2) * g * g
                nesterov = BETA1 * m / bc1 + (1.0 - BETA1) * g / bc1
                update_value = nesterov / ((v / bc2) ** 0.5 + 1e-8)
                out.append(p * (1.0 - LR * WEIGHT_DECAY) - LR * update_value)
                nm.append(m); nv.append(v)
            out_rows.append(out); moment_rows.append(nm); value_rows.append(nv)
        output.append(out_rows if matrix else out_rows[0])
        next_first.append(moment_rows if matrix else moment_rows[0])
        next_second.append(value_rows if matrix else value_rows[0])
    return [output, [next_first, next_second]]


def view(parameter_blocks, state, step):
    return parameter_blocks
