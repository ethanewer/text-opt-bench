"""Uncentered RMSProp baseline; constants are tuned by the baseline driver."""

LR = 0.01
BETA2 = 0.99
MOMENTUM = 0.0


def zeros(shape):
    return [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]


def init(parameter_shapes):
    return [[zeros(shape) for shape in parameter_shapes],
            [zeros(shape) for shape in parameter_shapes]]


def update(parameter_blocks, gradient_blocks, state, step):
    second, momentum = state
    output, next_second, next_momentum = [], [], []
    for parameters, gradients, values, moments in zip(
            parameter_blocks, gradient_blocks, second, momentum):
        matrix = parameters and isinstance(parameters[0], list)
        rows = zip(parameters if matrix else [parameters],
                   gradients if matrix else [gradients],
                   values if matrix else [values], moments if matrix else [moments])
        out_rows, value_rows, moment_rows = [], [], []
        for prow, grow, vrow, mrow in rows:
            out, nv, nm = [], [], []
            for p, g, v, m in zip(prow, grow, vrow, mrow):
                v = BETA2 * v + (1.0 - BETA2) * g * g
                m = MOMENTUM * m + g / (v ** 0.5 + 1e-8)
                out.append(p - LR * m); nv.append(v); nm.append(m)
            out_rows.append(out); value_rows.append(nv); moment_rows.append(nm)
        output.append(out_rows if matrix else out_rows[0])
        next_second.append(value_rows if matrix else value_rows[0])
        next_momentum.append(moment_rows if matrix else moment_rows[0])
    return [output, [next_second, next_momentum]]


def view(parameter_blocks, state, step):
    return parameter_blocks
