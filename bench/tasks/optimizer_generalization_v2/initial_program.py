def zeros(shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [[0.0] * shape[1] for _ in range(shape[0])]


def init(parameter_shapes):
    return [[zeros(shape) for shape in parameter_shapes],
            [zeros(shape) for shape in parameter_shapes]]


def update(parameter_blocks, gradient_blocks, state, step):
    first, second = state
    output, next_first, next_second = [], [], []
    for parameters, gradients, m, v in zip(
            parameter_blocks, gradient_blocks, first, second):
        matrix = parameters and isinstance(parameters[0], list)
        parameter_rows = parameters if matrix else [parameters]
        gradient_rows = gradients if matrix else [gradients]
        first_rows = m if matrix else [m]
        second_rows = v if matrix else [v]
        out_rows, new_m_rows, new_v_rows = [], [], []
        for prow, grow, mrow, vrow in zip(
                parameter_rows, gradient_rows, first_rows, second_rows):
            out, new_m, new_v = [], [], []
            for p, g, old_m, old_v in zip(prow, grow, mrow, vrow):
                mi = 0.9 * old_m + 0.1 * g
                vi = 0.999 * old_v + 0.001 * g * g
                corrected_m = mi / (1.0 - 0.9 ** step)
                corrected_v = vi / (1.0 - 0.999 ** step)
                out.append(p - 0.015 * corrected_m /
                           (corrected_v ** 0.5 + 1e-8))
                new_m.append(mi)
                new_v.append(vi)
            out_rows.append(out)
            new_m_rows.append(new_m)
            new_v_rows.append(new_v)
        output.append(out_rows if matrix else out_rows[0])
        next_first.append(new_m_rows if matrix else new_m_rows[0])
        next_second.append(new_v_rows if matrix else new_v_rows[0])
    return [output, [next_first, next_second]]


def view(parameter_blocks, state, step):
    return parameter_blocks
