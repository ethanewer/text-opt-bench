"""Adam baseline for the structured optimizer v2 API."""


def zeros(shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [[0.0] * shape[1] for _ in range(shape[0])]


def init(task_info, parameter_shapes):
    return [[zeros(shape) for shape in parameter_shapes],
            [zeros(shape) for shape in parameter_shapes]]


def update(parameter_blocks, gradient_blocks, state, step, task_info):
    first, second = state
    output, next_first, next_second = [], [], []
    for parameters, gradients, m, v, shape in zip(
            parameter_blocks, gradient_blocks, first, second, task_info[2]):
        matrix = len(shape) == 2
        prows = parameters if matrix else [parameters]
        grows = gradients if matrix else [gradients]
        mrows = m if matrix else [m]
        vrows = v if matrix else [v]
        out_rows, nm_rows, nv_rows = [], [], []
        for prow, grow, mrow, vrow in zip(prows, grows, mrows, vrows):
            out, nm, nv = [], [], []
            for p, g, old_m, old_v in zip(prow, grow, mrow, vrow):
                mi = 0.9 * old_m + 0.1 * g
                vi = 0.999 * old_v + 0.001 * g * g
                out.append(p - 0.01 * (mi / (1 - 0.9 ** step)) /
                           ((vi / (1 - 0.999 ** step)) ** 0.5 + 1e-8))
                nm.append(mi)
                nv.append(vi)
            out_rows.append(out); nm_rows.append(nm); nv_rows.append(nv)
        output.append(out_rows if matrix else out_rows[0])
        next_first.append(nm_rows if matrix else nm_rows[0])
        next_second.append(nv_rows if matrix else nv_rows[0])
    return [output, [next_first, next_second]]


def view(parameter_blocks, state, step, task_info):
    return parameter_blocks
