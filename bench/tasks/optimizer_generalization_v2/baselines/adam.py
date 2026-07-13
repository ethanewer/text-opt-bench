"""Adam with one globally tuned learning rate."""

LR = 0.03
BETA1 = 0.9
BETA2 = 0.999


def zeros(shape):
    return [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]


def init(parameter_shapes):
    return [[zeros(shape) for shape in parameter_shapes],
            [zeros(shape) for shape in parameter_shapes]]


def update(parameter_blocks, gradient_blocks, state, step):
    first, second = state
    output, next_first, next_second = [], [], []
    for parameters, gradients, m, v in zip(parameter_blocks, gradient_blocks,
                                           first, second):
        matrix = parameters and isinstance(parameters[0], list)
        rows = zip(parameters if matrix else [parameters],
                   gradients if matrix else [gradients],
                   m if matrix else [m], v if matrix else [v])
        out_rows, m_rows, v_rows = [], [], []
        for prow, grow, mrow, vrow in rows:
            out, nm, nv = [], [], []
            for p, g, mi, vi in zip(prow, grow, mrow, vrow):
                mi = BETA1 * mi + (1.0 - BETA1) * g
                vi = BETA2 * vi + (1.0 - BETA2) * g * g
                out.append(p - LR * (mi / (1 - BETA1 ** step)) /
                           ((vi / (1 - BETA2 ** step)) ** 0.5 + 1e-8))
                nm.append(mi); nv.append(vi)
            out_rows.append(out); m_rows.append(nm); v_rows.append(nv)
        output.append(out_rows if matrix else out_rows[0])
        next_first.append(m_rows if matrix else m_rows[0])
        next_second.append(v_rows if matrix else v_rows[0])
    return [output, [next_first, next_second]]


def view(parameter_blocks, state, step):
    return parameter_blocks
