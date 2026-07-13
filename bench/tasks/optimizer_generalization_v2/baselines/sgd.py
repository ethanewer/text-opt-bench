"""Globally clipped heavy-ball SGD baseline."""

LR = 0.03
MOMENTUM = 0.9
CLIP_NORM = 10.0


def zeros(shape):
    return [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]


def init(parameter_shapes):
    return [zeros(shape) for shape in parameter_shapes]


def update(parameter_blocks, gradient_blocks, state, step):
    squared = 0.0
    for block in gradient_blocks:
        rows = block if block and isinstance(block[0], list) else [block]
        for row in rows:
            squared += sum(value * value for value in row)
    gradient_scale = min(1.0, CLIP_NORM / (squared ** 0.5 + 1e-12))
    output, next_state = [], []
    for parameters, gradients, velocity in zip(parameter_blocks, gradient_blocks, state):
        matrix = parameters and isinstance(parameters[0], list)
        rows = zip(parameters if matrix else [parameters],
                   gradients if matrix else [gradients],
                   velocity if matrix else [velocity])
        out_rows, state_rows = [], []
        for prow, grow, vrow in rows:
            out, updated = [], []
            for p, g, old in zip(prow, grow, vrow):
                value = MOMENTUM * old + gradient_scale * g
                out.append(p - LR * value)
                updated.append(value)
            out_rows.append(out); state_rows.append(updated)
        output.append(out_rows if matrix else out_rows[0])
        next_state.append(state_rows if matrix else state_rows[0])
    return [output, next_state]


def view(parameter_blocks, state, step):
    return parameter_blocks
