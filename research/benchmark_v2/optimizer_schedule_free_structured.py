"""Schedule-Free AdamW reference update with its averaged evaluation iterate."""


def zeros(shape):
    return [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]


def init(task_info, parameter_shapes):
    return [None, None, [zeros(shape) for shape in parameter_shapes], 0.0]


def update(parameter_blocks, gradient_blocks, state, step, task_info):
    z, x, second, weight_sum = state
    if z is None:
        z = [([list(row) for row in block] if len(shape) == 2 else list(block))
             for block, shape in zip(parameter_blocks, task_info[2])]
        x = [([list(row) for row in block] if len(shape) == 2 else list(block))
             for block, shape in zip(parameter_blocks, task_info[2])]
    lr, beta1, beta2 = 0.025, 0.9, 0.999
    weight_sum += lr * lr
    c = lr * lr / weight_sum
    output, nz, nx, nv = [], [], [], []
    for parameters, gradients, zb, xb, vb, shape in zip(
            parameter_blocks, gradient_blocks, z, x, second, task_info[2]):
        matrix = len(shape) == 2
        rows = zip(parameters if matrix else [parameters],
                   gradients if matrix else [gradients],
                   zb if matrix else [zb], xb if matrix else [xb],
                   vb if matrix else [vb])
        yo, zo, xo, vo = [], [], [], []
        for prow, grow, zrow, xrow, vrow in rows:
            yr, zr, xr, vr = [], [], [], []
            for p, g, zi, xi, vi in zip(prow, grow, zrow, xrow, vrow):
                vi = beta2 * vi + (1 - beta2) * g * g
                zi = zi - lr * g / ((vi / (1 - beta2 ** step)) ** 0.5 + 1e-8)
                xi = (1 - c) * xi + c * zi
                yr.append(beta1 * xi + (1 - beta1) * zi)
                zr.append(zi); xr.append(xi); vr.append(vi)
            yo.append(yr); zo.append(zr); xo.append(xr); vo.append(vr)
        output.append(yo if matrix else yo[0]); nz.append(zo if matrix else zo[0])
        nx.append(xo if matrix else xo[0]); nv.append(vo if matrix else vo[0])
    return [output, [nz, nx, nv, weight_sum]]


def view(parameter_blocks, state, step, task_info):
    return parameter_blocks if state[1] is None else state[1]
