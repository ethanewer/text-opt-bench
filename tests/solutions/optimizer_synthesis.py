def init(task_info, parameter_count):
    return [[0.0] * parameter_count, [0.0] * parameter_count]


def update(parameters, gradients, state, step, task_info):
    first, second = state
    result, new_first, new_second = [], [], []
    for p, g, m, v in zip(parameters, gradients, first, second):
        m = .9 * m + .1 * g
        v = .999 * v + .001 * g * g
        mh = m / (1 - .9 ** step)
        vh = v / (1 - .999 ** step)
        result.append(p - .01 * mh / (vh ** .5 + 1e-8))
        new_first.append(m)
        new_second.append(v)
    return [result, [new_first, new_second]]
