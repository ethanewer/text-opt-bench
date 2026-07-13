def init(task_info, parameter_count):
    return [0.0] * parameter_count


def update(parameters, gradients, state, step, task_info):
    beta = 0.9
    lr = 0.01
    result = []
    new_state = []
    for p, g, momentum in zip(parameters, gradients, state):
        momentum = beta * momentum + (1.0 - beta) * g
        result.append(p - lr * momentum)
        new_state.append(momentum)
    return [result, new_state]
