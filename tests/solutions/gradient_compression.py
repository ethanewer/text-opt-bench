def init(parameter_count, worker_count, workload_info):
    return [None] * worker_count


def encode(corrected_gradient, max_items, worker_info, state):
    order = sorted(range(len(corrected_gradient)),
                   key=lambda i: abs(corrected_gradient[i]), reverse=True)
    indices = order[:max_items]
    return [indices, [corrected_gradient[i] for i in indices], state]
