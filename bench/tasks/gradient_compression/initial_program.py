def init(parameter_count, worker_count, workload_info):
    return [None] * worker_count


def encode(corrected_gradient, max_items, worker_info, state):
    indices = list(range(max_items))
    values = [corrected_gradient[i] for i in indices]
    return [indices, values, state]
