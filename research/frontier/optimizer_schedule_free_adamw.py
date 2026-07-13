"""Schedule-Free AdamW adapted to the optimizer-synthesis candidate API.

The update follows Meta's simplified reference implementation.  The benchmark
does not expose an eval-mode callback, so it necessarily observes the training
iterate y rather than the paper's averaged evaluation iterate x.  Results from
this adapter must therefore be labelled API-adapted, not a faithful frontier.
"""


def init(task_info, parameter_count):
    zeros = [0.0] * parameter_count
    # z, x, second moment, cumulative averaging weight
    return [list(zeros), list(zeros), list(zeros), 0.0]


def update(parameters, gradients, state, step, task_info):
    z, x, second, weight_sum = state
    # The reference recommends learning rates around 1x--10x AdamW.  The
    # local benchmark's established Adam baseline uses 0.01.
    lr = 0.025
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight = lr * lr
    weight_sum += weight
    ckp1 = weight / weight_sum
    correction = 1.0 - beta2 ** step
    new_z = []
    new_x = []
    new_second = []
    result = []
    for p, g, old_z, old_x, v in zip(parameters, gradients, z, x, second):
        # State is initialized before the evaluator supplies initial params.
        # Reconstruct the reference initialization on the first update.
        if step == 1:
            old_z = p
            old_x = p
        v = beta2 * v + (1.0 - beta2) * g * g
        denom = (v / correction) ** 0.5 + eps
        zi = old_z - lr * g / denom
        xi = (1.0 - ckp1) * old_x + ckp1 * zi
        yi = beta1 * xi + (1.0 - beta1) * zi
        new_z.append(zi)
        new_x.append(xi)
        new_second.append(v)
        result.append(yi)
    return [result, [new_z, new_x, new_second, weight_sum]]
