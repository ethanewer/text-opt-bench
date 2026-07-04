def _round_cost(k, req, config):
    target = (config["target_base"] + config["target_per_token"] * (k + 1)) * req["verify_scale"]
    draft = (config["draft_base"] + config["draft_per_token"] * k) * req["draft_scale"]
    return target + draft + config["stall_penalty"] * k * k


def _best_policy(req, config):
    n = req["output_tokens"]
    max_draft = config["max_draft"]
    value = [0.0] * (n + max_draft + 2)
    policy = [1] * n
    acc = req["accept"]
    for pos in range(n - 1, -1, -1):
        rem = n - pos
        best = None
        best_k = 1
        limit = min(max_draft, rem)
        for k in range(1, limit + 1):
            cost = _round_cost(k, req, config)
            prob = 1.0
            future = 0.0
            for accepted in range(k):
                reject = prob * (1.0 - acc[pos + accepted])
                future += reject * value[min(n, pos + accepted + 1)]
                prob *= acc[pos + accepted]
            future += prob * value[min(n, pos + k + 1)]
            total = cost + future
            if best is None or total < best:
                best = total
                best_k = k
        value[pos] = best
        policy[pos] = best_k
    return policy


def plan(requests, config):
    return [_best_policy(req, config) for req in requests]
