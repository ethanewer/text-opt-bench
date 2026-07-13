def fit(training_rows):
    model_count = len(training_rows[0][2])
    costs = []
    global_quality = [0.0] * model_count
    global_cost = [0.0] * model_count
    buckets = {}
    for prompt, embedding, quality, cost in training_rows:
        code = 0
        for index, value in enumerate(embedding[:8]):
            if value >= 0:
                code |= 1 << index
        if code not in buckets:
            buckets[code] = [[0.0] * model_count, [0.0] * model_count, 0]
        local_quality, local_cost, count = buckets[code]
        for model in range(model_count):
            local_quality[model] += quality[model]
            local_cost[model] += cost[model]
            global_quality[model] += quality[model]
            global_cost[model] += cost[model]
            if cost[model] > 0:
                costs.append(cost[model])
        buckets[code][2] = count + 1
    costs.sort()
    scale = costs[len(costs) // 2] if costs else 1.0
    return [buckets, global_quality, global_cost, len(training_rows), scale]


def route(prompt, embedding, model_stats, cost_preference, state):
    buckets, global_quality, global_cost, total, scale = state
    code = 0
    for index, value in enumerate(embedding[:8]):
        if value >= 0:
            code |= 1 << index
    local = buckets.get(code)
    if local is not None and local[2] >= 8:
        quality, cost, count = local
    else:
        quality, cost, count = global_quality, global_cost, total
    best = 0
    best_value = -1e100
    for model in range(len(model_stats)):
        value = quality[model] / count - cost_preference * cost[model] / count / scale
        if value > best_value:
            best_value = value
            best = model
    return best
