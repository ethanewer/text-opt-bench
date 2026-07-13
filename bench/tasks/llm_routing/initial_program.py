def fit(training_rows):
    return None


def route(prompt, model_stats, cost_penalty, state):
    best = 0
    best_value = -1e100
    for i, stats in enumerate(model_stats):
        value = stats[0] - cost_penalty * stats[1]
        if value > best_value:
            best_value = value
            best = i
    return best
