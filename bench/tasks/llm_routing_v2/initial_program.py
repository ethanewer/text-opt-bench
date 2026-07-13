def fit(training_rows):
    return None


def route(prompt, embedding, model_stats, cost_preference, state):
    best = 0
    best_value = -1e100
    for model, stats in enumerate(model_stats):
        value = stats[0] - cost_preference * stats[1]
        if value > best_value:
            best_value = value
            best = model
    return best
