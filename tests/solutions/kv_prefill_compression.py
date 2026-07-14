def select(model_family, layer_index, attention_scores, budget, observation_window):
    length = len(attention_scores[0]); old = budget - observation_window
    recent = list(range(length - observation_window, length)); result = []
    for scores in attention_scores:
        ranked = sorted(range(length - observation_window), key=lambda i: scores[i], reverse=True)
        result.append(ranked[:old] + recent)
    return result
