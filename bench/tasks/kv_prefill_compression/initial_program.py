def select(model_family, layer_index, attention_scores, budget,
           observation_window):
    heads = len(attention_scores)
    length = len(attention_scores[0])
    old_budget = budget - observation_window
    recent = list(range(length - observation_window, length))
    result = []
    for head in range(heads):
        old = sorted(range(length - observation_window),
                     key=lambda i: attention_scores[head][i], reverse=True)
        result.append(old[:old_budget] + recent)
    return result
