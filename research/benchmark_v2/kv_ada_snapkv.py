"""Ada-KV head-adaptive allocation on top of SnapKV token scores."""


def select(model_family, layer_index, attention_scores, budget,
           observation_window):
    heads = len(attention_scores)
    length = len(attention_scores[0])
    old_length = length - observation_window
    old_budget = budget - observation_window
    floor = max(1, old_budget // 2)
    recent = list(range(old_length, length))
    result = []
    remaining = []
    for head in range(heads):
        ranked = sorted(range(old_length),
                        key=lambda i: attention_scores[head][i], reverse=True)
        result.append(ranked[:floor] + recent)
        for index in ranked[floor:]:
            remaining.append((attention_scores[head][index], head, index))
    remaining.sort(reverse=True)
    for score, head, index in remaining[:heads * (old_budget - floor)]:
        result[head].append(index)
    return result
