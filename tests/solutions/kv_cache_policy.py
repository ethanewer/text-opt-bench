def select(token_ids, attention_scores, budget, absolute_position):
    sink = min(4, budget)
    recent = budget - sink
    keep = list(range(sink))
    keep.extend(range(max(sink, len(token_ids) - recent), len(token_ids)))
    return keep
