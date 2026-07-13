def select(token_ids, attention_scores, budget, absolute_position):
    return list(range(len(token_ids) - budget, len(token_ids)))
