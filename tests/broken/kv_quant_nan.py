def encode(cache, config):
    return 0


def attend(encoded, queries, config):
    return [
        [[float("nan") for _ in range(config["value_dim"])] for _ in q_layer]
        for q_layer in queries
    ]
