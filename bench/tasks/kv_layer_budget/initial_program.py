def allocate(cache_info, config):
    n = config["n_tokens"]
    keep = n * 40 // 100
    return [[keep, 49, 49] for _ in range(config["n_layers"])]
