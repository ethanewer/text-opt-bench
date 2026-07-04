def allocate(cache_info, config):
    n = config["n_tokens"]
    return [
        [n * 80 // 100, 97, 97],
        [n * 40 // 100, 65, 65],
        [n * 60 // 100, 97, 97],
        [n * 70 // 100, 97, 97],
    ]
