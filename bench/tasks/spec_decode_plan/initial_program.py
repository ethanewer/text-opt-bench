def plan(requests, config):
    out = []
    for req in requests:
        n = req["output_tokens"]
        k = config["max_draft"]
        out.append([min(k, n - pos) for pos in range(n)])
    return out
