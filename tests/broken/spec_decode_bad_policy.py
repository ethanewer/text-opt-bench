def plan(requests, config):
    return [[config["max_draft"] + 1] for _ in requests]
