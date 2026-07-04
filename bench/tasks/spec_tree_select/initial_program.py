def select(trees, config):
    out = []
    cap = config["max_nodes"]
    for tree in trees:
        chosen = []
        for node in tree["nodes"]:
            if len(chosen) < cap:
                chosen.append(node["id"])
        out.append(chosen)
    return out
