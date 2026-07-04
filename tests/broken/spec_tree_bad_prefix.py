def select(trees, config):
    out = []
    for tree in trees:
        child = None
        for node in tree["nodes"]:
            if node["parent"] != -1:
                child = node["id"]
                break
        out.append([child])
    return out
