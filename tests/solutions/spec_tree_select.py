def _merge(a, b, cap):
    out = [-1.0] * (cap + 1)
    for i in range(cap + 1):
        if a[i] < 0.0:
            continue
        limit = cap - i
        for j in range(limit + 1):
            if b[j] >= 0.0:
                v = a[i] + b[j]
                if v > out[i + j]:
                    out[i + j] = v
    return out


def _node_dp(nid, by_id, children, depth_limit, cap):
    node = by_id[nid]
    if node["depth"] > depth_limit:
        out = [-1.0] * (cap + 1)
        out[0] = 0.0
        return out
    inc = [-1.0] * (cap + 1)
    inc[1] = node["path_prob"]
    for child in children.get(nid, ()):
        inc = _merge(inc, _node_dp(child["id"], by_id, children, depth_limit, cap), cap)
    out = [-1.0] * (cap + 1)
    out[0] = 0.0
    for i in range(1, cap + 1):
        out[i] = inc[i]
    return out


def _select_one(tree, config):
    by_id = {}
    children = {}
    for node in tree["nodes"]:
        by_id[node["id"]] = node
        children.setdefault(node["parent"], []).append(node)
    cap = config["max_nodes"]
    best_depth = 1
    best_count = 0
    best_score = config["verify_base"]
    for depth in range(1, config["max_depth"] + 1):
        forest = [-1.0] * (cap + 1)
        forest[0] = 0.0
        for root in children.get(-1, ()):
            forest = _merge(forest, _node_dp(root["id"], by_id, children, depth, cap), cap)
        for count in range(1, cap + 1):
            gain = forest[count]
            if gain < 0.0:
                continue
            cost = (
                config["verify_base"]
                + (config["verify_per_node"] + config["draft_per_node"]) * count
                + config["depth_overhead"] * depth * depth
            )
            score = cost / (1.0 + gain)
            if score < best_score:
                best_score = score
                best_depth = depth
                best_count = count

    selected = []
    frontier = list(children.get(-1, ()))
    while frontier and len(selected) < best_count:
        best_i = 0
        best_value = -1.0
        for i, node in enumerate(frontier):
            if node["depth"] <= best_depth and node["path_prob"] > best_value:
                best_value = node["path_prob"]
                best_i = i
        node = frontier.pop(best_i)
        if node["depth"] > best_depth:
            continue
        selected.append(node["id"])
        for child in children.get(node["id"], ()):
            frontier.append(child)
    return sorted(selected)


def select(trees, config):
    return [_select_one(tree, config) for tree in trees]
