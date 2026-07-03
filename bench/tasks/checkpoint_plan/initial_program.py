"""Baseline: first feasible balanced-memory checkpoint plan."""


def plan(layers, memory_budget):
    n = len(layers)
    total = sum(l["act_mem"] for l in layers)
    for nseg in range(2, min(n, 14) + 1):
        target = total / nseg
        out = [0]
        acc = 0
        for i, layer in enumerate(layers):
            acc += layer["act_mem"]
            if acc >= target and i + 1 < n:
                out.append(i + 1)
                acc = 0
        out.append(n)
        stored = sum(layers[i - 1]["act_mem"] for i in set(out) if 0 < i < n)
        max_segment = 0
        for a, b in zip(out, out[1:]):
            max_segment = max(max_segment, sum(l["act_mem"] for l in layers[a:b]))
        if stored + max_segment <= memory_budget:
            return out
    return [0, n]
