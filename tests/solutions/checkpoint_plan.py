"""Reference solution: deterministic search over balanced memory segments."""


def _score(layers, memory_budget, b):
    n = len(layers)
    b = sorted(set([0, n] + [x for x in b if 0 <= x <= n]))
    stored = sum(layers[i - 1]["act_mem"] for i in b if 0 < i < n)
    max_segment = 0
    recompute = 0
    for a, c in zip(b, b[1:]):
        if c <= a:
            return None
        max_segment = max(max_segment, sum(l["act_mem"] for l in layers[a:c]))
        recompute += sum(l["fwd_cost"] for l in layers[a : max(a, c - 1)])
    if stored + max_segment > memory_budget:
        return None
    return recompute, stored + max_segment, b


def plan(layers, memory_budget):
    n = len(layers)
    best = None
    total_mem = sum(l["act_mem"] for l in layers)
    for nseg in range(2, min(n, 20) + 1):
        for skew in (0.75, 0.9, 1.0, 1.15, 1.35):
            target = total_mem / nseg * skew
            b = [0]
            acc = 0
            for i, layer in enumerate(layers):
                acc += layer["act_mem"]
                if acc >= target and i + 1 < n:
                    b.append(i + 1)
                    acc = 0
            b.append(n)
            cand = _score(layers, memory_budget, b)
            if cand is not None and (best is None or cand < best):
                best = cand
    if best is not None:
        return best[2]
    return [0, n]
