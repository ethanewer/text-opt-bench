def prepare(meta_tasks):
    totals = {}
    counts = {}
    for task in meta_tasks:
        if task[0] != "taskset":
            continue
        curves = task[3]
        order = sorted(range(len(curves)), key=lambda i: curves[i][-1])
        for rank, index in enumerate(order):
            totals[index] = totals.get(index, 0) + rank
            counts[index] = counts.get(index, 0) + 1
    return sorted(totals, key=lambda i: totals[i] / counts[i])


def suggest(task_info, configurations, observations, remaining_budget, state):
    fidelities = task_info[2]
    seen = set((row[0], row[1]) for row in observations)
    # Spend low fidelities on TaskSet, then promote the best observed curves.
    if fidelities > 1:
        for index in range(len(configurations)):
            if (index, fidelities - 1) not in seen:
                return [index, fidelities - 1]
    observed = [row[0] for row in observations]
    if not observed:
        center = [0.5] * len(configurations[0])
        return [min(range(len(configurations)),
                    key=lambda i: sum((a-b)**2 for a,b in zip(configurations[i], center))),
                fidelities - 1]
    candidates = [i for i in range(len(configurations))
                  if (i, fidelities - 1) not in seen]
    index = max(candidates, key=lambda i: min(
        sum((a-b)**2 for a,b in zip(configurations[i], configurations[j]))
        for j in observed))
    return [index, fidelities - 1]
