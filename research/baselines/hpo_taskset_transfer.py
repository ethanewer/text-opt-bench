"""TaskSet ordered portfolio plus space-filling HPO-B exploration."""


def prepare(meta_tasks):
    rank_sums = {}
    counts = {}
    for task in meta_tasks:
        if task[0] != "taskset":
            continue
        curves = task[3]
        for rank, index in enumerate(
                sorted(range(len(curves)), key=lambda i: curves[i][-1])):
            rank_sums[index] = rank_sums.get(index, 0) + rank
            counts[index] = counts.get(index, 0) + 1
    return sorted(rank_sums, key=lambda i: rank_sums[i] / counts[i])


def suggest(task_info, configurations, observations, remaining_budget, state):
    kind = task_info[0]
    fidelity = task_info[2] - 1
    seen = {row[0] for row in observations if row[1] == fidelity}
    if kind == "taskset":
        for index in state:
            if index < len(configurations) and index not in seen:
                return [index, fidelity]
    if not observations:
        center = [0.5] * len(configurations[0])
        index = min(range(len(configurations)), key=lambda i: sum(
            (a - b) ** 2 for a, b in zip(configurations[i], center)))
        return [index, fidelity]
    observed = {row[0] for row in observations}
    candidates = [i for i in range(len(configurations)) if i not in seen]
    index = max(candidates, key=lambda i: min(
        sum((a - b) ** 2 for a, b in
            zip(configurations[i], configurations[j]))
        for j in observed))
    return [index, fidelity]
